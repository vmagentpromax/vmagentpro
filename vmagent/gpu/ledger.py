"""Per-node GPU/CPU reservation ledger: what fits, what is taken, and the invariant.

This is the module the collaborator's defragmentation engine has to land on (M10).
Her `engine.py` recomputes occupancy from scratch at every event (`st.check()` walks
all pods of all hosts); this ledger keeps incremental per-node totals and asserts
them against the free vectors instead. Consolidating allocate/release here is also
M4's first item: the same five- and seven-line blocks used to be written out three
times, in `_place`, in `_process_due` and in `_evict`.

Priority bookkeeping (which HP/LP tasks sit on a node, which are preemptible spot
tasks) deliberately stays with the caller: it is SpotGPU state, not ledger state,
and a migration policy must not have to know about it.
"""
from .contract import EPS


class LedgerMixin:
    def _fits(self, task, index, empty=False):
        node = self.nodes[index]
        gpu = node.gpu_capacity if empty else self.free_gpu[index]
        cpu = node.cpu_capacity*(1-self.background) if empty else self.free_cpu[index]
        return ((task.gpu == 0 or node.gpu_type in task.allowed_gpu_types)
                and task.gpu <= gpu+EPS and task.cpu <= cpu+EPS)

    def _any_fits(self, task, max_free_cpu=None):
        if task.gpu == 0:
            if max_free_cpu is None:
                max_free_cpu = max(self.free_cpu)
            return task.cpu <= max_free_cpu+EPS
        return any(self._fits(task, j) for model in task.allowed_gpu_types
                   for j in self.model_nodes.get(model, ()))

    def allocate(self, pid, index):
        """Reserve a task's request on a node. Order of operations is preserved so
        that float results match the pre-split engine exactly."""
        task = self.by_id[pid]
        self.free_gpu[index] -= task.gpu
        self.free_cpu[index] -= task.cpu
        self.node_gpu_used[index] += task.gpu
        self.node_cpu_used[index] += task.cpu
        self.node_running_count[index] += 1

    def release(self, pid, index):
        """Return a task's request to a node, clamped to capacity.

        The clamp is not dead code: a double release (releasing a task that was
        already released) would otherwise silently inflate free capacity past the
        node's total and be caught only by the invariant below.
        """
        task = self.by_id[pid]
        node = self.nodes[index]
        self.free_gpu[index] += task.gpu
        self.free_cpu[index] += task.cpu
        self.free_gpu[index] = min(node.gpu_capacity, self.free_gpu[index])
        self.free_cpu[index] = min(node.cpu_capacity*(1-self.background), self.free_cpu[index])
        self.node_gpu_used[index] -= task.gpu
        self.node_cpu_used[index] -= task.cpu
        self.node_running_count[index] -= 1

    def _check_resources(self):
        # Incremental ledgers are updated at placement and completion; compare
        # them against the free-resource vectors (both accumulated in the same
        # event order, so float drift stays far below EPS).
        for j, node in enumerate(self.nodes):
            expected_gpu = node.gpu_capacity - self.node_gpu_used[j]
            expected_cpu = node.cpu_capacity*(1-self.background) - self.node_cpu_used[j]
            if (abs(expected_gpu-self.free_gpu[j]) > 1e-7 or abs(expected_cpu-self.free_cpu[j]) > 1e-7
                    or expected_gpu < -EPS or expected_cpu < -EPS):
                raise RuntimeError("Resource ledger violated on " + node.server_id)

    def _rejection_reason(self, task):
        compatible = [n for n in self.nodes if task.gpu == 0 or n.gpu_type in task.allowed_gpu_types]
        if not compatible:
            return "gpu_model_unavailable"
        if all(task.gpu > n.gpu_capacity+EPS for n in compatible):
            return "gpu_request_exceeds_single_node"
        if all(task.cpu > n.cpu_capacity*(1-self.background)+EPS for n in compatible):
            return "cpu_request_exceeds_single_node_after_background"
        return "joint_request_never_fits_one_node"
