"""SpotGPU preemption (OSDI'26 "Heterogeneity at Hyperscale", Section 4.2).

HP tasks may evict LP spot tasks. The victim set on a node is the
lowest-preemption-cost prefix. Preemption cost of a spot task t:

    C_t = G_t * (now - last_checkpoint)

with checkpoints every `checkpoint_interval` seconds since the task started, so a
task that has just checkpointed is cheap to evict and one that has run a long time
since is expensive. On eviction the task keeps `preserved_progress` (whole
checkpoints) and loses the remainder, which is what `wasted_gpu_sec` measures.

Two things a reader should not infer from this file:

- The checkpoint interval is a **model assumption**, not trace data. The ASI trace
  carries no checkpoint information; the paper treats it as a deployment parameter.
  Changing `checkpoint_interval` changes `wasted_gpu_sec` roughly proportionally, so
  any comparison across runs must pin it.
- There is no interaction with live migration here. "Preemption interrupts an
  in-flight migration" has no basis in the paper and is deliberately not
  implemented; migration semantics belong to M11's four reservation switches.
"""
from .contract import EPS


class PreemptionMixin:
    def _spotgpu_victim_cost(self, pid):
        task = self.by_id[pid]
        elapsed = self.current_time - self.running_start[pid]
        checkpoints = int(elapsed // self.checkpoint_interval)
        last_ck = self.running_start[pid] + checkpoints * self.checkpoint_interval
        return task.gpu * (self.current_time - last_ck)

    def _spotgpu_best_preemption(self, task):
        """Best (node, victims, cost) to reclaim LP spot GPUs for an HP task, or None."""
        best = None
        for j, node in enumerate(self.nodes):
            if not (task.gpu == 0 or node.gpu_type in task.allowed_gpu_types):
                continue
            if task.gpu > node.gpu_capacity+EPS or task.cpu > node.cpu_capacity*(1-self.background)+EPS:
                continue
            free_g, free_c = self.free_gpu[j], self.free_cpu[j]
            if free_g >= task.gpu-EPS and free_c >= task.cpu-EPS:
                continue  # fits non-preemptively; normal path handles it
            # Tie-break by pod_id so the victim order is a total order.
            #
            # Without it this line is not reproducible: node_spot_pids is a set of
            # strings, CPython randomises string hashes per process (PYTHONHASHSEED),
            # and `sorted` is stable, so equal-cost victims come out in whatever order
            # that process happened to iterate. Scene F warmup tasks all start at t=0,
            # so ties are the common case rather than an edge case. Measured on the full
            # 119,495-task scene: preemptions 232 under PYTHONHASHSEED=0 and 242 under
            # PYTHONHASHSEED=1, with wasted_gpu_sec differing by 6,417 GPU·s and
            # reserved_gpu_ratio by 3.2 points, on identical code and identical input.
            #
            # pod_id ascending is an arbitrary choice, not a paper mechanism: OSDI'26
            # §4.2 defines the cost C_t = G_t*(now-last_checkpoint) and says nothing
            # about equal-cost victims. Any fixed total order would do; the point is
            # that the run must not depend on which process it happens to run in.
            spot = sorted(self.node_spot_pids[j], key=lambda pid: (self._spotgpu_victim_cost(pid), pid))
            if not spot:
                continue
            victims, total, freed_g, freed_c = [], 0., 0., 0.
            for pid in spot:
                t = self.by_id[pid]
                victims.append(pid)
                total += self._spotgpu_victim_cost(pid)
                freed_g += t.gpu
                freed_c += t.cpu
                if free_g+freed_g >= task.gpu-EPS and free_c+freed_c >= task.cpu-EPS:
                    if best is None or total < best[2]:
                        best = (j, list(victims), total)
                    break
        return best

    def _evict(self, victims, node):
        """Preempt running LP spot tasks and return them to the waiting queue."""
        for pid in victims:
            task = self.by_id[pid]
            elapsed = self.current_time - self.running_start[pid]
            checkpoints = int(elapsed // self.checkpoint_interval)
            preserved_now = checkpoints * self.checkpoint_interval
            wasted = elapsed - preserved_now
            self.preserved_progress[pid] = self.preserved_progress.get(pid, 0.0) + preserved_now
            record = self.records[pid]
            record["preemption_count"] += 1
            record["wasted_sec"] += wasted
            record["running_total_sec"] += elapsed
            self.running.pop(pid, None)
            self.running_start.pop(pid, None)
            # Invalidate the pending completion event; a new one is pushed on
            # the next placement.
            self.completion_counter.pop(pid, None)
            self.release(pid, node)
            self.node_lp_running[node] -= 1
            self.node_spot_pids[node].discard(pid)
            self.node_evictions[node] += 1
            self._enqueue(pid)
            record["status"] = "waiting"
            self._record("preempt", pid)
        self._preempt_generation += 1
        self._check_resources()

    def _try_preempt(self):
        """When no waiting task fits, place the earliest blocked HP GPU task by
        preempting the lowest-cost spot tasks. Returns True if it acted."""
        if self._last_preempt_generation == self._preempt_generation:
            return False
        self._last_preempt_generation = self._preempt_generation
        if not any(self.node_spot_pids):
            return False
        candidates = []
        for index, group in enumerate(self._hp_gpu_wait_groups):
            cursor = self._hp_gpu_wait_group_cursors[index]
            while cursor < len(group):
                sequence, pid = group[cursor]
                if pid in self.waiting_set and self._queue_seq.get(pid) == sequence:
                    break
                cursor += 1
            self._hp_gpu_wait_group_cursors[index] = cursor
            if cursor < len(group):
                candidates.append(group[cursor])
        for _, pid in sorted(candidates):
            task = self.by_id[pid]
            cand = self._spotgpu_best_preemption(task)
            if cand is None:
                continue
            node, victims, _ = cand
            self._evict(victims, node)
            self._place(pid, node)
            return True
        return False
