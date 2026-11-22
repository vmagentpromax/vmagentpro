"""Waiting-queue indexes: who is next, and why the one before them could not go.

Three structures, all keyed by a monotonic enqueue sequence so that "earliest
arrived still waiting" is answerable without scanning the queue:

- FCFS needs only insertion order (`_waiting_order`, `_queue_seq`).
- CPU-only backfill needs the minimum enqueue sequence at or below a given CPU
  demand, which is a prefix-minimum query: a segment tree over demand buckets
  (`_cpu_wait_tree`), because scanning every waiting task per decision made the
  89,127-deep scenario F queue quadratic.
- GPU backfill groups tasks by identical (gpu, cpu, models) demand and keeps a
  cursor per group. A group's blocked head stays blocked while the resources are
  unchanged, so the cursor advances once instead of rescanning the group per event.

The blocking-reason classifier lives here too because it can only be observed
during this scan: `_next_ready` returns a task only when it already fits, so a
blocked task never reaches the decision point where a policy would see it.
"""
import heapq
import math
from bisect import bisect_right

from .contract import EPS


class QueueMixin:
    def _cpu_tree_set(self, index, sequence):
        position = self.cpu_tree_size+index
        self._cpu_wait_tree[position] = sequence
        position //= 2
        while position:
            self._cpu_wait_tree[position] = min(self._cpu_wait_tree[2*position],
                                                 self._cpu_wait_tree[2*position+1])
            position //= 2

    def _cpu_prefix_min(self, limit):
        left, right = self.cpu_tree_size, self.cpu_tree_size+limit
        result = math.inf
        while left < right:
            if left & 1:
                result = min(result, self._cpu_wait_tree[left])
                left += 1
            if right & 1:
                right -= 1
                result = min(result, self._cpu_wait_tree[right])
            left //= 2
            right //= 2
        return result

    def _cpu_wait_candidate(self, max_free_cpu):
        limit = bisect_right(self.cpu_only_demands, max_free_cpu+EPS)
        while limit:
            sequence = self._cpu_prefix_min(limit)
            if sequence == math.inf:
                return None
            pid = self._waiting_order[int(sequence)]
            if pid in self.waiting_set and self._queue_seq.get(pid) == sequence:
                return pid
            index = self.cpu_demand_index[self.by_id[pid].cpu]
            heap = self._cpu_wait_heaps[index]
            while heap and (heap[0][1] not in self.waiting_set
                            or self._queue_seq.get(heap[0][1]) != heap[0][0]):
                heapq.heappop(heap)
            self._cpu_tree_set(index, heap[0][0] if heap else math.inf)
        return None

    def _enqueue(self, pid):
        self.waiting[pid] = None
        self.waiting_set.add(pid)
        sequence = len(self._waiting_order)
        self._queue_seq[pid] = sequence
        self._waiting_order.append(pid)
        self._queue_seq_counter += 1
        task = self.by_id[pid]
        if self.queue_policy == "backfill":
            if task.gpu == 0:
                index = self.cpu_demand_index[task.cpu]
                heap = self._cpu_wait_heaps[index]
                heapq.heappush(heap, (sequence, pid))
                self._cpu_tree_set(index, heap[0][0])
            else:
                key = (task.gpu, task.cpu, task.allowed_gpu_types)
                index = self._gpu_wait_group_index.get(key)
                if index is None:
                    index = len(self._gpu_wait_groups)
                    self._gpu_wait_group_index[key] = index
                    self._gpu_wait_groups.append([])
                    self._gpu_wait_group_cursors.append(0)
                self._gpu_wait_groups[index].append((sequence, pid))
        if task.gpu > 0 and self.task_priority.get(pid) == "HP":
            key = (task.gpu, task.cpu, task.allowed_gpu_types)
            index = self._hp_gpu_wait_group_index.get(key)
            if index is None:
                index = len(self._hp_gpu_wait_groups)
                self._hp_gpu_wait_group_index[key] = index
                self._hp_gpu_wait_groups.append([])
                self._hp_gpu_wait_group_cursors.append(0)
            self._hp_gpu_wait_groups[index].append((sequence, pid))
            self._preempt_generation += 1

    def _group_head(self, group, cursors, index):
        """Advance one demand group's cursor to its earliest still-waiting entry.

        The cursor is the whole point: a group's blocked head stays blocked while the
        ledger is unchanged, so scanning once per group per event is enough. `sequence`
        is compared against `_queue_seq` because a preempted task re-enters its group
        with a new sequence and the stale entry has to be skipped, not honoured.
        """
        cursor = cursors[index]
        while cursor < len(group):
            sequence, pid = group[cursor]
            if pid in self.waiting_set and self._queue_seq.get(pid) == sequence:
                break
            cursor += 1
        cursors[index] = cursor
        return group[cursor] if cursor < len(group) else None

    def _hp_candidates(self):
        """Earliest still-waiting HP GPU task in each demand group."""
        result = []
        for index, group in enumerate(self._hp_gpu_wait_groups):
            head = self._group_head(group, self._hp_gpu_wait_group_cursors, index)
            if head is not None:
                result.append(head)
        return sorted(result)

    def _hp_first_ready(self):
        """The experimental queue rule: try every HP GPU group head before anything else.

        Placing an HP task here may evict on the spot, which is why this path and
        `_try_preempt` must not both fire for the same decision. The paper gives no
        global queue discipline, so this is a local rule and it is off by default.
        """
        for _, pid in self._hp_candidates():
            task = self.by_id[pid]
            if self._any_fits(task):
                return pid
            if self.preempt_enabled:
                candidate = self._spotgpu_best_preemption(task)
                if candidate:
                    node, victims, _ = candidate
                    self._evict(victims, node)
                    return pid
        return None

    def _next_ready(self):
        if self.hp_first_queue:
            pid = self._hp_first_ready()
            if pid is not None:
                return pid
        if self.queue_policy == "fcfs":
            pid = next(iter(self.waiting), None)
            if pid is not None:
                if self._any_fits(self.by_id[pid]):
                    return pid
                self._note_block(pid)
            return None
        max_free_cpu = max(self.free_cpu)
        candidate = self._cpu_wait_candidate(max_free_cpu)
        candidate_sequence = self._queue_seq[candidate] if candidate is not None else math.inf
        for index, group in enumerate(self._gpu_wait_groups):
            head = self._group_head(group, self._gpu_wait_group_cursors, index)
            if head is None:
                continue
            sequence, pid = head
            if sequence < candidate_sequence:
                if self._any_fits(self.by_id[pid]):
                    candidate, candidate_sequence = pid, sequence
                else:
                    # Only the earliest blocked task per group is recorded: it is the
                    # one backfill actually skipped over, and the guard keeps this
                    # O(nodes) once per task rather than once per event.
                    self._note_block(pid)
        return candidate

    def _note_block(self, pid):
        """Record the first observed blocking reason for a task, once.

        _next_ready only hands a task to the policy when it already fits, so a
        blocked task never reaches pending_id. The classification therefore has to
        happen here, on the branch where _any_fits failed; recording at the decision
        point alone would label every task "direct".
        """
        if pid not in self.first_block_class:
            self.first_block_class[pid] = self._classify_block(self.by_id[pid])

    def _classify_block(self, task):
        """Why this task cannot start at this instant. Observational only.

        Keeps the four-way split used by the archived defragmentation benchmark so
        results stay comparable, adapted to this engine: GPU eligibility is a
        whitelist rather than an exact label, and CPU-only tasks are compatible with
        every node. free_cpu already excludes the background reservation.

        This describes the current state only. Permanent impossibility is decided
        separately by _rejection_reason against empty nodes, so a task classified
        here as a capacity shortage may still be rejected later.
        """
        if task.gpu == 0:
            candidates = range(len(self.nodes))
        else:
            candidates = set()
            for model in task.allowed_gpu_types:
                candidates.update(self.model_nodes.get(model, ()))
            if not candidates:
                return "model_unavailable"
        if any(self._fits(task, j) for j in candidates):
            return "direct"
        if sum(self.free_gpu[j] for j in candidates)+EPS < task.gpu:
            return "gpu_capacity_shortage"
        if sum(self.free_cpu[j] for j in candidates)+EPS < task.cpu:
            return "cpu_capacity_shortage"
        return "fragmentation"
