"""Reactive defragmentation: evacuate a node so a blocked task can start.

This is the collaborator's benchmark engine's core mechanism, ported into the main
engine so that the two stop being two. Her `engine.py` keeps its own occupancy model
and recomputes everything at every event; here the same policy runs on the main
engine's incremental ledger, its heap, its queue index and its metrics.

Semantics preserved from her engine, because the acceptance test is reproducing her
archived `baseline_results.csv` and not the semantics means failing M10:

- trigger: the earliest waiting task whose blocking reason is `fragmentation`
  (not capacity shortage, not model unavailability) - the classifier in queues.py
  supplies exactly that four-way split;
- plan: choose a target node, try subsets of its migratable pods (bounded by
  `max_plan_moves` and by 12 pods on the target), greedily assign each evacuated pod
  to the best-fit destination, and score a plan by total GPU-seconds paused;
- execution is serial, one move in flight, and admission is frozen for the whole
  plan;
- a move reserves the destination immediately while the source stays occupied, so the
  pod is double-booked for the pause. That is make-before-break with the cost paid in
  capacity, and it is what `destination_reserved_before_source_release` means.

The four switches below are M11. Their defaults are her engine's behaviour, so the
default configuration is the configuration her results were produced with.

    migration_mode                     "none" by default: nothing here runs unless
                                        a caller asks for it
    reserve_destination       True     destination is booked at move start
    release_source_when       "end"    source is freed only when the move lands
    pause_service             True     the pause extends the task's finish time
    block_admission_during_plan True   no admissions while a plan is executing
"""
import itertools
from collections import defaultdict

from .contract import EPS


class MigrationMixin:
    def blocked_by_migration(self):
        """Admission is frozen for the whole plan, not just per move."""
        return (self.migration_mode != "none" and self.block_admission_during_plan
                and self.migration_operation is not None)

    def try_defragment(self):
        if self.migration_mode == "none":
            return False
        # One plan at a time. Without this guard the decision loop re-enters here on
        # every iteration while the plan is still executing, and each entry builds a
        # fresh plan and starts another move for the same pod, overwriting the
        # reservation and double-releasing the source. Her engine has the same rule:
        # the plan block sits inside `if operation is None`.
        if self.migration_operation is not None:
            return False
        return self._try_defragment()

    def _migratable(self, pid):
        task = self.by_id[pid]
        value = task.metadata.get("migratable", True)
        if isinstance(value, str):
            # CSV has no booleans: "False" is a non-empty string and therefore
            # truthy, so a naive bool() would make every pod migratable.
            return value.strip().lower() not in ("", "false", "0", "no", "nan", "none")
        return bool(value)

    def _move_pause(self, task):
        return (self.migration_base_sec + self.migration_per_gpu_sec*task.gpu
                + self.migration_per_cpu_core_sec*task.cpu)

    def _evacuation_plans(self, req, limit, first_only):
        """Port of her plans(): online, computed from current resources only.

        Uses no future arrivals and no frozen witness, and failure to find a plan is
        not proof that none exists - the subset search is bounded twice over.
        """
        byhost = defaultdict(list)
        for pid, j in self.running.items():
            if self._migratable(pid):
                byhost[j].append(pid)
        server_of = [node.server_id for node in self.nodes]
        candidates = []
        for target, node in enumerate(self.nodes):
            if req.gpu > 0 and node.gpu_type not in req.allowed_gpu_types:
                continue
            free_g, free_c = self.free_gpu[target], self.free_cpu[target]
            resident = sorted(byhost.get(target, ()),
                              key=lambda pid: (-self.by_id[pid].gpu, pid))
            if len(resident) > 12:
                continue
            for size in range(1, min(limit, len(resident))+1):
                for subset in itertools.combinations(resident, size):
                    if free_g+sum(self.by_id[pid].gpu for pid in subset)+EPS < req.gpu:
                        continue
                    if free_c+sum(self.by_id[pid].cpu for pid in subset)+EPS < req.cpu:
                        continue
                    spare_g, spare_c = list(self.free_gpu), list(self.free_cpu)
                    moves, feasible = [], True
                    for pid in subset:
                        task = self.by_id[pid]
                        options = [j for j in range(len(self.nodes)) if j != target
                                   and (task.gpu == 0 or self.nodes[j].gpu_type in task.allowed_gpu_types)
                                   and spare_g[j]+EPS >= task.gpu and spare_c[j]+EPS >= task.cpu]
                        if not options:
                            feasible = False
                            break
                        j = min(options, key=lambda k: (spare_g[k]-task.gpu, spare_c[k]-task.cpu,
                                                        server_of[k]))
                        spare_g[j] -= task.gpu
                        spare_c[j] -= task.cpu
                        spare_g[target] += task.gpu
                        spare_c[target] += task.cpu
                        moves.append({"pod_id": pid, "source": server_of[target],
                                      "destination": server_of[j],
                                      "pause_sec": self._move_pause(task), "gpu": task.gpu})
                    if not feasible:
                        continue
                    candidates.append({"target": target, "target_id": server_of[target],
                                       "moves": moves,
                                       "gpu_pause_sec": sum(m["pause_sec"]*m["gpu"] for m in moves)})
                    if first_only:
                        # Her first_multi returns the first plan the enumeration
                        # reaches, not the cheapest: same search, no ordering. That
                        # is a different policy, not a faster version of the same one.
                        return candidates
        return sorted(candidates, key=lambda c: (c["gpu_pause_sec"], len(c["moves"]),
                                                 c["target_id"], str(c["moves"])))

    def _try_defragment(self):
        """Start the cheapest evacuation plan for the earliest fragmented request."""
        if self.migration_moves_done >= self.migration_max_total_moves:
            return False
        for pid in list(self.waiting):
            task = self.by_id[pid]
            if self._classify_block(task) != "fragmentation":
                continue
            # one_move caps a plan at a single evacuation; first_multi stops the
            # search at the first plan found. Both are her engine's rules.
            limit = 1 if self.migration_mode == "one_move" else self.migration_max_plan_moves
            plans = self._evacuation_plans(task, min(limit, self.migration_max_total_moves
                                                   - self.migration_moves_done),
                                           self.migration_mode == "first_multi")
            if not plans:
                continue
            self.migration_plans_considered += 1
            self.migration_operation = {"pod_id": pid, "moves": list(plans[0]["moves"])}
            self._begin_next_move()
            return True
        return False

    def _begin_next_move(self):
        """Launch the next move of the plan, skipping pods that finished meanwhile."""
        while self.migration_operation["moves"]:
            move = self.migration_operation["moves"].pop(0)
            pid = move["pod_id"]
            if pid not in self.running:
                self._record("planned_move_skipped_completed", pid)
                continue
            task = self.by_id[pid]
            source = self.running[pid]
            destination = [j for j, node in enumerate(self.nodes)
                           if node.server_id == move["destination"]][0]
            pause = self._move_pause(task)
            if self.reserve_destination:
                # Double-book: the destination is held while the source is still
                # occupied, so no other task can be admitted into either slot.
                self.free_gpu[destination] -= task.gpu
                self.free_cpu[destination] -= task.cpu
                self.node_gpu_used[destination] += task.gpu
                self.node_cpu_used[destination] += task.cpu
                self.migration_reserved = (pid, destination)
            if self.release_source_when == "start":
                # Break before make: the source is given up at the start of the move,
                # so for the duration of the pause the pod occupies nothing at all and
                # its work is stalled. Kept as an option because it is the other half
                # of the question M11 asks, not because it is the default.
                self.release(pid, source)
                self.running[pid] = destination
            self.migration_current = {"pod_id": pid, "source": source,
                                      "destination": destination, "pause_sec": pause}
            self.migration_moves_done += 1
            self.migration_gpu_units_moved += task.gpu
            self.migration_gpu_pause_sec += pause*task.gpu
            self._push(self.current_time+pause, 0, "migration_complete", pid)
            if self.pause_service:
                # Extend the task's *current* finish, not its original one. A pod
                # moved twice must pay two pauses: recomputing from
                # running_start + service - preserved silently drops the first,
                # which is what made three pods in S1/cheap_multi finish exactly one
                # pause earlier than the benchmark engine had them.
                finish = self.finish_at[pid]
                self.finish_at[pid] = finish + pause
                self.completion_counter[pid] = self._push(finish+pause, 0, "complete", pid)
            self._record("migration_start", pid)
            self._check_resources()
            return
        self.migration_operation = None

    def _transfer_node_membership(self, pid, source, destination):
        """Move the per-node priority and spot bookkeeping along with the pod.

        `_place` and the completion branch both maintain `node_hp_running`,
        `node_lp_running` and `node_spot_pids`, and until now nothing did on a move.
        The drift is permanent and it is not cosmetic. Measured mid-run with three LP
        pods and one move completed (`tests -k spot_membership`), the truth is
        `node_lp_running == [0, 2, 1]` while the counter reads `[1, 1, 1]`, and
        `node_spot_pids` still lists the moved pod on the node it left; measured after
        everything drains, the counters do not return to zero at all. So SpotGPU's
        `co_locate` and `history` placement terms rank on figures that are wrong for
        every cell combining migration with LP/HP pods, and a later eviction of the
        phantom entry would `release(pid, source)` capacity the pod no longer holds
        there. (That last step is the reachable consequence, not a measured crash -
        nothing in the suite gets there, which is itself covered by `_evict` reading
        `self.running[pid]` instead of the caller's node.)
        """
        if source == destination:
            return
        priority = self.task_priority.get(pid)
        if priority == "HP":
            self.node_hp_running[source] -= 1
            self.node_hp_running[destination] += 1
        elif priority == "LP":
            self.node_lp_running[source] -= 1
            self.node_lp_running[destination] += 1
        if self.by_id[pid].gpu > 0 and priority == "LP":
            self.node_spot_pids[source].discard(pid)
            self.node_spot_pids[destination].add(pid)
            # Spot membership changed, so a previously impossible eviction may now be
            # possible; without this the generation guard suppresses the retry.
            self._preempt_generation += 1

    def _complete_migration(self, pid, ev_counter):
        """Land an in-flight move; return False when the event no longer refers to it.

        The caller must not record the event in that case. A queued move event
        survives the move it was created for whenever the pod is re-planned first, so
        recording it anyway puts a `migration_complete` in the trace at a moment when
        nothing completed - a phantom event, and precisely the kind that makes a
        metric derived from the event log disagree with one derived from the state.
        """
        if self.migration_current is None or self.migration_current["pod_id"] != pid:
            return False
        move = self.migration_current
        task = self.by_id[pid]
        destination = move["destination"]
        source = move["source"]
        if self.reserve_destination:
            # The reservation simply becomes the occupancy: the destination was
            # charged at move start, so charging it again here would double-count it
            # and leave the node's ledger one release short when the task finishes.
            # Her engine writes this as `used[dest] -= request` immediately followed
            # by `add(pod, dest)`, which nets to the same thing.
            self.migration_reserved = None
        else:
            self.allocate(pid, destination)
        if self.release_source_when == "end":
            self.running.pop(pid, None)
            self.release(pid, source)
        self.running[pid] = destination
        self._transfer_node_membership(pid, source, destination)
        self.records[pid]["server_id"] = self.nodes[destination].server_id
        self.migration_completed_moves += 1
        self.migration_current = None
        # No _record here: this event came out of the heap, and _process_due records
        # every event it pops. Recording in both places put two migration_complete
        # rows in the trace for one move.
        self._check_resources()
        if self.migration_operation is not None:
            self._begin_next_move()

    def _migration_metrics(self):
        """Numbers defined on her engine, kept name-for-name so results line up."""
        return {"mode": self.migration_mode,
                "count": self.migration_moves_done,
                "completed_count": self.migration_completed_moves,
                "gpu_units_moved": self.migration_gpu_units_moved,
                "gpu_pause_sec": self.migration_gpu_pause_sec,
            "gpu_pause_accrued_sec": self.migration_pause_accrued,
                "plans_started": self.migration_plans_considered,
                "inflight_at_horizon": self.migration_current is not None,
                "reserve_destination": self.reserve_destination,
                "release_source_when": self.release_source_when,
                "pause_service": self.pause_service,
                "block_admission_during_plan": self.block_admission_during_plan}
