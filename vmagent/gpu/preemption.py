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

    def _prefix_enough(self, task, gpu_available, cpu_available):
        """Whether an evacuated prefix is *done evacuating*.

        `joint` is this engine's historical rule: keep evacuating until both GPU and
        CPU are covered. `gpu` is OSDI'26 §4.2 taken literally - "the smallest prefix
        {t1..tk} such that sum(G_tj) >= G_hp" names only GPUs - so the prefix stops as
        soon as GPUs are covered.

        The two modes never differ in what may actually be placed: both refuse a node
        whose CPU cannot cover the task, because this engine models CPU and
        over-subscribing it would break the ledger invariant. The difference is in
        evacuation size and cost. Under `gpu` the scheduler evicts fewer spot tasks and
        then gives up on a node that is CPU-short; under `joint` it keeps evicting until
        the CPU is covered too, pays more preemption cost, and succeeds. On a pool where
        CPU is the binding resource - scenario F reserves CPU at 100% while 38% of the
        H20 GPUs stay unallocated - that is not a marginal distinction.
        """
        if gpu_available < task.gpu - EPS:
            return False
        return self.victim_prefix == "gpu" or cpu_available >= task.cpu - EPS

    def _spotgpu_best_preemption(self, task):
        """Best (node, victims, cost) to reclaim LP spot GPUs for an HP task, or None."""
        plans = []
        for j, node in enumerate(self.nodes):
            if not (task.gpu == 0 or node.gpu_type in task.allowed_gpu_types):
                continue
            if task.gpu > node.gpu_capacity+EPS or task.cpu > node.cpu_capacity*(1-self.background)+EPS:
                continue
            free_g, free_c = self.free_gpu[j], self.free_cpu[j]
            if free_g >= task.gpu-EPS and free_c >= task.cpu-EPS:
                continue  # fits non-preemptively; normal path handles it
            spot = sorted(self.node_spot_pids[j])
            if not spot:
                continue
            if self.migration_current is not None:
                # Evicting a pod whose move is in flight cannot be given a defensible
                # meaning here: its completion event is already queued against a node it
                # is leaving or arriving at, so it would end up in `waiting` and `running`
                # at once. "Preemption interrupts a migration" has no basis in the paper
                # and is deliberately unspecified (see this module's header), so the
                # in-flight pod is left alone rather than given a made-up semantics.
                # With the default `block_admission_during_plan=True` this cannot be
                # reached at all; it is reachable via --no-block-admission.
                spot = [pid for pid in spot if pid != self.migration_current["pod_id"]]
                if not spot:
                    continue
            if self.victim_selection == "random":
                self.rng.shuffle(spot)
            else:
                spot.sort(key=lambda pid: (self._spotgpu_victim_cost(pid), pid))
            victims, total, freed_g, freed_c = [], 0., 0., 0.
            for pid in spot:
                t = self.by_id[pid]
                victims.append(pid)
                total += self._spotgpu_victim_cost(pid)
                freed_g += t.gpu
                freed_c += t.cpu
                if self._prefix_enough(task, free_g+freed_g, free_c+freed_c):
                    if free_g+freed_g >= task.gpu-EPS and free_c+freed_c >= task.cpu-EPS:
                        plans.append((j, list(victims), total))
                    break
        if not plans:
            return None
        if self.victim_selection == "random":
            return self.rng.choice(plans)
        return min(plans, key=lambda plan: (plan[2], plan[0]))

    def _evict(self, victims, node=None):
        """Preempt running LP spot tasks and return them to the waiting queue.

        Releases each victim from `self.running[pid]`, not from the `node` the caller
        passed. The two are the same until a migration is in flight, and reading the
        argument for a pod that has already moved would release capacity on a node it
        left - inflating that node's free resources while the node actually holding the
        reservation keeps it. `_check_resources` compares the incremental totals against
        the free vectors on both sides of the ledger, so it does not catch a symmetric
        swap of the two. `node` stays as the fallback so existing callers are unchanged
        in behaviour when the bookkeeping is consistent.
        """
        for pid in victims:
            task = self.by_id[pid]
            here = self.running.get(pid, node)
            elapsed = self.current_time - self.running_start[pid]
            checkpoints = int(elapsed // self.checkpoint_interval)
            preserved_now = checkpoints * self.checkpoint_interval
            wasted = elapsed - preserved_now
            self.preserved_progress[pid] = self.preserved_progress.get(pid, 0.0) + preserved_now
            record = self.records[pid]
            record["preemption_count"] += 1
            # Counted separately from `preemption_count` because most evictions cost
            # nothing: under FCFS on scenario F, 210 of 244 evictions took a task that had
            # run 0 seconds - the admission path puts a waiting LP task on the freed
            # capacity and this path takes it back within the same timestamp - and 12 more
            # landed exactly on a checkpoint boundary. Only 22 lost any work, and they
            # account for all of `wasted_gpu_sec`. `preemptions` remains what it always
            # was so archived figures still reproduce; this is the number that says how
            # much disruption the policy actually caused.
            if wasted > EPS:
                record["costly_preemptions"] += 1
            record["wasted_sec"] += wasted
            record["running_total_sec"] += elapsed
            self.running.pop(pid, None)
            self.running_start.pop(pid, None)
            # Invalidate the pending completion event; a new one is pushed on
            # the next placement.
            self.completion_counter.pop(pid, None)
            self.release(pid, here)
            self.node_lp_running[here] -= 1
            self.node_spot_pids[here].discard(pid)
            self.node_evictions[here] += 1
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
        for _, pid in self._hp_candidates():
            task = self.by_id[pid]
            cand = self._spotgpu_best_preemption(task)
            if cand is None:
                continue
            node, victims, _ = cand
            self._evict(victims, node)
            self._place(pid, node)
            return True
        return False
