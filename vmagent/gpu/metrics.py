"""Metrics: time-integrated areas, the event trace, and the run summary.

Everything reported about a run is produced here. Two families of numbers, and they
answer different questions:

- **reserved_*_ratio** integrates what the ledger actually held over time, using the
  measurement window and, for incomplete runs, only the covered part as denominator.
- **offered_*_ratio** integrates what the workload *asked for* assuming every task ran
  the moment it submitted. It can exceed 100% and is not a utilisation figure.

Neither is GPU SM utilisation nor CPU execution utilisation: the engine models
reservation, not execution. That caveat is repeated inside `summary()` under an
explicit key because the distinction is the single most misread thing about this
project's output.

`_integrate_to` is called from the clock on every advance, and `_record` from every
state transition; both must stay cheap, so the per-node loops here are the reason
the ledger is incremental rather than recomputed.
"""
import math
import time

from .contract import EPS


def interval_overlap(start_a, end_a, start_b, end_b):
    """Length of the intersection of two half-open intervals, never negative."""
    return max(0., min(end_a, end_b) - max(start_a, start_b))


def run_segments(events, start_kind="start", end_kinds=("complete", "preempt")):
    """Rebuild each pod's occupied intervals from the event log, preemption-aware.

    A task that is preempted and restarted occupies two disjoint intervals, and a
    migration moves it between them. Anything derived from *when a pod held
    resources* therefore has to come from here rather than from `records[pid]`,
    whose `server_id` and `start_time_sec` are overwritten by exactly those events.
    The engine integrates its own ledger incrementally and so does not need this;
    external re-derivations do, and `tools/audit_metric_derivations.py` measures how
    badly the shortcut fails without it.

    A pod still running at the end of the episode is closed at `env.current_time`, so
    the segments cover the whole run and the integral is comparable to the engine's.
    """
    segments = {}
    open_at = {}
    if not events:
        return segments
    horizon = events[-1]["time_sec"]
    for event in events:
        pid, kind, when = event["pod_id"], event["event"], event["time_sec"]
        if pid is None:
            continue
        if kind == start_kind:
            open_at[pid] = when
        elif kind in end_kinds and pid in open_at:
            segments.setdefault(pid, []).append((open_at.pop(pid), when, kind))
    for pid, when in open_at.items():
        segments.setdefault(pid, []).append((when, horizon, "open"))
    return segments


def integrate_events(events, field, window=None):
    """Area under one event-carried aggregate counter, in unit-seconds.

    `_record` snapshots the whole ledger at every state transition, so the column is
    a step function that holds from one event's timestamp to the next. This is the
    cheap, correct way to turn it back into GPU-seconds or core-seconds: the value was
    read when it was true, so rewriting per-task state afterwards cannot corrupt it.
    The final event contributes nothing, matching the engine, which stops integrating
    at the last transition.

    With `window=(start, end)` only the covered part counts, which is what
    `env.gpu_area` and `env.cpu_area` hold; without it the whole episode is covered,
    which is what `env.gpu_area_full` holds. Comparing one against the other produced
    a spurious 0.5% CPU gap on a scenario F slice whose window ended well before the
    run drained, so pick the basis that matches and not whichever is nearer to hand.
    """
    area = 0.
    for index in range(len(events)-1):
        left, right = events[index]["time_sec"], events[index+1]["time_sec"]
        if window is not None:
            left = max(left, window[0])
            right = min(right, window[1])
        if right > left:
            area += events[index][field]*(right-left)
    return area


def percentile_nearest_rank(ordered, fraction):
    """The p95 this project reports by default.

    `ordered` must already be sorted; the index is the classic lower nearest-rank
    rule, so p95 of [1, 2, 3] is 3, not an interpolated value. Used by the engine's
    summary and by the archived preemption experiment suite.
    """
    if not ordered:
        return 0.
    return ordered[max(0, math.ceil(fraction*len(ordered))-1)]


def percentile_linear(values, fraction):
    """numpy.quantile(..., method='linear'), reimplemented so nothing here needs numpy.

    This is *not* the same number as percentile_nearest_rank and the two must never
    be mixed in one table: it interpolates between neighbours, so p95 of [1, 2, 3]
    is 2.9. It exists because the defragmentation benchmark's engine calls
    numpy.quantile, and reproducing its p95 requires reproducing its definition.
    """
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered)-1)*fraction
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


class MetricsMixin:
    def _integrate_to(self, target):
        if target < self.current_time-EPS:
            raise RuntimeError("Clock moved backwards")
        overlap = max(0., min(target, self.measure_end)-max(self.current_time, self.measure_start))
        reserved = self.total_gpu-sum(self.free_gpu)
        # Effective allocation discounts a pod that is mid-migration: her benchmark
        # defines A_GPU as "computing allocated GPU", and a pod whose service is
        # paused produces nothing. Discount it by however many times the ledger is
        # currently counting it - two under make-before-break, where the source is
        # still held and the destination is reserved, zero under a full
        # break-before-make. Subtracting one copy, which this did first, leaves the
        # pod counted once throughout its pause and reads A_GPU too high by exactly
        # the migrated GPU-seconds.
        paused = 0.
        if self.migration_current is not None:
            in_flight_gpu = self.by_id[self.migration_current["pod_id"]].gpu
            counted = (1 if self.reserve_destination else 0) \
                + (1 if self.release_source_when == "end" else 0)
            paused = in_flight_gpu*counted
            # Time-integrated, unlike gpu_pause_sec which is the planned figure booked
            # at move start. They agree for every completed move and differ only when
            # a move is still in flight at the horizon, which is the case her
            # benchmark reports separately and a reader comparing the two engines
            # needs to be able to see.
            self.migration_pause_accrued += in_flight_gpu*max(0., target-self.current_time)
        self.gpu_area += reserved*overlap
        self.effective_gpu_area += max(0., reserved-paused)*overlap
        # The benchmark also reports the same integral over the whole horizon rather
        # than over its allocation window, so keep an unclipped pair as well.
        span = max(0., target-self.current_time)
        self.gpu_area_full += reserved*span
        self.effective_gpu_area_full += max(0., reserved-paused)*span
        self.cpu_area += (self.total_cpu*(1-self.background)-sum(self.free_cpu))*overlap
        for model, idxs in self.model_nodes.items():
            self.model_gpu_area[model] += sum(self.nodes[j].gpu_capacity-self.free_gpu[j] for j in idxs)*overlap
            self.model_cpu_area[model] += sum(self.nodes[j].cpu_capacity*(1-self.background)-self.free_cpu[j] for j in idxs)*overlap
        self.queue_area += len(self.waiting)*overlap
        partial = 0
        stranded = 0.
        for j, node in enumerate(self.nodes):
            used = node.gpu_capacity-self.free_gpu[j]
            if EPS < used < node.gpu_capacity-EPS:
                partial += 1
                stranded += node.gpu_capacity-used
        self.partial_node_time += partial*overlap
        self.stranded_gpu_time += stranded*overlap
        self.current_time = target

    def _record(self, kind, pod_id=None):
        if pod_id is not None:
            previous = self.last_recorded_status[pod_id]
            current = self.records[pod_id]["status"]
            if current != previous:
                self.status_counts[previous] -= 1
                self.status_counts[current] += 1
                self.last_recorded_status[pod_id] = current
        gpu = self.total_gpu-sum(self.free_gpu)
        foreground_cpu = self.total_cpu*(1-self.background)-sum(self.free_cpu)
        self.peak_gpu = max(self.peak_gpu, gpu)
        self.peak_cpu = max(self.peak_cpu, foreground_cpu+self.total_cpu*self.background)
        self.peak_queue = max(self.peak_queue, len(self.waiting))
        counts = self._counts()
        event = {"time_sec": self.current_time, "event": kind, "pod_id": pod_id,
                 "gpu_reserved": max(0., gpu), "cpu_reserved_cores": max(0., foreground_cpu)+self.total_cpu*self.background,
                 "waiting": len(self.waiting), "running": len(self.running), "completed": counts["completed"],
                 "rejected": counts["rejected"]}
        self.events.append(event)
        self.history.append(dict(event))

    def _counts(self):
        return dict(self.status_counts)

    def _fragmentation_metrics(self):
        """Stranded-capacity observation: how much node-time sits partially used.

        A node counts as partially occupied when at least one GPU is in use but the
        node is not saturated. Fully idle nodes are excluded: they are idle capacity,
        not fragmentation. This is the quantity IPC is reported against ("reduces the
        number of not-fully-occupied nodes by 20.2%"), so it is recorded even when no
        defragmentation policy is active.
        """
        idle = partial = full = 0
        for j, node in enumerate(self.nodes):
            used = node.gpu_capacity-self.free_gpu[j]
            if used <= EPS:
                idle += 1
            elif used >= node.gpu_capacity-EPS:
                full += 1
            else:
                partial += 1
        return {"node_count": len(self.nodes), "nodes_idle_end": idle,
                "nodes_partially_occupied_end": partial, "nodes_fully_occupied_end": full,
                "partial_node_time_sec": self.partial_node_time,
                "stranded_gpu_time_sec": self.stranded_gpu_time}

    def _topology_metrics(self):
        """GPU entropy of workload placement over ASW domains (observational only).

        Entropy is a property of a group: in the ASI trace one multi-node job appears
        as several pods sharing a workload_id, and a single pod always lives in one
        node hence one ASW. So this metric is only meaningful when the input carries
        workload_id on tasks and asw_id on nodes; coverage counts are reported so the
        reader can see how much of the load the number actually covers.

        Placement means the final assigned server per pod. It is not a simultaneous
        co-residency measure, and it does not claim the pods ran at the same instant.
        Logarithm base is natural (nats); the field name states it explicitly.
        """
        index_of = {node.server_id: j for j, node in enumerate(self.nodes)}
        groups = {}
        for pid, task in self.by_id.items():
            if not task.workload_id:
                continue
            sid = self.records[pid]["server_id"]
            if sid is None:
                continue
            j = index_of.get(sid)
            if j is None:
                continue
            groups.setdefault(task.workload_id, []).append((self.nodes[j].asw_id, task.gpu))
        spanned, entropies, multi_asw_gpu, group_gpu = {}, [], 0., 0.
        unknown_asw_gpu = 0.
        for workload, placements in groups.items():
            total = sum(gpu for _, gpu in placements)
            if total <= EPS:
                continue
            by_asw = {}
            for asw, gpu in placements:
                if asw:
                    by_asw[asw] = by_asw.get(asw, 0.)+gpu
                else:
                    unknown_asw_gpu += gpu
            if not by_asw:
                continue
            known = sum(by_asw.values())
            n_asw = len(by_asw)
            spanned[n_asw] = spanned.get(n_asw, 0)+1
            if n_asw > 1:
                multi_asw_gpu += known
            group_gpu += known
            entropies.append(-sum((gpu/known)*math.log(gpu/known) for gpu in by_asw.values()))
        entropies.sort()
        labeled = sum(1 for task in self.tasks if task.workload_id)
        return {"nodes_with_asw": sum(1 for node in self.nodes if node.asw_id),
                "tasks_with_workload_id": labeled, "task_count": len(self.tasks),
                "workloads_placed": len(groups), "workloads_with_known_gpu_total": len(entropies),
                "asw_span_histogram": {str(k): v for k, v in sorted(spanned.items())},
                "workload_gpu_sec_in_multi_asw_workloads": multi_asw_gpu,
                "workload_gpu_in_measurable_groups": group_gpu,
                "gpu_on_nodes_without_asw": unknown_asw_gpu,
                "gpu_entropy_nats_mean": (sum(entropies)/len(entropies)) if entropies else 0.,
                "gpu_entropy_nats_p95": percentile_nearest_rank(entropies, .95),
                "gpu_entropy_nats_max": entropies[-1] if entropies else 0.,
                "topology_metrics_are_final_placement_not_coresidency": True}

    def _block_classification(self):
        """Aggregate of the first observed blocking reason per task that reached a
        decision point. Tasks never reached (still waiting at the horizon, or
        permanently rejected before any decision) are reported as uncovered rather
        than silently omitted."""
        counts = {}
        for reason in self.first_block_class.values():
            counts[reason] = counts.get(reason, 0)+1
        for reason in ("direct", "model_unavailable", "gpu_capacity_shortage",
                       "cpu_capacity_shortage", "fragmentation"):
            counts.setdefault(reason, 0)
        classified = len(self.first_block_class)
        return {"first_attempt_classes": counts, "tasks_classified": classified,
                "task_count": len(self.tasks),
                "tasks_never_reaching_a_decision": len(self.tasks)-classified,
                "fragmentation_share_of_classified": (counts["fragmentation"]/classified
                                                      if classified else 0.),
                "classification_is_instant_state_not_permanence": True}

    def summary(self):
        with self.lock:
            duration = self.measure_end-self.measure_start
            counts = self._counts()
            waits = sorted(r["wait_sec"] for r in self.records.values() if r["wait_sec"] is not None)
            offered_gpu = offered_cpu = 0.
            for task in self.tasks:
                overlap = max(0., min(self.measure_end, task.submit_time+task.service_time)-max(self.measure_start, task.submit_time))
                offered_gpu += task.gpu*overlap
                offered_cpu += task.cpu*overlap
            # Snapshot includes reservations during current wall time, even between events.
            extra = 0.
            if self.clock == "realtime" and not self.done:
                now = min(self.max_sim_time, max(self.current_time, (time.monotonic()-self.wall_start)*self.speed))
                extra = max(0., min(now, self.measure_end)-max(self.current_time, self.measure_start))
            gpu_area = self.gpu_area+(self.total_gpu-sum(self.free_gpu))*extra
            cpu_area = self.cpu_area+(self.total_cpu*(1-self.background)-sum(self.free_cpu))*extra
            end = self.current_time if self.done else min(self.max_sim_time, (time.monotonic()-self.wall_start)*self.speed) if self.clock == "realtime" else self.current_time
            measurement_covered = max(0., min(end, self.measure_end)-self.measure_start)
            # Completed runs have zero foreground after draining; incomplete runs are partial.
            denominator_sec = duration if self.end_reason == "drained" else measurement_covered
            per_model = {}
            for model, idxs in self.model_nodes.items():
                cap_g = sum(self.nodes[j].gpu_capacity for j in idxs)
                cap_c = sum(self.nodes[j].cpu_capacity for j in idxs)
                offered = sum(t.gpu*max(0., min(self.measure_end, t.submit_time+t.service_time)
                                        - max(self.measure_start, t.submit_time))
                              for t in self.tasks if t.gpu > 0 and t.allowed_gpu_types == (model,))
                per_model[model] = {
                    "gpu_capacity": cap_g, "cpu_capacity_cores": cap_c,
                    "offered_gpu_ratio": offered/(cap_g*duration) if cap_g else 0.,
                    "reserved_gpu_ratio": (self.model_gpu_area.get(model, 0.)/(cap_g*denominator_sec)
                                           if cap_g*denominator_sec else 0.),
                    "reserved_cpu_ratio": (self.model_cpu_area.get(model, 0.)/(cap_c*denominator_sec)+self.background
                                           if cap_c*denominator_sec else 0.)}
            return {"task_count": len(self.tasks), "node_count": len(self.nodes), "gpu_capacity": self.total_gpu,
                    "cpu_capacity_cores": self.total_cpu, **counts,
                    "unfinished": counts["not_arrived"]+counts["waiting"]+counts["running"],
                    "sim_time_sec": end, "wall_time_sec": (self.wall_end if self.wall_end is not None else time.monotonic())-self.wall_start,
                    "clock": self.clock, "speed": self.speed, "end_reason": self.end_reason,
                    "measurement_window_sec": [self.measure_start, self.measure_end],
                    "measurement_denominator_sec": denominator_sec,
                    "offered_gpu_ratio": offered_gpu/(self.total_gpu*duration),
                    "offered_cpu_ratio": offered_cpu/(self.total_cpu*duration)+self.background,
                    "reserved_gpu_ratio": gpu_area/(self.total_gpu*denominator_sec) if denominator_sec else 0.,
                    "reserved_cpu_ratio": cpu_area/(self.total_cpu*denominator_sec)+self.background if denominator_sec else 0.,
                    # Her benchmark's A_GPU: allocated GPU with a mid-migration pod
                    # discounted, because its service is paused while it holds two
                    # slots. Equal to reserved_gpu_ratio whenever nothing is migrating,
                    # which is why the two only diverge on the migration cells.
                    "effective_gpu_ratio": (self.effective_gpu_area
                                            / (self.total_gpu*denominator_sec) if denominator_sec else 0.),
                    # Full-horizon, not full-episode: her denominator is the horizon
                    # itself, so a run that drains early still divides by the whole
                    # horizon. Dividing by the last simulated instant instead inflates
                    # this ratio, which is how the first version of this field came out
                    # 0.11 too high on a 10,800 s benchmark instance.
                    "effective_gpu_ratio_full_horizon": (self.effective_gpu_area_full
                                                         / (self.total_gpu*self.max_sim_time)
                                                         if self.total_gpu else 0.),
                    "per_model": per_model,
                    "preemptions": sum(r["preemption_count"] for r in self.records.values()),
                    # See preemption.py: an eviction that cost the victim nothing still
                    # counts above. This one does not, and on scenario F the two differ by
                    # an order of magnitude (244 vs 22 under FCFS).
                    "preemptions_with_loss": sum(r["costly_preemptions"] for r in self.records.values()),
                    "preempted_task_count": sum(1 for r in self.records.values() if r["preemption_count"] > 0),
                    "disturbed_task_count": sum(1 for r in self.records.values() if r["costly_preemptions"] > 0),
                    "wasted_gpu_sec": sum(r["wasted_sec"]*t.gpu for r, t in zip(self.records.values(), self.tasks)),
                    "mean_wait_sec_started": sum(waits)/len(waits) if waits else 0.,
                    "p95_wait_sec_started": percentile_nearest_rank(waits, .95),
                    "peak_queue_length": int(self.peak_queue),
                    "peak_gpu_reserved": self.peak_gpu, "peak_cpu_reserved_cores": self.peak_cpu,
                    "cpu_background_fraction": self.background,
                    "fragmentation": dict(self._fragmentation_metrics(),
                        partial_node_time_ratio=(self.partial_node_time/(len(self.nodes)*denominator_sec)
                                                 if denominator_sec else 0.),
                        stranded_gpu_time_ratio=(self.stranded_gpu_time/(self.total_gpu*denominator_sec)
                                                 if self.total_gpu and denominator_sec else 0.)),
                    "topology": self._topology_metrics(),
                    "block_classification": self._block_classification(),
                    "migration": self._migration_metrics(),
                    "reservation_metrics_are_not_SM_or_CPU_execution_utilization": True}
