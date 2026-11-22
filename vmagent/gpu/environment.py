"""Timed CPU/GPU resource reservation simulator (Python standard library only).

The simulator used to be this one 1,009-line file. It is now six focused modules
plus this orchestrator, and the split is by *question*, not by convenience:

    contract.py     how a CSV/JSON becomes NodeSpec and TaskSpec, and what is rejected
    ledger.py       what fits on a node right now, and the reservation invariant
    queues.py       who is next, and why the task before them could not go
    clock.py        the event heap, time advance, fast vs realtime, tie-breaking
    placement.py    which feasible node a policy picks
    preemption.py   SpotGPU: who an HP task evicts and what that costs
    metrics.py      integrated areas, the event trace, and summary()

`GpuSchedEnv` below owns only the state machine that ties them together: `__init__`
validation, `reset`, the arrival/complete/reject event loop, `_place`, `step`, `run`,
and the three output surfaces (`observation`, `snapshot`, `export`). That is the part
with genuinely cross-cutting knowledge, because one transition touches the ledger,
the queue index, the heap and the record table at the same time.

Why the split exists: M10 absorbs the collaborator's defragmentation engine, and
migration needs a ledger to reuse and a place to hook placement. Neither had an
addressable home in a single file. Behaviour is unchanged by construction — the
method bodies moved verbatim, with one deliberate consolidation (the allocate/release
blocks that `_place`, `_process_due` and `_evict` each spelled out by hand are now
`ledger.allocate` / `ledger.release`, in the same operation order).
`tools/verify_golden_runs.py` re-runs the recorded scenario F results and diffs
every field to prove it.

Everything this module used to define is re-exported, so `from
vmagent.gpu.environment import GpuSchedEnv, load_workload` keeps working.
"""
import csv
import heapq
import json
import math
import operator
import random
import threading
import time
from collections import OrderedDict
from pathlib import Path

from .clock import ClockMixin
from .contract import (DEFAULT_NODES, DEFAULT_TASKS, EPS, ROOT, NodeSpec, TaskSpec,
                       load_initial_placement, load_nodes, load_workload, number,
                       optional_label, read_records, required)
from .ledger import LedgerMixin
from .metrics import MetricsMixin
from .migration import MigrationMixin
from .placement import PlacementMixin
from .preemption import PreemptionMixin
from .queues import QueueMixin
from .spotgpu import resolve


class GpuSchedEnv(LedgerMixin, QueueMixin, ClockMixin, PlacementMixin, PreemptionMixin,
                  MigrationMixin, MetricsMixin):
    """One action selects one server. reset/step expose a four-return Gym-style API.

    Tasks that cannot currently fit wait for future arrivals/completions. Invalid
    actions raise, permanently impossible tasks are rejected, and reset drains
    events until a placement decision is available. Service times are engine-only.
    """
    def __init__(self, nodes, tasks, clock="realtime", speed=1., queue_policy="backfill",
                 measurement_window=(0., 120.), max_sim_time=600., cpu_background_fraction=0., seed=42,
                 checkpoint_interval=3600., initial_placement=None,
                 migration="none", migration_base_sec=20., migration_per_gpu_sec=8.,
                 migration_per_cpu_core_sec=0.1, migration_max_plan_moves=6,
                 migration_max_total_moves=80, reserve_destination=True,
                 release_source_when="end", pause_service=True, block_admission_during_plan=True,
                 spotgpu=None):
        if clock not in ("realtime", "fast") or queue_policy not in ("backfill", "fcfs"):
            raise ValueError("Unknown clock or queue policy")
        if migration not in ("none", "one_move", "cheap_multi", "first_multi"):
            raise ValueError("Unknown migration mode")
        if release_source_when not in ("end", "start"):
            raise ValueError("release_source_when must be 'end' or 'start'")
        self.nodes, self.tasks = list(nodes), list(tasks)
        # pod_id -> server_id for tasks that are already running at t=0. Absent means
        # the engine places everything itself, which is the behaviour every scenario in
        # runs/ depends on.
        self.initial_placement = dict(initial_placement or {})
        self.cpu_only_demands = sorted({task.cpu for task in self.tasks if task.gpu == 0})
        self.cpu_demand_index = {demand: index for index, demand in enumerate(self.cpu_only_demands)}
        self.cpu_tree_size = 1
        while self.cpu_tree_size < len(self.cpu_only_demands):
            self.cpu_tree_size *= 2
        if not self.nodes or not self.tasks:
            raise ValueError("Nodes and tasks cannot be empty")
        if len({n.server_id for n in nodes}) != len(nodes) or len({t.pod_id for t in tasks}) != len(tasks):
            raise ValueError("Duplicate node or pod identity")
        if len({n.cluster_id for n in nodes}) != 1:
            raise ValueError("Node pool must use a single cluster")
        for task in tasks:
            cid = task.metadata.get("cluster_id")
            if cid and cid != self.nodes[0].cluster_id:
                raise ValueError("Task cluster_id differs from the selected node pool")
        self.clock, self.speed, self.queue_policy = clock, number(speed, "speed", True), queue_policy
        if not isinstance(measurement_window, (list, tuple)) or len(measurement_window) != 2:
            raise ValueError("measurement_window must contain [start, end]")
        self.measure_start = number(measurement_window[0], "measurement start")
        self.measure_end = number(measurement_window[1], "measurement end", True)
        if self.measure_end <= self.measure_start:
            raise ValueError("Measurement end must exceed start")
        self.max_sim_time = number(max_sim_time, "max_sim_time", True)
        self.background = number(cpu_background_fraction, "cpu_background_fraction")
        if self.background > 1:
            raise ValueError("CPU background fraction must lie in [0,1]")
        seed_number = number(seed, "seed")
        if not seed_number.is_integer():
            raise ValueError("seed must be an integer")
        self.seed = int(seed_number)
        self.checkpoint_interval = number(checkpoint_interval, "checkpoint_interval", True)
        # SpotGPU is a set of orthogonal knobs, not a family of named policies; see
        # spotgpu.py for what each one means and which paper sentence it comes from.
        self.spotgpu = resolve(spotgpu)
        self.victim_selection = self.spotgpu["victim_selection"]
        self.victim_prefix = self.spotgpu["victim_prefix"]
        self.hp_first_queue = self.spotgpu["hp_first_queue"]
        self.placement_terms = self.spotgpu["placement_terms"]
        self.placement_cpu_aware = self.spotgpu["placement_cpu_aware"]
        self.migration_mode = migration
        self.migration_base_sec = number(migration_base_sec, "migration_base_sec")
        self.migration_per_gpu_sec = number(migration_per_gpu_sec, "migration_per_gpu_sec")
        self.migration_per_cpu_core_sec = number(migration_per_cpu_core_sec,
                                                 "migration_per_cpu_core_sec")
        self.migration_max_plan_moves = int(migration_max_plan_moves)
        self.migration_max_total_moves = int(migration_max_total_moves)
        self.reserve_destination = bool(reserve_destination)
        self.release_source_when = release_source_when
        self.pause_service = bool(pause_service)
        self.block_admission_during_plan = bool(block_admission_during_plan)
        self.preempt_enabled = False
        self.total_gpu = sum(n.gpu_capacity for n in nodes)
        self.total_cpu = sum(n.cpu_capacity for n in nodes)
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.initialized = False

    def reset(self):
        with self.lock:
            if self.initialized:
                self.stop_requested.clear()
            self.current_time, self.done, self.end_reason = 0., False, None
            self.free_gpu = [n.gpu_capacity for n in self.nodes]
            self.free_cpu = [n.cpu_capacity*(1-self.background) for n in self.nodes]
            self.waiting, self.running, self.events, self.history = OrderedDict(), {}, [], []
            self.records = {t.pod_id: {"pod_id": t.pod_id, "gpu_request": t.gpu, "cpu_request_cores": t.cpu,
                            "gpu_spec_public": ",".join(t.allowed_gpu_types), "submit_time_sec": t.submit_time,
                            "service_sec": t.service_time, "priority_class": t.priority,
                            "status": "not_arrived", "start_time_sec": None,
                            "finish_time_sec": None, "wait_sec": None, "server_id": None,
                            "rejection_reason": None, "duration_match_status": t.metadata.get("duration_match_status", "unspecified"),
                            "preemption_count": 0, "wasted_sec": 0.0, "running_total_sec": 0.0}
                            for t in self.tasks}
            self.by_id = {t.pod_id: t for t in self.tasks}
            self.task_priority = {t.pod_id: t.priority for t in self.tasks}
            self.status_counts = {status: 0 for status in ("not_arrived", "waiting", "running", "completed", "rejected")}
            self.status_counts["not_arrived"] = len(self.tasks)
            self.last_recorded_status = {t.pod_id: "not_arrived" for t in self.tasks}
            self.heap, self.counter = [], 0
            for task in self.tasks:
                if task.pod_id in self.initial_placement:
                    continue  # placed by _bootstrap_initial below, never arrives
                self._push(task.submit_time, 1, "arrival", task.pod_id)
            # Incremental per-node ledgers so _check_resources stays O(nodes)
            # instead of O(nodes x running tasks) per event.
            self.node_gpu_used = [0.] * len(self.nodes)
            self.node_cpu_used = [0.] * len(self.nodes)
            self.node_running_count = [0] * len(self.nodes)
            self.gpu_area = self.cpu_area = self.queue_area = self.effective_gpu_area = 0.
            self.gpu_area_full = self.effective_gpu_area_full = 0.
            self.peak_gpu = self.peak_cpu = self.peak_queue = 0.
            self.model_nodes = {}
            for j, node in enumerate(self.nodes):
                self.model_nodes.setdefault(node.gpu_type, []).append(j)
            self.model_gpu_area = {m: 0. for m in self.model_nodes}
            self.model_cpu_area = {m: 0. for m in self.model_nodes}
            # Observational topology / fragmentation accounting. Neither feeds _fits,
            # placement scoring or rejection classification, so scheduling is unchanged.
            self.asw_nodes = {}
            for j, node in enumerate(self.nodes):
                if node.asw_id:
                    self.asw_nodes.setdefault(node.asw_id, []).append(j)
            self.partial_node_time = 0.
            self.stranded_gpu_time = 0.
            self.first_block_class = {}
            self.pending_id = None
            self._waiting_order = []
            self._gpu_wait_groups = []
            self._gpu_wait_group_index = {}
            self._gpu_wait_group_cursors = []
            self._hp_gpu_wait_groups = []
            self._hp_gpu_wait_group_index = {}
            self._hp_gpu_wait_group_cursors = []
            self.waiting_set = set()
            self._queue_seq = {}
            self._queue_seq_counter = 0
            self._cpu_wait_heaps = [[] for _ in self.cpu_only_demands]
            self._cpu_wait_tree = [math.inf] * (2*self.cpu_tree_size)
            self.node_evictions = [0] * len(self.nodes)
            self.node_hp_running = [0] * len(self.nodes)
            self.node_lp_running = [0] * len(self.nodes)
            self.node_spot_pids = [set() for _ in self.nodes]
            self._preempt_generation = 0
            self._last_preempt_generation = -1
            self.running_start = {}
            self.completion_counter = {}
            self.preserved_progress = {t.pod_id: 0.0 for t in self.tasks}
            self.finish_at = {}
            self.migration_operation = None
            self.migration_current = None
            self.migration_reserved = None
            self.migration_moves_done = 0
            self.migration_completed_moves = 0
            self.migration_gpu_units_moved = 0.
            self.migration_gpu_pause_sec = 0.
            self.migration_pause_accrued = 0.
            self.migration_plans_considered = 0
            self._bootstrap_initial()
            self.rng = random.Random(self.seed)
            self.wall_start = time.monotonic()
            self.wall_end = None
            self.initialized = True
            self._record("reset")
        self._advance_until_ready()
        return self.observation()

    def _bootstrap_initial(self):
        """Load a prescribed t=0 layout, without consulting any policy.

        This exists for the defragmentation benchmark, whose whole point is a starting
        state that is already fragmented. Routing those pods through the normal path
        would let first_fit/best_fit compact them and silently destroy the thing being
        measured, so the layout is treated as an input to validate, not a decision to
        re-make. Every check below is a contradiction between the layout and the
        workload, and raises rather than quietly re-placing the task.
        """
        if not self.initial_placement:
            return
        index_of = {node.server_id: j for j, node in enumerate(self.nodes)}
        unknown = sorted(set(self.initial_placement) - set(self.by_id))
        if unknown:
            raise ValueError("Initial placement references unknown pod_id: " + ", ".join(unknown[:3]))
        for task in self.tasks:
            pid = task.pod_id
            if pid not in self.initial_placement:
                continue
            if task.submit_time > EPS:
                raise ValueError("Initial placement task {} arrives at {} but must start at 0".format(
                    pid, task.submit_time))
            sid = self.initial_placement[pid]
            if sid not in index_of:
                raise ValueError("Initial placement references unknown server_id: " + sid)
            j = index_of[sid]
            if not self._fits(task, j):
                raise ValueError("Initial placement violates GPU model or capacity for " + pid)
            self.allocate(pid, j)
            priority = self.task_priority[pid]
            if priority == "HP":
                self.node_hp_running[j] += 1
            elif priority == "LP":
                self.node_lp_running[j] += 1
            if task.gpu > 0 and priority == "LP":
                self.node_spot_pids[j].add(pid)
                self._preempt_generation += 1
            self.running[pid] = j
            self.running_start[pid] = 0.
            record = self.records[pid]
            record.update(status="running", start_time_sec=0., wait_sec=0., server_id=sid)
            self.completion_counter[pid] = self._push(task.service_time, 0, "complete", pid)
            self.finish_at[pid] = task.service_time
        self._check_resources()

    def _process_due(self):
        """Drain every event due at the current instant, in heap order.

        Completion, arrival and rejection share one loop so that tie-breaking stays
        a property of the heap rather than of this function: see clock.py.
        """
        while self.heap and self.heap[0][0] <= self.current_time+EPS:
            _, _, ev_counter, kind, pid = heapq.heappop(self.heap)
            task, record = self.by_id[pid], self.records[pid]
            if kind == "complete":
                if self.completion_counter.get(pid) != ev_counter:
                    continue  # stale event from before a preemption
                index = self.running.pop(pid)
                # Take the start of this run segment from `running_start`, which is what
                # `_evict` also uses. Reading `record["start_time_sec"]` instead looked
                # equivalent and was not: a task whose final placement was at t=0 has
                # start_time_sec == 0.0, and `0.0 or self.current_time` yields
                # current_time, so the whole last segment was added as zero and
                # `running_total_sec` was published as 0.0 for every task that started at
                # the beginning of the episode - which is the entire warm-up cohort of
                # scenario F and every pod of an `--initial-placement` layout.
                started = self.running_start.pop(pid, self.current_time)
                self.release(pid, index)
                if self.task_priority.get(pid) == "HP":
                    self.node_hp_running[index] -= 1
                elif self.task_priority.get(pid) == "LP":
                    self.node_lp_running[index] -= 1
                if task.gpu > 0 and self.task_priority.get(pid) == "LP":
                    self.node_spot_pids[index].discard(pid)
                record["running_total_sec"] += self.current_time - started
                record.update(status="completed", finish_time_sec=self.current_time)
                self._preempt_generation += 1
            elif kind == "migration_complete":
                if not self._complete_migration(pid, ev_counter):
                    continue  # stale: the pod was re-planned before this move landed
            elif not any(self._fits(task, j, empty=True) for j in range(len(self.nodes))):
                record.update(status="rejected", rejection_reason=self._rejection_reason(task))
                kind = "reject"
            else:
                self._enqueue(pid)
                record["status"] = "waiting"
            self._record(kind, pid)
        self._check_resources()

    def _place(self, pid, action):
        """Place a waiting task on a node (engine-internal, used by step and SpotGPU).

        `remaining` subtracts preserved_progress, so a task that was preempted and
        checkpointed resumes instead of restarting; that subtraction is the whole
        difference between SpotGPU resume semantics and a plain requeue.
        """
        task = self.by_id[pid]
        remaining = task.service_time - self.preserved_progress.get(pid, 0.0)
        self.allocate(pid, action)
        priority = self.task_priority.get(pid, "Other")
        if priority == "HP":
            self.node_hp_running[action] += 1
        elif priority == "LP":
            self.node_lp_running[action] += 1
        if task.gpu > 0 and priority == "LP":
            self.node_spot_pids[action].add(pid)
            self._preempt_generation += 1
        del self.waiting[pid]
        self.waiting_set.discard(pid)
        self.running[pid] = action
        self.running_start[pid] = self.current_time
        record = self.records[pid]
        record.update(status="running", start_time_sec=self.current_time,
                      wait_sec=max(0., self.current_time-task.submit_time-record["running_total_sec"]),
                      server_id=self.nodes[action].server_id)
        self.completion_counter[pid] = self._push(self.current_time+remaining, 0, "complete", pid)
        self.finish_at[pid] = self.current_time + remaining
        self._record("start", pid)
        self._check_resources()

    def step(self, action):
        if not self.initialized or self.done:
            raise RuntimeError("Call reset before step; cannot step a finished episode")
        # Real scheduling computation consumes wall time; release due tasks first.
        if self.clock == "realtime":
            self._move_to(self.current_time)
        with self.lock:
            self._process_due()
            if self.current_time >= self.max_sim_time:
                self._finish("max_sim_time")
                return self.observation(), 0., True, {"end_reason": self.end_reason}
            if self.stop_requested.is_set():
                self._finish("cancelled")
                return self.observation(), 0., True, {"end_reason": self.end_reason}
            pid = self.pending_id
            task = self.by_id[pid]
            try:
                if isinstance(action, bool):
                    raise TypeError
                action = operator.index(action)
            except TypeError:
                raise ValueError("Action must be an integer server index")
            if not 0 <= action < len(self.nodes):
                raise ValueError("Action must be an integer server index")
            if not self._fits(task, action):
                raise ValueError("Action violates GPU model or resource capacity")
            self._place(pid, action)
        self._advance_until_ready()
        return self.observation(), 1., self.done, {"scheduled_pod_id": pid, "end_reason": self.end_reason}

    def run(self, policy="first_fit"):
        # Preemption is a property of the policy being run, not of the object. It used
        # to be set only on the spotgpu branch, so an environment reused for a second
        # run kept `preempt_enabled=True` and `first_fit` silently evicted LP tasks:
        # measured on a one-node case, preemptions 0 -> 1 and reserved_gpu_ratio
        # 0.350 -> 0.366667 for a run whose policy never preempts. Nothing in the
        # recorded pipelines reuses an environment, which is why no archived number
        # moves - but the Gym-style reset()/step() API and `schedgym/gpu_env.py` both
        # invite exactly that reuse. Assign unconditionally, on both branches.
        self.preempt_enabled = policy == "spotgpu" and self.spotgpu["preemption"]
        if policy == "spotgpu":
            self.reset()
            while not self.done:
                self.step(self._spotgpu_choose_action())
            return self.summary()
        self.reset()
        while not self.done:
            self.step(self.choose_action(policy))
        return self.summary()

    def observation(self):
        with self.lock:
            task = self.by_id[self.pending_id] if self.pending_id else None
            return {"time_sec": self.current_time,
                    "obs": [[self.free_gpu[j]/n.gpu_capacity, self.free_cpu[j]/n.cpu_capacity] for j, n in enumerate(self.nodes)],
                    "feat": [task.gpu, task.cpu] if task else [0., 0.],
                    "avail": [int(task is not None and self._fits(task, j)) for j in range(len(self.nodes))],
                    "node_ids": [n.server_id for n in self.nodes], "gpu_models": [n.gpu_type for n in self.nodes],
                    "task": {"pod_id": task.pod_id, "gpu_request": task.gpu, "cpu_request_cores": task.cpu,
                             "allowed_gpu_types": list(task.allowed_gpu_types)} if task else None,
                    "queue_length": len(self.waiting), "done": self.done}

    def get_attr(self, name):
        obs = self.observation()
        return obs[{"req": "feat"}.get(name, name)]

    def snapshot(self):
        with self.lock:
            records = list(self.records.values())
            truncated = len(records) > 2000
            return {"summary": self.summary(), "nodes": [
                {"server_id": n.server_id, "gpu_type": n.gpu_type, "gpu_capacity": n.gpu_capacity,
                 "cpu_capacity_cores": n.cpu_capacity, "free_gpu": max(0., self.free_gpu[j]),
                 "free_cpu_cores": max(0., self.free_cpu[j]), "cpu_background_cores": n.cpu_capacity*self.background}
                for j, n in enumerate(self.nodes)], "tasks": records[-2000:] if truncated else records,
                "tasks_truncated": truncated, "tasks_total": len(records),
                "history": [dict(row) for row in self.history[-2000:]], "events": [dict(e) for e in self.events[-300:]]}

    def export(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock:
            summary = self.summary()
            (folder/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            for filename, rows in [("tasks.csv", list(self.records.values())), ("events.csv", self.events),
                                   ("resource_history.csv", self.history), ("nodes_final.csv", self.snapshot()["nodes"])]:
                if rows:
                    with (folder/filename).open("w", encoding="utf-8-sig", newline="") as output:
                        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            return summary


__all__ = ["GpuSchedEnv", "NodeSpec", "TaskSpec", "EPS", "ROOT", "DEFAULT_NODES", "DEFAULT_TASKS",
           "number", "required", "optional_label", "read_records", "load_nodes", "load_workload"]
