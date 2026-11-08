"""Timed CPU/GPU resource reservation simulator (Python standard library only)."""
import csv
import heapq
import io
import json
import math
import operator
import random
import threading
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-9
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NODES = ROOT / "docs/server_selection/servers_20_H20.csv"
DEFAULT_TASKS = ROOT / "docs/server_selection/real_pod_sequences/scene_A_timing/scene_A_50_pods_timed.json"


def number(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError("{} must be a finite number".format(name)) from None
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError("{} must be {}".format(name, "positive and finite" if positive else "nonnegative and finite"))
    return result


def required(row, key):
    value = str(row.get(key, "")).strip()
    if not value or row.get(key) is None:
        raise ValueError("Missing {}".format(key))
    return value


def read_records(path, key):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, (list, dict)):
            raise ValueError("JSON input must be an array or an object")
        rows = data if isinstance(data, list) else data.get(key)
        metadata = data if isinstance(data, dict) else {}
    elif path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        metadata = {}
    else:
        raise ValueError("Only UTF-8 CSV and JSON inputs are supported")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("{} must contain a nonempty list of records".format(key))
    return rows, metadata


@dataclass(frozen=True)
class NodeSpec:
    server_id: str
    cluster_id: str
    gpu_type: str
    gpu_capacity: float
    cpu_capacity: float


@dataclass(frozen=True)
class TaskSpec:
    pod_id: str
    gpu: float
    cpu: float
    allowed_gpu_types: tuple
    submit_time: float
    service_time: float
    priority: str = "Other"
    metadata: dict = field(default_factory=dict, compare=False)


def load_nodes(path):
    rows, _ = read_records(path, "nodes")
    result, seen = [], set()
    for row in rows:
        sid = required(row, "server_id")
        if sid in seen:
            raise ValueError("Duplicate server_id: " + sid)
        seen.add(sid)
        result.append(NodeSpec(sid, required(row, "cluster_id"), required(row, "gpu_spec_public"),
                               number(row.get("gpu_count"), "gpu_count", True),
                               number(row.get("cpu_capacity_cores"), "cpu_capacity_cores", True)))
    if len({node.cluster_id for node in result}) != 1:
        raise ValueError("All servers must belong to the same cluster_id")
    return result


def load_workload(path):
    rows, metadata = read_records(path, "tasks")
    result, seen = [], set()
    for row in rows:
        pid = required(row, "pod_id")
        if pid in seen:
            raise ValueError("Duplicate pod_id: " + pid)
        seen.add(pid)
        gpu = number(row.get("gpu_request"), "gpu_request")
        allowed = row.get("allowed_gpu_types", row.get("gpu_spec_public", []))
        if isinstance(allowed, str):
            stripped = allowed.strip()
            if stripped.startswith("["):
                allowed = json.loads(stripped)
            elif stripped.lower() in ("", "nan", "none", "null"):
                allowed = []
            else:
                allowed = [stripped]
        if not isinstance(allowed, (list, tuple)) or any(not isinstance(x, str) or not x.strip() for x in allowed):
            raise ValueError("allowed_gpu_types must be a list of nonempty model labels")
        if gpu > 0 and not allowed:
            raise ValueError("GPU tasks must specify gpu_spec_public or allowed_gpu_types")
        priority = str(row.get("priority_class") or "Other").strip().upper()
        if priority not in ("HP", "LP", "OTHER"):
            priority = "OTHER"
        if priority == "OTHER":
            priority = "Other"
        result.append(TaskSpec(pid, gpu, number(row.get("cpu_request_cores"), "cpu_request_cores"),
                               tuple(allowed), number(row.get("submit_time_sec"), "submit_time_sec"),
                               number(row.get("service_sec"), "service_sec", True),
                               priority, dict(row)))
    result.sort(key=lambda task: (task.submit_time, task.pod_id))
    return result, {key: value for key, value in metadata.items() if key != "tasks"}


class GpuSchedEnv:
    """One action selects one server. reset/step expose a four-return Gym-style API.

    Tasks that cannot currently fit wait for future arrivals/completions. Invalid
    actions raise, permanently impossible tasks are rejected, and reset drains
    events until a placement decision is available. Service times are engine-only.
    """
    def __init__(self, nodes, tasks, clock="realtime", speed=1., queue_policy="backfill",
                 measurement_window=(0., 120.), max_sim_time=600., cpu_background_fraction=0., seed=42,
                 checkpoint_interval=3600.):
        if clock not in ("realtime", "fast") or queue_policy not in ("backfill", "fcfs"):
            raise ValueError("Unknown clock or queue policy")
        self.nodes, self.tasks = list(nodes), list(tasks)
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
        self.preempt_enabled = False
        self.total_gpu = sum(n.gpu_capacity for n in nodes)
        self.total_cpu = sum(n.cpu_capacity for n in nodes)
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.initialized = False

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

    def _rejection_reason(self, task):
        compatible = [n for n in self.nodes if task.gpu == 0 or n.gpu_type in task.allowed_gpu_types]
        if not compatible:
            return "gpu_model_unavailable"
        if all(task.gpu > n.gpu_capacity+EPS for n in compatible):
            return "gpu_request_exceeds_single_node"
        if all(task.cpu > n.cpu_capacity*(1-self.background)+EPS for n in compatible):
            return "cpu_request_exceeds_single_node_after_background"
        return "joint_request_never_fits_one_node"

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
                self._push(task.submit_time, 1, "arrival", task.pod_id)
            # Incremental per-node ledgers so _check_resources stays O(nodes)
            # instead of O(nodes x running tasks) per event.
            self.node_gpu_used = [0.] * len(self.nodes)
            self.node_cpu_used = [0.] * len(self.nodes)
            self.node_running_count = [0] * len(self.nodes)
            self.gpu_area = self.cpu_area = self.queue_area = 0.
            self.peak_gpu = self.peak_cpu = self.peak_queue = 0.
            self.model_nodes = {}
            for j, node in enumerate(self.nodes):
                self.model_nodes.setdefault(node.gpu_type, []).append(j)
            self.model_gpu_area = {m: 0. for m in self.model_nodes}
            self.model_cpu_area = {m: 0. for m in self.model_nodes}
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
            self.rng = random.Random(self.seed)
            self.wall_start = time.monotonic()
            self.wall_end = None
            self.initialized = True
            self._record("reset")
        self._advance_until_ready()
        return self.observation()

    def _push(self, when, priority, kind, pod_id):
        self.counter += 1
        heapq.heappush(self.heap, (when, priority, self.counter, kind, pod_id))
        return self.counter

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

    def _integrate_to(self, target):
        if target < self.current_time-EPS:
            raise RuntimeError("Clock moved backwards")
        overlap = max(0., min(target, self.measure_end)-max(self.current_time, self.measure_start))
        self.gpu_area += (self.total_gpu-sum(self.free_gpu))*overlap
        self.cpu_area += (self.total_cpu*(1-self.background)-sum(self.free_cpu))*overlap
        for model, idxs in self.model_nodes.items():
            self.model_gpu_area[model] += sum(self.nodes[j].gpu_capacity-self.free_gpu[j] for j in idxs)*overlap
            self.model_cpu_area[model] += sum(self.nodes[j].cpu_capacity*(1-self.background)-self.free_cpu[j] for j in idxs)*overlap
        self.queue_area += len(self.waiting)*overlap
        self.current_time = target

    def _move_to(self, target):
        target = min(target, self.max_sim_time)
        if self.clock == "realtime":
            while not self.stop_requested.is_set():
                delay = (target-(time.monotonic()-self.wall_start)*self.speed)/self.speed
                if delay <= 0:
                    break
                self.stop_requested.wait(min(delay, .1))
            elapsed = (time.monotonic()-self.wall_start)*self.speed
            actual = min(self.max_sim_time, max(self.current_time if self.stop_requested.is_set() else target, elapsed))
        else:
            actual = self.current_time if self.stop_requested.is_set() else target
        with self.lock:
            self._integrate_to(max(self.current_time, actual))

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

    def _process_due(self):
        while self.heap and self.heap[0][0] <= self.current_time+EPS:
            _, _, ev_counter, kind, pid = heapq.heappop(self.heap)
            task, record = self.by_id[pid], self.records[pid]
            if kind == "complete":
                if self.completion_counter.get(pid) != ev_counter:
                    continue  # stale event from before a preemption
                index = self.running.pop(pid)
                self.running_start.pop(pid, None)
                self.free_gpu[index] += task.gpu
                self.free_cpu[index] += task.cpu
                self.free_gpu[index] = min(self.nodes[index].gpu_capacity, self.free_gpu[index])
                self.free_cpu[index] = min(self.nodes[index].cpu_capacity*(1-self.background), self.free_cpu[index])
                self.node_gpu_used[index] -= task.gpu
                self.node_cpu_used[index] -= task.cpu
                self.node_running_count[index] -= 1
                if self.task_priority.get(pid) == "HP":
                    self.node_hp_running[index] -= 1
                elif self.task_priority.get(pid) == "LP":
                    self.node_lp_running[index] -= 1
                if task.gpu > 0 and self.task_priority.get(pid) == "LP":
                    self.node_spot_pids[index].discard(pid)
                record["running_total_sec"] += self.current_time - (record["start_time_sec"] or self.current_time)
                record.update(status="completed", finish_time_sec=self.current_time)
                self._preempt_generation += 1
            elif not any(self._fits(task, j, empty=True) for j in range(len(self.nodes))):
                record.update(status="rejected", rejection_reason=self._rejection_reason(task))
                kind = "reject"
            else:
                self._enqueue(pid)
                record["status"] = "waiting"
            self._record(kind, pid)
        self._check_resources()

    def _next_ready(self):
        if self.queue_policy == "fcfs":
            pid = next(iter(self.waiting), None)
            if pid is not None and self._any_fits(self.by_id[pid]):
                return pid
            return None
        max_free_cpu = max(self.free_cpu)
        candidate = self._cpu_wait_candidate(max_free_cpu)
        candidate_sequence = self._queue_seq[candidate] if candidate is not None else math.inf
        for index, group in enumerate(self._gpu_wait_groups):
            cursor = self._gpu_wait_group_cursors[index]
            while cursor < len(group):
                sequence, pid = group[cursor]
                if pid in self.waiting_set and self._queue_seq.get(pid) == sequence:
                    break
                cursor += 1
            self._gpu_wait_group_cursors[index] = cursor
            if cursor == len(group):
                continue
            sequence, pid = group[cursor]
            if sequence < candidate_sequence and self._any_fits(self.by_id[pid]):
                candidate, candidate_sequence = pid, sequence
        return candidate

    def _advance_until_ready(self):
        while True:
            with self.lock:
                if self.stop_requested.is_set():
                    self._finish("cancelled")
                if self.done:
                    return
                self._process_due()
                if self.current_time >= self.max_sim_time:
                    self._finish("drained" if not self.heap and not self.waiting and not self.running else "max_sim_time")
                    return
                self.pending_id = self._next_ready()
                if self.pending_id is not None:
                    return
                if self.preempt_enabled and self._try_preempt():
                    continue
                if not self.heap:
                    self._finish("drained" if not self.waiting and not self.running else "blocked")
                    return
                target = self.heap[0][0]
            self._move_to(target)

    def _finish(self, reason):
        if not self.done:
            self.done, self.end_reason, self.pending_id = True, reason, None
            self.wall_end = time.monotonic()
            self._record(reason)

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

    def choose_action(self, policy="first_fit"):
        obs = self.observation()
        valid = [i for i, allowed in enumerate(obs["avail"]) if allowed]
        if not valid:
            raise RuntimeError("No placement decision is ready")
        if policy == "first_fit":
            return valid[0]
        if policy == "random":
            return self.rng.choice(valid)
        if policy == "best_fit":
            task = self.by_id[self.pending_id]
            return min(valid, key=lambda j: ((self.free_gpu[j]-task.gpu)/self.nodes[j].gpu_capacity
                       +(self.free_cpu[j]-task.cpu)/self.nodes[j].cpu_capacity, j))
        if policy == "worst_fit":
            task = self.by_id[self.pending_id]
            return min(valid, key=lambda j: (-((self.free_gpu[j]-task.gpu)/self.nodes[j].gpu_capacity
                       +(self.free_cpu[j]-task.cpu)/self.nodes[j].cpu_capacity), j))
        raise ValueError("Unknown placement policy: " + str(policy))

    # ---- SpotGPU (OSDI'26, "Heterogeneity at Hyperscale", Section 4.2) ----
    # HP tasks may preempt LP spot tasks; the victim set on a node is the
    # lowest-preemption-cost prefix. Preemption cost of a spot task t:
    # C_t = G_t * (now - last_checkpoint), checkpoints every
    # self.checkpoint_interval seconds since the task's start.

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
            spot = sorted(self.node_spot_pids[j], key=self._spotgpu_victim_cost)
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
            self.free_gpu[node] += task.gpu
            self.free_cpu[node] += task.cpu
            self.free_gpu[node] = min(self.nodes[node].gpu_capacity, self.free_gpu[node])
            self.free_cpu[node] = min(self.nodes[node].cpu_capacity*(1-self.background), self.free_cpu[node])
            self.node_gpu_used[node] -= task.gpu
            self.node_cpu_used[node] -= task.cpu
            self.node_running_count[node] -= 1
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

    def _spotgpu_choose_action(self):
        """Non-preemptive placement ranking: task packing, homogeneous
        co-location, then eviction history (SpotGPU paper criteria 1-3)."""
        task = self.by_id[self.pending_id]
        priority = self.task_priority.get(self.pending_id, "Other")
        valid = [j for j in range(len(self.nodes)) if self._fits(task, j)]
        if not valid:
            raise RuntimeError("SpotGPU: pending task does not fit any node")
        if priority == "HP":
            def rank(j):
                packing = self.free_gpu[j] - task.gpu
                co_locate = 0 if self.node_hp_running[j] > 0 else 1
                history = -self.node_evictions[j]
                return (packing, co_locate, history, j)
        else:
            def rank(j):
                packing = self.free_gpu[j] - task.gpu
                co_locate = 0 if self.node_lp_running[j] > 0 else 1
                history = self.node_evictions[j]
                return (packing, co_locate, history, j)
        return min(valid, key=rank)

    def _place(self, pid, action):
        """Place a waiting task on a node (engine-internal, used by step and SpotGPU)."""
        task = self.by_id[pid]
        remaining = task.service_time - self.preserved_progress.get(pid, 0.0)
        self.free_gpu[action] -= task.gpu
        self.free_cpu[action] -= task.cpu
        self.node_gpu_used[action] += task.gpu
        self.node_cpu_used[action] += task.cpu
        self.node_running_count[action] += 1
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
        if policy == "spotgpu":
            self.preempt_enabled = True
            self.reset()
            while not self.done:
                self.step(self._spotgpu_choose_action())
            return self.summary()
        self.reset()
        while not self.done:
            self.step(self.choose_action(policy))
        return self.summary()

    def _counts(self):
        return dict(self.status_counts)

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
                    "per_model": per_model,
                    "preemptions": sum(r["preemption_count"] for r in self.records.values()),
                    "preempted_task_count": sum(1 for r in self.records.values() if r["preemption_count"] > 0),
                    "wasted_gpu_sec": sum(r["wasted_sec"]*t.gpu for r, t in zip(self.records.values(), self.tasks)),
                    "mean_wait_sec_started": sum(waits)/len(waits) if waits else 0.,
                    "p95_wait_sec_started": waits[max(0, math.ceil(.95*len(waits))-1)] if waits else 0.,
                    "peak_queue_length": int(self.peak_queue),
                    "peak_gpu_reserved": self.peak_gpu, "peak_cpu_reserved_cores": self.peak_cpu,
                    "cpu_background_fraction": self.background,
                    "reservation_metrics_are_not_SM_or_CPU_execution_utilization": True}

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
