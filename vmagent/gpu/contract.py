"""The input contract: how a node pool and a workload become engine objects.

Separated from the simulator on purpose. This is the layer M9 (parquet conversion),
M18 (contract standardisation) and the collaborator's engine all have to agree on,
so it should be readable and testable without loading any scheduling logic.

Two rules here are load-bearing and easy to break by accident:

- GPU eligibility is a *whitelist* (`allowed_gpu_types`), not a single label. A task
  with `gpu_request > 0` and an empty whitelist is rejected at load time, because
  silently treating "no model given" as "any model" would let GPU pods land on
  machines that cannot serve them.
- `asw_id` and `workload_id` are observational. Nothing in this module, and nothing
  downstream, may feed them into `_fits`, placement scoring or rejection.
"""
import csv
import io
import json
import math
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


def optional_label(row, key):
    """Read an observational label: blank, nan and null all mean unknown."""
    value = row.get(key)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "none", "null") else text


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
    # Optional access-switch / rack-domain label. Observational only: it never enters
    # _fits, placement scoring or rejection classification, so results are unchanged.
    asw_id: str = ""


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
    # Optional workload identity: in the ASI trace a multi-node job appears as several
    # pods sharing one workload_id. Observational only; absent or blank means unknown.
    workload_id: str = ""


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
                               number(row.get("cpu_capacity_cores"), "cpu_capacity_cores", True),
                               optional_label(row, "asw_id")))
    if len({node.cluster_id for node in result}) != 1:
        raise ValueError("All servers must belong to the same cluster_id")
    return result


def load_initial_placement(path):
    """Read a prescribed t=0 layout: pod_id -> server_id.

    Needed to run the defragmentation benchmark at all. Those instances start from a
    deliberately fragmented layout, and the engine's normal path re-derives placement
    from the policy, which would silently un-fragment the starting state and turn the
    experiment into a different one. Rows are `{pod_id, server_id}`; extra columns are
    ignored so the benchmark's own files can be passed directly.
    """
    rows, _ = read_records(path, "initial_pods")
    result, seen = {}, set()
    for row in rows:
        pid = required(row, "pod_id")
        sid = required(row, "server_id")
        if pid in seen:
            raise ValueError("Duplicate pod_id in initial placement: " + pid)
        seen.add(pid)
        result[pid] = sid
    if not result:
        raise ValueError("initial placement must not be empty")
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
                               priority, dict(row), optional_label(row, "workload_id")))
    result.sort(key=lambda task: (task.submit_time, task.pod_id))
    return result, {key: value for key, value in metadata.items() if key != "tasks"}
