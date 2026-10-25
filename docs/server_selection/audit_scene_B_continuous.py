"""Export all observed Pods on the fixed 20 servers for the 72-hour window.

One observed hour is one hour of exposure proxy, never an exact execution span.
No demand, task selection, arrival synthesis, or duration rescaling is applied.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "real_pod_sequences" / "scene_B_continuous_72h"
SOURCE = Path(r"E:\cluster-trace-gpu-v2026\data\asi_opensource_pod_hourly\asi_opensource_pod_hourly")
CLUSTER = "1f28e063c4e5e1bc48ff55269617dd2f"
COLS = ["pod_id", "server_id", "cluster_id", "gpu_spec_public", "gpu_request",
        "cpu_request_cores", "job_type_public", "priority_class"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(HERE / "servers_20_H20.csv")
    assert len(nodes) == 20 and nodes.server_id.is_unique
    ids = pa.array(nodes.server_id.tolist())
    frames = []
    exact_duplicates = 0
    for day in range(109, 112):
        for hour in range(24):
            files = sorted((SOURCE / f"day={day}" / f"hour={hour:02d}").glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"Missing {day}/{hour:02d}")
            parts = []
            for file in files:
                table = pq.ParquetFile(file).read(columns=COLS)
                parts.append(table.filter(pc.is_in(table.column("server_id"), value_set=ids)).to_pandas())
            frame = pd.concat(parts, ignore_index=True)
            assert frame.cluster_id.eq(CLUSTER).all()
            before = len(frame)
            frame = frame.drop_duplicates()
            exact_duplicates += before - len(frame)
            # A conflicting Pod/hour cannot be silently collapsed into a task.
            assert not frame.duplicated("pod_id").any(), f"Ambiguous Pod/hour {day}/{hour}"
            assert np.isfinite(frame[["gpu_request", "cpu_request_cores"]].to_numpy()).all()
            assert (frame[["gpu_request", "cpu_request_cores"]] >= 0).all().all()
            frame["source_day"] = day
            frame["source_hour"] = hour
            frame["window_hour_index"] = (day - 109) * 24 + hour
            frames.append(frame)
        print(f"Read day {day}: {sum(map(len, frames))} Pod-hour rows", flush=True)
    obs = pd.concat(frames, ignore_index=True).sort_values(["window_hour_index", "pod_id"])
    obs.to_csv(OUT / "all_pod_hourly_observations.csv", index=False, encoding="utf-8-sig")
    obs.to_parquet(OUT / "all_pod_hourly_observations.parquet", index=False)
    hourly = obs.groupby(["window_hour_index", "source_day", "source_hour"]).agg(
        pod_count=("pod_id", "nunique"), gpu_request_sum=("gpu_request", "sum"),
        cpu_request_cores_sum=("cpu_request_cores", "sum")).reset_index()
    assert len(hourly) == 72
    hourly["gpu_ratio_proxy"] = hourly.gpu_request_sum / 160
    hourly["cpu_ratio_proxy"] = hourly.cpu_request_cores_sum / 3840
    hourly.to_csv(OUT / "hourly_load_proxy.csv", index=False, encoding="utf-8-sig")
    records = []
    for pid, group in obs.groupby("pod_id", sort=True):
        hours = sorted(group.window_hour_index.unique().tolist())
        gv, cv = group.gpu_request.nunique(), group.cpu_request_cores.nunique()
        records.append(dict(pod_id=pid, gpu_request=group.gpu_request.iloc[0] if gv == 1 else None,
            cpu_request_cores=group.cpu_request_cores.iloc[0] if cv == 1 else None,
            resource_requests_constant=(gv == cv == 1),
            gpu_positive_observed_hours=int((group.gpu_request > 0).sum()),
            observed_hours_proxy=len(hours), first_observed_window_hour=hours[0],
            last_observed_window_hour=hours[-1],
            consecutive_observation_segments=1 + sum(b - a > 1 for a, b in zip(hours, hours[1:])),
            observed_hour_indices=json.dumps(hours),
            source_server_ids=";".join(sorted(group.server_id.unique())),
            source_gpu_labels=";".join(sorted(group.gpu_spec_public.dropna().unique())),
            gpu_hours_proxy=float(group.gpu_request.sum()),
            cpu_core_hours_proxy=float(group.cpu_request_cores.sum())))
    tasks = pd.DataFrame(records).sort_values(["first_observed_window_hour", "pod_id"])
    assert tasks.pod_id.is_unique
    tasks.to_csv(OUT / "unique_pods.csv", index=False, encoding="utf-8-sig")
    tasks[tasks.gpu_positive_observed_hours > 0].to_csv(
        OUT / "unique_gpu_pods.csv", index=False, encoding="utf-8-sig")
    overlaps = {}
    for scene in "ACDE":
        prior = pd.read_csv(HERE / "real_pod_sequences" / f"scene_{scene}_real_H20_pods.csv")
        overlaps[scene] = sorted(set(prior.pod_id) & set(tasks.pod_id))
    gpu_rows = obs[obs.gpu_request > 0]
    gh, ch = float(obs.gpu_request.sum()), float(obs.cpu_request_cores.sum())
    result = dict(window="day109/00 through day111/23", window_hours=72,
        servers=nodes.server_id.tolist(), gpu_capacity=160, cpu_capacity_cores=3840,
        all_pods_preserved=True, exact_duplicate_source_rows_removed=exact_duplicates,
        unique_pods=len(tasks), unique_gpu_positive_pods=int((tasks.gpu_positive_observed_hours > 0).sum()),
        pod_hour_observations=len(obs), tasks_with_variable_requests=int((~tasks.resource_requests_constant).sum()),
        tasks_with_observation_gaps=int((tasks.consecutive_observation_segments > 1).sum()),
        gpu_hours_proxy=gh, cpu_core_hours_proxy=ch, gpu_ratio_proxy=gh/(160*72),
        cpu_ratio_proxy=ch/(3840*72), gpu_positive_pods_cpu_core_hours_proxy=float(gpu_rows.cpu_request_cores.sum()),
        gpu_positive_pods_cpu_ratio_proxy=float(gpu_rows.cpu_request_cores.sum())/(3840*72),
        target_gpu_hours=10368, target_cpu_core_hours=82944,
        exact_target_satisfied=bool(np.isclose(gh,10368) and np.isclose(ch,82944)),
        high_gpu_low_cpu_range_satisfied=bool(.8 <= gh/(160*72) <= .95 and ch/(3840*72) <= .4),
        duration_method="count distinct observed hours per Pod; one hour exposure proxy for each observation",
        exact_execution_times_available=False, exact_arrival_times_available=False,
        overlaps_with_other_scene_pods=overlaps,
        interpretation="Hourly request exposure estimate; snapshot presence is not proof of full-hour running. Values above capacity can occur because same-hour observations need not be simultaneous. No simulator-ready arrivals or service durations are fabricated.")
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("servers", "overlaps_with_other_scene_pods")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
