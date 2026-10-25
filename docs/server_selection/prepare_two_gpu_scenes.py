"""Build compact, continuous real-Pod replay scenes with an explicit warm-up.

Original requests/identities are retained. Timing is reconstructed at hour
resolution and uniformly scaled, not claimed to be precise historical timing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "two_gpu_scenes"
BASE = Path(r"E:\cluster-trace-gpu-v2026\data\asi_opensource_pod_hourly\asi_opensource_pod_hourly")
COLS = ["pod_id", "server_id", "cluster_id", "state_public", "gpu_spec_public",
        "server_gpu_count", "server_cpu_capacity_cores", "gpu_request", "cpu_request_cores",
        "used_gpu_hours", "job_type_public", "priority_class"]


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(HERE / "servers_20_H20.csv")
    ids = pa.array(nodes.server_id.tolist())
    frames = []
    for day in range(109, 112):
        for hour in range(24):
            pieces = []
            for file in sorted((BASE / f"day={day}" / f"hour={hour:02d}").glob("*.parquet")):
                table = pq.ParquetFile(file).read(columns=COLS)
                pieces.append(table.filter(pc.is_in(table.column("server_id"), value_set=ids)).to_pandas())
            if not pieces:
                raise FileNotFoundError(f"Missing hour {day}/{hour:02d}")
            frame = pd.concat(pieces, ignore_index=True).drop_duplicates()
            assert not frame.pod_id.duplicated().any()
            assert frame.cluster_id.eq(nodes.cluster_id.iloc[0]).all()
            assert frame.server_gpu_count.eq(8).all() and frame.server_cpu_capacity_cores.eq(192).all()
            frame["hour_index"] = (day - 109) * 24 + hour
            frames.append(frame)
        print(f"Extracted day {day}", flush=True)
    obs = pd.concat(frames, ignore_index=True)
    obs.to_parquet(OUT / "source_72h.parquet", index=False)
    return obs


def extract_lower():
    """54h: 6h warm-up + 24h measurement + 24h duration look-ahead."""
    target = OUT / "source_lower_day167_54h.parquet"
    if target.exists():
        return pd.read_parquet(target)
    nodes = pd.read_csv(HERE / "servers_20_H20.csv")
    ids = pa.array(nodes.server_id.tolist())
    frames, checks = [], []
    server_base = BASE.parents[1] / "asi_opensource_server_hourly" / "asi_opensource_server_hourly"
    for absolute in range(166*24+18, 169*24):
        day, hour = divmod(absolute, 24)
        parts = []
        for file in sorted((BASE / f"day={day}" / f"hour={hour:02d}").glob("*.parquet")):
            table = pq.ParquetFile(file).read(columns=COLS)
            parts.append(table.filter(pc.is_in(table.column("server_id"), value_set=ids)).to_pandas())
        if not parts:
            raise FileNotFoundError(f"Missing {day}/{hour:02d}")
        frame = pd.concat(parts, ignore_index=True).drop_duplicates()
        assert not frame.pod_id.duplicated().any()
        assert frame.cluster_id.eq(nodes.cluster_id.iloc[0]).all()
        assert frame.server_gpu_count.eq(8).all() and frame.server_cpu_capacity_cores.eq(192).all()
        frame["absolute_hour"] = absolute
        frames.append(frame)
        # Inventory is checked independently, including servers without a Pod.
        inventory = []
        for file in sorted((server_base / f"day={day}" / f"hour={hour:02d}").glob("*.parquet")):
            table = pq.ParquetFile(file).read(columns=["server_id", "cluster_id", "gpu_spec_public", "gpu_count", "cpu_capacity_cores"])
            inventory.append(table.filter(pc.is_in(table.column("server_id"), value_set=ids)).to_pandas())
        if not inventory:
            raise FileNotFoundError(f"Missing inventory {day}/{hour:02d}")
        inv = pd.concat(inventory, ignore_index=True).drop_duplicates()
        assert len(inv) == 20 and inv.server_id.nunique() == 20
        assert inv.cluster_id.eq(nodes.cluster_id.iloc[0]).all() and inv.gpu_spec_public.eq("H20").all()
        assert inv.gpu_count.eq(8).all() and inv.cpu_capacity_cores.eq(192).all()
        checks.append(dict(day=day, hour=hour, matching_servers=20))
        if hour == 23:
            print(f"Read lower candidate through day {day}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(target, index=False)
    (OUT / "lower_inventory_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return result


def build_scenes(upper):
    lower = extract_lower()
    upper = upper.assign(absolute_hour=upper.hour_index + 109*24)
    # Full-day windows avoid selecting a tiny interval to manufacture contrast.
    for name, source, day in [("relative_low", lower, 167), ("relative_high", upper, 110)]:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        start, end, warm_start = day*24, (day+1)*24, day*24-6
        history = source[(source.absolute_hour >= warm_start) & (source.absolute_hour < end+24)].copy()
        admitted = history[history.absolute_hour < end]
        # All Pod IDs seen in the prefix or measurement are retained, including CPU-only.
        ids = set(admitted.pod_id)
        history = history[history.pod_id.isin(ids)].sort_values(["absolute_hour", "pod_id"])
        history.to_parquet(folder / "source_observations.parquet", index=False)
        scale = 900.  # 30 source hours -> 120 replay seconds; 1h -> 4s.
        tasks = []
        for pid, group in history.groupby("pod_id", sort=True):
            first = group.iloc[0]
            first_h, last_h = int(group.absolute_hour.min()), int(group.absolute_hour.max())
            seen = sorted(group.absolute_hour.unique().tolist())
            gpu, cpu = float(first.gpu_request), float(first.cpu_request_cores)
            assert np.isfinite([gpu, cpu]).all() and gpu >= 0 and cpu >= 0
            tasks.append(dict(pod_id=pid, gpu_spec_public="H20", gpu_request=gpu, cpu_request_cores=cpu,
                submit_time_sec=(first_h-warm_start)*3600/scale,
                service_sec=(last_h-first_h+1)*3600/scale,
                source_first_day=first_h//24, source_first_hour=first_h%24,
                source_last_day=last_h//24, source_last_hour=last_h%24,
                source_server_id=str(first.server_id), cluster_id=str(first.cluster_id),
                phase="warmup" if first_h < start else "measurement",
                first_seen_at_warmup_boundary=(first_h == warm_start),
                right_censored_at_lookahead_end=(last_h == end+23),
                gpu_request_variants=int(group.gpu_request.nunique()),
                cpu_request_variants=int(group.cpu_request_cores.nunique()),
                observation_gap_hours=(last_h-first_h+1-len(seen)),
                duration_match_status="hourly_presence_span_estimate",
                timing_is_reconstructed=True, request_policy="first_observed_real_pair_held_constant"))
        tasks.sort(key=lambda t: (t["submit_time_sec"], t["pod_id"]))
        assert len(tasks) == len(ids)
        measurement = source[(source.absolute_hour >= start) & (source.absolute_hour < end)]
        actual_ratio = float(measurement.used_gpu_hours.sum()/(160*24))
        offered_gpu = sum(t["gpu_request"]*max(0,min(120,t["submit_time_sec"]+t["service_sec"])-max(24,t["submit_time_sec"])) for t in tasks)/(160*96)
        offered_cpu = sum(t["cpu_request_cores"]*max(0,min(120,t["submit_time_sec"]+t["service_sec"])-max(24,t["submit_time_sec"])) for t in tasks)/(3840*96)
        metadata = dict(scenario=name, classification="relative GPU resource-time load; not absolute low/high threshold",
            source_measurement_window=f"day{day}/00 through day{day}/23", source_measurement_hours=24,
            source_warmup_hours=6, source_lookahead_hours=24, time_scale_factor=scale,
            gpu_capacity=160, cpu_capacity_cores=3840, measurement_window_sec=[24,120],
            warmup_window_sec=[0,24], default_max_sim_time_sec=600,
            source_gpu_resource_time_ratio=actual_ratio, replay_offered_gpu_ratio=offered_gpu,
            replay_offered_cpu_ratio=offered_cpu,
            synthetic_cpu_background_cores=0, initial_state="empty at warmup start; all prefix Pods scheduled by tested policy",
            arrival_policy="first hourly presence mapped to hour boundary; ties by pod_id; no random jitter",
            service_policy="last observed hour end minus first observed hour start, including gaps; 24h lookahead; boundary-censored spans are estimates",
            task_count=len(tasks), warmup_task_count=sum(t["phase"]=="warmup" for t in tasks),
            measurement_task_count=sum(t["phase"]=="measurement" for t in tasks),
            gpu_positive_task_count=sum(t["gpu_request"]>0 for t in tasks),
            request_changed_task_count=sum(t["gpu_request_variants"]>1 or t["cpu_request_variants"]>1 for t in tasks),
            right_censored_task_count=sum(t["right_censored_at_lookahead_end"] for t in tasks),
            observation_gap_task_count=sum(t["observation_gap_hours"]>0 for t in tasks))
        pd.DataFrame(tasks).to_csv(folder / "tasks.csv", index=False, encoding="utf-8-sig")
        compact = [{key: task[key] for key in ("pod_id", "gpu_spec_public", "gpu_request", "cpu_request_cores",
            "submit_time_sec", "service_sec", "phase", "cluster_id", "duration_match_status")} for task in tasks]
        (folder / "tasks.json").write_text(json.dumps(dict(metadata,tasks=compact), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        (folder / "construction_audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    obs = pd.read_parquet(OUT / "source_72h.parquet") if (OUT / "source_72h.parquet").exists() else extract()
    hourly = obs.groupby("hour_index").agg(pod_count=("pod_id", "size"),
        used_gpu_hours=("used_gpu_hours", "sum"), gpu_requests=("gpu_request", "sum"),
        cpu_requests=("cpu_request_cores", "sum"))
    hourly["used_gpu_ratio"] = hourly.used_gpu_hours / 160
    hourly.to_csv(OUT / "source_hourly.csv", encoding="utf-8-sig")
    print(hourly.describe().to_string(), flush=True)
    print("GPU-hours null:", int(obs.used_gpu_hours.isna().sum()),
          "states:", obs.state_public.value_counts().to_dict(), flush=True)
    if not args.extract_only:
        build_scenes(obs)


if __name__ == "__main__":
    main()
