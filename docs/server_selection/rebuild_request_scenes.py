"""Scan fixed-node hourly request pressure before constructing simulation timing.

Raw data are read only. Per-day caches make a full 185-day scan resumable.
No execution-duration field is read by the scan or window selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "request_load_scenes"
BASE = Path(r"E:\cluster-trace-gpu-v2026\data")
COLS = ["pod_id", "server_id", "cluster_id", "gpu_spec_public", "gpu_request",
        "cpu_request_cores", "used_gpu_hours", "avg_gpu_sm_util"]
GPU, CPU = 160., 3840.


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(data=BASE, out=OUT, workers=4):
    nodes = pd.read_csv(HERE / "servers_20_H20.csv")
    assert len(nodes) == 20 and nodes.server_id.is_unique
    assert nodes.gpu_count.sum() == GPU and nodes.cpu_capacity_cores.sum() == CPU
    ids = pa.array(nodes.server_id.tolist())
    root = data / "asi_opensource_pod_hourly" / "asi_opensource_pod_hourly"
    days = sorted(root.glob("day=*"), key=lambda p: int(p.name.split("=")[1]))
    if not days:
        raise FileNotFoundError(root)
    cache = out / "source_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = {"nodes_sha256": sha256(HERE / "servers_20_H20.csv"),
                   "source_root": str(root.resolve()), "columns": COLS, "version": 1}
    manifest = cache / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != fingerprint:
        raise ValueError("Cache belongs to another source/cohort; use a fresh --out directory")
    save_json(manifest, fingerprint)
    pa.set_cpu_count(4)

    def read_day(folder):
        day = int(folder.name.split("=")[1])
        target = cache / f"day_{day:03d}.parquet"
        audit_path = cache / f"day_{day:03d}.json"
        if target.exists() and audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if all(Path(r["path"]).stat().st_size == r["bytes"] and
                   Path(r["path"]).stat().st_mtime_ns == r["mtime_ns"] for r in audit["files"]):
                return day, True
        frames, files, missing = [], [], []
        for hour in range(24):
            paths = sorted((folder / f"hour={hour:02d}").glob("*.parquet"))
            if not paths:
                missing.append(day*24+hour)
                continue
            pieces = []
            for path in paths:
                stat = path.stat()
                files.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
                table = pq.ParquetFile(path).read(columns=COLS)
                pieces.append(table.filter(pc.is_in(table["server_id"], value_set=ids)).to_pandas())
            frame = pd.concat(pieces, ignore_index=True).drop_duplicates()
            # Reject contradictory duplicate pod-hour rows instead of picking one silently.
            if frame.pod_id.duplicated().any():
                raise ValueError(f"Conflicting duplicate Pod-hour rows: day={day}/hour={hour}")
            assert frame.cluster_id.eq(nodes.cluster_id.iloc[0]).all()
            resources = frame[["gpu_request", "cpu_request_cores"]].to_numpy()
            assert np.isfinite(resources).all() and (resources >= 0).all()
            frame["absolute_hour"] = day*24+hour
            frames.append(frame)
        if not frames:
            raise ValueError(f"Empty day {day}")
        pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
        save_json(audit_path, dict(day=day, files=files, missing_hours=missing))
        return day, False

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(read_day, day) for day in days]
        for count, future in enumerate(as_completed(futures), 1):
            day, cached = future.result()
            if count % 5 == 0 or count == len(days):
                print(f"Scanned {count}/{len(days)} days; last={day}; cached={cached}; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    return summarize(out)


def summarize(out=OUT):
    hourly, gpu_frames, missing = [], [], []
    for path in sorted((out / "source_cache").glob("day_*.parquet")):
        frame = pd.read_parquet(path)
        audit = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        missing += audit["missing_hours"]
        for hour, group in frame.groupby("absolute_hour"):
            pos = group[group.gpu_request > 0]
            hourly.append(dict(absolute_hour=int(hour), day=int(hour//24), hour=int(hour%24),
                pod_count=len(group), gpu_pod_count=len(pos), observed_servers=group.server_id.nunique(),
                gpu_request=float(group.gpu_request.sum()), cpu_request_cores=float(group.cpu_request_cores.sum()),
                gpu_pod_cpu_request_cores=float(pos.cpu_request_cores.sum()),
                rho_gpu=float(group.gpu_request.sum()/GPU), rho_cpu=float(group.cpu_request_cores.sum()/CPU),
                used_gpu_ratio=float(group.used_gpu_hours.sum()/GPU),
                used_gpu_null_count=int(group.used_gpu_hours.isna().sum())))
        # Keep zero-GPU observations of selected IDs too, for resource-change exclusion.
        gpu_frames.append(frame)
    hours = pd.DataFrame(hourly).sort_values("absolute_hour")
    hours.to_csv(out / "hourly_request_load.csv", index=False)
    observations = pd.concat(gpu_frames, ignore_index=True)
    observations.to_parquet(out / "fixed20_observations.parquet", index=False)
    save_json(out / "scan_audit.json", dict(
        hours_scanned=len(hours), day_min=int(hours.day.min()), day_max=int(hours.day.max()),
        missing_hours=missing, cohort="original 20 H20 servers; all Pods for hourly sums",
        gpu_capacity=GPU, cpu_capacity_cores=CPU, gpu_request_ratio_min=float(hours.rho_gpu.min()),
        gpu_request_ratio_max=float(hours.rho_gpu.max()),
        cpu_request_ratio_min=float(hours.rho_cpu.min()),
        observed_server_count_min=int(hours.observed_servers.min()),
        zero_observed_pod_hours_are_not_imputed=True,
        classification_uses_duration=False, hourly_csv_sha256=sha256(out / "hourly_request_load.csv")))
    print("Hourly scan:", json.dumps(json.loads((out / "scan_audit.json").read_text()), ensure_ascii=False), flush=True)
    return hours, observations


def windows(hours, obs, out=OUT):
    """Exhaust all contiguous windows of each integer length from 6 to 24h."""
    rows = []
    hour_groups = {int(h): g for h, g in obs.groupby("absolute_hour")}
    hmap = hours.set_index("absolute_hour")
    for start in sorted(hour_groups):
        identities = set()
        for length in range(1, 25):
            hour = start+length-1
            if hour not in hour_groups:
                break
            group = hour_groups[hour]
            identities.update(group.loc[group.gpu_request > 0, "pod_id"])
            if length < 6:
                continue
            chunk = hmap.loc[start:hour]
            gpu, cpu = float(chunk.rho_gpu.mean()), float(chunk.rho_cpu.mean())
            count_ok = 50 <= len(identities) <= 150
            high = gpu >= .8 and cpu >= .6
            low = gpu <= .35 and cpu <= .35
            rows.append(dict(start_hour=start, end_hour_exclusive=hour+1, hours=length,
                mean_rho_gpu=gpu, mean_rho_cpu=cpu, mean_used_gpu_ratio=float(chunk.used_gpu_ratio.mean()),
                unique_gpu_pods=len(identities), task_count_ok=count_ok,
                high_threshold_ok=high, low_threshold_ok=low,
                high_candidate=high and count_ok, low_candidate=low and count_ok))
    candidates = pd.DataFrame(rows)
    candidates.to_csv(out / "all_window_checks.csv", index=False)
    audit = dict(window_count=len(candidates), lengths_hours=list(range(6,25)),
        high_threshold="mean rho_gpu >= 0.8 and mean rho_cpu >= 0.6",
        low_threshold="mean rho_gpu <= 0.35 and mean rho_cpu <= 0.35",
        task_count_range=[50,150], high_candidates=int(candidates.high_candidate.sum()),
        low_candidates=int(candidates.low_candidate.sum()),
        minimum_window_mean_rho_gpu=float(candidates.mean_rho_gpu.min()),
        high_threshold_window_count=int(candidates.high_threshold_ok.sum()),
        low_threshold_window_count=int(candidates.low_threshold_ok.sum()),
        durations_used_in_window_selection=False)
    save_json(out / "window_selection_audit.json", audit)
    print("Window selection:", json.dumps(audit), flush=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args()
    hours, obs = summarize(args.out) if args.from_cache else scan(args.data, args.out, args.workers)
    windows(hours, obs, args.out)


if __name__ == "__main__":
    main()
