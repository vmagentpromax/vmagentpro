"""Scan observed H20 pod requests; never invent arrival times or durations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(r"E:\cluster-trace-gpu-v2026\data")
OUT = Path(__file__).resolve().parent / "real_pod_sequences"
CLUSTER = "1f28e063c4e5e1bc48ff55269617dd2f"
TABLE = "asi_opensource_pod_hourly"
COLS = ["pod_id", "server_id", "cluster_id", "gpu_spec_public",
        "gpu_request", "cpu_request_cores", "job_type_public", "priority_class"]


def read_hour(day, hour):
    folder = BASE / TABLE / TABLE / f"day={day}" / f"hour={hour:02d}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(folder)
    frame = pd.concat([pq.ParquetFile(f).read(columns=COLS).to_pandas() for f in files], ignore_index=True)
    frame = frame[(frame.cluster_id == CLUSTER) & (frame.gpu_spec_public == "H20")].copy()
    frame["source_day"] = day
    frame["source_hour"] = hour
    frame["observed_hour_index"] = day * 24 + hour
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    server_ids = set(pd.read_csv(Path(__file__).resolve().parent / "servers_20_H20.csv").server_id)
    frames = []
    checks = []
    for day in range(109, 112):
        for hour in range(24):
            f = read_hour(day, hour)
            positive = f[f.gpu_request > 0]
            frames.append(positive)
            local = f[f.server_id.isin(server_ids)].drop_duplicates("pod_id")
            checks.append({"day": day, "hour": hour, "cluster_H20_gpu_pods": positive.pod_id.nunique(),
                           "selected20_observed_pods": len(local),
                           "selected20_gpu_request_sum": float(local.gpu_request.sum()),
                           "selected20_cpu_request_sum": float(local.cpu_request_cores.sum()),
                           "selected20_used_servers": local.server_id.nunique()})
        print(f"Scanned day {day}: {sum(len(x) for x in frames):,} H20 GPU pod-hour rows", flush=True)
    all_obs = pd.concat(frames, ignore_index=True)
    # Exclude a pod if its observed resource request ever changes in this window.
    group = all_obs.groupby("pod_id", sort=False)
    stats = group.agg(observation_rows=("pod_id", "size"),
                      first_observed_hour=("observed_hour_index", "min"),
                      last_observed_hour=("observed_hour_index", "max"),
                      gpu_variants=("gpu_request", lambda x: x.nunique(dropna=False)),
                      cpu_variants=("cpu_request_cores", lambda x: x.nunique(dropna=False)))
    canonical = all_obs.sort_values(["observed_hour_index", "server_id"]).drop_duplicates("pod_id").set_index("pod_id")
    canonical = canonical.join(stats)
    valid = (canonical.gpu_variants == 1) & (canonical.cpu_variants == 1)
    valid &= np.isfinite(canonical.gpu_request) & np.isfinite(canonical.cpu_request_cores)
    valid &= (canonical.gpu_request > 0) & (canonical.gpu_request <= 8)
    valid &= (canonical.cpu_request_cores >= 0) & (canonical.cpu_request_cores <= 192)
    pool = canonical[valid].reset_index().copy()
    previous = read_hour(108, 23)
    previous_ids = set(previous[previous.gpu_request > 0].pod_id)
    pool["present_in_preceding_hour"] = pool.pod_id.isin(previous_ids)
    pool["first_observed_day"] = pool.first_observed_hour // 24
    pool["first_observed_hour_of_day"] = pool.first_observed_hour % 24
    pool["last_observed_day"] = pool.last_observed_hour // 24
    pool["last_observed_hour_of_day"] = pool.last_observed_hour % 24
    pool["cpu_gpu_ratio"] = pool.cpu_request_cores / pool.gpu_request
    pool.to_parquet(OUT / "unique_real_H20_pod_pool.parquet", index=False)
    all_obs[all_obs.pod_id.isin(pool.pod_id)].to_parquet(OUT / "real_H20_observations.parquet", index=False)
    pd.DataFrame(checks).to_csv(OUT / "hourly_observation_reference.csv", index=False, encoding="utf-8-sig")
    summary = {"cluster_id": CLUSTER, "days": [109, 110, 111], "hours": 72,
               "raw_gpu_positive_pod_hour_rows": len(all_obs), "unique_gpu_positive_pods": len(stats),
               "resource_changed_pods": int(((stats.gpu_variants > 1) | (stats.cpu_variants > 1)).sum()),
               "eligible_unique_pods": len(pool), "excluded_unique_pods": len(stats) - len(pool),
               "gpu_request_counts": {str(k): int(v) for k, v in pool.groupby("gpu_request").size().items()},
               "cpu_request_quantiles": {str(k): float(v) for k, v in pool.cpu_request_cores.quantile([0,.25,.5,.75,.95,1]).items()},
               "cpu_gpu_ratio_quantiles": {str(k): float(v) for k, v in pool.cpu_gpu_ratio.quantile([0,.25,.5,.75,.95,1]).items()},
               "job_type_counts": {str(k): int(v) for k, v in pool.groupby("job_type_public").size().items()},
               "notes": ["first/last observed hours are NOT submission/completion timestamps",
                         "pool includes H20 pods from the selected cluster, not only the selected 20 servers",
                         "no synthesized resource requests, arrivals, durations, or CPU background"]}
    (OUT / "pool_evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
