"""Select a reproducible homogeneous inventory subset from Alibaba GPU v2026.

Run from any directory with Python, pandas and pyarrow installed.
Only server inventory and one Pod snapshot are read; source data is never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def read_hour(base, table, day, hour, columns=None):
    folder = base / table / table / f"day={day}" / f"hour={hour:02d}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(folder)
    return pd.concat([pq.ParquetFile(f).read(columns=columns).to_pandas()
                      for f in files], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(r"E:\cluster-trace-gpu-v2026\data"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    cluster = "1f28e063c4e5e1bc48ff55269617dd2f"
    table = "asi_opensource_server_hourly"
    snapshot = read_hour(args.data, table, 109, 0)
    assert not snapshot.server_id.duplicated().any()
    candidates = snapshot[(snapshot.cluster_id == cluster) &
                          (snapshot.gpu_spec_public == "H20") &
                          (snapshot.gpu_count == 8) &
                          (snapshot.cpu_capacity_cores == 192)].copy()
    stable = set(candidates.server_id)
    frames = []
    for day in range(109, 112):
        for hour in range(24):
            frame = read_hour(args.data, table, day, hour)
            assert not frame.server_id.duplicated().any(), (day, hour)
            valid = frame[(frame.cluster_id == cluster) &
                          (frame.gpu_spec_public == "H20") &
                          (frame.gpu_count == 8) &
                          (frame.cpu_capacity_cores == 192)]
            stable.intersection_update(valid.server_id)
            frames.append((day, hour, valid))
    eligible = candidates[candidates.server_id.isin(stable)].copy()
    # Fixed hash ordering is independent of file/row order and library RNG versions.
    eligible["selection_key"] = eligible.server_id.map(
        lambda sid: hashlib.sha256(("42:" + sid).encode()).hexdigest())
    chosen = eligible.sort_values(["selection_key", "server_id"]).head(20).copy()
    assert len(chosen) == 20 and chosen.server_id.nunique() == 20
    ids = set(chosen.server_id)
    checks = []
    for day, hour, frame in frames:
        matches = frame[frame.server_id.isin(ids)]
        assert len(matches) == 20
        checks.append({"day": day, "hour": f"{hour:02d}", "matching_servers": len(matches)})
    columns = ["server_id", "cluster_id", "gpu_spec_public", "gpu_count", "cpu_capacity_cores", "asw_id"]
    chosen = chosen[columns].reset_index(drop=True)
    chosen.insert(0, "index", range(1, 21))
    chosen["snapshot_day"] = 109
    chosen["snapshot_hour"] = "00"
    chosen["verified_hours"] = 72
    pod = read_hour(args.data, "asi_opensource_pod_hourly", 109, 0,
                    ["pod_id", "server_id", "cluster_id", "gpu_spec_public",
                     "server_gpu_count", "server_cpu_capacity_cores", "gpu_request",
                     "cpu_request_cores", "job_type_public", "priority_class"])
    raw_pod_count = len(pod)
    pod = pod.drop_duplicates("pod_id", keep="first")
    model_pod = pod[(pod.cluster_id == cluster) & (pod.gpu_spec_public == "H20")]
    gpu_pod = model_pod[model_pod.gpu_request > 0]
    selected_pod = pod[pod.server_id.isin(ids)]
    assert (selected_pod.cluster_id == cluster).all()
    assert (selected_pod.gpu_spec_public == "H20").all()
    assert (selected_pod.server_gpu_count == 8).all()
    assert (selected_pod.server_cpu_capacity_cores == 192).all()
    evidence = {
        "cluster_id": cluster, "gpu_spec_public": "H20",
        "snapshot": {"day": 109, "hour": "00"},
        "verified_window": {"first": "109/00", "last": "111/23", "hours": 72},
        "snapshot_candidates": len(candidates), "stable_candidates": len(eligible),
        "selection_method": "First 20 ascending SHA256('42:' + server_id), tie-break server_id; no load-based selection",
        "total_servers": 20, "total_gpus": int(chosen.gpu_count.sum()),
        "total_cpu_cores": int(chosen.cpu_capacity_cores.sum()),
        "asw_domains": int(chosen.asw_id.nunique()),
        "snapshot_duplicate_pod_rows_removed": raw_pod_count - len(pod),
        "cluster_H20_gpu_positive_pods": len(gpu_pod),
        "cluster_H20_cpu_only_positive_pods": int(((model_pod.gpu_request == 0) & (model_pod.cpu_request_cores > 0)).sum()),
        "cluster_H20_gpu_request_counts": {str(k): int(v) for k, v in gpu_pod.groupby("gpu_request").size().items()},
        "cluster_H20_job_type_counts": {str(k): int(v) for k, v in gpu_pod.groupby("job_type_public").size().items()},
        "selected_servers_snapshot_pods": len(selected_pod),
        "selected_servers_snapshot_gpu_positive_pods": int((selected_pod.gpu_request > 0).sum()),
        "checks": checks,
        "sources": [
            "https://chatgpt.com/share/6adcaa27-8234-83ec-9346-462744d200bc",
            "https://www.usenix.org/conference/osdi26/presentation/li-suyi",
            "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"
        ],
        "limitations": ["Presence is observed inventory, not proof of health or free capacity.",
                        "Window verification covers 72 hours only, not the entire six-month trace.",
                        "GPU model bucket is not an exposed pod affinity specification.",
                        "No same-ASW requirement was imposed; multi-node topology is outside this selection."]
    }
    args.out.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.out / "servers_20_H20.csv", index=False, encoding="utf-8-sig")
    (args.out / "selection_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k != "checks"}, ensure_ascii=False, indent=2))
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()
