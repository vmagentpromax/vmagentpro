"""Create scenario-A simulation timing from authentic execution-span values.

Arrival times are synthetic. For a pod with multiple real execution spans, the
longest matching span is explicitly marked as an ambiguous service-time proxy.
No demand values, IDs, or raw execution-span values are invented or changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "real_pod_sequences" / "scene_A_real_H20_pods.csv"
OUT = HERE / "real_pod_sequences" / "scene_A_timing"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-service-sec", type=float, default=90.)
    parser.add_argument("--mean-interval-sec", type=float, default=.3)
    parser.add_argument("--measurement-sec", type=float, default=120.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.max_service_sec, args.mean_interval_sec, args.measurement_sec) <= 0:
        raise ValueError("All time parameters must be positive")
    tasks = pd.read_csv(SOURCE, float_precision="round_trip")
    spans_path = OUT / "matched_original_execution_spans.csv"
    spans = pd.read_csv(spans_path, float_precision="round_trip")
    assert len(tasks) == 50 and tasks.pod_id.is_unique
    joined = spans.merge(tasks[["pod_id", "gpu_spec_public", "gpu_request", "server_id", "job_type_public", "priority_class"]],
                         on="pod_id", suffixes=("", "_pod"), validate="many_to_one")
    for col in ["gpu_spec_public", "gpu_request", "server_id", "job_type_public", "priority_class"]:
        assert (joined[col] == joined[col+"_pod"]).all()
    assert np.isfinite(spans.duration_hours).all() and (spans.duration_hours > 0).all()
    assert spans.execution_summary_row_index_0based.is_unique
    counts = spans.groupby("pod_id").size()
    chosen = spans.sort_values(["pod_id", "duration_hours", "execution_summary_row_index_0based"],
                               ascending=[True, False, True]).drop_duplicates("pod_id")
    selected = chosen[["pod_id", "workload_id", "duration_hours", "execution_summary_row_index_0based"]].rename(
        columns={"duration_hours": "source_duration_hours"})
    frame = tasks.merge(selected, on="pod_id", validate="one_to_one")
    assert len(frame) == 50 and set(frame.pod_id) == set(tasks.pod_id)
    frame["matching_execution_span_count"] = frame.pod_id.map(counts)
    frame["duration_match_status"] = np.where(frame.matching_execution_span_count == 1,
                                              "single_matching_span", "ambiguous_longest_span_proxy")
    raw_seconds = frame.source_duration_hours.to_numpy() * 3600
    scale = max(1, math.ceil(float(raw_seconds.max()) / args.max_service_sec))
    frame["source_duration_sec"] = raw_seconds
    frame["time_scale_factor"] = scale
    frame["service_sec"] = raw_seconds / scale
    # Independent streams prevent arrival samples from depending on shuffle calls.
    order_seed, interval_seed = np.random.SeedSequence(args.seed).spawn(2)
    order_rng = np.random.default_rng(order_seed)
    interval_rng = np.random.default_rng(interval_seed)
    frame = frame.sort_values("pod_id").iloc[order_rng.permutation(len(frame))].reset_index(drop=True)
    frame["arrival_order"] = np.arange(1, len(frame)+1)
    frame["interarrival_sec"] = np.r_[0., interval_rng.exponential(args.mean_interval_sec, len(frame)-1)]
    frame["submit_time_sec"] = frame.interarrival_sec.cumsum()
    frame["ideal_finish_time_sec"] = frame.submit_time_sec + frame.service_sec
    frame["arrival_is_synthetic"] = True
    frame["source_row_index"] = frame.row_index
    frame = frame.drop(columns="row_index")
    first = ["arrival_order", "pod_id", "gpu_spec_public", "gpu_request", "cpu_request_cores",
             "submit_time_sec", "interarrival_sec", "service_sec", "source_duration_hours",
             "matching_execution_span_count", "duration_match_status", "time_scale_factor",
             "execution_summary_row_index_0based", "arrival_is_synthetic", "ideal_finish_time_sec"]
    frame = frame[first + [x for x in frame.columns if x not in first]]
    H = args.measurement_sec
    overlap = np.maximum(0, np.minimum(H, frame.ideal_finish_time_sec) - np.maximum(0, frame.submit_time_sec))
    gpu_ratio = float(np.dot(frame.gpu_request, overlap)/(160*H))
    cpu_ratio = float(np.dot(frame.cpu_request_cores, overlap)/(3840*H))
    events = []
    for row in frame.itertuples():
        events.append((row.submit_time_sec, 1, row.gpu_request, row.cpu_request_cores))
        events.append((row.ideal_finish_time_sec, 0, -row.gpu_request, -row.cpu_request_cores))
    g = c = max_g = max_c = 0.
    for _, _, dg, dc in sorted(events):
        g += dg; c += dc
        max_g = max(g, max_g); max_c = max(c, max_c)
    assert abs(g) < 1e-7 and abs(c) < 1e-7
    assert frame.pod_id.is_unique and len(frame) == 50
    assert frame.submit_time_sec.iloc[0] == 0 and (np.diff(frame.submit_time_sec) > 0).all()
    assert np.allclose(frame.service_sec*scale, frame.source_duration_hours*3600, rtol=1e-13)
    assert frame.service_sec.max() <= args.max_service_sec + 1e-9
    assert gpu_ratio <= .4 and cpu_ratio <= .4
    assert (frame.ideal_finish_time_sec <= H).all()
    original = tasks.set_index("pod_id")
    for row in frame.itertuples():
        orig = original.loc[row.pod_id]
        assert row.gpu_request == orig.gpu_request and row.cpu_request_cores == orig.cpu_request_cores
        assert row.gpu_spec_public == orig.gpu_spec_public
    output_path = OUT / "scene_A_50_pods_timed.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    # A compact simulator input omits raw match metadata, which remains in the CSV.
    simulator = {
        "scenario": "A", "gpu_model": "H20", "gpu_capacity": 160, "cpu_capacity_cores": 3840,
        "measurement_window_sec": [0, H], "synthetic_cpu_background_cores": 0,
        "time_scale_factor": scale, "seed": args.seed,
        "arrival_model": "first arrival at zero, then 49 independent exponential gaps; fixed task count, random arrival horizon",
        "mean_interarrival_sec": args.mean_interval_sec,
        "duration_policy": "real longest matching execution span per pod, ambiguous IDs flagged in CSV",
        "tasks": [{"pod_id": r.pod_id, "allowed_gpu_types": [r.gpu_spec_public],
                   "gpu_request": r.gpu_request, "cpu_request_cores": r.cpu_request_cores,
                   "submit_time_sec": r.submit_time_sec, "service_sec": r.service_sec,
                   "duration_match_status": r.duration_match_status} for r in frame.itertuples()]
    }
    (OUT / "scene_A_50_pods_timed.json").write_text(json.dumps(simulator, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "source_pod_csv_sha256": sha256(SOURCE), "matched_raw_spans_csv_sha256": sha256(spans_path),
        "output_csv_sha256": sha256(output_path), "pod_count": len(frame), "unique_pod_ids": frame.pod_id.nunique(),
        "original_ids_and_resource_pairs_preserved": True, "matched_real_execution_rows": len(spans),
        "pods_with_single_matching_span": int((frame.matching_execution_span_count == 1).sum()),
        "pods_with_ambiguous_multiple_matching_spans": int((frame.matching_execution_span_count > 1).sum()),
        "duration_selection_policy": "longest matching real execution span; no summing, averaging, clipping or arbitrary row-order choice",
        "exact_execution_instance_matching_confirmed": False, "time_scale_factor": scale,
        "seed": args.seed, "mean_interarrival_sec_configured": args.mean_interval_sec,
        "arrival_rate_per_sec_configured": 1/args.mean_interval_sec,
        "actual_mean_of_49_gaps_sec": float(frame.interarrival_sec.iloc[1:].mean()),
        "last_arrival_sec": float(frame.submit_time_sec.max()),
        "service_time_sec_quantiles": {str(k): float(v) for k, v in frame.service_sec.quantile([0,.25,.5,.75,.95,1]).items()},
        "ideal_last_completion_sec_no_queue": float(frame.ideal_finish_time_sec.max()),
        "measurement_window_sec": [0, H], "offered_gpu_ratio_over_measurement_window": gpu_ratio,
        "offered_cpu_ratio_over_measurement_window": cpu_ratio,
        "ideal_peak_gpu_request": float(max_g), "ideal_peak_cpu_request_cores": float(max_c),
        "ideal_peak_gpu_ratio": float(max_g/160), "ideal_peak_cpu_ratio": float(max_c/3840),
        "low_gpu_low_cpu_range_confirmed": True, "exact_30percent_30percent_confirmed": False,
        "arrival_times_are_synthetic": True, "durations_uniformly_rescaled_from_real_spans": True,
        "actual_scheduler_or_wallclock_runtime_tested": False, "cpu_background_cores": 0,
        "sources": ["https://github.com/alibaba/clusterdata/issues/259#issuecomment-5494018728",
                    "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"],
        "notes": ["Time scaling changes units; original real arrivals are unavailable and cannot be proportionally recovered.",
                  "Poisson timing is a new synthetic experiment, not an equivalent scaling of historical traffic.",
                  "Service time starts when a task is actually scheduled, not at submission.",
                  "Ideal finish times assume immediate starts; actual completion depends on the scheduler.",
                  "Sub-millisecond service times are retained without flooring or stretching.",
                  "Single matching summary row does not prove a uniquely identified historical execution instance."]
    }
    (OUT / "timing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
