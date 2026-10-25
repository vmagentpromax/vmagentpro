"""Construct only naturally qualifying windows selected by request pressure.

Selection is independent of durations and scheduler outcomes. Missing natural
classes stay missing. Timing is synthetic and durations are explicit proxies.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from rebuild_request_scenes import BASE, HERE, OUT, GPU, CPU, save_json, sha256


def extract_templates(obs, start, end):
    window = obs[obs.absolute_hour.between(start, end-1)].sort_values(["absolute_hour", "pod_id"])
    ids = set(window.loc[window.gpu_request > 0, "pod_id"])
    tasks, excluded = [], []
    for pid, group in window[window.pod_id.isin(ids)].groupby("pod_id", sort=True):
        first = group.iloc[0]
        reasons = []
        if group.gpu_request.nunique() != 1 or group.cpu_request_cores.nunique() != 1:
            reasons.append("resource_request_changed_in_window")
        if not group.gpu_spec_public.eq("H20").all():
            reasons.append("model_not_consistently_H20")
        if not (0 < first.gpu_request <= 8 and 0 <= first.cpu_request_cores <= 192):
            reasons.append("does_not_fit_single_empty_node")
        if reasons:
            excluded.append(dict(pod_id=pid, reasons=";".join(reasons)))
            continue
        tasks.append(dict(pod_id=pid, gpu_spec_public="H20", gpu_request=float(first.gpu_request),
            cpu_request_cores=float(first.cpu_request_cores), cluster_id=str(first.cluster_id),
            source_day=int(first.absolute_hour//24), source_hour=int(first.absolute_hour%24),
            source_server_id=str(first.server_id), source_first_absolute_hour=int(group.absolute_hour.min()),
            source_last_absolute_hour=int(group.absolute_hour.max()),
            source_observed_hours=len(group), source_server_ids=";".join(sorted(group.server_id.unique()))))
    return pd.DataFrame(tasks), pd.DataFrame(excluded, columns=["pod_id", "reasons"]), window


def verify_inventory(data, start, end):
    nodes = pd.read_csv(HERE / "servers_20_H20.csv")
    ids = pa.array(nodes.server_id.tolist())
    root = data / "asi_opensource_server_hourly" / "asi_opensource_server_hourly"
    rows = []
    for absolute in range(start, end):
        day, hour = divmod(absolute, 24)
        parts = []
        for path in sorted((root / f"day={day}" / f"hour={hour:02d}").glob("*.parquet")):
            table = pq.ParquetFile(path).read(columns=["server_id", "cluster_id", "gpu_spec_public", "gpu_count", "cpu_capacity_cores"])
            parts.append(table.filter(pc.is_in(table["server_id"], value_set=ids)).to_pandas())
        if not parts:
            return None
        frame = pd.concat(parts).drop_duplicates()
        valid = (len(frame) == 20 and frame.server_id.nunique() == 20 and
            frame.cluster_id.eq(nodes.cluster_id.iloc[0]).all() and frame.gpu_spec_public.eq("H20").all() and
            frame.gpu_count.eq(8).all() and frame.cpu_capacity_cores.eq(192).all())
        if not valid:
            return None
        rows.append(dict(absolute_hour=absolute, matching_servers=20, gpu_capacity=160, cpu_capacity_cores=3840))
    return rows


def select_windows(data=BASE, out=OUT):
    candidates = pd.read_csv(out / "all_window_checks.csv")
    obs = pd.read_parquet(out / "fixed20_observations.parquet")
    selected, selection = {}, {}
    for label in ("low", "high"):
        eligible = candidates[candidates[f"{label}_candidate"]].copy()
        eligible["count_distance"] = abs(eligible.unique_gpu_pods-100)
        eligible = eligible.sort_values(["hours", "count_distance", "start_hour"], ascending=[False, True, True])
        checked = 0
        for row in eligible.itertuples():
            checked += 1
            templates, excluded, window = extract_templates(obs, row.start_hour, row.end_hour_exclusive)
            if not 50 <= len(templates) <= 150:
                continue
            inventory = verify_inventory(data, row.start_hour, row.end_hour_exclusive)
            if inventory is None:
                continue
            folder = out / label
            folder.mkdir(exist_ok=True)
            templates.to_csv(folder / "source_templates.csv", index=False)
            excluded.to_csv(folder / "excluded_pods.csv", index=False)
            window.to_parquet(folder / "source_window_observations.parquet", index=False)
            pd.DataFrame(inventory).to_csv(folder / "inventory_checks.csv", index=False)
            selection[label] = dict(status="selected", start_hour=int(row.start_hour),
                end_hour_exclusive=int(row.end_hour_exclusive), hours=int(row.hours),
                mean_rho_gpu=float(row.mean_rho_gpu), mean_rho_cpu=float(row.mean_rho_cpu),
                mean_used_gpu_ratio=float(row.mean_used_gpu_ratio), unique_gpu_pods=int(row.unique_gpu_pods),
                retained_stable_gpu_pods=len(templates), excluded_pods=len(excluded), candidates_checked=checked,
                inventory_all_hours_verified=True,
                shape_fractions=shape_fractions(templates),
                gpu_pod_cpu_share_of_source_window=float(window.loc[window.gpu_request > 0, "cpu_request_cores"].sum()/window.cpu_request_cores.sum()))
            selected[label] = templates
            break
        if label not in selection:
            selection[label] = dict(status="missing", candidates_checked=checked,
                threshold_candidates=int(candidates[f"{label}_threshold_ok"].sum()),
                count_qualified_candidates=len(eligible), substitute_constructed=False)
    audit = dict(selection_rule="longest window, then closest GPU-Pod count to 100, then earliest start; reject resource-changing Pods and invalid inventory; no durations or scheduler results used",
        natural_windows=selection, cross_class_shape_comparison="available" if len(selected)==2 else "unavailable: a natural class is missing",
        low_threshold_relaxed=False, synthetic_substitute_allowed=False)
    save_json(out / "selected_windows.json", audit)
    return selected, audit


def shape_fractions(frame):
    return dict(fractional=float((frame.gpu_request < 1).mean()),
                single=float(frame.gpu_request.eq(1).mean()), multi=float((frame.gpu_request > 1).mean()))


def match_durations(selected, data=BASE, out=OUT):
    frames = list(selected.values())
    if not frames:
        return {}
    combined = pd.concat(frames).drop_duplicates("pod_id").set_index("pod_id")
    ids = pa.array(combined.index.tolist())
    path = data / "asi_opensource_job_execution_summary" / "asi_opensource_job_execution_summary" / "part-000.parquet"
    columns = ["pod_id", "workload_id", "server_id", "gpu_spec_public", "gpu_request", "duration_hours"]
    pieces, offset = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=262144, columns=columns):
        mask = pc.is_in(batch.column("pod_id"), value_set=ids)
        if pc.any(mask).as_py():
            matched = batch.filter(mask).to_pandas()
            matched["summary_row_index_0based"] = np.flatnonzero(mask.to_numpy(zero_copy_only=False))+offset
            pieces.append(matched)
        offset += len(batch)
    all_matches = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns+["summary_row_index_0based"])
    all_matches.to_csv(out / "all_id_matched_execution_spans.csv", index=False)
    results = {}
    for label, tasks in selected.items():
        matches = all_matches.merge(tasks[["pod_id", "gpu_spec_public", "gpu_request", "source_server_ids"]],
                                    on="pod_id", suffixes=("", "_pod"), validate="many_to_one")
        keep = ((matches.gpu_spec_public == matches.gpu_spec_public_pod) &
                (matches.gpu_request == matches.gpu_request_pod) &
                np.isfinite(matches.duration_hours) & (matches.duration_hours > 0))
        keep &= pd.Series([isinstance(s, str) and s in allowed.split(";")
                           for s, allowed in zip(matches.server_id, matches.source_server_ids)], index=matches.index)
        matches = matches[keep].copy()
        matches.to_csv(out / label / "matched_execution_spans.csv", index=False)
        counts = matches.groupby("pod_id").size()
        chosen = matches.sort_values(["pod_id", "duration_hours", "summary_row_index_0based"],
                                    ascending=[True, False, True]).drop_duplicates("pod_id")
        chosen = chosen[["pod_id", "duration_hours", "summary_row_index_0based", "workload_id"]]
        frame = tasks.merge(chosen, on="pod_id", how="left", validate="one_to_one")
        missing = frame[frame.duration_hours.isna()]
        missing.to_csv(out / label / "unmatched_duration_pods.csv", index=False)
        frame = frame[frame.duration_hours.notna()].copy()
        frame["matching_execution_span_count"] = frame.pod_id.map(counts).astype(int)
        frame["duration_proxy"] = True
        frame["duration_match_status"] = "longest_compatible_execution_span_proxy"
        if not 50 <= len(frame) <= 150:
            raise ValueError(f"{label}: only {len(frame)} duration-matched Pods; not replacing or inventing durations")
        save_json(out / label / "duration_match_audit.json", dict(source=str(path),
            summary_rows_scanned=offset, source_bytes=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
            templates_before_match=len(tasks), matched_templates=len(frame), unmatched_templates=len(missing),
            multiple_compatible_span_pods=int((frame.matching_execution_span_count > 1).sum()),
            policy="longest positive finite span matching pod_id, H20, exact GPU request and any server observed for this Pod in the selected window",
            cpu_verified_from_hourly_pair=True, exact_historical_execution_instance_identified=False,
            single_matching_row_is_also_a_proxy=True))
        results[label] = frame
    return results


def timing(frame, rho, seed, scale):
    """Fixed N, exponential gaps; horizon is random, not normalized afterwards."""
    if not 0 < rho <= 1 or scale <= 0 or len(frame) < 2:
        raise ValueError("Invalid arrival parameters")
    frame = frame.sort_values("pod_id").reset_index(drop=True).copy()
    order_seed, gap_seed = np.random.SeedSequence(seed).spawn(2)
    frame = frame.iloc[np.random.default_rng(order_seed).permutation(len(frame))].reset_index(drop=True)
    frame["service_sec"] = frame.duration_hours*3600/scale
    work = float(np.dot(frame.gpu_request, frame.service_sec))
    expected_span = work/(rho*GPU)
    delta = expected_span/(len(frame)-1)
    frame["submit_time_sec"] = np.r_[0., np.random.default_rng(gap_seed).exponential(delta, len(frame)-1)].cumsum()
    last = float(frame.submit_time_sec.iloc[-1])
    metadata = dict(target_offered_gpu_ratio=rho, expected_arrival_span_sec=expected_span,
        actual_arrival_span_sec=last, mean_interarrival_sec=delta, gpu_work_gpu_sec=work,
        realized_work_over_arrival_span_ratio=work/(GPU*last),
        cpu_work_over_expected_arrival_span_ratio=float(np.dot(frame.cpu_request_cores, frame.service_sec)/(CPU*expected_span)),
        service_mean_sec=float(frame.service_sec.mean()), service_p95_sec=float(frame.service_sec.quantile(.95)),
        service_max_sec=float(frame.service_sec.max()))
    return frame, metadata


def generate(frames, selected_audit, out=OUT):
    if not frames:
        save_json(out / "construction_audit.json", dict(status="no_qualified_natural_windows", substitute_constructed=False))
        return
    # One global K across both classes, never a separate per-class normalization.
    scale = math.ceil(max(float(f.duration_hours.max()*3600) for f in frames.values())/90.)
    constructed = {}
    for label, frame in frames.items():
        folder = out / label
        frame.to_csv(folder / "task_templates.csv", index=False)
        paths = []
        rhos = [.7,.8,.85,.9] if label == "high" else [.2,.3,.35,.4]
        # Main 90s cap plus uniform unit-scale sensitivity (60s and 120s).
        for max_service in (60,90,120):
            effective_scale = scale*90/max_service
            for rho in rhos:
                for seed in range(42,72):
                    tasks, params = timing(frame, rho, seed, effective_scale)
                    relative = Path(f"max{max_service}_rho{rho:g}") / f"seed_{seed}.json"
                    dest = folder / relative
                    dest.parent.mkdir(exist_ok=True)
                    # Fixed input-based transient boundary, common to all algorithms.
                    observation_end = float((tasks.submit_time_sec+tasks.service_sec).max())
                    cutoff = .1*float(tasks.submit_time_sec.max())
                    meta = dict(scenario=f"natural_{label}", seed=seed, gpu_capacity=GPU, cpu_capacity_cores=CPU,
                        task_count=len(tasks), initial_state="empty", warmup_hours=0, synthetic_cpu_background_cores=0,
                        time_scale_factor=effective_scale, source_natural_window=selected_audit["natural_windows"][label],
                        arrival_model="synthetic fixed-N exponential gaps, first arrival at zero; random horizon; no normalization",
                        arrival_is_synthetic=True, duration_proxy=True, exact_execution_instance_identified=False,
                        steady_state_claimed=False, target_is_input_intensity_not_measured_utilization=True,
                        transient_cutoff_sec=cutoff, measurement_window_sec=[cutoff, observation_end],
                        **params)
                    columns = ["pod_id", "cluster_id", "gpu_spec_public", "gpu_request", "cpu_request_cores",
                               "submit_time_sec", "service_sec", "duration_proxy", "duration_match_status"]
                    save_json(dest, dict(meta, tasks=tasks[columns].to_dict("records")))
                    paths.append(dict(path=str(relative), max_service_sec=max_service, rho=rho, seed=seed, **params))
                    if max_service == 90 and rho == (.85 if label == "high" else .35) and seed == 42:
                        save_json(folder / "tasks.json", dict(meta, tasks=tasks[columns].to_dict("records")))
                        tasks.to_csv(folder / "tasks.csv", index=False)
        pd.DataFrame(paths).to_csv(folder / "sequence_manifest.csv", index=False)
        constructed[label] = dict(task_count=len(frame), seeds=list(range(42,72)), time_scale_factor=scale,
            sequence_count=len(paths), main_rho=.85 if label == "high" else .35,
            shape_fractions=shape_fractions(frame), main_tasks_sha256=sha256(folder / "tasks.json"))
    save_json(out / "construction_audit.json", dict(natural_scenes=constructed,
        missing_natural_classes=[k for k in ("low","high") if k not in frames],
        substitute_constructed=False, paired_algorithm_seeds=list(range(42,72)),
        duration_proxy=True, cpu_only_background_added=False,
        caveat="W/(rho*C) calibrates input intensity only; finite empty-start simulations need measured utilization/queue validation; no steady state is asserted"))
    print("Constructed:", constructed, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    selected, audit = select_windows(args.data, args.out)
    frames = match_durations(selected, args.data, args.out)
    generate(frames, audit, args.out)


if __name__ == "__main__":
    main()
