"""Select real, mutually disjoint Pod request groups; no synthetic time fields.

The targets constrain request sums divided by capacity, NOT time-integrated load.
Each group is a subset of one real hourly snapshot, not a recovered arrival stream.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

OUT = Path(__file__).resolve().parent / "real_pod_sequences"
SPECS = [
    ("C", "GPU低_CPU高", 109, 14, 48., 3456.),
    ("B", "GPU高_CPU低", 109, 1, 144., 1152.),
    ("D", "GPU高_CPU高", 110, 1, 144., 3456.),
    ("A", "GPU低_CPU低", 109, 0, 48., 1152.),
    ("E", "GPU中_CPU中", 111, 0, 96., 2304.),
]


def select_counts(frame, gpu_sum, cpu_sum):
    shapes = frame.groupby(["gpu_request", "cpu_request_cores"], sort=True).size().reset_index(name="available")
    matrix = np.array([shapes.gpu_request, shapes.cpu_request_cores, np.ones(len(shapes))])
    # Minimal sample size within the previously discussed small-experiment range.
    result = milp(c=np.ones(len(shapes)), integrality=np.ones(len(shapes)),
                  bounds=Bounds(np.zeros(len(shapes)), shapes.available.to_numpy()),
                  constraints=LinearConstraint(matrix, [gpu_sum, cpu_sum, 50], [gpu_sum, cpu_sum, 150]),
                  options={"time_limit": 15., "mip_rel_gap": 0.})
    if result.x is None:
        return None
    shapes["selected_count"] = np.rint(result.x).astype(int)
    actual = matrix @ shapes.selected_count.to_numpy()
    if not np.allclose(actual[:2], [gpu_sum, cpu_sum], atol=1e-7):
        return None
    return shapes[shapes.selected_count > 0]


def prove_packing(frame):
    # First try a deterministic two-resource best-fit packing, then integer solve.
    gpu_left = np.full(20, 8.)
    cpu_left = np.full(20, 192.)
    assignments = {}
    order = frame.assign(size=np.maximum(frame.gpu_request/8, frame.cpu_request_cores/192)).sort_values(
        ["size", "cpu_request_cores", "gpu_request", "pod_id"], ascending=[False, False, False, True])
    for row in order.itertuples():
        feasible = np.where((gpu_left + 1e-8 >= row.gpu_request) & (cpu_left + 1e-8 >= row.cpu_request_cores))[0]
        if not len(feasible):
            break
        score = (gpu_left[feasible] - row.gpu_request)/8 + (cpu_left[feasible] - row.cpu_request_cores)/192
        j = feasible[np.argmin(score)]
        gpu_left[j] -= row.gpu_request
        cpu_left[j] -= row.cpu_request_cores
        assignments[row.pod_id] = int(j)
    if len(assignments) == len(frame):
        return assignments
    # Shape-based assignment reduces the number of binary/integer variables.
    shapes = frame.groupby(["gpu_request", "cpu_request_cores"], sort=True).size().reset_index(name="count")
    n = len(shapes)
    # Variables x[shape,node] count pods of each shape on each node.
    rows, lo, hi = [], [], []
    for k in range(n):
        row = np.zeros(n*20); row[k*20:(k+1)*20] = 1
        rows.append(row); lo.append(shapes.iloc[k]["count"]); hi.append(shapes.iloc[k]["count"])
    for j in range(20):
        for col, capacity in [("gpu_request", 8.), ("cpu_request_cores", 192.)]:
            row = np.zeros(n*20)
            for k in range(n):
                row[k*20+j] = shapes.iloc[k][col]
            rows.append(row); lo.append(0.); hi.append(capacity)
    result = milp(c=np.zeros(n*20), integrality=np.ones(n*20), bounds=Bounds(0, np.inf),
                  constraints=LinearConstraint(np.array(rows), lo, hi), options={"time_limit": 20.})
    if result.x is None:
        return None
    counts = np.rint(result.x).astype(int).reshape(n,20)
    assignments = {}
    for k, shape in shapes.iterrows():
        ids = sorted(frame[(frame.gpu_request == shape.gpu_request) &
                           (frame.cpu_request_cores == shape.cpu_request_cores)].pod_id)
        pos = 0
        for j in range(20):
            for pod_id in ids[pos:pos+counts[k,j]]:
                assignments[pod_id] = j
            pos += counts[k,j]
    return assignments


def main():
    pool = pd.read_parquet(OUT / "unique_real_H20_pod_pool.parquet")
    obs = pd.read_parquet(OUT / "real_H20_observations.parquet")
    eligible_ids = set(pool.pod_id)
    used = set()
    outputs, summaries = [], []
    nodes = pd.read_csv(Path(__file__).resolve().parent / "servers_20_H20.csv")
    for scene, label, day, hour, g, c in SPECS:
        source = obs[(obs.source_day == day) & (obs.source_hour == hour) &
                     obs.pod_id.isin(eligible_ids) & ~obs.pod_id.isin(used)].copy()
        # Reject conflicting duplicates rather than silently resolving them.
        if source.duplicated("pod_id").any():
            distinct = source.groupby("pod_id")[["gpu_request", "cpu_request_cores"]].nunique()
            conflicts = set(distinct[(distinct > 1).any(axis=1)].index)
            source = source[~source.pod_id.isin(conflicts)].drop_duplicates("pod_id")
        counts = select_counts(source, g, c)
        selection_scope = "subset_of_one_hourly_snapshot"
        if counts is None and scene == "C":
            # The user permits real task templates; broaden to the full three-day
            # pool rather than inventing CPU background or changing CPU requests.
            source = pool[~pool.pod_id.isin(used)][list(obs.columns)].copy()
            counts = select_counts(source, g, c)
            selection_scope = "subset_of_unique_real_pods_across_all_72_hours"
        if counts is None:
            summaries.append({"scene": scene, "status": "no_exact_request_group_found_at_candidate_hour", "day": day, "hour": hour})
            continue
        source["selection_key"] = source.pod_id.map(lambda x: hashlib.sha256(("42:"+x).encode()).hexdigest())
        rows = []
        for row in counts.itertuples():
            candidates = source[(source.gpu_request == row.gpu_request) & (source.cpu_request_cores == row.cpu_request_cores)]
            rows.append(candidates.sort_values(["selection_key", "pod_id"]).head(row.selected_count))
        selected = pd.concat(rows, ignore_index=True).drop(columns="selection_key")
        assert not used.intersection(selected.pod_id)
        assert selected.pod_id.is_unique
        selected = selected.merge(pool[["pod_id", "first_observed_day", "first_observed_hour_of_day", "observation_rows"]], on="pod_id", validate="one_to_one")
        selected = selected.sort_values(["first_observed_day", "first_observed_hour_of_day", "pod_id"]).reset_index(drop=True)
        selected.insert(0, "scene", scene)
        selected.insert(1, "row_index", range(1, len(selected)+1))
        first_columns = ["scene", "row_index", "pod_id", "gpu_spec_public", "gpu_request", "cpu_request_cores"]
        selected = selected[first_columns + [x for x in selected.columns if x not in first_columns]]
        assignments = prove_packing(selected)
        assert assignments is not None, f"No placement proof for {scene}"
        # Keep the packing proof separate: it is a hypothetical placement, not source history.
        packing = selected[["pod_id", "gpu_request", "cpu_request_cores"]].copy()
        packing["example_target_server_id"] = packing.pod_id.map(lambda p: nodes.iloc[assignments[p]].server_id)
        for _, group in packing.groupby("example_target_server_id"):
            assert group.gpu_request.sum() <= 8+1e-8 and group.cpu_request_cores.sum() <= 192+1e-8
        used.update(selected.pod_id)
        outputs.append(selected)
        selected.to_csv(OUT / f"scene_{scene}_real_H20_pods.csv", index=False, encoding="utf-8-sig")
        packing.to_parquet(OUT / f"scene_{scene}_hypothetical_packing_check.parquet", index=False)
        summaries.append({"scene": scene, "description": label, "status": "real_request_subset_found",
                          "selection_scope": selection_scope,
                          "source_day": day if selection_scope == "subset_of_one_hourly_snapshot" else None,
                          "source_hour": hour if selection_scope == "subset_of_one_hourly_snapshot" else None,
                          "earliest_source_day": int(selected.source_day.min()),
                          "latest_source_day": int(selected.source_day.max()), "pod_count": len(selected),
                          "gpu_request_sum": round(float(selected.gpu_request.sum()), 8),
                          "cpu_request_cores_sum": round(float(selected.cpu_request_cores.sum()), 8),
                          "gpu_request_capacity_ratio": round(float(selected.gpu_request.sum()/160), 8),
                          "cpu_request_capacity_ratio": round(float(selected.cpu_request_cores.sum()/3840), 8),
                          "fits20_empty_nodes_simultaneously_under_fractional_resource_model": True,
                          "time_integrated_offered_load_confirmed": False,
                          "synthetic_cpu_background": False,
                          "shapes": [{"gpu_request": r.gpu_request, "cpu_request_cores": r.cpu_request_cores, "pods": int(r.selected_count)} for r in counts.itertuples()]})
        print(scene, len(selected), selected.gpu_request.sum(), selected.cpu_request_cores.sum(), flush=True)
    merged = pd.concat(outputs, ignore_index=True).sort_values(["scene", "row_index"])
    assert merged.pod_id.is_unique
    # Equality check every exported field against its exact source-hour record.
    source_keys = obs.drop_duplicates(["pod_id", "source_day", "source_hour"])
    audit = merged.merge(source_keys, on=["pod_id", "source_day", "source_hour"], suffixes=("", "_source"), validate="one_to_one")
    assert len(audit) == len(merged)
    for col in ["gpu_request", "cpu_request_cores", "gpu_spec_public", "cluster_id", "server_id"]:
        assert (audit[col] == audit[col+"_source"]).all(), col
    merged.to_csv(OUT / "all_scenes_real_H20_pods.csv", index=False, encoding="utf-8-sig")
    summary = {"scenes": sorted(summaries, key=lambda x:x["scene"]), "total_unique_pods": len(merged),
               "duplicate_pod_ids_across_scenes": 0,
               "original_resource_pairs_preserved": True,
               "ordering": "first observation within day109-111, tie-break pod_id; NOT actual submission order",
               "interpretation": "Selected real Pod REQUEST GROUPS. Request/capacity ratios do not establish the document's time-integrated offered-load targets. No arrival/service/background values were synthesized."}
    (OUT / "scene_request_evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total_unique_pods": len(merged), "duplicates": 0,
                      "scenes": [{k:s.get(k) for k in ["scene", "pod_count", "source_day", "source_hour", "gpu_request_sum", "cpu_request_cores_sum", "selection_scope"]} for s in summary["scenes"]]}, indent=2))


if __name__ == "__main__":
    main()
