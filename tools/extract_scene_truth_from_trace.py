"""Pull the trace rows a scene was built from, so the simulator has something to match.

M16 is "does this simulator reproduce what the cluster did?". The blocker so far has
been that the answer was never computable: the scenes were derived from
`asi_opensource_pod_hourly`, then the source rows were left behind on D: and only the
derived workload CSV kept. This tool rebuilds the other side of the comparison - the
observed rows for exactly the servers a scene uses - and stores them next to the scene,
so every later fidelity check is a local join instead of a 192-partition scan of a USB
drive.

What it takes from the trace is deliberately the *measured* fields, not the ones the
scene builder already consumed:

- `schedule_delay_sec` - the only delay ground truth there is. The simulator's
  `wait_sec` is a modelled queueing delay; this is what Alibaba's scheduler actually
  took. Comparing the two distributions is the replay calibration.
- `used_gpu_hours` and `avg_gpu_sm_util` - what was reserved versus what computed, per
  reserved GPU-hour. `tools/trace_fidelity_probe.py` put the cluster-wide figure at
  about 20% of reserved GPU-hours actually busy, which is why no reservation number
  from this simulator may be called utilisation.
- `ready_delay_sec`, `state_public`, `priority_class`, `job_type_public` - so a
  mismatch can be attributed instead of merely observed.

Reads one partition at a time and filters immediately; the whole scan is bounded by the
servers in `nodes.csv`, not by the trace size.

    C:/Users/trw/anaconda3/python.exe tools/extract_scene_truth_from_trace.py \
        --nodes 场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv --start-day 108 --end-day 116
"""
import argparse
import json
from pathlib import Path

import pandas as pd

TRACE = Path("D:/cluster-trace-gpu-v2026")
POD = TRACE / "asi_opensource_pod_hourly"

COLUMNS = ["pod_id", "workload_id", "server_id", "cluster_id", "state_public",
           "gpu_spec_public", "server_gpu_count", "priority_class", "job_type_public",
           "gpu_request", "cpu_request_cores", "used_gpu_hours", "avg_gpu_sm_util",
           "avg_cpu_request_util", "schedule_delay_sec", "ready_delay_sec",
           "ready_status"]


def partitions(start_day, end_day, hours=None):
    # `day=` is unpadded and `hour=` is two digits in this release of the table, which
    # is inconsistent enough that a sorted() over the paths would not give day order.
    for day in range(start_day, end_day+1):
        for hour in (hours or range(24)):
            directory = POD/("day=%d" % day)/("hour=%02d" % hour)
            for path in sorted(directory.glob("*.parquet")):
                yield day, hour, path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", required=True, type=Path)
    parser.add_argument("--start-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--last-hour", type=int, default=None,
                        help="only this hour of the first day, for a warm-up partition")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    servers = set(nodes.server_id)
    clusters = set(nodes.cluster_id)
    out = args.out or args.nodes.parents[1]/"truth"/"pod_hourly_observed.parquet"
    print("servers=%d clusters=%s days=%d..%d" % (len(servers), sorted(clusters),
                                                  args.start_day, args.end_day))

    frames, scanned = [], 0
    for day, hour, path in partitions(args.start_day, args.end_day):
        if day == args.start_day and args.last_hour is not None and hour != args.last_hour:
            continue
        frame = pd.read_parquet(path, columns=COLUMNS)
        scanned += len(frame)
        keep = frame[frame.server_id.isin(servers)]
        if len(keep):
            keep = keep.assign(snapshot_day=day, snapshot_hour=hour)
            frames.append(keep)
        if (hour+1) % 6 == 0:
            print("    day=%03d hour=%02d rows_scanned=%d matched=%d"
                  % (day, hour, scanned, sum(len(f) for f in frames)), flush=True)
    observed = pd.concat(frames, ignore_index=True)
    if observed.empty:
        raise SystemExit("no trace row matched any server in %s - wrong day range, or "
                         "the scene predates this trace?" % args.nodes)
    assert set(observed.cluster_id) <= clusters, "servers straddle clusters: %s" % (
        sorted(set(observed.cluster_id)))

    out.parent.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(out, index=False)
    hours = (1 if args.last_hour is not None else 24) + 24*(args.end_day-args.start_day)
    if len(frames) != hours:
        print("    WARNING %d partitions carried these servers, expected %d over %d hours"
              % (len(frames), hours, hours))
    summary = summarise(observed, nodes, hours, scanned, len(frames), str(args.nodes))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (out.parent/"extraction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if observed.server_id.nunique() < len(servers):
        missing = sorted(servers-set(observed.server_id))
        print("    WARNING %d of %d scene servers never appear in these partitions: %s"
              % (len(missing), len(servers), missing[:3]))


def summarise(observed, nodes, hours, scanned, partitions, source):
    """The numbers a re-scan is not needed to check, so keep it callable.

    Split out of main() because a 193-partition pass over a USB drive takes 25 minutes,
    and a change to any figure here should be verifiable against the parquet that pass
    already wrote rather than by paying for the scan again.
    """
    # gpu_request == 0 also satisfies "% 1 == 0", and on scenario F those CPU-only rows
    # are 87% of the table, so without the >0 here a coverage figure silently divides by
    # a population the field could never describe.
    whole = observed[(observed.gpu_request > 0) & (observed.gpu_request % 1 == 0)]
    per_gpu = whole.avg_gpu_sm_util/whole.gpu_request
    weight = whole.used_gpu_hours
    good = per_gpu.notna() & (weight > 0)
    gpu = observed[observed.gpu_request > 0]
    # The scene this was built for keeps only rows that ask for something. Applying its
    # own stated filter here is what turns "I read the same servers" into "the scene is
    # reproducible from the trace": on scenario F all four counts its manifest records
    # come back exactly - 339,034 rows, 119,742 pods, 5,910 GPU pods, 113,832 CPU-only.
    # Without this line the extraction reads as 64,949 rows and 973 pods *more* than the
    # scene, which looks like a discrepancy and is only a filter nobody applied.
    requested = observed[(observed.gpu_request > 0) | (observed.cpu_request_cores > 0)]
    return {
        "nodes": source, "rows": int(len(observed)),
        "rows_scanned": int(scanned), "distinct_pods": int(observed.pod_id.nunique()),
        "distinct_servers_observed": int(observed.server_id.nunique()),
        "partitions_used": int(partitions), "hours_covered": int(hours),
        "scene_filter_gpu_or_cpu_request": {
            "rows": int(len(requested)), "pods": int(requested.pod_id.nunique()),
            "gpu_pods": int(requested[requested.gpu_request > 0].pod_id.nunique()),
            "cpu_only_pods": int(requested[requested.gpu_request <= 0].pod_id.nunique())},
        "schedule_delay_sec": {("p%.0f" % (q*100)): float(observed.schedule_delay_sec.quantile(q))
                               for q in (.5, .9, .95, .99)},
        "schedule_delay_nonnull_share": float(observed.schedule_delay_sec.notna().mean()),
        "reserved_gpu_hours": float(gpu.used_gpu_hours.sum()),
        "physical_gpu_hours": float(nodes.gpu_count.sum()*hours),
        # Above 1.0 is not the cluster overselling itself: used_gpu_hours credits a pod
        # that was present in an hour, so sequential pods on one card each fill it.
        # The ratio is the number that stops a reader treating the column as a
        # reservation integral.
        "reserved_gpu_hours_over_physical_capacity": float(
            gpu.used_gpu_hours.sum()/(nodes.gpu_count.sum()*hours)),
        "per_gpu_sm_util_gpu_hour_weighted": float(
            (per_gpu[good]*weight[good]).sum()/weight[good].sum()),
        # Denominator is GPU pod-hours, not `whole`: `whole` still carries the CPU-only
        # rows that survive `% 1 == 0`, and quoting coverage against them turns 20.78%
        # into 2.43% purely by including 87% of the table the field could never describe.
        "per_gpu_sm_util_row_coverage": float(
            good.sum()/max(1, int((weight > 0).sum()))),
        "gpu_pod_hours_with_util": int(good.sum()),
        "gpu_pod_hours": int((weight > 0).sum()),
        "reserved_gpu_hours_under_10pct_util": float(
            weight[good & (per_gpu < 10.)].sum()/weight[good].sum()),
    }


if __name__ == "__main__":
    main()
