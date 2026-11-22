"""M9 acceptance: does the main engine agree with the benchmark's own engine?

`tools/convert_benchmark_instances.py` only makes the two engines read the same
scenario. This checks whether they then *agree*, by recomputing the benchmark's own
metric definitions from the main engine's per-task records and comparing them with
`baseline_results.csv`, which the collaborator's engine produced.

Only the `migration=none` cells are comparable today. The main engine has no
migration yet (M10/M11/M12), so the one_move / cheap_multi / first_multi rows are
reported as not-yet-comparable rather than skipped silently.

Metric definitions are taken from her engine, not invented here:

    wait          = start - arrival, or horizon - arrival when never started
                    (right-censored, so an unstarted request is not dropped)
    p95           = numpy.quantile(waits, 0.95), i.e. linear interpolation
    deadline rate = fraction of all arrivals that start within start_deadline_sec
    reserved      = integral of allocated GPU over [0, allocation_window] /
                    (total_gpu * allocation_window)

    C:/Users/trw/anaconda3/python.exe tools/compare_benchmark_engines.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu.run import make_environment

# Measured agreement target. The two engines integrate the same quantity with the
# same event order, so anything larger than float noise here is a real difference.
REL_TOL = 1e-9


def quantile_linear(values, fraction):
    """numpy.quantile(..., method='linear'), reimplemented so this tool needs no numpy."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered)-1)*fraction
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def engine_metrics(env, arrivals, horizon, deadline_sec, window_sec):
    started = [env.records[pid]["start_time_sec"] for pid in arrivals if env.records[pid]["start_time_sec"] is not None]
    waits = []
    for pid in arrivals:
        record = env.records[pid]
        waits.append((record["start_time_sec"] if record["start_time_sec"] is not None else horizon)
                     - record["submit_time_sec"])
    summary = env.summary()
    return {"requests": len(arrivals),
            "started": len(started),
            "p95_wait_censored_sec": quantile_linear(waits, .95),
            "mean_wait_censored_sec": sum(waits)/len(waits) if waits else 0.,
            "start_within_deadline_rate": sum(1 for wait in waits if wait <= deadline_sec+1e-9)/len(arrivals),
            "gpu_reserved_allocation_rate": summary["reserved_gpu_ratio"],
            "gpu_effective_allocation_full_horizon_rate": summary["reserved_gpu_ratio"]
                if window_sec == horizon else None}


def compare(instance, converted, rel_tol=REL_TOL):
    instance, converted = Path(instance), Path(converted)
    config = json.loads((instance/"config.json").read_text(encoding="utf-8"))
    horizon, window = config["horizon_sec"], config["allocation_window_sec"]
    deadline = config["start_deadline_sec"]
    with (instance/"baseline_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["migration"] == "none"]
    inputs = converted/"inputs"
    results = []
    for row in rows:
        env, _ = make_environment(inputs/"nodes.csv", inputs/"jobs.csv",
                                  initial_placement=inputs/"initial_placement.csv",
                                  clock="fast", queue_policy="backfill",
                                  max_sim_time=horizon, measurement_window=[0., window])
        env.run(row["placement"])
        arrivals = [pid for pid, task in env.by_id.items()
                    if pid not in env.initial_placement]
        measured = engine_metrics(env, arrivals, horizon, deadline, window)
        differences = []
        for key, reference in row.items():
            if key not in measured or measured[key] is None:
                continue
            want, got = float(reference), float(measured[key])
            if abs(want-got) > rel_tol*max(1., abs(want)):
                differences.append((key, want, got))
        results.append((row["placement"], measured, differences))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=REPO/"benchmark任务/三指标数据评测工作文件_20261108/01_场景数据与构建/candidates")
    parser.add_argument("--converted", type=Path, default=REPO/"outputs/benchmark_csv")
    parser.add_argument("--rel-tol", type=float, default=REL_TOL)
    args = parser.parse_args()

    instances = sorted(p for c in sorted(Path(args.root).iterdir()) if c.is_dir()
                       for p in sorted(c.iterdir()) if p.is_dir() and p.name.startswith("S"))
    bad, total = 0, 0
    for instance in instances:
        out = args.converted/instance.parent.name/instance.name
        if not (out/"inputs/jobs.csv").is_file():
            print("[skip] %s/%s not converted yet" % (instance.parent.name, instance.name))
            continue
        for placement, measured, differences in compare(instance, out, args.rel_tol):
            total += 1
            tag = "agree" if not differences else "DIFF"
            print("[%s] %-32s %-9s requests=%d started=%d p95=%.4f mean=%.4f deadline=%.4f reserved=%.15f"
                  % (tag, instance.parent.name+"/"+instance.name, placement, measured["requests"],
                     measured["started"], measured["p95_wait_censored_sec"], measured["mean_wait_censored_sec"],
                     measured["start_within_deadline_rate"], measured["gpu_reserved_allocation_rate"]))
            for key, want, got in differences:
                print("        %-40s benchmark=%r engine=%r delta=%r" % (key, want, got, got-want))
                bad += 1
    print("\n%d cell(s) compared, %d field difference(s)" % (total, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
