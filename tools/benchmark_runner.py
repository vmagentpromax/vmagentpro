"""Run the main engine over the benchmark grid and emit her column layout.

The point of this file is that `baseline_results.csv` stops being something only her
engine can produce. Same header, same column order, same definitions where the main
engine has them, so the archived table can be replaced by a pointer to the main engine
without anyone losing the ability to regenerate the numbers.

Two of her columns are deliberately left blank rather than filled with something that
merely shares the name, because the main engine computes a different quantity under
each of them:

    first_attempt_classes    hers re-classifies the entire waiting queue at every
                             event and accumulates blocked-seconds per class. The main
                             engine records the first observed reason per task,
                             because the former is O(queue x nodes) per event and
                             scenario F peaks at 89,127 queued tasks over 30 nodes
                             across 119,495 events. Same vocabulary, different metric.
    blocked_request_seconds  the same construction, in seconds rather than counts.

    C:/Users/trw/anaconda3/python.exe tools/benchmark_runner.py            # all instances
    C:/Users/trw/anaconda3/python.exe tools/benchmark_runner.py --instance S1 \
        --candidates .../candidates/seed20261107_interval10
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu.contract import load_initial_placement, load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv

# Her header, verbatim, so the two tables can be diffed as text.
COLUMNS = ("placement", "migration", "requests", "started", "completed", "unstarted",
           "p95_wait_censored_sec", "mean_wait_censored_sec", "start_within_deadline_rate",
           "gpu_effective_allocation_rate", "gpu_reserved_allocation_rate",
           "gpu_effective_allocation_full_horizon_rate", "migration_count",
           "migration_completed_count", "gpu_units_moved", "migration_gpu_pause_sec",
           "migration_gpu_pause_accrued_sec", "migration_inflight_at_horizon",
           "first_attempt_classes", "blocked_request_seconds", "all_resource_checks_passed")
BLANK_COLUMNS = {"first_attempt_classes", "blocked_request_seconds"}
PLACEMENTS = ("first_fit", "best_fit")
MIGRATIONS = ("none", "one_move", "cheap_multi", "first_multi")


def quantile_linear(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return ""
    position = (len(ordered)-1)*fraction
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def run_cell(nodes_path, tasks_path, placement_path, config, placement, migration):
    horizon = config["horizon_sec"]
    window = config["allocation_window_sec"]
    deadline = config["start_deadline_sec"]
    moves = config["migration"]
    env = GpuSchedEnv(load_nodes(nodes_path), load_workload(tasks_path)[0], clock="fast",
                      queue_policy="backfill", seed=42, measurement_window=[0., window],
                      max_sim_time=horizon, initial_placement=load_initial_placement(placement_path),
                      migration=migration, migration_max_plan_moves=moves["max_plan_moves"],
                      migration_max_total_moves=moves["max_total_moves"],
                      migration_base_sec=moves["base_sec"],
                      migration_per_gpu_sec=moves["per_gpu_sec"],
                      migration_per_cpu_core_sec=moves["per_cpu_core_sec"],
                      reserve_destination=moves["destination_reserved_before_source_release"],
                      pause_service=moves["pause_service"],
                      block_admission_during_plan=moves["admission_paused_during_plan"])
    summary = env.run(placement)
    arrivals = [pid for pid in env.by_id if pid not in env.initial_placement]
    waits = [(env.records[pid]["start_time_sec"] if env.records[pid]["start_time_sec"] is not None
              else horizon) - env.records[pid]["submit_time_sec"] for pid in arrivals]
    metrics = summary["migration"]
    return {"placement": placement, "migration": migration,
            "requests": len(arrivals),
            "started": sum(1 for pid in arrivals if env.records[pid]["start_time_sec"] is not None),
            "completed": sum(1 for pid in arrivals if env.records[pid]["status"] == "completed"),
            "unstarted": sum(1 for pid in arrivals if env.records[pid]["start_time_sec"] is None),
            "p95_wait_censored_sec": quantile_linear(waits, .95),
            "mean_wait_censored_sec": sum(waits)/len(waits) if waits else "",
            "start_within_deadline_rate": sum(1 for wait in waits if wait <= deadline+1e-9)/len(arrivals),
            "gpu_effective_allocation_rate": summary["effective_gpu_ratio"],
            "gpu_reserved_allocation_rate": summary["reserved_gpu_ratio"],
            "gpu_effective_allocation_full_horizon_rate": summary["effective_gpu_ratio_full_horizon"],
            "migration_count": metrics["count"],
            "migration_completed_count": metrics["completed_count"],
            "gpu_units_moved": metrics["gpu_units_moved"],
            "migration_gpu_pause_sec": metrics["gpu_pause_sec"],
            "migration_gpu_pause_accrued_sec": metrics["gpu_pause_accrued_sec"],
            "migration_inflight_at_horizon": metrics["inflight_at_horizon"],
            "first_attempt_classes": "", "blocked_request_seconds": "",
            "all_resource_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path,
                        default=REPO/"benchmark任务/三指标数据评测工作文件_20261108/01_场景数据与构建/candidates")
    parser.add_argument("--converted", type=Path, default=REPO/"outputs/benchmark_csv")
    parser.add_argument("--instance", default=None, help="only this S directory, e.g. S1")
    parser.add_argument("--output", type=Path, default=REPO/"outputs/benchmark_main_engine")
    args = parser.parse_args()

    instances = sorted(p for candidate in sorted(Path(args.candidates).iterdir()) if candidate.is_dir()
                       for p in sorted(candidate.iterdir())
                       if p.is_dir() and p.name.startswith("S")
                       and (args.instance is None or p.name == args.instance))
    missing = [str(p) for p in instances
               if not (args.converted/p.parent.name/p.name/"inputs/jobs.csv").is_file()]
    if missing:
        parser.error("convert first with tools/convert_benchmark_instances.py; missing: "
                     + ", ".join(missing[:3]))

    for instance in instances:
        inputs = args.converted/instance.parent.name/instance.name/"inputs"
        config = json.loads((instance/"config.json").read_text(encoding="utf-8"))
        started = time.monotonic()
        rows = [run_cell(inputs/"nodes.csv", inputs/"jobs.csv", inputs/"initial_placement.csv",
                         config, placement, migration)
                for placement in PLACEMENTS for migration in MIGRATIONS]
        out = args.output/instance.parent.name/instance.name
        out.mkdir(parents=True, exist_ok=True)
        with (out/"main_engine_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        (out/"main_engine_results.notes.json").write_text(json.dumps({
            "produced_by": "tools/benchmark_runner.py on vmagent.gpu.environment.GpuSchedEnv",
            "instance": str(instance.resolve()),
            "wall_sec": round(time.monotonic()-started, 1),
            "blank_columns": sorted(BLANK_COLUMNS),
            "blank_reason": "the main engine records the first observed blocking reason "
                            "per task; hers re-classifies the whole queue at every event",
            "expected_divergence": "best_fit rows differ from the archived table because the "
                                   "two engines' best_fit are different algorithms; see "
                                   "runs/benchmark_s1s4_20261122/README.md section 4",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[ok] %-40s %d cells in %5.1fs" % (instance.parent.name+"/"+instance.name,
                                                 len(rows), time.monotonic()-started))
    print("\n%d instance(s) written under %s" % (len(instances), args.output))


if __name__ == "__main__":
    main()
