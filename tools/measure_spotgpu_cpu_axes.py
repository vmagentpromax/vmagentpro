"""Measure the two CPU-related SpotGPU axes on scenario F, without changing defaults.

The question came from reading OSDI'26 §4.2 against this trace: the paper ranks nodes by
idle GPU capacity and tests the evacuation prefix against GPUs only, while scenario F
reserves CPU at 100% and leaves 38% of the H20 GPUs unallocated. Both statements can be
true only if CPU-blindness has a measurable cost, so measure it.

Nothing here changes the engine's defaults; it runs the same workload under knob
settings and prints the table.

    python tools/measure_spotgpu_cpu_axes.py                 # full scenario F
    python tools/measure_spotgpu_cpu_axes.py --limit 40000   # quick look
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu.contract import load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv

# (label, knob bundle). `none` is what the engine does today and what the paper says.
CASES = [
    ("default (paper: GPU-only packing, joint prefix)", {}),
    ("victim_prefix=gpu (literal paper prefix test)", {"victim_prefix": "gpu"}),
    ("placement_cpu_aware=sum", {"placement_cpu_aware": "sum"}),
    ("placement_cpu_aware=lexicographic", {"placement_cpu_aware": "lexicographic"}),
    ("both cpu axes on", {"placement_cpu_aware": "sum", "victim_prefix": "gpu"}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv")
    parser.add_argument("--tasks", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv")
    parser.add_argument("--limit", type=int, default=119495)
    parser.add_argument("--queue", default="fcfs", choices=["fcfs", "backfill"])
    parser.add_argument("--output", type=Path, default=REPO/"outputs/spotgpu_cpu_axes.json")
    args = parser.parse_args()

    nodes = load_nodes(args.nodes)
    tasks = load_workload(args.tasks)[0][:args.limit]
    results = {}
    for label, knobs in CASES:
        env = GpuSchedEnv(nodes, list(tasks), clock="fast", queue_policy=args.queue, seed=42,
                          measurement_window=[0., 691200.], max_sim_time=4000000., spotgpu=knobs)
        started = time.monotonic()
        summary = env.run("spotgpu")
        block = summary["block_classification"]["first_attempt_classes"]
        per_model = summary["per_model"]
        results[label] = {
            "knobs": knobs, "wall_sec": round(time.monotonic()-started, 1),
            "preemptions": summary["preemptions"],
            "preempted_task_count": summary["preempted_task_count"],
            "wasted_gpu_sec": summary["wasted_gpu_sec"],
            "reserved_gpu_ratio": summary["reserved_gpu_ratio"],
            "reserved_cpu_ratio": summary["reserved_cpu_ratio"],
            "mean_wait_sec_started": summary["mean_wait_sec_started"],
            "p95_wait_sec_started": summary["p95_wait_sec_started"],
            "completed": summary["completed"], "unfinished": summary["unfinished"],
            "end_reason": summary["end_reason"],
            "blocked_fragmentation": block.get("fragmentation", 0),
            "blocked_cpu_shortage": block.get("cpu_capacity_shortage", 0),
            "blocked_gpu_shortage": block.get("gpu_capacity_shortage", 0),
            "per_model": {model: {"reserved_gpu_ratio": values["reserved_gpu_ratio"],
                                  "reserved_cpu_ratio": values["reserved_cpu_ratio"]}
                          for model, values in per_model.items()},
        }
        row = results[label]
        print("[%-49s] preemptions=%-5d wasted=%-11.1f reserved_gpu=%.6f reserved_cpu=%.6f "
              "mean_wait_h=%.2f frag=%-6d cpu_short=%-6d %s"
              % (label, row["preemptions"], row["wasted_gpu_sec"], row["reserved_gpu_ratio"],
                 row["reserved_cpu_ratio"], row["mean_wait_sec_started"]/3600,
                 row["blocked_fragmentation"], row["blocked_cpu_shortage"], row["end_reason"]))
        for model, values in sorted(row["per_model"].items()):
            print("        %-8s reserved_gpu=%.6f reserved_cpu=%.6f"
                  % (model, values["reserved_gpu_ratio"], values["reserved_cpu_ratio"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"queue_policy": args.queue, "task_count": len(tasks),
                                       "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten to " + str(args.output))


if __name__ == "__main__":
    main()
