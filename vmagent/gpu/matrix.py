"""Run a policy matrix in parallel: python -m vmagent.gpu.matrix --help.

Every (placement x queue) combination is an independent simulation, so the matrix
parallelises almost linearly. This matters because the project's safety practice is
to re-run the full matrix and diff it field by field after every engine change;
serially that costs about 15 minutes on scenario F, and it is the single biggest
tax on iteration speed.

Results are byte-comparable with running vmagent.gpu.run once per combination:
each worker builds its own environment from the same inputs and the same seed, so
parallelism cannot change any number. test_matrix_matches_serial asserts that.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .environment import DEFAULT_NODES, DEFAULT_TASKS, ROOT
from .run import environment_stamp, make_environment

PLACEMENTS = ("first_fit", "best_fit", "worst_fit", "random", "spotgpu")
QUEUES = ("fcfs", "backfill")


def _run_one(job):
    """Worker body. Must stay at module level and take plain data for Windows spawn."""
    nodes_path, tasks_path, policy, queue_policy, options, out_dir = job
    started = time.perf_counter()
    try:
        env, metadata = make_environment(nodes_path, tasks_path, clock="fast",
                                         queue_policy=queue_policy, **options)
        env.run(policy)
        summary = env.export(out_dir)
        manifest = {"nodes": str(Path(nodes_path).resolve()), "tasks": str(Path(tasks_path).resolve()),
                    "policy": policy, "queue_policy": queue_policy,
                    "workload_metadata": metadata, "seed": options.get("seed", 42),
                    "clock": "fast", "matrix_worker_wall_sec": time.perf_counter()-started,
                    "environment": environment_stamp()}
        (Path(out_dir)/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
        return {"policy": policy, "queue_policy": queue_policy, "ok": True,
                "wall_sec": time.perf_counter()-started, "summary": summary,
                "output": str(Path(out_dir).resolve())}
    except Exception as exc:  # a broken cell must not take down the whole matrix
        return {"policy": policy, "queue_policy": queue_policy, "ok": False,
                "wall_sec": time.perf_counter()-started,
                "error": "%s: %s" % (type(exc).__name__, exc)}


def build_jobs(nodes, tasks, placements, queues, options, out_root):
    return [(nodes, tasks, p, q, dict(options), Path(out_root)/("F-%s-%s" % (p, q)))
            for q in queues for p in placements]


def run_matrix(nodes, tasks, placements=PLACEMENTS, queues=QUEUES, out_root=ROOT/"outputs/gpu/matrix",
               workers=None, max_tasks_per_child=None, **options):
    jobs = build_jobs(nodes, tasks, placements, queues, options, out_root)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 2)-1))
    results, started = [], time.perf_counter()
    # Workers persist by default. On Windows every process spawn re-imports the whole
    # module tree, which measured 1.3 s against a 0.3 s cell: forcing one task per
    # child made a 4-cell matrix 5x SLOWER than serial. Pass max_tasks_per_child only
    # when the loaded workload is large enough that holding it per worker is the
    # binding constraint rather than the spawn cost.
    pool_kwargs = {"max_workers": workers}
    if max_tasks_per_child:
        pool_kwargs["max_tasks_per_child"] = max_tasks_per_child
    with ProcessPoolExecutor(**pool_kwargs) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            flag = "ok " if result["ok"] else "FAIL"
            print("[%s] %s/%s  %6.1fs  completed=%s rejected=%s unfinished=%s" % (
                flag, result["policy"], result["queue_policy"], result["wall_sec"],
                result["summary"]["completed"] if result["ok"] else "-",
                result["summary"]["rejected"] if result["ok"] else "-",
                result["summary"]["unfinished"] if result["ok"] else "-"), flush=True)
    elapsed = time.perf_counter()-started
    results.sort(key=lambda r: (r["queue_policy"], r["policy"]))
    index = {"matrix_wall_sec": elapsed, "workers": workers, "cells": len(results),
             "max_tasks_per_child": max_tasks_per_child,
             "serial_sum_sec": sum(r["wall_sec"] for r in results),
             "speedup_vs_serial_sum": (sum(r["wall_sec"] for r in results)/elapsed) if elapsed else 0.,
             "failures": [r for r in results if not r["ok"]],
             "rows": [{k: v for k, v in r.items() if k != "summary"} for r in results]}
    (Path(out_root)/"matrix_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print("\n%d cells in %.1f s using %d workers (sum of cell times %.1f s, ~%.1fx)"
          % (len(results), elapsed, workers, index["serial_sum_sec"], index["speedup_vs_serial_sum"]))
    if index["failures"]:
        print("%d cell(s) FAILED; see matrix_index.json" % len(index["failures"]))
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the placement x queue policy matrix in parallel")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--policies", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS))
    parser.add_argument("--queue-policies", nargs="+", choices=QUEUES, default=list(QUEUES))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--measurement-seconds", type=float)
    parser.add_argument("--max-sim-time", type=float, default=600.)
    parser.add_argument("--cpu-background-fraction", type=float, default=0.)
    parser.add_argument("--checkpoint-interval", type=float, default=3600.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/gpu/matrix")
    args = parser.parse_args()

    options = dict(max_sim_time=args.max_sim_time, cpu_background_fraction=args.cpu_background_fraction,
                   checkpoint_interval=args.checkpoint_interval, seed=args.seed)
    if args.measurement_seconds is not None:
        options["measurement_window"] = [0., args.measurement_seconds]
    run_matrix(args.nodes, args.tasks, tuple(args.policies), tuple(args.queue_policies),
               args.output, args.workers, **options)


if __name__ == "__main__":
    main()
