"""Run both fixed-server replay scenes through FirstFit and BestFit."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vmagent.gpu.run import make_environment

DATA = Path(__file__).resolve().parent / "two_gpu_scenes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["all", "relative_low", "relative_high"], default="all")
    parser.add_argument("--policy", choices=["all", "first_fit", "best_fit"], default="all")
    parser.add_argument("--clock", choices=["fast", "realtime"], default="fast")
    parser.add_argument("--speed", type=float, default=1.)
    args = parser.parse_args()
    result_dir = ROOT / "outputs/gpu/two_scenes"
    rows = []
    for scene in (["relative_low", "relative_high"] if args.scene == "all" else [args.scene]):
        for policy in (["first_fit", "best_fit"] if args.policy == "all" else [args.policy]):
            path = DATA / scene / "tasks.json"
            env, meta = make_environment(ROOT / "docs/server_selection/servers_20_H20.csv", path,
                clock=args.clock, speed=args.speed, queue_policy="backfill", max_sim_time=600, seed=42)
            print(f"Running {scene} / {policy}: {len(env.tasks)} unique Pods", flush=True)
            env.run(policy)
            folder = result_dir / scene / policy
            summary = env.export(folder)
            assert summary["end_reason"] == "drained", summary
            assert summary["unfinished"] == 0
            env._check_resources()
            assert all(abs(n.gpu_capacity-g) < 1e-7 and abs(n.cpu_capacity-c) < 1e-7
                       for n,g,c in zip(env.nodes,env.free_gpu,env.free_cpu))
            # Verify cached counts against the complete task ledger.
            for status, count in env._counts().items():
                assert count == sum(r["status"] == status for r in env.records.values())
            cohort = [r for pid,r in env.records.items() if env.by_id[pid].metadata.get("phase") == "measurement"]
            waits = sorted(r["wait_sec"] for r in cohort if r["wait_sec"] is not None)
            turnarounds = [r["finish_time_sec"]-r["submit_time_sec"] for r in cohort if r["status"] == "completed"]
            # State immediately before measurement arrivals, after boundary completions.
            boundary = None
            for event in env.events:
                if event["time_sec"] > env.measure_start or (event["time_sec"] == env.measure_start and event["event"] == "arrival"):
                    break
                boundary = event
            row = dict(scene=scene, policy=policy, source_gpu_resource_time_ratio=meta["source_gpu_resource_time_ratio"],
                task_count=summary["task_count"], warmup_task_count=meta["warmup_task_count"],
                measurement_task_count=len(cohort), completed=summary["completed"], rejected=summary["rejected"],
                unfinished=summary["unfinished"], replay_offered_gpu_ratio=summary["offered_gpu_ratio"],
                reserved_gpu_ratio=summary["reserved_gpu_ratio"], reserved_cpu_ratio=summary["reserved_cpu_ratio"],
                measurement_cohort_mean_wait_sec=sum(waits)/len(waits) if waits else 0,
                measurement_cohort_p95_wait_sec=waits[max(0,math.ceil(.95*len(waits))-1)] if waits else 0,
                measurement_cohort_mean_turnaround_sec=sum(turnarounds)/len(turnarounds) if turnarounds else 0,
                sim_time_sec=summary["sim_time_sec"], wall_time_sec=summary["wall_time_sec"])
            warmup = dict(measurement_start_sec=env.measure_start, measurement_end_sec=env.measure_end,
                state_before_measurement_arrivals=boundary,
                warmup_cohort_completed_by_boundary=sum(t.metadata.get("phase")=="warmup" and
                    env.records[t.pod_id]["finish_time_sec"] is not None and
                    env.records[t.pod_id]["finish_time_sec"] <= env.measure_start for t in env.tasks),
                initial_state="empty only at prefix start; prefix tasks warm the environment; queue/running tasks carry into measurement",
                warmup_excluded_from_resource_measurement=True,
                statistically_stationary_state_claimed=False)
            (folder / "warmup_audit.json").write_text(json.dumps(warmup, indent=2), encoding="utf-8")
            (folder / "measurement_cohort_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            manifest = dict(nodes=str(ROOT / "docs/server_selection/servers_20_H20.csv"), tasks=str(path),
                policy=policy, clock=args.clock, speed=args.speed, queue_policy="backfill", seed=42,
                workload_metadata=meta)
            (folder / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    name = "comparison" if args.scene == args.policy == "all" else f"comparison_{args.scene}_{args.policy}"
    (result_dir / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (result_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
