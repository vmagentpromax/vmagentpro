"""Build a small scenario that actually triggers SpotGPU preemption.

Why this exists: the engine has supported HP/LP preemption since d20f8b1, but the
built-in scenario A carries no priority_class at all (measured: 50 tasks, all None)
and scenario F does carry real priorities yet is 119,495 tasks over 30 nodes. So a
fresh clone could not demonstrate the project's newest capability without either
seeing nothing or waiting minutes.

Selection rule, stated so the result is not mistaken for a trace extract:
  - take the first N rows of scenario F's nodes.csv as the pool;
  - keep only tasks whose GPU model exists in that pool;
  - find T, the HP arrival whose concurrent LP load already meets pool capacity;
  - select the LP tasks genuinely running at T that outlive the horizon, capped so
    they fit the pool;
  - select the HP tasks arriving in [T, T + hp_after_hours].
Arrival times, service times, resource requests, models and priorities are copied
verbatim from scenario F. Nothing is synthesised.

Two rejected alternatives, both measured rather than assumed:
  - one shared arrival window sorted by time: _spotgpu_choose_action ranks HP above
    LP, so HP took all 24 slots, node_lp_running came out [0,0,0] and the demo
    reported preemptions=0 while the engine was behaving correctly;
  - fixed LP window followed by HP window: real LP service times are short, so by
    the time the HP window opened only 8 LP tasks were still alive and zero HP tasks
    of the pool's model had arrived, giving 0 tasks selected.
Selecting LP by "actually running at T" makes contention a property of the extracted
data instead of a hoped-for schedule.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NODES = ROOT / "场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv"
SOURCE_JOBS = ROOT / "场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv"
FIELDS = ("pod_id", "gpu_spec_public", "gpu_request", "cpu_request_cores",
          "submit_time_sec", "service_sec", "cluster_id", "duration_match_status",
          "job_type_public", "priority_class", "workload_id")


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(node_count, checkpoint_interval, max_tasks, hp_after_hours):
    pool = read(SOURCE_NODES)[:node_count]
    present = sorted({n["gpu_spec_public"] for n in pool})
    capacity_gpu = sum(float(n["gpu_count"]) for n in pool)
    capacity_cpu = sum(float(n["cpu_capacity_cores"]) for n in pool)

    gpu_jobs = []
    for job in read(SOURCE_JOBS):
        if float(job["gpu_request"]) <= 0 or job["gpu_spec_public"] not in present:
            continue
        job["_gpu"] = float(job["gpu_request"])
        job["_sub"] = float(job["submit_time_sec"])
        job["_svc"] = float(job["service_sec"])
        gpu_jobs.append(job)
    lp = [j for j in gpu_jobs if j["priority_class"] == "LP"]
    hp = [j for j in gpu_jobs if j["priority_class"] == "HP"]

    anchor = None
    for candidate in sorted(hp, key=lambda j: (j["_sub"], j["pod_id"])):
        running = sum(j["_gpu"] for j in lp if j["_sub"] <= candidate["_sub"] < j["_sub"] + j["_svc"])
        if running >= capacity_gpu:
            anchor = candidate["_sub"]
            concurrent = running
            break
    if anchor is None:
        raise SystemExit("no HP arrival has concurrent LP load >= pool capacity %.0f GPU; "
                         "increase --nodes or --hp-after-hours" % capacity_gpu)

    horizon = anchor + hp_after_hours * 3600
    victims = [j for j in lp if j["_sub"] <= anchor < j["_sub"] + j["_svc"]
               and j["_svc"] >= checkpoint_interval and j["_sub"] + j["_svc"] > horizon]
    victims.sort(key=lambda j: (-j["_gpu"], j["pod_id"]))
    selected, used = [], 0.0
    for job in victims:
        if len(selected) >= max_tasks or used + job["_gpu"] > capacity_gpu:
            continue
        selected.append(job)
        used += job["_gpu"]
    lp_kept = list(selected)

    arrivals = [j for j in hp if anchor <= j["_sub"] < horizon]
    hp_kept = arrivals[:max(0, max_tasks-len(lp_kept))]
    selected.extend(hp_kept)
    selected.sort(key=lambda j: (j["_sub"], j["pod_id"]))
    for job in selected:
        job.setdefault("workload_id", "")
        for key in ("_gpu", "_sub", "_svc"):
            job.pop(key, None)

    stats = {"nodes": len(pool), "node_models": present,
             "capacity_gpu": capacity_gpu, "capacity_cpu_cores": capacity_cpu,
             "anchor_T_sec": anchor, "concurrent_lp_gpu_at_T_in_full_trace": concurrent,
             "hp_window_sec": [anchor, horizon], "tasks": len(selected),
             "lp_tasks": len(lp_kept), "hp_tasks": len(hp_kept),
             "lp_gpu_committed": used, "checkpoint_interval_sec": checkpoint_interval,
             "hp_arrivals_in_window": len(arrivals),
             "victim_candidates_not_selected": len(victims)-len(lp_kept),
             "source": "subset of 场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv, values unmodified"}
    return pool, selected, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=float, default=3600.)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--hp-after-hours", type=float, default=2.)
    parser.add_argument("--output", type=Path, default=ROOT / "场景/04_演示场景/spotgpu_demo_20261121")
    args = parser.parse_args()

    nodes, jobs, stats = build(args.nodes, args.checkpoint_interval, args.max_tasks, args.hp_after_hours)
    out = args.output
    (out / "inputs").mkdir(parents=True, exist_ok=True)

    with open(out / "inputs/nodes.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nodes[0]))
        writer.writeheader()
        writer.writerows(nodes)
    with open(out / "inputs/jobs_spotgpu_demo.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
    (out / "inputs/construction_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
