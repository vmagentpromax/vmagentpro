"""Paired 30-seed evaluation with explicit finite-run reservation metrics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vmagent.gpu.run import make_environment
from vmagent.gpu.metrics import integrate_events, run_segments
from rebuild_request_scenes import OUT, save_json

POLICIES = ("first_fit", "best_fit", "worst_fit")
METRICS = ("mean_jct_sec", "p95_jct_sec", "makespan_sec", "mean_wait_sec", "agu_reserved", "cpu_reserved",
           "afr_joint", "gpu_fragmentation_fraction", "throughput_tasks_per_sec",
           "post_transient_mean_jct_sec", "post_transient_agu_reserved", "post_transient_afr_joint")


def metrics(env, cutoff):
    """Reconstruct free-node resources over positive-length event intervals.

    AFR_joint: pending, aggregate GPU AND CPU suffice for at least one waiting
    task, but no node accommodates that task. Pure aggregate CPU shortage is
    excluded. GPU fragmentation separately ignores CPU when testing packing.

    The time integrals come from the event trace, not from rebuilding the ledger out
    of `env.records`. That used to be the same loop you still see below for AFR, and it
    silently assumed a task never changes node - `records[pid]["server_id"]` is
    overwritten by preemption and by migration, and evicted tasks stayed in `running`.
    `tools/audit_metric_derivations.py` measured the damage: exact on every
    non-preempting run, +10,494.678 and +582,374.015 GPU-seconds on the two that
    preempt. `_record` already snapshots the whole ledger at each transition, so
    integrating those snapshots is both correct under preemption and cheaper.

    What is left in the rebuild loop is genuinely per-node, and cannot come from an
    aggregate, so its precondition is now stated rather than relied on.
    """
    end = env.current_time
    segments = run_segments(env.events)
    moved = [pid for pid, parts in segments.items() if len(parts) > 1]
    if moved:
        raise ValueError(
            "metrics() derives per-node free capacity from records[pid]['server_id'], "
            "which is the node a task is on *now*; %d task(s) ran in %d separate "
            "intervals, so that reading is no longer the node they were on *then*. "
            "Add preemption or migration to POLICIES only after this loop moves onto a "
            "per-interval placement, e.g. by recording the node in the start event."
            % (len(moved), max(len(segments[pid]) for pid in moved)))
    running, waiting = set(), set()
    frag_area = gpu_frag_area = pending_area = 0.
    post_frag = 0.
    ledger = env.records
    for index, event in enumerate(env.events[:-1]):
        pid, kind = event["pod_id"], event["event"]
        if kind == "arrival":
            waiting.add(pid)
        elif kind == "start":
            waiting.discard(pid)
            running.add(pid)
        elif kind == "complete":
            running.discard(pid)
        elif kind == "reject":
            waiting.discard(pid)
        left, right = event["time_sec"], env.events[index+1]["time_sec"]
        if right <= left:
            continue
        free_g = {n.server_id: n.gpu_capacity for n in env.nodes}
        free_c = {n.server_id: n.cpu_capacity for n in env.nodes}
        for rid in running:
            r = ledger[rid]
            free_g[r["server_id"]] -= r["gpu_request"]
            free_c[r["server_id"]] -= r["cpu_request_cores"]
        total_g, total_c = sum(free_g.values()), sum(free_c.values())
        joint = gpu_frag = False
        for wid in waiting:
            task = env.by_id[wid]
            gpu_enough = total_g+1e-8 >= task.gpu
            if gpu_enough and all(g+1e-8 < task.gpu for g in free_g.values()):
                gpu_frag = True
            if gpu_enough and total_c+1e-8 >= task.cpu and not any(
                    free_g[n.server_id]+1e-8 >= task.gpu and free_c[n.server_id]+1e-8 >= task.cpu
                    for n in env.nodes):
                joint = True
        duration = right-left
        post = max(0., right-max(left, cutoff))
        frag_area += joint*duration
        gpu_frag_area += gpu_frag*duration
        pending_area += bool(waiting)*duration
        post_frag += joint*post
    gpu_seconds = integrate_events(env.events, "gpu_reserved")
    cpu_seconds = integrate_events(env.events, "cpu_reserved_cores")
    post_gpu_seconds = integrate_events(env.events, "gpu_reserved", (cutoff, end))
    records = list(ledger.values())
    jcts = [r["finish_time_sec"]-r["submit_time_sec"] for r in records]
    cohort = [r for r in records if r["submit_time_sec"] >= cutoff]
    assert cohort and end > cutoff
    return dict(mean_jct_sec=float(np.mean(jcts)), p95_jct_sec=float(np.quantile(jcts,.95)),
        makespan_sec=end, mean_wait_sec=float(np.mean([r["wait_sec"] for r in records])),
        agu_reserved=gpu_seconds/(env.total_gpu*end), cpu_reserved=cpu_seconds/(env.total_cpu*end), afr_joint=frag_area/end,
        gpu_fragmentation_fraction=gpu_frag_area/end, pending_time_fraction=pending_area/end,
        throughput_tasks_per_sec=len(records)/end,
        post_transient_mean_jct_sec=float(np.mean([r["finish_time_sec"]-r["submit_time_sec"] for r in cohort])),
        post_transient_mean_wait_sec=float(np.mean([r["wait_sec"] for r in cohort])),
        post_transient_agu_reserved=post_gpu_seconds/(env.total_gpu*(end-cutoff)),
        post_transient_afr_joint=post_frag/(end-cutoff), post_transient_task_count=len(cohort))


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    return dict(n=len(values), mean=mean, ci95_low=mean-half, ci95_high=mean+half)


def evaluate(source=OUT, output=ROOT / "outputs/gpu/request_scenes", sensitivity=False):
    rows = []
    capacities = set()
    constructed = json.loads((source / "construction_audit.json").read_text(encoding="utf-8")).get("natural_scenes", {})
    for label in ("low", "high"):
        if label not in constructed:
            continue
        manifest = pd.read_csv(source / label / "sequence_manifest.csv")
        main_rho = .85 if label == "high" else .35
        if not sensitivity:
            manifest = manifest[(manifest.max_service_sec==90) & (manifest.rho==main_rho)]
        for seq in manifest.itertuples():
            path = source / label / seq.path
            for policy in POLICIES:
                # Sum(service)+last arrival is a conservative serial drain bound.
                data = json.loads(path.read_text(encoding="utf-8"))
                bound = max(t["submit_time_sec"] for t in data["tasks"])+sum(t["service_sec"] for t in data["tasks"])+1
                env, meta = make_environment(ROOT / "docs/server_selection/servers_20_H20.csv", path,
                    clock="fast", queue_policy="backfill", max_sim_time=bound, seed=int(seq.seed))
                capacities.add((env.total_gpu, env.total_cpu))
                summary = env.run(policy)
                assert summary["end_reason"] == "drained" and summary["rejected"] == summary["unfinished"] == 0
                assert summary["completed"] == len(env.tasks)
                env._check_resources()
                assert all(abs(f-n.gpu_capacity)<1e-7 and abs(c-n.cpu_capacity)<1e-7
                           for n,f,c in zip(env.nodes,env.free_gpu,env.free_cpu))
                result = metrics(env, meta["transient_cutoff_sec"])
                # Reservation work must equal the input resource-time workload after draining.
                # Capacity comes from the loaded node inventory, never from literals: this
                # script's 160/3840 matched servers_20_H20.csv by coincidence, and the repo
                # now holds three node pools with different totals (3840, 4480, 5760 cores).
                assert abs(result["agu_reserved"]*env.total_gpu*result["makespan_sec"]-meta["gpu_work_gpu_sec"]) < 1e-5
                cpu_work = sum(t.cpu*t.service_time for t in env.tasks)
                assert abs(result["cpu_reserved"]*env.total_cpu*result["makespan_sec"]-cpu_work) < 1e-5
                rows.append(dict(scene=label, seed=int(seq.seed), policy=policy, rho=seq.rho,
                    max_service_sec=int(seq.max_service_sec), task_count=len(env.tasks), **result))
                if seq.seed == 42 and seq.max_service_sec == 90 and seq.rho == main_rho:
                    folder = output / label / policy / "seed_42"
                    env.export(folder)
                    save_json(folder / "finite_run_metrics.json", result)
                    save_json(folder / "run_manifest.json", dict(input=str(path.resolve()), policy=policy,
                        queue_policy="backfill", initial_state="empty", background=0, transient_cutoff=meta["transient_cutoff_sec"]))
            if seq.seed % 10 == 1:
                print(f"Evaluated {label}: max={seq.max_service_sec}, rho={seq.rho}, seed={seq.seed}",flush=True)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_seed_metrics.csv", index=False)
    if not rows:
        save_json(output / "evaluation_audit.json", dict(status="no_natural_scenes", substitute_constructed=False))
        return
    summaries, paired = [], []
    keys = ["scene", "rho", "max_service_sec"]
    for group_key, group in frame.groupby(keys):
        base = dict(zip(keys,group_key))
        for policy, runs in group.groupby("policy"):
            for metric in METRICS:
                summaries.append(dict(**base,policy=policy,metric=metric,**interval(runs[metric])))
        for control in ("first_fit", "worst_fit"):
            best = group[group.policy=="best_fit"].set_index("seed")
            other = group[group.policy==control].set_index("seed")
            assert set(best.index)==set(other.index)
            for metric in METRICS:
                paired.append(dict(**base,comparison=f"best_fit - {control}",metric=metric,
                                   **interval(best[metric]-other[metric])))
    # One node pool per evaluation; if that ever stops holding, the AGU denominator
    # below would be ambiguous and the audit would silently describe only part of it.
    assert len(capacities) == 1, "runs mixed node pools: %s" % sorted(capacities)
    gpu_capacity, cpu_capacity = capacities.pop()
    pd.DataFrame(summaries).to_csv(output / "metrics_mean_ci95.csv",index=False)
    pd.DataFrame(paired).to_csv(output / "paired_differences_ci95.csv",index=False)
    save_json(output / "evaluation_audit.json", dict(runs=len(frame), policies=list(POLICIES),
        seeds=sorted(int(s) for s in frame.seed.unique()), all_runs_drained=True, all_resources_released=True,
        all_work_integrals_verified=True, ci="two-sided Student t 95%; paired per-seed differences",
        steady_state_claimed=False, node_pool_gpu_capacity=gpu_capacity,
        node_pool_cpu_capacity_cores=cpu_capacity,
        AGU="allocated GPU-equivalent time / (%g * makespan); not SM utilization" % gpu_capacity,
        AFR="fraction of time with a waiting task for which aggregate GPU and CPU suffice but no node jointly fits; excludes aggregate CPU shortage",
        transient="fixed 10% of last-arrival time; JCT cohort selected by submission >= cutoff; resource integral cutoff..drain; full metrics also retained",
        natural_low_missing="low" not in constructed))
    print(frame.groupby(["scene","rho","max_service_sec","policy"])[["mean_jct_sec","mean_wait_sec","agu_reserved","afr_joint"]].mean().to_string(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gpu/request_scenes")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    evaluate(args.source,args.output,args.sensitivity)


if __name__ == "__main__":
    main()
