"""M16 step 2: does the simulator's queueing delay look like the trace's?

`tools/extract_scene_truth_from_trace.py` rebuilds the observed side of scene F. This is
the comparison, and it is the first thing in this project that can *falsify* the
simulator rather than just exercise it: for the same pods, the engine reports a
`wait_sec` (submit to first admission under the modelled policy) and the trace reports
`schedule_delay_sec` (what Alibaba's scheduler actually took).

Two things are deliberately not done, because doing them would manufacture agreement:

- **No mean.** `schedule_delay_sec` is truncated at 24h-1s in the release and has a
  heavy tail, so a mean is mostly an artefact of where the clip sits. Both sides are
  compared on quantiles, both clipped to the same ceiling.
- **No time-axis alignment.** `jobs_F.csv` stores `submit_time_sec` relative to the
  window start, while the trace's delay is a per-pod quantity that does not depend on
  the window. A distribution comparison is well-defined across that shift; a per-pod
  scatter at a given wall-clock instant is not.

It also settles a claim the probe could only guess at. The 20-partition sample showed HP
with a much better median than LP (2 s vs 29 s) but a *worse* tail (p90 1,482 s vs 404
s), which contradicts a naive "HP is prioritised" story. The obvious explanation - big
requests queue longer - is testable now that the pods' `gpu_request` sits next to their
delay, and it is not the only candidate: whole-GPU pods and GPU-sharing pods may simply
be different populations, and only the former carry SM utilisation at all.

    C:/Users/trw/anaconda3/python.exe tools/calibrate_replay_delay.py --limit 20000
    C:/Users/trw/anaconda3/python.exe tools/calibrate_replay_delay.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SCENE = REPO/"runs/hetero8d_20261101"
CEILING = 86399.  # the release truncates schedule_delay_sec at 24h-1s
QUANTILES = (.5, .9, .95, .99)


def quantiles(values):
    values = np.clip(np.asarray(values, dtype=float), 0., CEILING)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size),
            **{"p%.0f" % (q*100): round(float(np.quantile(values, q)), 1) for q in QUANTILES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", type=Path, default=SCENE/"truth/pod_hourly_observed.parquet")
    parser.add_argument("--jobs", type=Path, default=SCENE/"inputs/jobs_F.csv")
    parser.add_argument("--nodes", type=Path, default=SCENE/"inputs/nodes.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--policies", default="first_fit,first_fit:fcfs,spotgpu",
                        help="policy or policy:queue_policy, comma separated")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    observed = pd.read_parquet(args.truth)
    jobs = pd.read_csv(args.jobs)
    if args.limit:
        jobs = jobs.head(args.limit)

    # schedule_delay_sec is a pod attribute copied onto each of its pod-hour rows, so
    # first-non-null is the pod's value; taking a mean over rows would just down-weight
    # long-lived pods. Verified in the probe: mean distinct values per pod was 0.9910.
    delay = (observed.sort_values(["pod_id", "snapshot_day", "snapshot_hour"])
             .groupby("pod_id").agg(schedule_delay_sec=("schedule_delay_sec", "first"),
                                    gpu_request=("gpu_request", "first"),
                                    priority=("priority_class", "first"),
                                    job_type=("job_type_public", "first"),
                                    reserved_gpu_hours=("used_gpu_hours", "sum"),
                                    rows=("pod_id", "size")))
    table = {
        "observed_pods": int(len(delay)),
        "scene_pods": int(len(jobs)),
        "join_overlap": int(len(set(jobs.pod_id) & set(delay.index))),
        "observed_with_delay": int(delay.schedule_delay_sec.notna().sum()),
    }

    print("1. what the trace's delay looks like, restricted to the pods the scene uses")
    keep = delay.join(jobs.set_index("pod_id")[["priority_class"]], how="inner",
                      rsuffix="_scene")
    cohorts = [("all", keep)] + [(priority, keep[keep.priority == priority])
                                 for priority in ("HP", "LP", "Other")]
    for label, subset in cohorts:
        print("    %-6s %s" % (label, json.dumps(quantiles(subset.schedule_delay_sec.dropna()))))

    print("\n2. is the HP tail about request size? (the probe could only guess)")
    # `shared(<1)` is GPU sharing, and the trace carries no SM utilisation for those
    # pods at all, so this bucket is also exactly the blind spot of section 3 of the
    # probe. Keeping it separate is what makes a claim like "bigger requests wait
    # longer" checkable instead of rhetorical.
    for size, subset in keep.groupby(pd.cut(keep.gpu_request,
                                            [-1, 0.999, 1.5, 2.5, 4.5, 8.5, 100],
                                            labels=["shared(<1)", "1", "2", "3-4", "5-8", ">8"],
                                            ordered=True)):
        for priority in ("HP", "LP"):
            cell = subset[subset.priority == priority]
            print("    %-11s %-3s %s" % (size, priority,
                                         json.dumps(quantiles(cell.schedule_delay_sec.dropna()))))

    print("\n3. simulated wait vs observed schedule delay, per policy")
    from vmagent.gpu.contract import load_nodes, load_workload
    from vmagent.gpu.environment import GpuSchedEnv

    nodes = load_nodes(args.nodes)
    tasks, _ = load_workload(args.jobs)
    tasks = [t for t in tasks if t.pod_id in set(jobs.pod_id)]
    observed_delay = delay.schedule_delay_sec.to_dict()
    table["policies"] = {}
    for policy in args.policies.split(","):
        name, _, queue = policy.partition(":")
        env = GpuSchedEnv(nodes, list(tasks), clock="fast",
                          queue_policy=queue or "backfill", seed=42,
                          cpu_background_fraction=0., checkpoint_interval=3600.,
                          measurement_window=[0., 691200.], max_sim_time=4000000.)
        summary = env.run(name.strip())
        waits = {pid: record["wait_sec"] for pid, record in env.records.items()}
        paired = [(waits[pid], observed_delay[pid]) for pid in set(waits) & set(observed_delay)
                  if observed_delay[pid] == observed_delay[pid]]
        simulated = [a for a, _ in paired]
        actual = [b for _, b in paired]
        by_priority = {}
        for priority in ("HP", "LP", "Other"):
            pods = set(delay.index[delay.priority == priority]) & set(waits)
            by_priority[priority] = {
                "n": len(pods),
                "simulated": quantiles([waits[pid] for pid in pods]),
                "observed": quantiles([observed_delay[pid] for pid in pods
                                       if pid in observed_delay])}
        row = {"queue_policy": queue or "backfill", "preemptions": summary["preemptions"],
               "paired_pods": len(paired),
               "simulated_all": quantiles(simulated), "observed_all": quantiles(actual),
               "ratio_p50": round(float(np.quantile(np.clip(simulated, 0, CEILING), .5) /
                                max(1e-9, np.quantile(np.clip(actual, 0, CEILING), .5))), 3)
                           if simulated and actual else None,
               "ratio_p99": round(float(np.quantile(np.clip(simulated, 0, CEILING), .99) /
                                max(1e-9, np.quantile(np.clip(actual, 0, CEILING), .99))), 3)
                           if simulated and actual else None,
               "by_priority": by_priority}
        table["policies"][name.strip()+(":"+queue if queue else "")] = row
        print("    %-10s paired=%d  simulated %s\n               observed  %s  ratio p50=%s p99=%s"
              % (name, len(paired), json.dumps(row["simulated_all"]),
                 json.dumps(row["observed_all"]), row["ratio_p50"], row["ratio_p99"]))

    print("\n%s" % json.dumps({k: v for k, v in table.items() if k != "policies"},
                              indent=2, ensure_ascii=False))
    print("""
What this can and cannot support. `schedule_delay_sec` is the delay of a pod arriving at
a cluster that was not empty, under a scheduler with admission control, gang semantics
and GPU sharing that this simulator does not model. The engine's `wait_sec` is queueing
in an isolated replay of one 30-node pool. So a ratio near 1 is evidence the model is
not obviously wrong; a ratio far from 1 is a finding only if `join_overlap` and
`observed_with_delay` are high enough for the quantiles to mean anything, which is why
both are printed above the comparison rather than below it.""")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
