"""Audit the four ways this repository computes "GPU-seconds reserved".

M1 is about converging the metric code scattered across the repository. This is the
measurement that decided how: the same quantity, computed from the same run by four
different pieces of code, does not come out the same.

A. `legacy_rederivation` - a transcription of what
   `docs/server_selection/run_request_scenes.py:metrics()` did until this audit fixed
   it. It rebuilds per-node free capacity from `records[pid]["server_id"]` for
   everything in its reconstructed `running` set. Two silent assumptions: a task never
   changes node, and an evicted task stops running. Preemption breaks both - it
   rewrites `server_id` and leaves the victim in `running`. Kept here so the finding
   stays executable after the code it describes is gone.
B. `harness.evaluate()` from `experiments/preemption_20261107/` says so in its own
   docstring ("never use overwritten start_time or flawed run totals") and rebuilds run
   segments from the event log. It asserts its agreement with the engine on every call.
C. `metrics()` today: the aggregate integrals come from the event trace through
   `vmagent.gpu.metrics.integrate_events`, and the per-node half that still needs the
   old reading now refuses to run rather than quietly returning a wrong number.
D. `integrate_events` itself, the primitive that A should have been using.

One trap worth naming, because the first cut of this audit fell into it: an integral has
to be compared against the engine value covering the same span. `env.gpu_area` is clipped
to the measurement window and `env.gpu_area_full` is not, and comparing a full-span
re-derivation against the windowed one reported a 5% divergence on a scenario F slice
with zero preemptions, where nothing was wrong at all. Each table below therefore names
the span on both sides.

    C:/Users/trw/anaconda3/python.exe tools/audit_metric_derivations.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs/server_selection"))

from vmagent.gpu.contract import load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv
from vmagent.gpu.metrics import integrate_events, run_segments

SCENES = {
    "scene_A": (REPO/"docs/server_selection/servers_20_H20.csv",
                REPO/"docs/server_selection/real_pod_sequences/scene_A_timing/scene_A_50_pods_timed.json",
                {"measurement_window": [0., 120.], "max_sim_time": 150000.}),
    "spotgpu_demo": (REPO/"runs/spotgpu_demo_20261121/inputs/nodes.csv",
                     REPO/"runs/spotgpu_demo_20261121/inputs/jobs_spotgpu_demo.csv",
                     {"measurement_window": [86400., 129600.], "max_sim_time": 400000.}),
    "F_slice_8k": (REPO/"runs/hetero8d_20261101/inputs/nodes.csv",
                    REPO/"runs/hetero8d_20261101/inputs/jobs_F.csv",
                    {"measurement_window": [0., 691200.], "max_sim_time": 4000000., "limit": 8000}),
}


def legacy_rederivation(env):
    """GPU-seconds reserved, the way metrics() computed it before this audit.

    Faithful to the original, including the part that is wrong: `preempt` is not one of
    the handled kinds, so an evicted task stays in `running` and goes on being charged
    to whichever node it lands on next.
    """
    running, area = set(), 0.
    ledger = env.records
    for index, event in enumerate(env.events[:-1]):
        kind, pid = event["event"], event["pod_id"]
        if kind == "start":
            running.add(pid)
        elif kind in ("complete", "reject"):
            running.discard(pid)
        left, right = event["time_sec"], env.events[index+1]["time_sec"]
        if right <= left:
            continue
        free = {n.server_id: n.gpu_capacity for n in env.nodes}
        for rid in running:
            free[ledger[rid]["server_id"]] -= ledger[rid]["gpu_request"]
        area += (env.total_gpu-sum(free.values()))*(right-left)
    return area


def table(title, basis, compute, reference, runs):
    print(title)
    print("%-14s %-12s %18s %18s %10s" % ("scene", "policy", "engine GPU-s", basis, "verdict"))
    for name, policy, env, preemptions in runs:
        truth = reference(env)
        try:
            derived = compute(env)
        except Exception:
            print("%-14s %-12s %18.3f %18s   REFUSES" % (name, policy, truth, "n/a"))
            continue
        gap = derived-truth
        verdict = "agree" if abs(gap) <= max(1e-5, abs(truth)*1e-9) else "DIVERGES"
        print("%-14s %-12s %18.3f %18.3f %10s  gap=%+.6f GPU-s  preemptions=%d"
              % (name, policy, truth, derived, verdict, gap, preemptions))
    print()


def main():
    from run_request_scenes import metrics
    sys.path.insert(0, str(REPO / "experiments/preemption_20261107"))
    from harness import evaluate

    runs = []
    for name, (nodes_path, tasks_path, options) in SCENES.items():
        options = dict(options)
        tasks = list(load_workload(tasks_path)[0])
        limit = options.pop("limit", None)
        if limit:
            tasks = tasks[:limit]
        for policy in ("first_fit", "spotgpu"):
            env = GpuSchedEnv(load_nodes(nodes_path), list(tasks), clock="fast",
                              queue_policy="backfill", seed=42, cpu_background_fraction=0.,
                              checkpoint_interval=3600., **options)
            summary = env.run(policy)
            runs.append((name, policy, env, summary["preemptions"]))

    table("A. the shortcut metrics() used to take, over [0, makespan], vs env.gpu_area_full",
          "legacy GPU-s", legacy_rederivation, lambda env: env.gpu_area_full, runs)
    table("B. harness.evaluate() over the measurement window, vs env.gpu_area",
          "segments GPU-s",
          lambda env: evaluate(env)[0]["allocated_gpu_ratio"]*env.total_gpu*(env.measure_end-env.measure_start),
          lambda env: env.gpu_area, runs)
    table("C. metrics() today, over [0, makespan], vs env.gpu_area_full",
          "metrics() GPU-s",
          lambda env: metrics(env, env.measure_start)["agu_reserved"]*env.total_gpu*env.current_time,
          lambda env: env.gpu_area_full, runs)
    table("D. integrate_events over the measurement window, vs env.gpu_area",
          "events GPU-s",
          lambda env: integrate_events(env.events, "gpu_reserved",
                                       (env.measure_start, env.measure_end)),
          lambda env: env.gpu_area, runs)

    print("Segment counts, which is where the shortcut loses the plot:")
    for name, policy, env, preemptions in runs:
        segments = run_segments(env.events)
        multi = sum(1 for parts in segments.values() if len(parts) > 1)
        print("    %-14s %-12s pods=%-6d segments=%-6d pods that ran in >1 interval=%-3d "
              "preemptions=%d" % (name, policy, len(segments),
                                  sum(len(p) for p in segments.values()), multi, preemptions))
    print("""
Reading: A is the number the ten-scenario tables would have carried the moment a
preempting policy was added to them, and it is wrong without anything complaining. B and
D are the same idea in two places - rebuild intervals from the event log, because
`_record` snapshots the whole ledger while that snapshot is true, so nothing happening
afterwards can corrupt it. C is what metrics() does now: correct where it can be, and
loud where it cannot.

Consequence for M1, and what this changes: `run_segments` and `integrate_events` live in
vmagent/gpu/metrics.py, run_request_scenes.metrics() integrates through them, and the
server_id shortcut is deleted rather than ported.""")


if __name__ == "__main__":
    main()
