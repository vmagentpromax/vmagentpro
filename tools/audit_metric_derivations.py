"""Audit the independent metric re-derivations against the engine's own state.

M1 is about converging the metric code scattered across the repository, but before
moving anything this checks whether the scattered re-derivations actually agree with
the engine, and what precondition each one silently relies on. Two of them matter:

- `docs/server_selection/run_request_scenes.py:metrics()` rebuilds per-node free
  capacity from `records[pid]["server_id"]` for everything in its reconstructed
  `running` set. That is only sound while a task never changes node. Preemption
  rewrites `server_id`, so the function is preconditioned on no preemption - and the
  script's own conservation assert is what catches a violation, loudly.
- `experiments/preemption_20261107/harness.py:evaluate()` says so in its docstring
  ("never use overwritten start_time or flawed run totals") and instead rebuilds
  run segments from the event log. It is the correct pattern for a preempting engine.

The third re-derivation, `harness.evaluate()`, is included for contrast because it
covers the same quantity and asserts its own agreement with the engine. Comparing all
three on one run is what turns "these two functions differ" into a statement about
which primitive is safe to keep.

    C:/Users/trw/anaconda3/python.exe tools/audit_metric_derivations.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs/server_selection"))

from vmagent.gpu.contract import load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv

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


def rederived_gpu_seconds(env, metrics_fn):
    """What run_request_scenes.metrics() integrates, in absolute GPU-seconds."""
    result = metrics_fn(env, env.measure_start)
    return result["agu_reserved"] * env.total_gpu * result["makespan_sec"], result


def main():
    from run_request_scenes import metrics as rederive
    sys.path.insert(0, str(REPO / "experiments/preemption_20261107"))
    from harness import evaluate

    # Each table compares a re-derivation against the engine integral covering the same
    # span. metrics() integrates over [0, makespan], so its reference is gpu_area_full;
    # evaluate() integrates over the measurement window, so its reference is gpu_area.
    # An earlier version of this audit mixed the two and reported a 5% divergence on a
    # scenario F slice with zero preemptions, where nothing was actually wrong.
    print("A. run_request_scenes.metrics() over [0, makespan]  vs  env.gpu_area_full")
    print("%-14s %-12s %18s %18s %10s" % ("scene", "policy", "engine GPU-s", "re-derived GPU-s", "verdict"))
    rows = []
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
            truth = env.gpu_area_full
            try:
                derived, _ = rederived_gpu_seconds(env, rederive)
            except AssertionError as exc:
                derived = float("nan")
                print("%-14s %-12s %18.3f %18s   CONSERVATION ASSERT FIRED (%s)"
                      % (name, policy, truth, "n/a", str(exc)[:40]))
                continue
            gap = derived - truth
            verdict = "agree" if abs(gap) <= max(1e-6, abs(truth) * 1e-9) else "DIVERGES"
            print("%-14s %-12s %18.3f %18.3f %10s  gap=%+.3f GPU-s  preemptions=%d"
                  % (name, policy, truth, derived, verdict, gap, summary["preemptions"]))
            rows.append((name, policy, env, summary))

    print("\nB. harness.evaluate() over the measurement window  vs  env.gpu_area")
    print("   (evaluate() asserts this equality itself; a fired assert is the finding)")
    print("%-14s %-12s %18s %18s %10s" % ("scene", "policy", "engine GPU-s", "segments GPU-s", "verdict"))
    for name, policy, env, summary in rows:
        try:
            result, _ = evaluate(env)
            derived = result["allocated_gpu_ratio"] * env.total_gpu * (env.measure_end - env.measure_start)
        except AssertionError as exc:
            print("%-14s %-12s %18.3f %18s   ASSERT FIRED (%s)"
                  % (name, policy, env.gpu_area, "n/a", str(exc)[:40]))
            continue
        gap = derived - env.gpu_area
        print("%-14s %-12s %18.3f %18.3f %10s  gap=%+.6f GPU-s  preemptions=%d"
              % (name, policy, env.gpu_area, derived,
                 "agree" if abs(gap) <= max(1e-5, env.gpu_area * 1e-10) else "DIVERGES",
                 gap, summary["preemptions"]))
    print("""
Reading: the two re-derivations are not equally trustworthy, and the difference is a
real trap - the same words, "reserved GPU-seconds", computed two ways, one of which
breaks under the feature the project added most recently.

- A is exact wherever no task changes node and over-counts under SpotGPU. It rebuilds
  per-node free capacity from records[pid]["server_id"], which preemption rewrites, and
  it keeps evicted tasks in its reconstructed `running` set, so the work a preempted
  task re-executes is attributed to the wrong interval. run_request_scenes.py survives
  today only because POLICIES contains no preempting policy; its conservation assert is
  the tripwire, and it would fire the moment that changes.
- B is exact in both cases, including with preemptions, because it rebuilds run segments
  from the event log instead of from mutated per-task state. It costs an assert per
  call, so a disagreement cannot be silently shipped.

Consequence for M1: converging this code into vmagent/gpu/metrics.py must not reuse the
server_id shortcut. harness.evaluate()'s event-segment reconstruction is the primitive
that should move into the shared module, and A should be deleted rather than ported.""")


if __name__ == "__main__":
    main()
