"""Check that the engine's SpotGPU knobs reproduce the five experiment policies exactly.

`experiments/preemption_20261107/harness.py` built those policies by subclassing the
engine and overriding three methods. `vmagent/gpu/spotgpu.py` says the same five are
just points in a six-knob space. That claim is only worth anything if running the
engine with the knob values gives the identical event trace, so this runs both sides
on the same workload and diffs summary, records and events.

    python tools/verify_spotgpu_variants.py                       # 40k-task slice
    python tools/verify_spotgpu_variants.py --limit 119495        # full scenario F
    python tools/verify_spotgpu_variants.py --variant paper_cost  # one of them
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"experiments/preemption_20261107"))

from vmagent.gpu.contract import load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv
from vmagent.gpu.spotgpu import VARIANTS, resolve
from harness import ExperimentEnv


def run(env, experiment=False):
    # The harness sets preempt_enabled inside run_experiment(); calling the base
    # run("spotgpu") on it would silently re-enable preemption and make no_preempt
    # diverge for a reason that has nothing to do with the knobs.
    summary = env.run_experiment() if experiment else env.run("spotgpu")
    return {"summary": summary, "records": list(env.records.values()),
            "events": list(env.events)}


def diff(a, b, path=""):
    if path.endswith("wall_time_sec"):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield ("%s.%s" % (path, key), a.get(key, "<absent>"), b.get(key, "<absent>"))
            else:
                for row in diff(a[key], b[key], "%s.%s" % (path, key)):
                    yield row
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (path, len(a), len(b))
            return
        for index, (left, right) in enumerate(zip(a, b)):
            for row in diff(left, right, "%s[%d]" % (path, index)):
                yield row
        return
    if a != b:
        yield (path, a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv")
    parser.add_argument("--tasks", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv")
    parser.add_argument("--limit", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", action="append", default=None)
    args = parser.parse_args()

    nodes = load_nodes(args.nodes)
    tasks = load_workload(args.tasks)[0][:args.limit]
    settings = dict(clock="fast", queue_policy="backfill", seed=args.seed,
                    measurement_window=[0., 691200.], max_sim_time=4000000.)
    wanted = args.variant or list(VARIANTS)
    failures = 0
    for name in wanted:
        knobs = resolve(VARIANTS[name])
        reference = run(ExperimentEnv(nodes, list(tasks), variant=name, **settings), experiment=True)
        candidate = run(GpuSchedEnv(nodes, list(tasks), spotgpu=VARIANTS[name], **settings))
        rows = list(diff(reference, candidate))
        failures += 1 if rows else 0
        summary = candidate["summary"]
        print("[%s] %-16s preemptions=%-4d wasted=%-10.1f reserved=%.12f  %s"
              % ("same" if not rows else "DIFF", name, summary["preemptions"],
                 summary["wasted_gpu_sec"], summary["reserved_gpu_ratio"],
                 "engine knobs reproduce the harness exactly"
                 if not rows else "%d field(s), first %s: %r vs %r" % (len(rows), rows[0][0], rows[0][1], rows[0][2])))
    print("\n%d/%d variants reproduced by the knob settings" % (len(wanted)-failures, len(wanted)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
