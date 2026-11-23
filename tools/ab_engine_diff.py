"""Prove two engine implementations produce identical results on identical input.

Built for M3 (environment.py split into seven modules) and reusable for M6 (columnar
data structures) and M10 (absorbing the collaborator's defragmentation engine).

The refactor's acceptance bar is "no number moves", and comparing against a *recorded*
summary is not enough: recordings can predate the code, and they only cover the fields
someone thought to save. So this runs a snapshot of the old implementation and the
current one over the same loaded tasks and compares everything the engine can emit —
summary(), the per-task record table, and the full event trace.

    python tools/ab_engine_diff.py                       # every cell, quick scenes
    python tools/ab_engine_diff.py --with-scene-f        # add the 119,495-task cells
    python tools/ab_engine_diff.py --cell F-first_fit-fcfs

`--before` names a module file exposing GpuSchedEnv; the current one is the default
comparison target. A git snapshot works as long as the file has no relative imports.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu.contract import load_nodes, load_workload
from vmagent.gpu.environment import GpuSchedEnv as NewEnv
from vmagent.gpu.run import make_environment

POLICIES = ("first_fit", "best_fit", "worst_fit", "random", "spotgpu")
QUEUES = ("fcfs", "backfill")

SCENES = {
    "scene_A": (REPO/"场景/05_场景构建与选型/server_selection/servers_20_H20.csv",
                REPO/"场景/05_场景构建与选型/server_selection/real_pod_sequences/scene_A_timing/scene_A_50_pods_timed.json",
                {"measurement_window": [0., 120.], "max_sim_time": 150000.}),
    "spotgpu_demo": (REPO/"场景/04_演示场景/spotgpu_demo_20261121/inputs/nodes.csv",
                     REPO/"场景/04_演示场景/spotgpu_demo_20261121/inputs/jobs_spotgpu_demo.csv",
                     {"measurement_window": [86400., 129600.], "max_sim_time": 400000.}),
    "F_mini": (REPO/"场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv",
               REPO/"场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv",
               {"measurement_window": [0., 691200.], "max_sim_time": 4000000., "limit": 4000}),
    "F_full": (REPO/"场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv",
               REPO/"场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv",
               {"measurement_window": [0., 691200.], "max_sim_time": 4000000.}),
    # The five-scene suite is the only input with CPU-only pods and permanent
    # rejections, so it exercises _cpu_wait_candidate and _rejection_reason, which
    # the GPU-only scenes never reach.
    "allpods_A": (REPO/"场景/01_五场景A_E/allpods_20261025/inputs/nodes.csv",
                  REPO/"场景/01_五场景A_E/allpods_20261025/inputs/jobs_A.csv",
                  {"measurement_window": [0., 21600.], "max_sim_time": 200000.}),
    "allpods_C": (REPO/"场景/01_五场景A_E/allpods_20261025/inputs/nodes.csv",
                  REPO/"场景/01_五场景A_E/allpods_20261025/inputs/jobs_C.csv",
                  {"measurement_window": [0., 21600.], "max_sim_time": 200000.}),
    "allpods_D": (REPO/"场景/01_五场景A_E/allpods_20261025/inputs/nodes.csv",
                  REPO/"场景/01_五场景A_E/allpods_20261025/inputs/jobs_D.csv",
                  {"measurement_window": [0., 21600.], "max_sim_time": 200000.}),
    # A background fraction shrinks every node's usable CPU, which changes both the
    # fit tests and the clamping in release(); it is not covered by any other scene.
    "allpods_D_bg30": (REPO/"场景/01_五场景A_E/allpods_20261025/inputs/nodes.csv",
                       REPO/"场景/01_五场景A_E/allpods_20261025/inputs/jobs_D.csv",
                       {"measurement_window": [0., 21600.], "max_sim_time": 200000.,
                        "cpu_background_fraction": 0.3}),
}


def load_module(path):
    spec = importlib.util.spec_from_file_location("engine_before", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(env_class, nodes, tasks, options, policy, queue_policy):
    settings = {"clock": "fast", "queue_policy": queue_policy, "seed": 42,
                "cpu_background_fraction": 0., "checkpoint_interval": 3600.}
    settings.update(options)
    env = env_class(nodes, tasks, **settings)
    env.run(policy)
    result = {"summary": env.summary(), "records": list(env.records.values()),
              "events": list(env.events)}
    del env
    return result


def diff(a, b, path=""):
    """Yield (path, kind) mismatches over dicts, lists and scalars."""
    if path.endswith("wall_time_sec"):
        return  # measured clock cost of the run itself, not a result
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                yield ("%s.%s" % (path, key), "presence")
            else:
                for row in diff(a[key], b[key], "%s.%s" % (path, key)):
                    yield row
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (path, "length %d vs %d" % (len(a), len(b)))
            return
        for index, (left, right) in enumerate(zip(a, b)):
            for row in diff(left, right, "%s[%d]" % (path, index)):
                yield row
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            yield (path, "bool")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # Exact, not tolerant: this is an equivalence check between two code paths
        # over the same float operations, so any difference at all is a behaviour
        # change. Tolerance belongs in golden comparisons, not here.
        if a != b:
            yield (path, "%r vs %r" % (a, b))
        return
    if a != b:
        yield (path, "%r vs %r" % (a, b))


def cells(scenes, policies, queues):
    for scene in scenes:
        for policy in policies:
            for queue in queues:
                yield scene, policy, queue


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True, help="module file with the old GpuSchedEnv")
    parser.add_argument("--scene", action="append", default=None,
                        help="scene name, repeatable (default: all except F_full)")
    parser.add_argument("--cell", default=None, help="run only '<scene>-<policy>-<queue>'")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--queues", default=",".join(QUEUES))
    args = parser.parse_args()

    before = load_module(args.before).GpuSchedEnv
    scenes = args.scene or [name for name in SCENES if name != "F_full"]
    policies = args.policies.split(",")
    queues = args.queues.split(",")

    cache, failures, checked = {}, [], 0
    for scene, policy, queue in cells(scenes, policies, queues):
        if args.cell and args.cell != "%s-%s-%s" % (scene, policy, queue):
            continue
        if scene not in cache:
            nodes_path, tasks_path, options = SCENES[scene]
            nodes = load_nodes(nodes_path)
            tasks = load_workload(tasks_path)[0]
            if options.get("limit"):
                tasks = tasks[:options["limit"]]
            cache[scene] = (nodes, tasks, {k: v for k, v in options.items() if k != "limit"})
        nodes, tasks, options = cache[scene]
        name = "%s/%s/%s" % (scene, policy, queue)
        reference = diff_before = run(before, nodes, tasks, options, policy, queue)
        candidate = run(NewEnv, nodes, tasks, options, policy, queue)
        rows = list(diff(reference, candidate))
        checked += 1
        if rows:
            failures.append((name, rows))
            print("[DIFF] %-34s %d field(s), first: %s = %s" % (name, len(rows), rows[0][0], rows[0][1]))
        else:
            print("[same] %-34s summary+%d records+%d events identical"
                  % (name, len(candidate["records"]), len(candidate["events"])))
        del reference, candidate
    print("\n%d/%d cells identical between the two engines" % (checked-len(failures), checked))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
