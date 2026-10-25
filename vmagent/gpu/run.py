"""Run a GPU trace experiment: python -m vmagent.gpu.run --help."""
import argparse
import json
from pathlib import Path

from .environment import DEFAULT_NODES, DEFAULT_TASKS, ROOT, GpuSchedEnv, load_nodes, load_workload


def make_environment(nodes_path, tasks_path, **options):
    nodes = load_nodes(nodes_path)
    tasks, metadata = load_workload(tasks_path)
    window = options.pop("measurement_window", metadata.get("measurement_window_sec", [0., 120.]))
    for key, actual in [("gpu_capacity", sum(n.gpu_capacity for n in nodes)),
                        ("cpu_capacity_cores", sum(n.cpu_capacity for n in nodes))]:
        if key in metadata and abs(float(metadata[key])-actual) > 1e-7:
            raise ValueError("Task JSON {} disagrees with node inventory".format(key))
    if float(metadata.get("synthetic_cpu_background_cores", 0)) != 0:
        raise ValueError("Use cpu_background_fraction explicitly; nonzero background in task JSON is not silently applied")
    return GpuSchedEnv(nodes, tasks, measurement_window=window, **options), metadata


def main():
    parser = argparse.ArgumentParser(description="Timed H20/CPU resource scheduling")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--policy", choices=["first_fit", "best_fit", "worst_fit", "random"], default="first_fit")
    parser.add_argument("--clock", choices=["realtime", "fast"], default="realtime")
    parser.add_argument("--speed", type=float, default=1., help="realtime simulated seconds per wall-clock second")
    parser.add_argument("--queue-policy", choices=["backfill", "fcfs"], default="backfill")
    parser.add_argument("--measurement-seconds", type=float, help="override measurement window to [0, seconds]")
    parser.add_argument("--max-sim-time", type=float, default=600.)
    parser.add_argument("--cpu-background-fraction", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/gpu/latest")
    args = parser.parse_args()
    options = dict(clock=args.clock, speed=args.speed, queue_policy=args.queue_policy,
                   max_sim_time=args.max_sim_time, cpu_background_fraction=args.cpu_background_fraction, seed=args.seed)
    if args.measurement_seconds is not None:
        options["measurement_window"] = [0., args.measurement_seconds]
    try:
        env, metadata = make_environment(args.nodes, args.tasks, **options)
        print("Loaded {} nodes, {} unique pods; clock={}, policy={}".format(len(env.nodes), len(env.tasks), args.clock, args.policy), flush=True)
        try:
            env.run(args.policy)
        except KeyboardInterrupt:
            env.stop_requested.set()
            with env.lock:
                env._finish("cancelled")
        summary = env.export(args.output)
        manifest = {"nodes": str(args.nodes.resolve()), "tasks": str(args.tasks.resolve()),
                    "policy": args.policy, "queue_policy": args.queue_policy,
                    "workload_metadata": metadata, "seed": args.seed}
        (args.output/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Results: " + str(args.output.resolve()))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
