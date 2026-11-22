"""Run a GPU trace experiment: python -m vmagent.gpu.run --help."""
import argparse
import json
import platform
import sys
from pathlib import Path

from .environment import DEFAULT_NODES, DEFAULT_TASKS, ROOT, GpuSchedEnv, load_initial_placement, load_nodes, load_workload
from .spotgpu import CPU_AWARE_MODES, VARIANTS as SPOTGPU_VARIANTS, VICTIM_PREFIXES, VICTIM_SELECTIONS


def environment_stamp():
    """Record what produced a run.

    CPython 3.12 switched float sum() to Neumaier compensated summation, so the
    last 1-2 ULP of any aggregate differs between interpreters on identical code
    and identical input. Measured: per_model.XPU-A.offered_gpu_ratio is
    1.040616417218244 under 3.11.11 and 1.0406164172182435 under 3.13.14. Without
    this stamp a golden diff across environments looks like a regression.
    """
    return {"python": sys.version.split()[0], "python_implementation": platform.python_implementation(),
            "platform": platform.platform()}


def make_environment(nodes_path, tasks_path, initial_placement=None, **options):
    nodes = load_nodes(nodes_path)
    tasks, metadata = load_workload(tasks_path)
    placement = load_initial_placement(initial_placement) if initial_placement else None
    window = options.pop("measurement_window", metadata.get("measurement_window_sec", [0., 120.]))
    for key, actual in [("gpu_capacity", sum(n.gpu_capacity for n in nodes)),
                        ("cpu_capacity_cores", sum(n.cpu_capacity for n in nodes))]:
        if key in metadata and abs(float(metadata[key])-actual) > 1e-7:
            raise ValueError("Task JSON {} disagrees with node inventory".format(key))
    if float(metadata.get("synthetic_cpu_background_cores", 0)) != 0:
        raise ValueError("Use cpu_background_fraction explicitly; nonzero background in task JSON is not silently applied")
    return GpuSchedEnv(nodes, tasks, measurement_window=window, initial_placement=placement,
                       **options), metadata


def main():
    parser = argparse.ArgumentParser(description="Timed H20/CPU resource scheduling")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--initial-placement", type=Path, default=None,
                        help="CSV/JSON of {pod_id, server_id} already running at t=0; the policy "
                             "does not get to re-place these, which is what makes the "
                             "defragmentation benchmark's starting state survive")
    parser.add_argument("--policy", choices=["first_fit", "best_fit", "worst_fit", "random", "spotgpu"], default="first_fit")
    parser.add_argument("--clock", choices=["realtime", "fast"], default="realtime")
    parser.add_argument("--speed", type=float, default=1., help="realtime simulated seconds per wall-clock second")
    parser.add_argument("--queue-policy", choices=["backfill", "fcfs"], default="backfill")
    parser.add_argument("--measurement-seconds", type=float,
                        help="measurement window length, starting at --measurement-start-seconds")
    parser.add_argument("--measurement-start-seconds", type=float, default=0.,
                        help="measurement window start; scenes whose pods arrive late (for example the "
                             "SpotGPU demo at ~87,000 s) need a nonzero start or every ratio reads 0%")
    parser.add_argument("--max-sim-time", type=float, default=600.)
    parser.add_argument("--cpu-background-fraction", type=float, default=0.)
    parser.add_argument("--checkpoint-interval", type=float, default=3600.,
                        help="SpotGPU checkpoint interval in simulated seconds (default 3600)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--migration", choices=["none", "one_move", "cheap_multi", "first_multi"],
                        default="none", help="reactive defragmentation mode (default off)")
    parser.add_argument("--migration-max-plan-moves", type=int, default=6)
    parser.add_argument("--migration-max-total-moves", type=int, default=80)
    parser.add_argument("--migration-base-sec", type=float, default=20.)
    parser.add_argument("--migration-per-gpu-sec", type=float, default=8.)
    parser.add_argument("--migration-per-cpu-core-sec", type=float, default=0.1)
    parser.add_argument("--release-source-when", choices=["end", "start"], default="end",
                        help="'end' is make-before-break with the source held through the "
                             "pause, which is what the benchmark archive was produced with")
    parser.add_argument("--no-pause-service", action="store_false", dest="pause_service",
                        help="do not extend the task's finish time by the migration pause")
    parser.add_argument("--no-block-admission", action="store_false", dest="block_admission",
                        help="keep admitting other tasks while a plan is executing")
    parser.add_argument("--no-reserve-destination", action="store_false", dest="reserve_destination",
                        help="break before make: book the destination only when the move lands")
    parser.add_argument("--spotgpu-variant", choices=sorted(SPOTGPU_VARIANTS), default=None,
                        help="load one of the five experiment policies as a knob bundle; "
                             "explicit --spotgpu-* flags below override it")
    parser.add_argument("--spotgpu-victim-selection", choices=VICTIM_SELECTIONS, default=None)
    parser.add_argument("--spotgpu-victim-prefix", choices=VICTIM_PREFIXES, default=None,
                        help="'gpu' is OSDI'26's literal prefix test sum(G_t)>=G_hp; "
                             "'joint' is this engine's historical behaviour and the stricter one")
    parser.add_argument("--spotgpu-hp-first-queue", action="store_true", default=None)
    parser.add_argument("--spotgpu-placement-terms", default=None,
                        help="comma separated prefix of packing,co_locate,history")
    parser.add_argument("--spotgpu-cpu-aware", choices=CPU_AWARE_MODES, default=None,
                        help="add CPU leftover to the packing term; 'none' is the paper")
    parser.add_argument("--no-spotgpu-preemption", action="store_false", dest="spotgpu_preemption",
                        default=None, help="keep SpotGPU's placement ranking but never evict")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/gpu/latest")
    args = parser.parse_args()
    spotgpu = dict(SPOTGPU_VARIANTS[args.spotgpu_variant]) if args.spotgpu_variant else {}
    for key, value in [("victim_selection", args.spotgpu_victim_selection),
                       ("victim_prefix", args.spotgpu_victim_prefix),
                       ("hp_first_queue", args.spotgpu_hp_first_queue or None),
                       ("placement_cpu_aware", args.spotgpu_cpu_aware),
                       ("preemption", args.spotgpu_preemption if args.spotgpu_preemption is not None else None)]:
        if value is not None:
            spotgpu[key] = value
    if args.spotgpu_placement_terms is not None:
        spotgpu["placement_terms"] = tuple(part.strip() for part in args.spotgpu_placement_terms.split(",") if part.strip())
    options = dict(clock=args.clock, speed=args.speed, queue_policy=args.queue_policy,
                   max_sim_time=args.max_sim_time, cpu_background_fraction=args.cpu_background_fraction,
                   checkpoint_interval=args.checkpoint_interval, seed=args.seed, spotgpu=spotgpu,
                   migration=args.migration, migration_max_plan_moves=args.migration_max_plan_moves,
                   migration_max_total_moves=args.migration_max_total_moves,
                   migration_base_sec=args.migration_base_sec,
                   migration_per_gpu_sec=args.migration_per_gpu_sec,
                   migration_per_cpu_core_sec=args.migration_per_cpu_core_sec,
                   release_source_when=args.release_source_when, pause_service=args.pause_service,
                   block_admission_during_plan=args.block_admission,
                   reserve_destination=args.reserve_destination)
    if args.measurement_seconds is not None:
        options["measurement_window"] = [args.measurement_start_seconds,
                                         args.measurement_start_seconds+args.measurement_seconds]
    try:
        env, metadata = make_environment(args.nodes, args.tasks,
                                         initial_placement=args.initial_placement, **options)
        print("Loaded {} nodes, {} unique pods; clock={}, policy={}".format(len(env.nodes), len(env.tasks), args.clock, args.policy), flush=True)
        try:
            env.run(args.policy)
        except KeyboardInterrupt:
            env.stop_requested.set()
            with env.lock:
                env._finish("cancelled")
        summary = env.export(args.output)
        manifest = {"nodes": str(args.nodes.resolve()), "tasks": str(args.tasks.resolve()),
                    "initial_placement": str(args.initial_placement.resolve()) if args.initial_placement else None,
                    "policy": args.policy, "queue_policy": args.queue_policy,
                    "workload_metadata": metadata, "seed": args.seed,
                    "environment": environment_stamp()}
        (args.output/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Results: " + str(args.output.resolve()))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
