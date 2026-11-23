"""Root-cause the residual: replay both engines' occupancy step functions and diff them.

The cross-engine comparison matches on every migration decision metric but leaves
31 of 84 migration cells with `gpu_reserved_allocation_rate` off by up to 2.2e-4, and
zero difference when migration is off. That is small enough to be a rounding story
and specific enough to be a real one, so instead of guessing, this reconstructs the
allocated-GPU-over-time curve from each engine's own event log and reports where they
first disagree.

Her engine exposes occupancy implicitly (start / completion / migration_start /
migration_complete), so the curve is rebuilt from those; the main engine writes
`gpu_reserved` into every event, so its curve is read directly.

    C:/Users/trw/anaconda3/python.exe tools/diff_migration_integral.py \
        --instance .../candidates/seed20261107_interval10/S1 --migration cheap_multi
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BENCHMARK = REPO / "场景/03_碎片整理与十场景/三指标数据评测工作文件_20261108/01_场景数据与构建"


def load_harness():
    spec = importlib.util.spec_from_file_location("benchmark_engine", BENCHMARK/"engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def her_curve(logs, gpu_of, window, initial_gpu):
    """Step function of allocated GPU, from her event log.

    Starts at the prescribed initial layout, not at zero: her engine places the
    initial pods inside State.__init__, so they never emit a start event.
    """
    level, points = initial_gpu, [(0.0, initial_gpu)]
    for event in logs:
        kind = event["event"]
        if kind == "start":
            level += gpu_of[event["pod_id"]]
        elif kind == "completion":
            level -= gpu_of[event["pod_id"]]
        elif kind == "migration_start":
            level += event["gpu"]          # destination reserved, source still held
        elif kind == "migration_complete":
            level -= event["gpu"]          # double booking ends
        elif kind == "planned_move_skipped_completed":
            continue
        else:
            continue
        if event["time_sec"] <= window:
            points.append((event["time_sec"], level))
    points.append((window, level))
    return points


def engine_curve(env_events, window):
    points = [(0.0, 0.0)]
    for event in env_events:
        if event["time_sec"] <= window:
            points.append((event["time_sec"], event["gpu_reserved"]))
    return points


def integrate(points, window):
    total, previous = 0.0, (0.0, 0.0)
    for time, level in points:
        total += previous[1] * (time - previous[0])
        previous = (time, level)
    total += previous[1] * (window - previous[0])
    return total


def level_at(points, moment):
    """Value of a step function just after `moment`.

    Points are appended in event order, so several can share one timestamp - the
    engine's reset event, for example, sits at t=0 right after the synthetic zero.
    Taking the last of them is what "just after" means; max() by time alone would
    return the first and misreport the opening interval as a divergence.
    """
    value = points[0][1]
    for time, level in points:
        if time <= moment:
            value = level
        else:
            break
    return value


def first_divergence(a, b, window):
    """Compare two step functions and return the first interval where they differ."""
    breaks = sorted({t for t, _ in a} | {t for t, _ in b} | {0.0, window})
    for left, right in zip(breaks, breaks[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2.0
        level_a, level_b = level_at(a, mid), level_at(b, mid)
        if abs(level_a - level_b) > 1e-9:
            return left, right, level_a, level_b
    return None


def divergences(a, b, window):
    """Every interval where the two step functions disagree, with its duration."""
    breaks = sorted({t for t, _ in a} | {t for t, _ in b} | {0.0, window})
    rows = []
    for left, right in zip(breaks, breaks[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2.0
        level_a, level_b = level_at(a, mid), level_at(b, mid)
        if abs(level_a - level_b) > 1e-9:
            rows.append((left, right, right-left, level_a, level_b))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path,
                        default=BENCHMARK/"candidates/seed20261107_interval10")
    parser.add_argument("--instance", default="S1")
    parser.add_argument("--placement", default="first_fit")
    parser.add_argument("--migration", default="cheap_multi")
    args = parser.parse_args()

    instance = Path(args.candidates)/args.instance
    import pandas as pd
    servers = pd.read_parquet(instance/"servers.parquet").to_dict("records")
    initial = pd.read_parquet(instance/"initial_pods.parquet").to_dict("records")
    arrivals = pd.read_parquet(instance/"arrivals.parquet").to_dict("records")
    config = json.loads((instance/"config.json").read_text(encoding="utf-8"))
    window = config["allocation_window_sec"]
    gpu_of = {row["pod_id"]: float(row["gpu_request"]) for row in initial + arrivals}

    engine = load_harness()
    metrics, logs = engine.simulate(servers, initial, arrivals, config, args.placement, args.migration)

    converted = REPO/"outputs/benchmark_csv"/Path(args.candidates).name/args.instance/"inputs"
    from vmagent.gpu.contract import load_nodes, load_initial_placement, load_workload
    from vmagent.gpu.environment import GpuSchedEnv
    nodes = load_nodes(converted/"nodes.csv")
    tasks = load_workload(converted/"jobs.csv")[0]
    placement = load_initial_placement(converted/"initial_placement.csv")
    env = GpuSchedEnv(nodes, tasks, clock="fast", queue_policy="backfill", seed=42,
                      measurement_window=[0., window], max_sim_time=config["horizon_sec"],
                      initial_placement=placement,
                      migration=args.migration,
                      migration_max_plan_moves=config["migration"]["max_plan_moves"],
                      migration_max_total_moves=config["migration"]["max_total_moves"])
    summary = env.run(args.placement)

    a, b = her_curve(logs, gpu_of, window, sum(gpu_of[row["pod_id"]] for row in initial)), \
        engine_curve(env.events, window)
    area_a, area_b = integrate(a, window), integrate(b, window)
    capacity = sum(float(row["gpu_count"]) for row in servers)
    print("benchmark gpu_reserved_allocation_rate %.15f" % metrics["gpu_reserved_allocation_rate"])
    print("rebuilt from her log                   %.15f" % (area_a/(capacity*window)))
    print("main engine summary                    %.15f" % summary["reserved_gpu_ratio"])
    print("rebuilt from engine events             %.15f" % (area_b/(capacity*window)))
    print("difference                             %.3e GPU·s = %.6f GPU·h"
          % (area_a-area_b, (area_a-area_b)/3600))
    gap = divergences(a, b, window)
    if gap:
        print("%d divergent interval(s), total %.4f GPU·s:" % (len(gap), sum(r[2]*(r[3]-r[4]) for r in gap)))
        for left, right, width, level_a, level_b in gap[:12]:
            print("        [%12.6f, %12.6f)  %8.4f s  benchmark=%6.1f engine=%6.1f  delta=%+5.1f GPU  %8.4f GPU·s"
                  % (left, right, width, level_a, level_b, level_a-level_b, width*(level_a-level_b)))
        if len(gap) > 12:
            print("        ... %d more" % (len(gap)-12))
    else:
        print("curves agree at every interval; the difference is in the denominator")
    print("her migration_count=%d engine=%d; her inflight_at_horizon=%s engine=%s"
          % (metrics["migration_count"], summary["migration"]["count"],
             metrics["migration_inflight_at_horizon"], summary["migration"]["inflight_at_horizon"]))


if __name__ == "__main__":
    main()
