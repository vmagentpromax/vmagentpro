"""Re-run recorded golden runs from their own manifests and diff every field.

The project's safety practice is "change the engine, then prove the numbers did not
move". Until now that proof was written by hand in each commit message, so it could
not be repeated by anyone else. This tool makes it a command.

Each golden directory holds summary.json plus the run_manifest.json that produced it,
and the shared run parameters live in metrics/provenance.json. Inputs are taken from
the in-repo copies, not the D: drive paths the manifests still name.

    python tools/verify_golden_runs.py                      # all 14 scenario F runs
    python tools/verify_golden_runs.py --only fcfs          # the five fcfs cells
    python tools/verify_golden_runs.py --root runs/allpods_20261025/metrics

Fields the reference predates (fragmentation, topology, block_classification) are
reported as additions rather than differences; a field the reference has and the
candidate lacks is always a failure. wall_time_sec and realtime sim_time_sec are
excluded because they legitimately track machine load.

**Two of the fourteen runs cannot match their archive, and saying so is the point of
the second reference directory.** `F-spotgpu-fcfs` and `F-spotgpu-backfill` were
recorded on 2026-11-01, a week before SpotGPU preemption existed in the engine
(`d20f8b1`), and the implementation that produced them sorted victims over a set of
pod_id strings - so ties came out differently per process until `2315feb` replaced that
with a `(cost, pod_id)` total order. Their archived numbers are a single draw from a
flipped coin; `runs/hetero8d_20261101/README.md` documents the spread it measured
(232 vs 242 preemptions, GPU reservation 3.2 points apart). So a bare DIFF on those two
cells is expected, and expecting it is worthless as a safety net: a real regression
would hide among the two known ones. `--record-replacement` writes the current
deterministic summary beside them, and a run is reported `[FIXED]` when it fails the
archive but matches that replacement field for field. Anything that matches neither is
a genuine `[DIFF]` and still fails.

    python tools/verify_golden_runs.py                      # all 14 scenario F runs
    python tools/verify_golden_runs.py --only fcfs          # the five fcfs cells
    python tools/verify_golden_runs.py --root runs/allpods_20261101/metrics
    python tools/verify_golden_runs.py --record-replacement --only spotgpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vmagent.gpu.compare import diff_summaries, format_diff
from vmagent.gpu.run import make_environment

REPO = Path(__file__).resolve().parents[1]


def load_golden(root):
    """Yield (name, recorded_summary, config) for every golden directory under root."""
    root = Path(root)
    provenance = json.loads((root.parent/"provenance.json").read_text(encoding="utf-8"))
    shared = provenance["run_parameters"]
    for folder in sorted(p for p in root.iterdir() if (p/"summary.json").is_file()):
        manifest = json.loads((folder/"run_manifest.json").read_text(encoding="utf-8"))
        window = float(shared.get("measurement_start_seconds", 0.))
        config = {"policy": manifest["policy"],
                  "options": {"clock": shared["clock"], "speed": shared.get("speed", 1.),
                              "queue_policy": manifest["queue_policy"],
                              "seed": manifest.get("seed", 42),
                              "max_sim_time": shared["max_sim_time"],
                              "cpu_background_fraction": shared.get("cpu_background_fraction", 0.),
                              "checkpoint_interval": shared.get("spotgpu_checkpoint_interval_sec", 3600.),
                              "measurement_window": [window, window+shared["measurement_seconds"]]}}
        yield folder.name, json.loads((folder/"summary.json").read_text(encoding="utf-8")), config


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO/"runs/hetero8d_20261101/metrics/summaries")
    parser.add_argument("--nodes", type=Path, default=REPO/"runs/hetero8d_20261101/inputs/nodes.csv")
    parser.add_argument("--tasks", type=Path, default=REPO/"runs/hetero8d_20261101/inputs/jobs_F.csv")
    parser.add_argument("--only", default="", help="keep golden runs whose name contains this")
    parser.add_argument("--rel-tol", type=float, default=1e-12)
    parser.add_argument("--replacement-root", type=Path,
                        default=REPO/"runs/hetero8d_20261101/metrics/post_fix_summaries",
                        help="summaries recorded after the victim-ordering fix, for runs "
                             "whose archive predates the feature being tested")
    parser.add_argument("--record-replacement", action="store_true",
                        help="write each selected run's summary into --replacement-root")
    args = parser.parse_args()

    failures, started = [], time.monotonic()
    selected = [row for row in load_golden(args.root) if args.only in row[0]]
    if not selected:
        print("no golden runs matched --only %r" % args.only)
        return 2
    for name, reference, config in selected:
        env, _ = make_environment(args.nodes, args.tasks, **config["options"])
        clock = time.monotonic()
        summary = env.run(config["policy"])
        if args.record_replacement:
            folder = args.replacement_root/name
            folder.mkdir(parents=True, exist_ok=True)
            (folder/"summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            (folder/"why.json").write_text(json.dumps({
                "name": name,
                "archive_is_unusable_because": "recorded before the feature existed, and "
                                               "its victim order was process-dependent "
                                               "until the (cost, pod_id) fix",
                "produced_by": "tools/verify_golden_runs.py --record-replacement",
                "parameters": config}, indent=2, sort_keys=True), encoding="utf-8")
            print("[RECORD] %-26s %5.1fs  wrote %s" % (name, time.monotonic()-clock, folder))
            continue
        differences = diff_summaries(reference, summary, rel_tol=args.rel_tol, allow_new_fields=True)
        path = args.replacement_root/name/"summary.json"
        replacement = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        residual = (diff_summaries(replacement, summary, rel_tol=args.rel_tol,
                                   allow_new_fields=True) if differences and replacement else None)
        if not differences:
            status, detail = "OK  ", "all %d recorded fields reproduce" % len(reference)
        elif residual is not None and not residual:
            status, detail = "FIXED", ("archive predates the feature; all %d fields match "
                                       "the post-fix reference at %s"
                                       % (len(replacement), path.parent.relative_to(REPO)))
        else:
            status, detail = "DIFF", format_diff(differences)
        print("[%s] %-26s %5.1fs  %s" % (status, name, time.monotonic()-clock, detail))
        if status == "DIFF":
            failures.append(name)
    if args.record_replacement:
        print("\nrecorded %d replacement summaries under %s" % (len(selected), args.replacement_root))
        return 0
    print("\n%d/%d golden runs reproduce within tolerance (%.1f s total)"
          % (len(selected)-len(failures), len(selected), time.monotonic()-started))
    if failures:
        print("FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
