"""M9: convert the defragmentation benchmark's parquet instances to the engine's CSV contract.

The collaborator's benchmark under `benchmark任务/` runs on its own engine and takes
`servers.parquet` + `initial_pods.parquet` + `arrivals.parquet` + `config.json`. The
main engine takes `nodes.csv` + a jobs table. Until both sides can read the same
thing there is no way to check that "one simulator" and "the other simulator" agree
on a scenario, which is the acceptance test M10 needs.

Run it with an interpreter that has pandas and pyarrow:

    C:/Users/trw/anaconda3/python.exe tools/convert_benchmark_instances.py --all
    C:/Users/trw/anaconda3/python.exe tools/convert_benchmark_instances.py \
        --instance 'benchmark任务/.../candidates/seed20261107_interval10/S2'

Field mapping, and why each choice is the only defensible one:

    servers.parquet      -> nodes.csv        server_id, cluster_id, gpu_spec_public,
                                             gpu_count, cpu_capacity_cores, asw_id
    initial_pods.parquet -> jobs.csv         submit_time_sec = simulation_start_sec (0),
                                             service_sec = remaining_duration_sec
    arrivals.parquet     -> jobs.csv         submit_time_sec = arrival_time_sec,
                                             service_sec = duration_sec
    both                 -> initial_placement.csv  pod_id, server_id, for the 294 pods
                                             that must start on a prescribed node

`remaining_duration_sec` and not `duration_sec` for the initial pods: the engine
starts an task's service clock when it places it, and these pods are placed at t=0,
so the time they still owe is the remaining one. Using duration_sec would extend
every initial pod's life by the part it had already run.

The initial layout goes through `--initial-placement`, not through the policy.
These instances exist because their starting state is fragmented; letting first_fit
re-place those 294 pods would compact them and measure something else.

What is dropped, recorded in each conversion_manifest.json rather than hidden:

    migratable                  the main engine has no migration yet (M10/M11/M12)
    gpu_mem_request_trace       the engine models no memory
    source_* / duration_*       provenance only; no scheduling effect
    config.start_deadline_sec   the engine reports no deadline-attainment rate
    config.decision_interval    the engine is event-driven, not tick-driven
    p95 wait definition         the benchmark censors unstarted requests at
                                horizon - arrival; the engine's p95_wait_sec_started
                                only covers started ones, so the two p95 numbers are
                                not comparable without restating one of them
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    import pandas as pd
except ImportError:
    sys.exit("this converter needs pandas+pyarrow; run it with the anaconda interpreter")

JOB_FIELDS = ("pod_id", "gpu_spec_public", "gpu_request", "cpu_request_cores", "submit_time_sec",
              "service_sec", "cluster_id", "duration_match_status", "job_type_public",
              "priority_class", "workload_id")
NODE_FIELDS = ("server_id", "cluster_id", "gpu_spec_public", "gpu_count", "cpu_capacity_cores", "asw_id")
DROP_COLUMNS = ("migratable", "gpu_mem_request_trace", "source_server_id", "source_cluster_id",
                "source_day", "source_hour", "source_gpu_request", "source_state_public",
                "source_ready_status", "source_used_gpu_hours", "duration_source_pod_id",
                "duration_summary_row_index", "duration_hours_raw", "duration_base_sec",
                "initial_state", "request_id")


def blank(value):
    """NaN/None to empty string: the engine reads blank, nan and null as unknown."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if not isinstance(value, str) and value is not None and str(value).lower() in ("nan", "none", "<na>"):
        return ""
    return str(value)


def convert(instance, output):
    instance = Path(instance)
    config = json.loads((instance/"config.json").read_text(encoding="utf-8"))
    servers = pd.read_parquet(instance/"servers.parquet")
    initial = pd.read_parquet(instance/"initial_pods.parquet")
    arrivals = pd.read_parquet(instance/"arrivals.parquet")

    nodes = pd.DataFrame({"server_id": servers.server_id, "cluster_id": servers.cluster_id,
                          "gpu_spec_public": servers.gpu_spec_public,
                          "gpu_count": servers.gpu_count.astype(float),
                          "cpu_capacity_cores": servers.cpu_capacity_cores.astype(float),
                          "asw_id": servers.asw_id.map(blank)})

    def row(pod, submit, service):
        return {"pod_id": pod.pod_id, "gpu_spec_public": pod.gpu_spec_public,
                "gpu_request": float(pod.gpu_request), "cpu_request_cores": float(pod.cpu_request_cores),
                "submit_time_sec": float(submit), "service_sec": float(service),
                "cluster_id": pod.source_cluster_id,
                "duration_match_status": blank(pod.duration_source_method),
                "job_type_public": blank(pod.job_type_public), "priority_class": blank(pod.priority_class),
                "workload_id": blank(pod.workload_id)}

    jobs = pd.DataFrame([row(pod, pod.simulation_start_sec, pod.remaining_duration_sec)
                         for _, pod in initial.iterrows()]
                        + [row(pod, pod.arrival_time_sec, pod.duration_sec)
                           for _, pod in arrivals.iterrows()])[list(JOB_FIELDS)]
    jobs = jobs.sort_values(["submit_time_sec", "pod_id"], kind="mergesort").reset_index(drop=True)
    placement = initial[["pod_id", "server_id"]].copy()

    # Contradictions the engine would otherwise only discover at run time.
    problems = []
    if (initial.simulation_start_sec != 0).any():
        problems.append("initial pods with simulation_start_sec != 0")
    if initial.pod_id.duplicated().any() or arrivals.pod_id.duplicated().any():
        problems.append("duplicate pod_id within one input table")
    if set(initial.pod_id) & set(arrivals.pod_id):
        problems.append("pod_id present in both initial_pods and arrivals")
    if not set(initial.server_id) <= set(servers.server_id):
        problems.append("initial layout references a server that is not in servers.parquet")
    if (jobs.service_sec <= 0).any():
        problems.append("non-positive service_sec")
    load = initial.groupby("server_id").agg(gpu=("gpu_request", "sum"), cpu=("cpu_request_cores", "sum"))
    capacity = servers.set_index("server_id")[["gpu_count", "cpu_capacity_cores"]]
    over = load.join(capacity)
    if (over.gpu > over.gpu_count + 1e-9).any() or (over.cpu > over.cpu_capacity_cores + 1e-9).any():
        problems.append("initial layout exceeds a node's capacity")

    output = Path(output)
    (output/"inputs").mkdir(parents=True, exist_ok=True)
    nodes[list(NODE_FIELDS)].to_csv(output/"inputs/nodes.csv", index=False)
    jobs.to_csv(output/"inputs/jobs.csv", index=False)
    placement.to_csv(output/"inputs/initial_placement.csv", index=False)
    manifest = {"source": str(instance.resolve()),
                "source_sha256": {name: __import__("hashlib").sha256(
                    (instance/(name+".parquet")).read_bytes()).hexdigest()
                    for name in ("servers", "initial_pods", "arrivals")},
                "scenario": config.get("scenario"), "seed": config.get("seed"),
                "rows": {"nodes": len(nodes), "jobs": len(jobs),
                         "jobs_initial": len(initial), "jobs_arrivals": len(arrivals),
                         "initial_placement": len(placement)},
                "engine_parameters": {"max_sim_time": config["horizon_sec"],
                                      "measurement_window": [0., config["allocation_window_sec"]],
                                      "queue_policy": "backfill",
                                      "cpu_background_fraction": 0.},
                "parameter_mapping": {"horizon_sec": "max_sim_time",
                                      "allocation_window_sec": "measurement window length",
                                      "migration.*": "no engine equivalent yet (M10/M11/M12)",
                                      "start_deadline_sec": "no engine metric",
                                      "decision_interval_sec": "engine is event-driven, not tick-driven"},
                "dropped_columns": list(DROP_COLUMNS),
                "semantics_matching": {
                    "queue": "benchmark 'FIFO scan with backfill' == engine backfill",
                    "completion_before_arrival": "engine pushes complete at heap priority 0, arrival at 1",
                    "gpu_model_exact": "single-model allowed_gpu_types whitelist",
                    "gpu_cards_whole": "engine accepts fractional shares; integers are a subset",
                    "preemption_false": "compare with first_fit/best_fit, not spotgpu",
                    "gang_false": "engine places one pod per node"},
                "known_metric_gaps": ["p95/mean wait: benchmark right-censors unstarted requests at "
                                      "horizon-arrival; engine reports started-only waits",
                                      "benchmark A_GPU excludes GPU paused during migration; the engine "
                                      "has no migration, so the exclusion cannot arise yet"],
                "problems": problems}
    (output/"conversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=REPO/"benchmark任务/三指标数据评测工作文件_20261108/01_场景数据与构建/candidates")
    parser.add_argument("--instance", type=Path, action="append", default=None,
                        help="a specific S directory, repeatable")
    parser.add_argument("--all", action="store_true", help="convert every S directory of every candidate")
    parser.add_argument("--output", type=Path, default=REPO/"outputs/benchmark_csv")
    args = parser.parse_args()

    instances = args.instance or ([p for c in sorted(Path(args.root).iterdir()) if c.is_dir()
                                   for p in sorted(c.iterdir()) if p.is_dir() and p.name.startswith("S")]
                                  if args.all else [])
    if not instances:
        parser.error("nothing to convert: pass --instance or --all")

    failed = 0
    for instance in instances:
        candidate = instance.parent.name
        out = args.output/candidate/instance.name
        manifest = convert(instance, out)
        status = "ok " if not manifest["problems"] else "BAD"
        print("[%s] %-34s nodes=%d jobs=%d (initial=%d + arrivals=%d) %s"
              % (status, candidate+"/"+instance.name, manifest["rows"]["nodes"], manifest["rows"]["jobs"],
                 manifest["rows"]["jobs_initial"], manifest["rows"]["jobs_arrivals"],
                 "" if not manifest["problems"] else "; ".join(manifest["problems"])))
        if manifest["problems"]:
            failed += 1
    print("\n%d instance(s) converted, %d with contradictions" % (len(instances), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
