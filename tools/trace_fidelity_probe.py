"""M16 first step: what the ASI trace actually says about delay and utilisation.

The simulator reports reservation, not execution. That is a defensible design
choice, but it has to be measured against the trace rather than asserted, because
the paper's own headline is that high demand does not mean high use. This tool
answers three questions with data before anyone commits to a calibration design:

1. `schedule_delay_sec` - how much of it is populated, and what does the
   simulator's own wait distribution have to match? This is the only delay ground
   truth in the trace, and it is what M16's replay calibration would target.
2. `avg_gpu_sm_util` vs reserved GPU-hours - the gap between "a GPU is reserved for
   you" and "a GPU is computing". The engine reports the first; the trace carries the
   second. If the gap is large, no reservation number from this simulator may be
   described as utilisation, which the summary already warns about. This field is a sum
   over the pod's GPUs in percent per GPU, so it has to be divided by `gpu_request`
   before it is a utilisation, and only whole-GPU pods carry it at all.
3. Whether these fields are per-pod-hour samples or per-pod totals, because that
   decides what kind of averaging is legitimate.

Reads a bounded sample of partitions, not the whole 4,440-file table.

    C:/Users/trw/anaconda3/python.exe tools/trace_fidelity_probe.py
    C:/Users/trw/anaconda3/python.exe tools/trace_fidelity_probe.py --files 400 --day 110
"""
import argparse
import glob
import random
from pathlib import Path

import pandas as pd

TRACE = Path("D:/cluster-trace-gpu-v2026")
POD = TRACE / "asi_opensource_pod_hourly"
SUMMARY = TRACE / "asi_opensource_job_execution_summary"

COLUMNS = ["pod_id", "server_id", "cluster_id", "gpu_spec_public", "priority_class",
           "job_type_public", "server_gpu_count", "gpu_request", "cpu_request_cores",
           "used_gpu_hours", "avg_gpu_sm_util", "avg_cpu_request_util",
           "schedule_delay_sec", "ready_delay_sec", "state_public", "ready_status"]


def quantiles(series, label, unit=""):
    clean = series.dropna()
    if clean.empty:
        print("    %-28s all null" % label)
        return
    q = clean.quantile([.5, .9, .99, 1.0])
    print("    %-28s n=%-9d share=%.4f  p50=%8.2f p90=%8.2f p99=%8.2f max=%10.2f%s"
          % (label, len(clean), len(clean)/len(series), q[.5], q[.9], q[.99], q[1.0], unit))


def weighted(series, weights):
    keep = series.notna() & weights.notna() & (weights > 0)
    if not keep.any():
        return float("nan")
    return float((series[keep]*weights[keep]).sum()/weights[keep].sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", type=int, default=200)
    parser.add_argument("--day", default=None, help="restrict to one snapshot day")
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()

    pattern = str(POD/("day=%s" % args.day if args.day else "day=*")/"hour=*"/"*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("no partitions matched %s - is the D: drive mounted?" % pattern)
    picked = random.Random(args.seed).sample(files, min(args.files, len(files)))
    print("sampling %d of %d partitions%s" % (len(picked), len(files),
                                              " (day=%s)" % args.day if args.day else ""))
    frames = [pd.read_parquet(path, columns=COLUMNS) for path in picked]
    df = pd.concat(frames, ignore_index=True)
    print("rows=%d  distinct pods=%d  distinct servers=%d  clusters=%d"
          % (len(df), df.pod_id.nunique(), df.server_id.nunique(), df.cluster_id.nunique()))

    gpu = df[df.gpu_request > 0]
    print("\n1. schedule_delay_sec (the trace's only delay ground truth)")
    quantiles(df.schedule_delay_sec, "all rows")
    quantiles(gpu.schedule_delay_sec, "GPU pods")
    for priority in ("HP", "LP", "Other"):
        subset = gpu[gpu.priority_class == priority]
        if len(subset):
            quantiles(subset.schedule_delay_sec, "GPU pods %s" % priority)
    print("    null share by state_public:")
    print(df.groupby("state_public").schedule_delay_sec.agg(
        n="size", nonnull=lambda s: s.notna().sum()).to_string())
    clipped = int((df.schedule_delay_sec >= 86399).sum())
    print("    rows pinned at 86399s: %d (%.5f of non-null); rows >= 3599: %.5f - the "
          "field is capped at 24h-1s, so the far tail is censored and a distribution "
          "match must censor it the same way"
          % (clipped, clipped/max(1, int(df.schedule_delay_sec.notna().sum())),
             float((df.schedule_delay_sec >= 3599).mean())))

    print("\n2. avg_gpu_sm_util vs reserved GPU-hours - the reservation/use gap")
    # avg_gpu_sm_util is a sum over the pod's GPUs in percent per GPU, not a per-GPU
    # percentage: its p50 is 9.2 at gpu_request=1 and 210 at gpu_request=8. Dividing by
    # the request is therefore required before the number means anything, and a value
    # above 100 afterwards means the device is shared and reported per pod.
    whole = gpu[gpu.gpu_request % 1 == 0]
    partial = gpu[gpu.gpu_request % 1 != 0]
    print("    SM util reported for integral gpu_request: %.4f of those rows; "
          "for fractional: %.4f"
          % (float(whole.avg_gpu_sm_util.notna().mean()),
             float(partial.avg_gpu_sm_util.notna().mean())))
    per_gpu = whole.avg_gpu_sm_util/whole.gpu_request
    gpu_hours = whole.used_gpu_hours
    print("    per-GPU SM util (%%), unweighted    n=%-9d p50=%6.2f p75=%6.2f p95=%6.2f"
          % (int(per_gpu.notna().sum()), per_gpu.quantile(.5), per_gpu.quantile(.75),
             per_gpu.quantile(.95)))
    print("    reserved GPU-hours weighted mean     %.2f%%  (this is the cluster-level use "
          "of what the scheduler handed out)" % (100*weighted(per_gpu/100., gpu_hours)))
    idle = per_gpu < 10.
    print("    share of reserved GPU-hours with per-GPU util <10%%: %.4f   <30%%: %.4f"
          % (float(gpu_hours[idle].sum()/gpu_hours.sum()),
             float(gpu_hours[per_gpu < 30.].sum()/gpu_hours.sum())))
    quantiles(gpu.avg_cpu_request_util, "GPU pods CPU request util")
    print("    residency within the hour: mean used_gpu_hours/gpu_request = %.4f"
          % float((gpu.used_gpu_hours/gpu.gpu_request).clip(upper=1.).mean()))
    whole = whole.assign(per_gpu=per_gpu)
    keep = whole.per_gpu.notna() & (whole.used_gpu_hours > 0)
    busy = whole[keep].copy()
    busy["weight"] = busy.used_gpu_hours
    busy["used"] = busy.per_gpu*busy.used_gpu_hours
    def grouped(keys):
        out = busy.groupby(keys).agg(n=("per_gpu", "size"),
                                     gpu_hours=("weight", "sum"),
                                     util_p50=("per_gpu", "median"))
        out["util_gpu_hour_weighted"] = out.index.map(
            busy.groupby(keys).used.sum()/busy.groupby(keys).weight.sum())
        return out
    by_type = grouped("job_type_public")
    by_type["cpu_util_mean"] = busy.groupby("job_type_public").avg_cpu_request_util.mean()
    print(by_type.to_string(float_format=lambda v: "%.4f" % v))
    print("    by priority_class (reserved GPU-hours weighted, i.e. the share of what the "
          "scheduler handed out that was actually computing):")
    print(grouped("priority_class").to_string(float_format=lambda v: "%.4f" % v))

    print("\n3. is this a per-pod-hour sample or a per-pod total?")
    per_pod = gpu.groupby("pod_id").agg(rows=("avg_gpu_sm_util", "size"),
                                        distinct_util=("avg_gpu_sm_util", "nunique"),
                                        distinct_delay=("schedule_delay_sec", "nunique"))
    print("    pods with >1 row: %.4f of %d; mean distinct avg_gpu_sm_util values per pod: %.4f"
          % (float((per_pod.rows > 1).mean()), len(per_pod), float(per_pod.distinct_util.mean())))
    print("    mean distinct schedule_delay_sec per pod: %.4f  (1.0 would mean the field is"
          " constant across a pod's hours)" % float(per_pod.distinct_delay.mean()))

    if SUMMARY.exists():
        summary = pd.read_parquet(SUMMARY)
        print("\n4. job_execution_summary (duration and delay per pod, one row each)")
        print("    rows=%d  null schedule_delay share=%.4f"
              % (len(summary), float(summary.schedule_delay_sec.isna().mean())))
        quantiles(summary.schedule_delay_sec, "all jobs")
        for priority in ("HP", "LP", "Other"):
            subset = summary[summary.priority_class == priority]
            if len(subset):
                quantiles(subset.schedule_delay_sec, "%s jobs" % priority)
        quantiles(summary.duration_hours*3600, "duration_sec")


if __name__ == "__main__":
    main()
