"""Where the workload loader actually spends its time, measured rather than assumed.

M5 was scoped as "make loading fast by letting pyarrow/pandas read the input instead
of csv.DictReader". Profiling says that premise is wrong about the bottleneck: the
parser is a third of the cost, and the largest single item is constructing 119,495
dataclass objects, which no input format removes. This tool exists so that conclusion
is checkable instead of being a claim in a commit message.

    python tools/profile_loader.py                          # scenario F
    python tools/profile_loader.py --tasks path/to/jobs.csv

Read the numbers together with the note printed at the end: the question M5 was
supposed to answer is now M6's, because only dropping per-task objects changes the
dominant term.
"""
import argparse
import io
import sys
import time
from csv import DictReader
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu.contract import TaskSpec, number, optional_label, required


def timed(label, fn):
    start = time.perf_counter()
    result = fn()
    print("  %-46s %6.2fs" % (label, time.perf_counter()-start))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.tasks)
    print("%s" % path)
    for attempt in range(args.repeat):
        print("  run %d" % (attempt+1))
        total = time.perf_counter()
        text = timed("read_text (utf-8-sig)", lambda: path.read_text(encoding="utf-8-sig"))
        rows = timed("csv.DictReader -> list of dicts", lambda: list(DictReader(io.StringIO(text))))
        tasks = timed("per-row validation + TaskSpec construction",
                      lambda: [TaskSpec(required(row, "pod_id"),
                                         number(row.get("gpu_request"), "gpu_request"),
                                         number(row.get("cpu_request_cores"), "cpu_request_cores"),
                                         (row["gpu_spec_public"],),
                                         number(row.get("submit_time_sec"), "submit_time_sec"),
                                         number(row.get("service_sec"), "service_sec", True),
                                         "Other", dict(row), optional_label(row, "workload_id"))
                                 for row in rows])
        timed("sort by (submit_time, pod_id)", lambda: sorted(tasks, key=lambda t: (t.submit_time, t.pod_id)))
        print("  %-46s %6.2fs  (%d rows)" % ("total", time.perf_counter()-total, len(rows)))
    print("""
Reading: the parser is one term, object construction is another of similar size, and
no input format removes either. A pyarrow/pandas fast path would replace the
DictReader line and leave the construction line untouched, so it buys a fraction of
the load, adds a dependency, and puts a second float parser between the trace and the
results. The term that actually disappears is per-task objects, which is M6
(columnar arrays), not M5. Recommendation: fold M5 into M6.""")


if __name__ == "__main__":
    main()
