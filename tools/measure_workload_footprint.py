"""What a loaded workload actually costs in memory, measured rather than asserted.

M6's acceptance line is "the biggest ASI pool fits under 1 GB", which is about 60 bytes
per task, so the first thing to know is where the money currently goes. Nobody had a
committed way to answer it: the numbers that circulated (1,051 B/task, 995 B of it
metadata) came from an ad-hoc `sys.getsizeof` session that was never checked in, and
`tools/profile_loader.py` times the loader without ever measuring what it leaves behind.

This walks the real objects, built through the real loader on truncated copies of the
input, so the accounting cannot drift away from what the engine holds:

    C:/Users/trw/anaconda3/python.exe tools/measure_workload_footprint.py
    C:/Users/trw/anaconda3/python.exe tools/measure_workload_footprint.py \
        --sizes 10000,40000,119495 --extrapolate 16260000

Two conventions worth knowing before quoting the output:

- Bytes are `sys.getsizeof` summed over a recursive walk with a shared-object set, so an
  object reachable from two fields is counted once. That makes the figure CPython-specific
  and it is *retained object size*, not allocator-resident memory: pymalloc keeps its
  arenas, so the process working set is larger than the sum and neither predicts the other.
  Quote both or neither.
- `metadata` is charged the *marginal* cost of keeping it: everything else is walked first
  and claims its objects, then the metadata dicts are walked against the same seen set. A
  value that the `pod_id` field already holds is therefore not double-charged, which is the
  honest reading of "drop metadata and you save X bytes per task".
"""
import argparse
import csv
import io
import sys
import tempfile
from pathlib import Path
from sys import getsizeof

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu import contract

INTERNED_INTS = range(-5, 257)


def walk(obj, seen, skip=()):
    """Retained bytes of `obj`, counting each reachable object at most once.

    `skip` names instance attributes not to descend into, which is how the metadata dict
    is charged separately instead of being silently absorbed into the TaskSpec figure.
    """
    if obj is None or isinstance(obj, bool) or (isinstance(obj, int) and obj in INTERNED_INTS):
        return 0                      # singletons, shared process-wide
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = getsizeof(obj)
    if isinstance(obj, (str, bytes, int, float, complex)):
        return total
    if isinstance(obj, dict):
        for key, value in obj.items():
            total += walk(key, seen) + walk(value, seen)
        return total
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            total += walk(item, seen)
        return total
    for name in list(getattr(obj, "__dict__", None) or ()):
        if name in skip:
            continue      # left entirely to the dedicated pass, so it is charged once
        total += walk(getattr(obj, name, None), seen)
    for name in getattr(type(obj), "__slots__", None) or ():
        if name in skip:
            continue
        total += walk(getattr(obj, name, None), seen)
    return total


def workloads(path, sizes):
    """Build each requested size through the real loader, on a truncated input file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    with io.StringIO(text) as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0].keys())
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            target = Path(tmp)/("jobs_%d.csv" % size)
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[:size])
            out[size] = contract.load_workload(target)[0]
    return out


class WorkingSet:
    """Process resident memory on Windows; reports None anywhere it cannot ask."""

    def __init__(self):
        self.available = sys.platform.startswith("win")
        self._impl = self._bind() if self.available else None

    def _bind(self):
        import ctypes
        import ctypes.wintypes

        class Counter(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        # `psapi.dll` still loads on desktop Windows but the documented entry point has
        # lived in kernel32 as K32GetProcessMemoryInfo since Vista, and some builds only
        # expose that one. Try both rather than reporting nothing. `GetCurrentProcess`
        # must be typed as a pointer: with ctypes' default c_int restype the pseudo-handle
        # (void*)(-1) is truncated on 64-bit and the call fails with ERROR_BAD_LENGTH,
        # which is exactly how this function ended up silently returning None once.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        lookups = []
        for lib, name in ((kernel32, "K32GetProcessMemoryInfo"),
                          (getattr(ctypes.windll, "psapi", None), "GetProcessMemoryInfo")):
            if lib is None:
                continue
            try:
                function = getattr(lib, name)
            except (AttributeError, OSError):
                continue
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counter), ctypes.wintypes.DWORD]
            function.restype = ctypes.wintypes.BOOL
            lookups.append(function)

        def read():
            if not lookups:
                return None
            info = Counter()
            info.cb = ctypes.sizeof(info)
            handle = kernel32.GetCurrentProcess()
            for function in lookups:
                if function(handle, ctypes.byref(info), info.cb):
                    return info.WorkingSetSize, info.PeakWorkingSetSize
            return None

        return read

    def __call__(self):
        return self._impl() if self._impl else None


def human(n):
    return "%.1f MB" % (n/1e6) if n < 1e9 else "%.2f GB" % (n/1e9)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/jobs_F.csv")
    parser.add_argument("--sizes", default="10000,40000,119495",
                        help="comma separated task counts to measure")
    parser.add_argument("--extrapolate", type=int, default=16260000,
                        help="task count to project onto; 0 to skip")
    parser.add_argument("--run", action="store_true",
                        help="also run one full episode over the whole input and report "
                             "wall time, event count and resident memory while running")
    parser.add_argument("--nodes", type=Path, default=REPO/"场景/02_主场景F/hetero8d_20261101/inputs/nodes.csv")
    parser.add_argument("--placement", default="first_fit")
    parser.add_argument("--queue-policy", default="fcfs", choices=("fcfs", "backfill"))
    parser.add_argument("--max-sim-time", type=float, default=4000000.,
                        help="scenario F's recorded horizon; see metrics/provenance.json")
    parser.add_argument("--measurement-seconds", type=float, default=691200.,
                        help="scenario F's recorded 8-day window")
    args = parser.parse_args()

    sizes = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    resident = WorkingSet()
    print("input      : %s" % args.tasks)
    print("interpreter: %s on %s" % (sys.version.split()[0], sys.platform))
    print("note       : getsizeof is CPython-specific, and this is retained size, "
          "not allocator-resident memory\n")

    built = workloads(args.tasks, sizes)
    results = {}
    for size in sizes:
        tasks = built[size]
        seen = set()
        body = sum(walk(task, seen, skip=("metadata",)) for task in tasks)
        with_meta = sum(walk(task.metadata, seen) for task in tasks)
        results[size] = (body, with_meta)
        print("%7d tasks  TaskSpec+values %9s   metadata %9s   total %9s   metadata share %.1f%%"
              % (size, human(body), human(with_meta), human(body+with_meta),
                 100.*with_meta/(body+with_meta)))
        now = resident()
        if now is not None:
            print("          %23s working set %s (peak %s)"
                  % ("", human(now[0]), human(now[1])))
        del tasks, built[size]
        built[size] = None

    low, high = sizes[0], sizes[-1]
    d_body = (results[high][0]-results[low][0])/(high-low)
    d_meta = (results[high][1]-results[low][1])/(high-low)
    per_task = d_body + d_meta
    average = (results[high][0]+results[high][1])/high
    print("\nmarginal bytes per task (two-point fit, %d -> %d):" % (low, high))
    print("    TaskSpec and the objects its fields point at   %8.1f B" % d_body)
    print("    metadata = dict(row), on top of that           %8.1f B  (%.1f%% of the total)"
          % (d_meta, 100.*d_meta/per_task))
    print("    total                                        %8.1f B" % per_task)
    print("    full-sample average over %s tasks             %8.1f B" % (f"{high:,}", average))
    print("  The fit and the average differ by the fixed cost of the list and the "
          "first-page\n  objects; quote the fit when reasoning about a larger pool.")

    if args.extrapolate:
        n = args.extrapolate
        projected = per_task*n
        print("\nat {:.0f} B/task, a pool of {:,} tasks holds {:.1f} GB decimal "
              "= {:.1f} GiB.".format(per_task, n, projected/1e9, projected/2**30))
        budget = 1e9/n
        print("M6's line is 1 GB for that pool, i.e. {:.0f} B/task - today's figure is "
              "{:.1f}x that.".format(budget, per_task/budget))
    now = resident()
    if now is not None:
        print("\nworking set at the end: {} (peak so far {}). Python gives little back to"
              " the OS,\nso the peak covers every sample built above and is an upper bound,"
              " not a\nper-task figure. The bytes/task numbers are the ones to quote."
              .format(human(now[0]), human(now[1])))

    if args.run:
        # The claim worth checking is not what loading costs but what a full episode
        # costs, because the per-task records and the event trace dominate it. Built
        # through vmagent.gpu.run.make_environment, i.e. the same construction path the
        # CLI and the dashboard use, so the figure cannot describe a different program.
        from time import monotonic
        from vmagent.gpu.run import make_environment
        env, _ = make_environment(args.nodes, args.tasks, clock="fast",
                                  queue_policy=args.queue_policy, seed=42,
                                  max_sim_time=args.max_sim_time,
                                  measurement_window=[0., args.measurement_seconds],
                                  cpu_background_fraction=0., checkpoint_interval=3600.)
        start = monotonic()
        env.run(args.placement)
        elapsed = monotonic()-start
        tasks, events, records = len(env.tasks), len(env.events), len(env.records)
        seen = set()
        sample = events if events <= 2000 else 2000
        per_event = sum(walk(event, seen) for event in env.events[:sample])/sample
        now = resident()
        print("\n--run %s/%s over %s tasks: %.1f s, %s events, %s records"
              % (args.placement, args.queue_policy, f"{tasks:,}", elapsed,
                 f"{events:,}", f"{records:,}"))
        print("   the event trace alone holds about %s of dicts" % human(per_event*events))
        if now is not None:
            print("   working set %.2f GB, peak %.2f GB - this is the figure to quote for"
                  " a run,\n   not the load-only number above." % (now[0]/1e9, now[1]/1e9))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
