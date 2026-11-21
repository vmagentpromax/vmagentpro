"""Tolerance-based comparison of simulation results.

The project's safety practice is to re-run a scenario and diff summary.json field by
field after every engine change. Exact equality is the wrong test for two measured
reasons:

1. CPython 3.12 changed float sum() to Neumaier compensated summation. On identical
   code and input, per_model.XPU-A.offered_gpu_ratio is 1.040616417218244 under
   3.11.11 and 1.0406164172182435 under 3.13.14: a 2 ULP difference that is not a
   regression.
2. Wall clock and realtime sim clock legitimately vary with machine load.

So diff_summaries() reports differences beyond a relative tolerance and always
separates the fields that are expected to drift.
"""
import json
from pathlib import Path

# Fields that legitimately vary run to run: wall clock always, and the sim clock in
# realtime mode because thread wake-up granularity feeds into it.
VOLATILE = ("wall_time_sec", "sim_time_sec")

DEFAULT_REL_TOL = 1e-12
DEFAULT_ABS_TOL = 1e-12


def _walk(a, b, path, rel_tol, abs_tol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = "%s.%s" % (path, key) if path else key
            if key not in a or key not in b:
                out.append((sub, a.get(key, "<absent>"), b.get(key, "<absent>"), "presence"))
                continue
            _walk(a[key], b[key], sub, rel_tol, abs_tol, out)
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append((path, a, b, "bool"))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a-b) > max(abs_tol, abs(a)*rel_tol, abs(b)*rel_tol):
            out.append((path, a, b, "value"))
        return
    if a != b:
        out.append((path, a, b, "value"))


def diff_summaries(a, b, rel_tol=DEFAULT_REL_TOL, abs_tol=DEFAULT_ABS_TOL, ignore_volatile=True):
    """Return a list of (field_path, reference, candidate, kind) differences."""
    out = []
    _walk(a, b, "", rel_tol, abs_tol, out)
    if ignore_volatile:
        out = [row for row in out if row[0] not in VOLATILE]
    return out


def compare_files(reference, candidate, rel_tol=DEFAULT_REL_TOL):
    """Convenience for golden checks: read two summary.json paths and diff them."""
    a = json.loads(Path(reference).read_text(encoding="utf-8"))
    b = json.loads(Path(candidate).read_text(encoding="utf-8"))
    return diff_summaries(a, b, rel_tol)


def format_diff(differences, limit=20):
    if not differences:
        return "identical within tolerance"
    lines = ["%d difference(s) beyond tolerance:" % len(differences)]
    for path, left, right, kind in differences[:limit]:
        lines.append("  %-46s %s  ref=%r cand=%r  (delta %r)" % (
            path, kind, left, right,
            (right-left) if isinstance(left, (int, float)) and isinstance(right, (int, float)) else "-"))
    if len(differences) > limit:
        lines.append("  ... %d more" % (len(differences)-limit))
    return "\n".join(lines)
