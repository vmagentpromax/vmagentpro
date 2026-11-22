"""Do the `placement.py:28-29` pointers in the docs still land inside the file?

Every architecture note in this project cites code by file and line, which is the most
useful form of citation and the most fragile: the engine is being edited continuously, so
a two-line insertion silently invalidates every reference below it. Two full audit passes
independently found stale line numbers in documents written days earlier, and neither pass
could say how many were left.

This turns "I checked" into a command. It indexes every source file under the given roots,
extracts each `name.py:N` or `name.py:N-M` citation from the markdown, resolves the name
(unique basename, else the explicit relative path) and reports:

    MISSING        no such file under any indexed root
    OUT OF RANGE   the line number is past the end of the file, i.e. the code moved
    BARE AMBIGUOUS the basename matches two files; informational when the citation is
                   inside at least one of them, a failure only when it is inside neither

    C:/Users/trw/anaconda3/python.exe tools/check_doc_line_references.py --quiet
    C:/Users/trw/anaconda3/python.exe tools/check_doc_line_references.py --explain --quiet

"In range but pointing at the wrong function" is the failure this cannot decide for you,
because deciding it means reading. `--explain` prints the enclosing `def` / `class` /
compound statement of each cited range so a mismatch is visible instead of assumed; it is
what to run after moving code, before re-numbering anything by hand.

Citations to files outside the repository (the sibling research folder) are picked up
automatically when that folder is present; pass --root to add more.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# `foo.py:12`, `foo.py:12-30`, and the Chinese-text variant `foo.py 第 12 行` are all
# written by hand; only the colon forms are machine-checkable without parsing prose.
#
# A citation may be revision-pinned: `1ffabd6:schedgym/sched_env.py:39-43`. That form
# exists because the upstream half of the tree was removed from the working copy on
# 2026-11-22, and documents that describe it still have to cite it. Those are checked
# against `git show`, not against the checkout - a historical citation that no longer
# resolves on disk is correct, and must not be reported as stale.
CITATION = re.compile(r"(?:([0-9a-f]{7,40}):)?"
                      r"([A-Za-z0-9_\-./\u4e00-\u9fff]+\.py):(\d+)(?:-(\d+))?")

# Superseded documents keep their original wording by policy - the numbers in them are
# history, and rewriting them would quietly rewrite the reasoning too. What they must do
# is say so: the marker below, inside the document's 时效性提示 block, turns its dead
# pointers from "failure" into "declared". A new stale citation anywhere else still
# turns this command red, which is the property worth keeping.
EXEMPT = re.compile("行号引用已豁免")

_rev_lines = {}


def revision_length(rev, name):
    """Lines in `name` as of `rev`, or None when git says the path did not exist."""
    key = (rev, name)
    if key not in _rev_lines:
        import subprocess
        out = subprocess.run(["git", "-C", str(REPO), "show", "%s:%s" % (rev, name)],
                             capture_output=True)
        _rev_lines[key] = None if out.returncode else \
            len(out.stdout.decode("utf-8", "replace").splitlines())
    return _rev_lines[key]


def revision_exists(rev, name):
    """Any suffix match: docs cite `sched_env.py` bare as well as by full path."""
    import subprocess
    listing = _rev_lines.get((rev, "__list__"))
    if listing is None:
        out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", rev],
                             capture_output=True)
        listing = out.stdout.decode("utf-8", "replace").splitlines()
        _rev_lines[(rev, "__list__")] = listing
    wanted = name.replace("\\", "/")
    return any(p == wanted or p.endswith("/" + wanted) for p in listing)



def index_sources(roots):
    """basename -> list of (path, line_count), plus a set of relative paths."""
    by_name = defaultdict(list)
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in (".git", "__pycache__", "site-packages", ".venv", "venv")
                   for part in path.parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            by_name[path.name].append((path, lines))
    return by_name


def markdown_files(roots):
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            if ".git" in path.parts:
                continue
            yield path


def enclosing(source, number):
    """The nearest `def`/`class` at or above `number`, so --explain can name the target."""
    label = "?"
    for index in range(min(number, len(source)), 0, -1):
        line = source[index-1]
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "for ", "if ", "while ", "with ")):
            indent = len(line)-len(stripped)
            if indent == 0 or stripped.startswith(("def ", "class ")):
                return "%s  (L%d)" % (stripped.rstrip()[:70], index)
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", default=None,
                        help="directory to index for sources and scan for markdown")
    parser.add_argument("--quiet", action="store_true", help="print only the failures")
    parser.add_argument("--explain", action="store_true",
                        help="print the function each citation brackets, so a citation that "
                             "is in range but pointing at the wrong thing can be eyeballed")
    args = parser.parse_args()

    roots = [REPO, REPO.parent/"vmagentpro调研与改进方案"]
    roots = [r for r in (args.root or roots) if r.exists()]
    sources = index_sources(roots)
    problems = refs = exempt_total = 0
    per_document = {}
    for document in markdown_files(roots):
        text = document.read_text(encoding="utf-8", errors="replace")
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            for rev, name, start, end in CITATION.findall(line):
                refs += 1
                low = int(start)
                high = int(end) if end else low
                cited = name+":"+start+("-"+end if end else "")
                if rev:
                    # Revision-pinned: the working copy is irrelevant, git is the source.
                    if not revision_exists(rev, name):
                        rows.append((number, "REV MISSING", name, " @ %s" % rev))
                    elif "/" in name:
                        length = revision_length(rev, name)
                        if length is None:
                            rows.append((number, "REV MISSING", name, " @ %s" % rev))
                        elif high > length:
                            rows.append((number, "REV OUT OF RANGE", cited,
                                         " @ %s (%d lines)" % (rev, length)))
                    continue
                candidates = sources.get(Path(name).name, [])
                if len(name) > len(Path(name).name):
                    wanted = name.replace("\\", "/")
                    # An explicit relative path is a stronger statement than a basename:
                    # if nothing ends with that path the citation is wrong, and matching
                    # some other same-named file would hide the mistake.
                    narrowed = [c for c in candidates if c[0].as_posix().endswith(wanted)]
                    if not narrowed:
                        rows.append((number, "MISSING", cited, " (no file ends with %s)" % wanted))
                        continue
                    candidates = narrowed
                if not candidates:
                    rows.append((number, "MISSING", name, ""))
                    continue
                # A bare basename that matches two files is only worth reporting when the
                # citation cannot be true of either of them; `sched_env.py` exists twice in
                # this checkout (upstream and packaged) and the reader's question is
                # "did the code move", which this answers either way.
                if all(high > len(lines) or high < 1 for _, lines in candidates):
                    detail = ("  (%d candidates, longest has %d lines)"
                              % (len(candidates), max(len(lines) for _, lines in candidates))
                              if len(candidates) > 1 else "")
                    rows.append((number, "OUT OF RANGE", cited,
                                 " -> %s%s" % (candidates[0][0].as_posix(), detail)))
                elif len(candidates) > 1 and len(name) == len(Path(name).name):
                    rows.append((number, "BARE AMBIGUOUS (in range)", name, ""))
                if args.explain:
                    rows.append((number, "explain", cited,
                                 enclosing(candidates[0][1], low)
                                 + ("   ..   "+enclosing(candidates[0][1], high)
                                    if high != low else "")))
        live = [row for row in rows if row[1] not in ("BARE AMBIGUOUS (in range)", "explain")]
        declared = len(live) if EXEMPT.search("\n".join(text.splitlines()[:60])) else 0
        soft_only = sum(1 for row in rows if row[1].startswith("BARE"))
        per_document[document] = (len(live) - declared, soft_only, declared)
        exempt_total += declared
        if args.quiet and len(live) <= declared:
            continue
        for number, kind, target, extra in rows:
            if declared:
                kind = "declared (banner)"
            problems += 1 if kind not in ("BARE AMBIGUOUS (in range)", "explain",
                                          "declared (banner)") else 0
            print("%s:%d  %-22s %-30s %s" % (document, number, kind, target, extra))
    print("\n%d citation(s) checked against %d source file(s) in %s"
          % (refs, sum(len(v) for v in sources.values()), ", ".join(str(r) for r in roots)))
    for document, (bad, soft, declared) in sorted(per_document.items()):
        if bad or soft or declared:
            print("  %-88s %d stale/unresolvable, %d bare-but-in-range, %d declared"
                  % (document.name, bad, soft, declared))
    print("%d stale or unresolvable, %d declared by a 时效性提示 banner"
          % (problems, exempt_total))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
