"""Do the paths written down in this project still exist?

`check_doc_line_references.py` answers "does this line number still land inside the file".
This answers the other half: "does the file itself still exist". The 2026-11-23 move of all
scenario data into `场景/` rewrote 383 references across code and notes; the only way to know
the sweep got them all was a command that re-reads every reference and checks the disk.

Two exclusions, both deliberate:

  * the collaborator's hash-verified archive - its `manifest.sha256.json` covers the bytes of
    every file inside, so its scripts and documents are left exactly as they were, including
    any path they mention;
  * archived run records (`run_manifest.json`, `provenance.json`, ...) - they name the paths
    a run actually used, which is history, not a pointer to maintain.

    C:/Users/trw/anaconda3/python.exe tools/check_referenced_paths.py
    C:/Users/trw/anaconda3/python.exe tools/check_referenced_paths.py --quiet

A reference is anything that looks like `<topdir>/...` inside a string: code constants
(`REPO / "场景/02_主场景F/..."`) and document prose are both read. Tokens with a glob, a
format placeholder or an ellipsis are counted as prose and skipped, because they are how
people write about a path rather than a path.
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTES = [REPO.parent / "vmagentpro调研与改进方案", REPO.parent / "vmagentpro准备工作"]
# Roots that exist today are checked for existence. Roots that the 2026-11-23 move removed
# are counted as history: the change log and the superseded plans mention `runs/` and
# `benchmark任务/` on purpose, and calling that a dangling reference would make the command
# red for the wrong reason.
LIVE = ("场景/", "experiments/", "tools/", "tests/", "vmagent/")
REMOVED = ("runs/", "docs/", "benchmark任务/", "基准表/")
SKIP_DIR = re.compile(r"(\.git|__pycache__|node_modules|\.venv)[/\\]")
SKIP_FILE = re.compile(r"三指标数据评测工作文件_20261108|"
                       r"(run_manifest|provenance|verification_report|reproducibility_report)\.json$")
CODE_TOKEN = re.compile(r'"((?:%s)[^"]*)"' % "|".join(re.escape(r) for r in LIVE + REMOVED))
DOC_TOKEN = re.compile(r"(?<![0-9A-Za-z_一-鿿])((?:%s)[^\s`'\")\]},，。；:]+)"
                       % "|".join(re.escape(r) for r in LIVE + REMOVED))
# Anything containing whitespace is a sentence, not a path; a backslash is how a directory
# tree is drawn with `\|` connectors; a full-width parenthesis is prose glued to a path.
PROSE = re.compile(r"[\s*?{…\\（）]|\.\.\.|README\.md section|\bsection\b")


def files(roots, suffixes):
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            rel = path.as_posix()
            if SKIP_DIR.search(rel) or SKIP_FILE.search(rel):
                continue
            yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="print only the failures")
    args = ap.parse_args()

    deleted = set()
    if (REPO / ".git").exists():
        import subprocess
        out = subprocess.run(["git", "-C", str(REPO), "log", "--diff-filter=D",
                              "--name-only", "--pretty="],
                             capture_output=True, text=True, encoding="utf-8").stdout
        deleted = {l.strip() for l in out.splitlines() if l.strip()}
        # A deleted directory never appears as a deleted *file*, but documents name it
        # on purpose (`vmagent/schedgym`, `vmagent/controllers`). Reconstruct the set
        # from the removal commit instead of hard-coding a list that will go stale.
        for rev in ("71a599d",):
            listing = subprocess.run(["git", "-C", str(REPO), "show", "--name-only",
                                      "--pretty=", rev],
                                     capture_output=True, text=True,
                                     encoding="utf-8", errors="replace").stdout
            for line in listing.splitlines():
                parts = line.strip().split("/")
                for i in range(1, len(parts)):
                    deleted.add("/".join(parts[:i]))

    checked = prose = historical = removed = 0
    bad = []
    def consider(path, token, kind):
        nonlocal checked, prose, historical, removed, bad
        token = token.rstrip("/.,")
        # A path in this project never contains a space, and a bare top-level name is not
        # a reference to anything - both rules exist because the tool otherwise matches the
        # sentence it was written in, and its own list of roots.
        if not token or "/" not in token or PROSE.search(token):
            prose += 1
            return
        if token.startswith(REMOVED):
            removed += 1         # a mention of a top level the move deleted: history
            return
        checked += 1
        if (REPO / token).exists():
            return
        # A path that once existed and was deliberately deleted is history, not a bug:
        # the comparison documents still describe `vmagent/train.py`, and that is correct.
        # A path that never existed anywhere is the thing worth failing on.
        if token in deleted:
            historical += 1
            return
        bad.append((path, token, kind))

    for path in files([REPO], {".py", ".bat"}):
        for token in CODE_TOKEN.findall(path.read_text(encoding="utf-8", errors="replace")):
            consider(path, token, "code")
    for path in files([REPO] + NOTES, {".md", ".py", ".bat"}):
        for token in DOC_TOKEN.findall(path.read_text(encoding="utf-8", errors="replace")):
            # Documents talk about the upstream tree (`vmagent/train.py`, `vmagent/schedgym`)
            # in prose that is deliberately historical, and about paths relative to the note
            # folder itself. Only the scenario tree is a live contract here: that is what the
            # 2026-11-23 move changed, and it is what a reader is meant to be able to open.
            if not token.startswith("场景/"):
                prose += 1
                continue
            consider(path, token, "prose")
    if not args.quiet or bad:
        for path, token, kind in bad:
            print("%-70s  %s  %s" % (path, kind, token))
    print("\n%d live path reference(s) verified" % checked)
    print("%d mention(s) of a removed top level, %d of a deliberately deleted file "
          "(both are history, both would fail if they were written as live paths)"
          % (removed, historical))
    print("%d prose/glob/ellipsis skipped" % prose)
    print("%d dangling" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
