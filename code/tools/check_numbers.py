"""Flag any superseded number still printed in the paper, the talk or the working notes.

Every engine fix moves the measured values. A number that was true last week reads exactly like a
number that is true today, so the only reliable guard is a list of values that are known to be
wrong now. code/tools/superseded_numbers.json holds that list, one entry per quantity, with the
results file the current value comes from and why the old one died.

    python3 code/tools/check_numbers.py            # check, exit 1 if anything stale is found
    python3 code/tools/check_numbers.py --current  # also print the current value of each quantity

Add a row whenever a printed number changes. That is the whole discipline.

A superseded value can also turn up as the current, correct value of a different quantity, since the
check is a substring match over lines. The manifest's "exceptions" list handles exactly that case: an
entry names the quantity id, the file and a marker the line must contain, and suppresses the hit only
there. Each one records why. Never add an exception to silence a number that really is stale.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST = os.path.join(os.path.dirname(__file__), "superseded_numbers.json")
TARGETS = ["paper/sections/*.tex", "paper/main.tex", "talk/index.html",
           "PAPER.md", "METHOD.md", "README.md", "findings/NARRATIVE.md"]


def dig(obj, path):
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def current_value(source):
    if ":" not in source:
        return None
    rel, path = source.split(":", 1)
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        return None
    try:
        data = json.load(open(full))
    except Exception:
        return None
    return dig(data, path.split(" ")[0])


def main():
    man = json.load(open(MANIFEST))
    files = []
    for pat in TARGETS:
        files.extend(sorted(glob.glob(os.path.join(ROOT, pat))))
    exempt = man.get("exceptions", [])

    def is_exempt(qid, rel, line):
        for e in exempt:
            if e["id"] == qid and e["file"] == rel and e["line_contains"] in line:
                return True
        return False

    hits = []
    for q in man["quantities"]:
        for f in files:
            try:
                text = open(f, encoding="utf-8").read()
            except Exception:
                continue
            for line_no, line in enumerate(text.split("\n"), start=1):
                rel = os.path.relpath(f, ROOT)
                for bad in q["superseded"]:
                    if bad in line and not is_exempt(q["id"], rel, line):
                        hits.append((rel, line_no, q["id"], bad, line.strip()[:110]))
    if "--current" in sys.argv:
        print("current values")
        for q in man["quantities"]:
            print("  %-22s %s" % (q["id"], current_value(q["source"])))
        print()
    if not hits:
        print("no superseded number found in %d files" % len(files))
        return 0
    print("STALE NUMBERS (%d)" % len(hits))
    for f, ln, qid, bad, line in hits:
        print("  %s:%d  [%s]  %r" % (f, ln, qid, bad))
        print("      %s" % line)
    print("\nEach one must be retyped from the results file named in %s."
          % os.path.relpath(MANIFEST, ROOT))
    return 1


if __name__ == "__main__":
    sys.exit(main())
