"""Compare a labelled run's results against the committed ones, ignoring only the
provenance stamp (which by design records the source md5 and the run date)."""
import json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
COMMITTED = os.path.join(SP, "baseline", "results_committed")

FILES = ["e7_immediate_vs_deferred.json", "e8_false_mint_null.json",
         "e9_support_curve.json", "e13_creation_bias.json",
         "wikipedia_demo.json", "e5_order_dependence.json",
         "llm_graph_formation.json"]
VOLATILE = {"provenance", "date", "runtime_seconds", "wall_s", "seconds_per_run",
            "workers", "mean_seconds", "repair_s"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diffs(a, b, p="", out=None):
    if out is None:
        out = []
    if type(a) is not type(b):
        out.append((p, a, b)); return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((p + "/" + str(k), a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                diffs(a[k], b[k], p + "/" + str(k), out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((p + "/len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diffs(x, y, f"{p}[{i}]", out)
    elif a != b:
        out.append((p, a, b))
    return out


def main():
    label = sys.argv[1]
    d = os.path.join(SP, label, "results")
    print(f"comparing {label} against committed results (provenance/date/timings ignored)")
    total = 0
    for f in FILES:
        pa, pb = os.path.join(d, f), os.path.join(COMMITTED, f)
        if not os.path.exists(pa):
            print(f"  {f:38s} NOT PRODUCED"); continue
        A, B = strip(json.load(open(pa))), strip(json.load(open(pb)))
        raw_same = open(pa, "rb").read() == open(pb, "rb").read()
        dd = diffs(A, B)
        total += len(dd)
        tag = "IDENTICAL" if not dd else f"{len(dd)} DIFFS"
        print(f"  {f:38s} {tag}   (raw bytes identical: {raw_same})")
        for p, x, y in dd[:12]:
            print(f"      {p}: new={x!r} committed={y!r}")
        if len(dd) > 12:
            print(f"      ... {len(dd)-12} more")
    print(f"TOTAL DIFFS: {total}")


main()
