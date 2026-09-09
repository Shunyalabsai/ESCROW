"""E71: does the agreement split actually predict being right, or is it decoration?

WHY THIS DECIDES THE WHOLE CONFIDENCE TRACK. Running R arrival orders gives, for free, a number per
record: how many of them placed it with the same neighbours. E58 noticed that every merge decision
landing at 0 of 9 or 8 of 9 was correct while every error sat in the 3-to-7 band, and that looked
like a confidence measure with nothing to set. It is only worth having if it tracks correctness. A
confidence that does not is worse than none, because it invites trust it has not earned.

WHAT IS MEASURED. For every pair of records, the fraction of R arrival orders that place them in the
same node, and whether the truth agrees they belong together. Bin by that fraction and read off the
observed accuracy in each bin. A calibrated signal gives a curve that rises; decoration gives a flat
line.

Pairs rather than records, because a record has no correctness on its own: what the metrics score,
and what the system actually asserts, is whether two records belong together.

THE TWO STREAMS ARE CHOSEN TO DISAGREE. The encyclopedia stream is where consensus helped at 320
records and hurt at 120, so the confidence has something real to say there. The planted stream is
where every order already agrees, so the curve should be almost entirely at the confident end with
almost nothing in the middle: a confidence that is well calibrated on easy data and silent about it
is behaving correctly.

WHAT WOULD REFUTE IT. A flat curve, or one that rises on one stream and not the other. Then the split
is not a confidence and C3, the abstention rule built on it, should not be written.
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e71_is_the_confidence_calibrated.json")
R = 9
MAX_PAIRS = 60000


def co_placement(recs, R, seed=0):
    """For each pair, how many of R arrival orders put them in the same node."""
    n = len(recs)
    together = collections.Counter()
    for r in range(R):
        order = list(range(n))
        random.Random(seed * 1009 + r).shuffle(order)
        g, _ = run_stream([recs[i] for i in order])
        lab_run = record_labels(g, n)
        lab = [None] * n
        for pos, orig in enumerate(order):
            lab[orig] = lab_run[pos]
        by = collections.defaultdict(list)
        for i, l in enumerate(lab):
            if l is not None and l != -1:
                by[l].append(i)
        for members in by.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    together[(members[a], members[b])] += 1
    return together


def calibrate(recs, truth, R, seed=0):
    n = len(recs)
    together = co_placement(recs, R, seed)
    rng = random.Random(0)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > MAX_PAIRS:
        all_pairs = rng.sample(all_pairs, MAX_PAIRS)
    # The first version of this scored accuracy, which came out at 1.0000 in every bin. That was the
    # metric failing, not the confidence succeeding: with three classes over hundreds of records
    # almost every pair is truly apart, the system says apart, and accuracy is dominated by true
    # negatives it was never in doubt about.
    #
    # The calibration question is the conditional one: given that N of R orders placed this pair
    # together, how often are they actually the same kind? A calibrated signal makes that fraction
    # rise with N. Decoration makes it flat.
    bins = collections.defaultdict(lambda: [0, 0])
    for i, j in all_pairs:
        c = together.get((i, j), 0)
        same = truth[i] == truth[j] and truth[i] is not None
        bins[c][0] += 1
        bins[c][1] += same
    return {str(c): {"pairs": v[0], "truly_same_kind": round(v[1] / v[0], 4), "hits": v[1]}
            for c, v in sorted(bins.items())}


def main():
    streams = {}
    import llm_graph_formation as L
    wf = L.build_wikipedia()
    streams["encyclopedia_320"] = (wf["records"], wf["truth"])
    streams["encyclopedia_120"] = (wf["records"][:120], wf["truth"][:120])
    from e4_baseline_army import planted8
    recs, truth = planted8(n=1500, seed=7)
    streams["planted8_1500"] = (recs, truth)

    report = {"experiment": "E71 is the agreement split a calibrated confidence",
              "R": R, "unit": "pairs of records", "streams": {}}

    for name, (recs, truth) in streams.items():
        cal = calibrate(recs, truth, R, seed=3)
        report["streams"][name] = cal
        print(f"\n{name}: {len(recs)} records, {R} arrival orders\n")
        print(f"  {'orders agreeing':>16} {'pairs':>10} {'P(same kind)':>12}")
        for c in sorted(cal, key=lambda x: int(x)):
            v = cal[c]
            bar = "#" * int(round(v["truly_same_kind"] * 40))
            print(f"  {c + ' of ' + str(R):>16} {v['pairs']:>10} {v['truly_same_kind']:>10.4f}  {bar}")

    # is the curve monotone at the ends, which is the property that matters
    summary = {}
    for name, cal in report["streams"].items():
        ks = sorted(cal, key=lambda x: int(x))
        summary[name] = {
            "P_same_when_0_of_R_agree": cal.get("0", {}).get("truly_same_kind"),
            "P_same_when_all_R_agree": cal.get(str(R), {}).get("truly_same_kind"),
            "P_same_in_the_split_bins": (
                round(sum(cal[k]["hits"] for k in ks if 0 < int(k) < R)
                      / max(sum(cal[k]["pairs"] for k in ks if 0 < int(k) < R), 1), 4)),
            "pairs_in_the_confident_bins": sum(cal[k]["pairs"] for k in ks if int(k) in (0, R)),
            "pairs_in_the_split_bins": sum(cal[k]["pairs"] for k in ks if 0 < int(k) < R)}
    report["headline"] = {
        "by_stream": summary,
        "reading": ("calibration is the conditional: given N of R orders agreed, how often are the "
                    "two records really the same kind. It must rise with N. If the split bins sit "
                    "between the two ends rather than at one of them, the split is telling the "
                    "caller something the verdict alone does not"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
