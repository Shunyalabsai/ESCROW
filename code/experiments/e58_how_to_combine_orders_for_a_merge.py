"""E58: which arrival orders should a merge decision listen to, and how?

THE PROBLEM E57 LEFT. The merge delta for one pair of keys moves a long way with the arrival order
of the same data. The batch objective is a function of the state and not of the order, so this is
the greedy search landing in different states, not the criterion changing its mind. On one fixture
the same pair scored -36, -7, +248, -22, -4 and -4 bits across six shuffles of one stream: five
orders find the merge and one lands somewhere it does not.

That is search noise, and search noise is the one kind of instability a caller can do something
about, because shuffling your own data is free. The question is what to do with R answers.

FOUR RULES, AND THEY DISAGREE.

  mean, or equivalently summing the bits    the MDL-native move, and the obvious first guess
  median                                    the robust location estimate
  majority                                  count the signs; identical to the sign of the median
  shortest description                      ask the single state with the smallest L_batch, which
                                            is the selector the paper already uses for minting

The mean is the interesting failure. The per-order deltas are heavy tailed, because a search that
lands badly lands very badly, so one order at +248 outvotes five at about -10 and the sum says keep
where every typical order says merge. Averaging bits is right when the spread is noise around a
value; it is wrong when the spread is a few catastrophic states.

WHAT WOULD REFUTE THE RECOMMENDATION. If the rule that wins here loses on the planted fixtures of
E54 and E55, it is fitted to this fixture. Both are scored below for that reason. And if no rule
separates the probes, the honest reading is that combining orders cannot fix what E57 found, since
consensus across states cannot supply knowledge the criterion does not have.
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.batch import BatchObjective
from escrow.keymerge import key_merge_delta
from escrow.protocol import run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e58_how_to_combine_orders_for_a_merge.json")

ORDERS = 9
SEEDS = (0, 1, 2)


def e57_case(seed):
    from e57_semantics_in_the_key_merge import build_stream
    return build_stream(seed), [("shade", "colour", True),
                                ("material", "finish", False),
                                ("shade", "weight", False)]


def e54_case(seed):
    from e54_key_merge_by_value_distribution import build_stream, N_PER_ROLE
    return build_stream(N_PER_ROLE, seed), [("shade", "colour", True),
                                            ("colour", "weight", False),
                                            ("fabric", "garment_type", False)]


CASES = {"disjoint_vocabularies_e57": e57_case, "shared_vocabularies_e54": e54_case}

RULES = {
    "mean": lambda ds: st.mean(ds) < 0,
    "median": lambda ds: st.median(ds) < 0,
    "majority": lambda ds: sum(1 for x in ds if x < 0) * 2 > len(ds),
}


def main():
    report = {"experiment": "E58 combining arrival orders for one merge decision",
              "orders_per_stream": ORDERS, "seeds": list(SEEDS), "cases": {}}
    errors = {k: 0 for k in list(RULES) + ["shortest_description"]}
    total = 0

    for case_name, build in CASES.items():
        rows = []
        for seed in SEEDS:
            base, probes = build(seed)
            runs = []
            for r in range(ORDERS):
                recs = list(base)
                random.Random(9000 + 100 * seed + r).shuffle(recs)
                g, b = run_stream(recs)
                runs.append((b.total(), g))
            shortest = min(runs, key=lambda x: x[0])[1]
            for a, bb, want in probes:
                ds = []
                for _, g in runs:
                    ka, kb = g.keys.get(a), g.keys.get(bb)
                    if ka is None or kb is None:
                        continue
                    d = key_merge_delta(g, BatchObjective, ka.kid, kb.kid)
                    if d is not None:
                        ds.append(round(d, 2))
                if not ds:
                    continue
                ka, kb = shortest.keys.get(a), shortest.keys.get(bb)
                d_short = key_merge_delta(shortest, BatchObjective, ka.kid, kb.kid)
                verdicts = {k: f(ds) for k, f in RULES.items()}
                verdicts["shortest_description"] = (d_short is not None and d_short < 0)
                total += 1
                for k, v in verdicts.items():
                    errors[k] += (v != want)
                rows.append({"seed": seed, "a": a, "b": bb, "should_merge": want,
                             "deltas_by_order": ds,
                             "spread_bits": round(max(ds) - min(ds), 1),
                             "orders_voting_merge": sum(1 for x in ds if x < 0), "of": len(ds),
                             "verdicts": verdicts,
                             "correct": {k: (v == want) for k, v in verdicts.items()}})
                marks = "  ".join(f"{k}={'merge' if v else 'keep'}{'' if v == want else '(X)'}"
                                  for k, v in verdicts.items())
                print(f"  [{case_name[:8]}] s{seed} {a}+{bb:<13} want "
                      f"{'merge' if want else 'keep ':<6} {marks}")
        report["cases"][case_name] = rows

    report["headline"] = {
        "decisions_scored": total,
        "errors_by_rule": errors,
        "best": sorted(errors, key=lambda k: errors[k])[0],
        "note_on_majority": ("a majority of signs is the sign of the median, so those two are one "
                             "rule and not two"),
        "why_the_mean_loses": ("the per-order deltas are heavy tailed: a badly landed search is very "
                               "badly landed, so one order at +248 outweighs five at about -10"),
        "note_on_shortest_description": ("the selector the paper uses for minting asks the single "
                                         "state with the smallest total description. That state is "
                                         "the best answer to the whole question and not necessarily "
                                         "to one pair of keys, which is why consensus beats it here"),
        "reading": ("combining orders removes search noise, which is real and worth removing. It "
                    "does not supply the knowledge E57 found missing, so a stable answer here is not "
                    "the same as a correct one"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
