"""E28: why the arrival order matters so much, and what to do about it without adding a parameter.

THE OBJECTION. On the one real stream the method's agreement with the category labels swings from
0.64 to 0.99 with the arrival order. A reviewer reading that, next to a density baseline whose score
barely moves, concludes that the rule is unstable where it counts. E21 makes it worse: given the same
categorical input, HDBSCAN at its own library default averages 0.7926 with a spread of 0.009, while a
single ESCROW run averages 0.8691 over sixty orders and swings across a third of the scale.

THE QUESTION THIS ANSWERS. Is that spread a modelling failure or a search failure? The paper proves
the batch objective is a function of the state and not of the arrival order, while the greedy one-pass
sequence that reaches a state is not. So there are two possibilities and they call for opposite
responses:

  If the objective cannot tell the good runs from the bad, the model is wrong and the theory has to
  change.

  If the objective ranks them correctly, the model is right and the search is what varies, and the
  answer is more search, not a tuned quantity.

WHAT THIS RUNS. Sixty arrival orders of the same 320 records, under the shipped protocol. For each
order it records the adjusted Rand index against the category labels, the node count, and the total
batch codelength, which is the objective the repair pass descends. Then two things:

  1. The correlation between accuracy and description length across orders. This is the diagnostic.

  2. The curve for a selection rule that uses no labels at all: run k orders, keep the one with the
     shortest description, report the accuracy of the run you kept. The rule is model selection by
     the paper's own criterion. It adds no quantity that has to be calibrated on data; it spends
     compute instead, and k is a budget the user sets knowing what it buys, in the same way the
     candidate pool budget is. The curve also reports the best run in each sample, because the
     difference between what the codelength picks and what was available is the price of the
     objective being a proxy for accuracy rather than accuracy itself.

WHAT WOULD HAVE FALSIFIED THE FIX. A correlation near zero, or of the wrong sign, would have meant
the objective is blind to the difference and that selecting on it is no better than selecting at
random. The number is reported whichever way it comes out.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.protocol import run_stream, describe as protocol_describe   # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e4_baseline_army import wikipedia, _ari                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e28_order_variance.json")

ORDERS = 60
KS = (1, 2, 3, 5, 10, 20, 40)

# E21, matched representation, the same 320 records: HDBSCAN at its library default.
HDBSCAN_DEFAULT = {"mean": 0.7926, "min": 0.7907, "max": 0.7993,
                   "source": "results/e21_matched_representation.json"}


def one_order(seed):
    recs, truth = wikipedia(RESULTS, seed)
    t0 = time.time()
    g, b = run_stream(recs)
    lab = [-1] * len(recs)
    for v in g.nodes.values():
        for m in v.members:
            lab[m - 1] = v.nid
    return {"seed": seed,
            "ARI": round(_ari(truth, lab), 4),
            "K": int(g.K),
            "bits": round(b.total(), 1),
            "background": int(sum(1 for x in lab if x == -1)),
            "seconds": round(time.time() - t0, 2)}


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / den if den else 0.0


def main():
    runs = []
    for s in range(ORDERS):
        r = one_order(s)
        runs.append(r)
        print(f"  order {s:2d}  ARI {r['ARI']:.4f}  K {r['K']}  {r['bits']:.1f} bits", flush=True)

    ari = [r["ARI"] for r in runs]
    bits = [r["bits"] for r in runs]
    rho = pearson(ari, bits)
    best = min(runs, key=lambda r: r["bits"])
    worst = max(runs, key=lambda r: r["bits"])

    # Subsets are drawn at random rather than enumerated in order. Enumerating combinations of a
    # small pool makes the large-k rows nearly deterministic, because almost every subset contains
    # the same shortest-description run, and that flatters both the mean and the spread.
    rng = random.Random(0)
    curve = []
    for k in KS:
        picks, oracle = [], []
        for _ in range(4000):
            samp = rng.sample(runs, k)
            picks.append(min(samp, key=lambda r: r["bits"])["ARI"])
            oracle.append(max(r["ARI"] for r in samp))
        curve.append({"k": k,
                      "mean_ARI": round(st.mean(picks), 4),
                      "sd": round(st.pstdev(picks), 4),
                      "min": round(min(picks), 4), "max": round(max(picks), 4),
                      "oracle_best_in_sample": round(st.mean(oracle), 4),
                      "gap_to_oracle": round(st.mean(oracle) - st.mean(picks), 4),
                      "draws": len(picks)})
        print(f"  keep the shortest of {k:2d}: mean ARI {curve[-1]['mean_ARI']:.4f} "
              f"sd {curve[-1]['sd']:.4f}  gap to the best run in the sample "
              f"{curve[-1]['gap_to_oracle']:.4f}", flush=True)

    report = {
        "experiment": "E28 order variance",
        "question": "is the spread across arrival orders a modelling failure or a search failure",
        "protocol": protocol_describe(),
        "stream": "the 320 Wikipedia infobox records, one permutation per seed",
        "orders": ORDERS,
        "per_order": runs,
        "single_run": {"mean_ARI": round(st.mean(ari), 4), "sd": round(st.pstdev(ari), 4),
                       "min": min(ari), "max": max(ari)},
        "diagnostic": {
            "correlation_ARI_vs_codelength": round(rho, 3),
            "shortest_description": best,
            "longest_description": worst,
            "spread_bits": round(max(bits) - min(bits), 1),
            "reading": ("A negative correlation means the objective prefers the accurate runs, so the "
                        "model is not what varies; the greedy sequence is. A correlation near zero "
                        "would have meant the objective is blind to the difference and that the "
                        "theory, not the search, needed changing."),
        },
        "selection_by_description_length": {
            "rule": ("run k arrival orders and keep the one whose total batch codelength is shortest; "
                     "the labels are never consulted"),
            "adds_a_calibrated_quantity": False,
            "what_it_costs": "k times the compute, and k passes rather than one",
            "curve": curve,
        },
        "hdbscan_default_same_representation": HDBSCAN_DEFAULT,
    }
    report["headline"] = {
        "correlation": round(rho, 3),
        "single_run_mean": report["single_run"]["mean_ARI"],
        "best_of_three_mean": next(c["mean_ARI"] for c in curve if c["k"] == 3),
        "gap_to_the_best_run_in_the_sample": next(c["gap_to_oracle"] for c in curve if c["k"] == 10),
        "hdbscan_default_mean": HDBSCAN_DEFAULT["mean"],
        "verdict": ("search, not model" if rho < -0.3 else
                    "the objective does not rank the runs, so the model is implicated"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
