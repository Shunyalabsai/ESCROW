"""E39: is the repair cadence a calibrated parameter, or a choice the objective can make itself?

THE PROBLEM, FOUND WHILE PROBING HIERARCHIES. The shipped protocol repairs every 100 records. That
number is declared as an operational choice, and E19 checked the candidate pool budget for exactly
this kind of effect, but nobody checked the cadence. It turns out to decide the answer. On a planted
two-level stream, repairing every 100 returns the three families and deferring every repair to one
final pass returns the eight or nine species, at about 9,300 fewer bits. On the real cover benchmark
of E37 the same deferral takes the run from omega 0.2044 to -0.0162. A quantity that swings the
output that far, with no rule for setting it, is a calibrated parameter in all but name, and the
paper claims not to have one.

THE QUESTION. E28 met the same shape of problem for arrival order and answered it without adding a
quantity: run several orders and keep the one whose total batch codelength is shortest, which is
model selection by the paper's own criterion, spending compute instead of calibration. The same
answer is available here. Run the stream at several cadences and keep the state with the shortest
description. The labels are never consulted.

WHAT DECIDES WHETHER THAT WORKS. Only one thing: whether the batch codelength ranks the cadences the
way the truth does. So this measures, per stream, the codelength AND the agreement with the truth at
every cadence, and then asks what the codelength would have picked.

  If shortest-description picks a state at or near the best available agreement, the cadence is not a
  parameter. It is a resource, like the pool budget and like the restart count, and the paper can say
  so with a number.

  If shortest-description picks a poor state, the objective cannot see the difference, the cadence is
  a genuine tuned quantity, and the paper must concede it.

WHAT WOULD FALSIFY THE FIX, stated before the run: a stream where the shortest description is not
among the better answers. Reported either way, per stream, with the gap to the best cadence available.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np                                                      # noqa: E402

from escrow.batch import BatchObjective                                 # noqa: E402
from escrow.protocol import new_run                                     # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e4_baseline_army import _ari, planted8, two_group, wikipedia   # noqa: E402
from experiments.e25_multi_membership import (omega_index, _shared_counts)      # noqa: E402
from experiments.e35_nesting import two_level                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
WIKIDATA = os.path.join(RESULTS, "wikidata_cover", "wikidata_people.json")
OUT = os.path.join(RESULTS, "e39_cadence_selection.json")

# The cadences offered. `None` means no repair during the stream, one final full pass. All of them
# end with the same final full repair, so they differ only in how often repair runs on the way.
CADENCES = (50, 100, 200, 500, 1000, None)


def run(recs, every):
    g, b = new_run(BatchObjective)
    for i, r in enumerate(recs):
        g.process(r)
        if every is not None and (i + 1) % every == 0:
            b.repair()
    b.repair(full=True)
    return g, b


def partition_labels(g, n):
    lab = [-1] * n
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    return lab


def cover_matrix(g, n):
    nids = sorted(g.nodes)
    M = np.zeros((n, len(nids)), dtype=bool)
    for c, nid in enumerate(nids):
        for m in g.nodes[nid].members:
            M[m - 1, c] = True
    return M


def load_wikidata():
    d = json.load(open(WIKIDATA, encoding="utf-8"))
    recs = [v["record"] for v in d.values()]
    labels = [set(v["labels"]) for v in d.values()]
    lab = sorted({x for s in labels for x in s})
    ix = {l: i for i, l in enumerate(lab)}
    M = np.zeros((len(recs), len(lab)), dtype=bool)
    for i, s in enumerate(labels):
        for l in s:
            M[i, ix[l]] = True
    return recs, M


def main():
    streams = []
    for sd in (0, 1):
        recs, fam, sp = two_level(sd)
        streams.append({"name": f"nesting_s{sd}", "recs": recs,
                        "score": ("ari", [f"{a}_{c}" for a, c in sp])})
    r, t = planted8(seed=7)
    streams.append({"name": "planted8", "recs": r, "score": ("ari", t)})
    r, t = two_group(seed=0)
    streams.append({"name": "two_group", "recs": r, "score": ("ari", t)})
    r, t = wikipedia(RESULTS, 0)
    streams.append({"name": "wikipedia", "recs": r, "score": ("ari", t)})
    if os.path.exists(WIKIDATA):
        recs, Mt = load_wikidata()
        streams.append({"name": "wikidata_cover", "recs": recs, "score": ("omega", Mt)})

    report = {"experiment": "E39 the repair cadence, selected rather than set",
              "question": ("the cadence decides the answer; can the batch codelength choose it "
                           "without ever seeing a label?"),
              "rule": ("run the stream at each cadence and keep the state whose total batch "
                       "codelength is shortest; the labels are never consulted"),
              "adds_a_calibrated_quantity": False,
              "what_it_costs": "one pass per cadence offered",
              "cadences": [c if c is not None else "final pass only" for c in CADENCES],
              "falsifier": "a stream where the shortest description is not among the better answers",
              "streams": []}

    for s in streams:
        recs = s["recs"]
        n = len(recs)
        kind, truth = s["score"]
        T = _shared_counts(truth) if kind == "omega" else None
        rows = []
        for every in CADENCES:
            g, b = run(recs, every)
            if kind == "ari":
                q = _ari(truth, partition_labels(g, n))
            else:
                q = omega_index(truth, cover_matrix(g, n), T=T)[0]
            rows.append({"cadence": every if every is not None else "final only",
                         "bits": round(b.total(), 1), "K": int(g.K), "quality": round(q, 4)})
            print(f"  {s['name']:16s} cadence {str(every):11s} bits {b.total():10.0f} "
                  f"K {g.K:3d} {kind} {q:.4f}", flush=True)
        picked = min(rows, key=lambda r: r["bits"])
        best = max(rows, key=lambda r: r["quality"])
        worst = min(rows, key=lambda r: r["quality"])
        cell = {"stream": s["name"], "metric": kind, "n": n, "rows": rows,
                "shortest_description": picked,
                "best_available": best, "worst_available": worst,
                "gap_to_best": round(best["quality"] - picked["quality"], 4),
                "spread_over_cadences": round(best["quality"] - worst["quality"], 4)}
        report["streams"].append(cell)
        print(f"  -> shortest description picks cadence {picked['cadence']} at "
              f"{picked['quality']:.4f}; best available {best['quality']:.4f}; "
              f"spread {cell['spread_over_cadences']:.4f}\n", flush=True)

    gaps = [c["gap_to_best"] for c in report["streams"]]
    spreads = [c["spread_over_cadences"] for c in report["streams"]]
    report["headline"] = {
        "mean_gap_between_what_the_codelength_picks_and_the_best_cadence": round(st.mean(gaps), 4),
        "worst_gap": round(max(gaps), 4),
        "mean_spread_across_cadences_if_chosen_badly": round(st.mean(spreads), 4),
        "streams_where_the_pick_is_the_best": sum(1 for g in gaps if g <= 1e-9),
        "streams": len(gaps),
        "verdict": ("the codelength ranks the cadences, so the cadence is a resource the user "
                    "spends rather than a quantity fitted to data"
                    if max(gaps) < 0.05 else
                    "the codelength does not reliably rank the cadences, so the cadence is a tuned "
                    "quantity and the paper must concede it"),
    }
    print(json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
