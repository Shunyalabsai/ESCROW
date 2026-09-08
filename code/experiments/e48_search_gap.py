"""E48: how much of the loss is the criterion, and how much is the search that looks for it?

WHY. The scope proposition explains MusicBrainz 20K and ReVerb45K, which ask for groups of about
two. It does not explain the cover fixture or the Wikidata cover benchmark, where both scope clauses
hold and the rule still loses. E46 removed three candidate explanations. E47 then found that on the
cover fixture the planted truth is a SHORTER description than the state the engine reaches, by about
3{,}800 bits, which says the criterion ranks the right answer above our own answer and the search is
what fails.

This decomposes the loss into its two parts, on the same objective and the same records:

  planted truth            what the data was generated from, omega 1 by construction
  repair from the truth    the best state the shipped moves can reach when handed a good start
  engine from scratch      what we actually report

The gap between the first two is the OBJECTIVE's gap: how far L_batch's own preference sits from
the truth. The gap between the last two is the SEARCH's gap: how much the one-pass greedy plus a
merge-only repair leaves on the table. Both are measured in bits and in agreement, so they can be
compared rather than argued about.

WHY IT MATTERS WHICH. An objective gap is a modelling defect and no amount of engineering repairs
it. A search gap is an engineering defect, and the paper's claim that the price is the contribution
survives it intact. They call for opposite next steps, so the paper should not guess.

WHAT WOULD FALSIFY THE READING. If the engine's state is already at or below the reachable optimum,
there is no search gap and the loss is the criterion's. Reported either way.
"""
from __future__ import annotations
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

from escrow.batch import BatchObjective
from escrow.provenance import stamped
from e47_does_the_objective_prefer_the_truth import (
    cover_fixture, background_only, install_oracle, engine_state)

OUT = os.path.join(ROOT, "results", "e48_search_gap.json")


def pred_sets(g, n):
    ps = [set() for _ in range(n)]
    for v in g.nodes.values():
        for m in v.members:
            ps[m - 1].add(v.nid)
    return [frozenset(x) for x in ps]


def omega_index(T, P, n):
    """The omega index: agreement on how many groups each PAIR of records shares, chance corrected."""
    tj, pj, agree, tot = {}, {}, 0, 0
    for i in range(n):
        Ti, Pi = T[i], P[i]
        for j in range(i + 1, n):
            t, p = len(Ti & T[j]), len(Pi & P[j])
            tj[t] = tj.get(t, 0) + 1
            pj[p] = pj.get(p, 0) + 1
            tot += 1
            if t == p:
                agree += 1
    obs = agree / tot
    exp = sum(tj.get(k, 0) * pj.get(k, 0) for k in set(tj) | set(pj)) / (tot * tot)
    return (obs - exp) / (1 - exp) if exp < 1 else 0.0


def three_states(recs, truth, gen):
    n = len(recs)
    g, b = background_only(recs)
    install_oracle(g, truth, gen)
    out = {"truth": {"K": int(g.K), "L": round(b.total(), 1),
                     "omega": round(omega_index(truth, pred_sets(g, n), n), 4)}}
    b.repair(full=True)
    out["repair_from_truth"] = {"K": int(g.K), "L": round(b.total(), 1),
                                "omega": round(omega_index(truth, pred_sets(g, n), n), 4)}
    g2, b2 = engine_state(recs)
    out["engine_from_scratch"] = {"K": int(g2.K), "L": round(b2.total(), 1),
                                  "omega": round(omega_index(truth, pred_sets(g2, n), n), 4)}
    out["objective_gap_bits"] = round(out["repair_from_truth"]["L"] - out["truth"]["L"], 1)
    out["search_gap_bits"] = round(out["engine_from_scratch"]["L"]
                                   - out["repair_from_truth"]["L"], 1)
    return out


def main():
    report = {"experiment": "E48 how much of the loss is the criterion and how much is the search",
              "definitions": {
                  "objective_gap_bits": "L(repair from truth) - L(truth); negative means the "
                                        "objective prefers something other than the truth",
                  "search_gap_bits": "L(engine from scratch) - L(repair from truth); positive means "
                                     "the search leaves bits on the table"},
              "cover_fixture": []}

    print("the E33/E34 cover fixture, both scope clauses met")
    print(f"{'seed':>4} | {'truth':>18} | {'repair from truth':>18} | {'engine':>18}")
    print(f"{'':>4} | {'K':>2} {'L':>8} {'om':>6} | {'K':>2} {'L':>8} {'om':>6} | {'K':>2} {'L':>8} {'om':>6}")
    for seed in (0, 1, 2):
        recs, truth, gen, _ = cover_fixture(shared=12, seed=seed)
        r = three_states(recs, truth, gen)
        r["seed"] = seed
        report["cover_fixture"].append(r)
        t, p, e = r["truth"], r["repair_from_truth"], r["engine_from_scratch"]
        print(f"{seed:>4} | {t['K']:>2} {t['L']:>8.0f} {t['omega']:>6.3f} | "
              f"{p['K']:>2} {p['L']:>8.0f} {p['omega']:>6.3f} | "
              f"{e['K']:>2} {e['L']:>8.0f} {e['omega']:>6.3f}", flush=True)

    rows = report["cover_fixture"]
    mean = lambda f: round(sum(f(r) for r in rows) / len(rows), 1)
    report["headline"] = {
        "objective_gap_bits_mean": mean(lambda r: r["objective_gap_bits"]),
        "search_gap_bits_mean": mean(lambda r: r["search_gap_bits"]),
        "omega_truth": mean(lambda r: r["truth"]["omega"]),
        "omega_reachable_by_the_shipped_moves": mean(lambda r: r["repair_from_truth"]["omega"]),
        "omega_we_report": mean(lambda r: r["engine_from_scratch"]["omega"]),
        "reading": ("the objective's own preference costs part of the agreement and the search "
                    "costs the rest; compare the two gaps in bits before deciding what to fix"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
