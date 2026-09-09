"""E76: the split move, on the fixture where the split move was already tried and found wanting.

WHAT THIS SETTLES. E51 asked how much of the measured search gap a split move would recover, proposed
the split by a two-way k-means over the node's own cells, and answered: about a tenth of it, with
zero splits accepted on two of the three seeds. That is a real measurement and it is not withdrawn.
It is a measurement of that proposal.

The paper's limitations section says a version two owes a split move. This is that move, and it
differs from E51's in one respect that turns out to be the whole of it. E51 proposes a bipartition
and prices it. The operator here names a candidate the way the mint names a node, by a (key, value)
cell, then **grows the seed under each half's own Krichevsky-Trofimov predictive before pricing it**.
A cell names a fraction of one kind at one key, so a bipartition priced at the moment it is named
prices a sliver and the gate refuses. That is the mint-on-first-miss degeneracy in a second place,
and it takes the same answer: a candidate is not priced on the evidence it has when it is named.

Measured on the value fixture E75 at full signal, over three arrival orders: the ungrown split takes
ARI from 0.6099 to 0.7906 and the node count to 24 against a true 8, and the grown split takes ARI to
0.9995 and the node count to exactly 8.

WHAT IS COMPARED HERE, all three scored by the same `BatchObjective.total` on the same records:

  the engine without the moves    the state E51 started from
  E51's k-means split             its published number, recomputed rather than quoted
  the engine with the moves       what ships

WHAT WOULD REFUTE IT. The new operator recovering no more of the gap than E51's did, or recovering
bits while agreement falls, which would be a finding about the objective rather than the search.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

from escrow import batch as B
from escrow.batch import BatchObjective
from escrow.provenance import stamped
from e47_does_the_objective_prefer_the_truth import (background_only, cover_fixture, engine_state,
                                                     install_oracle)
from e48_search_gap import omega_index, pred_sets
from e51_a_split_move import split_to_fixpoint

OUT = os.path.join(ROOT, "results", "e76_the_split_that_works.json")
SEEDS = (0, 1, 2)


class moves_off:
    def __enter__(self):
        self.was = B.SPLIT_MOVES
        B.SPLIT_MOVES = False

    def __exit__(self, *a):
        B.SPLIT_MOVES = self.was
        return False


def main():
    report = {"experiment": "E76 the split move that recovers the gap, against the one that did not",
              "fixture": "the E47 cover fixture, shared=12, the same three seeds as E51",
              "scored_by": "BatchObjective.total, one instrument for all three arms",
              "rows": []}
    print(f"{'seed':>4} | {'no split moves':>17} | {'E51 k-means split':>18} | "
          f"{'the grown split':>18} | {'truth':>17}")
    print(f"{'':>4} | {'K':>2} {'L':>9} {'om':>4} | {'K':>2} {'L':>9} {'om':>4} | "
          f"{'K':>2} {'L':>9} {'om':>4} | {'K':>2} {'L':>9}")
    for seed in SEEDS:
        recs, truth, gen, _ = cover_fixture(shared=12, seed=seed)
        n = len(recs)

        with moves_off():
            g0, b0 = engine_state(recs)
        L0, om0 = b0.total(), omega_index(truth, pred_sets(g0, n), n)

        with moves_off():
            g1, b1, L1, acc = split_to_fixpoint(recs, g0, b0, n)
        om1 = omega_index(truth, pred_sets(g1, n), n)

        t0 = time.time()
        g2, b2 = engine_state(recs)
        secs = time.time() - t0
        L2, om2 = b2.total(), omega_index(truth, pred_sets(g2, n), n)

        gT, bT = background_only(recs)
        install_oracle(gT, truth, gen)
        LT = bT.total()

        gap = L0 - LT
        row = {"seed": seed, "records": n,
               "no_split_moves": {"K": int(g0.K), "L": round(L0, 1), "omega": round(om0, 4)},
               "e51_kmeans_split": {"K": int(g1.K), "L": round(L1, 1), "omega": round(om1, 4),
                                    "splits_accepted": acc,
                                    "share_of_the_gap": round((L0 - L1) / gap, 4) if gap > 0 else 0.0},
               "the_grown_split": {"K": int(g2.K), "L": round(L2, 1), "omega": round(om2, 4),
                                   "share_of_the_gap": round((L0 - L2) / gap, 4) if gap > 0 else 0.0,
                                   "seconds": round(secs, 1)},
               "truth": {"K": int(gT.K), "L": round(LT, 1)},
               "gap_to_the_truth_before": round(gap, 1)}
        report["rows"].append(row)
        print(f"{seed:>4} | {g0.K:>2} {L0:>9.0f} {om0:>4.2f} | {g1.K:>2} {L1:>9.0f} {om1:>4.2f} | "
              f"{g2.K:>2} {L2:>9.0f} {om2:>4.2f} | {gT.K:>2} {LT:>9.0f}")

    def mean(path):
        vals = [r[path[0]][path[1]] for r in report["rows"]]
        return round(sum(vals) / len(vals), 4)

    report["headline"] = {
        "omega_without_the_moves": mean(("no_split_moves", "omega")),
        "omega_with_e51_kmeans_split": mean(("e51_kmeans_split", "omega")),
        "omega_with_the_grown_split": mean(("the_grown_split", "omega")),
        "share_of_the_gap_e51": mean(("e51_kmeans_split", "share_of_the_gap")),
        "share_of_the_gap_grown": mean(("the_grown_split", "share_of_the_gap")),
        "reading": ("E51's number stands as a measurement of its own proposal. What changed is not "
                    "that a split move became admissible, it always was, but that the candidate is "
                    "grown before it is priced, so the gate sees the move's value rather than its "
                    "first sliver"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
