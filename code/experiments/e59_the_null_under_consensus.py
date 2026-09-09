"""E59: does running several arrival orders and taking a consensus break the null?

WHY THIS RUNS BEFORE THE REST OF THE TRACK. Consensus is about to be applied to every decision the
system makes, and it is not obviously safe. Running R arrival orders gives R chances for the greedy
search to land somewhere that proposes a node, so a rule that accepts on a minority would find
structure in a stream that has none. The paper's central claim is that nothing is born on noise, and
it has to survive the protocol change or the protocol change does not happen.

THE THREE RULES, AND WHY THE SPREAD MATTERS MORE THAN THE VERDICT.

  any        a node is created if any of the R orders creates it. The dangerous one, and the reason
             this experiment exists: it is what a careless implementation of "run it a few times"
             does, and its false-mint rate should grow with R.
  majority   created if more than half the orders create it. The rule E58 recommends.
  all        created only if every order creates it. The safe one, and the one that would refuse
             real structure too often if the spread is wide.

Reporting all three on the same streams shows what consensus costs and buys, rather than asserting
that the recommended rule is safe.

WHAT WOULD REFUTE THE TRACK. Any node on any null stream under `majority`. If that happens the
consensus protocol is not shippable as specified, whatever it does for the merge decision.

THE POSITIVE CONTROL IS NOT OPTIONAL. A rule that returns zero on everything passes the null
trivially, so the same three rules are scored on a planted stream where the answer is eight groups.
A protocol that is silent on noise and also silent on structure has not been shown to be safe, only
to be deaf.
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
OUT = os.path.join(ROOT, "results", "e59_the_null_under_consensus.json")

R_VALUES = (3, 5, 9)
NULL_STREAMS = 12
NULL_LENGTHS = (2000, 5000)
KEYS, D = 4, 10


def null_stream(n, seed):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(D)}" for j in range(KEYS)} for _ in range(n)]


def consensus_pairs(recs, R, shuffle_seed):
    """How often each pair of records is placed in the same node, across R arrival orders.

    Pairs rather than node counts, because node identities are not comparable between runs: node 4
    in one order is not node 4 in another. A pair being co-placed is comparable, and it is also
    exactly what the agreement metrics score.
    """
    n = len(recs)
    together = collections.Counter()
    ks = []
    for r in range(R):
        order = list(range(n))
        random.Random(shuffle_seed * 1000 + r).shuffle(order)
        shuffled = [recs[i] for i in order]
        g, _ = run_stream(shuffled)
        lab_shuffled = record_labels(g, n)
        lab = [None] * n
        for pos, orig in enumerate(order):
            lab[orig] = lab_shuffled[pos]
        ks.append(int(g.K))
        by = collections.defaultdict(list)
        for i, l in enumerate(lab):
            if l is not None and l != -1:
                by[l].append(i)
        for members in by.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    together[(members[a], members[b])] += 1
    return together, ks


def groups_under(together, R, rule):
    """Number of co-placement pairs surviving each rule, and whether any node exists at all."""
    if rule == "any":
        keep = [p for p, c in together.items() if c >= 1]
    elif rule == "majority":
        keep = [p for p, c in together.items() if c * 2 > R]
    else:
        keep = [p for p, c in together.items() if c == R]
    return len(keep)


def main():
    report = {"experiment": "E59 the null under a consensus protocol",
              "null": {"streams": NULL_STREAMS, "lengths": list(NULL_LENGTHS),
                       "keys": KEYS, "values_per_key": D},
              "R_values": list(R_VALUES), "null_results": [], "positive_control": []}

    print("NULL. Truth is no node at all, so any surviving pair is a false grouping.\n")
    print(f"  {'R':>3} {'length':>7} {'streams':>8} {'any':>10} {'majority':>10} {'all':>8} "
          f"{'single-order K':>16}")
    for R in R_VALUES:
        for length in NULL_LENGTHS:
            tot = {"any": 0, "majority": 0, "all": 0}
            all_ks = []
            for s in range(NULL_STREAMS):
                recs = null_stream(length, 4000 + s)
                together, ks = consensus_pairs(recs, R, shuffle_seed=s + 1)
                all_ks += ks
                for rule in tot:
                    tot[rule] += groups_under(together, R, rule)
            row = {"R": R, "length": length, "streams": NULL_STREAMS,
                   "false_pairs": tot, "single_order_K": sorted(set(all_ks))}
            report["null_results"].append(row)
            print(f"  {R:>3} {length:>7} {NULL_STREAMS:>8} {tot['any']:>10} "
                  f"{tot['majority']:>10} {tot['all']:>8} {str(sorted(set(all_ks))):>16}")

    print("\nPOSITIVE CONTROL. A protocol silent on noise and on structure is deaf, not safe.\n")
    try:
        from e4_baseline_army import planted8
        for R in R_VALUES:
            recs, truth = planted8(n=1600, seed=0)
            together, ks = consensus_pairs(recs, R, shuffle_seed=77)
            same = sum(1 for i in range(len(truth)) for j in range(i + 1, len(truth))
                       if truth[i] == truth[j])
            kept = {rule: groups_under(together, R, rule) for rule in ("any", "majority", "all")}
            report["positive_control"].append({"R": R, "true_same_pairs": same, "kept": kept,
                                               "K_per_order": ks})
            print(f"  R={R}  truth has {same} same-group pairs.  kept: "
                  f"any {kept['any']}, majority {kept['majority']}, all {kept['all']}   "
                  f"K per order {ks}")
    except Exception as e:                                        # noqa: BLE001
        report["positive_control_error"] = repr(e)[:200]
        print(f"  positive control unavailable: {e!r}"[:120])

    worst = max((r["false_pairs"]["majority"] for r in report["null_results"]), default=0)
    worst_any = max((r["false_pairs"]["any"] for r in report["null_results"]), default=0)
    report["headline"] = {
        "false_pairs_on_noise_under_majority": worst,
        "false_pairs_on_noise_under_any": worst_any,
        "null_survives_majority": worst == 0,
        "reading": ("consensus gives the search R chances to propose a node, so the any rule is the "
                    "one at risk and is reported for that reason. The majority rule is the one the "
                    "track proposes to ship, and it is only shippable if this row is zero"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
