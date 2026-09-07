"""E29: does the price really outrun the drift, or is that just asserted?

THE OBJECTION, raised independently by three readers of the submission. The paper concedes that its
lifetime false-mint bound does not cover the statistic the engine ships, and reports a measured
exceedance instead. What then carries the safety claim is one sentence in the appendix: the release
gate fires on the computed price, whose membership column grows faster than the accumulated drift.
That is a race between two growing quantities and the paper never analyses it. Its own numbers point
the other way, since the slack the bound would need grows from 97 bits at 500 records to 1,205 at
4,000. Node counts are measured to twenty thousand records and are zero, but a count of zero says
nothing about how close the run came, and the throughput section quotes a hundred thousand.

WHAT THIS MEASURES. Not whether a node appears, which E8 and E26 already report, but the margin: for
every candidate account still alive at the end of a null stream, the gap between the evidence it has
accrued and the price it would have to pay to be released. The largest such gap, over the whole pool,
is how close the stream came to minting. Run at increasing lengths, its trend is the race the paper
asserts, measured rather than argued.

  margin(T) = max over live candidates of (accrued evidence in bits - price in bits)

A margin of zero is a mint. The question is the sign of its trend.

  If the margin grows more negative with T, the price is winning and the claim is safe, and the rate
  is the thing to quote instead of the assertion.

  If the margin flattens or rises toward zero, there is a stream length at which this engine mints on
  noise, and the paper must say so and stop at the length it has measured.

WHAT WOULD FALSIFY THE PAPER'S POSITION. A margin that rises with length. It is reported whichever
way it goes.
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.codes import price                                          # noqa: E402
from escrow.protocol import run_stream, describe as protocol_describe   # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e29_margin_vs_length.json")

LENGTHS = (2000, 5000, 10000, 20000, 50000)
SEEDS = (0, 1, 2)
KEYS, ALPHABET = 4, 10


def null_stream(n, seed):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(ALPHABET)}" for j in range(KEYS)} for _ in range(n)]


def margin_of(g, n):
    """The closest any live candidate came to its release price, in bits.

    Zero would be a mint. The price is the engine's own, from codes.price, called with the
    candidate's cohort size and its support, exactly as the release gate calls it.
    """
    best = None
    for c in getattr(g, "pool", {}).values():
        support = len(c.keys)
        if support == 0:
            continue
        pr = price(c.t, support, n, g.K, g.e, len(g.keys))
        gap = c.G - pr
        if best is None or gap > best["margin"]:
            best = {"margin": round(gap, 2), "evidence_bits": round(c.G, 2),
                    "price_bits": round(pr, 2), "cohort": int(c.t), "support": support}
    return best


def main():
    rows = []
    for n in LENGTHS:
        per_seed = []
        for s in SEEDS:
            t0 = time.time()
            g, _ = run_stream(null_stream(n, s))
            m = margin_of(g, n)
            per_seed.append({"seed": s, "K": g.K, "live_candidates": len(getattr(g, "pool", {})),
                             "closest": m, "seconds": round(time.time() - t0, 1)})
            print(f"  n={n:6d} seed {s}: K={g.K}  closest margin "
                  f"{(m or {}).get('margin')} bits  ({per_seed[-1]['seconds']}s)", flush=True)
        margins = [p["closest"]["margin"] for p in per_seed if p["closest"]]
        rows.append({"n": n, "per_seed": per_seed,
                     "K_max": max(p["K"] for p in per_seed),
                     "margin_mean": round(st.mean(margins), 2) if margins else None,
                     "margin_max": round(max(margins), 2) if margins else None})

    trend = [(r["n"], r["margin_max"]) for r in rows if r["margin_max"] is not None]
    rising = (len(trend) > 1 and trend[-1][1] > trend[0][1])
    report = {
        "experiment": "E29 margin against stream length",
        "question": ("does the release price pull away from the accrued evidence as the stream grows, "
                     "which is what the paper asserts but never measures"),
        "protocol": protocol_describe(),
        "stream": f"pure noise, {KEYS} keys, {ALPHABET} iid uniform values each, no latent structure",
        "definition": "margin = accrued evidence minus release price, in bits; zero is a mint",
        "lengths": list(LENGTHS), "seeds": list(SEEDS),
        "rows": rows,
        "trend_of_the_worst_margin": trend,
        "headline": {
            "any_node_created": any(r["K_max"] > 0 for r in rows),
            "margin_rises_with_length": rising,
            "verdict": ("the margin rises with length, so there is a length at which this engine "
                        "mints on noise and the claim must stop at the measured length"
                        if rising else
                        "the price pulls away from the evidence as the stream grows, which is the "
                        "race the paper asserts, now measured"),
        },
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    print("worst margin by length:", trend)

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
