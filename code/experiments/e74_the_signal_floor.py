"""E74: not a null and a positive control, but the whole curve between them.

WHY THIS REPLACES THE BINARY TEST. Every claim about abstention so far rests on two points: a stream
with no structure, where the answer is zero nodes, and a stream with planted structure, where the
answer is eight. Two points do not describe a rule. What a reader wants to know is where the
transition is, how sharp it is, and whether the method crosses it in the right place. A rule that
returns zero on nothing and eight on everything could still be wrong about every case in between,
and every real stream is in between.

These streams are **constructed with known truth**, not synthetic in the dismissive sense. Every
value is placed by a rule we wrote down, which is what makes the truth checkable at all; the point
of the sweep is that the amount of structure is a dial we turn rather than a condition we assert.

TWO WAYS TO DILUTE, BECAUSE THEY ASK DIFFERENT QUESTIONS.

  value corruption   every record comes from the planted generator, and each value is replaced by a
                     random one with probability 1-f. The groups are all still there and each is
                     harder to see. This asks how much evidence a group needs.

  record mixture     a fraction f of records come from the planted generator and 1-f are pure noise
                     with no group at all. The groups are as clear as ever and there are fewer
                     records carrying them. This asks whether unrelated traffic drowns real structure,
                     which is the shape of most real streams.

WHAT TO READ OFF. Where K departs from zero, whether it rises to the true eight or overshoots, and
whether agreement tracks the signal or collapses. The interesting number is not any single point but
whether the method is quiet where it should be and confident where it should be, with a transition
in between rather than a cliff at either end.

WHAT WOULD REFUTE THE ABSTENTION CLAIM. Nodes appearing at a signal level where there is nothing to
find, or the transition sitting far from where the seeding condition predicts it.
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
from e4_baseline_army import _ari, planted8

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e74_the_signal_floor.json")

N = int(os.environ.get("E74_N", "3000"))
SEEDS = (0, 1, 2)
SIGNAL = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85, 1.0)
KEYS, VALUES = 4, 10


def noise_record(rng):
    return {f"k{j}": f"v{rng.randrange(VALUES)}" for j in range(KEYS)}


def by_corruption(signal, seed, n=N):
    """Every record from the generator; each value random with probability 1 - signal."""
    return planted8(n=n, noise=1.0 - signal, seed=seed)


def by_mixture(signal, seed, n=N):
    """`signal` of the records carry the planted groups; the rest are pure noise with no group."""
    rng = random.Random(10_000 + seed)
    planted, truth = planted8(n=n, noise=0.0, seed=seed)
    recs, tr = [], []
    for i in range(n):
        if rng.random() < signal:
            recs.append(planted[i])
            tr.append(truth[i])
        else:
            recs.append(noise_record(rng))
            tr.append(None)                      # belongs to no group
    return recs, tr


def score(recs, truth):
    g, _ = run_stream(recs)
    lab = record_labels(g, len(recs))
    idx = [i for i, t in enumerate(truth) if t is not None]
    ari = _ari([truth[i] for i in idx], [lab[i] for i in idx]) if len(idx) > 2 else None
    covered_signal = sum(1 for i in idx if lab[i] != -1)
    noise_idx = [i for i, t in enumerate(truth) if t is None]
    covered_noise = sum(1 for i in noise_idx if lab[i] != -1)
    return {"K": int(g.K),
            "ARI_on_the_signal_records": None if ari is None else round(ari, 4),
            "signal_records": len(idx), "signal_records_in_a_node": covered_signal,
            "noise_records": len(noise_idx), "noise_records_in_a_node": covered_noise}


def main():
    report = {"experiment": "E74 the whole curve from no signal to all signal",
              "note": ("streams constructed with known truth; the amount of structure is a dial, "
                       "not an assertion"),
              "records": N, "seeds": list(SEEDS), "true_groups": 8, "sweeps": {}}

    for name, build in (("value corruption", by_corruption), ("record mixture", by_mixture)):
        rows = []
        print(f"\n{name.upper()}: {N} records, truth is 8 groups\n")
        print(f"  {'signal':>7} {'noise':>7} {'K':>6} {'ARI on signal':>15} "
              f"{'signal in a node':>17} {'noise in a node':>16}")
        for f in SIGNAL:
            per = [score(*build(f, s)) for s in SEEDS]
            ks = [p["K"] for p in per]
            aris = [p["ARI_on_the_signal_records"] for p in per
                    if p["ARI_on_the_signal_records"] is not None]
            cs = sum(p["signal_records_in_a_node"] for p in per)
            ss = sum(p["signal_records"] for p in per)
            cn = sum(p["noise_records_in_a_node"] for p in per)
            nn = sum(p["noise_records"] for p in per)
            row = {"signal": f, "noise": round(1 - f, 2),
                   "K_mean": round(sum(ks) / len(ks), 2), "K_per_seed": ks,
                   "ARI_mean": round(sum(aris) / len(aris), 4) if aris else None,
                   "signal_records_covered": f"{cs}/{ss}",
                   "noise_records_covered": f"{cn}/{nn}" if nn else "none present"}
            rows.append(row)
            print(f"  {f:>7.2f} {1-f:>7.2f} {row['K_mean']:>6.2f} "
                  f"{(row['ARI_mean'] if row['ARI_mean'] is not None else float('nan')):>15.4f} "
                  f"{row['signal_records_covered']:>17} {row['noise_records_covered']:>16}")
        report["sweeps"][name] = rows

    def first_node(rows):
        for r in rows:
            if r["K_mean"] > 0:
                return r["signal"]
        return None

    def first_exact(rows):
        for r in rows:
            if abs(r["K_mean"] - 8) < 0.5:
                return r["signal"]
        return None

    report["headline"] = {
        name: {"signal_at_which_the_first_node_appears": first_node(rows),
               "signal_at_which_K_reaches_the_true_8": first_exact(rows),
               "K_at_zero_signal": rows[0]["K_mean"]}
        for name, rows in report["sweeps"].items()}
    report["headline"]["reading"] = (
        "the number that matters is not either endpoint but where the transition sits and how sharp "
        "it is. K at zero signal must be zero or the abstention claim is gone; K reaching eight "
        "somewhere below full signal is what says the rule is not merely cautious")
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
