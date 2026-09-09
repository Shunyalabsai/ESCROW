"""E75: a constructed stream where the values carry the groups, and the whole curve across it.

WHAT THIS REPLACES, AND WHY IT HAD TO. The paper's headline constructed result is ARI 1.0 at K = 8
on `planted8`, with standard deviation 0 across twenty arrival orders. That fixture builds a record
of group g as

    {c{g}k0: ..., c{g}k1: ..., c{g}k2: ...}

so the **key name encodes the group**. Grouping records by their key signature alone, discarding
every value, scores **ARI 1.0000** on it, and does so at every setting of its own `noise` parameter
including 1.0, because that parameter corrupts values and never touches keys. Eight groups, eight key
signatures, a dictionary lookup.

So the fixture is separable by key presence, the value code is never exercised, and the sweep it
offers tests nothing. That is a criticism of an instrument we built, and it should be said before
anyone else says it.

THE REPLACEMENT. Every group draws from the **same shared key pool**, so a key signature carries no
information at all and the groups can only be told apart by what their values say. A group is a
characteristic distribution over the shared values, and `signal` is how concentrated that
distribution is:

  signal = 0.0   every group draws values uniformly from the whole vocabulary. There is nothing to
                 find, and the truth is zero nodes, even though every record has the same keys as
                 every other and looks structured.
  signal = 1.0   each group draws only from its own slice of the vocabulary. Fully separable.
  in between     each value comes from the group's own slice with probability `signal`, and from
                 the shared pool otherwise.

This is the harder null. The old one gives noise records four keys that no real group uses; this one
gives the noise the **same keys as the signal**, so nothing but the value evidence can tell them
apart, which is exactly the decision the price is supposed to make.

WHAT WOULD REFUTE THE ABSTENTION CLAIM. Any node at signal 0, where every record carries the same
keys and the values are uniform. That is a far more demanding null than a stream of unrelated keys,
and it is the one worth passing.
"""
from __future__ import annotations

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
from e4_baseline_army import _ari

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e75_a_fixture_that_tests_the_values.json")

N = int(os.environ.get("E75_N", "3000"))
GROUPS, KEYS, PER_GROUP = 8, 6, 8          # 8 groups, 6 shared keys, 8 values each in a group slice
SEEDS = (0, 1, 2)
SIGNAL = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)


def stream(signal, seed, n=N):
    """Every record carries the same six keys. Only the values differ by group."""
    rng = random.Random(5000 + seed)
    vocab = [f"v{i}" for i in range(GROUPS * PER_GROUP)]
    slice_of = {g: vocab[g * PER_GROUP:(g + 1) * PER_GROUP] for g in range(GROUPS)}
    recs, truth = [], []
    for _ in range(n):
        g = rng.randrange(GROUPS)
        rec = {}
        for j in range(KEYS):
            pool = slice_of[g] if rng.random() < signal else vocab
            rec[f"k{j}"] = rng.choice(pool)
        recs.append(rec)
        truth.append(g)
    return recs, truth


def keysig_baseline(recs, truth):
    """What the old fixture's shortcut scores here: group by key signature, ignore every value."""
    sigs, lab = {}, []
    for r in recs:
        s = tuple(sorted(r.keys()))
        sigs.setdefault(s, len(sigs))
        lab.append(sigs[s])
    return round(_ari(truth, lab), 4), len(sigs)


def main():
    report = {"experiment": "E75 a constructed stream whose groups live in the values",
              "why": ("planted8 encodes the group in the key name, so grouping on key signature "
                      "alone scores ARI 1.0000 on it at every noise setting and the value code is "
                      "never tested"),
              "records": N, "groups": GROUPS, "shared_keys": KEYS, "seeds": list(SEEDS),
              "sweep": []}

    ks_ari, ks_n = keysig_baseline(*stream(1.0, 0))
    report["key_signature_shortcut_on_this_fixture"] = {"ARI": ks_ari, "distinct_signatures": ks_n}
    print(f"the key-signature shortcut that solves planted8 scores ARI {ks_ari} here, "
          f"with {ks_n} signature(s)\n")

    print(f"  {'signal':>7} {'noise':>7} {'K':>7} {'ARI':>9} {'covered':>16}  K per seed")
    for f in SIGNAL:
        per = []
        for s in SEEDS:
            recs, truth = stream(f, s)
            g, _ = run_stream(recs)
            lab = record_labels(g, len(recs))
            per.append({"K": int(g.K), "ARI": round(_ari(truth, lab), 4),
                        "covered": sum(1 for x in lab if x != -1)})
        km = sum(p["K"] for p in per) / len(per)
        am = sum(p["ARI"] for p in per) / len(per)
        cov = sum(p["covered"] for p in per)
        row = {"signal": f, "noise": round(1 - f, 2), "K_mean": round(km, 2),
               "K_per_seed": [p["K"] for p in per], "ARI_mean": round(am, 4),
               "records_in_a_node": f"{cov}/{N * len(SEEDS)}"}
        report["sweep"].append(row)
        print(f"  {f:>7.2f} {1-f:>7.2f} {km:>7.2f} {am:>9.4f} {row['records_in_a_node']:>16}  "
              f"{row['K_per_seed']}")

    zero = report["sweep"][0]
    first = next((r["signal"] for r in report["sweep"] if r["K_mean"] > 0), None)
    exact = next((r["signal"] for r in report["sweep"] if abs(r["K_mean"] - GROUPS) < 0.5), None)
    report["headline"] = {
        "K_at_zero_signal": zero["K_mean"],
        "the_null_holds": zero["K_mean"] == 0,
        "signal_at_which_the_first_node_appears": first,
        "signal_at_which_K_reaches_the_true_8": exact,
        "key_signature_shortcut_ARI": ks_ari,
        "reading": ("this null is harder than the shipped one: every record carries the same keys, "
                    "so nothing but the value evidence separates a group from noise. A node at zero "
                    "signal would mean the abstention claim rests on the old fixture's key names"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
