"""E55: the scope condition for the key merge, found before any real-data number is quoted.

WHY THIS RUNS BEFORE THE REAL DATA. E54 got every probe right, but the margins were lopsided: the
planted synonym merged at -326 bits while the controls were kept by only +0.7 to +6.3. That gap is
structural rather than incidental. Pooling two value blocks costs roughly

    (n_a + n_b) * JS(p_a, p_b)

which scales with how often the two keys are used, while the model-side saving from dropping one
entry from the key inventory does not. So as the keys get rarer the cost of a wrong merge falls
towards nothing while the saving stays, and below some frequency every pair merges regardless of
what its values say. That is the same shape as Proposition 1: a decision is affordable only where
there is enough evidence to pay for it, and the price says so in advance.

WHAT THIS MEASURES. Two sweeps on a fixture where the truth is known by construction.

  (i) frequency. Two keys with disjoint vocabularies, which must never merge, swept from very rare
      to common. Find the count at which the margin changes sign. Below it the operator is not
      deciding, it is guessing cheaply.

  (ii) overlap. Two keys at a fixed, comfortable frequency whose vocabularies overlap by a
      controlled fraction, from disjoint to identical. The margin should fall smoothly and cross
      zero somewhere, and where it crosses is the operator's actual definition of "the same key".

WHAT WOULD REFUTE THE DESIGN. If (i) shows no crossover, the worry was wrong and rare keys are safe.
If it shows one at a frequency where real keys live, then the operator cannot be shipped as it
stands and A4 on the worklist is load-bearing rather than optional. Either way the number goes in
the paper before any Lazada result does.
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

from escrow.batch import BatchObjective
from escrow.keymerge import key_merge_delta
from escrow.protocol import run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e55_when_can_a_key_merge_be_decided.json")

BULK = 400                      # records of ordinary structure the two probe keys sit inside
FREQS = (4, 8, 16, 32, 64, 128, 256)
OVERLAPS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)

VOCAB_A = [f"a{i}" for i in range(8)]
VOCAB_B = [f"b{i}" for i in range(8)]


def bulk_records(rng, n):
    """Ordinary structure so the graph is not degenerate while the probe keys are swept."""
    out = []
    for _ in range(n):
        out.append({"kind": rng.choice(["shirt", "dress", "coat"]),
                    "fabric": rng.choice(["cotton", "silk", "wool"]),
                    "fit": rng.choice(["slim", "regular", "loose"])})
    return out


def probe_stream(rng, freq, overlap, n_bulk=BULK):
    """Two probe keys, `probe_a` and `probe_b`, each on `freq` records, never together.

    `overlap` is the fraction of b's vocabulary shared with a. At 0.0 the two keys say entirely
    different things and must never merge; at 1.0 they say exactly the same things and are one key
    under two names.
    """
    shared = int(round(overlap * len(VOCAB_A)))
    vocab_b = VOCAB_A[:shared] + VOCAB_B[shared:]
    recs = bulk_records(rng, n_bulk)
    for _ in range(freq):
        recs.append({"kind": rng.choice(["shirt", "dress", "coat"]),
                     "probe_a": rng.choice(VOCAB_A)})
    for _ in range(freq):
        recs.append({"kind": rng.choice(["shirt", "dress", "coat"]),
                     "probe_b": rng.choice(vocab_b)})
    rng.shuffle(recs)
    return recs


def margin(recs):
    g, b = run_stream(recs)
    ka = g.keys.get("probe_a")
    kb = g.keys.get("probe_b")
    if ka is None or kb is None:
        return None, g
    return key_merge_delta(g, BatchObjective, ka.kid, kb.kid), g


def main():
    report = {"experiment": "E55 the scope condition for the key merge",
              "fixture": {"bulk_records": BULK, "seeds": list(SEEDS),
                          "probe_vocabulary_size": len(VOCAB_A)},
              "frequency_sweep": [], "overlap_sweep": []}

    print("(i) frequency, with DISJOINT vocabularies. These must never merge, so every")
    print("    negative margin is a false merge the operator would take.\n")
    print(f"    {'records per key':>16} {'mean margin, bits':>20} {'merges':>8}")
    for f in FREQS:
        ds, merges = [], 0
        for s in SEEDS:
            d, g = margin(probe_stream(random.Random(1000 + s), f, overlap=0.0))
            if d is None:
                continue
            ds.append(d)
            merges += (d < 0)
        if not ds:
            continue
        mean = sum(ds) / len(ds)
        report["frequency_sweep"].append({"records_per_key": f, "overlap": 0.0,
                                          "mean_margin_bits": round(mean, 2),
                                          "margins": [round(x, 2) for x in ds],
                                          "false_merges": merges, "of": len(ds)})
        flag = "  <-- FALSE MERGE" if merges else ""
        print(f"    {f:>16} {mean:>20.1f} {merges:>4}/{len(ds)}{flag}")

    print("\n(ii) vocabulary overlap at a comfortable frequency. The margin should fall as the")
    print("     two keys start saying the same things, and where it crosses zero is the")
    print("     operator's working definition of one key.\n")
    print(f"    {'overlap':>10} {'mean margin, bits':>20} {'merges':>8}")
    FREQ_FIXED = 128
    for ov in OVERLAPS:
        ds, merges = [], 0
        for s in SEEDS:
            d, g = margin(probe_stream(random.Random(2000 + s), FREQ_FIXED, overlap=ov))
            if d is None:
                continue
            ds.append(d)
            merges += (d < 0)
        if not ds:
            continue
        mean = sum(ds) / len(ds)
        report["overlap_sweep"].append({"overlap": ov, "records_per_key": FREQ_FIXED,
                                       "mean_margin_bits": round(mean, 2),
                                       "margins": [round(x, 2) for x in ds],
                                       "merges": merges, "of": len(ds)})
        print(f"    {ov:>10.2f} {mean:>20.1f} {merges:>4}/{len(ds)}")

    fs = report["frequency_sweep"]
    bad = [r for r in fs if r["false_merges"]]
    safe = [r for r in fs if not r["false_merges"]]
    report["headline"] = {
        "false_merges_on_disjoint_vocabularies_at": [r["records_per_key"] for r in bad],
        "safe_at": [r["records_per_key"] for r in safe],
        "crossover_between": ([max(r["records_per_key"] for r in bad),
                               min(r["records_per_key"] for r in safe)]
                              if bad and safe else None),
        "overlap_at_which_it_merges": [r["overlap"] for r in report["overlap_sweep"]
                                       if r["merges"] == r["of"]],
        "reading": ("the cost of a wrong merge is about (n_a + n_b) times the divergence between "
                    "the two value distributions, so it vanishes with the counts while the saving "
                    "from dropping a key from the inventory does not. Below the crossover the "
                    "operator is not deciding, and the price says so before any data is seen"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
