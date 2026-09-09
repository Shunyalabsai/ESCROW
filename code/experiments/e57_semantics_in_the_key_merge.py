"""E57: the one place semantics should earn its keep, and a test that can say it does not.

WHERE E56 LEFT THIS. Given the same records, a semantic naming charge cost 11 bits more than the
categorical one on noise, which is correct and required, and 1,292 bits more on the encyclopedia
stream, which is a negative result. The mixture paid its one bit and fell back, so nothing was
harmed, but nothing was gained either. The reason is that the encyclopedia's blocks hold values that
are semantically alike without being predictable: knowing that every value of `director` is a
person's name does not help guess which person comes next.

SO WHERE SHOULD AN EMBEDDING PAY? Not in predicting the next value, but in recognising that two
vocabularies are the same vocabulary. That is the key merge of E54, and it is exactly the case the
categorical operator cannot reach: E55 measured its working definition of one key as more than about
half the value strings shared, which is a statement about characters. Two sellers using `shade` with
`red, blue, green` and `colour` with `crimson, azure, emerald` are describing one facet with zero
string overlap, and no amount of counting exact matches will ever see it.

THE FIXTURE. Three probes, all with zero string overlap between the two keys:

  synonymous   shade over a palette, colour over a different wording of the same palette.
               Must merge once values are compared by meaning, must not before.
  disjoint     shade over a palette, weight over masses. Must never merge under either code.
  lookalike    material over fabrics, and finish over surface treatments. Related words that are
               not the same facet, which is where a similarity with a cut-off would go wrong and
               where a code has to pay for its confidence.

WHAT WOULD REFUTE IT. If the semantic arm merges the disjoint or lookalike probes, the charge is
buying agreement rather than predicting it, and the embedding is doing what a threshold would do.
If it fails to merge the synonymous probe, semantics does not help here either, and the honest
conclusion from E56 and E57 together is that this embedding earns nothing anywhere and should be
left out.
"""
from __future__ import annotations

import collections
import json
import math
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
from escrow.semantic import Embedding, block_value_bits, mixture_bits

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e57_semantics_in_the_key_merge.json")
EMB = os.path.join(ROOT, "results", "embeddings", "keymerge_fixture.json")
VOCAB_OUT = os.path.join(ROOT, "results", "embeddings", "keymerge_fixture.values.txt")

PALETTE_A = ["red", "blue", "green", "black", "white"]
PALETTE_B = ["crimson", "azure", "emerald", "ebony", "ivory"]
MASSES = ["250 grams", "500 grams", "1 kilogram", "2 kilograms", "5 kilograms"]
FABRICS = ["cotton", "silk", "wool", "linen", "denim"]
FINISHES = ["matte", "glossy", "brushed", "polished", "satin"]
BULK = ["shirt", "dress", "coat", "jacket", "scarf"]

PROBES = [
    ("shade", "colour", True, "one palette, two wordings, no string overlap"),
    ("shade", "weight", False, "a palette against masses, unrelated"),
    ("material", "finish", False, "related words, different facets, where a cut-off goes wrong"),
]
SEEDS = (0, 1, 2)
PER_KEY = 150


def build_stream(seed):
    rng = random.Random(seed)
    recs = []
    # each record carries enough facets for the engine to have something to mint on. The first
    # version of this fixture gave every record two keys and returned K = 0, which meant the merge
    # was being priced against a graph with no nodes in it.
    def base():
        return {"item": rng.choice(BULK), "fit": rng.choice(FITS),
                "season": rng.choice(SEASONS)}
    # the two colour keys sit on records that are otherwise the same kind of thing, so a node forms
    # over both and the merge is priced against a real graph rather than against the background
    for _ in range(PER_KEY):
        recs.append({**base(), "sleeve": rng.choice(["short", "long"]),
                     "shade": rng.choice(PALETTE_A)})
    for _ in range(PER_KEY):
        recs.append({**base(), "sleeve": rng.choice(["short", "long"]),
                     "colour": rng.choice(PALETTE_B)})
    for _ in range(PER_KEY):
        recs.append({**base(), "pack": rng.choice(["box", "bag"]),
                     "weight": rng.choice(MASSES)})
    for _ in range(PER_KEY):
        recs.append({**base(), "weave": rng.choice(["plain", "twill"]),
                     "material": rng.choice(FABRICS)})
    for _ in range(PER_KEY):
        recs.append({**base(), "weave": rng.choice(["plain", "twill"]),
                     "finish": rng.choice(FINISHES)})
    rng.shuffle(recs)
    return recs


FITS = ["slim", "regular", "loose"]
SEASONS = ["summer", "winter", "all year"]


EXTRA = ["short", "long", "box", "bag", "plain", "twill"]


def all_values():
    return sorted(set(PALETTE_A + PALETTE_B + MASSES + FABRICS + FINISHES + BULK
                      + FITS + SEASONS + EXTRA))


def value_delta(emb, recs, key_a, key_b, semantic):
    """Change in the value-prediction term when the two keys become one, under one of the codes.

    This replaced an earlier version that moved the naming charge instead. Measured on this fixture,
    pooling two disjoint vocabularies costs +298 bits, of which the predictive carries +293 and the
    naming charge +8.7, so semantics acting on naming alone could never move the decision and E57's
    first run duly separated its probes by about one bit against a 243-bit barrier.
    """
    seq_a = [str(r[key_a]) for r in recs if key_a in r]
    seq_b = [str(r[key_b]) for r in recs if key_b in r]
    inv_a, inv_b = [], []
    for v in seq_a:
        if v not in inv_a:
            inv_a.append(v)
    for v in seq_b:
        if v not in inv_b:
            inv_b.append(v)
    inv_c = inv_a + [v for v in inv_b if v not in inv_a]
    before = (block_value_bits(emb, seq_a, inv_a, semantic)
              + block_value_bits(emb, seq_b, inv_b, semantic))
    merged_seq = []
    for r in recs:
        if key_a in r:
            merged_seq.append(str(r[key_a]))
        elif key_b in r:
            merged_seq.append(str(r[key_b]))
    after = block_value_bits(emb, merged_seq, inv_c, semantic)
    return after - before


def main():
    if not os.path.exists(EMB):
        os.makedirs(os.path.dirname(VOCAB_OUT), exist_ok=True)
        with open(VOCAB_OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(all_values()) + "\n")
        print(f"no embedding cache at {EMB}")
        print(f"wrote the fixture vocabulary to {VOCAB_OUT}. Build the cache with:")
        print(f"  python3 code/tools/embed_values.py --values {VOCAB_OUT} --out {EMB}")
        sys.exit(1)

    emb = Embedding.from_file(EMB)
    report = {"experiment": "E57 does an embedding let the key merge reach synonymous vocabularies",
              "embedding": json.load(open(EMB)).get("model", "?"),
              "fixture": {"records_per_key": PER_KEY, "seeds": list(SEEDS),
                          "string_overlap_between_every_probe_pair": 0.0},
              "per_seed": []}

    for seed in SEEDS:
        recs = build_stream(seed)
        g, b = run_stream(recs)
        row = {"seed": seed, "records": len(recs), "K": int(g.K), "probes": []}
        print(f"\nseed {seed}: {len(recs)} records, K = {g.K}")
        for a, bb, should, why in PROBES:
            ka, kb = g.keys.get(a), g.keys.get(bb)
            if ka is None or kb is None:
                continue
            cat = key_merge_delta(g, BatchObjective, ka.kid, kb.kid)
            if cat is None:
                continue
            d_cat = value_delta(emb, recs, a, bb, semantic=False)
            d_sem = value_delta(emb, recs, a, bb, semantic=True)
            sem = cat - d_cat + d_sem
            mix = mixture_bits(cat, sem)
            row["probes"].append({
                "a": a, "b": bb, "should_merge": should, "why": why,
                "categorical_bits": round(cat, 1), "semantic_bits": round(sem, 1),
                "mixture_bits": round(mix, 1),
                "merged_categorical": cat < 0, "merged_semantic": sem < 0,
                "merged_mixture": mix < 0,
                "correct_under_mixture": (mix < 0) == should})
            print(f"   {a:>9} + {bb:<9} want {'merge' if should else 'keep ':<5}   "
                  f"categorical {cat:>+9.1f}   semantic {sem:>+9.1f}   mixture {mix:>+9.1f}   "
                  f"{'ok' if ((mix < 0) == should) else 'WRONG'}")
        report["per_seed"].append(row)

    def agg(field):
        return {f"{p['a']}+{p['b']}": [r["probes"][i][field]
                                       for r in report["per_seed"]
                                       for i, q in enumerate(r["probes"]) if q["a"] == p["a"]
                                       and q["b"] == p["b"]][0]
                for r0 in report["per_seed"][:1] for i, p in enumerate(r0["probes"])}

    probes = [(p["a"], p["b"]) for p in report["per_seed"][0]["probes"]]
    report["headline"] = {
        "categorical_merges": [f"{a}+{b}" for a, b in probes
                               if all(any(q["a"] == a and q["b"] == b and q["merged_categorical"]
                                          for q in r["probes"]) for r in report["per_seed"])],
        "semantic_merges": [f"{a}+{b}" for a, b in probes
                            if all(any(q["a"] == a and q["b"] == b and q["merged_semantic"]
                                       for q in r["probes"]) for r in report["per_seed"])],
        "all_probes_correct_under_mixture_on_every_seed": all(
            q["correct_under_mixture"] for r in report["per_seed"] for q in r["probes"]),
        "reading": ("the two keys in every probe share no value string, so the categorical operator "
                    "cannot merge any of them. If the semantic arm merges the synonymous pair and "
                    "only that one, the embedding has bought a decision counting alone cannot make, "
                    "and it bought it in bits rather than at a cut-off"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
