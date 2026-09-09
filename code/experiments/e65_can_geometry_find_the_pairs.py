"""E65: can the graph's own geometry find the key pairs worth pricing, without a model and without
pricing them?

THE JOB GEOMETRY IS BEING ASKED TO DO. Not to decide anything. Pricing a key pair means rebuilding
the whole state and scoring it, which on 853 seller-typed keys is seconds per pair against 86,767
pairs that clear the structural checks. That is the search problem, and E48 already measured its
cost elsewhere in the system at 3,317 bits. A generator that puts the pairs worth pricing near the
top turns an impossible sweep into a short list, and it changes what gets found without touching
what is true, because the objective still decides.

So this scores ranking, not verdicts, and it needs no pricing at all.

WHAT IT RANKS AGAINST. E20's gold: a pair is positive exactly when the two key strings match after
lowercasing and stripping punctuation, which on this data means `Size`, `size`, `SIZE`, `Size.` and
`size.` are one key. That gold is a string function and the paper says so, which is what makes it
the right target here: **the geometry never reads the strings**, so recovering those pairs is a real
test of whether use follows spelling in the data.

THE FEATURES, THREE OF WHICH COST NOTHING.

  support Jaccard   overlap of the node sets that support each key, the graph's own shape
  record Jaccard    overlap of the records carrying each key
  co-key Jaccard    overlap of the OTHER keys appearing alongside each, the free analogue of an
                    attention context
  value Jaccard     overlap of the value vocabularies, which is what the objective's pooling term
                    actually reads

Against two baselines that must be beaten or the exercise is pointless: ranking at random, and
ranking by character n-gram similarity of the key names, which is a string method and therefore has
the gold's own definition baked in. Beating random is necessary; the n-gram row is there to show
what an unfair advantage looks like.

WHAT WOULD REFUTE IT. Precision at the top no better than the base rate. Then geometry is not a
generator on real data and the search gap needs a different attack.
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

from escrow.keymerge import cooccurring_pairs, merge_is_in_scope
from escrow.protocol import run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e65_can_geometry_find_the_pairs.json")
LIMIT = int(os.environ.get("ESCROW_E65_LIMIT", "4000"))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def char_ngrams(s, lo=3, hi=4):
    s = f"  {s.lower()}  "
    return {s[i:i + n] for n in range(lo, hi + 1) for i in range(len(s) - n + 1)}


def main():
    from lazada_c2_rawkey import load_records, norm_key
    recs, _, _ = load_records()
    recs = recs[:LIMIT]
    g, _ = run_stream(recs)
    print(f"{len(recs)} listings, {len(g.keys)} raw keys, K = {g.K}", flush=True)

    nodes_of = collections.defaultdict(set)
    for v in g.nodes.values():
        for kid in v.S:
            nodes_of[kid].add(v.nid)
    recs_of = collections.defaultdict(set)
    cokeys_of = collections.defaultdict(set)
    for rid, keys in g.record_keys.items():
        for kid in keys:
            recs_of[kid].add(rid)
            cokeys_of[kid] |= (set(keys) - {kid})
    values_of = {ki.kid: set(ki.inventory) for ki in g.keys.values()}
    ngrams_of = {kid: char_ngrams(g.key_name[kid]) for kid in g.key_by_id}

    cooc = cooccurring_pairs(g)
    kids = sorted(g.key_by_id)
    rows = []
    for i in range(len(kids)):
        for j in range(i + 1, len(kids)):
            a, b = kids[i], kids[j]
            if (a, b) in cooc:
                continue
            in_scope, _, _ = merge_is_in_scope(g, a, b)
            if not in_scope:
                continue
            rows.append({
                "a": g.key_name[a], "b": g.key_name[b],
                "gold": norm_key(g.key_name[a]) == norm_key(g.key_name[b]),
                "support_jaccard": jaccard(nodes_of[a], nodes_of[b]),
                "record_jaccard": jaccard(recs_of[a], recs_of[b]),
                "cokey_jaccard": jaccard(cokeys_of[a], cokeys_of[b]),
                "value_jaccard": jaccard(values_of[a], values_of[b]),
                "keyname_ngram_jaccard": jaccard(ngrams_of[a], ngrams_of[b]),
            })
    # A feature that is zero for nearly every pair cannot rank anything: sorting by it sorts ties,
    # and the "top 50" is whatever order the pairs were built in. The first run of this experiment
    # reported every model-free feature at exactly random and I read that as a finding. It was not.
    # On the encyclopedia stream support Jaccard is zero for 99.0 percent of pairs with 3 distinct
    # values, record Jaccard is zero for 100 percent by construction since the generator only offers
    # non-co-occurring pairs, and value Jaccard is zero for 99.6 percent. Only co-key Jaccard had
    # resolution, at 477 distinct values. Three of the four rows were measuring nothing.
    resolution = {}
    for f in ("support_jaccard", "record_jaccard", "cokey_jaccard", "value_jaccard",
              "keyname_ngram_jaccard"):
        vals = [r[f] for r in rows]
        z = sum(1 for x in vals if x == 0.0)
        resolution[f] = {"share_exactly_zero": round(z / len(vals), 4) if vals else None,
                         "distinct_values": len(set(vals)), "max": round(max(vals), 4) if vals else 0,
                         "usable_as_a_ranker": len(set(vals)) > 20 and z / max(len(vals), 1) < 0.95}
    print("feature resolution, checked before any ranking is reported:\n")
    print(f"  {'feature':<26} {'zero share':>11} {'distinct':>9} {'usable':>8}")
    for f, r in resolution.items():
        print(f"  {f:<26} {r['share_exactly_zero']:>11.4f} {r['distinct_values']:>9} "
              f"{str(r['usable_as_a_ranker']):>8}")
    print()

    gold = [r for r in rows if r["gold"]]
    base = len(gold) / len(rows) if rows else 0.0
    print(f"{len(rows)} pairs in scope and never co-occurring, of which "
          f"{len(gold)} are gold same-key pairs (base rate {100*base:.4f} percent)\n", flush=True)

    rng = random.Random(0)
    features = ["support_jaccard", "record_jaccard", "cokey_jaccard", "value_jaccard",
                "keyname_ngram_jaccard"]
    report = {"experiment": "E65 geometry as a candidate generator, scored by ranking",
              "records": len(recs), "raw_keys": len(g.keys), "K": int(g.K),
              "pairs_in_scope": len(rows), "gold_pairs": len(gold),
              "base_rate": round(base, 6), "feature_resolution": resolution, "by_feature": {}}

    print(f"  {'feature':<24} " + " ".join(f"{'p@' + str(k):>8}" for k in (10, 50, 200))
          + f" {'recall@200':>11} {'lift@50':>8}")
    for f in features + ["random"]:
        if f == "random":
            ranked = rows[:]
            rng.shuffle(ranked)
        else:
            ranked = sorted(rows, key=lambda r: -r[f])
        out = {}
        for k in (10, 50, 200):
            hits = sum(1 for r in ranked[:k] if r["gold"])
            out[f"precision_at_{k}"] = round(hits / k, 4)
            out[f"hits_at_{k}"] = hits
        out["recall_at_200"] = round(out["hits_at_200"] / len(gold), 4) if gold else None
        out["lift_at_50"] = round(out["precision_at_50"] / base, 1) if base else None
        report["by_feature"][f] = out
        print(f"  {f:<24} " + " ".join(f"{out['precision_at_' + str(k)]:>8.4f}"
                                        for k in (10, 50, 200))
              + f" {out['recall_at_200']:>11} {out['lift_at_50']:>8}")

    free = [f for f in features if f != "keyname_ngram_jaccard"
            and resolution[f]["usable_as_a_ranker"]]
    best = (max(free, key=lambda f: report["by_feature"][f]["precision_at_50"]) if free
            else "none of the model-free features has enough resolution to rank")
    report["headline"] = {
        "base_rate": round(base, 6),
        "best_model_free_feature": best,
        "usable_model_free_features": free,
        "its_lift_at_50": (report["by_feature"][best]["lift_at_50"] if free else None),
        "keyname_ngram_lift_at_50": report["by_feature"]["keyname_ngram_jaccard"]["lift_at_50"],
        "random_lift_at_50": report["by_feature"]["random"]["lift_at_50"],
        "reading": ("the key-name n-gram row has the gold's own definition baked into it and is here "
                    "to show what an unfair advantage looks like, not as a competitor. What matters "
                    "is whether a feature that never reads the key strings beats the base rate by "
                    "enough to turn 86,767 pairs into a short list"),
    }
    report["gold_pair_feature_values"] = [
        {"a": r["a"], "b": r["b"], **{f: round(r[f], 4) for f in features}} for r in gold[:25]]
    print("\nwhere the gold pairs actually sit, which decides whether they were ever findable:")
    print(f"  {'pair':<44} " + " ".join(f"{f.split('_')[0][:7]:>8}" for f in features))
    for r in gold[:12]:
        print(f"  {r['a'][:20] + ' + ' + r['b'][:20]:<44} "
              + " ".join(f"{r[f]:>8.4f}" for f in features))

    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
