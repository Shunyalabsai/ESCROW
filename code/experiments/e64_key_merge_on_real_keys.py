"""E64: the key merge on 3,033 raw seller-typed keys, with the scope condition in force.

WHY LAZADA AND NOT THE ENCYCLOPEDIA STREAM. With the scope condition wired in, the encyclopedia
stream yields zero merges from 356 priced pairs, which is the right answer and also an empty test:
there is nothing for a candidate generator to rank and nothing to check against a gold. Lazada is
where the decision actually lives. Sellers type their own keys, so the stream carries `Size`, `size`,
`SIZE`, `Size.` and `size.` as five separate keys, and each is frequent enough to clear the scope.

WHAT IS ASKED, IN ORDER.

  1. What does the operator propose, unprompted, over every pair that clears scope and never
     co-occurs. Reported in full, because a merge list is only honest if the wrong ones are shown
     next to the right ones.
  2. How does that compare with E20's gold. That gold is a string function, positive exactly when the
     two key strings match after lowercasing and stripping punctuation, so it scores normalisation
     and not use. The operator never reads the strings. Agreement is therefore interesting in both
     directions: a pair the gold has and the operator misses is a case where use does not follow
     spelling, and a pair the operator has and the gold lacks is a candidate for the decision the
     paper says is open.
  3. Whether the free geometry features rank the accepted pairs to the top, which E61 could not test
     on the encyclopedia stream because its accepted set was the singleton artefact.

WHAT WOULD REFUTE IT. The operator proposing merges that are obviously wrong to a reader, at a rate
the scope condition was supposed to have removed. Or the gold agreement being so low in both
directions that the operator is measuring nothing anybody can check.
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

from escrow.batch import BatchObjective
from escrow.keymerge import (MERGE_SCOPE_MIN_RECORDS, cooccurring_pairs, key_merge_delta,
                             merge_is_in_scope)
from escrow.protocol import run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e64_key_merge_on_real_keys.json")
LIMIT = int(os.environ.get("ESCROW_E64_LIMIT", "6000"))
MAX_PAIRS = int(os.environ.get("ESCROW_E64_MAX_PAIRS", "3000"))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main():
    from lazada_c2_rawkey import load_records, norm_key
    recs, _, stats = load_records()
    recs = recs[:LIMIT]
    g, b = run_stream(recs)
    print(f"{len(recs)} listings, {len(g.keys)} raw keys, K = {g.K}\n", flush=True)

    freq = collections.Counter(k for r in recs for k in r)
    cooc = cooccurring_pairs(g)
    nodes_of = collections.defaultdict(set)
    for v in g.nodes.values():
        for kid in v.S:
            nodes_of[kid].add(v.nid)
    cokeys_of = collections.defaultdict(set)
    for rid, keys in g.record_keys.items():
        for kid in keys:
            cokeys_of[kid] |= (set(keys) - {kid})

    kids = sorted(g.key_by_id)
    pairs = []
    for i in range(len(kids)):
        for j in range(i + 1, len(kids)):
            a, bb = kids[i], kids[j]
            if (a, bb) in cooc:
                continue
            in_scope, na, nb = merge_is_in_scope(g, a, bb)
            if not in_scope:
                continue
            pairs.append((a, bb))
    random.Random(0).shuffle(pairs)
    print(f"{len(pairs)} pairs clear the scope condition of {MERGE_SCOPE_MIN_RECORDS} "
          f"records and never co-occur; pricing {min(len(pairs), MAX_PAIRS)}\n", flush=True)

    rows = []
    for n, (a, bb) in enumerate(pairs[:MAX_PAIRS]):
        d = key_merge_delta(g, BatchObjective, a, bb)
        if d is None:
            continue
        na, nb = g.key_name[a], g.key_name[bb]
        rows.append({"a": na, "b": nb, "delta_bits": round(d, 1), "merge": d < 0,
                     "records_a": freq[na], "records_b": freq[nb],
                     "same_after_normalising": norm_key(na) == norm_key(nb),
                     "support_jaccard": round(jaccard(nodes_of[a], nodes_of[bb]), 4),
                     "cokey_jaccard": round(jaccard(cokeys_of[a], cokeys_of[bb]), 4)})
        if (n + 1) % 250 == 0:
            print(f"    priced {n + 1}", flush=True)

    acc = [r for r in rows if r["merge"]]
    gold_in_sample = [r for r in rows if r["same_after_normalising"]]
    tp = [r for r in acc if r["same_after_normalising"]]
    print(f"\npriced {len(rows)} pairs, {len(acc)} merges accepted")
    print(f"of those, {len(tp)} are pairs the string gold also calls the same key")
    print(f"the sample contained {len(gold_in_sample)} gold pairs in scope\n")
    print("every accepted merge, so the wrong ones sit beside the right ones:")
    for r in sorted(acc, key=lambda r: r["delta_bits"])[:40]:
        mark = "gold" if r["same_after_normalising"] else "    "
        print(f"   {mark}  {r['a']!r:<28} + {r['b']!r:<28} {r['delta_bits']:>9.1f} bits "
              f"({r['records_a']} + {r['records_b']} records)")

    report = {"experiment": "E64 the key merge on real seller-typed keys",
              "records": len(recs), "raw_keys": len(g.keys), "K": int(g.K),
              "scope_minimum_records": MERGE_SCOPE_MIN_RECORDS,
              "pairs_in_scope": len(pairs), "pairs_priced": len(rows),
              "accepted": len(acc),
              "accepted_that_the_string_gold_agrees_with": len(tp),
              "gold_pairs_in_the_priced_sample": len(gold_in_sample),
              "recall_on_gold_pairs_in_sample": (round(len(tp) / len(gold_in_sample), 4)
                                                 if gold_in_sample else None),
              "all_accepted": sorted(acc, key=lambda r: r["delta_bits"]),
              "closest_rejected": sorted([r for r in rows if not r["merge"]],
                                         key=lambda r: r["delta_bits"])[:20]}
    if acc:
        ranked_support = sorted(rows, key=lambda r: -r["support_jaccard"])
        ranked_cokey = sorted(rows, key=lambda r: -r["cokey_jaccard"])
        report["geometry_ranking"] = {
            "accepted_in_top_20_by_support_jaccard":
                sum(1 for r in ranked_support[:20] if r["merge"]),
            "accepted_in_top_20_by_cokey_jaccard":
                sum(1 for r in ranked_cokey[:20] if r["merge"]),
            "accepted_total": len(acc), "priced_total": len(rows)}
        print(f"\ngeometry: of the top 20 pairs by support Jaccard, "
              f"{report['geometry_ranking']['accepted_in_top_20_by_support_jaccard']} are accepted; "
              f"by co-key Jaccard, "
              f"{report['geometry_ranking']['accepted_in_top_20_by_cokey_jaccard']}. "
              f"Base rate is {len(acc)}/{len(rows)}.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
