"""E63: is there information for an embedding to extract, before anyone builds a code to extract it?

THE DISCIPLINE THIS RESTORES. E56 and E57 built a semantic code, measured it, watched it lose, and
then reasoned backwards about why. That is the wrong order and it produced two conclusions I had to
withdraw. The claim "an embedding helps" reduces to one statement that can be measured without
building anything: **values that are close in embedding space behave the same way in the data**. If
that correlation is absent, no code built on the embedding can help and the extractor is beside the
point. If it is present and the code still loses, the code is at fault, which is the case where
iterating is worth the time.

WHAT BEHAVING THE SAME WAY MEANS, MEASURED THREE WAYS. For pairs of values under one key:

  co-key         do they appear on records carrying the same other keys
  co-value       do they appear alongside the same other values
  same-node      do they end up in the same node of the graph the engine builds

Each is a Jaccard over sets the data already determines. None of them needs a model.

THE CONTROL THAT DECIDES WHETHER TO PAY FOR A MODEL AT ALL. Every correlation is computed twice,
once for the sentence embedding and once for **character three-to-five-gram cosine**, which needs no
model and cannot represent a word. E43 already found that code scoring 1.0000 on the encyclopedia
stream where the language model reached 0.9905, so it is not a straw man: if it carries the same
information about value behaviour, the embedding is buying nothing that string overlap does not
already give, and the honest recommendation is to use the free one.

WHAT WOULD REFUTE THE WHOLE SEMANTIC TRACK. Correlations at or near zero for the embedding on every
stream. Then the track stops, and that is a result worth having rather than a disappointment.
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

from escrow.protocol import run_stream
from escrow.provenance import stamped
from escrow.semantic import Embedding

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e63_is_the_information_there.json")
EMB_DIR = os.path.join(ROOT, "results", "embeddings")

MAX_PAIRS = 4000
MIN_OCCURRENCES = 3          # a value seen twice has almost no behaviour to measure


def char_ngrams(s, lo=3, hi=5):
    s = f"  {s.lower()}  "
    out = collections.Counter()
    for n in range(lo, hi + 1):
        for i in range(len(s) - n + 1):
            out[s[i:i + n]] += 1
    return out


def cos_counter(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else 0.0


def behaviour(recs, g):
    """For each (key, value): the other keys, the other values, and the nodes it appears with."""
    node_of = {}
    for v in g.nodes.values():
        for m in v.members:
            node_of.setdefault(m, set()).add(v.nid)
    cokeys = collections.defaultdict(set)
    covalues = collections.defaultdict(set)
    nodes = collections.defaultdict(set)
    counts = collections.Counter()
    for i, rec in enumerate(recs, start=1):
        items = [(k, str(v)) for k, v in rec.items()]
        for k, v in items:
            key = (k, v)
            counts[key] += 1
            cokeys[key] |= {kk for kk, _ in items if kk != k}
            covalues[key] |= {vv for kk, vv in items if kk != k}
            nodes[key] |= node_of.get(i, set())
    return cokeys, covalues, nodes, counts


def run_stream_case(name, recs, emb_path):
    emb = Embedding.from_file(emb_path) if emb_path and os.path.exists(emb_path) else None
    g, _ = run_stream(recs)
    cokeys, covalues, nodes, counts = behaviour(recs, g)

    by_key = collections.defaultdict(list)
    for (k, v), c in counts.items():
        if c >= MIN_OCCURRENCES:
            by_key[k].append(v)

    rng = random.Random(0)
    pairs = []
    for k, vals in by_key.items():
        if len(vals) < 2:
            continue
        for _ in range(min(200, len(vals) * (len(vals) - 1) // 2)):
            a, b = rng.sample(vals, 2)
            pairs.append((k, a, b))
    rng.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]

    ng = {}
    rows = []
    for k, a, b in pairs:
        for v in (a, b):
            if v not in ng:
                ng[v] = char_ngrams(v)
        row = {"char_ngram_cos": cos_counter(ng[a], ng[b]),
               "cokey_jaccard": jaccard(cokeys[(k, a)], cokeys[(k, b)]),
               "covalue_jaccard": jaccard(covalues[(k, a)], covalues[(k, b)]),
               "samenode_jaccard": jaccard(nodes[(k, a)], nodes[(k, b)])}
        if emb is not None and emb.has(a) and emb.has(b):
            row["embedding_cos"] = emb.cos(a, b)
        rows.append(row)

    out = {"records": len(recs), "K": int(g.K), "pairs_scored": len(rows),
           "values_with_at_least_%d_occurrences" % MIN_OCCURRENCES:
               sum(len(v) for v in by_key.values()),
           "correlations": {}}
    for sim in ("embedding_cos", "char_ngram_cos"):
        usable = [r for r in rows if sim in r]
        if len(usable) < 20:
            continue
        out["correlations"][sim] = {
            beh: round(pearson([r[sim] for r in usable], [r[beh] for r in usable]), 4)
            for beh in ("cokey_jaccard", "covalue_jaccard", "samenode_jaccard")}
        out["correlations"][sim]["pairs"] = len(usable)
    return out


def main():
    cases = {}
    try:
        import llm_graph_formation as L
        cases["wikipedia"] = (L.build_wikipedia()["records"],
                              os.path.join(EMB_DIR, "wikipedia.json"))
    except Exception as e:                                          # noqa: BLE001
        print(f"wikipedia unavailable: {e!r}"[:100])
    try:
        from lazada_c2_rawkey import load_records
        recs, _, _ = load_records()
        cases["lazada"] = (recs[:4000], os.path.join(EMB_DIR, "lazada.json"))
    except Exception as e:                                          # noqa: BLE001
        print(f"lazada unavailable: {e!r}"[:100])

    report = {"experiment": "E63 is there information for an embedding to extract",
              "min_occurrences": MIN_OCCURRENCES, "streams": {}}
    for name, (recs, emb_path) in cases.items():
        r = run_stream_case(name, recs, emb_path)
        report["streams"][name] = r
        print(f"\n{name}: {r['records']} records, K={r['K']}, {r['pairs_scored']} value pairs\n")
        print(f"  {'similarity':<18} {'co-key':>9} {'co-value':>10} {'same-node':>11} {'pairs':>7}")
        for sim, c in r["correlations"].items():
            print(f"  {sim:<18} {c['cokey_jaccard']:>+9.4f} {c['covalue_jaccard']:>+10.4f} "
                  f"{c['samenode_jaccard']:>+11.4f} {c['pairs']:>7}")

    report["headline"] = {
        "correlations": {n: r["correlations"] for n, r in report["streams"].items()},
        "reading": ("a correlation near zero for the embedding means no code built on it can help on "
                    "that stream, whatever the code. A correlation the character n-grams match means "
                    "the embedding is buying nothing string overlap does not already give, and the "
                    "free one should be used instead"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
