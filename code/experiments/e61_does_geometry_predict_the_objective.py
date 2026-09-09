"""E61: does the geometry of the graph predict what the objective will accept?

THE FRAMING, WHICH IS THE POINT. Geometry must not become a criterion. A cosine similarity with a
cut-off deciding whether two keys are one key is `tau_new` under a new name, and the whole line of
work exists to remove it. But geometry can be a *search heuristic*, and that is a different thing
with different rules: a proposal is not a decision. The objective still accepts or rejects, in bits,
exactly as it does now. A generator that proposes better candidates changes what gets found and
never what is true.

WHY THAT IS WORTH DOING RATHER THAN MERELY SAFE. E48 measured where the loss actually is: handing
the shipped moves a truth-seeded start reaches a description 3,317 bits shorter than they reach from
scratch, at omega 0.7445 against the 0.3466 we report. The criterion already prefers the better
answer; the search does not reach it. So a better proposal mechanism attacks the measured gap
directly, and it is the only kind of improvement that cannot cost the guarantee.

WHAT IS MEASURED. For every candidate key pair in a stream, three similarities that cost nothing,
and one that costs a model:

  support Jaccard      overlap of the node sets that support each key, which is the graph's own
                       shape and needs no embedding at all
  record Jaccard       overlap of the records carrying each key
  co-key Jaccard       overlap of the OTHER keys that appear alongside each, which is the closest
                       free analogue of an attention context
  value centroid cos   cosine between the mean embedding of each key's values, which costs a model

against the quantity that actually decides: the objective's own merge delta in bits. The question is
not whether these correlate loosely. It is whether ranking candidates by a similarity puts the pairs
the objective would accept at the top, because that is what a generator has to do.

WHAT WOULD REFUTE IT. A flat or negative rank correlation, or a top-k that misses the accepted pairs.
Then geometry is not a useful proposal mechanism here and the search gap needs a different attack.
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
from escrow.keymerge import cooccurring_pairs, key_merge_delta
from escrow.protocol import run_stream
from escrow.provenance import stamped
from escrow.semantic import Embedding

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e61_does_geometry_predict_the_objective.json")
EMB_DIR = os.path.join(ROOT, "results", "embeddings")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def geometry(g, recs):
    """Per key: the nodes supporting it, the records carrying it, and the keys beside it."""
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
    return nodes_of, recs_of, cokeys_of


def value_centroid_cos(emb, g, kid_a, kid_b):
    def centroid(kid):
        ki = g.key_by_id.get(kid)
        if ki is None:
            return None
        vecs = [emb.vectors[v] for v in ki.inventory if v in emb.vectors]
        if not vecs:
            return None
        d = len(vecs[0])
        c = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
        n = math.sqrt(sum(x * x for x in c)) or 1.0
        return [x / n for x in c]
    ca, cb = centroid(kid_a), centroid(kid_b)
    if ca is None or cb is None:
        return None
    return sum(x * y for x, y in zip(ca, cb))


def run_case(name, recs, emb):
    g, b = run_stream(recs)
    cooc = cooccurring_pairs(g)
    nodes_of, recs_of, cokeys_of = geometry(g, recs)
    kids = sorted(g.key_by_id)
    rows = []
    for i in range(len(kids)):
        for j in range(i + 1, len(kids)):
            a, bb = kids[i], kids[j]
            if (a, bb) in cooc:
                continue
            d = key_merge_delta(g, BatchObjective, a, bb)
            if d is None:
                continue
            row = {"a": g.key_name[a], "b": g.key_name[bb], "delta_bits": round(d, 2),
                   "accepted": d < 0,
                   "support_jaccard": round(jaccard(nodes_of[a], nodes_of[bb]), 4),
                   "record_jaccard": round(jaccard(recs_of[a], recs_of[bb]), 4),
                   "cokey_jaccard": round(jaccard(cokeys_of[a], cokeys_of[bb]), 4)}
            if emb is not None:
                c = value_centroid_cos(emb, g, a, bb)
                row["value_centroid_cos"] = None if c is None else round(c, 4)
            rows.append(row)
    return g, rows


def score(rows, feature):
    """How well does this similarity rank the pairs the objective accepts?"""
    usable = [r for r in rows if r.get(feature) is not None]
    if len(usable) < 3:
        return None
    xs = [-r[feature] for r in usable]        # higher similarity should mean lower delta
    ys = [r["delta_bits"] for r in usable]
    rho = spearman(xs, ys)
    accepted = [r for r in usable if r["accepted"]]
    ranked = sorted(usable, key=lambda r: -r[feature])
    out = {"spearman_vs_delta": round(rho, 4), "pairs": len(usable),
           "accepted_pairs": len(accepted)}
    for k in (1, 3, 5):
        if len(ranked) >= k and accepted:
            hit = sum(1 for r in ranked[:k] if r["accepted"])
            out[f"accepted_in_top_{k}"] = f"{hit}/{min(k, len(accepted))}"
    return out


def main():
    cases = {}

    from e57_semantics_in_the_key_merge import build_stream as ks
    cases["keymerge_fixture"] = (ks(0), os.path.join(EMB_DIR, "keymerge_fixture.json"))
    from e54_key_merge_by_value_distribution import build_stream as k54, N_PER_ROLE
    cases["shared_vocab_fixture"] = (k54(N_PER_ROLE, 0), None)
    try:
        import llm_graph_formation as L
        cases["wikipedia"] = (L.build_wikipedia()["records"],
                              os.path.join(EMB_DIR, "wikipedia.json"))
    except Exception:                                              # noqa: BLE001
        pass

    report = {"experiment": "E61 geometry as a proposal mechanism, never as a criterion",
              "features": ["support_jaccard", "record_jaccard", "cokey_jaccard",
                           "value_centroid_cos"],
              "cases": {}}

    for name, (recs, emb_path) in cases.items():
        emb = Embedding.from_file(emb_path) if emb_path and os.path.exists(emb_path) else None
        g, rows = run_case(name, recs, emb)
        acc = [r for r in rows if r["accepted"]]
        print(f"\n{name}: {len(recs)} records, K={g.K}, {len(g.keys)} keys, "
              f"{len(rows)} priced pairs, {len(acc)} accepted by the objective")
        feats = {}
        for f in report["features"]:
            sc = score(rows, f)
            if sc is None:
                continue
            feats[f] = sc
            tops = "  ".join(f"top{k.split('_')[-1]} {v}" for k, v in sc.items()
                             if k.startswith("accepted_in_top"))
            print(f"    {f:<22} spearman {sc['spearman_vs_delta']:>+7.4f}   {tops}")
        report["cases"][name] = {"records": len(recs), "K": int(g.K), "keys": len(g.keys),
                                 "priced_pairs": len(rows), "accepted": len(acc),
                                 "by_feature": feats,
                                 "accepted_pairs": [f"{r['a']}+{r['b']}" for r in acc][:20]}

    best = {}
    for name, c in report["cases"].items():
        if c["by_feature"]:
            best[name] = max(c["by_feature"], key=lambda f: c["by_feature"][f]["spearman_vs_delta"])
    report["headline"] = {
        "best_feature_by_case": best,
        "spearman_by_case": {n: {f: v["spearman_vs_delta"] for f, v in c["by_feature"].items()}
                             for n, c in report["cases"].items()},
        "reading": ("a similarity that ranks the accepted pairs to the top is a candidate generator "
                    "worth having, because it attacks the 3,317 bit search gap E48 measured without "
                    "touching the criterion. One that does not is a similarity looking for a "
                    "threshold, and this project has no use for that"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
