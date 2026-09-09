"""E56: what does an embedding actually buy, in bits, and where does it buy nothing?

THE QUESTION NOBODY REPORTS. Systems add an embedding and report an accuracy that went up. That
tells you the embedding helped somewhere, not how much, not where, and not what it cost. Because
semantics enter here as a code rather than as a score, the answer is a number in bits on every
stream, and the same number says when semantics are not worth having.

WHAT IS MEASURED. For each stream, the naming charge of every value block under two codes: the
shipped categorical one, which spends log2(inventory + 1) bits on each value new to a block, and the
semantic one, which spends -log2 P_sem(value | what this block already holds). Everything else about
the description is identical, so the difference is attributable to exactly one term.

Then the mixture, which is what would actually ship:

    L_mix = -log2( 2^-L_cat / 2 + 2^-L_sem / 2 )  <=  min(L_cat, L_sem) + 1 bit

THE NULL COMES FIRST, AS ALWAYS. On a stream of arbitrary symbols there is nothing for an embedding
to know, so the semantic arm must earn nothing and the mixture must cost about one bit. If instead
semantics buys a saving on noise, the charge is not normalised and the construction is a similarity
score in disguise, which would be fatal rather than disappointing. That is the first row of the
table and the reason to read it before any other.

WHAT WOULD REFUTE THE DESIGN. Semantics paying on the null; or the mixture costing materially more
than one bit over the better arm; or the saving on real streams being indistinguishable from zero,
which would mean the embedding adds nothing this data needs and the honest move is to leave it out.
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

from escrow.protocol import run_stream
from escrow.provenance import stamped
from escrow.semantic import Embedding, block_naming_bits, mixture_bits

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e56_what_does_semantics_buy.json")
EMB_DIR = os.path.join(ROOT, "results", "embeddings")


def noise_stream(n=200, keys=4, d=10, seed=0):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def wikipedia_stream():
    import llm_graph_formation as L
    return L.build_wikipedia()["records"]


def lazada_stream(limit=2000):
    from lazada_c2_rawkey import load_records
    recs, _, _ = load_records()
    return recs[:limit]


STREAMS = {
    "noise": noise_stream,
    "wikipedia": wikipedia_stream,
    "lazada": lazada_stream,
}


def blocks_in_arrival_order(recs, g):
    """For every (node, key) block, the value strings in the order that block first met them.

    The naming charge is levied once per value new to a block, so the arrival order is what the
    charge is a function of. Reading it off the records rather than off the stored counts keeps this
    honest: a block's counts have forgotten the order.
    """
    out = collections.defaultdict(list)
    member_of = {}
    for v in g.nodes.values():
        for m in v.members:
            member_of.setdefault(m, []).append(v.nid)
    for i, rec in enumerate(recs, start=1):
        for nid in member_of.get(i, ()):
            for k, val in rec.items():
                out[(nid, k)].append(str(val))
    # and the background, which is one block per key over the records no node covers
    for i, rec in enumerate(recs, start=1):
        if i in member_of:
            continue
        for k, val in rec.items():
            out[(-1, k)].append(str(val))
    return out


def inventories(recs):
    inv = collections.defaultdict(list)
    seen = collections.defaultdict(set)
    for rec in recs:
        for k, val in rec.items():
            s = str(val)
            if s not in seen[k]:
                seen[k].add(s)
                inv[k].append(s)
    return inv


def score(recs, emb):
    g, b = run_stream(recs)
    blocks = blocks_in_arrival_order(recs, g)
    inv = inventories(recs)
    cat = sem = 0.0
    covered = 0
    for (nid, key), seq in blocks.items():
        vocab = inv[key]
        cat += block_naming_bits(emb, seq, vocab, semantic=False)
        sem += block_naming_bits(emb, seq, vocab, semantic=True)
        covered += sum(1 for v in set(seq) if emb.has(v))
    return {"K": int(g.K), "records": len(recs), "blocks": len(blocks),
            "naming_bits_categorical": round(cat, 1),
            "naming_bits_semantic": round(sem, 1),
            "naming_bits_mixture": round(mixture_bits(cat, sem), 1),
            "semantics_saves_bits": round(cat - sem, 1),
            "mixture_costs_over_best": round(mixture_bits(cat, sem) - min(cat, sem), 3),
            "values_with_a_vector": covered}


def main():
    report = {"experiment": "E56 what an embedding buys, in bits, and where it buys nothing",
              "construction": ("two complete codes differing only in the naming charge, combined as "
                               "a Bayesian mixture with a half prior on each, so the combined code "
                               "is never worse than the better arm by more than one bit"),
              "streams": {}}
    for name, build in STREAMS.items():
        path = os.path.join(EMB_DIR, f"{name}.json")
        if not os.path.exists(path):
            report["streams"][name] = {"skipped": f"no embedding cache at {path}"}
            print(f"{name:<12} skipped, no embedding cache. Build it with "
                  f"code/tools/embed_values.py --stream {name}")
            continue
        try:
            recs = build()
        except Exception as e:                                     # noqa: BLE001
            report["streams"][name] = {"skipped": f"stream unavailable: {e!r}"[:160]}
            print(f"{name:<12} skipped, {e!r}"[:110])
            continue
        emb = Embedding.from_file(path)
        row = score(recs, emb)
        row["embedding"] = json.load(open(path)).get("model", "?")
        report["streams"][name] = row
        print(f"{name:<12} categorical {row['naming_bits_categorical']:>10.1f} bits   "
              f"semantic {row['naming_bits_semantic']:>10.1f}   "
              f"saves {row['semantics_saves_bits']:>+9.1f}   "
              f"mixture costs {row['mixture_costs_over_best']:.3f} over the better arm", flush=True)

    done = {k: v for k, v in report["streams"].items() if "skipped" not in v}
    report["headline"] = {
        "streams_measured": sorted(done),
        "semantics_saves_bits": {k: v["semantics_saves_bits"] for k, v in done.items()},
        "mixture_cost_over_the_better_arm": {k: v["mixture_costs_over_best"]
                                             for k, v in done.items()},
        "null_is_silent": (done["noise"]["semantics_saves_bits"] <= 0.0
                           if "noise" in done else None),
        "reading": ("a saving on the null would mean the semantic charge is not normalised and the "
                    "construction is a similarity score in disguise. A saving on a real stream is "
                    "what the embedding is worth there, and it is the same number that says when to "
                    "leave the embedding out"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
