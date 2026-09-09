"""E67: every method on both streams, all of them held to one record at a time.

WHY THIS EXISTS. Two asymmetries have been found in the paper's comparison table and neither was
stated where it mattered.

  The classical baselines are batch. E66 measured what that is worth: `kmeans_silhouette` makes about
  190 full passes over every record and then selects k with a score that needs the whole matrix. Held
  to one pass with the true k handed over for free, the same method falls from 0.9797 to 0.5436, and
  character n-grams fall from 1.0000 to 0.1555.

  The language model reads twenty records per call. The paper states this, in both directions, and
  then reports the number anyway. Twenty records of lookahead is not the question the rule answers.

So this table has one protocol and everything is in it: **one record, decide, next**. The model gets
no lookahead. The baselines get no second pass. The rule gets what it always had.

LAZADA IS HERE BECAUSE IT IS WHERE THE RULE LOOKS WORST. The paper reports ARI 0.0013 against the
silver product type. That label is model-derived, being AutoPKG's own knowledge graph rather than a
human gold, and it asks for **5,518 types over 21,260 records, 3.85 records each**. Proposition 1
prices groups that small out of a stream that long, so the paper predicts the loss before running.
A prediction is worth more when the thing it predicts is measured against every alternative under
one protocol, which is what this does.

WHAT WOULD REFUTE THE PAPER'S READING. A streaming method reaching a good score on the Lazada silver
label. That would mean the groups are findable one record at a time after all, and the scope
argument is an excuse rather than an explanation.
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

from escrow.protocol import record_labels, run_consensus, run_stream
from escrow.provenance import stamped
from e4_baseline_army import _ari

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e67_everything_streaming.json")
BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
MODEL = os.environ.get("ESCROW_E67_MODEL", "gemini-3.8-flash")
WIKI_N = int(os.environ.get("ESCROW_E67_WIKI_N", "120"))
LAZADA_N = int(os.environ.get("ESCROW_E67_LAZADA_N", "400"))
R = 3


def streaming_kmeans_ari(recs, truth, k, seed=0):
    """Sequential k-means, one pass, k handed over. The generous streaming baseline of E66."""
    import numpy as np
    cols = sorted({f"{a}={b}" for r in recs for a, b in r.items()}
                  | {f"KEY:{a}" for r in recs for a in r})
    ix = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(recs), len(cols)))
    for i, r in enumerate(recs):
        for a, b in r.items():
            X[i, ix[f"{a}={b}"]] = 1.0
            X[i, ix[f"KEY:{a}"]] = 1.0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    centres = np.zeros((k, X.shape[1]))
    counts = np.zeros(k)
    lab = [None] * len(X)
    filled = 0
    for idx in order:
        x = X[idx]
        if filled < k:
            centres[filled] = x
            counts[filled] = 1
            lab[idx] = filled
            filled += 1
            continue
        j = int(np.argmin(np.linalg.norm(centres - x, axis=1)))
        counts[j] += 1
        centres[j] += (x - centres[j]) / counts[j]
        lab[idx] = j
    return round(_ari(truth, lab), 4)


def escrow_arms(recs, truth):
    g0, _ = run_stream(recs)
    single = round(_ari(truth, record_labels(g0, len(recs))), 4)
    cons = run_consensus(recs, R=R, seed=11)
    best = cons["best"]
    return {"streaming_single_order": {"ARI": single, "K": int(g0.K)},
            f"streaming_shortest_of_{R}": {"ARI": round(_ari(truth, best["labels"]), 4),
                                           "K": int(best["K"])}}


def llm_arm(recs, truth, chunk, api_key):
    import llm_graph_formation as L
    old = L.CHUNK
    L.CHUNK = chunk
    try:
        ds = {"name": "stream", "records": recs, "truth": truth}
        r = L.run_llm_api(BASE, MODEL, api_key, ds, "greedy", 0,
                          extra_body={"reasoning_effort": "none"})
    finally:
        L.CHUNK = old
    a = r["assignments"]
    pred = [a[i] if a[i] is not None else f"__u{i}" for i in range(len(recs))]
    return {"ARI": round(_ari(truth, pred), 4), "K": len(r["nodes"]),
            "assigned": sum(1 for x in a if x is not None),
            "tokens": r["tokens_in"] + r["tokens_out"], "calls": r["n_chunks"]}


def main():
    key = os.environ.get("GEMINI_API_KEY")
    report = {"experiment": "E67 one protocol for every method: one record at a time",
              "protocol": "one record, decide, next. No lookahead, no second pass, no future.",
              "model": MODEL, "streams": {}}

    streams = {}
    import llm_graph_formation as L
    wf = L.build_wikipedia()
    streams["encyclopedia"] = (wf["records"][:WIKI_N], wf["truth"][:WIKI_N],
                               "three obvious kinds, human labelled")
    try:
        from lazada_c2_rawkey import load_records, load_silver_types
        recs, pids, _ = load_records()
        pid2type, _ = load_silver_types()
        recs, pids = recs[:LAZADA_N], pids[:LAZADA_N]
        truth = [pid2type.get(p) for p in pids]
        keep = [i for i, t in enumerate(truth) if t is not None]
        streams["lazada"] = ([recs[i] for i in keep], [truth[i] for i in keep],
                             "silver product type from AutoPKG's own graph, model derived")
    except Exception as e:                                        # noqa: BLE001
        print(f"lazada unavailable: {e!r}"[:120])

    for name, (recs, truth, label_note) in streams.items():
        k_true = len({t for t in truth if t is not None})
        row = {"records": len(recs), "true_kinds": k_true, "label": label_note,
               "records_per_kind": round(len(recs) / max(k_true, 1), 2), "arms": {}}
        print(f"\n{'=' * 70}\n{name}: {len(recs)} records, {k_true} kinds, "
              f"{row['records_per_kind']} records each\n  label: {label_note}\n{'=' * 70}")

        row["arms"]["escrow"] = escrow_arms(recs, truth)
        for k, v in row["arms"]["escrow"].items():
            print(f"  ESCROW {k:<28} ARI {v['ARI']:>8.4f}  K={v['K']}")

        row["arms"]["sequential_kmeans_true_k_given"] = {
            "ARI": streaming_kmeans_ari(recs, truth, max(k_true, 2)),
            "note": "one pass, and it is told the number of kinds, which ESCROW is not"}
        print(f"  sequential k-means, true k given     "
              f"ARI {row['arms']['sequential_kmeans_true_k_given']['ARI']:>8.4f}  K={k_true}")

        if key:
            for chunk in (1, 20):
                try:
                    r = llm_arm(recs, truth, chunk, key)
                    tag = "one record per call" if chunk == 1 else "twenty per call, ADVANTAGED"
                    row["arms"][f"llm_chunk_{chunk}"] = {**r, "note": tag}
                    print(f"  {MODEL}, {tag:<28} ARI {r['ARI']:>8.4f}  K={r['K']}  "
                          f"{r['calls']} calls, {r['tokens']} tokens")
                except Exception as e:                            # noqa: BLE001
                    row["arms"][f"llm_chunk_{chunk}"] = {"error": repr(e)[:160]}
                    print(f"  {MODEL} chunk {chunk} failed: {e!r}"[:110])
        else:
            print("  (no GEMINI_API_KEY, language model arms skipped)")
        report["streams"][name] = row

    report["headline"] = {
        n: {"records_per_kind": r["records_per_kind"],
            "escrow_shortest_of_3": r["arms"]["escrow"][f"streaming_shortest_of_{R}"]["ARI"],
            "sequential_kmeans_true_k_given":
                r["arms"]["sequential_kmeans_true_k_given"]["ARI"],
            "llm_one_record_per_call": r["arms"].get("llm_chunk_1", {}).get("ARI"),
            "llm_twenty_per_call_advantaged": r["arms"].get("llm_chunk_20", {}).get("ARI")}
        for n, r in report["streams"].items()}
    report["headline"]["reading"] = (
        "one protocol for everything. Where the label asks for groups of about four records in a "
        "stream of thousands, Proposition 1 prices them out and the paper says so before running; "
        "the question this answers is whether any streaming method does better there, because if "
        "none does, the scope argument is an explanation rather than an excuse")
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
