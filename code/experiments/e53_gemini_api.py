"""E53: the same two decisions, given to a served frontier model instead of a local checkpoint.

WHY. Every language model in the paper so far is an open-weight checkpoint we load ourselves, and
the largest is fourteen billion parameters. A fair reading of the abstention result is that it may
be a property of small open models rather than of prompting, and the only way to answer that is to
ask a served frontier model the identical question. This does that through Google's OpenAI
compatible endpoint, so the prompt, the chunking, the parser, the node list carried forward and the
record order are the ones the local runs use and only the call changes.

WHAT IS MEASURED. The same two streams. A noise stream with nothing in it, where the truth is zero
nodes and any node the model builds is invented. And the encyclopedia stream of three obvious types,
where a model that abstains by being incompetent is visible rather than counted as a success. Both
at chunk twenty, which is the protocol the paper reports, and at chunk one, which is the question the
rule answers with no lookahead at all.

WHAT IS LOGGED. Everything the endpoint reports: prompt tokens, completion tokens, wall time per
call, the node count after every chunk, and the finish reason. Served models bill by the token, so
the cost column is the one an engineer reads before any accuracy column, and it is recorded per
model per stream rather than summarised.

THE KEY. Read from the environment as GEMINI_API_KEY. It is never printed, never written into the
result file, and never committed. Run it as

    GEMINI_API_KEY=... python3 code/experiments/e53_gemini_api.py --list
    GEMINI_API_KEY=... python3 code/experiments/e53_gemini_api.py

The first lists the models the key can reach, so the model list below can be set from what is
actually available rather than from a guess.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))

import llm_graph_formation as L
from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped
from e4_baseline_army import _ari

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e53_gemini_api.json")
BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"

MODELS = [m.strip() for m in os.environ.get("ESCROW_E53_MODELS", "").split(",") if m.strip()]
# These models spend a private reasoning trace out of the same output budget as the answer, so left
# alone the answer is cut off part way through the JSON and the parser sees one record of twenty.
# That would measure the harness, not the model. Turning the trace off is the same choice the local
# runs make through the chat template, and it is recorded with every run.
REASONING = os.environ.get("ESCROW_E53_REASONING", "none")
EXTRA = {"reasoning_effort": REASONING} if REASONING else None
NULL_N = int(os.environ.get("ESCROW_E53_NULL_N", "100"))
WIKI_N = int(os.environ.get("ESCROW_E53_WIKI_N", "120"))
SEEDS = (0, 1)
CHUNKS = (20, 1)


def api_key():
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k:
        raise SystemExit("set GEMINI_API_KEY in the environment; it is never read from a file here")
    return k


def list_models(key):
    import urllib.request
    req = urllib.request.Request(LIST_URL, headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    rows = []
    for m in d.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            rows.append({"model": m["name"].replace("models/", ""),
                         "input_token_limit": m.get("inputTokenLimit"),
                         "output_token_limit": m.get("outputTokenLimit"),
                         "display_name": m.get("displayName")})
    return sorted(rows, key=lambda r: r["model"], reverse=True)


def null_stream(n, seed, keys=4, d=10):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def escrow_on(records, truth=None):
    g, _ = run_stream(records)
    if truth is None:
        return {"K": int(g.K)}
    return {"K": int(g.K), "ARI": round(_ari(truth, record_labels(g, len(records))), 4)}


def run_at_chunk(base, model, key, ds, chunk):
    """The shipped served runner with only CHUNK changed, restored afterwards."""
    old = L.CHUNK
    L.CHUNK = chunk
    try:
        t0 = time.time()
        r = L.run_llm_api(base, model, key, ds, "greedy", 0, extra_body=EXTRA)
        r["seconds"] = round(time.time() - t0, 1)
        return r
    finally:
        L.CHUNK = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the models this key can reach and exit")
    args = ap.parse_args()
    key = api_key()

    if args.list:
        for r in list_models(key):
            print(f"  {r['model']:<52} in {r['input_token_limit']}  out {r['output_token_limit']}")
        return

    if not MODELS:
        raise SystemExit("set ESCROW_E53_MODELS to a comma separated list; run --list first")

    wiki_full = L.build_wikipedia()
    wiki = {"name": "wikipedia", "records": wiki_full["records"][:WIKI_N],
            "truth": wiki_full["truth"][:WIKI_N]}
    ref = {"encyclopedia": escrow_on(wiki["records"], wiki["truth"]),
           "noise": {str(s): escrow_on(null_stream(NULL_N, s)) for s in SEEDS}}
    print(f"ESCROW on the same records: encyclopedia {ref['encyclopedia']}, "
          f"noise {[ref['noise'][str(s)]['K'] for s in SEEDS]}, 0 tokens\n", flush=True)

    prior = {}
    if os.path.exists(OUT):
        try:
            prior = json.load(open(OUT))
        except Exception:                                             # noqa: BLE001
            prior = {}
    kept = [m for m in prior.get("models", []) if m.get("runs")]
    done = {m["model"] for m in kept}

    report = {"experiment": "E53 the same two decisions given to a served frontier model",
              "endpoint": BASE,
              "held_fixed": ["the shipped system prompt", "the node list carried forward",
                             "the parser", "temperature 0", "the cap on output length",
                             "the records and their order"],
              "note_on_the_prompt": ("the served runner folds the system prompt into the user turn, "
                                     "because that is what the other served run in this paper does; "
                                     "the text is identical"),
              "reasoning_effort": REASONING,
              "note_on_reasoning": ("these models bill a private trace against the same output "
                                    "budget as the answer, so at the shipped cap the answer is cut "
                                    "off mid JSON and the parser sees one record of twenty. The "
                                    "trace is turned off, which is what the local runs do through "
                                    "the chat template. On the same probe chunk effort low built "
                                    "13 nodes where none built 9, so the setting moves the count "
                                    "and is reported rather than buried"),
              "stream_lengths": {"noise": NULL_N, "encyclopedia": WIKI_N, "seeds": list(SEEDS)},
              "escrow_reference": dict(ref, tokens=0),
              "models": list(kept)}

    for name in MODELS:
        if name in done:
            print(f"  {name}: already measured, kept\n", flush=True)
            continue
        row = {"model": name, "runs": []}
        report["models"].append(row)
        for chunk in CHUNKS:
            r_row = {"chunk": chunk, "noise": {}, "encyclopedia": None}
            try:
                for s in SEEDS:
                    ds = {"name": f"null_s{s}", "records": null_stream(NULL_N, s),
                          "truth": [None] * NULL_N}
                    r = run_at_chunk(BASE, name, key, ds, chunk)
                    r_row["noise"][str(s)] = {
                        "nodes": len(r["nodes"]), "seconds": r["seconds"],
                        "assigned": sum(1 for a in r["assignments"] if a is not None),
                        "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                        "nodes_after_each_chunk": [c["nodes_after"] for c in r["chunks"]],
                        "calls": r["n_chunks"]}
                    print(f"  {name} chunk {chunk:>2}, noise seed {s}: {len(r['nodes'])} nodes "
                          f"(truth 0, ESCROW {ref['noise'][str(s)]['K']}) in {r['seconds']}s, "
                          f"{r['tokens_in'] + r['tokens_out']} tokens", flush=True)
                r = run_at_chunk(BASE, name, key, wiki, chunk)
                a = r["assignments"]
                pred = [a[i] if a[i] is not None else f"__u{i}" for i in range(len(wiki["records"]))]
                r_row["encyclopedia"] = {
                    "ARI": round(_ari(wiki["truth"], pred), 4), "nodes": len(r["nodes"]),
                    "assigned": sum(1 for x in a if x is not None),
                    "seconds": r["seconds"], "tokens_in": r["tokens_in"],
                    "tokens_out": r["tokens_out"], "calls": r["n_chunks"]}
                print(f"  {name} chunk {chunk:>2}, encyclopedia: "
                      f"ARI {r_row['encyclopedia']['ARI']} at K={r_row['encyclopedia']['nodes']} "
                      f"in {r['seconds']}s, {r['tokens_in'] + r['tokens_out']} tokens "
                      f"(ESCROW {ref['encyclopedia']}, 0 tokens)\n", flush=True)
            except Exception as e:                                    # noqa: BLE001
                r_row["error"] = repr(e)[:300]
                print(f"  {name} chunk {chunk}: FAILED {e!r}\n", flush=True)
            row["runs"].append(r_row)
            json.dump(stamped(report), open(OUT, "w"), indent=2)

    def at(m, c):
        return next((r for r in m.get("runs", []) if r["chunk"] == c and not r.get("error")), None)

    ran = [m for m in report["models"] if any(not r.get("error") for r in m.get("runs", []))]
    report["headline"] = {
        "escrow_encyclopedia_ARI": ref["encyclopedia"].get("ARI"),
        "escrow_noise_nodes": {s: ref["noise"][s]["K"] for s in ref["noise"]},
        "escrow_tokens": 0,
        "noise_nodes_at_chunk_20": {m["model"]: {k: v["nodes"] for k, v in at(m, 20)["noise"].items()}
                                    for m in ran if at(m, 20)},
        "noise_nodes_at_chunk_1": {m["model"]: {k: v["nodes"] for k, v in at(m, 1)["noise"].items()}
                                   for m in ran if at(m, 1)},
        "encyclopedia_ARI_at_chunk_20": {m["model"]: at(m, 20)["encyclopedia"]["ARI"]
                                         for m in ran if at(m, 20)},
        "encyclopedia_ARI_at_chunk_1": {m["model"]: at(m, 1)["encyclopedia"]["ARI"]
                                        for m in ran if at(m, 1)},
        "tokens_by_model": {
            m["model"]: sum((v["tokens_in"] + v["tokens_out"])
                            for r in m["runs"] if not r.get("error")
                            for v in list(r["noise"].values()) +
                            ([r["encyclopedia"]] if r["encyclopedia"] else []))
            for m in ran},
        "reading": ("a served frontier model that returns zero or one node on the noise stream while "
                    "still scoring on the encyclopedia stream would show abstention is reachable by "
                    "prompting after all, and the paper's sentence would have to be narrowed"),
    }
    print(json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
