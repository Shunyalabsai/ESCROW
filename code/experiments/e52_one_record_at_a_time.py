"""E52: give the language model the stream the way the rule gets it, one record at a time.

WHY. Every comparison so far hands the model twenty records per call. It therefore decides with a
twenty-record lookahead the rule never has: it can see that four of the twenty share a key before it
names anything. The rule sees one record, decides, and cannot look forward. That difference has been
stated in the paper as an asymmetry and never measured, and it is the one asymmetry that could
account for the model's win on the encyclopedia stream.

THE CHANGE. One line: the chunk size goes from twenty to one. Everything else is untouched, the same
shipped prompt, the same node list carried forward, the same parser, greedy decoding, the same
records in the same order, the same cap on output length. So the model now answers exactly the
question the rule answers: this record, these nodes, attach or mint.

WHAT THE TWO STREAMS SEPARATE. On the encyclopedia stream a model can lean on what it knows about
the world: it has read about films and mountains, so "director" and "elevation_m" mean something to
it before any evidence arrives. The rule has no such knowledge and never will, by construction. If
the model's win survives at chunk one, that win is domain knowledge doing work the rule has given up
on purpose, and the paper should say so. If it does not survive, the win was the lookahead.

On the noise stream there is nothing to know. Whatever a model does there is not domain knowledge,
because there is no domain. The count it returns at chunk one against chunk twenty says whether
seeing twenty records at once was what drove it to name almost every one of them.

WHICH MODELS. The models the paper has used so far were released in 2024, which is a fair complaint
and not one worth arguing with. The default list here is the 2026 Qwen3.5 dense ladder, which is the
direct successor of the two Qwen2.5 checkpoints already in the abstention table, and it brackets the
five to seven billion band from both sides because no dense model released in 2026 sits inside it.

REASONING MODELS. Qwen3.5 and its family are hybrid: left alone they emit a long private trace
before the answer, which the shipped cap on output length would cut off mid-trace, so the parser
would see nothing and every record would go unassigned. That is a measurement of the harness and not
of the model. The trace is therefore switched off through the model's own chat template, which is
the documented way to ask for a direct answer, and the fact is recorded per model in the result.

COST. Twenty times the calls, so the streams are shorter here than in E41 and the models are run
one at a time with each downloaded checkpoint removed before the next, since the box is short of
disk.
"""
from __future__ import annotations
import json
import os
import random
import shutil
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
OUT = os.path.join(ROOT, "results", "e52_one_record_at_a_time.json")
HUB = os.path.expanduser("~/.cache/huggingface/hub")

# The 2026 dense ladder from one family, smallest first so a run cut short still answers. There is
# no dense 2026 release between four and nine billion parameters, so the pair brackets the band
# rather than sitting in it, and 9B is the successor of the Qwen2.5-7B already in the table.
MODELS = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
_env = os.environ.get("ESCROW_E52_MODELS")
if _env:
    MODELS = [m.strip() for m in _env.split(",") if m.strip()]

NULL_N = int(os.environ.get("ESCROW_E52_NULL_N", "100"))
WIKI_N = int(os.environ.get("ESCROW_E52_WIKI_N", "120"))
SEEDS = (0, 1)
CHUNKS = (20, 1)


def null_stream(n, seed, keys=4, d=10):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def cache_dir_for(name):
    return os.path.join(HUB, "models--" + name.replace("/", "--"))


def escrow_on(records, truth=None):
    g, _ = run_stream(records)
    if truth is None:
        return {"K": int(g.K)}
    return {"K": int(g.K),
            "ARI": round(_ari(truth, record_labels(g, len(records))), 4)}


def disable_thinking(tok):
    """Ask the model's own chat template for a direct answer rather than a private trace.

    Templates that have no such switch raise on the extra argument, so the switch is tried once and
    kept only if it works. Returns True when the trace was actually turned off, which the result
    file records per model, because a reader must be able to tell a measurement of the model from a
    measurement of how we called it.
    """
    probe = [{"role": "user", "content": "hi"}]
    try:
        off = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    except Exception:                                                 # noqa: BLE001
        return False
    on = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    if off == on:
        return False
    orig = tok.apply_chat_template

    def patched(*a, **k):
        k.setdefault("enable_thinking", False)
        return orig(*a, **k)

    tok.apply_chat_template = patched
    return True


def run_at_chunk(name, tok, model, ds, chunk):
    """The shipped runner with only CHUNK changed, restored afterwards."""
    old = L.CHUNK
    L.CHUNK = chunk
    try:
        t0 = time.time()
        r = L.run_llm(name, tok, model, ds, "greedy", 0)
        r["seconds"] = round(time.time() - t0, 1)
        return r
    finally:
        L.CHUNK = old


def load_named(name):
    """Load a checkpoint for text generation.

    The 2026 checkpoints ship as one multimodal wrapper whose text tower is the language model, so
    the plain causal head does not always bind to the weights on disk. The wrapper generates from
    text alone, which is all this experiment ever gives it, so the loader tries the plain head
    first and falls back to the wrapper. Which class bound is recorded with the result.
    """
    import torch
    import transformers
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    last = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
        except Exception as e:                                        # noqa: BLE001
            last = e
            print(f"  [load] {cls_name} did not bind: {e!r}"[:200], flush=True)
            continue
        model.eval()
        print(f"[model] {name} loaded in {time.time() - t0:.0f}s through {cls_name}", flush=True)
        return tok, model, cls_name
    raise last if last is not None else RuntimeError("no loader class available")


def main():
    wiki_full = L.build_wikipedia()
    wiki = {"name": "wikipedia", "records": wiki_full["records"][:WIKI_N],
            "truth": wiki_full["truth"][:WIKI_N]}

    ref = {"encyclopedia": escrow_on(wiki["records"], wiki["truth"]),
           "noise": {str(s): escrow_on(null_stream(NULL_N, s)) for s in SEEDS}}
    print(f"ESCROW on the same records: encyclopedia {ref['encyclopedia']}, "
          f"noise {[ref['noise'][str(s)]['K'] for s in SEEDS]}\n", flush=True)

    prior = {}
    if os.path.exists(OUT):
        try:
            prior = json.load(open(OUT))
        except Exception:                                             # noqa: BLE001
            prior = {}
    kept = [m for m in prior.get("models", []) if m.get("runs")]
    done = {m["model"] for m in kept}

    report = {"experiment": "E52 the language model at one record at a time",
              "changed": "chunk size only, 20 to 1",
              "held_fixed": ["the shipped system prompt", "the node list carried forward",
                             "the parser", "greedy decoding", "the cap on output length",
                             "the records and their order"],
              "stream_lengths": {"noise": NULL_N, "encyclopedia": WIKI_N, "seeds": list(SEEDS)},
              "escrow_reference": ref, "models": list(kept)}

    for name in MODELS:
        if name in done:
            print(f"  {name}: already measured, kept\n", flush=True)
            continue
        preexisting = os.path.isdir(cache_dir_for(name))
        row = {"model": name, "was_cached_before_this_run": preexisting, "runs": []}
        try:
            tok, model, loader = load_named(name)
        except Exception as e:                                        # noqa: BLE001
            row["error"] = repr(e)[:300]
            row.pop("runs")
            report["models"].append(row)
            print(f"  {name}: FAILED {e!r}\n", flush=True)
            json.dump(stamped(report), open(OUT, "w"), indent=2)
            continue
        row["loaded_through"] = loader
        row["reasoning_trace_turned_off"] = disable_thinking(tok)
        report["models"].append(row)
        print(f"  [chat template] private trace turned off: "
              f"{row['reasoning_trace_turned_off']}", flush=True)

        for chunk in CHUNKS:
            r_row = {"chunk": chunk, "noise": {}, "encyclopedia": None}
            for s in SEEDS:
                ds = {"name": f"null_s{s}", "records": null_stream(NULL_N, s),
                      "truth": [None] * NULL_N}
                r = run_at_chunk(name, tok, model, ds, chunk)
                r_row["noise"][str(s)] = {
                    "nodes": len(r["nodes"]), "seconds": r["seconds"],
                    "assigned": sum(1 for a in r["assignments"] if a is not None),
                    "tokens": r["tokens_in"] + r["tokens_out"]}
                print(f"  {name} chunk {chunk:>2}, noise seed {s}: {len(r['nodes'])} nodes "
                      f"(truth 0, ESCROW {ref['noise'][str(s)]['K']}) in {r['seconds']}s",
                      flush=True)
            r = run_at_chunk(name, tok, model, wiki, chunk)
            a = r["assignments"]
            pred = [a[i] if a[i] is not None else f"__u{i}" for i in range(len(wiki["records"]))]
            r_row["encyclopedia"] = {
                "ARI": round(_ari(wiki["truth"], pred), 4), "nodes": len(r["nodes"]),
                "assigned": sum(1 for x in a if x is not None),
                "seconds": r["seconds"], "tokens": r["tokens_in"] + r["tokens_out"]}
            print(f"  {name} chunk {chunk:>2}, encyclopedia: "
                  f"ARI {r_row['encyclopedia']['ARI']} at K={r_row['encyclopedia']['nodes']} "
                  f"in {r['seconds']}s (ESCROW {ref['encyclopedia']})\n", flush=True)
            row["runs"].append(r_row)
            json.dump(stamped(report), open(OUT, "w"), indent=2)

        del model, tok
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:                                             # noqa: BLE001
            pass
        if not preexisting:
            d = cache_dir_for(name)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                print(f"  [disk] removed {d}", flush=True)

    def at(m, c):
        return next((r for r in m.get("runs", []) if r["chunk"] == c), None)

    ran = [m for m in report["models"] if m.get("runs")]
    report["headline"] = {
        "escrow_encyclopedia_ARI": ref["encyclopedia"].get("ARI"),
        "escrow_noise_nodes": {s: ref["noise"][s]["K"] for s in ref["noise"]},
        "encyclopedia_ARI_at_chunk_20": {m["model"]: at(m, 20)["encyclopedia"]["ARI"]
                                         for m in ran if at(m, 20)},
        "encyclopedia_ARI_at_chunk_1": {m["model"]: at(m, 1)["encyclopedia"]["ARI"]
                                        for m in ran if at(m, 1)},
        "noise_nodes_at_chunk_20": {m["model"]: {k: v["nodes"] for k, v in at(m, 20)["noise"].items()}
                                    for m in ran if at(m, 20)},
        "noise_nodes_at_chunk_1": {m["model"]: {k: v["nodes"] for k, v in at(m, 1)["noise"].items()}
                                   for m in ran if at(m, 1)},
        "reading": ("at chunk one the model answers the question the rule answers, with no "
                    "lookahead. A win that survives is domain knowledge the rule has given up by "
                    "construction; a win that does not survive was the lookahead."),
    }
    print(json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
