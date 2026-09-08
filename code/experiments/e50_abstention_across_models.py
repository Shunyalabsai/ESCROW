"""E50: is the failure to abstain a property of one model, or of prompting?

WHY. E41 gives one language model the decision the price is built for and it builds 80, 195 and 195
nodes on a stream with nothing in it, where the rule builds none. E42 then shows the wording of the
instruction moves that count from 20 to 85, and that a prompt stating outright that a stream may hold
no kinds still builds 82. From those two the paper says abstention is not something a prompt can be
asked for. That sentence is broader than the evidence: one model, one family, one size.

This runs the same test across families and sizes, on the identical records, the identical shipped
prompt, the identical chunking and parser, and greedy decoding so nothing is sampled. Each model gets
the noise stream at three seeds, where the truth is zero nodes, and the encyclopedia stream, so a
model that abstains by being incompetent is visible rather than counted as a success.

WHAT WOULD FALSIFY THE CLAIM, and this is the point of running it. If any model returns zero or one
node on the noise stream while still scoring well on the encyclopedia stream, then abstention IS
reachable by prompting and the sentence must be narrowed to the models where it is not. Reported
either way, and the counts are printed per model rather than averaged.

DISK. The box has limited free space, so models are fetched one at a time and each downloaded model
is removed before the next. Models that were already cached before this run are never removed.
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
OUT = os.path.join(ROOT, "results", "e50_abstention_across_models.json")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
NULL_N = 200
SEEDS = (0, 1, 2)

# Open-weight instruction models that need no access token, chosen to vary the family and,
# within Qwen, the size, so a failure to abstain cannot be blamed on one lab's recipe or on
# capacity alone. Llama and Gemma are gated and are skipped rather than worked around; that
# exclusion is recorded in the result file.
# Open-weight instruction models that need no access token, ordered by what each one buys, so a
# run cut short still answers the most important question first. Qwen3 is a 2025 model and answers
# "you only tested last year's"; then a second and third family; then the Qwen size ladder, which
# separates family from capacity. Llama and Gemma are gated and are skipped rather than worked
# around, and that exclusion is recorded in the result file rather than left silent.
MODELS = [
    "Qwen/Qwen2.5-14B-Instruct",          # the paper's model, already cached, never removed
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",                      # 2025 frontier family, Apache 2.0
    "openai/gpt-oss-20b",                 # a different lab entirely, mixture of experts
    "allenai/Olmo-3-7B-Instruct",         # fully open: data, checkpoints and logs
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen3-14B",                     # same 2025 family, larger
    "Qwen/Qwen2.5-3B-Instruct",           # the size ladder inside one family
]
_env = os.environ.get("ESCROW_E50_MODELS")
if _env:
    MODELS = [m.strip() for m in _env.split(",") if m.strip()]


def null_stream(n, seed, keys=4, d=10):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def cache_dir_for(name):
    return os.path.join(HUB, "models--" + name.replace("/", "--"))


def load_named(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[model] {name} loaded in {time.time() - t0:.0f}s", flush=True)
    return tok, model


def main():
    wiki = L.build_wikipedia()
    escrow_null = {}
    for s in SEEDS:
        g, _ = run_stream(null_stream(NULL_N, s))
        escrow_null[s] = int(g.K)
    g, _ = run_stream(wiki["records"])
    escrow_wiki = round(_ari(wiki["truth"],
                             record_labels(g, len(wiki["records"]))), 4)
    print(f"ESCROW, for reference: {escrow_null} nodes on noise, "
          f"encyclopedia ARI {escrow_wiki}\n", flush=True)

    prior = {}
    if os.path.exists(OUT):
        try:
            prior = json.load(open(OUT))
        except Exception:                                             # noqa: BLE001
            prior = {}
    done = {m["model"] for m in prior.get("models", []) if "nodes_on_noise" in m}
    report = {"experiment": "E50 is the failure to abstain a property of one model or of prompting",
              "held_fixed": ["the shipped system prompt", "the records", "their order",
                             "chunk size 20", "greedy decoding", "the parser"],
              "escrow_reference": {"nodes_on_noise": escrow_null, "encyclopedia_ARI": escrow_wiki},
              "models": []}

    report["models"] = [m for m in prior.get("models", []) if "nodes_on_noise" in m]
    report["gated_and_skipped"] = sorted(
        {m["model"] for m in prior.get("models", []) if "gated" in str(m.get("error", ""))}
        | {m for m in ("meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it")})
    for name in MODELS:
        if name in done:
            print(f"  {name}: already measured in a previous batch, kept\n", flush=True)
            continue
        preexisting = os.path.isdir(cache_dir_for(name))
        row = {"model": name, "was_cached_before_this_run": preexisting}
        try:
            tok, model = load_named(name)
        except Exception as e:                                        # noqa: BLE001
            row["error"] = repr(e)[:300]
            report["models"].append(row)
            print(f"  {name}: FAILED {e!r}\n", flush=True)
            json.dump(stamped(report), open(OUT, "w"), indent=2)
            continue

        nulls = []
        for s in SEEDS:
            ds = {"name": f"null_s{s}", "records": null_stream(NULL_N, s),
                  "truth": [None] * NULL_N}
            r = L.run_llm(name, tok, model, ds, "greedy", s)
            nulls.append(len(r["nodes"]))
            print(f"  {name} noise seed {s}: {len(r['nodes'])} nodes "
                  f"(truth 0, ESCROW {escrow_null[s]})", flush=True)
        rw = L.run_llm(name, tok, model, wiki, "greedy", 0)
        a = rw["assignments"]
        pred = [a[i] if a[i] is not None else f"__u{i}" for i in range(len(wiki["records"]))]
        ari = round(_ari(wiki["truth"], pred), 4)
        row.update({"nodes_on_noise": nulls, "encyclopedia_ARI": ari,
                    "encyclopedia_nodes": len(rw["nodes"]),
                    "abstains": all(k <= 1 for k in nulls),
                    "competent_on_the_easy_stream": ari >= 0.8})
        print(f"  {name} encyclopedia: ARI {ari} at K={len(rw['nodes'])}\n", flush=True)
        report["models"].append(row)
        json.dump(stamped(report), open(OUT, "w"), indent=2)

        del model, tok
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:                                             # noqa: BLE001
            pass
        if not preexisting:
            d = cache_dir_for(name)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                print(f"  [disk] removed {d}", flush=True)

    ran = [m for m in report["models"] if "nodes_on_noise" in m]
    report["headline"] = {
        "models_run": len(ran),
        "models_that_abstained_on_noise": [m["model"] for m in ran if m["abstains"]],
        "nodes_on_noise_by_model": {m["model"]: m["nodes_on_noise"] for m in ran},
        "encyclopedia_ARI_by_model": {m["model"]: m["encyclopedia_ARI"] for m in ran},
        "escrow_nodes_on_noise": escrow_null,
        "reading": ("a model that returns zero or one node on the noise stream while still scoring "
                    "on the encyclopedia stream would show abstention is reachable by prompting, "
                    "and the paper's sentence would have to be narrowed"),
    }
    print(json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
