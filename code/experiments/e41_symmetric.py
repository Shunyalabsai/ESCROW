"""E41: the symmetric test. Give the language model OUR tasks, not only its own.

WHY. E20 handed a language model a task built for it, deciding whether two key strings are the same
attribute, and it beat us. That comparison is only half of one. This project generalises its own rule
across domains, one engine unchanged on listings, infoboxes, music metadata and noun phrases, and
then reports where it loses. The fair mirror is to ask the model to generalise onto the decisions
this rule is built for, and to report that honestly whichever way it goes.

THE TWO TASKS, both of which the engine settles by its price.

  1. ABSTENTION ON NOISE. The paper's headline safety result is that no node is created on a stream
     with no latent structure: K = 0 on 48 null streams and on 300 more. A language model is asked
     to build the graph from the same kind of stream, four keys with ten independent uniform values
     each and nothing to find. The question is not accuracy, it is whether the model declines. Our
     answer is zero nodes. A model that invents kinds here is inventing them everywhere.

  2. PLANTED STRUCTURE. Eight groups with their own keys and values, which the engine recovers
     exactly on every arrival order at the paper's length of 3,000 records. Note that planted8 seeds
     the generator, so the three seeds here are three independently drawn streams and not three
     orderings of one; the model and the engine get the same stream in the same order as each other.

The prompt, the chunking, the parser and the arrival order are the ones llm_graph_formation.py
already uses, imported rather than rewritten, so the only thing that changes is the stream.

WHAT WOULD FALSIFY OUR SIDE. If the model returns one node, or none, on the null stream, then
abstention is not something a computed price buys that a prompt cannot, and the paper should stop
implying it is. Reported either way.
"""
from __future__ import annotations
import json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))

import llm_graph_formation as L
from escrow.protocol import run_stream
from escrow.provenance import stamped
sys.path.insert(0, os.path.join(HERE))
from e4_baseline_army import _ari, planted8

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "e41_symmetric.json"))
N = 200                       # records per stream, kept small: the model reads every one
SEEDS = (0, 1, 2)


def null_stream(n, seed, keys=4, d=10):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def escrow_K(recs):
    g, _ = run_stream(recs)
    return int(g.K)


def escrow_labels(recs):
    g, _ = run_stream(recs)
    lab = [-1] * len(recs)
    for v in g.nodes.values():
        for m in v.members:
            lab[m - 1] = v.nid
    return lab, int(g.K)


def main():
    model_name, tok, model = L.load_model()
    report = {"experiment": "E41 the symmetric test: the language model on our tasks",
              "model": model_name, "records_per_stream": N, "seeds": list(SEEDS),
              "prompt_protocol": "imported from llm_graph_formation.py, unchanged",
              "null": [], "planted": []}

    print("1. abstention on a stream with no structure (ours: K = 0)")
    for s in SEEDS:
        recs = null_stream(N, s)
        ds = {"name": f"null_s{s}", "records": recs, "truth": [None] * len(recs)}
        t0 = time.time()
        run = L.run_llm(model_name, tok, model, ds, "greedy", s)
        k_llm = len(run["nodes"])
        k_esc = escrow_K(recs)
        report["null"].append({"seed": s, "llm_nodes": k_llm, "escrow_nodes": k_esc,
                               "llm_node_names": [n["name"] for n in run["nodes"]][:12],
                               "llm_nodes_after_each_chunk": [c["nodes_after"] for c in run["chunks"]],
                               "records_after_each_chunk": [(i + 1) * L.CHUNK
                                                            for i in range(len(run["chunks"]))],
                               "tokens": run["tokens_in"] + run["tokens_out"],
                               "seconds": round(time.time() - t0, 1)})
        print(f"   seed {s}: language model {k_llm} nodes, ESCROW {k_esc}", flush=True)

    print("2. planted eight-group structure (ours: ARI 1.0)")
    for s in SEEDS:
        recs, truth = planted8(n=N, seed=s)
        ds = {"name": f"planted8_s{s}", "records": recs, "truth": truth}
        run = L.run_llm(model_name, tok, model, ds, "greedy", s)
        a = run["assignments"]
        pred = [a[i] if a[i] is not None else f"__bg{i}" for i in range(len(recs))]
        lab, k_esc = escrow_labels(recs)
        report["planted"].append({
            "seed": s,
            "llm_ARI": round(_ari(truth, pred), 4), "llm_nodes": len(run["nodes"]),
            "escrow_ARI": round(_ari(truth, lab), 4), "escrow_nodes": k_esc,
            "tokens": run["tokens_in"] + run["tokens_out"]})
        r = report["planted"][-1]
        print(f"   seed {s}: language model ARI {r['llm_ARI']} K={r['llm_nodes']}, "
              f"ESCROW ARI {r['escrow_ARI']} K={r['escrow_nodes']}", flush=True)

    nl = report["null"]
    report["headline"] = {
        "llm_nodes_on_noise": [x["llm_nodes"] for x in nl],
        "escrow_nodes_on_noise": [x["escrow_nodes"] for x in nl],
        "llm_ARI_on_planted": [x["llm_ARI"] for x in report["planted"]],
        "escrow_ARI_on_planted": [x["escrow_ARI"] for x in report["planted"]],
        "reading": ("E20 gave the model a task built for it and it won. This gives it the two the "
                    "price is built for. Abstention is the one that matters: a rule that cannot "
                    "decline on noise will invent kinds on any stream."),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
