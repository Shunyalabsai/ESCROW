"""E42: how much of a language model's result is the prompt.

WHY. The paper reports one prompt, written once, and a number beside it. A reader is entitled to ask
what that number is a property of. This project claims its own rule carries no calibrated quantity,
and the honest comparison is not accuracy against accuracy, it is settled quantity against settled
quantity. If the prompt is a quantity someone has to set per use case, and if setting it differently
moves the answer, then the prompt is the language model's calibrated parameter and the comparison
must be stated that way.

THE DESIGN. Everything is held fixed: the model, the records, their arrival order, the chunk size,
greedy decoding so there is no sampling at all, the rendering of the records, the output format
instructions and the parser. One thing varies, the system prompt, and the variants split into two
classes that must be read separately.

  PARAPHRASE   the same instruction in different words. Nothing is added, removed or reversed.
               A practitioner who rewrote the prompt for tone or house style would produce these.
               Any movement here is movement caused by wording alone.
  POLICY       the instruction itself differs: be conservative, be liberal, or be told that a
               stream may have no structure in it. A practitioner tunes here, deliberately.

THE TWO STREAMS, one where the model wins and one where it should refuse.
  encyclopedia  320 infobox records of three obvious types, scored by agreement with the gold.
                The paper's shipped prompt reaches 0.9905 here, better than the engine.
  no structure  200 records with nothing to find, scored by how many nodes get built. The truth
                is zero, which is what the engine returns.

WHAT WOULD FALSIFY THE CLAIM. If every paraphrase lands within noise of the shipped prompt on both
streams, then the prompt is not a calibrated quantity in any meaningful sense and this paper should
stop implying that it is. Reported either way, and the shipped prompt is included unchanged so it
can be seen whether it is the lucky one.
"""
from __future__ import annotations
import json, os, random, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))

import llm_graph_formation as L
from escrow.protocol import run_stream
from escrow.provenance import stamped
from e4_baseline_army import _ari

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "e42_prompt_sensitivity.json"))
NULL_N = 200

SHIPPED = L.SYSTEM   # captured before anything is patched

VARIANTS = [
    ("shipped", "paraphrase", SHIPPED),

    ("paraphrase_reordered", "paraphrase",
     "Reply with JSON only. You are building a graph of nodes from a stream of records, where each "
     "record describes one item as key=value lines. One node stands for one kind of item, and "
     "records describing the same kind of item attach to the same node. Whenever an existing node "
     "fits, reuse it; create a new node only when none fits."),

    ("paraphrase_bulleted", "paraphrase",
     "You build a graph of nodes from a stream of records.\n"
     "- Each record describes one item as key=value lines.\n"
     "- A node stands for one kind of item.\n"
     "- Records of the same kind of item attach to the same node.\n"
     "- Reuse an existing node whenever one fits.\n"
     "- Create a new node only when no existing node fits.\n"
     "- Reply with JSON only."),

    ("paraphrase_persona", "paraphrase",
     "You are an experienced data engineer who organises catalogues. You build a graph of nodes "
     "from a stream of records. Each record describes one item as key=value lines. A node stands "
     "for one kind of item. Records of the same kind of item attach to the same node. Reuse an "
     "existing node whenever one fits. Create a new node only when no existing node fits. Reply "
     "with JSON only."),

    ("policy_conservative", "policy",
     "You build a graph of nodes from a stream of records. Each record describes one item as "
     "key=value lines. A node stands for one kind of item. Records of the same kind of item attach "
     "to the same node. Be conservative: strongly prefer reusing an existing node, and keep the "
     "number of nodes small. Create a new node only when no existing node could possibly fit. "
     "Reply with JSON only."),

    ("policy_liberal", "policy",
     "You build a graph of nodes from a stream of records. Each record describes one item as "
     "key=value lines. A node stands for one kind of item. Records of the same kind of item attach "
     "to the same node. Be precise: create a new node whenever a record does not clearly match an "
     "existing one, so that each node stays a single well defined kind. Reply with JSON only."),

    ("policy_may_be_no_structure", "policy",
     "You build a graph of nodes from a stream of records. Each record describes one item as "
     "key=value lines. A node stands for one kind of item. Records of the same kind of item attach "
     "to the same node. Reuse an existing node whenever one fits. Some streams have no kinds in "
     "them at all, in which case every record belongs to one single node and you should create no "
     "others. Create a new node only when no existing node fits. Reply with JSON only."),
]


def null_stream(n, seed, keys=4, d=10):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def spread(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return {"min": round(min(xs), 4), "max": round(max(xs), 4),
            "range": round(max(xs) - min(xs), 4), "sd": round(statistics.pstdev(xs), 4)}


def main():
    model_name, tok, model = L.load_model()
    wiki = L.build_wikipedia()
    nul = {"name": "null", "records": null_stream(NULL_N, 0), "truth": [None] * NULL_N}

    g, _ = run_stream(nul["records"])
    escrow_null_K = int(g.K)

    report = {"experiment": "E42 how much of a language model's result is the prompt",
              "model": model_name, "decoding": "greedy, no sampling",
              "held_fixed": ["model", "records", "arrival order", "chunk size 20", "greedy decoding",
                             "record rendering", "output format instructions", "parser"],
              "varied": "the system prompt only",
              "escrow_for_reference": {"encyclopedia_ARI": 0.8744, "no_structure_nodes": escrow_null_K},
              "variants": []}

    for vname, vclass, text in VARIANTS:
        L.SYSTEM = text
        t0 = time.time()
        rw = L.run_llm(model_name, tok, model, wiki, "greedy", 0)
        a = rw["assignments"]
        pred = [a[i] if a[i] is not None else f"__u{i}" for i in range(len(wiki["records"]))]
        ari = round(_ari(wiki["truth"], pred), 4)
        kw = len(rw["nodes"])
        rn = L.run_llm(model_name, tok, model, nul, "greedy", 0)
        kn = len(rn["nodes"])
        row = {"prompt": vname, "class": vclass, "encyclopedia_ARI": ari, "encyclopedia_nodes": kw,
               "no_structure_nodes": kn, "system_prompt_words": len(text.split()),
               "seconds": round(time.time() - t0, 1)}
        report["variants"].append(row)
        print(f"  {vname:28s} [{vclass:10s}] encyclopedia ARI {ari:.4f} K={kw:3d} | "
              f"no structure K={kn:4d} (truth 0, ESCROW {escrow_null_K})", flush=True)
        json.dump(stamped(report), open(OUT, "w"), indent=2)

    L.SYSTEM = SHIPPED
    par = [r for r in report["variants"] if r["class"] == "paraphrase"]
    allv = report["variants"]
    report["headline"] = {
        "paraphrases_only_wording_differs": {
            "n": len(par),
            "encyclopedia_ARI": spread([r["encyclopedia_ARI"] for r in par]),
            "no_structure_nodes": spread([r["no_structure_nodes"] for r in par])},
        "all_variants_including_policy_changes": {
            "n": len(allv),
            "encyclopedia_ARI": spread([r["encyclopedia_ARI"] for r in allv]),
            "no_structure_nodes": spread([r["no_structure_nodes"] for r in allv])},
        "escrow": {"encyclopedia_ARI": 0.8744, "no_structure_nodes": escrow_null_K,
                   "quantity_that_moves_it": "none"},
        "reading": ("the spread within the paraphrase group is caused by wording alone, since "
                    "nothing else differs and decoding is greedy"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
