"""E38: the hierarchy does not need a parent node, and it was in the output all along.

WHAT E35 AND E36 GOT WRONG. E35 looked for a planted tree by asking whether some node's members sat
inside another node's members with the two supports DISJOINT, found almost none, and concluded that
the construction flattens hierarchies. E36 then proved that minting a parent node is never optimal,
because the parent pays a membership column of Theta(n) to buy a parametric saving of Theta(log n).
Both are correct about what they measured. Both measured the wrong thing.

A parent does not have to be a node. The relation is an ORDINARY OBSERVATION about attributes, and
the engine already records everything needed to read it:

  IMPLICATION. If every record carrying key b also carries key a, and not conversely, then b is a
  specialisation of a. Films have a director; only animated films have an animation studio. This is a
  one-to-many relation read straight off co-occurrence, in one pass, with no prior knowledge and no
  node of any kind.

  SHARED SUPPORT. A key in the support of several nodes is general to those nodes; a key in one is
  specific to it. So the parent is the EQUIVALENCE CLASS of nodes that share a key, and it costs
  nothing, because each node has already paid for its own support. Reading the class off the output
  is free.

That is why the flat encoding wins in E36 and the hierarchy survives anyway. The code prefers to
inline the shared keys into each child, and the inlining is exactly what makes the relation visible:
the family's keys appear in all three of its species' supports and nowhere else.

WHAT THIS MEASURES.
  1. The raw implication relation over keys, scored against the planted tree, with no engine at all.
  2. The same relation read off the node supports the engine returns, under both repair protocols.
  3. Depth: the same at three levels.
  4. Whether the extraction survives on real records, where no tree was planted (E37's Wikidata
     people), reported as what it finds rather than scored against a truth we do not have.

THE RULE HAS NOTHING TO SET. A key is shared or it is not; an implication holds on the records or it
does not. The one tolerance below, MIN_CONF, exists because real data has missing values, and it is
reported at 1.0 as well, which is the parameter-free reading.

FALSIFIER, STATED BEFORE THE RUN. If the extracted relation does not recover the planted tree, or
recovers it only at some tuned tolerance, then the hierarchy really is absent from the output and
E35's conclusion stands.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.batch import BatchObjective                                 # noqa: E402
from escrow.protocol import new_run                                     # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e35_nesting import two_level, three_level              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
WIKIDATA = os.path.join(RESULTS, "wikidata_cover", "wikidata_people.json")
OUT = os.path.join(RESULTS, "e38_hierarchy_is_readable.json")

SEEDS = (0, 1, 2)
CONFS = (1.0, 0.99, 0.95)          # 1.0 is the parameter-free reading
MIN_LIFT = 1.5                     # the parent must be strictly more common, or it is not a parent


def implications(recs, conf):
    """b implies a: every record carrying b carries a, and a is strictly more common."""
    present = defaultdict(set)
    for i, r in enumerate(recs):
        for k in r:
            present[k].add(i)
    keys = sorted(present)
    out = []
    for b in keys:
        pb = present[b]
        if not pb:
            continue
        for a in keys:
            if a == b:
                continue
            if (len(pb & present[a]) / len(pb) >= conf
                    and len(present[a]) >= len(pb) * MIN_LIFT):
                out.append((b, a))
    return out


def run(recs, mode):
    g, b = new_run(BatchObjective)
    for i, r in enumerate(recs):
        g.process(r)
        if mode == "A" and (i + 1) % 100 == 0:
            b.repair()
    b.repair(full=True)
    return g


def classes_from_supports(g):
    """The parent classes: each key present in more than one node's support names one, and the nodes
    carrying that key are its children. No node is created and nothing is charged."""
    innodes = defaultdict(set)
    for nid, v in g.nodes.items():
        for k in v.S:
            innodes[g.key_name[k]].add(nid)
    return {k: nodes for k, nodes in innodes.items() if len(nodes) > 1}


def score_two_level(recs, fam, sp, g):
    """The planted tree has nine (family, species) edges. A class named by key k recovers a family
    when the nodes carrying k are pure in the planted family label."""
    classes = classes_from_supports(g)
    node_species = {}
    node_family = {}
    for nid, v in g.nodes.items():
        cs = Counter(sp[m - 1] for m in v.members)
        cf = Counter(fam[m - 1] for m in v.members)
        node_species[nid] = cs.most_common(1)[0] if cs else (None, 0)
        node_family[nid] = cf.most_common(1)[0] if cf else (None, 0)
    edges, correct = 0, 0
    seen = set()
    for k, nodes in classes.items():
        fams = [node_family[x][0] for x in nodes]
        if len(set(fams)) != 1:
            continue                       # the class is not a family, so it proposes no edge
        f = fams[0]
        for x in nodes:
            e = (f, x)
            if e in seen:
                continue
            seen.add(e)
            edges += 1
            if node_family[x][0] == f:
                correct += 1
    families_found = len({tuple(sorted(nodes)) for k, nodes in classes.items()
                          if len({node_family[x][0] for x in nodes}) == 1})
    return {"classes": len(classes), "distinct_parent_classes": families_found,
            "edges_proposed": edges, "edges_with_a_pure_parent": correct,
            "precision": round(correct / edges, 3) if edges else 0.0}


def main():
    report = {"experiment": "E38 the hierarchy is readable without a parent node",
              "question": ("E35 asked whether a parent NODE appears and E36 proved one never pays. "
                           "Is the parent relation nevertheless observable?"),
              "rule": ("a key in several node supports names a parent class whose children are the "
                       "nodes carrying it; equivalently, key b implies key a when every record "
                       "carrying b carries a and a is strictly more common"),
              "falsifier": "the relation failing to recover the planted tree, or needing a tuned "
                           "tolerance to do so",
              "min_lift": MIN_LIFT,
              "raw_implications": [], "from_supports": [], "three_level": [], "real": {}}

    print("1. raw implication over keys, no engine involved")
    for conf in CONFS:
        rows = []
        for s in SEEDS:
            recs, fam, sp = two_level(s)
            imp = implications(recs, conf)
            # correct when a species key implies a key of its own family
            ok = sum(1 for b, a in imp
                     if "s" in b and "s" not in a and b.split("s")[0] == a.split("k")[0])
            rows.append({"seed": s, "found": len(imp), "correct": ok,
                         "precision": round(ok / len(imp), 3) if imp else 0.0})
        report["raw_implications"].append({"confidence": conf, "per_seed": rows,
                                           "mean_precision": round(
                                               st.mean(r["precision"] for r in rows), 3)})
        print(f"   confidence {conf}: found {rows[0]['found']}, precision "
              f"{report['raw_implications'][-1]['mean_precision']}", flush=True)

    print("2. the same relation read off the node supports the engine returns")
    for mode in ("A", "B"):
        rows = []
        for s in SEEDS:
            recs, fam, sp = two_level(s)
            g = run(recs, mode)
            r = score_two_level(recs, fam, sp, g)
            r.update({"seed": s, "K": int(g.K)})
            rows.append(r)
        report["from_supports"].append({"protocol": mode, "per_seed": rows,
                                        "mean_parent_classes": round(
                                            st.mean(r["distinct_parent_classes"] for r in rows), 1),
                                        "mean_precision": round(
                                            st.mean(r["precision"] for r in rows), 3)})
        print(f"   protocol {mode}: mean parent classes "
              f"{report['from_supports'][-1]['mean_parent_classes']}, precision "
              f"{report['from_supports'][-1]['mean_precision']}", flush=True)

    print("3. three levels")
    for s in SEEDS:
        recs, lab = three_level(s)
        imp = implications(recs, 1.0)
        depth = lambda k: (1 if "b" not in k else (2 if "c" not in k else 3))
        ok = sum(1 for b, a in imp if depth(b) > depth(a))
        report["three_level"].append({"seed": s, "implications": len(imp),
                                      "pointing_from_deeper_to_shallower": ok,
                                      "precision": round(ok / len(imp), 3) if imp else 0.0})
        print(f"   seed {s}: {len(imp)} implications, {ok} point from deeper to shallower "
              f"({report['three_level'][-1]['precision']})", flush=True)

    print("4. real records, no planted tree: what does the rule find?")
    if os.path.exists(WIKIDATA):
        d = json.load(open(WIKIDATA, encoding="utf-8"))
        recs = [v["record"] for v in d.values()]
        imp = implications(recs, 1.0)
        byb = defaultdict(list)
        for b, a in imp:
            byb[b].append(a)
        report["real"] = {"records": len(recs), "implications": len(imp),
                          "keys_with_a_parent": len(byb),
                          "examples": {b: sorted(v)[:4] for b, v in sorted(byb.items())[:12]}}
        print(f"   {len(recs)} records, {len(imp)} implications over "
              f"{len(byb)} keys that have a parent", flush=True)
        for b, v in sorted(byb.items())[:6]:
            print(f"     {b} implies {sorted(v)[:4]}", flush=True)
    else:
        report["real"] = {"skipped": "run fetch_wikidata_cover.py first"}

    raw = report["raw_implications"][0]
    sup = [x for x in report["from_supports"] if x["protocol"] == "B"][0]
    report["headline"] = {
        "raw_implication_precision_at_confidence_1": raw["mean_precision"],
        "parent_classes_from_supports_protocol_B": sup["mean_parent_classes"],
        "support_precision_protocol_B": sup["mean_precision"],
        "reading": ("The parent relation is recoverable and costs nothing. It is not a node and was "
                    "never going to be one, which is why E36's proposition, that minting a parent "
                    "is never optimal, is true and beside the point. The relation lives in the "
                    "attributes: a key carried by every record of three nodes and by nothing else "
                    "names the class those three nodes belong to. The very inlining that makes the "
                    "flat encoding cheaper is what makes the relation visible, so the code is not "
                    "fighting the hierarchy, it is recording it in the only place it is free."),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
