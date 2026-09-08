"""E49: is the real-world loss the criterion, or the search that looks for it?

WHY. E47 and E48 decomposed the loss on the synthetic cover fixture and found the larger half is
search: the shipped moves, handed a good starting state, reach a description 6{,}539 bits shorter
than the same moves reach from scratch, and that difference is worth omega 0.347 against 0.686. The
question that decides what the paper claims is whether the same holds on real records, because the
scope proposition already explains MusicBrainz 20K and ReVerb45K and does not explain this.

THE THREE STATES, on each real benchmark, all scored by the same objective and the same metric:

  engine from scratch     the shipped protocol, which is what every number in the paper reports
  a truth-seeded state    one node per label, each cell given to the label-node that holds most of
                          the record's labels, then the shipped repair run to fixpoint. This is not
                          an optimum and is not claimed as one; it is a reachable state built from
                          the answer, so its description length is an upper bound on what the
                          objective would accept
  best of R orders        the same engine, R arrival orders, keeping the run with the SHORTEST
                          description. This consults no label at all, so it is a search improvement
                          a practitioner could actually run

WHAT EACH OUTCOME MEANS.
  If the truth-seeded state has a SHORTER description than the engine's, the criterion ranks a
  better answer above ours and the search is the limit. Future work is search.
  If the engine's is already shortest, the criterion prefers what we produce and the loss is the
  criterion's. Future work is the code.
  If best-of-R closes part of the gap with no labels, that is direct evidence a practitioner can
  recover some of it today.
"""
from __future__ import annotations
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

import numpy as np

from escrow.engine import EscrowGraph, Node, ValueBlock
from escrow.batch import BatchObjective
from escrow.provenance import stamped
from e25_multi_membership import omega_index

OUT = os.path.join(ROOT, "results", "e49_search_gap_on_real_data.json")
RESTARTS = 5


# ------------------------------------------------------------------ data ---- #
def load_wikidata():
    p = os.path.join(ROOT, "results", "wikidata_cover", "wikidata_people.json")
    d = json.load(open(p, encoding="utf-8"))
    recs = [v["record"] for v in d.values()]
    labels = [set(v["labels"]) for v in d.values()]
    return "wikidata_people", recs, labels


def load_wikipedia():
    recs, labels = [], []
    for cat in ("film", "person", "mountain"):
        p = os.path.join(ROOT, "results", f"wiki_cache.{cat}.json")
        for r in json.load(open(p, encoding="utf-8")):
            recs.append(r)
            labels.append({cat})
    order = list(range(len(recs)))
    random.Random(0).shuffle(order)
    return "wikipedia_infoboxes", [recs[i] for i in order], [labels[i] for i in order]


# ---------------------------------------------------------------- scoring --- #
def truth_matrix(labels):
    lab = sorted({x for s in labels for x in s})
    ix = {l: i for i, l in enumerate(lab)}
    M = np.zeros((len(labels), len(lab)), dtype=bool)
    for i, s in enumerate(labels):
        for l in s:
            M[i, ix[l]] = True
    return M


def pred_matrix(g, n):
    nids = sorted(g.nodes)
    M = np.zeros((n, max(len(nids), 1)), dtype=bool)
    for c, nid in enumerate(nids):
        for m in g.nodes[nid].members:
            M[m - 1, c] = True
    return M


def omega(Mt, Mp):
    return round(float(omega_index(Mt, Mp, T=None)[0]), 4)


# ------------------------------------------------------------------ states -- #
def run_engine(recs, cadence=100):
    g = EscrowGraph()
    b = BatchObjective(g).install()
    for i, r in enumerate(recs, 1):
        g.process(r)
        if i % cadence == 0:
            b.repair()
    b.repair(full=True)
    return g, b


def background_only(recs):
    g = EscrowGraph()
    b = BatchObjective(g).install()
    g._mint = lambda *a, **k: None
    for r in recs:
        g.process(r)
    return g, b


def install_truth_seed(g, labels):
    """One node per label. A record's cell goes to whichever of its labels holds the most records,
    so the assignment is deterministic and every cell is claimed once."""
    size = {}
    for s in labels:
        for l in s:
            size[l] = size.get(l, 0) + 1
    members = {}
    for rid, s in enumerate(labels, start=1):
        for l in s:
            members.setdefault(l, set()).add(rid)
    owner_of = [max(s, key=lambda l: (size[l], l)) if s else None for s in labels]
    for lab in sorted(members, key=lambda l: (-size[l], l)):
        g.next_id += 1
        w = Node(nid=g.next_id, t=len(members[lab]), birth_n=0)
        w.members = set(members[lab])
        for rid in w.members:
            if owner_of[rid - 1] != lab:
                continue
            for kid, vid in g.record_vals.get(rid, {}).items():
                if g.record_owner.get(rid, {}).get(kid, 0) != 0:
                    continue
                w.S.add(kid)
                w.pub.setdefault(kid, set()).add(rid)
                blk = w.blocks.get(kid)
                if blk is None:
                    blk = w.blocks[kid] = ValueBlock()
                blk.observe(vid)
                g.key_by_id[kid].background.unobserve(vid)
                g.record_owner.setdefault(rid, {})[kid] = w.nid
        # Presence is a property of MEMBERSHIP, not of who owns the value cell: the engine sets
        # pub[kid] to every member publishing kid (engine._mint), independent of ownership.
        # Corrected 2026-09-08.
        for kid in w.S:
            w.pub[kid] = {r for r in w.members if kid in g.record_keys.get(r, ())}
        for kid in w.S:
            w.p[kid] = len(w.pub.get(kid, ()))
            g.ix2.setdefault(kid, set()).add(w.nid)
            for vid in w.blocks[kid].counts:
                g.ix1.setdefault((kid, vid), set()).add(w.nid)
        g.nodes[w.nid] = w
        g.K += 1
    return g


def main():
    report = {"experiment": "E49 is the real-world loss the criterion or the search",
              "restarts": RESTARTS, "benchmarks": []}

    for loader in (load_wikidata, load_wikipedia):
        name, recs, labels = loader()
        n = len(recs)
        Mt = truth_matrix(labels)
        print(f"\n=== {name}: {n} records, {Mt.shape[1]} labels ===", flush=True)

        g, b = run_engine(recs)
        shipped = {"K": int(g.K), "L": round(b.total(), 1), "omega": omega(Mt, pred_matrix(g, n))}
        print(f"  engine from scratch     K={shipped['K']:>4} L={shipped['L']:>12.1f} "
              f"omega={shipped['omega']}", flush=True)

        g1, b1 = background_only(recs)
        install_truth_seed(g1, labels)
        seeded = {"K": int(g1.K), "L": round(b1.total(), 1), "omega": omega(Mt, pred_matrix(g1, n))}
        print(f"  truth-seeded            K={seeded['K']:>4} L={seeded['L']:>12.1f} "
              f"omega={seeded['omega']}", flush=True)
        b1.repair(full=True)
        seeded_rep = {"K": int(g1.K), "L": round(b1.total(), 1),
                      "omega": omega(Mt, pred_matrix(g1, n))}
        print(f"  truth-seeded + repair   K={seeded_rep['K']:>4} L={seeded_rep['L']:>12.1f} "
              f"omega={seeded_rep['omega']}", flush=True)

        best = None
        for r in range(RESTARTS):
            idx = list(range(n))
            random.Random(1000 + r).shuffle(idx)
            perm = [recs[i] for i in idx]
            gr, br = run_engine(perm)
            L = br.total()
            if best is None or L < best["L"]:
                Mp = np.zeros_like(pred_matrix(gr, n))
                nids = sorted(gr.nodes)
                Mp = np.zeros((n, max(len(nids), 1)), dtype=bool)
                for c, nid in enumerate(nids):
                    for m in gr.nodes[nid].members:
                        Mp[idx[m - 1], c] = True       # undo the permutation before scoring
                best = {"K": int(gr.K), "L": round(L, 1), "omega": omega(Mt, Mp), "order": r}
        print(f"  best of {RESTARTS} orders       K={best['K']:>4} L={best['L']:>12.1f} "
              f"omega={best['omega']}  (chosen by shortest description, no label)", flush=True)

        row = {"benchmark": name, "records": n, "labels": int(Mt.shape[1]),
               "engine_from_scratch": shipped, "truth_seeded": seeded,
               "truth_seeded_after_repair": seeded_rep, "best_of_orders": best,
               "search_gap_bits": round(shipped["L"] - seeded_rep["L"], 1),
               "restart_gain_bits": round(shipped["L"] - best["L"], 1),
               "restart_gain_omega": round(best["omega"] - shipped["omega"], 4)}
        report["benchmarks"].append(row)
        print(f"  -> search gap {row['search_gap_bits']:+.1f} bits, "
              f"restarts alone recover {row['restart_gain_bits']:+.1f} bits and "
              f"{row['restart_gain_omega']:+.4f} omega", flush=True)

    report["headline"] = {
        b["benchmark"]: {"search_gap_bits": b["search_gap_bits"],
                         "omega_shipped": b["engine_from_scratch"]["omega"],
                         "omega_truth_seeded_after_repair": b["truth_seeded_after_repair"]["omega"],
                         "omega_best_of_orders": b["best_of_orders"]["omega"]}
        for b in report["benchmarks"]}
    print("\n" + json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
