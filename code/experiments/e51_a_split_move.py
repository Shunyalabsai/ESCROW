"""E51: how much of the search gap would a split move recover?

WHY. E47 and E48 localise the loss on the cover fixture: the planted truth is a shorter description
than what the engine returns, and the larger half of the gap belongs to the search rather than the
criterion. The mechanism is in the move set. The repair pass offers merge, delete and reassign and no
split, and a fresh mint is offered to an existing node at birth, so after minting the node count can
only fall and an early merge cannot be undone. That makes "a version 2 owes a split move" the
obvious next sentence, and an obvious sentence is worth a number.

WHAT THIS IS AND IS NOT. This is not a shipped feature and it is not an incremental delta. It is an
offline check of whether an improving split EXISTS, evaluated exactly: propose a bipartition of a
node's members, rebuild the whole state from the resulting allocation, and score it with the shipped
`BatchObjective.total()`. Because the state is rebuilt rather than patched, no incremental
bookkeeping can be wrong; the cost is that it is far too slow to ship, which is the point. A version
2 needs the exact delta. This says whether that work is worth doing.

THE PROPOSAL. For each node, its members are described by the cells they publish on the node's own
support, and a two-way k-means on that matrix proposes the split. The proposal is a heuristic and is
declared as one; what is exact is the acceptance test, which is the shipped objective on the rebuilt
state. A split is accepted only when it strictly lowers L_batch, and the pass repeats to fixpoint.

WHAT WOULD MAKE THE ANSWER "NO". If no split lowers L_batch, or if the ones that do recover little of
the gap, then the missing move is not what costs us and the search failure is elsewhere. Reported
either way, with the bits recovered and the agreement alongside, because a move that lowers the
description and lowers agreement would be a finding about the objective, not about the search.
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
from sklearn.cluster import KMeans

from escrow.engine import EscrowGraph, Node, ValueBlock
from escrow.batch import BatchObjective
from escrow.provenance import stamped
from e47_does_the_objective_prefer_the_truth import cover_fixture, background_only, engine_state
from e48_search_gap import omega_index, pred_sets

OUT = os.path.join(ROOT, "results", "e51_a_split_move.json")
MIN_SPLIT = 8            # a node smaller than this is not worth proposing a split for, declared


def allocation_of(g, n):
    """record id -> set of node labels, and the per-cell owner as (rid, kid) -> label."""
    alloc = {r: set() for r in range(1, n + 1)}
    for v in g.nodes.values():
        for m in v.members:
            alloc[m].add(v.nid)
    owner = {}
    for rid, d in g.record_owner.items():
        for kid, w in d.items():
            if w:
                owner[(rid, kid)] = w
    return alloc, owner


def build_state(recs, alloc, owner):
    """Rebuild a graph from scratch for the given allocation, so nothing is patched.

    A cell goes to the node its recorded owner names, and to no node at all when the engine left it
    in the background. Claiming background cells for the largest label was a bug: it made the
    rebuild of the engine's own allocation cost 2{,}400 bits more than the engine's state, which is
    larger than the effects being measured. Verified: with this rule the rebuild reproduces the
    engine's description length exactly.
    """
    g, b = background_only(recs)
    size = {}
    for s_ in alloc.values():
        for l in s_:
            size[l] = size.get(l, 0) + 1
    node_of = {}
    for lab in sorted(size, key=lambda l: (-size[l], str(l))):
        members = {r for r, s_ in alloc.items() if lab in s_}
        if not members:
            continue
        g.next_id += 1
        w = Node(nid=g.next_id, t=len(members), birth_n=0)
        w.members = set(members)
        node_of[lab] = w
    for lab, w in node_of.items():
        for rid in w.members:
            for kid, vid in g.record_vals.get(rid, {}).items():
                if owner.get((rid, kid)) != lab:
                    continue                      # the engine gave this cell to someone else
                w.S.add(kid)
                blk = w.blocks.get(kid)
                if blk is None:
                    blk = w.blocks[kid] = ValueBlock()
                blk.observe(vid)
                g.key_by_id[kid].background.unobserve(vid)
                g.record_owner.setdefault(rid, {})[kid] = w.nid
    for lab, w in list(node_of.items()):
        if not w.S:
            continue                              # a node that codes nothing is not a node
        # Presence is a property of MEMBERSHIP, not of who owns the value cell: the engine sets
        # pub[kid] to every member publishing kid, independent of ownership (engine._mint, and
        # measured to hold in the engine's final state: 30 of 30 node-key pairs).
        for kid in w.S:
            w.pub[kid] = {r for r in w.members if kid in g.record_keys.get(r, ())}
            w.p[kid] = len(w.pub[kid])
            g.ix2.setdefault(kid, set()).add(w.nid)
            for vid in w.blocks[kid].counts:
                g.ix1.setdefault((kid, vid), set()).add(w.nid)
        g.nodes[w.nid] = w
        g.K += 1
    return g, b


def propose_split(g, v, seed=0):
    """Two-way k-means over the node's members described by the cells they publish on its support.
    A heuristic proposal; the acceptance test below is exact."""
    mem = sorted(v.members)
    if len(mem) < MIN_SPLIT:
        return None
    kids = sorted(v.S)
    if not kids:
        return None
    cols = {}
    for r in mem:
        for kid in kids:
            vid = g.record_vals.get(r, {}).get(kid)
            if vid is not None:
                cols.setdefault((kid, vid), len(cols))
    if len(cols) < 2:
        return None
    X = np.zeros((len(mem), len(cols)), dtype=np.float32)
    for i, r in enumerate(mem):
        for kid in kids:
            vid = g.record_vals.get(r, {}).get(kid)
            j = cols.get((kid, vid))
            if j is not None:
                X[i, j] = 1.0
    lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(X)
    a = {mem[i] for i in range(len(mem)) if lab[i] == 0}
    b = set(mem) - a
    if not a or not b:
        return None
    return a, b


def split_to_fixpoint(recs, g0, b0, n, max_rounds=12):
    alloc, owner = allocation_of(g0, n)
    g, b = build_state(recs, alloc, owner)
    best = b.total()
    accepted = 0
    for _ in range(max_rounds):
        improved = False
        # build_state renumbers nodes, so map each rebuilt node back to the label it carries
        label_of = {}
        for lab in {l for s_ in alloc.values() for l in s_}:
            mem = {r for r, s_ in alloc.items() if lab in s_}
            for x in g.nodes.values():
                if x.members == mem:
                    label_of[x.nid] = lab
                    break
        for v in sorted(g.nodes.values(), key=lambda x: -x.t):
            if v.nid not in label_of:
                continue
            pr = propose_split(g, v)
            if pr is None:
                continue
            left, right = pr
            lab = label_of[v.nid]
            tag_l, tag_r = (lab, "L"), (lab, "R")
            trial = {r: set(x) for r, x in alloc.items()}
            for r in list(left) + list(right):
                if lab in trial[r]:
                    trial[r].discard(lab)
                    trial[r].add(tag_l if r in left else tag_r)
            # The cells this node owned must follow their record to the side it went to, or the
            # split would hand them back to the background and be scored as a deletion.
            trial_owner = dict(owner)
            for (rid, kid), w_ in owner.items():
                if w_ == lab:
                    trial_owner[(rid, kid)] = tag_l if rid in left else tag_r
            g2, b2 = build_state(recs, trial, trial_owner)
            L2 = b2.total()
            if L2 < best - 1e-9:
                best, alloc, owner, g, b = L2, trial, trial_owner, g2, b2
                accepted += 1
                improved = True
                break
        if not improved:
            break
    return g, b, best, accepted


def main():
    report = {"experiment": "E51 how much of the search gap a split move would recover",
              "proposal": "two-way k-means on the node's members over its own support, a heuristic",
              "acceptance": "exact: the state is rebuilt and scored with BatchObjective.total()",
              "rows": []}
    print(f"{'seed':>4} | {'engine':>18} | {'engine + split':>18} | {'truth':>18} | recovered")
    print(f"{'':>4} | {'K':>2} {'L':>8} {'om':>5} | {'K':>2} {'L':>8} {'om':>5} | "
          f"{'K':>2} {'L':>8} {'om':>5} |")
    for seed in (0, 1, 2):
        recs, truth, gen, _ = cover_fixture(shared=12, seed=seed)
        n = len(recs)
        g0, b0 = engine_state(recs)
        L0 = b0.total()
        om0 = omega_index(truth, pred_sets(g0, n), n)

        gS, bS, LS, acc = split_to_fixpoint(recs, g0, b0, n)
        omS = omega_index(truth, pred_sets(gS, n), n)

        from e47_does_the_objective_prefer_the_truth import install_oracle
        gT, bT = background_only(recs)
        install_oracle(gT, truth, gen)
        LT, omT, KT = bT.total(), omega_index(truth, pred_sets(gT, n), n), gT.K

        gap = L0 - LT
        rec = (L0 - LS) / gap if gap > 0 else 0.0
        report["rows"].append({
            "seed": seed,
            "engine": {"K": int(g0.K), "L": round(L0, 1), "omega": round(om0, 4)},
            "engine_plus_split": {"K": int(gS.K), "L": round(LS, 1), "omega": round(omS, 4),
                                  "splits_accepted": acc},
            "truth": {"K": int(KT), "L": round(LT, 1), "omega": round(omT, 4)},
            "bits_recovered": round(L0 - LS, 1),
            "share_of_the_gap_to_the_truth": round(rec, 3)})
        print(f"{seed:>4} | {g0.K:>2} {L0:>8.0f} {om0:>5.3f} | {gS.K:>2} {LS:>8.0f} {omS:>5.3f} | "
              f"{KT:>2} {LT:>8.0f} {omT:>5.3f} | {L0-LS:+.0f} bits, {rec:.0%} of the gap",
              flush=True)

    rows = report["rows"]
    report["headline"] = {
        "mean_bits_recovered": round(sum(r["bits_recovered"] for r in rows) / len(rows), 1),
        "mean_share_of_the_gap": round(
            sum(r["share_of_the_gap_to_the_truth"] for r in rows) / len(rows), 3),
        "omega_before": round(sum(r["engine"]["omega"] for r in rows) / len(rows), 4),
        "omega_after": round(sum(r["engine_plus_split"]["omega"] for r in rows) / len(rows), 4),
        "reading": ("a move the shipped repair does not have, evaluated exactly on the shipped "
                    "objective, and how much of the measured search gap it closes"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
