"""E70: can a level be priced, and does a two-level description compress better than a flat one?

THE GAP THIS ADDRESSES. Recursion recovers the planted tree exactly, ARI 0.984 to 0.995 against the
nine leaves, and the objective correctly accepts the real level at about -6,000 bits and refuses the
invented one at about +2,000. But the gate only ever decides whether to REPLACE a node with its
sub-nodes. It cannot say "keep both and link them", because the objective has no term for a two-level
description as one object.

Proposition on hierarchy explains why nobody wrote one: minting a parent never pays, because its
membership column is Theta(n) while the saving from factorising is only Theta(log n). That is true,
and it prices a **node**. A level is not one more vertex.

WHAT A LEVEL ACTUALLY IS, AND WHY IT SHOULD BE CHEAPER. E35 measured the redundancy: every species
node carries its family's keys as well as its own, so the family structure is written down three
times. A two-level description removes exactly that. A leaf stops carrying its parent's keys and
points at the parent instead, so:

  the parent pays once      its support, its presence columns, its value blocks
  each leaf pays for        only the keys its parent does not already explain, plus a pointer
  the pointer costs         log2(number of parents) bits per leaf, not per record

The membership column does not double, because a record is a member of its leaf and its parent
membership is implied by the pointer rather than transmitted again. That is the reparameterisation
`prop:hier` does not consider, and whether it pays is arithmetic rather than opinion.

WHAT IS MEASURED, on the planted two-level fixture where the truth is known:

  flat, families only      the shipped protocol's answer
  flat, leaves only        the same records described as nine groups with no parents
  two level                parents and leaves, leaves inheriting their parent's keys

All three scored by the same code on the same records, so the comparison is three descriptions of
one dataset and the shortest wins.

WHAT WOULD REFUTE IT. The two-level description costing more than the better flat one. Then the
hierarchy is a way of reading the answer and never a shorter description, `prop:hier` covers the case
after all, and the paper should say that a tree is output structure rather than model structure.
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.batch import BatchObjective, L_vblock
from escrow.codes import L_col, L_N, L_supp, lg2


def supp(size_T: int, A_n: int) -> float:
    """`L_supp` asserts a support of at least one key. A leaf whose parent already explains every
    key it carries has an EMPTY residual support, which is a real and important state here: it is a
    leaf that adds nothing, and the code has to be able to say so rather than crash. The empty case
    keeps the size code and drops the subset choice, since there is exactly one empty subset."""
    if size_T <= 0:
        return L_N(1)
    return L_supp(size_T, A_n)
from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped
from e35_nesting import two_level

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e70_pricing_a_level.json")
SEEDS = (0, 1, 2)


def group_code(recs, groups, n_total, key_universe, label=""):
    """Codelength of describing `recs` as the given groups, with no parents.

    One structure term for the group count, then per group: a membership column over the whole
    stream, a support drawn from the key universe, a presence column per supported key, and a value
    block per supported key. This is the same shape as `BatchObjective.node_local` and is written out
    here so the two-level variant can share it exactly and the two numbers stay comparable.
    """
    A_n = len(key_universe)
    bits = L_N(len(groups) + 1) - (lg2(len(groups) + 1.0) if len(groups) > 1 else 0.0)
    detail = collections.Counter()
    for members in groups.values():
        t = len(members)
        keys = collections.Counter()
        vals = collections.defaultdict(collections.Counter)
        for i in members:
            for k, v in recs[i].items():
                keys[k] += 1
                vals[k][v] += 1
        bits += L_col(t, n_total)
        detail["membership"] += L_col(t, n_total)
        bits += supp(len(keys), A_n)
        detail["support"] += supp(len(keys), A_n)
        for k, p in keys.items():
            bits += L_col(p, t)
            detail["presence"] += L_col(p, t)
            naming = math.log2(len(key_universe[k]) + 1.0)
            b = L_vblock(dict(vals[k]), naming)
            bits += b
            detail["values"] += b
    return bits, detail


def two_level_code(recs, parents, children, n_total, key_universe):
    """The same data described as parents with leaves under them.

    A leaf transmits only the keys its parent does not already support, and a pointer naming which
    parent it hangs from. The parent's keys are written once instead of once per leaf, which is the
    redundancy E35 measured.
    """
    A_n = len(key_universe)
    n_par = len(parents)
    bits = L_N(n_par + 1) - (lg2(n_par + 1.0) if n_par > 1 else 0.0)
    detail = collections.Counter()

    # WHICH KEYS BELONG TO THE PARENT, decided by bits rather than by assumption.
    #
    # The first version of this code gave the parent every key any of its members carried. That cost
    # 24,975 bits of parent presence against 271 for the flat description, because a key carried by
    # only one child is present on a third of the parent's members and a sparse presence column is
    # expensive, while the same key inside its own leaf is present on nearly all of them and a dense
    # column is nearly free.
    #
    # So a key is placed where it is cheaper. Held by the parent it costs one presence column over
    # the parent's members plus one value block; pushed to the leaves it costs a column and a block
    # in each leaf that carries it. Comparing those two numbers is what decides, and nothing is set.
    parent_keys = {}
    for pid, members in parents.items():
        mem = set(members)
        child_of = {c: [i for i in ms if i in mem] for c, ms in children.items() if c[0] == pid}
        keep = set()
        for k in {k for i in members for k in recs[i]}:
            at_parent = 0.0
            p_here = sum(1 for i in members if k in recs[i])
            at_parent += L_col(p_here, len(members))
            vals_p = collections.Counter(recs[i][k] for i in members if k in recs[i])
            naming = math.log2(len(key_universe[k]) + 1.0)
            at_parent += L_vblock(dict(vals_p), naming)
            at_leaves = 0.0
            for c, ms in child_of.items():
                p_c = sum(1 for i in ms if k in recs[i])
                if p_c == 0:
                    continue
                at_leaves += L_col(p_c, len(ms))
                vc = collections.Counter(recs[i][k] for i in ms if k in recs[i])
                at_leaves += L_vblock(dict(vc), naming)
            if at_parent <= at_leaves:
                keep.add(k)
        parent_keys[pid] = keep

    for pid, members in parents.items():
        t = len(members)
        keys = collections.Counter()
        vals = collections.defaultdict(collections.Counter)
        for i in members:
            for k, v in recs[i].items():
                if k not in parent_keys[pid]:
                    continue
                keys[k] += 1
                vals[k][v] += 1
        bits += L_col(t, n_total)
        detail["parent membership"] += L_col(t, n_total)
        bits += supp(len(keys), A_n)
        detail["parent support"] += supp(len(keys), A_n)
        for k, p in keys.items():
            bits += L_col(p, t)
            detail["parent presence"] += L_col(p, t)
            naming = math.log2(len(key_universe[k]) + 1.0)
            b = L_vblock(dict(vals[k]), naming)
            bits += b
            detail["parent values"] += b

    bits += L_N(len(children) + 1) - (lg2(len(children) + 1.0) if len(children) > 1 else 0.0)
    for (pid, cid), members in children.items():
        t = len(members)
        # the pointer: which parent, out of the parents that exist
        bits += math.log2(max(n_par, 1))
        detail["pointer"] += math.log2(max(n_par, 1))
        # membership inside the parent, not inside the whole stream: the saving that makes a level
        bits += L_col(t, len(parents[pid]))
        detail["leaf membership inside parent"] += L_col(t, len(parents[pid]))
        keys = collections.Counter()
        vals = collections.defaultdict(collections.Counter)
        for i in members:
            for k, v in recs[i].items():
                if k in parent_keys[pid]:
                    continue                       # the parent already explains this key
                keys[k] += 1
                vals[k][v] += 1
        bits += supp(len(keys), A_n)
        detail["leaf support"] += supp(len(keys), A_n)
        for k, p in keys.items():
            bits += L_col(p, t)
            detail["leaf presence"] += L_col(p, t)
            naming = math.log2(len(key_universe[k]) + 1.0)
            b = L_vblock(dict(vals[k]), naming)
            bits += b
            detail["leaf values"] += b
    return bits, detail


def main():
    report = {"experiment": "E70 pricing a level rather than a node",
              "construction": ("a leaf transmits only the keys its parent does not support, plus a "
                               "pointer naming the parent, and its membership column is taken inside "
                               "the parent rather than over the whole stream"),
              "per_seed": []}

    for seed in SEEDS:
        recs, fam, sp = two_level(seed)
        n = len(recs)
        key_universe = collections.defaultdict(set)
        for r in recs:
            for k, v in r.items():
                key_universe[k].add(v)

        fams = collections.defaultdict(list)
        leaves = collections.defaultdict(list)
        kids = collections.defaultdict(list)
        for i, (f, s) in enumerate(zip(fam, sp)):
            fams[f].append(i)
            leaves[(f, s)].append(i)
            kids[(f, s)].append(i)

        flat_fam, d1 = group_code(recs, fams, n, key_universe)
        flat_leaf, d2 = group_code(recs, leaves, n, key_universe)
        nested, d3 = two_level_code(recs, fams, kids, n, key_universe)

        best_flat = min(flat_fam, flat_leaf)
        row = {"seed": seed, "records": n,
               "flat_families_bits": round(flat_fam, 1),
               "flat_leaves_bits": round(flat_leaf, 1),
               "two_level_bits": round(nested, 1),
               "two_level_minus_best_flat": round(nested - best_flat, 1),
               "pays": nested < best_flat,
               "where_the_bits_go_two_level": {k: round(v, 1) for k, v in d3.items()},
               "where_the_bits_go_flat_leaves": {k: round(v, 1) for k, v in d2.items()}}
        report["per_seed"].append(row)
        print(f"\nseed {seed}: {n} records, 3 families x 3 species")
        print(f"   flat, families only   {flat_fam:>11.1f} bits")
        print(f"   flat, nine leaves     {flat_leaf:>11.1f} bits")
        print(f"   TWO LEVEL             {nested:>11.1f} bits   "
              f"{nested - best_flat:+.1f} against the better flat one   "
              f"{'PAYS' if nested < best_flat else 'does not pay'}")

    pays = sum(1 for r in report["per_seed"] if r["pays"])
    report["headline"] = {
        "seeds_where_the_two_level_code_is_shortest": pays, "of": len(SEEDS),
        "mean_saving_bits": round(
            sum(r["two_level_minus_best_flat"] for r in report["per_seed"]) / len(SEEDS), 1),
        "reading": ("if the two-level code is shorter, a level can be priced and the hierarchy is "
                    "model structure rather than a way of reading the output. If it is not, the "
                    "hierarchy proposition covers this case after all and the paper should say a "
                    "tree is output structure only"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
