"""E36: under this code a hierarchy is never the cheapest description, and the reason is a redundancy.

WHAT E35 LEFT OPEN. E35 measured that the engine returns one level of a planted two-level tree and
that a diagnostic build which does return both levels pays MORE bits for them. That located the
problem in the objective rather than in the search, and stopped there. This file asks the question
properly: is the flat description actually cheaper at the optimum, or was the engine simply failing
to reach a good nested state?

THE TWO DESCRIPTIONS OF THE SAME TRUTH. Three families, three species in each. A family owns keys
every one of its species carries; a species owns keys of its own.

  FLAT    nine nodes. Node (f, s) supports its family's keys AND its species' keys, and holds the
          records of that species.
  NESTED  three family nodes, each supporting only its family's keys and holding every record of
          that family, plus nine species nodes supporting only their own keys.

Both describe every (record, key) cell exactly once, under whichever node owns that key, so this is
a like-for-like comparison of two valid codes for the same data and not a comparison of two answers.

WHAT THE ARITHMETIC SAYS BEFORE ANY RUN. The difference decomposes into two terms of different
orders.

  The membership columns. NESTED pays for the parent's column on top of the children's, and nothing
  else changes, so it pays an extra sum over families of L_col(n_f, n). That is Theta(n).

  The value blocks. NESTED codes each family key once over n/3 rows where FLAT codes it three times
  over n/9 rows each. The data term is identical, since the same cells are coded either way, and
  only the Krichevsky-Trofimov parametric part differs, which is Theta(log n) in NESTED's favour.

So the penalty grows linearly in the stream and the saving grows logarithmically. A hierarchy is not
merely hard to find under this code. It is never optimal, and it gets further from optimal with every
record that arrives. No improvement to the search can change that.

THE REDUNDANCY, AND THE REPAIR. The parent's members are the union of its children's members. The
decoder already has the children's columns, so transmitting the parent's column transmits what it can
already derive, and a code that transmits derivable information is not minimal. Charge the parent
only for the residual its children do not determine, which is empty when the children partition it,
and the Theta(n) penalty disappears. What is left is the parametric saving, so the hierarchy becomes
strictly cheaper at every n, by more as the stream grows.

This file computes both descriptions exactly, with the code's own L_col, L_supp, L_vblock and
structure cost, on a real generated stream with noise, and reports the decomposition and the effect
of the repair.

WHAT WOULD FALSIFY THE ACCOUNT. If FLAT and NESTED came out within the parametric term of each other
under the shipped code, the linear penalty would not be real and E35's result would need another
explanation. If the repaired code did not reverse the ordering, the redundancy would not be what is
carrying it.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.batch import L_vblock                                       # noqa: E402
from escrow.codes import L_col, L_supp, L_N, lg2                        # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e36_hierarchy_is_never_optimal.json")

FAMILIES, SPECIES = 3, 3
FAMILY_KEYS, SPECIES_KEYS = 3, 3
VALUES = 6
NOISE = 0.1
SIZES = (900, 3000, 9000, 27000)
SEEDS = (0, 1, 2)


def stream(n, seed):
    rng = random.Random(seed)
    recs, fam, sp = [], [], []
    for _ in range(n):
        f = rng.randrange(FAMILIES)
        s = rng.randrange(SPECIES)
        rec = {}
        for j in range(FAMILY_KEYS):
            of = rng.randrange(FAMILIES) if rng.random() < NOISE else f
            rec[f"f{f}k{j}"] = f"f{of}v{rng.randrange(VALUES)}"
        for j in range(SPECIES_KEYS):
            os_ = rng.randrange(SPECIES) if rng.random() < NOISE else s
            rec[f"f{f}s{s}k{j}"] = f"f{f}s{os_}v{rng.randrange(VALUES)}"
        recs.append(rec)
        fam.append(f)
        sp.append((f, s))
    return recs, fam, sp


def structure_cost(K):
    return L_N(K + 1) - (lg2(K + 1.0) if K > 1 else 0.0)


def node_cost(members, support, recs, n, A_n, inventory, charge_membership=True):
    """One node's contribution, term by term, exactly as BatchObjective.node_local computes it."""
    t = len(members)
    col = L_col(t, n) if charge_membership else 0.0
    sup = L_supp(len(support), A_n)
    presence, values = 0.0, 0.0
    for key in support:
        counts = Counter()
        present = 0
        for i in members:
            v = recs[i].get(key)
            if v is not None:
                present += 1
                counts[v] += 1
        presence += L_col(present, t)
        if counts:
            values += L_vblock(counts, math.log2(len(inventory[key]) + 1.0))
    return {"membership_column": col, "support": sup, "presence": presence, "values": values,
            "total": col + sup + presence + values}


def describe(recs, fam, sp, charge_parent=True):
    """The two descriptions, priced with the same functions."""
    n = len(recs)
    keys = sorted({k for r in recs for k in r})
    A_n = len(keys)
    inventory = {}
    for r in recs:
        for k, v in r.items():
            inventory.setdefault(k, set()).add(v)

    by_species = {}
    by_family = {}
    for i, (f, s) in enumerate(sp):
        by_species.setdefault((f, s), []).append(i)
        by_family.setdefault(f, []).append(i)

    fam_keys = {f: [f"f{f}k{j}" for j in range(FAMILY_KEYS)] for f in range(FAMILIES)}
    sp_keys = {(f, s): [f"f{f}s{s}k{j}" for j in range(SPECIES_KEYS)]
               for f in range(FAMILIES) for s in range(SPECIES)}

    flat = {"K": FAMILIES * SPECIES, "nodes": [], "structure": structure_cost(FAMILIES * SPECIES)}
    for (f, s), members in sorted(by_species.items()):
        flat["nodes"].append(node_cost(members, fam_keys[f] + sp_keys[(f, s)],
                                       recs, n, A_n, inventory))
    flat["total"] = flat["structure"] + sum(x["total"] for x in flat["nodes"])

    K = FAMILIES + FAMILIES * SPECIES
    nested = {"K": K, "nodes": [], "structure": structure_cost(K)}
    for f, members in sorted(by_family.items()):
        # the parent's column is derivable from its children's when they partition it
        nested["nodes"].append(node_cost(members, fam_keys[f], recs, n, A_n, inventory,
                                         charge_membership=charge_parent))
    for (f, s), members in sorted(by_species.items()):
        nested["nodes"].append(node_cost(members, sp_keys[(f, s)], recs, n, A_n, inventory))
    nested["total"] = nested["structure"] + sum(x["total"] for x in nested["nodes"])

    def part(d, field):
        return sum(x[field] for x in d["nodes"])

    return {
        "flat_total": round(flat["total"], 1), "nested_total": round(nested["total"], 1),
        "nested_minus_flat": round(nested["total"] - flat["total"], 1),
        "by_term": {
            "membership_column": round(part(nested, "membership_column")
                                       - part(flat, "membership_column"), 1),
            "support": round(part(nested, "support") - part(flat, "support"), 1),
            "presence": round(part(nested, "presence") - part(flat, "presence"), 1),
            "values": round(part(nested, "values") - part(flat, "values"), 1),
            "structure": round(nested["structure"] - flat["structure"], 1),
        },
    }


def main():
    report = {
        "experiment": "E36 a hierarchy is never optimal under this code",
        "question": ("E35 found the engine returns one level and that a build which returns both "
                     "pays more. Is the flat description actually cheaper AT THE OPTIMUM?"),
        "method": ("both descriptions of the same planted truth are priced with the code's own "
                   "L_col, L_supp, L_vblock and structure cost, on a real stream with noise"),
        "falsifier": ("flat and nested within the parametric term of each other under the shipped "
                      "code, or the repair failing to reverse the ordering"),
        "generator": {"families": FAMILIES, "species_per_family": SPECIES,
                      "family_keys": FAMILY_KEYS, "species_keys": SPECIES_KEYS,
                      "values_per_key": VALUES, "noise": NOISE},
        "rows": [],
    }
    print(f"{'n':>7} {'seed':>5} | {'shipped: nested - flat':>23} | "
          f"{'parent column derived':>22} | {'membership term':>16}")
    for n in SIZES:
        for seed in SEEDS:
            recs, fam, sp = stream(n, seed)
            shipped = describe(recs, fam, sp, charge_parent=True)
            repaired = describe(recs, fam, sp, charge_parent=False)
            row = {"n": n, "seed": seed,
                   "shipped": shipped, "parent_column_derived": repaired}
            report["rows"].append(row)
            print(f"{n:>7} {seed:>5} | {shipped['nested_minus_flat']:>+23.0f} | "
                  f"{repaired['nested_minus_flat']:>+22.0f} | "
                  f"{shipped['by_term']['membership_column']:>+16.0f}", flush=True)

    import statistics as st
    by_n = {}
    for r in report["rows"]:
        by_n.setdefault(r["n"], []).append(r)
    trend = [{"n": n,
              "shipped_penalty_for_nesting":
                  round(st.mean(x["shipped"]["nested_minus_flat"] for x in rows), 1),
              "repaired_gain_for_nesting":
                  round(st.mean(x["parent_column_derived"]["nested_minus_flat"] for x in rows), 1)}
             for n, rows in sorted(by_n.items())]
    report["trend"] = trend
    first, last = trend[0], trend[-1]
    report["headline"] = {
        "shipped_penalty_grows_with_n": bool(last["shipped_penalty_for_nesting"]
                                             > first["shipped_penalty_for_nesting"]),
        "shipped_penalty_first_last": [first["shipped_penalty_for_nesting"],
                                       last["shipped_penalty_for_nesting"]],
        "repaired_prefers_nesting_everywhere": bool(
            all(t["repaired_gain_for_nesting"] < 0 for t in trend)),
        "repaired_gain_first_last": [first["repaired_gain_for_nesting"],
                                     last["repaired_gain_for_nesting"]],
        "reading": ("Under the shipped code the nested description costs more than the flat one at "
                    "every size and the gap grows with the stream, because the parent is charged a "
                    "membership column the decoder could derive from its children. That is a "
                    "property of the code and not of the search, so no better search finds a tree. "
                    "Charging the parent only for what its children do not determine reverses the "
                    "ordering at every size, and the margin grows with the stream because what is "
                    "left is the parametric saving from coding a shared key once instead of once "
                    "per child."),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
