"""E62: how much evidence does a key need before a merge can be decided at all?

WHAT THIS IS FOR. E61 found the merge operator accepting 866 of 5,662 key pairs on the encyclopedia
stream, and a sampled sweep put every false merge on a singleton key: 29.7 percent of pairs merge
when the rarer key appears once, 0.0 percent when it appears five times or more. That is not a defect
to patch. It is the same shape as Proposition 1: a decision has to be affordable before it can be
made, and the price says which decisions those are before any data is seen. This measures where the
line is, so it can be stated rather than chosen.

THE PREDICTION, MADE BEFORE THE RUN. Two keys merge when the description gets shorter. Against the
merge sits the value evidence, which for disjoint vocabularies is about one bit per observation and
so grows like `n_a + n_b`. For the merge sits the model saving: one entry leaves the key inventory,
which shrinks every node's support charge and folds two background presence columns into one. That
saving barely moves with `n`. So the crossover should sit where

    n_a + n_b  is about equal to the model saving in bits

which makes the scope condition a statement about record counts, with a closed form, and not a
minimum somebody picks. If instead the crossover drifts with the stream length or the schema size in
some way the saving does not explain, it is a calibrated quantity and has to be reported as one.

WHAT IS SWEPT. Records per key, and distinct values per key, because a key seen once necessarily has
one value and the two axes are confounded on real data. Both keys always have disjoint vocabularies,
so the truth is always "do not merge" and every merge is a false one.

WHAT WOULD REFUTE IT. A crossover that moves with the schema size in a way the measured saving does
not predict; or no crossover at all, which would mean the operator cannot be scoped and needs a
different fix.
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.batch import BatchObjective
from escrow.keymerge import key_merge_delta
from escrow.protocol import run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e62_the_merge_scope_condition.json")

BULK = 400
FREQS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
VALUES_PER_KEY = (1, 2, 4, 8)
SCHEMA_SIZES = (6, 20, 60)          # how many other keys the stream carries
SEEDS = (0, 1, 2)


def stream(rng, freq, n_values, schema_size, n_bulk=BULK):
    """Two probe keys on disjoint records with disjoint vocabularies, inside a stream whose
    schema has `schema_size` other keys, so the inventory saving can be varied independently."""
    va = [f"a{i}" for i in range(n_values)]
    vb = [f"b{i}" for i in range(n_values)]
    filler = [f"f{i}" for i in range(schema_size)]
    recs = []
    for _ in range(n_bulk):
        r = {"kind": rng.choice(["shirt", "dress", "coat"]),
             "fabric": rng.choice(["cotton", "silk", "wool"])}
        for f in rng.sample(filler, min(3, len(filler))):
            r[f] = rng.choice(["x", "y", "z"])
        recs.append(r)
    for _ in range(freq):
        recs.append({"kind": rng.choice(["shirt", "dress", "coat"]),
                     "probe_a": rng.choice(va)})
    for _ in range(freq):
        recs.append({"kind": rng.choice(["shirt", "dress", "coat"]),
                     "probe_b": rng.choice(vb)})
    rng.shuffle(recs)
    return recs


def margin(recs):
    g, _ = run_stream(recs)
    ka, kb = g.keys.get("probe_a"), g.keys.get("probe_b")
    if ka is None or kb is None:
        return None, None
    return key_merge_delta(g, BatchObjective, ka.kid, kb.kid), len(g.keys)


def main():
    report = {"experiment": "E62 the scope condition for the key merge",
              "prediction": ("the crossover sits where n_a + n_b is about the model saving in bits, "
                             "so it is a record count with a closed form and not a chosen minimum"),
              "fixture": {"bulk_records": BULK, "seeds": list(SEEDS),
                          "vocabularies": "always disjoint, so every merge is a false one"},
              "sweep": []}

    for schema in SCHEMA_SIZES:
        print(f"\nschema with {schema} filler keys")
        print(f"    {'values/key':>11} " + " ".join(f"{f:>6}" for f in FREQS))
        for nv in VALUES_PER_KEY:
            cells = []
            for f in FREQS:
                if nv > max(1, f):
                    cells.append(None)
                    continue
                ds = []
                for s in SEEDS:
                    d, nkeys = margin(stream(random.Random(7000 + s), f, nv, schema))
                    if d is not None:
                        ds.append(d)
                if not ds:
                    cells.append(None)
                    continue
                mean = sum(ds) / len(ds)
                merges = sum(1 for x in ds if x < 0)
                cells.append({"freq": f, "values_per_key": nv, "schema": schema,
                              "mean_margin_bits": round(mean, 1),
                              "false_merges": merges, "of": len(ds)})
            report["sweep"] += [c for c in cells if c]
            row = " ".join((f"{c['mean_margin_bits']:>6.0f}" if c else "     .") for c in cells)
            flag = "".join("!" if c and c["false_merges"] else " " for c in cells)
            print(f"    {nv:>11} {row}")
            print(f"    {'':>11} " + " ".join(f"{'MERGE' if c and c['false_merges'] else '':>6}"
                                              for c in cells))

    bad = [c for c in report["sweep"] if c["false_merges"]]
    good = [c for c in report["sweep"] if not c["false_merges"]]
    by_schema = {}
    for schema in SCHEMA_SIZES:
        b = [c["freq"] for c in bad if c["schema"] == schema]
        g_ = [c["freq"] for c in good if c["schema"] == schema]
        by_schema[schema] = {"merges_up_to_freq": max(b) if b else None,
                             "safe_from_freq": min([x for x in g_ if not b or x > max(b)],
                                                   default=None)}
    report["headline"] = {
        "crossover_by_schema_size": by_schema,
        "total_cells": len(report["sweep"]),
        "cells_with_a_false_merge": len(bad),
        "reading": ("if the crossover frequency is stable across schema sizes the scope condition is "
                    "a property of the price; if it grows with the schema, the inventory saving is "
                    "what drives it and the condition has to be stated in terms of that saving"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
