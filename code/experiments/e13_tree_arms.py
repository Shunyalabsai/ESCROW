"""E13 comparison arms: VFDT and EFDT on the PLANTED streams, not only the null.

Why this file exists. The E8 tree arms (e8_tree_arms.py) run the Very Fast and
Extremely Fast Decision Trees on pure noise, where every split they make is
false structure. That measures one half of the trade: at a loose delta the
trees invent structure on noise, and at river's default delta of 1e-7 they stay
silent. It does not measure the other half, which the paper asserts: that the
silence of a strict delta is bought with detection delay when real structure
does arrive. This file measures the delay, on the same planted streams as the
creation-bias grid, at the same two deltas and five seeds.

Supervision conversion. The trees need a label and these records have none, so
the conversion is the one in e8_tree_arms.py, applied to E13's records: one
feature per key the record carries, named by that key and valued by that key's
value, and the target is the value of the lexicographically first key, which is
dropped from the features. The trees therefore see the record's key names and
its values, which is everything ESCROW sees minus the one cell being predicted.
Once the noise is on, this target carries real signal: a foreign key or a
foreign value ties one group's alphabet to another's, so a split on a key name
predicts the target better than the majority class and the first split index is
a detection delay and not a false positive. As the noise rises further the
signal is destroyed exactly as it is for ESCROW.

A positional variant of this conversion, one feature per slot carrying the key
name and one carrying the value, was tried and discarded: it hands the tree two
features of near equal merit, so the Hoeffding test ties and neither delta
splits inside 10000 records, which measures the tie breaker and not the delay.

The value target has one blind spot, and it is the clean stream. At noise 0 a
key belongs to exactly one group, so a feature is present only on that group's
records, and inside that group the target value is uniform: the split merit is
zero and neither delta splits, so no delay can be read off the clean cell. The
second target closes that hole. Target "group" hands the tree the planted group
label itself, which is supervision ESCROW never receives, and asks it to
predict that label from the whole record. Real structure is then guaranteed to
be present and the first split index is a clean detection delay. Both targets
are run on the same streams and reported side by side, so the arm is not
resting on one conversion.

Recorded per cell: the final node count (river counts the root, so a silent
tree reads 1 and not 0) and the index of the first record at which the tree
holds more than one node, which is the detection delay. A first-split index of
null means the tree never split within the stream.

river's default delta is 1e-7 for both classes (checked on river 0.26.1), so
the delta = 0.01 arm is the loose setting and not the library default.

Requires river, which is not a dependency of anything else here:
  python3 -m venv .venv && .venv/bin/pip install river
Run: .venv/bin/python e13_tree_arms.py [n_workers]
"""
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import river
    from river import tree
except ImportError:
    sys.exit("river is required: pip install river")

from escrow.provenance import stamped
from e13_creation_bias import planted_stream

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

K_STAR = 8                       # the headline row of the creation-bias grid
LENGTHS = {                      # per model, see RUNTIME below
    "vfdt": (2000, 30000),
    "efdt": (2000,),
}
NOISES = (0.0, 0.2, 0.4)         # clean, half broken, past the ESCROW breakdown
DELTAS = (1e-2, 1e-7)            # the "usual setting" quoted in the paper, and river's default
SEEDS = (11, 12, 13, 14, 15)     # the seeds of the creation-bias grid
GRACE = 200                      # as in e8_tree_arms.py

# RUNTIME: the two models get different stream lengths and the reason is measured,
# not assumed. EFDT reconsiders the split at every node on the path for every
# record, and on this fixture every key name is a nominal level, so one EFDT fit
# costs about 79 s at T = 2000 and did not finish inside twenty five minutes at
# T = 10000 on one core. VFDT costs a few seconds at T = 30000. Both models
# therefore run at T = 2000, which is where they are compared to each other and
# is also the shortest length in the E8 null table, and VFDT alone is extended to
# T = 30000, which is where the strict delta's delay becomes visible: on the group
# target the first split lands near 8400 records at delta = 0.01 and near 29200 at
# delta = 1e-7, so a short stream would report the strict arm as silent when it is
# in fact merely late, the exact confusion this file exists to remove.

MAKE = {
    "vfdt": lambda d: tree.HoeffdingTreeClassifier(delta=d, grace_period=GRACE),
    "efdt": lambda d: tree.ExtremelyFastDecisionTreeClassifier(delta=d, grace_period=GRACE),
}


def to_xy_value(rec, label):
    """One feature per key, named by the key; predict the first key's value from the rest."""
    items = sorted(rec.items())
    return {k: v for k, v in items[1:]}, items[0][1]


def to_xy_group(rec, label):
    """The whole record; predict the planted group label, or "noise" for a random record."""
    return dict(rec), ("noise" if label == -1 else f"g{label}")


TARGETS = {"value": to_xy_value, "group": to_xy_group}


def one_run(job):
    target, model, delta, noise, seed, T = job
    recs, truth = planted_stream(T, K_STAR, noise, noise, noise, seed)
    to_xy = TARGETS[target]
    m = MAKE[model](delta)
    first = None
    t0 = time.time()
    for i, rec in enumerate(recs):
        x, y = to_xy(rec, truth[i])
        m.learn_one(x, y)
        if first is None and m.n_nodes > 1:
            first = i + 1
    return {"target": target, "model": model, "delta": delta, "noise": noise, "seed": seed,
            "T": T, "n_nodes": int(m.n_nodes), "first_split_index": first,
            "seconds": round(time.time() - t0, 2)}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    jobs = [(t, m, d, n, s, T) for t in TARGETS for m in MAKE for d in DELTAS
            for n in NOISES for s in SEEDS for T in LENGTHS[m]]
    jobs.sort(key=lambda j: (j[1] != "efdt", -j[5]))   # the slow model first, longest first
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_run, jobs)):
            rows.append(r)
            print(f"[{i + 1}/{len(jobs)}] {r['target']} {r['model']} T={r['T']} "
                  f"delta={r['delta']} noise={r['noise']} seed={r['seed']}: "
                  f"nodes={r['n_nodes']} first={r['first_split_index']} {r['seconds']}s",
                  flush=True)
    wall = time.time() - t0

    cells = {}
    for target in TARGETS:
        for model in MAKE:
          for T in LENGTHS[model]:
            for delta in DELTAS:
                for noise in NOISES:
                    got = [r for r in rows if r["target"] == target and r["model"] == model
                           and r["delta"] == delta and r["noise"] == noise and r["T"] == T]
                    nodes = [r["n_nodes"] for r in got]
                    firsts = [r["first_split_index"] for r in got]
                    seen = [f for f in firsts if f is not None]
                    cells[f"target={target} {model} T={T} delta={delta} noise={noise}"] = {
                        "n_nodes": sorted(nodes),
                        "mean_nodes": round(sum(nodes) / len(nodes), 1),
                        "min_nodes": min(nodes), "max_nodes": max(nodes),
                        "first_split_index": firsts,
                        "mean_first_split": round(sum(seen) / len(seen), 1) if seen else None,
                        "min_first_split": min(seen) if seen else None,
                        "max_first_split": max(seen) if seen else None,
                        "seeds_that_never_split": len(firsts) - len(seen),
                        "mean_seconds": round(sum(r["seconds"] for r in got) / len(got), 2),
                    }

    def delay(target, model, T, noise):
        a = cells[f"target={target} {model} T={T} delta=0.01 noise={noise}"]["mean_first_split"]
        b = cells[f"target={target} {model} T={T} delta=1e-07 noise={noise}"]["mean_first_split"]
        if a is None or b is None:
            return None
        return round(b - a, 1)

    delays = {f"target={t} {m} T={T} noise={n}": delay(t, m, T, n)
              for t in TARGETS for m in MAKE for T in LENGTHS[m] for n in NOISES}
    measured = {k: v for k, v in delays.items() if v is not None}

    split_cells = sum(1 for c in cells.values() if c["mean_first_split"] is not None)
    reading = (
        f"{split_cells} of the {len(cells)} cells split at least once, so on planted structure "
        "the trees are not silent the way they are on the E8 null, and the price of a strict "
        "delta shows up as a delay in the first split index rather than as silence. Every cell "
        "where both deltas split gives that delay in records: "
        + ("; ".join(f"{k} {v}" for k, v in measured.items())
           if measured else "no cell split at both deltas, so no delay is measurable here")
        + ". Cells where the strict delta never split inside the stream are recorded with a null "
          "first split index and a seeds_that_never_split count, and they are the same statement "
          "in a stronger form. The node counts are the other half of the trade, since the loose "
          "delta ends with more nodes, which on the E8 null was pure false structure. Both files "
          "use the same generator, seeds and deltas, so the null table and this table are two "
          "halves of one trade-off and not two experiments."
    )

    out = {
        "design": (
            f"E13 planted streams, K* = {K_STAR}, lengths {LENGTHS}, noise driving value noise, key noise "
            "and the random record fraction together, as in e13_creation_bias.py. The trees "
            "predict the lexicographically first key's value from the rest of the record, one "
            "feature per remaining key named by that key, as in e8_tree_arms.py."
        ),
        "why": (
            "e8_tree_arms.py runs these trees on the null only, where every split is false. "
            "This file supplies the planted arm, which is where a strict delta pays detection "
            "delay."
        ),
        "river_version": river.__version__,
        "river_default_delta": 1e-7,
        "grid": {"K_star": K_STAR, "T_by_model": {k: list(v) for k, v in LENGTHS.items()},
                 "noise": list(NOISES), "delta": list(DELTAS),
                 "targets": list(TARGETS), "seeds": list(SEEDS), "grace_period": GRACE,
                 "runs": len(jobs)},
        "targets": {
            "value": "predict the lexicographically first key's value from the other key value "
                     "pairs, the e8_tree_arms.py conversion",
            "group": "predict the planted group label from the whole record, supervision ESCROW "
                     "never receives; random records carry the label noise",
        },
        "note_on_node_scale": (
            "river counts the root, so a tree that never splits reads 1 node while ESCROW's "
            "silence reads 0 minted nodes."
        ),
        "note_on_lengths": (
            "VFDT runs at T = 2000 and T = 30000; EFDT at T = 2000 only. Measured reason: one "
            "EFDT fit costs about 79 s at T = 2000 and did not finish inside twenty five minutes "
            "at T = 10000 on one core, because EFDT reconsiders every split at every node on the "
            "path for every record and every key name here is a nominal level. T = 2000 is the "
            "length at which the two models are compared; T = 30000 is where the strict delta's "
            "delay becomes visible for VFDT."
        ),
        "cells": cells,
        "detection_delay_records_strict_minus_loose": delays,
        "runtime_seconds": round(wall, 1),
        "workers": workers,
        "reading": reading,
    }
    stamped(out)                                  # the repo wide engine md5 and date stamp
    path = os.path.join(OUT, "e13_tree_arms.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwall {wall:.1f}s over {workers} workers")
    print(reading)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
