"""E54: can the same counting decide that two key names are one key?

THE DECISION. E20 scored us on key identity and we returned one true pair against 2,105 false,
because version 1 ships no operator that prices two keys as one. E45 then showed the benchmark that
scored us records string normalisation rather than the decision we care about: whether two different
names carry one role. This asks whether the objective already contains the answer.

THE FIXTURE, AND WHAT IT PLANTS. A stream of records over four roles. One role is written under two
different names on disjoint records, `shade` on some and `colour` on others, drawn from the same
value distribution: that is the pair a correct operator must merge, and no string rule can, because
the names share no characters. Against it sit three controls that a merge must leave alone:

  same name, different role   `size` on garments in {S,M,L} and `size` on screens in inches,
                              which a string rule merges and which are two keys
  different name, different role  `colour` against `weight`, disjoint vocabularies
  same role, but co-occurring  `input_tray` and `output_tray`, same vocabulary, but a record
                              carries both, which is direct evidence they are two keys

The last is the one E45 found AutoPKG's own declared merges failing, so it is planted deliberately
rather than hoped for.

WHAT WOULD REFUTE THE CLAIM. If the objective merges any control, or fails to merge the planted
synonym pair, then the description length does not contain this decision and version 2 needs
something other than counting. The number to read is the bit margin on each pair, not just the
verdict, because a margin near zero on the planted pair would mean the criterion is right by luck.
"""
from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

from escrow.batch import BatchObjective
from escrow.keymerge import key_merge_delta, propose_key_merges
from escrow.protocol import run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e54_key_merge_by_value_distribution.json")

N_PER_ROLE = 180
SEEDS = (0, 1, 2)

GARMENT_SIZE = ["S", "M", "L", "XL"]
SCREEN_SIZE = ["13 inch", "15 inch", "17 inch", "24 inch"]
COLOURS = ["red", "blue", "green", "black", "white"]
WEIGHTS = ["250 g", "500 g", "1 kg", "2 kg"]
TRAYS = ["A4", "A3", "Letter", "Legal"]


def build_stream(n_per_role, seed):
    """Four roles. The colour role is written under two names on disjoint records."""
    rng = random.Random(seed)
    recs = []
    for i in range(n_per_role):
        # the planted synonym: one role, two names, never on the same record
        name = "shade" if i % 2 == 0 else "colour"
        recs.append({"garment_type": rng.choice(["shirt", "dress", "coat"]),
                     "fabric": rng.choice(["cotton", "silk", "wool"]),
                     name: rng.choice(COLOURS),
                     "size": rng.choice(GARMENT_SIZE)})
    for _ in range(n_per_role):
        # `size` again, same string, a different role entirely
        recs.append({"panel": rng.choice(["IPS", "OLED", "TN"]),
                     "refresh": rng.choice(["60 Hz", "120 Hz", "144 Hz"]),
                     "size": rng.choice(SCREEN_SIZE)})
    for _ in range(n_per_role // 2):
        # weight, to sit against colour as a different name for a different role
        recs.append({"grocery_type": rng.choice(["rice", "flour", "sugar"]),
                     "organic": rng.choice(["yes", "no"]),
                     "weight": rng.choice(WEIGHTS)})
    for _ in range(n_per_role // 2):
        # two trays on one record: same vocabulary, and provably two keys
        recs.append({"printer_type": rng.choice(["laser", "inkjet"]),
                     "input_tray": rng.choice(TRAYS),
                     "output_tray": rng.choice(TRAYS)})
    rng.shuffle(recs)
    return recs


PROBES = [
    ("shade", "colour", True, "one role written under two names, the decision that is open"),
    ("colour", "weight", False, "different names, different roles, disjoint vocabularies"),
    ("shade", "weight", False, "the same, from the other synonym"),
    ("input_tray", "output_tray", False, "same vocabulary, but a record carries both"),
    ("fabric", "garment_type", False, "two roles on the same records, both garment facets"),
]


def kid_of(g, name):
    ki = g.keys.get(name)
    return None if ki is None else ki.kid


def main():
    report = {"experiment": "E54 can the objective decide that two key names are one key",
              "fixture": {"records_per_role": N_PER_ROLE, "seeds": list(SEEDS),
                          "planted_synonym": ["shade", "colour"],
                          "controls": [p[0] + " vs " + p[1] for p in PROBES[1:]]},
              "per_seed": []}

    for seed in SEEDS:
        recs = build_stream(N_PER_ROLE, seed)
        g, b = run_stream(recs)
        row = {"seed": seed, "records": len(recs), "K": int(g.K),
               "distinct_keys": len(g.keys), "probes": []}
        print(f"\nseed {seed}: {len(recs)} records, K = {g.K}, {len(g.keys)} distinct keys")
        for a, bb, should_merge, why in PROBES:
            ka, kb = kid_of(g, a), kid_of(g, bb)
            if ka is None or kb is None:
                row["probes"].append({"a": a, "b": bb, "error": "key not in the stream"})
                continue
            d = key_merge_delta(g, BatchObjective, ka, kb)
            blocked = d is None
            merged = (not blocked) and d < 0
            ok = (merged == should_merge)
            row["probes"].append({"a": a, "b": bb, "should_merge": should_merge, "why": why,
                                  "blocked_by_cooccurrence": blocked,
                                  "delta_bits": None if blocked else round(d, 2),
                                  "merged": merged, "correct": ok})
            verdict = "blocked, a record carries both" if blocked else f"{d:+9.1f} bits"
            print(f"   {a:>12} + {bb:<12} want {'merge ' if should_merge else 'keep  '} "
                  f"got {'merge' if merged else 'keep ':<6} {verdict:>28}  {'ok' if ok else 'WRONG'}")
        row["all_probes_correct"] = all(p.get("correct") for p in row["probes"] if "correct" in p)
        report["per_seed"].append(row)

    # and what it proposes with nothing planted: every candidate pair in the stream, priced
    recs = build_stream(N_PER_ROLE, SEEDS[0])
    g, b = run_stream(recs)
    sweep = propose_key_merges(g, BatchObjective)
    report["unprompted_sweep_on_seed_0"] = {
        "candidates_priced": sweep["candidates_priced"],
        "cooccurring_pairs_excluded": sweep["cooccurring_pairs_excluded"],
        "accepted": sweep["accepted"],
        "closest_rejected": sweep["all"][len(sweep["accepted"]):len(sweep["accepted"]) + 5]}
    print(f"\nunprompted sweep on seed 0: {sweep['candidates_priced']} pairs priced, "
          f"{len(sweep['accepted'])} accepted")
    for r in sweep["accepted"]:
        print(f"   MERGE {r['a']!r} + {r['b']!r}  {r['delta_bits']:+.1f} bits")

    report["headline"] = {
        "seeds_with_every_probe_correct": sum(1 for r in report["per_seed"]
                                              if r["all_probes_correct"]),
        "of": len(SEEDS),
        "planted_pair_margin_bits": [next(p["delta_bits"] for p in r["probes"]
                                          if p.get("a") == "shade" and p.get("b") == "colour")
                                     for r in report["per_seed"]],
        "unprompted_accepts": [f"{r['a']}+{r['b']}"
                               for r in report["unprompted_sweep_on_seed_0"]["accepted"]],
        "reading": ("a merge is taken exactly when the description gets shorter, and the two key "
                    "strings are never compared. The planted pair shares no characters, so no string "
                    "rule can reach it; the co-occurring pair shares a vocabulary, so every "
                    "distributional rule without the structural check would take it"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
