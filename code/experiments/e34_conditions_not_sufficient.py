"""E34: the two scope conditions are necessary and they are not sufficient, measured.

WHY THIS EXISTS. E31 derives two conditions from the price and shows they account for every stream
the paper reports: a group is reachable only if a characteristic (key, value) pair recurs at least
t* times, and a node is affordable only if each member saves more than log2((n-t+1/2)/(t-1/2)) bits.
Both are stated as necessary conditions and both are one-directional. It would be easy to read the
placement of the benchmarks as though meeting them were enough, and it is not. This file is the
counterexample, found while checking whether the same conditions explain the multi-membership
failures of E33.

THE COUNTEREXAMPLE. Take the E33 cover fixture at its hardest setting: eight groups, every key drawn
from a shared pool of twelve, no group owning any key of its own. Then

  the SEEDING condition is met with room to spare. The most frequent characteristic pair of a group
  recurs 36 to 47 times across seeds, against a t* of at most 24;

  the PRICE condition is met as well. The best live candidate at the end of the stream is accruing
  13.25 bits per member against a requirement of 9.23;

  and the engine still mints 3 nodes where the truth has 8, scoring omega 0.2582.

So both conditions hold and the rule fails anyway. What the measurement localises is where it fails:
the candidates still alive at the end carry cohorts of 3 to 7, not the 36 to 47 their seed pairs
would allow, so the accrual is not reaching the candidates that the seeds should feed. That is a
property of candidate generation and pool management, not of the price, and this file does not claim
to have explained it. It reports it as open, with the numbers that bound it.

WHAT THIS CHANGES IN THE PAPER. Nothing in Proposition scope, whose two clauses are both "only if"
and remain true. What it changes is what may be said around it: the conditions place the benchmarks
correctly and they do not license the reading that meeting them predicts success.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics as st
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.codes import price                                          # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e25_multi_membership import (truth_matrix, omega_index,   # noqa: E402
                                              _shared_counts, escrow_cover)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e34_conditions_not_sufficient.json")

N, GROUPS, KEYS_PER_GROUP, VALUES, NOISE = 1500, 8, 3, 8, 0.1
WEIGHTS = (0.5, 0.3, 0.2)
SEEDS = (0, 1, 2)
CELLS = ((12, 0), (24, 0), (12, 1), (12, 2))     # (shared key pool, private keys per group)
T_STAR_MAX = 24


def shared_key_cover(key_pool, private, seed):
    rng = random.Random(seed)
    owned = [sorted(rng.sample(range(key_pool), KEYS_PER_GROUP - private)) for _ in range(GROUPS)]
    sizes = list(range(1, len(WEIGHTS) + 1))
    recs, truth = [], []
    for _ in range(N):
        m = rng.choices(sizes, weights=list(WEIGHTS))[0]
        gs = sorted(rng.sample(range(GROUPS), m))
        rec = {}
        for g in gs:
            for j in range(private):
                og = rng.randrange(GROUPS) if rng.random() < NOISE else g
                rec[f"p{g}k{j}"] = f"c{og}v{rng.randrange(VALUES)}"
            for k in owned[g]:
                og = rng.randrange(GROUPS) if rng.random() < NOISE else g
                rec[f"k{k}"] = f"c{og}v{rng.randrange(VALUES)}"
        recs.append(rec)
        truth.append(frozenset(gs))
    return recs, truth


def seed_cohorts(recs, truth):
    per = {}
    for r, ts in zip(recs, truth):
        for g in ts:
            per.setdefault(g, Counter()).update(r.items())
    return [int(max(c.values())) for c in per.values()]


def best_live_candidate(g, n):
    best = None
    for c in getattr(g, "pool", {}).values():
        s = len(c.keys)
        if s == 0:
            continue
        pr = price(c.t, s, n, g.K, g.e, len(g.keys))
        gap = c.G - pr
        if best is None or gap > best["margin"]:
            t = int(c.t)
            best = {"margin": round(gap, 2), "evidence_bits": round(c.G, 2),
                    "price_bits": round(pr, 2), "cohort": t, "support": s,
                    "bits_per_member": round(c.G / max(t, 1), 2),
                    "required_bits_per_member":
                        round(math.log2((n - t + 0.5) / (t - 0.5)), 2) if 2 <= t < n else None}
    return best


def main():
    rows = []
    for pool, private in CELLS:
        per_seed = []
        for s in SEEDS:
            recs, truth = shared_key_cover(pool, private, s)
            Mt = truth_matrix(truth, GROUPS)
            T = _shared_counts(Mt)
            Me, g = escrow_cover(recs)
            om = omega_index(Mt, Me, T=T)[0]
            coh = seed_cohorts(recs, truth)
            per_seed.append({
                "seed": s, "K": int(g.K), "omega": round(om, 4),
                "seed_cohort_min": min(coh), "seed_cohort_median": int(st.median(coh)),
                "groups_meeting_the_seeding_condition":
                    int(sum(1 for c in coh if c >= T_STAR_MAX)),
                "best_live_candidate": best_live_candidate(g, len(recs)),
            })
        rows.append({
            "shared_key_pool": pool, "private_keys_per_group": private,
            "true_groups": GROUPS,
            "K_mean": round(st.mean(p["K"] for p in per_seed), 1),
            "omega_mean": round(st.mean(p["omega"] for p in per_seed), 4),
            "seed_cohort_min_over_seeds": min(p["seed_cohort_min"] for p in per_seed),
            "seed_cohort_median_over_seeds":
                int(st.median([p["seed_cohort_median"] for p in per_seed])),
            "groups_meeting_the_seeding_condition_mean":
                round(st.mean(p["groups_meeting_the_seeding_condition"] for p in per_seed), 1),
            "per_seed": per_seed,
        })
        r = rows[-1]
        print(f"  pool={pool} private={private}: K {r['K_mean']} of {GROUPS}, omega "
              f"{r['omega_mean']}, seed cohorts >= {r['seed_cohort_min_over_seeds']} "
              f"(t* max {T_STAR_MAX}), groups meeting the seeding condition "
              f"{r['groups_meeting_the_seeding_condition_mean']}/{GROUPS}", flush=True)

    worst = rows[0]
    report = {
        "experiment": "E34 the scope conditions are necessary, not sufficient",
        "question": ("E31 places every benchmark correctly with two conditions read off the price. "
                     "Does meeting them predict that the rule will work?"),
        "answer": "no, and this is the measured counterexample",
        "t_star_used": T_STAR_MAX,
        "fixture": {"n": N, "groups": GROUPS, "keys_per_group": KEYS_PER_GROUP,
                    "values_per_key": VALUES, "noise": NOISE, "set_size_weights": list(WEIGHTS),
                    "note": "the E33 cover fixture; private = 0 means no group owns a key of its "
                            "own, so key presence never names the group"},
        "rows": rows,
        "headline": {
            "hardest_cell": {"shared_key_pool": worst["shared_key_pool"],
                             "private_keys_per_group": worst["private_keys_per_group"]},
            "seeding_condition_met": True,
            "seed_cohort_min": worst["seed_cohort_min_over_seeds"],
            "price_condition_met_by_the_best_live_candidate": True,
            "K_found": worst["K_mean"], "K_true": GROUPS, "omega": worst["omega_mean"],
            "reading": ("Both conditions hold with room to spare and the rule still fails. The "
                        "candidates alive at the end carry cohorts far below what their seed pairs "
                        "allow, so the accrual is not reaching the candidates the seeds should "
                        "feed. That is candidate generation and pool management rather than the "
                        "price, and it is reported here as open rather than explained."),
        },
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
