"""E9: the support curve - cohort size at birth t* against evidence rate and
alphabet size. 'Our gamma is a computed recurrence count' made checkable: the
figure that replaces an accuracy table.

For each (k, d): run the two-group fixture, record the FIRST mint's cohort size
t*, its realised per-member escrow rate rhat = G/t*, and the self-consistency
prediction t*_pred = min{ t : t * rhat >= Price(t, k, n_t) } with n_t = t/rho
(rho = slice density 0.5/d). The one directional claim, asserted and used by the
paper, is that t* falls as k grows; it holds on seed 0 and on the five-seed
means. Two claims an earlier version of this docstring made are false and are
withdrawn: t* does NOT rise with d (five-seed means at k=2 are 16.6, 23.6, 16.6
for d = 4, 8, 16, and at k=3 they are 5.8, 6.2, 5.6), and t* is not within the
granularity of arrival of t*_pred at k=2, where the self-consistent prediction
overshoots the measurement by a factor of 2 to 3.5 (16.6 measured against 49.8
predicted at d=4). The prediction tracks within about 1.5x for k >= 3.

Every cell now runs over seeds 0 to 4 (audit Block 13: no variance anywhere).
The existing single-seed keys are seed 0 and are unchanged; the per-seed lists
and their mean, min and max sit beside them, next to the closed-form prediction.
"""
import json
import math
import random
import statistics
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.codes import price
from escrow.protocol import new_run, REPAIR_EVERY, describe as protocol_describe
from escrow.provenance import stamped

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)


def first_mint(k, d, n_max=40000, seed=0):
    """The one run protocol, driven record by record so the stream can stop at
    the first mint. Repair before the first mint has no node to move, so the
    quantity measured is the same as the loop this replaced."""
    rng = random.Random(seed)
    g, b = new_run()
    for i in range(n_max):
        grp = i % 2
        g.process({f"g{grp}k{j}": f"g{grp}v{rng.randrange(d)}" for j in range(k)})
        if g.mint_log:
            m = g.mint_log[0]
            # realised rate: recover G at mint from the receipt's per-key bits
            return {"k": k, "d": d, "t_star": m["members"], "n_at_mint": m["n"],
                    "G_at_mint": m["G_total"], "price_at_mint": m["price_paid"],
                    "rate_bits_per_member": round(m["G_total"] / m["members"], 3)}
        if (i + 1) % REPAIR_EVERY == 0:
            b.repair()
    return {"k": k, "d": d, "t_star": None, "n_at_mint": None,
            "rate_bits_per_member": None}


def predicted_t(rate, k, d, A_n, t_max=5000):
    rho = 0.5 / d
    for t in range(2, t_max):
        n_t = max(t + 1, int(t / rho))
        if t * rate >= price(t, k, n_t, 0, 0, max(A_n, 2 * k)):
            return t
    return None


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(statistics.fmean(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}


grid = []
for k in (2, 3, 4, 6):
    for d in (4, 8, 16):
        per_seed = []
        for s in SEEDS:
            r = first_mint(k, d, seed=s)
            if r["t_star"] is not None and r["rate_bits_per_member"]:
                r["t_pred_selfconsistent"] = predicted_t(
                    r["rate_bits_per_member"], k, d, 2 * k)
            r["seed"] = s
            per_seed.append(r)
        row = dict(per_seed[0])                       # seed 0 keeps the old keys
        row.pop("seed", None)
        row["seeds"] = {
            "seeds": list(SEEDS),
            "per_seed": [{"seed": r["seed"], "t_star": r["t_star"],
                          "n_at_mint": r["n_at_mint"],
                          "rate_bits_per_member": r["rate_bits_per_member"],
                          "t_pred_selfconsistent": r.get("t_pred_selfconsistent")}
                         for r in per_seed],
            "t_star": spread(r["t_star"] for r in per_seed),
            "n_at_mint": spread(r["n_at_mint"] for r in per_seed),
            "rate_bits_per_member": spread(r["rate_bits_per_member"] for r in per_seed),
            "t_pred_selfconsistent": spread(r.get("t_pred_selfconsistent")
                                            for r in per_seed),
        }
        grid.append(row)
        print(k, d, "t_star", [r["t_star"] for r in per_seed],
              "pred", [r.get("t_pred_selfconsistent") for r in per_seed])

# monotonicity checks (qualitative falsifiers)
by = {(r["k"], r["d"]): r["t_star"] for r in grid if r["t_star"]}
mono_k = all(by.get((k2, d), 10**9) <= by.get((k1, d), 0) or by.get((k1, d)) is None
             for d in (4, 8, 16) for k1, k2 in ((2, 4), (3, 6)) if (k2, d) in by and (k1, d) in by)
by_mean = {(r["k"], r["d"]): r["seeds"]["t_star"]["mean"] for r in grid
           if r["seeds"]["t_star"]}
mono_k_mean = all(by_mean[(k2, d)] <= by_mean[(k1, d)]
                  for d in (4, 8, 16) for k1, k2 in ((2, 4), (3, 6))
                  if (k2, d) in by_mean and (k1, d) in by_mean)
res = {"grid": grid, "monotone_in_k": mono_k,
       "seeds": list(SEEDS),
       "monotone_in_k_on_seed_means": mono_k_mean,
       "protocol": protocol_describe()}
res["withdrawn_claims"] = {
    "t_star_rises_with_d": "false on five-seed means: k=2 gives 16.6, 23.6, 16.6 and k=3 gives "
                           "5.8, 6.2, 5.6 for d = 4, 8, 16",
    "prediction_within_granularity_of_arrival": "false at k=2, where the self-consistent prediction "
                                                "overshoots t* by 2 to 3.5 times; within about 1.5 "
                                                "times for k >= 3",
    "note": "the paper claims only that t* falls as k grows, which holds on seed 0 and on the "
            "five-seed means",
}
print(json.dumps({"monotone_in_k": mono_k,
                  "monotone_in_k_on_seed_means": mono_k_mean}, indent=2))
with open(os.path.join(OUT, "e9_support_curve.json"), "w") as f:
    json.dump(stamped(res), f, indent=2)
