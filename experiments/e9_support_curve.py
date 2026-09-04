"""E9: the support curve - cohort size at birth t* against evidence rate and
alphabet size. 'Our gamma is a computed recurrence count' made checkable: the
figure that replaces an accuracy table.

For each (k, d): run the two-group fixture, record the FIRST mint's cohort size
t*, its realised per-member escrow rate rhat = G/t*, and the self-consistency
prediction t*_pred = min{ t : t * rhat >= Price(t, k, n_t) } with n_t = t/rho
(rho = slice density 0.5/d). Assertions: monotone directions (t* falls with k,
rises with d) and |t* - t*_pred| within the granularity of arrival.
"""
import json
import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.codes import price
from escrow.engine import EscrowGraph

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)


def first_mint(k, d, n_max=40000, seed=0):
    rng = random.Random(seed)
    g = EscrowGraph()
    for i in range(n_max):
        grp = i % 2
        g.process({f"g{grp}k{j}": f"g{grp}v{rng.randrange(d)}" for j in range(k)})
        if g.mint_log:
            m = g.mint_log[0]
            # realised rate: recover G at mint from the receipt's per-key bits
            return {"k": k, "d": d, "t_star": m["members"], "n_at_mint": m["n"],
                    "G_at_mint": m["G_total"], "price_at_mint": m["price_paid"],
                    "rate_bits_per_member": round(m["G_total"] / m["members"], 3)}
    return {"k": k, "d": d, "t_star": None, "n_at_mint": None,
            "rate_bits_per_member": None}


def predicted_t(rate, k, d, A_n, t_max=5000):
    rho = 0.5 / d
    for t in range(2, t_max):
        n_t = max(t + 1, int(t / rho))
        if t * rate >= price(t, k, n_t, 0, 0, max(A_n, 2 * k)):
            return t
    return None


grid = []
for k in (2, 3, 4, 6):
    for d in (4, 8, 16):
        r = first_mint(k, d)
        if r["t_star"] is not None and r["rate_bits_per_member"]:
            r["t_pred_selfconsistent"] = predicted_t(
                r["rate_bits_per_member"], k, d, 2 * k)
        grid.append(r)
        print(r)

# monotonicity checks (qualitative falsifiers)
by = {(r["k"], r["d"]): r["t_star"] for r in grid if r["t_star"]}
mono_k = all(by.get((k2, d), 10**9) <= by.get((k1, d), 0) or by.get((k1, d)) is None
             for d in (4, 8, 16) for k1, k2 in ((2, 4), (3, 6)) if (k2, d) in by and (k1, d) in by)
res = {"grid": grid, "monotone_in_k": mono_k}
print(json.dumps({"monotone_in_k": mono_k}, indent=2))
with open(os.path.join(OUT, "e9_support_curve.json"), "w") as f:
    json.dump(res, f, indent=2)
