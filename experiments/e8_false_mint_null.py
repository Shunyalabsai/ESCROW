"""E8: false-mint behaviour under a synthetic null.

(i)  The Ville bound, measured on THIS implementation's escrow accumulator:
     20,000 simulated null candidates, P(sup_t G_t >= b) vs 2^-b.
(ii) The plateau: mean final K on iid null streams as stream length grows
     2k -> 20k. Theorem 1 predicts E[#spurious] <= 1, flat in T.
     (The VFDT/EFDT arms - node counts growing in T at fixed delta - are added
     when the river dependency is installed; this file records our side.)
"""
import json
import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.codes import ValueBlock, kt
from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

# ---- (i) Ville: null candidate trajectories over one categorical key -------- #
def ville(runs=20000, T=200, d=10, seed=0):
    rng = random.Random(seed)
    sups = []
    for _ in range(runs):
        bg = ValueBlock(); cand = ValueBlock()
        # background warmed on 200 null draws first (the incumbent has history)
        for _ in range(200):
            bg.observe(rng.randrange(d))
        G, sup = 0.0, 0.0
        naming = math.log2(d + 1.0)
        for _ in range(T):
            x = rng.randrange(d)
            G += bg.cost(x, naming) - cand.cost(x, naming)
            cand.observe(x); bg.observe(x)
            sup = max(sup, G)
        sups.append(sup)
    out = {}
    for b in (1, 2, 4, 8, 12):
        emp = sum(1 for s in sups if s >= b) / len(sups)
        out[b] = {"empirical": emp, "bound": 2.0 ** -b}
        assert emp <= 2.0 ** -b + 3e-3, (b, emp)
    return out

# ---- (ii) the plateau ------------------------------------------------------- #
def plateau(lengths=(2000, 5000, 10000, 20000), seeds=12, d=10, keys=4):
    res = {}
    for T in lengths:
        ks = []
        for s in range(seeds):
            rng = random.Random(1000 + s)
            g = EscrowGraph(); b = BatchObjective(g)
            for i in range(T):
                g.process({f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)})
                if (i + 1) % 200 == 0:
                    b.repair()
            b.repair()
            ks.append(g.K)
        res[T] = {"mean_K": sum(ks) / len(ks), "max_K": max(ks), "all": ks}
    return res

v = ville()
p = plateau()
res = {"ville": v, "plateau": p}
print(json.dumps(res, indent=2))
with open(os.path.join(OUT, "e8_false_mint_null.json"), "w") as f:
    json.dump(res, f, indent=2)
