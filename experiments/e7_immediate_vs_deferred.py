"""E7: immediate vs deferred mint, run as a falsification, not a horse race.

Predictions (plan section 5.10; DERIVATIONS.md Lemma 2/3):
  P1  UNSEEDED immediate mint produces ZERO nodes; the realised per-attempt
      deficit tracks sum_a KL(qhat_a || Uniform_a) + C(u).
  P2  RECORD-SEEDED immediate mint's rate tracks the fraction of cells with
      qhat_a(x) < 1/m_a, and on a typo-injected stream the nodes it mints are
      typos that are never reused.
  P3  The DEFERRED mint (the shipped engine) mints the true structure and
      ignores the typos.
"""
import json
import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.codes import ValueBlock, kt, price
from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)

# ---- the stream: two latent groups + occasional typo records ---------------- #
def stream(n, k=3, d=8, typo_rate=0.01, seed=0):
    r = random.Random(seed)
    out = []
    for i in range(n):
        grp = i % 2
        rec = {f"g{grp}k{j}": f"g{grp}v{r.randrange(d)}" for j in range(k)}
        if r.random() < typo_rate:                       # one corrupted value
            key = r.choice(list(rec))
            rec[key] = f"TYPO{r.randrange(10**6)}"       # a value seen once, ever
        out.append(rec)
    return out

N = 6000
recs = stream(N)

# ---- Arm 1: unseeded immediate mint ---------------------------------------- #
# Mint iff the record pays for a fresh EMPTY node on arrival: the fresh node's
# blocks are empty, so its cost on every key is the uninformed one.
def arm_unseeded(records):
    g = EscrowGraph()
    mints, deficits = 0, []
    for rec in records:
        n = g.n + 1
        # fresh-node cost vs background cost, computed pre-update via one probe
        # of the engine's own primitives
        saving = 0.0
        for a, x in rec.items():
            ki = g.keys.get(a)
            if ki is None or g.n == 0:
                continue
            naming = math.log2(len(ki.inventory) + 1.0)
            bg = ki.background.cost(str(x) and ki.inventory.get(str(x)) is not None
                                     and ki.inventory[str(x)] or -1, naming) \
                 if False else ki.background.cost(ki.inventory.get(str(x), -1), naming)
            fresh = ValueBlock().cost(0, naming)         # empty block: naming only
            saving += bg - fresh
        pr = price(1, max(1, len(rec)), n, g.K, g.e, max(len(g.keys), len(rec)))
        if saving > pr:
            mints += 1
        else:
            deficits.append(pr - saving)
        g.process(rec)                                    # graph evolves normally
    return mints, deficits

m1, deficits = arm_unseeded(recs)

# ---- Arm 2: record-seeded immediate mint ----------------------------------- #
# The fresh node is seeded FROM the record (DP-means style): its cost on every
# key is ~0, so it mints exactly when the record's own background cost exceeds
# the price - i.e. on rare values.
def arm_seeded(records):
    g = EscrowGraph()
    minted = []                                          # (n, rec, was_typo)
    for rec in records:
        n = g.n + 1
        cost_bg = 0.0
        for a, x in rec.items():
            ki = g.keys.get(a)
            if ki is None or g.n == 0:
                continue
            naming = math.log2(len(ki.inventory) + 1.0)
            cost_bg += ki.background.cost(ki.inventory.get(str(x), -1), naming)
        pr = price(1, max(1, len(rec)), n, g.K, g.e, max(len(g.keys), len(rec)))
        if cost_bg > pr:
            minted.append((n, any(v.startswith("TYPO") for v in rec.values())))
        g.process(rec)
    return minted

seeded_mints = arm_seeded(recs)

# ---- Arm 3: the deferred mint (the engine as shipped) ----------------------- #
g = EscrowGraph()
b = BatchObjective(g)
for i, rec in enumerate(recs):
    g.process(rec)
    if (i + 1) % 100 == 0:
        b.repair()
b.repair()
typo_nodes = [m for m in g.mint_log
              if any(str(v).startswith("TYPO") for v in m["per_key_bits"])]

res = {
    "n": N,
    "unseeded": {"mints": m1, "mean_deficit_bits": round(sum(deficits) / len(deficits), 2),
                 "prediction": "0 mints; deficit = sum KL(q||U) + C(u)"},
    "seeded": {"mints": len(seeded_mints),
               "typo_fraction": round(sum(1 for _, t in seeded_mints if t)
                                      / max(1, len(seeded_mints)), 3),
               "prediction": "mints on rare values, i.e. on typos"},
    "deferred": {"final_K": g.K, "total_mints": len(g.mint_log),
                 "typo_justified_nodes": len(typo_nodes),
                 "supports": sorted(sorted(g.key_name[k] for k in v.S)
                                    for v in g.nodes.values()),
                 "prediction": "K=2, zero typo nodes"},
}
print(json.dumps(res, indent=2))
with open(os.path.join(OUT, "e7_immediate_vs_deferred.json"), "w") as f:
    json.dump(res, f, indent=2)
