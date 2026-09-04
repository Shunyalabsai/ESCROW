"""E13: the creation-bias curve - the answer to Kearns, Mansour, Ng & Ron.

Fix a true latent structure of K* groups; sweep stream length T at several noise
rates; plot K_hat(T) against T.
  - Kearns failure signature: K_hat grows linearly in T (noise coded as structure).
  - Dual failure (SLIQ's verdict on QR/WP): K_hat plateaus below K*.
  - The claim: K_hat -> K* by the Wilks-vs-log-n gap, with no significance level.
The abstract's sentence "we do not claim it has no bias, and we measure its
creation bias directly" is unsupported without this file.
"""
import json
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

K_STAR = 8
KEYS_PER_GROUP = 3
D = 8


def stream(T, noise, seed):
    rng = random.Random(seed)
    for i in range(T):
        grp = rng.randrange(K_STAR)
        rec = {}
        for j in range(KEYS_PER_GROUP):
            if rng.random() < noise:
                # noise: a value from ANOTHER group's alphabet for this slot
                og = rng.randrange(K_STAR)
                rec[f"c{grp}k{j}"] = f"c{og}v{rng.randrange(D)}"
            else:
                rec[f"c{grp}k{j}"] = f"c{grp}v{rng.randrange(D)}"
        yield rec


res = {}
for noise in (0.0, 0.1, 0.2):
    row = {}
    for T in (1000, 3000, 10000, 30000):
        g = EscrowGraph()
        b = BatchObjective(g)
        for i, rec in enumerate(stream(T, noise, seed=7)):
            g.process(rec)
            if (i + 1) % 250 == 0:
                b.repair()
        b.repair()
        row[T] = g.K
        print(f"noise={noise} T={T}: K_hat={g.K} (K*={K_STAR})")
    res[str(noise)] = row

res["K_star"] = K_STAR
with open(os.path.join(OUT, "e13_creation_bias.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
