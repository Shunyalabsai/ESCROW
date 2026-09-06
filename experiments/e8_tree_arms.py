"""E8 comparison arms: VFDT and EFDT node counts on the identical null stream.

The paper's claim: our expected spurious node count is bounded by one over the
whole stream (measured zero), flat in stream length. The Hoeffding-gate trees
control false splits per test at a significance level delta and then test
unboundedly often, so their structure grows with stream length on pure noise.
This file measures that growth on the same null streams as e8_false_mint_null.

Supervision conversion (the trees need a label, we have none): the target is
the value of the first key, predicted from the remaining keys. On an iid null
stream there is no signal, so every split is false structure.

Requires: river (pip install river). Run where cores are plentiful.
"""
import json
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.provenance import stamped  # noqa: E402

try:
    from river import tree
except ImportError:
    sys.exit("river is required: pip install river")

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

D = 10
KEYS = 4
LENGTHS = (2000, 5000, 10000, 20000)
DELTAS = (1e-2, 1e-7)
SEEDS = 5


def null_stream(T, seed):
    rng = random.Random(seed)
    for _ in range(T):
        vals = [rng.randrange(D) for _ in range(KEYS)]
        x = {f"k{j}": str(vals[j]) for j in range(1, KEYS)}
        y = str(vals[0])
        yield x, y


def n_nodes(model):
    for attr in ("n_nodes",):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    s = getattr(model, "summary", None)
    if isinstance(s, dict) and "n_nodes" in s:
        return int(s["n_nodes"])
    return None


def run(make, name):
    res = {}
    for delta in DELTAS:
        for T in LENGTHS:
            counts = []
            for seed in range(SEEDS):
                m = make(delta)
                for x, y in null_stream(T, 1000 + seed):
                    m.learn_one(x, y)
                counts.append(n_nodes(m))
            res[f"delta={delta} T={T}"] = {
                "mean_nodes": sum(counts) / len(counts), "all": counts}
            print(name, f"delta={delta} T={T}: mean nodes = {sum(counts)/len(counts):.1f}")
    return res


out = {
    "design": "predict key k0 from k1..k3 on iid uniform null; every split is false structure",
    "vfdt": run(lambda d: tree.HoeffdingTreeClassifier(delta=d, grace_period=200), "VFDT"),
    "efdt": run(lambda d: tree.ExtremelyFastDecisionTreeClassifier(delta=d, grace_period=200), "EFDT"),
    "ours_reference": "results/e8_false_mint_null.json: mean K = 0.0 at every length",
}
with open(os.path.join(OUT, "e8_tree_arms.json"), "w") as f:
    json.dump(stamped(out), f, indent=2)
print("written: results/e8_tree_arms.json")
