"""E5: order dependence. The same records in 20 random orders, ESCROW under the one protocol.

Reports, per stream, the adjusted Rand index against the true labels and the node count K over
20 permutations: mean, standard deviation, minimum, maximum. This is the measurement the paper
owes for a one-pass method: the objective is order-invariant, the greedy sequence is not, and the
spread below is how much that costs. Streams are the E4 ones: two-group (2,000 records), the
eight-group noisy stream (3,000), and the 320 Wikipedia infoboxes. Pure Python, about a minute.
Writes results/e5_order_dependence.json.
"""
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from escrow.protocol import run_stream, describe as protocol_describe        # noqa: E402
from escrow.provenance import stamped
from experiments.e4_baseline_army import two_group, planted8, wikipedia, _ari  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
PERMS = 20


def labels(g, n):
    lab = [-1] * n
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    return lab


def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else OUT
    data = {"twogroup": two_group(), "planted8": planted8(), "wikipedia": wikipedia(cache_dir)}
    report = {"protocol": protocol_describe(), "permutations": PERMS, "streams": {}}
    for name, (recs, truth) in data.items():
        aris, ks, secs = [], [], []
        for p in range(PERMS):
            order = list(range(len(recs)))
            random.Random(1000 + p).shuffle(order)
            r = [recs[i] for i in order]
            t = [truth[i] for i in order]
            t0 = time.time()
            g, _ = run_stream(r)
            secs.append(time.time() - t0)
            aris.append(round(_ari(t, labels(g, len(r))), 4))
            ks.append(g.K)
        row = {"n": len(recs), "true_K": len(set(truth)),
               "ARI": {"mean": round(statistics.mean(aris), 4), "sd": round(statistics.pstdev(aris), 4),
                       "min": min(aris), "max": max(aris), "all": aris},
               "K": {"mean": round(statistics.mean(ks), 2), "sd": round(statistics.pstdev(ks), 2),
                     "min": min(ks), "max": max(ks), "all": ks},
               "seconds_per_run": round(statistics.mean(secs), 2)}
        report["streams"][name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "n"}, default=str)[:300])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e5_order_dependence.json"), "w") as f:
        json.dump(stamped(report), f, indent=2)


if __name__ == "__main__":
    main()
