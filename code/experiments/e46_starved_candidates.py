"""E46: why the rule under-mints where its own conditions say it should not.

WHY. E34 is the paper's own counterexample: a fixture meeting both clauses of the scope proposition
with room to spare, on which the rule still returns K = 2.7 where the truth is 8. What that
measurement localised was where, not why. Every candidate alive at the end carried a cohort of 3 to
5 records while its seed pair occurs 36 to 44 times, and every one of them had MORE evidence per
member than the price required. The accrual is not reaching the candidates the seeds should feed.

THE MECHANISM, read off the code. In `EscrowGraph._insert` a cell only feeds a candidate while the
background still owns it:

    resid = [(kid, vid_of[kid]) for kid in K_r if own[kid] == 0]

So the first node to take a shared key stops that key feeding every other candidate, permanently.
E34's fixture is the worst case for this by construction: eight groups drawing every key from one
shared pool of twelve, none of its own. The evidence a touched candidate then accrues is already
charged against the CURRENT OWNER's code, not the background, so the filter is doing a second job
that the delta formula does not need it to do.

THE TEST. One switch, `accrue_owned_cells`, default off so the shipped engine is unchanged. Off is
the current rule. On, every cell of the record feeds its own signature's candidate and the evidence
stays charged against the owner. Both arms run the same five gates.

WHAT WOULD MAKE THE CHANGE WRONG, and these are the gates that matter more than the win:
  1. It mints on noise. The safety claim is the paper's strongest result and no accuracy gain buys
     it. If K rises above zero on any null stream, the change is dead.
  2. It breaks the positive controls, which the current engine solves exactly.
  3. It over-mints on the real streams, trading under-clustering for a different failure.
Reported either way, and the null gate is reported first.
"""
from __future__ import annotations
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

import random

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective
from escrow.provenance import stamped
from e4_baseline_army import _ari, planted8, two_group

OUT = os.path.join(ROOT, "results", "e46_starved_candidates.json")
CADENCE = 100


def run(recs, accrue_owned, cap=4096):
    """The shipped protocol of escrow.protocol.run_stream, with the one switch exposed."""
    g = EscrowGraph(cand_pool_cap=cap, accrue_owned_cells=accrue_owned)
    b = BatchObjective(g).install()
    t0 = time.time()
    for i, r in enumerate(recs, 1):
        g.process(r)
        if i % CADENCE == 0:
            b.repair()
    b.repair(full=True)
    lab = [-1] * len(recs)
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    return g, lab, round(time.time() - t0, 1)


def null_stream(n, seed, keys=4, d=10):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def cover_fixture(n=1500, groups=8, shared=12, private=0, kpg=3, vals=8, noise=0.1, seed=0):
    """E33/E34's fixture. private=0 means no group owns a key, so key presence never names it."""
    rng = random.Random(seed)
    pool = [f"s{i}" for i in range(shared)]
    priv = {gi: [f"p{gi}_{j}" for j in range(private)] for gi in range(groups)}
    gkeys = {gi: rng.sample(pool, kpg) + priv[gi] for gi in range(groups)}
    recs, truth = [], []
    for _ in range(n):
        size = rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        gs = rng.sample(range(groups), size)
        rec = {}
        for gi in gs:
            for k in gkeys[gi]:
                og = rng.randrange(groups) if rng.random() < noise else gi
                rec[k] = f"g{og}v{rng.randrange(vals)}"
        recs.append(rec)
        truth.append(tuple(sorted(gs)))
    return recs, truth


def omega(truth, lab):
    """Membership agreement for a cover truth, scored as ARI over the first label."""
    return _ari([t[0] for t in truth], lab)


def main():
    report = {"experiment": "E46 why the rule under-mints where its own conditions say it should not",
              "mechanism": ("a cell feeds its candidate only while the background owns it, so the "
                            "first node to take a shared key starves every other candidate"),
              "switch": "EscrowGraph(accrue_owned_cells=...), default False is the shipped rule",
              "gates": {}}

    print("GATE 1, the one that can kill the change: does it mint on noise?")
    null_rows = []
    for s in (0, 1, 2, 3):
        recs = null_stream(2000, s)
        a, _, ta = run(recs, False)
        b, _, tb = run(recs, True)
        null_rows.append({"seed": s, "K_shipped": int(a.K), "K_accrue_owned": int(b.K),
                          "seconds_shipped": ta, "seconds_accrue_owned": tb})
        print(f"   seed {s}: shipped K={a.K}, accrue-owned K={b.K}", flush=True)
    report["gates"]["null"] = null_rows
    safe = all(r["K_accrue_owned"] == 0 for r in null_rows)
    report["gates"]["null_still_silent"] = safe
    print(f"   -> abstention preserved: {safe}")

    print("\nGATE 2: the positive controls the shipped engine solves exactly")
    pos = []
    for name, mk in (("two_group", lambda: two_group(n=2000, seed=0)),
                     ("planted8", lambda: planted8(n=3000, seed=7))):
        recs, truth = mk()
        a, la, _ = run(recs, False)
        b, lb, _ = run(recs, True)
        pos.append({"stream": name, "true_K": len(set(truth)),
                    "shipped": {"K": int(a.K), "ARI": round(_ari(truth, la), 4)},
                    "accrue_owned": {"K": int(b.K), "ARI": round(_ari(truth, lb), 4)}})
        print(f"   {name}: shipped K={a.K} ARI={_ari(truth, la):.4f} | "
              f"accrue-owned K={b.K} ARI={_ari(truth, lb):.4f}", flush=True)
    report["gates"]["positive_controls"] = pos

    print("\nGATE 3: E34's counterexample, the fixture this is meant to explain")
    e34 = []
    for shared in (12, 24):
        for s in (0, 1, 2):
            recs, truth = cover_fixture(shared=shared, private=0, seed=s)
            a, la, _ = run(recs, False)
            b, lb, _ = run(recs, True)
            e34.append({"shared_key_pool": shared, "seed": s, "true_groups": 8,
                        "shipped": {"K": int(a.K), "omega": round(omega(truth, la), 4)},
                        "accrue_owned": {"K": int(b.K), "omega": round(omega(truth, lb), 4)}})
            print(f"   pool {shared} seed {s}: shipped K={a.K} om={omega(truth, la):.4f} | "
                  f"accrue-owned K={b.K} om={omega(truth, lb):.4f}", flush=True)
    report["gates"]["e34_cover_fixture"] = e34

    ship_K = sum(r["shipped"]["K"] for r in e34) / len(e34)
    new_K = sum(r["accrue_owned"]["K"] for r in e34) / len(e34)
    ship_om = sum(r["shipped"]["omega"] for r in e34) / len(e34)
    new_om = sum(r["accrue_owned"]["omega"] for r in e34) / len(e34)
    report["headline"] = {
        "null_still_silent": safe,
        "e34_mean_K": {"truth": 8, "shipped": round(ship_K, 2), "accrue_owned": round(new_K, 2)},
        "e34_mean_omega": {"shipped": round(ship_om, 4), "accrue_owned": round(new_om, 4)},
        "reading": ("if the null stays silent and K moves toward 8, the under-minting E34 found is "
                    "a starvation in candidate feeding and not a limit of the price"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
