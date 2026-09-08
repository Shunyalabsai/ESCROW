"""E18, cost side. What does a price-side seed naming charge cost the mint?

The charge does not touch the accumulator, only the level it must clear, so its whole
cost is measured on the positive controls: how many nodes still mint, and how good the
partition still is, as the charge grows.

Three things are measured.
  1. The two principled codes at their natural size: "av" (log2|A_n| + log2 d_a, the
     form Theorem 1(iii) writes) and "ln" (Rissanen's universal code on the two intern
     ranks, the open-alphabet form), alone and with the tight naming charge.
  2. A sweep over the uniform slack Gamma of Theorem 1(iii), which is the cost curve:
     it says how many bits the mint can absorb before it stops happening, and can be
     read against the Gamma the null exceedance actually needs (e18_seed_charge_null).
  3. The positive controls the paper reports: the two synthetic streams and Wikipedia
     under the canonical order, the same three over the 20 arrival orders of E5, the
     deferred arm of E7, and the Wikipedia demonstration.

Writes results/e18_charge_cost.json.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow import codes as C                                            # noqa: E402
from escrow.protocol import run_stream, describe as protocol_describe    # noqa: E402
from experiments.e4_baseline_army import (two_group, planted8,           # noqa: E402
                                          wikipedia, _ari)
from experiments.e7_immediate_vs_deferred import stream as e7_stream     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
RESULTS = os.path.join(ROOT, "results")
PERMS = 20
E7_SEEDS = (0, 1, 2, 3, 4)
E7_N = 6000

# label -> (tight_naming, seed_charge_on, mode, gamma)
ARMS = [
    ("off",                 (False, False, "av", 0.0)),
    ("seed_av",             (False, True,  "av", 0.0)),
    ("seed_ln",             (False, True,  "ln", 0.0)),
    ("seed_av_naming",      (True,  True,  "av", 0.0)),
    ("seed_ln_naming",      (True,  True,  "ln", 0.0)),
]
GAMMAS = tuple(float(x) for x in os.environ.get(
    "E18_GAMMAS", "0,2,4,8,16,32,64,128,256,512").split(","))
ONLY = os.environ.get("E18_ONLY", "")            # "controls" runs the sweep alone


def set_arm(tight, seed_on, mode, gamma):
    C.set_flags(tight_naming=tight, unselected_statistic=False,
                seed_naming_charge=seed_on, seed_naming_mode=mode,
                seed_naming_gamma=gamma)


def labels(g, n):
    lab = [-1] * n
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    return lab


def score(recs, truth):
    g, b = run_stream(recs)
    lab = labels(g, len(recs))
    by = {}
    for i, x in enumerate(lab):
        by.setdefault(x, {}).setdefault(truth[i], 0)
        by[x][truth[i]] += 1
    purity = sum(max(c.values()) for c in by.values()) / len(recs)
    sizes = sorted((sum(c.values()) for k, c in by.items() if k != -1), reverse=True)
    charges = [m.get("seed_naming_charge", 0.0) for m in g.mint_log]
    margins = [m["released"] - m["price_paid"] for m in g.mint_log
               if m.get("released") is not None]
    return {"K": g.K, "ari": round(_ari(truth, lab), 4), "mints": len(g.mint_log),
            "background_records": sum(1 for x in lab if x == -1),
            "purity_incl_background": round(purity, 4),
            "node_sizes": sizes[:8], "n": len(recs),
            "mean_seed_charge_bits": (round(statistics.fmean(charges), 3)
                                      if charges else None),
            "min_release_margin_bits": (round(min(margins), 3) if margins else None)}


def fixtures():
    return (("twogroup", two_group), ("planted8", planted8),
            ("wikipedia", lambda: wikipedia(RESULTS)))


def controls(out):
    """The three canonical-order controls, once per arm and once per Gamma."""
    out["controls_by_arm"] = {}
    for label, cfg in ARMS:
        set_arm(*cfg)
        row = {"flags": C.flags()}
        for name, fx in fixtures():
            recs, truth = fx()
            row[name] = score(recs, truth)
        out["controls_by_arm"][label] = row
        print("arm", label, {k: (v["K"], v["ari"]) for k, v in row.items()
                             if k != "flags"}, flush=True)
    out["controls_by_gamma"] = {}
    for gam in GAMMAS:
        set_arm(False, True, "none", gam)
        row = {"gamma_bits": gam}
        for name, fx in fixtures():
            recs, truth = fx()
            row[name] = score(recs, truth)
        out["controls_by_gamma"][str(gam)] = row
        print("gamma", gam, {k: (v["K"], v["ari"]) for k, v in row.items()
                             if k != "gamma_bits"}, flush=True)


def e5(out, arms=("off", "seed_av", "seed_ln", "seed_av_naming")):
    """E5's design: the same records in 20 random orders."""
    cfg = dict(ARMS)
    out["e5_order_dependence"] = {"permutations": PERMS,
                                  "protocol": protocol_describe(), "arms": {}}
    data = {name: fx() for name, fx in fixtures()}
    for label in arms:
        set_arm(*cfg[label])
        arm_row = {}
        for name, (recs, truth) in data.items():
            aris, ks = [], []
            for p in range(PERMS):
                order = list(range(len(recs)))
                random.Random(1000 + p).shuffle(order)
                r = [recs[i] for i in order]
                t = [truth[i] for i in order]
                g, _ = run_stream(r)
                aris.append(round(_ari(t, labels(g, len(r))), 4))
                ks.append(g.K)
            arm_row[name] = {
                "true_K": len(set(truth)),
                "ARI": {"mean": round(statistics.fmean(aris), 4),
                        "sd": round(statistics.pstdev(aris), 4),
                        "min": min(aris), "max": max(aris)},
                "K": {"mean": round(statistics.fmean(ks), 2),
                      "sd": round(statistics.pstdev(ks), 2),
                      "min": min(ks), "max": max(ks), "all": ks}}
            print("e5", label, name, arm_row[name]["K"]["mean"],
                  arm_row[name]["ARI"]["mean"], flush=True)
        out["e5_order_dependence"]["arms"][label] = arm_row


def e7_deferred(out, arms=("off", "seed_av", "seed_ln", "seed_av_naming")):
    """E7's deferred arm: the shipped engine on the typo stream, five seeds."""
    cfg = dict(ARMS)
    out["e7_deferred"] = {"n": E7_N, "seeds": list(E7_SEEDS), "arms": {}}
    for label in arms:
        set_arm(*cfg[label])
        rows = []
        for s in E7_SEEDS:
            recs = e7_stream(E7_N, seed=s)
            g, b = run_stream(recs)
            typo = [m for m in g.mint_log
                    if any(str(v).startswith("TYPO") for v in m["per_key_bits"])]
            rows.append({"seed": s, "final_K": g.K, "total_mints": len(g.mint_log),
                         "typo_justified_nodes": len(typo),
                         "supports": sorted(sorted(g.key_name[k] for k in v.S)
                                            for v in g.nodes.values())})
        out["e7_deferred"]["arms"][label] = {
            "per_seed": rows,
            "final_K": [r["final_K"] for r in rows],
            "total_mints": [r["total_mints"] for r in rows],
            "typo_justified_nodes": sum(r["typo_justified_nodes"] for r in rows),
            "supports_identical_across_seeds":
                len({json.dumps(r["supports"]) for r in rows}) == 1,
            "supports_seed0": rows[0]["supports"]}
        print("e7", label, out["e7_deferred"]["arms"][label]["final_K"],
              out["e7_deferred"]["arms"][label]["total_mints"], flush=True)


def wiki_demo(out, arms=("off", "seed_av", "seed_ln", "seed_av_naming")):
    """The Wikipedia demonstration: nodes and the truth mix inside each."""
    cfg = dict(ARMS)
    recs, truth = wikipedia(RESULTS)
    out["wikipedia_demo"] = {"n": len(recs), "arms": {}}
    for label in arms:
        set_arm(*cfg[label])
        g, b = run_stream(recs)
        nodes = []
        for v in sorted(g.nodes.values(), key=lambda x: -x.t):
            mix = {}
            for m in v.members:
                mix[truth[m - 1]] = mix.get(truth[m - 1], 0) + 1
            nodes.append({"nid": v.nid, "t": v.t,
                          "mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
                          "pure": len(mix) == 1})
        covered = len({m for v in g.nodes.values() for m in v.members})
        out["wikipedia_demo"]["arms"][label] = {
            "K": g.K, "mints": len(g.mint_log), "nodes": nodes,
            "background_records": len(recs) - covered,
            "all_nodes_pure": all(x["pure"] for x in nodes)}
        print("wikidemo", label, g.K, [x["t"] for x in nodes], flush=True)


def main():
    t0 = time.time()
    out = {"experiment": "E18-seed-naming-charge-cost",
           "question": ("what a price-side seed naming charge costs the mint, "
                        "given that it cannot change the accumulator"),
           "protocol": protocol_describe(),
           "arms": {k: {"tight_naming": v[0], "seed_naming_charge": v[1],
                        "mode": v[2], "gamma": v[3]} for k, v in ARMS},
           "gamma_sweep_bits": list(GAMMAS)}
    controls(out)
    if ONLY != "controls":
        e5(out)
        e7_deferred(out)
        wiki_demo(out)
    set_arm(False, False, "av", 0.0)                 # leave the module shipped
    out["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(os.environ.get("ARMS_OUT", RESULTS),
                        os.environ.get("E18_COST_NAME", "e18_charge_cost.json"))
    json.dump(out, open(path, "w"), indent=2)
    print("written", path)


if __name__ == "__main__":
    main()
