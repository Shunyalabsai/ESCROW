"""The three E13 cells the tight-naming drive never reached: K* = 20 at noise 0.3, 0.4
and 0.5, T = 10000, the same five seeds. Uses E13's own one_run and its fork pool."""
import json, os, statistics, sys, time
from multiprocessing import get_context
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from escrow import codes as C
from experiments.e13_creation_bias import one_run, SEEDS

if __name__ == "__main__":
    jobs = [(20, noise, 10000, s) for noise in (0.3, 0.4, 0.5) for s in SEEDS]
    t0 = time.time()
    with get_context("fork").Pool(8) as pool:
        rs = pool.map(one_run, jobs)
    cells = {}
    for r in rs:
        cells.setdefault(r["noise"], []).append(r)
    out = {"flags": C.flags(), "T": 10000, "cells": {}}
    for noise, v in sorted(cells.items()):
        out["cells"][str(noise)] = {
            "mean_K": statistics.fmean(x["K"] for x in v),
            "K_all": [x["K"] for x in v],
            "mean_ari": round(statistics.fmean(x["ari"] for x in v), 4)}
        print(f"K*=20 noise={noise}: mean_K {out['cells'][str(noise)]['mean_K']} "
              f"{out['cells'][str(noise)]['K_all']} ari {out['cells'][str(noise)]['mean_ari']}",
              flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    o = os.environ.get("ARMS_OUT", ".")
    json.dump(out, open(os.path.join(o, f"e13_k20_tail_{os.environ.get('ARM','x')}.json"),
                        "w"), indent=2)
    print("done", out["runtime_seconds"], "s")
