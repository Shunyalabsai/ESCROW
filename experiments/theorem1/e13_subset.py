"""A fast subset of E13's grid: K* in {2, 8}, noise in {0.0, 0.1, 0.2, 0.3}, T = 3000,
the same five seeds. Enough to say whether a flag setting keeps the creation-bias curve."""
import json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from escrow import codes as C
from experiments.e13_creation_bias import one_run, SEEDS

out = {"flags": C.flags(), "cells": {}}
for K_star in (2, 8):
    for noise in (0.0, 0.1, 0.2, 0.3):
        rs = [one_run((K_star, noise, 3000, s)) for s in SEEDS]
        cell = {"mean_K": statistics.fmean(r["K"] for r in rs),
                "K_all": [r["K"] for r in rs],
                "mean_ari": round(statistics.fmean(r["ari"] for r in rs), 4)}
        out["cells"][f"K{K_star}_noise{noise}"] = cell
        print(f"K*={K_star} noise={noise}: mean_K {cell['mean_K']} {cell['K_all']} "
              f"ari {cell['mean_ari']}", flush=True)
o = os.environ.get("ARMS_OUT", ".")
json.dump(out, open(os.path.join(o, f"e13_subset_{os.environ.get('ARM','x')}.json"), "w"),
          indent=2)
