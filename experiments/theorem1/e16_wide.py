"""The E16 exceedance on MANY INDEPENDENT null streams.

E16's plateau design reuses one random generator per seed across its four lengths, so the
T = 5000 stream begins with the T = 2000 stream: the four length blocks are the same 480
trajectories counted four times, and the committed file's identical rows across lengths
show it. The supremum is set inside the first 2,000 records in every case, so this file
holds T = 2,000 and sweeps the seed instead, giving genuinely independent streams.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow import codes as C                                     # noqa: E402
from experiments.e16_arms import (LEVELS, exceedance, null_stream,  # noqa: E402
                                  run_stream_tracked)


def main():
    arm = os.environ.get("ARM", "unlabelled")
    seeds = int(os.environ.get("SEEDS", "300"))
    T = int(os.environ.get("T", "2000"))
    t0 = time.time()
    sups, Ks, mints = [], [], 0
    for s in range(seeds):
        g = run_stream_tracked(null_stream(T, s))
        sups += list(g.sup.values())
        Ks.append(g.K)
        mints += len(g.minted_sigs)
        if (s + 1) % 50 == 0:
            print(f"  {s+1}/{seeds} seeds {round(time.time()-t0,1)}s "
                  f"cands={len(sups)} mints={mints}", flush=True)
    n = len(sups)
    exc = exceedance(sups)
    for b in LEVELS:
        p = exc[str(b)]["empirical"]
        exc[str(b)]["standard_error"] = round((p * (1 - p) / n) ** 0.5, 6)
    report = {"experiment": "E16-wide", "arm": arm, "flags": C.flags(),
              "design": f"{seeds} independent null streams of T={T}, 4 categorical keys, "
                        "10 iid uniform values each",
              "statistic": ("sup_t (G_t - g_t[seed key])" if C.UNSELECTED_STATISTIC
                            else "sup_t G_t"),
              "streams": seeds, "candidates": n, "mints": mints,
              "mean_K": sum(Ks) / len(Ks), "max_K": max(Ks),
              "exceedance": exc,
              "sup_stats": {"max": round(max(sups), 4), "mean": round(sum(sups) / n, 4)},
              "runtime_seconds": round(time.time() - t0, 1)}
    out = os.environ.get("ARMS_OUT", ".")
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, f"e16_wide_{arm}.json"), "w"), indent=2)
    print(json.dumps({"arm": arm, "n": n, "mints": mints, "maxK": report["max_K"],
                      "exceedance": {k: v["empirical"] for k, v in exc.items()}}, indent=1))


if __name__ == "__main__":
    main()
