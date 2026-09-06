"""E16 exceedance protocol, re-run with the CORRECTIONS ACTUALLY IN THE ENGINE.

The committed E16 computes its four columns from one shipped run and corrects two of
them arithmetically after the fact. This file instead runs the same 48 E8 null streams
four times, once per flag configuration, and measures the accumulator the engine really
carries:

  arm            ESCROW_TIGHT_NAMING  ESCROW_UNSELECTED_STATISTIC  statistic
  shipped        off                  off                          sup_t G_t
  naming         on                   off                          sup_t G_t
  unselected     off                  on                           sup_t (G_t - g_t[seed])
  both           on                   on                           sup_t (G_t - g_t[seed])

against the nominal Ville bound 2^-b at b = 1, 2, 4, 8, 12.

The trajectory bookkeeping matches E16: sup starts at 0, is updated after every record
the candidate accrued on, and a released candidate's final value is the one at release.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow import codes as C                                            # noqa: E402
from escrow.engine import EscrowGraph                                    # noqa: E402
from escrow.batch import BatchObjective                                  # noqa: E402
from escrow.protocol import REPAIR_EVERY, describe as protocol_describe  # noqa: E402

LEVELS = (1, 2, 4, 8, 12)


def null_stream(T: int, seed: int, d: int = 10, keys: int = 4):
    """The E8 plateau null: iid uniform values on `keys` independent categorical keys."""
    rng = random.Random(1000 + seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(T)]


class TrackedGraph(EscrowGraph):
    """EscrowGraph that records, per candidate seed, the running supremum of the
    accumulator the release gate actually uses under the current flags."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sup: dict = {}            # sig -> sup_t of the statistic
        self.minted_sigs: set = set()

    @staticmethod
    def _g_of(c, kid):
        if c.g_vec is not None and kid < len(c.g_vec):
            return float(c.g_vec[kid])
        return c.g.get(kid, 0.0)

    def _stat(self, c) -> float:
        # A price-side charge of Delta bits does not touch the accumulator; it
        # raises the level the accumulator has to clear. Testing sup_t G_t against
        # b + Delta is the same event as testing sup_t (G_t - Delta) against b, and
        # subtracting keeps the levels b comparable with the committed table, so
        # the seed naming charge appears here as a deduction. Zero when off.
        d = C.seed_naming_charge(c.sig[0], c.sig[1], len(self.keys),
                                 len(self.key_by_id[c.sig[0]].inventory))
        if C.UNSELECTED_STATISTIC:
            return c.G - self._g_of(c, c.sig[0]) - d
        return c.G - d

    def _mint(self, c, support, n, released=None, g_outside=None):
        self.minted_sigs.add(c.sig)
        self.sup[c.sig] = max(self.sup.get(c.sig, 0.0), self._stat(c))
        return super()._mint(c, support, n, released=released, g_outside=g_outside)

    def process(self, record):
        r = super().process(record)
        for sig, c in self.pool.items():
            # every candidate that has ever existed gets a row, even one whose
            # statistic never rose above zero: E16's denominator is the number of
            # candidate TRAJECTORIES, not the number that ever went positive.
            v = self._stat(c)
            prev = self.sup.get(sig, 0.0)
            self.sup[sig] = v if v > prev else prev
        return r


def run_stream_tracked(records, every=REPAIR_EVERY):
    g = TrackedGraph()
    b = BatchObjective(g).install()
    for i, rec in enumerate(records):
        g.process(rec)
        if (i + 1) % every == 0:
            b.repair()
    b.repair(full=True)
    return g


def exceedance(sups: list) -> dict:
    out = {}
    for b in LEVELS:
        emp = sum(1 for s in sups if s >= b) / max(1, len(sups))
        out[str(b)] = {"empirical": round(emp, 6), "bound_2^-b": 2.0 ** -b,
                       "holds": emp <= 2.0 ** -b + 3e-3}
    return out


def main():
    lengths = (2000, 5000, 10000, 20000)
    seeds = 12
    t0 = time.time()
    arm = os.environ.get("ARM", "unlabelled")
    sups, Ks, mints, per_length = [], [], 0, {}
    for T in lengths:
        sT, kT, mT = [], [], 0
        for s in range(seeds):
            g = run_stream_tracked(null_stream(T, s))
            sT += list(g.sup.values())
            kT.append(g.K)
            mT += len(g.minted_sigs)
        per_length[str(T)] = {"streams": seeds, "candidates": len(sT),
                              "mean_K": sum(kT) / len(kT), "max_K": max(kT),
                              "mints": mT, "exceedance": exceedance(sT)}
        sups += sT
        Ks += kT
        mints += mT
        print(f"  T={T} done {round(time.time()-t0,1)}s  candidates={len(sT)} "
              f"maxK={max(kT)} mints={mT}", flush=True)
    report = {
        "experiment": "E16-arms",
        "arm": arm,
        "flags": C.flags(),
        "protocol": protocol_describe(),
        "design": "E8 plateau: lengths 2000, 5000, 10000, 20000 by 12 seeds = 48 streams, "
                  "4 independent categorical keys, 10 iid uniform values each",
        "statistic": (("sup_t (G_t - g_t[seed key])" if C.UNSELECTED_STATISTIC
                       else "sup_t G_t")
                      + (" - seed naming charge" if C.SEED_NAMING_CHARGE else "")),
        "candidates": len(sups),
        "streams": len(lengths) * seeds,
        "mean_K": sum(Ks) / len(Ks),
        "max_K": max(Ks),
        "mints": mints,
        "exceedance": exceedance(sups),
        "sup_stats": {"max": round(max(sups), 4), "min": round(min(sups), 4),
                      "mean": round(sum(sups) / len(sups), 4)},
        "by_length": per_length,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out = os.environ.get("ARMS_OUT", ".")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"e16_arms_{arm}.json")
    json.dump(report, open(path, "w"), indent=2)
    print("written", path)
    print(json.dumps({"arm": arm, "flags": report["flags"],
                      "exceedance": {k: v["empirical"] for k, v in report["exceedance"].items()},
                      "mints": mints, "max_K": report["max_K"]}, indent=1))


if __name__ == "__main__":
    main()
