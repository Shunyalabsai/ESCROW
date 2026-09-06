"""Is the corrected accumulator a supermartingale STEP BY STEP?

Ville's bound needs E[2^{dG} | F_{t-1}] <= 1 at every accrual step. This file measures
that expectation directly, and separately for the value terms and the presence terms, by
re-deriving each step's delta from a pre-record snapshot of the same blocks the engine
priced. It reports the empirical mean of 2^{dG} over every accrual step of every
candidate, which is 1 for a martingale and at most 1 for a supermartingale.

The average of 2^{dG} over realised steps is an estimate of E_true[2^{dG}], not of
E_P[2^{dG}]; Theorem 1 needs the latter, so this file computes BOTH: the realised mean
and, per step, the exact expectation under the incumbent's own predictive summed over the
value alphabet (which is what the theorem's H0 asserts generates the record).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow import codes as C                                    # noqa: E402
from escrow.codes import ValueBlock, kt                          # noqa: E402
from escrow.engine import EscrowGraph                            # noqa: E402
from escrow.batch import BatchObjective                          # noqa: E402
from experiments.e16_arms import null_stream                     # noqa: E402

D = 10                          # alphabet of every key in the null


class StepProbe(EscrowGraph):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.real_val, self.real_pres, self.real_all = [], [], []
        self.exp_val = []           # exact E_P[2^dval] per key per step

    def process(self, record):
        # snapshot everything the accrual will read, before any update
        snap_keys = {kid: (len(ki.inventory), ki.P,
                           dict(ki.background.counts), ki.background.u, ki.background.N)
                     for kid, ki in self.key_by_id.items()}
        sigs = []
        for a, x in record.items():
            ki = self.keys.get(a)
            if ki is not None:
                inv = len(ki.inventory)
                sigs.append((ki.kid, ki.inventory.get(str(x), inv)))
        snap_c = {}
        for sig in sigs:
            c = self.pool.get(sig)
            if c is not None:
                snap_c[sig] = (c.t, dict(c.p),
                               {k: (dict(b.counts), b.u, b.N) for k, b in c.blocks.items()})
        past = self.n
        r = super().process(record)
        n = self.n
        vals = self.record_vals.get(n, {})
        owners = self.record_owner.get(n, {})
        K_r = set(vals)
        for kid in sorted(K_r):
            if owners.get(kid, 0) != 0:
                continue                                    # not a residual
            sig = (kid, vals[kid])
            if sig not in snap_c:
                continue                                    # candidate's first step
            t_c, p_c, blks_c = snap_c[sig]
            dval_tot = dpres_tot = 0.0
            for k2 in K_r:
                if k2 == kid:
                    continue                                # the seed key is excluded
                inv2, P2, bgc, bgu, bgN = snap_keys[k2]
                naming2 = math.log2(inv2 + 1.0)
                bg = ValueBlock(); bg.counts = dict(bgc); bg.u = bgu; bg.N = bgN
                cc, cu, cN = blks_c.get(k2, ({}, 0, 0))
                cb = ValueBlock(); cb.counts = dict(cc); cb.u = cu; cb.N = cN
                v2 = vals[k2]
                dval_tot += bg.cost(v2, naming2) - cb.cost(v2, naming2)
                dpres_tot += (kt(P2, past, 2) if past > 0 else 1.0) \
                    - kt(p_c.get(k2, 0), t_c, 2)
                # exact E_P[2^dval] for this key: sum over the incumbent's alphabet
                s = 0.0
                for vv in list(range(inv2)) + [inv2]:
                    lp = 2.0 ** -bg.cost(vv, naming2)
                    s += lp * 2.0 ** (bg.cost(vv, naming2) - cb.cost(vv, naming2))
                self.exp_val.append(s)
            self.real_val.append(2.0 ** dval_tot)
            self.real_pres.append(2.0 ** dpres_tot)
            self.real_all.append(2.0 ** (dval_tot + dpres_tot))
        return r


def run(seeds=40, T=2000):
    g_all = StepProbe()
    rv, rp, ra, ev = [], [], [], []
    for s in range(seeds):
        g = StepProbe()
        b = BatchObjective(g).install()
        for i, rec in enumerate(null_stream(T, s)):
            g.process(rec)
            if (i + 1) % 100 == 0:
                b.repair()
        b.repair(full=True)
        rv += g.real_val; rp += g.real_pres; ra += g.real_all; ev += g.exp_val
    return rv, rp, ra, ev


def main():
    rv, rp, ra, ev = run()
    out = {"flags": C.flags(), "accrual_steps": len(ra),
           "mean_2^dG_value_terms_realised": round(statistics.mean(rv), 6),
           "mean_2^dG_presence_terms_realised": round(statistics.mean(rp), 6),
           "mean_2^dG_all_nonseed_realised": round(statistics.mean(ra), 6),
           "mean_exact_E_P[2^dval]_per_key": round(statistics.mean(ev), 6),
           "key_terms": len(ev),
           "reading": ("<= 1 means the step is a supermartingale on average; the exact "
                       "column is the theorem's own expectation, under the incumbent's "
                       "predictive, and must be exactly 1 when both blocks are "
                       "normalised")}
    print(json.dumps(out, indent=2))
    o = os.environ.get("ARMS_OUT", ".")
    json.dump(out, open(os.path.join(o, f"e16_stepwise_{os.environ.get('ARM','x')}.json"),
                        "w"), indent=2)


if __name__ == "__main__":
    main()
