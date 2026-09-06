"""E18. Can a PRICE-side seed naming charge restore the lifetime bound?

The design to test, from findings/THEOREM1.md: keep the seed key's evidence, which is
load-bearing, and pay for the selection by adding to the mint price a prefix codeword
that names the seed pair. A price-side charge does not touch the accumulator; it raises
the level the accumulator has to clear, so testing sup_t G_t against b + Delta is the
same event as testing sup_t (G_t - Delta) against b.

This file records, for every candidate trajectory on a null stream, the RAW accumulator
supremum sup_t G_t and sup_t (G_t - g_t[seed key]), together with the two candidate
naming codelengths, so every arm and every uniform slack Gamma can be evaluated exactly
offline from one run: the exceedance of a uniform-Gamma arm at level b is the empirical
frequency of sup_t G_t >= b + Gamma.

It sweeps the stream length T, because that is the question a constant charge lives or
dies on. Under a martingale, sup_t G_t is almost surely finite and E[sup] is bounded
uniformly in T. If the accrual index set is chosen by the data so the accumulator has
positive drift, sup_T grows with T, and then no constant charge can work.

Environment: ARM (label), SEEDS, T_LIST, ESCROW_TIGHT_NAMING.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow import codes as C                                            # noqa: E402
from escrow.codes import L_N                                            # noqa: E402
from escrow.engine import EscrowGraph                                    # noqa: E402
from escrow.batch import BatchObjective                                  # noqa: E402
from escrow.protocol import REPAIR_EVERY                                 # noqa: E402
from e16_arms import LEVELS, null_stream                                 # noqa: E402

QUANTILES = (0.5, 0.75, 0.9, 0.9375, 0.99, 0.99609, 0.999, 0.99976, 1.0)


class RawTracked(EscrowGraph):
    """Records sup_t of the RAW accumulator and of the accumulator without the seed
    key's own ledger entry, independently of which correction flags are set, so one
    run scores every price-side arm."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sup: dict = {}          # sig -> [sup G, sup (G - g_seed), t_last]
        self.minted_sigs: set = set()

    @staticmethod
    def _g_of(c, kid):
        if c.g_vec is not None and kid < len(c.g_vec):
            return float(c.g_vec[kid])
        return c.g.get(kid, 0.0)

    def _touch(self, c):
        row = self.sup.get(c.sig)
        if row is None:
            row = self.sup[c.sig] = [0.0, 0.0, 0]
        gs = c.G - self._g_of(c, c.sig[0])
        if c.G > row[0]:
            row[0] = c.G
        if gs > row[1]:
            row[1] = gs
        row[2] = c.t

    def _mint(self, c, support, n, released=None, g_outside=None):
        self.minted_sigs.add(c.sig)
        self._touch(c)
        return super()._mint(c, support, n, released=released, g_outside=g_outside)

    def process(self, record):
        r = super().process(record)
        for c in self.pool.values():
            self._touch(c)
        return r


def run(records, every=REPAIR_EVERY):
    g = RawTracked()
    b = BatchObjective(g).install()
    for i, rec in enumerate(records):
        g.process(rec)
        if (i + 1) % every == 0:
            b.repair()
    b.repair(full=True)
    return g


def quantiles(xs):
    s = sorted(xs)
    n = len(s)
    return {str(q): round(s[min(n - 1, int(math.ceil(q * n)) - 1 if q < 1 else n - 1)], 4)
            for q in QUANTILES}


def exceed(sups, offset):
    """Empirical Pr(sup_t G_t - offset >= b) at each level, with the nominal 2^-b."""
    n = max(1, len(sups))
    out = {}
    for b in LEVELS:
        p = sum(1 for s in sups if s - offset >= b) / n
        out[str(b)] = {"empirical": round(p, 6), "bound_2^-b": 2.0 ** -b,
                       "standard_error": round((p * (1 - p) / n) ** 0.5, 6),
                       "holds": p <= 2.0 ** -b + 3e-3}
    return out


def main():
    arm = os.environ.get("ARM", "unlabelled")
    seeds = int(os.environ.get("SEEDS", "300"))
    t_list = [int(x) for x in os.environ.get("T_LIST", "500,1000,2000,4000").split(",")]
    small = int(os.environ.get("SEEDS_LARGE_T", "100"))
    t0 = time.time()
    by_T = {}
    d_keys, d_vals = 4, 10                       # the null's schema, fixed by design
    delta_av = math.log2(d_keys) + math.log2(d_vals)
    # the "ln" code names the seed pair by its two intern ranks; on this null the
    # ranks run over 4 keys and 10 values, so the charge is bounded by
    delta_ln_max = L_N(d_keys) + L_N(d_vals)
    delta_ln_mean = (sum(L_N(k + 1) for k in range(d_keys)) / d_keys
                     + sum(L_N(v + 1) for v in range(d_vals)) / d_vals)
    for T in t_list:
        ns = seeds if T <= 2000 else small
        raw, unsel, steps, mints, Ks = [], [], [], 0, []
        for s in range(ns):
            g = run(null_stream(T, s))
            for row in g.sup.values():
                raw.append(row[0])
                unsel.append(row[1])
                steps.append(row[2])
            Ks.append(g.K)
            mints += len(g.minted_sigs)
        mean_steps = sum(steps) / max(1, len(steps))
        by_T[str(T)] = {
            "streams": ns, "candidates": len(raw), "mints": mints,
            "mean_K": sum(Ks) / len(Ks), "max_K": max(Ks),
            "accrual_steps_per_candidate": round(mean_steps, 2),
            "drift_bits_per_accrual_step": round(sum(raw) / len(raw) / max(1e-9, mean_steps), 4),
            "sup_G": {"mean": round(sum(raw) / len(raw), 4),
                      "min": round(min(raw), 4), "max": round(max(raw), 4),
                      "quantiles": quantiles(raw)},
            "sup_G_minus_seed_key": {"mean": round(sum(unsel) / len(unsel), 4),
                                     "max": round(max(unsel), 4),
                                     "quantiles": quantiles(unsel)},
            # the exceedance of the shipped statistic under each candidate charge
            "exceedance_charge_0": exceed(raw, 0.0),
            "exceedance_charge_av": exceed(raw, delta_av),
            "exceedance_charge_ln_mean": exceed(raw, delta_ln_mean),
            # and of the unselected statistic with the same charge added, which is
            # the only arm where a charge of this size can matter
            "exceedance_unselected_charge_0": exceed(unsel, 0.0),
            "exceedance_unselected_charge_av": exceed(unsel, delta_av),
            # what uniform slack Gamma WOULD be needed at each level
            "required_gamma": {str(b): round(_required_gamma(raw, b), 2) for b in LEVELS},
            "required_gamma_unselected": {str(b): round(_required_gamma(unsel, b), 2)
                                          for b in LEVELS},
        }
        print(f"  T={T} n={len(raw)} meanG={by_T[str(T)]['sup_G']['mean']} "
              f"steps={by_T[str(T)]['accrual_steps_per_candidate']} "
              f"bits/step={by_T[str(T)]['drift_bits_per_accrual_step']} "
              f"req_gamma_b8={by_T[str(T)]['required_gamma']['8']} "
              f"{round(time.time()-t0,1)}s", flush=True)
    report = {
        "experiment": "E18-seed-naming-charge-null",
        "arm": arm, "flags": C.flags(),
        "design": (f"{seeds} independent null streams per T (({small} for T>2000)), "
                   "4 independent categorical keys, 10 iid uniform values each; "
                   "E16's null_stream and protocol, so the numbers are like for like"),
        "charges": {"av_log2A_plus_log2d": round(delta_av, 4),
                    "ln_rank_code_mean": round(delta_ln_mean, 4),
                    "ln_rank_code_max": round(delta_ln_max, 4)},
        "by_T": by_T,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out = os.environ.get("ARMS_OUT",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", "..", "results"))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"e18_seed_charge_null_{arm}.json")
    json.dump(report, open(path, "w"), indent=2)
    print("written", path)


def _required_gamma(sups, b):
    """The smallest uniform slack Gamma whose empirical Pr(sup G - Gamma >= b) is at
    most 2^-b. At most m = floor(2^-b n) trajectories may exceed, so the (m+1)-th
    largest supremum must fall below b + Gamma."""
    s = sorted(sups)
    n = len(s)
    m = int(math.floor(2.0 ** -b * n))           # exceedances the bound still allows
    return max(0.0, s[n - m - 1] - b)            # the first one that must not exceed


if __name__ == "__main__":
    main()
