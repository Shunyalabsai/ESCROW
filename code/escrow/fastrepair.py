"""Performance-only fast path for BatchObjective.reassign_pass.

SEMANTICS ARE UNCHANGED: the same candidate set, in the same iteration order,
with the same acceptance rule; the value-block terms are the analytic O(1)
difference of the same L_vblock closed form the stock pass evaluates by full
re-summation (the profile on the Lazada 2,600-record prefix put 86% of all
wall time in exactly that re-summation: 196M lg2 calls). Per-kid coverage and
publication aggregates replace the per-call union over every node, and are
updated after every applied move so intra-pass reads match the stock pass.

Equivalence is asserted by tests/check below (run as __main__: random-block
delta checks vs stock L_vblock to 1e-9) and by the prefix A/B in the runner
log (same K, same mints, same node member sets, same L_batch total).
"""
from __future__ import annotations

import math

from .batch import BatchObjective, L_vblock
from .codes import L_col, lg2, naming_charge


def dec_delta(u: int, N: int, c0: int, naming: float) -> float:
    """L_vblock after removing one observation of a value with count c0 >= 1,
    minus L_vblock before. Matches L_vblock(bg2)-L_vblock(bg) of the stock pass."""
    if u == 1 and c0 == 1:                      # block empties (then N == 1)
        return -(L_col(1, N) + naming_charge(naming, 0))
    Nr = N - u
    if c0 == 1:                                 # value disappears, u -> u-1
        return (L_col(u - 1, N - 1) - L_col(u, N)
                + (lg2(Nr + (u - 1) / 2.0) - lg2((u - 1) / 2.0))
                - (lg2(Nr + u / 2.0) - lg2(u / 2.0))
                - naming_charge(naming, u - 1))     # the u-th value un-named
    return (L_col(u, N - 1) - L_col(u, N)       # c0 > 1: u unchanged, Nr -> Nr-1
            + lg2(Nr - 1 + u / 2.0) - lg2(Nr + u / 2.0)
            + lg2(c0 - 0.5) - lg2(c0 - 1.5))


def inc_delta(u: int, N: int, c1: int, naming: float) -> float:
    """L_vblock after adding one observation of a value with current count c1,
    minus L_vblock before. Matches L_vblock(nbc+vid)-L_vblock(nbc)."""
    if N == 0:
        return L_col(1, 1) + naming_charge(naming, 0)
    Nr = N - u
    if c1 == 0:                                 # new value, u -> u+1, Nr unchanged
        return (L_col(u + 1, N + 1) - L_col(u, N)
                + (lg2(Nr + (u + 1) / 2.0) - lg2((u + 1) / 2.0))
                - (lg2(Nr + u / 2.0) - lg2(u / 2.0))
                + naming_charge(naming, u))         # the (u+1)-th value named
    return (L_col(u, N + 1) - L_col(u, N)       # existing value: Nr -> Nr+1
            + lg2(Nr + 1 + u / 2.0) - lg2(Nr + u / 2.0)
            + lg2(c1 - 0.5) - lg2(c1 + 0.5))


class FastBatchObjective(BatchObjective):
    def _fast_gain(self, rec_n, v, covered_by_kid, pub_by_kid) -> float:
        g = self.g
        keys_r = g.record_keys.get(rec_n, frozenset())
        vals_r = g.record_vals.get(rec_n, {})
        d = L_col(v.t + 1, g.n) - L_col(v.t, g.n)
        for kid in v.S:
            pub = 1 if kid in keys_r else 0
            p = v.p.get(kid, 0)
            d += L_col(p + pub, v.t + 1) - L_col(p, v.t)
            covered = covered_by_kid.get(kid, ())
            if rec_n in covered:
                continue
            ki = g.key_by_id[kid]
            n0b = g.n - len(covered)
            p0b = max(0, ki.P - len(pub_by_kid.get(kid, ())))
            d -= L_col(min(p0b, n0b), n0b) if n0b > 0 else 0.0
            n0a, p0a = n0b - 1, p0b - pub
            d += L_col(min(max(p0a, 0), n0a), n0a) if n0a > 0 else 0.0
            if pub and kid in vals_r:
                vid = vals_r[kid]
                bg = ki.background
                c0 = bg.counts.get(vid, 0)
                if c0 > 0:
                    naming = math.log2(len(ki.inventory) + 1.0)
                    d += dec_delta(bg.u, bg.N, c0, naming)
                    nb = v.blocks.get(kid)
                    if nb is None:
                        d += L_col(1, 1) + naming_charge(naming, 0)
                    else:
                        d += inc_delta(nb.u, nb.N, nb.counts.get(vid, 0), naming)
        return d

    def reassign_pass(self) -> int:
        g = self.g
        if not g.nodes:
            return 0
        covered_all = set().union(*(v.members for v in g.nodes.values()))
        covered_by_kid: dict = {}
        pub_by_kid: dict = {}
        for v in g.nodes.values():
            for kid in v.S:
                covered_by_kid.setdefault(kid, set()).update(v.members)
                pub_by_kid.setdefault(kid, set()).update(v.pub.get(kid, ()))
        moved = 0
        for rec_n in list(g.record_keys):
            if rec_n in covered_all:
                continue
            keys_r = g.record_keys[rec_n]
            best, best_v = -1e-9, None
            for v in g.nodes.values():          # stock iteration order preserved
                if not (v.S & keys_r):
                    continue
                d = self._fast_gain(rec_n, v, covered_by_kid, pub_by_kid)
                if d < best:
                    best, best_v = d, v
            if best_v is not None:
                self._apply_reassign(rec_n, best_v)
                for kid in best_v.S:
                    covered_by_kid.setdefault(kid, set()).add(rec_n)
                    if kid in keys_r:
                        pub_by_kid.setdefault(kid, set()).add(rec_n)
                moved += 1
        return moved


if __name__ == "__main__":                      # delta self-test vs stock L_vblock
    import random
    rng = random.Random(1)
    worst = 0.0
    for trial in range(2000):
        u = rng.randrange(1, 12)
        counts = {i: rng.randrange(1, 9) for i in range(u)}
        naming = math.log2(rng.randrange(1, 4000) + 1.0)
        N = sum(counts.values())
        vid = rng.randrange(u)
        # decrement check
        before = L_vblock(counts, naming)
        after_c = dict(counts)
        after_c[vid] -= 1
        if after_c[vid] == 0:
            del after_c[vid]
        got = dec_delta(u, N, counts[vid], naming)
        want = L_vblock(after_c, naming) - before
        worst = max(worst, abs(got - want))
        # increment check (existing and new value)
        vid2 = rng.randrange(u + 1)             # u == new value
        after_c = dict(counts)
        after_c[vid2] = after_c.get(vid2, 0) + 1
        got = inc_delta(u, N, counts.get(vid2, 0), naming)
        want = L_vblock(after_c, naming) - before
        worst = max(worst, abs(got - want))
    # empty-block increment
    worst = max(worst, abs(inc_delta(0, 0, 0, 3.0) - (L_vblock({0: 1}, 3.0) - 0.0)))
    print(f"self-test worst |delta error| = {worst:.3e}")
    assert worst < 1e-9
    print("fastrepair delta self-test PASSED")
