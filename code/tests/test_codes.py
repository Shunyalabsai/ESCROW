"""UT-1, UT-2, UT-6, UT-10, UT-19, UT-20: the code-level gates (paper unit-test suite)."""
import itertools
import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.codes import (L_KT_block, L_col, ValueBlock, attach, kt,
                          km_multinomial_complexity, price, L_supp)


def comb(n, k):
    return math.comb(n, k)


def test_ut1_kraft_column_code():
    for n in (1, 2, 3, 5, 8, 12):
        s = sum(comb(n, t) * 2.0 ** (-L_col(t, n)) for t in range(n + 1))
        assert abs(s - 1.0) < 1e-12, (n, s)


def test_ut2_kraft_value_block():
    d = 3
    for N in range(1, 7):
        s = 0.0
        for seq in itertools.product(range(d), repeat=N):
            counts = [seq.count(i) for i in range(d)]
            s += 2.0 ** (-L_KT_block(counts, d))
        assert abs(s - 1.0) < 1e-12, (N, s)


def test_ut6_incremental_equals_block():
    rng = random.Random(0)
    worst = 0.0
    for _ in range(20000):
        d = rng.randint(2, 12)
        counts = [rng.randint(0, 60) for _ in range(d)]
        j = rng.randrange(d)
        N = sum(counts)
        inc = L_KT_block([c + (1 if i == j else 0) for i, c in enumerate(counts)], d) \
              - L_KT_block(counts, d)
        pred = kt(counts[j], N, d)
        worst = max(worst, abs(inc - pred))
    assert worst < 1e-9, worst


def test_ut10_singleton_price_two_families_and_the_forbidden_plugin():
    for d, want in ((2, 1.0), (3, math.log2(3)), (10, math.log2(10)),
                    (50, math.log2(50)), (1000, math.log2(1000))):
        got = L_KT_block([1] + [0] * (d - 1), d)
        assert abs(got - want) < 1e-9, (d, got, want)
        C = km_multinomial_complexity(1, d)
        assert abs(C - d) < 1e-6, (d, C)
    # the plug-in entropy of the singleton is 0.0 and is NOT a code
    counts = [1] + [0] * 9
    plugin = -sum((c / 1) * math.log2(c / 1) for c in counts if c)
    assert plugin == 0.0


def test_ut19_attach_price_consistency_and_sign():
    # The incremental form is canonical (implementation_notes #1: never compute a
    # delta as a difference of totals). The block-difference cross-check loses
    # precision through lgamma cancellation as n grows, so it is asserted tightly
    # at moderate n and loosely at large n; the DIRECT form is what ships.
    rng = random.Random(1)
    worst_mod, worst_big = 0.0, 0.0
    for _ in range(10000):
        n_past = rng.randint(2, 10 ** 4)
        t = rng.randint(1, n_past - 1)
        d = abs(attach(t, n_past + 1) - (L_col(t + 1, n_past + 1) - L_col(t, n_past + 1)))
        worst_mod = max(worst_mod, d)
    for _ in range(2000):
        n_past = rng.randint(10 ** 4, 10 ** 6)
        t = rng.randint(1, n_past - 1)
        d = abs(attach(t, n_past + 1) - (L_col(t + 1, n_past + 1) - L_col(t, n_past + 1)))
        worst_big = max(worst_big, d)
    assert worst_mod < 1e-9, worst_mod
    assert worst_big < 1e-7, worst_big
    assert attach(50, 101) == 0.0
    assert attach(51, 101) < 0.0 < attach(49, 101)


def test_ut20_escape_closure():
    rng = random.Random(2)
    b = ValueBlock()
    seen_once_cost = None
    for i in range(200):
        assert abs(b.p_mass_check() - 1.0) < 1e-12, (i, b.p_mass_check())
        v = rng.randint(0, 30)
        if b.counts.get(v) == 1 and seen_once_cost is None:
            seen_once_cost = b.cost(v, 5.0)
        b.observe(v)
    # a value seen twice is strictly cheaper than the same value seen once
    v2 = next(v for v, c in b.counts.items() if c >= 3)
    once = ValueBlock(); once.observe(v2)
    twice = ValueBlock(); twice.observe(v2); twice.observe(v2)
    assert twice.cost(v2, 5.0) < once.cost(v2, 5.0)
    # an unseen value costs a finite amount
    assert math.isfinite(b.cost("never-seen", 5.0))


def test_ut4_kraft_mint_price_upper_bound():
    # sum over subsets S (via L_col over t) and supports T of 2^-Price <= 1, small n
    n, A_n, K, e = 8, 4, 1, 0
    total = 0.0
    for t in range(0, n + 1):
        for sz in range(1, A_n + 1):
            total += (comb(n, t) * comb(A_n, sz)
                      * 2.0 ** (-price(t, sz, n, K, e, A_n)))
    assert total <= 1.0 + 1e-9, total


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
