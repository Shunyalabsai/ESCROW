"""Codelength primitives for the escrow insertion process.

Every function returns BITS. Every distribution is a valid prefix code: it sums to
exactly 1 over its alphabet including any escape symbol, and nothing is normalised by
a quantity the decoder cannot compute from the past (see the ESCROW paper,
encoding specification). lgamma, never factorials (implementation_notes #2).
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from math import lgamma


# --------------------------------------------------------------------------- #
# Two optional corrections, both OFF by default. The shipped default is exactly
# the code that produced every committed results file; with both flags off no
# branch below changes a single bit of any computation.
#
#   ESCROW_TIGHT_NAMING=1        stage-(iii) escape names the value from the
#                                inv - u + 1 slots that are actually reachable,
#                                not from all inv + 1. Makes each value block's
#                                predictive sum to exactly one (E16 measured the
#                                shipped block at 1 - u(u+1/2)/((N+1)(inv+1))).
#   ESCROW_UNSELECTED_STATISTIC=1  the escrow released at the gate excludes the
#                                seed key's own ledger entry, so the quantity
#                                compared against the price is the evidence from
#                                keys OTHER than the one that created the
#                                candidate (engine.py, step 7).
#
# The flags are read once at import. Set them in the environment before the
# process starts, or call set_flags() before building a graph.
# --------------------------------------------------------------------------- #
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


TIGHT_NAMING = _env_flag("ESCROW_TIGHT_NAMING")
UNSELECTED_STATISTIC = _env_flag("ESCROW_UNSELECTED_STATISTIC")


def set_flags(tight_naming: bool = None, unselected_statistic: bool = None) -> dict:
    """Set the correction flags from code (tests and sweeps). Returns the state."""
    global TIGHT_NAMING, UNSELECTED_STATISTIC
    if tight_naming is not None:
        TIGHT_NAMING = bool(tight_naming)
    if unselected_statistic is not None:
        UNSELECTED_STATISTIC = bool(unselected_statistic)
    return flags()


def flags() -> dict:
    return {"tight_naming": TIGHT_NAMING,
            "unselected_statistic": UNSELECTED_STATISTIC}


def naming_charge(naming: float, u: int) -> float:
    """The stage-(iii) charge for one escape out of a block that has already seen
    u distinct values.

    `naming` arrives from every caller as log2(inv + 1), with inv the key's global
    value inventory as of the past, so inv is recoverable and the tight form needs
    no change of signature anywhere. Shipped: log2(inv + 1), uniform over inv + 1
    slots of which only inv - u + 1 can be reached, because the u values already in
    the block are coded by the no-escape branch; the block therefore keeps only
    1 - u(u + 1/2)/((N + 1)(inv + 1)) of its mass. Tight: log2(inv - u + 1),
    uniform over exactly the reachable slots (the inv - u inventory values this
    block has not seen, plus one slot for a value new to the corpus), which makes
    the block's predictive sum to exactly one."""
    if not TIGHT_NAMING:
        return naming
    inv = round(2.0 ** naming - 1.0)
    return math.log2(max(1.0, inv - u + 1.0))


LN2 = 0.6931471805599453
LOG2_C0 = 1.5165  # log2(2.865064), Rissanen 1983 universal integer code constant
LOG2_PI = math.log2(math.pi)


@lru_cache(maxsize=1 << 20)
def lg2(x: float) -> float:
    """log2 of the Gamma function. Cached: the arguments repeat massively (small
    counts plus half-integer offsets), and the repair pass was spending a third
    of its time inside lgamma before the cache."""
    return lgamma(x) / LN2


def log2_star(x: float) -> float:
    """Iterated log: log2 x + log2 log2 x + ... (positive terms only)."""
    s = 0.0
    v = float(x)
    while True:
        v = math.log2(v)
        if v <= 0.0:
            return s
        s += v


def L_N(k: int) -> float:
    """Rissanen's universal code for the integer k >= 1. NEVER call with 0: the
    caller transmits L_N(x+1) for any count that may be zero (SPEC, primitives)."""
    assert k >= 1, "L_N is defined for k >= 1; transmit L_N(x+1) for zero-based counts"
    return log2_star(k) + LOG2_C0


def L_col(t: int, n: int) -> float:
    """KT / Beta(1/2,1/2) block code for a binary membership column with t ones in n.
    -log2 [ Gamma(t+1/2) Gamma(n-t+1/2) / (pi Gamma(n+1)) ]. Kraft-tight (UT-1)."""
    return LOG2_PI + lg2(n + 1.0) - lg2(t + 0.5) - lg2(n - t + 0.5)


def attach(t: int, n_incl: int) -> float:
    """Differential membership cost for the current record joining a node that has t
    members, with n_incl = stream length INCLUDING the current record: the cost of
    'member' minus the cost of 'not member' under the column KT predictive.
    Equals L_col(t+1, n_past+1) - L_col(t, n_past+1) with n_past = n_incl - 1.
    Zero at t = n_past/2, negative above (the derived Matthew effect, UT-19)."""
    n_past = n_incl - 1
    return math.log2((n_past - t + 0.5) / (t + 0.5))


def kt(c: int, N: int, d: float) -> float:
    """Predictive KT cost of one symbol whose count so far is c, in a block with N
    total observations and alphabet size d: -log2((c + 1/2)/(N + d/2))."""
    return -math.log2((c + 0.5) / (N + d / 2.0))


def L_KT_block(counts, d: int) -> float:
    """KT block (sequence) code for a categorical block with the given counts and
    FIXED alphabet d. No multinomial coefficient: this is the sequence code (UT-2)."""
    N = sum(counts)
    s = lg2(N + d / 2.0) - lg2(d / 2.0)
    for c in counts:
        s += lg2(0.5) - lg2(c + 0.5)
    return s


def L_supp(size_T: int, A_n: int) -> float:
    """Support code: L_N(|T|) + log2 C(A_n, |T|), uniform over subsets (E3)."""
    assert 1 <= size_T <= A_n
    return L_N(size_T) + (lg2(A_n + 1.0) - lg2(size_T + 1.0) - lg2(A_n - size_T + 1.0))


def L_ann(e: int, n: int, announce: bool) -> float:
    """Announcement (edit) KT Bernoulli over the edit history (E11)."""
    if announce:
        return -math.log2((e + 0.5) / (n + 1.0))
    return -math.log2((n - e + 0.5) / (n + 1.0))


def price(t: int, size_T: int, n: int, K: int, e: int, A_n: int) -> float:
    """The full node price (E12): what replaces lambda.
    Price(t,T,n) = Lann(n) + 2 + L_col(t,n) + L_supp(T) + (L_N(K+1)-L_N(K)) + log2(K+1)."""
    return (L_ann(e, n, True) + 2.0 + L_col(t, n) + L_supp(size_T, A_n)
            + (L_N(K + 1) - L_N(K) if K >= 1 else L_N(1)) + math.log2(K + 1.0))


def km_multinomial_complexity(n: int, K: int) -> float:
    """Exact NML parametric complexity C(n, K) for the multinomial, by the linear-time
    Kontkanen-Myllymaki recursion C_K = C_{K-1} + (n/(K-2)) C_{K-2}. AUDIT PATH ONLY:
    the shipped code is KT. Returns C, not log C (UT-10 asserts C(1,K) = K)."""
    if K == 1:
        return 1.0
    # C_2(n) = sum_k binom(n,k) (k/n)^k ((n-k)/n)^(n-k)
    c2 = 0.0
    for k in range(n + 1):
        logterm = lg2(n + 1.0) - lg2(k + 1.0) - lg2(n - k + 1.0)
        if k > 0:
            logterm += k * math.log2(k / n)
        if n - k > 0:
            logterm += (n - k) * math.log2((n - k) / n)
        c2 += 2.0 ** logterm
    if K == 2:
        return c2
    cm2, cm1 = 1.0, c2
    for j in range(3, K + 1):
        cm2, cm1 = cm1, cm1 + (n / (j - 2.0)) * cm2
    return cm1


class ValueBlock:
    """A categorical value block with the three-stage past-only escape (E7).

    Stage (i): 'is this value new to this block?' - KT Bernoulli on (u, N - u).
    Stage (ii): if seen, its index among the u seen values - KT over alphabet u.
    Stage (iii): if new, name it from the key's global inventory (the same
                 construction, handled by the caller via `naming_cost`).
    The predictive over {seen values} + {escape} sums to exactly 1 (UT-20).
    """

    __slots__ = ("counts", "N", "u")

    def __init__(self) -> None:
        self.counts: dict = {}
        self.N = 0
        self.u = 0

    def cost(self, vid, naming: float) -> float:
        """Bits to code vid under this block. `naming` is the stage-(iii) cost of
        identifying WHICH value, from the key's global inventory as of the PAST
        (the decoder shares it), charged whenever the value is new to this block.
        Omitting it makes a thin block's escapes nearly free without ever
        identifying the value - a Kraft violation that manufactures evidence
        (measured: sibling-node explosion, K=204 on the k=4 positive control).
        A block's first observation is new with certainty: cost = naming only."""
        c = self.counts.get(vid)
        if self.N == 0:
            return naming_charge(naming, self.u)
        if c is None:
            # stage (i) escape: P(new | past) via KT on (u novel events, N trials)
            return kt(self.u, self.N, 2) + naming_charge(naming, self.u)
        # stage (i) no-escape, then stage (ii): this one among the u seen
        return kt(self.N - self.u, self.N, 2) + kt(c, self.N, self.u)

    def p_mass_check(self) -> float:
        """Total predictive mass over {seen} + {escape}; must be 1.0 (UT-20).

        Stages (i) and (ii) only: the escape branch is counted as one symbol, so
        this is 1.0 under both naming rules. p_mass_full is the whole three-stage
        block, which is where the shipped rule loses mass."""
        if self.N == 0:
            return 1.0
        p_new = (self.u + 0.5) / (self.N + 1.0)
        p_old = (self.N - self.u + 0.5) / (self.N + 1.0)
        s = p_new
        for c in self.counts.values():
            s += p_old * (c + 0.5) / (self.N + self.u / 2.0)
        return s

    def p_mass_full(self, inv: int) -> float:
        """Brute-force total predictive mass of the WHOLE three-stage block over the
        reachable alphabet: the u values this block has seen, the inv - u inventory
        values it has not, and one slot for a value new to the corpus. Summed from
        self.cost itself, so it measures the shipped code rather than a formula.
        Exactly 1 under tight naming; 1 - u(u + 1/2)/((N + 1)(inv + 1)) under the
        shipped rule (E16, self_check)."""
        naming = math.log2(inv + 1.0)
        seen = list(self.counts)
        unseen = [("__unseen__", i) for i in range(inv - len(seen))]
        fresh = [("__fresh__", 0)]
        return sum(2.0 ** -self.cost(v, naming) for v in seen + unseen + fresh)

    def observe(self, vid) -> None:
        if vid in self.counts:
            self.counts[vid] += 1
        else:
            self.counts[vid] = 1
            self.u += 1
        self.N += 1

    def unobserve(self, vid) -> None:
        """Remove one observation of vid. Used when the cell changes owner: a
        record's value for a key is coded by exactly one block, so whoever gives
        it up must lose the count the new owner gains."""
        c = self.counts.get(vid, 0)
        if c <= 0:
            return
        if c == 1:
            del self.counts[vid]
            self.u -= 1
        else:
            self.counts[vid] = c - 1
        self.N -= 1
