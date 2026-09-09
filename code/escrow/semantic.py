"""Semantics as a code, never as a score.

THE TEMPTATION AND WHY IT IS REFUSED. A categorical block sees `red` and `crimson` as two unrelated
symbols, so any identity that lives below that granularity is invisible to it. The obvious repair is
a cosine similarity with a cut-off: call two values the same when they are close enough. That repair
would put back exactly the calibrated quantity this whole line of work exists to remove. `tau_new`
under a new name is still `tau_new`.

So semantics enter only in a form that can be charged in bits. An embedding here does not score a
pair, it defines a probability distribution over values, and the decision is made the way every other
decision is made: whichever description is shorter.

THE CONSTRUCTION, WHICH COSTS ONE BIT. Two complete codes for the same stream, the shipped
categorical one and a semantic one, are combined as a Bayesian mixture

    P_mix(data) = 1/2 P_cat(data) + 1/2 P_sem(data)
    L_mix       = -log2( 2^-L_cat / 2 + 2^-L_sem / 2 )  <=  min(L_cat, L_sem) + 1

That inequality is the whole argument. There is no mixture weight to tune, no threshold, and no
temperature at the top level: the combined code is never worse than the better of its two parts by
more than a single bit over the entire stream, and it needs no help deciding which part is better.
If the embedding is useless, or the domain is nothing like what it was trained on, the mixture pays
one bit and behaves exactly like today's method. A cut-off cannot degrade that gracefully.

WHERE THE TWO CODES DIFFER. Only in the naming step. When a block meets a value it has not held
before, the shipped code spends log2(inventory + 1) bits, a uniform choice from the key's inventory.
The semantic code spends -log2 P_sem(v | values this block already holds), which is short when the
new value sits near what the block has already seen and long when it does not. Everything else, the
novelty column and the KT block over repeats, is untouched.

NO BANDWIDTH IS CHOSEN EITHER. P_sem needs a scale on which distance is read. Rather than pick one,
it is a uniform mixture over a fixed grid of scales, which is itself one valid distribution. A
mixture is not a choice.

THE EMBEDDING IS STILL A CHOICE, AND IT IS DECLARED. Like the choice of prior, it is a modelling
decision whose cost is measured rather than hidden. What it must never become is a quantity fitted to
the data it is then evaluated on.
"""
from __future__ import annotations

import json
import math

# Distance is read on several scales at once and the results averaged, so no bandwidth is chosen.
# The grid spans "only near-identical values count" to "everything counts a little".
SCALE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)


def mixture_bits(l_a: float, l_b: float, prior_a: float = 0.5) -> float:
    """Codelength of the Bayesian mixture of two codes, in bits.

    Computed in the log domain because 2**-l underflows for any real stream: at a few thousand
    records these codelengths run to tens of thousands of bits.
    """
    la = l_a - math.log2(prior_a)
    lb = l_b - math.log2(1.0 - prior_a)
    lo = min(la, lb)
    return lo - math.log2(1.0 + 2.0 ** (lo - max(la, lb)))


class Embedding:
    """Value string to unit vector. Loaded from a cache file so the engine stays stdlib-only.

    The cache is written by `code/tools/embed_values.py` on a machine that has the model. Nothing in
    the insertion process imports a neural library, and a missing cache degrades to no semantics
    rather than to an error.
    """

    def __init__(self, vectors: dict | None = None):
        self.vectors = vectors or {}
        self._norm_cache: dict = {}

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls({k: list(map(float, v)) for k, v in raw.get("vectors", raw).items()})

    def has(self, value: str) -> bool:
        return value in self.vectors

    def cos(self, a: str, b: str) -> float:
        """Cosine similarity, or 0.0 when either value is not in the cache."""
        va, vb = self.vectors.get(a), self.vectors.get(b)
        if va is None or vb is None:
            return 0.0
        num = sum(x * y for x, y in zip(va, vb))
        na = self._norm_cache.get(a)
        if na is None:
            na = self._norm_cache[a] = math.sqrt(sum(x * x for x in va)) or 1.0
        nb = self._norm_cache.get(b)
        if nb is None:
            nb = self._norm_cache[b] = math.sqrt(sum(x * x for x in vb)) or 1.0
        return num / (na * nb)


def semantic_naming_bits(emb: Embedding, value: str, seen: list, inventory: list) -> float:
    """Bits to name `value` from `inventory`, given the values this block already holds.

    The alphabet is exactly the one the shipped charge uses: the key's inventory plus one slot for a
    value not yet in it. That slot keeps the mass a uniform code would give it, so the two codes are
    compared over the same alphabet and semantics only ever redistributes mass among known values.

    With nothing seen yet there is nothing to be near, and this returns the uniform charge, so a
    block's first value costs exactly what it costs today.
    """
    m = len(inventory)
    if m == 0:
        return 0.0
    uniform = math.log2(m + 1.0)
    if not seen:
        return uniform
    keep = 1.0 / (m + 1.0)                       # mass reserved for a value outside the inventory
    budget = 1.0 - keep

    def nearest(v: str) -> float:
        # distance in [0, 2], from cosine similarity against the closest value already held
        return 1.0 - max((emb.cos(v, s) for s in seen), default=0.0)

    d = {v: nearest(v) for v in inventory}
    total = 0.0
    for beta in SCALE_GRID:
        w = {v: math.exp(-beta * dv) for v, dv in d.items()}
        z = sum(w.values())
        if z <= 0.0:
            continue
        total += (w.get(value, 0.0) / z) / len(SCALE_GRID)
    p = budget * total
    if p <= 0.0:
        return uniform
    return -math.log2(p)


def block_naming_bits(emb, counts_in_arrival_order, inventory, semantic: bool) -> float:
    """Total naming charge for one value block, under either code.

    `counts_in_arrival_order` is the sequence of value strings as the block first met them, which is
    what the naming charge is levied on: one charge per value new to the block. The categorical arm
    reproduces the shipped `u * log2(inventory + 1)` exactly, so the two arms differ in this term and
    nowhere else.
    """
    seen: list = []
    total = 0.0
    for v in counts_in_arrival_order:
        if v in seen:
            continue
        if semantic:
            total += semantic_naming_bits(emb, v, seen, inventory)
        else:
            total += math.log2(len(inventory) + 1.0)
        seen.append(v)
    return total


# --------------------------------------------------------------------------- #
# Where semantics actually belongs, found by measurement rather than by design.
#
# The first version of this module put the embedding in the naming charge, the bits spent when a
# block meets a value it has not held before. E56 and E57 then measured what that term is worth. On
# a block of 150 records over 5 values, pooled with another of the same shape and a disjoint
# vocabulary, the cost of pooling is +298 bits, of which the KT block over repeats carries +293 and
# the naming charge carries +8.7. So the naming charge cannot move a merge decision, and E57's
# probes came back separated by about one bit against a barrier of 243.
#
# The decision lives in the predictive over values. That is also where similarity ought to act: the
# claim "crimson is a kind of red" is a claim that seeing `red` should make `crimson` cheaper to
# transmit, which is a statement about prediction and not about naming. Smoothing the KT counts by
# similarity says exactly that, and says it in bits.
# --------------------------------------------------------------------------- #


def _kernel(emb: Embedding, inventory: list, sharpness: float) -> dict:
    """K[v][w] in [0, 1], with K[v][v] = 1. Cosine below zero is clamped, because a negative
    similarity is not evidence against a value, it is an absence of evidence for it."""
    k = {}
    for v in inventory:
        row = {}
        for w in inventory:
            if v == w:
                row[w] = 1.0
            else:
                c = emb.cos(v, w)
                row[w] = max(0.0, c) ** sharpness if c > 0.0 else 0.0
        k[v] = row
    return k


def block_value_bits(emb: Embedding, seq, inventory: list, semantic: bool,
                     alpha: float = 0.5) -> float:
    """Prequential codelength of a value sequence, in bits, under one of the two predictives.

    Categorical, which is the Krichevsky-Trofimov estimator the engine ships:

        P(v) = (c_v + alpha) / (N + alpha * m)

    Semantic, the same estimator with the counts smoothed by similarity, so a value borrows strength
    from the values it resembles:

        P(v) = (sum_w c_w K(v, w) + alpha) / (sum_x sum_w c_w K(x, w) + alpha * m)

    Both are normalised over the inventory at every step, so both are valid codes and the comparison
    between them is a comparison of two descriptions of the same data. With K the identity the second
    reduces to the first exactly, which is what makes an unhelpful embedding harmless rather than
    harmful. The sharpness of K is not chosen: the predictive is a uniform mixture over a grid, and a
    mixture is not a choice.
    """
    m = len(inventory)
    if m == 0 or not seq:
        return 0.0
    idx = {v: i for i, v in enumerate(inventory)}
    counts = [0.0] * m
    total = 0.0
    bits = 0.0

    if not semantic:
        for v in seq:
            i = idx.get(v)
            if i is None:
                continue
            p = (counts[i] + alpha) / (total + alpha * m)
            bits += -math.log2(p)
            counts[i] += 1.0
            total += 1.0
        return bits

    kernels = [_kernel(emb, inventory, s) for s in SCALE_GRID]
    rows = [[[k[v][w] for w in inventory] for v in inventory] for k in kernels]
    for v in seq:
        i = idx.get(v)
        if i is None:
            continue
        p = 0.0
        for kr in rows:
            smoothed = [sum(counts[j] * kr[a][j] for j in range(m)) for a in range(m)]
            z = sum(smoothed) + alpha * m
            p += ((smoothed[i] + alpha) / z) / len(rows)
        bits += -math.log2(p)
        counts[i] += 1.0
        total += 1.0
    return bits
