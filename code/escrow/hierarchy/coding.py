"""Closed-form score, independent of the bitstream encoder and decoder.

All graph-dependent decisions are paid. The background histograms are sent in the
common header, then frozen. Nodes use fixed-alphabet Dirichlet-half block codes.
This is a prefix description of an observed prefix, not an anytime test process.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache

LN2 = math.log(2)


def integer_bits(value):
    """Elias gamma length for a nonnegative integer encoded as value + 1."""
    if value < 0:
        raise ValueError("integer must be nonnegative")
    return 2 * (value + 1).bit_length() - 1


def string_bits(value):
    raw = value.encode("utf-8")
    return integer_bits(len(raw)) + 8 * len(raw)


@lru_cache(maxsize=262144)
def _lgamma(x):
    return math.lgamma(x) / LN2


def kt_bits(counts):
    """Sequence probability, not a histogram probability. Counts include zero cells."""
    d = len(counts)
    if d <= 1:
        return 0.0
    n = sum(counts)
    return (_lgamma(n + d / 2) - _lgamma(d / 2)
            + sum(_lgamma(.5) - _lgamma(c + .5) for c in counts))


def binary_bits(ones, total):
    return kt_bits((total - ones, ones))


class Description:
    """Cache only the common data header; a scorer belongs to one immutable prefix."""
    def __init__(self, records):
        self.records = records
        self.n = len(records)
        self.keys = sorted({k for r in records for k in r})
        self.hist = {k: Counter(r[k] for r in records if k in r) for k in self.keys}
        self.vocab = {k: sorted(h) for k, h in self.hist.items()}
        self.index = {k: {v: i for i, v in enumerate(vs)} for k, vs in self.vocab.items()}
        header = 8 * 45 + integer_bits(self.n) + integer_bits(len(self.keys))
        for key in self.keys:
            header += string_bits(key) + integer_bits(len(self.vocab[key]))
            for value in self.vocab[key]:
                header += string_bits(value) + integer_bits(self.hist[key][value])
        self.header = header

    def score(self, state, validate=True):
        if len(state.records) != self.n or state.records != self.records:
            raise ValueError("description cache belongs to a different record prefix")
        if validate:
            state.validate()
        order = state.topological()
        terms = dict(header=float(self.header), structure=float(integer_bits(len(order))),
                     membership=0.0, ownership=0.0, presence=0.0, values=0.0)
        terms["structure"] += binary_bits(sum(len(v.parents) for v in state.nodes.values()),
                                          len(order) * (len(order) - 1) // 2)
        for i, nid in enumerate(order):
            v = state.nodes[nid]
            terms["structure"] += binary_bits(len(v.support), len(self.keys))
            eligible = state.eligible_rows(nid)
            terms["membership"] += binary_bits(len(v.members), len(eligible))

        for key in self.keys:
            supporting = [nid for nid in order if key in state.nodes[nid].support]
            routes, counts, present = {}, defaultdict(Counter), defaultdict(lambda: [0, 0])
            for r, record in enumerate(self.records):
                eligible = (0,) + tuple(nid for nid in supporting if r in state.nodes[nid].members)
                owner = state.owners[r].get(key, 0)
                bucket = routes.setdefault(eligible, Counter())
                bucket[owner] += 1
                pub = key in record
                present[owner][int(pub)] += 1
                if pub:
                    counts[owner][record[key]] += 1
            terms["ownership"] += sum(kt_bits(tuple(c.get(o, 0) for o in eligible))
                                        for eligible, c in routes.items())
            p_total = sum(self.hist[key].values())
            for owner, pc in present.items():
                if owner == 0:
                    # Known at decode time: the common header transmitted these counts.
                    p = (p_total + .5) / (self.n + 1)
                    terms["presence"] -= pc[1] * math.log2(p) + pc[0] * math.log2(1 - p)
                    d = len(self.vocab[key])
                    terms["values"] -= sum(c * math.log2((self.hist[key][v] + .5) /
                                                         (p_total + d / 2))
                                            for v, c in counts[owner].items())
                else:
                    terms["presence"] += kt_bits(tuple(pc))
                    terms["values"] += kt_bits(tuple(counts[owner].get(v, 0)
                                                        for v in self.vocab[key]))
        terms["total"] = sum(terms.values())
        return terms


def score(state):
    return Description(state.records).score(state)


def delta(before, after):
    if before.records != after.records:
        raise ValueError("a structural move must describe the identical records")
    obj = Description(before.records)
    left, right = obj.score(before), obj.score(after)
    return {k: right[k] - left[k] for k in left}
