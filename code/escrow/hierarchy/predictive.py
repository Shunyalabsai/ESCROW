"""Causal observation code, kept separate from repaired prefix descriptions.

At each arrival a mixture over current member distributions is evaluated before
the state sees that record. Weights are rebuilt from past membership, so no
whole-sequence fixed-expert mixture guarantee is asserted for this predictor.
"""
from __future__ import annotations

import math
from collections import Counter

from .coding import integer_bits, string_bits


def value_probabilities(counts, alphabet):
    """Normalised probabilities on established symbols plus an escape action."""
    denom = sum(counts.values()) + (len(alphabet) + 1) / 2
    return {v: (counts.get(v, 0) + .5) / denom for v in alphabet}, .5 / denom


def record_loss(state, record):
    known = state.keys
    new = sorted(set(record) - set(known))
    # Integer and literal codes are prefix codes, including newly encountered keys.
    common = integer_bits(len(new)) + sum(string_bits(k) + string_bits(record[k]) for k in new)
    rows = [set(range(len(state.records)))] + [v.members for v in state.nodes.values()]
    weights = [len(r) + .5 for r in rows]
    expert_bits = []
    for members in rows:
        loss = 0.0
        for key in known:
            counts = Counter(state.records[r][key] for r in members if key in state.records[r])
            p = (sum(counts.values()) + .5) / (len(members) + 1)
            loss -= math.log2(p if key in record else 1 - p)
            if key in record:
                alphabet = sorted({r[key] for r in state.records if key in r})
                probabilities, escape = value_probabilities(counts, alphabet)
                value = record[key]
                loss -= math.log2(probabilities.get(value, escape))
                if value not in probabilities:
                    loss += string_bits(value)
        expert_bits.append(loss)
    best = min(expert_bits)
    mixture = sum(w * 2 ** (best - b) for w, b in zip(weights, expert_bits)) / sum(weights)
    return common + best - math.log2(mixture)
