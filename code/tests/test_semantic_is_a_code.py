"""The semantic arm must be a code, not a score. These are the properties that make that true.

If any of these fails, the construction has become a similarity with a threshold wearing a
codelength for a hat, and the central claim of the work is gone.
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from escrow.semantic import (SCALE_GRID, Embedding, block_naming_bits, mixture_bits,
                             semantic_naming_bits)


def toy_embedding():
    """Three colours near each other, two sizes near each other, the groups far apart."""
    return Embedding({
        "red":     [1.0, 0.0, 0.0], "crimson": [0.98, 0.20, 0.0], "scarlet": [0.96, 0.28, 0.0],
        "small":   [0.0, 0.0, 1.0], "tiny":    [0.0, 0.20, 0.98],
    })


class MixtureIsAValidCode(unittest.TestCase):

    def test_never_worse_than_the_better_arm_by_more_than_one_bit(self):
        """The entire argument for adding semantics at all."""
        for a in (10.0, 500.0, 40_000.0):
            for b in (10.0, 500.0, 40_000.0, 1e6):
                m = mixture_bits(a, b)
                self.assertLessEqual(m, min(a, b) + 1.0 + 1e-9,
                                     f"mixture of {a} and {b} cost {m}")
                self.assertGreaterEqual(m, min(a, b) - 1e-9)

    def test_survives_the_codelengths_a_real_stream_produces(self):
        """A naive 2**-L implementation underflows to zero here and returns inf."""
        m = mixture_bits(120_000.0, 121_000.0)
        self.assertTrue(math.isfinite(m))
        self.assertLessEqual(m, 120_001.0)

    def test_kraft_holds_over_the_alphabet(self):
        """The semantic naming charge must be a distribution over the same alphabet the shipped
        charge uses: the inventory plus one slot for a value outside it. If it sums past 1 it is
        buying its saving by cheating rather than by predicting."""
        emb = toy_embedding()
        inv = ["red", "crimson", "scarlet", "small", "tiny"]
        for seen in ([], ["red"], ["red", "crimson"], ["small"]):
            mass = sum(2.0 ** -semantic_naming_bits(emb, v, seen, inv) for v in inv)
            self.assertLessEqual(mass, 1.0 + 1e-9,
                                 f"naming mass {mass} exceeds 1 with seen={seen}")


class SemanticsOnlyHelpsWhereThereIsStructure(unittest.TestCase):

    def test_a_near_value_is_cheaper_than_a_far_one(self):
        emb = toy_embedding()
        inv = ["red", "crimson", "scarlet", "small", "tiny"]
        near = semantic_naming_bits(emb, "crimson", ["red"], inv)
        far = semantic_naming_bits(emb, "small", ["red"], inv)
        self.assertLess(near, far, "the embedding is not being used")

    def test_the_first_value_costs_exactly_the_shipped_charge(self):
        """With nothing seen there is nothing to be near, so the two codes must agree."""
        emb = toy_embedding()
        inv = ["red", "crimson", "scarlet", "small", "tiny"]
        self.assertAlmostEqual(semantic_naming_bits(emb, "red", [], inv),
                               math.log2(len(inv) + 1.0), places=9)

    def test_an_unknown_embedding_degrades_to_the_shipped_charge(self):
        """A value the cache has never seen must not be quietly treated as near everything."""
        emb = Embedding({})
        inv = [f"v{i}" for i in range(10)]
        sem = block_naming_bits(emb, ["v1", "v2", "v3"], inv, semantic=True)
        cat = block_naming_bits(emb, ["v1", "v2", "v3"], inv, semantic=False)
        self.assertAlmostEqual(sem, cat, places=6)

    def test_on_arbitrary_symbols_semantics_earns_nothing(self):
        """The null. Values with no relation to each other must give the semantic arm no
        advantage, or it would start inventing structure in noise, which is the one thing this
        whole system exists not to do."""
        rng = random.Random(0)
        vals = [f"v{i}" for i in range(10)]
        emb = Embedding({v: [rng.gauss(0, 1) for _ in range(16)] for v in vals})
        seq = [rng.choice(vals) for _ in range(200)]
        sem = block_naming_bits(emb, seq, vals, semantic=True)
        cat = block_naming_bits(emb, seq, vals, semantic=False)
        self.assertGreater(sem, cat - 1.0,
                           "random vectors bought a saving, so the charge is not normalised")

    def test_the_scale_grid_is_a_mixture_and_not_a_choice(self):
        """Averaging over scales must not collapse to one scale, or a bandwidth has been picked."""
        emb = toy_embedding()
        inv = ["red", "crimson", "scarlet", "small", "tiny"]
        mixed = semantic_naming_bits(emb, "crimson", ["red"], inv)
        singles = []
        for beta in SCALE_GRID:
            w = {v: math.exp(-beta * (1.0 - emb.cos(v, "red"))) for v in inv}
            z = sum(w.values())
            keep = 1.0 / (len(inv) + 1.0)
            singles.append(-math.log2((1.0 - keep) * w["crimson"] / z))
        self.assertGreater(mixed, min(singles) - 1e-9)
        self.assertLess(mixed, max(singles) + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
