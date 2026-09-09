"""Historical key-merge controls and two preserved failed semantic claims.

After the symbol-alignment correction, disjoint vocabularies can determine their
original key spelling and can still compress. The two stronger semantic assertions
are marked as expected failures and retained as negative results, not coding guarantees.

The key merge must compare two descriptions of the SAME data.

Folding two keys into one makes the merged key's presence column denser, and one dense column is
much cheaper than two sparse ones. That saving has nothing to do with what the two keys say, so if
the merged description is allowed to forget which spelling each record used, the operator merges
every pair it is offered. Measured on E55's fixture before the fix: two keys with completely
disjoint vocabularies merged at every frequency tested, and the false merges grew worse as the keys
got commoner, reaching -156 bits at 256 records each.

The reason is that such a merged state is not a description of the input. A reader cannot tell
whether record 7 said `shade` or `colour`, so it cannot reconstruct the data, and a shorter
description that loses the data is not a shorter description.

These tests fail if the spelling column is ever dropped or weakened.
"""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from escrow.batch import BatchObjective
from escrow.keymerge import key_merge_delta, spelling_column_bits
from escrow.protocol import run_stream

VOCAB_A = [f"a{i}" for i in range(8)]
VOCAB_B = [f"b{i}" for i in range(8)]


def stream(overlap, freq=128, bulk=400, seed=0):
    """Two probe keys on disjoint records, sharing `overlap` of their vocabulary."""
    rng = random.Random(seed)
    shared = int(round(overlap * len(VOCAB_A)))
    vocab_b = VOCAB_A[:shared] + VOCAB_B[shared:]
    recs = [{"kind": rng.choice(["shirt", "dress", "coat"]),
             "fabric": rng.choice(["cotton", "silk", "wool"]),
             "fit": rng.choice(["slim", "regular", "loose"])} for _ in range(bulk)]
    recs += [{"kind": rng.choice(["shirt", "dress", "coat"]),
              "probe_a": rng.choice(VOCAB_A)} for _ in range(freq)]
    recs += [{"kind": rng.choice(["shirt", "dress", "coat"]),
              "probe_b": rng.choice(vocab_b)} for _ in range(freq)]
    rng.shuffle(recs)
    g, _ = run_stream(recs)
    return g


def delta(g, decompose=False):
    return key_merge_delta(g, BatchObjective, g.keys["probe_a"].kid, g.keys["probe_b"].kid,
                           decompose=decompose)


class KeyMergeIsLossless(unittest.TestCase):

    @unittest.expectedFailure
    def test_disjoint_vocabularies_never_merge(self):
        """Known invalid semantic claim: disjoint symbols can determine their spelling.

        Preserved as a negative result. The original passing result used unrelated
        key-local integer identifiers as shared symbols. See test_keymerge_symbols.py.
        """
        for freq in (8, 32, 128, 256):
            g = stream(overlap=0.0, freq=freq)
            d = delta(g)
            self.assertGreater(d, 0.0,
                               f"keys with disjoint vocabularies merged at {freq} records each "
                               f"({d:+.1f} bits); the spelling column is missing or too cheap")

    def test_identical_vocabularies_do_merge(self):
        """And the operator must still be able to say yes, or it is just a refusal."""
        g = stream(overlap=1.0, freq=128)
        self.assertLess(delta(g), 0.0, "two names for one role failed to merge")

    @unittest.expectedFailure
    def test_the_spelling_column_is_what_decides_it(self):
        """Known historical claim invalidated by correcting symbol alignment."""
        g = stream(overlap=0.0, freq=256)
        parts = delta(g, decompose=True)
        self.assertGreater(parts["delta_bits"], 0.0)
        without = parts["after_without_spelling"] - parts["before"]
        self.assertLess(without, 0.0,
                        "the fixture no longer reproduces the defect, so this test guards nothing")
        self.assertGreater(parts["spelling_column"], abs(without),
                           "the spelling column no longer covers the presence saving")

    def test_a_record_carrying_both_keys_blocks_the_merge(self):
        """Structural, with no level to set: a record with both keys is evidence of two keys."""
        rng = random.Random(0)
        recs = [{"printer": rng.choice(["laser", "inkjet"]),
                 "probe_a": rng.choice(VOCAB_A),
                 "probe_b": rng.choice(VOCAB_A)} for _ in range(200)]
        g, _ = run_stream(recs)
        self.assertIsNone(delta(g), "a co-occurring pair was priced instead of being refused")

    def test_the_charge_is_symmetric_in_the_two_keys(self):
        """Merging a into b and b into a describe the same state, so they cost the same."""
        g = stream(overlap=0.5, freq=128)
        a, b = g.keys["probe_a"].kid, g.keys["probe_b"].kid
        self.assertAlmostEqual(spelling_column_bits(g, a, b),
                               spelling_column_bits(g, b, a), places=9)
        self.assertAlmostEqual(key_merge_delta(g, BatchObjective, a, b),
                               key_merge_delta(g, BatchObjective, b, a), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
