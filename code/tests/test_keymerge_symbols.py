"""Regression for independently interned value identifiers across keys."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from escrow.keymerge import spelling_column_bits


def graph(reverse):
    inv_a = {"red": 1, "blue": 2}
    inv_b = {"blue": 1, "red": 2} if reverse else dict(inv_a)
    result = SimpleNamespace(keys={"a": SimpleNamespace(inventory=inv_a),
                                   "b": SimpleNamespace(inventory=inv_b)},
                             key_name={1: "a", 2: "b"}, record_keys={}, record_vals={})
    for kid, inv in ((1, inv_a), (2, inv_b)):
        for value in ["red"] * 40 + ["blue"] * 10:
            row = len(result.record_keys)
            result.record_keys[row] = {kid}
            result.record_vals[row] = {kid: inv[value]}
    return result


class SymbolTests(unittest.TestCase):
    def test_local_identifier_permutation_cannot_change_spelling_price(self):
        self.assertAlmostEqual(spelling_column_bits(graph(False), 1, 2),
                               spelling_column_bits(graph(True), 1, 2), places=12)


if __name__ == "__main__":
    unittest.main()
