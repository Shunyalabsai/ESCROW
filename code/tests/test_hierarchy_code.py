"""Independent coding gates for the hierarchical categorical description."""
import itertools
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from escrow.hierarchy.model import HierarchyState
from escrow.hierarchy.coding import score, delta, kt_bits
from escrow.hierarchy.codec import encode, decode


def example(seed):
    rng = random.Random(seed)
    state = HierarchyState()
    for r in range(40):
        state.append({k: rng.choice(["red", "blue", "green"])
                      for k in ("a", "b", "c") if rng.random() < .7})
    a = state.add_node(range(30), {"a", "b"})
    b = state.add_node(range(10, 40), {"b", "c"})
    c = state.add_node(range(15, 25), {"a", "c"}, [a, b])
    for r in range(40):
        for key in state.keys:
            state.owners[r][key] = rng.choice(state.eligible_owners(r, key))
    return state


class ReferenceCodeTests(unittest.TestCase):
    def test_roundtrip_with_overlap_absence_and_two_parents(self):
        for seed in range(12):
            state = example(seed)
            encoded = encode(state)
            recovered = decode(encoded)
            self.assertEqual(state.canonical(), recovered.canonical())
            # Finite precision arithmetic termination and byte padding only.
            self.assertLess(abs(8 * len(encoded) - score(state)["total"]), 10)

    def test_empty_and_new_symbols(self):
        state = HierarchyState()
        for rec in ({}, {"a": ""}, {"a": "é", "new": "new"}, {}, {"late": "x"}):
            state.append(rec)
            self.assertEqual(decode(encode(state)).records, state.records)
        self.assertEqual(decode(encode(HierarchyState())).records, [])

    def test_kraft_small_binary_sequences(self):
        for n in range(1, 7):
            mass = sum(2 ** -kt_bits((n - sum(seq), sum(seq)))
                       for seq in itertools.product((0, 1), repeat=n))
            self.assertAlmostEqual(mass, 1, places=12)

    def test_move_delta_and_node_relabelling(self):
        state = example(3)
        other = state.clone()
        other.nodes[3].parents.remove(2)
        d = delta(state, other)
        self.assertAlmostEqual(d["total"], score(other)["total"] - score(state)["total"])
        before = encode(other)
        remap = {1: 73, 2: 9, 3: 108}
        for node in other.nodes.values():
            node.nid = remap[node.nid]
            node.parents = {remap[p] for p in node.parents}
        other.nodes = {v.nid: v for v in other.nodes.values()}
        other.owners = [{k: remap.get(o, 0) for k, o in row.items()} for row in other.owners]
        other.next_id = 109
        self.assertEqual(encode(other), before)

    def test_invalid_ownership_cycle_and_noncontainment_rejected(self):
        state = example(4)
        state.owners[39]["a"] = 1
        with self.assertRaises(ValueError):
            encode(state)
        state = example(4)
        state.nodes[1].parents.add(3)
        with self.assertRaises(ValueError):
            encode(state)
        state = example(4)
        state.nodes[3].members.add(0)
        with self.assertRaises(ValueError):
            encode(state)

    def test_corrupt_and_truncated_streams_rejected(self):
        data = encode(example(8))
        with self.assertRaises(ValueError):
            decode(data[:-1])
        with self.assertRaises(ValueError):
            decode(data[:8] + bytes([data[8] ^ 1]) + data[9:])


if __name__ == "__main__":
    unittest.main()
