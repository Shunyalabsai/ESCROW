"""Independent reconstruction and predictive checks for EHR05 and its ablations."""
import itertools
import math
import os
import random
import sys
import unittest

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.dirname(__file__)]
from test_hierarchy_code import example
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.branch_codec import encode, decode
from escrow.hierarchy.coding import score as old_score
from escrow.hierarchy.model import HierarchyState
from escrow.hierarchy.moves import propose, route_to


class BranchCodeTests(unittest.TestCase):
    def test_compound_refinement_crosses_a_costly_intermediate(self):
        state = HierarchyState()
        for r in range(120):
            state.append({"common": "c", "x": "a" if r < 60 else "b", "y": "a" if r < 60 else "b"})
        state.add_node(range(120), state.keys)
        route_to(state, 1)
        objective = BranchDescription(state.records)
        first = ("refine", dict(parent=1, members=list(range(60)), support=["x", "y"]))
        second = ("refine", dict(parent=1, members=list(range(60, 120)), support=["x", "y"]))
        single = propose(state, first[0], **first[1])
        compound = propose(state, "compound", moves=[first, second])
        sequential = propose(single, second[0], **second[1])
        self.assertEqual(compound.canonical(), sequential.canonical())
        self.assertGreater(objective.score(single)["total"], objective.score(state)["total"])
        self.assertLess(objective.score(compound)["total"], objective.score(state)["total"])
        self.assertLess(len(encode(compound)), len(encode(state)))
        self.assertEqual(decode(encode(compound)).canonical(), compound.canonical())
        self.assertEqual(len(state.nodes), 1)
        with self.assertRaises(ValueError):
            propose(state, "compound", moves=[("compound", dict(moves=[]))])

    def test_balanced_exposure_repairs_the_blocked_sibling_counterexample(self):
        state = HierarchyState()
        for r in range(120):
            state.append({"common": "c", "x": "a" if r < 60 else "b",
                          "y": "a" if r < 60 else "b"})
        state.add_node(range(120), {"common"})
        state.add_node(range(60), {"x", "y"}, [1])
        state.add_node(range(60, 120), {"x", "y"}, [1])
        for nid in state.nodes:
            route_to(state, nid)
        collapsed = propose(state, "merge", nodes=[1, 2])
        old = BranchDescription(state.records, balanced_records=False)
        new = BranchDescription(state.records)
        self.assertLess(old.score(collapsed)["total"], old.score(state)["total"])
        self.assertGreater(new.score(collapsed)["total"], new.score(state)["total"])
        order = new.data_order(collapsed, collapsed.topological())
        self.assertEqual({int(r >= 60) for r in order[:2]}, {0, 1})
        self.assertEqual(collapsed.canonical(), decode(encode(collapsed)).canonical())

    def test_all_ablations_roundtrip_with_overlap_and_multiple_parents(self):
        for seed, joint, trained, balanced in itertools.product(range(8), (False, True), (False, True), (False, True)):
            state = example(seed)
            data = encode(state, joint, trained, balanced)
            self.assertEqual(decode(data).canonical(), state.canonical())
            bits = BranchDescription(state.records, joint, trained, balanced).score(state)["total"]
            self.assertLess(abs(8 * len(data) - bits), 10.)

    def test_disabled_ablations_reproduce_ehr03_ideal_code(self):
        for seed in range(8):
            state = example(seed)
            old = old_score(state)
            new = BranchDescription(state.records, False, False).score(state)
            for key in old:
                self.assertAlmostEqual(new[key] - (8 if key in ("total", "header") else 0), old[key], places=8)

    def test_empty_records_and_empty_graph(self):
        state = HierarchyState()
        for record in ({}, {"new": "é"}, {}, {"new": "x", "late": ""}):
            state.append(record)
            self.assertEqual(decode(encode(state)).canonical(), state.canonical())
        self.assertEqual(decode(encode(HierarchyState())).canonical(), HierarchyState().canonical())

    def test_normalisation_when_unowned_observations_update_the_parent(self):
        # Fix the graph, owners, keys and vocabulary. Sum the categorical data code
        # over every binary presence/value sequence. It must remain a probability law.
        # Background is not used; therefore no fitted header counts enter this check.
        state = HierarchyState()
        for _ in range(3):
            state.append({"k": "0"})
        state.add_node(range(3), {"k"})
        state.add_node([0, 2], {"k"}, [1])
        state.owners = [{"k": 2}, {"k": 1}, {"k": 2}]
        total = 0.
        for outcomes in itertools.product((None, "0", "1"), repeat=3):
            records = [({"k": v} if v is not None else {}) for v in outcomes]
            state.records = records
            obj = BranchDescription(records)
            # Condition on an externally fixed alphabet for this component test.
            obj.keys, obj.vocab, obj.hist = ["k"], {"k": ["0", "1"]}, {"k": {"0": 0, "1": 0}}
            terms = obj.score(state, validate=False)
            total += 2 ** -(terms["presence"] + terms["values"])
        self.assertAlmostEqual(total, 1., places=12)

    def test_move_deltas_and_record_order_are_explicit(self):
        state = example(3)
        obj = BranchDescription(state.records)
        for op, args in [("remove_link", dict(child=3, parent=2)),
                         ("ownership", dict(assignments=[(18, "a", 0)])),
                         ("introduce_parent", dict(children=[1, 2], support=[]))]:
            other = propose(state, op, **args)
            diff = obj.score(other)["total"] - obj.score(state)["total"]
            self.assertLess(abs(diff - 8 * (len(encode(other)) - len(encode(state)))), 20.)
        # Alphabet and graph stay fixed, but all-member predictive histories use arrival order.
        permutation = list(reversed(range(len(state.records))))
        other = state.clone()
        other.records = [state.records[i] for i in permutation]
        other.owners = [state.owners[i] for i in permutation]
        inverse = {old: new for new, old in enumerate(permutation)}
        for node in other.nodes.values():
            node.members = {inverse[r] for r in node.members}
        self.assertEqual(decode(encode(other)).canonical(), other.canonical())
        self.assertNotAlmostEqual(obj.score(state)["total"], BranchDescription(other.records).score(other)["total"], places=4)


if __name__ == "__main__":
    unittest.main()
