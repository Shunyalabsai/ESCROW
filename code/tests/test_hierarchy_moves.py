"""Structural invariants and streaming protocol, independent of planted-label agreement."""
import math
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from escrow.hierarchy.model import HierarchyState
from escrow.hierarchy.moves import propose
from escrow.hierarchy.codec import encode, decode
from escrow.hierarchy.coding import score, delta
from escrow.hierarchy.engine import HierarchicalEscrowGraph
from escrow.hierarchy.predictive import value_probabilities
from escrow.hierarchy.proposals import candidates


def fixture():
    state = HierarchyState()
    for r in range(32):
        state.append({"a": str(r % 2), "b": str(r // 8)})
    state.add_node(range(24), {"a"})
    state.add_node(range(8, 32), {"b"})
    return state


class MoveTests(unittest.TestCase):
    def checked(self, before, op, **args):
        saved = before.canonical()
        after = propose(before, op, **args)
        self.assertEqual(before.canonical(), saved)
        self.assertEqual(after.canonical(), decode(encode(after)).canonical())
        d = delta(before, after)
        self.assertAlmostEqual(d["total"], score(after)["total"] - score(before)["total"])
        self.assertLess(abs(8 * (len(encode(after)) - len(encode(before))) - d["total"]), 20)
        return after

    def test_all_operations_and_ancestor_closure(self):
        s = fixture()
        s = self.checked(s, "refine", parent=1, parents=[2], members=list(range(10, 20)), support=["a"])
        self.assertEqual(s.nodes[3].parents, {1, 2})
        s = self.checked(s, "introduce_parent", children=[1, 2], support=[])
        self.assertEqual(s.nodes[4].members, set(range(32)))
        s = self.checked(s, "remove_link", child=3, parent=2)
        s = self.checked(s, "add_link", child=3, parent=2)
        s = self.checked(s, "reparent", child=3, parents=[2])
        s = self.checked(s, "membership", node=3, members=list(range(10, 22)), route=True)
        s = self.checked(s, "ownership", assignments=[(10, "a", 0)])
        s = self.checked(s, "support", node=3, support=["a", "b"], route=True)
        s = self.checked(s, "merge", nodes=[1, 2])
        s = self.checked(s, "delete", node=3)
        self.assertEqual(s.next_id, 5)
        s = self.checked(s, "create", members=[0], support=["a"])
        self.assertIn(5, s.nodes)

    def test_cycle_and_noncontainment(self):
        s = fixture()
        with self.assertRaises(ValueError):
            propose(s, "add_link", child=1, parent=2)
        s = propose(s, "refine", parent=1, members=[0, 1], support=["a"])
        with self.assertRaises(ValueError):
            propose(s, "add_link", child=1, parent=3)
        with self.assertRaises(ValueError):
            propose(s, "ownership", assignments=[(31, "a", 1)])

    def test_refinement_proposals_continue_beyond_four_levels(self):
        s = HierarchyState()
        for r in range(12):
            s.append({"a": str(r % 2)})
        for depth in range(8):
            s.add_node(range(12 - depth), {"a"}, [depth] if depth else [])
        self.assertEqual(s.canonical(), decode(encode(s)).canonical())
        self.assertTrue(any(op == "refine" and args["parent"] == 8
                            for op, args in candidates(s)))

    def test_predictive_actions_normalise(self):
        for counts in (Counter(), Counter(a=30, b=1), Counter(a=1)):
            probabilities, escape = value_probabilities(counts, ["a", "b"])
            self.assertAlmostEqual(sum(probabilities.values()) + escape, 1.)

    def test_fixed_ancestry_proposals_do_not_rewrite_surviving_links(self):
        state = fixture()
        state = propose(state, "refine", parent=1, members=[0, 1], support=["a"])
        state = propose(state, "refine", parent=2, members=[20, 21], support=["b"])
        old = {nid: set(v.parents) for nid, v in state.nodes.items()}
        for op, args in candidates(state, fixed_ancestry=True):
            try:
                candidate = propose(state, op, **args)
            except ValueError:
                continue
            for nid in set(old) & set(candidate.nodes):
                self.assertEqual(candidate.nodes[nid].parents, old[nid])

    def test_streaming_and_final_repair_separate(self):
        g = HierarchicalEscrowGraph(repair_every=3, candidate_budget=20)
        for rec in ({"a": "0"}, {"a": "1"}, {"a": "0"}, {"b": "new"}):
            g.process(rec)
        self.assertEqual(len(g.repair_log), 1)
        before = g.predictive_loss_bits
        g.final_repair()
        self.assertEqual(g.predictive_loss_bits, before)
        self.assertEqual(g.repair_log[-1]["phase"], "final_repair")
        self.assertAlmostEqual(sum(g.arrival_losses), g.predictive_loss_bits)
        with self.assertRaises(TypeError):
            g.process({"bad": 4})
        self.assertEqual(len(g.state.records), 4)

    def test_unresolved_tie_and_budget_reported(self):
        g = HierarchicalEscrowGraph(candidate_budget=1)
        g.state = fixture()
        # Two interchangeable parents with identical membership and no data ownership.
        g.state.nodes[1].members = set(range(24))
        g.state.nodes[2].members = set(range(24))
        g.state = propose(g.state, "create", members=list(range(12)), support=[])
        before = g.state.canonical()
        g.repair(proposals=[("reparent", dict(child=3, parents=[1])),
                            ("reparent", dict(child=3, parents=[2]))])
        self.assertTrue(g.repair_log[-1]["search_budget_exhausted"])
        # Full-budget comparison never resolves equal descriptions by node id.
        h = HierarchicalEscrowGraph(candidate_budget=10)
        h.state = fixture()
        h.state.nodes[1].members = set(range(24))
        h.state.nodes[2].members = set(range(24))
        h.state = propose(h.state, "create", members=list(range(12)), support=[])
        h.repair(proposals=[("reparent", dict(child=3, parents=[1])),
                            ("reparent", dict(child=3, parents=[2]))])
        self.assertEqual(h.state.nodes[3].parents, set())


if __name__ == "__main__":
    unittest.main()
