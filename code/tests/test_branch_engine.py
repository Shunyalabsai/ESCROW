"""The EHR05 engine must accept, report and serialise under the same code."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from escrow.hierarchy import BranchEscrowGraph
from escrow.hierarchy.branch_codec import encode, decode
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.engine import HierarchicalEscrowGraph
from escrow.hierarchy.branch_proposals import candidates, parent_arrangements
from escrow.hierarchy.moves import propose
from escrow.hierarchy.benchmarks import intersecting


class BranchEngineTests(unittest.TestCase):
    def test_joint_parent_arrangements_can_withdraw_old_links(self):
        fixture = intersecting(400, 0)
        graph = BranchEscrowGraph(repair_every=1000)
        for record in fixture.records:
            graph.state.append(record)
        graph.state = propose(graph.state, "create", members=fixture.members["joint"], support=graph.state.keys)
        outside = set(range(400)) - fixture.members["joint"]
        graph.state = propose(graph.state, "create", members=outside, support=["k6", "k7", "k8"])
        for members, support in ((fixture.members["a1"], ["k0", "k1", "k2"]),
                                 (fixture.members["b1"], ["k3", "k4", "k5"]),
                                 (fixture.members["b0"] - fixture.members["joint"], ["k0", "k1", "k2", "k3", "k4", "k5"]),
                                 (fixture.members["a0"] - fixture.members["joint"], ["k0", "k1", "k2"])):
            graph.state = propose(graph.state, "refine", parent=2, members=members, support=support)
        objective = BranchDescription(graph.state.records)
        before = objective.score(graph.state)["total"]
        found = False
        for operation, arguments in parent_arrangements(graph.state):
            after = propose(graph.state, operation, **arguments)
            if len(after.nodes[1].parents) == 2 and objective.score(after)["total"] < before:
                self.assertLess(len(encode(after)), len(encode(graph.state)))
                self.assertEqual(after.canonical(), decode(encode(after)).canonical())
                found = True
                break
        self.assertTrue(found)
        self.assertEqual(list(parent_arrangements(graph.state, allow_links=False)), [])
        self.assertEqual(list(parent_arrangements(graph.state, fixed_ancestry=True)), [])

    def test_shared_structure_can_reuse_an_existing_parent(self):
        graph = BranchEscrowGraph(repair_every=1000)
        for r in range(120):
            graph.process({"common": str(r % 2), "x": str(r // 60), "y": str(r // 60)})
        graph.state = propose(graph.state, "create", members=list(range(120)), support=["common"])
        for members in (range(60), range(60, 120)):
            graph.state = propose(graph.state, "refine", parent=1, members=members, support=graph.state.keys)
        before = graph.describe(graph.state)["total"]
        reuse = []
        for operation, arguments in candidates(graph.state):
            if operation == "compound":
                after = propose(graph.state, operation, **arguments)
                if set(after.nodes) == set(graph.state.nodes) and graph.describe(after)["total"] < before:
                    reuse.append(after)
        self.assertTrue(reuse)
        self.assertTrue(any(all(s.owners[r]["common"] == 1 for r in range(120)) for s in reuse))

    def test_refinement_can_reuse_global_membership_with_conditional_keys(self):
        graph = BranchEscrowGraph(repair_every=1000)
        for r in range(160):
            graph.process({key: str(r // 80) for key in "abc"} |
                          {key: str(r // 40) for key in "defghi"})
        for members in (range(80), range(80, 160)):
            graph.state = propose(graph.state, "create", members=members, support=list("abcdefghi"))
        before = graph.describe(graph.state)["total"]
        paying = []
        for operation, arguments in candidates(graph.state):
            if operation == "refine" and arguments["support"] == list("defghi"):
                after = propose(graph.state, operation, **arguments)
                if graph.describe(after)["total"] < before and len(encode(after)) < len(encode(graph.state)):
                    paying.append(after)
        self.assertTrue(paying)
        for state in paying:
            self.assertEqual(state.canonical(), decode(encode(state)).canonical())

    def test_reference_acceptance_uses_its_own_description(self):
        graph = BranchEscrowGraph(repair_every=100)
        for r in range(80):
            graph.process({key: str(r // 40) for key in "abcdef"})
        self.assertFalse(graph.events)
        before = graph.describe(graph.state)
        old_bytes = len(encode(graph.state))
        graph.repair(proposals=[("create", dict(members=list(range(40)), support=list("abcdef")))])
        self.assertEqual(len(graph.events), 1)
        event = graph.events[0]
        after = BranchDescription(graph.state.records).score(graph.state)
        for part in before:
            self.assertAlmostEqual(event["delta_bits"][part], after[part] - before[part])
        self.assertEqual(event["serialised_delta_bits"], 8 * (len(encode(graph.state)) - old_bytes))
        self.assertEqual(graph.state.canonical(), decode(encode(graph.state)).canonical())
        snapshot = graph.snapshot()
        self.assertEqual(snapshot["protocol"]["code"], "EHR05")
        self.assertIn("membership_patterns", snapshot["description_bits"])
        self.assertAlmostEqual(snapshot["predictive_loss_bits"], sum(graph.arrival_losses))
        self.assertEqual(HierarchicalEscrowGraph().snapshot()["protocol"]["code"], "EHR03")


if __name__ == "__main__":
    unittest.main()
