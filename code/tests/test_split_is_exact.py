"""The split and the residual mint must price exactly the state they create.

Every earlier operator bug in this project had the same shape: a delta that described a state the
engine did not actually build. The key merge scored a merged state that had forgotten which spelling
each record used, and merged everything. E47 and E48 published a conclusion drawn from a hand-built
state the engine never produced. So the standard for a new move is not that it improves a number, it
is that the number it predicts equals the change in `BatchObjective.total` to floating-point
tolerance, on a state the engine built itself.

These tests fail if a split or a residual mint ever prices a state other than the one it applies.
"""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

from escrow import batch as B
from escrow.batch import BatchObjective
from escrow.protocol import run_stream
from e75_a_fixture_that_tests_the_values import stream as value_stream


class no_split_moves:
    """Build the state the repair pass used to leave, so the operator's own arithmetic can be
    checked against it. With the moves wired in, run_stream now empties the residual itself, which
    is the point of them and would otherwise leave these tests with nothing to price."""

    def __enter__(self):
        self.was = B.SPLIT_MOVES
        B.SPLIT_MOVES = False

    def __exit__(self, *a):
        B.SPLIT_MOVES = self.was
        return False
from escrow.split import (apply_residual_mint, apply_split, best_residual_mint, best_split,
                          grow_seed, residual_mint_delta, residual_records, split_candidates,
                          split_delta, split_naming_charge, split_proposals, two_means_seed)

TOL = 1e-6


def fused(n=600, groups=4, keys=4, per=6, seed=0):
    """Groups told apart only by their values, over one shared set of keys."""
    rng = random.Random(seed)
    vocab = [f"v{i}" for i in range(groups * per)]
    sl = {g: vocab[g * per:(g + 1) * per] for g in range(groups)}
    recs, truth = [], []
    for _ in range(n):
        g = rng.randrange(groups)
        recs.append({f"k{j}": rng.choice(sl[g]) for j in range(keys)})
        truth.append(g)
    return recs, truth


def noise(n=400, keys=4, vals=10, seed=0):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(vals)}" for j in range(keys)} for _ in range(n)]


class SplitPricesWhatItBuilds(unittest.TestCase):
    def test_split_delta_equals_the_change_in_the_objective(self):
        recs, _ = fused(seed=1)
        with no_split_moves():
            g, _ = run_stream(recs)
        obj = BatchObjective(g)
        checked = 0
        for v in sorted(g.nodes.values(), key=lambda x: -x.t):
            cands = split_candidates(obj, v, cap=6)
            if not cands:
                continue
            grown = grow_seed(obj, v.members, cands[0][1])
            if not grown or len(grown) == len(v.members):
                continue
            before = obj.total()
            predicted = split_delta(obj, v, grown)
            apply_split(obj, v, grown)
            self.assertAlmostEqual(obj.total() - before, predicted, delta=TOL,
                                   msg="the split priced a state it did not build")
            checked += 1
            if checked == 2:
                break
        self.assertGreater(checked, 0, "no split was exercised, so nothing was tested")

    def test_the_two_halves_own_exactly_the_cells_the_parent_owned(self):
        recs, _ = fused(seed=2)
        with no_split_moves():
            g, _ = run_stream(recs)
        obj = BatchObjective(g)
        v = max(g.nodes.values(), key=lambda x: x.t)
        cands = split_candidates(obj, v, cap=4)
        if not cands:
            self.skipTest("no candidate cell on this stream")
        grown = grow_seed(obj, v.members, cands[0][1])
        before = {kid: dict(b.counts) for kid, b in v.blocks.items()}
        w = apply_split(obj, v, grown)
        for kid, counts in before.items():
            joint = dict(v.blocks.get(kid).counts) if kid in v.blocks else {}
            for vid, c in (w.blocks.get(kid).counts.items() if kid in w.blocks else ()):
                joint[vid] = joint.get(vid, 0) + c
            self.assertEqual(joint, counts,
                             "a value cell was lost or duplicated by the split")

    def test_a_split_is_refused_when_there_is_nothing_to_split(self):
        g, _ = run_stream(noise(seed=3))
        obj = BatchObjective(g)
        for v in g.nodes.values():
            self.assertIsNone(best_split(obj, v, cap=10),
                              "a split was accepted on a stream with no structure")


class ResidualMintPricesWhatItBuilds(unittest.TestCase):
    def test_residual_mint_delta_equals_the_change_in_the_objective(self):
        # This fixture is chosen because the engine leaves 681 records in no node and a mint out
        # of them pays about 1,020 bits. A stream where nothing pays would let the test pass by
        # never running the arithmetic it exists to check.
        recs, _ = fused(n=2000, groups=6, keys=6, per=8, seed=1)
        with no_split_moves():
            g, _ = run_stream(recs)
        obj = BatchObjective(g)
        res = residual_records(g)
        self.assertGreater(len(res), 20, "this stream left no residual to mint from")
        best = best_residual_mint(obj, cap=8)
        self.assertIsNotNone(best, "the residual mint did not fire, so nothing was checked")
        _, members, _, S = best
        before = obj.total()
        predicted, S2, _ = residual_mint_delta(obj, members, S)
        apply_residual_mint(obj, members, S2)
        self.assertAlmostEqual(obj.total() - before, predicted, delta=TOL,
                               msg="the residual mint priced a state it did not build")

    def test_the_residual_is_exactly_the_records_in_no_node(self):
        recs, _ = fused(n=500, seed=5)
        with no_split_moves():
            g, _ = run_stream(recs)
        res = residual_records(g)
        for v in g.nodes.values():
            self.assertFalse(res & v.members, "a member of a node was called residual")
        covered = set()
        for v in g.nodes.values():
            covered |= v.members
        self.assertEqual(len(res) + len(covered), g.n)


class OnlyTheLastPassMoves(unittest.TestCase):
    """The moves must fire on a caller's deliberate final pass and not on the schedule's own.

    Twenty-seven experiments hand-roll their protocol and end with `repair(full=True)`. If the moves
    needed a second flag, every one of them would silently run without them, which is the defect
    shape this project keeps finding: correct code with no caller. The signature already separates
    the two cases, because a self-scheduled full pass arrives as full=None.
    """

    def _count_moves(self, **kw):
        recs, _ = value_stream(1.0, 0, n=900)
        with no_split_moves():
            g, _ = run_stream(recs)
        before = int(g.K)
        BatchObjective(g).repair(**kw)
        return before, int(g.K)

    def test_an_explicit_full_pass_moves(self):
        before, after = self._count_moves(full=True)
        self.assertGreater(after, before,
                           "a deliberate final pass did not run the split")

    def test_a_caller_can_still_ask_for_no_moves(self):
        before, after = self._count_moves(full=True, final=False)
        self.assertEqual(after, before,
                         "final=False did not suppress the moves")

    def test_the_schedules_own_full_pass_does_not_move(self):
        # full=None is how the power-of-two checkpoint arrives, and it must not split: mid-stream a
        # node is still accumulating. Measured on the encyclopedia, moving there costs 32 bits.
        recs, _ = value_stream(1.0, 0, n=900)
        with no_split_moves():
            g, _ = run_stream(recs)
        before = int(g.K)
        g.last_full_repair_n = 0            # force the schedule to choose a full pass
        BatchObjective(g).repair()
        self.assertEqual(int(g.K), before,
                         "the schedule's own full pass ran the split")


class TheNamingChargeIsTheCorrection(unittest.TestCase):
    def test_naming_grows_with_the_number_of_candidates(self):
        self.assertLess(split_naming_charge(4), split_naming_charge(400))
        self.assertAlmostEqual(split_naming_charge(1), 0.0, delta=TOL)

    def test_a_split_that_pays_by_less_than_its_naming_charge_is_refused(self):
        # The value stream at 900 records, where the engine leaves five nodes and a split of the
        # largest pays about 1,490 bits. `fused` accepts no split at any seed, so a test written on
        # it passes without ever running the arithmetic.
        recs, _ = value_stream(1.0, 0, n=900)
        with no_split_moves():
            g, _ = run_stream(recs)
        obj = BatchObjective(g)
        checked = 0
        for v in sorted(g.nodes.values(), key=lambda x: -x.t)[:3]:
            b = best_split(obj, v, cap=8)
            if b is None:
                continue
            raw = split_delta(obj, v, b[1])
            # the charge names the winner out of every proposal actually offered, which is what
            # `split_proposals` reports, so the test cannot drift from the operator
            _, naming = split_proposals(obj, v, cap=8)
            self.assertAlmostEqual(b[0], raw + naming, delta=TOL,
                                   msg="the accepted split did not carry its naming charge")
            checked += 1
        self.assertGreater(checked, 0, "no split was accepted, so the charge was never checked")

    def test_the_two_means_proposal_reaches_a_bipartition_the_cells_do_not(self):
        # On this stream the winning proposal IS the two-means one, at about -1,490 bits, which no
        # cell-grown candidate matched. That is the whole reason a second proposal is offered.
        recs, _ = value_stream(1.0, 0, n=900)
        with no_split_moves():
            g, _ = run_stream(recs)
        obj = BatchObjective(g)
        v = max(g.nodes.values(), key=lambda x: x.t)
        part = two_means_seed(obj, v.members)
        self.assertIsNotNone(part, "the two-means proposal never fired, so nothing was tested")
        self.assertTrue(0 < len(part) < len(v.members),
                        "the two-means proposal returned a degenerate bipartition")
        self.assertTrue(part <= v.members, "it proposed records the node does not hold")
        b = best_split(obj, v, cap=64)
        self.assertIsNotNone(b, "no split was accepted on a stream where one pays")
        self.assertIn(b[2][0], ("two-means", "binary-two-means"),
                      "a bipartition proposal used to win here; if a cell now wins, re-anchor "
                      "this test rather than deleting it")

    def test_the_two_means_proposal_is_deterministic(self):
        recs, _ = value_stream(1.0, 0, n=900)
        with no_split_moves():
            g, _ = run_stream(recs)
        obj = BatchObjective(g)
        v = max(g.nodes.values(), key=lambda x: x.t)
        self.assertEqual(two_means_seed(obj, v.members),
                         two_means_seed(obj, v.members),
                         "the proposal is not deterministic, so a run is not reproducible")


if __name__ == "__main__":
    unittest.main(verbosity=2)
