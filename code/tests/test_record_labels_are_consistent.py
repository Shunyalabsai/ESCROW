"""Every experiment must turn a cover into record labels the same way.

The engine returns a cover, so a record can sit in several nodes, while ARI and the other
single-label metrics want one label each. The rule, written once in escrow.protocol.record_labels,
is that a record takes the node with the most members that contains it, with the lowest node id
breaking a tie, and a record in no node takes -1.

The rule was originally written out by hand in sixteen experiments and thirteen of them left the
loop unsorted, so the label of a multi-node record fell to dictionary order. That is not a harmless
difference. On the 320 record encyclopedia stream it moves 9 records, and across sixty arrival
orders it changes the reported agreement in 27 of them, moving the mean from 0.8764 to 0.8691 and
the shortest-of-three headline from 0.9367 to 0.9281. Two numbers printed side by side in the paper
were computed under different rules because of it.

These two tests keep that from coming back: the rule still does what it says, and no experiment
writes its own unsorted copy.
"""
import ast
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
LOOP = "for v in g.nodes.values():"
ASSIGN = "lab[m - 1] = v.nid"


class Node:
    def __init__(self, nid, members):
        self.nid, self.members, self.t = nid, set(members), len(members)


class Graph:
    def __init__(self, nodes):
        self.nodes = {n.nid: n for n in nodes}


class TestTheRule(unittest.TestCase):
    def test_largest_node_wins_and_lowest_id_breaks_the_tie(self):
        import sys
        sys.path.insert(0, CODE)
        from escrow.protocol import record_labels
        # record 1 is in all three nodes; node 7 is the largest, so it takes 7
        g = Graph([Node(7, [1, 2, 3, 4]), Node(3, [1, 2]), Node(9, [1])])
        self.assertEqual(record_labels(g, 5)[0], 7)
        # record 5 is in no node, so it joins the one background cluster
        self.assertEqual(record_labels(g, 5)[4], -1)
        # two nodes of equal size: the lower id wins
        g2 = Graph([Node(8, [1, 2]), Node(2, [1, 2])])
        self.assertEqual(record_labels(g2, 2)[0], 2)

    def test_ties_do_not_depend_on_insertion_order(self):
        import sys
        sys.path.insert(0, CODE)
        from escrow.protocol import record_labels
        a = Graph([Node(8, [1, 2]), Node(2, [1, 2])])
        b = Graph([Node(2, [1, 2]), Node(8, [1, 2])])
        self.assertEqual(record_labels(a, 2), record_labels(b, 2))


class TestNoExperimentWritesItsOwn(unittest.TestCase):
    def test_no_unsorted_label_loop_anywhere(self):
        offenders = []
        for root, _, files in os.walk(CODE):
            if "__pycache__" in root:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(root, f)
                with open(p, encoding="utf-8") as fh:
                    lines = fh.read().split("\n")
                for i, ln in enumerate(lines):
                    if ln.strip() == LOOP and ASSIGN in "\n".join(lines[i:i + 3]):
                        offenders.append(f"{os.path.relpath(p, CODE)}:{i + 1}")
        self.assertEqual(offenders, [],
                         "these build record labels without sorting by (t, -nid), so a record in "
                         "more than one node gets whichever label dictionary order happens to "
                         "leave last. Call escrow.protocol.record_labels instead: " +
                         ", ".join(offenders))


if __name__ == "__main__":
    unittest.main()
