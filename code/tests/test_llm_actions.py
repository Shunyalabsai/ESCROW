"""Offline protocol tests. No model, account, network request or paid call is used."""
import collections
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments import llm_graph_formation as harness


class ActionTests(unittest.TestCase):
    def test_background_deferral_and_parse_errors_are_distinct(self):
        nodes, counts, decisions = {}, collections.Counter(), {}
        objects = harness.parse_answer('[{"record":0,"action":"background"},'
            '{"record":1,"action":"defer"},{"record":2,"node":"unknown"}]')
        assignments = harness.apply_answer(objects, nodes, [0, 1, 2], counts, 0, decisions)
        self.assertEqual(assignments, {0: None, 1: None})
        self.assertEqual(decisions, {0: "background", 1: "defer", 2: "parse_failure"})
        self.assertEqual(counts["parse_failure_records"], 1)
        self.assertEqual(nodes, {})

    def test_explicit_create_and_attach(self):
        nodes, counts, decisions = {}, collections.Counter(), {}
        harness.apply_answer([dict(record=0, node="a", new=True, description="recurring")],
                              nodes, [0], counts, 0, decisions)
        assignment = harness.apply_answer([dict(record=1, node="a")], nodes, [1], counts, 1, decisions)
        self.assertEqual(assignment, {1: "a"})
        self.assertEqual(len(nodes), 1)
        self.assertEqual(decisions, {0: "created", 1: "assigned"})

    def test_invalid_response_is_not_abstention(self):
        counts, decisions = collections.Counter(), {}
        harness.apply_answer(harness.parse_answer("not JSON"), {}, [0], counts, 0, decisions)
        self.assertEqual(decisions[0], "parse_failure")
        self.assertEqual(counts["deferred_records"], 0)
        self.assertEqual(counts["background_records"], 0)

    def test_streaming_default_and_prompt(self):
        self.assertEqual(harness.CHUNK, 1)
        prompt = harness.make_prompt({}, [{"key": "value"}], 0)
        self.assertIn('"background"', prompt)
        self.assertIn('"defer"', prompt)


if __name__ == "__main__":
    unittest.main()
