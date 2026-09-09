"""Coding gate and exhaustive descriptions on every small binary record multiset."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from escrow.hierarchy.exact import minimum_description
from escrow.hierarchy.engine import HierarchicalEscrowGraph
from escrow.hierarchy.codec import encode, decode, MAGIC
from escrow.hierarchy.coding import score
from escrow.provenance import experiment_stamp


def run(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("choose a new result path")
    provenance = experiment_stamp([], [], dict(max_records=4, max_nodes=3, max_keys=2,
        values_per_key=2, state_budget=None, repair_every=100, candidate_budget=4096), [__file__])
    suite = unittest.defaultTestLoader.discover(str(ROOT / "code" / "tests"), pattern="test_hierarchy*.py")
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful():
        raise AssertionError("coding tests failed; exhaustive and broader runs were not started")
    started = time.perf_counter()
    # Nine records: each of two keys is absent, zero, or one.
    rows = [{k: value for k, value in zip(("a", "b"), values) if value is not None}
            for values in itertools.product((None, "0", "1"), repeat=2)]
    results = []
    for n in range(5):
        for instance in itertools.combinations_with_replacement(range(len(rows)), n):
            records = [rows[i] for i in instance]
            exact = minimum_description(records, max_nodes=3)
            g = HierarchicalEscrowGraph()
            for record in records:
                g.process(record)
            primary = score(g.state)["total"]
            g.final_repair()
            reached = score(g.state)["total"]
            recovered = decode(encode(exact["state"]))
            assert recovered.canonical() == exact["state"].canonical()
            results.append(dict(instance=list(instance), complete=exact["complete"],
                exact_bits=exact["description"]["total"], primary_bits=primary,
                final_repair_bits=reached, final_repair_gap_bits=reached - exact["description"]["total"],
                exact_nodes=len(exact["state"].nodes), evaluated=exact["evaluated"],
                branches_pruned_by_lower_bound=exact["branches_pruned_by_lower_bound"]))
        print(f"completed all multisets of {n} records", flush=True)
    result = dict(code=MAGIC.decode(), unit_tests=tests.testsRun, unit_tests_pass=True,
        instances=results, count=len(results), exhaustive_complete=all(r["complete"] for r in results),
        max_final_repair_gap_bits=max(r["final_repair_gap_bits"] for r in results),
        searched_descriptions=sum(r["evaluated"] for r in results),
        elapsed_seconds=time.perf_counter() - started,
        scope="All multisets of zero to four records over two optional binary keys, zero to three nodes, all valid parent sets and all slot owners. Record permutations have the same description optimum; streaming search order is evaluated separately.",
        provenance=provenance)
    result["passed"] = result["exhaustive_complete"] and result["max_final_repair_gap_bits"] < 1e-8
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("instances", "provenance")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
