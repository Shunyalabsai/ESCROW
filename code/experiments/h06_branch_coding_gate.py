"""Branch-code reconstruction and exhaustive small-instance gate. No structural campaign."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.branch_codec import encode, decode, MAGIC
from escrow.hierarchy.branch_engine import BranchEscrowGraph
from escrow.hierarchy.exact import minimum_description
from escrow.provenance import experiment_stamp


def run(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError("choose a fresh result path")
    rows = [{k: v for k, v in zip(("a", "b"), values) if v is not None}
            for values in itertools.product((None, "0", "1"), repeat=2)]
    provenance = experiment_stamp(rows, range(len(rows)), dict(code=MAGIC.decode(),
        input_role="enumerated record universe; actual input hashes and orders are per instance",
        joint_membership=True, member_predictors=True, balanced_records=True, max_nodes=3, max_records=4,
        candidate_budget=4096, repair_every=100, exhaustive_state_budget=None), [__file__])
    suite = unittest.defaultTestLoader.discover(str(ROOT / "code" / "tests"), pattern="test_branch_code.py")
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful():
        raise AssertionError("branch-code unit gate failed")
    started, cases = time.perf_counter(), []
    for n in range(5):
        for instance in itertools.combinations_with_replacement(range(9), n):
            records = [rows[i] for i in instance]
            exact = minimum_description(records, description_cls=BranchDescription)
            g = BranchEscrowGraph()
            for record in records:
                g.process(record)
            primary = g.describe(g.state)["total"]
            g.final_repair()
            bits = g.describe(g.state)["total"]
            best = exact["state"]
            assert decode(encode(best)).canonical() == best.canonical()
            assert abs(8 * len(encode(best)) - exact["description"]["total"]) < 10
            cases.append(dict(instance=list(instance), input_sha256=hashlib.sha256(json.dumps(records,
                sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                arrival_order=list(range(n)), complete=exact["complete"], evaluated=exact["evaluated"],
                exact_nodes=len(best.nodes), exact_bits=exact["description"]["total"],
                primary_bits=primary, final_repair_bits=bits, gap_bits=bits - exact["description"]["total"],
                pruned_by_valid_lower_bound=exact["branches_pruned_by_lower_bound"]))
        print(f"{MAGIC.decode()} completed all multisets of {n} records", flush=True)
    result = dict(code=MAGIC.decode(), cases=cases, count=len(cases), unit_tests=tests.testsRun,
        exhaustive_complete=all(c["complete"] for c in cases), max_gap_bits=max(c["gap_bits"] for c in cases),
        all_optima_empty=all(c["exact_nodes"] == 0 for c in cases),
        evaluated_descriptions=sum(c["evaluated"] for c in cases),
        elapsed_seconds=time.perf_counter() - started, provenance=provenance)
    result["passed"] = result["exhaustive_complete"] and result["max_gap_bits"] < 1e-8
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "provenance")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
