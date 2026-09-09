"""Factorial coding diagnosis after the coding gate, before any large campaign."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(Path(__file__).resolve().parent)]
from h04_nesting_diagnosis import planted_state
from escrow.hierarchy.benchmarks import tree
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.branch_codec import encode, decode, MAGIC
from escrow.hierarchy.moves import propose
from escrow.provenance import experiment_stamp


def require_gate(path):
    gate = json.loads(Path(path).read_text())
    if not gate.get("passed") or gate.get("code") != MAGIC.decode():
        raise RuntimeError("the current branch coding gate must pass first")
    for module in ("model", "codec", "branch_codec", "branch_coding", "moves"):
        name = "code/escrow/hierarchy/" + module + ".py"
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != gate["provenance"]["source_sha256"].get(name):
            raise RuntimeError("the coding gate is stale for " + name)


def reorder(state, order):
    result = state.clone()
    inverse = {old: new for new, old in enumerate(order)}
    result.records = [state.records[i] for i in order]
    result.owners = [state.owners[i] for i in order]
    for node in result.nodes.values():
        node.members = {inverse[r] for r in node.members}
    return result


def run(args):
    require_gate(args.coding_gate)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a fresh output path")
    provenance = experiment_stamp([], [], dict(code=MAGIC.decode(),
        input_role="independent constructed streams identified per row", sizes=[40, 100, 400, 1000, 4000],
        seed=0, arrival_orders="original, reverse, shuffle-1, leaf-blocked; scored separately", use_of_truth="counterfactual graphs and stress-test arrival orders only",
        balanced_records=True,
        ablations="both, member predictors only, joint membership only, neither"), [__file__])
    rows = []
    for n in (40, 100, 400, 1000, 4000):
        benchmark = tree(2, n=n, seed=0)
        full, ids = planted_state(benchmark, "hierarchy")
        flat, _ = planted_state(benchmark, "leaves")
        collapsed = {}
        for a, b in itertools.product(("00", "01"), ("10", "11")):
            candidate = propose(full, "merge", nodes=[ids["0"], ids[a]])
            collapsed[a + "_" + b] = propose(candidate, "merge", nodes=[ids["1"], ids[b]])
        leaf = {r: name for name, members in benchmark.members.items() if len(name) == 2 for r in members}
        orders = {"original": list(range(n)), "reverse": list(reversed(range(n))), "shuffle_1": list(range(n)),
                  "leaf_blocked": sorted(range(n), key=lambda r: leaf[r])}
        random.Random(1).shuffle(orders["shuffle_1"])
        for order_name, order in orders.items():
            states = {"explicit": reorder(full, order), "flat": reorder(flat, order)}
            states.update({"collapsed_" + name: reorder(state, order) for name, state in collapsed.items()})
            for joint in (False, True):
                for trained in (False, True):
                    objective = BranchDescription(states["explicit"].records, joint, trained)
                    parts = {name: objective.score(state) for name, state in states.items()}
                    actual = {name: 8 * len(encode(state, joint, trained)) for name, state in states.items()}
                    best = min((name for name in states if name.startswith("collapsed_")),
                               key=lambda name: parts[name]["total"])
                    for name, state in states.items():
                        assert decode(encode(state, joint, trained)).canonical() == state.canonical()
                        assert abs(actual[name] - parts[name]["total"]) < 10
                    rows.append(dict(n=n, arrival_order_name=order_name, arrival_order=order,
                        input_sha256=hashlib.sha256(json.dumps(benchmark.records, sort_keys=True,
                            separators=(",", ":")).encode()).hexdigest(),
                        joint_membership=joint, member_predictors=trained, descriptions=parts,
                        serialised_bits=actual, best_collapsed=best, collapsed_minus_explicit={
                            k: parts[best][k] - parts["explicit"][k] for k in parts["explicit"]},
                        flat_minus_explicit=parts["flat"]["total"] - parts["explicit"]["total"]))
        print(f"counterfactuals complete at {n} records", flush=True)
    result = dict(code=MAGIC.decode(), provenance=provenance, rows=rows,
        note="Diagnostic graphs use truth only for evaluation. Orders are separate descriptions, not pooled evidence or a shortest-replay selection.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([dict(n=r["n"], order=r["arrival_order_name"],
        collapsed_minus_explicit=r["collapsed_minus_explicit"]["total"], flat_minus_explicit=r["flat_minus_explicit"])
        for r in rows if r["joint_membership"] and r["member_predictors"]], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-gate", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
