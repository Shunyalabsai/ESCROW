"""Separate a refinement proposal failure from a coding barrier on the depth-two audit."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(Path(__file__).resolve().parent)]
from h04_nesting_diagnosis import planted_state
from h07_residual_branch_code import require_gate
from escrow.hierarchy.benchmarks import tree
from escrow.hierarchy.model import HierarchyState, HierarchyNode
from escrow.hierarchy.moves import propose
from escrow.hierarchy.proposals import candidates
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.branch_codec import encode, decode, MAGIC
from escrow.provenance import experiment_stamp


def run(args):
    require_gate(args.coding_gate)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a fresh output path")
    observed = json.loads(Path(args.observed).read_text())
    if observed["fixture"] != "02_tree_depth_2":
        raise ValueError("this diagnosis is for the depth-two audit")
    snapshot = observed["primary"]
    b = tree(2, snapshot["prefix"], observed["metadata"]["seed"])
    state = HierarchyState(b.records,
        {n["id"]: HierarchyNode(n["id"], set(n["members"]), set(n["support"]), set(n["parents"]))
         for n in snapshot["nodes"]}, snapshot["ownership"],
        1 + max(n["id"] for n in snapshot["nodes"]))
    state.validate()
    objective = BranchDescription(b.records)
    before = objective.score(state)
    rows = []
    for label, members in sorted(b.members.items()):
        if len(label) != 2:
            continue
        for parent in state.nodes:
            if not members < state.nodes[parent].members:
                continue
            for support in (state.keys, state.keys[3:]):
                after = propose(state, "refine", parent=parent, members=members, support=support)
                parts = objective.score(after)
                assert decode(encode(after)).canonical() == after.canonical()
                rows.append(dict(label=label, parent=parent, members=len(members), support=support,
                    delta={k: parts[k] - before[k] for k in before},
                    serialised_delta_bits=8 * (len(encode(after)) - len(encode(state)))))
    generated = []
    for op, arguments in candidates(state):
        if op not in ("create", "refine"):
            continue
        members = set(arguments["members"])
        mismatch, label = min((len(members ^ truth), name) for name, truth in b.members.items() if len(name) == 2)
        after = propose(state, op, **arguments)
        generated.append(dict(operation=op, arguments=arguments, nearest_leaf=label,
            symmetric_difference=mismatch, delta_bits=objective.score(after)["total"] - before["total"]))
    full, _ = planted_state(b, "hierarchy")
    result = dict(code=MAGIC.decode(), observed_source=args.observed,
        observed_sha256=hashlib.sha256(Path(args.observed).read_bytes()).hexdigest(),
        current_description=before, explicit_description=objective.score(full),
        exact_child_alternatives=rows, generated_candidates=generated,
        provenance=experiment_stamp(b.records, range(len(b.records)), dict(code=MAGIC.decode(),
            use_of_truth="evaluation-only counterfactual memberships and mismatch metrics",
            learner_input="records only"), [__file__]))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(current_bits=before["total"], explicit_bits=objective.score(full)["total"],
        exact_children=[dict(label=r["label"], support=r["support"], delta=r["delta"]["total"]) for r in rows],
        nearest_generated=[dict(operation=r["operation"], nearest_leaf=r["nearest_leaf"],
            symmetric_difference=r["symmetric_difference"], delta_bits=r["delta_bits"])
            for r in sorted(generated, key=lambda r: r["symmetric_difference"])[:4]]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-gate", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
