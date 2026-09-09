"""Evaluation-only coordinated parent changes at the observed overlap local minimum."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(Path(__file__).resolve().parent)]
from h07_residual_branch_code import require_gate
from h10_deeper_branch_diagnosis import restore, checked
from escrow.hierarchy.benchmarks import intersecting
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.branch_codec import MAGIC
from escrow.hierarchy.moves import propose, route_to
from escrow.provenance import experiment_stamp


def run(args):
    require_gate(args.coding_gate)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a fresh output path")
    audit = json.loads(Path(args.observed).read_text())
    snapshot = audit["final_audit_snapshot"]
    b = intersecting(snapshot["prefix"], args.seed)
    state = restore(b.records, snapshot)
    obj = BranchDescription(b.records)
    def find(members):
        found = [nid for nid, node in state.nodes.items() if node.members == members]
        if len(found) != 1:
            raise ValueError("diagnostic membership is not uniquely represented")
        return found[0]
    joint = find(b.members["joint"])
    outside = {name: find(b.members[name] - b.members["joint"]) for name in ("a0", "b0")}
    complement = find(set(range(len(b.records))) - b.members["joint"])
    negative = {name: find(b.members[name]) for name in ("a1", "b1")}
    def insert(start, name):
        offset = 0 if name == "a0" else 3
        return propose(start, "introduce_parent", children=[joint, outside[name]],
            support=["k" + str(k) for k in range(offset, offset + 3)], route=True, inherit_support=True)
    states = dict(current=state, insert_a0=insert(state, "a0"), insert_b0=insert(state, "b0"))
    both = insert(states["insert_a0"], "b0")
    a0, b0 = state.next_id, state.next_id + 1
    assert both.nodes[joint].parents == {a0, b0}
    states["both_parents"] = both
    released = both
    for name in ("a1", "b1"):
        released = propose(released, "remove_link", child=negative[name], parent=complement)
        states["both_release_through_" + name] = released
    compact = released
    for nid in outside.values():
        compact = propose(compact, "delete", node=nid)
    for nid in (a0, b0, *negative.values(), joint, complement):
        route_to(compact, nid)
    compact.validate()
    states["coordinated_six_node_alternative"] = compact
    results = {name: checked(obj, candidate) for name, candidate in states.items()}
    result = dict(code=MAGIC.decode(), states=results,
        provenance=experiment_stamp(b.records, range(len(b.records)), dict(code=MAGIC.decode(),
            seed=args.seed, use_of_truth="evaluation-only memberships and supports for coordinated alternatives",
            source=args.observed, source_sha256=hashlib.sha256(Path(args.observed).read_bytes()).hexdigest()), [__file__]),
        note="These alternatives are not supplied to the learner. They diagnose a barrier in the evaluated neighbourhood, not a globally optimal or unique taxonomy.")
    out.write_text(json.dumps(result, indent=2) + "\n")
    base = results["current"]["description"]["total"]
    print(json.dumps({name: dict(bits=value["description"]["total"], delta=value["description"]["total"] - base)
        for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-gate", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())
