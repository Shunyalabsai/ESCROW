"""Diagnose explicit sibling suppression using evaluation-only counterfactual graphs."""
from __future__ import annotations

import argparse
import json
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from escrow.hierarchy.benchmarks import tree
from escrow.hierarchy.model import HierarchyState, HierarchyNode
from escrow.hierarchy.moves import propose, route_to
from escrow.hierarchy.coding import score, delta, binary_bits
from escrow.hierarchy.codec import encode, decode, MAGIC
from escrow.provenance import experiment_stamp


def planted_state(benchmark, mode):
    state = HierarchyState()
    for record in benchmark.records:
        state.append(record)
    ids = {}
    for name in sorted(benchmark.members, key=lambda v: (len(v), v)):
        if mode == "empty" or (mode == "parents" and len(name) != 1) or (mode == "leaves" and len(name) != 2):
            continue
        support = set(state.keys) if mode == "leaves" else {
            "k" + str(k) for k in range(3 * (len(name) - 1), 3 * len(name))}
        parents = [ids[name[:-1]]] if mode == "hierarchy" and len(name) > 1 else []
        nid = state.add_node(benchmark.members[name], support, parents)
        ids[name] = nid
        route_to(state, nid)
    return state, ids


def run(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a new result path")
    provenance = experiment_stamp([], [], dict(code=MAGIC.decode(), use_of_truth="evaluation-only counterfactuals",
        sizes=[40, 100, 400, 1000, 4000], seed=0), [__file__])
    result = dict(code=MAGIC.decode(), diagnostic_only=True, sizes=[], provenance=provenance)
    for n in (40, 100, 400, 1000, 4000):
        b = tree(2, n=n, seed=0)
        descriptions = {}
        for mode in ("empty", "parents", "leaves", "hierarchy"):
            state, _ = planted_state(b, mode)
            descriptions[mode] = score(state)
        explicit, ids = planted_state(b, "hierarchy")
        collapsed = propose(explicit, "merge", nodes=[ids["0"], ids["00"]])
        collapsed = propose(collapsed, "merge", nodes=[ids["1"], ids["11"]])
        descriptions["one_child_implicit_per_parent"] = score(collapsed)
        assert decode(encode(explicit)).canonical() == explicit.canonical()
        assert decode(encode(collapsed)).canonical() == collapsed.canonical()
        result["sizes"].append(dict(n=n, descriptions=descriptions,
            input_sha256=hashlib.sha256(json.dumps(b.records, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
            arrival_order=list(range(n)), fixture_metadata=b.metadata,
            collapsed_minus_explicit=delta(explicit, collapsed),
            serialised_collapsed_minus_explicit=8 * (len(encode(collapsed)) - len(encode(explicit))),
            removed_membership_bits=sum(binary_bits(len(explicit.nodes[ids[c]].members),
                len(explicit.nodes[ids[c[0]]].members)) for c in ("00", "11"))))
    if args.observed:
        observed = json.loads(Path(args.observed).read_text())
        snapshot = observed["primary"]
        b = tree(2, n=snapshot["prefix"], seed=0)
        state = HierarchyState()
        for record in b.records:
            state.append(record)
        state.nodes = {v["id"]: HierarchyNode(v["id"], set(v["members"]), set(v["support"]),
                                              set(v["parents"])) for v in snapshot["nodes"]}
        state.next_id = max(state.nodes, default=0) + 1
        state.owners = snapshot["ownership"]
        state.validate()
        result["observed_run"] = dict(source=str(args.observed), current_code_description=score(state),
            recorded_description=snapshot["description_bits"],
            primary_metrics=observed["primary_metrics"],
            final_repair_metrics=observed["final_repair_metrics"],
            repair_log=snapshot["repair_log"],
            translation=("Same binary format." if observed["provenance"]["configuration"]["code"] == MAGIC.decode()
                else "EHR03 adds a common 64-bit length frame to the EHR02 descriptions."))
    result["structural_gate_passed"] = False
    result["diagnosis"] = (
        "The current objective can strictly prefer omitting a supported sibling. Its data blocks "
        "survive under the parent as residual-owned cells, while its conditional membership column "
        "and node disappear. Extra ownership cost grows logarithmically while saved membership "
        "cost can grow linearly. Increasing the proposal budget cannot force the complete explicit "
        "hierarchy to win this objective. This is a representation and objective failure, not an "
        "ownership-accounting omission.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(structural_gate_passed=False, sizes=[dict(n=r["n"],
        collapsed_minus_explicit=r["collapsed_minus_explicit"]) for r in result["sizes"]]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--observed")
    run(parser.parse_args())
