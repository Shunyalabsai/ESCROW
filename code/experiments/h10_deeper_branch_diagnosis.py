"""Evaluation-only depth and overlap counterfactuals after failed structural searches."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(Path(__file__).resolve().parent)]
from h04_nesting_diagnosis import planted_state
from h07_residual_branch_code import require_gate
from escrow.hierarchy.benchmarks import tree, intersecting
from escrow.hierarchy.model import HierarchyState, HierarchyNode
from escrow.hierarchy.moves import propose, route_to
from escrow.hierarchy.branch_coding import BranchDescription
from escrow.hierarchy.branch_codec import encode, decode, MAGIC
from escrow.provenance import experiment_stamp


def restore(records, snapshot):
    return HierarchyState(records, {n["id"]: HierarchyNode(n["id"], set(n["members"]),
        set(n["support"]), set(n["parents"])) for n in snapshot["nodes"]},
        snapshot["ownership"], max(n["id"] for n in snapshot["nodes"]) + 1).validate()


def checked(objective, state):
    parts = objective.score(state)
    data = encode(state)
    assert decode(data).canonical() == state.canonical()
    assert abs(8 * len(data) - parts["total"]) < 10
    return dict(description=parts, serialised_bits=8 * len(data))


def run(args):
    require_gate(args.coding_gate)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a fresh output path")
    td, md = [json.loads(Path(path).read_text()) for path in (args.depth3, args.multiple)]
    b = tree(3, td["primary"]["prefix"], td["metadata"]["seed"])
    obj = BranchDescription(b.records)
    primary, final = [restore(b.records, td[phase]) for phase in ("primary", "final_repair")]
    full, _ = planted_state(b, "hierarchy")
    rows = []
    mapping = td["final_repair_metrics"]["node_matching"]
    for nid, node in final.nodes.items():
        label = mapping.get(str(nid))
        if label is None or len(label) != 2 or node.members != b.members[label]:
            continue
        children = [label + "0", label + "1"]
        singles = [propose(final, "refine", parent=nid, members=b.members[c], support=b.records[0].keys() & {"k6", "k7", "k8"}) for c in children]
        paired = final
        for c in children:
            paired = propose(paired, "refine", parent=nid, members=b.members[c], support=["k6", "k7", "k8"])
        trimmed = propose(paired, "support", node=nid, support=sorted(node.support - {"k6", "k7", "k8"}))
        variants = {children[0]: singles[0], children[1]: singles[1], "pair": paired, "pair_trimmed": trimmed}
        rows.append(dict(parent=label, alternatives={name: checked(obj, state) for name, state in variants.items()}))
    depth = dict(primary=checked(obj, primary), final_repair=checked(obj, final), full=checked(obj, full),
        refinements=rows, provenance=experiment_stamp(b.records, range(len(b.records)),
            dict(code=MAGIC.decode(), truth_use="diagnostic alternatives only", observed=args.depth3), [__file__]))

    b = intersecting(md["primary"]["prefix"], md["metadata"]["seed"])
    obj = BranchDescription(b.records)
    full = HierarchyState()
    for record in b.records:
        full.append(record)
    ids = {}
    for name in ("a0", "a1", "b0", "b1", "joint"):
        offset = 0 if name.startswith("a") else 3 if name.startswith("b") else 6
        ids[name] = full.add_node(b.members[name], {"k" + str(k) for k in range(offset, offset + 3)},
            [ids[p] for p in b.parents[name]])
        route_to(full, ids[name])
    complement = propose(full, "create", members=sorted(set(range(len(b.records))) - b.members["joint"]),
                         support=["k6", "k7", "k8"])
    nested = complement
    for name in ("a1", "b1"):
        nested = propose(nested, "add_link", child=ids[name], parent=complement.next_id - 1)
    overlap = dict(primary=checked(obj, restore(b.records, md["primary"])),
        final_repair=checked(obj, restore(b.records, md["final_repair"])),
        planted_five=checked(obj, full), with_complement=checked(obj, complement),
        with_complement_parent=checked(obj, nested),
        note="The fixture gives non-joint records a shared disjoint vocabulary on three keys. Its complement is observable structure omitted from the five reference labels; unmatched is not proof of unsupported structure.",
        provenance=experiment_stamp(b.records, range(len(b.records)), dict(code=MAGIC.decode(),
            truth_use="diagnostic alternatives only", observed=args.multiple), [__file__]))
    out.write_text(json.dumps(dict(depth3=depth, multiple_parents=overlap), indent=2) + "\n")
    print(json.dumps(dict(depth3={name: depth[name]["description"]["total"] for name in ("primary", "final_repair", "full")},
        child_deltas=[dict(parent=r["parent"], deltas={k: v["description"]["total"] - depth["final_repair"]["description"]["total"] for k, v in r["alternatives"].items()}) for r in rows],
        overlap={name: overlap[name]["description"]["total"] for name in ("primary", "final_repair", "planted_five", "with_complement", "with_complement_parent")}), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-gate", required=True)
    parser.add_argument("--depth3", required=True)
    parser.add_argument("--multiple", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
