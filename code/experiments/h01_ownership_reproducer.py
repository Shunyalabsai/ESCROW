"""Reproduce the historical 3,000-record defect and price the identical assignments."""
from __future__ import annotations

import copy
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(Path(__file__).resolve().parent)]
from escrow import batch as legacy_batch
from escrow.batch import BatchObjective
from escrow.codes import ValueBlock
from escrow.protocol import run_stream
from escrow.provenance import experiment_stamp
from escrow.hierarchy.model import HierarchyState, HierarchyNode
from escrow.hierarchy.coding import score, kt_bits
from escrow.hierarchy.codec import encode, decode, MAGIC
from e75_a_fixture_that_tests_the_values import stream


def convert(graph, records):
    state = HierarchyState()
    for record in records:
        state.append(record)
    for nid, node in graph.nodes.items():
        state.nodes[nid] = HierarchyNode(nid, {r - 1 for r in node.members},
                                         {graph.key_name[k] for k in node.S})
    state.next_id = max(state.nodes, default=0) + 1
    for rid, row in graph.record_owner.items():
        state.owners[rid - 1] = {graph.key_name[k]: owner for k, owner in row.items()}
    return state.validate()


def run(output=None, code="EHR03"):
    if code == "EHR05":
        from escrow.hierarchy.branch_coding import score
        from escrow.hierarchy.branch_codec import encode, decode, MAGIC
    elif code == "EHR03":
        from escrow.hierarchy.coding import score
        from escrow.hierarchy.codec import encode, decode, MAGIC
    else:
        raise ValueError("unknown reference code")
    original, _ = stream(0., 0, n=3000)
    order = list(range(len(original)))
    random.Random(1).shuffle(order)
    records = [original[i] for i in order]
    old_split = legacy_batch.SPLIT_MOVES
    legacy_batch.SPLIT_MOVES = False
    try:
        graph, objective = run_stream(records)
    finally:
        legacy_batch.SPLIT_MOVES = old_split
    empty_graph = copy.deepcopy(graph)
    empty_graph.nodes, empty_graph.K = {}, 0
    for ki in empty_graph.keys.values():
        ki.background = ValueBlock()
    for row in graph.record_vals.values():
        for kid, vid in row.items():
            empty_graph.key_by_id[kid].background.observe(vid)
    converted = convert(graph, records)
    empty = HierarchyState()
    for record in records:
        empty.append(record)
    actual_owners = sum(owner != 0 for row in converted.owners for owner in row.values())
    route_bits = score(converted)["ownership"]
    result = dict(historical_nodes=graph.K, node_owned_cells=actual_owners,
                  background_cells=sum(map(len, records)) - actual_owners,
                  historical_empty_bits=BatchObjective(empty_graph).total(),
                  historical_selected_bits=objective.total(),
                  explicit_route_bits=route_bits,
                  historical_plus_route_bits=objective.total() + route_bits,
                  reference_empty=score(empty), reference_selected=score(converted),
                  reference_serialised_empty_bits=8 * len(encode(empty)),
                  reference_serialised_selected_bits=8 * len(encode(converted)))
    result["roundtrip"] = decode(encode(converted)).canonical() == converted.canonical()
    result["defect_reproduced"] = result["historical_selected_bits"] < result["historical_empty_bits"]
    result["routing_correction_rejects"] = result["historical_plus_route_bits"] > result["historical_empty_bits"]
    result["reference_rejects"] = result["reference_selected"]["total"] > result["reference_empty"]["total"]
    result["provenance"] = experiment_stamp(original, order, dict(signal=0, seed=0,
        legacy_split_moves=False, repair_every=100, legacy_final_repair=True,
        reference_code=MAGIC.decode()), [__file__])
    out = Path(output) if output else ROOT / "results" / "hierarchy_reference" / "h01_ownership.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError("results are immutable; choose a fresh output directory")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2))
    assert all(result[k] for k in ("roundtrip", "defect_reproduced", "routing_correction_rejects",
                                  "reference_rejects"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--code", choices=("EHR03", "EHR05"), default="EHR03")
    args = parser.parse_args()
    run(args.output, args.code)
