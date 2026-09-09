"""A separately labelled finite search audit; never a replacement streaming result."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(Path(__file__).resolve().parent)]
from h07_residual_branch_code import require_gate
from h10_deeper_branch_diagnosis import restore
from escrow.hierarchy.benchmarks import intersecting
from escrow.hierarchy.branch_engine import BranchEscrowGraph
from escrow.hierarchy.branch_codec import encode, decode, MAGIC
from escrow.hierarchy.evaluation import measure
from escrow.provenance import experiment_stamp


def run(args):
    require_gate(args.coding_gate)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a fresh output path")
    original = json.loads(Path(args.observed).read_text())
    if original["fixture"] != "07_intersecting_multiple_parents":
        raise ValueError("this audit is for the intersecting fixture")
    source = original["final_repair"]
    b = intersecting(source["prefix"], original["metadata"]["seed"])
    graph = BranchEscrowGraph()
    graph.state = restore(b.records, source)
    graph.predictive_loss_bits = source["predictive_loss_bits"]
    provenance = experiment_stamp(b.records, range(len(b.records)), dict(code=MAGIC.decode(),
        phase="additional repair audit starting after the separately reported final repair",
        candidate_budget=4096, extra_repairs_cap=args.repairs, defaults_changed=False,
        primary_source=args.observed, source_sha256=hashlib.sha256(Path(args.observed).read_bytes()).hexdigest()), [__file__])
    checkpoints = []
    for i in range(args.repairs):
        report = graph.final_repair()
        assert decode(encode(graph.state)).canonical() == graph.state.canonical()
        row = dict(additional_repair=i + 1, search=report, description=graph.describe(graph.state),
                   metrics=measure(graph.state, b.members, b.parents))
        checkpoints.append(row)
        result = dict(provenance=provenance, source_final_bits=source["description_bits"]["total"],
            checkpoints=checkpoints, final_audit_snapshot=graph.snapshot(),
            primary_result_replaced=False, more_arrivals_observed=0)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(repair=i + 1, nodes=len(graph.state.nodes),
            bits=row["description"]["total"], budget_exhausted=report["search_budget_exhausted"],
            links=row["metrics"]["direct_links"])), flush=True)
        if not report["search_budget_exhausted"]:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-gate", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repairs", type=int, default=4)
    run(parser.parse_args())
