"""Constructed EHR05 search gate after the residual-branch coding audit."""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
import resource
import platform
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from escrow.hierarchy.benchmarks import suite
from escrow.hierarchy.branch_engine import BranchEscrowGraph
from h07_residual_branch_code import require_gate
from escrow.hierarchy.evaluation import measure
from escrow.hierarchy.branch_codec import encode, decode, MAGIC
from escrow.provenance import experiment_stamp


def run(args):
    require_gate(args.coding_gate)
    residual = json.loads(Path(args.residual_gate).read_text())
    if residual.get("code") != MAGIC.decode():
        raise RuntimeError("residual audit uses another format")
    rows = [r for r in residual["rows"] if r["n"] >= 400 and
            r["joint_membership"] and r["member_predictors"]]
    if len(rows) != 12 or any(r["collapsed_minus_explicit"]["total"] <= 0 or
                             r["flat_minus_explicit"] <= 0 for r in rows):
        raise RuntimeError("the residual counterexample remains unresolved")
    for name, digest in residual["provenance"]["source_sha256"].items():
        if name in {"code/escrow/hierarchy/" + module + ".py" for module in
                    ("model", "coding", "codec", "branch_coding", "branch_codec")} and hashlib.sha256(
                (ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("residual audit is stale for " + name)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("choose a new result directory")
    out.mkdir(parents=True)
    modes = dict(hierarchy={}, corrected_flat=dict(allow_links=False),
                 fixed_ancestry=dict(fixed_ancestry=True))
    summary = []
    for benchmark in suite(args.records, args.seed):
        if args.fixture is not None and not any(benchmark.name.startswith(f) for f in args.fixture.split(",")):
            continue
        for method in args.methods.split(","):
            g = BranchEscrowGraph(candidate_budget=args.budget, **modes[method])
            provenance = experiment_stamp(benchmark.records, range(len(benchmark.records)),
                dict(code=MAGIC.decode(), repair_every=100, candidate_budget=args.budget,
                     seed=args.seed, records=args.records, method=method,
                     trace_memory=args.trace_memory), [__file__])
            if args.trace_memory:
                tracemalloc.start()
            started = time.perf_counter()
            checkpoints = []
            for i, record in enumerate(benchmark.records):
                g.process(record)
                if (i + 1) % 100 == 0:
                    prefix_truth = {k: {r for r in members if r <= i} for k, members in benchmark.members.items()}
                    prefix_truth = {k: v for k, v in prefix_truth.items() if v}
                    checkpoints.append(dict(prefix=i + 1, metrics=measure(g.state, prefix_truth,
                        {c: ps for c, ps in benchmark.parents.items() if c in prefix_truth})))
                    print(f"{benchmark.name} {method} prefix={i+1} nodes={len(g.state.nodes)}", flush=True)
            primary = g.snapshot()
            metrics = measure(g.state, benchmark.members, benchmark.parents)
            primary_seconds = time.perf_counter() - started
            if args.trace_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_kind = "tracemalloc_peak_python_allocations"
            else:
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if platform.system() != "Darwin":
                    peak *= 1024
                memory_kind = "process_peak_resident_bytes; isolate fixtures for memory comparisons"
            assert decode(encode(g.state)).canonical() == g.state.canonical()
            g.final_repair()
            result = dict(fixture=benchmark.name, method=method, metadata=benchmark.metadata,
                primary=primary, primary_metrics=metrics, primary_seconds=primary_seconds,
                peak_memory_bytes=peak, memory_measurement=memory_kind,
                runtime_includes_memory_tracing=args.trace_memory,
                checkpoints=checkpoints, final_repair=g.snapshot(),
                final_repair_metrics=measure(g.state, benchmark.members, benchmark.parents),
                provenance=provenance)
            (out / (benchmark.name + "_" + method + ".json")).write_text(json.dumps(result, indent=2) + "\n")
            row = dict(fixture=benchmark.name, method=method, **metrics,
                       seconds=primary_seconds, peak_memory_bytes=peak, memory_measurement=memory_kind,
                       budget_exhaustions=sum(r["search_budget_exhausted"] for r in primary["repair_log"]))
            summary.append(row)
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--residual-gate", required=True)
    parser.add_argument("--records", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--methods", default="hierarchy")
    parser.add_argument("--fixture")
    parser.add_argument("--trace-memory", action="store_true")
    run(parser.parse_args())
