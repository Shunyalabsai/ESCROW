"""E19: the candidate-pool budget curve.

THE OBJECTION. The method removes lambda, the new-node penalty, but it holds a candidate
pool at a declared operational budget (`EscrowGraph.cand_pool_cap`, engine.py:80, default
4096) and evicts the lowest-escrow account when the pool is full (engine.py:207-221). A
reviewer can say the parameter did not go away, it moved from the penalty to the budget.
The only evidence in the repository today is a two-point stress test in
code/tests/test_batch.py (no node at a cap of 1, eight nodes at a cap of 8, on the shuffled
planted stream), which is an extreme and not a curve, and answers the objection neither way.

WHAT THIS RUNS. The budget over
  {1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384, 32768}
on four streams, under the shipped protocol (escrow.protocol: BatchObjective(g).install(),
repair() every 100 records, one final repair(full=True)):
  twogroup   2,000 records, 2 planted groups, generator reseeded over seeds 0..4
  planted8   3,000 records, 8 planted groups at 10 percent value noise, seeds 0..4
  wikipedia  the 320 cached infoboxes (film 120, person 113, mountain 87); the record set
             is fixed, so a seed permutes the ARRIVAL ORDER, seeds 0..4, seed 0 being the
             E4 order
  lazada     the first 1,000 spec-bearing listings of the seed-0 shuffle of 21,365, read
             from results/llm_gf_runs/data_lazada.json with its silver of_type truth; the
             record set is fixed, so order 0 is the shipped file order and orders 1..4 are
             permutations of it

PER CELL. Final node count K, adjusted Rand index against the truth, mints, records left in
the background (no node), peak pool occupancy, exact eviction count, wall time.

READING. Two questions are answered in the report's `reading` block: the budget at which the
answer stops moving, per stream, and whether the shipped default of 4096 sits on the flat
part for every stream tested. "The answer" is the tuple (K, ARI, mints, background records);
the report also carries the stricter test, whether the node partition is identical to the
partition at the largest budget, seed by seed.

The eviction count is exact and the engine is untouched: `EscrowGraph.pool` is swapped for a
dict subclass that counts __delitem__, which the eviction branch uses (engine.py:219) and the
post-mint pop (engine.py:399) does not.

Run: python3 code/experiments/e19_budget_curve.py [--quick]
Writes results/e19_budget_curve.json.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.protocol import new_run, REPAIR_EVERY, describe as protocol_describe
from escrow.provenance import stamped
from experiments.e4_baseline_army import two_group, planted8, wikipedia, _ari

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e19_budget_curve.json")

# The requested grid, plus four refinement points. The first pass put the Wikipedia and
# Lazada plateau exactly at 4096, the shipped default, because the requested grid jumps
# 1024 -> 4096; 128, 512, 2048 and 3072 locate the plateau inside that jump, which is the
# number a practitioner actually wants.
REQUESTED_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384, 32768)
REFINEMENT_BUDGETS = (128, 512, 2048, 3072)
BUDGETS = tuple(sorted(set(REQUESTED_BUDGETS) | set(REFINEMENT_BUDGETS)))
SHIPPED_DEFAULT = 4096
SEEDS = (0, 1, 2, 3, 4)
LAZADA_JSON = os.path.join(RESULTS, "llm_gf_runs", "data_lazada.json")


# --------------------------------------------------------------------------- #
# exact eviction counting without touching the engine
# --------------------------------------------------------------------------- #
class _CountingPool(dict):
    """A dict that counts `del d[k]`. The engine evicts with `del self.pool[worst.sig]`
    and pops a released candidate with `self.pool.pop(c.sig, None)`; dict.pop does not
    route through __delitem__, so this counts evictions and only evictions."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.evictions = 0

    def __delitem__(self, k):
        self.evictions += 1
        super().__delitem__(k)


# --------------------------------------------------------------------------- #
# streams
# --------------------------------------------------------------------------- #
def _lazada(order_seed):
    with open(LAZADA_JSON) as fh:
        d = json.load(fh)
    recs, truth = d["records"], d["truth"]
    if order_seed == 0:                       # order 0 is the shipped file order
        return list(recs), list(truth)
    ix = list(range(len(recs)))
    random.Random(order_seed).shuffle(ix)
    return [recs[i] for i in ix], [truth[i] for i in ix]


def lazada_available():
    return os.path.exists(LAZADA_JSON)


STREAMS = {
    "twogroup": dict(
        make=lambda s: two_group(seed=s),
        seed_kind="stream generator seed (records resampled)",
        truth="planted group id, K* = 2",
        n=2000),
    "planted8": dict(
        make=lambda s: planted8(seed=s),
        seed_kind="stream generator seed (records resampled)",
        truth="planted group id, K* = 8, 10 percent value noise",
        n=3000),
    "wikipedia": dict(
        make=lambda s: wikipedia(RESULTS, seed=s),
        seed_kind="arrival-order shuffle seed (the 320 records are fixed); seed 0 is the E4 order",
        truth="infobox category (film, person, mountain), K* = 3",
        n=320),
    "lazada": dict(
        make=_lazada,
        seed_kind=("arrival-order seed (the 1,000 records are fixed); order 0 is the shipped "
                   "file order, orders 1 to 4 are permutations of it"),
        truth=("silver of_type from the AutoPKG KG, 759 types over 1,000 records; ARI is "
               "reported but is uninformative at that type count (see findings/TODO.md item 5)"),
        n=1000),
}


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #
def run_one(records, truth, cap):
    """One stream at one budget, under the shipped protocol. protocol.new_run gives the
    engine and the installed batch objective; the loop below is protocol.run_stream with a
    per-record pool observation, which the protocol docstring sanctions ('scripts that need
    per-record hooks call new_run and drive the loop themselves, but must keep the same
    three choices')."""
    g, b = new_run(cand_pool_cap=cap)
    g.pool = _CountingPool(g.pool)
    peak = 0
    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        g.process(rec)
        if len(g.pool) > peak:
            peak = len(g.pool)
        if (i + 1) % REPAIR_EVERY == 0:
            b.repair()
    b.repair(full=True)
    seconds = time.perf_counter() - t0

    labels = [-1] * len(records)               # background is one cluster, as in E4
    for v in g.nodes.values():
        for m in v.members:
            labels[m - 1] = v.nid
    parts = sorted(tuple(sorted(v.members)) for v in g.nodes.values())
    return dict(
        K=g.K,
        ARI=round(_ari(truth, labels), 6) if truth is not None else None,
        mints=len(g.mint_log),
        background=sum(1 for x in labels if x == -1),
        peak_pool=peak,
        evictions=g.pool.evictions,
        cap_bound=bool(g.pool.evictions > 0),
        seconds=round(seconds, 3),
        _partition=parts)


def _spread(vals):
    return dict(mean=round(sum(vals) / len(vals), 4), min=min(vals), max=max(vals))


def _answer(cell):
    """'The answer' for the stop-moving test: node count, ARI, mints, background."""
    return (cell["K"], cell["ARI"], cell["mints"], cell["background"])


# --------------------------------------------------------------------------- #
# one stream: the whole curve
# --------------------------------------------------------------------------- #
def sweep(name, budgets, seeds):
    cfg = STREAMS[name]
    streams = {s: cfg["make"](s) for s in seeds}
    cells = []
    for cap in budgets:
        per_seed = []
        for s in seeds:
            recs, truth = streams[s]
            r = run_one(recs, truth, cap)
            r["seed"] = s
            per_seed.append(r)
            print(f"    [{name}] budget={cap:<6d} seed={s} K={r['K']:<4d} "
                  f"ARI={r['ARI']} mints={r['mints']:<4d} bg={r['background']:<5d} "
                  f"peak={r['peak_pool']:<6d} evict={r['evictions']:<7d} {r['seconds']}s",
                  flush=True)
        cells.append(dict(
            budget=cap,
            per_seed=[{k: v for k, v in r.items() if k != "_partition"} for r in per_seed],
            _partitions=[r["_partition"] for r in per_seed],
            K=_spread([r["K"] for r in per_seed]),
            ARI=_spread([r["ARI"] for r in per_seed]),
            mints=_spread([r["mints"] for r in per_seed]),
            background=_spread([r["background"] for r in per_seed]),
            peak_pool=_spread([r["peak_pool"] for r in per_seed]),
            evictions=_spread([r["evictions"] for r in per_seed]),
            seeds_with_eviction=sum(1 for r in per_seed if r["cap_bound"]),
            seconds=_spread([r["seconds"] for r in per_seed]),
            seconds_total=round(sum(r["seconds"] for r in per_seed), 3)))

    # reference = the largest budget run
    ref = cells[-1]
    for c in cells:
        c["seeds_answer_equals_largest_budget"] = sum(
            1 for a, bb in zip(c["per_seed"], ref["per_seed"]) if _answer(a) == _answer(bb))
        c["seeds_partition_identical_to_largest_budget"] = sum(
            1 for a, bb in zip(c["_partitions"], ref["_partitions"]) if a == bb)

    ns = len(seeds)
    # plateau: the smallest budget from which EVERY larger budget in the grid, and the budget
    # itself, gives the same answer on every seed as the largest budget.
    plateau_answer = None
    for i, c in enumerate(cells):
        if all(cc["seeds_answer_equals_largest_budget"] == ns for cc in cells[i:]):
            plateau_answer = c["budget"]
            break
    plateau_partition = None
    for i, c in enumerate(cells):
        if all(cc["seeds_partition_identical_to_largest_budget"] == ns for cc in cells[i:]):
            plateau_partition = c["budget"]
            break

    # Is the budget a quality knob? Compare how much the ARI moves with the budget, once the
    # budget is past the smallest grid point that stops the pool being crippled, against how
    # much it moves with the arrival order or the generator seed at the shipped default.
    def _rng(vals):
        return round(max(vals) - min(vals), 4) if vals else None

    at_default = next((c for c in cells if c["budget"] == SHIPPED_DEFAULT), cells[-1])
    ari_over_seeds_at_default = _rng([r["ARI"] for r in at_default["per_seed"]])
    ari_over_budgets_ge_128 = _rng([c["ARI"]["mean"] for c in cells if c["budget"] >= 128])
    knob = dict(
        ari_range_over_seeds_at_default=ari_over_seeds_at_default,
        ari_range_over_budget_means_from_128_up=ari_over_budgets_ge_128,
        budget_moves_ari_less_than_the_seed_does=(
            ari_over_budgets_ge_128 is not None and ari_over_seeds_at_default is not None
            and ari_over_budgets_ge_128 <= ari_over_seeds_at_default),
        note=("the budget is not monotone in ARI; what is reported is the size of the movement, "
              "not its direction"))

    max_peak = max(r["peak_pool"] for c in cells for r in c["per_seed"])
    smallest_never_binding = None
    for c in cells:
        if c["seeds_with_eviction"] == 0:
            smallest_never_binding = c["budget"]
            break

    out = dict(
        stream=name, n_records=cfg["n"], seeds=list(seeds), seed_kind=cfg["seed_kind"],
        truth=cfg["truth"], cells=[{k: v for k, v in c.items() if k != "_partitions"}
                                   for c in cells],
        plateau_budget=plateau_answer,
        plateau_budget_partition_identical=plateau_partition,
        smallest_budget_that_never_evicts=smallest_never_binding,
        max_peak_pool_over_all_cells=max_peak,
        is_the_budget_a_quality_knob=knob,
        default_on_flat_part=(plateau_answer is not None and plateau_answer <= SHIPPED_DEFAULT),
        never_stops_moving=(plateau_answer is None),
        seconds_total=round(sum(c["seconds_total"] for c in cells), 2))
    return out


# --------------------------------------------------------------------------- #
def build_reading(report):
    st = report["streams"]
    per = {}
    for name, s in st.items():
        per[name] = dict(
            plateau_budget=s["plateau_budget"],
            plateau_budget_partition_identical=s["plateau_budget_partition_identical"],
            smallest_budget_that_never_evicts=s["smallest_budget_that_never_evicts"],
            max_peak_pool=s["max_peak_pool_over_all_cells"],
            is_the_budget_a_quality_knob=s["is_the_budget_a_quality_knob"],
            default_4096_on_flat_part=s["default_on_flat_part"],
            never_stops_moving=s["never_stops_moving"])
    moving = [n for n, v in per.items() if v["never_stops_moving"]]
    not_flat = [n for n, v in per.items() if not v["never_stops_moving"]
                and not v["default_4096_on_flat_part"]]
    all_flat = not moving and not not_flat
    plateaus = {n: v["plateau_budget"] for n, v in per.items()}
    order = [n for n in ("twogroup", "planted8", "wikipedia", "lazada") if n in plateaus]
    plateau_str = ", ".join(f"{n} {plateaus[n]}" for n in order)
    lines = []
    if all_flat:
        worst = max(plateaus[n] for n in order)
        rel = ("equal to" if worst == SHIPPED_DEFAULT else
               f"{SHIPPED_DEFAULT // worst}x below" if worst and SHIPPED_DEFAULT % worst == 0 else
               "below")
        lines.append(
            f"On every stream tested the answer stops moving at or below a budget of {worst}, "
            f"which is {rel} the shipped default of {SHIPPED_DEFAULT}, so the default is on the "
            "flat part of the curve for all four.")
        lines.append("Smallest budget that reaches the plateau, per stream: " + plateau_str + ".")
    else:
        if moving:
            lines.append("The answer never stops moving on: " + ", ".join(moving)
                         + ". That concedes the objection on those streams and is reported as such.")
        if not_flat:
            lines.append("The plateau is ABOVE the shipped default of 4096 on: "
                         + ", ".join(not_flat) + ", so the default is not on the flat part there.")
    lines.append(
        "The budget stops mattering once it exceeds the peak number of live candidate accounts "
        "the stream produces, which is a property of the stream and not a tuned value: peaks here "
        "are " + ", ".join(f"{n} {per[n]['max_peak_pool']}" for n in per) + ".")
    lines.append(
        "Below the plateau the budget does move the answer, and not monotonically: on Wikipedia "
        "the mean ARI rises to a peak at a small budget and settles lower at the plateau. From a "
        "budget of 128 up, the ARI moves with the budget by "
        + ", ".join(
            f"{n} {per[n]['is_the_budget_a_quality_knob']['ari_range_over_budget_means_from_128_up']} "
            f"against {per[n]['is_the_budget_a_quality_knob']['ari_range_over_seeds_at_default']} "
            f"with the seed at the default" for n in order)
        + ".")
    lines.append(
        "Scope: these four streams are 320 to 3,000 records. The cap is documented to bind on the "
        "full Lazada stream (21,365 records) and on MusicBrainz 20K (results/mb20k_full2.json "
        "records the cap raised from 4096 to 32768 mid-run), so the flat-part claim is made for "
        "streams of this size and is not a claim about arbitrarily long streams.")
    return dict(
        question=("At what budget does the answer stop moving, and is the shipped default of "
                  f"{SHIPPED_DEFAULT} on the flat part of that curve for every stream tested?"),
        answer_definition=("the tuple (final node count K, ARI against the truth, number of mints, "
                           "records left in the background); 'stops moving' means this tuple equals "
                           "its value at the largest budget in the grid, on every seed, at this "
                           "budget and at every larger budget in the grid"),
        default_on_flat_part_for_every_stream=all_flat,
        streams_where_it_never_stops_moving=moving,
        streams_where_the_plateau_is_above_the_default=not_flat,
        smallest_budget_reaching_the_plateau=plateaus,
        smallest_budget_reaching_the_plateau_partition_identical={
            n: v["plateau_budget_partition_identical"] for n, v in per.items()},
        smallest_budget_that_never_evicts={
            n: v["smallest_budget_that_never_evicts"] for n, v in per.items()},
        per_stream=per,
        summary=" ".join(lines))


def validity_check(report):
    """The budget-4096 column must reproduce results/e4_first_tier.json exactly, because 4096
    is the shipped default and E4 is the committed run of the same protocol on three of these
    streams. If it does not, this sweep is measuring something else and the table is void."""
    path = os.path.join(RESULTS, "e4_first_tier.json")
    if not os.path.exists(path):
        return dict(ran=False, why=f"{path} not present")
    with open(path) as fh:
        e4 = json.load(fh)
    rows, ok = [], True
    for name in ("twogroup", "planted8", "wikipedia"):
        if name not in report["streams"] or name not in e4:
            continue
        cell = next((c for c in report["streams"][name]["cells"]
                     if c["budget"] == SHIPPED_DEFAULT), None)
        per = {r["seed"]: r for r in e4[name]["ours_seeds"]["per_seed"]}
        if cell is None:
            continue
        for r in cell["per_seed"]:
            a = per.get(r["seed"])
            if a is None:
                continue
            m = (round(r["ARI"], 4) == a["ARI"]) and (r["K"] == a["K"])
            ok = ok and m
            rows.append(dict(stream=name, seed=r["seed"], e4_ARI=a["ARI"],
                             e19_ARI=round(r["ARI"], 4), e4_K=a["K"], e19_K=r["K"], match=m))
    return dict(ran=True, all_match=ok, compared=len(rows), rows=rows,
                what=("the budget-4096 column of this sweep against the ours_seeds rows of "
                      "results/e4_first_tier.json, ARI to 4 decimals and K exactly"))


def main():
    quick = "--quick" in sys.argv
    budgets = (1, 8, 4096, 32768) if quick else BUDGETS
    seeds = (0, 1) if quick else SEEDS

    names = ["twogroup", "planted8", "wikipedia"]
    skipped = {}
    if lazada_available():
        names.append("lazada")
    else:
        skipped["lazada"] = f"{LAZADA_JSON} not present"

    t0 = time.time()
    report = dict(
        experiment="e19_budget_curve",
        title="the candidate-pool budget curve",
        purpose=("answer the objection that the new-node penalty lambda was not removed but "
                 "moved into the candidate-pool budget, by measuring the curve instead of the "
                 "two-point stress test in code/tests/test_batch.py"),
        parameter="EscrowGraph(cand_pool_cap=...) (engine.py:80), shipped default 4096",
        eviction_rule=("drop the live candidate with the lowest accrued escrow G, never one "
                       "touched by the current record (engine.py:207-221)"),
        protocol=protocol_describe(),
        protocol_note=("every run goes through escrow.protocol.new_run(cand_pool_cap=c); the "
                       "cand_pool_cap keyword was added to protocol.new_run and "
                       "protocol.run_stream as a defaulted additive argument, so no existing "
                       "caller and no default changed"),
        budgets=list(budgets),
        budgets_requested=list(REQUESTED_BUDGETS),
        budgets_added=sorted(b for b in budgets if b in REFINEMENT_BUDGETS),
        budgets_added_why=("the requested grid jumps 1024 to 4096, which put the Wikipedia and "
                           "Lazada plateau exactly at the shipped default; 128, 512, 2048 and "
                           "3072 locate the plateau inside that jump"),
        budgets_dropped=[],
        shipped_default=SHIPPED_DEFAULT,
        seeds=list(seeds),
        metrics=dict(
            K="final node count after the final full repair pass",
            ARI="adjusted Rand index of record labels against the truth; background = one cluster",
            mints="len(g.mint_log), releases attempted over the stream (nodes can later merge)",
            background="records that end in no node",
            peak_pool="max len(g.pool) observed after any record",
            evictions="exact count of candidate accounts dropped by the eviction rule",
            seconds="wall time of the stream plus its repair passes"),
        streams={}, skipped=skipped)

    for name in names:
        print(f"  [{name}] starting", flush=True)
        report["streams"][name] = sweep(name, budgets, seeds)
        print(f"  [{name}] done in {report['streams'][name]['seconds_total']}s", flush=True)

    report["validity_check"] = validity_check(report)
    report["reading"] = build_reading(report)
    report["seconds_total"] = round(time.time() - t0, 2)
    stamped(report)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    vc = report["validity_check"]
    print(f"\nvalidity check (budget {SHIPPED_DEFAULT} column against E4): "
          f"{vc.get('all_match')} over {vc.get('compared')} cells")
    print("\n" + report["reading"]["summary"])
    print(f"\nwrote {OUT} in {report['seconds_total']}s")


if __name__ == "__main__":
    main()
