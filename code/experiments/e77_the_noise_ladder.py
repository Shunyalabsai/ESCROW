"""E77: the whole noise ladder, for both arms of the engine, with and without majority voting.

WHY. E75 built a stream whose groups live only in the values, because the shipped `planted8` fixture
puts the group name inside the key name and a dictionary lookup on the key signature scores a perfect
agreement on it at every noise setting. E76 then built a split move that is grown before it is
priced. Adjusted Rand index, or ARI, is the agreement measure used throughout; 1.0 is the truth and
0.0 is chance.

Two questions were left open and both belong on one graph. First, nobody has run this ladder for both
arms. The stored sweep in results/e75_a_fixture_that_tests_the_values.json was produced with the
split flag left at its default, which is on, so that file is the split arm and not the engine before
the move; this file reproduces it level for level below. The only base arm number on this fixture is
the single point at full signal, mean ARI 0.6099 at mean K 5.0, quoted in the split comment in
escrow/batch.py. One curve and one headline from two engines cannot be read together, so the base
ladder is measured here from end to end. Second, every number so far is a single arrival order, and
the protocol also offers a shortest of R selection over R arrival orders, which the paper calls
majority voting. Nobody has said which of the two a quoted number is. So this runs the full ladder
four ways: the engine before the split move and the engine with it, each in a single arrival order
and each under the shortest of three orders.

WHAT IS MEASURED. For each signal level, each of three fixture seeds, and each arm: the number of
nodes K against the true 8, ARI against the planted truth, how many of the 3000 records ended up in
any node at all, and L_batch, the total description length in bits that the criterion minimises. The
ladder runs signal 1.0 down to 0.0, which is noise 0.0 up to 1.0. At signal 0 the truth is zero
nodes, so K is the score there and ARI is not meaningful. Nothing is tuned on the labels: the engine
has no calibrated parameter, and each arm gets one run per order.

The two arms differ in exactly one flag, `escrow.batch.SPLIT_MOVES`, which is set explicitly before
every single run so no number here depends on a default:

  escrow_base    the split move and the residual mint switched off, which is the engine the
                 paper's earlier numbers come from
  escrow_split   the same engine with the split move and the residual mint switched on

The flag now defaults to on, so the base arm exists only because the flag is written before the run.
Neither arm is left to the default.

Apple to apple holds by construction. Both arms read the same records, in the same orders, from the
same fixture, and are scored by the same ARI and the same L_batch. Neither arm is told the number of
groups, and neither is told which records are noise. No baseline appears in this file, so there is
nothing here that one method was given and another was not. The classical field on these same eleven
rungs, with the same records and the E4 triple on every knob, is E78.

WHAT WOULD REFUTE IT. A node at signal 0 in either arm, which would break the abstention claim on the
harder null. Or the split arm buying agreement while spending more bits, since the split move is
supposed to be a search fix and not a change of the criterion, so it must come out shorter wherever
it comes out better. Or majority voting moving the answer so much that the single order numbers
already quoted in the paper cannot stand as quoted.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

from escrow import batch as B
from escrow.protocol import record_labels, run_consensus, run_stream
from escrow.provenance import stamped
from e4_baseline_army import _ari
from e75_a_fixture_that_tests_the_values import stream

OUT = os.path.join(ROOT, "results", "e77_the_noise_ladder.json")

N = int(os.environ.get("E77_N", "3000"))
SEEDS = (0, 1, 2)
R = 3
TRUE_K = 8
SIGNAL = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)
ARMS = (("escrow_base", False), ("escrow_split", True))

# The published numbers for the single arrival order at signal 1.0, n = 3000, seeds 0, 1, 2.
# The base row is the before number quoted in the split comment in escrow/batch.py, whose mean is
# ARI 0.6099 at mean K 5.0. The split row is the per seed form of the stored E75 sweep, whose mean
# is ARI 0.9986 at K 8, 7, 8. If this file cannot reproduce them the engine has moved and no results
# file should be written.
#
# One stale number to know about. E76's docstring and the split comment in escrow/batch.py both say
# the grown split takes this fixture to ARI 0.9995 at a node count of exactly 8. E76's own code only
# runs the E47 cover fixture, so that pair is not in any results file, and these three seeds do not
# give it: they give 0.9986 at K 8, 7, 8. Quote the seed numbers below, not 0.9995.
REFERENCE = {
    "escrow_base": {"ARI": [0.4815, 0.8655, 0.4828], "K": [5, 6, 4]},
    "escrow_split": {"ARI": [0.9996, 1.0000, 0.9961], "K": [8, 7, 8]},
}

# The whole stored E75 ladder, signal 0.0 up to 1.0, which was run with the split flag on. The split
# arm here must reproduce it at every level, or one of the two files is stale.
E75_LADDER = os.path.join(ROOT, "results", "e75_a_fixture_that_tests_the_values.json")


def score(g, bits, truth):
    lab = record_labels(g, len(truth))
    return {"K": int(g.K), "ARI": round(_ari(truth, lab), 4),
            "in_a_node": sum(1 for x in lab if x != -1), "L_batch": round(bits, 1)}


def one_order(recs, truth):
    g, b = run_stream(recs)
    return score(g, b.total(), truth)


def voted(recs, truth, seed):
    """Shortest of R shuffled orders, plus what each of those R orders scored on its own.

    The per order numbers are here so the voting column can be read against the mean of the very
    same orders. Voting minus single order mixes two things, more tries and different orders, since
    run_consensus shuffles all R and none of them is the fixture order. Voting minus the mean of its
    own orders is the effect of keeping the shortest, with nothing else moving."""
    cons = run_consensus(recs, R=R, seed=seed)
    best = cons["best"]
    lab = best["labels"]
    per_order = [{"K": int(r["K"]), "ARI": round(_ari(truth, r["labels"]), 4),
                  "in_a_node": sum(1 for x in r["labels"] if x not in (None, -1)),
                  "L_batch": round(r["bits"], 1)} for r in cons["runs"]]
    return {"K": int(best["K"]), "ARI": round(_ari(truth, lab), 4),
            "in_a_node": sum(1 for x in lab if x not in (None, -1)),
            "L_batch": round(best["bits"], 1),
            "K_by_order": [o["K"] for o in per_order],
            "ARI_by_order": [o["ARI"] for o in per_order],
            "in_a_node_by_order": [o["in_a_node"] for o in per_order],
            "L_batch_by_order": [o["L_batch"] for o in per_order]}


def mean(xs):
    return sum(xs) / len(xs)


def bar(x, width=34, top=1.0):
    filled = int(round(max(0.0, min(x, top)) / top * width))
    return "#" * filled + "." * (width - filled)


def k_line(k, width=26, top=13):
    cells = ["-"] * width
    mark = int(round(TRUE_K / top * (width - 1)))
    cells[mark] = "|"
    if k is not None:
        pos = int(round(min(k, top) / top * (width - 1)))
        cells[pos] = "*" if pos != mark else "+"
    return "".join(cells)


def main():
    started = time.time()
    report = {
        "experiment": "E77 the noise ladder, both engine arms, single order and majority voting",
        "fixture": ("E75 stream(signal, seed, n=%d): 8 groups, 6 shared keys, every record carries "
                    "the same six keys, so only the values separate a group" % N),
        "records": N, "groups": TRUE_K, "seeds": list(SEEDS), "orders_for_voting": R,
        "signal_ladder": list(SIGNAL),
        "arms": {"escrow_base": "escrow.batch.SPLIT_MOVES = False, the engine before the split move",
                 "escrow_split": "escrow.batch.SPLIT_MOVES = True, split move and residual mint on"},
        "what_each_method_was_given": (
            "both arms read the same records in the same orders, with the same keys and the same "
            "values. Neither is told the number of groups, neither is told which records are noise, "
            "and neither has a calibrated parameter to set. The single order arm reads each stream "
            "once in fixture order. The voting arm reads the same records in three shuffled orders "
            "and keeps the state with the shortest description, which is a choice made on bits and "
            "never on the labels. No baseline runs in this file, so there is nothing here that one "
            "method was given and another was not. The classical field on these same eleven rungs, "
            "with the same records and a swept knob reported twice, is E78"),
        "metrics": ("K against the true 8, ARI against the planted truth, records placed in any "
                    "node out of %d, and L_batch in bits. At signal 0 the truth is zero nodes, so K "
                    "is the score there and ARI is not meaningful" % N),
        "levels": [],
    }

    print(f"records {N}, seeds {list(SEEDS)}, {R} orders for voting, true K = {TRUE_K}\n")
    head = (f"  {'signal':>6} {'noise':>6} | {'base K':>7} {'base ARI':>9} | "
            f"{'split K':>8} {'split ARI':>10} | {'vote base K':>12} {'vote base ARI':>14} | "
            f"{'vote split K':>13} {'vote split ARI':>15}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    for f in SIGNAL:
        cell = {}
        for arm, flag in ARMS:
            single, vote = [], []
            for s in SEEDS:
                recs, truth = stream(f, s, n=N)
                B.SPLIT_MOVES = flag                      # set before every run, never defaulted
                single.append(one_order(recs, truth))
                assert B.SPLIT_MOVES == flag, "the arm flag moved while the stream was running"
                B.SPLIT_MOVES = flag
                vote.append(voted(recs, truth, s))
                assert B.SPLIT_MOVES == flag, "the arm flag moved while the orders were running"
            cell[arm] = {"single_order": single, "majority_voting": vote}

        row = {"signal": f, "noise": round(1 - f, 2), "per_arm": {}}
        for arm, _ in ARMS:
            for mode in ("single_order", "majority_voting"):
                per = cell[arm][mode]
                row["per_arm"].setdefault(arm, {})[mode] = {
                    "K_mean": round(mean([p["K"] for p in per]), 2),
                    "K_per_seed": [p["K"] for p in per],
                    "ARI_mean": round(mean([p["ARI"] for p in per]), 4),
                    "ARI_per_seed": [p["ARI"] for p in per],
                    "in_a_node_total": sum(p["in_a_node"] for p in per),
                    "in_a_node_per_seed": [p["in_a_node"] for p in per],
                    "in_a_node_out_of": N * len(SEEDS),
                    "L_batch_mean": round(mean([p["L_batch"] for p in per]), 1),
                    "L_batch_per_seed": [p["L_batch"] for p in per],
                }
                if mode == "majority_voting":
                    cellrow = row["per_arm"][arm][mode]
                    cellrow["K_by_order_per_seed"] = [p["K_by_order"] for p in per]
                    cellrow["ARI_by_order_per_seed"] = [p["ARI_by_order"] for p in per]
                    cellrow["in_a_node_by_order_per_seed"] = [p["in_a_node_by_order"] for p in per]
                    cellrow["L_batch_by_order_per_seed"] = [p["L_batch_by_order"] for p in per]
                    flat_k = [k for p in per for k in p["K_by_order"]]
                    flat_a = [a_ for p in per for a_ in p["ARI_by_order"]]
                    cellrow["every_order_K_mean"] = round(mean(flat_k), 2)
                    cellrow["every_order_ARI_mean"] = round(mean(flat_a), 4)
                    cellrow["orders_that_minted_a_node"] = sum(1 for k in flat_k if k > 0)
                    cellrow["orders_run"] = len(flat_k)
        for mode in ("single_order", "majority_voting"):
            a = row["per_arm"]["escrow_base"][mode]
            c = row["per_arm"]["escrow_split"][mode]
            bs = cell["escrow_base"][mode]
            cs = cell["escrow_split"][mode]
            row.setdefault("split_minus_base", {})[mode] = {
                "K": round(c["K_mean"] - a["K_mean"], 2),
                "ARI": round(c["ARI_mean"] - a["ARI_mean"], 4),
                "L_batch": round(c["L_batch_mean"] - a["L_batch_mean"], 1),
                "ARI_per_seed": [round(y["ARI"] - x["ARI"], 4) for x, y in zip(bs, cs)],
                "L_batch_per_seed": [round(y["L_batch"] - x["L_batch"], 1)
                                     for x, y in zip(bs, cs)],
            }
        report["levels"].append(row)

        p = row["per_arm"]
        print(f"  {f:>6.2f} {1-f:>6.2f} | "
              f"{p['escrow_base']['single_order']['K_mean']:>7.2f} "
              f"{p['escrow_base']['single_order']['ARI_mean']:>9.4f} | "
              f"{p['escrow_split']['single_order']['K_mean']:>8.2f} "
              f"{p['escrow_split']['single_order']['ARI_mean']:>10.4f} | "
              f"{p['escrow_base']['majority_voting']['K_mean']:>12.2f} "
              f"{p['escrow_base']['majority_voting']['ARI_mean']:>14.4f} | "
              f"{p['escrow_split']['majority_voting']['K_mean']:>13.2f} "
              f"{p['escrow_split']['majority_voting']['ARI_mean']:>15.4f}")

    # ---------------- the reference check, before anything is written ---------------- #
    top = next(r for r in report["levels"] if r["signal"] == 1.0)
    zero = next(r for r in report["levels"] if r["signal"] == 0.0)
    problems = []
    for arm, want in REFERENCE.items():
        got = top["per_arm"][arm]["single_order"]
        if got["K_per_seed"] != want["K"]:
            problems.append(f"{arm} K at signal 1.0 is {got['K_per_seed']}, expected {want['K']}")
        for i, (x, y) in enumerate(zip(got["ARI_per_seed"], want["ARI"])):
            if abs(x - y) > 0.002:
                problems.append(f"{arm} ARI at signal 1.0 seed {SEEDS[i]} is {x}, expected {y}")
    for arm, _ in ARMS:
        ks = zero["per_arm"][arm]["single_order"]["K_per_seed"]
        if any(k != 0 for k in ks):
            problems.append(f"{arm} at signal 0.0 gives K {ks}, the null needs 0 on every seed")

    # the whole stored E75 ladder, which was run with the split flag on, level for level
    ladder_rows, ladder_note = [], "results/e75_a_fixture_that_tests_the_values.json not found"
    if os.path.exists(E75_LADDER):
        stored = json.load(open(E75_LADDER))["sweep"]
        ladder_note = ("every level of the stored E75 sweep against the split arm here, single "
                       "arrival order. The stored file was run with the split flag at its default, "
                       "which is on, so the split arm is the one it has to match")
        for s_row in stored:
            here = next((r for r in report["levels"] if r["signal"] == s_row["signal"]), None)
            if here is None:
                continue
            got = here["per_arm"]["escrow_split"]["single_order"]
            same = (got["K_per_seed"] == s_row["K_per_seed"]
                    and abs(got["ARI_mean"] - s_row["ARI_mean"]) <= 0.002)
            ladder_rows.append({"signal": s_row["signal"], "stored_K": s_row["K_per_seed"],
                                "split_arm_K": got["K_per_seed"],
                                "stored_ARI_mean": s_row["ARI_mean"],
                                "split_arm_ARI_mean": got["ARI_mean"], "same": same})
            if not same:
                problems.append(
                    f"split arm at signal {s_row['signal']} gives K {got['K_per_seed']} ARI "
                    f"{got['ARI_mean']}, stored E75 has K {s_row['K_per_seed']} ARI "
                    f"{s_row['ARI_mean']}")
    report["reference_check"] = {"reference": REFERENCE, "problems": problems,
                                 "reproduced": not problems,
                                 "stored_E75_ladder_note": ladder_note,
                                 "stored_E75_ladder": ladder_rows}

    # ---------------- the plain text charts ---------------- #
    lines = []
    lines.append("ARI against noise, single arrival order. b = base, s = split. "
                 "left is ARI 0, right is ARI 1")
    for r in report["levels"]:
        p = r["per_arm"]
        lines.append(f"  noise {r['noise']:>4.2f}  b {bar(p['escrow_base']['single_order']['ARI_mean'])}"
                     f" {p['escrow_base']['single_order']['ARI_mean']:>6.4f}")
        lines.append(f"              s {bar(p['escrow_split']['single_order']['ARI_mean'])}"
                     f" {p['escrow_split']['single_order']['ARI_mean']:>6.4f}")
    lines.append("")
    lines.append("ARI against noise, shortest of three orders. b = base, s = split")
    for r in report["levels"]:
        p = r["per_arm"]
        lines.append(f"  noise {r['noise']:>4.2f}  b "
                     f"{bar(p['escrow_base']['majority_voting']['ARI_mean'])}"
                     f" {p['escrow_base']['majority_voting']['ARI_mean']:>6.4f}")
        lines.append(f"              s {bar(p['escrow_split']['majority_voting']['ARI_mean'])}"
                     f" {p['escrow_split']['majority_voting']['ARI_mean']:>6.4f}")
    lines.append("")
    lines.append("K against noise, single arrival order. | is the true 8, * is the mean K, "
                 "+ is a mean K on the true 8")
    for r in report["levels"]:
        p = r["per_arm"]
        lines.append(f"  noise {r['noise']:>4.2f}  b {k_line(p['escrow_base']['single_order']['K_mean'])}"
                     f" {p['escrow_base']['single_order']['K_mean']:>5.2f}")
        lines.append(f"              s {k_line(p['escrow_split']['single_order']['K_mean'])}"
                     f" {p['escrow_split']['single_order']['K_mean']:>5.2f}")
    report["chart"] = lines
    print("\n" + "\n".join(lines))

    # ---------------- headline ---------------- #
    def first_node(arm, mode):
        for r in report["levels"]:
            if r["per_arm"][arm][mode]["K_mean"] > 0:
                return r["signal"]
        return None

    def reaches_eight(arm, mode):
        for r in report["levels"]:
            if abs(r["per_arm"][arm][mode]["K_mean"] - TRUE_K) < 0.5:
                return r["signal"]
        return None

    votes_help = {}
    for arm, _ in ARMS:
        d = [round(r["per_arm"][arm]["majority_voting"]["ARI_mean"]
                   - r["per_arm"][arm]["single_order"]["ARI_mean"], 4)
             for r in report["levels"]]
        # the clean half of it: the kept state against the mean of the very same three orders
        k = [round(r["per_arm"][arm]["majority_voting"]["ARI_mean"]
                   - r["per_arm"][arm]["majority_voting"]["every_order_ARI_mean"], 4)
             for r in report["levels"]]
        votes_help[arm] = {"ARI_voting_minus_single_per_level": d,
                           "mean": round(mean(d), 4), "worst": min(d), "best": max(d),
                           "ARI_kept_minus_mean_of_the_same_three_orders_per_level": k,
                           "kept_minus_own_orders_mean": round(mean(k), 4),
                           "reading": ("voting minus single order mixes more tries with different "
                                       "orders, because run_consensus shuffles all three and none "
                                       "of them is the fixture order. The second list holds the "
                                       "orders fixed and is the effect of keeping the shortest")}

    # every place a single node swallowed the stream, which is the failure shape to watch
    swallowed = []
    for r in report["levels"]:
        for arm, _ in ARMS:
            for m in ("single_order", "majority_voting"):
                cellrow = r["per_arm"][arm][m]
                for i, s_ in enumerate(SEEDS):
                    if cellrow["K_per_seed"][i] == 1 and cellrow["in_a_node_per_seed"][i] > 0.9 * N:
                        swallowed.append({"signal": r["signal"], "arm": arm, "mode": m, "seed": s_,
                                          "in_a_node": cellrow["in_a_node_per_seed"][i],
                                          "of": N, "ARI": cellrow["ARI_per_seed"][i]})
    # what abstaining costs on the very same records, so the state can be priced against it
    for case in swallowed:
        r = next(x for x in report["levels"] if x["signal"] == case["signal"])
        i = SEEDS.index(case["seed"])
        empty_bits, own_bits = [], None
        for arm, _ in ARMS:
            s_ = r["per_arm"][arm]["single_order"]
            if s_["K_per_seed"][i] == 0:
                empty_bits.append(s_["L_batch_per_seed"][i])
            v_ = r["per_arm"][arm]["majority_voting"]
            for kk, ll in zip(v_["K_by_order_per_seed"][i], v_["L_batch_by_order_per_seed"][i]):
                if kk == 0:
                    empty_bits.append(ll)
            if arm == case["arm"]:
                own_bits = (s_["L_batch_per_seed"][i] if case["mode"] == "single_order"
                            else v_["L_batch_per_seed"][i])
        case["L_batch"] = own_bits
        case["L_batch_of_the_empty_state"] = min(empty_bits) if empty_bits else None
        case["bits_the_one_node_state_saves"] = (
            round(min(empty_bits) - own_bits, 1) if empty_bits and own_bits is not None else None)
    saved = [c["bits_the_one_node_state_saves"] for c in swallowed
             if c["bits_the_one_node_state_saves"] is not None]
    report["one_node_swallows_the_stream"] = {
        "cases": swallowed,
        "levels_where_it_happened": sorted({c["signal"] for c in swallowed}),
        "bits_saved_over_abstaining": {"min": min(saved), "max": max(saved)} if saved else None,
        "reading": ("a single node holding almost every record is not a wrong node count, it is the "
                    "everything in one node state, and it scores ARI near zero. Where the empty "
                    "state was also reached on the same records, the saving says how much the "
                    "criterion prefers the one node state over saying nothing. A positive saving "
                    "means the criterion accepts the state and the search only has to find it. The "
                    "levels listed here say where that happened, and it is worth one experiment of "
                    "its own")}

    bits_ok = all(x <= 0 for r in report["levels"]
                  for m in ("single_order", "majority_voting")
                  for x in r["split_minus_base"][m]["L_batch_per_seed"])
    report["about_the_bit_comparison"] = {
        "split_arm_never_spends_more_bits_in_any_seed": bits_ok,
        "reading": ("this one is a consistency check and not a finding. The split move and the "
                    "residual mint run in the final repair pass only, so both arms follow the same "
                    "path up to that pass, and a move is applied only when it strictly lowers "
                    "L_batch. The split arm therefore cannot end above the base arm on the same "
                    "records in the same order. What the check would catch is the move being "
                    "applied on something other than the criterion, which would show up as a cell "
                    "where the split arm bought agreement and paid bits for it")}

    null_voting = {}
    for arm, _ in ARMS:
        v = zero["per_arm"][arm]["majority_voting"]
        null_voting[arm] = {"K_of_the_kept_state_per_seed": v["K_per_seed"],
                            "K_of_every_order_per_seed": v["K_by_order_per_seed"],
                            "bits_of_every_order_per_seed": v["L_batch_by_order_per_seed"],
                            "single_order_K_per_seed": zero["per_arm"][arm]["single_order"]["K_per_seed"]}
    report["null_under_voting"] = {
        "holds": all(k == 0 for arm, _ in ARMS
                     for ks in null_voting[arm]["K_of_every_order_per_seed"] for k in ks),
        "detail": null_voting,
        "reading": ("shortest of three keeps the state with the fewest bits, so it takes a minimum "
                    "over three tries, and more tries make an unwanted state easier to find. That "
                    "is only half of it. Where a node does appear at zero signal the objective "
                    "prefers it to the empty state on the same records, by the bit saving listed "
                    "in one_node_swallows_the_stream, and the same one node state is reached in "
                    "the fixture order alone at the levels listed there. So the criterion accepts "
                    "the state and the extra tries only find it. The per order counts say whether "
                    "a node appeared in one order or in all three")}

    node_at_zero = [{"arm": arm, "mode": m, "K_per_seed": zero["per_arm"][arm][m]["K_per_seed"]}
                    for arm, _ in ARMS for m in ("single_order", "majority_voting")
                    if any(k > 0 for k in zero["per_arm"][arm][m]["K_per_seed"])]
    bought_and_paid = [{"signal": r["signal"], "mode": m}
                       for r in report["levels"] for m in ("single_order", "majority_voting")
                       if r["split_minus_base"][m]["ARI"] > 0
                       and r["split_minus_base"][m]["L_batch"] > 0]
    biggest_vote_move = max(abs(x) for arm, _ in ARMS
                            for x in votes_help[arm]["ARI_voting_minus_single_per_level"])
    report["refutation_conditions"] = {
        "a node at zero signal in either arm": {
            "fired": bool(node_at_zero), "where": node_at_zero,
            "fired_in_a_single_arrival_order": any(c["mode"] == "single_order"
                                                   for c in node_at_zero),
            "fired_under_the_shortest_of_three_orders": any(c["mode"] == "majority_voting"
                                                            for c in node_at_zero),
            "note": ("this is the condition the docstring names first. The paper quotes the single "
                     "arrival order, so that is the line the abstention claim rests on, and the "
                     "voting line is a separate statement about a minimum over three tries")},
        "the split arm buying agreement while spending more bits": {
            "fired": bool(bought_and_paid), "where": bought_and_paid},
        "voting moving the answer": {
            "largest_ARI_move_from_voting": round(biggest_vote_move, 4),
            "note": ("how far a single order number moves if the same run is voted. Read it with "
                     "voting_effect_on_ARI, which separates more tries from different orders")},
    }

    report["headline"] = {
        "K_at_zero_signal": {arm: zero["per_arm"][arm]["single_order"]["K_per_seed"]
                             for arm, _ in ARMS},
        "K_at_zero_signal_with_voting": {arm: zero["per_arm"][arm]["majority_voting"]["K_per_seed"]
                                         for arm, _ in ARMS},
        "the_null_holds_in_every_arm": all(
            k == 0 for arm, _ in ARMS for m in ("single_order", "majority_voting")
            for k in zero["per_arm"][arm][m]["K_per_seed"]),
        "signal_at_which_the_first_node_appears": {
            f"{arm}.{m}": first_node(arm, m) for arm, _ in ARMS
            for m in ("single_order", "majority_voting")},
        "signal_at_which_mean_K_reaches_the_true_8": {
            f"{arm}.{m}": reaches_eight(arm, m) for arm, _ in ARMS
            for m in ("single_order", "majority_voting")},
        "voting_effect_on_ARI": votes_help,
        "split_arm_never_spends_more_bits": bits_ok,
        "orders_at_zero_signal_that_minted_a_node": {
            arm: f"{zero['per_arm'][arm]['majority_voting']['orders_that_minted_a_node']} of "
                 f"{zero['per_arm'][arm]['majority_voting']['orders_run']}"
            for arm, _ in ARMS},
        "one_node_swallowed_the_stream_in_this_many_cells": len(swallowed),
        "refutation_conditions_that_fired": [k for k, v in report["refutation_conditions"].items()
                                             if v.get("fired")],
        "reference_reproduced": not problems,
        "reading": ("read this with `null_under_voting` and `one_node_swallows_the_stream`. The "
                    "single order arms are the ones the abstention claim rests on, and "
                    "K_at_zero_signal is that claim. Where a node does get in at zero signal, the "
                    "objective prefers that one node state to the empty state on the same records, "
                    "so what fails there is the criterion and not only the search"),
    }
    # escrow.provenance stamps engine.py, batch.py and codes.py. The split move lives in split.py,
    # which is the file the two arms differ over, so its digest is recorded here as well.
    try:
        import hashlib
        sp = os.path.join(ROOT, "code", "escrow", "split.py")
        report["split_py_md5"] = hashlib.md5(open(sp, "rb").read()).hexdigest()
    except OSError as e:
        report["split_py_md5"] = f"not read: {e}"
    report["runtime_seconds"] = round(time.time() - started, 1)

    print("\n" + json.dumps(report["headline"], indent=2))

    if problems:
        print("\nREFERENCE CHECK FAILED, no results file written:")
        for p_ in problems:
            print("  " + p_)
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("\nwritten", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
