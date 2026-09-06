"""E18 assembly. Puts every arm's exceedance and every positive control in one file.

Reads the arm outputs produced by e16_wide.py (committed arms in results/theorem1_arms,
new arms in ARMS_OUT), by e18_seed_charge_null.py and by e18_charge_cost.py, and writes
results/e18_seed_naming_charge.json: the algebra, the charge chosen, the exceedance table
for every arm measured so far, the positive-control table, and the verdict.

Nothing is computed here that was not measured somewhere else; this file only collects.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(ROOT, "results")
COMMITTED_ARMS = os.path.join(RESULTS, "theorem1_arms")
ARMS = os.environ.get("ARMS_OUT", COMMITTED_ARMS)
LEVELS = ("1", "2", "4", "8", "12")


def load(path):
    with open(path) as f:
        return json.load(f)


def wide_row(path, label, note):
    d = load(path)
    return {"arm": label, "flags": d["flags"], "statistic": d["statistic"],
            "candidates": d["candidates"], "streams": d["streams"],
            "mints": d["mints"], "max_K": d["max_K"],
            "sup_stats": d["sup_stats"],
            "exceedance": {b: d["exceedance"][b]["empirical"] for b in LEVELS},
            "standard_error": {b: d["exceedance"][b].get("standard_error")
                               for b in LEVELS},
            "holds_at": [b for b in LEVELS
                         if d["exceedance"][b]["empirical"] <= 2.0 ** -int(b)],
            "note": note}


def main():
    rows = []
    for fn, label, note in [
        (os.path.join(COMMITTED_ARMS, "e16_wide_shipped.json"), "shipped",
         "the default; measured before this experiment"),
        (os.path.join(COMMITTED_ARMS, "e16_wide_naming.json"), "tight_naming",
         "correction (a) alone; measured before this experiment"),
        (os.path.join(COMMITTED_ARMS, "e16_wide_unselected.json"), "unselected",
         "correction (b) alone; measured before this experiment"),
        (os.path.join(COMMITTED_ARMS, "e16_wide_both.json"), "tight_naming+unselected",
         "corrections (a)+(b); measured before this experiment"),
        (os.path.join(ARMS, "e16_wide_seed_av.json"), "seed_naming_av",
         "NEW: keep the seed evidence, add log2|A_n| + log2 d_a to the price"),
        (os.path.join(ARMS, "e16_wide_seed_ln.json"), "seed_naming_ln",
         "NEW: the same, with Rissanen's rank code instead of the uniform one"),
        (os.path.join(ARMS, "e16_wide_seed_av_naming.json"),
         "seed_naming_av+tight_naming", "NEW: the charge with correction (a)"),
        (os.path.join(ARMS, "e16_wide_seed_av_unselected.json"),
         "seed_naming_av+unselected", "NEW: the charge with correction (b)"),
        (os.path.join(ARMS, "e16_wide_seed_av_both.json"),
         "seed_naming_av+tight_naming+unselected",
         "NEW: the charge with both earlier corrections"),
    ]:
        if os.path.exists(fn):
            rows.append(wide_row(fn, label, note))
        else:
            print("missing", fn, file=sys.stderr)

    nulls = {}
    for fn, label in [(os.path.join(ARMS, "e18_seed_charge_null_shipped.json"),
                       "shipped"),
                      (os.path.join(ARMS, "e18_seed_charge_null_shipped_T3000.json"),
                       "shipped_T3000"),
                      (os.path.join(ARMS, "e18_seed_charge_null_tightnaming.json"),
                       "tight_naming")]:
        if os.path.exists(fn):
            d = load(fn)
            for T, r in d["by_T"].items():
                nulls.setdefault(label, {})[T] = {
                    "candidates": r["candidates"],
                    "mean_sup_G": r["sup_G"]["mean"], "max_sup_G": r["sup_G"]["max"],
                    "accrual_steps_per_candidate":
                        r.get("accrual_steps_per_candidate", int(T) / 10),
                    "drift_bits_per_accrual_step":
                        r.get("drift_bits_per_accrual_step",
                              round(r["sup_G"]["mean"] / (int(T) / 10), 4)),
                    "required_gamma_shipped_statistic": r["required_gamma"],
                    "required_gamma_unselected_statistic":
                        r["required_gamma_unselected"],
                }

    cost = {}
    for fn, key in [(os.path.join(ARMS, "e18_charge_cost.json"), "charge_cost"),
                    (os.path.join(ARMS, "e18_charge_cost_finegamma.json"),
                     "charge_cost_fine_gamma")]:
        if os.path.exists(fn):
            cost[key] = load(fn)

    controls = {}
    for label in ("off", "naming", "unsel", "both"):
        fn = os.path.join(COMMITTED_ARMS, f"controls_{label}.json")
        if os.path.exists(fn):
            controls[label] = load(fn)
    for label in ("seed_av", "seed_ln", "seed_av_naming", "seed_av_unselected"):
        fn = os.path.join(ARMS, f"controls_{label}.json")
        if os.path.exists(fn):
            controls[label] = load(fn)

    out = {
        "experiment": "E18: can a price-side seed naming charge make Theorem 1 true?",
        "question": ("Keep the seed key's evidence, which is load-bearing, and pay for "
                     "the selection on the price side by naming the seed pair. Does the "
                     "lifetime bound hold, and at what cost in recovery?"),
        "algebra": {
            "family_bound": ("E[# spurious mints, ever] = sum_p Pr(p ever mints) "
                             "<= sum_p 2^-c_p with c_p the level candidate p must "
                             "clear; writing c_p = Lambda_p + Delta_p gives "
                             "sum_p 2^-c_p <= (max_p 2^-Lambda_p) sum_p 2^-Delta_p"),
            "requirement_on_the_charge": "sum_p 2^-Delta_p <= 1, the Kraft inequality "
                                         "over the candidate index set: Delta must be "
                                         "the length of a prefix codeword naming the "
                                         "seed pair, and nothing else is forced",
            "av": ("Delta = log2|A| + log2 d_a. Kraft sum = sum_a sum_{v<=d_a} "
                   "1/(|A| d_a) = sum_a 1/|A| = 1 exactly, but only for a closed "
                   "schema: naming with the counts current at each candidate's own "
                   "seed time is not a prefix code over a growing candidate set "
                   "(one new value per arrival on one key gives sum_j 1/j)"),
            "ln": ("Delta = L_N(rank(a)) + L_N(rank_a(v)), Rissanen's universal "
                   "integer code on the two intern ranks. sum_k 2^-L_N(k) <= 1 for "
                   "each factor, so the product is Kraft over the unbounded candidate "
                   "set; frozen at seed time and free of the stream length T"),
            "gamma": ("the uniform slack Gamma of Theorem 1(iii) is the same device "
                      "with the alphabet sizes replaced by a bound on them: the family "
                      "bound reads |A| dbar 2^-Gamma"),
            "boundary": ("Ville with a non-decreasing predictable boundary c_t is "
                         "controlled by inf_t c_t, so a charge that grows with the "
                         "stream buys nothing: only its seed-time value counts"),
            "what_decides_it": ("all three charges are O(1) in T. If the accumulator "
                               "were a martingale, sup_t G_t would satisfy "
                               "Pr(sup >= c) <= 2^-c and E[sup_T] would be bounded "
                               "uniformly in T. If the accrual index set is chosen by "
                               "the data so the accumulator drifts, sup_T grows with T "
                               "and no constant charge can work: the required charge "
                               "would have to know the stream length, which is exactly "
                               "the property Theorem 1(iii) holds against "
                               "mint-on-first-miss"),
        },
        "exceedance_all_arms": rows,
        "nominal": {b: 2.0 ** -int(b) for b in LEVELS},
        "null_drift_and_required_charge": nulls,
        "positive_controls_quick": controls,
        "cost": cost,
        "verdict": {
            "does_the_bound_hold": "No, and not at any price.",
            "why": ("A naming charge is O(1) in the stream length by construction. "
                    "Under the null the shipped accumulator drifts upward at 1.42 "
                    "bits per accrual step at T=500 rising to 2.76 at T=4000, so its "
                    "supremum grows with the stream (mean 71 bits at T=500, 1103 at "
                    "T=4000) and the uniform slack that would restore the nominal "
                    "level at b=8 grows with it (97 bits at T=500, 1205 at T=4000). "
                    "No constant can close a gap of that shape. The defect is a "
                    "drift induced by the accrual index set, not a family-wise "
                    "error rate."),
            "what_the_charge_does_pay_for": (
                "Exactly what it is for. With the seed key's own ledger entry out "
                "of the statistic, the slack the exceedance still needs is 3.4 to "
                "3.6 bits and is FLAT in T from 500 to 4000 records; the 5.32-bit "
                "log2|A|+log2 d_a charge covers it and brings the exceedance to "
                "0.132, 0.079, 0.026, 0.0015, 0.000, inside the nominal at all five "
                "levels, which no arm measured before this experiment achieved. "
                "Theorem 1(iii)'s charge is a correct multiple-comparisons "
                "correction; it is being asked to pay for a subsequence choice, "
                "which is not a multiplicity."),
            "cost_is_not_the_binding_constraint": (
                "At its natural size (5.3 to 12.2 bits) the charge preserves or "
                "improves every positive control: both synthetic streams keep the "
                "exact node count and a perfect index under the canonical order and "
                "over 20 arrival orders, E7's deferred arm keeps K=2 with identical "
                "supports and zero typo nodes on all five seeds, no node mints on "
                "any null stream, and Wikipedia improves from ARI 0.679 to 0.967 "
                "and purity 0.900 to 0.994. A uniform slack of 256 bits still "
                "leaves both synthetic controls exact; the two-group stream holds "
                "to 448 bits and the planted eight-group stream to 288. The price "
                "can absorb a large constant. It cannot absorb a quantity that "
                "grows with the stream."),
            "crossing": ("planted8 (n=3000) holds K=8 to Gamma=288 and returns "
                         "nothing at 352; the null of the same length needs 787 "
                         "bits at b=1 and 875 at b=8. twogroup (n=2000) holds to "
                         "448 and the null of that length needs 480 at b=1, so the "
                         "weakest level is within about 30 bits on that one control "
                         "and out of reach at every higher level and on the harder "
                         "control."),
            "recommendation": ("Do not adopt the charge as a repair for Theorem "
                               "1(ii). It may still be adopted on its own merits, "
                               "since it is close to free and improves the real "
                               "data, but that is a separate decision and the "
                               "shipped default is unchanged here."),
        },
        "paper_wording": {
            "theorem_1_iii_addition": (
                "This charge corrects multiplicity, and multiplicity alone. It is "
                "O(1) in the stream length by construction, and the deficit the "
                "engine's statistic exhibits is not: measured under the null, the "
                "slack that would restore the nominal level at b = 8 is 97 bits at "
                "T = 500 and 1,205 bits at T = 4,000, while the same slack for the "
                "statistic with the seed key's own term removed is 3.6 and 3.4. So "
                "(iii) is not available as a repair for (ii), and this paper does "
                "not use it as one."),
            "theory_tex_measured_paragraph": (
                "What the engine measures instead of the bound, and why the price "
                "cannot repair it. Over 300 independent null streams and 12,000 "
                "candidate trajectories the exceedance Pr(sup_t G_t >= b) is 1.000 "
                "at every b in {1, 2, 4, 8, 12} for the shipped statistic, against "
                "a nominal 2^-b. Adding the seed naming charge of Theorem 1(iii) to "
                "the price, which is the correction the theorem itself contemplates "
                "and which leaves the load-bearing seed evidence in place, does not "
                "move it: the exceedance is 1.000 at every level under log|A_n| + "
                "log d_a, under the universal code on the seed pair's intern ranks, "
                "and under either together with the tight naming charge. The reason "
                "is measurable and it is not multiplicity. Under the null the "
                "accumulator drifts upward at 1.42 bits per accrual step at T = 500 "
                "rising to 2.76 at T = 4,000, so its supremum grows with the "
                "stream, from a mean of 71 bits to 1,103, and the uniform slack "
                "that would restore the nominal level at b = 8 grows with it, from "
                "97 bits to 1,205. A naming charge is constant in T by "
                "construction, so no charge of that kind can close a gap of that "
                "shape. What the charge does pay for is exactly what it is for: "
                "with the seed key's own ledger entry removed from the statistic, "
                "the slack the exceedance still needs is 3.4 to 3.6 bits and is "
                "flat in T across 500 to 4,000 records, and the 5.32-bit charge "
                "covers it, bringing the exceedance to 0.132, 0.079, 0.026, 0.0015 "
                "and 0.000, inside the nominal at all five levels. The shipped "
                "statistic's failure is therefore a drift induced by the accrual "
                "index set, not a family-wise error rate, and the price side cannot "
                "absorb it."),
            "appendix_replacement_for_the_next_measurement_paragraph": (
                "That design has now been run, and it does not make the theorem "
                "true. Charging the seed pair's own naming cost on the price side, "
                "log|A_n| + log d_a or the universal code on its intern ranks, "
                "leaves the exceedance at 1.000 at every level, alone and with the "
                "tight naming charge. It is also close to free: at its natural "
                "size, 5.3 to 12.2 bits, both synthetic controls keep their exact "
                "node count and a perfect index under the canonical order and over "
                "twenty arrival orders, the deferred arm of E7 keeps its two nodes "
                "with identical supports and no typo node on all five seeds, no "
                "node is minted on any of the 48 null streams of E8 nor on any of "
                "the 300 wider ones, and the Wikipedia demonstration improves: its "
                "index rises from 0.679 to 0.967, its purity from 0.900 to 0.994, "
                "and the one impure node of 156 records splits into a pure film "
                "node of 117 and a pure person node of 111. A uniform slack of 256 "
                "bits still leaves both synthetic controls exact. The charge is "
                "affordable and it is ineffective, and those two facts together "
                "identify the defect: the price can be raised by a constant, and "
                "what the shipped statistic needs is not a constant. The one "
                "combination that does satisfy the bound is the charge added to the "
                "unselected statistic, which holds at all five levels and is the "
                "first arm in this series to do so, and which is not available as a "
                "default because deleting the seed key's evidence destroys the "
                "mint."),
        },
    }
    path = os.path.join(RESULTS, "e18_seed_naming_charge.json")
    json.dump(out, open(path, "w"), indent=2)
    print("written", path)
    for r in rows:
        print(f"{r['arm']:42s} " +
              " ".join(f"{r['exceedance'][b]:.5f}" for b in LEVELS) +
              f"   holds at {r['holds_at']}")


if __name__ == "__main__":
    main()
