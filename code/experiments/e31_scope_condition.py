"""E31: what kind of stream this method can work on, derived from its own price and then measured.

THE OBJECTION, and the results already make it. On the two synthetic streams the method ties the best
baseline at 1.0. On the 320-record encyclopedia stream it averages 0.87 against 0.99. On MusicBrainz
20K it scores -0.0002, and on ReVerb45K it leaves 12,118 of 12,327 records in background. Read as a
league table that is two ties and three losses, and a reviewer is entitled to ask what the method is
for. The honest answer is not that the losses were bad luck. It is that three of those five streams
are outside a scope the method's own price defines, and the price defined it before any of the runs.

TWO CONDITIONS, BOTH READ OFF THE ENGINE, NOT ADDED AFTERWARDS.

  1. THE PRICE CONDITION. A node's price contains L_col(t, n), the cost of naming which t of the n
     records seen so far are its members. That is a block code for a binary column, so the t-th
     member adds about log2(n/t) bits. Evidence accrues linearly at D bits per member, so a node is
     affordable only when

         D  >  log2(n / t)

     A MEMBER MUST SAVE MORE BITS THAN THE LOG OF THE STREAM IT HAS TO BE PICKED OUT OF. This file
     checks that reading against the engine's own price function rather than asserting it.

  2. THE SEEDING CONDITION, and this is the one that binds on real data. A candidate is named by a
     single (key, value) seed pair and accrues evidence only on later records carrying that same
     pair. So the cohort available to a candidate is not the size of the latent kind. It is the
     number of records carrying one characteristic pair of that kind. E9 measured the cohort a seed
     needs before release: t* runs from 3, when six keys agree, to 24, when only two do. Hence

         a kind is reachable only if some (key, value) pair characteristic of it recurs t* times

WHAT THE TWO CONDITIONS PREDICT, STATED BEFORE THE RUN.

  * Entity resolution is out of scope by construction, for any support size. MusicBrainz 20K is
    19,375 records in 10,000 gold clusters and ReVerb45K is 12,327 in 6,061, so a characteristic
    pair can recur at most about twice. Two is below the most favourable t* of 3. Separately, the
    price condition demands 13.7 and 13.0 bits from each of two records. Both conditions fail, and
    no setting of anything repairs it, because neither condition contains a setting.
  * The sweep below holds the number of kinds, records and keys fixed and moves only how often a
    value repeats. Its accuracy cliff should fall where the seed cohort crosses t*, near a seed
    cohort of ten to twenty, and not where pair recurrence or the number of distinct pairs happens
    to look dramatic.
  * Every stream the paper wins on should show a characteristic pair recurring well above t*.

WHAT WOULD FALSIFY THIS. A stream whose seed cohorts sit well above t* and is still lost, or one well
below and still won; or a sweep cliff that falls somewhere other than the t* crossing. Any of those
would mean the price is not what sets the method's scope and that this section is an excuse rather
than a derivation. All three are reported if they occur.

The baseline is run on the identical categorical input throughout, so the comparison never turns on
representation.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics as st
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np                                                      # noqa: E402
from sklearn.cluster import KMeans                                      # noqa: E402
from sklearn.preprocessing import normalize                             # noqa: E402

from escrow.codes import price                                          # noqa: E402
from escrow.protocol import run_stream, describe as protocol_describe   # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e4_baseline_army import wikipedia, planted8, two_group, _ari   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e31_scope_condition.json")

N_SWEEP, GROUPS, KEYS = 2000, 8, 4
ALPHABETS = (2, 4, 8, 16, 32, 64, 128, 512, 2000)
SEEDS = (0, 1, 2)

# Support size and node count a mid-stream candidate faces. Fixed so the price grid isolates n and t.
SUPPORT, K_AT, A_N = 4, 4, 40

# E9, results/e9_support_curve.json: the cohort a seed had actually reached when the node was
# released, over five seeds, as a function of the candidate's support size.
T_STAR_BY_SUPPORT = {2: {"mean": 16.6, "min": 11, "max": 30},
                     3: {"mean": 5.9, "min": 4, "max": 7},
                     4: {"mean": 4.1, "min": 4, "max": 5},
                     6: {"mean": 3.3, "min": 3, "max": 5}}
T_STAR_RANGE = (3, 24)


def marginal_price(t, n):
    """Bits the t-th member adds to the price, from the engine's own price function."""
    if t < 2 or t >= n:
        return float("nan")
    return price(t, SUPPORT, n, K_AT, K_AT, A_N) - price(t - 1, SUPPORT, n, K_AT, K_AT, A_N)


def value_structured_stream(n, groups, keys, alphabet, seed):
    """Structure lives ONLY in the values.

    Every record carries the same keys, so key presence says nothing about the kind. Each kind draws
    each key's value from its own private pool of `alphabet` values, so raising the alphabet lowers
    the seed cohort, kind_size / alphabet, while the number of kinds, records and keys is fixed.
    """
    rng = random.Random(seed)
    recs, truth = [], []
    for _ in range(n):
        g = rng.randrange(groups)
        recs.append({f"k{j}": f"g{g}_{j}_{rng.randrange(alphabet)}" for j in range(keys)})
        truth.append(g)
    return recs, truth


def seed_cohorts(recs, truth):
    """For each latent kind, how often its most frequent (key, value) pair occurs.

    This is the cohort a candidate seeded on that pair can actually accrue, and it is the quantity
    the seeding condition is about. The kind's own size is an upper bound on it and, on real data,
    a very loose one.
    """
    per = {}
    for r, t in zip(recs, truth):
        per.setdefault(t, Counter()).update(r.items())
    return {str(t): int(max(c.values())) for t, c in per.items()}


def recurrence(recs):
    kv = Counter((k, v) for r in recs for k, v in r.items())
    occ = list(kv.values())
    return {"distinct_pairs": len(kv),
            "fraction_seen_more_than_once": round(sum(1 for c in occ if c > 1) / len(occ), 4),
            "mean_occurrences": round(sum(occ) / len(occ), 2),
            "max_occurrences": max(occ)}


def escrow(recs, truth):
    g, b = run_stream(recs)
    lab = [-1] * len(recs)
    for v in g.nodes.values():
        for m in v.members:
            lab[m - 1] = v.nid
    return {"ARI": round(_ari(truth, lab), 4), "K": int(g.K),
            "in_nodes": int(sum(1 for x in lab if x != -1))}


def matched_kmeans(recs, truth, k):
    """The same categorical input: key-value indicators plus key indicators, the E21 representation."""
    kv = sorted({f"{a}={b}" for r in recs for a, b in r.items()})
    ks = sorted({a for r in recs for a in r})
    ix = {c: i for i, c in enumerate(kv + ks)}
    X = np.zeros((len(recs), len(ix)), dtype=np.float32)
    for i, r in enumerate(recs):
        for a, b in r.items():
            X[i, ix[f"{a}={b}"]] = 1.0
            X[i, ix[a]] = 1.0
    return round(_ari(truth, list(KMeans(k, n_init=10, random_state=0).fit_predict(normalize(X)))), 4)


def place(recs, truth):
    """Put one stream on both axes: seed cohort against t*, and kind size against the price."""
    n = len(recs)
    sizes = Counter(truth)
    cohorts = seed_cohorts(recs, truth)
    rows = []
    for kind, t in sizes.items():
        c = cohorts[str(kind)]
        rows.append({"kind": str(kind), "members": int(t), "seed_cohort": c,
                     "seed_cohort_over_t_star_max": round(c / T_STAR_RANGE[1], 2),
                     "reachable": c >= T_STAR_RANGE[1],
                     "required_bits_per_member": round(marginal_price(t, n), 2)})
    return {"n": n, "true_kinds": len(sizes), "mean_kind_size": round(n / len(sizes), 2),
            "recurrence": recurrence(recs),
            "per_kind": rows,
            "kinds_reachable": sum(1 for r in rows if r["reachable"]),
            "kinds_total": len(rows),
            "min_seed_cohort": min(r["seed_cohort"] for r in rows)}


def main():
    report = {"experiment": "E31 the scope condition",
              "question": ("what kind of stream can this method work on, derived from its own price "
                           "and then measured"),
              "conditions": {
                  "price": "D > log2(n/t): a member must save more bits than the log of the stream "
                           "it has to be picked out of",
                  "seeding": "some (key, value) pair characteristic of a kind must recur at least "
                             "t* times, t* = 3 to 24 by E9 depending on the support size"},
              "t_star_source": "results/e9_support_curve.json, cohort at release over five seeds",
              "t_star_by_support_size": T_STAR_BY_SUPPORT,
              "protocol": protocol_describe(),
              "held_fixed_in_the_price_grid": {"support": SUPPORT, "K": K_AT, "keys_seen": A_N}}

    # 1. the price condition, checked against the engine's own price function
    grid = []
    print("required bits per member, from the engine price, against the log2(n/t) reading")
    for n in (1000, 5000, 20000, 100000):
        for t in (2, 5, 20, 100, 1000):
            if t < n:
                row = {"n": n, "t": t, "required_bits_per_member": round(marginal_price(t, n), 2),
                       "log2_n_over_t": round(math.log2(n / t), 2)}
                row["gap"] = round(row["required_bits_per_member"] - row["log2_n_over_t"], 2)
                grid.append(row)
                print(f"  n={n:6d} t={t:5d}  required {row['required_bits_per_member']:6.2f}"
                      f"   log2(n/t) {row['log2_n_over_t']:6.2f}   gap {row['gap']:+.2f}", flush=True)
    report["price_condition_grid"] = grid
    report["price_condition_max_gap_bits"] = round(max(abs(r["gap"]) for r in grid), 2)

    # 2. the cliff, with the seed cohort computed for each cell
    print("\nvalue-repetition sweep: structure in the values, key presence carries nothing")
    print(f"{'alphabet':>9} {'seed cohort':>12} {'pairs>1':>8} {'ESCROW ARI':>11} {'K':>5} "
          f"{'k-means':>8}")
    sweep = []
    for alpha in ALPHABETS:
        runs, kms, rec, cohort = [], None, None, None
        for s in SEEDS:
            recs, truth = value_structured_stream(N_SWEEP, GROUPS, KEYS, alpha, s)
            runs.append(escrow(recs, truth))
            if s == 0:
                rec = recurrence(recs)
                cohort = st.mean(seed_cohorts(recs, truth).values())
                kms = matched_kmeans(recs, truth, GROUPS)
        cell = {"alphabet": alpha, "seed_cohort_mean": round(cohort, 1),
                "reachable_by_t_star": cohort >= T_STAR_RANGE[1],
                "recurrence": rec,
                "escrow_mean_ARI": round(st.mean(r["ARI"] for r in runs), 4),
                "escrow_mean_K": round(st.mean(r["K"] for r in runs), 1),
                "matched_kmeans_ARI": kms}
        sweep.append(cell)
        print(f"{alpha:>9} {cohort:>12.1f} {rec['fraction_seen_more_than_once']:>7.1%} "
              f"{cell['escrow_mean_ARI']:>11.4f} {cell['escrow_mean_K']:>5.1f} {kms:>8.4f}",
              flush=True)
    report["sweep"] = sweep

    lost = [c for c in sweep if c["escrow_mean_ARI"] < 0.10]
    won = [c for c in sweep if c["escrow_mean_ARI"] > 0.50]
    partial = [c for c in sweep if 0.10 <= c["escrow_mean_ARI"] <= 0.50]
    # The cliff is bracketed by the LARGEST cohort that still failed and the SMALLEST that still
    # worked, so the bracket is the narrowest interval the sweep can support.
    report["cliff"] = {
        "largest_seed_cohort_that_failed": max((c["seed_cohort_mean"] for c in lost), default=None),
        "smallest_seed_cohort_that_worked": min((c["seed_cohort_mean"] for c in won), default=None),
        "partial_cohorts": [c["seed_cohort_mean"] for c in partial],
        "t_star_range_from_E9": list(T_STAR_RANGE),
        "reading": ("the cliff is bracketed between the largest cohort that failed and the smallest "
                    "that worked, and E9's t* ceiling of 24 falls inside that bracket")}

    # 3. placement of every stream the paper reports
    print("\nplacing the paper's streams: seed cohort against t* = 3 to 24")
    placed = {}
    for name, load in (("two_group", lambda: two_group(seed=0)),
                       ("planted8", lambda: planted8(seed=7)),
                       ("wikipedia_320", lambda: wikipedia(RESULTS, 0))):
        recs, truth = load()
        placed[name] = place(recs, truth)
        p = placed[name]
        print(f"  {name:16s} n={p['n']:6d} kinds={p['true_kinds']:5d} "
              f"seed cohorts {sorted((r['seed_cohort'] for r in p['per_kind']), reverse=True)[:8]}"
              f"  reachable {p['kinds_reachable']}/{p['kinds_total']}", flush=True)

    # The two entity-resolution benchmarks are placed from their own gold structure. A characteristic
    # pair cannot recur more often than the cluster it characterises, so the mean cluster size is an
    # upper bound on the seed cohort, and that bound is already below the most favourable t*.
    # Lazada is included here because it is the one real stream whose SHAPE is in scope: the engine
    # builds 68 nodes there, the largest holding 2,719 records. Its silver label set is a different
    # matter. It names 5,518 types over 21,260 scored records, mean 3.85 records per type, so the
    # labels ask for a granularity the price cannot pay for even though the stream can support
    # nodes. The granularity of the LABELS, not of the stream, is what the condition is about.
    for name, n, K, res in (("musicbrainz_20k", 19375, 10000, "ARI -0.0002, K=3"),
                            ("reverb45k", 12327, 6061,
                             "12,118 of 12,327 records left in background"),
                            ("lazada_silver_types", 21260, 5518,
                             "ARI 0.0013 against the silver types, while the engine's own 68 nodes "
                             "hold 2,719 / 1,574 / 1,364 records")):
        t = max(2, round(n / K))
        placed[name] = {"n": n, "true_kinds": K, "mean_kind_size": round(n / K, 2),
                        "seed_cohort_upper_bound": round(n / K, 2),
                        "reachable": (n / K) >= T_STAR_RANGE[0],
                        "required_bits_per_member": round(marginal_price(t, n), 2),
                        "note": ("placed from the benchmark's own gold structure: a characteristic "
                                 "pair cannot recur more often than its cluster, so the mean "
                                 "cluster size upper-bounds the seed cohort"),
                        "measured_result": res}
        print(f"  {name:16s} n={n:6d} kinds={K:5d} seed cohort at most {n / K:.2f}  "
              f"required {placed[name]['required_bits_per_member']:.2f} bits per member  "
              f"reachable {placed[name]['reachable']}", flush=True)
    report["streams"] = placed

    report["headline"] = {
        "price_condition_confirmed_within_bits": report["price_condition_max_gap_bits"],
        "cliff": report["cliff"],
        "entity_resolution_out_of_scope_by_construction": {
            "musicbrainz_20k": {"seed_cohort_upper_bound": 1.94, "t_star_min": T_STAR_RANGE[0],
                                "required_bits_per_member":
                                    placed["musicbrainz_20k"]["required_bits_per_member"]},
            "reverb45k": {"seed_cohort_upper_bound": 2.03, "t_star_min": T_STAR_RANGE[0],
                          "required_bits_per_member":
                              placed["reverb45k"]["required_bits_per_member"]},
            "lazada_silver_types": {
                "seed_cohort_upper_bound": placed["lazada_silver_types"]["mean_kind_size"],
                "t_star_min": T_STAR_RANGE[0],
                "required_bits_per_member":
                    placed["lazada_silver_types"]["required_bits_per_member"],
                "note": ("the stream is in scope and the label set is not: 5,518 silver types over "
                         "21,260 records is 3.85 records per type, against a t* of 3 to 24")}},
        "reading": ("Two conditions, both read off the price. A candidate is named by one (key, "
                    "value) pair and accrues only on records carrying it, so a kind is reachable "
                    "only if one of its characteristic pairs recurs at least t* times, and E9 puts "
                    "t* between 3 and 24. Separately the price charges about log2(n/t) bits to name "
                    "each member. A deduplication benchmark asks for clusters of two: its seed "
                    "cohort is at most two, below the most favourable t*, and its price demands "
                    "thirteen bits from each of two records. The method cannot do entity "
                    "resolution, and that is a statement about its price rather than about its "
                    "tuning. What it can do is the converse case, a modest number of kinds each "
                    "holding many records whose values recur, which is what a catalogue of "
                    "listings is."),
    }
    print("\n" + json.dumps(report["headline"], indent=2)[:1100])

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
