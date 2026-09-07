"""E32: the same real stream, made bigger, tested against the scope condition's own prediction.

WHY THIS AND NOT A NEW DATASET. E31 says a latent kind is reachable only if one of its characteristic
(key, value) pairs recurs at least t* times, t* between 3 and 24 by E9. The 320-record encyclopedia
stream clears that bar three times over on its top pair per kind, and still averages only 0.87 with a
spread across a third of the scale, because it clears it ONCE per kind: 92% of its pairs are seen
exactly once, so a node is born and then has almost nothing with which to recruit. That is a property
of a 320-record sample, not of encyclopedias. Swapping in a different dataset here would confound the
question with a change of domain, and would look like shopping for a stream that flatters the method.

So the stream is held fixed and only its size moves. Same source, same categories, same parser,
character for character. What changes is how many pages of each kind the stream contains.

THE PREDICTION, AND IT IS NOT A SAFE ONE. Two quantities move in opposite directions as the stream
grows:

    the seed cohort of a kind grows LINEARLY in n, since a fixed fraction of records carry a
    characteristic pair;
    the price of a member grows as log2(n / t), which for a kind of fixed proportion is CONSTANT,
    and the cost of finding the node among more candidates grows only logarithmically.

So the method should get BETTER on the same data as the sample grows, and its order-variance should
shrink, because more of its pairs clear t* and the search has more evidence to select on. This is the
opposite of what happens to a method whose difficulty grows with the stream, and it is the sharp
version of the E29 race. If accuracy is flat or falls with n, the scope condition is not what governs
this stream and E31's account of the losses is wrong.

WHAT IS MEASURED, at each stream size and over several arrival orders: the seed cohort per kind, the
fraction of (key, value) pairs that clear t*, ESCROW's agreement with the categories and its spread,
and the same baselines on the identical categorical input that E21 defines. The baselines see the
whole vocabulary up front, which is more than ESCROW gets.

WHAT WOULD FALSIFY IT. Accuracy flat or falling in n, or a spread that does not shrink. Reported
either way.
"""
from __future__ import annotations

import glob
import json
import os
import random
import statistics as st
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.protocol import run_stream, describe as protocol_describe   # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e4_baseline_army import _ari                           # noqa: E402
from experiments.e21_matched_representation import (matched_representation,   # noqa: E402
                                                    baseline_suite)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
LARGE = os.path.join(RESULTS, "wiki_large")
OUT = os.path.join(RESULTS, "e32_scale_in_scope.json")

SIZES = (320, 640, 1280, 2560, 4500)
ORDERS = 8
T_STAR_MAX = 24          # E9's least favourable cohort at release, used as the bar throughout

# The matched representation is dense and one column wide per observed (key, value) pair, so it
# reaches 4,500 by 32,253 at the top of the ladder. The full baseline suite sweeps k from 2 to 20 for
# its silhouette selection and then sweeps HDBSCAN, agglomerative, DP-means and a cosine graph over
# their own grids, all on that matrix; at 2,560 records that did not finish in forty minutes of CPU,
# which is why the cap is where it is. Above the cap, k-means handed the true number of types is
# still run at every size. That is the ORACLE condition and the hardest baseline in the suite to
# beat, so what is skipped is the easier comparisons, and they are skipped by a declared rule rather
# than dropped quietly.
BASELINE_FULL_MAX_N = 1280

# Two declared choices about the pool, both made before any accuracy was read.
#
# MIN_CATEGORY. Four of the first ten categories turned out to be container categories holding
# almost no articles, which the category name does not reveal. A populous category of the same type
# was added beside each rather than swapping it out, and a category enters the pool only if it
# yielded at least this many records. The rule is stated here so the pool is not chosen by looking
# at results.
MIN_CATEGORY = 100
#
# THE LABEL. A trailing digit marks a second category of a TYPE already present: "settlement2" is
# more cities, not a new kind. Those share one label, since two labels for one infobox template
# would be a ground truth that no method could satisfy.


def _label(cat):
    return cat.rstrip("0123456789")


def load_pool():
    """Every fetched record with its type label, subject to the two rules declared above."""
    by_cat = {}
    for f in sorted(glob.glob(os.path.join(LARGE, "wiki_cache.*.json"))):
        cat = os.path.basename(f).split("wiki_cache.")[1].split(".json")[0]
        rows = []
        for r in json.load(open(f, encoding="utf-8")):
            rec = {str(k): str(v) for k, v in r.items() if isinstance(v, (str, int, float))}
            if rec:
                rows.append(rec)
        by_cat[cat] = rows
    dropped = {c: len(v) for c, v in by_cat.items() if len(v) < MIN_CATEGORY}
    pool = [(_label(c), r) for c, v in by_cat.items() if len(v) >= MIN_CATEGORY for r in v]
    return pool, dropped


def subsample(pool, n, seed):
    """n records, drawn so the category mix matches the pool's. Holding the mix fixed is what makes
    the only moving part the stream size."""
    rng = random.Random(seed)
    by_cat = {}
    for cat, r in pool:
        by_cat.setdefault(cat, []).append(r)
    total = sum(len(v) for v in by_cat.values())
    picked = []
    for cat, rows in sorted(by_cat.items()):
        take = min(len(rows), max(1, round(n * len(rows) / total)))
        picked += [(cat, r) for r in rng.sample(rows, take)]
    rng.shuffle(picked)
    picked = picked[:n]
    return [r for _, r in picked], [c for c, _ in picked]


def stream_stats(recs, truth):
    kv = Counter((k, v) for r in recs for k, v in r.items())
    per = {}
    for r, t in zip(recs, truth):
        per.setdefault(t, Counter()).update(r.items())
    cohorts = {t: int(max(c.values())) for t, c in per.items()}
    occ = list(kv.values())
    return {"distinct_pairs": len(kv),
            "pairs_clearing_t_star": int(sum(1 for c in occ if c >= T_STAR_MAX)),
            "fraction_pairs_clearing_t_star": round(sum(1 for c in occ if c >= T_STAR_MAX) / len(occ), 4),
            "fraction_pairs_seen_once": round(sum(1 for c in occ if c == 1) / len(occ), 4),
            "seed_cohort_min": min(cohorts.values()),
            "seed_cohort_median": int(st.median(cohorts.values())),
            "kinds_reachable": int(sum(1 for c in cohorts.values() if c >= T_STAR_MAX)),
            "kinds_total": len(cohorts)}


def escrow_once(recs, truth):
    t0 = time.time()
    g, b = run_stream(recs)
    lab = [-1] * len(recs)
    for v in g.nodes.values():
        for m in v.members:
            lab[m - 1] = v.nid
    return {"ARI": round(_ari(truth, lab), 4), "K": int(g.K),
            "bits": round(b.total(), 1),
            "background": int(sum(1 for x in lab if x == -1)),
            "seconds": round(time.time() - t0, 1)}


def main():
    pool, dropped = load_pool()
    if not pool:
        print("no records in", LARGE, "- run fetch_wiki_large.py first")
        return
    cats = Counter(c for c, _ in pool)
    print(f"pool: {len(pool)} records over {len(cats)} types "
          f"(dropped below {MIN_CATEGORY}: {dropped})")
    for c, k in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:12s} {k}")

    report = {"experiment": "E32 the same real stream at increasing size",
              "question": ("does the method improve on the SAME data as the stream grows, which is "
                           "what the scope condition predicts and what a difficulty-grows-with-n "
                           "method would not do"),
              "prediction": ("seed cohorts grow linearly in n while the price of a member is flat "
                             "for a kind of fixed proportion, so accuracy should rise and the "
                             "spread across arrival orders should shrink"),
              "falsifier": "accuracy flat or falling in n, or a spread that does not shrink",
              "protocol": protocol_describe(),
              "source": "same categories, same parser as the 320-record stream; only n moves",
              "t_star_bar": T_STAR_MAX,
              "pool": {"records": len(pool), "categories": dict(cats),
                       "min_category_size": MIN_CATEGORY,
                       "categories_dropped_below_minimum": dropped,
                       "label_rule": ("a trailing digit marks a second category of a type already "
                                      "present, and those share one label")},
              "rows": []}

    for n in SIZES:
        if n > len(pool):
            print(f"n={n} exceeds the pool of {len(pool)}, stopping")
            break
        runs, stats = [], None
        for s in range(ORDERS):
            recs, truth = subsample(pool, n, s)
            if s == 0:
                stats = stream_stats(recs, truth)
            runs.append(escrow_once(recs, truth))
            print(f"  n={n:5d} order {s}: ARI {runs[-1]['ARI']:.4f} K {runs[-1]['K']} "
                  f"({runs[-1]['seconds']}s)", flush=True)
        ari = [r["ARI"] for r in runs]

        recs, truth = subsample(pool, n, 0)
        X, _dim = matched_representation(recs)      # returns (matrix, column count)
        if n <= BASELINE_FULL_MAX_N:
            base, _checks, _n = baseline_suite(X, truth, len(set(truth)))
        else:
            from sklearn.cluster import KMeans as _KM
            lab = _KM(len(set(truth)), n_init=10, random_state=0).fit_predict(X)
            base = {"kmeans_oracleK": dict(condition="ORACLE", knob="n_clusters",
                                           value=int(len(set(truth))),
                                           ARI=round(_ari(truth, list(lab)), 4),
                                           K=int(len(set(lab)))),
                    "_note": {"skipped": "the rest of the suite, above the declared size cap",
                              "cap": BASELINE_FULL_MAX_N}}
        row = {"n": n, "stream": stats,
               "escrow": {"mean_ARI": round(st.mean(ari), 4), "sd": round(st.pstdev(ari), 4),
                          "min": min(ari), "max": max(ari),
                          "spread": round(max(ari) - min(ari), 4),
                          "mean_K": round(st.mean(r["K"] for r in runs), 1),
                          "per_order": runs},
               # Keep every field except the bulky per-parameter curves. The first version of
               # this kept only a fixed list of key names, which silently dropped the silhouette
               # baseline's score, since that one reports under by_cosine_silhouette and
               # by_euclidean_silhouette rather than under ARI. That is the fairest baseline in the
               # suite, so losing it was the one omission that mattered.
               "baselines_matched_input": {
                   k: {kk: vv for kk, vv in v.items()
                       if kk not in ("sweep", "K_at", "silhouette_curve", "grid", "grid_e4",
                                     "grid_extended_beyond_e4")}
                   for k, v in base.items() if isinstance(v, dict)}}
        report["rows"].append(row)
        print(f"  n={n:5d}: ESCROW {row['escrow']['mean_ARI']:.4f} "
              f"(sd {row['escrow']['sd']:.4f}, spread {row['escrow']['spread']:.4f})  "
              f"pairs clearing t* {stats['fraction_pairs_clearing_t_star']:.1%}  "
              f"kinds reachable {stats['kinds_reachable']}/{stats['kinds_total']}", flush=True)

    rows = report["rows"]
    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        report["headline"] = {
            "n_from_to": [first["n"], last["n"]],
            "ARI_from_to": [first["escrow"]["mean_ARI"], last["escrow"]["mean_ARI"]],
            "spread_from_to": [first["escrow"]["spread"], last["escrow"]["spread"]],
            "pairs_clearing_t_star_from_to": [first["stream"]["fraction_pairs_clearing_t_star"],
                                              last["stream"]["fraction_pairs_clearing_t_star"]],
            "accuracy_rose": last["escrow"]["mean_ARI"] > first["escrow"]["mean_ARI"],
            "spread_shrank": last["escrow"]["spread"] < first["escrow"]["spread"],
        }
        report["headline"]["verdict"] = (
            "the prediction holds: the same stream gets easier as it grows"
            if report["headline"]["accuracy_rose"] and report["headline"]["spread_shrank"]
            else "the prediction does not hold, and the scope condition does not govern this stream")
        print("\n" + json.dumps(report["headline"], indent=2))

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
