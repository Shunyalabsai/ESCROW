"""E21: the same input for everyone, and what the baselines do at their own defaults.

THE OBJECTION, from an ICLR reviewer. code/experiments/e4_baseline_army.py encodes each
record with the sentence transformer all-MiniLM-L6-v2 and runs every distance baseline on
that embedding. ESCROW never sees the embedding; it reads the categorical (key, value)
facets. So E4 compares rule plus representation against rule alone, and cannot tell you
which one won. Separately, E4 reports only oracle-tuned and transferred knobs, never the
condition a practitioner actually gets, which is the library default.

WHAT THIS RUNS. On the three E4 streams (twogroup, planted8, wikipedia), every baseline is
run on TWO representations and, where it has a knob, under THREE conditions.

  Representations, both L2 normalised so cosine and Euclidean order pairs the same way:
    matched    binary multi-hot over the observed (key, value) pairs of that stream. This is
               exactly the information ESCROW codes: the record is the set of its facets, and
               nothing else is supplied. Vocabulary sizes: twogroup 80, planted8 779,
               wikipedia 2607.
    embedding  all-MiniLM-L6-v2 over "k=v k=v ..." with the keys sorted, which is E4's own
               encoding, recomputed here so both arms sit in one file under one metric.

  Metric. Cosine, on both arms. On a multi-hot indicator vector the cosine of two records is
  the count of shared (key, value) pairs divided by the geometric mean of their facet counts,
  which is the natural set overlap and is what a categorical record supports; Euclidean
  distance on unnormalised counts would make a record with many facets far from everything.
  The E4 embeddings were already unit norm, so cosine is also E4's own metric there. Where a
  library's shipped default names a different metric, that default is run as shipped as well
  and reported separately (`default_library_metric`).

  Baselines: k-means handed the true number of groups, k-means with k chosen by silhouette
  over k = 2..20, HDBSCAN, average-linkage agglomerative clustering cut at a distance
  threshold, DP-means (e4_baseline_army.dp_means), and cosine connected components (extra,
  carried over from E4 for parity).

  Conditions, per knobbed baseline:
    ORACLE       the knob swept, the best score on the test labels taken. This is an upper
                 bound and is labelled as one. k-means handed the true K is oracle by
                 construction, since the true K is test-label information.
    TRANSFERRED  the oracle knob of another stream applied unchanged, worst over the two
                 available sources, as in E4.
    DEFAULT      the value the library ships, recorded literally in the JSON
                 (HDBSCAN min_cluster_size=5; KMeans n_clusters=8; AgglomerativeClustering
                 n_clusters=2, linkage ward, metric euclidean, distance_threshold None).
                 DP-means and cosine components are not library estimators and ship no
                 default, which is recorded as such rather than invented.

  ESCROW is one run of the shipped protocol per stream, no knob, on the records themselves.

THE FALSIFIER. If a distance baseline on ESCROW's own representation matches or beats ESCROW
without its knob being tuned on the test labels, then the criterion is not what is doing the
work. Two readings are evaluated in the JSON and both are reported whatever they say:
  strict   only conditions that touch no test labels at all: DEFAULT, and k-means by
           silhouette. If any of these matches or beats ESCROW, the falsifier fires.
  lenient  strict plus TRANSFERRED, which tunes on another stream's labels but not on this
           stream's.
ORACLE and k-means handed the true K are excluded from both, because both read the test
labels of the stream being scored.

Run: python3 code/experiments/e21_matched_representation.py [--quick] [--no-embedding]
Writes results/e21_matched_representation.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

from escrow.protocol import run_stream, describe as protocol_describe
from escrow.provenance import stamped
from experiments.e4_baseline_army import (two_group, planted8, wikipedia, _ari,
                                          stream_for_seed, dp_means, cosine_cc)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e21_matched_representation.json")
WIKI_CACHE = RESULTS

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# E4's default streams, so the matched arm sits beside the numbers already committed in
# results/e4_first_tier.json. two_group() and wikipedia() default to seed 0; planted8()
# defaults to seed 7.
PRIMARY_STREAMS = ("twogroup", "planted8", "wikipedia")
SEED_FAMILY = (0, 1, 2, 3, 4)

KS_FOR_SILHOUETTE = tuple(range(2, 21))

# Grids. Every E4 grid point is kept so a transfer means the same thing in both files. The
# agglomerative and DP-means grids are EXTENDED upward, because on a multi-hot indicator most
# record pairs are orthogonal (cosine distance 1.0) and E4's top point of 0.7 would have cut
# the sweep off before the useful region. Extending a baseline's grid can only help the
# baseline, which is the permitted direction.
GRIDS = {
    "hdbscan": [3, 5, 10, 20, 50, 100],
    "agglo_threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99],
    "dp_means": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9, 1.99],
    "cosine_components": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
}
GRIDS_E4 = {
    "hdbscan": [3, 5, 10, 20, 50, 100],
    "agglo_threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
    "dp_means": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "cosine_components": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
}

# The literal shipped defaults, read off the sklearn 1.9 constructors.
LIBRARY_DEFAULTS = {
    "kmeans": dict(library="sklearn.cluster.KMeans", knob="n_clusters", default=8,
                   other_defaults="init k-means++, n_init auto, metric euclidean"),
    "hdbscan": dict(library="sklearn.cluster.HDBSCAN", knob="min_cluster_size", default=5,
                    other_defaults="metric euclidean, cluster_selection_method eom, "
                                   "min_samples None"),
    "agglo_threshold": dict(library="sklearn.cluster.AgglomerativeClustering",
                            knob="distance_threshold", default=None,
                            other_defaults="n_clusters 2, linkage ward, metric euclidean",
                            note=("distance_threshold has no shipped value: the constructor "
                                  "default is n_clusters=2 with ward linkage on euclidean "
                                  "distance, so that is what a practitioner gets and that is "
                                  "what is run")),
    "dp_means": dict(library=None, knob="lambda", default=None,
                     note=("DP-means is not a library estimator and Kulis and Jordan 2012 give "
                           "no default lambda, so there is no DEFAULT condition to report. That "
                           "absence is itself the cost of the method and is reported, not "
                           "papered over with an invented value")),
    "cosine_components": dict(library=None, knob="similarity cut", default=None,
                              note="not a library estimator, no shipped default"),
}


# --------------------------------------------------------------------------- #
# representations
# --------------------------------------------------------------------------- #
def matched_representation(recs):
    """Binary multi-hot over the observed (key, value) pairs AND over the keys themselves,
    rows L2 normalised.

    The key columns are not decoration and leaving them out was a real error in the first
    version of this experiment. The engine prices key PRESENCE separately from the value a key
    carries: a node states how reliably its members publish each key in its support, and a
    member that stays silent on a supported key pays a silence bill. A representation with only
    (key, value) columns cannot see that two records carry the same keys with different values,
    so it withholds from the baselines something ESCROW reads, which is the wrong direction. On
    the encyclopedia stream adding the key columns takes HDBSCAN at its shipped default from
    0.7909 to 0.9745, so the omission was not academic.

    Every baseline still gets the whole vocabulary of the stream up front, which is MORE than
    ESCROW gets, since ESCROW meets the vocabulary one record at a time."""
    vocab = sorted({(k, v) for r in recs for k, v in r.items()})
    keys = sorted({k for r in recs for k in r})
    ix = {p: i for i, p in enumerate(vocab)}
    kx = {k: len(vocab) + i for i, k in enumerate(keys)}
    X = np.zeros((len(recs), len(vocab) + len(keys)), dtype=np.float64)
    for i, r in enumerate(recs):
        for k, v in r.items():
            X[i, ix[(k, v)]] = 1.0
            X[i, kx[k]] = 1.0
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm, X.shape[1]


def record_text(r):
    return " ".join(f"{k}={v}" for k, v in sorted(r.items()))


def embedding_representation(recs, model):
    """E4's own encoding, recomputed: all-MiniLM-L6-v2 on the sorted 'k=v' text, unit norm."""
    texts = [record_text(r) for r in recs]
    X = np.asarray(model.encode(texts, batch_size=256, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float64)
    return X, X.shape[1]


# --------------------------------------------------------------------------- #
# DP-means, vectorised over centres, verified identical to the E4 implementation
# --------------------------------------------------------------------------- #
def dp_means_fast(X, lam):
    """Same algorithm and same arithmetic as e4_baseline_army.dp_means, with the per-centre
    python loop replaced by one numpy reduction over the live centres. Verified against the
    original at every grid point of every stream (see `dp_means_implementation_check`)."""
    C = np.empty_like(X)
    C[0] = X[0]
    nc = 1
    assign = [0]
    for t in range(1, len(X)):
        x = X[t]
        d2 = np.sum((C[:nc] - x) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] > lam:
            C[nc] = x
            assign.append(nc)
            nc += 1
        else:
            assign.append(j)
    return assign


def check_dp_means_impl(X, grid, n=400):
    """Run both implementations on the first n records at every grid point and compare the
    assignment lists exactly. If this is not True the fast path is not usable."""
    Xs = X[:n]
    rows = []
    for lam in grid:
        rows.append(dict(lam=lam, identical=bool(dp_means(Xs, lam) == dp_means_fast(Xs, lam))))
    return dict(n_records_checked=int(min(n, len(X))), per_lam=rows,
                all_identical=all(r["identical"] for r in rows))


# --------------------------------------------------------------------------- #
# agglomerative
# --------------------------------------------------------------------------- #
def agglo_sweeper(X):
    """The library estimator itself, one fit per distance threshold.

    A first version of this script cut a single scipy average-linkage tree with fcluster,
    which is much faster. It was rejected: on twogroup and wikipedia the two disagreed at
    thresholds where many record pairs sit at exactly that cosine distance (two records
    sharing 2 of 4 facets are at distance exactly 0.5), because scipy cuts with <= and
    sklearn with <. The disagreement was small (agreement ARI 0.89 to 1.00) but it is a
    disagreement, so the baseline is run as the library ships it. The rejected comparison is
    recorded under `agglo_implementation_check` for the record."""
    def cut(thr):
        return list(AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                            linkage="average", metric="cosine").fit_predict(X))
    return cut


def check_agglo_impl(X, cut, thrs):
    """The record of why the scipy shortcut was rejected: sklearn (what is used everywhere in
    this file) against one scipy average-linkage tree cut at the same distance."""
    Z = linkage(pdist(X, "cosine"), method="average")
    rows = []
    for thr in thrs:
        sk = cut(thr)
        sp = list(fcluster(Z, t=thr, criterion="distance"))
        rows.append(dict(threshold=thr, agreement_ARI=round(_ari(sk, sp), 6),
                         K_sklearn=len(set(sk)), K_scipy=len(set(sp))))
    return dict(what=("sklearn AgglomerativeClustering, which is what this experiment uses, "
                      "against a scipy linkage cut at the same distance, which was rejected"),
                used="sklearn.cluster.AgglomerativeClustering",
                per_threshold=rows,
                all_identical=all(abs(r["agreement_ARI"] - 1.0) < 1e-9 for r in rows))


# --------------------------------------------------------------------------- #
# the baseline suite on one (representation, stream)
# --------------------------------------------------------------------------- #
def baseline_suite(X, truth, true_K, verify=False):
    n = len(X)
    out = {}
    Dcos = pairwise_distances(X, metric="cosine")
    Deuc = pairwise_distances(X, metric="euclidean")
    cut = agglo_sweeper(X)

    # ---- k-means handed the true K (ORACLE by construction) ----
    lab = KMeans(true_K, n_init=10, random_state=0).fit_predict(X)
    out["kmeans_oracleK"] = dict(
        condition="ORACLE", knob="n_clusters", value=int(true_K),
        ARI=round(_ari(truth, list(lab)), 4), K=int(len(set(lab))),
        note="handed the true number of groups, which is test-label information")

    # ---- k-means with k by silhouette (no test labels touched) ----
    sil_rows = {}
    kmeans_labels = {}
    for k in KS_FOR_SILHOUETTE:
        l = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        kmeans_labels[k] = l
        sil_rows[k] = dict(cosine=float(silhouette_score(Dcos, l, metric="precomputed")),
                           euclidean=float(silhouette_score(Deuc, l, metric="precomputed")))
    sel = {}
    for m in ("cosine", "euclidean"):
        k_sel = max(sil_rows, key=lambda k: sil_rows[k][m])
        l = KMeans(k_sel, n_init=10, random_state=0).fit_predict(X)
        sel[m] = dict(K_chosen=int(k_sel), ARI=round(_ari(truth, list(l)), 4))
    out["kmeans_silhouette"] = dict(
        condition="SELECTED, no test labels", k_range=[KS_FOR_SILHOUETTE[0], KS_FOR_SILHOUETTE[-1]],
        by_cosine_silhouette=sel["cosine"], by_euclidean_silhouette=sel["euclidean"],
        silhouette_curve={str(k): {m: round(v, 4) for m, v in r.items()}
                          for k, r in sil_rows.items()},
        note="E4 selected by the euclidean silhouette; both metrics are reported here")

    # ---- k-means at the library default k = 8 ----
    kdef = LIBRARY_DEFAULTS["kmeans"]["default"]
    l = KMeans(kdef, n_init=10, random_state=0).fit_predict(X)
    out["kmeans_default"] = dict(condition="DEFAULT", knob="n_clusters", value=kdef,
                                 ARI=round(_ari(truth, list(l)), 4), K=int(len(set(l))),
                                 note="sklearn.cluster.KMeans() ships n_clusters=8")

    # ---- the knobbed sweeps ----
    def hdb(p):
        return list(HDBSCAN(min_cluster_size=int(p), metric="cosine",
                            copy=True).fit_predict(X))

    def agg(p):
        return cut(p)

    def dpm(p):
        return dp_means_fast(X, p)

    def ccc(p):
        return cosine_cc(X, p)

    for bname, fn in (("hdbscan", hdb), ("agglo_threshold", agg),
                      ("dp_means", dpm), ("cosine_components", ccc)):
        sweep, Ks = {}, {}
        for p in GRIDS[bname]:
            lp = fn(p)
            sweep[p] = round(_ari(truth, lp), 4)
            Ks[p] = int(len(set(lp)))
        best_p = max(sweep, key=lambda p: sweep[p])
        row = dict(
            grid=GRIDS[bname], grid_e4=GRIDS_E4[bname],
            grid_extended_beyond_e4=[p for p in GRIDS[bname] if p not in GRIDS_E4[bname]],
            sweep={str(p): sweep[p] for p in GRIDS[bname]},
            K_at={str(p): Ks[p] for p in GRIDS[bname]},
            oracle=dict(condition="ORACLE", value=best_p, ARI=sweep[best_p], K=Ks[best_p],
                        note="best on the test labels, an upper bound"),
            library_default=LIBRARY_DEFAULTS[bname])
        # DEFAULT
        d = LIBRARY_DEFAULTS[bname]["default"]
        if bname == "hdbscan":
            row["default"] = dict(condition="DEFAULT", value=d, ARI=sweep[d], K=Ks[d],
                                  metric="cosine",
                                  note="min_cluster_size=5 as shipped, on the cosine metric "
                                       "this experiment uses for both representations")
            lm = list(HDBSCAN(min_cluster_size=5, metric="euclidean", copy=True).fit_predict(X))
            row["default_library_metric"] = dict(
                condition="DEFAULT, constructor exactly", value=5, metric="euclidean",
                ARI=round(_ari(truth, lm), 4), K=int(len(set(lm))))
        elif bname == "agglo_threshold":
            lm = list(AgglomerativeClustering().fit_predict(X))
            row["default"] = dict(
                condition="DEFAULT, constructor exactly",
                value="n_clusters=2, linkage ward, metric euclidean, distance_threshold None",
                ARI=round(_ari(truth, lm), 4), K=int(len(set(lm))),
                note="AgglomerativeClustering() ships no distance_threshold at all")
        else:
            row["default"] = dict(condition="DEFAULT", value=None, ARI=None,
                                  available=False,
                                  note=LIBRARY_DEFAULTS[bname]["note"])
        out[bname] = row

    checks = {}
    if verify:
        checks["dp_means_implementation_check"] = check_dp_means_impl(X, GRIDS["dp_means"])
        checks["agglo_implementation_check"] = check_agglo_impl(
            X, cut, [0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    return out, checks, n


# --------------------------------------------------------------------------- #
def escrow_row(recs, truth):
    t0 = time.perf_counter()
    g, b = run_stream(recs)
    lab = [-1] * len(recs)
    for v in g.nodes.values():
        for m in v.members:
            lab[m - 1] = v.nid
    return dict(ARI=round(_ari(truth, lab), 4), K=int(g.K),
                background=int(sum(1 for x in lab if x == -1)),
                seconds=round(time.perf_counter() - t0, 2),
                protocol=protocol_describe(),
                representation="the records themselves, read as (key, value) facets; no vector")


# --------------------------------------------------------------------------- #
def add_transfers(arm):
    """TRANSFERRED: the oracle knob of another stream applied unchanged to this one, worst
    over the available sources, exactly as E4 does it."""
    names = list(arm.keys())
    for bname in ("hdbscan", "agglo_threshold", "dp_means", "cosine_components"):
        for tgt in names:
            per_source = {}
            for src in names:
                if src == tgt:
                    continue
                p = arm[src][bname]["oracle"]["value"]
                per_source[src] = dict(value=p, ARI=arm[tgt][bname]["sweep"][str(p)])
            if not per_source:
                arm[tgt][bname]["transferred"] = dict(
                    condition="TRANSFERRED", available=False,
                    note="only one stream in this run, so there is nothing to transfer from")
                continue
            worst = min(per_source, key=lambda s: per_source[s]["ARI"])
            arm[tgt][bname]["transferred"] = dict(
                condition="TRANSFERRED", per_source=per_source,
                worst_source=worst, worst_ARI=per_source[worst]["ARI"],
                worst_value=per_source[worst]["value"],
                best_source=max(per_source, key=lambda s: per_source[s]["ARI"]),
                best_ARI=max(v["ARI"] for v in per_source.values()),
                note="E4 reports the worst transfer; the best is carried here too")


# --------------------------------------------------------------------------- #
def falsifier(arm, escrow, rep_name):
    """Did any baseline match or beat ESCROW without tuning its knob on this stream's labels?

    strict   DEFAULT conditions and k-means by silhouette. No test labels anywhere.
    lenient  strict plus TRANSFERRED, which uses another stream's labels.
    """
    verdict = {}
    for name, row in arm.items():
        e = escrow[name]["ARI"]
        strict, lenient = [], []

        def add(bucket, label, ari):
            if ari is None:
                return
            bucket.append(dict(baseline=label, ARI=ari, beats_escrow=bool(ari > e),
                               matches_or_beats=bool(ari >= e - 1e-9)))

        add(strict, "kmeans_silhouette (cosine)",
            row["kmeans_silhouette"]["by_cosine_silhouette"]["ARI"])
        add(strict, "kmeans_silhouette (euclidean)",
            row["kmeans_silhouette"]["by_euclidean_silhouette"]["ARI"])
        add(strict, "kmeans_default (k=8)", row["kmeans_default"]["ARI"])
        for b in ("hdbscan", "agglo_threshold", "dp_means", "cosine_components"):
            d = row[b].get("default", {})
            if d.get("ARI") is not None:
                add(strict, f"{b} DEFAULT ({d.get('value')})", d["ARI"])
            dm = row[b].get("default_library_metric")
            if dm and dm.get("ARI") is not None:
                add(strict, f"{b} DEFAULT library metric", dm["ARI"])
        lenient = list(strict)
        for b in ("hdbscan", "agglo_threshold", "dp_means", "cosine_components"):
            t = row[b].get("transferred")
            if t and t.get("available", True):
                add(lenient, f"{b} TRANSFERRED worst ({t['worst_value']} from {t['worst_source']})",
                    t["worst_ARI"])
                add(lenient, f"{b} TRANSFERRED best ({t['best_source']})", t["best_ARI"])
        best_strict = max(strict, key=lambda r: r["ARI"]) if strict else None
        best_lenient = max(lenient, key=lambda r: r["ARI"]) if lenient else None
        oracle_best = max(
            [row["kmeans_oracleK"]["ARI"]]
            + [row[b]["oracle"]["ARI"] for b in ("hdbscan", "agglo_threshold",
                                                 "dp_means", "cosine_components")])
        verdict[name] = dict(
            representation=rep_name,
            escrow_ARI=e, escrow_K=escrow[name]["K"],
            best_untuned_strict=best_strict, best_untuned_lenient=best_lenient,
            best_oracle_ARI=oracle_best,
            falsifier_fires_strict=bool(best_strict and best_strict["matches_or_beats"]),
            falsifier_fires_lenient=bool(best_lenient and best_lenient["matches_or_beats"]),
            oracle_beats_escrow=bool(oracle_best >= e - 1e-9),
            all_untuned_strict=sorted(strict, key=lambda r: -r["ARI"]),
            all_untuned_lenient_extra=sorted(
                [r for r in lenient if r not in strict], key=lambda r: -r["ARI"]))
    return verdict


# --------------------------------------------------------------------------- #
def run_arm(streams, rep_fn, rep_name, verify_first=True):
    arm, checks = {}, {}
    for i, (name, (recs, truth)) in enumerate(streams.items()):
        t0 = time.perf_counter()
        X, dim = rep_fn(recs)
        true_K = len(set(truth))
        suite, chk, n = baseline_suite(X, truth, true_K, verify=verify_first)
        suite["_meta"] = dict(n=int(n), dim=int(dim), true_K=int(true_K),
                              representation=rep_name,
                              seconds=round(time.perf_counter() - t0, 2))
        arm[name] = suite
        if chk:
            checks[name] = chk
        print(f"  [{rep_name}/{name}] n={n} dim={dim} trueK={true_K} "
              f"{suite['_meta']['seconds']}s", flush=True)
        for b in ("hdbscan", "agglo_threshold", "dp_means", "cosine_components"):
            print(f"      {b:18s} oracle {suite[b]['oracle']['ARI']} at "
                  f"{suite[b]['oracle']['value']}   default "
                  f"{suite[b].get('default', {}).get('ARI')}", flush=True)
        print(f"      kmeans_oracleK     {suite['kmeans_oracleK']['ARI']}   "
              f"silhouette(cos) {suite['kmeans_silhouette']['by_cosine_silhouette']['ARI']} "
              f"at k={suite['kmeans_silhouette']['by_cosine_silhouette']['K_chosen']}   "
              f"k=8 default {suite['kmeans_default']['ARI']}", flush=True)
    add_transfers(arm)
    return arm, checks


# --------------------------------------------------------------------------- #
def summarise(report):
    lines = []
    esc = report["escrow"]
    for rep in [r for r in ("matched", "embedding") if r in report["falsifier"]]:
        v = report["falsifier"][rep]
        for name in PRIMARY_STREAMS:
            if name not in v:
                continue
            row = v[name]
            bs = row["best_untuned_strict"]
            lines.append(
                f"{rep}/{name}: ESCROW {row['escrow_ARI']} (K={row['escrow_K']}), best untuned "
                f"baseline {bs['ARI']} ({bs['baseline']}), best oracle "
                f"{row['best_oracle_ARI']}.")
    fired = [f"{rep}/{n}" for rep in report["falsifier"]
             for n, r in report["falsifier"][rep].items() if r["falsifier_fires_strict"]]
    fired_m = [n for n, r in report["falsifier"].get("matched", {}).items()
               if r["falsifier_fires_strict"]]
    fired_len_m = [n for n, r in report["falsifier"].get("matched", {}).items()
                   if r["falsifier_fires_lenient"]]
    prose = []
    if fired_m:
        prose.append(
            "THE FALSIFIER FIRES on the matched representation, on " + ", ".join(fired_m)
            + ". A distance baseline reaches or passes ESCROW on ESCROW's own representation "
              "with no knob tuned on the test labels, so on those streams the criterion is not "
              "what is doing the work. This is reported because it is what was measured.")
    else:
        prose.append("The falsifier does not fire on the matched representation under the "
                     "strict reading on any stream.")
    if fired_len_m and set(fired_len_m) - set(fired_m):
        prose.append("Under the lenient reading, which also allows a knob transferred from "
                     "another stream, it additionally fires on "
                     + ", ".join(sorted(set(fired_len_m) - set(fired_m))) + ".")
    prose.append(
        "Read the two arms against each other rather than each on its own: the point of the "
        "experiment is that a baseline's score is a property of the pair (rule, representation) "
        "and moves a long way when the representation changes with the rule held fixed.")
    return dict(per_cell=lines,
                summary=" ".join(prose),
                falsifier_fires_strict_anywhere=bool(fired),
                cells_where_it_fires_strict=fired,
                matched_cells_where_it_fires_strict=fired_m,
                matched_cells_where_it_fires_lenient=fired_len_m)


# --------------------------------------------------------------------------- #
def main():
    quick = "--quick" in sys.argv
    do_embed = "--no-embedding" not in sys.argv
    t_start = time.time()

    streams = {"twogroup": two_group(), "planted8": planted8(),
               "wikipedia": wikipedia(WIKI_CACHE)}
    if quick:
        streams = {"wikipedia": streams["wikipedia"]}

    report = dict(
        experiment="e21_matched_representation",
        title="the same input for everyone, and what the baselines do at their own defaults",
        objection=("E4 gives every distance baseline a sentence-transformer embedding that "
                   "ESCROW never sees, so it compares rule plus representation against rule "
                   "alone; and E4 reports only oracle-tuned and transferred knobs, never the "
                   "library default a practitioner actually gets"),
        falsifier=("if a distance baseline on ESCROW's own representation matches or beats "
                   "ESCROW without its knob being tuned on the test labels, the criterion is "
                   "not what is doing the work"),
        protocol=protocol_describe(),
        representations=dict(
            matched=("binary multi-hot over the observed (key, value) pairs of the stream and over "
                     "the keys themselves, because the engine prices key presence too, rows "
                     "L2 normalised; exactly the information ESCROW codes, given to the "
                     "baselines in full and up front, which is more than ESCROW gets"),
            embedding=(f"{EMBED_MODEL} over the sorted 'k=v' text of the record, unit norm; "
                       "E4's own encoding, recomputed here")),
        metric=("cosine on both representations. On a multi-hot indicator the cosine of two "
                "records is the number of shared (key, value) pairs over the geometric mean of "
                "their facet counts, which is the natural overlap for a categorical record; "
                "unnormalised euclidean would push any record with many facets away from "
                "everything. Both representations are unit norm, so cosine and euclidean order "
                "pairs identically and only the silhouette value differs; where a library's "
                "shipped default names euclidean, that is run as shipped and reported under "
                "default_library_metric."),
        conditions=dict(
            ORACLE="knob swept, best score on the test labels taken; an upper bound",
            TRANSFERRED="oracle knob of another stream applied unchanged, worst over sources",
            DEFAULT="the value the library ships, recorded literally"),
        grids=GRIDS, grids_e4=GRIDS_E4,
        grids_note=("every E4 grid point is retained so a transfer means the same thing in both "
                    "files; the agglomerative, DP-means and cosine-component grids are extended "
                    "upward because on a multi-hot indicator most record pairs are orthogonal "
                    "and E4's top point cut the sweep off before the useful region. Extending a "
                    "baseline's grid can only help the baseline."),
        library_defaults=LIBRARY_DEFAULTS,
        silhouette_k_range=[KS_FOR_SILHOUETTE[0], KS_FOR_SILHOUETTE[-1]],
        environment=("the repository venv carries numpy, scipy and scikit-learn but not torch or "
                     "sentence-transformers, so the embedding arm needs both. They were installed "
                     "with pip --target into a scratch directory put on PYTHONPATH for the run; "
                     "the repository venv was not modified. torch 2.14.0, sentence-transformers "
                     "6.0.1, transformers 5.16.1, scikit-learn 1.9.0, numpy 2.5.2. These are "
                     "newer than whatever produced results/e4_first_tier.json, which is why the "
                     "embedding arm is checked against it rather than assumed identical."),
        deviations={}, escrow={}, arms={}, implementation_checks={})

    # ---- ESCROW ----
    print("ESCROW on the primary streams", flush=True)
    for name, (recs, truth) in streams.items():
        report["escrow"][name] = escrow_row(recs, truth)
        print(f"  [escrow/{name}] ARI={report['escrow'][name]['ARI']} "
              f"K={report['escrow'][name]['K']} bg={report['escrow'][name]['background']}",
              flush=True)

    # ---- matched arm ----
    print("\nmatched representation", flush=True)
    arm_m, chk_m = run_arm(streams, matched_representation, "matched", verify_first=True)
    report["arms"]["matched"] = arm_m
    report["implementation_checks"]["matched"] = chk_m

    # ---- embedding arm ----
    if do_embed:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBED_MODEL)
            print("\nembedding representation", flush=True)
            arm_e, chk_e = run_arm(streams, lambda r: embedding_representation(r, model),
                                   "embedding", verify_first=False)
            report["arms"]["embedding"] = arm_e
            report["implementation_checks"]["embedding"] = chk_e
        except Exception as exc:
            report["deviations"]["embedding_arm"] = (
                f"skipped: {type(exc).__name__}: {exc}. The embedding numbers already in "
                "results/e4_first_tier.json are the comparison.")
            print("embedding arm skipped:", exc, flush=True)
    else:
        report["deviations"]["embedding_arm"] = (
            "skipped by --no-embedding. The embedding numbers already in "
            "results/e4_first_tier.json are the comparison.")

    # ---- falsifier ----
    report["falsifier"] = {rep: falsifier(report["arms"][rep], report["escrow"], rep)
                           for rep in report["arms"]}
    report["reading"] = summarise(report)

    # ---- seed family on the matched arm ----
    if not quick:
        print("\nseed family, matched representation, seeds 0 to 4", flush=True)
        fam = {}
        for name in PRIMARY_STREAMS:
            rows = []
            for s in SEED_FAMILY:
                recs, truth = stream_for_seed(name, s, WIKI_CACHE)
                X, _ = matched_representation(recs)
                true_K = len(set(truth))
                suite, _, _ = baseline_suite(X, truth, true_K, verify=False)
                e = escrow_row(recs, truth)
                untuned = [suite["kmeans_silhouette"]["by_cosine_silhouette"]["ARI"],
                           suite["kmeans_silhouette"]["by_euclidean_silhouette"]["ARI"],
                           suite["kmeans_default"]["ARI"],
                           suite["hdbscan"]["default"]["ARI"],
                           suite["hdbscan"]["default_library_metric"]["ARI"],
                           suite["agglo_threshold"]["default"]["ARI"]]
                rows.append(dict(seed=s, escrow_ARI=e["ARI"], escrow_K=e["K"],
                                 best_untuned_strict_ARI=max(untuned),
                                 kmeans_oracleK_ARI=suite["kmeans_oracleK"]["ARI"],
                                 best_oracle_ARI=max(
                                     [suite["kmeans_oracleK"]["ARI"]]
                                     + [suite[b]["oracle"]["ARI"] for b in
                                        ("hdbscan", "agglo_threshold", "dp_means",
                                         "cosine_components")]),
                                 oracle_values={b: suite[b]["oracle"]["value"] for b in
                                                ("hdbscan", "agglo_threshold", "dp_means",
                                                 "cosine_components")}))
                print(f"  [{name} seed {s}] escrow {rows[-1]['escrow_ARI']} "
                      f"(K={rows[-1]['escrow_K']})  best untuned "
                      f"{rows[-1]['best_untuned_strict_ARI']}  best oracle "
                      f"{rows[-1]['best_oracle_ARI']}", flush=True)
            fam[name] = dict(
                per_seed=rows,
                escrow_ARI_mean=round(sum(r["escrow_ARI"] for r in rows) / len(rows), 4),
                best_untuned_ARI_mean=round(
                    sum(r["best_untuned_strict_ARI"] for r in rows) / len(rows), 4),
                seeds_where_untuned_matches_or_beats_escrow=sum(
                    1 for r in rows if r["best_untuned_strict_ARI"] >= r["escrow_ARI"] - 1e-9))
        report["seed_family_matched"] = fam
    else:
        report["deviations"]["seed_family"] = "skipped in --quick"

    # ---- reproduction check against E4 ----
    e4path = os.path.join(RESULTS, "e4_first_tier.json")
    if os.path.exists(e4path) and "embedding" in report["arms"]:
        e4 = json.load(open(e4path))
        rows = []
        for name in report["arms"]["embedding"]:
            if name not in e4:
                continue
            rows.append(dict(
                stream=name,
                e4_ours_ARI=e4[name]["ours"]["ARI"], e21_escrow_ARI=report["escrow"][name]["ARI"],
                e4_kmeans_oracleK=e4[name]["kmeans_oracleK"]["ARI"],
                e21_kmeans_oracleK=report["arms"]["embedding"][name]["kmeans_oracleK"]["ARI"],
                e4_kmeans_silhouette=e4[name]["kmeans_silhouette"]["ARI"],
                e21_kmeans_silhouette_euclidean=report["arms"]["embedding"][name][
                    "kmeans_silhouette"]["by_euclidean_silhouette"]["ARI"],
                e4_hdbscan_oracle=e4[name]["hdbscan"]["oracle_ARI"],
                e21_hdbscan_oracle=report["arms"]["embedding"][name]["hdbscan"]["oracle"]["ARI"]))
        report["e4_reproduction_check"] = dict(
            what=("the embedding arm here against results/e4_first_tier.json. The sentence "
                  "transformer library version is not pinned in this repository, so small "
                  "differences are expected and are reported rather than hidden; the ESCROW "
                  "rows must match exactly."),
            rows=rows,
            escrow_rows_match=all(abs(r["e4_ours_ARI"] - r["e21_escrow_ARI"]) < 1e-9
                                  for r in rows))

    report["seconds_total"] = round(time.time() - t_start, 2)
    stamped(report)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\n" + "\n".join(report["reading"]["per_cell"]))
    print("\n" + report["reading"]["summary"])
    print("\nfalsifier fires (strict) on matched:",
          report["reading"]["matched_cells_where_it_fires_strict"])
    print("falsifier fires (lenient) on matched:",
          report["reading"]["matched_cells_where_it_fires_lenient"])
    print(f"\nwrote {OUT} in {report['seconds_total']}s")


if __name__ == "__main__":
    main()
