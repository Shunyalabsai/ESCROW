"""E78: the classical field on the same noise ladder, so one graph holds every method.

WHY. E75 built a stream whose groups live in the values and walked it from full signal to pure
noise. It reported one method. A ladder with one curve on it says how our rule behaves, and says
nothing about whether the behaviour is worth anything, because the reader cannot see what an
ordinary method does on the same rungs. E72 put a field of streaming methods against us, but on a
single encyclopedia stream at one operating point, so it could not show what happens when the signal
is turned down. This experiment puts both together: every method from E4 and E72 that can be run
here, on all eleven rungs of the E75 ladder, at the same three seeds.

THE POINT OF THE LADDER, AND WHY THE BOTTOM RUNG IS THE ONE THAT MATTERS. At signal 1.0 each group
draws only from its own slice of the shared vocabulary, and the groups are separable. At signal 0.0
every group draws from the whole vocabulary, every record carries the same six keys, and the correct
answer is zero nodes. Nothing distinguishes one record from another, so a method that reports eight
groups there has invented all eight. The top of the ladder measures accuracy. The bottom measures
whether a method knows when to say nothing. Most methods cannot say nothing: k-means is handed a k,
DP-means (the Dirichlet process means algorithm) returns at least one cluster, leader clustering
returns at least one cluster. HDBSCAN (hierarchical density based spatial clustering of applications
with noise) can call every point noise, so it is the one classical method here that can report zero
nodes, and it does report zero once its smallest cluster size is raised far enough. So passing the
bottom rung on its own is not the test. The test is whether ONE setting of a knob passes both rungs,
and this file measures that for every knobbed method at every value of its grid.

WHAT EVERY METHOD IS GIVEN, WHICH IS THE SAME THING. Every method sees the same records. The
baselines see them through the multi-hot vector E72 builds: one column per observed (key, value)
pair and one column per observed key, then the row is scaled to unit length. That is exactly what
the record contains and nothing else. No embedding, no text, no order information, no count of
groups except where a row says the true k was handed over. `representation_check` turns that
sentence into a measurement: it rebuilds every record from its own row and requires the rebuilt
stream to equal the original, so the baselines are given the record and not a summary of it. The
batch methods are also allowed to see all three thousand records at once and to revisit them, which
ESCROW never does, so where the comparison is uneven it is uneven in their favour.

THE E4 TRIPLE, APPLIED AT EVERY RUNG. Where a method has a knob it is reported twice. "Oracle" means
the knob was swept and the best value picked by looking at the test labels, which is not available in
use and is stated as such. "Transferred" means the knob was picked on a different stream, by sweeping
on that stream's own labels, and then carried here unchanged. That is what a practitioner actually
holds. ESCROW has no calibrated parameter and gets one run per arm. HDBSCAN is swept the same way,
on its smallest cluster size, because it is the only method here that can report nothing and so the
only one whose knob could pass both rungs. Running it at its library defaults alone would have hidden
that, and an earlier version of this file did exactly that.

THE TWO ESCROW ARMS. `escrow_base` is the shipped engine before the split move. `escrow_split` adds
the split move and the residual mint. Both are run fresh here, not quoted from another file, and the
flag is set explicitly before each run.

WHAT WOULD REFUTE THE PAPER'S CLAIM. A baseline that is not handed k and whose knob was not picked
on these labels, tracking ESCROW across the whole ladder: high agreement where there is signal AND
no invented groups at signal 0. If that happens the claim that the rule earns something by needing
nothing told is wrong, and this file has to say so plainly.

A SECOND AND WEAKER REFUTATION, ALSO MEASURED HERE. One value of some knob that does both halves,
even if finding that value needs the labels. That would say the classical method is capable of the
behaviour and only lacks a way to reach the setting, which is a much smaller claim for us than the
one the paper makes. It is reported on its own line so the two are never confused.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

from escrow import batch as B
from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped

from e4_baseline_army import _ari, dp_means
from e72_more_streaming_baselines import birch_like, leader, multihot, order, seq_kmeans
from e75_a_fixture_that_tests_the_values import keysig_baseline, stream

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e78_baselines_on_the_noise_ladder.json")

N = int(os.environ.get("E78_N", "3000"))
SEEDS = (0, 1, 2)
GROUPS = 8
SIGNAL = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)

# The development stream the transferred knobs are chosen on. A different seed, so different
# records, and full signal, so it is the most helpful development set a practitioner could have.
DEV_SIGNAL, DEV_SEED = 1.0, 100

# Grids. E4 and E72 chose theirs for a sentence embedding, where the distances sit elsewhere. On
# this multi-hot representation every record shares all six key columns, so two unrelated records
# already agree by half and the interesting band is narrow. The grids below are the E4 and E72 grids
# with extra values added inside that band. A grid that cannot reach the right answer would make the
# baseline look worse than it is, which is the wrong direction to be unfair in.
AGGLO_CUTS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.495, 0.5, 0.7)
LAMBDAS = (0.1, 0.25, 0.5, 0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25, 1.5)
RADII = (0.2, 0.35, 0.5, 0.65, 0.8, 0.88, 0.92, 0.95, 0.98, 0.99, 1.0, 1.1, 1.3)
K_RANGE = range(2, 21)

# HDBSCAN's smallest cluster size. 5 is the library default, so the row called `hdbscan_defaults` is
# read off this grid rather than fitted again. The top of the grid is above 3000 / 8 = 375, the size
# of a true group, because that is where the method starts refusing to call anything a cluster, and
# refusing is the behaviour the bottom rung is about.
HDBSCAN_MIN_SIZES = (5, 25, 100, 200, 400)
HDBSCAN_DEFAULT_MIN_SIZE = 5

# The range k-means is allowed when silhouette picks k. The main row keeps 2 to 20, which is what
# every method is told. WIDE_K is used once, at signal 0, only to answer whether that row's answer is
# its own or the top of the range it was handed.
WIDE_K = tuple(range(2, 61))


# --------------------------------------------------------------------------- #
# The same algorithms as E4 and E72, written so they finish on 3000 records.
#
# E4's dp_means and E72's leader walk their centre list in Python, which is fine on the 320 record
# encyclopedia stream and quadratic here. These three do the identical arithmetic with the centre
# list held as one array. `equivalence_check` below runs the originals and these side by side on a
# smaller stream and requires the labels to match exactly, so the speed is not bought with a change
# of method. Everything else is imported from E4 and E72 unchanged.
# --------------------------------------------------------------------------- #
class Centres:
    """A growable array of centres, so the distance to all of them is one call."""

    def __init__(self, dim):
        self.buf = np.zeros((64, dim))
        self.n = 0

    def add(self, x):
        if self.n == len(self.buf):
            self.buf = np.vstack([self.buf, np.zeros_like(self.buf)])
        self.buf[self.n] = x
        self.n += 1
        return self.n - 1

    @property
    def view(self):
        return self.buf[:self.n]


def dp_means_fast(X, lam):
    """E4's dp_means: attach to the nearest centre, or open a new one if the squared distance to it
    is above lam. Centres are the record that opened them and are never moved."""
    C = Centres(X.shape[1])
    C.add(X[0])
    assign = [0]
    for x in X[1:]:
        d2 = ((C.view - x) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        if d2[j] > lam:
            assign.append(C.add(x))
        else:
            assign.append(j)
    return assign


def leader_fast(X, radius, seed):
    """E72's leader clustering: attach to the FIRST centre within the radius, else become one."""
    o = order(len(X), seed)
    C = Centres(X.shape[1])
    lab = [None] * len(X)
    for idx in o:
        x = X[idx]
        hit = np.flatnonzero(np.linalg.norm(C.view - x, axis=1) <= radius)
        lab[idx] = int(hit[0]) if hit.size else C.add(x)
    return lab


def birch_fast(X, radius, seed, branching=50):
    """E72's BIRCH-like pass: absorb into the nearest summary if within the radius, else open a new
    one, and never keep more than `branching` of them."""
    o = order(len(X), seed)
    S = Centres(X.shape[1])
    counts = []
    lab = [None] * len(X)
    for idx in o:
        x = X[idx]
        if S.n:
            d = np.linalg.norm(S.view / np.asarray(counts)[:, None] - x, axis=1)
            best, bd = int(np.argmin(d)), float(d.min())
        else:
            best, bd = None, None
        if best is not None and bd <= radius:
            S.buf[best] += x
            counts[best] += 1
            lab[idx] = best
        elif S.n < branching:
            j = S.add(x)
            counts.append(1)
            lab[idx] = j
        else:
            S.buf[best] += x
            counts[best] += 1
            lab[idx] = best
    return lab


def representation_check(n=400, signal=0.7, seed=9):
    """Does the multi-hot vector carry the whole record, so the baselines see what ESCROW sees?

    This is the apple to apple rule turned into a measurement. Every record is rebuilt from its own
    row and the rebuilt stream has to equal the original one. If it does, the baselines were handed
    the record itself and nothing was held back from them. The second line asks whether the key
    columns are on in every record, which they are on this fixture because every record carries the
    same six keys, so those columns separate no two records for either side."""
    recs, _ = stream(signal, seed, n=n)
    X = multihot(recs)
    cols = sorted({f"{k}={v}" for r in recs for k, v in r.items()}
                  | {f"KEY:{k}" for r in recs for k in r})
    rebuilt = []
    for row in X:
        rec = {}
        for j in np.flatnonzero(row > 0):
            c = cols[j]
            if not c.startswith("KEY:"):
                k, v = c.split("=", 1)
                rec[k] = v
        rebuilt.append(rec)
    key_cols = [j for j, c in enumerate(cols) if c.startswith("KEY:")]
    return {"records": n, "columns": len(cols),
            "the_multi_hot_rebuilds_every_record": rebuilt == recs,
            "the_key_columns_are_on_in_every_record": bool((X[:, key_cols] > 0).all()),
            "note": ("the first line says the baselines were given the record and not a summary of "
                     "it. The second says the key names carry nothing here, so neither side gains "
                     "from seeing them")}


def equivalence_check(n=400, signal=0.7, seed=9):
    """Run the originals and the fast copies on the same small stream and require equal labels."""
    recs, _ = stream(signal, seed, n=n)
    X = multihot(recs)
    out = {"records": n, "signal": signal, "seed": seed}
    out["dp_means"] = all(list(dp_means(X, lam)) == list(dp_means_fast(X, lam)) for lam in LAMBDAS)
    out["leader"] = all(list(leader(X, r, 0)) == list(leader_fast(X, r, 0)) for r in RADII)
    out["birch_like"] = all(list(birch_like(X, r, 0)) == list(birch_fast(X, r, 0)) for r in RADII)
    out["hdbscan_default_is_the_smallest_cluster_size_5"] = (
        list(HDBSCAN().fit_predict(X)) ==
        list(HDBSCAN(min_cluster_size=HDBSCAN_DEFAULT_MIN_SIZE).fit_predict(X)))
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(truth, lab):
    """ARI (the adjusted Rand index), the number of nodes, and how many records are in a node.

    A label of -1 means the record is in no node. ESCROW leaves the background at -1 and HDBSCAN
    calls it noise, and both are counted the same way here: the background is one cluster for ARI,
    which is the convention escrow.protocol.record_labels already documents, and it is not counted
    as a node."""
    lab = [int(x) for x in lab]
    return {"ARI": round(_ari(truth, lab), 4),
            "K": len({x for x in lab if x != -1}),
            "placed": sum(1 for x in lab if x != -1)}


def mean_of(rows, field):
    return round(sum(r[field] for r in rows) / len(rows), 4)


# --------------------------------------------------------------------------- #
# One stream: every method, every knob value.
# --------------------------------------------------------------------------- #
def run_one_stream(signal, seed, sweeps_only=False):
    """Everything measured on a single (signal, seed) stream, keyed by method name.

    `sweeps_only` returns the knob sweeps and skips every method that has no knob. The development
    stream that fixes the transferred values reads the sweeps and nothing else, so running the rest
    there would only cost time."""
    recs, truth = stream(signal, seed, n=N)
    X = multihot(recs)
    Dc = pairwise_distances(X, metric="cosine")
    res, sweeps = {}, {}

    # --- the knobs, swept so both the oracle and the transferred value can be read off --- #
    sweeps["agglomerative"] = {c: score(truth, AgglomerativeClustering(
        n_clusters=None, distance_threshold=c, linkage="average",
        metric="precomputed").fit_predict(Dc)) for c in AGGLO_CUTS}
    sweeps["dp_means"] = {l: score(truth, dp_means_fast(X, l)) for l in LAMBDAS}
    sweeps["leader"] = {r: score(truth, leader_fast(X, r, seed)) for r in RADII}
    sweeps["birch_like"] = {r: score(truth, birch_fast(X, r, seed)) for r in RADII}
    sweeps["hdbscan"] = {m: score(truth, HDBSCAN(min_cluster_size=m).fit_predict(X))
                         for m in HDBSCAN_MIN_SIZES}
    if sweeps_only:
        return res, sweeps

    # --- the two ESCROW arms, fresh, flag set explicitly before each run --- #
    B.SPLIT_MOVES = False
    g, _ = run_stream(recs)
    res["escrow_base"] = score(truth, record_labels(g, len(recs)))
    B.SPLIT_MOVES = True
    g, _ = run_stream(recs)
    res["escrow_split"] = score(truth, record_labels(g, len(recs)))

    # --- the degenerate nulls and the key signature shortcut --- #
    res["null_one_node_per_record"] = score(truth, list(range(len(recs))))
    res["null_one_node_total"] = score(truth, [0] * len(recs))
    ks_ari, ks_n = keysig_baseline(recs, truth)
    res["key_signature"] = {"ARI": ks_ari, "K": ks_n, "placed": len(recs)}

    # --- told the true number of groups --- #
    res["kmeans_true_k"] = score(truth, KMeans(GROUPS, n_init=10, random_state=0).fit_predict(X))
    res["seq_kmeans_true_k"] = score(truth, seq_kmeans(X, GROUPS, seed))
    res["seq_kmeans_spread_true_k"] = score(truth, seq_kmeans(X, GROUPS, seed, plusplus=True))

    # --- label free --- #
    De = pairwise_distances(X)
    sil = {}
    for k in K_RANGE:
        lk = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        sil[k] = float(silhouette_score(De, lk, metric="precomputed"))
    k_sil = max(sil, key=sil.get)
    res["kmeans_silhouette"] = score(truth, KMeans(k_sil, n_init=10, random_state=0).fit_predict(X))
    res["kmeans_silhouette"]["k_chosen"] = int(k_sil)
    res["hdbscan_defaults"] = dict(sweeps["hdbscan"][HDBSCAN_DEFAULT_MIN_SIZE])
    return res, sweeps


def silhouette_cap_check(signal=0.0, seeds=SEEDS):
    """Is the kmeans_silhouette row's answer at signal 0 its own, or the top of the range it got?

    The main row hands k-means a range of 2 to 20 and lets silhouette pick inside it with no labels.
    At signal 0 it picks 20 every time, which is the top of that range, so the reported number of
    nodes may be the range talking rather than the method. This runs the same choice with the range
    opened to 60 and keeps the whole curve, so a reader can see whether the score is still climbing
    at 20. It is a diagnostic and does not change the main row, which stays at 2 to 20 because that
    is what the method was told."""
    out = {"signal": signal, "range": [WIDE_K[0], WIDE_K[-1]], "per_seed": []}
    for s in seeds:
        recs, truth = stream(signal, s, n=N)
        X = multihot(recs)
        De = pairwise_distances(X)
        sil = {}
        for k in WIDE_K:
            lk = KMeans(k, n_init=4, random_state=0).fit_predict(X)
            sil[k] = round(float(silhouette_score(De, lk, metric="precomputed")), 5)
        k_best = int(max(sil, key=sil.get))
        row = dict(score(truth, KMeans(k_best, n_init=10, random_state=0).fit_predict(X)))
        row["seed"] = s
        row["k_chosen"] = k_best
        row["score_at_the_top_of_the_told_range"] = sil.get(max(K_RANGE))
        row["score_at_the_top_of_the_wide_range"] = sil[WIDE_K[-1]]
        row["curve"] = {str(k): v for k, v in sil.items()}
        out["per_seed"].append(row)
    out["k_chosen_per_seed"] = [r["k_chosen"] for r in out["per_seed"]]
    out["reading"] = ("if the chosen k sits at the top of the wide range too, then the number of "
                      "nodes this method reports at signal 0 is set by the range it is given and "
                      "not by the data, and the 20 in the main table understates what it invents")
    return out


KNOBBED = {"agglomerative": ("a cut on the cosine distance", AGGLO_CUTS),
           "dp_means": ("a lambda", LAMBDAS),
           "leader": ("a radius", RADII),
           "birch_like": ("a radius", RADII),
           "hdbscan": ("a smallest cluster size", HDBSCAN_MIN_SIZES)}

TOLD = {
    "escrow_base": "nothing",
    "escrow_split": "nothing",
    "null_one_node_per_record": "nothing",
    "null_one_node_total": "nothing",
    "key_signature": "nothing",
    "kmeans_true_k": "the true number of groups",
    "seq_kmeans_true_k": "the true number of groups",
    "seq_kmeans_spread_true_k": "the true number of groups",
    "kmeans_silhouette": "a range of k to search, 2 to 20",
    "hdbscan_defaults": "nothing",
}
for _m, (_w, _g) in KNOBBED.items():
    TOLD[_m + "_oracle"] = _w + ", picked on the test labels"
    TOLD[_m + "_transferred"] = _w + ", picked on a different stream"

TOLD_SHORT = {
    "escrow_base": "nothing",
    "escrow_split": "nothing",
    "null_one_node_per_record": "nothing",
    "null_one_node_total": "nothing",
    "key_signature": "nothing",
    "kmeans_true_k": "the true k",
    "seq_kmeans_true_k": "the true k",
    "seq_kmeans_spread_true_k": "the true k",
    "kmeans_silhouette": "a k range, 2 to 20",
    "hdbscan_defaults": "nothing",
    "agglomerative_oracle": "a cut, oracle",
    "agglomerative_transferred": "a cut, transferred",
    "dp_means_oracle": "a lambda, oracle",
    "dp_means_transferred": "a lambda, transferred",
    "leader_oracle": "a radius, oracle",
    "leader_transferred": "a radius, transferred",
    "birch_like_oracle": "a radius, oracle",
    "birch_like_transferred": "a radius, transferred",
    "hdbscan_oracle": "a size, oracle",
    "hdbscan_transferred": "a size, transferred",
}

STREAMING = {"escrow_base", "escrow_split", "seq_kmeans_true_k", "seq_kmeans_spread_true_k",
             "dp_means_oracle", "dp_means_transferred", "leader_oracle", "leader_transferred",
             "birch_like_oracle", "birch_like_transferred", "key_signature"}

ROW_ORDER = ["escrow_base", "escrow_split", "kmeans_silhouette", "hdbscan_defaults",
             "hdbscan_oracle", "hdbscan_transferred",
             "kmeans_true_k", "seq_kmeans_true_k", "seq_kmeans_spread_true_k",
             "agglomerative_oracle", "agglomerative_transferred",
             "dp_means_oracle", "dp_means_transferred",
             "leader_oracle", "leader_transferred",
             "birch_like_oracle", "birch_like_transferred",
             "key_signature", "null_one_node_per_record", "null_one_node_total"]


def choose_transferred():
    """Sweep each knob on the development stream and keep the value with the best ARI there.

    A tie is broken by the first value in the grid, which is the smallest one."""
    _, sw = run_one_stream(DEV_SIGNAL, DEV_SEED, sweeps_only=True)
    picked = {}
    for name, grid in sw.items():
        best = max(grid, key=lambda p: grid[p]["ARI"])
        picked[name] = {"value": best, "ARI_on_the_development_stream": grid[best]["ARI"],
                        "K_on_the_development_stream": grid[best]["K"]}
    return picked


def one_knob_value_test(rows, with_signal):
    """For every knobbed method, walk its grid and ask each value to do both halves at once.

    The oracle column picks a new value at every rung, so it never has to hold one setting across
    the ladder. This does. For each value it reports the mean agreement over the rungs that have
    signal and the number of nodes that same value reports at signal 0, where the answer is zero.
    A value that scores well and reports zero would be a classical method doing both halves, which
    is the weaker refutation this file is looking for."""
    out = {}
    at_zero = rows[0]["knob_sweep"]
    for m, (_w, grid) in KNOBBED.items():
        every = []
        for v in grid:
            key = str(v)
            aris = [r["knob_sweep"][m][key]["ARI_mean"] for r in with_signal]
            every.append({"knob": v,
                          "mean_ARI_where_there_is_signal": round(sum(aris) / len(aris), 4),
                          "nodes_at_signal_zero": at_zero[m][key]["K_mean"]})
        silent = [e for e in every if e["nodes_at_signal_zero"] == 0]
        out[m] = {
            "every_value": every,
            "values_reporting_zero_nodes_at_signal_zero": [e["knob"] for e in silent],
            "best_mean_ARI_among_the_silent_values":
                max((e["mean_ARI_where_there_is_signal"] for e in silent), default=None),
            "best_mean_ARI_over_the_whole_grid":
                max(e["mean_ARI_where_there_is_signal"] for e in every),
        }
    return out


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.time()
    report = {
        "experiment": "E78 the classical field on the E75 noise ladder",
        "why": ("E75 walked the ladder with one method on it, so the reader could not see what an "
                "ordinary method does on the same rungs"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor} (the one with scikit-learn)",
        "records_per_stream": N, "groups": GROUPS, "seeds": list(SEEDS),
        "signal_ladder": list(SIGNAL),
        "representation": ("the E72 multi-hot vector: one column per observed (key, value) pair and "
                           "one column per observed key, row scaled to unit length. Every baseline "
                           "gets this and nothing else, so it sees exactly the records ESCROW sees"),
        "reused_from": {
            "e4_baseline_army": "_ari, dp_means",
            "e72_more_streaming_baselines": "multihot, order, seq_kmeans, leader, birch_like",
            "e75_a_fixture_that_tests_the_values": "stream, keysig_baseline",
            "scikit_learn": "KMeans, AgglomerativeClustering, HDBSCAN, silhouette_score",
            "rewritten_for_speed_only": ("dp_means_fast, leader_fast, birch_fast do the same "
                                         "arithmetic with the centre list held as one array, and "
                                         "the equivalence check requires equal labels"),
        },
        "what_each_method_was_given": TOLD,
        "methods_that_read_one_record_at_a_time_and_never_revisit": sorted(STREAMING),
        "methods_that_see_the_whole_batch": sorted(set(ROW_ORDER) - STREAMING),
        "development_stream_for_the_transferred_knobs":
            f"the same fixture at signal {DEV_SIGNAL}, seed {DEV_SEED}, which is a different stream",
    }
    # escrow.provenance stamps engine.py, batch.py and codes.py. The split arm is decided by
    # split.py and every run here goes through protocol.py, so their md5s are recorded too, or the
    # stamp would not identify the code that produced the escrow_split row.
    _esc = os.path.dirname(os.path.abspath(B.__file__))
    report["engine_files_the_provenance_stamp_does_not_cover"] = {
        name: md5_of(os.path.join(_esc, name)) for name in ("split.py", "protocol.py")}

    print("checking that the multi-hot vector carries the whole record ...")
    report["representation_check"] = representation_check()
    print("  ", json.dumps(report["representation_check"]))
    if not report["representation_check"]["the_multi_hot_rebuilds_every_record"]:
        raise SystemExit("the baselines are not being given the whole record, stopping")

    print("checking the fast copies against the originals ...")
    report["equivalence_check"] = equivalence_check()
    print("  ", json.dumps(report["equivalence_check"]))
    if not all(v for k, v in report["equivalence_check"].items() if isinstance(v, bool)):
        raise SystemExit("a fast copy does not match its original, stopping")

    print(f"\npicking the transferred knobs on the development stream "
          f"(signal {DEV_SIGNAL}, seed {DEV_SEED}) ...")
    transferred = choose_transferred()
    report["transferred_knobs"] = transferred
    for m, v in transferred.items():
        print(f"   {m:<16} {v['value']}   (ARI {v['ARI_on_the_development_stream']} there)")

    print(f"\nwalking the ladder, {len(SIGNAL)} rungs by {len(SEEDS)} seeds "
          f"by {N} records ...\n")
    rows = []
    for sig in SIGNAL:
        per_method, smallest_at_zero, per_value = {}, {}, {}
        for s in SEEDS:
            res, sw = run_one_stream(sig, s)
            if sig == 0.0:
                for name, grid in sw.items():
                    smallest_at_zero.setdefault(name, []).append(
                        min(v["K"] for v in grid.values()))
            for name, grid in sw.items():
                for val, sc in grid.items():
                    per_value.setdefault(name, {}).setdefault(str(val), []).append(sc)
            for name, grid in sw.items():
                best = max(grid, key=lambda p: grid[p]["ARI"])
                res[name + "_oracle"] = dict(grid[best], knob=best)
                tv = transferred[name]["value"]
                res[name + "_transferred"] = dict(grid[tv], knob=tv)
            for name, v in res.items():
                per_method.setdefault(name, []).append(v)
        row = {"signal": sig, "noise": round(1 - sig, 2), "methods": {}}
        for name, vals in per_method.items():
            row["methods"][name] = {
                "ARI_mean": mean_of(vals, "ARI"), "ARI_per_seed": [v["ARI"] for v in vals],
                "K_mean": mean_of(vals, "K"), "K_per_seed": [v["K"] for v in vals],
                "placed": sum(v["placed"] for v in vals), "of": N * len(SEEDS),
                "knob_per_seed": [v.get("knob") for v in vals] if "knob" in vals[0] else None,
            }
            if "k_chosen" in vals[0]:
                row["methods"][name]["k_chosen_per_seed"] = [v["k_chosen"] for v in vals]
        row["knob_sweep"] = {m: {v: {"ARI_mean": mean_of(sc, "ARI"), "K_mean": mean_of(sc, "K")}
                                 for v, sc in d.items()} for m, d in per_value.items()}
        if sig == 0.0:
            report["smallest_number_of_nodes_any_knob_value_gives_at_signal_zero"] = {
                m: {"per_seed": v, "note": "the truth at signal 0 is zero nodes"}
                for m, v in smallest_at_zero.items()}
        rows.append(row)
        top = row["methods"]
        print(f"  signal {sig:<5} escrow_base ARI {top['escrow_base']['ARI_mean']:.4f} "
              f"K {top['escrow_base']['K_mean']:<5} | escrow_split ARI "
              f"{top['escrow_split']['ARI_mean']:.4f} K {top['escrow_split']['K_mean']:<5} | "
              f"kmeans_silhouette ARI {top['kmeans_silhouette']['ARI_mean']:.4f} "
              f"K {top['kmeans_silhouette']['K_mean']:<5} | hdbscan ARI "
              f"{top['hdbscan_defaults']['ARI_mean']:.4f} K {top['hdbscan_defaults']['K_mean']}")
    report["rows"] = rows

    # ---------------- the two tables ---------------- #
    def table(field, fmt):
        head = f"  {'method':<28}{'told':<24}" + "".join(f"{s:>8.2f}" for s in SIGNAL)
        lines = [head, "  " + "-" * (len(head) - 2)]
        for name in ROW_ORDER:
            cells = "".join(fmt(r["methods"][name][field]) for r in rows)
            lines.append(f"  {name:<28}{TOLD_SHORT.get(name, ''):<24}{cells}")
        return "\n".join(lines)

    ari_table = table("ARI_mean", lambda v: f"{v:>8.3f}")
    k_table = table("K_mean", lambda v: f"{v:>8.1f}")
    report["ARI_table"] = ari_table
    report["K_table"] = k_table
    print("\nARI against the true 8 groups, mean over three seeds, by signal\n")
    print(ari_table)
    print("\nnumber of nodes reported, mean over three seeds, by signal. "
          "The truth is 8 everywhere except signal 0.0, where it is 0\n")
    print(k_table)

    # ---------------- is the silhouette row's 20 its own answer or the top of its range? ------ #
    print("\nopening the k range for the silhouette row at signal 0 ...")
    report["kmeans_silhouette_when_the_range_is_opened"] = silhouette_cap_check()
    cap = report["kmeans_silhouette_when_the_range_is_opened"]
    print("   k chosen with the range opened to 60:", cap["k_chosen_per_seed"])

    # ---------------- the refutation test ---------------- #
    # Not handed k, and no knob picked on THESE labels. The transferred arms are in this set because
    # their value came from a different stream, which is what a practitioner can actually do. They
    # are not label free in the strict sense and the key name says so.
    not_tuned_here = [m for m in ROW_ORDER
                      if m not in ("escrow_base", "escrow_split")
                      and "oracle" not in m and "true_k" not in m]
    with_signal = [r for r in rows if r["signal"] >= 0.5]
    zero = rows[0]["methods"]

    def mean_ari(m):
        return round(sum(r["methods"][m]["ARI_mean"] for r in with_signal) / len(with_signal), 4)

    ours = mean_ari("escrow_split")
    matched = [m for m in not_tuned_here if mean_ari(m) >= ours - 0.02]
    abstained = [m for m in ROW_ORDER if zero[m]["K_mean"] == 0]
    both = [m for m in matched if zero[m]["K_mean"] == 0]

    one_value = one_knob_value_test(rows, with_signal)
    report["one_knob_value_that_has_to_do_both_halves"] = one_value
    can_be_silent = {m: v["values_reporting_zero_nodes_at_signal_zero"]
                     for m, v in one_value.items() if v["values_reporting_zero_nodes_at_signal_zero"]}
    does_both = {m: v["best_mean_ARI_among_the_silent_values"] for m, v in one_value.items()
                 if v["best_mean_ARI_among_the_silent_values"] is not None
                 and v["best_mean_ARI_among_the_silent_values"] >= ours - 0.02}

    report["headline"] = {
        "mean_ARI_over_the_rungs_with_signal": {m: mean_ari(m) for m in ROW_ORDER},
        "nodes_at_signal_zero_where_the_truth_is_zero": {m: zero[m]["K_mean"] for m in ROW_ORDER},
        "escrow_base_mean_ARI": mean_ari("escrow_base"),
        "escrow_split_mean_ARI": ours,
        "methods_not_handed_k_and_not_tuned_on_these_labels": not_tuned_here,
        "of_those_the_ones_matching_escrow_split_where_there_is_signal": matched,
        "what_those_methods_report_at_signal_zero": {m: zero[m]["K_mean"] for m in matched},
        "methods_reporting_zero_nodes_at_signal_zero": abstained,
        "of_those_the_ones_that_were_not_tuned_on_these_labels": both,
        "the_claim_is_refuted": bool(both),
        "knobbed_methods_with_a_value_that_reports_zero_nodes_at_signal_zero": can_be_silent,
        "knobbed_methods_with_ONE_value_that_does_both_halves": does_both,
        "the_weaker_claim_is_refuted": bool(does_both),
        "the_half_that_is_not_ours": (
            "a baseline that is not handed k beating us where there is signal is a real result and "
            "is reported here as one. It only refutes the claim if the same run also reports "
            "nothing at signal 0"),
        "a_note_on_the_oracle_rows_at_signal_zero": (
            "at signal 0 the oracle knob is still picked by agreement, and agreement is 0 exactly "
            "for a method that reports nothing and slightly negative for one that cuts noise into "
            "groups, so the oracle does weakly prefer silence there. Any oracle row in the silent "
            "list is silent because the labels were consulted, which is not available in use"),
        "reading": ("the claim needs a method to do both halves: agree with the groups where there "
                    "are groups, and report nothing where there is nothing. A method that is handed "
                    "k, or a knob picked on the test labels, has been told the answer to the second "
                    "half. If a method that was told neither does both, the paper is wrong and this "
                    "line says so"),
        "seconds": round(time.time() - t_start, 1),
    }
    print("\nthe knob values that report nothing at signal 0, and what they score where there "
          "is signal\n")
    for m, v in one_value.items():
        for e in v["every_value"]:
            mark = "   silent at signal 0" if e["nodes_at_signal_zero"] == 0 else ""
            print(f"  {m:<16}{str(e['knob']):>7}   mean ARI with signal "
                  f"{e['mean_ARI_where_there_is_signal']:>7.3f}   nodes at signal 0 "
                  f"{e['nodes_at_signal_zero']:>8.1f}{mark}")
    print("\n" + json.dumps({k: v for k, v in report["headline"].items()
                             if k != "mean_ARI_over_the_rungs_with_signal"}, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
