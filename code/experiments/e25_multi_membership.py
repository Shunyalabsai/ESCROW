"""E25: is multi-membership load-bearing?

THE OBJECTION, from an ICLR reviewer. The paper's stated reason that an existing codelength
criterion cannot be reused is that a record activates a SET of nodes rather than landing in one
block of a partition, and that is the paper's whole justification for building a new code. No
experiment anywhere in the repository tests that this matters. Every fixture shipped so far
(twogroup, planted8, wikipedia, lazada) is a partition in the truth, so a partition method is
never asked to do the thing it cannot do, and the justification is asserted rather than measured.

WHAT THIS RUNS. A new fixture built the way the other synthetic fixtures in
experiments/e4_baseline_army.py are built, but with a genuine cover in the truth:

    multi_membership8(n=3000, groups=8, keys_per_group=3, values_per_key=8, noise=0.1,
                      set_size_weights=(0.5, 0.3, 0.2))

Each record draws a SET of 1 to 3 latent groups uniformly without replacement, with the set size
drawn from set_size_weights, and emits keys_per_group keys for EVERY group it belongs to; each
emitted value carries its own group tag, which with probability `noise` is replaced by a uniformly
random group tag, exactly as planted8 corrupts values. Group g owns the keys c{g}k{j}, so the
groups a record belongs to are visible in its key set, as in planted8. All generator parameters
are written into the JSON so the fixture is reproducible from the report alone.

A CONTROL fixture is generated from the same generator with set_size_weights=(1, 0, 0), which
makes it a partition and is otherwise identical. It isolates multi-membership as the cause of any
gap: if the gap is present on the cover and absent on the control, the cover is what produced it.

Arms, five seeds (0 to 4) on each fixture:
  ESCROW            escrow.protocol.run_stream, whose node set is a cover: a record can be a
                    member of several nodes at once.
  kmeans_trueG      k-means handed the true number of latent GROUPS (8).
  kmeans_trueC      k-means handed the true number of distinct group COMBINATIONS, which is the
                    number of blocks in the finest partition consistent with the truth. This is a
                    second and larger oracle, given because a partition method cannot represent
                    the cover and the combination partition is the best it could possibly aim at.
  agglomerative     average linkage on cosine distance, cut at an ORACLE distance threshold.
  dpmeans           sequential DP-means at an ORACLE lambda.
Every baseline is run on three lossless multi-hot encodings of the same records (key indicators,
key=value indicators, and their union), and its ORACLE representation is taken too. So each
baseline is scored at the best cell of (representation x parameter) chosen on the test labels,
which is strictly more than ESCROW gets: ESCROW is one run with no calibrated parameter, on the
raw records. Any input ESCROW gets, the baselines get, and here they get more.

SCORING. ARI needs a partition, so it is the wrong metric for a cover and is reported only as a
degraded secondary with its caveat attached. The two primary scores are both cover scores:

  OMEGA INDEX (Collins and Dent 1988). For every unordered pair of records (i, j) count t_ij, the
  number of groups they share in the truth, and p_ij, the number they share in the prediction.
  Let A_obs be the fraction of pairs with t_ij == p_ij, and let A_exp = sum_j f_t(j) f_p(j) where
  f_t(j) is the fraction of pairs sharing exactly j groups in the truth and f_p(j) the same in the
  prediction. Omega = (A_obs - A_exp) / (1 - A_exp). It is 1 for a perfect cover, 0 for chance,
  and it can go negative. A partition method can only ever emit p_ij in {0, 1}, so every pair that
  truly shares 2 or 3 groups is scored wrong by construction, which is exactly the quantity under
  test.

  PER-MEMBERSHIP PRECISION, RECALL AND F1 over the set of (record, group) memberships. Predicted
  groups are matched to true groups by best overlap, in two ways, both reported for every method:
    one-to-one   a maximum-total-overlap one-to-one matching (scipy.optimize.linear_sum_assignment).
                 Memberships of predicted groups left unmatched are counted as false positives.
    many-to-one  every predicted group is relabelled to the true group it overlaps most, several
                 predicted groups may map to the same true group, and the relabelled memberships
                 are deduplicated. This is the more generous reading and it is the one to quote
                 against us. It does not punish over-splitting: many pure small clusters all
                 relabel onto the same true group and their union is scored once, so a baseline
                 can reach F1 1.0 on a partition by cutting far too finely. That is why the omega
                 index and the one-to-one F1 are reported beside it.
  Precision = |TP| / |predicted memberships after relabelling|, recall = |TP| / |true memberships|,
  F1 the harmonic mean. A record that ESCROW leaves in the background simply contributes no
  predicted membership and costs recall.

Also reported: the distribution of how many groups a record belongs to, in the truth and in each
prediction, so a reader can check the fixture is a cover and not a partition in disguise, and can
check whether ESCROW's output is a cover at all.

THE FALSIFIER, stated in advance. If a partition method matches or beats ESCROW on the omega
index and on membership F1 on this genuine cover, then multi-membership is not load-bearing, the
new code is not justified by it, and the paper's justification for building a new code has to
change. The report evaluates this in reading.falsifier_fired, per fixture, using the generous
many-to-one F1 and the omega index, and it is evaluated against the ORACLE-tuned baselines, which
is the reading least favourable to us.

Run: python3 code/experiments/e25_multi_membership.py [--quick]
Writes results/e25_multi_membership.json.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from escrow.protocol import run_stream, describe as protocol_describe
from escrow.provenance import stamped
from experiments.e4_baseline_army import _ari, dp_means as _e4_dp_means

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e25_multi_membership.json")

SEEDS = (0, 1, 2, 3, 4)

GEN = dict(n=3000, groups=8, keys_per_group=3, values_per_key=8, noise=0.1,
           set_size_weights=(0.5, 0.3, 0.2), max_set_size=3)

AGGLO_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DPMEANS_LAMBDAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
REPRESENTATIONS = ("key", "keyvalue", "key_plus_keyvalue")


# --------------------------------------------------------------------------- #
# the fixture
# --------------------------------------------------------------------------- #
def multi_membership8(n=GEN["n"], groups=GEN["groups"], keys_per_group=GEN["keys_per_group"],
                      values_per_key=GEN["values_per_key"], noise=GEN["noise"],
                      set_size_weights=GEN["set_size_weights"], seed=0):
    """A cover version of planted8. Each record draws a SET of latent groups of size 1 to
    len(set_size_weights) and emits the keys and values of every group in that set. Group g owns
    the keys c{g}k{j} and the untainted values c{g}v{m}; with probability `noise` a value's group
    tag is redrawn uniformly, which is planted8's corruption. Returns (records, truth) where truth
    is a list of frozensets of group ids."""
    rng = random.Random(seed)
    sizes = list(range(1, len(set_size_weights) + 1))
    recs, truth = [], []
    for _ in range(n):
        m = rng.choices(sizes, weights=list(set_size_weights))[0]
        gs = sorted(rng.sample(range(groups), m))
        rec = {}
        for g in gs:
            for j in range(keys_per_group):
                og = rng.randrange(groups) if rng.random() < noise else g
                rec[f"c{g}k{j}"] = f"c{og}v{rng.randrange(values_per_key)}"
        recs.append(rec)
        truth.append(frozenset(gs))
    return recs, truth


def features(recs, kind):
    """A lossless multi-hot encoding of the same records the engine sees, L2 normalised so that
    the euclidean geometry the baselines use is the cosine geometry. `kind` selects key
    indicators, key=value indicators, or their union."""
    def toks(r):
        if kind == "key":
            return [k for k in r]
        if kind == "keyvalue":
            return [f"{k}={v}" for k, v in r.items()]
        return [k for k in r] + [f"{k}={v}" for k, v in r.items()]

    vocab = {}
    rows = []
    for r in recs:
        ts = toks(r)
        for t in ts:
            if t not in vocab:
                vocab[t] = len(vocab)
        rows.append(ts)
    X = np.zeros((len(recs), len(vocab)), dtype=np.float64)
    for i, ts in enumerate(rows):
        for t in ts:
            X[i, vocab[t]] = 1.0
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


# --------------------------------------------------------------------------- #
# membership matrices
# --------------------------------------------------------------------------- #
def truth_matrix(truth, groups):
    M = np.zeros((len(truth), groups), dtype=bool)
    for i, gs in enumerate(truth):
        for g in gs:
            M[i, g] = True
    return M


def escrow_cover(recs):
    """ESCROW's node set as a membership matrix. A record may be a member of several nodes; a
    record in no node is an all-False row, which is the honest reading of the background."""
    g, _ = run_stream(recs)
    nids = sorted(g.nodes)
    M = np.zeros((len(recs), len(nids)), dtype=bool)
    for col, nid in enumerate(nids):
        for m in g.nodes[nid].members:
            M[m - 1, col] = True
    return M, g


def partition_matrix(labels, n):
    """A partition baseline's labels as a membership matrix with exactly one True per row."""
    uniq = sorted(set(labels))
    idx = {c: j for j, c in enumerate(uniq)}
    M = np.zeros((n, len(uniq)), dtype=bool)
    for i, c in enumerate(labels):
        M[i, idx[c]] = True
    return M


# --------------------------------------------------------------------------- #
# the two cover scores
# --------------------------------------------------------------------------- #
def _shared_counts(M):
    """Pairwise counts of shared groups. The multiply is done in float32 so it goes through BLAS
    (numpy's integer matmul is a generic single-threaded loop, and a prediction can have as many
    groups as records); every entry is a sum of products of exact 0/1 values and is far below
    2**24, so the float32 result is exact and the cast back to int16 is lossless."""
    F = M.astype(np.float32)
    return (F @ F.T).astype(np.int16)


def omega_index(Mt, Mp, T=None):
    """Omega index between two covers. See the module docstring for the definition. `T`, the
    truth's pairwise shared-group counts, may be passed in because it is the same for every
    prediction scored against the same truth."""
    n = Mt.shape[0]
    T = _shared_counts(Mt) if T is None else T
    P = _shared_counts(Mp)
    jmax = int(max(T.max(initial=0), P.max(initial=0)))
    width = jmax + 1
    joint = np.zeros(width * width, dtype=np.int64)
    for i in range(n - 1):
        t = T[i, i + 1:].astype(np.int64)
        p = P[i, i + 1:].astype(np.int64)
        joint += np.bincount(t * width + p, minlength=width * width)
    joint = joint.reshape(width, width)
    npairs = int(joint.sum())
    if npairs == 0:
        return 0.0, {}, {}
    agree = int(np.trace(joint))
    ft = joint.sum(axis=1) / npairs
    fp = joint.sum(axis=0) / npairs
    a_obs = agree / npairs
    a_exp = float(np.dot(ft, fp))
    om = 1.0 if a_exp >= 1.0 else (a_obs - a_exp) / (1.0 - a_exp)
    hist_t = {str(j): int(v) for j, v in enumerate(joint.sum(axis=1)) if v}
    hist_p = {str(j): int(v) for j, v in enumerate(joint.sum(axis=0)) if v}
    return float(om), hist_t, hist_p


def membership_prf(Mt, Mp):
    """Per-membership precision, recall and F1 under both matchings. See the module docstring."""
    n, kt = Mt.shape
    kp = Mp.shape[1]
    true_total = int(Mt.sum())
    O = (Mp.astype(np.int64).T @ Mt.astype(np.int64))          # kp x kt overlaps

    out = {}
    if kp == 0 or true_total == 0:
        zero = dict(precision=0.0, recall=0.0, f1=0.0, matched_groups=0,
                    predicted_memberships=int(Mp.sum()), true_memberships=true_total,
                    true_positives=0)
        return dict(one_to_one=dict(zero), many_to_one=dict(zero))

    # one to one, maximum total overlap
    r, c = linear_sum_assignment(-O)
    mapped = np.zeros((n, kt), dtype=bool)
    matched_rows = set()
    for a, b in zip(r, c):
        mapped[:, b] |= Mp[:, a]
        matched_rows.add(int(a))
    unmatched_pred = sum(int(Mp[:, a].sum()) for a in range(kp) if a not in matched_rows)
    tp = int((mapped & Mt).sum())
    pred_total = int(mapped.sum()) + unmatched_pred
    prec = tp / pred_total if pred_total else 0.0
    rec = tp / true_total
    out["one_to_one"] = dict(
        precision=round(prec, 4), recall=round(rec, 4),
        f1=round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        matched_groups=len(matched_rows), predicted_memberships=pred_total,
        true_memberships=true_total, true_positives=tp)

    # many to one, every predicted group relabelled to its best true group, then deduplicated
    best = O.argmax(axis=1)
    mapped2 = np.zeros((n, kt), dtype=bool)
    for a in range(kp):
        mapped2[:, best[a]] |= Mp[:, a]
    tp2 = int((mapped2 & Mt).sum())
    pred_total2 = int(mapped2.sum())
    prec2 = tp2 / pred_total2 if pred_total2 else 0.0
    rec2 = tp2 / true_total
    out["many_to_one"] = dict(
        precision=round(prec2, 4), recall=round(rec2, 4),
        f1=round(2 * prec2 * rec2 / (prec2 + rec2), 4) if (prec2 + rec2) else 0.0,
        distinct_true_groups_used=int(len(set(best.tolist()))),
        predicted_memberships=pred_total2, true_memberships=true_total, true_positives=tp2)
    return out


def size_histogram(M):
    c = Counter(M.sum(axis=1).tolist())
    return {str(int(k)): int(v) for k, v in sorted(c.items())}


def ari_secondary(truth, Mp):
    """ARI with the truth read as the partition by distinct group COMBINATION, and the prediction
    read as one label per record. This is the degraded reading, kept only because a reviewer will
    look for it. For a cover prediction a record in several groups is given its lowest-indexed
    group, which is arbitrary, and that arbitrariness is the point: ARI cannot see a cover."""
    lab = []
    for i in range(Mp.shape[0]):
        nz = np.nonzero(Mp[i])[0]
        lab.append(int(nz[0]) if len(nz) else -1)
    return round(_ari([tuple(sorted(t)) for t in truth], lab), 4)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def dp_means_fast(X, lam):
    """Vectorised DP-means, identical in semantics to experiments.e4_baseline_army.dp_means:
    sequential pass, squared euclidean distance to the nearest existing centre, open a new centre
    at the point when that distance exceeds lam. Ties break to the lowest index in both. The
    equivalence is checked at run time and recorded in the report."""
    centers = np.empty((max(16, 8), X.shape[1]), dtype=X.dtype)
    centers[0] = X[0]
    k = 1
    assign = [0]
    for i in range(1, X.shape[0]):
        x = X[i]
        d2 = np.sum((centers[:k] - x) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] > lam:
            if k == centers.shape[0]:
                centers = np.vstack([centers, np.empty_like(centers)])
            centers[k] = x
            assign.append(k)
            k += 1
        else:
            assign.append(j)
    return assign


def check_dp_means_equivalence(X, lams=(0.25, 1.0)):
    rows = []
    Xs = X[:300]
    for lam in lams:
        a = _e4_dp_means(Xs, lam)
        b = dp_means_fast(Xs, lam)
        rows.append(dict(lam=lam, n=len(Xs), identical=bool(a == b)))
    return rows


def baseline_labels(X, truth, groups):
    """All partition arms on one representation. Returns {arm: {param: labels}}."""
    n = X.shape[0]
    n_comb = len({tuple(sorted(t)) for t in truth})
    out = {"kmeans_trueG": {str(groups): list(KMeans(groups, n_init=10, random_state=0)
                                              .fit_predict(X))},
           "kmeans_trueC": {str(n_comb): list(KMeans(n_comb, n_init=10, random_state=0)
                                              .fit_predict(X))}}
    D = pdist(X, metric="cosine")
    Z = linkage(D, method="average")
    out["agglomerative"] = {str(t): list(fcluster(Z, t=t, criterion="distance"))
                            for t in AGGLO_THRESHOLDS}
    out["dpmeans"] = {str(l): dp_means_fast(X, l) for l in DPMEANS_LAMBDAS}
    return out, n_comb


def score_prediction(Mt, Mp, truth, T=None):
    om, hist_t, hist_p = omega_index(Mt, Mp, T=T)
    prf = membership_prf(Mt, Mp)
    return dict(omega=round(om, 4),
                membership=prf,
                groups_predicted=int(Mp.shape[1]),
                memberships_per_record=size_histogram(Mp),
                records_with_no_group=int((Mp.sum(axis=1) == 0).sum()),
                ARI_secondary=ari_secondary(truth, Mp),
                _pair_hist_truth=hist_t, _pair_hist_pred=hist_p)


# --------------------------------------------------------------------------- #
# one fixture, one seed
# --------------------------------------------------------------------------- #
def run_seed(fixture_kwargs, seed, reps):
    recs, truth = multi_membership8(seed=seed, **fixture_kwargs)
    groups = fixture_kwargs.get("groups", GEN["groups"])
    Mt = truth_matrix(truth, groups)
    T_truth = _shared_counts(Mt)
    n = len(recs)

    t0 = time.perf_counter()
    Me, g = escrow_cover(recs)
    escrow_secs = time.perf_counter() - t0
    row = {"n": n,
           "truth": dict(groups=groups,
                         memberships=int(Mt.sum()),
                         memberships_per_record=size_histogram(Mt),
                         distinct_combinations=len({tuple(sorted(t)) for t in truth})),
           "escrow": score_prediction(Mt, Me, truth, T=T_truth)}
    row["escrow"]["K"] = int(g.K)
    row["escrow"]["mints"] = int(len(g.mint_log))
    row["escrow"]["seconds"] = round(escrow_secs, 3)
    row["escrow"]["is_a_cover"] = bool(int(Me.sum()) > n - row["escrow"]["records_with_no_group"])

    baselines = {}
    dp_check = None
    for rep in reps:
        X = features(recs, rep)
        if dp_check is None:
            dp_check = check_dp_means_equivalence(X)
        arms, n_comb = baseline_labels(X, truth, groups)
        for arm, byparam in arms.items():
            for param, labels in byparam.items():
                Mp = partition_matrix(labels, n)
                cell = score_prediction(Mt, Mp, truth, T=T_truth)
                cell.pop("_pair_hist_truth")
                baselines.setdefault(arm, {})[f"{rep}|{param}"] = cell
    row["baselines_all_cells"] = baselines
    row["dp_means_equivalence_check"] = dp_check
    return row


def _best(cells, key):
    """The oracle cell: the (representation, parameter) cell with the largest value of `key`,
    chosen on the test labels, which is stated as an advantage given to the baseline."""
    def val(c):
        return c["omega"] if key == "omega" else c["membership"]["many_to_one"]["f1"]
    k = max(cells, key=lambda c: val(cells[c]))
    return dict(cell=k, value=round(val(cells[k]), 4), detail=cells[k])


def summarise_seed(row):
    """Per seed, reduce each baseline's grid to its two oracle cells."""
    out = {"escrow": {k: v for k, v in row["escrow"].items() if not k.startswith("_pair_hist")},
           "truth": row["truth"], "n": row["n"]}
    out["escrow"]["pair_hist_truth"] = row["escrow"]["_pair_hist_truth"]
    out["escrow"]["pair_hist_pred"] = row["escrow"]["_pair_hist_pred"]
    out["dp_means_equivalence_check"] = row["dp_means_equivalence_check"]
    for arm, cells in row["baselines_all_cells"].items():
        out[arm] = dict(oracle_omega=_best(cells, "omega"),
                        oracle_f1_many_to_one=_best(cells, "f1"),
                        n_cells=len(cells))
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 4)


def _spread(xs):
    return dict(mean=_mean(xs), min=round(min(xs), 4), max=round(max(xs), 4))


# --------------------------------------------------------------------------- #
def run_fixture(name, fixture_kwargs, seeds, reps):
    per_seed = []
    for s in seeds:
        t0 = time.time()
        row = run_seed(fixture_kwargs, s, reps)
        summ = summarise_seed(row)
        summ["seed"] = s
        summ["seconds"] = round(time.time() - t0, 2)
        per_seed.append(summ)
        e = summ["escrow"]
        print(f"    [{name}] seed={s} K={e['K']} omega={e['omega']} "
              f"F1(1-1)={e['membership']['one_to_one']['f1']} "
              f"F1(m-1)={e['membership']['many_to_one']['f1']} "
              f"bg={e['records_with_no_group']} {summ['seconds']}s", flush=True)
        for arm in ("kmeans_trueG", "kmeans_trueC", "agglomerative", "dpmeans"):
            b = summ[arm]
            print(f"        {arm:<14s} best omega={b['oracle_omega']['value']} "
                  f"at {b['oracle_omega']['cell']}   best F1(m-1)="
                  f"{b['oracle_f1_many_to_one']['value']} at "
                  f"{b['oracle_f1_many_to_one']['cell']}", flush=True)

    arms = ["escrow", "kmeans_trueG", "kmeans_trueC", "agglomerative", "dpmeans"]
    agg = {}
    for arm in arms:
        if arm == "escrow":
            om = [r["escrow"]["omega"] for r in per_seed]
            f1m = [r["escrow"]["membership"]["many_to_one"]["f1"] for r in per_seed]
            f11 = [r["escrow"]["membership"]["one_to_one"]["f1"] for r in per_seed]
            pr = [r["escrow"]["membership"]["many_to_one"]["precision"] for r in per_seed]
            rc = [r["escrow"]["membership"]["many_to_one"]["recall"] for r in per_seed]
            ari = [r["escrow"]["ARI_secondary"] for r in per_seed]
            ng = [float(r["escrow"]["groups_predicted"]) for r in per_seed]
            ngf = list(ng)
        else:
            om = [r[arm]["oracle_omega"]["value"] for r in per_seed]
            f1m = [r[arm]["oracle_f1_many_to_one"]["value"] for r in per_seed]
            f11 = [r[arm]["oracle_f1_many_to_one"]["detail"]["membership"]["one_to_one"]["f1"]
                   for r in per_seed]
            pr = [r[arm]["oracle_f1_many_to_one"]["detail"]["membership"]["many_to_one"]
                  ["precision"] for r in per_seed]
            rc = [r[arm]["oracle_f1_many_to_one"]["detail"]["membership"]["many_to_one"]["recall"]
                  for r in per_seed]
            ari = [r[arm]["oracle_omega"]["detail"]["ARI_secondary"] for r in per_seed]
            ng = [float(r[arm]["oracle_omega"]["detail"]["groups_predicted"]) for r in per_seed]
            ngf = [float(r[arm]["oracle_f1_many_to_one"]["detail"]["groups_predicted"])
                   for r in per_seed]
        agg[arm] = dict(omega=_spread(om), f1_many_to_one=_spread(f1m),
                        f1_one_to_one=_spread(f11),
                        groups_predicted_at_f1_cell=_spread(ngf),
                        precision_many_to_one=_spread(pr), recall_many_to_one=_spread(rc),
                        ARI_secondary=_spread(ari), groups_predicted=_spread(ng),
                        tuned_on_test_labels=(arm != "escrow"),
                        what_was_tuned=("nothing; one run, no calibrated parameter"
                                        if arm == "escrow" else
                                        "representation and parameter, both chosen on the test "
                                        "labels, separately for each scored quantity"))
    return dict(generator=dict(fixture_kwargs), seeds=list(seeds), representations=list(reps),
                per_seed=per_seed, aggregate=agg)


def build_reading(report):
    lines, per_fixture = [], {}
    for name, fx in report["fixtures"].items():
        agg = fx["aggregate"]
        e_om = agg["escrow"]["omega"]["mean"]
        e_f1 = agg["escrow"]["f1_many_to_one"]["mean"]
        e_f11 = agg["escrow"]["f1_one_to_one"]["mean"]
        beats_om = [a for a in agg if a != "escrow" and agg[a]["omega"]["mean"] >= e_om]
        beats_f1 = [a for a in agg if a != "escrow" and agg[a]["f1_many_to_one"]["mean"] >= e_f1]
        beats_f11 = [a for a in agg if a != "escrow"
                     and agg[a]["f1_one_to_one"]["mean"] >= e_f11]
        both = sorted(set(beats_om) & set(beats_f1))
        best_base_om = max((a for a in agg if a != "escrow"),
                           key=lambda a: agg[a]["omega"]["mean"])
        best_base_f1 = max((a for a in agg if a != "escrow"),
                           key=lambda a: agg[a]["f1_many_to_one"]["mean"])
        best_base_f11 = max((a for a in agg if a != "escrow"),
                            key=lambda a: agg[a]["f1_one_to_one"]["mean"])
        per_fixture[name] = dict(
            escrow_omega=e_om, escrow_f1_many_to_one=e_f1, escrow_f1_one_to_one=e_f11,
            best_baseline_omega=dict(arm=best_base_om,
                                     value=agg[best_base_om]["omega"]["mean"],
                                     groups_predicted=agg[best_base_om]["groups_predicted"]["mean"]),
            best_baseline_f1_many_to_one=dict(
                arm=best_base_f1, value=agg[best_base_f1]["f1_many_to_one"]["mean"],
                groups_predicted_at_that_cell=agg[best_base_f1]["groups_predicted_at_f1_cell"]["mean"]),
            best_baseline_f1_one_to_one=dict(arm=best_base_f11,
                                             value=agg[best_base_f11]["f1_one_to_one"]["mean"]),
            omega_gap=round(e_om - agg[best_base_om]["omega"]["mean"], 4),
            f1_gap=round(e_f1 - agg[best_base_f1]["f1_many_to_one"]["mean"], 4),
            f1_one_to_one_gap=round(e_f11 - agg[best_base_f11]["f1_one_to_one"]["mean"], 4),
            baselines_matching_or_beating_escrow_on_omega=sorted(beats_om),
            baselines_matching_or_beating_escrow_on_f1=sorted(beats_f1),
            baselines_matching_or_beating_escrow_on_f1_one_to_one=sorted(beats_f11),
            arms_with_generous_f1_but_lower_omega=sorted(
                [dict(arm=a, f1_many_to_one=agg[a]["f1_many_to_one"]["mean"],
                      f1_one_to_one=agg[a]["f1_one_to_one"]["mean"],
                      omega=agg[a]["omega"]["mean"],
                      groups_predicted_at_that_cell=agg[a]["groups_predicted_at_f1_cell"]["mean"])
                 for a in agg
                 if a != "escrow" and agg[a]["f1_many_to_one"]["mean"] >= e_f1
                 and agg[a]["omega"]["mean"] < e_om],
                key=lambda d: d["arm"]),
            escrow_ARI_secondary=agg["escrow"]["ARI_secondary"]["mean"],
            baselines_beating_escrow_on_ARI_secondary=sorted(
                [dict(arm=a, ARI_secondary=agg[a]["ARI_secondary"]["mean"],
                      omega=agg[a]["omega"]["mean"],
                      f1_one_to_one=agg[a]["f1_one_to_one"]["mean"])
                 for a in agg if a != "escrow"
                 and agg[a]["ARI_secondary"]["mean"] > agg["escrow"]["ARI_secondary"]["mean"]],
                key=lambda d: -d["ARI_secondary"]),
            falsifier_applies=(name == "cover"),
            falsifier_applies_note=(
                "the falsifier is defined on the cover fixture; on the partition control a "
                "partition method matching ESCROW is the expected and desired outcome and is not "
                "a falsification of anything"),
            falsifier_fired=bool(both), falsifier_fired_by=both)
        lines.append(
            f"{name}: ESCROW omega {e_om}, membership F1 {e_f1}; best oracle-tuned partition "
            f"baseline omega {agg[best_base_om]['omega']['mean']} ({best_base_om}), "
            f"F1 {agg[best_base_f1]['f1_many_to_one']['mean']} ({best_base_f1}).")

    cover = per_fixture.get("cover")
    ctrl = per_fixture.get("control_partition")
    verdict = []
    if cover and cover["falsifier_fired"]:
        verdict.append(
            "THE FALSIFIER FIRED on the cover fixture: " + ", ".join(cover["falsifier_fired_by"])
            + " matched or beat ESCROW on both cover scores while being oracle-tuned. On this "
            "evidence multi-membership is not load-bearing and the paper's justification for a "
            "new code has to change.")
    elif cover:
        verdict.append(
            "The falsifier did not fire on the cover fixture: no partition baseline matched "
            "ESCROW on both cover scores, even with its representation and its parameter chosen "
            "on the test labels. The gap is "
            f"{cover['omega_gap']} omega and {cover['f1_gap']} membership F1 against the best "
            "baseline on each score.")
    if cover and ctrl:
        verdict.append(
            "Control: on the single-membership fixture from the same generator the gaps are "
            f"{ctrl['omega_gap']} omega and {ctrl['f1_gap']} F1, against "
            f"{cover['omega_gap']} and {cover['f1_gap']} on the cover. "
            + ("The gap is smaller on the partition control, which is what attributes it to "
               "multi-membership rather than to the generator, the representation or the "
               "baselines themselves."
               if (ctrl["omega_gap"] < cover["omega_gap"]
                   and ctrl["f1_gap"] < cover["f1_gap"])
               else "The gap is NOT smaller on the partition control, so it cannot be attributed "
                    "to multi-membership alone and this experiment does not support the claim."))
        if ctrl["omega_gap"] <= 0.0 and ctrl["f1_gap"] <= 0.0:
            verdict.append(
                "Stated plainly against the method: on the partition control ESCROW does not beat "
                "the partition baselines at all, they tie it exactly, so nothing here says ESCROW "
                "is the better clusterer. The whole difference appears when, and only when, the "
                "truth is a cover.")
    if cover and cover["baselines_beating_escrow_on_ARI_secondary"]:
        worst = cover["baselines_beating_escrow_on_ARI_secondary"][0]
        verdict.append(
            "ESCROW LOSES ON ARI on the cover fixture and it is reported here rather than buried: "
            f"{worst['arm']} reaches ARI {worst['ARI_secondary']} against ESCROW's "
            f"{cover['escrow_ARI_secondary']}. That arm is k-means handed the true number of "
            "distinct group COMBINATIONS, and on this fixture the records of one combination are "
            "identical in their key set, so it recovers the combination partition exactly. This "
            "is the whole reason ARI is the wrong instrument for a cover: recovering the 92 "
            "combination blocks is not recovering the 8 groups, and the same arm scores omega "
            f"{worst['omega']} and one-to-one membership F1 {worst['f1_one_to_one']} against "
            f"ESCROW's {cover['escrow_omega']} and {cover['escrow_f1_one_to_one']}. A reader who "
            "trusts ARI alone would read this fixture backwards.")
    exploited = {n: v["arms_with_generous_f1_but_lower_omega"] for n, v in per_fixture.items()
                 if v["arms_with_generous_f1_but_lower_omega"]}
    caveat = ("Caveat on the generous score, stated against ourselves: the many-to-one membership "
              "F1 does not punish over-splitting, because many pure small clusters all relabel "
              "onto the same true group and their union is scored once, so a baseline can reach a "
              "high F1 by cutting far too finely. That is why the omega index and the one-to-one "
              "F1 are reported beside it.")
    if exploited:
        caveat += (" Arms that reached ESCROW's many-to-one F1 while scoring below it on omega, "
                   "which is that effect: "
                   + "; ".join(f"{n}: " + ", ".join(
                       f"{d['arm']} at {d['groups_predicted_at_that_cell']} predicted groups"
                       for d in v) for n, v in exploited.items()) + ".")
    else:
        caveat += " No arm reached ESCROW's many-to-one F1 while scoring below it on omega here."
    verdict.append(caveat)
    return dict(
        question=("does a record belonging to several groups at once actually defeat partition "
                  "methods, which is the paper's stated reason for building a new code?"),
        per_fixture=per_fixture,
        falsifier=("a partition method that matches or beats ESCROW on BOTH the omega index and "
                   "the many-to-one membership F1 on the cover fixture falsifies the claim"),
        falsifier_fired_on_cover=bool(cover and cover["falsifier_fired"]),
        summary=" ".join(lines + verdict))


def main():
    quick = "--quick" in sys.argv
    seeds = (0, 1) if quick else SEEDS
    reps = ("keyvalue",) if quick else REPRESENTATIONS
    kw = dict(GEN)
    kw.pop("max_set_size")
    if quick:
        kw["n"] = 600
    cover_kw = dict(kw)
    control_kw = dict(kw)
    control_kw["set_size_weights"] = (1.0, 0.0, 0.0)

    t0 = time.time()
    report = dict(
        experiment="e25_multi_membership",
        title="is multi-membership load-bearing?",
        purpose=("test the paper's stated justification for a new code, that a record activates a "
                 "set of nodes rather than landing in one block of a partition, by building a "
                 "fixture whose truth is a genuine cover and scoring covers directly"),
        protocol=protocol_describe(),
        quick=quick,
        generator=dict(
            function="multi_membership8 (this file)",
            built_like="experiments.e4_baseline_army.planted8, with a SET of latent groups per record",
            cover_parameters=cover_kw,
            control_parameters=control_kw,
            value_noise_meaning=("with probability `noise` an emitted value's group tag is redrawn "
                                 "uniformly over all groups, as in planted8"),
            key_ownership="group g owns the keys c{g}k{j}",
            seeds=list(seeds)),
        baselines=dict(
            kmeans_trueG="k-means handed the true number of latent groups",
            kmeans_trueC=("k-means handed the true number of distinct group combinations, the "
                          "finest partition consistent with the truth"),
            agglomerative=("average linkage on cosine distance, scipy.cluster.hierarchy.linkage "
                           "plus fcluster, which is the same rule as the sklearn "
                           "AgglomerativeClustering(linkage='average', metric='cosine') arm used "
                           "in E4; cut at an oracle distance threshold"),
            dpmeans="sequential DP-means at an oracle lambda",
            bpmeans=("NOT RUN. The task allows BP-means only if code/experiments/e22_bpmeans.py "
                     "exists; it does not exist in this repository at run time, and it was left "
                     "out rather than reimplemented."),
            representations=list(reps),
            oracle_note=("every baseline is scored at the best (representation, parameter) cell "
                         "chosen on the test labels, separately for the omega index and for the "
                         "membership F1; ESCROW is one run with no calibrated parameter")),
        grids=dict(agglomerative_thresholds=list(AGGLO_THRESHOLDS),
                   dpmeans_lambdas=list(DPMEANS_LAMBDAS)),
        metrics=dict(
            omega=("Collins and Dent omega index: over record pairs, agreement between the number "
                   "of groups shared in the truth and in the prediction, corrected for chance"),
            membership_prf=("precision, recall and F1 over (record, group) memberships after "
                            "matching predicted groups to true groups by best overlap, reported "
                            "both one-to-one (Hungarian, unmatched predicted groups are false "
                            "positives) and many-to-one (generous, deduplicated)"),
            ARI_secondary=("adjusted Rand index with the truth read as the partition by distinct "
                           "group combination; reported only as a degraded secondary because ARI "
                           "cannot represent a cover")),
        fixtures={})

    print("  [cover] starting", flush=True)
    report["fixtures"]["cover"] = run_fixture("cover", cover_kw, seeds, reps)
    print("  [control_partition] starting", flush=True)
    report["fixtures"]["control_partition"] = run_fixture("control_partition", control_kw,
                                                          seeds, reps)
    report["reading"] = build_reading(report)
    report["seconds_total"] = round(time.time() - t0, 2)
    stamped(report)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\n" + report["reading"]["summary"])
    print(f"\nwrote {OUT} in {report['seconds_total']}s")


if __name__ == "__main__":
    main()
