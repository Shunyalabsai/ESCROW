"""E22: BP-means, the method this paper's whole argument is built against.

THE OBJECTION, from an ICLR reviewer. The paper's framing is that BP-means obtains its
new-feature penalty by taking a small-variance limit which deletes the computable part of the
price and keeps the hand-set part (PAPER.md section 3, paper/sections/appendix_theory.tex
Lemma "the limit deletes the computable price"). BP-means is run nowhere in the paper. It is
also the only multi-membership competitor in the literature, so its absence leaves two central
claims untested at once: that the hand-set constant does not transfer, and that a multi-
membership competitor does not reach what the computed price reaches.

WHAT IS IMPLEMENTED, and where it came from. BP-means from Broderick, Kulis and Jordan,
"MAD-Bayes: MAP-based Asymptotic Derivations from Bayes" (ICML 2013). The PDF was read
(arXiv:1212.2126v2). The objective implemented is the paper's Eq. (6),

    argmin over K+, Z, A of  tr[(X - ZA)'(X - ZA)] + K+ lambda^2,

and the algorithm implemented is the paper's own box, quoted from page 4:

    BP-means algorithm. Iterate the following two steps until no changes are made:
    1. For n = 1, ..., N
       - For k = 1, ..., K+, choose the optimizing value (0 or 1) of z_nk.
       - Let Z' equal Z but with one new feature (labeled K+ + 1) containing only data index
         n. Set A' = A but with one new row: A'_{K+ +1,.} <- X_n,. - Z_n,. A.
       - If the triplet (K+ + 1, Z', A') lowers the objective from the triplet (K+, Z, A),
         replace the latter triplet with the former.
    2. Set A <- (Z'Z)^{-1} Z'X.

This is the uncollapsed, unbounded-cardinality BP-means, not the collapsed variant (their
Section 4.1) and not the parametric K-features variant (their Section 4.2). Empty features are
dropped and features with identical support are merged before the least-squares update, which
the paper requires ("Z'Z is invertible so long as two features do not have the same collection
of indices; in this case we simply combine the two features"). The least-squares update is
numpy.linalg.lstsq, which is the pseudo-inverse solution the paper's (Z'Z)^{-1}Z'X asks for.

WHAT IS RUN. On the multi-hot (key, value) representation of the three E4 streams, which is
the same input ESCROW gets: one binary column per distinct (attribute, value) pair seen in that
stream, one row per record. twogroup n=2000 d=80, planted8 n=3000 d=779, wikipedia n=320
d=2607. Seeds 0 to 4, exactly the E4 seed family (the two synthetic generators are resampled,
the Wikipedia record set is fixed and the seed permutes arrival order). Arms:

  ORACLE lambda^2: swept over a fixed grid, best score taken ON THE TEST LABELS. This is an
    upper bound and is labelled as one everywhere it appears. ESCROW never gets this.
  TRANSFERRED lambda^2: the oracle value of another stream applied unchanged.
  THE PAPER'S OWN VALUES: lambda^2 = 1, used for every algorithm in their tabletop experiment
    (Section 5.1, "all with lambda^2 = 1"), and lambda^2 = 5, used in their faces experiment
    (Section 5.2). The paper prescribes no universal default and says so: "choosing the
    relative penalty effect lambda^2. One option is to solve for lambda^2 from a proposed K
    value via a heuristic (Kulis and Jordan, 2012) or validation on a data subset. Rather than
    assume K and return to it in this roundabout way, in the following we aim merely to
    demonstrate that there exist reasonable values of lambda^2 that return meaningful results."
    Both values are reported as what they are: constants chosen on image data at another scale.
  THE K-SEEDED HEURISTIC: the first option in that quotation, solved for the TRUE K, using the
    paper's own feature initialization (base feature = mean of all the data, then the residual
    of the furthest point, repeatedly). This arm is given the true number of groups, which is
    domain knowledge ESCROW is never given. It is included because a baseline may have more
    than ESCROW, never less.

  Restarts. Each lambda^2 is run from two initializations, the empty allocation and the paper's
  base-feature initialization, and the run with the LOWER OBJECTIVE is kept. That is an
  internal criterion, no labels, and it is the local-optimum defence the paper itself proposes.

  Controls. (a) a cap sensitivity control: the run is capped at 250 features for tractability,
  and the smallest grid price is rerun uncapped on seed 0 to show which direction the cap moves
  the answer; (b) a representation control on seed 0: the same sweep on L2-normalized rows, so
  the collapse cannot be blamed on unnormalized row norms.

SCORING. BP-means returns a cover, not a partition, so it is scored three ways and all three
are reported: by_dot, each record to its active feature of largest x_n . a_k; by_norm, each
record to its active feature of largest ||a_k||^2; by_pattern, records with the same binary
feature row in one cluster. A record with no active feature gets -1 and all such records are
one cluster, which is the same honest convention E4 uses for ESCROW's background. The score
credited to BP-means in every comparison is the BEST of the three, which is generous to it.
Feature count K is reported against the true group count. _ari from e4_baseline_army is used
for every number so it is comparable with the rest of the repository.

THE FALSIFIER. If BP-means at a TRANSFERRED lambda^2 matches ESCROW on these streams, then the
computed price buys nothing a chosen constant does not, and the paper's central claim is in
serious trouble. "Matches" is evaluated in the JSON as best transferred BP-means ARI >= ESCROW
ARI on a stream, mean over the seed family, and separately for the oracle upper bound and for
the paper's own constants. Every such comparison is reported whichever way it comes out.

Run: python3 code/experiments/e22_bpmeans.py [--quick]
Writes results/e22_bpmeans.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.protocol import run_stream, describe as protocol_describe
from escrow.provenance import stamped
from experiments.e4_baseline_army import (two_group, planted8, wikipedia, _ari, stream_for_seed)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e22_bpmeans.json")

STREAMS = ("twogroup", "planted8", "wikipedia")
TRUE_K = {"twogroup": 2, "planted8": 8, "wikipedia": 3}
SEEDS = (0, 1, 2, 3, 4)

# One grid, shared by all three streams, so a transferred value is always a value the target
# stream was also swept at and the transfer costs no extra run. It spans two decades and is
# refined either side of the row-norm scale of the synthetic streams (||x||^2 = 4 and 3), where
# the answer turns over, and it contains the paper's own two values 1 and 5 exactly.
GRID = (0.25, 0.5, 1.0, 2.0, 2.5, 2.9, 2.99, 3.0, 3.5, 3.9, 3.99, 4.0, 4.5, 5.0,
        6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 22.0, 25.0, 30.0)
# The same sweep for the L2-normalized representation control, where every row norm is 1.
GRID_UNIT = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2)

PAPER_VALUES = {"paper_tabletop_lambda2_1": 1.0, "paper_faces_lambda2_5": 5.0}

K_CAP = 250          # tractability cap on the feature count; its effect is measured, not assumed
MAX_ITER = 15
INITS = ("empty", "base")


# --------------------------------------------------------------------------- #
# the representation: multi-hot over distinct (attribute, value) pairs
# --------------------------------------------------------------------------- #
def multihot(records):
    vocab, rows = {}, []
    for r in records:
        idx = []
        for k, v in r.items():
            key = (str(k), str(v))
            j = vocab.get(key)
            if j is None:
                j = len(vocab)
                vocab[key] = j
            idx.append(j)
        rows.append(idx)
    X = np.zeros((len(records), len(vocab)), dtype=np.float64)
    for i, idx in enumerate(rows):
        X[i, idx] = 1.0
    return X


# --------------------------------------------------------------------------- #
# BP-means (Broderick, Kulis and Jordan 2013, Eq. 6 and the algorithm box, page 4)
# --------------------------------------------------------------------------- #
def bp_means(X, lam2, k_cap=K_CAP, max_iter=MAX_ITER, init="empty"):
    """One run of BP-means at penalty lam2 = lambda^2. Returns (Z, A, info).

    The record loop is the paper's step 1 verbatim: a 0/1 coordinate pass over existing
    features, then the new-feature trial, accepted exactly when it lowers the objective, which
    for a fresh feature carrying only record n is when the record's squared residual exceeds
    lambda^2. Step 2 is the least-squares update, after dropping empty features and merging
    features of identical support."""
    n, d = X.shape
    xsq = (X * X).sum(1)
    if init == "base":                       # the paper's own feature initialization, base first
        A = X.mean(0)[None, :].copy()
        Z = np.ones((n, 1), dtype=bool)
    else:
        A = np.zeros((0, d))
        Z = np.zeros((n, 0), dtype=bool)

    capped = False
    converged = False
    obj = None
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        k = A.shape[0]
        cols = list((X @ A.T).T) if k else []            # cols[j][i] = x_i . a_j
        AA = list(A @ A.T) if k else []                  # AA[j][l] = a_j . a_l
        Arows = list(A)
        act_sets = [set(np.nonzero(Z[i])[0]) for i in range(n)]
        changed = False

        for i in range(n):
            act = act_sets[i]
            kk = len(Arows)
            xa = np.array([cols[j][i] for j in range(kk)]) if kk else np.zeros(0)
            c = xa.copy()                                # c[j] = residual . a_j
            for j in act:
                c -= AA[j]
            rsq = xsq[i] - 2.0 * sum(xa[j] for j in act) + sum(AA[a][b] for a in act for b in act)
            for j in range(kk):
                djj = AA[j][j]
                if j not in act:
                    delta = -2.0 * c[j] + djj            # cost of switching z_ij from 0 to 1
                    if delta < -1e-12:
                        act.add(j); rsq += delta; c = c - AA[j]; changed = True
                else:
                    delta = 2.0 * c[j] + djj             # cost of switching z_ij from 1 to 0
                    if delta < -1e-12:
                        act.discard(j); rsq += delta; c = c + AA[j]; changed = True
            if rsq > lam2 + 1e-12:                       # the new-feature trial
                if len(Arows) >= k_cap:
                    capped = True
                else:
                    r = X[i].copy()
                    for j in act:
                        r -= Arows[j]
                    newcol = X @ r
                    newrow = np.array([float(Arows[j] @ r) for j in range(len(Arows))]
                                      + [float(r @ r)])
                    for j in range(len(Arows)):
                        AA[j] = np.append(AA[j], newrow[j])
                    AA.append(newrow)
                    Arows.append(r)
                    cols.append(newcol)
                    act.add(len(Arows) - 1)
                    rsq = 0.0
                    changed = True

        kk = len(Arows)
        Z = np.zeros((n, kk), dtype=bool)
        for i in range(n):
            for j in act_sets[i]:
                Z[i, j] = True
        if kk:
            Z = Z[:, Z.any(0)]                           # drop empty features
            seen, keep = set(), []
            for j in range(Z.shape[1]):                  # merge identical supports
                key = Z[:, j].tobytes()
                if key not in seen:
                    seen.add(key); keep.append(j)
            Z = Z[:, keep]
        if Z.shape[1]:
            A, *_ = np.linalg.lstsq(Z.astype(np.float64), X, rcond=None)
        else:
            A = np.zeros((0, d))
        E = X - (Z.astype(np.float64) @ A if Z.shape[1] else 0.0)
        new_obj = float((E * E).sum() + lam2 * Z.shape[1])
        if (not changed) and obj is not None and abs(new_obj - obj) < 1e-9:
            obj = new_obj
            converged = True
            break
        obj = new_obj
    return Z, A, dict(K=int(Z.shape[1]), objective=round(obj, 4), iters=iters,
                      converged=bool(converged), cap_bound=bool(capped),
                      mean_features_per_record=round(float(Z.sum(1).mean()), 4),
                      records_with_no_feature=int((Z.sum(1) == 0).sum()))


def label_sets(X, Z, A):
    """The three partitions read off a cover. -1 is the no-feature background, one cluster."""
    n = X.shape[0]
    k = Z.shape[1]
    by_dot, by_norm = [], []
    if k:
        XA = X @ A.T
        anorm = (A * A).sum(1)
        for i in range(n):
            act = np.nonzero(Z[i])[0]
            if len(act) == 0:
                by_dot.append(-1); by_norm.append(-1)
            else:
                by_dot.append(int(act[int(np.argmax(XA[i, act]))]))
                by_norm.append(int(act[int(np.argmax(anorm[act]))]))
    else:
        by_dot = [-1] * n
        by_norm = [-1] * n
    by_pattern = [Z[i].tobytes() for i in range(n)]
    return dict(by_dot=by_dot, by_norm=by_norm, by_pattern=by_pattern)


def cell(X, truth, lam2, k_cap=K_CAP, max_iter=MAX_ITER):
    """One price on one stream: both initializations, the lower objective kept."""
    best = None
    for init in INITS:
        t0 = time.perf_counter()
        Z, A, info = bp_means(X, lam2, k_cap=k_cap, max_iter=max_iter, init=init)
        info["seconds"] = round(time.perf_counter() - t0, 3)
        info["init"] = init
        labs = label_sets(X, Z, A)
        info["ARI"] = {k: round(_ari(truth, v), 6) for k, v in labs.items()}
        info["ARI_best_of_three"] = round(max(info["ARI"].values()), 6)
        if best is None or info["objective"] < best["objective"]:
            best = info
    best["lambda2"] = lam2
    return best


# --------------------------------------------------------------------------- #
# the K-seeded heuristic: the paper's first option, solved for the true K
# --------------------------------------------------------------------------- #
def heuristic_lambda2(X, target_k):
    """The paper's own feature initialization (page 5, right column) run to target_k features,
    with the furthest point taken deterministically rather than sampled proportional to squared
    distance, then lambda^2 set to the largest squared residual that remains, which is the
    largest penalty at which that construction would have stopped at target_k. This arm is
    handed the true K, which ESCROW is not."""
    res = X - X.mean(0)[None, :]                     # base feature = mean of all the data
    built = 1
    while built < target_k:
        rn = (res * res).sum(1)
        i = int(np.argmax(rn))
        a = res[i].copy()
        gain = 2.0 * (res @ a) - float(a @ a)        # drop in squared residual from taking a
        res[gain > 0] -= a
        built += 1
    rn = (res * res).sum(1)
    return float(rn.max()), built


# --------------------------------------------------------------------------- #
# ESCROW on the same streams, same seeds, no knob
# --------------------------------------------------------------------------- #
def escrow_cell(records, truth):
    t0 = time.perf_counter()
    g, _ = run_stream(records)
    labels = [-1] * len(records)
    for v in g.nodes.values():
        for m in v.members:
            labels[m - 1] = v.nid
    return dict(K=int(g.K), ARI=round(_ari(truth, labels), 6),
                background=int(sum(1 for x in labels if x == -1)),
                seconds=round(time.perf_counter() - t0, 3))


def k_curve(report, seeds):
    """The mechanism behind the result, derived from the swept cells already in the report and
    from nothing else: BP-means' feature count as a function of the price, and the price at which
    it falls off. On binary multi-hot records with disjoint value vocabularies, a fresh feature is
    minted whenever a record's squared residual exceeds lambda^2, and a record's squared norm is
    just its attribute count, so the run has two regimes and almost nothing in between: below the
    row norm every record mints, above it nothing is ever minted after the first."""
    out = {}
    for name in report["streams"]:
        row = report["streams"][name]
        per_price = []
        for i, lam2 in enumerate(report["grid"]):
            ks = [row["sweep"][str(s)][i]["K"] for s in seeds]
            aris = [row["sweep"][str(s)][i]["ARI_best_of_three"] for s in seeds]
            per_price.append(dict(lambda2=lam2, K_mean=_mean([float(k) for k in ks]),
                                  K_min=min(ks), K_max=max(ks),
                                  ARI_mean=_mean(aris)))
        big = [c["lambda2"] for c in per_price if c["K_mean"] >= K_CAP]
        small = [c["lambda2"] for c in per_price if c["K_mean"] <= 2]
        mid = [c for c in per_price if 2 < c["K_mean"] < K_CAP]
        out[name] = dict(
            mean_row_sq_norm=row["mean_row_sq_norm"], true_K=row["true_K"],
            curve=per_price,
            largest_price_at_the_feature_cap=max(big) if big else None,
            smallest_price_collapsing_to_two_or_fewer=min(small) if small else None,
            prices_with_a_feature_count_between_3_and_the_cap=[c["lambda2"] for c in mid],
            best_ARI_among_those=(max((c["ARI_mean"] for c in mid), default=None)))
    return out


def validity_check(report, seeds):
    """The ESCROW arm here must reproduce results/e4_first_tier.json seed for seed, because it is
    the same protocol on the same three streams. If it does not, this table is measuring something
    else and the comparison is void."""
    path = os.path.join(RESULTS, "e4_first_tier.json")
    if not os.path.exists(path):
        return dict(ran=False, why=f"{path} not present")
    with open(path) as fh:
        e4 = json.load(fh)
    rows, ok = [], True
    for name in STREAMS:
        if name not in e4:
            continue
        per = {r["seed"]: r for r in e4[name]["ours_seeds"]["per_seed"]}
        for s in seeds:
            a = per.get(s)
            if a is None:
                continue
            mine = report["streams"][name]["escrow"][str(s)]
            m = (round(mine["ARI"], 4) == a["ARI"]) and (mine["K"] == a["K"])
            ok = ok and m
            rows.append(dict(stream=name, seed=s, e4_ARI=a["ARI"], e22_ARI=round(mine["ARI"], 4),
                             e4_K=a["K"], e22_K=mine["K"], match=m))
    return dict(ran=True, all_match=ok, compared=len(rows), rows=rows,
                what="the ESCROW arm of this experiment against the ours_seeds rows of "
                     "results/e4_first_tier.json, ARI to 4 decimals and K exactly")


def _mean(vals):
    return round(sum(vals) / len(vals), 6)


def _spread(vals):
    return dict(mean=_mean(vals), min=round(min(vals), 6), max=round(max(vals), 6))


# --------------------------------------------------------------------------- #
def main():
    quick = "--quick" in sys.argv
    seeds = (0, 1) if quick else SEEDS
    grid = (1.0, 2.9, 5.0, 20.0) if quick else GRID
    grid_unit = (0.2, 0.9) if quick else GRID_UNIT

    t_start = time.time()
    report = dict(
        experiment="e22_bpmeans",
        title="BP-means, the method this paper's argument is built against",
        purpose=("run the only multi-membership competitor in the literature on the same input "
                 "ESCROW gets, at an oracle price, at a transferred price, and at the prices its "
                 "own paper used, and report every comparison whichever way it comes out"),
        source=dict(
            paper="Broderick, Kulis and Jordan, MAD-Bayes: MAP-based Asymptotic Derivations from "
                  "Bayes, ICML 2013",
            read="arXiv:1212.2126v2 PDF, read in full for this experiment",
            objective="Eq. (6): argmin over K+, Z, A of tr[(X - ZA)'(X - ZA)] + K+ lambda^2",
            algorithm="the BP-means algorithm box, page 4, uncollapsed and unbounded cardinality",
            not_implemented=["the collapsed BP-means of Section 4.1",
                             "the parametric K-features and repeated-feature variants of "
                             "Section 4.2"],
            lambda_guidance=("the paper prescribes no universal value. It uses lambda^2 = 1 for "
                             "every algorithm in the tabletop experiment (Section 5.1) and "
                             "lambda^2 = 5 in the faces experiment (Section 5.2), both on the top "
                             "100 principal components of image data, and it says: 'choosing the "
                             "relative penalty effect lambda^2. One option is to solve for "
                             "lambda^2 from a proposed K value via a heuristic (Kulis and Jordan, "
                             "2012) or validation on a data subset. Rather than assume K and "
                             "return to it in this roundabout way, in the following we aim merely "
                             "to demonstrate that there exist reasonable values of lambda^2 that "
                             "return meaningful results.'")),
        representation=("multi-hot over the distinct (attribute, value) pairs of the stream, one "
                        "binary column per pair; this is the same record content ESCROW is given, "
                        "with no embedding, no normalization and no extra field"),
        protocol_escrow=protocol_describe(),
        grid=list(grid),
        grid_unit_norm_control=list(grid_unit),
        seeds=list(seeds),
        restarts=("each price is run from the empty allocation and from the paper's base-feature "
                  "initialization; the run with the LOWER OBJECTIVE is kept, which is an internal "
                  "criterion and uses no labels"),
        feature_cap=K_CAP,
        max_iter=MAX_ITER,
        scoring=dict(
            by_dot="record to its active feature of largest x_n . a_k",
            by_norm="record to its active feature of largest ||a_k||^2",
            by_pattern="records sharing a binary feature row form one cluster",
            background="a record with no active feature gets -1, all such records are one cluster",
            credited="the best of the three is credited to BP-means in every comparison",
            ari="_ari from experiments/e4_baseline_army.py, the same function as every other "
                "number in this repository"),
        streams={})

    # ------------------------------------------------------------------ sweep
    data = {}
    for name in STREAMS:
        print(f"[{name}] building", flush=True)
        per_seed = {}
        for s in seeds:
            recs, truth = stream_for_seed(name, s, RESULTS)
            per_seed[s] = (multihot(recs), truth, recs)
        data[name] = per_seed
        X0 = per_seed[seeds[0]][0]
        row = dict(n=int(X0.shape[0]), d=int(X0.shape[1]), true_K=TRUE_K[name],
                   mean_row_sq_norm=round(float((X0 * X0).sum(1).mean()), 4),
                   seeds={}, escrow={}, sweep={})
        report["streams"][name] = row

    for name in STREAMS:
        row = report["streams"][name]
        for s in seeds:
            X, truth, recs = data[name][s]
            row["escrow"][str(s)] = escrow_cell(recs, truth)
            print(f"[{name}] seed {s} ESCROW {row['escrow'][str(s)]}", flush=True)
            cells = []
            for lam2 in grid:
                c = cell(X, truth, lam2)
                cells.append(c)
                print(f"  [{name}] seed={s} lambda2={lam2:<6} K={c['K']:<5} "
                      f"obj={c['objective']:<12} ARIbest={c['ARI_best_of_three']:<9} "
                      f"init={c['init']:<5} cap={int(c['cap_bound'])} conv={int(c['converged'])} "
                      f"{c['seconds']}s", flush=True)
            row["sweep"][str(s)] = cells

    # ------------------------------------------------------- oracle, per stream
    for name in STREAMS:
        row = report["streams"][name]
        per_seed_oracle = {}
        for s in seeds:
            cells = row["sweep"][str(s)]
            best = max(cells, key=lambda c: c["ARI_best_of_three"])
            per_seed_oracle[str(s)] = dict(lambda2=best["lambda2"], K=best["K"],
                                           ARI=best["ARI_best_of_three"], ARI_by=best["ARI"],
                                           cap_bound=best["cap_bound"])
        # the oracle value the stream would hand another stream: best on the seed-mean ARI
        mean_by_lambda = {}
        for i, lam2 in enumerate(grid):
            mean_by_lambda[lam2] = _mean([row["sweep"][str(s)][i]["ARI_best_of_three"]
                                          for s in seeds])
        oracle_lambda = max(mean_by_lambda, key=lambda p: mean_by_lambda[p])
        row["oracle"] = dict(
            what="best ARI on the TEST LABELS over the grid; an upper bound, not an operating point",
            per_seed=per_seed_oracle,
            ARI=_spread([v["ARI"] for v in per_seed_oracle.values()]),
            K=_spread([float(v["K"]) for v in per_seed_oracle.values()]),
            seed_mean_ARI_by_lambda2={str(k): v for k, v in mean_by_lambda.items()},
            oracle_lambda2_on_seed_mean=oracle_lambda,
            oracle_seed_mean_ARI=mean_by_lambda[oracle_lambda])

    # -------------------------------------------------- transferred and fixed
    idx = {lam2: i for i, lam2 in enumerate(grid)}
    for name in STREAMS:
        row = report["streams"][name]
        transfers = {}
        for src in STREAMS:
            if src == name:
                continue
            lam2 = report["streams"][src]["oracle"]["oracle_lambda2_on_seed_mean"]
            cells = [row["sweep"][str(s)][idx[lam2]] for s in seeds]
            transfers[src] = dict(lambda2=lam2,
                                  ARI=_spread([c["ARI_best_of_three"] for c in cells]),
                                  K=_spread([float(c["K"]) for c in cells]),
                                  cap_bound_seeds=sum(1 for c in cells if c["cap_bound"]))
        row["transferred"] = dict(
            what="the oracle price of another stream, applied unchanged",
            sources=transfers,
            best_source=max(transfers, key=lambda s: transfers[s]["ARI"]["mean"]),
            best_ARI=max(t["ARI"]["mean"] for t in transfers.values()))

        fixed = {}
        for label, lam2 in PAPER_VALUES.items():
            cells = [row["sweep"][str(s)][idx[lam2]] for s in seeds]
            fixed[label] = dict(lambda2=lam2,
                                ARI=_spread([c["ARI_best_of_three"] for c in cells]),
                                K=_spread([float(c["K"]) for c in cells]),
                                cap_bound_seeds=sum(1 for c in cells if c["cap_bound"]))
        row["paper_values"] = dict(
            what="the two values the MAD-Bayes paper itself used, on image principal components",
            values=fixed,
            best_ARI=max(v["ARI"]["mean"] for v in fixed.values()))

    # ------------------------------------------- the K-seeded heuristic arm
    for name in STREAMS:
        row = report["streams"][name]
        per_seed = {}
        for s in seeds:
            X, truth, _ = data[name][s]
            lam2, built = heuristic_lambda2(X, TRUE_K[name])
            c = cell(X, truth, lam2)
            per_seed[str(s)] = dict(lambda2=round(lam2, 6), features_built=built, K=c["K"],
                                    ARI=c["ARI_best_of_three"], ARI_by=c["ARI"],
                                    cap_bound=c["cap_bound"])
            print(f"  [{name}] seed={s} heuristic lambda2={lam2:.4f} K={c['K']} "
                  f"ARIbest={c['ARI_best_of_three']}", flush=True)
        row["k_seeded_heuristic"] = dict(
            what=("the paper's first stated option, solved for the TRUE K by its own feature "
                  "initialization; this arm gets the true group count, which ESCROW never gets"),
            per_seed=per_seed,
            ARI=_spread([v["ARI"] for v in per_seed.values()]),
            K=_spread([float(v["K"]) for v in per_seed.values()]))

    # -------------------------------------------------------------- controls
    controls = {}
    lam_small = grid[0]
    for name in STREAMS:
        X, truth, _ = data[name][seeds[0]]
        t0 = time.perf_counter()
        Z, A, info = bp_means(X, lam_small, k_cap=X.shape[0] + 1, max_iter=4, init="empty")
        labs = label_sets(X, Z, A)
        aris = {k: round(_ari(truth, v), 6) for k, v in labs.items()}
        capped_cell = report["streams"][name]["sweep"][str(seeds[0])][0]
        controls[name] = dict(
            lambda2=lam_small, seed=seeds[0],
            uncapped=dict(K=info["K"], ARI_best_of_three=round(max(aris.values()), 6), ARI=aris,
                          iters=info["iters"], converged=info["converged"],
                          seconds=round(time.perf_counter() - t0, 3),
                          note="max_iter reduced to 4 for this control, stated as a reduction"),
            capped=dict(K=capped_cell["K"], ARI_best_of_three=capped_cell["ARI_best_of_three"],
                        cap=K_CAP),
            cap_helps_bpmeans=bool(capped_cell["ARI_best_of_three"]
                                   >= round(max(aris.values()), 6)))
        print(f"[control cap {name}] uncapped K={info['K']} "
              f"ARI={round(max(aris.values()), 6)} vs capped K={capped_cell['K']} "
              f"ARI={capped_cell['ARI_best_of_three']}", flush=True)
    report["control_feature_cap"] = dict(
        what=("the smallest price on the grid, rerun with no feature cap on the first seed, to "
              "show which direction the tractability cap of 250 moves BP-means"),
        per_stream=controls)

    norm_control = {}
    for name in STREAMS:
        X, truth, _ = data[name][seeds[0]]
        nrm = np.sqrt((X * X).sum(1, keepdims=True))
        nrm[nrm == 0] = 1.0
        Xn = X / nrm
        cells = []
        for lam2 in grid_unit:
            c = cell(Xn, truth, lam2)
            cells.append(dict(lambda2=lam2, K=c["K"], ARI_best_of_three=c["ARI_best_of_three"],
                              cap_bound=c["cap_bound"], objective=c["objective"]))
            print(f"  [norm {name}] lambda2={lam2} K={c['K']} "
                  f"ARIbest={c['ARI_best_of_three']}", flush=True)
        best = max(cells, key=lambda c: c["ARI_best_of_three"])
        norm_control[name] = dict(cells=cells, oracle=best,
                                  escrow_ARI=report["streams"][name]["escrow"][str(seeds[0])]["ARI"])
    report["control_l2_normalized_rows"] = dict(
        what=("the same sweep on L2-normalized rows, seed 0 only, so the outcome cannot be "
              "blamed on unnormalized row norms; the oracle here is again taken on test labels"),
        per_stream=norm_control)

    # -------------------------------------------------------------- verdicts
    escrow_mean = {n: _mean([report["streams"][n]["escrow"][str(s)]["ARI"] for s in seeds])
                   for n in STREAMS}
    escrow_K = {n: _mean([float(report["streams"][n]["escrow"][str(s)]["K"]) for s in seeds])
                for n in STREAMS}
    verdict = {}
    for name in STREAMS:
        row = report["streams"][name]
        tr = row["transferred"]["best_ARI"]
        orc = row["oracle"]["ARI"]["mean"]
        pap = row["paper_values"]["best_ARI"]
        heur = row["k_seeded_heuristic"]["ARI"]["mean"]
        verdict[name] = dict(
            escrow_ARI=escrow_mean[name], escrow_K=escrow_K[name], true_K=TRUE_K[name],
            bpmeans_oracle_ARI=orc, bpmeans_oracle_K=row["oracle"]["K"]["mean"],
            bpmeans_transferred_ARI=tr, bpmeans_transferred_K=(
                row["transferred"]["sources"][row["transferred"]["best_source"]]["K"]["mean"]),
            bpmeans_paper_value_ARI=pap,
            bpmeans_k_seeded_heuristic_ARI=heur,
            transferred_matches_escrow=bool(tr >= escrow_mean[name]),
            oracle_matches_escrow=bool(orc >= escrow_mean[name]),
            paper_value_matches_escrow=bool(pap >= escrow_mean[name]),
            k_seeded_heuristic_matches_escrow=bool(heur >= escrow_mean[name]))
    fired = [n for n, v in verdict.items() if v["transferred_matches_escrow"]]
    oracle_fired = [n for n, v in verdict.items() if v["oracle_matches_escrow"]]
    report["falsifier"] = dict(
        statement=("if BP-means at a TRANSFERRED lambda^2 matches ESCROW, the computed price buys "
                   "nothing a chosen constant does not and the paper's central claim is in serious "
                   "trouble"),
        test="mean ARI over the seed family, BP-means credited its best of three scorings",
        per_stream=verdict,
        streams_where_transferred_matches_escrow=fired,
        streams_where_the_oracle_upper_bound_matches_escrow=oracle_fired,
        falsifier_fires=bool(fired),
        escrow_beaten_without_test_labels=bool(fired or any(
            v["paper_value_matches_escrow"] or v["k_seeded_heuristic_matches_escrow"]
            for v in verdict.values())))

    lines = []
    for name in STREAMS:
        v = verdict[name]
        lines.append(
            f"{name}: ESCROW ARI {v['escrow_ARI']} at K {v['escrow_K']} (true K "
            f"{v['true_K']}); BP-means oracle {v['bpmeans_oracle_ARI']} at K "
            f"{v['bpmeans_oracle_K']}, transferred {v['bpmeans_transferred_ARI']} at K "
            f"{v['bpmeans_transferred_K']}, paper value {v['bpmeans_paper_value_ARI']}, "
            f"true-K heuristic {v['bpmeans_k_seeded_heuristic_ARI']}.")
    if fired:
        lines.append("THE FALSIFIER FIRES on: " + ", ".join(fired)
                     + ". A transferred constant reaches what the computed price reaches there, "
                       "and that is reported as a loss for the method.")
    else:
        lines.append("The falsifier does not fire: on no stream does a transferred constant reach "
                     "ESCROW. Neither does the oracle upper bound, which is chosen on the test "
                     "labels.")
    report["reading"] = dict(
        question=("does BP-means, the small-variance method whose hand-set penalty this paper "
                  "replaces, reach ESCROW on ESCROW's own input at an oracle price, a transferred "
                  "price, its own paper's price, or a price solved from the true K?"),
        summary=" ".join(lines))

    for name in STREAMS:
        row = report["streams"][name]
        row["escrow_summary"] = dict(
            ARI=_spread([row["escrow"][str(s)]["ARI"] for s in seeds]),
            K=_spread([float(row["escrow"][str(s)]["K"]) for s in seeds]),
            knob="none; ESCROW is one run of the shipped protocol with no calibrated parameter")

    report["mechanism_k_against_price"] = dict(
        what=("BP-means' feature count against the price, per stream, derived from the "
              "swept cells in this same report; it is the mechanism behind every number "
              "above"),
        per_stream=k_curve(report, seeds))
    report["validity_check"] = validity_check(report, seeds)
    report["seconds_total"] = round(time.time() - t_start, 2)
    stamped(report)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    vc = report["validity_check"]
    print(f"\nvalidity check (ESCROW arm against E4): {vc.get('all_match')} over "
          f"{vc.get('compared')} cells")
    print("\n" + report["reading"]["summary"])
    print(f"\nwrote {OUT} in {report['seconds_total']}s")


if __name__ == "__main__":
    main()
