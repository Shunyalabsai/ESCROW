"""E66: the baselines under the constraint the rule actually works under.

THE ASYMMETRY THIS EXISTS TO FIX, AND IT IS OURS TO OWN. The paper is scrupulous about the language
model's advantage, stating in both directions that it reads twenty records per call while the rule
reads one. It says nothing at all about a much larger advantage held by the classical baselines.

Per stream, `kmeans_silhouette` does this:

    for k in 2..20:                      19 candidate clusterings
        KMeans(n_clusters=k, n_init=10)  10 restarts each, every one a full pass over every record
        silhouette_score(X, lab)         a global score needing the whole matrix at once

That is about 190 complete passes over the entire dataset, followed by a model-selection step that
cannot be computed without holding every record simultaneously. The rule sees each record once, in
an order it does not choose, and decides irrevocably.

Reporting those two numbers side by side and calling one a loss is not a comparison of methods. It
is a comparison of access to the data, and the paper should have said so.

WHAT THIS MEASURES. The same records and the same representation, with the baselines held to the
rule's own protocol: one record at a time, no second pass, no future. Three arms per baseline, and
the third is the one that matters:

  batch, label free      what the paper reports today: full data, k chosen by silhouette
  batch, oracle k        k chosen on the test labels, an upper bound it could never have in use
  streaming              one record at a time, and for the methods that need k, k given for free

The streaming column is generous on purpose. Sequential k-means cannot choose its own k from a
stream, so it is handed the true number, which the rule has to discover. Where a method cannot be
made streaming at all it is reported as such, because "cannot run" is the honest entry and is more
informative than a blank.

WHAT WOULD REFUTE THE COMPLAINT. If the baselines keep their scores under the streaming constraint,
the asymmetry was not load bearing and the paper's current table is fair as it stands.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from escrow.protocol import record_labels, run_consensus, run_stream
from escrow.provenance import stamped

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e66_the_baselines_under_our_own_constraint.json")
KS = list(range(2, 21))
SEEDS = (0, 1, 2)


def multihot(recs):
    """The matched representation of E21: the same information the engine reads."""
    cols = sorted({f"{k}={v}" for r in recs for k, v in r.items()}
                  | {f"KEY:{k}" for r in recs for k in r})
    ix = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(recs), len(cols)), dtype=float)
    for i, r in enumerate(recs):
        for k, v in r.items():
            X[i, ix[f"{k}={v}"]] = 1.0
            X[i, ix[f"KEY:{k}"]] = 1.0
    return X


def char_ngram_matrix(recs, lo=3, hi=5):
    texts = [" ".join(f"{k}={v}" for k, v in r.items()) for r in recs]
    vocab = {}
    grams = []
    for t in texts:
        s = f"  {t.lower()}  "
        g = {}
        for n in range(lo, hi + 1):
            for i in range(len(s) - n + 1):
                tok = s[i:i + n]
                vocab.setdefault(tok, len(vocab))
                g[tok] = g.get(tok, 0) + 1
        grams.append(g)
    X = np.zeros((len(recs), len(vocab)), dtype=float)
    for i, g in enumerate(grams):
        for tok, c in g.items():
            X[i, vocab[tok]] = c
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9)


def batch_kmeans(X, truth):
    best_sil, pick, oracle = -2.0, None, -2.0
    for k in KS:
        if k >= len(X):
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        a = adjusted_rand_score(truth, lab)
        oracle = max(oracle, a)
        try:
            s = silhouette_score(X, lab)
        except ValueError:
            s = -2.0
        if s > best_sil:
            best_sil, pick = s, {"k": k, "ARI": round(float(a), 4)}
    passes = len([k for k in KS if k < len(X)]) * 10
    return {"label_free_by_silhouette": pick, "oracle_best_k": round(float(oracle), 4),
            "full_passes_over_the_data": passes}


def streaming_kmeans(X, truth, k, seed=0):
    """Sequential k-means, MacQueen 1967: one record at a time, assign to the nearest centre and
    move that centre. One pass, no revisits, no future. k has to be given, which is the whole point:
    the method cannot discover it from a stream, and the rule has to."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    centres = np.zeros((k, X.shape[1]))
    counts = np.zeros(k)
    lab = np.full(len(X), -1)
    filled = 0
    for idx in order:
        x = X[idx]
        if filled < k:                                   # seed centres from the first k records
            centres[filled] = x
            counts[filled] = 1
            lab[idx] = filled
            filled += 1
            continue
        d = np.linalg.norm(centres - x, axis=1)
        j = int(np.argmin(d))
        counts[j] += 1
        centres[j] += (x - centres[j]) / counts[j]
        lab[idx] = j
    return adjusted_rand_score(truth, lab)


def main():
    import llm_graph_formation as L
    ds = L.build_wikipedia()
    recs, truth = ds["records"], ds["truth"]
    k_true = len(set(truth))
    reps = {"matched multi-hot (E21)": multihot(recs),
            "character 3-to-5-grams (E43)": char_ngram_matrix(recs)}

    report = {"experiment": "E66 the baselines held to the rule's own protocol",
              "records": len(recs), "true_kinds": k_true,
              "why": ("kmeans_silhouette makes about 190 full passes over every record and then "
                      "selects k with a score that needs the whole matrix at once; the rule sees "
                      "each record once, in an order it does not choose, and cannot revisit"),
              "representations": {}}

    print(f"{len(recs)} encyclopedia records, truth is {k_true} kinds\n")
    for name, X in reps.items():
        b = batch_kmeans(X, truth)
        stream_true_k = [streaming_kmeans(X, truth, k_true, s) for s in SEEDS]
        stream_oracle = max(
            sum(streaming_kmeans(X, truth, k, s) for s in SEEDS) / len(SEEDS) for k in KS)
        row = {"batch_label_free": b["label_free_by_silhouette"],
               "batch_oracle_k": b["oracle_best_k"],
               "full_passes_the_batch_arm_makes": b["full_passes_over_the_data"],
               "streaming_with_true_k_given": {
                   "mean_ARI": round(float(np.mean(stream_true_k)), 4),
                   "per_seed": [round(float(x), 4) for x in stream_true_k]},
               "streaming_best_k_on_labels": round(float(stream_oracle), 4)}
        report["representations"][name] = row
        print(f"{name}")
        print(f"   batch, k by silhouette, {b['full_passes_over_the_data']} full passes"
              f"       ARI {b['label_free_by_silhouette']['ARI']:.4f} at k="
              f"{b['label_free_by_silhouette']['k']}")
        print(f"   batch, k chosen on the test labels                    "
              f"ARI {b['oracle_best_k']:.4f}")
        print(f"   STREAMING, one pass, true k handed to it              "
              f"ARI {np.mean(stream_true_k):.4f}   {[round(x,3) for x in stream_true_k]}")
        print(f"   STREAMING, one pass, best k chosen on the labels      "
              f"ARI {stream_oracle:.4f}\n")

    g0, _ = run_stream(recs)
    ours_single = round(adjusted_rand_score(truth, record_labels(g0, len(recs))), 4)
    cons = run_consensus(recs, R=3, seed=11)
    ours_cons = round(adjusted_rand_score(truth, cons["best"]["labels"]), 4)
    report["ours"] = {"streaming_single_order": ours_single, "streaming_shortest_of_3": ours_cons,
                      "K_discovered": cons["best"]["K"], "k_never_given": True}
    print(f"ESCROW, one record at a time, k never given")
    print(f"   single order                                          ARI {ours_single:.4f}")
    print(f"   shortest of three orders                              ARI {ours_cons:.4f} "
          f"at K={cons['best']['K']}")

    mh = report["representations"]["matched multi-hot (E21)"]
    report["headline"] = {
        "batch_label_free_ARI": mh["batch_label_free"]["ARI"],
        "same_method_streaming_with_true_k_given": mh["streaming_with_true_k_given"]["mean_ARI"],
        "ours_streaming_k_never_given": ours_cons,
        "full_passes_the_batch_arm_makes": mh["full_passes_the_batch_arm_makes"],
        "reading": ("the batch column is what the paper reports today. The streaming column holds "
                    "the same method to the rule's protocol and hands it the true number of kinds "
                    "for free, which the rule has to discover. If the streaming number falls below "
                    "ours, the table in the paper is comparing access to the data and not methods"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
