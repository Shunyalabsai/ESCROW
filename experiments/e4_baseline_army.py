"""E4 first tier: the oracle-tuned / transferred / ours triple on local datasets.

Every baseline gets the same record text and the same frozen embedding. Where a
baseline has a knob, it gets its ORACLE-BEST value chosen on the test labels
(stated as such), and separately the value transferred from another dataset.
Ours is one run with no knob. Metric: ARI against the planted or held-out truth.

Run: CUDA_VISIBLE_DEVICES=<dev> python e4_baseline_army.py <wiki_cache_dir>
"""
import glob
import json
import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sentence_transformers import SentenceTransformer

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


# ---------------- datasets (records + truth) ---------------- #
def two_group(n=2000, k=4, d=10, seed=0):
    rng = random.Random(seed)
    recs, truth = [], []
    for i in range(n):
        g = i % 2
        recs.append({f"g{g}k{j}": f"g{g}v{rng.randrange(d)}" for j in range(k)})
        truth.append(g)
    return recs, truth


def planted8(n=3000, noise=0.1, seed=7):
    rng = random.Random(seed)
    recs, truth = [], []
    for _ in range(n):
        g = rng.randrange(8)
        rec = {}
        for j in range(3):
            og = rng.randrange(8) if rng.random() < noise else g
            rec[f"c{g}k{j}"] = f"c{og}v{rng.randrange(8)}"
        recs.append(rec); truth.append(g)
    return recs, truth


def wikipedia(cache_dir):
    recs, truth = [], []
    for f in sorted(glob.glob(os.path.join(cache_dir, "wiki_cache.*.json"))):
        cat = f.split("wiki_cache.")[1].split(".json")[0]
        for r in json.load(open(f)):
            recs.append(r); truth.append(cat)
    order = list(range(len(recs)))
    random.Random(0).shuffle(order)
    return [recs[i] for i in order], [truth[i] for i in order]


# ---------------- ours ---------------- #
def escrow_labels(recs):
    g = EscrowGraph(); b = BatchObjective(g)
    for i, r in enumerate(recs):
        g.process(r)
        if (i + 1) % 100 == 0:
            b.repair()
    b.repair()
    lab = [-1] * len(recs)                      # background = one cluster, honest
    for v in g.nodes.values():
        for m in v.members:
            lab[m - 1] = v.nid
    return lab, g.K


# ---------------- baselines ---------------- #
def dp_means(X, lam):
    centers = [X[0]]
    assign = [0]
    for x in X[1:]:
        d2 = [float(np.sum((x - c) ** 2)) for c in centers]
        j = int(np.argmin(d2))
        if d2[j] > lam:
            centers.append(x.copy()); assign.append(len(centers) - 1)
        else:
            assign.append(j)
    return assign


def cosine_cc(X, thr):
    S = X @ X.T
    n = len(X)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in np.nonzero(S[i] >= thr)[0]:
            if j > i:
                parent[find(int(j))] = find(i)
    return [find(i) for i in range(n)]


def sweep(fn, grid, truth):
    scores = {}
    for p in grid:
        try:
            scores[p] = adjusted_rand_score(truth, fn(p))
        except Exception:
            scores[p] = float("nan")
    best = max((v, k) for k, v in scores.items() if v == v)
    return scores, best[1], best[0]


def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    data = {
        "twogroup": two_group(),
        "planted8": planted8(),
        "wikipedia": wikipedia(cache_dir),
    }
    report = {}
    sweeps = {}                                  # (baseline, dataset) -> (scores, best_param, best_ari)
    for name, (recs, truth) in data.items():
        texts = [" ".join(f"{k}={v}" for k, v in sorted(r.items())) for r in recs]
        X = np.asarray(model.encode(texts, batch_size=256, normalize_embeddings=True,
                                    show_progress_bar=False))
        truK = len(set(truth))
        ours_lab, ours_K = escrow_labels(recs)
        row = {"n": len(recs), "true_K": truK,
               "ours": {"ARI": round(adjusted_rand_score(truth, ours_lab), 4), "K": ours_K}}

        row["kmeans_oracleK"] = {"ARI": round(adjusted_rand_score(
            truth, KMeans(truK, n_init=10, random_state=0).fit_predict(X)), 4),
            "note": "given the true K"}
        ks = range(2, 21)
        sil = {k: silhouette_score(X, KMeans(k, n_init=4, random_state=0).fit_predict(X))
               for k in ks}
        k_sil = max(sil, key=sil.get)
        row["kmeans_silhouette"] = {"ARI": round(adjusted_rand_score(
            truth, KMeans(k_sil, n_init=10, random_state=0).fit_predict(X)), 4),
            "K_chosen": k_sil}

        for bname, fn, grid in [
            ("hdbscan", lambda p: HDBSCAN(min_cluster_size=p).fit_predict(X),
             [3, 5, 10, 20, 50, 100]),
            ("agglo_threshold", lambda p: AgglomerativeClustering(
                n_clusters=None, distance_threshold=p, linkage="average",
                metric="cosine").fit_predict(X),
             [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]),
            ("cosine_components", lambda p: cosine_cc(X, p),
             [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
            ("dp_means", lambda p: dp_means(X, p),
             [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        ]:
            scores, best_p, best_a = sweep(fn, grid, truth)
            sweeps[(bname, name)] = (scores, best_p, best_a)
            row[bname] = {"oracle_param": best_p, "oracle_ARI": round(best_a, 4),
                          "sweep": {str(k): round(v, 4) for k, v in scores.items()}}
        report[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "n"}, default=str)[:400])

    # transferred: oracle param from A applied to B (worst over sources)
    for bname in ("hdbscan", "agglo_threshold", "cosine_components", "dp_means"):
        for tgt in data:
            transfers = {}
            for src in data:
                if src == tgt:
                    continue
                p = sweeps[(bname, src)][1]
                transfers[src] = round(sweeps[(bname, tgt)][0].get(p, float("nan")), 4)
            report[tgt][bname]["transferred_ARI"] = transfers
    with open(os.path.join(OUT, "e4_first_tier.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("written results/e4_first_tier.json")


if __name__ == "__main__":
    main()
