"""E43: methods older than the transformer, on the tasks a language model wins.

WHY. Two results in this paper are read as evidence that a language model brings something the
counting cannot: it reaches ARI 0.9905 on the encyclopedia stream where the engine reaches 0.8744,
and AutoPKG's agent reaches F1 0.6355 on key identity where the engine reaches 0.0008. Both are
called semantic tasks. Before conceding either, the tasks deserve a floor: what does a method with
no model in it at all score on the same records? If a bag of words from 1972 or a lowercase from
1960 reaches the same number, the win is not evidence about semantics, it is evidence that the task
was easy, and the paper must say so.

THE FLOOR, six of them, on the identical records the language model read:
  words         term frequency over the record rendered as key=value text, then k-means
  chars         character 3-to-5-grams, which cannot represent a word, let alone a meaning
  keyset        Jaccard over the set of keys, ignoring every value
  multihot      the paper's own matched representation (E21), for continuity
  keys_only     multi-hot over key presence alone
  values_only   multi-hot over (key, value) pairs alone

Every one chooses its own k by silhouette, consulting no label, and the oracle column takes the best
k on the test labels and is stated as the upper bound it could not have in use. The fairness rule
holds in the direction that costs us: any input the engine gets, these get.

WHAT WOULD FALSIFY THE READING. If every floor sits far below 0.99 on the encyclopedia stream, the
language model really is buying something, and the paper should say that plainly instead of
explaining it away.
"""
from __future__ import annotations
import collections, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

RUNS = os.path.join(ROOT, "results", "llm_gf_runs")
OUT = os.path.join(ROOT, "results", "e43_traditional_on_semantic.json")
KS = list(range(2, 21))

# the two numbers this is a floor for, read from the paper's own result files
REFERENCE = {"wikipedia": {"llm_sampled_mean_ARI": 0.9905, "llm_greedy_ARI": 0.9896,
                           "escrow_ARI": 0.8744},
             "lazada": {"llm_agreement_across_seeds": 0.1647, "escrow_agreement": 1.0}}


def as_text(rec):
    return " ".join(f"{k}={v}" for k, v in sorted(rec.items()))


def features(recs):
    """Six representations of the same records, none of them a model."""
    texts = [as_text(r) for r in recs]
    F = {}
    F["words"] = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_]+", min_df=2).fit_transform(texts).toarray()
    F["chars"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=4000).fit_transform(texts).toarray()
    keys = sorted({k for r in recs for k in r})
    ki = {k: i for i, k in enumerate(keys)}
    K = np.zeros((len(recs), len(keys)))
    for i, r in enumerate(recs):
        for k in r:
            K[i, ki[k]] = 1.0
    F["keys_only"] = K
    pairs = sorted({(k, str(v)) for r in recs for k, v in r.items()})
    pi = {p: i for i, p in enumerate(pairs)}
    V = np.zeros((len(recs), len(pairs)))
    for i, r in enumerate(recs):
        for k, v in r.items():
            V[i, pi[(k, str(v))]] = 1.0
    F["values_only"] = V
    F["multihot"] = np.hstack([K, V])                 # the E21 matched representation
    # keyset Jaccard, expressed as a normalised key vector so cosine k-means is Jaccard-like
    n = np.linalg.norm(K, axis=1, keepdims=True)
    F["keyset"] = K / np.where(n == 0, 1, n)
    return F


def run_one(X, truth):
    """Pick k by silhouette with no label, and separately by the labels, and report both."""
    rows, best_sil, pick, oracle = [], -2.0, None, -2.0
    for k in KS:
        if k >= len(X):
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        a = adjusted_rand_score(truth, lab)
        try:
            s = silhouette_score(X, lab)
        except ValueError:
            s = -2.0
        rows.append({"k": k, "silhouette": round(float(s), 4), "ARI": round(float(a), 4)})
        if s > best_sil:
            best_sil, pick = s, {"k": k, "ARI": round(float(a), 4)}
        oracle = max(oracle, a)
    return {"label_free_by_silhouette": pick, "oracle_best_k_on_test_labels": round(float(oracle), 4),
            "sweep": rows}


def main():
    report = {"experiment": "E43 methods older than the transformer on the tasks a language model wins",
              "note": ("every method reads the identical records the language model read, in the "
                       "identical file, and chooses its own k with no label; the oracle column is "
                       "an upper bound it could not have in use"),
              "reference_numbers_this_is_a_floor_for": REFERENCE, "streams": {}}

    for name in ("wikipedia", "lazada"):
        path = os.path.join(RUNS, f"data_{name}.json")
        if not os.path.exists(path):
            print("missing", path); continue
        ds = json.load(open(path))
        recs, truth = ds["records"], ds["truth"]
        keep = [i for i, t in enumerate(truth) if t is not None]
        recs = [recs[i] for i in keep]; truth = [truth[i] for i in keep]
        print(f"\n{name}: {len(recs)} records, {len(set(truth))} true types")
        F = features(recs)
        out = {}
        for fname, X in F.items():
            r = run_one(X, truth)
            out[fname] = r
            p = r["label_free_by_silhouette"]
            print(f"  {fname:12s} label free k={p['k']:2d} ARI {p['ARI']:.4f}   "
                  f"oracle {r['oracle_best_k_on_test_labels']:.4f}", flush=True)
        best_free = max(out.items(), key=lambda kv: kv[1]["label_free_by_silhouette"]["ARI"])
        report["streams"][name] = {
            "records": len(recs), "true_types": len(set(truth)), "methods": out,
            "best_label_free": {"method": best_free[0],
                                **best_free[1]["label_free_by_silhouette"]},
            "best_oracle": max(v["oracle_best_k_on_test_labels"] for v in out.values())}

    w = report["streams"].get("wikipedia", {})
    if w:
        bf = w["best_label_free"]
        report["headline"] = {
            "wikipedia_language_model_best": REFERENCE["wikipedia"]["llm_sampled_mean_ARI"],
            "wikipedia_best_method_with_no_model_in_it": bf["ARI"],
            "wikipedia_that_method": bf["method"],
            "wikipedia_escrow": REFERENCE["wikipedia"]["escrow_ARI"],
            "reading": ("if the floor is level with the language model then the encyclopedia "
                        "stream does not measure semantics and no paper should read it that way"),
        }
        print("\n" + json.dumps(report["headline"], indent=2))
    json.dump(report, open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
