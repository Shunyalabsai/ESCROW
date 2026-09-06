"""MusicBrainz 20K (musicbrainz-20-A01): ESCROW v1 vs embedding baselines.

Conditions:
  A: categorical-only facets {source, number, year, language, length_band}
  B: A + declared presence-token facets from title/artist/album
Truth: CID (10,000 clusters over 19,375 records). NEVER fed to any method's input.

Run from the code directory:
  CUDA_VISIBLE_DEVICES=2 ../.venv/bin/python experiments/mb20k_benchmark.py \
      [--n 500] [--cond both|A|B] [--skip-escrow] [--skip-baselines] [--tag dry]
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.environ.get("ESCROW_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# data, baselines and results live under ESCROW_ROOT (default: the directory above this code directory)

from escrow.provenance import stamped  # noqa: E402

import numpy as np

DATA = os.path.join(ROOT, "data/musicbrainz/musicbrainz-20-A01.csv.dapo")
OUT_DIR = os.path.join(ROOT, "results")

# The run that produced results/mb20k_full2.json raised the engine's candidate pool cap from
# its 4096 default to 32768 (recorded in that file's "deviations") so that no candidate is ever
# evicted on this dataset; the first attempt crashed when a candidate touched by the current
# record was evicted. The script now sets it, instead of only describing it.
CAND_POOL_CAP = 32768
REPAIR_EVERY = 200          # the cadence results/mb20k_full2.json records
FORCE_REPAIR_K = 300        # forced repair above this many nodes
FORCE_REPAIR_GAP = 60       # ... but never closer together than this many records

# ---------------------------------------------------------------- parsing ----
MISSING = {"", "null", "unk.", "[unknown]", "unknown"}


def is_missing(v: str) -> bool:
    return v.strip().lower() in MISSING


RE_MMSS = re.compile(r"^(\d+):(\d{1,2})$")
RE_MSEC = re.compile(r"^(\d+)\s*m\s*(\d+)\s*sec$", re.I)
RE_DIG = re.compile(r"^\d+$")
RE_DEC = re.compile(r"^\d+\.\d+$")


def parse_len_seconds(raw: str):
    """Deterministic length parse -> seconds, or None if missing/unparseable.
    mm:ss -> m*60+s (0 => placeholder, missing); 'Xm Ysec' -> m*60+s;
    pure digits >= 1000 -> milliseconds/1000, < 1000 -> seconds (0 => missing);
    decimal 'x.y' -> minutes*60; anything else -> missing."""
    v = raw.strip().lower()
    if v in MISSING:
        return None
    m = RE_MMSS.match(v)
    if m:
        s = int(m.group(1)) * 60 + int(m.group(2))
        return float(s) if s > 0 else None
    m = RE_MSEC.match(v)
    if m:
        return float(int(m.group(1)) * 60 + int(m.group(2)))
    if RE_DIG.match(v):
        x = int(v)
        if x <= 0:
            return None
        return x / 1000.0 if x >= 1000 else float(x)
    if RE_DEC.match(v):
        return float(v) * 60.0
    return None


TOK_SPLIT = re.compile(r"[\W_]+", re.UNICODE)


def tokenize(s: str):
    return [t for t in TOK_SPLIT.split(s.lower()) if len(t) >= 2]


A_KEYS = ("source", "number", "year", "language", "length_band")


def load(n_slice=None):
    rows = list(csv.DictReader(open(DATA)))
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)          # stream order, seed 0, once
    rows = [rows[i] for i in order]
    if n_slice:
        rows = rows[:n_slice]
    truth = [r["CID"] for r in rows]

    # token global frequency over the records in this run (full corpus in the
    # main run); >= 2 kept
    tf = Counter()
    for r in rows:
        for f in ("title", "artist", "album"):
            if not is_missing(r[f]):
                tf.update(tokenize(r[f]))
    kept = {t for t, c in tf.items() if c >= 2}

    recA, recB, txtA, txtB = [], [], [], []
    for r in rows:
        a = {}
        if not is_missing(r["SourceID"]):
            a["source"] = r["SourceID"].strip()
        if not is_missing(r["number"]):
            a["number"] = r["number"].strip()
        if not is_missing(r["year"]):
            a["year"] = r["year"].strip()
        if not is_missing(r["language"]):
            a["language"] = r["language"].strip()
        sec = parse_len_seconds(r["length"])
        if sec is not None:
            a["length_band"] = "b%d" % int(sec // 30)
        recA.append(a)
        ta = " ".join("%s=%s" % (k, a[k]) for k in A_KEYS if k in a)
        txtA.append(ta if ta else "empty")

        b = dict(a)
        for f, pre in (("title", "t"), ("artist", "a"), ("album", "b")):
            if is_missing(r[f]):
                continue
            for t in tokenize(r[f]):
                if t in kept:
                    b["%s:%s" % (pre, t)] = "1"
        recB.append(b)
        raws = [r[f].strip() for f in ("title", "artist", "album")
                if not is_missing(r[f])]
        tb = (ta + " | " + " | ".join(raws)) if raws else ta
        txtB.append(tb.strip(" |") or "empty")
    return rows, truth, recA, recB, txtA, txtB, len(kept)


# ---------------------------------------------------------------- metrics ----
def pairwise(truth, pred):
    tp = sum(c * (c - 1) // 2 for c in Counter(zip(truth, pred)).values())
    t = sum(c * (c - 1) // 2 for c in Counter(truth).values())
    p = sum(c * (c - 1) // 2 for c in Counter(pred).values())
    prec = tp / p if p else 1.0            # vacuous precision when no pairs asserted
    rec = tp / t if t else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"pairwise_precision": round(prec, 4), "pairwise_recall": round(rec, 4),
            "pairwise_F1": round(f1, 4), "predicted_pairs": int(p),
            "true_pairs": int(t), "tp_pairs": int(tp)}


def score(truth, pred):
    from sklearn.metrics import adjusted_rand_score
    d = pairwise(truth, pred)
    d["ARI"] = round(adjusted_rand_score(truth, pred), 4)
    d["K_found"] = len(set(pred))
    return d


def singletonize(labels):
    """Noise labels (-1) become unique singleton clusters (the null assertion)."""
    out, nxt = [], 0
    for l in labels:
        if l == -1:
            nxt += 1
            out.append("s%d" % nxt)
        else:
            out.append(int(l))
    return out


# ---------------------------------------------------------------- escrow -----
def run_escrow(recs, repair_every=REPAIR_EVERY):
    from escrow.engine import EscrowGraph
    from escrow.batch import BatchObjective
    g = EscrowGraph(cand_pool_cap=CAND_POOL_CAP)
    b = BatchObjective(g).install()
    t0 = time.time()
    rep_time, rep_calls = 0.0, 0
    last_rep = 0
    k_traj = []
    for i, r in enumerate(recs, 1):
        g.process(r)
        forced = g.K > FORCE_REPAIR_K and i - last_rep >= FORCE_REPAIR_GAP
        if (i - last_rep >= repair_every or forced) and i < len(recs):
            rt = time.time()
            b.repair()
            rep_time += time.time() - rt
            rep_calls += 1
            last_rep = i
            k_traj.append([i, g.K])
            print("  [escrow] rec %d/%d K=%d pool=%d elapsed=%.0fs repair_s=%.0f"
                  % (i, len(recs), g.K, len(g.pool), time.time() - t0, rep_time),
                  flush=True)
        elif i % 1000 == 0:
            print("  [escrow] rec %d/%d K=%d elapsed=%.0fs"
                  % (i, len(recs), g.K, time.time() - t0), flush=True)
    rt = time.time()
    b.repair()
    rep_time += time.time() - rt
    rep_calls += 1
    total = time.time() - t0
    k_traj.append([len(recs), g.K])

    n = len(recs)
    best = [None] * n
    memberships = [0] * n
    for v in g.nodes.values():
        sz = len(v.members)
        for m in v.members:
            i = m - 1
            memberships[i] += 1
            cand = (sz, v.nid)
            if best[i] is None or cand < best[i]:
                best[i] = cand
    labels, nxt = [], 0
    for i in range(n):
        if best[i] is None:
            nxt += 1
            labels.append("u%d" % nxt)      # unassigned = its own singleton
        else:
            labels.append(best[i][1])
    stats = {"K_nodes_final": g.K,
             "nodes_minted_total": len(g.mint_log),
             "records_in_some_node": n - nxt,
             "records_unassigned_singleton": nxt,
             "records_in_multiple_nodes": sum(1 for m in memberships if m > 1),
             "repair_calls": rep_calls,
             "repair_seconds": round(rep_time, 1),
             "runtime_seconds": round(total, 1),
             "repair_every": repair_every,
             "cand_pool_cap": CAND_POOL_CAP,
             "K_trajectory_tail": k_traj[-15:]}
    return labels, stats


# ---------------------------------------------------------------- baselines --
def embed(texts):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    X = model.encode(texts, batch_size=512, normalize_embeddings=True,
                     show_progress_bar=False)
    return np.asarray(X, dtype=np.float32)


def kmeans_oracle_k(X, K, seed=0, iters=100):
    """torch GPU kmeans++ (seeded) + Lloyd. Deterministic given seed/device."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X).to(dev)
    n, d = Xt.shape
    K = min(K, n)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    C = torch.empty((K, d), device=dev)
    idx = torch.randint(n, (1,), generator=gen, device=dev)
    C[0] = Xt[idx[0]]
    closest = ((Xt - C[0]) ** 2).sum(1)
    for k in range(1, K):
        probs = closest.clamp_min(0)
        if float(probs.sum()) <= 0:
            idx = torch.randint(n, (1,), generator=gen, device=dev)
        else:
            idx = torch.multinomial(probs, 1, generator=gen)
        C[k] = Xt[idx[0]]
        closest = torch.minimum(closest, ((Xt - C[k]) ** 2).sum(1))
        if k % 2000 == 0:
            print("    kmeans++ seeded %d/%d" % (k, K), flush=True)
    assign = None
    for it in range(iters):
        scores = (C * C).sum(1).unsqueeze(0) - 2.0 * (Xt @ C.t())
        newa = scores.argmin(1)
        del scores
        if assign is not None and bool(torch.equal(newa, assign)):
            break
        assign = newa
        sums = torch.zeros_like(C)
        cnt = torch.zeros(K, device=dev)
        sums.index_add_(0, assign, Xt)
        cnt.index_add_(0, assign, torch.ones(n, device=dev))
        mask = cnt > 0
        C = torch.where(mask.unsqueeze(1), sums / cnt.clamp_min(1).unsqueeze(1), C)
    return assign.cpu().numpy().tolist()


def dp_means(X, lam):
    """Greedy one-pass dp-means, squared euclidean vs lambda, first record seeds."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X).to(dev)
    n, d = Xt.shape
    C = torch.empty((n, d), device=dev)
    C[0] = Xt[0]
    k = 1
    assign = np.empty(n, dtype=np.int64)
    assign[0] = 0
    for i in range(1, n):
        x = Xt[i]
        sc = (C[:k] * C[:k]).sum(1) - 2.0 * (C[:k] @ x)
        j = int(sc.argmin())
        d2 = float(sc[j]) + float(x @ x)
        if d2 > lam:
            C[k] = x
            assign[i] = k
            k += 1
        else:
            assign[i] = j
    return assign.tolist()


def cosine_dist_matrix(X):
    D = 1.0 - X @ X.T
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float64)


def sweep(truth, fns):
    """fns: {param: callable -> labels}. Oracle-best by pairwise F1 on TEST labels."""
    per = {}
    for p, fn in fns.items():
        t0 = time.time()
        try:
            lab = fn()
            sc = score(truth, lab)
        except Exception as ex:
            sc = {"error": ("%s: %s" % (type(ex).__name__, ex))[:300]}
        sc["seconds"] = round(time.time() - t0, 1)
        per[str(p)] = sc
        print("    param=%s -> %s" % (p, json.dumps(sc)), flush=True)
    ok = {p: v for p, v in per.items() if "pairwise_F1" in v}
    best = max(ok, key=lambda p: ok[p]["pairwise_F1"]) if ok else None
    return {"oracle_param": best,
            "oracle_score": ok.get(best),
            "oracle_note": "parameter chosen on the test labels (max pairwise F1): ORACLE",
            "sweep": per}


def run_baselines(texts, truth, true_k):
    from sklearn.cluster import HDBSCAN
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    out = {}
    t0 = time.time()
    X = embed(texts)
    out["embedding_seconds"] = round(time.time() - t0, 1)
    print("  embedded %d texts in %.0fs" % (len(texts), out["embedding_seconds"]),
          flush=True)

    t0 = time.time()
    lab = kmeans_oracle_k(X, true_k)
    sc = score(truth, lab)
    sc["seconds"] = round(time.time() - t0, 1)
    out["kmeans_oracleK"] = {"score": sc,
                             "note": "given the TRUE cluster count K=%d: ORACLE" % true_k}
    print("  kmeans_oracleK -> %s" % json.dumps(sc), flush=True)

    D = cosine_dist_matrix(X)

    def hdb(p):
        lab = HDBSCAN(min_cluster_size=p, metric="precomputed").fit_predict(D)
        return singletonize(lab)
    print("  hdbscan sweep:", flush=True)
    out["hdbscan"] = sweep(truth, {p: (lambda p=p: hdb(p)) for p in (3, 5, 10, 20, 50)})
    out["hdbscan"]["note"] = ("precomputed cosine distances; noise points counted "
                              "as singleton clusters")

    print("  agglomerative (average cosine) sweep:", flush=True)
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="average")
    del cond
    out["agglo_avg_cosine"] = sweep(
        truth, {p: (lambda p=p: fcluster(Z, t=p, criterion="distance").tolist())
                for p in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)})
    out["agglo_avg_cosine"]["note"] = ("scipy average linkage on cosine pdist, cut at "
                                       "distance_threshold")
    del D, Z

    print("  dp-means sweep:", flush=True)
    out["dp_means"] = sweep(
        truth, {p: (lambda p=p: dp_means(X, p))
                for p in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)})
    out["dp_means"]["note"] = "greedy one-pass, squared-euclidean vs lambda"
    return out


# ---------------------------------------------------------------- main -------
DECLARED_CHOICES = [
    "stream order: rows shuffled once with random.Random(0); the same order feeds ESCROW and every baseline",
    "ground truth: CID; it never appears in any engine facet or baseline input text",
    "missing markers, case-insensitive: '', 'null', 'unk.', '[unknown]', 'unknown' (the last added because the raw data uses it 299x as an evident placeholder); missing fields are omitted from engine records and baseline texts identically",
    "length parse (deterministic, applied identically everywhere): 'mm:ss' -> m*60+s; 'Xm Ysec' -> m*60+s; pure digits >=1000 -> milliseconds/1000, digits <1000 -> seconds; decimal 'x.y' -> minutes*60 (value distribution makes minutes the only plausible unit); parse failure or <=0 seconds -> missing ('00:00' and '0' are placeholders)",
    "length_band = floor(seconds/30), value 'b<band>'",
    "no normalization of number/year/language values: raw strings ('Eng.' stays distinct from 'English'); the same handicap for every method",
    "condition A engine record: {source, number, year, language, length_band}, missing omitted; NO text fields",
    "condition B adds presence-token facets: lowercase, split on non-word chars and underscore (unicode), token length >=2, global instance frequency >=2 counted over title+artist+album across the full corpus being run; keys 't:<tok>'/'a:<tok>'/'b:<tok>', value '1'; missing-marker text fields skipped",
    "baseline text A: 'source=.. number=.. year=.. language=.. length_band=..' with missing fields omitted (identical band value the engine sees); fully-missing record encodes as 'empty'",
    "baseline text B: text A + ' | ' + raw title + ' | ' + raw artist + ' | ' + raw album (missing-marker fields omitted). ASYMMETRY AGAINST ESCROW: baselines see the raw strings (word order, full vocabulary, no frequency filter); ESCROW sees only the filtered presence tokens",
    "embeddings: sentence-transformers all-MiniLM-L6-v2, normalize_embeddings=True, one embedding per condition shared by all baselines",
    "every swept baseline parameter is chosen on the test labels by max pairwise F1 and is marked ORACLE",
    "kmeans oracle-K: given the true cluster count; torch GPU kmeans++ (seed 0) + Lloyd <=100 iters (sklearn KMeans at K=10000/n=19375 is impractical; same algorithm, declared implementation)",
    "agglomerative: scipy average linkage on the cosine distance matrix, fcluster at the swept distance thresholds (equivalent cut to sklearn distance_threshold)",
    "hdbscan: sklearn HDBSCAN on the precomputed cosine distance matrix; min_cluster_size swept; noise (-1) points become singleton clusters (the null assertion), the same convention as unassigned ESCROW records",
    "ESCROW partition for scoring: a record in multiple nodes is assigned to its smallest containing node (tie: lowest node id); records in no node are singletons; multi-membership counts reported",
    "ESCROW repair cadence: BatchObjective.repair() every 200 records, plus a forced repair whenever K > 300 and at least 60 records have passed since the last repair, plus one final repair; on this dataset K never exceeds 3, so the forced-repair guard never fires and the cadence is a flat 200",
    "ESCROW candidate pool cap: EscrowGraph(cand_pool_cap=32768); the engine default is 4096, at which a candidate touched by the current record can be evicted mid-record, which crashed the first attempt. 32768 removes eviction entirely on this dataset and is set by the script",
    "pairwise P/R/F1 over induced record pairs vs CID; if a method asserts zero pairs its precision is vacuous (reported 1.0) and F1 is 0 via recall",
]

CONTEXT = ("FAMER-class published results on MusicBrainz-20K reach pairwise F1 ~0.8-0.9 "
           "with tuned similarity graphs and full text access; not run here, not "
           "head-to-head comparable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--cond", default="both", choices=["both", "A", "B"])
    ap.add_argument("--skip-escrow", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    rows, truth, recA, recB, txtA, txtB, n_kept_tokens = load(args.n)
    true_k = len(set(truth))
    print("loaded %d records, true K=%d, kept tokens=%d"
          % (len(rows), true_k, n_kept_tokens), flush=True)

    report = {"dataset": "musicbrainz-20-A01 (MusicBrainz 20K)",
              "n_records": len(rows), "true_K": true_k,
              "tag": args.tag,
              "kept_tokens_freq_ge2": n_kept_tokens,
              "declared_choices": DECLARED_CHOICES,
              "context_published_results": CONTEXT,
              "deviations": [],
              "conditions": {}}

    conds = []
    if args.cond in ("both", "A"):
        conds.append(("A_categorical_only", recA, txtA))
    if args.cond in ("both", "B"):
        conds.append(("B_token_bridge", recB, txtB))

    out_path = os.path.join(OUT_DIR, "mb20k_%s.json" % args.tag)
    for cname, recs, texts in conds:
        print("=== condition %s ===" % cname, flush=True)
        cres = {}
        if not args.skip_escrow:
            lab, stats = run_escrow(recs)
            cres["escrow"] = {"score": score(truth, lab), **stats}
            print("  ESCROW -> %s" % json.dumps(cres["escrow"]["score"]), flush=True)
        if not args.skip_baselines:
            cres["baselines"] = run_baselines(texts, truth, true_k)
        report["conditions"][cname] = cres
        with open(out_path, "w") as f:
            json.dump(stamped(report), f, indent=2)
        print("checkpointed %s" % out_path, flush=True)

    report["honest_reading"] = ""            # filled in by the analyst after the run
    with open(out_path, "w") as f:
        json.dump(stamped(report), f, indent=2)
    print(json.dumps(report, indent=2))
    print("written %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
