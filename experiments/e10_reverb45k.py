"""E10: REVERB45K NP-canonicalization under CESI's own protocol.

Unit: one ESCROW record per unique normalized SUBJECT string (CESI's own
granularity). Keys: each relation phrase (normalized, spaces->underscores),
value = sorted-joined normalized objects for that relation under that subject;
plus surface-token presence facets "s:<tok>" (value "1") for lowercase alnum
tokens of the subject, length >= 2, global frequency >= 2 over the record set.
Stream order: unique subjects sorted, then shuffled with seed 0.

Scoring: CESI's OWN metrics code (baselines/cesi/src/metrics.py), with
C_ele2clust etc. constructed exactly as cesi_main.py's np_evaluate does:
elements are "subject|_id" mentions, gold from true_link.subject mids, and each
per-string cluster expanded to all its mentions.

Ground truth (true_link) is used ONLY for evaluation and for the oracle knobs
of the baselines, all labelled oracle. Nothing gold ever reaches the engine or
any input text.

Run from the code directory:
  CUDA_VISIBLE_DEVICES=2 python \
    experiments/e10_reverb45k.py [--limit 500] [--out NAME.json]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = os.environ.get("ESCROW_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# data, baselines and results live under ESCROW_ROOT (default: the directory above this code directory)
sys.path.insert(0, os.path.join(ROOT, "baselines/cesi/src"))

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective
from metrics import evaluate                       # CESI's own metrics code

DATA = os.path.join(ROOT, "baselines/cesi/data/reverb45k/reverb45k_test")
OUT_DIR = os.path.join(ROOT, "results")

DECLARED_CHOICES = [
    "normalization: the dataset's own triple_norm strings are used for subject, relation and object "
    "(lowercased+lemmatized by the dataset authors); whitespace-collapsed and stripped; triples with "
    "any empty normalized part would be skipped (none are in this file)",
    "record unit: one record per unique normalized subject string; mention identity for scoring is "
    "'<normalized subject>|<_id>' exactly as cesi_main.py builds triple_unique[0]",
    "relation keys: relation phrase with spaces replaced by underscores; value = the lexicographically "
    "sorted UNIQUE normalized objects for that relation under that subject, joined with ' ; '",
    "surface facets: key 's:<tok>' value '1' for lowercase alnum tokens (re [a-z0-9]+) of the subject "
    "string, token length >= 2, token frequency >= 2 counted over the unique subject strings of this "
    "run's record set (each string counts each token type once)",
    "stream order: unique subjects sorted lexicographically then shuffled with random.Random(0)",
    "escrow repair schedule: BatchObjective.repair() every 200 records (the engine contract's band is "
    "200-500; 200 chosen after a 2000-record probe at cadence 500 showed per-repair cost exploding "
    "superlinearly in the un-repaired mint backlog), plus a forced repair when K > 300 and at least 60 "
    "records passed since the last repair, plus once at the end",
    "escrow label extraction: a record in multiple nodes' member sets is assigned to the node with the "
    "most members (tie: lowest nid); records in no node (background) are SINGLETON clusters in the "
    "primary reading, and one shared background cluster in a secondary reading reported alongside",
    "baseline input text: '<subject> ' + ' '.join('<rel_underscored>=<joined objects>') over the "
    "subject's relation keys in sorted key order -- the same normalized strings the engine receives; "
    "the subject string subsumes the engine's token facets, so no method sees signal the others lack",
    "embeddings: sentence-transformers all-MiniLM-L6-v2, normalize_embeddings=True, batch 256",
    "kmeans oracle-K: K = number of distinct gold subject mids in the record set (oracle, gold-derived), "
    "n_init=1, random_state=0",
    "hdbscan: sklearn HDBSCAN, euclidean on the normalized embeddings, min_cluster_size swept "
    "{3,5,10,20,50}, oracle-best; noise points (-1) become singleton clusters",
    "agglomerative: average linkage, cosine metric, distance_threshold swept "
    "{0.05,0.1,0.2,0.3,0.4,0.5,0.7}, oracle-best",
    "dp-means: e4_baseline_army.py's greedy (centers are first points, no centroid update), squared "
    "Euclidean on normalized embeddings (= 2-2cos), lambda swept {0.1,0.25,0.5,0.75,1.0,1.25,1.5}, "
    "oracle-best",
    "oracle-best selection: parameter maximizing (macro_f1+micro_f1+pair_f1)/3 on the test labels; the "
    "full sweep is reported so every per-parameter number is visible",
    "all clusterings are over unique subject strings and expanded to mentions for scoring, identically "
    "for ESCROW and every baseline",
]

DEVIATIONS = [
    "CESI's proc_ent normalization calls gensim.utils.lemmatize, which needs the dead 'pattern' "
    "package (Python<=3.6); not installable on this box's Python 3.12. The dataset's own triple_norm "
    "field (the authors' lowercase+lemma normalization, which CESI itself uses for relations) is used "
    "for all strings instead, applied identically to ESCROW and all baselines. proc_ent additionally "
    "drops non-NN/VB/JJ/RB tokens (e.g. determiners), so CESI merges slightly more surface variants "
    "for free than we do; our string space is strictly harder, and published CESI numbers are context, "
    "not a matched comparison (they also consume side information we do not).",
]


# ------------------------- data ------------------------- #
def load_triples():
    triples = []
    for line in open(DATA, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        s, r, o = (" ".join(str(x).split()) for x in t["triple_norm"])
        if not s or not r or not o:
            continue
        triples.append({"sub": s, "rel": r, "obj": o, "_id": t["_id"],
                        "mid": t["true_link"]["subject"]})
    return triples


def build_records(triples, limit=0):
    import random
    subjects = sorted({t["sub"] for t in triples})
    random.Random(0).shuffle(subjects)
    if limit:
        subjects = subjects[:limit]
    keep = set(subjects)
    triples = [t for t in triples if t["sub"] in keep]

    rel_objs = defaultdict(lambda: defaultdict(set))
    for t in triples:
        rel_objs[t["sub"]][t["rel"]].add(t["obj"])

    tok_freq = defaultdict(int)
    for s in subjects:
        for w in set(re.findall(r"[a-z0-9]+", s.lower())):
            if len(w) >= 2:
                tok_freq[w] += 1

    records, texts = [], []
    for s in subjects:
        rec = {}
        for rel in sorted(rel_objs[s]):
            rec[rel.replace(" ", "_")] = " ; ".join(sorted(rel_objs[s][rel]))
        text = s + " " + " ".join(f"{k}={v}" for k, v in sorted(rec.items()))
        for w in set(re.findall(r"[a-z0-9]+", s.lower())):
            if len(w) >= 2 and tok_freq[w] >= 2:
                rec[f"s:{w}"] = "1"
        records.append(rec)
        texts.append(text.strip())
    return subjects, records, texts, triples


def build_gold(triples):
    true_ent2clust = defaultdict(set)
    sub2mentions = defaultdict(list)
    for t in triples:
        ele = t["sub"] + "|" + str(t["_id"])
        true_ent2clust[ele].add(t["mid"])
        sub2mentions[t["sub"]].append(ele)
    true_clust2ent = defaultdict(set)
    for ele, cl in true_ent2clust.items():
        for c in cl:
            true_clust2ent[c].add(ele)
    return dict(true_ent2clust), dict(true_clust2ent), dict(sub2mentions)


# ------------------------- scoring (CESI construction) ------------------------- #
def score(subjects, labels, sub2mentions, true_ent2clust, true_clust2ent):
    C_ele2clust, C_clust2ele = {}, defaultdict(set)
    for s, cid in zip(subjects, labels):
        for m in sub2mentions[s]:
            C_ele2clust[m] = {cid}
            C_clust2ele[cid].add(m)
    r = evaluate(C_ele2clust, dict(C_clust2ele), true_ent2clust, true_clust2ent)
    r["system_clusters"] = len(C_clust2ele)
    r["system_singletons"] = sum(1 for v in C_clust2ele.values() if len(v) == 1)
    return r


# ------------------------- ours ------------------------- #
def run_escrow(records, repair_every=300):
    g = EscrowGraph()
    b = BatchObjective(g).install()
    t0 = time.time()
    repair_s, last_repair = 0.0, 0
    for i, rec in enumerate(records, 1):
        g.process(rec)
        if (i % repair_every == 0) or (g.K > 300 and i - last_repair >= 60):
            rt = time.time()
            b.repair()
            repair_s += time.time() - rt
            last_repair = i
        if i % 250 == 0:
            print(f"[escrow] {i}/{len(records)} K={g.K} keys={len(g.keys)} "
                  f"pool={len(g.pool)} mints={len(g.mint_log)} "
                  f"elapsed={time.time()-t0:.0f}s (repair {repair_s:.0f}s)", flush=True)
    rt = time.time()
    b.repair()
    repair_s += time.time() - rt
    total_s = time.time() - t0

    assign, overlap = {}, 0
    for v in sorted(g.nodes.values(), key=lambda x: (-x.t, x.nid)):
        for m in v.members:
            if m in assign:
                overlap += 1
            else:
                assign[m] = v.nid
    n = len(records)
    lab_singleton = [assign.get(i, f"bg:{i}") for i in range(1, n + 1)]
    lab_onebg = [assign.get(i, "bg") for i in range(1, n + 1)]
    stats = {"K_nodes_final": g.K, "mints_total": len(g.mint_log),
             "records_in_nodes": len(assign), "background_records": n - len(assign),
             "multi_membership_extra": overlap, "keys_registered": len(g.keys),
             "runtime_s": round(total_s, 1), "repair_s": round(repair_s, 1)}
    return lab_singleton, lab_onebg, stats


# ------------------------- baselines ------------------------- #
def dp_means(X, lam):
    n = X.shape[0]
    C = np.empty_like(X)
    m = 0
    assign = np.empty(n, dtype=np.int64)
    for i in range(n):
        x = X[i]
        if m == 0:
            C[0] = x
            assign[i] = 0
            m = 1
            continue
        d2 = 2.0 - 2.0 * (C[:m] @ x)
        j = int(np.argmin(d2))
        if d2[j] > lam:
            C[m] = x
            assign[i] = m
            m += 1
        else:
            assign[i] = j
    return assign.tolist()


def as_singletons(labels):
    out, nxt = [], 0
    for l in labels:
        if l == -1:
            out.append(f"noise:{nxt}")
            nxt += 1
        else:
            out.append(int(l))
    return out


def sweep(name, fit, grid, subjects, sub2mentions, tec, tce):
    rows = {}
    for p in grid:
        t0 = time.time()
        try:
            labels = fit(p)
            r = score(subjects, labels, sub2mentions, tec, tce)
            r["fit_s"] = round(time.time() - t0, 1)
        except Exception as ex:
            r = {"error": f"{type(ex).__name__}: {ex}"}
        rows[str(p)] = r
        print(f"[{name}] param={p} -> " + json.dumps(
            {k: r.get(k) for k in ("macro_f1", "micro_f1", "pair_f1", "system_clusters", "error")}),
            flush=True)
    ok = {p: r for p, r in rows.items() if "error" not in r}
    if not ok:
        return {"sweep": rows, "oracle_param": None}
    best = max(ok, key=lambda p: (ok[p]["macro_f1"] + ok[p]["micro_f1"] + ok[p]["pair_f1"]) / 3)
    return {"sweep": rows, "oracle_param": best, "oracle": ok[best],
            "note": "oracle: parameter chosen on test labels"}


# ------------------------- main ------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="e10_reverb45k.json")
    ap.add_argument("--repair-every", type=int, default=200)
    ap.add_argument("--skip-escrow", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    triples = load_triples()
    subjects, records, texts, triples = build_records(triples, args.limit)
    tec, tce, sub2mentions = build_gold(triples)
    gold_singletons = sum(1 for v in tce.values() if len(v) == 1)
    print(f"[data] triples={len(triples)} records={len(records)} gold_clusters={len(tce)} "
          f"gold_singletons={gold_singletons}", flush=True)

    report = {
        "benchmark": "REVERB45K test, CESI protocol (NP canonicalization)",
        "data": {"triples": len(triples), "records_unique_subjects": len(records),
                 "gold_clusters": len(tce), "gold_singletons": gold_singletons,
                 "limit": args.limit},
        "declared_choices": DECLARED_CHOICES,
        "deviations": list(DEVIATIONS),
        "cesi_published_context": {
            "macro_f1": 0.627, "micro_f1": 0.844, "pair_f1": 0.819,
            "note": "CESI (WWW'18) on ReVerb45k test; consumes side information "
                    "(entity linking, PPDB, WordNet, AMIE, KBP) that no method here sees"},
    }

    # ---- ours ---- #
    if not args.skip_escrow:
        print("[escrow] start", flush=True)
        lab1, lab2, stats = run_escrow(records, args.repair_every)
        esc = score(subjects, lab1, sub2mentions, tec, tce)
        esc_bg = score(subjects, lab2, sub2mentions, tec, tce)
        report["escrow"] = {"primary_background_as_singletons": esc,
                            "secondary_background_as_one_cluster": esc_bg,
                            "engine": stats,
                            "K_found_nodes": stats["K_nodes_final"],
                            "K_true_gold_clusters": len(tce)}
        print("[escrow] " + json.dumps(esc), flush=True)

    if args.skip_baselines:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, args.out)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print("[done] written " + out, flush=True)
        return

    # ---- embeddings ---- #
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = np.asarray(model.encode(texts, batch_size=256, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float64)
    print(f"[embed] {X.shape} in {time.time()-t0:.0f}s", flush=True)

    # ---- kmeans oracle-K ---- #
    K_true = min(len(tce), len(records))
    t0 = time.time()
    km = KMeans(n_clusters=K_true, n_init=1, random_state=0).fit_predict(X)
    r = score(subjects, km.tolist(), sub2mentions, tec, tce)
    r["fit_s"] = round(time.time() - t0, 1)
    report["kmeans_oracleK"] = {"K_given": K_true, "result": r,
                                "note": "oracle: given the true gold cluster count"}
    print("[kmeans] " + json.dumps(r), flush=True)

    # ---- swept baselines ---- #
    report["hdbscan"] = sweep(
        "hdbscan", lambda p: as_singletons(HDBSCAN(min_cluster_size=p).fit_predict(X)),
        [3, 5, 10, 20, 50], subjects, sub2mentions, tec, tce)
    report["agglo_avg_cosine"] = sweep(
        "agglo", lambda p: AgglomerativeClustering(
            n_clusters=None, distance_threshold=p, linkage="average",
            metric="cosine").fit_predict(X).tolist(),
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7], subjects, sub2mentions, tec, tce)
    report["dp_means"] = sweep(
        "dpmeans", lambda p: dp_means(X, p),
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], subjects, sub2mentions, tec, tce)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, args.out)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("[done] written " + out, flush=True)


if __name__ == "__main__":
    main()
