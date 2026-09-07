"""E37: the multi-membership question on real records, against baselines that can answer it.

WHY THIS EXISTS. Every cover number in this project came from one synthetic fixture, and E33 showed
that fixture hands the answer to any cover-capable method because it writes the group identity into
the key NAMES: the key matrix satisfies X = Z B exactly, so the truth is a noiseless function of
which keys a record carries. A real benchmark cannot be rigged that way.

THE BENCHMARK. People from Wikidata, 3,362 of them. A record is a person's properties, one
(property, value) pair per property, with times coarsened to the year. The label set is that person's
occupations, restricted to the twelve most frequent in a 20,000 row sample so that the groups are
chosen by frequency rather than by us. 39.2 percent of the records carry more than one occupation and
the mean is 1.74, so the truth is a genuine cover. Occupation itself is removed from every record,
along with field of work, position held, professorship and affiliation, so nothing in the input names
the answer. Licence CC0.

IN SCOPE BY OUR OWN CONDITION, which matters because E31 says a group is unreachable when its
characteristic pairs are too rare. The most frequent characteristic pair of each of the twelve labels
recurs between 282 and 1,033 times against a t* of at most 24, so all twelve clear it. If the method
loses here it does not get to plead scope.

WHO IT IS RUN AGAINST. Methods that can emit overlapping memberships, which partition methods cannot:
non-negative matrix factorisation and latent Dirichlet allocation, each converted to a cover by
keeping every component within a fraction of the row maximum. They are run in the paper's three
standing conditions, ORACLE with components and cut chosen on the test labels, DEFAULT with a
label-free rule fixed in advance, and k-means at the true label count as a partition reference.
ESCROW is run under both repair protocols, since E35 found the cadence decides the granularity.

FALSIFIER, STATED BEFORE THE RUN. A cover-capable baseline matching ESCROW under the DEFAULT
condition. Under ORACLE the baselines have three choices made on the test labels and ESCROW has none,
so an ORACLE loss is expected and is not the test.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np                                                      # noqa: E402
from sklearn.cluster import KMeans                                      # noqa: E402
from sklearn.decomposition import NMF, LatentDirichletAllocation        # noqa: E402

from escrow.batch import BatchObjective                                 # noqa: E402
from escrow.protocol import new_run                                     # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e25_multi_membership import (omega_index, membership_prf,   # noqa: E402
                                              _shared_counts)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
DATA = os.path.join(RESULTS, "wikidata_cover", "wikidata_people.json")
OUT = os.path.join(RESULTS, "e37_real_cover.json")

KS = (6, 9, 12, 16, 24)
CUTS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
DEFAULT_CUT = 0.5
MIN_PAIR = 2          # a (key, value) column is kept when it occurs at least twice, declared


def load():
    d = json.load(open(DATA, encoding="utf-8"))
    recs, labels = [], []
    for v in d.values():
        recs.append(v["record"])
        labels.append(set(v["labels"]))
    return recs, labels


def truth_matrix(labels):
    lab = sorted({x for s in labels for x in s})
    ix = {l: i for i, l in enumerate(lab)}
    M = np.zeros((len(labels), len(lab)), dtype=bool)
    for i, s in enumerate(labels):
        for l in s:
            M[i, ix[l]] = True
    return M, lab


def features(recs, kind="key_plus_keyvalue"):
    """The same information the engine reads, as indicators. Columns for (key, value) pairs seen
    only once carry no information any method can use and would make the matrix five times larger,
    so they are dropped by a rule declared before the run. Key columns are always kept."""
    kv = Counter((k, v) for r in recs for k, v in r.items())
    pairs = sorted(p for p, c in kv.items() if c >= MIN_PAIR)
    keys = sorted({k for r in recs for k in r})
    cols = {}
    if kind in ("keyvalue", "key_plus_keyvalue"):
        for p in pairs:
            cols[p] = len(cols)
    if kind in ("key", "key_plus_keyvalue"):
        for k in keys:
            cols[k] = len(cols)
    X = np.zeros((len(recs), len(cols)), dtype=np.float32)
    for i, r in enumerate(recs):
        for k, v in r.items():
            j = cols.get((k, v))
            if j is not None:
                X[i, j] = 1.0
            j = cols.get(k)
            if j is not None:
                X[i, j] = 1.0
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def escrow_cover(recs, mode):
    g, b = new_run(BatchObjective)
    for i, r in enumerate(recs):
        g.process(r)
        if mode == "A" and (i + 1) % 100 == 0:
            b.repair()
    b.repair(full=True)
    nids = sorted(g.nodes)
    M = np.zeros((len(recs), len(nids)), dtype=bool)
    for c, nid in enumerate(nids):
        for m in g.nodes[nid].members:
            M[m - 1, c] = True
    return M, g, b


def score(Mt, Mp, T):
    om = omega_index(Mt, Mp, T=T)[0]
    prf = membership_prf(Mt, Mp)
    return {"omega": round(om, 4),
            "membership_f1_one_to_one": round(prf["one_to_one"]["f1"], 4),
            "membership_f1_many_to_one": round(prf["many_to_one"]["f1"], 4),
            "groups_predicted": int(Mp.shape[1])}


def cover_of(W, cut):
    m = W.max(axis=1, keepdims=True)
    m[m <= 0] = 1.0
    M = (W / m) >= cut
    e = ~M.any(axis=1)
    if e.any():
        M[e, W[e].argmax(axis=1)] = True
    return M


def elbow_k(X, ks):
    errs = []
    for k in ks:
        m = NMF(k, init="nndsvda", max_iter=250, random_state=0)
        m.fit(X)
        errs.append(m.reconstruction_err_)
    drops = [errs[i - 1] - errs[i] for i in range(1, len(errs))]
    return ks[int(np.argmax(drops)) + 1]


def main():
    recs, labels = load()
    Mt, lab = truth_matrix(labels)
    T = _shared_counts(Mt)
    n = len(recs)
    per_label = Counter(x for s in labels for x in s)
    print(f"{n} records, {len(lab)} labels, "
          f"{round(st.mean(len(s) for s in labels), 2)} labels per record, "
          f"{round(sum(1 for s in labels if len(s) > 1) / n, 3)} multi-label", flush=True)

    report = {"experiment": "E37 the cover question on real records",
              "benchmark": {"source": "Wikidata people, occupation as the label set", "licence": "CC0",
                            "records": n, "labels": len(lab),
                            "mean_labels_per_record": round(st.mean(len(s) for s in labels), 2),
                            "multi_label_share": round(sum(1 for s in labels if len(s) > 1) / n, 3),
                            "records_per_label": dict(per_label),
                            "label_removed_from_the_record": ["P106", "P101", "P39", "P803",
                                                              "P1416"]},
              "falsifier": "a cover-capable baseline matching ESCROW under the DEFAULT condition",
              "escrow": {}, "baselines": {}}

    for mode, name in (("A", "repair every 100 records, the shipped protocol"),
                       ("B", "all repair deferred to one final pass")):
        M, g, b = escrow_cover(recs, mode)
        r = score(Mt, M, T)
        r["K"] = int(g.K)
        r["bits"] = round(b.total(), 1)
        r["protocol"] = name
        report["escrow"][mode] = r
        print(f"  ESCROW {mode}: K={r['K']} omega {r['omega']} F1 "
              f"{r['membership_f1_one_to_one']} bits {r['bits']:.0f}", flush=True)

    X = features(recs)
    print(f"  matched representation {X.shape}", flush=True)

    grid = {}
    for k in KS:
        for nm, mdl in (("nmf", NMF(k, init="nndsvda", max_iter=300, random_state=0)),
                        ("lda", LatentDirichletAllocation(k, random_state=0, max_iter=20))):
            try:
                W = mdl.fit_transform(np.clip(X, 0, None))
            except Exception as exc:
                print(f"    {nm} k={k} failed {type(exc).__name__}", flush=True)
                continue
            for cut in CUTS:
                grid[f"{nm}|{k}|{cut}"] = score(Mt, cover_of(W, cut), T)
            print(f"    {nm} k={k} done", flush=True)

    for nm in ("nmf", "lda"):
        cells = {c: v for c, v in grid.items() if c.startswith(nm + "|")}
        if not cells:
            continue
        best = max(cells.items(), key=lambda kv: kv[1]["omega"])
        report["baselines"][nm + "_oracle"] = {"cell": best[0], "condition":
                                               "components and cut chosen on the test labels",
                                               **best[1]}
        print(f"  {nm} ORACLE {best[0]}: omega {best[1]['omega']}", flush=True)

    ke = elbow_k(np.clip(X, 0, None), list(KS))
    W = NMF(ke, init="nndsvda", max_iter=300, random_state=0).fit_transform(np.clip(X, 0, None))
    d = score(Mt, cover_of(W, DEFAULT_CUT), T)
    report["baselines"]["nmf_default"] = {"condition": "label-free: components by the "
                                          "reconstruction-error elbow, cut at half the row maximum",
                                          "k_by_elbow": int(ke), "cut": DEFAULT_CUT, **d}
    print(f"  nmf DEFAULT k={ke} cut={DEFAULT_CUT}: omega {d['omega']}", flush=True)

    km = KMeans(len(lab), n_init=10, random_state=0).fit_predict(X)
    P = np.zeros((n, len(set(km))), dtype=bool)
    for i, c in enumerate(sorted(set(km))):
        P[km == c, i] = True
    report["baselines"]["kmeans_true_label_count"] = {
        "condition": "ORACLE partition reference, handed the true number of labels", **score(Mt, P, T)}
    print(f"  kmeans at the true label count: omega "
          f"{report['baselines']['kmeans_true_label_count']['omega']}", flush=True)

    best_esc = max(report["escrow"].values(), key=lambda r: r["omega"])
    dflt = report["baselines"]["nmf_default"]["omega"]
    report["headline"] = {
        "escrow_best_omega": best_esc["omega"],
        "escrow_best_protocol": best_esc["protocol"],
        "cover_capable_default_omega": dflt,
        "cover_capable_oracle_omega": max(
            (v["omega"] for k, v in report["baselines"].items() if k.endswith("_oracle")),
            default=None),
        "falsifier_fired": bool(dflt >= best_esc["omega"]),
        "reading": ("The synthetic cover fixture wrote the answer into the key names. This one "
                    "cannot, and every label clears the seeding condition by a factor of at least "
                    "eleven, so the scope condition does not excuse a loss here."),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
