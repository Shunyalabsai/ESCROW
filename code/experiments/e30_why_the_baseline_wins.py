"""E30: why an untuned baseline beats this method on the encyclopedia stream, and one fix refuted.

THE FACT TO EXPLAIN. On the matched representation (E21) k-means with a silhouette-chosen k reaches
0.9898 on the 320 encyclopedia records, without ever seeing a label, and returns the same answer on
every arrival order. A single run of this method averages 0.8691 over sixty orders and three
restarts reach 0.9284 (E28). On the two synthetic streams the two are level at 1.0. A paper that
reports that and does not explain it is reporting a defeat, not a result.

WHAT THIS MEASURES.

  1. Where the baseline's accuracy comes from. The multi-hot has two kinds of column, one per
     observed (key, value) pair and one per key. Running k-means on each half separately says which
     half carries the signal.

  2. Why this method cannot use the same signal. A candidate here is named by a seed pair, one
     (key, value) whose value the graph failed to explain, and it accrues only on later records
     carrying that same pair. So the rate at which values repeat is the rate at which the candidate
     generator gets anything to work with. That rate is measured per stream and put beside the
     method's accuracy on that stream.

  3. The obvious fix, and whether it works. If the signal is key presence and the generator can only
     seed on values, then encoding each key's presence as a facet of its own should hand the
     generator the signal. It is a three-line transform and it is tested here rather than assumed.

THE PREDICTION, STATED BEFORE THE RUN. If the diagnosis is right, accuracy should track how often a
(key, value) pair recurs, and the presence-facet transform should lift the encyclopedia stream toward
the baseline. The first held. The second did not, and the way it failed is the useful part: the
engine already prices key presence through its absence code, so encoding presence as a facet makes
every record pay for the same fact twice. The description gets longer, the node count explodes and
accuracy halves. What the diagnosis actually licenses is a change to the candidate generator, letting
a key seed a candidate on the absence evidence the engine already computes, with no new facet. That
is not implemented here and is stated as what it is, untested.
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
from sklearn.preprocessing import normalize                             # noqa: E402

from escrow.protocol import run_stream, describe as protocol_describe   # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402
from experiments.e4_baseline_army import (wikipedia, planted8, two_group,   # noqa: E402
                                          _ari)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e30_why_the_baseline_wins.json")

ORDERS = 10


def streams():
    return [("wikipedia", lambda s: wikipedia(RESULTS, s), 3),
            ("planted8", lambda s: planted8(seed=s), 8),
            ("two_group", lambda s: two_group(seed=s), 2)]


def halves(recs, truth, true_k):
    """k-means on the key columns alone, the (key, value) columns alone, and both."""
    kv = sorted({f"{k}={v}" for r in recs for k, v in r.items()})
    ks = sorted({k for r in recs for k in r})
    ix = {c: i for i, c in enumerate(kv + ks)}
    X = np.zeros((len(recs), len(ix)), dtype=np.float32)
    for i, r in enumerate(recs):
        for k, v in r.items():
            X[i, ix[f"{k}={v}"]] = 1.0
            X[i, ix[k]] = 1.0
    out = {}
    for name, M in (("keys_only", X[:, len(kv):]),
                    ("key_value_only", X[:, :len(kv)]),
                    ("both", X)):
        lab = KMeans(true_k, n_init=10, random_state=0).fit_predict(normalize(M))
        out[name] = round(_ari(truth, list(lab)), 4)
    return out


def repetition(recs):
    kv = Counter((k, v) for r in recs for k, v in r.items())
    keys = Counter(k for r in recs for k in r)
    return {"distinct_keys": len(keys),
            "distinct_key_value_pairs": len(kv),
            "fraction_of_pairs_seen_more_than_once": round(sum(1 for c in kv.values() if c > 1) / len(kv), 4),
            "mean_occurrences_of_a_pair": round(sum(kv.values()) / len(kv), 2),
            "mean_occurrences_of_a_key": round(sum(keys.values()) / len(keys), 1)}


def escrow(recs, truth):
    g, b = run_stream(recs)
    lab = [-1] * len(recs)
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    return _ari(truth, lab), g.K, b.total()


def with_presence_facets(recs):
    out = []
    for r in recs:
        d = dict(r)
        for k in r:
            d[f"has:{k}"] = "1"
        out.append(d)
    return out


def main():
    report = {"experiment": "E30 why the baseline wins",
              "protocol": protocol_describe(),
              "orders": ORDERS, "streams": {}}

    for name, load, true_k in streams():
        recs0, truth0 = load(0)
        cell = {"where_the_baseline_signal_is": halves(recs0, truth0, true_k),
                "value_repetition": repetition(recs0)}
        a, ap = [], []
        for s in range(ORDERS):
            recs, truth = load(s)
            r1 = escrow(recs, truth)
            r2 = escrow(with_presence_facets(recs), truth)
            a.append(r1[0]); ap.append(r2[0])
            if s == 0:
                cell["shipped_example"] = {"ARI": round(r1[0], 4), "K": r1[1], "bits": round(r1[2], 1)}
                cell["presence_facets_example"] = {"ARI": round(r2[0], 4), "K": r2[1],
                                                   "bits": round(r2[2], 1)}
        cell["escrow_shipped"] = {"mean_ARI": round(st.mean(a), 4), "sd": round(st.pstdev(a), 4)}
        cell["escrow_with_presence_facets"] = {"mean_ARI": round(st.mean(ap), 4),
                                               "sd": round(st.pstdev(ap), 4)}
        report["streams"][name] = cell
        print(f"{name:10s} pairs repeating {cell['value_repetition']['fraction_of_pairs_seen_more_than_once']:.1%}"
              f"  ESCROW {cell['escrow_shipped']['mean_ARI']:.4f}"
              f"  with presence facets {cell['escrow_with_presence_facets']['mean_ARI']:.4f}"
              f"  baseline keys-only {cell['where_the_baseline_signal_is']['keys_only']:.4f}", flush=True)

    rep = [report["streams"][n]["value_repetition"]["fraction_of_pairs_seen_more_than_once"]
           for n, _, _ in streams()]
    acc = [report["streams"][n]["escrow_shipped"]["mean_ARI"] for n, _, _ in streams()]
    report["reading"] = {
        "the_baseline_signal": ("on the encyclopedia stream the key columns alone give the baseline "
                                "everything and the value columns give it almost nothing"),
        "why_this_method_cannot_use_it": ("a candidate is named by a (key, value) seed and accrues "
                                          "only on records carrying that pair, so accuracy tracks how "
                                          "often a pair recurs"),
        "repetition_by_stream": dict(zip([n for n, _, _ in streams()], rep)),
        "accuracy_by_stream": dict(zip([n for n, _, _ in streams()], acc)),
        "the_refuted_fix": ("encoding key presence as a facet of its own makes each record pay twice "
                            "for a fact the absence code already prices: the description lengthens, "
                            "the node count explodes and accuracy falls"),
        "what_the_diagnosis_licenses": ("letting a key seed a candidate on the absence evidence the "
                                        "engine already computes, with no new facet. Not implemented "
                                        "here and not claimed"),
    }
    print("\n" + json.dumps(report["reading"], indent=2)[:600])
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
