"""Membership and relationship evaluation. This module alone receives truth."""
from __future__ import annotations

import math
from collections import Counter


def prf(predicted, expected):
    overlap = len(predicted & expected)
    p = overlap / len(predicted) if predicted else float(not expected)
    r = overlap / len(expected) if expected else float(not predicted)
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.)


def ancestors(parents):
    result = set()
    for child in parents:
        todo, seen = list(parents[child]), set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            result.add((p, child))
            todo.extend(parents.get(p, ()))
    return result


def measure(state, truth, truth_parents):
    # Maximum weight one-to-one matching using membership F1, fixed for every edge metric.
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    predicted_ids, true_ids = sorted(state.nodes), sorted(truth)
    weights = np.zeros((len(predicted_ids), len(true_ids)))
    for i, nid in enumerate(predicted_ids):
        for j, tid in enumerate(true_ids):
            weights[i, j] = prf(state.nodes[nid].members, truth[tid])["f1"]
    row, col = linear_sum_assignment(-weights)
    mapping = {predicted_ids[i]: true_ids[j] for i, j in zip(row, col) if weights[i, j] > 0}
    mapped = lambda nid: mapping.get(nid, ("unmatched", nid))
    predicted_membership = {(mapped(nid), r) for nid, v in state.nodes.items() for r in v.members}
    expected_membership = {(tid, r) for tid, members in truth.items() for r in members}
    edges = {(mapped(p), mapped(c)) for c, v in state.nodes.items() for p in v.parents}
    expected_edges = {(p, c) for c, ps in truth_parents.items() for p in ps}
    predicted_parents = {nid: v.parents for nid, v in state.nodes.items()}
    transitive = {(mapped(p), mapped(c)) for p, c in ancestors(predicted_parents)}
    expected_ancestors = ancestors(truth_parents)
    coverage = {r for v in state.nodes.values() for r in v.members}
    expected_coverage = set.union(*truth.values()) if truth else set()
    counts = Counter(r for v in state.nodes.values() for r in v.members)
    true_counts = Counter(r for members in truth.values() for r in members)
    depth = {}
    for nid in state.topological():
        depth[nid] = 1 + max((depth[p] for p in state.nodes[nid].parents), default=0)
    true_depth = {}
    def depth_of(nid):
        if nid not in true_depth:
            true_depth[nid] = 1 + max((depth_of(p) for p in truth_parents.get(nid, ())), default=0)
        return true_depth[nid]
    for tid in truth:
        depth_of(tid)
    unsupported = sum(1 for nid, d in depth.items() if nid not in mapping or d > true_depth[mapping[nid]])
    return dict(membership=prf(predicted_membership, expected_membership),
        direct_links=prf(edges, expected_edges), ancestor_links=prf(transitive, expected_ancestors),
        coverage=prf(coverage, expected_coverage), coverage_fraction=len(coverage) / max(1, len(state.records)),
        overlapping_membership=prf({r for r, c in counts.items() if c > 1},
                                  {r for r, c in true_counts.items() if c > 1}),
        predicted_nodes=len(state.nodes), true_nodes=len(truth),
        predicted_depth=max(depth.values(), default=0), true_depth=max(true_depth.values(), default=0),
        unsupported_depth_nodes=unsupported, node_matching={str(k): v for k, v in mapping.items()},
        unsupported_depth_interpretation="Reference-based unmatched-or-deeper proxy. Incomplete reference groups cannot certify that these nodes lack data support.",
        matching_rule="maximum summed membership F1, one-to-one; unmatched groups remain errors")
