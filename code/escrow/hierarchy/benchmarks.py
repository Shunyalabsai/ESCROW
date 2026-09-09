"""Constructed evaluation records. Truth is a separate object, never learner input."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Benchmark:
    name: str
    records: list
    members: dict
    parents: dict
    metadata: dict


def tree(depth, n=400, seed=0, signal=1., missingness=0., background=0.,
         uneven=False, children_first=False):
    if not 0 <= depth <= 4:
        raise ValueError("fixture depth must be zero through four")
    rng = random.Random(seed)
    records, members, parents = [], {}, {}
    levels = max(depth, 1)
    for r in range(n):
        path = "".join(str(rng.randrange(2)) for _ in range(levels))
        actual_depth = 1 if uneven and path[0] == "0" else depth
        is_background = depth == 0 or rng.random() < background
        record = {}
        for level in range(1, levels + 1):
            own = int(path[:level], 2)
            for repeat in range(3):
                key = "k" + str(3 * (level - 1) + repeat)
                if rng.random() < missingness:
                    continue
                if children_first and level < levels and r < n // 2:
                    continue
                structured = not is_background and level <= actual_depth and rng.random() < signal
                value = own if structured else rng.randrange(2 ** level)
                record[key] = "v" + str(2 * value + rng.randrange(2))
        records.append(record)
        if not is_background:
            for level in range(1, actual_depth + 1):
                node = path[:level]
                members.setdefault(node, set()).add(r)
                parents[node] = {node[:-1]} if level > 1 else set()
    return Benchmark("tree_depth_" + str(depth), records, members, parents,
        dict(depth=depth, seed=seed, signal=signal, missingness=missingness,
             background=background, uneven=uneven, children_first=children_first,
             sampling="independent uniform draws within a two-symbol group vocabulary",
             unsupported_leaf_refinement_is_failure=True))


def intersecting(n=400, seed=0):
    rng = random.Random(seed)
    records, members = [], {k: set() for k in ("a0", "a1", "b0", "b1", "joint")}
    for r in range(n):
        a, b = rng.randrange(2), rng.randrange(2)
        record = {}
        for offset, bit in ((0, a), (3, b)):
            for repeat in range(3):
                record["k" + str(offset + repeat)] = "v" + str(2 * bit + rng.randrange(2))
        for repeat in range(3):
            value = rng.randrange(2) if a == b == 0 else rng.randrange(2, 10)
            record["k" + str(6 + repeat)] = "v" + str(value)
        records.append(record)
        members["a" + str(a)].add(r)
        members["b" + str(b)].add(r)
        if a == b == 0:
            members["joint"].add(r)
    parents = {name: set() for name in members}
    parents["joint"] = {"a0", "b0"}
    return Benchmark("intersecting_multiple_parents", records, members, parents,
                     dict(seed=seed, joint_specialisation=True, ordinary_overlap_also_present=True,
                          reference_groups_exhaustive=False,
                          reference_audit="Non-joint records also share a disjoint vocabulary on k6, k7 and k8. Their complement group is observable but omitted from the five reference labels. Evaluate recovery of known groups and links; do not interpret unmatched groups alone as unsupported."))


def ambiguous(n=400, seed=0, later_identifiable=False):
    rng = random.Random(seed)
    records, members = [], {k: set() for k in ("a", "b", "child")}
    for r in range(n):
        a = rng.randrange(2) == 0
        b = a if not later_identifiable or r < n // 2 else rng.randrange(2) == 0
        child = a and rng.randrange(2) == 0
        record = {}
        for offset, bit in ((0, a), (3, b), (6, child)):
            for repeat in range(3):
                record["k" + str(offset + repeat)] = "v" + str(2 * int(bit) + rng.randrange(2))
        records.append(record)
        for name, active in (("a", a), ("b", b), ("child", child)):
            if active:
                members[name].add(r)
    parents = {name: set() for name in members}
    if later_identifiable:
        parents["child"] = {"a"}
    return Benchmark("parentage_resolves" if later_identifiable else "indistinguishable_parentage",
        records, members, parents, dict(seed=seed, ambiguous_until=n // 2 if later_identifiable else n,
            ambiguity="Identical parent membership before the reveal; the two groups may also merge.",
            admissible_unresolved_parent_sets=[[], ["a"], ["b"], ["a", "b"]],
            edge_truth_complete=later_identifiable))


def suite(n=400, seed=0):
    out = [tree(depth, n=n, seed=seed) for depth in range(5)]
    out += [tree(4, n=n, seed=seed, uneven=True),
            tree(3, n=n, seed=seed, children_first=True), intersecting(n, seed),
            tree(3, n=n, seed=seed, missingness=.2),
            tree(3, n=n, seed=seed, background=.2),
            ambiguous(n, seed), ambiguous(n, seed, later_identifiable=True)]
    for i, fixture in enumerate(out):
        fixture.name = f"{i:02d}_" + fixture.name
    return out
