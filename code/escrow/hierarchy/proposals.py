"""Label-free search proposals. Complete description lengths alone decide acceptance."""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

from .coding import kt_bits


def useful_keys(records, members, keys):
    """Suggest keys whose node block is shorter than the transmitted background model."""
    useful = set()
    for key in keys:
        total = Counter(r.get(key) for r in records)
        selected = Counter(records[r].get(key) for r in members)
        d = len(total)
        background = -sum(c * math.log2((total[v] + .5) / (len(records) + d / 2))
                          for v, c in selected.items())
        if kt_bits(tuple(selected.get(v, 0) for v in total)) < background:
            useful.add(key)
    return useful


def pooled_keys(state, children):
    keys = state.keys
    result = set()
    for key in keys:
        vocab = sorted({r[key] for r in state.records if key in r})
        counts = [Counter(state.records[r][key] for r in state.nodes[c].members
                          if key in state.records[r]) for c in children]
        pooled = sum(counts, Counter())
        if kt_bits(tuple(pooled[v] for v in vocab)) < sum(
                kt_bits(tuple(c[v] for v in vocab)) for c in counts):
            result.add(key)
    return result


def grow(records, members, universe, keys):
    """One categorical likelihood assignment step, used only to suggest a membership."""
    if not members or members == universe:
        return set(members)
    n, t = len(universe), len(members)
    inside, outside, alphabets = {}, {}, {}
    for key in keys:
        all_counts = Counter(records[r].get(key) for r in universe)
        inside[key] = Counter(records[r].get(key) for r in members)
        outside[key] = all_counts - inside[key]
        alphabets[key] = len(all_counts)
    result = set()
    for r in sorted(universe):
        odds = math.log((t + .5) / (n - t + .5))
        for key in keys:
            value, d = records[r].get(key), alphabets[key]
            odds += math.log((inside[key][value] + .5) / (t + d / 2))
            odds -= math.log((outside[key][value] + .5) / (n - t + d / 2))
        if odds > 0:
            result.add(r)
    return result


def candidates(state, allow_links=True, fixed_ancestry=False):
    nodes, records, keys = state.nodes, state.records, state.keys
    order = state.topological()
    has_children = {p for node in nodes.values() for p in node.parents}
    # Structural corrections are available at every prefix repair.
    for child in order:
        if not fixed_ancestry or child not in has_children:
            yield "delete", dict(node=child)
        v = nodes[child]
        if allow_links and not fixed_ancestry:
            for parent in sorted(v.parents):
                yield "remove_link", dict(child=child, parent=parent)
            eligible = [p for p in order if p != child and v.members <= nodes[p].members
                        and child not in state.ancestors(p)]
            for p in eligible:
                if p not in v.parents:
                    yield "add_link", dict(child=child, parent=p)
                    yield "reparent", dict(child=child, parents=[p])
            for parents in itertools.combinations(eligible, 2):
                if set(parents) != v.parents:
                    yield "reparent", dict(child=child, parents=list(parents))
        for support in (set(keys), set()):
            if support != v.support:
                yield "support", dict(node=child, support=sorted(support), route=True)
        for key in sorted(v.support):
            yield "ownership", dict(assignments=[(r, key, 0) for r in sorted(v.members)])
            yield "ownership", dict(assignments=[(r, key, child) for r in sorted(v.members)])
        universe = state.eligible_rows(child)
        grown = grow(records, v.members, universe, keys)
        if grown and grown != v.members:
            yield "membership", dict(node=child, members=sorted(grown), route=True)
    for a, b in itertools.combinations(order, 2):
        if not fixed_ancestry or (a not in has_children and b not in has_children
                                  and nodes[a].parents == nodes[b].parents):
            yield "merge", dict(nodes=[a, b])
        if allow_links and not fixed_ancestry and a not in state.ancestors(b) and b not in state.ancestors(a):
            common = nodes[a].parents & nodes[b].parents
            yield "introduce_parent", dict(children=[a, b], support=[], parents=sorted(common))
            shared = nodes[a].support & nodes[b].support
            if shared:
                yield "introduce_parent", dict(children=[a, b], support=sorted(shared),
                                                 parents=sorted(common), route=False)
            pooled = pooled_keys(state, [a, b])
            if pooled:
                yield "introduce_parent", dict(children=[a, b], support=sorted(pooled),
                    parents=sorted(common), route=True, inherit_support=True)

    if allow_links and not fixed_ancestry:
        shared_modes = defaultdict(set)
        for child in order:
            for key in keys:
                counts = Counter(records[r][key] for r in nodes[child].members if key in records[r])
                if counts:
                    mode = min(counts, key=lambda v: (-counts[v], v))
                    shared_modes[key, mode].add(child)
        for children in shared_modes.values():
            if len(children) <= 2 or any(a in state.ancestors(b) for a in children for b in children):
                continue
            pooled = pooled_keys(state, children)
            if pooled:
                yield "introduce_parent", dict(children=sorted(children), support=sorted(pooled),
                                                route=True, inherit_support=True)

    # Presence and value carriers are constructed without access to evaluation labels.
    presence, values, signatures = defaultdict(set), defaultdict(set), defaultdict(set)
    for r, record in enumerate(records):
        signatures[tuple(sorted(record))].add(r)
        for key, value in record.items():
            presence[key].add(r)
            values[key, value].add(r)
    seeds = [set(range(len(records)))] + [signatures[s] for s in sorted(signatures)]
    seeds += [presence[k] for k in sorted(presence)]
    seeds += [values[kv] for kv in sorted(values)]
    scopes = [(None, set(range(len(records))))]
    if allow_links:
        scopes += [(nid, nodes[nid].members) for nid in order]
    # Interleave scopes, so existing leaves get proposals before the root uses the budget.
    for seed in seeds:
        for parent, universe in scopes:
            base = seed & universe
            if not base:
                continue
            grown = grow(records, base, universe, keys)
            for members in (base, grown):
                if not members or (parent is not None and members == universe):
                    continue
                present_keys = {k for r in members for k in records[r]}
                useful = useful_keys(records, members, keys)
                for support in (set(keys), present_keys, useful):
                    if not support:
                        continue
                    if any(v.members == members and v.support == support for v in nodes.values()):
                        continue
                    args = dict(members=sorted(members), support=sorted(support))
                    if parent is not None:
                        args["parent"] = parent
                    yield "refine" if parent is not None else "create", args
