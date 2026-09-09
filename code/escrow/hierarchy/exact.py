"""Finite exhaustive audit for at most four rows, three nodes and two binary keys.

Only a valid nonnegative lower bound prunes descriptions. A resource limit marks
the answer incomplete, never exact. This module is not used for discovery.
"""
from __future__ import annotations

import itertools

from .coding import Description, binary_bits, integer_bits
from .model import HierarchyState, HierarchyNode


def subsets(items, nonempty=False):
    items = list(items)
    for mask in range(int(nonempty), 1 << len(items)):
        yield {item for i, item in enumerate(items) if mask & (1 << i)}


def minimum_description(records, max_nodes=3, state_budget=None, description_cls=Description):
    if len(records) > 4 or max_nodes > 3:
        raise ValueError("exact audit is restricted to four records and three nodes")
    empty = HierarchyState()
    for record in records:
        empty.append(record)
    if len(empty.keys) > 2 or any(len({r[k] for r in records if k in r}) > 2 for k in empty.keys):
        raise ValueError("exact audit supports two keys and two values per key")
    objective = description_cls(records)
    best, best_terms = empty, objective.score(empty)
    evaluated, pruned, complete = 1, 0, True
    keys, n = empty.keys, len(records)

    class BudgetReached(Exception):
        pass

    def evaluate_graph(state, count):
        nonlocal best, best_terms, evaluated, pruned
        # Graph and membership terms are exact; all remaining terms are nonnegative.
        prior = integer_bits(count) + binary_bits(sum(len(v.parents) for v in state.nodes.values()),
                                                   count * (count - 1) // 2)
        for v in state.nodes.values():
            prior += binary_bits(len(v.support), len(keys))
            prior += binary_bits(len(v.members), len(state.eligible_rows(v.nid)))
        if hasattr(objective, "model_terms"):
            prior = sum(objective.model_terms(state).values()) - objective.header
        if objective.header + prior > best_terms["total"]:
            pruned += 1
            return
        slots = [(r, k) for r in range(n) for k in keys]
        options = [state.eligible_owners(r, k) for r, k in slots]
        for owners in itertools.product(*options):
            if state_budget is not None and evaluated >= state_budget:
                raise BudgetReached()
            state.owners = [{} for _ in range(n)]
            for (r, k), owner in zip(slots, owners):
                if owner:
                    state.owners[r][k] = owner
            terms = objective.score(state)
            evaluated += 1
            if terms["total"] < best_terms["total"]:
                best, best_terms = state.clone(), terms

    def graphs(state, count):
        nid = len(state.nodes) + 1
        if nid > count:
            evaluate_graph(state, count)
            return
        for parents in subsets(range(1, nid)):
            if any((parents - {p}) & state.ancestors(p) for p in parents):
                continue
            eligible = set(range(n)) if not parents else set.intersection(
                *(state.nodes[p].members for p in parents))
            for members in subsets(sorted(eligible), nonempty=True):
                for support in subsets(keys):
                    state.nodes[nid] = HierarchyNode(nid, members, support, parents)
                    graphs(state, count)
        state.nodes.pop(nid, None)

    try:
        for count in range(1, max_nodes + 1):
            # Each nonempty membership column costs at least one bit.
            lower = integer_bits(count) + count * binary_bits(0, len(keys))
            # A nonempty joint family sends at least one pattern (gamma(1)=3)
            # and one pattern bit (at least one bit). Remaining costs are nonnegative.
            lower += 4 if getattr(objective, "joint_membership", False) else count
            lower += binary_bits(0, count * (count - 1) // 2)
            if objective.header + lower > best_terms["total"] or not n:
                pruned += 1
                continue
            work = empty.clone()
            work.next_id = count + 1
            graphs(work, count)
    except BudgetReached:
        complete = False
    return dict(state=best, description=best_terms, evaluated=evaluated,
                branches_pruned_by_lower_bound=pruned, complete=complete, max_nodes=max_nodes)
