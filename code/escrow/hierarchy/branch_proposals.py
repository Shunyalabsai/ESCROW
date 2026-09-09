"""Additional EHR05 proposals relative to established parent distributions.

These scores only nominate supports. The full independent description decides every
acceptance. No benchmark metadata enters this module.
"""
import math
import itertools
from collections import Counter, deque

from .coding import binary_bits, kt_bits
from .proposals import candidates as historical_candidates, grow, pooled_keys
from .moves import propose


def conditional_keys(state, members, parent_members):
    selected = set()
    for key in state.keys:
        vocabulary = sorted({record[key] for record in state.records if key in record})
        local = Counter(state.records[r][key] for r in members if key in state.records[r])
        parent = Counter(state.records[r][key] for r in parent_members if key in state.records[r])
        pubs, parent_pubs = sum(local.values()), sum(parent.values())
        probability = (parent_pubs + .5) / (len(parent_members) + 1)
        contextual = -pubs * math.log2(probability) - (len(members) - pubs) * math.log2(1 - probability)
        contextual -= sum(count * math.log2((parent[value] + .5) /
            (parent_pubs + len(vocabulary) / 2)) for value, count in local.items())
        separate = binary_bits(pubs, len(members)) + kt_bits(tuple(local[value] for value in vocabulary))
        if separate < contextual:
            selected.add(key)
    return selected


def local_candidates(state, allow_links=True, fixed_ancestry=False):
    # Joint refresh avoids making new arrivals pay for many individually evaluated
    # repairs before the graph can again explain the whole current prefix.
    refreshed, updates = state, []
    for nid in state.topological():
        members = grow(refreshed.records, refreshed.nodes[nid].members,
                       refreshed.eligible_rows(nid), refreshed.keys)
        if members and members != refreshed.nodes[nid].members:
            arguments = dict(node=nid, members=sorted(members), route=True)
            refreshed = propose(refreshed, "membership", **arguments)
            updates.append(("membership", arguments))
    if len(updates) > 1:
        yield "compound", dict(moves=updates)
    for nid in state.topological():
        used = {key for r in state.nodes[nid].members for key, owner in state.owners[r].items() if owner == nid}
        if used != state.nodes[nid].support:
            yield "support", dict(node=nid, support=sorted(used), route=False)
    # A membership found globally can also be proposed under an established parent.
    # Its support is then nominated relative to that parent, not only to background.
    seen, pairs = set(), set()
    for operation, arguments in historical_candidates(state, allow_links, fixed_ancestry):
        yield operation, arguments
        if operation == "introduce_parent" and arguments.get("route"):
            children = set(arguments["children"])
            members = set.union(*(state.nodes[c].members for c in children))
            pooled = set(arguments["support"])
            for parent in state.topological():
                if parent in children or state.nodes[parent].members != members or any(
                        c in state.ancestors(parent) for c in children):
                    continue
                # Evaluate reuse alongside insertion, preserving an established
                # identity when it can already explain the proposed shared block.
                moves = [("support", dict(node=parent,
                    support=sorted(state.nodes[parent].support | pooled), route=True))]
                for child in sorted(children):
                    if parent not in state.nodes[child].parents:
                        moves.append(("add_link", dict(child=child, parent=parent)))
                    if arguments.get("inherit_support"):
                        moves.append(("support", dict(node=child,
                            support=sorted(state.nodes[child].support - pooled), route=False)))
                yield "compound", dict(moves=moves)
        if not allow_links or operation not in ("create", "refine"):
            continue
        members = set(arguments["members"])
        for parent in state.topological():
            if not members < state.nodes[parent].members:
                continue
            identity = (parent, frozenset(members))
            if identity in seen:
                continue
            seen.add(identity)
            support = conditional_keys(state, members, state.nodes[parent].members)
            if support:
                yield "refine", dict(parent=parent, members=sorted(members), support=sorted(support))
                remainder = state.nodes[parent].members - members
                pair = (parent, frozenset((frozenset(members), frozenset(remainder))))
                if pair in pairs:
                    continue
                pairs.add(pair)
                other_support = conditional_keys(state, remainder, state.nodes[parent].members)
                if other_support:
                    moves = [("refine", dict(parent=parent, members=sorted(members), support=sorted(support))),
                             ("refine", dict(parent=parent, members=sorted(remainder), support=sorted(other_support)))]
                    # Both new children take these slots, so the parent may release
                    # their shared support. The full compound must still pay.
                    retained = state.nodes[parent].support - (support & other_support)
                    if retained != state.nodes[parent].support:
                        moves.append(("support", dict(node=parent, support=sorted(retained), route=False)))
                    yield "compound", dict(moves=moves)


def parent_arrangements(state, allow_links=True, fixed_ancestry=False):
    """Joint shared parents and withdrawals, without accepting intermediate debts."""
    if not allow_links or fixed_ancestry:
        return
    order = state.topological()
    proposals = []
    for a, b in itertools.combinations(order, 2):
        if a in state.ancestors(b) or b in state.ancestors(a):
            continue
        support = pooled_keys(state, [a, b])
        if support:
            proposals.append(dict(children=[a, b], support=sorted(support), route=True,
                inherit_support=True, parents=sorted(state.nodes[a].parents & state.nodes[b].parents)))
    old_edges = [(child, parent) for child in order for parent in sorted(state.nodes[child].parents)]
    for left, right in itertools.combinations(proposals, 2):
        # Distinct supports can describe different aspects of their shared child.
        if not set(left["children"]) & set(right["children"]) or set(left["support"]) & set(right["support"]):
            continue
        moves = [("introduce_parent", left), ("introduce_parent", right)]
        try:
            paired = propose(state, "compound", moves=moves)
        except (ValueError, KeyError):
            continue
        remaining = [(c, p) for c, p in old_edges if p in paired.nodes[c].parents]
        for size in range(len(remaining) + 1):
            for removed in itertools.combinations(remaining, size):
                yield "compound", dict(moves=moves + [
                    ("remove_link", dict(child=c, parent=p)) for c, p in removed])


def candidates(state, allow_links=True, fixed_ancestry=False):
    # Interleave proposal families so one combinatorial source cannot take the
    # entire evaluated-state budget before ordinary repair has been considered.
    sources = deque(iter(source(state, allow_links, fixed_ancestry))
                    for source in (local_candidates, parent_arrangements))
    while sources:
        source = sources.popleft()
        candidate = next(source, None)
        if candidate is not None:
            yield candidate
            sources.append(source)
