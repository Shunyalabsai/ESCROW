"""EHR05: joint memberships, all-member predictors, and balanced data exposure.

The three switches are explicit diagnostic ablations, transmitted in the format.
The revised reference uses all three. No switch is chosen from evaluation labels.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque

from .coding import Description as PriorDescription, binary_bits, integer_bits, kt_bits


def families(state, order):
    """Equal-parent families in a dependency order, using only the transmitted graph."""
    groups = {}
    for nid in order:
        groups.setdefault(frozenset(state.nodes[nid].parents), []).append(nid)
    done = set()
    while groups:
        ready = [ps for ps in groups if ps <= done]
        if not ready:
            raise ValueError("family dependencies contain a cycle")
        ps = min(ready, key=lambda p: min(order.index(v) for v in groups[p]))
        children = groups.pop(ps)
        eligible = set(range(len(state.records))) if not ps else set.intersection(
            *(state.nodes[p].members for p in ps))
        yield children, sorted(eligible)
        done.update(children)


class BranchDescription(PriorDescription):
    def __init__(self, records, joint_membership=True, member_predictors=True, balanced_records=True):
        super().__init__(records)
        self.joint_membership = joint_membership
        self.member_predictors = member_predictors
        self.balanced_records = balanced_records
        self.header += 8  # The transmitted ablation byte is common to all variants.

    def data_order(self, state, order):
        if not self.balanced_records:
            return list(range(self.n))
        cohorts = defaultdict(list)
        for r in range(self.n):
            pattern = tuple(r in state.nodes[nid].members for nid in order)
            cohorts[pattern].append(r)
        active = deque(iter(cohorts[p]) for p in sorted(cohorts))
        result = []
        while active:
            cohort = active.popleft()
            row = next(cohort, None)
            if row is not None:
                result.append(row)
                active.append(cohort)
        return result

    def model_terms(self, state):
        order = state.topological()
        k = len(order)
        terms = dict(header=float(self.header), structure=float(integer_bits(k)),
                     membership_patterns=0.0, membership=0.0,
                     ownership=0.0, presence=0.0, values=0.0)
        terms["structure"] += binary_bits(sum(len(v.parents) for v in state.nodes.values()), k * (k - 1) // 2)
        terms["structure"] += sum(binary_bits(len(v.support), len(self.keys)) for v in state.nodes.values())
        if self.joint_membership:
            for children, eligible in families(state, order):
                patterns = Counter(tuple(int(r in state.nodes[c].members) for c in children) for r in eligible)
                terms["membership_patterns"] += integer_bits(len(patterns))
                terms["membership_patterns"] += sum(binary_bits(sum(p), len(children)) for p in patterns)
                terms["membership"] += kt_bits(tuple(patterns.values()))
        else:
            terms["membership"] += sum(binary_bits(len(v.members), len(state.eligible_rows(v.nid)))
                                        for v in state.nodes.values())
        return terms

    def score(self, state, validate=True):
        if len(state.records) != self.n or state.records != self.records:
            raise ValueError("description belongs to a different prefix")
        if validate:
            state.validate()
        terms = self.model_terms(state)
        order = state.topological()
        rows = self.data_order(state, order)
        for key in self.keys:
            supporting = [nid for nid in order if key in state.nodes[nid].support]
            # Separate histories from the encoder: dictionaries of categorical counts.
            history_present = {nid: [0, 0] for nid in supporting}
            history_value = {nid: Counter() for nid in supporting}
            routes = {}
            pubs = sum(self.hist[key].values())
            p_background = (pubs + .5) / (self.n + 1)
            d = len(self.vocab[key])
            for r in rows:
                record = state.records[r]
                members = [nid for nid in supporting if r in state.nodes[nid].members]
                eligible = (0,) + tuple(members)
                owner = state.owners[r].get(key, 0)
                routes.setdefault(eligible, Counter())[owner] += 1
                published = int(key in record)
                if owner == 0:
                    terms["presence"] -= math.log2(p_background if published else 1 - p_background)
                    if published:
                        terms["values"] -= math.log2((self.hist[key][record[key]] + .5) / (pubs + d / 2))
                else:
                    pc = history_present[owner]
                    terms["presence"] -= math.log2((pc[published] + .5) / (sum(pc) + 1))
                    if published:
                        counts = history_value[owner]
                        terms["values"] -= math.log2((counts[record[key]] + .5) / (pc[1] + d / 2))
                trained = members if self.member_predictors else ([owner] if owner else [])
                for nid in trained:
                    history_present[nid][published] += 1
                    if published:
                        history_value[nid][record[key]] += 1
            terms["ownership"] += sum(kt_bits(tuple(counts.get(o, 0) for o in options))
                                        for options, counts in routes.items())
        terms["total"] = sum(terms.values())
        return terms


def score(state, **options):
    return BranchDescription(state.records, **options).score(state)
