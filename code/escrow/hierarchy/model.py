"""State and invariants. Record indices are zero based; node 0 is the background."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class HierarchyNode:
    nid: int
    members: set[int] = field(default_factory=set)
    support: set[str] = field(default_factory=set)
    parents: set[int] = field(default_factory=set)


@dataclass
class HierarchyState:
    records: list[dict[str, str]] = field(default_factory=list)
    nodes: dict[int, HierarchyNode] = field(default_factory=dict)
    # Includes absent as well as published slots. Omitted entries mean background.
    owners: list[dict[str, int]] = field(default_factory=list)
    next_id: int = 1

    @property
    def keys(self):
        return sorted({key for record in self.records for key in record})

    def clone(self):
        return HierarchyState(list(self.records), copy.deepcopy(self.nodes),
                              [dict(row) for row in self.owners], self.next_id)

    def append(self, record):
        if not isinstance(record, dict) or any(not isinstance(k, str) or
                not isinstance(v, str) for k, v in record.items()):
            raise TypeError("categorical records must be dictionaries of strings to strings")
        self.records.append(dict(record))
        self.owners.append({})

    def topological(self):
        """Canonical order independent of node identifiers. No data is inferred by the decoder."""
        remaining, out = set(self.nodes), []
        while remaining:
            ready = [nid for nid in remaining if self.nodes[nid].parents <= set(out)]
            if not ready:
                raise ValueError("parent links contain a cycle or a missing parent")
            def key(nid):
                v = self.nodes[nid]
                return (tuple(sorted(v.support)), tuple(sorted(v.members)),
                        tuple(sorted(out.index(p) for p in v.parents)))
            chosen = min(ready, key=key)
            out.append(chosen)
            remaining.remove(chosen)
        return out

    def ancestors(self, nid):
        found, todo = set(), list(self.nodes[nid].parents)
        while todo:
            p = todo.pop()
            if p in found:
                continue
            found.add(p)
            todo.extend(self.nodes[p].parents)
        return found

    def eligible_rows(self, nid):
        parents = self.nodes[nid].parents
        if not parents:
            return set(range(len(self.records)))
        return set.intersection(*(self.nodes[p].members for p in parents))

    def eligible_owners(self, row, key):
        return (0,) + tuple(nid for nid in self.topological()
                            if row in self.nodes[nid].members and key in self.nodes[nid].support)

    def close_ancestors(self):
        for nid in reversed(self.topological()):
            v = self.nodes[nid]
            for p in v.parents:
                self.nodes[p].members.update(v.members)

    def reduce_edges(self):
        self.topological()
        for v in self.nodes.values():
            redundant = set()
            for p in v.parents:
                redundant.update(v.parents & self.ancestors(p))
            v.parents.difference_update(redundant)

    def clean_owners(self):
        for r, row in enumerate(self.owners):
            for key, owner in list(row.items()):
                v = self.nodes.get(owner)
                if owner and (v is None or r not in v.members or key not in v.support):
                    del row[key]

    def validate(self):
        n, keys = len(self.records), set(self.keys)
        if len(self.owners) != n:
            raise ValueError("one ownership row is required per record")
        self.topological()
        if any(nid <= 0 or v.nid != nid for nid, v in self.nodes.items()):
            raise ValueError("node identifiers must be positive and consistent")
        if self.nodes and self.next_id <= max(self.nodes):
            raise ValueError("next_id must exceed every live node identifier")
        for v in self.nodes.values():
            if not v.members or not v.members <= set(range(n)):
                raise ValueError("node membership must be nonempty and inside the prefix")
            if not v.support <= keys:
                raise ValueError("support names an unknown key")
            if not v.members <= self.eligible_rows(v.nid):
                raise ValueError("a child must belong to every established parent")
            if any((v.parents - {p}) & self.ancestors(p) for p in v.parents):
                raise ValueError("store direct parent links without transitive redundancy")
        for r, row in enumerate(self.owners):
            for key, owner in row.items():
                if key not in keys:
                    raise ValueError("ownership names an unknown key")
                if owner:
                    v = self.nodes.get(owner)
                    if v is None or r not in v.members or key not in v.support:
                        raise ValueError("owner is not an eligible member node")
        return self

    def add_node(self, members, support, parents=()):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = HierarchyNode(nid, set(members), set(support), set(parents))
        return nid

    def canonical(self):
        order = self.topological()
        keys = self.keys
        ids = {nid: i + 1 for i, nid in enumerate(order)}
        return {"records": self.records,
                "nodes": [{"members": sorted(self.nodes[nid].members),
                           "support": sorted(self.nodes[nid].support),
                           "parents": sorted(ids[p] for p in self.nodes[nid].parents)}
                          for nid in order],
                "owners": [{k: ids.get(row.get(k, 0), 0) for k in keys}
                           for row in self.owners]}
