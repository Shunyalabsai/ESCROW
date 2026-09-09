"""Structural proposals. These functions never accept a move or change their input."""
from __future__ import annotations


def route_to(state, nid):
    node = state.nodes[nid]
    for row in node.members:
        for key in node.support:
            state.owners[row][key] = nid


def propose(state, operation, **args):
    out = state.clone()
    if operation == "compound":
        for move, arguments in args["moves"]:
            if move == "compound":
                raise ValueError("compound proposals cannot nest")
            out = propose(out, move, **arguments)
    elif operation in ("create", "refine"):
        parents = args.get("parents", ())
        if operation == "refine":
            parents = set(parents) | {args["parent"]}
        nid = out.add_node(args["members"], args["support"], parents)
        if args.get("route", True):
            route_to(out, nid)
    elif operation == "introduce_parent":
        children = set(args["children"])
        if not children:
            raise ValueError("parent insertion requires children")
        members = set.union(*(out.nodes[c].members for c in children))
        nid = out.add_node(members, args.get("support", ()), args.get("parents", ()))
        for c in children:
            out.nodes[c].parents.add(nid)
        out.close_ancestors()
        out.reduce_edges()
        if args.get("route", False):
            route_to(out, nid)
        if args.get("inherit_support", False):
            for c in children:
                out.nodes[c].support.difference_update(out.nodes[nid].support)
    elif operation in ("add_link", "remove_link", "reparent"):
        node = out.nodes[args["child"]]
        if operation == "add_link":
            node.parents.add(args["parent"])
        elif operation == "remove_link":
            node.parents.remove(args["parent"])
        else:
            node.parents = set(args["parents"])
        out.reduce_edges()
    elif operation == "delete":
        nid = args["node"]
        parents = out.nodes[nid].parents
        for node in out.nodes.values():
            if nid in node.parents:
                node.parents = (node.parents - {nid}) | parents
        del out.nodes[nid]
        out.reduce_edges()
    elif operation == "merge":
        a, b = args["nodes"]
        if a == b:
            raise ValueError("merge requires distinct nodes")
        left, right = out.nodes[a], out.nodes[b]
        left.members |= right.members
        left.support |= right.support
        left.parents = (left.parents & right.parents) - {a, b}
        for node in out.nodes.values():
            if b in node.parents:
                node.parents = (node.parents - {b}) | ({a} if node.nid != a else set())
        del out.nodes[b]
        for row in out.owners:
            for key, owner in row.items():
                if owner == b:
                    row[key] = a
        out.close_ancestors()
        out.reduce_edges()
    elif operation == "membership":
        out.nodes[args["node"]].members = set(args["members"])
        out.close_ancestors()
        if args.get("route", False):
            route_to(out, args["node"])
    elif operation == "ownership":
        for row, key, owner in args["assignments"]:
            out.owners[row][key] = owner
        out.validate()
    elif operation == "support":
        out.nodes[args["node"]].support = set(args["support"])
        if args.get("route", False):
            route_to(out, args["node"])
    else:
        raise ValueError("unknown structural operation: " + operation)
    out.clean_owners()
    return out.validate()


def revisions(before, after):
    """Exact edits of the observed prefix, including assignments of absent slots."""
    memberships = []
    for nid in sorted(set(before.nodes) | set(after.nodes)):
        old = before.nodes[nid].members if nid in before.nodes else set()
        new = after.nodes[nid].members if nid in after.nodes else set()
        if old != new:
            memberships.append(dict(node=nid, added=sorted(new - old), removed=sorted(old - new)))
    ownership = []
    for r in range(len(before.records)):
        for key in before.keys:
            old, new = before.owners[r].get(key, 0), after.owners[r].get(key, 0)
            if old != new:
                ownership.append(dict(record=r, key=key, before=old, after=new))
    old_links = {(p, c) for c, v in before.nodes.items() for p in v.parents}
    new_links = {(p, c) for c, v in after.nodes.items() for p in v.parents}
    return dict(membership_revisions=memberships, ownership_revisions=ownership,
                links_added=sorted(new_links - old_links), links_removed=sorted(old_links - new_links),
                nodes_added=sorted(set(after.nodes) - set(before.nodes)),
                nodes_removed=sorted(set(before.nodes) - set(after.nodes)))
