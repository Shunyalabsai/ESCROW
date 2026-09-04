"""L_batch: the one named, order-invariant batch objective, and the repair pass.

L_batch is a function of the current state's sufficient statistics ONLY - no
arrival position appears anywhere (test UT-15). The insertion step is a
greedy move on it; every repair move is an exact evaluation of it; a move is
accepted iff it strictly lowers it, which with a finite state space and a lower
bound gives termination (the DP-means Theorem 3.1 analogue).

v1 scope: categorical facets; merge moves between nodes with DISJOINT member sets
(the sibling-slice case the mint produces); the owner-index term E6 is omitted
from the objective (declared: it is zero for disjoint-support graphs and bounded
by log2(1+overlap) per cell otherwise).
"""
from __future__ import annotations

import math

from .codes import L_KT_block, L_N, L_col, L_supp, lg2


def L_vblock(counts: dict, naming: float) -> float:
    """Canonical order-invariant code for a categorical value block given counts.
    Novelty column (which of the N observations were first-sights: u of N) +
    KT block over the u seen values for the N-u repeat picks + u times the
    stage-(iii) naming cost. Decodable given the key's global inventory; within
    O(u log u) bits of any realised arrival order (the escape drift the spec
    concedes for expanding alphabets)."""
    u = len(counts)
    if u == 0:
        return 0.0
    N = sum(counts.values())
    repeats = [c - 1 for c in counts.values()]
    return L_col(u, N) + L_KT_block(repeats, u) + u * naming


class BatchObjective:
    def __init__(self, graph) -> None:
        self.g = graph

    # ---- per-node local terms (everything a merge can change) ---- #
    def node_local(self, t: int, S: set, p: dict, blocks: dict) -> float:
        g = self.g
        n, A_n = g.n, len(g.keys)
        s = L_col(t, n) + L_supp(len(S), A_n)
        for kid in S:
            s += L_col(p.get(kid, 0), t)                      # presence block
            blk = blocks.get(kid)
            if blk is not None and blk.counts:
                naming = math.log2(len(g.key_by_id[kid].inventory) + 1.0)
                s += L_vblock(blk.counts, naming)
        return s

    def structure_cost(self, K: int) -> float:
        return L_N(K + 1) - (lg2(K + 1.0) if K > 1 else 0.0)  # L_N(K+1) - log2 K!

    def total(self) -> float:
        """Full L_batch of the current state (background terms included)."""
        g = self.g
        s = self.structure_cost(g.K)
        for v in g.nodes.values():
            s += self.node_local(v.t, v.S, v.p, v.blocks)
        covered: dict[int, set] = {}
        for v in g.nodes.values():
            for kid in v.S:
                covered.setdefault(kid, set()).update(v.members)
        for name, ki in g.keys.items():
            n0 = g.n - len(covered.get(ki.kid, ()))
            p0 = max(0, ki.P - sum(v.p.get(ki.kid, 0) for v in g.nodes.values()
                                   if ki.kid in v.S))
            if n0 > 0:
                s += L_col(min(p0, n0), n0)
            naming = math.log2(len(ki.inventory) + 1.0)
            s += L_vblock(ki.background.counts, naming)
        return s

    # ---- the merge move ---- #
    def merge_delta(self, v, w) -> float | None:
        """Exact dL_batch for merging nodes v and w. None if out of v1 scope
        (overlapping members). Negative means the merge pays."""
        members_m = v.members | w.members
        t_m = len(members_m)
        S_m = v.S | w.S
        # presence from exact publication sets: no double counting, any overlap
        p_m = {kid: len(v.pub.get(kid, set()) | w.pub.get(kid, set())) for kid in S_m}
        blocks_m = {}
        for kid in S_m:
            c = dict(v.blocks[kid].counts) if kid in v.blocks else {}
            if kid in w.blocks:
                for vid, cnt in w.blocks[kid].counts.items():
                    c[vid] = c.get(vid, 0) + cnt
            blk = type(next(iter(v.blocks.values()))) () if v.blocks else None
            blocks_m[kid] = _CountsOnly(c)
        before = self.node_local(v.t, v.S, v.p, v.blocks) \
               + self.node_local(w.t, w.S, w.p, w.blocks)
        after = self.node_local(t_m, S_m, p_m, blocks_m)
        d_struct = self.structure_cost(self.g.K - 1) - self.structure_cost(self.g.K)
        # Background presence: a key supported by only one side now covers the
        # union's members, so records newly covered leave the background block.
        d_background = 0.0
        g = self.g
        rk = g.record_keys
        for kid in S_m:
            if kid in v.S and kid in w.S:
                continue
            donor = w if kid in v.S else v          # the side that did NOT support kid
            others = [x for x in g.nodes.values()
                      if kid in x.S and x.nid not in (v.nid, w.nid)]
            covered = set().union(*(x.members for x in others)) if others else set()
            covered |= (v if kid in v.S else w).members
            newly = donor.members - covered
            if not newly:
                continue
            ki = g.key_by_id[kid]
            n0_before = g.n - len(covered)
            pub_all = sum(x.p.get(kid, 0) for x in g.nodes.values() if kid in x.S)
            p0_before = max(0, ki.P - pub_all)
            newly_pub = sum(1 for r in newly if kid in rk.get(r, ()))
            n0_after = n0_before - len(newly)
            p0_after = max(0, p0_before - newly_pub)
            if n0_before > 0:
                d_background -= L_col(min(p0_before, n0_before), n0_before)
            if n0_after > 0:
                d_background += L_col(min(p0_after, n0_after), n0_after)
        return after - before + d_struct + d_background

    # ---- the delete move a node the objective no longer wants (test UT-13b) ---- #
    def delete_delta(self, v) -> float:
        """Exact dL_batch for deleting node v: its members' cells return to the
        background. Negative means the deletion pays - the node never earned its
        keep (e.g. a single atypical record's presence vector, minted then never
        reused)."""
        g = self.g
        d = -self.node_local(v.t, v.S, v.p, v.blocks)
        d += self.structure_cost(g.K - 1) - self.structure_cost(g.K)
        rk = g.record_keys
        for kid in v.S:
            others = [x for x in g.nodes.values() if kid in x.S and x.nid != v.nid]
            covered_wo = set().union(*(x.members for x in others)) if others else set()
            newly_uncovered = v.members - covered_wo
            ki = g.key_by_id[kid]
            pub_all = sum(x.p.get(kid, 0) for x in g.nodes.values() if kid in x.S)
            n0_before = g.n - len(covered_wo | v.members)
            p0_before = max(0, ki.P - pub_all)
            n0_after = n0_before + len(newly_uncovered)
            p0_after = p0_before + sum(1 for r in newly_uncovered
                                       if kid in rk.get(r, ()))
            if n0_before > 0:
                d -= L_col(min(p0_before, n0_before), n0_before)
            if n0_after > 0:
                d += L_col(min(p0_after, n0_after), n0_after)
            # the values v coded return to the background block
            if kid in v.blocks and v.blocks[kid].counts:
                naming = math.log2(len(ki.inventory) + 1.0)
                bgc = dict(ki.background.counts)
                d -= L_vblock(bgc, naming) + 0.0
                for vid, c in v.blocks[kid].counts.items():
                    bgc[vid] = bgc.get(vid, 0) + c
                d += L_vblock(bgc, naming)
                d -= L_vblock(v.blocks[kid].counts, naming) * 0.0  # already in node_local
        return d

    def _apply_delete(self, v) -> None:
        g = self.g
        for kid in v.S:
            ki = g.key_by_id[kid]
            if kid in v.blocks:
                bg = ki.background
                for vid, c in v.blocks[kid].counts.items():
                    if vid in bg.counts:
                        bg.counts[vid] += c
                    else:
                        bg.counts[vid] = c
                        bg.u += 1
                    bg.N += c
            g.ix2.get(kid, set()).discard(v.nid)
        for key, ids in g.ix1.items():
            ids.discard(v.nid)
        del g.nodes[v.nid]
        g.K -= 1
        g.e += 1

    # ---- the reassign move: a past record joins a node it predates ---------- #
    def reassign_gain(self, rec_n: int, v) -> float:
        """Exact dL_batch for retro-attaching record rec_n to node v (membership,
        presence, and value-ownership for the record's keys in v's support move
        from the background to v). Negative means it pays."""
        g = self.g
        keys_r = g.record_keys.get(rec_n, frozenset())
        vals_r = g.record_vals.get(rec_n, {})
        d = L_col(v.t + 1, g.n) - L_col(v.t, g.n)          # the membership column
        for kid in v.S:
            pub = 1 if kid in keys_r else 0
            # node presence block gains one trial
            d += L_col(v.p.get(kid, 0) + pub, v.t + 1) - L_col(v.p.get(kid, 0), v.t)
            # background presence block loses this record (if it was uncovered)
            others = [x for x in g.nodes.values() if kid in x.S]
            covered = set().union(*(x.members for x in others)) if others else set()
            if rec_n in covered:
                continue
            ki = g.key_by_id[kid]
            pub_all = sum(x.p.get(kid, 0) for x in g.nodes.values() if kid in x.S)
            n0b = g.n - len(covered)
            p0b = max(0, ki.P - pub_all)
            d -= L_col(min(p0b, n0b), n0b) if n0b > 0 else 0.0
            n0a, p0a = n0b - 1, p0b - pub
            d += L_col(min(max(p0a, 0), n0a), n0a) if n0a > 0 else 0.0
            # value ownership moves from the background block to v's block
            if pub and kid in vals_r:
                vid = vals_r[kid]
                naming = math.log2(len(ki.inventory) + 1.0)
                bg = ki.background.counts
                if bg.get(vid, 0) > 0:
                    d -= L_vblock(bg, naming)
                    bg2 = dict(bg); bg2[vid] -= 1
                    if bg2[vid] == 0: del bg2[vid]
                    d += L_vblock(bg2, naming)
                    nb = v.blocks.get(kid)
                    nbc = dict(nb.counts) if nb is not None else {}
                    d -= L_vblock(nbc, naming)
                    nbc[vid] = nbc.get(vid, 0) + 1
                    d += L_vblock(nbc, naming)
        return d

    def _apply_reassign(self, rec_n: int, v) -> None:
        g = self.g
        keys_r = g.record_keys.get(rec_n, frozenset())
        vals_r = g.record_vals.get(rec_n, {})
        v.members.add(rec_n)
        v.t = len(v.members)
        for kid in v.S:
            if kid in keys_r:
                v.p[kid] = v.p.get(kid, 0) + 1
                v.pub.setdefault(kid, set()).add(rec_n)
                vid = vals_r.get(kid)
                ki = g.key_by_id[kid]
                bg = ki.background
                if vid is not None and bg.counts.get(vid, 0) > 0:
                    bg.counts[vid] -= 1
                    bg.N -= 1
                    if bg.counts[vid] == 0:
                        del bg.counts[vid]
                        bg.u -= 1
                    blk = v.blocks.get(kid)
                    if blk is None:
                        from .codes import ValueBlock
                        blk = v.blocks[kid] = ValueBlock()
                    if vid in blk.counts:
                        blk.counts[vid] += 1
                    else:
                        blk.counts[vid] = 1
                        blk.u += 1
                    blk.N += 1
                    g.ix1.setdefault((kid, vid), set()).add(v.nid)

    def reassign_pass(self) -> int:
        """Sweep uncovered records against every node sharing a key; apply the
        best strictly-improving retro-attachment per record. One pass."""
        g = self.g
        if not g.nodes:
            return 0
        covered = set().union(*(v.members for v in g.nodes.values()))
        moved = 0
        for rec_n in list(g.record_keys):
            if rec_n in covered:
                continue
            keys_r = g.record_keys[rec_n]
            best, best_v = -1e-9, None
            for v in g.nodes.values():
                if not (v.S & keys_r):
                    continue
                d = self.reassign_gain(rec_n, v)
                if d < best:
                    best, best_v = d, v
            if best_v is not None:
                self._apply_reassign(rec_n, best_v)
                moved += 1
        return moved

    def repair(self, max_rounds: int = 50) -> int:
        """Greedy best-merge-first pass; accept iff dL < 0; terminate at fixpoint.
        Returns the number of merges applied."""
        g = self.g
        moves = 0
        for _ in range(max_rounds):
            progressed = False
            # 1. merges to fixpoint (consolidate slices into features)
            while True:
                best = None
                nodes = sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid))
                for i, v in enumerate(nodes):
                    for w in nodes[i + 1:]:
                        if not (v.S & w.S):
                            continue
                        d = self.merge_delta(v, w)
                        if d is not None and d < -1e-9 and (best is None or d < best[0]):
                            best = (d, v, w)
                if best is None:
                    break
                self._apply_merge(best[1], best[2])
                moves += 1
                progressed = True
            # 2. reassign: stragglers join the consolidated nodes BEFORE any
            #    delete is evaluated - a partially-attached true node loses to
            #    the background (measured: exact delta -50.8 on UT-12) while the
            #    fully-attached one wins; deleting before reassigning destroys
            #    true structure on the strength of its own incompleteness.
            r = self.reassign_pass()
            moves += r
            progressed = progressed or r > 0
            # 3. deletes: nodes the completed objective still does not want
            while True:
                best = None
                for v in sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid)):
                    d = self.delete_delta(v)
                    if d < -1e-9 and (best is None or d < best[0]):
                        best = (d, v)
                if best is None:
                    break
                self._apply_delete(best[1])
                moves += 1
                progressed = True
            if not progressed:
                break
        return moves

    def _apply_merge(self, v, w) -> None:
        g = self.g
        v.members |= w.members
        v.t = len(v.members)
        for kid, s in w.pub.items():
            v.pub.setdefault(kid, set()).update(s)
        for kid in w.S:
            v.S.add(kid)
        for kid in v.S:
            v.p[kid] = len(v.pub.get(kid, ()))
            if kid in w.blocks:
                if kid in v.blocks:
                    for vid, cnt in w.blocks[kid].counts.items():
                        vb = v.blocks[kid]
                        if vid in vb.counts:
                            vb.counts[vid] += cnt
                        else:
                            vb.counts[vid] = cnt
                            vb.u += 1
                        vb.N += cnt
                else:
                    v.blocks[kid] = w.blocks[kid]
        for key, ids in g.ix1.items():
            ids.discard(w.nid)
        for kid in w.S:
            g.ix2.get(kid, set()).discard(w.nid)
            g.ix2.setdefault(kid, set()).add(v.nid)
            if kid in v.blocks:
                for vid in v.blocks[kid].counts:
                    g.ix1.setdefault((kid, vid), set()).add(v.nid)
        del g.nodes[w.nid]
        g.K -= 1
        g.e += 1


class _CountsOnly:
    """Minimal block view for merge evaluation."""
    __slots__ = ("counts",)
    def __init__(self, counts: dict) -> None:
        self.counts = counts
