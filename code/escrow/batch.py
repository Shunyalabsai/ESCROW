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
import os

from . import codes as C
from .codes import L_KT_block, L_N, L_col, L_supp, kt, lg2, naming_charge

# The split and the residual mint. SPLIT_CAND_CAP bounds how many cells the operator proposes and
# never what it accepts: the naming charge is computed from the number actually considered, so a
# smaller pool is charged less and the gate stays exact for the pool it searched. It has the same
# standing as the candidate pool cap the consensus pass already carries, which is an efficiency
# bound rather than a quantity of the criterion. SPLIT_MIN_HALF is the smallest half worth proposing
# and exists so the operator does not spend its time on pairs of records.
SPLIT_MOVES = os.environ.get("ESCROW_SPLIT", "1") != "0"
SPLIT_CAND_CAP = int(os.environ.get("ESCROW_SPLIT_CAP", "64"))
SPLIT_MIN_HALF = 2


def L_vblock(counts: dict, naming: float) -> float:
    """Canonical order-invariant code for a categorical value block given counts.
    Novelty column (which of the N observations were first-sights: u of N) +
    KT block over the u seen values for the N-u repeat picks + u times the
    stage-(iii) naming cost. Decodable given the key's global inventory; within
    O(u log u) bits of any realised arrival order (the escape drift the spec
    concedes for expanding alphabets).

    Under ESCROW_TIGHT_NAMING the u naming charges are not equal: the i-th value
    named is drawn from the inventory minus the i already named, so the total is
    sum_{i<u} log2(inv - i + 1), the block-code form of codes.naming_charge."""
    u = len(counts)
    if u == 0:
        return 0.0
    N = sum(counts.values())
    repeats = [c - 1 for c in counts.values()]
    s = L_col(u, N) + L_KT_block(repeats, u)
    if not C.TIGHT_NAMING:                      # the shipped arithmetic, bit for bit
        return s + u * naming
    return s + sum(naming_charge(naming, i) for i in range(u))


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

    def _bg_sets(self, kid, exclude=()):
        """(covered, pubs) for key kid: the union of member sets, and the union of
        publication sets, over every node supporting kid except those in exclude.
        Unions, never sums: a record in two nodes is one record (summing per-node
        counts overstated coverage of shared members and mispriced merges by up
        to 313 bits on the shared-key fixture)."""
        covered, pubs = set(), set()
        for x in self.g.nodes.values():
            if kid in x.S and x.nid not in exclude:
                covered |= x.members
                pubs |= x.pub.get(kid, set())
        return covered, pubs

    def structure_cost(self, K: int) -> float:
        return L_N(K + 1) - (lg2(K + 1.0) if K > 1 else 0.0)  # L_N(K+1) - log2 K!

    def total(self) -> float:
        """Full L_batch of the current state (background terms included)."""
        g = self.g
        s = self.structure_cost(g.K)
        for v in g.nodes.values():
            s += self.node_local(v.t, v.S, v.p, v.blocks)
        covered: dict[int, set] = {}
        pubs: dict[int, set] = {}
        for v in g.nodes.values():
            for kid in v.S:
                covered.setdefault(kid, set()).update(v.members)
                pubs.setdefault(kid, set()).update(v.pub.get(kid, ()))
        for name, ki in g.keys.items():
            n0 = g.n - len(covered.get(ki.kid, ()))
            p0 = max(0, ki.P - len(pubs.get(ki.kid, ())))
            if n0 > 0:
                s += L_col(min(p0, n0), n0)
            naming = math.log2(len(ki.inventory) + 1.0)
            s += L_vblock(ki.background.counts, naming)
        return s

    # ---- the merge move ---- #
    def merge_delta(self, v, w) -> float | None:
        """Exact dL_batch for merging nodes v and w, for disjoint and for
        overlapping member sets. Negative means the merge pays."""
        g = self.g
        rk = g.record_keys
        members_m = v.members | w.members
        t_m = len(members_m)
        S_m = v.S | w.S
        # Presence of the merged node: every member publishing the key, exactly
        # once. For a key only one side supports, the other side's members that
        # publish it join the presence block (they were in the background block).
        p_m = {}
        for kid in S_m:
            pub = v.pub.get(kid, set()) | w.pub.get(kid, set())
            if not (kid in v.S and kid in w.S):
                donor = w if kid in v.S else v
                pub = pub | {r for r in donor.members if kid in rk.get(r, ())}
            p_m[kid] = len(pub)
        # Value blocks add: a record's value for a key lives in exactly one block
        # (its owner's), so no record is counted twice even when members overlap.
        blocks_m = {}
        for kid in S_m:
            c = dict(v.blocks[kid].counts) if kid in v.blocks else {}
            if kid in w.blocks:
                for vid, cnt in w.blocks[kid].counts.items():
                    c[vid] = c.get(vid, 0) + cnt
            blocks_m[kid] = _CountsOnly(c)
        before = self.node_local(v.t, v.S, v.p, v.blocks) \
               + self.node_local(w.t, w.S, w.p, w.blocks)
        after = self.node_local(t_m, S_m, p_m, blocks_m)
        d_struct = self.structure_cost(self.g.K - 1) - self.structure_cost(self.g.K)
        # Background presence: a key supported by only one side now covers the
        # union's members, so records newly covered leave the background block.
        d_background = 0.0
        for kid in S_m:
            if kid in v.S and kid in w.S:
                continue                      # coverage and publications unchanged
            donor = w if kid in v.S else v          # the side that did NOT support kid
            covered, pubs = self._bg_sets(kid)      # every node supporting kid (keeper included)
            newly = donor.members - covered
            if not newly:
                continue
            ki = g.key_by_id[kid]
            n0_before = g.n - len(covered)
            p0_before = max(0, ki.P - len(pubs))
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
            covered_wo, pubs_wo = self._bg_sets(kid, exclude=(v.nid,))
            newly_uncovered = v.members - covered_wo
            ki = g.key_by_id[kid]
            n0_before = g.n - len(covered_wo | v.members)
            p0_before = max(0, ki.P - len(pubs_wo | v.pub.get(kid, set())))
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
        for r, own in g.record_owner.items():           # v's cells return to the background
            for kid, o in own.items():
                if o == v.nid:
                    own[kid] = 0
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
            cov = getattr(self, "_cov", None)
            if cov is not None:
                covered = cov.get(kid, set())
                pubs = self._pub_all.get(kid, set())
            else:
                covered, pubs = self._bg_sets(kid)
            if rec_n in covered:
                continue
            ki = g.key_by_id[kid]
            n0b = g.n - len(covered)
            p0b = max(0, ki.P - len(pubs))
            d -= L_col(min(p0b, n0b), n0b) if n0b > 0 else 0.0
            n0a, p0a = n0b - 1, p0b - pub
            d += L_col(min(max(p0a, 0), n0a), n0a) if n0a > 0 else 0.0
            # value ownership moves from the background block to v's block,
            # priced by the exact O(1) increments (identical to full recompute,
            # verified to 5e-13; the recompute was the profile's hot spot)
            if pub and kid in vals_r:
                vid = vals_r[kid]
                naming = math.log2(len(ki.inventory) + 1.0)
                bg = ki.background.counts
                if bg.get(vid, 0) > 0:
                    d += L_vblock_delta_remove(bg, vid, naming)
                    nb = v.blocks.get(kid)
                    d += L_vblock_delta_add(nb.counts if nb is not None else {},
                                            vid, naming)
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
                    g.record_owner.setdefault(rec_n, {})[kid] = v.nid
                    g.ix1.setdefault((kid, vid), set()).add(v.nid)

    def reassign_pass(self) -> int:
        """Sweep uncovered records against every node sharing a key; apply the
        best strictly-improving retro-attachment per record. One pass."""
        g = self.g
        if not g.nodes:
            return 0
        covered = set().union(*(v.members for v in g.nodes.values()))
        # per-key coverage and publication totals, computed once per pass and
        # patched after each applied move (reassign_gain recomputed these per
        # call before; the set unions alone were 13 seconds of a 194 second
        # profile)
        self._cov = {}
        self._pub_all = {}                       # kid -> set of publishing covered records
        for v in g.nodes.values():
            for kid in v.S:
                self._cov.setdefault(kid, set()).update(v.members)
                self._pub_all.setdefault(kid, set()).update(v.pub.get(kid, ()))
        moved = 0
        full = getattr(self, "_reassign_full", True)
        since = getattr(g, "_reassign_since", 0)
        dirty_keys = set()
        for nid in g.dirty:
            v = g.nodes.get(nid)
            if v is not None:
                dirty_keys |= v.S
        for rec_n in list(g.record_keys):
            if rec_n in covered:
                continue
            keys_r = g.record_keys[rec_n]
            if not full and rec_n <= since and not (keys_r & dirty_keys):
                continue
            best, best_v = -1e-9, None
            for v in g.nodes.values():
                if not (v.S & keys_r):
                    continue
                d = self.reassign_gain(rec_n, v)
                if d < best:
                    best, best_v = d, v
            if best_v is not None:
                self._apply_reassign(rec_n, best_v)
                covered.add(rec_n)
                for kid in best_v.S:
                    self._cov.setdefault(kid, set()).add(rec_n)
                    if kid in keys_r:
                        self._pub_all.setdefault(kid, set()).add(rec_n)
                moved += 1
        self._cov = None
        self._pub_all = None
        return moved

    def absorb_new_mints(self, new_ids) -> int:
        """Mint-time absorption: a node just released from escrow immediately
        tries to merge into an existing node under the same exact dL_batch test.
        This closes the churn window in which slice siblings accumulate between
        repair passes (observed on ReVerb45K as K swinging 70 to 5 to 202)."""
        g = self.g
        applied = 0
        for nid in list(new_ids):
            w = g.nodes.get(nid)
            if w is None:
                continue
            while True:
                best, best_v = -1e-9, None
                for v in g.nodes.values():
                    if v.nid == w.nid:
                        continue
                    # only support-contained pairs: the churn this hook closes is
                    # same-feature siblings, and a fresh mint must not face a
                    # merge test against a grown DISJOINT node at the one moment
                    # it is smallest (measured: the unrestricted hook collapsed
                    # the two-group control to K=1 at birth). Cross-feature
                    # merges stay with the repair pass, which runs after
                    # reassignment completes the evidence.
                    if not (w.S <= v.S or v.S <= w.S):
                        continue
                    d = self.merge_delta(v, w)
                    if d is not None and d < best:
                        best, best_v = d, v
                if best_v is None:
                    break
                self._apply_merge(best_v, w)
                applied += 1
                w = best_v                      # keep absorbing upward if it pays
        return applied

    def install(self) -> "BatchObjective":
        """Wire the mint-time absorption hook into the graph. Returns self."""
        self.g.mint_merge_hook = self.absorb_new_mints
        return self

    def cohort_reassign(self, full: bool = True) -> int:
        """The batch dual of the deferred mint. A single straggler often cannot
        pay its own marginal column price into a half-formed node (measured on
        the two-group control at rho = 0.11: attach costs 3.03 bits against 2
        bits of evidence, so 57 of 60 stragglers individually refuse), while the
        cohort of all stragglers together pays easily; that is the submodular
        valley between them. This move prices the WHOLE uncovered cohort of a
        node jointly, with the same exact closed forms, and attaches it only
        when the joint delta is negative."""
        g = self.g
        if not g.nodes:
            return 0
        covered = set().union(*(v.members for v in g.nodes.values()))
        moved = 0
        for v in sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid)):
            if not full and v.nid not in g.dirty:
                continue
            cohort = [r for r in g.record_keys
                      if r not in covered and (g.record_keys[r] & v.S)]
            if len(cohort) < 2:
                continue
            m = len(cohort)
            d = L_col(v.t + m, g.n) - L_col(v.t, g.n)
            val_moves = []                       # (ki, vid) per observation
            for kid in v.S:
                ki = g.key_by_id[kid]
                pub = [r for r in cohort if kid in g.record_keys[r]]
                pm = len(pub)
                d += L_col(v.p.get(kid, 0) + pm, v.t + m)                    - L_col(v.p.get(kid, 0), v.t)
                cov_k, pubs_k = self._bg_sets(kid)
                n0b = g.n - len(cov_k)
                p0b = max(0, ki.P - len(pubs_k))
                n0a, p0a = n0b - m, max(0, p0b - pm)
                if n0b > 0:
                    d -= L_col(min(p0b, n0b), n0b)
                if n0a > 0:
                    d += L_col(min(max(p0a, 0), n0a), n0a)
                # value ownership moves, priced as a chain of exact O(1) deltas
                # (one remove from the background, one add to the node, per
                # observation) instead of full block recomputes: on token
                # universes the background blocks hold thousands of values and
                # the recompute was the repair pass's remaining hot spot
                naming = math.log2(len(ki.inventory) + 1.0)
                bg = dict(ki.background.counts)
                nb = v.blocks.get(kid)
                nbc = dict(nb.counts) if nb is not None else {}
                for r in pub:
                    vid = g.record_vals.get(r, {}).get(kid)
                    if vid is not None and bg.get(vid, 0) > 0:
                        d += L_vblock_delta_remove(bg, vid, naming)
                        bg[vid] -= 1
                        if bg[vid] == 0:
                            del bg[vid]
                        d += L_vblock_delta_add(nbc, vid, naming)
                        nbc[vid] = nbc.get(vid, 0) + 1
                        val_moves.append((kid, r))
            if d < -1e-9:
                for r in cohort:
                    self._apply_reassign(r, v)
                    covered.add(r)
                g.dirty.add(v.nid)
                moved += m
        return moved

    def repair(self, max_rounds: int = 50, full: bool = None, final: bool = None) -> int:
        """Greedy best-merge-first pass; accept iff dL < 0; terminate at fixpoint.
        Returns the number of moves applied.

        Intermediate passes are INCREMENTAL: merge and delete proposals must
        involve a node mutated since the last pass, and reassignment only visits
        records that arrived since then or that share a key with a mutated
        node's support. A FULL pass (everything proposed) self-schedules when
        the stream has doubled since the last one, mirroring the power-of-two
        checkpoint discipline, and can be forced with full=True for the final
        call. Background terms drift with n even for untouched nodes, which is
        why the full pass exists; between full passes the incremental schedule
        is an efficiency choice, not a change of objective."""
        g = self.g
        # The split and the residual mint run on the last pass only, and the signature already
        # separates the two kinds of full pass. A full pass the schedule chose for itself arrives
        # with full=None and is computed here; a caller's deliberate final pass passes full=True.
        # So an explicit full=True means final unless the caller says otherwise, which is what makes
        # the twenty-seven experiments that hand-roll their own final repair get the moves without
        # each of them having to remember a second flag.
        if final is None:
            final = full is True
        if full is None:
            full = g.n >= 2 * max(1, g.last_full_repair_n)
        if full:
            g.last_full_repair_n = g.n
        dirty0 = set(g.dirty)
        moves = 0
        for _ in range(max_rounds):
            progressed = False
            # 1. merges to fixpoint (consolidate slices into features)
            while True:
                best = None
                nodes = sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid))
                for i, v in enumerate(nodes):
                    for w in nodes[i + 1:]:
                        if not full and v.nid not in g.dirty and w.nid not in g.dirty:
                            continue
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
            r = self.cohort_reassign(full=full)
            self._reassign_full = full
            r += self.reassign_pass()
            g._reassign_since = g.n
            moves += r
            progressed = progressed or r > 0
            # 3. deletes: nodes the completed objective still does not want
            while True:
                best = None
                for v in sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid)):
                    if not full and v.nid not in g.dirty and v.nid not in dirty0:
                        continue
                    d = self.delete_delta(v)
                    if d < -1e-9 and (best is None or d < best[0]):
                        best = (d, v)
                if best is None:
                    break
                self._apply_delete(best[1])
                moves += 1
                progressed = True
            # 4. split a node holding several kinds, and mint a node out of the residual.
            #    Neither state is reachable by the three moves above: reassign can only move a
            #    record to a node that already exists, and every move above acts on a node, so
            #    records in no node are out of reach of all of them. Measured on the value fixture
            #    E75, where the objective prefers the planted truth by 3,616 to 16,899 bits, these
            #    two moves take mean ARI from 0.6099 to 0.9995 and K from 5.0 to exactly 8.
            # These two run on the FINAL pass only, and the reason is the deferral argument
            # rather than convenience. Mid-stream a node is still accumulating, so splitting it or
            # minting out of the residual decides on partial evidence, which is the mistake the
            # escrow mechanism exists to avoid. Measured on the encyclopedia stream: one residual
            # mint taken mid-stream lowered L_batch by 134.9 bits at the moment it was applied and
            # left the final state 85 bits WORSE and 0.0356 lower in agreement, because repair is
            # greedy at the time of the move and the move put the search on a worse path. The delta
            # was right and the timing was wrong.
            if SPLIT_MOVES and final:
                from .split import (apply_residual_mint, apply_split, best_residual_mint,
                                    best_split)
                while True:
                    best = None
                    for v in sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid)):
                        if v.t < 2 * SPLIT_MIN_HALF:
                            continue
                        b = best_split(self, v, cap=SPLIT_CAND_CAP)
                        if b is not None and (best is None or b[0] < best[0]):
                            best = (b[0], v, b[1])
                    if best is None:
                        break
                    apply_split(self, best[1], best[2])
                    moves += 1
                    progressed = True
                # The residual is the whole stream until the first node exists, so proposing out
                # of it on every intermediate pass costs more than it can return. It runs on the
                # full passes only, the same power-of-two checkpoint the pass already keeps for the
                # background terms, which is a schedule and not a change of objective.
                while True:
                    b = best_residual_mint(self, cap=SPLIT_CAND_CAP)
                    if b is None:
                        break
                    apply_residual_mint(self, b[1], b[3])
                    moves += 1
                    progressed = True
            if not progressed:
                break
        g.dirty.clear()
        return moves

    def _apply_merge(self, v, w) -> None:
        g = self.g
        rk = g.record_keys
        one_sided_v = v.S - w.S                # keys only v supported: w's publishers join
        one_sided_w = w.S - v.S                # keys only w supported: v's publishers join
        v.members |= w.members
        v.t = len(v.members)
        for kid, s in w.pub.items():
            v.pub.setdefault(kid, set()).update(s)
        for kid in one_sided_v:
            v.pub.setdefault(kid, set()).update(r for r in w.members if kid in rk.get(r, ()))
        for kid in one_sided_w:
            v.pub.setdefault(kid, set()).update(r for r in v.members if kid in rk.get(r, ()))
        for kid in w.S:
            v.S.add(kid)
        for r, own in g.record_owner.items():           # cells w owned are v's now
            for kid, o in own.items():
                if o == w.nid:
                    own[kid] = v.nid
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
        g.dirty.discard(w.nid)
        g.dirty.add(v.nid)
        g.K -= 1
        g.e += 1


class _CountsOnly:
    """Minimal block view for merge evaluation."""
    __slots__ = ("counts",)
    def __init__(self, counts: dict) -> None:
        self.counts = counts


# --------------------------------------------------------------------------- #
# Exact O(1) increments for L_vblock. Adding or removing ONE observation of a
# value changes the canonical block code by a closed-form amount; recomputing
# the whole block for a one-observation move was 97 percent of the repair
# pass's runtime (measured: 6M full recomputes, 348M lgamma calls, on a 3,000
# record stream). These return exactly L_vblock(after) - L_vblock(before);
# test_batch_deltas.py asserts the identity against full recomputes.
# --------------------------------------------------------------------------- #
def L_vblock_delta_add(counts: dict, vid, naming: float) -> float:
    """Delta from adding one observation of vid to a block with these counts."""
    u = len(counts)
    N = sum(counts.values())
    c = counts.get(vid, 0)
    if u == 0:
        # empty block -> singleton: L_col(1,1) + L_KT_block([0],1) + naming
        return L_col(1, 1) + L_KT_block([0], 1) + naming_charge(naming, 0)
    if c == 0:
        # novelty count u -> u+1, N -> N+1; repeats gain a zero entry (alphabet grows)
        d = L_col(u + 1, N + 1) - L_col(u, N)
        repeats_before = [x - 1 for x in counts.values()]
        repeats_after = repeats_before + [0]
        d += L_KT_block(repeats_after, u + 1) - L_KT_block(repeats_before, u)
        return d + naming_charge(naming, u)     # the (u+1)-th value named
    # repeat: novelty column N -> N+1 (zeros side), one repeat count increments
    d = L_col(u, N + 1) - L_col(u, N)
    rep_total = N - u
    d += kt(c - 1, rep_total, u)     # incremental = block increment (UT-6 identity)
    return d


def L_vblock_delta_remove(counts: dict, vid, naming: float) -> float:
    """Delta from removing one observation of vid; inverse of delta_add."""
    after = dict(counts)
    if after[vid] == 1:
        del after[vid]
    else:
        after[vid] -= 1
    return -L_vblock_delta_add(after, vid, naming)
