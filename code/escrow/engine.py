"""The escrow insertion process (v1: categorical facets only).

Implements the per-record steps of the ESCROW paper's algorithm, with one
deliberate reordering: ALL evidence terms are computed from pre-record statistics
(snapshotted before any update), because the escrow accumulator G_t is a sequential
log-likelihood ratio and a statistic that incorporates the current record would
invalidate the martingale behind the Ville guarantee (paper, Theorem 1; test UT-17).

v1 scope, declared: categorical values only (numeric Student-t and text PPM are
later phases); posting lists untruncated (B_node = infinity, reported); repair pass
not yet wired (insertion only). Nothing here reads a statistic from the future.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import codes as C
from .codes import ValueBlock, attach, kt, price

try:                                            # optional fast path only; the
    import numpy as _np                         # public package stays stdlib-only
except ImportError:                             # and this branch simply never runs
    _np = None


@dataclass
class KeyInfo:
    kid: int
    P: int = 0                                  # records that published this key
    background: ValueBlock = field(default_factory=ValueBlock)
    inventory: dict = field(default_factory=dict)   # value string -> vid
    first_seen_n: int = 0

    def intern(self, value: str) -> int:
        vid = self.inventory.get(value)
        if vid is None:
            vid = len(self.inventory)
            self.inventory[value] = vid
        return vid


@dataclass
class Node:
    nid: int
    t: int = 0                                  # members
    S: set = field(default_factory=set)         # support: key ids
    p: dict = field(default_factory=dict)       # kid -> members publishing kid
    blocks: dict = field(default_factory=dict)  # kid -> ValueBlock
    members: set = field(default_factory=set)   # record ids (repair needs these)
    pub: dict = field(default_factory=dict)     # kid -> member ids publishing kid
    birth_n: int = 0                            # diagnostics ONLY; never in L_batch


@dataclass
class Candidate:
    sig: tuple                                  # (kid, vid) seed signature
    t: int = 0
    keys: set = field(default_factory=set)
    p: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=dict)
    g: dict = field(default_factory=dict)       # per-key escrow, bits
    G: float = 0.0                              # total escrow, bits
    members: list = field(default_factory=list)
    p_vec: object = None                        # numpy fast path: presence counts
    g_vec: object = None                        # numpy fast path: per-key escrow


@dataclass
class Receipt:
    """The per-record audit trail: what was decided and which facet paid for it."""
    n: int
    active: list                                # node ids attached (excl. background)
    owners: dict                                # key string -> node id (0 = background)
    per_key_bits: dict                          # key string -> signed bits vs background
    minted: list                                # node ids minted while processing this record


class EscrowGraph:
    def __init__(self, cand_pool_cap: int = 4096,
                 accrue_owned_cells: bool = False) -> None:
        # accrue_owned_cells is an experiment switch, default off so the shipped
        # behaviour is unchanged. Off: a (key, value) cell feeds its candidate only
        # while the background still owns it, so the first node to take a shared key
        # stops that key feeding every other candidate. On: every cell of the record
        # feeds its own signature's candidate, with the evidence still charged
        # against the CURRENT OWNER's code, which is what the delta below already
        # computes. E46 measures which is right.
        self.n = 0
        self.K = 0
        self.next_id = 0                        # monotone node ids: NEVER reused
        self.e = 0                              # edits (mints) so far
        self.H = 0.0
        self.keys: dict[str, KeyInfo] = {}
        self.key_by_id: dict[int, KeyInfo] = {}
        self.key_name: dict[int, str] = {}
        self.nodes: dict[int, Node] = {}        # latent nodes only, 1-based ids
        self.ix1: dict[tuple, set] = {}         # (kid, vid) -> node ids
        self.ix2: dict[int, set] = {}           # kid -> node ids supporting kid
        self.pool: dict[tuple, Candidate] = {}
        self.dirty: set[int] = set()            # nodes mutated since the last repair
        self.last_full_repair_n = 0             # power-of-two full-pass scheduler
        self.mint_merge_hook = None             # set by BatchObjective: absorb a
        #                                         fresh mint into an existing node
        #                                         before siblings can accumulate
        self.record_vals: dict[int, dict] = {}        # n -> {kid: vid} (v1, reassign)
        self.record_owner: dict[int, dict] = {}       # n -> {kid: node id or 0}
        # One explainer per (record, key) cell: whoever codes the value holds the
        # count. Without this ledger a minted node copied its candidate's counts
        # while the same cells stayed in the background, so cells were coded
        # twice and L_batch stopped being a function of the partition (measured:
        # 43 percent of Wikipedia cells double counted, 841 bits between two
        # arrival orders that reach the same partition).
        self.record_keys: dict[int, frozenset] = {}   # n -> published kids (v1: kept
        # in memory for the repair pass's overlap accounting; a later phase replaces
        # this with per-node published-member bitmaps)
        self.cand_pool_cap = cand_pool_cap
        self.accrue_owned_cells = accrue_owned_cells
        self.mint_log: list = []

    # ------------------------------------------------------------------ #
    def _key(self, name: str) -> KeyInfo:
        ki = self.keys.get(name)
        if ki is None:
            ki = KeyInfo(kid=len(self.keys), first_seen_n=self.n)
            self.keys[name] = ki
            self.key_by_id[ki.kid] = ki
            self.key_name[ki.kid] = name
        return ki

    # ------------------------------------------------------------------ #
    def process(self, record: dict) -> Receipt:
        """record: {key string: value string}. Returns the receipt."""
        self.n += 1
        n = self.n
        self.H += 1.0 / n
        past = n - 1                            # size of the past, for every predictive

        # --- 0. PARSE (register keys, intern values; all reads below are pre-update)
        items = []                              # (kid, vid, keyinfo)
        naming = {}                             # kid -> stage-(iii) bits, PRE-intern
        for a, x in record.items():
            ki = self._key(a)
            naming[ki.kid] = math.log2(len(ki.inventory) + 1.0)
            items.append((ki.kid, ki.intern(str(x)), ki))
        K_r = {kid for kid, _, _ in items}
        self.record_keys[n] = frozenset(K_r)
        self.record_vals[n] = dict(vid_of) if False else {kid: vid for kid, vid, _ in items}
        vid_of = {kid: vid for kid, vid, _ in items}

        # --- pre-record background costs per published key (the incumbent's code) --
        bg_val = {}                             # kid -> bits to code the value under background
        bg_pres = {}                            # kid -> bits to code "present" under background
        bg_abs = {}                             # kid -> bits to code "absent" under background
        for kid, vid, ki in items:
            bg_val[kid] = ki.background.cost(vid, naming[kid])
            bg_pres[kid] = kt(ki.P, past, 2) if past > 0 else 1.0
        # (bg_abs is filled lazily for candidate silence bills)

        # --- 1. RETRIEVE CANDIDATE NODES ---------------------------------------- #
        cand: set[int] = set()
        for kid, vid, _ in items:
            cand |= self.ix1.get((kid, vid), set())
            cand |= self.ix2.get(kid, set())
        cand &= self.nodes.keys()               # guard against merged-away ids

        # --- 2. PER-ATTRIBUTE EVIDENCE, pre-update ------------------------------ #
        D: dict[int, dict[int, float]] = {}     # kid -> {nid: signed bits vs background}
        Dabs: dict[int, float] = {}             # nid -> silence bill (keys v expects, r lacks)
        for v in cand:
            node = self.nodes[v]
            for kid in K_r & node.S:
                blk = node.blocks[kid]
                dval = bg_val[kid] - blk.cost(vid_of[kid], naming[kid])
                dpres = bg_pres[kid] - kt(node.p.get(kid, 0), node.t, 2)
                D.setdefault(kid, {})[v] = dval + dpres
            silence = 0.0
            for kid in node.S - K_r:
                ki = self.key_by_id[kid]
                b_abs = kt(past - ki.P, past, 2) if past > 0 else 1.0
                silence += b_abs - kt(node.t - node.p.get(kid, 0), node.t, 2)
            Dabs[v] = silence

        # --- 3. GREEDY ACTIVE SET ----------------------------------------------- #
        own = {kid: 0 for kid in K_r}           # 0 = background
        Dcur = {kid: 0.0 for kid in K_r}        # background baseline is 0 by definition
        q = {kid: 0 for kid in K_r}
        S_r: list[int] = []
        remaining = set(cand)
        while remaining:
            best_v, best_gain = None, 0.0
            for v in sorted(remaining, key=lambda w: (self.nodes[w].birth_n, w)):
                node = self.nodes[v]
                g = Dabs[v] - attach(node.t, n)
                for kid in K_r & node.S:
                    g += max(0.0, D.get(kid, {}).get(v, -math.inf) - Dcur[kid])
                    g -= math.log2(2 + q[kid]) - math.log2(1 + q[kid])
                if g > best_gain:
                    best_v, best_gain = v, g
            if best_v is None:
                break
            S_r.append(best_v)
            remaining.discard(best_v)
            node = self.nodes[best_v]
            for kid in K_r & node.S:
                dv = D.get(kid, {}).get(best_v, -math.inf)
                if dv > Dcur[kid]:
                    own[kid] = best_v
                    Dcur[kid] = dv
                    q[kid] += 1

        # --- 6 (moved before updates). ACCRUE ESCROW with pre-record statistics -- #
        resid = [(kid, vid_of[kid]) for kid in K_r
                 if self.accrue_owned_cells or own[kid] == 0]
        touched: list[Candidate] = []
        for sig in resid:
            c = self.pool.get(sig)
            if c is None:
                if len(self.pool) >= self.cand_pool_cap:
                    # v1 eviction: drop the lowest-escrow candidate (declared; the
                    # spec's Space-Saving with deterministic eviction comes later).
                    # Never evict a candidate touched by THIS record: it is about
                    # to be updated and possibly released (measured crash on
                    # MusicBrainz at the 4096 cap: KeyError on the release).
                    touched_sigs = {tc.sig for tc in touched}
                    victims = [cc for cc in self.pool.values() if cc.sig not in touched_sigs]
                    if victims:
                        worst = min(victims, key=lambda cc: cc.G)
                        del self.pool[worst.sig]
                c = Candidate(sig=sig)
                self.pool[sig] = c
            touched.append(c)
        # The incumbent in the escrow log-likelihood ratio is the CURRENT OWNER's
        # code (paper, Theorem 1: the ratio is against the owner's code), not the background. The
        # pseudocode's c[0] shorthand is correct only when the owner IS the
        # background; using the background for owned keys double-counts evidence a
        # minted node already explains and mints sibling nodes (measured: K=332 on
        # the k=4 UT-12 fixture before this fix, K=2 after).
        own_val = {}                            # kid -> incumbent value cost, pre-update
        own_pres = {}                           # kid -> incumbent presence cost, pre-update
        for kid, vid, ki in items:
            w = own[kid]
            if w == 0:
                own_val[kid] = bg_val[kid]
                own_pres[kid] = bg_pres[kid]
            else:
                node = self.nodes[w]
                own_val[kid] = node.blocks[kid].cost(vid, naming[kid])
                own_pres[kid] = kt(node.p.get(kid, 0), node.t, 2)
        # The background absence vector and the published-key mask depend only on
        # statistics frozen for this record (accrual runs before the updates of
        # step 5), so they are the same for every touched candidate. Building them
        # once per record instead of once per candidate is the difference between
        # 16 million and 2 million generator steps on a 600-key stream; on the
        # real marketplace stream this is where the wall time was going.
        _bg_abs_vec = _mask_rec = None
        if _np is not None and touched:
            _nk = len(self.key_by_id)
            _bg_abs_vec = _np.empty(_nk)
            if past > 0:
                _P_vec = _np.fromiter(
                    (self.key_by_id[k].P for k in range(_nk)), float, _nk)
                _bg_abs_vec[:] = -_np.log2((past - _P_vec + 0.5) / (past + 1.0))
            else:
                _bg_abs_vec[:] = 1.0
            _mask_rec = _np.ones(_nk, bool)
            for kid in K_r:
                _mask_rec[kid] = False

        for c in touched:
            for kid, vid, ki in items:
                blk = c.blocks.get(kid)
                if blk is None:
                    blk = c.blocks[kid] = ValueBlock()
                    c.keys.add(kid)
                delta = (own_val[kid] - blk.cost(vid, naming[kid])) \
                        + (own_pres[kid] - kt(c.p.get(kid, 0), c.t, 2))
                c.G += delta
                c.g[kid] = c.g.get(kid, 0.0) + delta
                if c.g_vec is not None and kid < len(c.g_vec):
                    c.g_vec[kid] += delta       # the fast path's per-key ledger
                blk.observe(vid)
                c.p[kid] = c.p.get(kid, 0) + 1
                if c.p_vec is not None and kid < len(c.p_vec):
                    c.p_vec[kid] = c.p[kid]     # keep the fast path current: a
                    # stale presence count makes later absence deltas wrong for
                    # keys the candidate published before but not now
            # Absence runs against the GLOBAL key set, not just the keys the
            # candidate's members have published: a candidate whose members never
            # publish key b earns the bits the background wastes predicting b
            # might appear. Without this credit the objective's margin is
            # invisible to the release rule at small k (measured: the k=2
            # positive control never mints). Keys nobody publishes contribute
            # ~0 by themselves (UT-9). With numpy available this is one vector
            # operation over the key universe (token facets push |A| into the
            # thousands, where the python loop was the wall); the pure-python
            # loop is the fallback and computes the identical quantity.
            if _np is not None:
                nk = len(self.key_by_id)
                bg_abs_vec = _bg_abs_vec
                mask = _mask_rec
                if c.p_vec is None or len(c.p_vec) < nk:
                    pv = _np.zeros(nk)
                    gv = _np.zeros(nk)
                    if c.p_vec is not None:
                        pv[:len(c.p_vec)] = c.p_vec
                        gv[:len(c.g_vec)] = c.g_vec
                    old_len = 0 if c.g_vec is None else len(c.g_vec)
                    for kid, cnt in c.p.items():
                        pv[kid] = cnt
                    for kid, val in c.g.items():
                        if kid >= old_len:      # keys the ledger has never held
                            gv[kid] = val
                    c.p_vec, c.g_vec = pv, gv
                # published keys were already incremented this record and are
                # masked out below; clip them so the log never sees a negative
                # argument (their entries are computed but never used)
                cand_abs = -_np.log2(
                    (_np.minimum(c.p_vec, c.t) * -1.0 + c.t + 0.5) / (c.t + 1.0))
                delta_vec = _np.where(mask, bg_abs_vec - cand_abs, 0.0)
                c.g_vec += delta_vec
                c.G += float(delta_vec.sum())
                # g_vec is the full per-key ledger on this path: publication deltas
                # are mirrored into it above and absence credit is added here. The
                # earlier overwrite g_vec[kid] = g[kid] for published keys threw away
                # the absence credit a support key had earned on records that did
                # not carry it (measured: 21 against 38 mints on a mixed stream).
            else:
                for kid, ki in self.key_by_id.items():
                    if kid in K_r:
                        continue
                    b_abs = kt(past - ki.P, past, 2) if past > 0 else 1.0
                    delta = b_abs - kt(c.t - c.p.get(kid, 0), c.t, 2)
                    c.G += delta
                    c.g[kid] = c.g.get(kid, 0.0) + delta
            c.t += 1
            c.members.append(n)

        # --- 5. UPDATE STATISTICS ------------------------------------------------ #
        for v in S_r:
            node = self.nodes[v]
            node.t += 1
            node.members.add(n)
            self.dirty.add(v)
            for kid in node.S:
                if kid in K_r:
                    node.p[kid] = node.p.get(kid, 0) + 1
                    node.pub.setdefault(kid, set()).add(n)
        owners_n = self.record_owner.setdefault(n, {})
        for kid, vid, ki in items:
            ki.P += 1
            owner = own[kid]
            owners_n[kid] = owner
            if owner == 0:
                ki.background.observe(vid)
            else:
                node = self.nodes[owner]
                node.blocks[kid].observe(vid)
                self.ix1.setdefault((kid, vid), set()).add(owner)

        # --- 7. ESCROW RELEASE (the deferred mint) ------------------------------- #
        minted = []
        for c in touched:
            if _np is not None and c.g_vec is not None:
                gmap = {k: float(c.g_vec[k]) for k in c.keys}
                out_mask = _np.ones(len(c.g_vec), bool)
                for k in c.keys:
                    out_mask[k] = False
                gv = c.g_vec[out_mask]
                g_outside = float(gv[gv > 0].sum())
            else:
                gmap = {k: v for k, v in c.g.items() if k in c.keys}
                g_outside = sum(v for k, v in c.g.items()
                                if k not in c.keys and v > 0)
            # The UNSELECTED statistic. The candidate exists because the seed pair
            # (a+, v+) was a residual, and it accrues ONLY on records carrying that
            # pair, so on every accrual step the seed key is published with the same
            # value. Its per-key ledger entry g[a+] (value terms and presence terms
            # both) is therefore positive by construction and is not a fair bet: it
            # is the evidence that CHOSE this candidate, and the naming charge in the
            # price is what pays for that choice. Counting it again in the released
            # statistic counts it twice. With the flag on, g[a+] is dropped from the
            # quantity compared against the price; the seed key still enters the
            # minted node's support (the node is the records with a+ = v+, and its
            # cells must move), so the support term prices |T| + 1 keys.
            seed_kid = c.sig[0] if C.UNSELECTED_STATISTIC else None
            if seed_kid is not None:
                gmap.pop(seed_kid, None)
            ordered = sorted(gmap.items(), key=lambda kv: -kv[1])
            named_extra = 1 if seed_kid is not None else 0
            # The SEED NAMING CHARGE, the opposite trade to the unselected
            # statistic. The seed key's evidence is load-bearing (about 43 percent
            # of a candidate's bits), so instead of deleting it from the statistic
            # this pays for the selection on the PRICE side: the candidate must
            # also afford a prefix codeword naming its seed pair out of the
            # candidate set, which is the multiple-comparisons correction of
            # Theorem 1(iii) made explicit. Zero unless the flag is on.
            seed_charge = C.seed_naming_charge(
                c.sig[0], c.sig[1], len(self.keys),
                len(self.key_by_id[c.sig[0]].inventory))
            best_T, best_margin, running, best_running = None, 0.0, 0.0, 0.0
            for j, (kid, gk) in enumerate(ordered, start=1):
                if gk <= 0:
                    break
                running += gk
                margin = (running + g_outside
                          - price(c.t, j + named_extra, n, self.K, self.e,
                                  len(self.keys)) - seed_charge)
                if margin > best_margin:
                    best_T, best_margin, best_running = ordered[:j], margin, running
            if best_T is None:
                continue
            support = [kid for kid, _ in best_T]
            if seed_kid is not None and seed_kid not in support:
                support.append(seed_kid)
            self._mint(c, support, n,
                       released=best_running + g_outside, g_outside=g_outside)
            minted.append(self.next_id)
            self.pool.pop(c.sig, None)
        if minted and self.mint_merge_hook is not None:
            self.mint_merge_hook(minted)

        return Receipt(
            n=n, active=S_r,
            owners={self.key_name[kid]: own[kid] for kid in K_r},
            per_key_bits={self.key_name[kid]: Dcur[kid] for kid in K_r},
            minted=minted)

    # ------------------------------------------------------------------ #
    def _mint(self, c: Candidate, support: list, n: int,
              released: float = None, g_outside: float = None) -> None:
        self.K += 1
        self.e += 1
        self.next_id += 1
        w = Node(nid=self.next_id, t=c.t, birth_n=n)
        w.members = set(c.members)
        for kid in support:
            w.pub[kid] = {r for r in c.members if kid in self.record_keys.get(r, ())}
        # The node takes ownership of its members' cells for its support keys:
        # the escrow evidence was accrued against exactly those owners, so the
        # counts must move, not be copied.
        for kid in support:
            for r in w.pub.get(kid, ()):
                vid = self.record_vals.get(r, {}).get(kid)
                if vid is None:
                    continue
                prev = self.record_owner.get(r, {}).get(kid, 0)
                if prev == w.nid:
                    continue
                if prev == 0:
                    self.key_by_id[kid].background.unobserve(vid)
                else:
                    pv = self.nodes.get(prev)
                    if pv is not None and kid in pv.blocks:
                        pv.blocks[kid].unobserve(vid)
                self.record_owner.setdefault(r, {})[kid] = w.nid
        for kid in support:
            w.S.add(kid)
            w.p[kid] = len(w.pub.get(kid, ()))
            blk = c.blocks.get(kid)
            w.blocks[kid] = blk if blk is not None else ValueBlock()
            self.ix2.setdefault(kid, set()).add(w.nid)
            for vid in w.blocks[kid].counts:
                self.ix1.setdefault((kid, vid), set()).add(w.nid)
        self.nodes[w.nid] = w
        def _g(k):
            if c.g_vec is not None:
                return float(c.g_vec[k])
            return c.g.get(k, 0.0)
        # recomputed rather than passed in, so the _mint signature (and every
        # subclass that overrides it) is untouched: it depends only on the key
        # table and the seed pair, neither of which _mint changes.
        seed_charge = C.seed_naming_charge(
            c.sig[0], c.sig[1], len(self.keys),
            len(self.key_by_id[c.sig[0]].inventory))
        entry = {
            "node": w.nid, "n": n, "members": c.t,
            "G_total": round(c.G, 3),
            "released": None if released is None else round(released, 3),
            "g_outside": None if g_outside is None else round(g_outside, 3),
            "price_paid": round(price(c.t, len(support), n, self.K - 1, self.e - 1,
                                      len(self.keys)) + seed_charge, 3),
            "support": [self.key_name[k] for k in support],
            "justifying_key": self.key_name[max(support, key=_g)],
            "per_key_bits": {self.key_name[k]: round(_g(k), 3) for k in support},
            "seed": (self.key_name[c.sig[0]], c.sig[1]),
        }
        if seed_charge:                         # reported only when the flag is on,
            entry["seed_naming_charge"] = round(seed_charge, 3)   # so the shipped
        self.mint_log.append(entry)             # receipt is byte for byte the same
