"""The two moves the repair pass was missing: split a node, and mint a node out of the residual.

WHY THEY ARE NEEDED, MEASURED. `BatchObjective` offers merge, reassign and delete. On the value
fixture E75, where the groups live in the values and not in the key names, the objective prefers the
planted truth on every seed by 3,616 to 16,899 bits, so the shortfall is search and not criterion.
Two states the existing moves cannot leave:

  a node holding several true groups   reassign moves a record to a node that ALREADY EXISTS, so if
                                       the node that should hold it was never minted there is
                                       nowhere to move it to. No sequence of merges, deletions and
                                       single-record moves takes one node holding four groups to
                                       four nodes.
  two groups lying in the residual     every existing move acts on a node, so records in no node are
                                       out of reach of all of them. On seed 1 that is two whole
                                       groups and 3,616 bits.

WHAT A CANDIDATE IS. The same thing it is for the mint: a (key, value) cell. A feature is named by an
attribute value under the IBP stance, which is what makes the hypothesis space bounded, and it is
what makes the naming charge O(1) rather than growing with the stream.

WHY THE SEED IS GROWN BEFORE IT IS PRICED. A cell names a fraction of one group at one key. Pricing
the bipartition at the moment the cell is named prices a sliver, the gate refuses, and the search
stops one move short of every answer. That is precisely the mint-on-first-miss degeneracy, and it
takes the same answer: a candidate is not priced on the evidence it has when it is named. The seed is
grown by one assignment pass under each half's own Krichevsky-Trofimov predictive, and only the grown
bipartition is priced. Nothing is calibrated by this. The growth is a search proposal built from the
code the method already uses, and the accept or reject is the exact objective plus the naming charge.
Measured on E75 at full signal, over three seeds: without growth the split takes ARI 0.6099 to 0.7906
and K to 24 against a true 8; with growth it takes ARI to 0.9995 and K to exactly 8, and on one seed
it reaches the truth's own codelength to the bit.

WHAT WOULD REFUTE THE OPERATORS. A split accepted on a stream with no structure, or a delta that does
not match the change in `BatchObjective.total`. Both are unit tests.
"""
from __future__ import annotations

import collections
import math

from .batch import L_vblock
from .codes import L_col, ValueBlock


def split_naming_charge(n_candidates: int) -> float:
    """Bits to say WHICH cell named the split, out of those the operator could have proposed.

    The same argument as the key merge and as the mint. A hypothesis selected out of a set pays for
    the selection, and by Kraft that payment is log2 of the set, which is the multiple-comparisons
    correction rather than a safety margin someone chose.

    This must be `math.log2` and not `codes.lg2`, which is log2 of the Gamma function. Writing lg2
    here charged log2(64!) = 290 bits to name one of 64 cells instead of 6, and 290 bits is larger
    than most splits are worth: on the E47 cover fixture the best split of every node scored between
    +7 and +62 bits against it and the operator refused all of them, which read as the move not
    working on cover data. It was the charge, not the move."""
    return math.log2(max(n_candidates, 1))


def _owned_cells(g, members, nid):
    """kid -> [(record, value id)] for the cells these records own through node `nid`.
    `nid` 0 means the background. A record's value for a key lives in exactly one block, so
    partitioning the members partitions these cells with nothing counted twice."""
    out = collections.defaultdict(list)
    for r in members:
        own = g.record_owner.get(r, {})
        vals = g.record_vals.get(r, {})
        for kid, o in own.items():
            if o == nid and kid in vals:
                out[kid].append((r, vals[kid]))
    return out


def _counts(pairs, keep):
    c = {}
    for r, vid in pairs:
        if r in keep:
            c[vid] = c.get(vid, 0) + 1
    return c


class _Blk:
    __slots__ = ("counts",)

    def __init__(self, counts):
        self.counts = counts


def grow_seed(obj, members, seed):
    """One assignment pass: each record joins the half whose KT predictive spends fewer bits on it.

    Both halves are scored with the same 1/2 pseudocount the value blocks use everywhere else, so no
    quantity enters here that is not already in the code. The logarithm is `math.log2`; `codes.lg2`
    is log2 of the Gamma function and putting it here is the same slip that made the naming charge
    290 bits.

    Every key a member carries is read, NOT only the keys in the node's support. Restricting to the
    support blinded the operator exactly where it was needed: a record that does not belong in a node
    is, almost by definition, one carrying keys the node does not support, so those are the cells
    that identify it. Measured on the encyclopedia film node, 87 of the 98 cells carried by its 11
    misfit members lay outside its 14-key support, the best reachable proposal was +7.47 bits and was
    refused, and the split that peels those 11 off is worth -128.6 bits and takes agreement from
    0.7905 to 0.9349. The support still decides what the resulting NODES look like, in `split_delta`
    and `apply_split`; it has no business deciding what the search may look at."""
    g = obj.g
    ca = collections.defaultdict(collections.Counter)
    cb = collections.defaultdict(collections.Counter)
    na, nb = collections.Counter(), collections.Counter()
    for r in members:
        d, t = (ca, na) if r in seed else (cb, nb)
        for kid, vid in g.record_vals.get(r, {}).items():
            d[kid][vid] += 1
            t[kid] += 1

    def bits(r, d, t):
        s = 0.0
        for kid, vid in g.record_vals.get(r, {}).items():
            m = len(g.key_by_id[kid].inventory)
            s -= math.log2((d[kid].get(vid, 0) + 0.5) / (t[kid] + 0.5 * m))
        return s

    return {r for r in members if bits(r, ca, na) <= bits(r, cb, nb)}


def two_means_seed(obj, members):
    """A second proposal: a two-means bipartition of the members over their own cells.

    The seed-and-grow proposal starts from one cell, so it can only reach bipartitions that some
    single cell points at. On a cover, where a record belongs to several nodes and a node holds one
    feature rather than one kind, that is often not where the improving split lives: measured on the
    E47 cover fixture, every cell-grown split of every node costs between 7 and 102 bits, while E51's
    two-means found one worth 931. Two proposals into one gate is strictly more search and no change
    of criterion, and the wider candidate set pays for itself through the naming charge.

    Deterministic and stdlib only, so the engine keeps both properties. Cosine over the cell
    profiles, seeded by the member furthest from the whole and then the member furthest from that.
    """
    g = obj.g
    cells = {}
    for r in members:
        cells[r] = {(kid, vid) for kid, vid in g.record_vals.get(r, {}).items()}
    live = [r for r in sorted(members) if cells[r]]
    if len(live) < 4:
        return None

    def cos(a, b_prof, b_norm):
        if not a or b_norm <= 0:
            return 0.0
        dot = sum(b_prof.get(c, 0.0) for c in a)
        return dot / (math.sqrt(len(a)) * b_norm)

    def profile(rs):
        p = collections.Counter()
        for r in rs:
            for c in cells[r]:
                p[c] += 1
        n = max(len(rs), 1)
        prof = {c: k / n for c, k in p.items()}
        return prof, math.sqrt(sum(x * x for x in prof.values()))

    whole, whole_n = profile(live)
    far = min(live, key=lambda r: (cos(cells[r], whole, whole_n), r))
    fp, fn = profile([far])
    far2 = min(live, key=lambda r: (cos(cells[r], fp, fn), r))
    a = {far}
    b = {far2}
    for _ in range(10):
        pa, na = profile(a or [far])
        pb, nb = profile(b or [far2])
        na_, nb_ = set(), set()
        for r in live:
            (na_ if cos(cells[r], pa, na) >= cos(cells[r], pb, nb) else nb_).add(r)
        if not na_ or not nb_:
            return None
        if (na_, nb_) == (a, b):
            break
        a, b = na_, nb_
    a |= {r for r in members if r not in live and r in a}
    rest = set(members) - a
    if not a or not rest:
        return None
    return a


def binary_two_means(obj, members, restarts=4):
    """A third proposal: Lloyd on the binary cell indicators under Euclidean distance.

    This is the proposal E51 used through sklearn, and it reaches bipartitions the other two do not.
    Measured on the E47 cover fixture, seed 2, the largest node: E51's bipartition prices -680.4 bits
    under this gate, which the gate would accept, while the cell-grown proposals price between +7 and
    +102 and the cosine two-means prices +253.5. So the cover fixture refusal was never the criterion
    and never the pricing path, it was the proposal, and the honest fix is to propose better rather
    than to charge less.

    Deterministic and stdlib only. Seeds come from a furthest-point traversal rather than from a
    random draw, so a run reproduces. Every restart is returned and the gate picks among them, which
    is what keeps the extra search from being a second criterion.
    """
    g = obj.g
    cells = {}
    for r in sorted(members):
        c = frozenset((kid, vid) for kid, vid in g.record_vals.get(r, {}).items())
        if c:
            cells[r] = c
    live = sorted(cells)
    if len(live) < 4:
        return []

    def d2(a, b):
        return len(a) + len(b) - 2 * len(a & b)

    # furthest-point traversal: deterministic, and it spreads the seeds the way k-means++ tries to
    picked = [max(live, key=lambda r: (len(cells[r]), -r))]
    while len(picked) < min(2 * restarts, len(live)):
        far = max(live, key=lambda r: (min(d2(cells[r], cells[q]) for q in picked), -r))
        if far in picked:
            break
        picked.append(far)

    out = []
    for i in range(0, len(picked) - 1, 2):
        ca, cb = cells[picked[i]], cells[picked[i + 1]]
        a = set()
        for _ in range(12):
            na = {r for r in live if d2(cells[r], ca) <= d2(cells[r], cb)}
            nb = set(live) - na
            if not na or not nb:
                break
            if na == a:
                break
            a = na
            ca = _centroid_cells(cells, na)
            cb = _centroid_cells(cells, nb)
        if a and len(a) < len(members):
            out.append(a | (set(members) - set(live)) if False else set(a))
    return out


def _centroid_cells(cells, group):
    """The cells carried by at least half the group. The Euclidean centroid of binary vectors
    rounded back to a set, which is what keeps the distance above an integer count."""
    cnt = collections.Counter()
    for r in group:
        cnt.update(cells[r])
    half = len(group) / 2.0
    return frozenset(c for c, k in cnt.items() if k >= half)


def split_delta(obj, v, part_a) -> float | None:
    """Exact dL_batch of replacing node v by two nodes holding `part_a` and the rest.

    Both halves inherit v's support unchanged, so the union of their members and the set of nodes
    supporting each key are both exactly what they were. Coverage and publication sets are therefore
    identical and the background contributes nothing to this delta. A half that ends up supporting a
    key none of its members publish pays L_col(0, t) for it and the existing merge and delete moves
    can take it back; that cost is real and is charged here rather than assumed away."""
    g = obj.g
    part_b = v.members - part_a
    if not part_a or not part_b:
        return None
    owned = _owned_cells(g, v.members, v.nid)
    before = obj.node_local(v.t, v.S, v.p, v.blocks)
    after = 0.0
    for half in (part_a, part_b):
        p = {kid: len(v.pub.get(kid, set()) & half) for kid in v.S}
        blocks = {}
        for kid in v.S:
            c = _counts(owned.get(kid, ()), half)
            if c:
                blocks[kid] = _Blk(c)
        after += obj.node_local(len(half), v.S, p, blocks)
    return after - before + obj.structure_cost(g.K + 1) - obj.structure_cost(g.K)


def split_candidates(obj, v, cap=None):
    """Every cell present among v's members, largest first. `cap` bounds the work and never the
    criterion: it is the same efficiency bound the consensus pass already carries, and the naming
    charge is computed from the number actually considered, so a smaller pool is charged less and
    the gate stays exact for the pool it searched."""
    g = obj.g
    cells = collections.defaultdict(set)
    for r in v.members:
        for kid, vid in g.record_vals.get(r, {}).items():
            cells[(kid, vid)].add(r)          # every key a member carries, not only the support
    live = [(c, s) for c, s in cells.items() if 1 < len(s) < len(v.members) - 1]
    live.sort(key=lambda x: (-len(x[1]), x[0]))
    return live if cap is None else live[:cap]


def presence_seeds(obj, members):
    """Seeds named by carrying a key at all, rather than by carrying a particular value.

    A record that does not belong in a node is usually identified by the KEY it carries and not by
    the value: the nine person records sitting in the encyclopedia film node each carry
    `birth_place`, but each carries a different birth place, so every one of those cells has a single
    carrier and `split_candidates` drops it as a singleton. The cells that identify a misfit are
    exactly the ones a value-named proposal cannot see, which is why removing the support filter
    alone changed nothing: with every cell visible and no cap at all, still no split of that node
    paid, while peeling its eleven misfits is worth -128.6 bits.

    A key-presence seed is the natural proposal for a support-based split and costs one more entry in
    the naming charge, which the gate pays for out of the same pool as every other proposal."""
    g = obj.g
    by = collections.defaultdict(set)
    for r in members:
        for kid in g.record_keys.get(r, ()):
            by[kid].add(r)
    out = [(("key", kid), rs) for kid, rs in by.items() if 1 < len(rs) < len(members) - 1]
    out.sort(key=lambda x: (-len(x[1]), x[0][1]))
    return out


def split_proposals(obj, v, cap=None):
    """Every bipartition of v the operator will consider, and the bits to name the winner.

    Three sources, one gate. A cell-named seed grown under each half's KT predictive; a cosine
    two-means over the cell profiles; and a Lloyd on the binary cell indicators. They reach different
    bipartitions on different data shapes, and none of them is a criterion: the objective decides,
    and the charge is computed from how many were offered, so searching wider costs more to name.
    """
    live = split_candidates(obj, v, cap)
    proposals = [(cell, grow_seed(obj, v.members, carriers)) for cell, carriers in live]
    n_offered = len(live)
    pres = presence_seeds(obj, v.members)
    if cap is not None:
        pres = pres[:cap]
    for cell, carriers in pres:
        proposals.append((cell, grow_seed(obj, v.members, carriers)))
        proposals.append((("key-raw", cell[1]), set(carriers)))
        n_offered += 2
    tm = two_means_seed(obj, v.members)
    if tm is not None:
        proposals.append((("two-means", 0), tm))
        n_offered += 1
    for i, part in enumerate(binary_two_means(obj, v.members)):
        proposals.append((("binary-two-means", i), part))
        n_offered += 1
    proposals = [(c, p) for c, p in proposals if p and 0 < len(p) < len(v.members)]
    return proposals, split_naming_charge(n_offered)


def best_split(obj, v, cap=None):
    """The best split of v that any proposal reaches, priced with its naming charge."""
    proposals, naming = split_proposals(obj, v, cap)
    if not proposals:
        return None
    best, seen = None, set()
    for cell, part in proposals:
        key = frozenset(part)
        if key in seen:
            continue
        seen.add(key)
        d = split_delta(obj, v, part)
        if d is None:
            continue
        d += naming
        if d < -1e-9 and (best is None or d < best[0]):
            best = (d, part, cell)
    return best


def apply_split(obj, v, part_a) -> object:
    """Replace v by v (holding part_a) and a new node holding the rest, and move the bookkeeping."""
    from .engine import Node
    g = obj.g
    part_b = set(v.members) - set(part_a)
    owned = _owned_cells(g, v.members, v.nid)
    nid = max(g.nodes) + 1 if g.nodes else 1
    w = Node(nid=nid, birth_n=g.n)
    w.members = set(part_b)
    w.t = len(w.members)
    w.S = set(v.S)
    for kid in v.S:
        w.pub[kid] = set(v.pub.get(kid, set())) & part_b
        w.p[kid] = len(w.pub[kid])
    for kid, pairs in owned.items():
        for r, vid in pairs:
            if r not in part_b:
                continue
            blk = w.blocks.get(kid)
            if blk is None:
                blk = w.blocks[kid] = ValueBlock()
            if vid in blk.counts:
                blk.counts[vid] += 1
            else:
                blk.counts[vid] = 1
                blk.u += 1
            blk.N += 1
            g.record_owner.setdefault(r, {})[kid] = nid
            g.ix1.setdefault((kid, vid), set()).add(nid)
    # v keeps part_a: rebuild its own presence and blocks from the cells it still owns
    v.members = set(part_a)
    v.t = len(v.members)
    for kid in list(v.S):
        v.pub[kid] = set(v.pub.get(kid, set())) & v.members
        v.p[kid] = len(v.pub[kid])
        c = _counts(owned.get(kid, ()), v.members)
        if c:
            blk = ValueBlock()
            blk.counts = c
            blk.u = len(c)
            blk.N = sum(c.values())
            v.blocks[kid] = blk
        else:
            v.blocks.pop(kid, None)
    for kid in w.S:
        g.ix2.setdefault(kid, set()).add(nid)
    # v keeps its id in ix1 for cells that went to w, and that is inert rather than tidy. The only
    # reader is the candidate lookup in `engine.process`, which unions ix1 with ix2, and both halves
    # keep the parent's support, so ix2 already returns both nodes for every key either could match.
    # It is also only reached while records are arriving, which the final pass no longer overlaps.
    # If these moves are ever run mid-stream, remove v.nid from the cells it no longer owns first.
    g.nodes[nid] = w
    g.K += 1
    g.e += 1
    g.dirty.add(v.nid)
    g.dirty.add(nid)
    return w


# ---------------------------------------------------------------- residual mint -- #

def residual_records(g):
    """Records in no node. Every existing move acts on a node, so these are out of reach of all
    of them, and on E75 seed 1 that is two whole groups."""
    covered = set()
    for v in g.nodes.values():
        covered |= v.members
    return set(range(1, g.n + 1)) - covered


def residual_mint_delta(obj, members, S=None):
    """Exact dL_batch of minting one node from records currently in no node.

    The inverse of `delete_delta`: the node's own local cost and the structure increment are paid,
    and the background stops paying for the coverage, the publications and the value cells that move
    into the node."""
    g = obj.g
    if not members:
        return None, None, None
    if S is None:
        S = set()
        for r in members:
            S |= set(g.record_keys.get(r, ()))
    if not S:
        return None, None, None
    owned = _owned_cells(g, members, 0)
    p, blocks = {}, {}
    for kid in S:
        p[kid] = sum(1 for r in members if kid in g.record_keys.get(r, ()))
        c = _counts(owned.get(kid, ()), members)
        if c:
            blocks[kid] = _Blk(c)
    d = obj.node_local(len(members), S, p, blocks)
    d += obj.structure_cost(g.K + 1) - obj.structure_cost(g.K)
    for kid in S:
        ki = g.key_by_id[kid]
        covered, pubs = obj._bg_sets(kid)
        newly = members - covered
        n0b = g.n - len(covered)
        p0b = max(0, ki.P - len(pubs))
        newly_pub = sum(1 for r in newly if kid in g.record_keys.get(r, ()))
        n0a = n0b - len(newly)
        p0a = max(0, p0b - newly_pub)
        if n0b > 0:
            d -= L_col(min(p0b, n0b), n0b)
        if n0a > 0:
            d += L_col(min(p0a, n0a), n0a)
        c = _counts(owned.get(kid, ()), members)
        if c:
            naming = math.log2(len(ki.inventory) + 1.0)
            bgc = dict(ki.background.counts)
            d -= L_vblock(bgc, naming)
            for vid, k in c.items():
                bgc[vid] = bgc.get(vid, 0) - k
                if bgc[vid] <= 0:
                    del bgc[vid]
            d += L_vblock(bgc, naming)
    return d, S, blocks


def best_residual_mint(obj, cap=None):
    """The best grown node the residual can pay for, with its naming charge. None if none pays."""
    g = obj.g
    res = residual_records(g)
    if len(res) < 2:
        return None
    S = set()
    for r in res:
        S |= set(g.record_keys.get(r, ()))
    cells = collections.defaultdict(set)
    for r in res:
        for kid, vid in g.record_vals.get(r, {}).items():
            cells[(kid, vid)].add(r)
    live = [(c, s) for c, s in cells.items() if 1 < len(s) < len(res)]
    live.sort(key=lambda x: (-len(x[1]), x[0]))
    if cap is not None:
        live = live[:cap]
    if not live:
        return None
    naming = split_naming_charge(len(live))
    best, seen = None, set()
    for cell, carriers in live:
        grown = grow_seed(obj, res, carriers)
        if not grown or len(grown) == len(res):
            continue
        key = frozenset(grown)
        if key in seen:
            continue
        seen.add(key)
        d, S_new, _ = residual_mint_delta(obj, grown)
        if d is None:
            continue
        d += naming
        if d < -1e-9 and (best is None or d < best[0]):
            best = (d, grown, cell, S_new)
    return best


def apply_residual_mint(obj, members, S):
    """Mint the node and move the cells out of the background."""
    from .engine import Node
    g = obj.g
    nid = max(g.nodes) + 1 if g.nodes else 1
    w = Node(nid=nid, birth_n=g.n)
    w.members = set(members)
    w.t = len(w.members)
    w.S = set(S)
    for kid in S:
        w.pub[kid] = {r for r in members if kid in g.record_keys.get(r, ())}
        w.p[kid] = len(w.pub[kid])
    for r in members:
        own = g.record_owner.setdefault(r, {})
        for kid, vid in g.record_vals.get(r, {}).items():
            if kid not in S or own.get(kid, 0) != 0:
                continue
            bg = g.key_by_id[kid].background
            if bg.counts.get(vid, 0) <= 0:
                continue
            bg.counts[vid] -= 1
            bg.N -= 1
            if bg.counts[vid] == 0:
                del bg.counts[vid]
                bg.u -= 1
            blk = w.blocks.get(kid)
            if blk is None:
                blk = w.blocks[kid] = ValueBlock()
            if vid in blk.counts:
                blk.counts[vid] += 1
            else:
                blk.counts[vid] = 1
                blk.u += 1
            blk.N += 1
            own[kid] = nid
            g.ix1.setdefault((kid, vid), set()).add(nid)
    for kid in S:
        g.ix2.setdefault(kid, set()).add(nid)
    g.nodes[nid] = w
    g.K += 1
    g.e += 1
    g.dirty.add(nid)
    return w
