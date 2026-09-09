"""The operator version 1 does not ship: pricing two key names as one key.

WHY THIS EXISTS. The insertion process asks whether a record earns a node. The same counting asks a
second question the engine never poses: do these two key names earn separate existence? E20 scored
us on that decision and we returned one true pair against 2,105 false, because there is no code path
in the engine that answers it at all. This is that code path.

THE RULE, AND WHY IT NEEDS NOTHING SET. Merging keys a and b changes the description in both
directions. It saves on the model side, because the key inventory loses an entry and every node's
support charge L_supp(|S|, A_n) is drawn from a smaller universe, and any node that supported both
now supports one. It pays on the data side, because the two value blocks pool into one, and pooling
two distributions costs their Jensen-Shannon divergence weighted by their masses. So the criterion is
the one the rest of the system already uses: merge exactly when the total description gets shorter.

Nothing here compares the two key strings. `Shade` and `Colour` merge if the records that carry them
say the same things; `Colour` and `Weight` do not, however similar a lexical rule finds their names.
That is the decision the paper says is open, and it is the one a string gold cannot record.

THE ONE HARD CONSTRAINT, WHICH IS STRUCTURAL RATHER THAN CHOSEN. A candidate pair is rejected
outright if any record publishes both keys. A record carrying both `input_tray` and `output_tray`
is direct evidence they are two keys, not one, and no amount of distributional similarity should
overturn it. E45 found AutoPKG's own declared merges failing exactly this check. The constraint has
no level to set: either a record carries both or it does not.

COST. The delta is computed by building the merged state and scoring it with the shipped objective,
rather than by a closed form. That is O(state) per candidate and slower than a derived delta, but it
is exact by construction and cannot drift from `BatchObjective.total()`. Version 2 can derive the
delta once this is known to be right.
"""
from __future__ import annotations

import copy
import math

from .codes import L_col, ValueBlock


def cooccurring_pairs(g) -> set:
    """Key id pairs that some record publishes together. These can never be one key."""
    out = set()
    for keys in g.record_keys.values():
        ks = sorted(keys)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                out.add((ks[i], ks[j]))
    return out


def candidate_key_pairs(g, cooccur=None):
    """Pairs worth pricing: they share at least one value string and never co-occur.

    Sharing a value is a search heuristic and not part of the accept test, exactly as the candidate
    pool is for minting. Two keys whose vocabularies are disjoint cannot pool cheaply, so pricing
    them wastes the search budget; if one ever could, the accept test would still be the arbiter.
    Restricting the generator changes what is found, never what is accepted.
    """
    if cooccur is None:
        cooccur = cooccurring_pairs(g)
    by_value = {}
    for name, ki in g.keys.items():
        for v in ki.inventory:
            by_value.setdefault(v, set()).add(ki.kid)
    pairs = set()
    for kids in by_value.values():
        ks = sorted(kids)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                p = (ks[i], ks[j])
                if p not in cooccur:
                    pairs.add(p)
    return pairs


def _merged_graph(g, kid_a, kid_b):
    """A copy of g in which kid_b has been folded into kid_a. No record carries both, which the
    caller has already checked, so every per-record and per-node structure folds without collision."""
    h = copy.deepcopy(g)
    name_a, name_b = h.key_name[kid_a], h.key_name[kid_b]
    ki_a, ki_b = h.keys[name_a], h.keys[name_b]

    # the merged inventory: b's value strings take ids in a's numbering, sharing an id where the
    # string already exists, which is what makes pooling cheap when the two keys say the same things
    remap = {}
    for value, vid_b in ki_b.inventory.items():
        remap[vid_b] = ki_a.intern(value)

    def fold_block(dst: ValueBlock, src: ValueBlock) -> None:
        for vid_b, c in src.counts.items():
            vid = remap[vid_b]
            dst.counts[vid] = dst.counts.get(vid, 0) + c
        dst.N = sum(dst.counts.values())
        dst.u = len(dst.counts)

    fold_block(ki_a.background, ki_b.background)
    ki_a.P += ki_b.P
    ki_a.first_seen_n = min(ki_a.first_seen_n, ki_b.first_seen_n)

    for v in h.nodes.values():
        if kid_b not in v.S:
            continue
        v.S.discard(kid_b)
        v.S.add(kid_a)
        pub_b = v.pub.pop(kid_b, set())
        v.pub[kid_a] = set(v.pub.get(kid_a, set())) | set(pub_b)
        v.p[kid_a] = len(v.pub[kid_a])
        v.p.pop(kid_b, None)
        blk_b = v.blocks.pop(kid_b, None)
        if blk_b is not None:
            blk_a = v.blocks.get(kid_a)
            if blk_a is None:
                blk_a = ValueBlock()
                v.blocks[kid_a] = blk_a
            fold_block(blk_a, blk_b)

    for rid, vals in h.record_vals.items():
        if kid_b in vals:
            vals[kid_a] = remap[vals.pop(kid_b)]
    for rid, keys in list(h.record_keys.items()):
        if kid_b in keys:
            h.record_keys[rid] = frozenset((keys - {kid_b}) | {kid_a})
    for rid, own in h.record_owner.items():
        if kid_b in own:
            own[kid_a] = own.pop(kid_b)

    del h.keys[name_b]
    del h.key_name[kid_b]
    h.key_by_id = {ki.kid: ki for ki in h.keys.values()}
    return h


def spelling_column_bits(g, kid_a, kid_b) -> float:
    """What the merged description still owes: which of the two spellings each record used.

    This term is not decoration and leaving it out inverts the operator. Folding two keys into one
    makes the merged key's presence column denser, and one dense column is cheaper than two sparse
    ones by a wide margin: on a fixture with two keys on 256 records each out of 912, the background
    presence term alone falls by 665 bits. That saving is available whatever the two keys say,
    because it comes from the shape of the column and not from the values, so without this term the
    operator merges every pair of keys it is offered, including keys with disjoint vocabularies, and
    the false merges get *worse* as the keys get commoner (E55, first run).

    The reason is that the merged state was not a description of the same data. A reader given it
    cannot tell whether record 7 said `shade` or `colour`, so it cannot reconstruct the input, and a
    shorter description that loses the data is not a shorter description. Charging the choice of
    spelling restores the comparison: it costs a column of length n_a + n_b carrying n_a ones, which
    very nearly cancels the presence saving, and what is left to decide the merge is the value-block
    pooling cost. That is the term that actually knows whether the two keys say the same things.
    """
    per_value = {}
    symbols = {kid: {vid: value for value, vid in g.keys[g.key_name[kid]].inventory.items()}
               for kid in (kid_a, kid_b)}
    for rid, keys in g.record_keys.items():
        which = 0 if kid_a in keys else (1 if kid_b in keys else None)
        if which is None:
            continue
        vals = g.record_vals.get(rid, {})
        kid = kid_a if which == 0 else kid_b
        # Value identifiers are local to each key. Only the actual symbol is shared.
        value = symbols[kid][vals[kid]]
        slot = per_value.setdefault(value, [0, 0])
        slot[which] += 1
    if not per_value:
        return 0.0
    # Conditioning on the value is not an optimisation, it is the difference between two claims.
    # Charging log2 of the whole column says the spelling is independent of what was said, which is
    # true when both keys draw on the same vocabulary: `Size` and `size.` both take S, M, L, so
    # knowing the value tells you nothing about which spelling produced it and the merge must pay in
    # full. It is false when the vocabularies are disjoint: if `shade` only ever says `red` and
    # `colour` only ever says `crimson`, the value already determines the spelling and a code that
    # charged for it twice would be refusing merges it has already paid to detect.
    return sum(L_col(min(a, b), a + b) for a, b in per_value.values())


# The evidence two keys can offer against being merged is about one bit per observation, since the
# most their value distributions can disagree is completely. The saving from merging is the model
# side: one entry leaves the key inventory. That saving does not grow with the data, so below some
# total record count the saving always wins and the merge is taken for reasons that have nothing to
# do with what the two keys mean.
#
# E62 measures where that is. Sweeping the two keys independently, the margin is about
# (n_a + n_b) minus a constant of three to five bits, the crossover sits at n_a + n_b of about six,
# and it does not move with the schema size: identical margins at 6, 20 and 60 filler keys. The
# fixture prediction then holds on real data. On the encyclopedia stream, bucketed by n_a + n_b:
# 23 of 23 pairs merge at 2, 45 of 86 merge at 3 to 6, and **0 of 391 merge at 7 or above**.
#
# This is Proposition 1 in different clothes. A decision has to be affordable before it can be made,
# and the price says which those are in advance. Below the scope the operator does not return a
# merge nobody can defend; it declines, which is what the rest of the system does when the evidence
# is not there.
MERGE_SCOPE_MIN_RECORDS = 7


def merge_naming_charge(g, cooccur=None) -> float:
    "The bits it costs to say WHICH pair of keys is being merged.\n\n    This is the term the operator was missing, and leaving it out is the same mistake the paper\n    attacks everywhere else: a hypothesis selected out of a large set has to pay for the\n    selection. Section 4 already makes the argument for minting, that because the hypotheses are\n    indexed by seed pairs and the prices satisfy Kraft, the price is itself the\n    multiple-comparisons correction. A key merge is a hypothesis too, drawn from the set of key\n    pairs the operator could have proposed, and by Kraft naming the winner costs log2 of that set.\n\n    Why it matters, measured. On 1,500 Lazada listings 86,767 pairs clear the structural checks,\n    so naming one costs about 16.4 bits. The nine merges the operator accepted there without this\n    term all scored exactly -0.5 bits, and the closest rejected scored +0.1 to +0.3: the whole\n    decision band was under one bit wide, which is not a decision but the sign of a rounding-scale\n    quantity. All nine were plainly wrong to a reader, print_speed with scent, motor_type with\n    Title. Sixteen bits of naming charge removes every one and needs no constant.\n\n    This also retires MERGE_SCOPE_MIN_RECORDS as the guard. That constant was read off a symmetric\n    fixture sweep, validated on the encyclopedia stream, and then failed to transfer to Lazada,\n    where every false merge sat exactly on its boundary. A derived price is what the rest of the\n    system uses and what this should have used from the start."
    # Cached on the graph. The co-occurrence set is O(records times keys squared) to build, and the
    # charge is the same for every pair of a given state, so recomputing it inside the pricing loop
    # turned a minutes-long sweep into an unfinishable one.
    cached = getattr(g, "_merge_naming_charge", None)
    if cached is not None and cooccur is None:
        return cached
    if cooccur is None:
        cooccur = cooccurring_pairs(g)
    n = len(g.key_by_id)
    total = n * (n - 1) // 2 - len(cooccur)
    charge = math.log2(max(total, 1))
    try:
        g._merge_naming_charge = charge
    except AttributeError:                      # a slotted graph: recompute rather than fail
        pass
    return charge


def merge_is_in_scope(g, kid_a, kid_b):
    """Whether the two keys carry enough records between them for the merge to be decidable."""
    na = sum(1 for keys in g.record_keys.values() if kid_a in keys)
    nb = sum(1 for keys in g.record_keys.values() if kid_b in keys)
    return (na + nb) >= MERGE_SCOPE_MIN_RECORDS, na, nb


def key_merge_delta(g, objective_cls, kid_a, kid_b, decompose=False, enforce_scope=True):
    """Bits saved by treating the two keys as one. Negative means the merge is the shorter
    description and should be taken.

    Returns None when the pair cannot be decided: either a record publishes both keys, which is
    direct evidence they are two keys, or the two together carry fewer records than the scope
    condition needs. Pass enforce_scope=False to price a pair anyway, which is what E62 does to
    measure where the scope is.
    """
    a, b = (kid_a, kid_b) if kid_a < kid_b else (kid_b, kid_a)
    for keys in g.record_keys.values():
        if a in keys and b in keys:
            return None
    if enforce_scope:
        in_scope, _, _ = merge_is_in_scope(g, a, b)
        if not in_scope:
            return None
        charge = merge_naming_charge(g)
    else:
        charge = 0.0
    before = objective_cls(g).total()
    h = _merged_graph(g, a, b)
    spelling = spelling_column_bits(g, a, b)
    after = objective_cls(h).total() + spelling + charge
    if decompose:
        return {"delta_bits": after - before, "before": before,
                "after_without_spelling": after - spelling - charge,
                "spelling_column": spelling, "naming_charge": charge}
    return after - before


def propose_key_merges(g, objective_cls, budget=None, verbose=False):
    """Every candidate pair, priced. Returns the accepted ones first, then the rest, so a caller can
    see how close the rejected pairs came rather than only the verdict."""
    cooccur = cooccurring_pairs(g)
    pairs = sorted(candidate_key_pairs(g, cooccur))
    if budget is not None:
        pairs = pairs[:budget]
    rows = []
    for a, b in pairs:
        d = key_merge_delta(g, objective_cls, a, b)
        if d is None:
            continue
        rows.append({"a": g.key_name[a], "b": g.key_name[b], "delta_bits": round(d, 2),
                     "merge": d < 0})
        if verbose:
            print(f"    {g.key_name[a]!r} + {g.key_name[b]!r}: {d:+.1f} bits", flush=True)
    rows.sort(key=lambda r: r["delta_bits"])
    return {"candidates_priced": len(rows),
            "cooccurring_pairs_excluded": len(cooccur),
            "accepted": [r for r in rows if r["merge"]],
            "all": rows}
