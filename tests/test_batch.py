"""Batch-objective and engine-consistency gates (paper unit-test suite, batch layer).

Every repair move is an exact evaluation of L_batch, so every reported delta must
equal total(after) - total(before) on an applied copy; the numpy fast path in the
engine must be invisible; the fast repair must be invisible; the release receipt
must reconstruct the release. Two defects fixed on 2026-09-05 are guarded here:
  (1) engine.py numpy path: absence credit lived in c.g_vec and publication
      evidence in c.g, and g_vec was overwritten for published keys, losing the
      absence credit a support key earned on records that did not carry it
      (test_numpy_path_equals_pure_path; the shared-key stream is the sensitive
      fixture: with the old code it minted 29 against 30 and the totals differed).
  (2) batch.py merge accounting for records that belong to two nodes: per-node
      counts were summed where unions were required, mispricing accepted merges
      (test_merge_delta_exact_for_overlapping_members, which asserts that at
      least one overlapping pair is actually evaluated, else it fails as vacuous).
      The fix that stands prices background presence through BatchObjective._bg_sets
      (unions of member and publication sets over the nodes supporting a key).
test_cells_coded_once is the open finding of the same night: the mint path leaves
every (record, key) cell of a minted member in two value blocks (see its docstring).
Plain functions, no pytest; run this file to get PASS/FAIL per test.
"""
import copy
import glob
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import escrow.engine as engine_mod
from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective
from escrow.fastrepair import FastBatchObjective
from escrow.protocol import run_stream, REPAIR_EVERY
from escrow.codes import price

RESULTS = os.path.join(HERE, "..", "..", "results")
TOL = 1e-6


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
K_STAR, KEYS_PER_GROUP, D = 8, 3, 8            # copied from experiments/e13_creation_bias.py


def planted_stream(T=3000, noise=0.1, seed=7):
    """Eight planted groups, three keys each, `noise` of the slots carry another
    group's value alphabet. Verbatim generator of e13 (not imported: e13 runs on
    import)."""
    rng = random.Random(seed)
    out = []
    for _ in range(T):
        grp = rng.randrange(K_STAR)
        rec = {}
        for j in range(KEYS_PER_GROUP):
            if rng.random() < noise:
                og = rng.randrange(K_STAR)
                rec[f"c{grp}k{j}"] = f"c{og}v{rng.randrange(D)}"
            else:
                rec[f"c{grp}k{j}"] = f"c{grp}v{rng.randrange(D)}"
        out.append(rec)
    return out


def shared_key_stream(T=2000, seed=11, frac2=0.15, d=8):
    """Three groups. Every record carries three SHARED keys s0..s2 with a shared
    value alphabet, plus its own group's three keys; `frac2` of the records also
    carry a second group's three keys. This is the fixture on which records end
    up as members of two nodes (the overlapping-member case of the merge move)."""
    rng = random.Random(seed)
    out = []
    for _ in range(T):
        grp = rng.randrange(3)
        rec = {f"s{j}": f"sv{rng.randrange(d)}" for j in range(3)}
        for j in range(3):
            rec[f"g{grp}k{j}"] = f"g{grp}v{rng.randrange(d)}"
        if rng.random() < frac2:
            og = (grp + 1 + rng.randrange(2)) % 3
            for j in range(3):
                rec[f"g{og}k{j}"] = f"g{og}v{rng.randrange(d)}"
        out.append(rec)
    return out


def wiki_stream():
    """The Wikipedia demo stream: the three cached categories, sorted glob,
    shuffled with random.Random(0) exactly as experiments/wikipedia_demo.py does.
    Returns None when the cache is absent."""
    files = sorted(glob.glob(os.path.join(RESULTS, "wiki_cache.*.json")))
    if not files:
        return None
    recs = []
    for f in files:
        with open(f) as fh:
            recs += json.load(fh)
    random.Random(0).shuffle(recs)
    return recs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def mint_view(g):
    return [(m["n"], m["members"], frozenset(m["support"])) for m in g.mint_log]


def partition(g, remap=None):
    """Sorted member sets of the nodes; `remap` maps arrival index -> record id."""
    f = (lambda r: r) if remap is None else (lambda r: remap[r])
    return sorted(sorted(f(r) for r in v.members) for v in g.nodes.values())


def support_names(g, v):
    return frozenset(g.key_name[k] for k in v.S)


def drive(records, every=REPAIR_EVERY, objective_cls=BatchObjective, graph=None,
          install=True, at_repair=None, final_full=True, stop_after=None):
    """The protocol loop (engine + objective, repair every `every`, final full
    pass) with a hook `at_repair(g, b)` invoked BEFORE each periodic repair, so a
    test can inspect the pre-repair state the repair pass will price."""
    g = graph if graph is not None else EscrowGraph()
    b = objective_cls(g)
    if install:
        b.install()
    for i, rec in enumerate(records):
        if stop_after is not None and i >= stop_after:
            break
        g.process(rec)
        if (i + 1) % every == 0:
            if at_repair is not None:
                at_repair(g, b)
            b.repair()
    if final_full:
        if at_repair is not None:
            at_repair(g, b)
        b.repair(full=True)
    return g, b


def applied_total(g, apply):
    """total() after `apply(g_copy, b_copy)` on a deepcopy of the graph."""
    g2 = copy.deepcopy(g)
    b2 = BatchObjective(g2)
    apply(g2, b2)
    return b2.total()


# --------------------------------------------------------------------------- #
# 1. the numpy fast path is invisible
# --------------------------------------------------------------------------- #
def test_numpy_path_equals_pure_path():
    """Same stream through run_stream with numpy present and with engine._np set
    to None: K, mint count, mint log (n, members, support) and total() agree.
    The pure-python loop is the definition; the vector path must reproduce it.
    Guards defect (1): with the old g_vec overwrite the shared-key stream minted
    29 against 30 on the two paths and the totals differed."""
    if engine_mod._np is None:
        print("  note: numpy not installed, test_numpy_path_equals_pure_path skipped")
        return
    for name, recs in (("planted", planted_stream()), ("shared", shared_key_stream())):
        g_np, b_np = run_stream(recs)
        saved = engine_mod._np
        engine_mod._np = None
        try:
            g_py, b_py = run_stream(recs)
        finally:
            engine_mod._np = saved
        assert g_np.K == g_py.K, (name, g_np.K, g_py.K)
        assert len(g_np.mint_log) == len(g_py.mint_log), \
            (name, len(g_np.mint_log), len(g_py.mint_log))
        assert mint_view(g_np) == mint_view(g_py), (name, mint_view(g_np), mint_view(g_py))
        assert abs(b_np.total() - b_py.total()) < TOL, (name, b_np.total(), b_py.total())
        # the two paths must also agree on the per-key ledger that the receipts expose
        for a, p in zip(g_np.mint_log, g_py.mint_log):
            assert a["per_key_bits"] == p["per_key_bits"], (name, a, p)
            assert a["g_outside"] == p["g_outside"], (name, a, p)
        assert len(g_np.mint_log) > 0, name


# --------------------------------------------------------------------------- #
# 2. merge_delta is exact, overlapping and disjoint members alike
# --------------------------------------------------------------------------- #
def test_merge_delta_exact_for_overlapping_members():
    """At every repair point of the shared-key stream (and after the final
    pass), for every pair of nodes: merge_delta(v, w) equals total(after) minus
    total(before) with _apply_merge applied on a deepcopy, to 1e-6 bits. Pairs
    whose member sets intersect are counted; the test is vacuous and FAILS if no
    such pair is ever evaluated."""
    stats = {"overlap": 0, "disjoint": 0, "worst_overlap": 0.0, "worst_disjoint": 0.0,
             "max_shared": 0}

    def check(g, b):
        before = b.total()
        nodes = sorted(g.nodes.values(), key=lambda x: x.nid)
        for v, w in itertools.combinations(nodes, 2):
            d = b.merge_delta(v, w)
            assert d is not None and math.isfinite(d), (g.n, v.nid, w.nid, d)
            after = applied_total(
                g, lambda g2, b2: b2._apply_merge(g2.nodes[v.nid], g2.nodes[w.nid]))
            err = abs((after - before) - d)
            shared = len(v.members & w.members)
            kind = "overlap" if shared else "disjoint"
            stats[kind] += 1
            stats["worst_" + kind] = max(stats["worst_" + kind], err)
            stats["max_shared"] = max(stats["max_shared"], shared)
            assert err < TOL, (f"n={g.n} merge {v.nid}+{w.nid} ({kind}, shared={shared}, "
                               f"S_v&S_w={len(v.S & w.S)}): delta={d:.6f} "
                               f"actual={after - before:.6f} err={err:.3e}")

    drive(shared_key_stream(), at_repair=check)
    assert stats["overlap"] >= 1, f"vacuous: no overlapping-member pair evaluated ({stats})"
    print(f"  merge pairs: overlap={stats['overlap']} (worst {stats['worst_overlap']:.1e}, "
          f"max shared members {stats['max_shared']}), "
          f"disjoint={stats['disjoint']} (worst {stats['worst_disjoint']:.1e})")


# --------------------------------------------------------------------------- #
# 3. delete, reassign and cohort deltas are exact
# --------------------------------------------------------------------------- #
class _CohortSpy(BatchObjective):
    """Records, for every cohort cohort_reassign applies, its internal joint delta
    d and the totals just before and just after the m applications. The delta is
    a local of cohort_reassign, read off the caller frame at the first apply."""

    def __init__(self, graph):
        super().__init__(graph)
        self.cohorts = []                      # (n, nid, m, d, before, after)
        self._pending = None

    def _apply_reassign(self, rec_n, v):
        f = sys._getframe(1)
        if f.f_code is BatchObjective.cohort_reassign.__code__:
            if self._pending is None:
                self._pending = [self.g.n, v.nid, f.f_locals["m"], f.f_locals["d"],
                                 self.total(), 0]
            self._pending[5] += 1
        super()._apply_reassign(rec_n, v)
        if self._pending is not None and self._pending[5] == self._pending[2]:
            n, nid, m, d, before, _ = self._pending
            self.cohorts.append((n, nid, m, d, before, self.total()))
            self._pending = None


def test_delete_and_reassign_deltas_exact():
    """At every repair point of the shared-key stream: delete_delta(v) against
    _apply_delete for every node; reassign_gain(rec, v) against _apply_reassign
    for a seeded sample of (uncovered record, node sharing a key) pairs; and the
    joint delta cohort_reassign computes for each cohort it applies against the
    total before and after the applications. All to 1e-6 bits."""
    rng = random.Random(3)
    stats = {"delete": 0, "reassign": 0, "worst_delete": 0.0, "worst_reassign": 0.0,
             "reassign_pay": 0}

    def check(g, b):
        before = b.total()
        nodes = sorted(g.nodes.values(), key=lambda x: x.nid)
        for v in nodes:
            d = b.delete_delta(v)
            after = applied_total(g, lambda g2, b2: b2._apply_delete(g2.nodes[v.nid]))
            err = abs((after - before) - d)
            stats["delete"] += 1
            stats["worst_delete"] = max(stats["worst_delete"], err)
            assert err < TOL, (f"n={g.n} delete {v.nid} (t={v.t}): delta={d:.6f} "
                               f"actual={after - before:.6f} err={err:.3e}")
        covered = set().union(*(v.members for v in nodes)) if nodes else set()
        pairs = [(r, v) for r in g.record_keys if r not in covered
                 for v in nodes if v.S & g.record_keys[r]]
        for r, v in rng.sample(pairs, min(12, len(pairs))):
            d = b.reassign_gain(r, v)
            after = applied_total(g, lambda g2, b2: b2._apply_reassign(r, g2.nodes[v.nid]))
            err = abs((after - before) - d)
            stats["reassign"] += 1
            stats["reassign_pay"] += d < 0
            stats["worst_reassign"] = max(stats["worst_reassign"], err)
            assert err < TOL, (f"n={g.n} reassign rec {r} -> {v.nid}: delta={d:.6f} "
                               f"actual={after - before:.6f} err={err:.3e}")

    g, b = drive(shared_key_stream(), objective_cls=_CohortSpy, at_repair=check)
    assert stats["delete"] > 0 and stats["reassign"] > 0, stats
    worst_cohort = 0.0
    for n, nid, m, d, before, after in b.cohorts:
        err = abs((after - before) - d)
        worst_cohort = max(worst_cohort, err)
        assert d < -1e-9, (n, nid, m, d)
        assert err < TOL, (f"n={n} cohort of {m} -> node {nid}: delta={d:.6f} "
                           f"actual={after - before:.6f} err={err:.3e}")
    print(f"  deletes={stats['delete']} (worst {stats['worst_delete']:.1e}); "
          f"reassigns={stats['reassign']}, {stats['reassign_pay']} paying "
          f"(worst {stats['worst_reassign']:.1e}); cohorts applied={len(b.cohorts)} "
          f"(worst {worst_cohort:.1e})")
    if not b.cohorts:
        print("  note: no cohort was applied on this fixture; the cohort identity is untested here")


# --------------------------------------------------------------------------- #
# 4. the fast repair is invisible
# --------------------------------------------------------------------------- #
def test_fastrepair_equals_batch():
    """run_stream with BatchObjective and with FastBatchObjective: K, mint count,
    node member sets and total() identical (to 1e-6). Both fixtures, so the
    overlapping-member coverage sets of the fast pass are exercised too."""
    for name, recs in (("planted", planted_stream()), ("shared", shared_key_stream())):
        g1, b1 = run_stream(recs, objective_cls=BatchObjective)
        g2, b2 = run_stream(recs, objective_cls=FastBatchObjective)
        assert g1.K == g2.K, (name, g1.K, g2.K)
        assert len(g1.mint_log) == len(g2.mint_log), (name, len(g1.mint_log), len(g2.mint_log))
        assert mint_view(g1) == mint_view(g2), name
        assert partition(g1) == partition(g2), (name, partition(g1), partition(g2))
        assert abs(b1.total() - b2.total()) < TOL, (name, b1.total(), b2.total())
        assert g1.K > 0, name


# --------------------------------------------------------------------------- #
# 5. incremental repair against full repair
# --------------------------------------------------------------------------- #
def test_incremental_repair_matches_full():
    """From a mid-stream state with mutated (dirty) nodes, two deepcopies: one is
    repaired with repair(full=False) until it reports no moves, the other with
    repair(full=True) until it reports no moves.

    What is asserted and why (from batch.repair): every move is accepted iff its
    exact delta is < -1e-9, so total() is non-increasing across every call of
    either kind; that is asserted for each call. The incremental pass proposes a
    SUBSET of the full pass's moves (merges and deletes must touch a dirty node,
    reassignment visits new records or records sharing a key with a dirty node),
    so the full chain must end at least as low as the incremental chain, up to
    1e-6; that is asserted as full_total <= inc_total + 1e-6 (equality when the
    subset was already everything). A full pass applied on top of the incremental
    fixpoint must not raise the total either. Nothing stronger is provable: both
    are greedy descents and can stop at different local optima."""
    for name, recs in (("shared", shared_key_stream()), ("planted", planted_stream())):
        # Snapshot the pre-repair state at every periodic repair point; keep the
        # latest one at which the repair actually applied moves AND only a strict
        # subset of the nodes was dirty (the early points have every node dirty,
        # so incremental and full coincide there and would prove nothing).
        g = EscrowGraph()
        b = BatchObjective(g).install()
        snap = None
        for i, rec in enumerate(recs):
            g.process(rec)
            if (i + 1) % REPAIR_EVERY == 0:
                pre = copy.deepcopy(g)
                if b.repair() > 0 and pre.dirty and len(pre.dirty) < len(pre.nodes):
                    snap = pre
        assert snap is not None, (name, "no repair point with moves and a strict dirty subset")
        g = snap
        g_inc, g_full = copy.deepcopy(g), copy.deepcopy(g)
        b_inc, b_full = BatchObjective(g_inc), BatchObjective(g_full)
        t_inc = [b_inc.total()]
        for _ in range(20):
            moves = b_inc.repair(full=False)
            t_inc.append(b_inc.total())
            assert t_inc[-1] <= t_inc[-2] + TOL, (name, "incremental raised total", t_inc)
            if moves == 0:
                break
        t_full = [b_full.total()]
        for _ in range(20):
            moves = b_full.repair(full=True)
            t_full.append(b_full.total())
            assert t_full[-1] <= t_full[-2] + TOL, (name, "full raised total", t_full)
            if moves == 0:
                break
        assert abs(t_inc[0] - t_full[0]) < TOL
        inc_total, full_total = t_inc[-1], t_full[-1]
        assert full_total <= inc_total + TOL, \
            (name, f"full ended higher than incremental: full={full_total:.6f} inc={inc_total:.6f}")
        extra = b_inc.repair(full=True)
        assert b_inc.total() <= inc_total + TOL, (name, b_inc.total(), inc_total)
        print(f"  {name}: snapshot n={g.n} K={g.K} dirty={len(g.dirty)}/{len(g.nodes)}: "
              f"start={t_inc[0]:.3f} inc={inc_total:.3f} ({len(t_inc) - 1} calls) "
              f"full={full_total:.3f} ({len(t_full) - 1} calls) gap={inc_total - full_total:.3f}; "
              f"full pass on the incremental fixpoint: {extra} more moves, K={g_inc.K} vs {g_full.K}")


# --------------------------------------------------------------------------- #
# 6. L_batch is a function of the state, not of its labels or arrival order
# --------------------------------------------------------------------------- #
def test_lbatch_order_invariant():
    """batch.total() reads only counts: node t, S, p, blocks, members, the keys'
    P, inventory sizes and background blocks. Provable: relabelling nodes and
    reordering every dict leaves it unchanged (only the float summation order
    moves, so 1e-9). Asserted on the final shared-key graph with node ids
    permuted, node dict order shuffled, ix1/ix2 remapped, and every value block's
    count dict rebuilt in shuffled order. The arrival-order form is reported, not
    asserted: the greedy insertion path can reach different states, and even the
    same member partition can carry different value-block ownership."""
    g, b = drive(shared_key_stream())
    base = b.total()
    rng = random.Random(5)
    g2 = copy.deepcopy(g)
    old_ids = sorted(g2.nodes)
    new_ids = [1000 + i for i in range(len(old_ids))]
    rng.shuffle(new_ids)
    perm = dict(zip(old_ids, new_ids))
    nodes = {}
    for nid in rng.sample(old_ids, len(old_ids)):
        v = g2.nodes[nid]
        v.nid = perm[nid]
        v.blocks = {k: v.blocks[k] for k in rng.sample(list(v.blocks), len(v.blocks))}
        for blk in v.blocks.values():
            items = list(blk.counts.items())
            rng.shuffle(items)
            blk.counts = dict(items)
        v.p = {k: v.p[k] for k in rng.sample(list(v.p), len(v.p))}
        nodes[v.nid] = v
    g2.nodes = nodes
    g2.ix1 = {k: {perm.get(i, i) for i in s} for k, s in g2.ix1.items()}
    g2.ix2 = {k: {perm.get(i, i) for i in s} for k, s in g2.ix2.items()}
    for ki in g2.key_by_id.values():
        items = list(ki.background.counts.items())
        rng.shuffle(items)
        ki.background.counts = dict(items)
    relabelled = BatchObjective(g2).total()
    assert abs(relabelled - base) < 1e-9, (base, relabelled)
    assert g2.K == g.K and len(g2.nodes) == len(g.nodes)

    # arrival-order form: 20 random orders of the same records, fresh engines
    recs = planted_stream(1500)
    groups = {}
    for trial in range(20):
        order = list(range(len(recs)))
        random.Random(100 + trial).shuffle(order)
        gg, bb = run_stream([recs[i] for i in order])
        remap = {pos + 1: order[pos] for pos in range(len(order))}
        key = json.dumps(partition(gg, remap))
        groups.setdefault(key, []).append(bb.total())
    spreads = [max(ts) - min(ts) for ts in groups.values() if len(ts) > 1]
    print(f"  relabel diff={abs(relabelled - base):.1e}; arrival-order form: "
          f"{len(groups)} distinct partitions in 20 orders, "
          f"within-partition total spread max={max(spreads) if spreads else 0.0:.3e}")


# --------------------------------------------------------------------------- #
# 7. determinism across hash seeds
# --------------------------------------------------------------------------- #
def _hash_seed_worker():
    recs = wiki_stream()
    if recs is None:
        print(json.dumps(None))
        return
    g, b = run_stream(recs)
    print(json.dumps({"K": g.K, "n": g.n,
                      "mints": [(m["n"], m["members"], sorted(m["support"])) for m in g.mint_log],
                      "total": b.total()}))


def test_determinism_across_hash_seeds():
    """The Wikipedia cache stream in three subprocesses with PYTHONHASHSEED 0, 1
    and 12345: K and the mint log (n, members, support) must be identical. A set
    of strings iterated anywhere on the decision path would show up here."""
    if wiki_stream() is None:
        print("  note: results/wiki_cache.*.json not found, test_determinism_across_hash_seeds skipped")
        return
    outs = {}
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--hash-seed-worker"],
                           env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, (seed, p.stderr[-2000:])
        outs[seed] = json.loads(p.stdout.strip().splitlines()[-1])
        assert outs[seed] is not None, seed
    ref = outs["0"]
    for seed, o in outs.items():
        assert o["K"] == ref["K"], (seed, o["K"], ref["K"])
        assert o["mints"] == ref["mints"], (seed, o["mints"][:3], ref["mints"][:3])
        assert abs(o["total"] - ref["total"]) < TOL, (seed, o["total"], ref["total"])
    assert ref["K"] > 0 and len(ref["mints"]) > 0, ref
    print(f"  wiki n={ref['n']} K={ref['K']} mints={len(ref['mints'])} identical under 3 hash seeds")


# --------------------------------------------------------------------------- #
# 8. the receipt reconstructs the release
# --------------------------------------------------------------------------- #
class _ReceiptSpy(EscrowGraph):
    """Captures the UNROUNDED release quantities the engine hands to _mint, plus
    the price it paid (pre-mint K, e and key universe), so the receipt identity
    can be checked exactly; the logged receipt rounds each number to 3 decimals."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.raw = []

    def _mint(self, c, support, n, released=None, g_outside=None):
        if c.g_vec is not None:
            per_key = {k: float(c.g_vec[k]) for k in support}
        else:
            per_key = {k: c.g.get(k, 0.0) for k in support}
        paid = price(c.t, len(support), n, self.K, self.e, len(self.keys))
        self.raw.append({"released": released, "g_outside": g_outside,
                         "per_key": per_key, "price": paid, "n": n})
        super()._mint(c, support, n, released=released, g_outside=g_outside)


def test_receipt_reconstructs_release():
    """Every mint_log entry carries released and g_outside; released - price_paid
    is strictly positive; released equals the sum of per_key_bits over the
    support plus g_outside. Exact (1e-9) on the unrounded values captured at
    _mint; on the logged, 3-decimal-rounded values the identity holds within the
    rounding bound (|support| + 2) * 5e-4, which exceeds 1e-3 for supports of
    more than two keys (measured on the shared-key stream: 2.0e-3 with 9-key
    supports), so 1e-3 is asserted only where the bound allows it."""
    n_mints, worst_exact, worst_logged, min_margin = 0, 0.0, 0.0, math.inf
    for recs in (planted_stream(), shared_key_stream()):
        g, b = drive(recs, graph=_ReceiptSpy())
        assert len(g.mint_log) == len(g.raw) and g.mint_log, (len(g.mint_log), len(g.raw))
        for m, raw in zip(g.mint_log, g.raw):
            n_mints += 1
            assert "released" in m and "g_outside" in m, m
            assert m["released"] is not None and m["g_outside"] is not None, m
            # exact identity on the unrounded values
            recon = sum(raw["per_key"].values()) + raw["g_outside"]
            worst_exact = max(worst_exact, abs(recon - raw["released"]))
            assert abs(recon - raw["released"]) < 1e-9, (m, recon, raw)
            assert raw["released"] - raw["price"] > 0, (m, raw)
            assert raw["g_outside"] >= 0.0, raw
            min_margin = min(min_margin, raw["released"] - raw["price"])
            assert abs(m["price_paid"] - raw["price"]) <= 5e-4 + 1e-9, (m, raw)
            # the logged receipt, within its own rounding
            logged = sum(m["per_key_bits"].values()) + m["g_outside"]
            err = abs(logged - m["released"])
            worst_logged = max(worst_logged, err)
            bound = (len(m["support"]) + 2) * 5e-4 + 1e-9
            assert err <= bound, (m, err, bound)
            if bound <= 1e-3:
                assert err <= 1e-3, (m, err)
            assert m["released"] - m["price_paid"] > -1e-3, m
            assert set(m["per_key_bits"]) == set(m["support"]), m
    print(f"  {n_mints} mints: exact identity worst={worst_exact:.1e}, logged (rounded) "
          f"worst={worst_logged:.1e}, min released-price={min_margin:.3f} bits")


# --------------------------------------------------------------------------- #
# 9. install(): the mint-time absorption gate
# --------------------------------------------------------------------------- #
def test_install_positive_control():
    """Shared-key stream with and without install(). The gate's purpose is to
    absorb a fresh mint into an existing node whose support contains (or is
    contained in) its own, before siblings accumulate. Asserted: the hook is
    wired and fires; and after the final full repair every support-contained
    pair that survives has merge_delta > 0, i.e. the objective itself refuses
    the merge (a surviving contained pair with a paying merge would be a repair
    defect). Whether any contained pair survives at all is REPORTED: on this
    fixture two nodes with identical 9-key supports but different value
    distributions coexist both with and without the gate (merge_delta of order
    +3700 bits), so 'no contained pairs' is not what the gate delivers here."""
    fired = {"calls": 0, "absorbed": 0}

    class _Counting(BatchObjective):
        def absorb_new_mints(self, new_ids):
            fired["calls"] += 1
            k = super().absorb_new_mints(new_ids)
            fired["absorbed"] += k
            return k

    recs = shared_key_stream()
    out = {}
    for inst in (True, False):
        g, b = drive(recs, objective_cls=_Counting, install=inst)
        if inst:
            assert g.mint_merge_hook is not None
        else:
            assert g.mint_merge_hook is None
        nodes = sorted(g.nodes.values(), key=lambda x: x.nid)
        contained = [(v, w) for v, w in itertools.combinations(nodes, 2)
                     if v.S <= w.S or w.S <= v.S]
        for v, w in contained:
            d = b.merge_delta(v, w)
            assert d > 1e-9, (f"install={inst}: contained pair {v.nid},{w.nid} survives the "
                              f"final full repair with a paying merge_delta={d:.3f}")
        out[inst] = dict(K=g.K, mints=len(g.mint_log), contained=len(contained),
                         sizes=[v.t for v in nodes],
                         deltas=[round(b.merge_delta(v, w), 1) for v, w in contained])
    assert fired["calls"] > 0, "install() did not wire the absorption hook"
    print(f"  with install: {out[True]} (hook fired {fired['calls']} times, absorbed "
          f"{fired['absorbed']} mints); without: {out[False]}")
    if out[True]["contained"]:
        print(f"  note: {out[True]['contained']} support-contained pair(s) survive with "
              f"install(); merge_delta {out[True]['deltas']} bits, so the objective refuses them")


# --------------------------------------------------------------------------- #
# 10. candidate pool eviction under maximal pressure
# --------------------------------------------------------------------------- #
def test_pool_eviction_keeps_touched_candidate():
    """EscrowGraph(cand_pool_cap=1) puts the eviction rule on every record: no
    exception, the pool never exceeds cap + |record| (only candidates touched by
    the current record survive the cap), and the touched candidates are the ones
    that mint. Measured behaviour, asserted as such: on the SHUFFLED planted
    stream a cap of 1 mints nothing (each record's fresh candidates evict the
    previous record's, so no escrow can accumulate: K=0); on the same records
    sorted by group the touched candidate survives its run and mints (K > 0);
    and a cap of 8 already recovers the planted K = 8."""
    recs = planted_stream()

    def run(records, cap):
        g = EscrowGraph(cand_pool_cap=cap)
        b = BatchObjective(g).install()
        maxpool = 0
        for i, r in enumerate(records):
            g.process(r)
            maxpool = max(maxpool, len(g.pool))
            assert len(g.pool) <= cap + len(r), (cap, len(g.pool), len(r))
            if (i + 1) % REPAIR_EVERY == 0:
                b.repair()
        b.repair(full=True)
        return g, maxpool

    g1, mp1 = run(recs, 1)
    assert mp1 <= 1 + KEYS_PER_GROUP, mp1
    blocked = sorted(recs, key=lambda r: sorted(r)[0])
    g2, mp2 = run(blocked, 1)
    assert len(g2.mint_log) > 0 and g2.K > 0, (g2.K, len(g2.mint_log))
    g3, mp3 = run(recs, 8)
    assert g3.K == K_STAR and len(g3.mint_log) >= K_STAR, (g3.K, len(g3.mint_log))
    print(f"  cap=1 shuffled: K={g1.K} mints={len(g1.mint_log)} maxpool={mp1}; "
          f"cap=1 group-blocked: K={g2.K} mints={len(g2.mint_log)}; "
          f"cap=8 shuffled: K={g3.K} mints={len(g3.mint_log)}")
    if len(g1.mint_log) == 0:
        print("  note: cap=1 on the shuffled planted stream mints nothing (expected under the "
              "declared v1 eviction, reported not asserted)")


# --------------------------------------------------------------------------- #
# 11. every (record, key) cell is coded by exactly one value block
# --------------------------------------------------------------------------- #
def _cell_audit(g):
    """Per key: background N plus the node blocks' N minus P is the number of
    cells coded more than once (0 under the one-explainer-per-cell invariant).
    Per node and key: a value block holding more observations than the node has
    members publishing the key is a block that counts some record twice."""
    excess = 0
    for ki in g.keys.values():
        held = ki.background.N + sum(v.blocks[ki.kid].N for v in g.nodes.values()
                                     if ki.kid in v.blocks)
        excess += held - ki.P
    over = [(v.nid, g.key_name[kid], blk.N, v.p.get(kid, 0))
            for v in g.nodes.values() for kid, blk in v.blocks.items()
            if blk.N > v.p.get(kid, 0)]
    return excess, over


def test_cells_coded_once():
    """METHOD.md step 2: one explainer per (record, facet) cell, 'which is what
    keeps the code valid (no double counting)'; SPEC line 133 updates the
    background value block only when own[a] == 0; _apply_reassign moves a value
    from the background to the node and _apply_delete moves it back. So after
    any run: background N + sum of node block N == P for every key, and no node
    block may hold more observations than members publishing its key. Asserted
    on a two-type alternating stream (the smallest fixture that mints) and on the
    three streams. Written 2026-09-05 as the finding behind the arrival-order
    spread reported by test_lbatch_order_invariant (same member partition, totals
    hundreds of bits apart): the mint path breaks the invariant twice, (a) _mint
    hands the candidate's blocks to the node without removing those members'
    values from the background blocks that observed them while the owner was 0,
    and (b) sibling candidates touched by the same records each observe every
    value, and merging two such mints adds their blocks, so shared records are
    counted twice."""
    alternating = [{"a": "x", "b": "y"} if i % 2 == 0 else {"c": "p", "d": "q"}
                   for i in range(400)]
    report = []
    failures = []
    for name, recs in (("alternating", alternating), ("planted", planted_stream()),
                       ("shared", shared_key_stream()), ("wiki", wiki_stream() or [])):
        if not recs:
            continue
        g, b = run_stream(recs)
        cells = sum(len(v) for v in g.record_vals.values())
        excess, over = _cell_audit(g)
        worst = max(over, key=lambda o: o[2] - o[3]) if over else None
        report.append(f"{name}: n={g.n} K={g.K} cells={cells} coded-more-than-once={excess} "
                      f"({100.0 * excess / cells:.1f}%), node blocks with N > publishers: "
                      f"{len(over)}, worst (nid,key,N,p)={worst}")
        if excess != 0 or over:
            failures.append(report[-1])
    for line in report:
        print("  " + line)
    assert not failures, "cells coded by more than one value block: " + " | ".join(failures)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if "--hash-seed-worker" in sys.argv:
        _hash_seed_worker()
        sys.exit(0)
    t_all = time.time()
    failed = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except AssertionError as e:
                failed.append(name)
                print(f"FAIL {name} ({time.time() - t0:.1f}s): {e}")
    print(f"{len(failed)} failed, total {time.time() - t_all:.1f}s"
          + (f": {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)
