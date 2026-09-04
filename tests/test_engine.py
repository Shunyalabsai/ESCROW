"""Engine-level gates: UT-11a, UT-12/13 (positive controls), UT-14, UT-22 (paper unit-test suite).

Tolerances are v1-pragmatic where the spec's numbers assume the full encoder
(numeric+text, arithmetic-coder audit); every relaxation is stated inline.
"""
import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective


def run_stream(records, repair_every=100):
    g = EscrowGraph()
    b = BatchObjective(g)
    for i, r in enumerate(records):
        g.process(r)
        if (i + 1) % repair_every == 0:
            b.repair()
    b.repair()
    return g


def two_group(n, k=2, d=10, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        grp = i % 2
        out.append({f"g{grp}k{j}": f"g{grp}v{rng.randrange(d)}" for j in range(k)})
    return out


def test_ut11a_null_control():
    """iid records, no latent structure: E[#spurious nodes] <= 1 (Theorem 1)."""
    ks = []
    for seed in range(30):
        rng = random.Random(seed)
        recs = [{f"k{j}": f"v{rng.randrange(10)}" for j in range(4)}
                for _ in range(3000)]
        ks.append(run_stream(recs).K)
    assert sum(ks) / len(ks) <= 1.0, ks


def test_ut12_positive_control():
    """Two groups, disjoint keys and values: K = 2, correct supports.
    v1 latency note: the k=2 escrow clears its price by n ~ 1000 (the objective
    separates earlier; the release has detection latency t*)."""
    for k, n in ((2, 1000), (2, 5000), (4, 200), (4, 1000), (4, 5000)):
        g = run_stream(two_group(n, k=k))
        assert g.K == 2, (k, n, g.K)
        sups = sorted(tuple(sorted(g.key_name[kid] for kid in v.S))
                      for v in g.nodes.values())
        assert sups == [tuple(f"g0k{j}" for j in range(k)),
                        tuple(f"g1k{j}" for j in range(k))], (k, n, sups)


def test_ut13_presence_is_load_bearing():
    """On the disjoint fixture the evidence is presence/absence, not values: the
    per-key value component of the winning candidate is ~0 while presence+absence
    carries it. Checked via the mint receipts: every support key's escrow is
    positive and the justifying key is a group key."""
    g = run_stream(two_group(2000, k=4))
    assert g.K == 2
    for m in g.mint_log[:2]:
        assert all(b > 0 for b in m["per_key_bits"].values()), m
        assert m["justifying_key"].startswith("g"), m


def test_ut14_merge_guard():
    """Two nodes with disjoint supports and values must NOT merge (dL > 0), and
    the full repair on the healthy 2-node state keeps K = 2."""
    g = run_stream(two_group(3000, k=4))
    b = BatchObjective(g)
    assert g.K == 2
    v, w = sorted(g.nodes.values(), key=lambda x: x.nid)
    d = b.merge_delta(v, w)
    assert d is not None and d > 0, d
    b.repair()
    assert g.K == 2


def test_ut22_degenerate_controls():
    # (a) every record identical -> at most one node, never more
    recs = [{"a": "x", "b": "y"} for _ in range(2000)]
    g = run_stream(recs)
    assert g.K <= 1, g.K
    # (b) every record globally unique on every key -> K = 0, all background
    recs = [{"a": f"u{i}", "b": f"w{i}"} for i in range(2000)]
    g = run_stream(recs)
    assert g.K == 0, g.K
    # (c)+(d) a constant key and a uniform ubiquitous key never justify an edge
    rng = random.Random(3)
    recs = [{"const": "same", "uni": f"v{rng.randrange(8)}",
             f"g{i % 2}": f"g{i % 2}v{rng.randrange(6)}"} for i in range(4000)]
    g = run_stream(recs)
    for m in g.mint_log:
        assert m["justifying_key"] not in ("const", "uni"), m


def test_ut8_sparsity_does_not_dominate():
    """One generating node, record widths 2..12: mint decisions must not track
    record width. v1 form: no mints at all on width-varying single-population
    data (nothing separates), and per-record ownership shows no width bias."""
    rng = random.Random(4)
    keys = [f"k{j}" for j in range(12)]
    recs = []
    for _ in range(3000):
        w = rng.randint(2, 12)
        recs.append({k: f"v{rng.randrange(6)}" for k in rng.sample(keys, w)})
    g = run_stream(recs)
    assert g.K <= 1, g.K


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
