"""E12: encoding stability. Does the choice of code decide the graph?

The strongest objection to the whole thesis is that the parameter is not removed, only
hidden in the choice of code. The appendix says the minted-set agreement across code
choices "is measured by ablation rather than asserted" (appendix.tex:270-272) and
appendix_theory.tex:393-397 states the claim: the code choice "is designed to change how
long a node takes to earn its existence and not which nodes eventually earn it". This
file is that measurement.

Every discretionary encoding choice is varied, each independently and then in
combination, and the OUTPUT is compared:

  1. the universal integer code for the node count and the support size:
     Rissanen's L_N as shipped, ceil(log2(x+1)) + 1, Elias gamma, Elias delta, and the
     bare log-star form (Rissanen without the normalising constant);
  2. the KT prior beta in {1/2, 1}, separately for the presence and membership columns
     (which carry the predictive kt(., ., 2), the announcement code and the support
     scope code) and for the value blocks (the escape Bernoulli, the index among seen
     values, the novelty column and the repeat block);
  3. the support code: the shipped combinatorial L_supp = L_N(|T|) + log2 C(A_n, |T|)
     against a factorised per-key Bernoulli scope code L_col(|T|, A_n);
  4. the value-naming term, the stage-three cost of the escape. Switching the
     inventory-size charge OFF breaks the Kraft identity and is reported as a finding
     with the measured masses, not as a graph arm. A THIRD form is measured instead:
     the tight naming charge log2(inv - u + 1), uniform over the values this block has
     not yet seen plus one fresh slot, which makes the three-stage escape sum to
     exactly one instead of the shipped sub-probability.

Nothing under code/escrow/ is modified. The variants are installed as module
attributes on escrow.codes, escrow.engine and escrow.batch for the duration of one
run and then restored; a self-check asserts that the shipped configuration
reinstalled through this machinery reproduces the unpatched run bit for bit.

Streams: the planted eight-group stream (3,000 records), the shared-key stream where
records fall in two nodes (2,000), and the 320 Wikipedia infoboxes from
results/wiki_cache.*.json.

Writes results/e12_encoding_stability.json. Pure Python plus the optional numpy fast
path, about a minute.
"""
import json
import math
import os
import random
import sys
import time
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow import batch as B                                        # noqa: E402
from escrow import codes as C                                        # noqa: E402
from escrow import engine as E                                       # noqa: E402
from escrow.batch import BatchObjective                              # noqa: E402
from escrow.protocol import run_stream, describe as protocol_describe  # noqa: E402
from escrow.provenance import stamped                              # noqa: E402
from experiments.e4_baseline_army import planted8, wikipedia, _ari   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------- #
# streams
# --------------------------------------------------------------------------- #
def shared_key_stream(T=2000, seed=11, frac2=0.15, d=8):
    """Verbatim generator of code/tests/test_batch.py:72 (copied, not imported: the
    test module pulls in subprocess helpers and a results glob). Three groups. Every
    record carries three SHARED keys s0..s2 with a shared value alphabet, plus its own
    group's three keys; frac2 of the records also carry a second group's three keys.
    This is the fixture on which records end up as members of two nodes. Returns
    (records, primary-group labels)."""
    rng = random.Random(seed)
    out, truth = [], []
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
        truth.append(grp)
    return out, truth


def streams(cache_dir):
    p_recs, p_truth = planted8()
    s_recs, s_truth = shared_key_stream()
    w_recs, w_truth = wikipedia(cache_dir)
    return {
        "planted8": (p_recs, p_truth),
        "shared_key": (s_recs, s_truth),
        "wikipedia": (w_recs, w_truth),
    }


# --------------------------------------------------------------------------- #
# the universal integer codes
# --------------------------------------------------------------------------- #
def ic_rissanen(k):
    """As shipped: iterated log plus log2(2.865064), the normalising constant that
    makes the Kraft sum exactly one."""
    return C.log2_star(k) + C.LOG2_C0


def ic_simple_ceil(k):
    """The 'simpler' flat log code named in the objection."""
    return math.ceil(math.log2(k + 1)) + 1.0


def ic_elias_gamma(k):
    return 2.0 * math.floor(math.log2(k)) + 1.0


def ic_elias_delta(k):
    m = math.floor(math.log2(k))
    return m + 2.0 * math.floor(math.log2(m + 1)) + 1.0


def ic_logstar_bare(k):
    """Rissanen's form with the normalising constant removed."""
    return C.log2_star(k)


INT_CODES = {
    "rissanen": ic_rissanen,
    "simple_ceil": ic_simple_ceil,
    "elias_gamma": ic_elias_gamma,
    "elias_delta": ic_elias_delta,
    "logstar_bare": ic_logstar_bare,
}

KRAFT_LIMIT = 10 ** 6


def kraft_sum(fn, limit=KRAFT_LIMIT):
    s = 0.0
    for k in range(1, limit + 1):
        s += 2.0 ** (-fn(k))
    return s


INT_CODE_VERDICT = {
    "rissanen": ("valid", "1 (converges slowly from below)",
                 "Kraft sum is one by construction: the constant log2(2.865064) is exactly "
                 "the normaliser of the iterated log. The truncated sum below is a partial "
                 "sum of a series with a heavy tail, so a value under one at 10^6 is the "
                 "truncation and not slack."),
    "elias_gamma": ("valid", "1 (exactly)",
                    "Kraft-tight: the 2^j integers of length 2j+1 contribute 2^-(j+1) each "
                    "octave, so the sum telescopes to one."),
    "elias_delta": ("valid", "below 1",
                    "A prefix code; the sum stays below one at every truncation."),
    "simple_ceil": ("invalid", "divergent",
                    "NOT a prefix code: every octave [2^j, 2^(j+1)-1] contributes exactly "
                    "1/4, so the Kraft sum diverges like (1/4) log2 x and already exceeds "
                    "one at x = 8. It is a flat log code, not a universal integer code."),
    "logstar_bare": ("invalid", "2.865064",
                     "NOT a prefix code: this is Rissanen's code with the normaliser "
                     "removed, so its Kraft sum is the Rissanen constant 2.865064, an "
                     "overspend of 1.5165 bits of licence per use."),
}


# --------------------------------------------------------------------------- #
# the parametric code family
# --------------------------------------------------------------------------- #
def build(cfg):
    """Return the full set of replacement codelength functions for one variant.

    cfg keys: intcode, beta_col, beta_val, supp, naming.
      beta_col governs every binary column and its predictive: the membership column
      L_col(t, n), the presence blocks L_col(p, t), the presence/absence predictive
      kt(., ., 2) in the engine, the announcement code L_ann, and (in bernoulli mode)
      the support scope code.
      beta_val governs the categorical value code: the escape Bernoulli and the index
      among seen values in ValueBlock.cost, the novelty column and the repeat block of
      L_vblock.
    """
    ic = INT_CODES[cfg["intcode"]]
    bc = float(cfg["beta_col"])
    bv = float(cfg["beta_val"])
    tight = (cfg["naming"] == "tight")
    lg2 = C.lg2

    def L_N(k):
        assert k >= 1, "L_N is defined for k >= 1"
        return ic(k)

    def L_col(t, n):
        """Beta(bc, bc) block code for a binary column with t ones in n.
        At bc = 1/2 this is the shipped KT column code; at bc = 1 it is the Laplace
        code log2((n+1) C(n,t)). Kraft-tight over binary columns for every bc > 0."""
        return (lg2(n + 2.0 * bc) - lg2(t + bc) - lg2(n - t + bc)
                + 2.0 * lg2(bc) - lg2(2.0 * bc))

    def L_colv(u, N):
        """The same column code at the value-block prior, for the novelty column."""
        return (lg2(N + 2.0 * bv) - lg2(u + bv) - lg2(N - u + bv)
                + 2.0 * lg2(bv) - lg2(2.0 * bv))

    def attach(t, n_incl):
        n_past = n_incl - 1
        return math.log2((n_past - t + bc) / (t + bc))

    def kt_pres(c, N, d):
        return -math.log2((c + bc) / (N + d * bc))

    def kt_val(c, N, d):
        return -math.log2((c + bv) / (N + d * bv))

    def L_KT_block(counts, d):
        N = sum(counts)
        s = lg2(N + d * bv) - lg2(d * bv)
        for c in counts:
            s += lg2(bv) - lg2(c + bv)
        return s

    def L_ann(e, n, announce):
        if announce:
            return -math.log2((e + bc) / (n + 2.0 * bc))
        return -math.log2((n - e + bc) / (n + 2.0 * bc))

    if cfg["supp"] == "combinatorial":
        def L_supp(size_T, A_n):
            assert 1 <= size_T <= A_n
            return L_N(size_T) + (lg2(A_n + 1.0) - lg2(size_T + 1.0)
                                  - lg2(A_n - size_T + 1.0))
    elif cfg["supp"] == "bernoulli":
        def L_supp(size_T, A_n):
            """Factorised per-key Bernoulli scope code: each of the A_n keys is in or
            out under one KT Bernoulli, in block form. Kraft-tight over all 2^A_n
            subsets (this is exactly UT-1's identity), so no size announcement is
            needed and none is charged."""
            assert 1 <= size_T <= A_n
            return L_col(size_T, A_n)
    else:
        raise ValueError(cfg["supp"])

    def _nm(naming, u):
        """The stage-three naming charge. `naming` arrives from the caller as
        log2(inv + 1) with inv the key's global inventory size as of the past, so the
        inventory size is recoverable and the tight form can be built without touching
        the engine."""
        if not tight:
            return naming
        inv = 2.0 ** naming - 1.0
        return math.log2(max(1.0, inv - u + 1.0))

    def vb_cost(self, vid, naming):
        c = self.counts.get(vid)
        if self.N == 0:
            return _nm(naming, self.u)
        if c is None:
            return kt_val(self.u, self.N, 2) + _nm(naming, self.u)
        return kt_val(self.N - self.u, self.N, 2) + kt_val(c, self.N, self.u)

    def L_vblock(counts, naming):
        u = len(counts)
        if u == 0:
            return 0.0
        N = sum(counts.values())
        repeats = [c - 1 for c in counts.values()]
        s = L_colv(u, N) + L_KT_block(repeats, u)
        if tight:
            inv = 2.0 ** naming - 1.0
            for i in range(u):
                s += math.log2(max(1.0, inv - i + 1.0))
        else:
            s += u * naming
        return s

    def L_vblock_delta_add(counts, vid, naming):
        u = len(counts)
        N = sum(counts.values())
        c = counts.get(vid, 0)
        if u == 0:
            return L_colv(1, 1) + L_KT_block([0], 1) + _nm(naming, 0)
        if c == 0:
            d = L_colv(u + 1, N + 1) - L_colv(u, N)
            rb = [x - 1 for x in counts.values()]
            ra = rb + [0]
            d += L_KT_block(ra, u + 1) - L_KT_block(rb, u)
            return d + _nm(naming, u)
        d = L_colv(u, N + 1) - L_colv(u, N)
        rep_total = N - u
        d += kt_val(c - 1, rep_total, u)
        return d

    def L_vblock_delta_remove(counts, vid, naming):
        after = dict(counts)
        if after[vid] == 1:
            del after[vid]
        else:
            after[vid] -= 1
        return -L_vblock_delta_add(after, vid, naming)

    return {
        "L_N": L_N, "L_col": L_col, "L_supp": L_supp, "L_ann": L_ann,
        "attach": attach, "kt_pres": kt_pres, "kt_val": kt_val,
        "L_KT_block": L_KT_block, "vb_cost": vb_cost, "L_vblock": L_vblock,
        "L_vblock_delta_add": L_vblock_delta_add,
        "L_vblock_delta_remove": L_vblock_delta_remove,
    }


# --------------------------------------------------------------------------- #
# install / restore (nothing under code/escrow/ is edited on disk)
# --------------------------------------------------------------------------- #
_C_NAMES = ("L_N", "L_col", "L_supp", "L_ann", "attach", "kt", "L_KT_block")
_E_NAMES = ("attach", "kt", "price", "ValueBlock")
_B_NAMES = ("L_N", "L_col", "L_supp", "kt", "L_KT_block", "lg2",
            "L_vblock", "L_vblock_delta_add", "L_vblock_delta_remove")

ORIG = {
    "C": {n: getattr(C, n) for n in _C_NAMES},
    "E": {n: getattr(E, n) for n in _E_NAMES},
    "B": {n: getattr(B, n) for n in _B_NAMES},
    "vb_cost": C.ValueBlock.cost,
}


def apply_variant(cfg):
    f = build(cfg)
    # escrow.codes: price() resolves L_ann, L_col, L_supp and L_N from this namespace,
    # and engine holds the same price object, so patching here reaches both.
    C.L_N, C.L_col, C.L_supp, C.L_ann = f["L_N"], f["L_col"], f["L_supp"], f["L_ann"]
    C.attach, C.kt, C.L_KT_block = f["attach"], f["kt_val"], f["L_KT_block"]
    # ValueBlock is captured by KeyInfo's dataclass default_factory at import time, so
    # the CLASS must keep its identity: replace the method, never the class.
    C.ValueBlock.cost = f["vb_cost"]
    # escrow.engine imported attach and kt by value; every engine use of kt is a
    # presence or absence predictive, so it takes beta_col.
    E.attach, E.kt = f["attach"], f["kt_pres"]
    # escrow.batch imported its primitives by value; its single kt call is the value
    # repeat increment inside L_vblock_delta_add, so it takes beta_val.
    B.L_N, B.L_col, B.L_supp = f["L_N"], f["L_col"], f["L_supp"]
    B.kt, B.L_KT_block = f["kt_val"], f["L_KT_block"]
    B.L_vblock = f["L_vblock"]
    B.L_vblock_delta_add = f["L_vblock_delta_add"]
    B.L_vblock_delta_remove = f["L_vblock_delta_remove"]


def restore():
    for n, v in ORIG["C"].items():
        setattr(C, n, v)
    for n, v in ORIG["E"].items():
        setattr(E, n, v)
    for n, v in ORIG["B"].items():
        setattr(B, n, v)
    C.ValueBlock.cost = ORIG["vb_cost"]


# --------------------------------------------------------------------------- #
# the variants
# --------------------------------------------------------------------------- #
SHIPPED = {"intcode": "rissanen", "beta_col": 0.5, "beta_val": 0.5,
           "supp": "combinatorial", "naming": "shipped"}


def variant(name, valid, note, **over):
    cfg = dict(SHIPPED)
    cfg.update(over)
    return {"name": name, "cfg": cfg, "kraft_valid": valid, "note": note}


ARMS = [
    variant("shipped", True, "the shipped code: Rissanen L_N, KT beta=1/2 everywhere, "
            "combinatorial support, naming = log2(inventory+1)"),
    variant("intcode_elias_gamma", True, "Elias gamma for L_N", intcode="elias_gamma"),
    variant("intcode_elias_delta", True, "Elias delta for L_N", intcode="elias_delta"),
    variant("beta_col_laplace", True, "Laplace (beta=1) on the membership column, the "
            "presence blocks, the presence predictive and the announcement code; this is "
            "the c=2 end of the paper's c in [1,2] family", beta_col=1.0),
    variant("beta_val_laplace", True, "Laplace (beta=1) on the value blocks: escape "
            "Bernoulli, seen-value index, novelty column and repeat block", beta_val=1.0),
    variant("supp_bernoulli", True, "factorised per-key Bernoulli scope code instead of "
            "L_N(|T|) + log2 C(A_n,|T|)", supp="bernoulli"),
    variant("naming_tight", True, "stage-three naming charged over the values this block "
            "has not seen plus one fresh slot, log2(inv-u+1); makes the three-stage escape "
            "Kraft-tight instead of the shipped sub-probability", naming="tight"),
    variant("combo_gamma", True, "worst-case combination of valid changes, Elias gamma arm",
            intcode="elias_gamma", beta_col=1.0, beta_val=1.0, supp="bernoulli",
            naming="tight"),
    variant("combo_delta", True, "worst-case combination of valid changes, Elias delta arm",
            intcode="elias_delta", beta_col=1.0, beta_val=1.0, supp="bernoulli",
            naming="tight"),
    variant("intcode_simple_ceil", False, "ceil(log2(x+1))+1 for L_N; NOT a prefix code, "
            "reported as a labelled diagnostic only", intcode="simple_ceil"),
    variant("intcode_logstar_bare", False, "bare log-star for L_N, Rissanen without the "
            "normaliser; NOT a prefix code, reported as a labelled diagnostic only",
            intcode="logstar_bare"),
]


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def snapshot(g, b, secs, truth, remap=None):
    """Everything the comparison needs, so the graph can be dropped.

    `truth` is indexed by ORIGINAL record position. `remap` maps a 1-based arrival
    index to a 1-based original position, so a permuted run is comparable with an
    unpermuted one record by record; None means the stream was not permuted."""
    n = g.n
    rm = (lambda i: i) if remap is None else (lambda i: remap[i])
    memb = [set() for _ in range(n + 1)]
    for v in g.nodes.values():
        for m in v.members:
            memb[rm(m)].add(v.nid)
    order = {v.nid: (v.birth_n, v.nid) for v in g.nodes.values()}
    labels = []
    for i in range(1, n + 1):
        st = memb[i]
        labels.append(min(st, key=lambda nid: order[nid]) if st else -1)
    ident = []
    for v in sorted(g.nodes.values(), key=lambda x: (x.birth_n, x.nid)):
        tags = {}
        for m in v.members:
            t = truth[rm(m) - 1]
            tags[t] = tags.get(t, 0) + 1
        maj, cnt = max(tags.items(), key=lambda kv: (kv[1], str(kv[0])))
        ident.append({"majority_truth_group": maj, "t": v.t,
                      "purity": round(cnt / v.t, 4), "birth_n": v.birth_n})
    return {
        "K": g.K,
        "n_mints": len(g.mint_log),
        "mint_ns": [m["n"] for m in g.mint_log],
        "seconds": round(secs, 2),
        "nodes": {v.nid: {"t": v.t, "birth_n": v.birth_n,
                          "support": sorted(g.key_name[k] for k in v.S)}
                  for v in g.nodes.values()},
        "members": {v.nid: frozenset(rm(m) for m in v.members)
                    for v in g.nodes.values()},
        "labels": labels,
        "memb": [frozenset(st) for st in memb[1:]],
        "truth_ari": round(_ari(truth, labels), 4),
        "covered": sum(1 for st in memb[1:] if st),
        "multi": sum(1 for st in memb[1:] if len(st) > 1),
        "identity": ident,
        "identity_key": tuple(sorted(str(d["majority_truth_group"]) for d in ident)),
    }


PERMS = 20


def order_control(recs, truth):
    """The apples-to-apple control: the SHIPPED code, the same records, PERMS random
    arrival orders. Every statistic below is computed exactly as it is for the code
    sweep, so the between-code spread can be read against the between-order spread the
    method already has and already reports (E5)."""
    restore()
    snaps = []
    for p in range(PERMS):
        order = list(range(len(recs)))
        random.Random(1000 + p).shuffle(order)
        remap = {j + 1: order[j] + 1 for j in range(len(order))}
        g, b = run_stream([recs[i] for i in order])
        snaps.append(snapshot(g, b, 0.0, truth, remap=remap))
    aris, fracs = [], []
    for i, j in combinations(range(PERMS), 2):
        c = compare(snaps[i], snaps[j])
        aris.append(c["ARI"])
        fracs.append(c["frac_records_changing_node"])
    Ks = [s["K"] for s in snaps]
    ids = {s["identity_key"] for s in snaps}
    return {
        "permutations": PERMS,
        "K": {"min": min(Ks), "max": max(Ks), "all": Ks},
        "pairwise_ARI": {"min": round(min(aris), 6), "mean": round(sum(aris) / len(aris), 6),
                         "max": round(max(aris), 6)},
        "pairwise_frac_records_changing_node": {
            "min": round(min(fracs), 6), "mean": round(sum(fracs) / len(fracs), 6),
            "max": round(max(fracs), 6)},
        "truth_ARI": {"min": min(s["truth_ari"] for s in snaps),
                      "max": max(s["truth_ari"] for s in snaps)},
        "distinct_node_identity_multisets": len(ids),
        "node_identity_multisets": sorted("+".join(k) for k in ids),
    }


def match_nodes(ma, mb):
    """Greedy maximum-overlap matching between two node sets, by member sets."""
    pairs = []
    for a, sa in ma.items():
        for bnid, sb in mb.items():
            ov = len(sa & sb)
            if ov:
                pairs.append((-ov, a, bnid))
    pairs.sort()
    used_a, used_b, m = set(), set(), {}
    for _, a, bnid in pairs:
        if a in used_a or bnid in used_b:
            continue
        used_a.add(a)
        used_b.add(bnid)
        m[bnid] = a
    return m


def big_survival(sa, sb, thr):
    """Of the nodes in sa holding at least `thr` records, how many have a counterpart in
    sb with Jaccard at least 1/2. This is the question the paper's claim is really about:
    a node that has earned existence should not vanish when the code changes."""
    out, worst = 0, 1.0
    big = [nid for nid, ms in sa["members"].items() if len(ms) >= thr]
    for nid in big:
        A = sa["members"][nid]
        best = max((len(A & Bm) / len(A | Bm) for Bm in sb["members"].values()),
                   default=0.0)
        worst = min(worst, best)
        if best >= 0.5:
            out += 1
    return len(big), out, (round(worst, 4) if big else None)


def compare(sa, sb, big_frac=0.05):
    m = match_nodes(sa["members"], sb["members"])
    n = len(sa["labels"])
    thr = max(2, int(round(big_frac * n)))
    fresh = {}
    def mapb(nid):
        if nid in m:
            return m[nid]
        return fresh.setdefault(nid, -1000 - len(fresh))
    changed = sum(1 for i in range(n)
                  if sa["memb"][i] != frozenset(mapb(x) for x in sb["memb"][i]))
    jac, shifts = [], []
    for bnid, a in m.items():
        A, Bm = sa["members"][a], sb["members"][bnid]
        jac.append(len(A & Bm) / len(A | Bm))
        shifts.append(sb["nodes"][bnid]["birth_n"] - sa["nodes"][a]["birth_n"])
    same_support = sum(1 for bnid, a in m.items()
                       if sa["nodes"][a]["support"] == sb["nodes"][bnid]["support"])
    ba, sab, wab = big_survival(sa, sb, thr)
    bb, sba, wba = big_survival(sb, sa, thr)
    return {
        "ARI": round(_ari(sa["labels"], sb["labels"]), 6),
        "frac_records_changing_node": round(changed / n, 6),
        "n_records_changing_node": changed,
        "K_a": sa["K"], "K_b": sb["K"],
        "matched_nodes": len(m),
        "unmatched_a": sa["K"] - len(m), "unmatched_b": sb["K"] - len(m),
        "mean_matched_jaccard": round(sum(jac) / len(jac), 6) if jac else None,
        "min_matched_jaccard": round(min(jac), 6) if jac else None,
        "matched_nodes_same_support": same_support,
        "big_node_threshold_records": thr,
        "big_nodes_a": ba, "big_nodes_a_surviving_in_b": sab,
        "big_nodes_b": bb, "big_nodes_b_surviving_in_a": sba,
        "worst_big_node_jaccard": (min(x for x in (wab, wba) if x is not None)
                                   if (wab is not None or wba is not None) else None),
        "birth_shift_records": {
            "all": sorted(shifts),
            "max_abs": max((abs(x) for x in shifts), default=0),
            "mean_abs": round(sum(abs(x) for x in shifts) / len(shifts), 2) if shifts else 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# the value-naming Kraft finding (variant 4)
# --------------------------------------------------------------------------- #
def naming_kraft_report(g):
    """Total predictive mass of the three-stage escape over {values seen in the block}
    union {values in the key's inventory but not in the block} union {one fresh value},
    under the three naming rules. The shipped rule is a valid sub-probability; dropping
    the inventory charge puts the mass above one, which is the Kraft violation the
    engine's docstring records."""
    def mass(u, N, inv, mode):
        if N == 0:
            return 1.0
        p_new = (u + 0.5) / (N + 1.0)
        p_old = (N - u + 0.5) / (N + 1.0)
        slots = inv - u + 1                     # unseen inventory values plus one fresh
        if mode == "off":
            return p_old + p_new * slots
        if mode == "shipped":
            return p_old + p_new * slots / (inv + 1.0)
        return p_old + p_new                    # tight: log2(inv - u + 1)

    rows = []
    worst = {"off": 0.0, "shipped": 2.0, "tight": 0.0}
    for name, ki in g.keys.items():
        blocks = [("background", ki.background)]
        for v in g.nodes.values():
            if ki.kid in v.blocks:
                blocks.append((f"node{v.nid}", v.blocks[ki.kid]))
        inv = len(ki.inventory)
        for who, blk in blocks:
            if blk.N == 0:
                continue
            m_off = mass(blk.u, blk.N, inv, "off")
            m_ship = mass(blk.u, blk.N, inv, "shipped")
            m_tight = mass(blk.u, blk.N, inv, "tight")
            worst["off"] = max(worst["off"], m_off)
            worst["shipped"] = min(worst["shipped"], m_ship)
            worst["tight"] = max(worst["tight"], abs(m_tight - 1.0))
            rows.append((m_off, name, who, blk.u, blk.N, inv, m_ship))
    rows.sort(reverse=True)
    top = [{"key": r[1], "block": r[2], "u": r[3], "N": r[4], "inventory": r[5],
            "mass_naming_off": round(r[0], 4), "mass_shipped": round(r[6], 4),
            "mass_naming_tight": 1.0} for r in rows[:5]]
    return {
        "worst_mass_with_naming_off": round(worst["off"], 4),
        "tightest_mass_shipped": round(worst["shipped"], 4),
        "max_abs_deviation_from_one_tight": round(worst["tight"], 12),
        "worst_blocks": top,
    }


# --------------------------------------------------------------------------- #
def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else OUT
    t_start = time.time()

    # --- the integer codes, checked for Kraft before anything is run ---------- #
    intcode_table = {}
    for name, fn in INT_CODES.items():
        verdict, limit, why = INT_CODE_VERDICT[name]
        intcode_table[name] = {
            "kraft_sum_truncated_at_1e6": round(kraft_sum(fn), 6),
            "kraft_sum_analytic_limit": limit,
            "verdict": verdict, "why": why,
            "bits": {str(k): round(fn(k), 4) for k in (1, 2, 3, 4, 5, 8, 16, 64, 256)},
        }

    report = {
        "experiment": "E12 encoding stability",
        "question": ("Does the choice of code decide which nodes exist, or only when "
                     "they are born?"),
        "protocol": protocol_describe(),
        "integer_codes": intcode_table,
        "variants": {a["name"]: {"cfg": a["cfg"], "kraft_valid": a["kraft_valid"],
                                 "note": a["note"]} for a in ARMS},
        "streams": {},
    }

    data = streams(cache_dir)

    # --- self-check: the machinery must reproduce the shipped run exactly ----- #
    sc_recs, sc_truth = data["wikipedia"]
    restore()
    g0, b0 = run_stream(sc_recs)
    base = snapshot(g0, b0, 0.0, sc_truth)
    L0 = b0.total()
    apply_variant(SHIPPED)
    g1, b1 = run_stream(sc_recs)
    L1 = b1.total()
    s1 = snapshot(g1, b1, 0.0, sc_truth)
    restore()
    grid = [(t, n) for n in (1, 2, 3, 5, 8, 40, 400) for t in range(0, n + 1)]
    f_ship = build(SHIPPED)
    max_col = max(abs(f_ship["L_col"](t, n) - ORIG["C"]["L_col"](t, n)) for t, n in grid)
    max_ln = max(abs(f_ship["L_N"](k) - ORIG["C"]["L_N"](k)) for k in range(1, 1000))
    report["self_check"] = {
        "shipped_config_reproduces_unpatched_run": (
            base["K"] == s1["K"]
            and sorted(map(sorted, base["members"].values())) == sorted(map(sorted, s1["members"].values()))
            and abs(L0 - L1) < 1e-6),
        "K_unpatched": base["K"], "K_reinstalled": s1["K"],
        "L_batch_unpatched": round(L0, 6), "L_batch_reinstalled": round(L1, 6),
        "max_abs_L_col_error_vs_shipped": max_col,
        "max_abs_L_N_error_vs_shipped": max_ln,
    }
    assert report["self_check"]["shipped_config_reproduces_unpatched_run"], \
        "the variant machinery does not reproduce the shipped run; refusing to report"

    # --- the sweep ------------------------------------------------------------ #
    naming_finding_done = False
    worst_pairs_valid, worst_pairs_all = [], []
    for sname, (recs, truth) in data.items():
        snaps, rows = {}, {}
        for arm in ARMS:
            apply_variant(arm["cfg"])
            t0 = time.time()
            g, b = run_stream(recs)
            secs = time.time() - t0
            L_own = b.total()
            restore()
            L_ship = BatchObjective(g).total()
            s = snapshot(g, b, secs, truth)
            snaps[arm["name"]] = s
            rows[arm["name"]] = {
                "kraft_valid": arm["kraft_valid"],
                "K": s["K"], "n_mints": s["n_mints"], "mint_ns": s["mint_ns"],
                "truth_ARI": s["truth_ari"],
                "records_in_a_node": s["covered"], "records_in_two_nodes": s["multi"],
                "L_batch_own_code_bits": round(L_own, 2),
                "L_batch_shipped_yardstick_bits": round(L_ship, 2),
                "nodes": [{"t": v["t"], "birth_n": v["birth_n"],
                           "support": v["support"][:10]}
                          for v in sorted(s["nodes"].values(),
                                          key=lambda x: (x["birth_n"],))],
                "node_identity": s["identity"],
                "node_identity_multiset": "+".join(s["identity_key"]),
                "seconds": s["seconds"],
            }
            if sname == "wikipedia" and arm["name"] == "shipped" and not naming_finding_done:
                report["value_naming_term_finding"] = {
                    "question": ("variant 4: can the inventory-size charge be switched "
                                 "off without breaking the Kraft identity?"),
                    "answer": "No. It is not a discretionary choice and no graph arm is run for it.",
                    "why": ("The stage-three charge is what identifies WHICH new value the "
                            "escape stands for. With it, the predictive over the seen values, "
                            "the unseen inventory values and one fresh slot sums to at most "
                            "one. With naming = 0 the same escape mass is handed to every one "
                            "of the inv - u + 1 unseen candidates at once, so the total "
                            "predictive mass exceeds one and the code is not a code."),
                    "measured_on": "the shipped Wikipedia run, over every value block at end of stream",
                    "masses": naming_kraft_report(g),
                    "already_recorded": ("codes.py ValueBlock.cost docstring and appendix.tex:154-156: "
                                         "dropping the naming stage minted 204 sibling nodes on the "
                                         "k=4 positive control."),
                    "valid_alternative_measured_instead": (
                        "naming_tight, log2(inv - u + 1), which makes the same escape sum to "
                        "exactly one and closes the sub-probability noted in GAPS Block 7(g)."),
                }
                naming_finding_done = True

        # pairwise agreement
        names = [a["name"] for a in ARMS]
        pairs = {}
        for a, bn in combinations(names, 2):
            pairs[f"{a} vs {bn}"] = compare(snaps[a], snaps[bn])

        valid = [a["name"] for a in ARMS if a["kraft_valid"]]
        vpairs = {k: v for k, v in pairs.items()
                  if k.split(" vs ")[0] in valid and k.split(" vs ")[1] in valid}
        worst_v = min(vpairs.items(), key=lambda kv: (kv[1]["ARI"],
                                                      -kv[1]["frac_records_changing_node"]))
        worst_all = min(pairs.items(), key=lambda kv: (kv[1]["ARI"],
                                                      -kv[1]["frac_records_changing_node"]))
        worst_pairs_valid.append((worst_v[1]["ARI"], sname, worst_v[0], worst_v[1]))
        worst_pairs_all.append((worst_all[1]["ARI"], sname, worst_all[0], worst_all[1]))
        Ks = {a["name"]: snaps[a["name"]]["K"] for a in ARMS}
        Lown = {n: rows[n]["L_batch_own_code_bits"] for n in names}

        oc = order_control(recs, truth)
        vids = {snaps[a["name"]]["identity_key"] for a in ARMS if a["kraft_valid"]}
        wider_K = ((max(Ks[n] for n in valid) - min(Ks[n] for n in valid))
                   > (oc["K"]["max"] - oc["K"]["min"]))
        lower_ARI = worst_v[1]["ARI"] < oc["pairwise_ARI"]["min"]
        report["streams"][sname] = {
            "n_records": len(recs),
            "true_groups": len(set(truth)),
            "order_control_shipped_code": oc,
            "code_vs_order": {
                "between_code_worst_ARI_valid": worst_v[1]["ARI"],
                "between_order_worst_ARI": oc["pairwise_ARI"]["min"],
                "between_code_K_range_valid": [min(Ks[n] for n in valid),
                                               max(Ks[n] for n in valid)],
                "between_order_K_range": [oc["K"]["min"], oc["K"]["max"]],
                "code_choice_moves_the_graph_more_than_arrival_order": bool(
                    wider_K or lower_ARI),
                "distinct_node_identity_multisets_across_valid_codes": len(vids),
                "distinct_node_identity_multisets_across_orders":
                    oc["distinct_node_identity_multisets"],
                "node_identity_multisets_across_valid_codes":
                    sorted("+".join(k) for k in vids),
            },
            "arms": rows,
            "K_by_variant": Ks,
            "K_range_valid_codes": [min(Ks[n] for n in valid), max(Ks[n] for n in valid)],
            "K_range_all_arms": [min(Ks.values()), max(Ks.values())],
            "L_batch_own_code_spread_bits": round(max(Lown.values()) - min(Lown.values()), 2),
            "L_batch_own_code_spread_percent": round(
                100.0 * (max(Lown.values()) - min(Lown.values())) / min(Lown.values()), 3),
            "worst_pair_valid_codes": {"pair": worst_v[0], **worst_v[1]},
            "worst_pair_all_arms": {"pair": worst_all[0], **worst_all[1]},
            "pairs": pairs,
        }

    # --- headline ------------------------------------------------------------- #
    n_valid = sum(1 for a in ARMS if a["kraft_valid"])
    valid_names = [a["name"] for a in ARMS if a["kraft_valid"]]

    def vpairs_of(sname):
        d = report["streams"][sname]["pairs"]
        return {k: v for k, v in d.items()
                if k.split(" vs ")[0] in valid_names and k.split(" vs ")[1] in valid_names}

    single_names = [a["name"] for a in ARMS
                    if a["kraft_valid"] and not a["name"].startswith("combo")]

    # anchored on the shipped graph: of the nodes the SHIPPED code mints that hold at
    # least five percent of the stream, how many still have a counterpart (Jaccard at
    # least one half) under the least favourable valid variant.
    shipped_big = {}
    for sname in report["streams"]:
        d = report["streams"][sname]["pairs"]
        rows_ = [(d[f"shipped vs {a}"]["big_nodes_a"],
                  d[f"shipped vs {a}"]["big_nodes_a_surviving_in_b"], a)
                 for a in valid_names if a != "shipped"]
        tot = rows_[0][0]
        worst = min(rows_, key=lambda r: r[1])
        shipped_big[sname] = (tot, worst[1], worst[2])

    per = {}
    for sname in report["streams"]:
        vp = vpairs_of(sname)
        sp = {k: v for k, v in vp.items()
              if k.split(" vs ")[0] in single_names and k.split(" vs ")[1] in single_names}
        oc = report["streams"][sname]["order_control_shipped_code"]
        Ks = report["streams"][sname]["K_by_variant"]
        per[sname] = {
            "worst_ARI": min(v["ARI"] for v in vp.values()),
            "worst_pair": min(vp.items(), key=lambda kv: kv[1]["ARI"])[0],
            "worst_frac": max(v["frac_records_changing_node"] for v in vp.values()),
            "max_birth_shift": max(v["birth_shift_records"]["max_abs"] for v in vp.values()),
            "K_range": [min(Ks[n] for n in valid_names), max(Ks[n] for n in valid_names)],
            "order_K_range": [oc["K"]["min"], oc["K"]["max"]],
            "order_worst_ARI": oc["pairwise_ARI"]["min"],
            "order_worst_frac": oc["pairwise_frac_records_changing_node"]["max"],
            "worst_ARI_one_change_at_a_time": min(v["ARI"] for v in sp.values()),
            "worst_pair_one_change_at_a_time": min(sp.items(), key=lambda kv: kv[1]["ARI"])[0],
            "big_node_threshold_records": list(vp.values())[0]["big_node_threshold_records"],
            "shipped_big_nodes": shipped_big[sname][0],
            "shipped_big_nodes_surviving_in_the_worst_valid_variant": shipped_big[sname][1],
            "worst_variant_for_big_nodes": shipped_big[sname][2],
            "worst_big_node_jaccard": min(
                [v["worst_big_node_jaccard"] for v in vp.values()
                 if v["worst_big_node_jaccard"] is not None] or [None]),
            "identities": report["streams"][sname]["code_vs_order"][
                "node_identity_multisets_across_valid_codes"],
            "order_identities": oc["node_identity_multisets"],
            "unchanged": (min(v["ARI"] for v in vp.values()) >= 1.0
                          and max(v["frac_records_changing_node"] for v in vp.values()) == 0.0),
        }

    clean = [k for k, v in per.items() if v["unchanged"]]
    moved = [k for k, v in per.items() if not v["unchanged"]]
    worst_stream = min(per, key=lambda k: per[k]["worst_ARI"])
    w = per[worst_stream]
    order_dominates = all(
        per[k]["worst_ARI"] >= per[k]["order_worst_ARI"]
        and (per[k]["K_range"][1] - per[k]["K_range"][0])
        <= (per[k]["order_K_range"][1] - per[k]["order_K_range"][0])
        for k in per)
    spread = {k: report["streams"][k]["L_batch_own_code_spread_percent"] for k in report["streams"]}
    spread_bits = {k: report["streams"][k]["L_batch_own_code_spread_bits"] for k in report["streams"]}

    report["headline"] = {
        "arms_total": len(ARMS), "arms_kraft_valid": n_valid,
        "streams_where_every_valid_code_returns_the_same_partition": clean,
        "streams_where_the_code_choice_moves_the_partition": moved,
        "per_stream": per,
        "worst_disagreement_over_valid_codes": {
            "stream": worst_stream, "pair": w["worst_pair"], "ARI": w["worst_ARI"],
            "frac_records_changing_node": w["worst_frac"]},
        "encoding_spread_stays_inside_the_arrival_order_spread": order_dominates,
        "L_batch_own_code_spread_bits_by_stream": spread_bits,
        "L_batch_own_code_spread_percent_by_stream": spread,
    }

    mass_off = report["value_naming_term_finding"]["masses"]["worst_mass_with_naming_off"]
    big_tot = sum(per[k]["shipped_big_nodes"] for k in per)
    big_surv = sum(per[k]["shipped_big_nodes_surviving_in_the_worst_valid_variant"]
                   for k in per)
    report["headline"]["substantial_nodes_of_the_shipped_graph"] = big_tot
    report["headline"]["substantial_nodes_surviving_the_worst_valid_variant"] = big_surv
    report["headline"]["substantial_node_threshold"] = "five percent of the stream"
    report["headline"]["worst_ARI_one_change_at_a_time"] = {
        k: per[k]["worst_ARI_one_change_at_a_time"] for k in per}

    s_clean = ("; ".join(
        f"on {k} every arm returns K = {per[k]['K_range'][0]}, adjusted Rand index 1.0000 "
        f"between every pair of codes and against the planted truth, the same support keys, "
        f"and not one record in {report['streams'][k]['n_records']} changing node, with birth "
        f"times moving by up to {per[k]['max_birth_shift']} records"
        for k in clean) if clean else "no stream is completely unmoved")
    ARI_TOL = 0.02                    # "tracks" the order band rather than beating it
    inside_streams = [k for k in moved
                      if per[k]["worst_ARI"] >= per[k]["order_worst_ARI"] - ARI_TOL
                      and (per[k]["K_range"][1] - per[k]["K_range"][0])
                      <= (per[k]["order_K_range"][1] - per[k]["order_K_range"][0])]
    outside_streams = [k for k in moved if k not in inside_streams]
    if not outside_streams:
        s_order = "tracks on every stream"
    elif not inside_streams:
        s_order = "exceeds on " + " and ".join(outside_streams)
    else:
        s_order = ("tracks on " + " and ".join(inside_streams)
                   + " and exceeds on " + " and ".join(outside_streams))
    report["headline"]["streams_where_the_code_band_tracks_the_arrival_order_band"] = \
        inside_streams
    report["headline"]["streams_where_the_code_band_exceeds_the_arrival_order_band"] = \
        outside_streams
    s_moved = "; ".join(
        f"{k} spans K = {per[k]['K_range'][0]} to {per[k]['K_range'][1]} against "
        f"K = {per[k]['order_K_range'][0]} to {per[k]['order_K_range'][1]} for the shipped "
        f"code over {PERMS} arrival orders of the same records, and a worst pairwise "
        f"adjusted Rand index of {per[k]['worst_ARI']:.4f} "
        f"({per[k]['worst_ARI_one_change_at_a_time']:.4f} when only one choice is changed "
        f"at a time) against {per[k]['order_worst_ARI']:.4f} across orders"
        for k in moved)
    ident_changes = [k for k in moved if len(per[k]["identities"]) > 1]
    if ident_changes:
        s_ident = (
            f"On {' and '.join(ident_changes)} the honest answer to the sharper question is "
            f"that the code choice does change which nodes exist and not only when they are "
            f"born: {ident_changes[0]} yields "
            f"{len(per[ident_changes[0]]['identities'])} different node identity sets across "
            f"the {len(valid_names)} valid codes (for instance "
            f"{per[ident_changes[0]]['identities'][0]} against "
            f"{per[ident_changes[0]]['identities'][-1]}), but the same substitution happens "
            f"under a change of arrival order alone, which produces "
            f"{' and '.join(str(len(per[k]['order_identities'])) + ' on ' + k for k in ident_changes)} "
            f"identity sets over {PERMS} permutations of the shipped code; the nodes that "
            f"move are the marginal ones, since {big_surv} of the {big_tot} nodes the shipped "
            f"code mints that hold at least five percent of their stream still have a "
            f"counterpart under the least favourable valid variant.")
    else:
        s_ident = ("No variant changes which nodes exist: every node minted under one code is "
                   "minted under all of them, with the same support keys, and only the birth "
                   "times move.")

    report["reading"] = (
        f"{len(ARMS)} encodings of the same three streams, {n_valid} of them valid prefix "
        f"codes and {len(ARMS) - n_valid} kept only as labelled diagnostics, buy visibly "
        f"different numbers of bits and agree on the graph wherever the graph is determined: "
        f"the description length of the stream moves by "
        f"{', '.join(f'{spread_bits[k]:.0f} bits ({spread[k]:.2f} percent) on {k}' for k in spread)} "
        f"across the arms, while {s_clean}. "
        f"On the {'stream' if len(moved) == 1 else 'streams'} where the shipped code does "
        f"not itself recover the truth the codes disagree, and the comparison that decides "
        f"whether that is a hidden knob is against the movement the same code already shows "
        f"under a change of arrival order alone, a band it {s_order}: {s_moved}; "
        f"the worst disagreement anywhere in the sweep is adjusted Rand index "
        f"{w['worst_ARI']:.4f} on {worst_stream} ({w['worst_pair']}), with "
        f"{100.0 * w['worst_frac']:.1f} percent of records changing node. "
        f"{s_ident} "
        f"One item on the list turned out not to be a choice at all: the stage-three naming "
        f"charge cannot be switched off, because without it the escape hands its whole mass "
        f"to every unseen value at once and the block predictive sums to {mass_off:.1f} "
        f"instead of one, so it is reported here as a measured Kraft violation rather than "
        f"as an ablation arm."
    )

    report["runtime_seconds"] = round(time.time() - t_start, 1)

    def enc(o):
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError(type(o))

    path = os.path.join(OUT, "e12_encoding_stability.json")
    with open(path, "w") as f:
        json.dump(stamped(report), f, indent=2, default=enc)
    print(json.dumps(report["headline"], indent=2, default=enc))
    print()
    print(report["reading"])
    print()
    print("wrote", path, "in", report["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
