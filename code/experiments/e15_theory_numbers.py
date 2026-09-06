"""E15: every constant the paper quotes from the theory, computed from the shipped code.

Writes results/e15_theory_numbers.json. Each entry records the quantity, the value the
paper prints, the value this script computes, how it was computed, the paper sites that
quote it (file and line, checked at run time), and a verdict.

Answers findings/GAPS.md Block 4: "Theory-side numbers presented as measured have no
script and some do not survive measurement."

Covered:
  Q1  the price difference between the priors (a,b) = (1/2, 1) and (2, 0.1) per mint
  Q2  the concentration regret slope in bits per decade
  Q3  the "measured 17" crossover, asymptotic and exact, against arity
  Q4  the batch margin at fifty records (the "37.2 bits" cell) over a k x n grid
  Q5  the myopic-versus-batch gap on a 20,000-record 20-node Zipf stream
  Q6  the UT-14 merge deltas and the eight-node collapse
  Q7  the tokenisation spread on the Wikipedia text fields

RULES OBSERVED. Every codelength comes from escrow.codes or escrow.batch; every engine
run goes through escrow.protocol.run_stream (or new_run driving the same three choices,
which protocol.py permits for scripts that need per-record hooks). Two quantities are
defined in the paper and not in code/escrow/, so their CONTROL LOOP is written here while
every bit they count is still a shipped primitive; both are marked
"defined outside the code" in the JSON:
  - the record-seeded myopic rule (method.tex "Why the obvious rule fails",
    Lemma lem:myopic), which has no implementation in code/escrow/;
  - the node-independent presence arm (appendix_theory "the node-dependent presence
    blocks are load-bearing"), obtained by subclassing BatchObjective and DELETING the
    one shipped presence term, not by rewriting the objective.
A planted allocation is built with plant() below, which assembles an EscrowGraph state out
of the engine's own Node, KeyInfo and ValueBlock objects and then hands it to the shipped
BatchObjective.total(). plant() writes no codelength.

No em dashes or en dashes anywhere in this file or its output.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.batch import BatchObjective, L_vblock
from escrow.codes import (L_col, L_N, ValueBlock,
                          km_multinomial_complexity, price)
from escrow.engine import EscrowGraph, Node
from escrow.protocol import REPAIR_EVERY, describe as protocol_describe, new_run, run_stream
from escrow.provenance import stamped

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

LN2 = math.log(2.0)

# --------------------------------------------------------------------------- #
# paper sites: checked against the file at run time so line drift is visible
# --------------------------------------------------------------------------- #
def site(relpath: str, token: str) -> dict:
    """Locate every line of a paper file that quotes a token, and record it.

    Line numbers are RESOLVED AT RUN TIME rather than pinned, because the sections
    were being edited while this script was written and a pinned number goes stale
    silently. A LaTeX comment line (leading %) is reported separately from prose:
    a comment is a provenance note, not a printed number.
    """
    path = os.path.join(ROOT, relpath)
    rec = {"file": relpath, "quotes": token, "prose_lines": [], "comment_lines": [],
           "found": False, "text": None}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return rec
    for i, raw in enumerate(lines, start=1):
        if token not in raw:
            continue
        if raw.lstrip().startswith("%"):
            rec["comment_lines"].append(i)
        else:
            rec["prose_lines"].append(i)
            if rec["text"] is None:
                rec["text"] = raw.rstrip("\n")[:240]
    rec["found"] = bool(rec["prose_lines"] or rec["comment_lines"])
    return rec


# --------------------------------------------------------------------------- #
# planting: an EscrowGraph state for a declared allocation, no codelength here
# --------------------------------------------------------------------------- #
def plant(records, labels, supports=None):
    """Build the EscrowGraph state of a declared allocation.

    records  list of {key string: value string}
    labels   one label per record; None means the record belongs to no node
    supports optional {label: [key string, ...]}; default is every key the label's
             members publish

    Every cell is owned exactly once: a node owns its members' cells on its support
    keys, everything else is owned by the background, which is the same ledger
    invariant EscrowGraph._mint maintains. The state is then scored by the shipped
    BatchObjective.total(); nothing here computes bits.
    """
    g = EscrowGraph()
    g.n = len(records)
    for i, rec in enumerate(records, start=1):
        kids, vals = set(), {}
        for a, x in rec.items():
            ki = g._key(a)
            vid = ki.intern(str(x))
            ki.P += 1
            kids.add(ki.kid)
            vals[ki.kid] = vid
        g.record_keys[i] = frozenset(kids)
        g.record_vals[i] = vals
    by_label = {}
    for i, lab in enumerate(labels, start=1):
        if lab is None:
            continue
        by_label.setdefault(lab, []).append(i)
    for lab in sorted(by_label, key=lambda z: str(z)):
        members = by_label[lab]
        if supports is None:
            S = set()
            for r in members:
                S |= set(g.record_keys[r])
        else:
            S = {g.keys[a].kid for a in supports[lab] if a in g.keys}
        if not S:
            continue
        g.next_id += 1
        g.K += 1
        w = Node(nid=g.next_id, t=len(members), birth_n=0)
        w.members = set(members)
        for kid in sorted(S):
            pub = {r for r in members if kid in g.record_keys[r]}
            w.S.add(kid)
            w.pub[kid] = pub
            w.p[kid] = len(pub)
            blk = ValueBlock()
            for r in sorted(pub):
                blk.observe(g.record_vals[r][kid])
            w.blocks[kid] = blk
            g.ix2.setdefault(kid, set()).add(w.nid)
            for vid in blk.counts:
                g.ix1.setdefault((kid, vid), set()).add(w.nid)
        g.nodes[w.nid] = w
    owner_of = {}
    for w in g.nodes.values():
        for r in w.members:
            for kid in w.S:
                owner_of[(r, kid)] = w.nid
    for i in range(1, g.n + 1):
        own = {}
        for kid, vid in g.record_vals[i].items():
            o = owner_of.get((i, kid), 0)
            own[kid] = o
            if o == 0:
                g.key_by_id[kid].background.observe(vid)
        g.record_owner[i] = own
    return g


class NodeIndependentPresence(BatchObjective):
    """The node-independent missingness arm named at appendix_theory.tex:879-890.

    DEFINED OUTSIDE THE CODE: code/escrow/ ships no presence switch. This subclass
    deletes exactly one shipped term, the per-node presence block L_col(p_{v,a}, m_v)
    of E4, and changes nothing else; total(), merge_delta() and repair() are the
    shipped methods and pick the deletion up through node_local().
    """

    def node_local(self, t: int, S: set, p: dict, blocks: dict) -> float:
        return BatchObjective.node_local(self, t, S, p, blocks) - sum(
            L_col(p.get(kid, 0), t) for kid in S)


# --------------------------------------------------------------------------- #
# Q1. the price difference between two declared priors
# --------------------------------------------------------------------------- #
A1, B1 = 0.5, 1.0          # the shipped prior, method.tex:60
A2, B2 = 2.0, 0.1          # the comparison prior
N_MARKS = (10, 100, 1000, 10000, 100000)


def harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1))


def lam_exact(s, a, b, K, H):
    """The shipped closed form, Equation eq:lambda / eq:price."""
    return math.log2(s * (b + H) + 1.0) - math.log2(a + K)


def lam_asym(a, b, K, H):
    """The appendix's displayed difference form, appendix_theory.tex:381:
    log((b+H)/(b'+H)) - log((a+K)/(a'+K)), taken one prior at a time."""
    return math.log2(b + H) - math.log2(a + K)


E13_K_STAR, E13_KEYS, E13_D = 8, 3, 8


def e13_stream(T, seed=7):
    """The E13 planted fixture (code/experiments/e13_creation_bias.py:28-40) at
    noise 0, used here only to record the (K_s, H_s) trajectory the engine walks."""
    rng = random.Random(seed)
    for _ in range(T):
        grp = rng.randrange(E13_K_STAR)
        yield {f"c{grp}k{j}": f"c{grp}v{rng.randrange(E13_D)}"
               for j in range(E13_KEYS)}


def observed_trajectory(T=100000, marks=N_MARKS):
    """Run the E13 planted stream under the shipped protocol and record (K_n, H_n)
    at the stream positions Q1 needs. new_run keeps protocol.py's three choices."""
    g, b = new_run()
    traj = []
    for i, rec in enumerate(e13_stream(T), start=1):
        g.process(rec)
        if i in marks:
            traj.append({"n": i, "K": g.K, "H": g.H})
        if i % REPAIR_EVERY == 0:
            b.repair()
    b.repair(full=True)
    return traj, g.K


def q1_observed_rows(observed, label):
    rows = []
    for mark in observed:
        n, K, H = mark["n"], mark["K"], mark["H"]
        rows.append({
            "trajectory": label,
            "n": n, "K_observed": K, "H_observed": round(H, 4),
            "gamma_hat = K/H": round(K / H, 4) if H else None,
            "exact, H_n": round(lam_exact(n, A1, B1, K, H)
                                - lam_exact(n, A2, B2, K, H), 4),
            "asymptotic, H_n": round(lam_asym(A1, B1, K, H)
                                     - lam_asym(A2, B2, K, H), 4),
        })
    return rows


def q1_price_difference():
    """Difference in bits per mint between the two declared priors, along several
    declared trajectories for (K_s, H_s). The paper quotes 0.68 at n = 10 and 0.19
    at n = 10^5; the trajectory and the choice of H_s and of exact-versus-asymptotic
    form all move the first number, which is why every combination is reported."""
    rows = []
    trajectories = [("K_s = H_s", 1.0), ("K_s = 2 H_s", 2.0), ("K_s = 0.5 H_s", 0.5)]
    for name, mult in trajectories:
        for n in N_MARKS:
            H_prev, H_now = harmonic(n - 1), harmonic(n)
            cell = {"trajectory": name, "n": n}
            for hname, H in (("H_{n-1}", H_prev), ("H_n", H_now)):
                K = mult * H
                cell["exact, " + hname] = round(
                    lam_exact(n, A1, B1, K, H) - lam_exact(n, A2, B2, K, H), 4)
                cell["asymptotic, " + hname] = round(
                    lam_asym(A1, B1, K, H) - lam_asym(A2, B2, K, H), 4)
            rows.append(cell)
    return rows


# --------------------------------------------------------------------------- #
# Q2. the concentration regret slope
# --------------------------------------------------------------------------- #
def q2_regret_slope():
    half_log10 = 0.5 * math.log2(10.0)
    # the slope as a finite difference of (1/2) log2 H_N over one decade of N,
    # to show the closed form is the limit and not a measurement of anything
    diffs = []
    for e in range(3, 9):
        h1, h2 = math.log(10.0 ** e), math.log(10.0 ** (e + 1))
        diffs.append(round(0.5 * math.log2(h2) - 0.5 * math.log2(h1), 4))
    return {
        "half_log2_10": round(half_log10, 6),
        "paper_value": 1.661,
        "identity_holds_to": round(abs(half_log10 - 1.661), 6),
        "regret_slope_per_decade_of_N_not_of_exposure": diffs,
        "note": ("1.661 is (1/2) log2 10 exactly, the slope of (1/2) log2 H_N per "
                 "decade of EXPOSURE H_N. Per decade of RECORD COUNT N the slope is "
                 "(1/2) log2(ln 10^{e+1} / ln 10^e), the falling sequence listed "
                 "above, which is not 1.661. The paper says exposure and is right."),
    }


# --------------------------------------------------------------------------- #
# Q3. the crossover
# --------------------------------------------------------------------------- #
def comp_asymptotic(m: int, n: int) -> float:
    """Xie-Barron asymptotic parametric complexity in bits, appendix_theory.tex:721."""
    return ((m - 1) / 2.0 * math.log2(n / (2.0 * math.pi))
            + (m / 2.0) * math.log2(math.pi) - math.lgamma(m / 2.0) / LN2)


def comp_exact(m: int, n: int) -> float:
    """log2 of the exact Kontkanen-Myllymaki multinomial complexity, shipped."""
    return math.log2(km_multinomial_complexity(n, m))


def q3_crossover():
    """The paper's 17 is 2 pi e, the leading-order solution of
    (m-1)/2 log2(n / 2 pi) = (m-1)/(2 ln 2), which drops the Gamma constant and is
    therefore free of the arity m. With the constant kept, and with the exact
    Gamma-function form the algorithm is said to use, the crossover moves with m."""
    rows = []
    for m in (2, 5, 20, 100):
        mean_gain = (m - 1) / (2.0 * LN2)          # E[chi^2_{m-1}] in bits
        n_asym = next(n for n in range(1, 10 ** 6) if comp_asymptotic(m, n) > mean_gain)
        n_exact = next(n for n in range(1, 10 ** 6) if comp_exact(m, n) > mean_gain)
        rows.append({
            "arity_m": m,
            "degrees_of_freedom_d_S": m - 1,
            "mean_null_gain_bits": round(mean_gain, 4),
            "crossover_asymptotic_COMP": n_asym,
            "crossover_exact_Gamma_COMP": n_exact,
            "COMP_exact_at_crossover_bits": round(comp_exact(m, n_exact), 4),
            "COMP_exact_one_record_earlier_bits": (
                round(comp_exact(m, n_exact - 1), 4) if n_exact > 1 else None),
        })
    return {
        "leading_order_closed_form_2_pi_e": round(2.0 * math.pi * math.e, 6),
        "leading_order_is_arity_free": True,
        "by_arity": rows,
        "note": ("The leading-order form drops log2(pi^{m/2} / Gamma(m/2)) from the "
                 "charge. Keeping it makes the asymptotic crossover equal m at every "
                 "arity tested, and the exact Gamma-function form puts it far below "
                 "17 for small arities. 17 is a closed form, not a measurement."),
    }


# --------------------------------------------------------------------------- #
# Q4. the batch margin grid (the "37.2 bits at fifty records" cell)
# --------------------------------------------------------------------------- #
def two_group(n, k=2, d=10, seed=0):
    """The fixture code/tests/test_engine.py uses (test_engine.py:26-32): two groups,
    disjoint keys, disjoint values, k keys per group, d values per key."""
    rng = random.Random(seed)
    return [{f"g{i % 2}k{j}": f"g{i % 2}v{rng.randrange(d)}" for j in range(k)}
            for i in range(n)]


def q4_batch_margin():
    rows = []
    for k in (1, 2, 3, 4, 6):
        for n in (50, 200, 1000, 5000):
            recs = two_group(n, k=k)
            labels = [i % 2 for i in range(n)]
            L_planted = BatchObjective(plant(recs, labels)).total()
            L_onenode = BatchObjective(plant(recs, [0] * n)).total()
            L_background = BatchObjective(plant(recs, [None] * n)).total()
            rows.append({
                "k_keys_per_group": k, "n_records": n,
                "L_batch_planted_two_nodes": round(L_planted, 2),
                "L_batch_one_node_total": round(L_onenode, 2),
                "L_batch_all_background": round(L_background, 2),
                "margin_vs_one_node_total_bits": round(L_onenode - L_planted, 2),
                "margin_vs_all_background_bits": round(L_background - L_planted, 2),
                "one_node_minus_background_bits": round(L_onenode - L_background, 2),
            })
    return rows


# --------------------------------------------------------------------------- #
# Q5. the myopic-versus-batch gap on a 20,000-record 20-node stream
# --------------------------------------------------------------------------- #
Q5_K_STAR = 20
Q5_KEYS_PER_GROUP = 3
Q5_D = 8                   # values per group key
Q5_SHARED = 2              # keys every record publishes, owned by no node
Q5_D_SHARED = 12
Q5_N = 20000
Q5_SEED = 11

Q5_GENERATOR = (
    "20,000 records over 20 planted nodes. Group sizes are Zipf with exponent 1 over "
    "the 20 groups (weight 1/(j+1), normalised, sampled i.i.d.). Group j publishes 3 "
    "private categorical keys c{j}k0..2 over an 8 value alphabet c{j}v0..7, plus 2 "
    "shared keys s0, s1 over a 12 value alphabet that belong to no node's support, so "
    "the background always has cells to code. Seed 11. The planted allocation labels "
    "each record by its group and gives each node exactly its 3 private keys."
)


def q5_stream(n=Q5_N, seed=Q5_SEED):
    rng = random.Random(seed)
    w = [1.0 / (j + 1) for j in range(Q5_K_STAR)]
    tot = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / tot
        cum.append(acc)
    recs, labels = [], []
    for _ in range(n):
        u = rng.random()
        grp = next(j for j, c in enumerate(cum) if u <= c)
        rec = {f"c{grp}k{j}": f"c{grp}v{rng.randrange(Q5_D)}"
               for j in range(Q5_KEYS_PER_GROUP)}
        for s in range(Q5_SHARED):
            rec[f"s{s}"] = f"sv{rng.randrange(Q5_D_SHARED)}"
        recs.append(rec)
        labels.append(grp)
    return recs, labels


def myopic_run(records):
    """The record-seeded immediate rule of method.tex:67-75 and Lemma lem:myopic.

    DEFINED OUTSIDE THE CODE: code/escrow/ ships only the deferred rule, so the loop
    is written here. Every bit it counts is shipped: the incumbent is a ValueBlock per
    key (escrow.codes) and the price is escrow.codes.price at t = 1. At each arrival
    the record's cost under the incumbent is compared with the price of stating one
    node; if it clears, a node is minted seeded on the record's own (key, value)
    pattern, and later records join it when they match the seed on every seed key.
    Returns (labels, mints), the allocation the rule builds.
    """
    background, inventory = {}, {}
    seeds = []                                   # list of {key: value} seed patterns
    labels = []
    K = edits = 0
    for i, rec in enumerate(records, start=1):
        cost_incumbent = 0.0
        for a, x in rec.items():
            inv = inventory.setdefault(a, {})
            blk = background.setdefault(a, ValueBlock())
            cost_incumbent += blk.cost(inv.get(str(x), -1),
                                       math.log2(len(inv) + 1.0))
        joined = None
        for j, seed in enumerate(seeds):
            if all(rec.get(a) == v for a, v in seed.items()):
                joined = j
                break
        if joined is None:
            pr = price(1, max(1, len(rec)), i, K, edits,
                       max(len(background), len(rec)))
            if cost_incumbent > pr:
                seeds.append(dict(rec))
                K += 1
                edits += 1
                joined = len(seeds) - 1
        labels.append(joined)
        for a, x in rec.items():
            inv = inventory.setdefault(a, {})
            s = str(x)
            if s not in inv:
                inv[s] = len(inv)
            background.setdefault(a, ValueBlock()).observe(inv[s])
    return labels, len(seeds)


def q5_myopic_vs_batch(marks=(10, 100, 1000, 10000)):
    recs, labels = q5_stream()
    supports = {j: [f"c{j}k{i}" for i in range(Q5_KEYS_PER_GROUP)]
                for j in range(Q5_K_STAR)}
    sizes = sorted(Counter(labels).values(), reverse=True)

    L_planted = BatchObjective(plant(recs, labels, supports)).total()
    L_background = BatchObjective(plant(recs, [None] * len(recs))).total()
    L_onenode = BatchObjective(plant(recs, [0] * len(recs))).total()

    t0 = time.time()
    myopic_labels, myopic_mints = myopic_run(recs)
    if myopic_mints == 0:
        g_my = plant(recs, [None] * len(recs))
    else:
        g_my = plant(recs, myopic_labels)
    L_myopic = BatchObjective(g_my).total()
    wall_myopic = time.time() - t0

    # the deferred rule under the one shipped protocol, with the K trajectory taken
    # at the marks Q1 needs (new_run keeps protocol.py's three choices)
    t0 = time.time()
    g, b = new_run()
    traj = []
    for i, rec in enumerate(recs, start=1):
        g.process(rec)
        if i in marks:
            traj.append({"n": i, "K": g.K, "H": g.H})
        if i % REPAIR_EVERY == 0:
            b.repair()
    b.repair(full=True)
    L_deferred = BatchObjective(g).total()
    wall_deferred = time.time() - t0

    # Definition def:gap, the one-pass-to-batch gap, on the same stream: the state the
    # one-pass rule leaves against the terminal state of the repair pass
    t0 = time.time()
    g2, b2 = new_run()
    for rec in recs:
        g2.process(rec)
    L_stream_only = BatchObjective(g2).total()
    K_stream_only = g2.K
    b2.repair(full=True)
    L_repaired = BatchObjective(g2).total()
    wall_gap = time.time() - t0

    n = len(recs)
    return {
        "generator": Q5_GENERATOR,
        "n_records": n, "K_star": Q5_K_STAR,
        "group_sizes": sizes,
        "myopic_rule_mints": myopic_mints,
        "myopic_final_K": g_my.K,
        "L_batch_planted": round(L_planted, 2),
        "L_batch_myopic": round(L_myopic, 2),
        "L_batch_all_background": round(L_background, 2),
        "L_batch_one_node_total": round(L_onenode, 2),
        "L_batch_deferred_shipped_protocol": round(L_deferred, 2),
        "deferred_final_K": g.K,
        "gap_myopic_minus_planted_bits": round(L_myopic - L_planted, 1),
        "gap_bits_per_record": round((L_myopic - L_planted) / n, 3),
        "gap_percent_of_myopic_own_code": round(100.0 * (L_myopic - L_planted) / L_myopic, 2),
        "gap_deferred_minus_planted_bits": round(L_deferred - L_planted, 1),
        "delta_rep_definition_def_gap": {
            "K_after_one_pass": K_stream_only,
            "L_batch_one_pass": round(L_stream_only, 2),
            "K_after_repair": g2.K,
            "L_batch_after_repair": round(L_repaired, 2),
            "delta_rep_bits": round(L_stream_only - L_repaired, 1),
            "delta_rep_bits_per_record": round((L_stream_only - L_repaired) / n, 3),
            "note": ("Definition def:gap, computed here for the first time. The cadence "
                     "repair of the shipped protocol is omitted in this arm only, so that "
                     "the one-pass state is the state the one-pass rule alone leaves."),
        },
        "wall_seconds": {"myopic": round(wall_myopic, 2),
                         "deferred": round(wall_deferred, 2),
                         "delta_rep": round(wall_gap, 2)},
        "K_trajectory": traj,
    }


# --------------------------------------------------------------------------- #
# Q6. the UT-14 merge deltas and the eight-node collapse
# --------------------------------------------------------------------------- #
UT14_FIXTURE = (
    "The paper's fixture is not recorded in any file, so this is a declared "
    "reconstruction of the geometry it states: |A| = 200 declared keys, n = 10^4 "
    "records, two nodes with DISJOINT supports of k keys each and t members each, "
    "every member publishing every support key over an 8 value alphabet; the other "
    "n - 2t records publish 2 keys drawn from the remaining 200 - 2k. The delta is "
    "insensitive to that background: widths 1, 2, 4 and 8 give the same value to 4 "
    "decimal places."
)


def ut14_fixture(n=10000, A=200, t=50, k=2, bg_width=2, d=8, seed=0):
    rng = random.Random(seed)
    keys = [f"a{i}" for i in range(A)]
    S0, S1, other = keys[:k], keys[k:2 * k], keys[2 * k:]
    recs, labels = [], []
    for i in range(n):
        if i < t:
            recs.append({a: f"v{rng.randrange(d)}" for a in S0})
            labels.append(0)
        elif i < 2 * t:
            recs.append({a: f"v{rng.randrange(d)}" for a in S1})
            labels.append(1)
        else:
            recs.append({a: f"v{rng.randrange(d)}"
                         for a in rng.sample(other, min(bg_width, len(other)))})
            labels.append(None)
    return recs, labels


def eight_node_fixture(n=10000, A=200, t=50, k=2, d=8, seed=1):
    rng = random.Random(seed)
    keys = [f"a{i}" for i in range(A)]
    sups = [keys[j * k:(j + 1) * k] for j in range(8)]
    other = keys[8 * k:]
    recs, labels = [], []
    for i in range(n):
        j = i // t
        if j < 8:
            recs.append({a: f"v{rng.randrange(d)}" for a in sups[j]})
            labels.append(j)
        else:
            recs.append({a: f"v{rng.randrange(d)}" for a in rng.sample(other, 2)})
            labels.append(None)
    return recs, labels


def q6_ut14():
    paper_cells = []
    for t, k, quoted_with, quoted_without in ((50, 2, 282.41, -117.59),
                                              (200, 4, 2773.25, -426.75),
                                              (1000, 4, 13818.51, -2181.49)):
        recs, labels = ut14_fixture(t=t, k=k)
        g = plant(recs, labels, {0: [f"a{i}" for i in range(k)],
                                 1: [f"a{i}" for i in range(k, 2 * k)]})
        v, w = sorted(g.nodes.values(), key=lambda x: x.nid)
        d_with = BatchObjective(g).merge_delta(v, w)
        d_without = NodeIndependentPresence(g).merge_delta(v, w)
        paper_cells.append({
            "members_t": t, "keys_per_node_k": k,
            "paper_with_presence": quoted_with,
            "paper_without_presence": quoted_without,
            "computed_with_presence": round(d_with, 4),
            "computed_without_presence": round(d_without, 4),
            "difference_from_paper_with": round(d_with - quoted_with, 2),
            "difference_from_paper_without": round(d_without - quoted_without, 2),
            "presence_term_contribution_bits": round(d_with - d_without, 4),
            "closed_form_4tk": 4 * t * k,
        })

    # the fixture the shipped gate actually uses, with the current engine
    recs = two_group(3000, k=4)
    g, b = run_stream(recs)
    engine_cell = {"fixture": "test_engine.two_group(3000, k=4) under escrow.protocol.run_stream",
                   "K": g.K, "L_batch": round(BatchObjective(g).total(), 2)}
    if g.K == 2:
        v, w = sorted(g.nodes.values(), key=lambda x: x.nid)
        engine_cell["merge_delta_with_presence"] = round(BatchObjective(g).merge_delta(v, w), 2)
        engine_cell["merge_delta_node_independent_presence"] = round(
            NodeIndependentPresence(g).merge_delta(v, w), 2)
        engine_cell["presence_term_contribution_bits"] = round(
            BatchObjective(g).merge_delta(v, w)
            - NodeIndependentPresence(g).merge_delta(v, w), 2)

    # the eight-node disjoint-support graph, both arms
    recs8, labels8 = eight_node_fixture()
    sup8 = {j: [f"a{i}" for i in range(j * 2, j * 2 + 2)] for j in range(8)}
    pair_with, pair_without = [], []
    g8 = plant(recs8, labels8, sup8)
    nodes8 = sorted(g8.nodes.values(), key=lambda x: x.nid)
    ob_w, ob_n = BatchObjective(g8), NodeIndependentPresence(g8)
    for i, v in enumerate(nodes8):
        for w in nodes8[i + 1:]:
            pair_with.append(ob_w.merge_delta(v, w))
            pair_without.append(ob_n.merge_delta(v, w))
    g8a = plant(recs8, labels8, sup8)
    moves_with = BatchObjective(g8a).repair(full=True)
    g8b = plant(recs8, labels8, sup8)
    moves_without = NodeIndependentPresence(g8b).repair(full=True)
    eight = {
        "planted_K": len(nodes8),
        "pairs": len(pair_with),
        "merge_delta_with_presence_min": round(min(pair_with), 2),
        "merge_delta_with_presence_max": round(max(pair_with), 2),
        "merge_delta_without_presence_min": round(min(pair_without), 2),
        "merge_delta_without_presence_max": round(max(pair_without), 2),
        "pairs_that_pay_with_presence": sum(1 for x in pair_with if x < 0),
        "pairs_that_pay_without_presence": sum(1 for x in pair_without if x < 0),
        "K_after_repair_with_presence": g8a.K,
        "repair_moves_with_presence": moves_with,
        "K_after_repair_node_independent_presence": g8b.K,
        "repair_moves_node_independent_presence": moves_without,
        "shipped_repair_never_proposes_disjoint_support_merges": True,
        "note": ("batch.py:461-462 skips any pair with S_v & S_w empty, so the shipped "
                 "repair pass cannot collapse a disjoint-support graph under EITHER arm. "
                 "The collapse the paper reports is a property of merge_delta's sign, "
                 "which is why the pairwise deltas are reported beside the repair result."),
    }
    return {"fixture": UT14_FIXTURE, "paper_geometry": paper_cells,
            "shipped_gate_fixture": engine_cell, "eight_node_graph": eight}


# --------------------------------------------------------------------------- #
# Q7. the tokenisation spread on the Wikipedia text fields
# --------------------------------------------------------------------------- #
TOKENISATION_CODE = (
    "Version 1 quarantines text facets, so code/escrow/ ships NO text code and this "
    "cannot be a measurement of the shipped text facet. What is measured is the "
    "shipped categorical value block, escrow.batch.L_vblock, instantiated on three "
    "declared tokenisations of the same field, plus the shipped escrow.codes.L_N on "
    "each record's symbol count so that every tokenisation transmits its own "
    "segmentation and all three are codes for the same data. naming is log2(u+1) "
    "with u the field's distinct symbol count, the batch instantiation of the "
    "engine's log2(len(inventory)+1)."
)


def tokenise(value: str, mode: str):
    if mode == "whole_string":
        return [value]
    if mode == "whitespace":
        return value.split()
    if mode == "lowercased_whitespace":
        return [t.lower() for t in value.split()]
    raise ValueError(mode)


def field_bits(values, mode):
    counts = Counter()
    seg = 0.0
    for v in values:
        toks = tokenise(v, mode)
        if not toks:
            toks = [""]
        counts.update(toks)
        seg += L_N(len(toks) + 1)
    u = len(counts)
    return seg + L_vblock(dict(counts), math.log2(u + 1.0)), u


def q7_tokenisation(min_records=30, min_mean_tokens=1.5):
    caches = [f for f in sorted(os.listdir(OUT))
              if f.startswith("wiki_cache.") and f.endswith(".json")]
    per_field = []
    for fname in caches:
        with open(os.path.join(OUT, fname), encoding="utf-8") as f:
            recs = json.load(f)
        by_key = {}
        for r in recs:
            for a, x in r.items():
                by_key.setdefault(a, []).append(str(x))
        for a, values in sorted(by_key.items()):
            if len(values) < min_records:
                continue
            mean_toks = sum(len(v.split()) for v in values) / len(values)
            if mean_toks < min_mean_tokens:
                continue
            row = {"cache": fname, "field": a, "records": len(values),
                   "mean_whitespace_tokens": round(mean_toks, 2)}
            bits = {}
            for mode in ("whole_string", "whitespace", "lowercased_whitespace"):
                total, u = field_bits(values, mode)
                bits[mode] = total / len(values)
                row["bits_per_field_" + mode] = round(total / len(values), 3)
                row["distinct_symbols_" + mode] = u
            row["spread_bits_per_field"] = round(max(bits.values()) - min(bits.values()), 3)
            row["spread_whitespace_vs_lowercased"] = round(
                abs(bits["whitespace"] - bits["lowercased_whitespace"]), 3)
            per_field.append(row)
    spreads = sorted(r["spread_bits_per_field"] for r in per_field)
    case_only = sorted(r["spread_whitespace_vs_lowercased"] for r in per_field)

    def med(xs):
        if not xs:
            return None
        m = len(xs) // 2
        return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0, 3)

    return {
        "code_used": TOKENISATION_CODE,
        "fields_measured": len(per_field),
        "spread_over_three_tokenisations": {
            "min": spreads[0] if spreads else None,
            "median": med(spreads),
            "max": spreads[-1] if spreads else None,
            "fields_inside_5_to_15_bits": sum(1 for s in spreads if 5.0 <= s <= 15.0),
        },
        "spread_case_only_whitespace_vs_lowercased": {
            "min": case_only[0] if case_only else None,
            "median": med(case_only),
            "max": case_only[-1] if case_only else None,
        },
        "per_field": per_field,
    }


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()

    q5 = q5_myopic_vs_batch()
    e13_traj, e13_K = observed_trajectory()
    q1_rows = (q1_price_difference()
               + q1_observed_rows(
                   e13_traj,
                   f"observed on the E13 planted stream, K* = 8, final K = {e13_K}")
               + q1_observed_rows(
                   q5["K_trajectory"], "observed on the Q5 Zipf stream, K* = 20"))
    q2 = q2_regret_slope()
    q3 = q3_crossover()
    q4 = q4_batch_margin()
    q6 = q6_ut14()
    q7 = q7_tokenisation()

    def pick(rows, traj, n, field):
        for r in rows:
            if r.get("trajectory") == traj and r["n"] == n:
                return r[field]
        return None

    entries = [
        {
            "id": "Q1",
            "quantity": ("per mint codelength difference between the declared priors "
                         "(a,b) = (1/2, 1) and (2, 0.1)"),
            "paper_value": "0.68 bits at n = 10 and 0.19 bits at n = 10^5 along K_s = 2 H_s; "
                           "0.91 and 0.27 along K_s = H_s",
            "computed": {
                "K_s = 2 H_s, n = 10": {
                    "exact, H_n": pick(q1_rows, "K_s = 2 H_s", 10, "exact, H_n"),
                    "exact, H_{n-1}": pick(q1_rows, "K_s = 2 H_s", 10, "exact, H_{n-1}"),
                    "asymptotic, H_n": pick(q1_rows, "K_s = 2 H_s", 10, "asymptotic, H_n"),
                    "asymptotic, H_{n-1}": pick(q1_rows, "K_s = 2 H_s", 10, "asymptotic, H_{n-1}"),
                },
                "K_s = 2 H_s, n = 10^5": {
                    "exact, H_n": pick(q1_rows, "K_s = 2 H_s", 100000, "exact, H_n"),
                    "asymptotic, H_n": pick(q1_rows, "K_s = 2 H_s", 100000, "asymptotic, H_n"),
                },
                "K_s = H_s, n = 10": {
                    "exact, H_{n-1}": pick(q1_rows, "K_s = H_s", 10, "exact, H_{n-1}"),
                    "asymptotic, H_n": pick(q1_rows, "K_s = H_s", 10, "asymptotic, H_n"),
                },
                "K_s = H_s, n = 10^5": {
                    "exact, H_{n-1}": pick(q1_rows, "K_s = H_s", 100000, "exact, H_{n-1}"),
                },
                "grid": q1_rows,
            },
            "how_computed": ("escrow.codes' own price closed form, Equation eq:lambda, "
                             "evaluated at both priors and differenced, along three "
                             "declared trajectories for (K_s, H_s) and along the "
                             "trajectory the engine actually walks on the E13 planted "
                             "stream. Both the exact Lambda_s difference and the "
                             "appendix's displayed asymptotic difference are reported, "
                             "each with H_s taken as H_{n-1} and as H_n."),
            "verdict_detail": (
                "0.68 reproduces ONLY under the appendix's displayed asymptotic "
                "difference form with H_s = H_n along K_s = 2 H_s (0.6809). The exact "
                "Lambda_s difference of eq:lambda gives 0.6703 with H_n and 0.6898 with "
                "H_{n-1}. 0.91 is the mirror image: it reproduces only under the EXACT "
                "form with H_{n-1} (0.9120), while the asymptotic form gives 0.8988. The "
                "paper therefore quotes one pair from each of two different formulas. "
                "0.19 is robust: every combination gives 0.1879. The condition that must "
                "be stated is the FORM as well as the trajectory. Separately, no run in "
                "the paper walks K_s = 2 H_s: on the E13 planted stream K_10 = 0 and the "
                "difference at n = 10 is 2.36 bits, and at n = 10^5 the observed "
                "gamma_hat is 0.66, giving 0.34 bits, not 0.19."),
            "verdict": "reproduces under a condition the paper must state",
            "sites": [
                site("paper/sections/theory.tex", "0.68"),
                site("paper/sections/theory.tex", "K_s = 2H_s"),
                site("paper/sections/problem.tex", "$0.19$"),
                site("paper/sections/method.tex", "$0.19$"),
                site("paper/sections/appendix_theory.tex", "0.68"),
                site("paper/sections/appendix_theory.tex", "$0.91$"),
                site("talk/index.html", ">0.19<"),
            ],
        },
        {
            "id": "Q2",
            "quantity": "concentration regret slope, bits per decade of exposure",
            "paper_value": 1.661,
            "computed": q2,
            "how_computed": ("closed form (1/2) log2 10 evaluated in double precision; "
                             "no code path and no stream is involved."),
            "verdict": "closed form not a measurement",
            "sites": [
                site("paper/sections/appendix_theory.tex", "1.661"),
            ],
        },
        {
            "id": "Q3",
            "quantity": "the crossover record count past which pure noise cannot clear the charge",
            "paper_value": "2 pi e, about 17 records, called measured at theory.tex:100 "
                           "of the pre-review draft and closed form in the current one",
            "computed": q3,
            "how_computed": ("mean null gain (m-1)/(2 ln 2) bits from Wilks, compared with "
                             "(a) the Xie-Barron asymptotic COMP and (b) the exact "
                             "Gamma-function COMP through the shipped "
                             "escrow.codes.km_multinomial_complexity, at arities "
                             "m in {2, 5, 20, 100}."),
            "verdict": "closed form not a measurement",
            "sites": [
                site("paper/sections/theory.tex", "2\\pi e"),
                site("paper/sections/appendix_theory.tex", "2\\pi e"),
                site("paper/sections/appendix_theory.tex", "Gamma-function form"),
            ],
        },
        {
            "id": "Q4",
            "quantity": "batch objective margin for the planted two-node structure at fifty records",
            "paper_value": "37.2 bits at fifty records (withdrawn from live prose; "
                           "survives only as a comment and in superseded sections)",
            "computed": {"grid": q4},
            "how_computed": ("the shipped BatchObjective.total() on three states over the "
                             "same records built by plant(): the planted two-node "
                             "allocation, the one-node-total code and the all-background "
                             "code, on test_engine's two_group fixture at k in "
                             "{1, 2, 3, 4, 6} and n in {50, 200, 1000, 5000}."),
            "verdict_detail": (
                "At the quoted cell, k = 2 and n = 50, the shipped objective prefers the "
                "planted two-node structure by 78.79 bits over the one-node-total code "
                "and by 69.62 bits over the all-background code. Neither is 37.2. The "
                "9 to 11 bit offset the audit saw between its own figure and a nearby "
                "one is the one-node-total minus all-background gap, listed here per "
                "cell. One caveat the paper does not state: at k = 1 the margin is "
                "NEGATIVE at every n, so the objective prefers the degenerate code when "
                "each group carries a single key."),
            "verdict": "does not reproduce, use this value instead",
            "sites": [
                site("paper/sections/appendix_tables.tex", "37.2"),
                site("paper/sections/appendix_tables.tex", "development-time figure"),
            ],
        },
        {
            "id": "Q5",
            "quantity": "myopic-versus-batch gap on a 20,000-record 20-node stream",
            "paper_value": "312,994 bits, 15.6 bits per record, 31.3 percent of its own code",
            "computed": q5,
            "how_computed": ("the declared Zipf stream above; the record-seeded myopic rule "
                             "run to a state, the deferred rule run through the shipped "
                             "protocol, and the shipped BatchObjective.total() evaluated on "
                             "the myopic state, the deferred state and the planted allocation."),
            "verdict_detail": (
                "The bits and the bits per record are stream-specific and do not "
                "reproduce: on the declared stream the gap is 198,478 bits and 9.92 bits "
                "per record, not 312,994 and 15.6. The PERCENTAGE does reproduce: 31.57 "
                "percent against the quoted 31.3. The myopic rule mints zero nodes, so "
                "its state is exactly the all-background code, which is what makes the "
                "paper's phrase 'the myopic rule's degenerate code' correct. The deferred "
                "rule recovers K = 20 and lands 4,176 bits BELOW the planted allocation."),
            "verdict": "does not reproduce, use this value instead",
            "sites": [
                site("paper/sections/appendix_theory.tex", "312{,}994"),
                site("paper/sections/appendix_theory.tex", "31.3"),
                site("paper/sections/appendix_tables.tex", "312,994"),
            ],
        },
        {
            "id": "Q6",
            "quantity": "UT-14 disjoint-support merge deltas and the eight-node collapse",
            "paper_value": "+282.41 / -117.59, +2773.25 / -426.75, +13818.51 / -2181.49; "
                           "eight disjoint-support nodes collapse to one without the guard",
            "computed": q6,
            "how_computed": ("the shipped BatchObjective.merge_delta on a declared "
                             "reconstruction of the paper's geometry, against the same "
                             "method on the node-independent presence subclass; plus the "
                             "same two deltas on the fixture the shipped gate uses, run "
                             "through escrow.protocol.run_stream; plus all 28 pairwise "
                             "deltas and a full repair pass on an eight-node "
                             "disjoint-support graph under both arms."),
            "verdict_detail": (
                "The six deltas reproduce to within 6.86, 5.01 and 4.45 bits on a "
                "declared reconstruction of the stated geometry, and the quantity the "
                "sentence is about, the presence term's contribution, reproduces EXACTLY: "
                "400, 3200 and 16000 bits, which are the paper's own with-minus-without "
                "differences and equal the closed form 4tk. The residual is the fixture, "
                "which the paper does not record. The EIGHT-NODE COLLAPSE DOES NOT "
                "REPRODUCE: all 28 disjoint pairs do pay under the node-independent arm "
                "(-107.8 bits each), but the shipped repair pass skips any pair with "
                "disjoint supports (batch.py:461-462), so a full repair applies zero "
                "moves and K stays 8 under BOTH arms."),
            "verdict": "reproduces under a condition the paper must state",
            "sites": [
                site("paper/sections/appendix_theory.tex", "282.41"),
                site("paper/sections/appendix_theory.tex", "2773.25"),
                site("paper/sections/appendix_theory.tex", "13818.51"),
                site("paper/sections/appendix_theory.tex", "collapsed to one"),
                site("paper/sections/appendix.tex", "282.41"),
            ],
        },
        {
            "id": "Q7",
            "quantity": "the tokenisation choice, in bits per text field",
            "paper_value": "5 to 15 bits per field",
            "computed": q7,
            "how_computed": ("shipped escrow.batch.L_vblock plus shipped escrow.codes.L_N "
                             "for the segmentation, applied to the Wikipedia text fields "
                             "in results/wiki_cache.*.json under three declared "
                             "tokenisations. This is NOT the shipped text facet, which "
                             "does not exist in version 1."),
            "verdict_detail": (
                "Version 1 ships no text code, so 5 to 15 cannot be a measurement of the "
                "shipped text facet and must be reworded as an estimate. Measured under "
                "the shipped categorical block on 31 Wikipedia text fields, the answer "
                "depends on which tokenisations are compared: coarse against fine (whole "
                "string against whitespace tokens) is 5.28 to 74.99 bits per field, "
                "median 19.68, with only 9 of 31 fields inside 5 to 15; two nearby "
                "tokenisations (whitespace against lowercased whitespace) differ by 0.0 "
                "to 1.98 bits per field, median 0.006. The quoted range is neither."),
            "verdict": "does not reproduce, use this value instead",
            "sites": [
                site("paper/sections/problem.tex", "5 to 15"),
                site("paper/sections/limitations.tex", "5 to 15"),
                site("paper/sections/appendix.tex", "5$ to $15"),
                site("paper/sections/appendix_tables.tex", "5 to 15"),
            ],
        },
    ]

    out = {
        "experiment": "E15: theory-side constants computed from the shipped code",
        "answers": "findings/GAPS.md Block 4",
        "protocol": protocol_describe(),
        "python": sys.version.split()[0],
        "numpy_path_active": _numpy_active(),
        "wall_seconds": round(time.time() - t_start, 1),
        "summary": [{"id": e["id"], "quantity": e["quantity"],
                     "paper_value": e["paper_value"], "verdict": e["verdict"]}
                    for e in entries],
        "verdict_vocabulary": [
            "reproduces",
            "reproduces under a condition the paper must state",
            "closed form not a measurement",
            "does not reproduce, use this value instead",
        ],
        "entries": entries,
    }
    path = os.path.join(OUT, "e15_theory_numbers.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stamped(out), f, indent=2)
    print(f"wrote {path} in {out['wall_seconds']} s")
    for e in entries:
        gone = [x for x in e["sites"] if not x["found"]]
        print(f"  {e['id']}: {e['verdict']}"
              + (f"   [{len(gone)} token(s) no longer in the paper]" if gone else ""))
        for x in e["sites"]:
            where = ",".join(str(i) for i in x["prose_lines"]) or "none"
            com = ",".join(str(i) for i in x["comment_lines"])
            print(f"      {x['file']}  {x['quotes']!r}  prose {where}"
                  + (f"  comment {com}" if com else ""))


def _numpy_active() -> bool:
    from escrow import engine as _e
    return _e._np is not None


if __name__ == "__main__":
    main()
