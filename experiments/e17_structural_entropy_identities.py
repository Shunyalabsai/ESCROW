"""E17: the artefact behind Proposition C.1, the relation to structural entropy.

Every displayed identity of the structural-entropy subsection of the theory appendix
(paper/sections/appendix_theory.tex) is proved there by hand. This script is the
independent numerical witness for each of them, so that no line of that subsection rests
on a check with no shipped artefact. It writes results/e17_structural_entropy_identities.json.

What is checked, in the order the appendix proves it.

  C1  1-D closed form.       H^(1)(B) = 1 + (1/2)[H(p_R) + H(p_A)] on random bipartite B.
                             Identity. Expected residual: floating point only.

  C2  Tree slack identity.   H^(1)(B) - H^T(B) = sum_{a != root} ((V_a - g_a)/vol)
                             log(V_{a-}/V_a), for encoding trees of height 1, 2 and 3.
                             Identity. This is the general form; C3 is its height-2 case.

  C3  2-D gain identity.     H^(1)(B) - H^P(B) = sum_j (w(X_j)/M) log(2M/V_j)
                             = sum_a (s_{P(a),a}/M) log(2M/V_{P(a)}), per module and then
                             per atom. Identity, and the second form is the additivity
                             over attribute atoms that the paper concedes.

  C4  The two extremes.      The all-singleton and one-module partitions both evaluate to
                             H^(1)(B) exactly, and a planted co-cluster lies strictly below.

  C5  The bracketing.        0 <= H^T(B) <= H^(1)(B) <= log |R u A|, on random trees.
                             Inequality. The middle one is C2 with g_a <= V_a.

  C6  Coincidence identity.  2M H^{P_Z}(B) = sum_v sum_k q_{v,k} H(c_{v,k})
                             + sum_v Q_v H(kappa_v)
                             + sum_v sum_{r in v} m_r log(2 Q_v / m_r) + M,
                             on synthetic streams with single membership and node-disjoint
                             value supports. Identity.

  C7  KT regret, sign.       L_KT_block(counts, A) - q H(counts/q) >= 0 always, and it
                             tracks ((A-1)/2) log q + O(1). The sign is the proved
                             inequality; the asymptote is the cited standard result, and
                             this is the measurement of the O(1).

  C8  Column price.          L_col(1, n) from code/escrow/codes.py against the exact
                             Gamma-ratio form (identity) and against (3/2) log n
                             + log(2 sqrt(pi)) (bounded gap), with the two-sided Wendel
                             bracket the appendix proves:
                             (3/2)log n + log(2 sqrt pi) - d_n <= L <= (3/2)log n
                             + log(2 sqrt pi), d_n = log(n/(n-1/2)) = O(1/n).

  C9  Key-type invariance.   The Phi-fibre pair of Proposition C.1(iv): two datasets that
                             differ only in how atoms are typed into keys give the same B
                             (identical degree, volume and cut statistics, and an explicit
                             isomorphism), while the key-merge test separates them.
                             Includes the disjoint-alphabet identity Q JS_pi = Q H(pi).

  C10 First-record price.    Lambda_1 = log((b+1)/a) = 2 bits at the declared (a,b)=(1/2,1),
                             the quantity that exists when vol(B) = 0 and H^T does not.

Nothing here is fitted and nothing reads a dataset: C1 to C5 and C9 are graph arithmetic on
seeded random instances, C6 is graph arithmetic on a seeded synthetic stream, and C7, C8 and
C10 call the shipped codelength primitives in code/escrow/codes.py rather than re-implementing
them.

Run:  python code/experiments/e17_structural_entropy_identities.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if os.path.join(_ROOT, "code") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "code"))

from escrow.codes import L_col, L_KT_block  # noqa: E402  (shipped primitives, not re-derived)

LOG2 = math.log(2.0)
TOL = 1e-9


def log2(x: float) -> float:
    return math.log(x) / LOG2


def entropy(counts) -> float:
    """Base-2 Shannon entropy of a count vector, normalised inside."""
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / tot
            h -= p * log2(p)
    return h


# ----------------------------------------------------------------------------------
# Bipartite graph B and Li-Pan structural entropy on it.
# ----------------------------------------------------------------------------------

class Bipartite:
    """B = (R u A, E_B): one unit edge (r, a) iff record r carries atom a."""

    def __init__(self, incidence):
        # incidence: dict record -> set of atoms
        self.inc = {r: set(a) for r, a in incidence.items()}
        self.records = sorted(self.inc)
        atoms = set()
        for a in self.inc.values():
            atoms |= a
        self.atoms = sorted(atoms)
        self.deg = {}
        for r in self.records:
            self.deg[("r", r)] = len(self.inc[r])
        na = defaultdict(int)
        for r in self.records:
            for a in self.inc[r]:
                na[a] += 1
        for a in self.atoms:
            self.deg[("a", a)] = na[a]
        self.M = sum(len(self.inc[r]) for r in self.records)
        self.vol = 2 * self.M
        self.vertices = [("r", r) for r in self.records] + [("a", a) for a in self.atoms]
        self.edges = [(("r", r), ("a", a)) for r in self.records for a in self.inc[r]]

    def h1(self) -> float:
        """H^(1)(B) straight from the definition."""
        return sum(-(self.deg[v] / self.vol) * log2(self.deg[v] / self.vol)
                   for v in self.vertices if self.deg[v] > 0)

    def h1_closed(self) -> float:
        """1 + (1/2)[H(p_R) + H(p_A)]."""
        pr = [self.deg[("r", r)] for r in self.records]
        pa = [self.deg[("a", a)] for a in self.atoms]
        return 1.0 + 0.5 * (entropy(pr) + entropy(pa))

    def volume(self, vset) -> int:
        return sum(self.deg[v] for v in vset)

    def cut(self, vset) -> int:
        s = set(vset)
        return sum(1 for (u, v) in self.edges if (u in s) != (v in s))

    def internal(self, vset) -> int:
        s = set(vset)
        return sum(1 for (u, v) in self.edges if (u in s) and (v in s))


def tree_nodes(tree, out=None, parent=None):
    """Walk a nested-list encoding tree. Yields (vertex set, parent vertex set).

    A tree is a nested list: the root is the list of its children, a child is either a
    vertex (a leaf) or again a list. Every non-root node is reported once.
    """
    if out is None:
        out = []
    flat = flatten(tree)
    for child in tree:
        cset = flatten(child) if isinstance(child, list) else [child]
        out.append((cset, flat))
        if isinstance(child, list):
            tree_nodes(child, out, cset)
    return out


def flatten(node):
    if not isinstance(node, list):
        return [node]
    acc = []
    for c in node:
        acc.extend(flatten(c))
    return acc


def h_tree(B: Bipartite, tree) -> float:
    """H^T(B) = sum over non-root alpha of -(g_alpha/vol) log(V_alpha / V_{alpha^-})."""
    total = 0.0
    for cset, pset in tree_nodes(tree):
        Va, Vp = B.volume(cset), B.volume(pset)
        if Va <= 0 or Vp <= 0:
            continue
        total += -(B.cut(cset) / B.vol) * log2(Va / Vp)
    return total


def h_tree_slack(B: Bipartite, tree) -> float:
    """sum over non-root alpha of ((V_alpha - g_alpha)/vol) log(V_{alpha^-}/V_alpha)."""
    total = 0.0
    for cset, pset in tree_nodes(tree):
        Va, Vp = B.volume(cset), B.volume(pset)
        if Va <= 0 or Vp <= 0:
            continue
        total += ((Va - B.cut(cset)) / B.vol) * log2(Vp / Va)
    return total


def random_bipartite(rng, n_rec=14, n_atoms=9, p=0.35) -> Bipartite:
    while True:
        inc = {}
        for r in range(n_rec):
            carried = {a for a in range(n_atoms) if rng.random() < p}
            if not carried:
                carried = {rng.randrange(n_atoms)}
            inc[r] = carried
        B = Bipartite(inc)
        # Li-Pan structural entropy needs every vertex to have positive degree.
        if all(B.deg[v] > 0 for v in B.vertices) and B.M > 0:
            return B


def random_partition(rng, items, L):
    parts = [[] for _ in range(L)]
    for it in items:
        parts[rng.randrange(L)].append(it)
    return [p for p in parts if p]


# ----------------------------------------------------------------------------------
# C1, C3, C4: closed forms, the gain identity, the two extremes.
# ----------------------------------------------------------------------------------

def check_c1_c3_c4(rng, trials=200):
    e1 = e_mod = e_atom = 0.0
    e_single = e_onemod = 0.0
    strict_below = 0
    for _ in range(trials):
        B = random_bipartite(rng)
        e1 = max(e1, abs(B.h1() - B.h1_closed()))

        P = random_partition(rng, B.vertices, rng.randrange(2, 5))
        HP = h_tree(B, [list(p) for p in P])
        gain = B.h1() - HP

        per_module = sum((B.internal(X) / B.M) * log2(B.vol / B.volume(X))
                         for X in P if B.volume(X) > 0)
        e_mod = max(e_mod, abs(gain - per_module))

        # per-atom form: each internal edge has exactly one atom endpoint
        owner = {}
        for j, X in enumerate(P):
            for v in X:
                owner[v] = j
        per_atom = 0.0
        for a in B.atoms:
            j = owner[("a", a)]
            s_ja = sum(1 for r in B.records
                       if a in B.inc[r] and owner[("r", r)] == j)
            per_atom += (s_ja / B.M) * log2(B.vol / B.volume(P[j]))
        e_atom = max(e_atom, abs(gain - per_atom))

        singles = [[v] for v in B.vertices]
        e_single = max(e_single, abs(h_tree(B, singles) - B.h1()))
        e_onemod = max(e_onemod, abs(h_tree(B, [list(B.vertices)]) - B.h1()))
        if HP < B.h1() - 1e-12:
            strict_below += 1

    # a planted co-cluster, to show the gain is strictly positive somewhere
    inc = {}
    for r in range(8):
        inc[r] = {0, 1, 2} if r < 4 else {3, 4, 5}
    Bc = Bipartite(inc)
    Pc = [[("r", r) for r in range(4)] + [("a", a) for a in (0, 1, 2)],
          [("r", r) for r in range(4, 8)] + [("a", a) for a in (3, 4, 5)]]
    planted_h1, planted_hp = Bc.h1(), h_tree(Bc, [list(x) for x in Pc])

    return {
        "trials": trials,
        "C1_h1_closed_form_max_abs_err": e1,
        "C3_gain_per_module_max_abs_err": e_mod,
        "C3_gain_per_atom_max_abs_err": e_atom,
        "C4_all_singleton_minus_h1_max_abs": e_single,
        "C4_one_module_minus_h1_max_abs": e_onemod,
        "C4_partitions_strictly_below_h1": strict_below,
        "C4_planted_cocluster_h1": planted_h1,
        "C4_planted_cocluster_hp": planted_hp,
        "C4_planted_gain_positive": planted_hp < planted_h1 - 1e-12,
        "verdict": "PASS" if max(e1, e_mod, e_atom, e_single, e_onemod) < TOL else "FAIL",
    }


# ----------------------------------------------------------------------------------
# C2, C5: the general tree slack identity and the bracketing.
# ----------------------------------------------------------------------------------

def random_tree(rng, B):
    """A random encoding tree of height 1, 2 or 3 over B's vertices."""
    height = rng.choice([1, 2, 3])
    verts = list(B.vertices)
    rng.shuffle(verts)
    if height == 1:
        return [[v] for v in verts], height
    mods = random_partition(rng, verts, rng.randrange(2, 5))
    if height == 2:
        return [[[v] for v in X] for X in mods], height
    out = []
    for X in mods:
        if len(X) <= 2:
            out.append([[v] for v in X])
        else:
            subs = random_partition(rng, X, rng.randrange(2, min(4, len(X)) + 1))
            out.append([[[v] for v in S] for S in subs])
    return out, height


def check_c2_c5(rng, trials=300):
    e_slack = 0.0
    bracket_ok = True
    worst = None
    by_height = defaultdict(int)
    for _ in range(trials):
        B = random_bipartite(rng)
        T, height = random_tree(rng, B)
        by_height[height] += 1
        HT = h_tree(B, T)
        H1 = B.h1()
        e = abs((H1 - HT) - h_tree_slack(B, T))
        if e > e_slack:
            e_slack, worst = e, height
        cap = log2(len(B.vertices))
        if not (-1e-12 <= HT <= H1 + 1e-12 <= cap + 1e-12):
            bracket_ok = False
    return {
        "trials": trials,
        "heights_sampled": dict(by_height),
        "C2_slack_identity_max_abs_err": e_slack,
        "C2_worst_at_height": worst,
        "C5_bracket_0_le_HT_le_H1_le_log_n_holds": bracket_ok,
        "verdict": "PASS" if e_slack < TOL and bracket_ok else "FAIL",
    }


# ----------------------------------------------------------------------------------
# C6: the coincidence identity.
# ----------------------------------------------------------------------------------

def synth_disjoint_stream(rng, K=4, n_keys=5, n_rec=40):
    """Single membership, node-disjoint value supports.

    Every node owns a private value alphabet per key, so no atom (k, u) is ever carried by
    records of two different nodes: this is exactly the hypothesis of Proposition C.1(ii).
    """
    records = []
    for i in range(n_rec):
        v = rng.randrange(K)
        keys = [k for k in range(n_keys) if rng.random() < 0.6]
        if not keys:
            keys = [rng.randrange(n_keys)]
        facets = {}
        for k in keys:
            u = "v%d_k%d_u%d" % (v, k, rng.randrange(3))
            facets[k] = u
        records.append((i, v, facets))
    return records


def check_c6(rng, trials=40):
    worst = 0.0
    sample = None
    for _ in range(trials):
        records = synth_disjoint_stream(rng)
        inc = {i: {(k, u) for k, u in f.items()} for (i, v, f) in records}
        B = Bipartite(inc)

        members = defaultdict(list)
        for (i, v, f) in records:
            members[v].append((i, f))

        # the induced partition P_Z: records of v, plus the atoms they carry
        P = []
        for v, mem in sorted(members.items()):
            X = [("r", i) for (i, f) in mem]
            owned = set()
            for (i, f) in mem:
                owned |= {(k, u) for k, u in f.items()}
            X += [("a", a) for a in sorted(owned)]
            P.append(X)
        # sanity: the modules must partition R u A and every edge must be internal
        seen = set()
        for X in P:
            assert not (seen & set(X)), "modules overlap: supports were not disjoint"
            seen |= set(X)
        assert seen == set(B.vertices)
        assert sum(B.cut(X) for X in P) == 0, "an edge crossed a module"

        lhs = B.vol * h_tree(B, [list(X) for X in P])

        term_value = term_key = term_rec = 0.0
        for v, mem in sorted(members.items()):
            q = defaultdict(int)
            c = defaultdict(lambda: defaultdict(int))
            for (i, f) in mem:
                for k, u in f.items():
                    q[k] += 1
                    c[k][u] += 1
            Q = sum(q.values())
            for k in q:
                term_value += q[k] * entropy(list(c[k].values()))
            term_key += Q * entropy([q[k] for k in sorted(q)])
            for (i, f) in mem:
                m_r = len(f)
                term_rec += m_r * log2(2.0 * Q / m_r)
        rhs = term_value + term_key + term_rec + B.M

        err = abs(lhs - rhs)
        if err > worst:
            worst = err
            sample = {"2M_HPZ": lhs, "ml_plugin_value_cost": term_value,
                      "key_landing": term_key, "record_landing": term_rec, "M": B.M,
                      "rhs": rhs}
    return {
        "trials": trials,
        "C6_coincidence_max_abs_err": worst,
        "C6_worst_instance": sample,
        "verdict": "PASS" if worst < 1e-8 else "FAIL",
    }


# ----------------------------------------------------------------------------------
# C7: the KT regret against the maximum-likelihood plug-in.
# ----------------------------------------------------------------------------------

def check_c7(rng, trials=400):
    min_regret = float("inf")
    rows = []
    negative = 0
    for _ in range(trials):
        A = rng.randrange(2, 9)
        q = rng.choice([10, 50, 200, 1000, 5000])
        counts = [0] * A
        for _ in range(q):
            counts[rng.randrange(A)] += 1
        plug_in = q * entropy(counts)
        kt = L_KT_block(counts, A)
        regret = kt - plug_in
        if regret < -1e-9:
            negative += 1
        min_regret = min(min_regret, regret)
        rows.append((A, q, regret - 0.5 * (A - 1) * log2(q)))
    # the residual after subtracting ((A-1)/2) log q, which the appendix calls O(1)
    resid = [r for (_, _, r) in rows]
    return {
        "trials": trials,
        "C7_min_regret_over_ml_plugin": min_regret,
        "C7_count_of_negative_regrets": negative,
        "C7_residual_after_half_A_minus_1_log_q": {
            "min": min(resid), "max": max(resid),
            "mean": sum(resid) / len(resid),
        },
        "verdict": "PASS" if negative == 0 and min_regret >= -1e-9 else "FAIL",
    }


# ----------------------------------------------------------------------------------
# C8: the fresh membership column price.
# ----------------------------------------------------------------------------------

def check_c8():
    const = log2(2.0 * math.sqrt(math.pi))
    rows = []
    ok = True
    for n in [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]:
        shipped = L_col(1, n)                       # code/escrow/codes.py
        gamma_form = (log2(2.0 * n * math.sqrt(math.pi))
                      + (math.lgamma(n) - math.lgamma(n - 0.5)) / LOG2)
        asymptote = 1.5 * log2(n) + const
        delta_n = log2(n / (n - 0.5))               # the Wendel bracket width
        rows.append({
            "n": n,
            "shipped_L_col_1_n": shipped,
            "exact_gamma_ratio_form": gamma_form,
            "identity_abs_err": abs(shipped - gamma_form),
            "asymptote_1p5_log_n_plus_log_2sqrtpi": asymptote,
            "gap_asymptote_minus_exact": asymptote - shipped,
            "wendel_bracket_width_delta_n": delta_n,
            "within_bracket": (asymptote - delta_n - 1e-12) <= shipped <= (asymptote + 1e-12),
        })
        # the identity is exact in exact arithmetic; the tolerance is relative because
        # lgamma(n) - lgamma(n - 1/2) loses absolute precision as n grows.
        rel = rows[-1]["identity_abs_err"] / max(1.0, abs(shipped))
        rows[-1]["identity_rel_err"] = rel
        if not rows[-1]["within_bracket"] or rel > 1e-9:
            ok = False
    return {"rows": rows,
            "C8_constant_log2_2sqrtpi": const,
            "verdict": "PASS" if ok else "FAIL"}


# ----------------------------------------------------------------------------------
# C9: key-type invariance, the Phi-fibre pair.
# ----------------------------------------------------------------------------------

def js_pi(p, q, pi):
    """Jensen-Shannon divergence at mixing weight pi, in bits."""
    keys = set(p) | set(q)
    mix = {u: pi * p.get(u, 0.0) + (1 - pi) * q.get(u, 0.0) for u in keys}
    h = lambda d: -sum(v * log2(v) for v in d.values() if v > 0)
    return h(mix) - pi * h(p) - (1 - pi) * h(q)


def check_c9(t=50):
    # D  : one key with two values.   D' : two keys with one value each.
    # Both give the same B: 2t record vertices of degree 1, two atom vertices of degree t.
    incD = {r: {("k", "u")} for r in range(t)}
    incD.update({t + r: {("k", "uprime")} for r in range(t)})
    incDp = {r: {("k1", "u")} for r in range(t)}
    incDp.update({t + r: {("k2", "uprime")} for r in range(t)})
    BD, BDp = Bipartite(incD), Bipartite(incDp)

    inv = lambda B: (sorted(B.deg.values()), B.M, B.vol,
                     sorted(B.cut([v]) for v in B.vertices))
    same_graph = inv(BD) == inv(BDp) and abs(BD.h1() - BDp.h1()) < 1e-12

    # every partition of one lifts to the other under the obvious relabelling
    relabel = {("a", ("k", "u")): ("a", ("k1", "u")),
               ("a", ("k", "uprime")): ("a", ("k2", "uprime"))}
    relabel.update({("r", r): ("r", r) for r in range(2 * t)})
    rng = random.Random(7)
    max_tree_gap = 0.0
    for _ in range(200):
        P = random_partition(rng, BD.vertices, rng.randrange(2, 5))
        Pp = [[relabel[v] for v in X] for X in P]
        max_tree_gap = max(max_tree_gap,
                           abs(h_tree(BD, [list(X) for X in P])
                               - h_tree(BDp, [list(X) for X in Pp])))

    # the key-merge test on D', where the question is well posed
    Q = 2 * t
    pi = 0.5
    lhs = Q * js_pi({"u": 1.0}, {"uprime": 1.0}, pi)     # disjoint alphabets
    disjoint_identity_err = abs(js_pi({"u": 1.0}, {"uprime": 1.0}, pi) - entropy([1, 1]))
    # parameter saving: |V_k1| = |V_k2| = 1, |V_union| = 2
    rhs = (0.0 + 0.0) - 0.5 * log2(Q)
    merge = lhs < rhs

    # the value-level penalty in D: a group-two record at a group-one node, A_k = 2 + escape
    A_k = 3
    q_vk = t
    value_penalty = log2((0 + 0.5) / (q_vk + A_k / 2.0))

    return {
        "t_records_per_group": t,
        "C9_same_graph_invariants": same_graph,
        "C9_max_HT_gap_over_200_partitions": max_tree_gap,
        "C9_disjoint_alphabet_JS_equals_H_pi_abs_err": disjoint_identity_err,
        "C9_merge_divergence_side_Q_JS": lhs,
        "C9_merge_parameter_side": rhs,
        "C9_merge_verdict": "merge" if merge else "distinct",
        "C9_value_level_penalty_bits": value_penalty,
        "C9_value_level_penalty_vs_minus_log_2q": value_penalty + log2(2 * q_vk),
        "verdict": "PASS" if (same_graph and max_tree_gap < 1e-12
                              and disjoint_identity_err < 1e-12
                              and not merge) else "FAIL",
    }


# ----------------------------------------------------------------------------------
# C10: the price that exists when vol(B) = 0.
# ----------------------------------------------------------------------------------

def check_c10(a=0.5, b=1.0):
    H0 = 0.0
    K0 = 0
    lam1 = log2(1 * (b + H0) + 1) - log2(a + K0)
    return {"a": a, "b": b, "Lambda_1_bits": lam1,
            "verdict": "PASS" if abs(lam1 - 2.0) < 1e-12 else "FAIL"}


def main():
    rng = random.Random(20260906)
    out = {
        "what": "numerical witness for every displayed identity of Proposition C.1 "
                "(the relation to structural entropy) in paper/sections/appendix_theory.tex",
        "seed": 20260906,
        "C1_C3_C4_closed_forms_gain_extremes": check_c1_c3_c4(rng),
        "C2_C5_tree_slack_and_bracket": check_c2_c5(rng),
        "C6_coincidence_identity": check_c6(rng),
        "C7_kt_regret_sign_and_size": check_c7(rng),
        "C8_fresh_column_price": check_c8(),
        "C9_key_type_invariance": check_c9(),
        "C10_first_record_price": check_c10(),
    }
    verdicts = {k: v["verdict"] for k, v in out.items() if isinstance(v, dict) and "verdict" in v}
    out["verdicts"] = verdicts
    out["all_pass"] = all(v == "PASS" for v in verdicts.values())

    path = os.path.join(_ROOT, "results", "e17_structural_entropy_identities.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)

    for k, v in verdicts.items():
        print("%-42s %s" % (k, v))
    print("all_pass:", out["all_pass"])
    print("wrote", path)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
