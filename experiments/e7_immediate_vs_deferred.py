"""E7: immediate vs deferred mint, run as a falsification, not a horse race.

Predictions (plan section 5.10; DERIVATIONS.md Lemma 2/3):
  P1  UNSEEDED immediate mint produces ZERO nodes; the realised per-attempt
      deficit tracks sum_a KL(qhat_a || Uniform_a) + C(u).
  P2  RECORD-SEEDED immediate mint's rate tracks the fraction of cells with
      qhat_a(x) < 1/m_a, and on a typo-injected stream the nodes it mints are
      typos that are never reused.
  P3  The DEFERRED mint (the shipped engine) mints the true structure and
      ignores the typos.

Three things this file now measures that it did not before.
  (a) Variance. Every arm runs over seeds 0 to 4 and the file records per-seed
      values with mean, min and max (audit Block 13: no variance anywhere).
  (b) The P2 curve. A sweep over alphabet size d and typo rate records the
      seeded arm's mint count in each cell beside the fraction of cells whose
      value has empirical probability below 1/m_a, which is the quantity P2
      predicted the mint rate would track. Under the full node price the answer
      is zero everywhere, so the paper can print a measured zero instead of
      asserting a failure that does not happen (audit Block 10).
  (c) A fourth arm. The record-seeded immediate rule with a CONSTANT price in
      bits is the DP-means shape the method section actually describes. That is
      the rule under which minting on typos does happen, and this arm measures
      at which constant it happens, how many of the minted nodes are never
      reused, and how many are seeded by a typo record.
"""
import json
import math
import random
import statistics
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.codes import ValueBlock, kt, price
from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective
from escrow.protocol import run_stream, new_run, describe as protocol_describe
from escrow.provenance import stamped

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)

SEEDS = (0, 1, 2, 3, 4)

# ---- the stream: two latent groups + occasional typo records ---------------- #
def stream(n, k=3, d=8, typo_rate=0.01, seed=0):
    r = random.Random(seed)
    out = []
    for i in range(n):
        grp = i % 2
        rec = {f"g{grp}k{j}": f"g{grp}v{r.randrange(d)}" for j in range(k)}
        if r.random() < typo_rate:                       # one corrupted value
            key = r.choice(list(rec))
            rec[key] = f"TYPO{r.randrange(10**6)}"       # a value seen once, ever
        out.append(rec)
    return out

N = 6000

# ---- Arm 1: unseeded immediate mint ---------------------------------------- #
# Mint iff the record pays for a fresh EMPTY node on arrival: the fresh node's
# blocks are empty, so its cost on every key is the uninformed one.
def arm_unseeded(records):
    g = EscrowGraph()
    mints, deficits = 0, []
    for rec in records:
        n = g.n + 1
        # fresh-node cost vs background cost, computed pre-update via one probe
        # of the engine's own primitives
        saving = 0.0
        for a, x in rec.items():
            ki = g.keys.get(a)
            if ki is None or g.n == 0:
                continue
            naming = math.log2(len(ki.inventory) + 1.0)
            bg = ki.background.cost(str(x) and ki.inventory.get(str(x)) is not None
                                     and ki.inventory[str(x)] or -1, naming) \
                 if False else ki.background.cost(ki.inventory.get(str(x), -1), naming)
            fresh = ValueBlock().cost(0, naming)         # empty block: naming only
            saving += bg - fresh
        pr = price(1, max(1, len(rec)), n, g.K, g.e, max(len(g.keys), len(rec)))
        if saving > pr:
            mints += 1
        else:
            deficits.append(pr - saving)
        g.process(rec)                                    # graph evolves normally
    return mints, deficits


# ---- Arm 2: record-seeded immediate mint ----------------------------------- #
# The fresh node is seeded FROM the record (DP-means style): its cost on every
# key is ~0, so it mints exactly when the record's own background cost exceeds
# the price - i.e. on rare values.
def arm_seeded(records):
    g = EscrowGraph()
    minted = []                                          # (n, was_typo)
    prices = []
    for rec in records:
        n = g.n + 1
        cost_bg = 0.0
        for a, x in rec.items():
            ki = g.keys.get(a)
            if ki is None or g.n == 0:
                continue
            naming = math.log2(len(ki.inventory) + 1.0)
            cost_bg += ki.background.cost(ki.inventory.get(str(x), -1), naming)
        pr = price(1, max(1, len(rec)), n, g.K, g.e, max(len(g.keys), len(rec)))
        prices.append(pr)
        if cost_bg > pr:
            minted.append((n, any(v.startswith("TYPO") for v in rec.values())))
        g.process(rec)
    return minted, prices


# ---- why the seeded arm never mints: the size of the gap, in bits ----------- #
def seeded_cost_profile(records):
    """The two quantities the zero is made of: what a record is worth to the
    seeded rule (its whole cost under the incumbent background) against what a
    node costs (the full price). Split by whether the record carries a typo, and
    reported for the single rarest cell as well, since 'mints on rare values'
    is a claim about one cell, not about a whole record."""
    g = EscrowGraph()
    rows = {True: [], False: []}
    cell_max = {True: [], False: []}
    prices = []
    for rec in records:
        n = g.n + 1
        cost_bg, cells = 0.0, [0.0]
        for a, x in rec.items():
            ki = g.keys.get(a)
            if ki is None or g.n == 0:
                continue
            naming = math.log2(len(ki.inventory) + 1.0)
            c = ki.background.cost(ki.inventory.get(str(x), -1), naming)
            cost_bg += c
            cells.append(c)
        is_typo = any(str(v).startswith("TYPO") for v in rec.values())
        rows[is_typo].append(cost_bg)
        cell_max[is_typo].append(max(cells))
        prices.append(price(1, max(1, len(rec)), n, g.K, g.e,
                            max(len(g.keys), len(rec))))
        g.process(rec)
    def mean(v):
        return round(sum(v) / len(v), 2) if v else None
    return {"mean_price_bits": mean(prices),
            "typo_records": len(rows[True]),
            "mean_record_cost_bits_typo": mean(rows[True]),
            "mean_record_cost_bits_clean": mean(rows[False]),
            "mean_rarest_cell_bits_typo": mean(cell_max[True]),
            "mean_rarest_cell_bits_clean": mean(cell_max[False]),
            "max_record_cost_bits": round(max(rows[True] + rows[False]), 2),
            "deficit_at_the_most_expensive_record_bits":
                round(mean(prices) - max(rows[True] + rows[False]), 2)}


# ---- Arm 4: the record-seeded rule at a CONSTANT price (the DP-means shape) -- #
def arm_constant_price(records, lam, max_nodes=10**9):
    """The rule method.tex describes: a fresh node seeded from the record is
    created whenever no incumbent codes the record within `lam` bits, and the
    record then joins the cheapest incumbent. Bookkeeping is this arm's own so
    the deferred engine cannot move cells underneath it: every cell is coded by
    exactly one block, its owner's, exactly as in L_batch."""
    inventory, background, nodes = {}, {}, []
    cap_hit = False
    for i, rec in enumerate(records, start=1):
        cells = []
        for a, x in rec.items():
            inv = inventory.setdefault(a, {})
            bg = background.setdefault(a, ValueBlock())
            naming = math.log2(len(inv) + 1.0)           # PRE-intern, as the engine
            xs = str(x)
            vid = inv.get(xs)
            if vid is None:
                vid = inv[xs] = len(inv)
            cells.append((a, vid, naming, bg))
        best, best_cost = None, sum(bg.cost(v, nm) for _, v, nm, bg in cells)
        for nd in nodes:
            c = 0.0
            for a, v, nm, bg in cells:
                blk = nd["blocks"].get(a)
                c += blk.cost(v, nm) if blk is not None else bg.cost(v, nm)
                if c >= best_cost:
                    break
            if c < best_cost:
                best, best_cost = nd, c
        if best_cost > lam:
            if len(nodes) < max_nodes:
                best = {"blocks": {a: ValueBlock() for a, _, _, _ in cells}, "t": 0,
                        "birth": i,
                        "typo_seed": any(str(v).startswith("TYPO") for v in rec.values())}
                nodes.append(best)
            else:
                cap_hit = True
        if best is None:
            for _, v, nm, bg in cells:
                bg.observe(v)
        else:
            best["t"] += 1
            for a, v, nm, bg in cells:
                blk = best["blocks"].get(a)
                (blk if blk is not None else bg).observe(v)
    return nodes, cap_hit


# ---- the P2 statistic: cells rarer than uniform over their key's alphabet ---- #
def rare_cell_fraction(records):
    """Two readings of 'qhat_a(x) < 1/m_a'. `final` uses the whole-stream
    empirical distribution; `prequential` uses only the past, which is what the
    arrival-time decision actually sees. m_a is the number of distinct values of
    key a (final: over the whole stream; prequential: over the past)."""
    counts, totals = {}, {}
    for rec in records:
        for a, x in rec.items():
            counts.setdefault(a, {})[str(x)] = counts.setdefault(a, {}).get(str(x), 0) + 1
            totals[a] = totals.get(a, 0) + 1
    cells = rare = 0
    for rec in records:
        for a, x in rec.items():
            cells += 1
            m = len(counts[a])
            if counts[a][str(x)] / totals[a] < 1.0 / m:
                rare += 1
    seen, tot2 = {}, {}
    cells2 = rare2 = 0
    for rec in records:
        for a, x in rec.items():
            xs = str(x)
            c = seen.setdefault(a, {})
            t = tot2.get(a, 0)
            m = max(1, len(c))
            cells2 += 1
            if t == 0 or (c.get(xs, 0) / t) < 1.0 / m:
                rare2 += 1
            c[xs] = c.get(xs, 0) + 1
            tot2[a] = t + 1
    return {"final": round(rare / max(1, cells), 4),
            "prequential": round(rare2 / max(1, cells2), 4)}


def spread(vals):
    vals = list(vals)
    return {"mean": round(statistics.fmean(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def one_seed(seed, n=N):
    recs = stream(n, seed=seed)
    m1, deficits = arm_unseeded(recs)
    seeded_mints, prices = arm_seeded(recs)
    g, b = run_stream(recs)
    typo_nodes = [m for m in g.mint_log
                  if any(str(v).startswith("TYPO") for v in m["per_key_bits"])]
    return recs, {
        "seed": seed,
        "unseeded_mints": m1,
        "unseeded_mean_deficit_bits": round(sum(deficits) / len(deficits), 2),
        "seeded_mints": len(seeded_mints),
        "seeded_typo_fraction": round(sum(1 for _, t in seeded_mints if t)
                                      / max(1, len(seeded_mints)), 3),
        "mean_price_bits": round(sum(prices) / len(prices), 2),
        "deferred_final_K": g.K,
        "deferred_total_mints": len(g.mint_log),
        "deferred_typo_justified_nodes": len(typo_nodes),
        "deferred_supports": sorted(sorted(g.key_name[k] for k in v.S)
                                    for v in g.nodes.values()),
    }


def main():
    per_seed = []
    recs0 = None
    for s in SEEDS:
        recs, row = one_seed(s)
        if s == 0:
            recs0 = recs
        per_seed.append(row)
        print("seed", s, {k: v for k, v in row.items() if k != "deferred_supports"})

    z = per_seed[0]
    res = {
        "n": N,
        "protocol": protocol_describe(),
        "unseeded": {"mints": z["unseeded_mints"],
                     "mean_deficit_bits": z["unseeded_mean_deficit_bits"],
                     "prediction": "0 mints; deficit = sum KL(q||U) + C(u)"},
        "seeded": {"mints": z["seeded_mints"],
                   "typo_fraction": z["seeded_typo_fraction"],
                   "prediction": "mints on rare values, i.e. on typos"},
        "deferred": {"final_K": z["deferred_final_K"],
                     "total_mints": z["deferred_total_mints"],
                     "typo_justified_nodes": z["deferred_typo_justified_nodes"],
                     "supports": z["deferred_supports"],
                     "prediction": "K=2, zero typo nodes"},
    }

    # ---- (a) variance over seeds 0 to 4 ------------------------------------- #
    quantities = ["unseeded_mints", "unseeded_mean_deficit_bits", "seeded_mints",
                  "mean_price_bits", "deferred_final_K", "deferred_total_mints",
                  "deferred_typo_justified_nodes"]
    res["seeds"] = {
        "seeds": list(SEEDS),
        "note": "the single-seed rows above are seed 0; only the stream seed moves",
        "per_seed": [{k: v for k, v in r.items() if k != "deferred_supports"}
                     for r in per_seed],
        "spread": {q: spread(r[q] for r in per_seed) for q in quantities},
        "deferred_supports_identical_across_seeds":
            len({json.dumps(r["deferred_supports"]) for r in per_seed}) == 1,
    }
    print("spread", json.dumps(res["seeds"]["spread"], indent=2))

    # ---- (b) the P2 sweep: alphabet size by typo rate ----------------------- #
    sweep = []
    for d in (2, 4, 8, 32, 100):
        for tr in (0.0, 0.01, 0.05, 0.2):
            recs = stream(N, k=3, d=d, typo_rate=tr, seed=0)
            minted, prices = arm_seeded(recs)
            frac = rare_cell_fraction(recs)
            sweep.append({"d": d, "typo_rate": tr, "seeded_mints": len(minted),
                          "rare_cell_fraction_final": frac["final"],
                          "rare_cell_fraction_prequential": frac["prequential"],
                          "mean_price_bits": round(sum(prices) / len(prices), 2)})
            print("sweep", sweep[-1])
    res["seeded_deficit_profile"] = seeded_cost_profile(stream(N, seed=0))
    print("deficit profile", json.dumps(res["seeded_deficit_profile"]))
    res["seeded_sweep"] = {
        "design": "d in {2,4,8,32,100} x typo_rate in {0,0.01,0.05,0.2}, k=3, "
                  f"n={N}, seed 0; arm 2 (record-seeded immediate) unchanged",
        "p2_statistic": "rare_cell_fraction is the fraction of published cells whose "
                        "value has empirical probability below 1/m_a; P2 predicted the "
                        "seeded mint rate would track it",
        "grid": sweep,
        "total_seeded_mints": sum(c["seeded_mints"] for c in sweep),
        "rare_cell_fraction_range": [min(c["rare_cell_fraction_final"] for c in sweep),
                                     max(c["rare_cell_fraction_final"] for c in sweep)],
        "verdict": None,      # filled below
    }

    # ---- (c) arm 4: the same rule at a constant price ----------------------- #
    lam_rows = []
    for lam in (2.0, 4.0, 8.0, 16.0, 39.0):
        nodes, cap_hit = arm_constant_price(stream(N, seed=0), lam)
        never = sum(1 for nd in nodes if nd["t"] <= 1)
        typo = sum(1 for nd in nodes if nd["typo_seed"])
        typo_never = sum(1 for nd in nodes if nd["typo_seed"] and nd["t"] <= 1)
        lam_rows.append({
            "price_bits": lam, "mints": len(nodes),
            "never_reused": never,
            "never_reused_fraction": round(never / max(1, len(nodes)), 3),
            "typo_seeded_nodes": typo,
            "typo_seeded_fraction": round(typo / max(1, len(nodes)), 3),
            "typo_seeded_and_never_reused": typo_never,
            "node_cap_hit": cap_hit})
        print("constant price", lam_rows[-1])
    typo_recs = sum(1 for r in stream(N, seed=0)
                    if any(str(v).startswith("TYPO") for v in r.values()))
    res["seeded_constant_price"] = {
        "rule": "record-seeded immediate mint with a CONSTANT price of lam bits "
                "(the DP-means shape): mint iff no incumbent codes the record "
                "within lam bits; the record then joins the cheapest incumbent",
        "design": f"k=3, d=8, typo_rate=0.01, n={N}, seed 0; own cell ownership "
                  "ledger, the deferred engine is not run underneath",
        "typo_records_in_stream": typo_recs,
        "grid": lam_rows,
    }

    res["seeded_sweep"]["verdict"] = (
        "zero seeded mints in all 20 cells at the full node price"
        if res["seeded_sweep"]["total_seeded_mints"] == 0
        else f"{res['seeded_sweep']['total_seeded_mints']} seeded mints over 20 cells")

    with open(os.path.join(OUT, "e7_immediate_vs_deferred.json"), "w") as f:
        json.dump(stamped(res), f, indent=2)
    print("written results/e7_immediate_vs_deferred.json")


if __name__ == "__main__":
    main()
