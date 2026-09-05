"""LAZADA RAW-KEY STREAM: the real-world C2 preview (structure exhibit, no leaderboard).

One record per spec-bearing listing; keys are the RAW spec keys exactly as published
(case variants like color/Color kept distinct). Stream order: shuffle seed 0.
Repair every 200 records and once at the end. Ground-truth (of_type silver labels)
is NEVER fed to the engine; it is used only to score ARI afterwards.

Deliverables:
  (1) structure exhibit: K, node sizes, top nodes' supports and top values;
  (2) case-variant receipt survey: for every raw-key pair that is a pure
      case/spacing variant (normalise lowercase+strip non-alnum), how often the two
      keys are supported by the SAME latent node vs different ones, plus three
      concrete mint receipts with both variants inside one support;
  (3) product-type sanity: ARI of our record clustering vs the KG's of_type
      silver product-type clustering (labelled silver).

Run from the code directory:
      CUDA_VISIBLE_DEVICES=2 python experiments/lazada_c2_rawkey.py [--limit N]
"""
import argparse
import collections
import csv
import itertools
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.environ.get("ESCROW_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# data, baselines and results live under ESCROW_ROOT (default: the directory above this code directory)

from sklearn.metrics import adjusted_rand_score

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective
from escrow.fastrepair import FastBatchObjective

DATA_DIR = os.path.join(ROOT, "baselines/autopkg/data")
CSV_MAIN = os.path.join(DATA_DIR, "lazada_autopkg_product_data_url.csv")
CSV_NODES = os.path.join(DATA_DIR, "lazada_autopkg_kg_nodes.csv")
CSV_EDGES = os.path.join(DATA_DIR, "lazada_autopkg_kg_edges.csv")
OUT_DIR = os.path.join(ROOT, "results")

DECLARED_CHOICES = [
    "spec parsing: bracket-aware scan of the Java-map string '{k1=[v1, v2], ...}': "
    "key = text between the previous ']' (or the opening '{') and the next '=[', "
    "stripped of leading ',' and whitespace; value list = text to the first following "
    "']'; pairs with an empty key or an unclosed bracket are skipped",
    "multi-valued lists: split on ', ' (comma-space, the Java List.toString separator), "
    "strip each item, drop empties, sort lexicographically, join with ', ' -> one "
    "categorical value string (declared)",
    "records: one per spec-bearing listing (specifications field present and not the "
    "literal 'null'); listings whose spec string parses to zero pairs are dropped and "
    "counted",
    "keys: RAW spec keys exactly as published; no case folding, no trimming beyond the "
    "parser's whitespace strip; ground-truth columns (KG) never enter the record",
    "stream order: random.Random(0).shuffle over the parsed record list (seed 0); the "
    "dry-run slice is a prefix of this same stream",
    "repair cadence: BatchObjective.repair() every 200 records and once at the end; "
    "guard: if K exceeds 300 right after a record, repair immediately (contract's "
    "K-explosion clause)",
    "record label for ARI: each record is assigned to the live node containing it with "
    "the most members (tie -> lowest node id); records in no node form one background "
    "cluster labelled -1 (honest, same convention as e4)",
    "silver product types: of_type edges (source=Product node, target='Product Type' "
    "node); every product has at most one type in this KG; ARI computed over streamed "
    "records whose product_id has a silver type, labelled silver",
    "case-variant normalisation: lowercase + strip every non-alphanumeric character; "
    "families are normalised forms with >=2 distinct raw keys; the survey counts all "
    "unordered raw-key pairs inside families",
    "no embedding baselines: this benchmark is a structure exhibit per protocol",
]


# ------------------------------------------------------------------ parsing --
def parse_specs(s: str) -> dict:
    """Bracket-aware parse of '{k1=[v1, v2], k2=[..]}' -> {raw key: joined value}."""
    out = {}
    i = 1 if s.startswith("{") else 0
    while True:
        j = s.find("=[", i)
        if j == -1:
            break
        key = s[i:j].strip()
        if key.startswith(","):
            key = key[1:].strip()
        k = s.find("]", j + 2)
        if k == -1:
            break
        raw_vals = s[j + 2:k]
        items = sorted(v.strip() for v in raw_vals.split(", ") if v.strip())
        if key:
            out[key] = ", ".join(items)
        i = k + 1
    return out


def load_records():
    csv.field_size_limit(10 ** 9)
    rows_total = spec_rows = dropped_empty = 0
    recs, pids = [], []
    with open(CSV_MAIN, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_total += 1
            s = row["specifications"]
            if not s or s == "null":
                continue
            spec_rows += 1
            rec = parse_specs(s)
            if not rec:
                dropped_empty += 1
                continue
            recs.append(rec)
            pids.append(row["product_id"])
    order = list(range(len(recs)))
    random.Random(0).shuffle(order)
    recs = [recs[i] for i in order]
    pids = [pids[i] for i in order]
    return recs, pids, dict(rows_total=rows_total, spec_rows=spec_rows,
                            dropped_empty_parse=dropped_empty,
                            parsed_records=len(recs))


def load_silver_types():
    csv.field_size_limit(10 ** 9)
    prod_name, type_name = {}, {}
    with open(CSV_NODES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["node_type"] == "Product":
                prod_name[row["node_id"]] = row["node_name"]
            elif row["node_type"] == "Product Type":
                type_name[row["node_id"]] = row["node_name"]
    pid2type = {}
    with open(CSV_EDGES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["edge_type"] != "of_type":
                continue
            pid = prod_name.get(row["source_node_id"])
            if pid is None:
                continue
            t = row["target_node_id"]
            # at most one type per product in this KG; keep smallest id if repeated
            if pid not in pid2type or t < pid2type[pid]:
                pid2type[pid] = t
    return pid2type, type_name


# ------------------------------------------------------------------ survey --
NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_key(k: str) -> str:
    return NORM_RE.sub("", k.lower())


def case_variant_survey(g: EscrowGraph):
    families = collections.defaultdict(list)
    for name in g.keys:
        families[norm_key(name)].append(name)
    families = {nk: sorted(v) for nk, v in families.items() if len(v) >= 2 and nk}

    # live support map: kid -> set of live node ids supporting it
    supp = collections.defaultdict(set)
    for v in g.nodes.values():
        for kid in v.S:
            supp[kid].add(v.nid)

    buckets = dict(same_node=0, both_supported_no_shared_node=0,
                   one_supported=0, neither_supported=0)
    pair_rows = []
    for nk, names in sorted(families.items()):
        for a, b in itertools.combinations(names, 2):
            ka, kb = g.keys[a].kid, g.keys[b].kid
            na, nb = supp.get(ka, set()), supp.get(kb, set())
            shared = na & nb
            if shared:
                bucket = "same_node"
            elif na and nb:
                bucket = "both_supported_no_shared_node"
            elif na or nb:
                bucket = "one_supported"
            else:
                bucket = "neither_supported"
            buckets[bucket] += 1
            pair_rows.append(dict(
                family=nk, key_a=a, key_b=b,
                P_a=g.keys[a].P, P_b=g.keys[b].P,
                nodes_a=len(na), nodes_b=len(nb),
                shared_nodes=sorted(shared)[:5], bucket=bucket))

    both = buckets["same_node"] + buckets["both_supported_no_shared_node"]
    survey = dict(
        variant_families=len(families),
        variant_pairs=sum(buckets.values()),
        buckets=buckets,
        co_support_rate_given_both_supported=(
            round(buckets["same_node"] / both, 4) if both else None),
        families_sample={nk: names for nk, names in
                         sorted(families.items(),
                                key=lambda kv: -max(g.keys[x].P for x in kv[1]))[:15]},
    )
    # the strongest concrete same-node pairs, by evidence mass
    same = [r for r in pair_rows if r["bucket"] == "same_node"]
    same.sort(key=lambda r: -(min(r["P_a"], r["P_b"])))
    survey["same_node_examples"] = same[:10]
    if len(pair_rows) <= 400:
        survey["pairs_listed"] = pair_rows
    else:
        survey["pairs_listed"] = same[:200]
        survey["pairs_listed_note"] = ("pair list truncated to the top-200 same_node "
                                       "pairs; bucket counts above cover ALL pairs")

    # three mint receipts whose support holds both variants of one pair
    receipts = []
    for entry in g.mint_log:
        seen = collections.defaultdict(list)
        for name in entry["support"]:
            seen[norm_key(name)].append(name)
        hits = {nk: v for nk, v in seen.items() if len(v) >= 2 and nk}
        if hits:
            receipts.append(dict(variant_hits=hits, receipt=entry))
        if len(receipts) >= 3:
            break
    survey["mint_receipts_with_both_variants"] = receipts
    return survey


# ------------------------------------------------------------------ exhibit --
def structure_exhibit(g: EscrowGraph, top_n=10, top_keys=8, top_vals=3):
    sizes = sorted((v.t for v in g.nodes.values()), reverse=True)
    inv_cache = {}

    def val_name(kid, vid):
        if kid not in inv_cache:
            inv_cache[kid] = {v: k for k, v in g.key_by_id[kid].inventory.items()}
        return inv_cache[kid].get(vid, "?")

    top = []
    for v in sorted(g.nodes.values(), key=lambda x: (-x.t, x.nid))[:top_n]:
        keys = sorted(v.S, key=lambda kid: -v.p.get(kid, 0))[:top_keys]
        kd = {}
        for kid in keys:
            blk = v.blocks.get(kid)
            vals = []
            if blk is not None:
                vals = [(val_name(kid, vid)[:60], c) for vid, c in
                        sorted(blk.counts.items(), key=lambda kv: -kv[1])[:top_vals]]
            kd[g.key_name[kid]] = dict(p=v.p.get(kid, 0), top_values=vals)
        top.append(dict(node=v.nid, members=v.t, support_size=len(v.S),
                        top_support=kd))
    hist = collections.Counter()
    for s in sizes:
        b = ("1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20 else
             "21-100" if s <= 100 else "101-1000" if s <= 1000 else ">1000")
        hist[b] += 1
    return dict(K=g.K, node_sizes_top20=sizes[:20], size_histogram=dict(hist),
                covered_records=len(set().union(*(v.members for v in g.nodes.values()))
                                     ) if g.nodes else 0,
                top_nodes=top)


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="dry-run record cap")
    ap.add_argument("--repair-every", type=int, default=200)
    ap.add_argument("--out", default="lazada_c2_rawkey.json")
    args = ap.parse_args()

    t0 = time.time()
    recs, pids, data_stats = load_records()
    if args.limit:
        recs, pids = recs[:args.limit], pids[:args.limit]
    print(f"[data] {data_stats} streaming {len(recs)} records", flush=True)
    all_keys = set()
    for r in recs:
        all_keys.update(r)
    print(f"[data] unique raw keys in stream: {len(all_keys)}", flush=True)

    g = EscrowGraph()
    b = FastBatchObjective(g)
    deviations = [
        "repair uses FastBatchObjective (escrow/fastrepair.py), a "
        "PERFORMANCE-ONLY override of reassign_pass: identical candidate set, "
        "iteration order, and acceptance rule; the L_vblock terms are the exact "
        "analytic O(1) difference of the same closed form the stock pass "
        "re-sums per call. Verified: delta self-test worst error 1.0e-13; "
        "stock-vs-fast A/B on the true 2,600-record stream prefix gives "
        "identical K (15), identical mints (45), identical node member sets, "
        "and L_batch total difference 0.0 (stock 49.7s, fast 10.5s). Motivated "
        "by a cProfile showing 86% of wall time inside the stock re-summation "
        "(196M lg2 calls at n=2,600); without it the declared every-200 "
        "cadence does not finish overnight on 21,365 records.",
        "engine.py patched before this run (recorded, minimal): the release step "
        "now guard-deletes a minting candidate from the pool "
        "(if pool.get(sig) is c) instead of an unconditional del; the unguarded "
        "del raised KeyError at n~3.6k on this stream when the 4096-cap pool "
        "capacity-evicted a candidate that later minted within the same record. "
        "No same-sig candidate can be recreated before release within a record, "
        "so the guard is exact; mint accounting is unchanged."
    ]
    t_proc = t_rep = 0.0
    repair_calls = guard_repairs = 0
    t_stream0 = time.time()
    for i, r in enumerate(recs):
        ts = time.time()
        g.process(r)
        t_proc += time.time() - ts
        due = (i + 1) % args.repair_every == 0
        guard = g.K > 300
        if due or guard:
            ts = time.time()
            b.repair()
            t_rep += time.time() - ts
            repair_calls += 1
            guard_repairs += guard and not due
        if (i + 1) % 500 == 0:
            el = time.time() - t_stream0
            print(f"[stream] n={i+1} K={g.K} pool={len(g.pool)} "
                  f"elapsed={el:.1f}s rate={(i+1)/el:.1f} rec/s "
                  f"(proc {t_proc:.1f}s repair {t_rep:.1f}s)", flush=True)
    ts = time.time()
    b.repair()
    t_rep += time.time() - ts
    repair_calls += 1
    t_stream = time.time() - t_stream0
    if guard_repairs:
        deviations.append(f"K-explosion guard fired {guard_repairs} extra repair(s) "
                          f"beyond the every-{args.repair_every} cadence (declared guard)")

    print(f"[done] K={g.K} mints={len(g.mint_log)} stream={t_stream:.1f}s "
          f"rate={len(recs)/t_stream:.2f} rec/s", flush=True)

    exhibit = structure_exhibit(g)
    survey = case_variant_survey(g)

    # ---- product-type sanity (silver; REPORTING ONLY, never fed to the engine) ----
    pid2type, type_name = load_silver_types()
    # record label: node with most members containing it, tie -> lowest nid
    lab = [-1] * len(recs)
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid          # larger t iterated later -> wins
    silver, ours = [], []
    for i, pid in enumerate(pids):
        t = pid2type.get(pid)
        if t is not None:
            silver.append(t)
            ours.append(lab[i])
    ari = adjusted_rand_score(silver, ours) if silver else None
    sanity = dict(
        label="silver (KG of_type; the KG itself is model-derived)",
        n_scored=len(silver),
        n_records_streamed=len(recs),
        silver_type_count=len(set(silver)),
        our_cluster_count_on_scored=len(set(ours)),
        background_records_on_scored=sum(1 for x in ours if x == -1),
        ari_vs_silver=round(ari, 4) if ari is not None else None)

    # reporting-only interpretability: dominant silver type per top node
    for tn in exhibit["top_nodes"]:
        v = g.nodes[tn["node"]]
        c = collections.Counter(type_name.get(pid2type.get(pids[m - 1]), "?")
                                for m in v.members if pids[m - 1] in pid2type)
        tn["silver_top_types_reporting_only"] = c.most_common(3)

    out = dict(
        benchmark="LAZADA RAW-KEY STREAM (C2 preview, structure exhibit)",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        limit=args.limit or None,
        declared_choices=DECLARED_CHOICES,
        deviations=deviations,
        data=dict(**data_stats, unique_raw_keys_in_stream=len(all_keys)),
        runtime=dict(stream_wall_s=round(t_stream, 1),
                     process_s=round(t_proc, 1), repair_s=round(t_rep, 1),
                     rec_per_s=round(len(recs) / t_stream, 2),
                     repair_calls=repair_calls,
                     repair_every=args.repair_every,
                     total_wall_s=round(time.time() - t0, 1)),
        structure=exhibit,
        case_variant_survey=survey,
        product_type_sanity=sanity,
        baselines=dict(skipped="no embedding baselines by protocol: this is a "
                               "structure exhibit, not a leaderboard"),
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
