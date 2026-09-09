"""E68: is the Lazada number a real loss, or the wrong question asked of the wrong label?

WHY PROBE RATHER THAN ACCEPT. The paper reports ARI 0.0013 against the silver product type and
explains it with Proposition 1. E67 then showed that sequential k-means, handed the true number of
kinds for free, scores -0.0012 on the same slice, which says the label is unfindable one record at a
time. That is a good defence and it is still only a defence. A number that low deserves to be taken
apart rather than explained.

FOUR THINGS THIS ASKS.

  1. What does the label actually look like. How many types, how big, how many are singletons. A
     label that is nearly one type per record cannot be scored by any partition metric, and if that
     is what it is, ARI was never the right instrument and the paper should say so.

  2. Does the stream length matter. E67 ran 399 records and found K = 2. The full stream is 21,365
     records and finds K = 68. The seeding condition says a group must recur before it can be
     minted, so a short slice starves the engine of exactly the evidence it needs. Sweeping the
     length separates "the method cannot do this" from "the slice was too short to try".

  3. What is actually found. Not the score, the nodes: how many records they cover, how pure they
     are against the silver type, and what their supports name. A graph that is 68 coherent product
     templates is not a failure just because the label wanted 5,518 fine-grained types.

  4. Is there a coarser question with a real answer. The silver types have names, so grouping them by
     their leading word gives a coarser label nobody tuned. If agreement rises sharply against the
     coarse label, the fine one was asking for a resolution the stream does not carry.

WHAT WOULD REFUTE THE PAPER'S READING. Agreement staying near zero at every stream length and every
label granularity, with nodes that are neither pure nor interpretable. Then it is a real loss and the
scope argument is covering for one.
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped
from e4_baseline_army import _ari

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e68_probe_lazada.json")
LENGTHS = [int(x) for x in os.environ.get(
    "ESCROW_E68_LENGTHS", "400,1000,2500,5000,10000").split(",")]


def coarse(t, name_of):
    """A coarser label nobody tuned: the leading word of the silver type's own name."""
    n = name_of.get(t)
    if not n:
        return None
    return n.strip().split()[0].lower() if n.strip() else None


def main():
    from lazada_c2_rawkey import load_records, load_silver_types
    recs_all, pids_all, _ = load_records()
    pid2type, type_name = load_silver_types()

    report = {"experiment": "E68 taking the Lazada number apart",
              "label_shape": {}, "by_length": [], "nodes_at_the_longest": []}

    # ---- 1. what the label is ------------------------------------------------
    types = [pid2type.get(p) for p in pids_all]
    have = [t for t in types if t is not None]
    sizes = collections.Counter(have)
    singles = sum(1 for c in sizes.values() if c == 1)
    report["label_shape"] = {
        "records_with_a_label": len(have), "distinct_types": len(sizes),
        "records_per_type": round(len(have) / max(len(sizes), 1), 2),
        "types_with_exactly_one_record": singles,
        "share_of_types_that_are_singletons": round(singles / max(len(sizes), 1), 4),
        "records_inside_singleton_types": singles,
        "largest_types": [(type_name.get(t, t), c) for t, c in sizes.most_common(8)]}
    print("1. WHAT THE LABEL IS\n")
    for k, v in report["label_shape"].items():
        if k != "largest_types":
            print(f"   {k}: {v}")
    print(f"   largest types: {report['label_shape']['largest_types'][:5]}")

    # ---- 2 and 3. length sweep ----------------------------------------------
    print("\n2. DOES THE STREAM LENGTH MATTER\n")
    print(f"   {'records':>8} {'K':>4} {'covered':>8} {'ARI fine':>9} {'ARI coarse':>11} "
          f"{'purity':>8} {'types/record':>13}")
    for n in LENGTHS:
        recs, pids = recs_all[:n], pids_all[:n]
        truth = [pid2type.get(p) for p in pids]
        keep = [i for i, t in enumerate(truth) if t is not None]
        if len(keep) < 20:
            continue
        g, _ = run_stream(recs)
        lab = record_labels(g, len(recs))
        t_fine = [truth[i] for i in keep]
        l_keep = [lab[i] for i in keep]
        t_coarse = [coarse(truth[i], type_name) for i in keep]
        ok = [i for i, t in enumerate(t_coarse) if t is not None]
        ari_fine = round(_ari(t_fine, l_keep), 4)
        ari_coarse = round(_ari([t_coarse[i] for i in ok], [l_keep[i] for i in ok]), 4) if ok else None
        by = collections.defaultdict(collections.Counter)
        for l, t in zip(l_keep, t_fine):
            by[l][t] += 1
        maj = sum(c.most_common(1)[0][1] for c in by.values())
        purity = round(maj / max(len(l_keep), 1), 4)
        covered = sum(1 for x in lab if x != -1)
        row = {"records": len(recs), "K": int(g.K), "covered_records": covered,
               "ARI_fine": ari_fine, "ARI_coarse_leading_word": ari_coarse,
               "purity_against_fine": purity, "labelled_records": len(keep),
               "distinct_fine_types": len(set(t_fine)),
               "records_per_fine_type": round(len(keep) / max(len(set(t_fine)), 1), 2)}
        report["by_length"].append(row)
        print(f"   {len(recs):>8} {g.K:>4} {covered:>8} {ari_fine:>9.4f} "
              f"{(ari_coarse if ari_coarse is not None else float('nan')):>11.4f} {purity:>8.4f} "
              f"{row['records_per_fine_type']:>13}")

        if n == LENGTHS[-1]:
            print(f"\n3. WHAT IS ACTUALLY FOUND at {len(recs)} records\n")
            for l, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
                tot = sum(c.values())
                top, topc = c.most_common(1)[0]
                sup = sorted(g.key_name[k] for k in g.nodes[l].S)[:6] if l in g.nodes else []
                report["nodes_at_the_longest"].append(
                    {"node": int(l), "members": tot,
                     "majority_type": type_name.get(top, str(top)),
                     "majority_share": round(topc / tot, 3), "distinct_types_inside": len(c),
                     "support": sup})
                print(f"   node {l:>4}: {tot:>5} records, {len(c):>4} silver types inside, "
                      f"majority {type_name.get(top, top)!r} at {100*topc/tot:.0f}%")
                print(f"              support: {sup}")

    report["headline"] = {
        "label_is_nearly_unique_per_record": report["label_shape"]["records_per_type"],
        "share_of_types_that_are_singletons":
            report["label_shape"]["share_of_types_that_are_singletons"],
        "K_grows_with_length": [(r["records"], r["K"]) for r in report["by_length"]],
        "ARI_fine_by_length": [(r["records"], r["ARI_fine"]) for r in report["by_length"]],
        "ARI_coarse_by_length": [(r["records"], r["ARI_coarse_leading_word"])
                                 for r in report["by_length"]],
        "purity_by_length": [(r["records"], r["purity_against_fine"]) for r in report["by_length"]],
        "reading": ("if the label is nearly one type per record then no partition metric can score "
                    "it and ARI was the wrong instrument, not the method the wrong method. Purity "
                    "and the coarse label say whether the nodes are coherent even when the fine "
                    "label cannot be matched"),
    }
    print("\n" + json.dumps(report["headline"], indent=2)[:1400])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
