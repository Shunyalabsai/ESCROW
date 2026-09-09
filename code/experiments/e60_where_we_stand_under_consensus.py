"""E60: every stream we have, re-scored under the consensus protocol, against the single-order numbers.

WHY. The consensus protocol changes how an answer is produced, so every number the paper reports is
conditional on the old one. E59 established that it is safe at both ends, silent on noise and lossless
on planted structure. This asks what it does in the middle, which is where every real stream lives.

WHAT IS COMPARED, ON THE SAME RECORDS EVERY TIME.

  single order      one arrival order, which is what the paper reports today
  shortest of R     R orders, keep the state with the smallest description. No label is consulted
                    and nothing is calibrated: this is the selector the protocol already uses
  best of R         the best ARI any of the R orders reached. NOT a method, since choosing it needs
                    the labels. It is the ceiling that says how much the selector is leaving behind
  confidence        the fraction of records every order agrees about, which is free and is the
                    output the system has never had

Reporting the oracle ceiling beside the selector is the point of the table. A selector that reaches
the ceiling has solved the search problem for this stream; one that does not has quantified what is
left, and that gap is the honest measure of what consensus is worth.

WHAT WOULD REFUTE IT. Consensus costing accuracy anywhere, or the confidence failing to track
correctness, which is the calibration question T3.3 asks directly.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.protocol import consensus_confidence, run_consensus, run_stream, record_labels
from escrow.provenance import stamped
from e4_baseline_army import _ari

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e60_where_we_stand_under_consensus.json")

R_VALUES = (3, 5, 9)


def stream_wikipedia():
    import llm_graph_formation as L
    ds = L.build_wikipedia()
    return ds["records"], ds["truth"]


def stream_planted8():
    from e4_baseline_army import planted8
    return planted8(n=3000, seed=7)


def stream_twogroup():
    from e4_baseline_army import two_group
    return two_group(seed=0)


# The Wikidata cover benchmark is deliberately absent. Its truth is a set of occupations per record
# and 39 percent of records carry more than one, so it is scored by omega and not by ARI. Forcing a
# single-label metric onto it to get it into this table would be measuring the wrong thing; it gets
# its own consensus run against E37's own metric.

STREAMS = {
    "wikipedia": stream_wikipedia,
    "planted8": stream_planted8,
    "twogroup": stream_twogroup,
}


def main():
    report = {"experiment": "E60 every stream under the consensus protocol",
              "R_values": list(R_VALUES), "streams": {}}

    for name, build in STREAMS.items():
        try:
            recs, truth = build()
        except Exception as e:                                     # noqa: BLE001
            report["streams"][name] = {"skipped": repr(e)[:150]}
            print(f"{name:<16} skipped: {e!r}"[:110])
            continue

        g0, b0 = run_stream(recs)
        single = round(_ari(truth, record_labels(g0, len(recs))), 4)
        row = {"records": len(recs), "single_order_ARI": single, "single_order_K": int(g0.K),
               "by_R": {}}
        print(f"\n{name}  ({len(recs)} records)   single order: ARI {single} at K = {g0.K}")
        print(f"    {'R':>3} {'shortest of R':>15} {'K':>4} {'best of R (oracle)':>20} "
              f"{'decided':>9} {'split':>7}")
        for R in R_VALUES:
            cons = run_consensus(recs, R=R, seed=11)
            aris = [round(_ari(truth, r["labels"]), 4) for r in cons["runs"]]
            best_state = cons["best"]
            chosen = round(_ari(truth, best_state["labels"]), 4)
            conf = consensus_confidence(cons)
            row["by_R"][str(R)] = {
                "shortest_of_R_ARI": chosen, "shortest_of_R_K": best_state["K"],
                "best_of_R_ARI_oracle": max(aris), "worst_of_R_ARI": min(aris),
                "ARI_by_order": aris, "K_by_order": cons["K_by_order"],
                "bits_spread": [round(x, 1) for x in cons["bits_spread"]],
                "confidence": conf}
            print(f"    {R:>3} {chosen:>15.4f} {best_state['K']:>4} {max(aris):>20.4f} "
                  f"{conf['fraction_decided']:>9.3f} {conf['split']:>7}")
        report["streams"][name] = row

    done = {k: v for k, v in report["streams"].items() if "skipped" not in v}
    report["headline"] = {
        "single_order_ARI": {k: v["single_order_ARI"] for k, v in done.items()},
        "shortest_of_9_ARI": {k: v["by_R"]["9"]["shortest_of_R_ARI"] for k, v in done.items()
                              if "9" in v["by_R"]},
        "oracle_best_of_9_ARI": {k: v["by_R"]["9"]["best_of_R_ARI_oracle"] for k, v in done.items()
                                 if "9" in v["by_R"]},
        "fraction_of_records_every_order_agrees_on": {
            k: v["by_R"]["9"]["confidence"]["fraction_decided"] for k, v in done.items()
            if "9" in v["by_R"]},
        "reading": ("the gap between shortest-of-R and best-of-R is what the label-free selector "
                    "leaves on the table, and it is the honest size of the remaining search problem. "
                    "The agreed fraction is free and is what a confidence would be built on"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
