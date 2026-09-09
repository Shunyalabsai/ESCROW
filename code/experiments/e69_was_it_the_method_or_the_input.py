"""E69: the Lazada loss, is it the method or the input it was given?

WHAT LOOKING AT THE DATA FOUND. The source CSV has seven columns. We parse exactly one of them,
`specifications`. The silver product type we are scored against tracks a different column,
`product_name`, which we never see:

    spec keys ['color_family', 'interest', 'model', 'recommended_age']
      silver type : 'Beach Ball'
      product name: 'ANTEDATE Big Inflatable Beach Ball PVC 30cm Blow Up Beach Balls...'

    spec keys ['color', 'model']
      silver type : 'Auto Trailing Spoiler'
      product name: 'YESMILE Auto Supply Black/ White/ Red/ Blue/ Grey ABS Mini Spoiler'

The type name appears verbatim inside the product name for 810 of 5,968 labelled records, and the
derivation is plain in the rest. There is nothing in `color=Black, model=SYS_MCDB3R14` that says
Waist Chain, and no method reading only that column can recover it. That is not a scope condition
and it is not a hard problem. It is a benchmark scored against a field we were never given.

WHAT THIS MEASURES. The same engine, the same protocol, the same records, with one thing changed:
whether the product name is in the input. The name enters the only way v1 can take it, as presence
keys over its tokens, one key per word.

  specifications only     what the paper reports today
  plus the product name   the same records with the field the label is derived from

WHY THIS IS NOT A FREE WIN, AND MUST BE SAID. Tokenising a title is exactly the text facet v1
declares out of scope, and the paper's own limitation says the tokenisation residual is 5.3 to 75.0
bits per field. So the second arm is outside the shipped method. It is a diagnostic, not a result to
quote: it separates "the criterion cannot do this" from "the criterion was never shown the column".

WHAT WOULD REFUTE THE DIAGNOSIS. The second arm staying near zero. Then the input was not the
problem and the loss is the method's, which is worth knowing and would have to be said plainly.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped
from e4_baseline_army import _ari

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e69_was_it_the_method_or_the_input.json")
LENGTHS = [int(x) for x in os.environ.get("ESCROW_E69_LENGTHS", "1000,4000").split(",")]
TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "and", "for", "with", "of", "in", "a", "to", "cm", "mm", "pcs", "pc", "set", "new",
        "100", "1", "2", "3", "x"}


def name_tokens(name, cap=8):
    toks = [t for t in TOKEN.findall((name or "").lower()) if t not in STOP and len(t) > 2]
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= cap:
            break
    return out


def main():
    from lazada_c2_rawkey import CSV_MAIN, load_records, load_silver_types
    recs_all, pids_all, _ = load_records()
    pid2type, type_name = load_silver_types()
    csv.field_size_limit(10 ** 9)
    name_of = {}
    with open(CSV_MAIN, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name_of[row["product_id"]] = row.get("product_name") or ""

    report = {"experiment": "E69 the Lazada loss, method or input",
              "what_we_parse": "the specifications column",
              "what_the_label_tracks": "the product_name column, which the engine never sees",
              "the_second_arm_is_out_of_scope": ("tokenising a title is the text facet v1 declares "
                                                 "out of scope; this is a diagnostic, not a result "
                                                 "to quote"),
              "by_length": []}

    for n in LENGTHS:
        recs, pids = recs_all[:n], pids_all[:n]
        truth = [pid2type.get(p) for p in pids]
        keep = [i for i, t in enumerate(truth) if t is not None]
        if len(keep) < 20:
            continue
        t_keep = [truth[i] for i in keep]
        row = {"records": len(recs), "labelled": len(keep),
               "distinct_types": len(set(t_keep)),
               "records_per_type": round(len(keep) / max(len(set(t_keep)), 1), 2), "arms": {}}

        def score(rs, tag):
            g, _ = run_stream(rs)
            lab = record_labels(g, len(rs))
            lk = [lab[i] for i in keep]
            by = collections.defaultdict(collections.Counter)
            for l, t in zip(lk, t_keep):
                by[l][t] += 1
            purity = sum(c.most_common(1)[0][1] for c in by.values()) / max(len(lk), 1)
            row["arms"][tag] = {"ARI": round(_ari(t_keep, lk), 4), "K": int(g.K),
                                "purity_against_silver": round(purity, 4),
                                "covered": sum(1 for x in lab if x != -1)}
            return row["arms"][tag]

        a = score(recs, "specifications_only")
        withname = []
        for r, p in zip(recs, pids):
            rr = dict(r)
            for t in name_tokens(name_of.get(p, "")):
                rr[f"name:{t}"] = "1"
            withname.append(rr)
        b = score(withname, "plus_product_name_tokens")

        report["by_length"].append(row)
        print(f"\n{len(recs)} records, {row['distinct_types']} silver types, "
              f"{row['records_per_type']} records each")
        print(f"   specifications only        ARI {a['ARI']:>8.4f}  K={a['K']:<5} "
              f"purity {a['purity_against_silver']:.4f}  covered {a['covered']}")
        print(f"   plus the product name      ARI {b['ARI']:>8.4f}  K={b['K']:<5} "
              f"purity {b['purity_against_silver']:.4f}  covered {b['covered']}")
        mult = (b["ARI"] / a["ARI"]) if a["ARI"] > 0 else None
        print(f"   -> {'x%.0f' % mult if mult else 'n/a'} on ARI")

    report["headline"] = {
        "by_length": [{"records": r["records"],
                       "specifications_only": r["arms"]["specifications_only"]["ARI"],
                       "plus_product_name": r["arms"]["plus_product_name_tokens"]["ARI"],
                       "K_before": r["arms"]["specifications_only"]["K"],
                       "K_after": r["arms"]["plus_product_name_tokens"]["K"]}
                      for r in report["by_length"]],
        "reading": ("if the second arm rises sharply, the Lazada number was measuring an input that "
                    "does not contain the answer, not a criterion that cannot find it. That does not "
                    "make the second arm quotable, because tokenising a title is outside v1, but it "
                    "does change what the first number means"),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
