"""E45: is there a benchmark for the decision this paper says is open?

WHY. The paper's one open decision is key identity: whether two differently named keys carry the
same role. E20 measures it and we lose, and E44 then shows the gold E20 uses is the output of a
string function we wrote, so it rewards normalisation and does not record that decision at all. The
obvious next question is not "how do we win it" but "what would we measure a win on". This asks
whether the data holds any key-identity labels that are not a string function.

THE ONLY CANDIDATE IN THIS DATA. AutoPKG ships its own product knowledge graph, and 3,285 of its
15,964 "Attribute Key" nodes carry a non-empty `synonyms` list. Those lists are key merges its
pipeline decided, and some of them are exactly the decision the paper wants: "Stopcock Style" with
"Stopcock Type", "Convection Technology" with "Convection Feature". A string function sees neither.

WHAT MAKES THEM USABLE OR NOT, and this is what the experiment measures:
  1. Coverage. A pair is only evaluable if BOTH of its strings occur as raw spec keys in the stream
     every method reads. Their canonical names are their model's invention, so this is not given.
  2. Independence. The labels are their pipeline's output, so their agent must never be scored on
     them. Ours and the string rules can be, because neither produced them.
  3. Correctness. AutoPKG's own paper concedes that "an incorrect MERGE may collapse distinct
     concepts". Any pair that does so is not a label, it is an error, and the experiment prints every
     evaluable pair so a reader can judge rather than take our word.

WHAT WOULD MAKE THIS A BENCHMARK. Enough evaluable pairs, mostly correct, mostly invisible to a
string function. If instead there are a handful and some are wrong, then no key-identity benchmark
exists on this data, which is worth stating plainly: it is why E20's gold was a string function in
the first place, and it means a version 2 operator has nothing to be scored against until someone
labels keys by hand.
"""
from __future__ import annotations
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

from lazada_c2_rawkey import load_records, norm_key

from escrow.provenance import stamped

NODES = os.path.join(ROOT, "baselines", "autopkg", "data", "lazada_autopkg_kg_nodes.csv")
OUT = os.path.join(ROOT, "results", "e45_is_there_a_key_identity_gold.json")


def synonym_pairs(path):
    """Every (canonical name, synonym) pair the shipped graph declares on an Attribute Key node."""
    pairs, names = [], set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["node_type"] != "Attribute Key":
                continue
            names.add(row["node_name"])
            syn = row["synonyms"]
            if syn in ("[]", "null", "", None):
                continue
            for s in (x.strip() for x in syn.strip("[]").split(",")):
                if s:
                    pairs.append((row["node_name"], s))
    return pairs, names


def main():
    recs, _pids, _stats = load_records()
    freq = collections.Counter()
    for r in recs:
        freq.update(r.keys())
    by_norm = collections.defaultdict(set)
    for k in freq:
        by_norm[norm_key(k)].add(k)

    pairs, names = synonym_pairs(NODES)
    aligned = sum(1 for n in names if norm_key(n) in by_norm)
    evaluable = [(a, b) for a, b in pairs if norm_key(a) in by_norm and norm_key(b) in by_norm]
    string_visible = [(a, b) for a, b in evaluable if norm_key(a) == norm_key(b)]
    residue = [(a, b) for a, b in evaluable if norm_key(a) != norm_key(b)]

    report = {
        "experiment": "E45 is there a key-identity gold in this data that is not a string function",
        "source": "baselines/autopkg/data/lazada_autopkg_kg_nodes.csv, node_type 'Attribute Key'",
        "label_status": ("silver: these merges are AutoPKG's pipeline output, not human labels, so "
                         "their own agent must never be scored on them"),
        "counts": {
            "raw_keys_in_the_stream": len(freq),
            "attribute_key_nodes": len(names),
            "synonym_pairs_declared": len(pairs),
            "canonical_names_that_align_with_a_raw_stream_key": aligned,
            "pairs_with_both_sides_on_a_raw_stream_key": len(evaluable),
            "of_those_already_merged_by_the_string_function": len(string_visible),
            "of_those_a_string_function_cannot_see": len(residue),
        },
        "every_evaluable_pair": [
            {"a": a, "b": b,
             "string_function_merges_them": norm_key(a) == norm_key(b),
             "records_carrying_a": freq[sorted(by_norm[norm_key(a)])[0]],
             "records_carrying_b": freq[sorted(by_norm[norm_key(b)])[0]]}
            for a, b in evaluable],
    }

    n = len(evaluable)
    report["verdict"] = {
        "usable_as_a_benchmark": n >= 50,
        "reading": (
            "%d evaluable pairs is not a benchmark. Their canonical names are their model's own "
            "invention, so only %d of %d align with a key any method actually reads, and the "
            "synonym lists mostly relate one invented name to another. Read the pair list above "
            "before using any of it: some collapse distinct concepts, which AutoPKG's own paper "
            "names as a failure mode of its merge step. So no key-identity gold exists on this "
            "data that is not a string function, which is why E20's gold is one, and a version 2 "
            "operator has nothing to be scored against until keys are labelled by hand."
            % (n, aligned, len(names))),
    }
    print(json.dumps(report["counts"], indent=2))
    print("\nevery evaluable pair, for the reader to judge:")
    for p in report["every_evaluable_pair"]:
        flag = "string" if p["string_function_merges_them"] else "SEMANTIC"
        print(f"  [{flag:8s}] {p['a']!r} ({p['records_carrying_a']} records)"
              f"  ==  {p['b']!r} ({p['records_carrying_b']} records)")
    print("\n" + report["verdict"]["reading"])
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
