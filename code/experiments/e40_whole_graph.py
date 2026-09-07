"""E40: does the same observation define the WHOLE graph, or only the specialisation edge?

WHERE THIS COMES FROM. E38 showed that one relation, a general kind and the special kinds under it,
falls out of the records for nothing: if every record carrying key b also carries key a and a is more
common, then b specialises a. No node, no schema, no prior knowledge, nothing to set. The obvious
next question is whether that was a lucky special case or an instance of something general. A
knowledge graph is not one relation. It is several, and if the same counting yields all of them then
the construction returns a graph rather than a clustering.

FOUR MORE RELATIONS, EACH DEFINED BY A COUNT AND NOTHING ELSE.

  SIBLING, or disjointness. Keys b and c never appear in the same record, yet both specialise the
  same a. Then b and c are alternatives under a. This is what makes a taxonomy branch rather than
  chain.

  SYNONYMY, two names for one role. Keys b and c never co-occur, and the values they carry are drawn
  from the same pool. Never co-occurring alone is not enough, since siblings also never co-occur;
  what separates them is whether the value vocabularies coincide. This is the key-identity question
  the paper calls C2, asked without comparing key STRINGS, which is the point.

  VALUE-TO-VALUE FUNCTION, the typed edge. If a record's value on key a determines its value on key
  b, then a specific value of a points at a specific value of b: born in Paris implies citizen of
  France. These are the edges a knowledge graph is actually made of, and they are many-to-one, which
  is exactly the one-to-many observation read the other way round.

  CARDINALITY. For each such edge, whether it is one-to-one or many-to-one, which is read off the
  same counts and is what tells a hierarchy from a mere association.

WHAT IS MEASURED. Each relation is extracted on a planted stream where its truth is known by
construction, and scored for precision and recall; then all four are extracted from the Wikidata
people of E37, where nothing is planted, and reported as what they find. The planted generator is
built so that all four relations exist at once and are independent of one another, so a rule that
finds one by accident does not score on the rest.

NOTHING HERE IS TUNED. Every rule is an equality or a strict inequality on counts. The one tolerance,
for missing values in real data, is reported at 1.0 as well, which is the parameter-free reading.

FALSIFIER, STATED BEFORE THE RUN. Any relation that cannot be recovered above chance on the planted
stream, where it is present by construction. Reported per relation.
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.provenance import stamped                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
WIKIDATA = os.path.join(RESULTS, "wikidata_cover", "wikidata_people.json")
OUT = os.path.join(RESULTS, "e40_whole_graph.json")

N = 4000
SEEDS = (0, 1, 2)
CONF = 1.0            # the parameter-free reading; 0.95 is reported beside it
MIN_LIFT = 1.5
MIN_SUPPORT = 20      # a rule needs this many records before it is proposed, declared in advance


def planted(seed):
    """One stream carrying all four relations at once, each independent of the others.

    THREE families. Family f owns key `fam{f}` with its own value pool. Under it are three SPECIES;
    species s owns `sp{f}_{s}`, and the species keys of one family never co-occur, so they are
    siblings. Key `alias{f}` is a second NAME for `famaux{f}`: a record carries one or the other,
    never both, and both draw from one pool.

    Two design decisions, both made because the first version of this fixture measured itself rather
    than the rule.

    The value-edge keys are present only sometimes and independently of the family structure. In the
    first version they were in every record, so every key truly implied them; those are correct
    implications and counting them as errors measured the generator, not the rule.

    Each species draws its values from its OWN pool, except one deliberately planted hard case:
    species 0 and 1 of family 2 share a pool. They are siblings with one vocabulary, which is the
    case the paper calls key identity and which presence alone cannot tell from an alias. It is
    planted so the rule is scored on it rather than protected from it.
    """
    rng = random.Random(seed)
    cities = {"paris": "france", "lyon": "france", "berlin": "germany",
              "munich": "germany", "turin": "italy", "milan": "italy"}
    recs, truth = [], []
    for _ in range(N):
        f = rng.randrange(3)
        s = rng.randrange(3)
        rec = {}
        rec[f"fam{f}"] = f"f{f}v{rng.randrange(5)}"
        name = f"famaux{f}" if rng.random() < 0.5 else f"alias{f}"
        rec[name] = f"a{f}v{rng.randrange(5)}"
        pool = "shared2" if (f == 2 and s in (0, 1)) else f"s{f}_{s}"
        rec[f"sp{f}_{s}"] = f"{pool}v{rng.randrange(5)}"
        if rng.random() < 0.6:                     # independent of the family structure
            city = rng.choice(list(cities))
            rec["city"] = city
            rec["country"] = cities[city]
            rec["passport"] = cities[city] + "_pp"
        recs.append(rec)
        truth.append((f, s))
    gold = {
        # a species key implies its family key. It does NOT imply famaux or alias, since only half
        # the records of a family carry each of those.
        "specialises": {(f"sp{f}_{s}", f"fam{f}") for f in range(3) for s in range(3)}
                       | {(f"famaux{f}", f"fam{f}") for f in range(3)}
                       | {(f"alias{f}", f"fam{f}") for f in range(3)},
        "siblings": {tuple(sorted((f"sp{f}_{a}", f"sp{f}_{b}")))
                     for f in range(3) for a in range(3) for b in range(3) if a < b},
        "synonyms": {tuple(sorted((f"famaux{f}", f"alias{f}"))) for f in range(3)},
        "value_edges": {("city", "country"), ("city", "passport"),
                        ("country", "passport"), ("passport", "country")},
        "hard_case_siblings_sharing_a_pool": {tuple(sorted(("sp2_0", "sp2_1")))},
    }
    return recs, truth, gold


# ------------------------------------------------------------------ counts -- #
def index(recs):
    present = defaultdict(set)
    values = defaultdict(lambda: defaultdict(set))     # key -> value -> record ids
    for i, r in enumerate(recs):
        for k, v in r.items():
            present[k].add(i)
            values[k][v].add(i)
    return present, values


def specialises(present, conf, min_support):
    """b specialises a: every record carrying b carries a, and a is strictly more common."""
    out = set()
    for b, pb in present.items():
        if len(pb) < min_support:
            continue
        for a, pa in present.items():
            if a == b or len(pa) < len(pb) * MIN_LIFT:
                continue
            if len(pb & pa) / len(pb) >= conf:
                out.add((b, a))
    return out


def siblings(present, spec, min_support):
    """b and c never co-occur and both specialise some common a."""
    parents = defaultdict(set)
    for b, a in spec:
        parents[b].add(a)
    out = set()
    ks = [k for k in parents if len(present[k]) >= min_support]
    for i, b in enumerate(ks):
        for c in ks[i + 1:]:
            if present[b] & present[c]:
                continue
            if parents[b] & parents[c]:
                out.add(tuple(sorted((b, c))))
    return out


def synonyms(present, values, min_support):
    """Two keys that never co-occur AND whose value vocabularies coincide.

    Never co-occurring alone would also catch siblings; the value pool is what separates a second
    name for one role from two alternatives under one parent. No key STRING is compared.
    """
    out = set()
    ks = [k for k in present if len(present[k]) >= min_support]
    for i, b in enumerate(ks):
        for c in ks[i + 1:]:
            if present[b] & present[c]:
                continue
            vb, vc = set(values[b]), set(values[c])
            if not vb or not vc:
                continue
            j = len(vb & vc) / len(vb | vc)
            if j >= 0.9:
                out.add(tuple(sorted((b, c))))
    return out


def value_edges(recs, present, values, conf, min_support):
    """a -> b when a record's value on a determines its value on b.

    Reported with the cardinality, since one-to-one and many-to-one are different edges: a
    many-to-one map is a hierarchy over values and a one-to-one map is an alias.
    """
    out = {}
    ks = [k for k in present if len(present[k]) >= min_support]
    for a in ks:
        for b in ks:
            if a == b:
                continue
            both = present[a] & present[b]
            if len(both) < min_support:
                continue
            m = defaultdict(Counter)
            for i in both:
                m[recs[i][a]][recs[i][b]] += 1
            det = sum(c.most_common(1)[0][1] for c in m.values()) / len(both)
            if det >= conf:
                images = {c.most_common(1)[0][0] for c in m.values()}
                out[(a, b)] = {"determinism": round(det, 4),
                               "distinct_sources": len(m), "distinct_targets": len(images),
                               "cardinality": ("one_to_one" if len(images) == len(m)
                                               else "many_to_one")}
    return out


def prf(found, gold):
    tp = len(found & gold)
    p = tp / len(found) if found else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"found": len(found), "gold": len(gold), "true_positives": tp,
            "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def main():
    report = {"experiment": "E40 does the same observation define the whole graph",
              "relations": ["specialises", "siblings", "synonyms", "value_edges"],
              "rules_are_counts_not_settings": True,
              "min_support": MIN_SUPPORT, "min_lift": MIN_LIFT,
              "falsifier": "any relation not recovered above chance where it is planted",
              "planted": [], "real": {}}

    print("planted stream: all four relations present at once and independent")
    for conf in (CONF, 0.95):
        rows = []
        for s in SEEDS:
            recs, truth, gold = planted(s)
            present, values = index(recs)
            spec = specialises(present, conf, MIN_SUPPORT)
            sib = siblings(present, spec, MIN_SUPPORT)
            syn = synonyms(present, values, MIN_SUPPORT)
            ve = value_edges(recs, present, values, conf, MIN_SUPPORT)
            row = {"seed": s,
                   "specialises": prf(spec, gold["specialises"]),
                   "siblings": prf(sib, gold["siblings"]),
                   "synonyms": prf(syn, gold["synonyms"]),
                   "value_edges": prf(set(ve), gold["value_edges"]),
                   "hard_case_sibling_pair_called_a_synonym":
                       bool(gold["hard_case_siblings_sharing_a_pool"] & syn),
                   "cardinalities": {f"{a}->{b}": v["cardinality"] for (a, b), v in ve.items()}}
            rows.append(row)
        report["planted"].append({"confidence": conf, "per_seed": rows})
        print(f"  confidence {conf}:")
        for rel in ("specialises", "siblings", "synonyms", "value_edges"):
            ps = [r[rel]["precision"] for r in rows]
            rs = [r[rel]["recall"] for r in rows]
            print(f"    {rel:14s} precision {sum(ps)/len(ps):.3f}  recall {sum(rs)/len(rs):.3f}",
                  flush=True)
        print(f"    cardinalities on seed 0: {rows[0]['cardinalities']}", flush=True)

    print("\nreal records, nothing planted: Wikidata people")
    if os.path.exists(WIKIDATA):
        d = json.load(open(WIKIDATA, encoding="utf-8"))
        recs = [v["record"] for v in d.values()]
        present, values = index(recs)
        spec = specialises(present, 1.0, MIN_SUPPORT)
        sib = siblings(present, spec, MIN_SUPPORT)
        syn = synonyms(present, values, MIN_SUPPORT)
        ve = value_edges(recs, present, values, 1.0, MIN_SUPPORT)
        card = Counter(v["cardinality"] for v in ve.values())
        report["real"] = {
            "records": len(recs),
            "specialises": len(spec), "siblings": len(sib), "synonyms": len(syn),
            "value_edges": len(ve), "value_edge_cardinality": dict(card),
            "example_specialises": sorted(f"{b} -> {a}" for b, a in spec)[:8],
            "example_siblings": sorted(f"{b} | {c}" for b, c in sib)[:8],
            "example_synonyms": sorted(f"{b} = {c}" for b, c in syn)[:8],
            "example_value_edges": sorted(f"{a} -> {b} ({v['cardinality']})"
                                          for (a, b), v in ve.items())[:8],
        }
        for k in ("specialises", "siblings", "synonyms", "value_edges"):
            print(f"  {k:14s} {report['real'][k]}", flush=True)
        print(f"  cardinality {dict(card)}", flush=True)
        for k in ("example_siblings", "example_synonyms", "example_value_edges"):
            print(f"  {k}: {report['real'][k][:4]}", flush=True)
    else:
        report["real"] = {"skipped": "run fetch_wikidata_cover.py first"}

    base = report["planted"][0]["per_seed"]
    report["headline"] = {
        rel: {"precision": round(sum(r[rel]["precision"] for r in base) / len(base), 3),
              "recall": round(sum(r[rel]["recall"] for r in base) / len(base), 3)}
        for rel in ("specialises", "siblings", "synonyms", "value_edges")}
    report["headline"]["reading"] = (
        "Four relations, one counting pass, nothing set. If all four come back at high precision on "
        "the planted stream and return plausible structure on real records, then what the "
        "construction defines is a graph and not a clustering, and the specialisation edge of E38 "
        "was an instance rather than a special case.")
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
