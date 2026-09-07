"""Fetch a REAL cover benchmark: people, described by their properties, labelled by occupation.

WHY. Every multi-membership number this project has is from a synthetic fixture, and E33 showed that
fixture hands the answer to any cover-capable method by writing the group into the key names. A real
benchmark cannot be rigged that way. Wikidata gives one: a person is one record, the categorical
facets are that person's other properties, and the label set is their occupations (P106), which is
genuinely multi-valued for most people.

The label is REMOVED from the record. P106 never appears as a facet, and neither does P21 or any
property whose value set is the occupation vocabulary, so nothing in the input names the answer.

Licence: Wikidata is CC0. The API is public and the user agent identifies the run.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "wikidata_cover"))

UA = {"User-Agent": "escrow-research-benchmark/0.1 (structure-learning research)"}
SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"

# The label set. Occupations common enough that a group can hold many records, which is what the
# price needs; picking them by frequency rather than by hand keeps the choice out of our hands.
N_OCCUPATIONS = 12
N_PEOPLE = 6000

# Properties that must never enter a record, because they are the label or trivially name it.
BANNED = {"P106",           # occupation, which IS the label
          "P101",           # field of work
          "P39",            # position held
          "P803",           # professorship
          "P1416",          # affiliation
          }


def sparql(query):
    q = urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(f"{SPARQL}?{q}",
                                 headers={**UA, "Accept": "application/sparql-results+json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"  sparql retry {attempt + 1}: {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as exc:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


SAMPLE_ROWS = 20000          # one scan; the aggregate over all humans times the endpoint out


def sample_people():
    """One scan of (person, occupation) pairs, cached to disk.

    The occupation list is then taken by frequency WITHIN this sample, so which groups the benchmark
    contains is decided by the data rather than by us. The aggregate query over all of Wikidata times
    the endpoint out, and the query service rate-limits hard during its outages, so the raw response
    of the one call made to it is kept beside the built benchmark and reused.
    """
    cache = os.path.join(OUT, "scan_20000.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            rows = json.load(fh)["results"]["bindings"]
        print(f"scan: {len(rows)} rows from cache", flush=True)
    else:
        rows = sparql(f"""
          SELECT ?p ?occ WHERE {{ ?p wdt:P31 wd:Q5 ; wdt:P106 ?occ . }} LIMIT {SAMPLE_ROWS}
        """)["results"]["bindings"]
        os.makedirs(OUT, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump({"results": {"bindings": rows}}, fh)
    people = {}
    for r in rows:
        qid = r["p"]["value"].rsplit("/", 1)[-1]
        occ = r["occ"]["value"].rsplit("/", 1)[-1]
        people.setdefault(qid, set()).add(occ)
    return people


def claims_for(qids, keep):
    """The record for each person: every property they carry, as property=value strings.

    A value is the target entity id for an entity-valued claim, or the literal for a time or
    quantity, coarsened to the year for a time so that a date is a facet rather than a unique
    string. Occupations are pulled out separately as the LABEL and removed from the record.
    """
    out = {}
    for i in range(0, len(qids), 40):
        batch = qids[i:i + 40]
        d = api({"action": "wbgetentities", "ids": "|".join(batch), "props": "claims"})
        for qid, ent in (d.get("entities") or {}).items():
            rec, labels = {}, set()
            for pid, claims in (ent.get("claims") or {}).items():
                for c in claims:
                    snak = c.get("mainsnak", {})
                    dv = snak.get("datavalue")
                    if not dv:
                        continue
                    v = dv.get("value")
                    if dv.get("type") == "wikibase-entityid":
                        val = v.get("id")
                    elif dv.get("type") == "time":
                        val = str(v.get("time", ""))[1:5]           # the year
                    elif dv.get("type") == "quantity":
                        continue
                    elif dv.get("type") == "string":
                        continue                                    # identifiers, never a facet
                    else:
                        continue
                    if not val:
                        continue
                    if pid == "P106":
                        if val in keep:
                            labels.add(val)
                        continue
                    if pid in BANNED:
                        continue
                    rec.setdefault(pid, val)                        # first value per property
            if rec and labels:
                out[qid] = {"record": rec, "labels": sorted(labels)}
        time.sleep(0.4)
        print(f"  claims {min(i + 40, len(qids))}/{len(qids)}", flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "wikidata_people.json")
    if os.path.exists(path):
        print(f"cached: {len(json.load(open(path)))} records")
        return
    people = sample_people()
    counts = {}
    for occs in people.values():
        for o in occs:
            counts[o] = counts.get(o, 0) + 1
    keep = set(sorted(counts, key=lambda o: -counts[o])[:N_OCCUPATIONS])
    print(f"{len(people)} people sampled; keeping the {len(keep)} most frequent occupations",
          flush=True)
    for o in sorted(keep, key=lambda o: -counts[o]):
        print(f"  {o}: {counts[o]} people", flush=True)
    qids = sorted(q for q, occs in people.items() if occs & keep)
    print(f"{len(qids)} people carry at least one of them", flush=True)
    data = claims_for(qids, keep)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print(f"wrote {len(data)} records to {path}")


if __name__ == "__main__":
    main()
