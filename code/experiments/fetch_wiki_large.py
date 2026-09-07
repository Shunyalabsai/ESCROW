"""Fetch a larger real encyclopedia stream, same source and same parser as the 320-record one.

WHY. The 320-record stream is the paper's one real in-scope stream and it is small: 7.8% of its
(key, value) pairs are ever seen twice, so almost no seed can reach the cohort its release price
demands. That is a property of the SAMPLE, not of encyclopedias. Fetching more pages of the same kind
raises the recurrence of exactly the pairs that carry a type (country, language, genre) while leaving
the parser, the source and the ground truth definition alone.

The cache is written to its own directory. The three files behind the 320-record stream are not
touched, so every number already computed from it stands.
"""
from __future__ import annotations

import json
import os
import sys
import time

import re
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# The three helpers are copied from wikipedia_demo.py rather than imported: that file is a script
# with its work at module level, so importing it re-runs the 320-record demo and rewrites its
# result. The parser below must stay character for character the same as the one that built the
# 320-record cache, or the two streams are not the same measurement.
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "escrow-research-demo/0.1 (structure-learning research)"}


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json", "maxlag": 5})
    req = urllib.request.Request(f"{API}?{q}", headers=UA)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def category_titles(cat, limit):
    titles, cont = [], {}
    while len(titles) < limit:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                 "cmlimit": min(200, limit), "cmnamespace": 0, **cont})
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        cont = d.get("continue") or {}
        if not cont:
            break
    return titles[:limit]


def fetch_wikitext(titles):
    out = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        d = api({"action": "query", "prop": "revisions", "rvprop": "content",
                 "rvslots": "main", "titles": "|".join(batch)})
        for p in d["query"]["pages"].values():
            try:
                out[p["title"]] = p["revisions"][0]["slots"]["main"]["*"]
            except (KeyError, IndexError):
                pass
        time.sleep(1.2)
    return out


def parse_infobox(text):
    """Raw top-level params of the first {{Infobox ...}} template."""
    m = re.search(r"\{\{\s*Infobox", text, re.I)
    if not m:
        return None
    i, depth, start = m.start(), 0, m.start()
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1; i += 2
        elif text.startswith("}}", i):
            depth -= 1; i += 2
            if depth == 0:
                break
        else:
            i += 1
    body = text[start + 2:i - 2]
    # split on top-level pipes only
    parts, depth, buf = [], 0, []
    for ch in body:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    rec = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = re.sub(r"<[^>]+>|\{\{[^}]*\}\}|\[\[|\]\]|&nbsp;", " ", v)
        v = re.sub(r"\s+", " ", v).strip()
        if k and v and len(k) < 40:
            rec[k] = v[:60]
    return rec if len(rec) >= 3 else None

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "wiki_large"))

# Ten categories that use ten different infobox templates. The label is the category we fetched
# from, which is the same ground truth convention the 320-record stream uses.
CATS = {
    "film":       "Category:1994 films",
    "album":      "Category:1994 albums",
    "novel":      "Category:1994 novels",
    "videogame":  "Category:1994 video games",
    "actor":      "Category:English male film actors",
    "mountain":   "Category:Mountains of the Alps",
    "river":      "Category:Rivers of Switzerland",
    "university": "Category:Universities and colleges in California",
    "company":    "Category:Companies based in San Francisco",
    "settlement": "Category:Cities in Alameda County, California",
    # Four of the ten above are container categories with almost no article members, which is not
    # visible from the category name. Rather than silently swap them out, they are kept and a
    # populous category of the SAME type is added beside them. E32 states a minimum size for a
    # category to enter its pool, so the choice is declared rather than made by deletion.
    "settlement2": "Category:Cities in California",
    "river2":      "Category:Rivers of Bavaria",
    "university2": "Category:Universities and colleges in Massachusetts",
    "building":    "Category:Skyscrapers in New York City",
    "aircraft":    "Category:Boeing aircraft",
    "bird":        "Category:Birds of Europe",
    # Growing the stream means more records of the SAME kinds, not more kinds, so these are second
    # and third categories of types already present. Year categories are used because they hold
    # their articles directly; several of the topic categories above turned out to be containers
    # whose articles sit in subcategories, which the category name does not reveal.
    "film2":      "Category:1995 films",
    "film3":      "Category:1996 films",
    "album2":     "Category:1995 albums",
    "album3":     "Category:1996 albums",
    "videogame2": "Category:1995 video games",
    "videogame3": "Category:1996 video games",
    "mountain2":  "Category:Mountains of Switzerland",
    "company2":   "Category:Companies based in New York City",
}
PER_CAT = 300


def main():
    os.makedirs(OUT, exist_ok=True)
    for cat, title in CATS.items():
        path = os.path.join(OUT, f"wiki_cache.{cat}.json")
        if os.path.exists(path):
            print(f"{cat}: cached ({len(json.load(open(path)))} records)", flush=True)
            continue
        t0 = time.time()
        try:
            titles = category_titles(title, PER_CAT)
            text = fetch_wikitext(titles)
        except Exception as exc:                      # a category that fails is skipped, not faked
            print(f"{cat}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        recs = []
        for t, wt in text.items():
            box = parse_infobox(wt)
            if box:
                rec = {str(k): str(v) for k, v in box.items() if isinstance(v, (str, int, float))}
                if rec:
                    recs.append(rec)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(recs, fh)
        print(f"{cat}: {len(titles)} titles -> {len(recs)} infobox records "
              f"({time.time() - t0:.0f}s)", flush=True)
    total = 0
    for f in sorted(os.listdir(OUT)):
        if f.startswith("wiki_cache."):
            total += len(json.load(open(os.path.join(OUT, f))))
    print("total records:", total)


if __name__ == "__main__":
    main()
