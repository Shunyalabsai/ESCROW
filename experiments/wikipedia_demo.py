"""Generality demo: raw Wikipedia infoboxes -> ESCROW -> does it invent the types?

Fetches pages from a few unrelated categories, parses the FIRST infobox's raw
top-level `| key = value` parameters (no cleaning, no schema), shuffles them into
one stream, and runs the unmodified engine. If the method is generic, it should
mint one node per page type with the type's own raw keys as support - without
ever being told that films, people and mountains exist.
"""
import json
import random
import re
import sys, os, time
import urllib.parse
import urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective
from escrow.protocol import run_stream, describe as protocol_describe
from escrow.provenance import stamped

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
CACHE = os.path.join(OUT, "wiki_cache.json")
os.makedirs(OUT, exist_ok=True)

API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "escrow-research-demo/0.1 (structure-learning research)"}

CATS = {
    "film": "Category:1994 films",
    "person": "Category:English male film actors",
    "mountain": "Category:Alpine four-thousanders",
}
PER_CAT = 120


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


# ---- fetch (cached) --------------------------------------------------------- #
records = []
for cat_name, cat in CATS.items():
    ccache = CACHE.replace(".json", f".{cat_name}.json")
    if os.path.exists(ccache):
        recs = json.load(open(ccache))
    else:
        titles = category_titles(cat, PER_CAT)
        texts = fetch_wikitext(titles)
        recs = [r for r in (parse_infobox(tx) for tx in texts.values()) if r]
        json.dump(recs, open(ccache, "w"))
        print(f"{cat_name}: {len(titles)} pages -> {len(recs)} infobox records")
    records += [(cat_name, r) for r in recs]

rng = random.Random(0)
rng.shuffle(records)
allkeys = {k for _, r in records for k in r}
print(f"stream: {len(records)} records, {len(allkeys)} distinct raw keys")

# ---- run -------------------------------------------------------------------- #
g, b = run_stream([rec for _, rec in records])

# score nodes against the (held-out) category labels
truth = {}
for idx, (cat, rec) in enumerate(records, start=1):
    truth[idx] = cat
report = {"n": g.n, "K": g.K, "keys": len(g.keys), "protocol": protocol_describe(), "nodes": []}
for v in sorted(g.nodes.values(), key=lambda x: -x.t):
    cats = {}
    for r in v.members:
        cats[truth.get(r, "?")] = cats.get(truth.get(r, "?"), 0) + 1
    report["nodes"].append({
        "t": v.t, "purity": cats,
        "support": sorted(g.key_name[k] for k in v.S)[:14]})
print(json.dumps(report, indent=2)[:3500])
json.dump(stamped(report), open(os.path.join(OUT, "wikipedia_demo.json"), "w"), indent=2)
