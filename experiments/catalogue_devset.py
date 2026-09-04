"""First real-data run: ESCROW over raw catalogue records (development set).

Streams the RAW spec dicts from the e-commerce catalogue - keys exactly as
published ('Saree Fabric', 'color', 'Brand Color', ...). v1 scope: one source
department at a time, free-text-ish values kept as opaque tokens, stream capped
(the O(|A|) candidate absence loop is unoptimised; the Kahan absent-baseline
trick is queued engineering).
"""
import csv
import json
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from escrow.engine import EscrowGraph
from escrow.batch import BatchObjective

CSV = os.environ.get("ESCROW_CATALOG_CSV", "catalog.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(OUT, exist_ok=True)

DEPT = sys.argv[1] if len(sys.argv) > 1 else "Footwear"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

csv.field_size_limit(10 ** 7)
records = []
for row in csv.DictReader(open(CSV)):
    if row.get("department") != DEPT:
        continue
    try:
        sp = json.loads(row.get("specs") or "{}")
    except Exception:
        continue
    rec = {str(k): str(v)[:60] for k, v in sp.items()
           if isinstance(v, (str, int, float)) and str(v).strip()}
    if len(rec) >= 2:
        records.append(rec)
    if len(records) >= CAP:
        break

print(f"department={DEPT}: {len(records)} records, "
      f"{len({k for r in records for k in r})} distinct raw keys")

t0 = time.time()
g = EscrowGraph()
b = BatchObjective(g)
for i, rec in enumerate(records):
    g.process(rec)
    if (i + 1) % 500 == 0:
        b.repair()
        if (i + 1) % 2000 == 0:
            print(f"  n={i+1} K={g.K} pool={len(g.pool)} keys={len(g.keys)} "
                  f"({(i+1)/(time.time()-t0):.0f} rec/s)")
b.repair()
dt = time.time() - t0

nodes = sorted(g.nodes.values(), key=lambda v: -v.t)
inv_val = {ki.kid: {vid: s for s, vid in ki.inventory.items()} for ki in g.keys.values()}
summary = {
    "department": DEPT, "n": g.n, "K": g.K, "keys": len(g.keys),
    "records_per_sec": round(g.n / dt, 1),
    "total_mints": len(g.mint_log),
    "nodes": [{
        "t": v.t,
        "support": sorted(g.key_name[k] for k in v.S)[:12],
        "top_values": {g.key_name[k]: [(inv_val[k].get(vid, vid), c) for vid, c in
                                       sorted(v.blocks[k].counts.items(),
                                              key=lambda x: -x[1])[:3]]
                       for k in list(v.S)[:4] if k in v.blocks},
    } for v in nodes[:12]],
    "justifying_keys": [m["justifying_key"] for m in g.mint_log][:40],
}
print(json.dumps(summary, indent=2, default=str)[:4000])
with open(os.path.join(OUT, f"catalogue_{DEPT.lower().replace(' ','_')}.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
