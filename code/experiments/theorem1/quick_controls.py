"""The positive controls, run under whatever flags are set: two-group (true K=2),
planted eight-group (true K=8), Wikipedia infoboxes (three categories)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from escrow import codes as C
from escrow.protocol import run_stream
from experiments.e4_baseline_army import two_group, planted8, wikipedia, _ari

OUT = os.path.join(os.environ.get("ESCROW_ROOT", "../.."), "results")
rows = {"flags": C.flags()}
for name, fx in (("twogroup", lambda: two_group()),
                 ("planted8", lambda: planted8()),
                 ("wikipedia", lambda: wikipedia(OUT))):
    recs, truth = fx()
    g, b = run_stream(recs)
    lab = [-1] * len(recs)
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    bg = sum(1 for x in lab if x == -1)
    by = {}
    for i, x in enumerate(lab):
        by.setdefault(x, {}).setdefault(truth[i], 0)
        by[x][truth[i]] += 1
    purity = sum(max(c.values()) for c in by.values()) / len(recs)
    sizes = sorted((sum(c.values()) for k, c in by.items() if k != -1), reverse=True)
    rows[name] = {"K": g.K, "ari": round(_ari(truth, lab), 4), "mints": len(g.mint_log),
                  "background_records": bg, "purity_incl_background": round(purity, 4),
                  "node_sizes": sizes[:8], "n": len(recs)}
    print(name, json.dumps(rows[name]), flush=True)
o = os.environ.get("ARMS_OUT", ".")
json.dump(rows, open(os.path.join(o, f"controls_{os.environ.get('ARM','x')}.json"), "w"),
          indent=2)
