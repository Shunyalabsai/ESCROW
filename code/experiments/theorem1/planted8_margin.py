"""How far short does the planted eight-group stream fall under the corrections?

Sweeps the fixture's noise rate and, at the end of each run, reports for the best
candidate in the pool the evidence it holds against the price it must clear, both with
the seed key's ledger entry included (the shipped statistic) and excluded (the corrected
one), and with the support term charged for |T| and for |T| + 1 keys.
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from escrow import codes as C
from escrow.codes import price
from escrow.protocol import run_stream
from experiments.e4_baseline_army import planted8

rows = []
for noise in (0.0, 0.02, 0.05, 0.1, 0.2):
    recs, truth = planted8(noise=noise)
    g, b = run_stream(recs)
    best = None
    for sig, c in g.pool.items():
        gm = {k: (float(c.g_vec[k]) if c.g_vec is not None else c.g.get(k, 0.0))
              for k in c.keys}
        seed = c.sig[0]
        g_seed = gm.get(seed, 0.0)
        out = 0.0
        if c.g_vec is not None:
            for k in range(len(c.g_vec)):
                if k not in c.keys and float(c.g_vec[k]) > 0:
                    out += float(c.g_vec[k])
        else:
            out = sum(v for k, v in c.g.items() if k not in c.keys and v > 0)
        def scan(drop_seed, extra):
            gg = dict(gm)
            if drop_seed:
                gg.pop(seed, None)
            order = sorted(gg.items(), key=lambda kv: -kv[1])
            run_, bm = 0.0, None
            for j, (k, v) in enumerate(order, start=1):
                if v <= 0:
                    break
                run_ += v
                m = run_ + out - price(c.t, j + extra, g.n, g.K, g.e, len(g.keys))
                if bm is None or m > bm:
                    bm = m
            return bm
        row = {"sig": [g.key_name[seed], c.sig[1]], "t": c.t,
               "G": round(c.G, 2), "g_seed": round(g_seed, 2),
               "G_minus_seed": round(c.G - g_seed, 2),
               "best_margin_shipped": scan(False, 0),
               "best_margin_unselected_T_plus_1": scan(True, 1),
               "best_margin_unselected_T": scan(True, 0)}
        if best is None or (row["G_minus_seed"] > best["G_minus_seed"]):
            best = row
    for k in ("best_margin_shipped", "best_margin_unselected_T_plus_1",
              "best_margin_unselected_T"):
        if best and best[k] is not None:
            best[k] = round(best[k], 2)
    rows.append({"noise": noise, "K": g.K, "mints": len(g.mint_log),
                 "pool": len(g.pool), "best_candidate": best})
    print(json.dumps(rows[-1]), flush=True)
o = os.environ.get("ARMS_OUT", ".")
json.dump({"flags": C.flags(), "rows": rows},
          open(os.path.join(o, f"planted8_margin_{os.environ.get('ARM','x')}.json"), "w"),
          indent=2)
