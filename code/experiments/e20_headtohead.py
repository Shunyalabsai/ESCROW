"""E20: AutoPKG vs ESCROW head to head on AutoPKG's own Lazada data.

Shared decision: for each candidate attribute key, MERGE it into an existing key node
or MINT (ADD) a new one. Both systems see exactly the same record stream, built by
lazada_c2_rawkey.load_records() so record construction is identical to the paper's
other Lazada numbers.

AutoPKG arm is a faithful reimplementation of baselines/autopkg/src/kgd_agent/kgd.py
(the repository ships no runnable entry point: main.py is 0 bytes and 4 of the 6
relative imports in kgd.py are missing modules).
"""
import argparse, collections, csv, itertools, json, os, re, sys, time

# The repository root, two levels above code/experiments/, which is how every other runner here
# resolves it. It was an absolute path to the machine this was first run on, which is a private path
# and does not belong in a released file; ESCROW_ROOT overrides it if the checkout is elsewhere.
ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "experiments"))
os.environ.setdefault("ESCROW_ROOT", ROOT)

from lazada_c2_rawkey import load_records, norm_key

APKG = os.path.join(ROOT, "baselines/autopkg/src")
sys.path.insert(0, APKG)
# their prompts, imported from their file verbatim
_pp = os.path.join(APKG, "kgd_agent", "kgd_prompts.py")
_ns = {}
exec(compile(open(_pp, encoding="utf-8").read(), _pp, "exec"), _ns)
KGD_AM_PROMPT = _ns["KGD_AM_PROMPT"]
KEY_DESC_PROMPT = _ns["KEY_DESC_PROMPT"]

EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"      # their paper, Section: "using Qwen3-Embedding-0.6B"
AGENT_MODEL = "Qwen/Qwen2.5-14B-Instruct"    # DEVIATION: paper uses Qwen3-30B/Next-80B
NODE_TYPE = "Attribute Key"                  # single global key namespace, as in their shipped KG
TOPK = 10                                    # kgd.py: _check_candidate(..., topk=10)


# ---------------------------------------------------------------- gold ------
def build_gold(recs):
    keyfreq = collections.Counter()
    for r in recs:
        keyfreq.update(r.keys())
    fam = collections.defaultdict(set)
    for k in keyfreq:
        fam[norm_key(k)].add(k)
    pos = set()
    for n, v in fam.items():
        if not n or len(v) < 2:
            continue
        vs = sorted(v)
        for a, b in itertools.combinations(vs, 2):
            pos.add((a, b))
    neg = set()
    for r in recs:
        ks = sorted(r.keys())
        for a, b in itertools.combinations(ks, 2):
            neg.add((a, b))
    collide = pos & neg
    neg -= collide                    # a pair attested as co-present is not a pure variant
    pos -= collide
    return keyfreq, pos, neg, len(collide)


def stratum(f):
    return ("head(>=100)" if f >= 100 else "mid(10-99)" if f >= 10
            else "tail(2-9)" if f >= 2 else "singleton(1)")


def score(pred_same, pos, neg, keyfreq):
    """pred_same(a,b) -> bool. Pairwise P/R/F1 over gold-labelled pairs only."""
    def bucket(a, b):
        return stratum(min(keyfreq[a], keyfreq[b]))
    rows = collections.defaultdict(lambda: dict(tp=0, fn=0, fp=0, tn=0))
    for a, b in pos:
        k = "tp" if pred_same(a, b) else "fn"
        rows[bucket(a, b)][k] += 1; rows["ALL"][k] += 1
    for a, b in neg:
        k = "fp" if pred_same(a, b) else "tn"
        rows[bucket(a, b)][k] += 1; rows["ALL"][k] += 1
    out = {}
    for s, c in rows.items():
        tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        out[s] = dict(tp=tp, fn=fn, fp=fp, tn=tn,
                      precision=round(p, 4) if p is not None else None,
                      recall=round(r, 4) if r is not None else None,
                      f1=round(f1, 4) if f1 is not None else None,
                      gold_positives=tp + fn, gold_negatives=fp + tn)
    return out


# ------------------------------------------------------------- autopkg ------
class ApkgKGD:
    """Faithful reimplementation of the key-task decision loop in kgd.py."""

    def __init__(self, agent, emb, log):
        self.agent, self.emb, self.log = agent, emb, log
        self.nodes = {}          # node_id -> dict(name, desc, examples)
        self.idx = {}            # (node_type, name) -> node_id   (_register_idx, synonym aware)
        self.vecs = []           # parallel to self.order
        self.order = []
        self.counter = 0
        self.stats = collections.Counter()

    def next_node_id(self):
        self.counter += 1
        return "K%05d" % self.counter

    def to_candidate(self, name, desc, examples, node_id=None):
        # RECONSTRUCTED: data.py (which defines NodeData.to_candidate) is missing
        d = {"node_name": name, "description": desc, "examples": examples}
        if node_id:
            d = {"node_id": node_id, **d}
        return str(d)

    def process(self, name, desc, examples, vec):
        import numpy as np
        # kgd.py _process_logic step 1: exact match (synonym aware) -> MERGE, no LLM
        nid = self.idx.get((NODE_TYPE, name))
        if nid:
            self.stats["exact_merge"] += 1
            return nid, "MERGE(exact)"
        # _check_candidate: top-k nearest existing nodes of this node_type
        if self.order:
            M = np.stack(self.vecs)
            d = ((M - vec[None, :]) ** 2).sum(1)          # IndexFlatL2 == squared L2
            top = np.argsort(d)[:TOPK]
            cands = [self.order[i] for i in top]
        else:
            cands = []
        if not cands:
            nid = self.next_node_id()
            self._add(nid, name, desc, examples, vec)
            self.stats["add_empty_kg"] += 1
            return nid, "ADD(empty)"
        pretty_nodes = [self.to_candidate(self.nodes[c]["name"], self.nodes[c]["desc"],
                                          self.nodes[c]["examples"], c) for c in cands]
        prompt = KGD_AM_PROMPT.format(pretty_nodes=pretty_nodes, node_type=NODE_TYPE,
                                      pretty_candidate=self.to_candidate(name, desc, examples))
        raw = self.agent(prompt, max_new_tokens=12)
        action = self._parse(raw, cands)
        if action == "ADD":
            nid = self.next_node_id()
            self._add(nid, name, desc, examples, vec)
            self.stats["add_llm"] += 1
            return nid, "ADD"
        if action.startswith("MERGE "):
            tgt = action.split()[1]
            self.idx[(NODE_TYPE, name)] = tgt            # merge_node -> add_synonym + register
            self.stats["merge_llm"] += 1
            return tgt, "MERGE"
        # kgd.py _apply_action: invalid action -> `pass`, nothing is committed
        self.stats["invalid"] += 1
        return None, "INVALID:" + raw[:40].replace("\n", " ")

    def _parse(self, raw, cands):
        # kgd.py _split_thought_action then action_split = action.split()
        s = raw
        if "</think>" in s:
            s = s.split("</think>", 1)[1]
        s = s.strip()
        tok = s.split()
        if not tok:
            return "INVALID"
        a = tok[0].strip().upper()
        if a == "ADD":
            return "ADD"
        if a == "MERGE" and len(tok) > 1:
            t = tok[1].strip().strip(".,;:`'\"<>")
            if t in cands:
                return "MERGE " + t
            return "INVALID"
        return "INVALID"

    def _add(self, nid, name, desc, examples, vec):
        self.nodes[nid] = dict(name=name, desc=desc, examples=examples)
        self.idx[(NODE_TYPE, name)] = nid
        self.order.append(nid)
        self.vecs.append(vec)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys-limit", type=int, default=0)
    ap.add_argument("--out", default="e20_autopkg_head_to_head.json")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    recs, pids, dstats = load_records()
    if args.limit:
        recs, pids = recs[:args.limit], pids[:args.limit]
    print("[data]", dstats, "streaming", len(recs), flush=True)

    keyfreq, pos, neg, collided = build_gold(recs)
    print("[gold] positives=%d negatives=%d dropped_collisions=%d unique_keys=%d"
          % (len(pos), len(neg), collided, len(keyfreq)), flush=True)

    # candidate key stream: first appearance in the same seed-0 record stream
    seen, kstream = set(), []
    for r in recs:
        for k in r:
            if k not in seen:
                seen.add(k); kstream.append(k)
    if args.keys_limit:
        kstream = kstream[:args.keys_limit]
    print("[apkg] candidate key stream:", len(kstream), flush=True)

    results = {}

    # ---------- arm 1: AutoPKG exact-match-only floor (no LLM) ----------
    exact_part = {k: k for k in kstream}
    results["autopkg_exact_only"] = score(
        lambda a, b: exact_part.get(a, a) == exact_part.get(b, b), pos, neg, keyfreq)

    apkg_summary = {}
    if not args.skip_llm:
        import torch, numpy as np
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sentence_transformers import SentenceTransformer

        print("[emb] loading", EMB_MODEL, flush=True)
        st = SentenceTransformer(EMB_MODEL, device="cuda")
        E = st.encode(kstream, batch_size=64, show_progress_bar=False,
                      convert_to_numpy=True, normalize_embeddings=False)
        E = E.astype("float32")
        print("[emb] done", E.shape, flush=True)
        del st; torch.cuda.empty_cache()

        print("[agent] loading", AGENT_MODEL, flush=True)
        tok = AutoTokenizer.from_pretrained(AGENT_MODEL, padding_side="left")
        mdl = AutoModelForCausalLM.from_pretrained(AGENT_MODEL, torch_dtype=torch.bfloat16,
                                                   device_map="cuda")
        mdl.eval()

        def chat(prompts, max_new_tokens):
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             tokenize=False, add_generation_prompt=True)
                     for p in prompts]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    for o in out]

        # _update_properties for task 'key': description + examples, batched
        print("[agent] generating key descriptions (their KEY_DESC_PROMPT)", flush=True)
        descs, exs = {}, {}
        B = 32
        for i in range(0, len(kstream), B):
            chunk = kstream[i:i + B]
            outs = chat([KEY_DESC_PROMPT.format(attribute_key=k) for k in chunk], 150)
            for k, o in zip(chunk, outs):
                m = re.search(r"Description: ([^\n]+)", o, re.DOTALL)
                e = re.search(r"Examples: ([^\n]+)", o, re.DOTALL)
                descs[k] = m.group(1).strip() if m else o.strip()[:200]
                exs[k] = e.group(1).strip() if e else ""
            if (i // B) % 20 == 0:
                print("  desc %d/%d  %.0fs" % (i + len(chunk), len(kstream), time.time() - t0), flush=True)

        agent_fn = lambda p, max_new_tokens=12: chat([p], max_new_tokens)[0]
        kgd = ApkgKGD(agent_fn, None, print)
        assign = {}
        print("[agent] KGD decision loop", flush=True)
        for i, k in enumerate(kstream):
            nid, act = kgd.process(k, descs[k], exs[k], E[i])
            assign[k] = nid if nid else "__INVALID__%d" % i
            if (i + 1) % 250 == 0:
                print("  kgd %d/%d nodes=%d %s %.0fs"
                      % (i + 1, len(kstream), len(kgd.nodes), dict(kgd.stats), time.time() - t0),
                      flush=True)
        apkg_summary = dict(actions=dict(kgd.stats), canonical_key_nodes=len(kgd.nodes),
                            candidates=len(kstream))
        print("[agent] done", apkg_summary, flush=True)
        results["autopkg_kgd_llm"] = score(
            lambda a, b: assign.get(a, a) == assign.get(b, b), pos, neg, keyfreq)
        json.dump({k: v for k, v in assign.items()},
                  open(os.path.join(ROOT, "results", "e20_apkg_assign.json"), "w"), indent=0)
        del mdl; torch.cuda.empty_cache()

    # ---------- arm 2: ESCROW under escrow.protocol ----------
    from escrow import protocol
    from escrow.fastrepair import FastBatchObjective
    print("[escrow] run_stream", flush=True)
    te = time.time()
    g, b = protocol.run_stream(recs, every=200, objective_cls=FastBatchObjective,
                               on_progress=lambda n, gg: print("  escrow n=%d K=%d %.0fs"
                                                               % (n, gg.K, time.time() - te), flush=True)
                               if n % 2000 == 0 else None)
    esc_wall = time.time() - te
    print("[escrow] K=%d mints=%d %.0fs" % (g.K, len(g.mint_log), esc_wall), flush=True)

    supp = collections.defaultdict(set)
    for v in g.nodes.values():
        for kid in v.S:
            supp[kid].add(v.nid)

    def esc_same(a, b):
        ka = g.keys.get(a); kb = g.keys.get(b)
        if ka is None or kb is None:
            return False
        return bool(supp.get(ka.kid, set()) & supp.get(kb.kid, set()))

    # ---- dump ESCROW key-level structure so any read-out can be rescored offline ----
    owner_hist = collections.defaultdict(collections.Counter)
    for n_, om in g.record_owner.items():
        for kid, ow in om.items():
            owner_hist[kid][ow] += 1
    kdump = {}
    for name, ki in g.keys.items():
        kid = ki.kid
        kdump[name] = dict(
            kid=kid, P=ki.P, n_values=len(ki.inventory),
            supporting_nodes={str(v.nid): v.p.get(kid, 0)
                              for v in g.nodes.values() if kid in v.S},
            owners={str(o): c for o, c in owner_hist.get(kid, {}).items()},
        )
    json.dump(dict(keys=kdump,
                   nodes={str(v.nid): dict(t=v.t, support=sorted(v.S)) for v in g.nodes.values()},
                   key_name={str(k): v for k, v in g.key_name.items()}),
              open(os.path.join(ROOT, "results", "e20_escrow_keydump.json"), "w"))
    print("[escrow] key dump written", flush=True)

    results["escrow"] = score(esc_same, pos, neg, keyfreq)

    json.dump(dict(positives=sorted(pos), negatives=sorted(neg),
                   keyfreq=dict(keyfreq)),
              open(os.path.join(ROOT, "results", "e20_gold.json"), "w"))

    out = dict(
        benchmark="E20 AutoPKG vs ESCROW, AutoPKG's Lazada data, merge/mint decision",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        data=dict(**dstats, unique_raw_keys=len(keyfreq),
                  gold_positive_pairs=len(pos), gold_negative_pairs=len(neg),
                  dropped_collisions=collided, candidate_key_stream=len(kstream)),
        autopkg_run=apkg_summary,
        escrow_run=dict(K=g.K, mints=len(g.mint_log), wall_s=round(esc_wall, 1),
                        protocol=protocol.describe(200, FastBatchObjective)),
        results=results,
        models=dict(embedder=EMB_MODEL, agent=AGENT_MODEL),
        runtime_s=round(time.time() - t0, 1),
    )
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    p = os.path.join(ROOT, "results", args.out)
    json.dump(out, open(p, "w"), indent=1)
    print("[saved]", p, flush=True)
    for arm, tab in results.items():
        a = tab.get("ALL", {})
        print("  %-22s P=%s R=%s F1=%s" % (arm, a.get("precision"), a.get("recall"), a.get("f1")), flush=True)


if __name__ == "__main__":
    main()
