"""LLM GRAPH FORMATION vs ESCROW, apple to apple, on public data.

The author's question: "test the public data for any ill-formed graph formation. What is
the accuracy of that?"  We build the same latent-node graph two ways on the SAME records
in the SAME order and score the failure modes the paper's opening names: duplicate nodes
(one kind of thing under two names), wrong merges, non-determinism across runs, token
cost, plus plain accuracy against ground truth.

LLM arm (the honest streaming analogue of AutoSchemaKG / AutoPKG): records arrive in
chunks of 20; the prompt shows the CURRENT node list (name + one-line description) and
the new records as key=value lines; the model returns, per record, an existing node name
or a NEW node (name + description) as JSON.  The node list is carried across chunks.  No
cluster count is given.  This is exactly the create-or-attach decision ESCROW makes, made
by prompting.  Every token in and out is counted.
  (a) temperature 0.7, seeds 0,1,2 (pure temperature sampling: top_p/top_k off,
      repetition_penalty 1.0);  (b) greedy, once.

ESCROW arm: EscrowGraph + BatchObjective(g).install(), repair every 100 records, final
repair(full=True).  Deterministic, zero tokens.  Run three times to show identical output.

Datasets:
  wikipedia  320 raw infobox records (film 120, person 113, mountain 87), shuffled seed 0,
             truth = category (gold).
  lazada     first 1,000 spec-bearing listings of the seed-0 shuffle of 21,365; truth =
             silver of_type labels from the AutoPKG KG (labelled silver).

Usage, from the code directory:
  .venv python experiments/llm_graph_formation.py data
  CUDA_VISIBLE_DEVICES=2 .venv python experiments/llm_graph_formation.py llm \
        --jobs wikipedia:sampled:0,wikipedia:sampled:1,wikipedia:greedy:0
  .venv python experiments/llm_graph_formation.py escrow
  .venv python experiments/llm_graph_formation.py score
"""
import argparse
import collections
import copy
import itertools
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.environ.get("ESCROW_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# data, baselines and results live under ESCROW_ROOT (default: the directory above this code directory)

from escrow.provenance import stamped  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results")
# the Wikipedia infobox caches wiki_cache.<category>.json are written by
# experiments/wikipedia_demo.py and are committed under results/, not under a wiki/ directory
WIKI_DIR = OUT_DIR
RUN_DIR = os.path.join(OUT_DIR, "llm_gf_runs")
FINAL = os.path.join(OUT_DIR, "llm_graph_formation.json")
MODEL_PREF = ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
CHUNK = 20
MAX_NEW = 1600
LAZADA_LIMIT = 1000
LAZADA_TOTAL = 21365


# =============================================================== data ======
def build_wikipedia():
    records = []
    for cat in ["film", "person", "mountain"]:            # same order as wikipedia_demo.py
        recs = json.load(open(os.path.join(WIKI_DIR, f"wiki_cache.{cat}.json")))
        records += [(cat, r) for r in recs]
    random.Random(0).shuffle(records)
    return dict(name="wikipedia", truth_label="category (gold)",
                records=[r for _, r in records], truth=[c for c, _ in records],
                truth_names={c: c for c in ["film", "person", "mountain"]},
                total_stream=len(records))


def build_lazada(limit=LAZADA_LIMIT):
    from experiments.lazada_c2_rawkey import load_records, load_silver_types
    recs, pids, stats = load_records()                    # seed-0 shuffle, raw keys
    pid2type, type_name = load_silver_types()
    recs, pids = recs[:limit], pids[:limit]
    truth = [pid2type.get(p) for p in pids]
    return dict(name="lazada", truth_label="silver of_type (AutoPKG KG, model-derived)",
                records=recs, truth=truth, pids=pids,
                truth_names={t: type_name.get(t, "?") for t in set(truth) if t},
                total_stream=stats["parsed_records"], data_stats=stats)


def cmd_data(_):
    os.makedirs(RUN_DIR, exist_ok=True)
    for ds in (build_wikipedia(), build_lazada()):
        path = os.path.join(RUN_DIR, f"data_{ds['name']}.json")
        json.dump(ds, open(path, "w"))
        n = len(ds["records"])
        keys = {k for r in ds["records"] for k in r}
        tv = [t for t in ds["truth"] if t]
        print(f"[data] {ds['name']}: {n} records, {len(keys)} raw keys, "
              f"{len(set(tv))} truth classes over {len(tv)} labelled records -> {path}")


def load_data(name):
    return json.load(open(os.path.join(RUN_DIR, f"data_{name}.json")))


# ================================================================ LLM ======
SYSTEM = ("You build a graph of nodes from a stream of records. Each record describes one "
          "item as key=value lines. A node stands for one kind of item. Records of the same "
          "kind of item attach to the same node. Reuse an existing node whenever one fits. "
          "Create a new node only when no existing node fits. Reply with JSON only.")


def make_prompt(nodes, chunk, start):
    L = ["Existing nodes (name: description):"]
    if nodes:
        L += [f"- {n}: {d}" for n, d in nodes.items()]
    else:
        L.append("(none yet)")
    L += ["", "New records:"]
    for i, rec in enumerate(chunk):
        L.append(f"[record {start + i}]")
        L += [f"{k}={v}" for k, v in rec.items()]
        L.append("")
    L += ["For every record above, decide which node it attaches to. Output a JSON array "
          "with one object per record, in record order:",
          '- existing node: {"record": <id>, "node": "<existing node name, exactly as listed>"}',
          '- new node: {"record": <id>, "node": "<new node name>", "new": true, '
          '"description": "<one line saying what kind of item the node stands for>"}',
          "A node created for an earlier record in this same answer may be reused by later "
          "records by name. Output the JSON array only."]
    return "\n".join(L)


def parse_answer(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    i, j = t.find("["), t.rfind("]")
    objs = None
    if i != -1 and j > i:
        try:
            objs = json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            objs = None
    if objs is None:                                       # truncated or dirty: salvage objects
        objs = []
        for m in re.finditer(r"\{[^{}]*\}", t):
            try:
                objs.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
    return [o for o in objs if isinstance(o, dict)]


NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_name(s):
    return NORM_RE.sub("", str(s).lower())


def apply_answer(objs, nodes, chunk_ids, C, chunk_no):
    """nodes: OrderedDict name -> {'description', 'created_chunk'}; returns {rid: node}."""
    assign = {}
    loose = {norm_name(n): n for n in nodes}
    ids = set(chunk_ids)
    for o in objs:
        try:
            rid = int(o.get("record"))
        except (TypeError, ValueError):
            C["bad_record_id"] += 1
            continue
        if rid not in ids:
            C["bad_record_id"] += 1
            continue
        if rid in assign:
            C["duplicate_record_entry"] += 1
            continue
        name = o.get("node")
        if not isinstance(name, str) or not name.strip():
            C["missing_node_name"] += 1
            continue
        name = name.strip()
        is_new = bool(o.get("new"))
        if name in nodes:
            if is_new:
                C["new_flag_on_existing_name"] += 1
            assign[rid] = name
            continue
        key = norm_name(name)
        if key in loose:
            C["loose_name_match"] += 1                    # case/punct variant of a listed name
            assign[rid] = loose[key]
            continue
        if not is_new:
            C["implicit_new_node"] += 1                   # referenced a node that does not exist
        desc = o.get("description")
        nodes[name] = dict(description=desc if isinstance(desc, str) and desc.strip()
                           else "(no description given)", created_chunk=chunk_no)
        loose[key] = name
        C["new_nodes"] += 1
        assign[rid] = name
    C["unparsed_records"] += sum(1 for r in chunk_ids if r not in assign)
    return assign


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    last = None
    for name in MODEL_PREF:
        try:
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                         device_map="cuda")
            model.eval()
            print(f"[model] {name} loaded in {time.time() - t0:.0f}s on "
                  f"{torch.cuda.get_device_name(0)}", flush=True)
            return name, tok, model
        except Exception as e:                            # noqa: BLE001
            last = e
            print(f"[model] {name} failed: {e!r}", flush=True)
    raise RuntimeError(f"no model loaded: {last!r}")


def gen_config(model, arm):
    gc = copy.deepcopy(model.generation_config)
    gc.max_new_tokens = MAX_NEW
    gc.repetition_penalty = 1.0
    gc.pad_token_id = model.generation_config.eos_token_id if not isinstance(
        model.generation_config.eos_token_id, list) else model.generation_config.eos_token_id[-1]
    gc.top_p = None
    gc.top_k = None
    if arm == "sampled":
        gc.do_sample = True
        gc.temperature = 0.7
    else:
        gc.do_sample = False
        gc.temperature = None
    return gc


def run_llm_api(base_url, model_name, api_key, ds, arm, seed, max_chunks=0):
    """The same experiment against an OpenAI-compatible endpoint instead of a local
    checkpoint. Prompts, chunking, parsing, counters and the record order are the
    ones above, so the two arms are comparable; only the call changes. Used for the
    second model, which is served rather than loaded."""
    import urllib.request
    random.seed(seed)
    recs = ds["records"]
    N = len(recs)
    nodes = collections.OrderedDict()
    assign = [None] * N
    C = collections.Counter()
    chunks_log, raw = [], []
    sampled = (arm == "sampled")
    tin = tout = 0
    t0 = time.time()
    n_chunks = (N + CHUNK - 1) // CHUNK
    if max_chunks:
        n_chunks = min(n_chunks, max_chunks)
    for c in range(n_chunks):
        start = c * CHUNK
        chunk = recs[start:start + CHUNK]
        ids = list(range(start, start + len(chunk)))
        # This server takes neither a system role nor a seed, and reports zero usage,
        # so the system prompt is folded into the user turn and the tokens are counted
        # from the text with the same 4 characters per token rule used for the
        # extrapolation in the cost figure. Both facts are recorded in the run file.
        user = SYSTEM + "\n\n" + make_prompt(
            {n: v["description"] for n, v in nodes.items()}, chunk, start)
        body = {"model": model_name,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": MAX_NEW,
                "temperature": 0.7 if sampled else 0.0}
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + (api_key or "EMPTY")})
        tc = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            ans = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage") or {}
            n_in = int(usage.get("prompt_tokens") or 0) or max(1, len(user) // 4)
            n_out = int(usage.get("completion_tokens") or 0) or max(1, len(ans) // 4)
            finished = (resp["choices"][0].get("finish_reason") == "stop")
        except Exception as e:                              # noqa: BLE001
            print(f"[api {ds['name']} {arm} s{seed}] chunk {c} failed: {e!r}", flush=True)
            ans, n_in, n_out, finished = "", 0, 0, False
        objs = parse_answer(ans)
        K0 = len(nodes)
        a = apply_answer(objs, nodes, ids, C, c)
        for rid, name in a.items():
            assign[rid] = name
        tin += n_in
        tout += n_out
        dt = time.time() - tc
        chunks_log.append(dict(chunk=c, records=len(chunk), tokens_in=n_in, tokens_out=n_out,
                               finished=finished, parsed_objects=len(objs), assigned=len(a),
                               nodes_before=K0, nodes_after=len(nodes), seconds=round(dt, 1)))
        raw.append(ans)
        print(f"[api {ds['name']} {arm} s{seed}] chunk {c + 1}/{n_chunks} in={n_in} "
              f"out={n_out} assigned={len(a)}/{len(chunk)} K={len(nodes)} {dt:.1f}s", flush=True)
    wall = time.time() - t0
    return dict(dataset=ds["name"], method="llm", arm=arm, seed=seed, model=model_name,
                served_by=base_url,
                generation=dict(do_sample=sampled, temperature=0.7 if sampled else 0.0,
                                top_p=None, top_k=None, repetition_penalty=1.0,
                                max_new_tokens=MAX_NEW),
                chunk_size=CHUNK, n_records=N, n_chunks=n_chunks,
                tokens_in=tin, tokens_out=tout, wall_s=round(wall, 1),
                counters=dict(C), nodes=[dict(name=n, **v) for n, v in nodes.items()],
                assignments=assign, chunks=chunks_log, raw_outputs=raw)


def run_llm(model_name, tok, model, ds, arm, seed, max_chunks=0):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    recs = ds["records"]
    N = len(recs)
    nodes = collections.OrderedDict()
    assign = [None] * N
    C = collections.Counter()
    chunks_log, raw = [], []
    gc = gen_config(model, arm)
    eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
    tin = tout = 0
    t0 = time.time()
    n_chunks = (N + CHUNK - 1) // CHUNK
    if max_chunks:
        n_chunks = min(n_chunks, max_chunks)
    for c in range(n_chunks):
        start = c * CHUNK
        chunk = recs[start:start + CHUNK]
        ids = list(range(start, start + len(chunk)))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": make_prompt(
                        {n: v["description"] for n, v in nodes.items()}, chunk, start)}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        n_in = enc.input_ids.shape[1]
        tc = time.time()
        with torch.no_grad():
            out = model.generate(**enc, generation_config=gc)
        gen = out[0, n_in:].tolist()
        finished = any(t in eos for t in gen)
        n_out = len(gen)
        ans = tok.decode(gen, skip_special_tokens=True)
        objs = parse_answer(ans)
        K0 = len(nodes)
        a = apply_answer(objs, nodes, ids, C, c)
        for rid, name in a.items():
            assign[rid] = name
        tin += n_in
        tout += n_out
        dt = time.time() - tc
        chunks_log.append(dict(chunk=c, records=len(chunk), tokens_in=n_in, tokens_out=n_out,
                               finished=finished, parsed_objects=len(objs), assigned=len(a),
                               nodes_before=K0, nodes_after=len(nodes), seconds=round(dt, 1)))
        raw.append(ans)
        print(f"[llm {ds['name']} {arm} s{seed}] chunk {c + 1}/{n_chunks} in={n_in} "
              f"out={n_out} fin={finished} assigned={len(a)}/{len(chunk)} K={len(nodes)} "
              f"{dt:.1f}s", flush=True)
    wall = time.time() - t0
    return dict(dataset=ds["name"], method="llm", arm=arm, seed=seed, model=model_name,
                generation=dict(do_sample=gc.do_sample, temperature=gc.temperature,
                                top_p=gc.top_p, top_k=gc.top_k,
                                repetition_penalty=gc.repetition_penalty,
                                max_new_tokens=gc.max_new_tokens),
                chunk_size=CHUNK, n_records=N, n_chunks=n_chunks,
                tokens_in=tin, tokens_out=tout, wall_s=round(wall, 1),
                counters=dict(C), nodes=[dict(name=n, **v) for n, v in nodes.items()],
                assignments=assign, chunks=chunks_log, raw_outputs=raw)


def cmd_llm(args):
    os.makedirs(RUN_DIR, exist_ok=True)
    jobs = [j.split(":") for j in args.jobs.split(",") if j]
    use_api = bool(getattr(args, "api", ""))
    tag = getattr(args, "tag", "") or ""
    if use_api:
        model_name, tok, model = getattr(args, "api_model", "") or "served", None, None
        print(f"[model] served {model_name} at {args.api}", flush=True)
    else:
        model_name, tok, model = load_model()
    data = {}
    for dsn, arm, seed in jobs:
        seed = int(seed)
        if dsn not in data:
            data[dsn] = load_data(dsn)
        path = os.path.join(RUN_DIR, f"llm_{dsn}_{arm}_s{seed}{tag}.json")
        if os.path.exists(path) and not args.force:
            print(f"[skip] {path} exists", flush=True)
            continue
        if use_api:
            res = run_llm_api(args.api, model_name, args.api_key, data[dsn], arm, seed,
                              args.max_chunks)
        else:
            res = run_llm(model_name, tok, model, data[dsn], arm, seed, args.max_chunks)
        if args.max_chunks:
            path = path.replace(".json", f"_smoke{args.max_chunks}.json")
        json.dump(res, open(path, "w"), indent=1)
        print(f"[saved] {path} tokens_in={res['tokens_in']} tokens_out={res['tokens_out']} "
              f"wall={res['wall_s']}s K={len(res['nodes'])} counters={res['counters']}",
              flush=True)


# ============================================================= ESCROW ======
def run_escrow(ds, rep):
    from escrow.engine import EscrowGraph
    from escrow.batch import BatchObjective
    recs = ds["records"]
    N = len(recs)
    g = EscrowGraph()
    b = BatchObjective(g).install()
    merges = []
    orig = b._apply_merge

    def logged(v, w):
        merges.append((v.nid, w.nid))
        orig(v, w)
    b._apply_merge = logged
    t0 = time.time()
    t_rep = 0.0
    for i, r in enumerate(recs):
        g.process(r)
        if (i + 1) % 100 == 0:
            ts = time.time()
            b.repair()
            t_rep += time.time() - ts
    ts = time.time()
    b.repair(full=True)
    t_rep += time.time() - ts
    wall = time.time() - t0
    # record labels: node with most members containing it (tie -> lowest nid); else -1
    lab = [-1] * N
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    # receipts: mint entry of the survivor plus the entries it absorbed (transitively)
    absorbed = collections.defaultdict(list)
    for v, w in merges:
        absorbed[v].append(w)

    def closure(nid):
        out, stack = [], list(absorbed.get(nid, []))
        while stack:
            w = stack.pop()
            out.append(w)
            stack += absorbed.get(w, [])
        return out
    mint_by = {e["node"]: e for e in g.mint_log}
    inv = {}

    def val_name(kid, vid):
        if kid not in inv:
            inv[kid] = {v: k for k, v in g.key_by_id[kid].inventory.items()}
        return inv[kid].get(vid, "?")
    receipts = []
    for v in sorted(g.nodes.values(), key=lambda x: (-x.t, x.nid)):
        keys = sorted(v.S, key=lambda k: -v.p.get(k, 0))
        own = mint_by.get(v.nid)
        abs_ids = closure(v.nid)
        top_vals = {}
        for kid in keys[:5]:
            blk = v.blocks.get(kid)
            if blk is not None:
                top_vals[g.key_name[kid]] = [(val_name(kid, vid)[:40], c) for vid, c in
                                             sorted(blk.counts.items(),
                                                    key=lambda kv: -kv[1])[:3]]
        receipts.append(dict(
            node=v.nid, members=v.t, support_size=len(v.S),
            support=[(g.key_name[k], v.p.get(k, 0)) for k in keys[:12]],
            justifying_key=own["justifying_key"] if own else None,
            mint_bits=own["G_total"] if own else None,
            mint_price=own["price_paid"] if own else None,
            mint_seed=own["seed"] if own else None,
            mint_at_record=own["n"] if own else None,
            per_key_bits=own.get("per_key_bits") if own else None,
            absorbed_nodes=[dict(node=w, justifying_key=mint_by[w]["justifying_key"],
                                 members_at_mint=mint_by[w]["members"])
                            for w in abs_ids if w in mint_by],
            top_values=top_vals))
    return dict(dataset=ds["name"], method="escrow", rep=rep, n_records=N,
                K=g.K, mints=len(g.mint_log), merges=len(merges), keys=len(g.keys),
                tokens_in=0, tokens_out=0, wall_s=round(wall, 2), repair_s=round(t_rep, 2),
                assignments=lab, receipts=receipts,
                node_members={v.nid: sorted(m - 1 for m in v.members) for v in g.nodes.values()})


def cmd_escrow(args):
    os.makedirs(RUN_DIR, exist_ok=True)
    for dsn in args.datasets.split(","):
        ds = load_data(dsn)
        for rep in range(3):
            res = run_escrow(ds, rep)
            path = os.path.join(RUN_DIR, f"escrow_{dsn}_r{rep}.json")
            json.dump(res, open(path, "w"), indent=1)
            print(f"[escrow {dsn} r{rep}] K={res['K']} mints={res['mints']} "
                  f"merges={res['merges']} wall={res['wall_s']}s -> {path}", flush=True)


# ============================================================== score ======
def as_int_labels(labels):
    m, out = {}, []
    for x in labels:
        if x is None:
            x = -1
        out.append(m.setdefault(x, len(m)))
    return out


def cluster_metrics(labels, truth):
    from sklearn.metrics import adjusted_rand_score
    labels = [(-1 if x is None else x) for x in labels]
    idx = [i for i in range(len(labels)) if truth[i] is not None]
    T = [truth[i] for i in idx]
    L = [labels[i] for i in idx]
    ari = adjusted_rand_score(T, as_int_labels(L))
    cov = [i for i in idx if labels[i] != -1]
    ari_cov = (adjusted_rand_score([truth[i] for i in cov], as_int_labels([labels[i] for i in cov]))
               if len(cov) > 1 else None)
    # purity: background/unparsed = one cluster
    by = collections.defaultdict(collections.Counter)
    for i in idx:
        by[labels[i]][truth[i]] += 1
    purity = sum(c.most_common(1)[0][1] for c in by.values()) / max(1, len(idx))
    nodes = {k: c for k, c in by.items() if k != -1}
    K = len(set(x for x in labels if x != -1))
    maj, wrong, n_ge5, wrong_ge5 = {}, 0, 0, 0
    for k, c in nodes.items():
        top, cnt = c.most_common(1)[0]
        tot = sum(c.values())
        maj[k] = top
        bad = (1 - cnt / tot) > 0.10
        wrong += bad
        if tot >= 5:
            n_ge5 += 1
            wrong_ge5 += bad
    table = []
    for k, c in sorted(nodes.items(), key=lambda kv: -sum(kv[1].values())):
        top, cnt = c.most_common(1)[0]
        tot = sum(c.values())
        table.append(dict(node=str(k), members=tot, majority=str(top),
                          share=round(cnt / tot, 3), n_types=len(c)))
    Kt = len(nodes)
    sizes = sorted((sum(c.values()) for c in nodes.values()), reverse=True)
    singleton_nodes = sum(1 for x in sizes if x == 1)
    largest_share = round(sizes[0] / max(1, len(idx)), 4) if sizes else 0.0
    pairs = Kt * (Kt - 1) // 2
    dup = sum(1 for a, b in itertools.combinations(maj, 2) if maj[a] == maj[b])
    per_type = collections.Counter(maj.values())
    return dict(
        n_records=len(labels), n_scored=len(idx),
        background_or_unparsed=sum(1 for i in idx if labels[i] == -1),
        ari=round(ari, 4), ari_covered_only=None if ari_cov is None else round(ari_cov, 4),
        purity=round(purity, 4), nodes=K, nodes_with_truth=Kt,
        truth_classes=len(set(T)),
        duplicate_pairs=dup, node_pairs=pairs,
        duplicate_rate=round(dup / pairs, 4) if pairs else 0.0,
        surplus_nodes=Kt - len(per_type),
        types_with_multiple_nodes=sum(1 for v in per_type.values() if v > 1),
        wrong_merge_nodes=wrong, wrong_merge_rate=round(wrong / Kt, 4) if Kt else 0.0,
        wrong_merge_nodes_ge5=wrong_ge5, nodes_ge5=n_ge5,
        wrong_merge_rate_ge5=round(wrong_ge5 / n_ge5, 4) if n_ge5 else None,
        singleton_nodes=singleton_nodes, largest_node_share=largest_share,
        node_table=table, majority_type_per_node=maj)


def pairwise_ari(label_sets):
    from sklearn.metrics import adjusted_rand_score
    vals = []
    for a, b in itertools.combinations(label_sets, 2):
        vals.append(adjusted_rand_score(as_int_labels(a), as_int_labels(b)))
    return [round(v, 4) for v in vals]


def name_variant_dups(names):
    """Node names that collapse to one string after lowercase, strip non-alnum, strip
    a trailing plural 's'/'es'. Groups of >1 are the same thing under two names."""
    def canon(s):
        s = norm_name(s)
        if s.endswith("es") and len(s) > 4:
            s = s[:-2]
        elif s.endswith("s") and len(s) > 3:
            s = s[:-1]
        return s
    groups = collections.defaultdict(list)
    for n in names:
        groups[canon(n)].append(n)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    pairs = sum(len(v) * (len(v) - 1) // 2 for v in dup.values())
    return dict(groups=len(dup), pairs=pairs, examples=list(dup.values())[:20])


_ST = {}


def name_synonym_probe(names, truth_maj=None):
    """Probe, not a verdict: cosine similarity of node-name embeddings
    (sentence-transformers all-MiniLM-L6-v2). Pairs at >=0.80 and >=0.90 are listed so a
    reader can judge whether two nodes name one kind of thing. Labelled as a probe."""
    try:
        if "m" not in _ST:
            from sentence_transformers import SentenceTransformer
            _ST["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        import numpy as np
        names = list(names)
        if len(names) < 2:
            return dict(pairs_ge_0_80=0, pairs_ge_0_90=0, examples=[])
        E = _ST["m"].encode(names, normalize_embeddings=True, batch_size=256,
                            show_progress_bar=False)
        S = E @ E.T
        iu = np.triu_indices(len(names), 1)
        sims = S[iu]
        order = np.argsort(-sims)
        ex = []
        for o in order[:40]:
            if sims[o] < 0.80:
                break
            a, b = names[iu[0][o]], names[iu[1][o]]
            row = dict(a=a, b=b, cos=round(float(sims[o]), 3))
            if truth_maj is not None:
                row["same_majority_truth"] = truth_maj.get(a) == truth_maj.get(b)
            ex.append(row)
        return dict(pairs_ge_0_80=int((sims >= 0.80).sum()),
                    pairs_ge_0_90=int((sims >= 0.90).sum()),
                    total_pairs=int(len(sims)), examples=ex)
    except Exception as e:                                 # noqa: BLE001
        return dict(error=repr(e))


def case_variant_cocluster(records, labels):
    """Raw-key case/spacing variant pairs: do the records that carry key a and the
    records that carry key b land in the same modal cluster? Symmetric for both methods."""
    labels = [(-1 if x is None else x) for x in labels]
    fam = collections.defaultdict(set)
    carriers = collections.defaultdict(list)
    for i, r in enumerate(records):
        for k in r:
            fam[norm_name(k)].add(k)
            carriers[k].append(i)
    rows, co, tot, both_cov = [], 0, 0, 0
    for nk, ks in fam.items():
        if len(ks) < 2 or not nk:
            continue
        for a, b in itertools.combinations(sorted(ks), 2):
            la = collections.Counter(labels[i] for i in carriers[a] if labels[i] != -1)
            lb = collections.Counter(labels[i] for i in carriers[b] if labels[i] != -1)
            tot += 1
            ma = la.most_common(1)[0][0] if la else None
            mb = lb.most_common(1)[0][0] if lb else None
            if ma is not None and mb is not None:
                both_cov += 1
                if ma == mb:
                    co += 1
            rows.append(dict(key_a=a, key_b=b, n_a=len(carriers[a]), n_b=len(carriers[b]),
                             modal_a=str(ma), modal_b=str(mb)))
    return dict(variant_pairs=tot, pairs_both_covered=both_cov, co_clustered=co,
                co_cluster_rate_given_both_covered=round(co / both_cov, 4) if both_cov else None,
                pairs=rows[:40])


def headline(out):
    """Per dataset, one row per metric with llm_sampled_mean, llm_greedy, escrow and the
    winner by the stated direction. Purely computed from the measured numbers."""
    H = {}
    for dsn, D in out["datasets"].items():
        M = D["methods"]
        sm = D.get("llm_sampled_mean", {})
        gr = M.get("llm_greedy_s0", {})
        es = M.get("escrow", {})
        rows = []

        def row(metric, hi_is_good, a, b, c, note=""):
            vals = {"llm_sampled_mean": a, "llm_greedy": b, "escrow": c}
            have = {k: v for k, v in vals.items() if isinstance(v, (int, float))}
            win = None
            if have:
                best = (max if hi_is_good else min)(have.values())
                win = sorted(k for k, v in have.items() if v == best)
                win = "tie" if len(win) > 1 else win[0]
            rows.append(dict(metric=metric, better="higher" if hi_is_good else "lower",
                             **vals, winner=win, note=note))
        row("ari", True, sm.get("ari"), gr.get("ari"), es.get("ari"))
        row("purity", True, sm.get("purity"), gr.get("purity"), es.get("purity"))
        row("nodes", None, sm.get("nodes"), gr.get("nodes"), es.get("nodes"),
            f"truth classes = {D['truth_classes']}")
        rows[-1]["better"] = "closer to truth classes"
        rows[-1]["winner"] = None
        row("duplicate_rate", False, sm.get("duplicate_rate"), gr.get("duplicate_rate"),
            es.get("duplicate_rate"))
        row("wrong_merge_rate", False, sm.get("wrong_merge_rate"), gr.get("wrong_merge_rate"),
            es.get("wrong_merge_rate"))
        det_llm = D.get("llm_determinism", {}).get("mean_ari")
        det_es = D.get("escrow_determinism", {}).get("mean_ari")
        row("determinism_mean_ari_across_3_runs", True, det_llm, None, det_es,
            "greedy run once by design; ESCROW identical output on 3 runs")
        row("tokens_total", False, sm.get("tokens_total"), gr.get("tokens_total"),
            es.get("tokens_total"))
        wall_es = es.get("wall_s")
        if isinstance(wall_es, list):
            wall_es = round(sum(wall_es) / len(wall_es), 2)
        row("wall_s", False, sm.get("wall_s"), gr.get("wall_s"), wall_es,
            "LLM: one A100-80GB, HF generate, batch 1; ESCROW: one CPU core, includes repair")
        row("singleton_nodes", False, sm.get("singleton_nodes"), gr.get("singleton_nodes"),
            es.get("singleton_nodes"))
        row("largest_node_share", False, sm.get("largest_node_share"),
            gr.get("largest_node_share"), es.get("largest_node_share"),
            "catch-all indicator; ESCROW background not counted as a node")
        vn = [M[k]["value_named_nodes"]["count"] for k in M if k.startswith("llm_sampled")]
        rows.append(dict(metric="value_named_nodes", better="lower",
                         llm_sampled_mean=round(sum(vn) / len(vn), 2) if vn else None,
                         llm_greedy=gr.get("value_named_nodes", {}).get("count"),
                         escrow=None, winner=None,
                         note="not applicable to ESCROW (nodes carry no names)"))
        if "background_or_unparsed" in es:
            rows.append(dict(metric="records_without_a_node", better="context",
                             llm_sampled_mean=None, llm_greedy=gr.get("background_or_unparsed"),
                             escrow=es.get("background_or_unparsed"), winner=None,
                             note="ESCROW leaves records in the background until evidence "
                                  "accrues; the LLM attaches every record it parses"))
        H[dsn] = rows
    return H


def value_named_nodes(records, assignments, node_names):
    """LLM nodes whose name is a VALUE (or key+value) taken from one of its own member
    records, e.g. node "Model_SYS_CARA7A67/PILVA" for a record with model=SYS_CARA7A67/PILVA.
    A value promoted to a node is the same kind of thing under a record-specific name,
    the paper's duplicate-node failure at its sharpest. Not applicable to ESCROW (no names)."""
    members = collections.defaultdict(list)
    for i, a in enumerate(assignments):
        if a is not None:
            members[a].append(i)
    hits, ex = 0, []
    for n in node_names:
        cn = norm_name(n)
        found = None
        for i in members.get(n, []):
            for k, v in records[i].items():
                nv, nk = norm_name(v), norm_name(k)
                if nv and (cn == nv or cn == nk + nv or cn.endswith(nv) and len(nv) >= 5):
                    found = (k, str(v)[:40])
                    break
            if found:
                break
        if found:
            hits += 1
            if len(ex) < 12:
                ex.append(dict(node=n, members=len(members.get(n, [])), key=found[0],
                               value=found[1]))
    return dict(count=hits, of_nodes=len(node_names),
                rate=round(hits / len(node_names), 4) if node_names else 0.0, examples=ex)


def cmd_score(_):
    out = dict(
        title="LLM graph formation vs ESCROW on public data",
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        question=("test the public data for any ill-formed graph formation. What is the "
                  "accuracy of that?"),
        protocol=dict(
            llm=("streaming create-or-attach by prompting: chunks of 20 records in arrival "
                 "order; prompt = current node list (name + one-line description) + records "
                 "as key=value lines; JSON answer per record = existing node name or new node "
                 "(name, description); node list carried across chunks; no cluster count; "
                 "node-name matching = exact, then case/punctuation-insensitive (counted as "
                 "loose_name_match); a referenced name that does not exist creates the node "
                 "(counted as implicit_new_node); records with no parsable answer = -1"),
            llm_arms=["sampled: temperature 0.7, top_p off, top_k off, repetition_penalty 1.0, "
                      "seeds 0,1,2", "greedy: do_sample False, once"],
            escrow=("EscrowGraph + BatchObjective(g).install(); repair() every 100 records; "
                    "final repair(full=True); record label = node with most members "
                    "containing it, else -1 (background); three repetitions"),
            metrics=dict(
                ari="adjusted Rand index of record labels vs truth; -1 records form one cluster",
                ari_covered_only="same, restricted to records with a node",
                purity="sum over clusters (incl. the -1 cluster) of the majority-truth count / n",
                duplicate_rate=("fraction of node pairs whose majority truth type is the same "
                                "(the same kind of thing under two nodes)"),
                surplus_nodes="nodes minus distinct majority types",
                wrong_merge_rate=("fraction of nodes whose minority truth share exceeds 10 "
                                  "percent; also restricted to nodes with >=5 members"),
                determinism="mean pairwise ARI between the three seeded runs' labelings",
                singleton_nodes="nodes holding exactly one record (node-per-record formation)",
                largest_node_share="fraction of scored records in the largest node (catch-all "
                                   "indicator; ESCROW's background is reported separately as "
                                   "background_or_unparsed)",
                value_named_nodes=("LLM only: nodes whose name is a value (or key+value) copied "
                                   "from a member record; a value promoted to a node"),
                tokens="prompt + completion tokens of the model's own tokenizer; ESCROW = 0"),
            escrow_receipt_fields=dict(
                justifying_key="the support key with the largest accrued evidence at mint",
                per_key_bits="accrued evidence (bits) per support key at mint",
                mint_bits=("the candidate's TOTAL accrued evidence over all keys at mint, "
                           "negative keys included; can sit below mint_price because the mint "
                           "test is on the positive support subset plus outside evidence "
                           "(engine.process)"),
                mint_price="price(t, |support|, n, K, e, A_n) paid at mint",
                mint_seed="(key, value-id) the candidate was opened on",
                members="node size at the end (records may sit in more than one node; the "
                        "record label is the largest node containing it)",
                absorbed_nodes="earlier mints merged into this node by repair, with their "
                               "justifying keys"),
        ),
        datasets={}, deviations=[])
    for dsn in ["wikipedia", "lazada"]:
        ds = load_data(dsn)
        truth = ds["truth"]
        D = dict(truth_label=ds["truth_label"], n_records=len(ds["records"]),
                 truth_classes=len(set(t for t in truth if t)),
                 raw_keys=len({k for r in ds["records"] for k in r}),
                 total_stream=ds["total_stream"], methods={})
        if dsn == "lazada":
            c = collections.Counter(t for t in truth if t)
            D["truth_granularity_note"] = (
                f"silver types on these {len(ds['records'])} records: {len(c)} distinct, "
                f"{sum(1 for v in c.values() if v == 1)} of them singletons, largest "
                f"{c.most_common(1)[0][1]}; ARI against this truth rewards one node per "
                f"listing and is a granularity mismatch for any method that forms shared "
                f"nodes; read duplicate/wrong-merge/determinism/tokens as the primary "
                f"comparison here. The AutoPKG KG has no level above Product Type (node "
                f"types: Product 36927, Product Type 16733, Attribute Key 15964, Brand "
                f"12985; edge types: has_attribute, has_key, has_value, of_type), so no "
                f"coarser silver truth exists to score either method against. The silver "
                f"types were derived by AutoPKG's LLM from titles and descriptions that "
                f"neither method sees here; both see only the specifications map.")
        # ---- LLM
        runs = {}
        for arm, seeds in (("sampled", [0, 1, 2]), ("greedy", [0])):
            for s in seeds:
                p = os.path.join(RUN_DIR, f"llm_{dsn}_{arm}_s{s}.json")
                if os.path.exists(p):
                    runs[(arm, s)] = json.load(open(p))
                else:
                    out["deviations"].append(f"missing run {dsn} llm {arm} seed {s}")
        for (arm, s), r in runs.items():
            m = cluster_metrics(r["assignments"], truth)
            m.update(tokens_in=r["tokens_in"], tokens_out=r["tokens_out"],
                     tokens_total=r["tokens_in"] + r["tokens_out"], wall_s=r["wall_s"],
                     counters=r["counters"], model=r["model"],
                     chunks_unfinished=sum(1 for c in r["chunks"] if not c["finished"]),
                     node_names=[n["name"] for n in r["nodes"]],
                     name_variant_duplicates=name_variant_dups([n["name"] for n in r["nodes"]]),
                     value_named_nodes=value_named_nodes(ds["records"], r["assignments"],
                                                         [n["name"] for n in r["nodes"]]))
            if dsn == "lazada":
                m["case_variant_cocluster"] = case_variant_cocluster(ds["records"], r["assignments"])
                per_k = m["tokens_total"] / (len(ds["records"]) / 1000)
                m["tokens_per_1000_records"] = round(per_k)
                m["tokens_extrapolated_21365_linear_lower_bound"] = round(
                    per_k * ds["total_stream"] / 1000)
            maj = m.pop("majority_type_per_node")
            m["name_synonym_probe"] = name_synonym_probe([n["name"] for n in r["nodes"]], maj)
            D["methods"][f"llm_{arm}_s{s}"] = m
        sampled = [runs[k]["assignments"] for k in sorted(runs) if k[0] == "sampled"]
        if len(sampled) >= 2:
            pa = pairwise_ari(sampled)
            D["llm_determinism"] = dict(pairwise_ari=pa, mean_ari=round(sum(pa) / len(pa), 4),
                                        node_counts=[len(runs[k]["nodes"]) for k in sorted(runs)
                                                     if k[0] == "sampled"])
            # node-name set overlap across seeds
            sets = [set(n["name"] for n in runs[k]["nodes"]) for k in sorted(runs) if k[0] == "sampled"]
            jac = [round(len(a & b) / len(a | b), 4) for a, b in itertools.combinations(sets, 2)]
            D["llm_determinism"]["node_name_jaccard"] = jac
        if "sampled" in {k[0] for k in runs}:
            ms = [D["methods"][f"llm_sampled_s{s}"] for s in [0, 1, 2] if f"llm_sampled_s{s}" in D["methods"]]
            D["llm_sampled_mean"] = {k: round(sum(m[k] for m in ms) / len(ms), 4)
                                     for k in ["ari", "purity", "nodes", "duplicate_rate",
                                               "surplus_nodes", "wrong_merge_rate",
                                               "singleton_nodes", "largest_node_share",
                                               "tokens_total", "wall_s"]}
        # ---- ESCROW
        reps = []
        for rep in range(3):
            p = os.path.join(RUN_DIR, f"escrow_{dsn}_r{rep}.json")
            if os.path.exists(p):
                reps.append(json.load(open(p)))
            else:
                out["deviations"].append(f"missing run {dsn} escrow rep {rep}")
        if reps:
            r = reps[0]
            m = cluster_metrics(r["assignments"], truth)
            m.update(tokens_in=0, tokens_out=0, tokens_total=0,
                     wall_s=[x["wall_s"] for x in reps], repair_s=[x["repair_s"] for x in reps],
                     mints=r["mints"], merges=r["merges"],
                     receipts=r["receipts"])
            if dsn == "lazada":
                m["case_variant_cocluster"] = case_variant_cocluster(ds["records"], r["assignments"])
            # receipts annotated with truth
            tn = ds.get("truth_names", {})
            for rc in m["receipts"]:
                mem = r["node_members"][str(rc["node"])] if str(rc["node"]) in r["node_members"] \
                    else r["node_members"][rc["node"]]
                c = collections.Counter(tn.get(truth[i], truth[i]) for i in mem if truth[i])
                rc["truth_top3"] = c.most_common(3)
            maj = m.pop("majority_type_per_node")
            D["methods"]["escrow"] = m
            identical = all(x["assignments"] == r["assignments"] for x in reps[1:])
            identical_members = all(x["node_members"] == r["node_members"] for x in reps[1:])
            pa = pairwise_ari([x["assignments"] for x in reps]) if len(reps) >= 2 else []
            D["escrow_determinism"] = dict(
                runs=len(reps), identical_assignments=identical,
                identical_node_members=identical_members, pairwise_ari=pa,
                mean_ari=round(sum(pa) / len(pa), 4) if pa else None,
                K_per_run=[x["K"] for x in reps])
        out["datasets"][dsn] = D
    out["headline"] = headline(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(stamped(out), open(FINAL, "w"), indent=1)
    print(f"[saved] {FINAL}")
    slim = json.loads(json.dumps(out))
    for dsn in slim["datasets"]:
        for k, m in slim["datasets"][dsn]["methods"].items():
            m.pop("receipts", None)
            m.pop("node_names", None)
            if "node_table" in m:
                m["node_table"] = m["node_table"][:12]
            if "name_synonym_probe" in m and "examples" in m["name_synonym_probe"]:
                m["name_synonym_probe"]["examples"] = m["name_synonym_probe"]["examples"][:12]
            if "case_variant_cocluster" in m:
                m["case_variant_cocluster"].pop("pairs", None)
    print(json.dumps(slim, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    p = sub.add_parser("llm")
    p.add_argument("--api", default="", help="OpenAI-compatible base URL; served model instead of a local checkpoint")
    p.add_argument("--api-model", default="", help="model id at that endpoint")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--tag", default="", help="suffix for the run files, so a second model does not overwrite the first")
    p.add_argument("--jobs", required=True, help="dataset:arm:seed,...")
    p.add_argument("--max-chunks", type=int, default=0, help="smoke test cap")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("escrow")
    p.add_argument("--datasets", default="wikipedia,lazada")
    sub.add_parser("score")
    args = ap.parse_args()
    {"data": cmd_data, "llm": cmd_llm, "escrow": cmd_escrow, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
