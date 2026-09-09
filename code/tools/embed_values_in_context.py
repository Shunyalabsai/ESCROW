"""Embed a value in the context of the record it sits in, rather than on its own.

WHY THIS IS THE PART WORTH BUILDING. `code/tools/embed_values.py` embeds each distinct value string
alone, so `large` gets one vector whether the record is a shirt or a monitor, and `size` gets one
meaning whether it runs over {S, M, L} or over inches. That flat vector is what E56 measured, and it
loses: 11 bits on noise, which is correct and required, and 1,292 bits on the encyclopedia stream,
which is not.

The flat similarity kernel already IS single-head self-attention over the embedding space, verified
by ranking candidate values two ways and getting the same order, so re-deriving it with a softmax
changes nothing. What is genuinely missing is not the mechanism but the input: attention over the
OTHER key-value pairs of the same record, so a value's vector knows the structure it sits in.

HOW. Each record is serialised as its key-value pairs and passed through the encoder once. The
transformer's own self-attention mixes the pairs, and the vector for a given value is the mean of
its own token positions in that contextualised sequence. So `large` in
`item=shirt fabric=cotton size=large` and `large` in `panel=OLED refresh=120Hz size=large` come out
as different vectors, which is the whole point.

WHAT THIS COSTS, STATED PLAINLY. The flat cache is one vector per distinct value string, a few
thousand for these streams. This is one forward pass per record and one vector per value occurrence,
so it is larger and it has to be recomputed for a new stream rather than looked up. It also makes the
embedding a heavier declared choice. Whether it is worth that is what E61 measures, and the answer is
allowed to be no.

    python3 code/tools/embed_values_in_context.py --stream wikipedia \
        --out results/embeddings/wikipedia.context.json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def noise_stream(n=200, keys=4, d=10, seed=0):
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def wikipedia_stream():
    import llm_graph_formation as L
    return L.build_wikipedia()["records"]


def keymerge_fixture_stream():
    from e57_semantics_in_the_key_merge import build_stream
    return build_stream(0)


STREAMS = {"noise": noise_stream, "wikipedia": wikipedia_stream,
           "keymerge_fixture": keymerge_fixture_stream}


def serialise(rec):
    """One record as a flat string, and the character span each value occupies inside it."""
    parts, spans = [], {}
    pos = 0
    for k, v in rec.items():
        s = f"{k}={v}"
        if parts:
            pos += 1                                  # the separating space
        spans.setdefault(str(v), []).append((pos + len(str(k)) + 1, pos + len(s)))
        parts.append(s)
        pos += len(s)
    return " ".join(parts), spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, choices=sorted(STREAMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-records", type=int, default=5000)
    a = ap.parse_args()

    recs = STREAMS[a.stream]()[:a.max_records]
    print(f"{len(recs)} records, embedding each value in the context of its own record", flush=True)

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModel.from_pretrained(a.model)
    mdl.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev)

    # a value's vector is the mean of its contextual vectors over every record it appears in, so the
    # cache stays one vector per distinct value string and the consuming code is unchanged
    acc = collections.defaultdict(lambda: None)
    cnt = collections.Counter()
    per_occurrence = collections.defaultdict(list)
    B = 64
    with torch.no_grad():
        for i in range(0, len(recs), B):
            batch = recs[i:i + B]
            texts, spanss = zip(*(serialise(r) for r in batch))
            enc = tok(list(texts), padding=True, truncation=True, max_length=256,
                      return_tensors="pt", return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            out = mdl(**{k: v.to(dev) for k, v in enc.items()}).last_hidden_state.cpu()
            for b, spans in enumerate(spanss):
                offs = offsets[b].tolist()
                for value, ranges in spans.items():
                    picks = [t for t, (s0, s1) in enumerate(offs)
                             if s1 > s0 and any(s0 >= lo and s1 <= hi for lo, hi in ranges)]
                    if not picks:
                        continue
                    vec = out[b, picks].mean(0)
                    vec = vec / (vec.norm() + 1e-9)
                    acc[value] = vec if acc[value] is None else acc[value] + vec
                    cnt[value] += 1
                    if len(per_occurrence[value]) < 40:
                        per_occurrence[value].append(vec.clone())
            if (i // B) % 10 == 0:
                print(f"    {min(i + B, len(recs))}/{len(recs)} records", flush=True)

    # Dispersion first, because it decides whether averaging is defensible at all. If a value's
    # occurrences all point the same way, the mean loses nothing and a one-vector cache is honest.
    # If they spread, the mean is destroying exactly the signal contextualising created, and the
    # cache format is the bug rather than the embedding.
    import torch as _t
    dispersion = {}
    for v, vecs in per_occurrence.items():
        if len(vecs) < 2:
            continue
        M = _t.stack(vecs)
        M = M / (M.norm(dim=1, keepdim=True) + 1e-9)
        sims = (M @ M.T)
        iu = _t.triu_indices(len(vecs), len(vecs), offset=1)
        pw = sims[iu[0], iu[1]]
        dispersion[v] = {"occurrences": len(vecs),
                         "mean_self_similarity": round(float(pw.mean()), 4),
                         "min_self_similarity": round(float(pw.min()), 4)}

    vectors = {}
    for v, s in acc.items():
        m = s / cnt[v]
        m = m / (m.norm() + 1e-9)
        vectors[v] = [round(float(x), 5) for x in m.tolist()]

    out = {"model": a.model, "stream": a.stream, "contextual": True,
           "date": datetime.date.today().isoformat(),
           "records_seen": len(recs), "n_values": len(vectors),
           "dim": len(next(iter(vectors.values()))) if vectors else 0,
           "occurrences_per_value": {v: cnt[v] for v in list(cnt)[:50]},
           "dispersion": dispersion,
           "vectors": vectors}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"written {a.out}  ({len(vectors)} contextual vectors from {len(recs)} records)")


if __name__ == "__main__":
    main()
