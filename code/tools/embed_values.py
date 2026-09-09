"""Embed the value strings of a stream once, to a JSON cache the engine can read.

The insertion process stays stdlib-only. Nothing in `escrow/` imports a neural library, so the
embedding is computed here, on a machine that has the model, and shipped as a file of vectors. That
keeps the two apart for a reason beyond packaging: the embedding is a declared modelling choice, and
a choice that lives in its own artefact with its own provenance is one a reader can check, swap, or
refuse.

    python3 code/tools/embed_values.py --stream wikipedia --out results/embeddings/wikipedia.json
    python3 code/tools/embed_values.py --stream noise     --out results/embeddings/noise.json
    python3 code/tools/embed_values.py --values a.txt     --out results/embeddings/custom.json

The default model is small, multilingual and already on the box. It is named in the output file
alongside the date, because a number produced under one embedding is not a number produced under
another.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def noise_values(n=200, keys=4, d=10, seed=0):
    rng = random.Random(seed)
    recs = [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]
    return sorted({v for r in recs for v in r.values()}), recs


def wikipedia_values():
    import llm_graph_formation as L
    ds = L.build_wikipedia()
    return sorted({str(v) for r in ds["records"] for v in r.values()}), ds["records"]


def lazada_values(limit=2000):
    from lazada_c2_rawkey import load_records
    recs, _, _ = load_records()
    recs = recs[:limit]
    return sorted({str(v) for r in recs for v in r.values()}), recs


STREAMS = {"noise": noise_values, "wikipedia": wikipedia_values, "lazada": lazada_values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=sorted(STREAMS))
    ap.add_argument("--values", help="a file of value strings, one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-values", type=int, default=20000)
    a = ap.parse_args()

    if a.values:
        with open(a.values, encoding="utf-8") as f:
            values = sorted({line.strip() for line in f if line.strip()})
    elif a.stream:
        values, _ = STREAMS[a.stream]()
    else:
        ap.error("give --stream or --values")

    if len(values) > a.max_values:
        print(f"{len(values)} values, truncating to the {a.max_values} most frequent is NOT done "
              f"here; raise --max-values or narrow the stream", file=sys.stderr)
        sys.exit(2)

    print(f"{len(values)} distinct value strings, embedding with {a.model}", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(a.model)
        vecs = model.encode(values, batch_size=256, show_progress_bar=True,
                            normalize_embeddings=True)
        vecs = [list(map(float, v)) for v in vecs]
    except ImportError:
        # sentence-transformers is a thin wrapper and is not installed everywhere. Mean pooling over
        # the last hidden state with the attention mask, then L2 normalisation, is exactly what it
        # does for this family of models, so the vectors are the same either way.
        import torch
        from transformers import AutoModel, AutoTokenizer
        print("  sentence-transformers absent, pooling with transformers directly", flush=True)
        tok = AutoTokenizer.from_pretrained(a.model)
        mdl = AutoModel.from_pretrained(a.model)
        mdl.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev)
        vecs = []
        B = 256
        with torch.no_grad():
            for i in range(0, len(values), B):
                batch = values[i:i + B]
                enc = tok(batch, padding=True, truncation=True, max_length=64,
                          return_tensors="pt").to(dev)
                out = mdl(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vecs.extend(pooled.cpu().tolist())
                if (i // B) % 10 == 0:
                    print(f"    {min(i + B, len(values))}/{len(values)}", flush=True)

    import datetime
    out = {"model": a.model,
           "stream": a.stream or a.values,
           "date": datetime.date.today().isoformat(),
           "n_values": len(values),
           "dim": int(len(vecs[0])),
           "vectors": {v: [round(float(x), 5) for x in vec] for v, vec in zip(values, vecs)}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"written {a.out}  ({len(values)} vectors of {out['dim']} dimensions)")


if __name__ == "__main__":
    main()
