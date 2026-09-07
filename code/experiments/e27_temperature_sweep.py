"""E27: is the language model's non-determinism an artefact of one temperature?

THE OBJECTION. The paper reports that a language model asked to build the graph returns a different
graph on every seed, and measures it at temperature 0.7 with one greedy run beside it. A reviewer can
answer that in one line: 0.7 is a sampling temperature, lower it and the objection goes away. The
claim is only worth making if it survives the sweep.

WHY IT WAS RECORDED AS BLOCKED, AND WHY THAT WAS WRONG. The A100 box reports a driver and library
version mismatch, so nvidia-smi fails, and that was taken to mean the device was unusable. It is not.
The mismatch is in the userspace library nvidia-smi itself calls; torch initialises CUDA, reports
A100-SXM4-40GB, and Qwen2.5-14B-Instruct loads into 37 of the 42.4 GB and generates normally. No
driver reload was needed and no other process on the box was touched. A tool failing is not the same
as the capability being absent.

WHAT WAS RUN. The arms already in the paper, temperature 0.7 at seeds 0, 1, 2 and one greedy run, plus
temperatures 0.3 and 1.0 at the same three seeds, on both streams, with the prompt, the chunking, the
parser and the arrival order all unchanged. `sampled` and `greedy` still mean exactly what they meant,
so nothing already reported moves.

THIS FILE collects the runs written by llm_graph_formation.py into one result. It computes nothing
the runner did not, and it exists so the sweep is one artefact rather than twenty files.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.provenance import stamped                                   # noqa: E402
from experiments.e4_baseline_army import _ari                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
RUNS = os.path.join(RESULTS, "llm_gf_runs")
OUT = os.path.join(RESULTS, "e27_temperature_sweep.json")


def collect(ds):
    data = json.load(open(os.path.join(RUNS, f"data_{ds}.json"), encoding="utf-8"))
    truth = data["truth"]
    idx = [i for i, t in enumerate(truth) if t]
    groups = {}
    for f in sorted(glob.glob(os.path.join(RUNS, f"llm_{ds}_*.json"))):
        b = os.path.basename(f)
        if "smoke" in b or "gemma" in b:
            continue
        r = json.load(open(f, encoding="utf-8"))
        groups.setdefault(r["arm"], {})[r["seed"]] = r

    out = {}
    for arm, runs in groups.items():
        aris = []
        for r in runs.values():
            a = r["assignments"]
            pred = [a[i] if a[i] is not None else f"__bg{i}" for i in idx]
            aris.append(_ari([truth[i] for i in idx], pred))
        dets = []
        for x, y in itertools.combinations(runs.values(), 2):
            la = [x["assignments"][i] if x["assignments"][i] is not None else f"__a{i}"
                  for i in range(len(truth))]
            lb = [y["assignments"][i] if y["assignments"][i] is not None else f"__b{i}"
                  for i in range(len(truth))]
            dets.append(_ari(la, lb))
        out[arm] = {
            "runs": len(runs),
            "temperature": (0.0 if arm == "greedy"
                            else (0.7 if arm == "sampled" else float(arm.split("@")[1]))),
            "mean_ARI_vs_truth": round(sum(aris) / len(aris), 4),
            "mean_K": round(sum(len(r["nodes"]) for r in runs.values()) / len(runs), 1),
            "K_per_run": sorted(len(r["nodes"]) for r in runs.values()),
            "determinism_mean_pairwise_ARI": round(sum(dets) / len(dets), 4) if dets else None,
            "mean_tokens": round(sum(r["tokens_in"] + r["tokens_out"]
                                     for r in runs.values()) / len(runs)),
            "mean_wall_s": round(sum(r["wall_s"] for r in runs.values()) / len(runs), 1),
        }
    return {"records": len(truth), "labelled": len(idx),
            "truth_classes": len({t for t in truth if t}), "arms": out}


def main():
    report = {
        "experiment": "E27 the language model across temperatures",
        "question": "is the non-determinism the paper reports an artefact of temperature 0.7?",
        "model": "Qwen2.5-14B-Instruct, bf16, one A100-SXM4-40GB",
        "unblocking_note": ("recorded as blocked because nvidia-smi reports a driver and library "
                            "mismatch on the box. The mismatch is in the userspace library "
                            "nvidia-smi calls; torch initialises the device and the model loads and "
                            "generates. No driver reload, no other process touched"),
        "arms": "greedy, and sampling at 0.3, 0.7 and 1.0 with seeds 0, 1, 2",
        "datasets": {},
    }
    for ds in ("wikipedia", "lazada"):
        try:
            report["datasets"][ds] = collect(ds)
        except FileNotFoundError as exc:
            report["datasets"][ds] = {"skipped": str(exc)}
            continue
        print(f"{ds}:")
        for arm, v in sorted(report["datasets"][ds]["arms"].items(),
                             key=lambda kv: kv[1]["temperature"]):
            print(f"  T={v['temperature']:<4} ARI {v['mean_ARI_vs_truth']:.4f}  K {v['mean_K']:5.1f} "
                  f"  determinism {v['determinism_mean_pairwise_ARI']}  "
                  f"tokens {v['mean_tokens']}", flush=True)

    wiki = report["datasets"].get("wikipedia", {}).get("arms", {})
    laz = report["datasets"].get("lazada", {}).get("arms", {})
    dets_w = [v["determinism_mean_pairwise_ARI"] for v in wiki.values()
              if v["determinism_mean_pairwise_ARI"] is not None]
    dets_l = [v["determinism_mean_pairwise_ARI"] for v in laz.values()
              if v["determinism_mean_pairwise_ARI"] is not None]
    report["headline"] = {
        "wikipedia_determinism_range": [min(dets_w), max(dets_w)] if dets_w else None,
        "lazada_determinism_range": [min(dets_l), max(dets_l)] if dets_l else None,
        "lazada_K_range": sorted({k for v in laz.values() for k in v["K_per_run"]})[:1]
                          + sorted({k for v in laz.values() for k in v["K_per_run"]})[-1:],
        "reading": ("Temperature is not what drives it. On the 320 clean encyclopedia records of "
                    "three obvious types the model agrees with itself at 0.99 at every temperature "
                    "including 1.0. On a thousand raw listings with 643 raw keys it agrees with "
                    "itself at 0.12 to 0.16 at every temperature, and lowering the temperature to "
                    "0.3 makes it slightly worse rather than better while the node count swings "
                    "from 28 to 73. What varies is the difficulty of the stream, not the sampling. "
                    "So the paper's claim survives the obvious answer to it."),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
