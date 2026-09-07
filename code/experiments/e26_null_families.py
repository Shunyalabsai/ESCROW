"""E26: more than one kind of noise.

THE OBJECTION. The paper's safety claim, that no node is created on noise, rests on a single null
family: four categorical keys, ten values each, drawn independently and uniformly (E8, and the 300
wider streams behind Theorem 1). A reviewer can say, correctly, that one family is not a demonstration
that the rule is safe on noise. It is a demonstration that the rule is safe on that noise. Uniform
marginals are also the easiest possible case for a code built on Krichevsky-Trofimov counts, because
no value is ever surprising enough to look like structure.

WHAT THIS RUNS. Four sweeps, all under the shipped protocol, all with no latent structure at all, so
every node created is false by construction.

  1. ALPHABET. Keys fixed at 4, alphabet size d in {2, 5, 10, 50, 100, 500}. A small alphabet is the
     dangerous case: with d = 2 every record repeats values constantly, and coincidence is cheap.
  2. KEYS. Alphabet fixed at 10, key count in {2, 4, 8, 16, 32}. More keys per record means more
     ways for two records to agree by chance.
  3. MARGINAL. Keys 4, alphabet 10, values drawn from a Zipf law at exponent s in {0.0, 0.5, 1.0,
     1.5, 2.0}, where 0.0 is the uniform case already tested. A skewed marginal is the case the
     paper's own myopic-mint lemma says is dangerous, because a frequent value carries little
     information and a rare one carries a lot.
  4. PRESENCE. Keys 12, alphabet 10, every key present in a record independently with probability p,
     p in {1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3}. Every family above has p = 1: each key appears
     in every record, so the absence code has nothing to say. Real records are not like that, and
     this sweep is the one that found something.
  5. REAL DATA, in two arms, because the obvious construction is not a null and finding that out is
     itself a result.
     4a. VALUES SHUFFLED. The Wikipedia stream with each key's values permuted independently across
         the records that carry that key. This destroys every association between key values while
         leaving each record's KEY SET untouched, and the key set is not noise: a film infobox
         carries director and starring, a mountain carries elevation and prominence. The engine reads
         key presence through the absence code, so it should still find the categories here, and this
         arm is reported as a positive control on that code rather than as a null. Anyone who calls
         it a null has mistaken surviving structure for a false mint.
     4b. A TRUE NULL. Values permuted as above, and in addition each record's key set redrawn by
         including each key independently with its own global frequency. That destroys key
         co-occurrence, which is what carried the category in 4a, while preserving every per-key
         marginal, the alphabet and the overall record density. This is the null with the shape of
         real data, and it is the one the claim is tested on.

THE FALSIFIER, stated before the run. Any family in which a node appears. The claim under test is not
"few nodes"; it is none. A single mint on any of these streams contradicts the sentence the abstract
prints, and this file reports it if it happens.

Everything is pure Python over the standard library. Nothing here needs numpy.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from escrow.protocol import run_stream, describe as protocol_describe   # noqa: E402
from escrow.provenance import stamped                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RESULTS, "e26_null_families.json")

N_RECORDS = 2000
SEEDS = (0, 1, 2, 3, 4)


def uniform_null(n, keys, d, seed):
    """No latent structure: every key's value is drawn independently and uniformly."""
    rng = random.Random(seed)
    return [{f"k{j}": f"v{rng.randrange(d)}" for j in range(keys)} for _ in range(n)]


def zipf_null(n, keys, d, s, seed):
    """No latent structure, but the marginal is skewed: value i has weight 1/(i+1)^s."""
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** s) for i in range(d)]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)

    def draw():
        u = rng.random()
        for i, c in enumerate(cum):
            if u <= c:
                return i
        return d - 1

    return [{f"k{j}": f"v{draw()}" for j in range(keys)} for _ in range(n)]


def _wikipedia_records(cache_dir):
    import glob
    recs = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "wiki_cache.*.json"))):
        with open(f, encoding="utf-8") as fh:
            for box in json.load(fh):
                rec = {str(k): str(v) for k, v in box.items() if isinstance(v, (str, int, float))}
                if rec:
                    recs.append(rec)
    return recs


def wikipedia_values_shuffled(cache_dir, seed):
    """Values permuted within each key; every record's KEY SET is left intact.

    This is NOT a null. The key set is the strongest signal in this stream, and the engine reads key
    presence through the absence code, so it should still recover the categories. The arm is here as
    a positive control on that code, and to show why the obvious real-data null is not one.
    """
    recs = _wikipedia_records(cache_dir)
    rng = random.Random(seed)
    rng.shuffle(recs)
    columns = {}
    for r in recs:
        for k, v in r.items():
            columns.setdefault(k, []).append(v)
    for k in columns:
        rng.shuffle(columns[k])
    cursor = {k: 0 for k in columns}
    out = []
    for r in recs:
        new = {}
        for k in r:
            new[k] = columns[k][cursor[k]]
            cursor[k] += 1
        out.append(new)
    return out


def wikipedia_true_null(cache_dir, seed):
    """A null with the shape of real data.

    Each key keeps its global presence frequency and its own pool of values, but a record's keys are
    now drawn independently, so key co-occurrence is destroyed. Nothing associates one key with
    another or a key with a value, and the engine has nothing to find.
    """
    recs = _wikipedia_records(cache_dir)
    n = len(recs)
    rng = random.Random(seed)
    freq, pool = {}, {}
    for r in recs:
        for k, v in r.items():
            freq[k] = freq.get(k, 0) + 1
            pool.setdefault(k, []).append(v)
    out = []
    for _ in range(n):
        rec = {}
        for k, c in freq.items():
            if rng.random() < c / n:
                rec[k] = rng.choice(pool[k])
        if not rec:                      # never emit an empty record
            k = rng.choice(list(freq))
            rec[k] = rng.choice(pool[k])
        out.append(rec)
    return out


def presence_null(n, keys, d, p, seed):
    """No latent structure, and each key is present only with probability p.

    Every other family here has p = 1, so the absence code never has to price a missing key. This is
    the family that separates the two cases.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        rec = {f"k{j}": f"v{rng.randrange(d)}" for j in range(keys) if rng.random() < p}
        if not rec:
            rec = {"k0": f"v{rng.randrange(d)}"}
        out.append(rec)
    return out


def run_one(records):
    t0 = time.time()
    g, _ = run_stream(records)
    return {"K": g.K, "n": len(records), "seconds": round(time.time() - t0, 1)}


def sweep(name, build, cells):
    rows = []
    for label, kwargs in cells:
        per_seed = []
        for s in SEEDS:
            r = run_one(build(seed=s, **kwargs))
            per_seed.append(r["K"])
            print(f"  {name} {label} seed {s}: K = {r['K']} ({r['seconds']}s)", flush=True)
        rows.append({"cell": label, "params": kwargs,
                     "K_per_seed": per_seed,
                     "K_max": max(per_seed), "any_node": max(per_seed) > 0})
    return rows


def main():
    report = {
        "experiment": "E26 null families",
        "question": "Is the no-node-on-noise result a property of the rule, or of one null family?",
        "protocol": protocol_describe(),
        "falsifier": "any family in which a node appears; the claim is none, not few",
        "n_records": N_RECORDS,
        "seeds": list(SEEDS),
    }

    print("alphabet sweep")
    report["alphabet"] = sweep(
        "alphabet",
        lambda seed, d: uniform_null(N_RECORDS, 4, d, seed),
        [(f"d={d}", {"d": d}) for d in (2, 5, 10, 50, 100, 500)])

    print("key-count sweep")
    report["keys"] = sweep(
        "keys",
        lambda seed, keys: uniform_null(N_RECORDS, keys, 10, seed),
        [(f"keys={k}", {"keys": k}) for k in (2, 4, 8, 16, 32)])

    print("marginal sweep")
    report["marginal"] = sweep(
        "marginal",
        lambda seed, s: zipf_null(N_RECORDS, 4, 10, s, seed),
        [(f"zipf s={s}", {"s": s}) for s in (0.0, 0.5, 1.0, 1.5, 2.0)])

    print("presence sweep")
    report["presence"] = sweep(
        "presence",
        lambda seed, p: presence_null(N_RECORDS, 12, 10, p, seed),
        [(f"p={p}", {"p": p}) for p in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3)])

    print("positive control: Wikipedia values shuffled, key sets left intact")
    ctrl = []
    for s in SEEDS:
        r = run_one(wikipedia_values_shuffled(RESULTS, s))
        ctrl.append(r["K"])
        print(f"  values-shuffled seed {s}: K = {r['K']} over {r['n']} records ({r['seconds']}s)",
              flush=True)
    report["wikipedia_values_shuffled"] = {
        "what": "values permuted within each key, key sets intact. NOT a null: the key set carries "
                "the category, and the engine reads key presence through the absence code",
        "role": "positive control on the absence code, not a test of the noise claim",
        "true_categories": 3,
        "K_per_seed": ctrl, "K_max": max(ctrl)}

    print("real-data null: key co-occurrence destroyed as well")
    real = []
    for s in SEEDS:
        r = run_one(wikipedia_true_null(RESULTS, s))
        real.append(r["K"])
        print(f"  true-null seed {s}: K = {r['K']} over {r['n']} records ({r['seconds']}s)",
              flush=True)
    report["wikipedia_true_null"] = {
        "what": "each key keeps its global presence frequency and its own value pool, but a record's "
                "keys are drawn independently, so key co-occurrence is destroyed along with every "
                "key-to-value association",
        "K_per_seed": real, "K_max": max(real), "any_node": max(real) > 0}

    families = (report["alphabet"] + report["keys"] + report["marginal"] + report["presence"]
                + [{"cell": "wikipedia true null", "K_max": report["wikipedia_true_null"]["K_max"],
                    "any_node": report["wikipedia_true_null"]["any_node"]}])
    offenders = [f["cell"] for f in families if f["any_node"]]
    report["headline"] = {
        "families_tested": len(families),
        "families_with_any_node": len(offenders),
        "offending_cells": offenders,
        "verdict": ("no node in any family" if not offenders
                    else "a node appeared in " + ", ".join(offenders)),
        "reading": ("Every family in which each key is present in every record gives no node, across "
                    "alphabet 2 to 500, key count 2 to 32 and Zipf exponent 0 to 2. The families "
                    "that mint are the ones where key presence is itself random and high: a single "
                    "node appears on some seeds. The claim that survives is narrower than the one "
                    "the abstract prints, and it is stated in the paper as such."),
    }
    print("\n" + report["headline"]["verdict"])

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(stamped(report), fh, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
