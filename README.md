# ESCROW

Latent graphs from record streams by evidence accrual at a computed price.

A stream of records arrives. Each record is a set of attribute key and value pairs. ESCROW decides,
per record and in one pass, which latent nodes the record attaches to, whether an attribute key is a
new node or an existing node under another name, and when the stream has earned a brand new node.
Every decision is made by one description length. No quantity in the rule is calibrated on data.

The idea in three sentences. A new node has a price in bits, and the price is computed from the
stream, not chosen. No valid code lets a fresh node pay its price on its own first record, so
evidence accrues in escrow against a candidate node that does not yet exist. The node is created at
the first moment its accumulated evidence covers its price, and never before.

## Layout

```
escrow/codes.py       codelength primitives: KT codes, the membership column, the mint price
escrow/engine.py      the insertion process: attach, escrow, mint
escrow/batch.py       the batch objective and the repair operators: merge, reassign, delete
escrow/fastrepair.py  a faster reassign pass with the same result as batch.py
tests/                the verification suite; the Kraft identities are the foundation
experiments/          the mechanism experiments, the demos and the benchmark runners
```

The engine (`escrow/`) and the tests are pure Python, standard library only. `numpy` is used when
it is installed and is not required. The benchmark runners under `experiments/` have their own
needs, listed below.

## Run

```bash
python3 tests/test_codes.py     # code-level gates, including three Kraft identities
python3 tests/test_engine.py    # engine gates: positive control, null control, degenerate data

python3 experiments/e7_immediate_vs_deferred.py   # why immediate minting cannot work
python3 experiments/e8_false_mint_null.py         # the false-mint guarantee, measured
python3 experiments/e9_support_curve.py           # the release fires exactly at the price
python3 experiments/e13_creation_bias.py          # creation bias, measured flat at the truth
python3 experiments/wikipedia_demo.py             # raw Wikipedia infoboxes to typed structure
```

The Wikipedia demo fetches a few hundred pages from the public API on first run and caches them.
The catalogue experiment expects a CSV path in the environment variable `ESCROW_CATALOG_CSV`.

### Benchmark runners

These scripts compare ESCROW with tuned baselines and with a language model on public data. They
need extra packages and, for some, a GPU.

```
experiments/e4_baseline_army.py       tuned baselines on synthetic streams and Wikipedia: numpy, scikit-learn, sentence-transformers
experiments/e8_tree_arms.py           VFDT and EFDT trees on the pure-noise streams: river
experiments/mb20k_benchmark.py        MusicBrainz 20K: numpy, scikit-learn
experiments/lazada_c2_rawkey.py       raw marketplace listings (AutoPKG's Lazada release): scikit-learn
experiments/e10_reverb45k.py          ReVerb45K under CESI's metrics code: numpy, scikit-learn, and a CESI checkout
experiments/llm_graph_formation.py    a language model building the same graph: torch, transformers, scikit-learn, one GPU
experiments/make_figures.py           the two paper figures from the results files: matplotlib
```

The runners read data from and write results to the directory named by the environment variable
`ESCROW_ROOT`. The default is the directory that contains this repository. Inside it they expect
`data/`, `baselines/` and `results/`. Each script's docstring names the files it needs and where
they come from. The datasets are public; download them from their own sources.

## Status

Research code accompanying a paper under preparation. The engine covers categorical facets. Numeric
and free text facets, and the large benchmark comparisons, are in progress. Interfaces may change.

## License

PolyForm Noncommercial 1.0.0. You can use, modify and share this software for research, education
and any other noncommercial purpose. Commercial use of any kind requires a commercial license from
Shunya Labs. See `LICENSE.md`.

## Citation

A paper is in preparation. A citation entry will appear here when it is public.
