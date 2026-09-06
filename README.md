# ESCROW

**When does a record stream earn a new node?**

A stream of records arrives. Each record is a set of attribute key and value pairs. ESCROW decides,
per record and in one pass, which existing nodes the record joins and when the stream has earned a
brand new one. Every decision is made by one description length, and no quantity in the rule is
calibrated on data.

The idea in three sentences. A new node has a price in bits, and that price is computed from the
stream rather than chosen. No fresh node can pay its price on its own first record under any valid
code, so evidence accrues in escrow against a node that does not yet exist. The node is created at
the first moment its account covers its price, and never before, and it carries a receipt in bits
naming the key that paid for it.

The paper is in this repository: [paper/escrow.pdf](paper/escrow.pdf).

## Why this exists

Almost every system that builds a graph from a stream answers the same question with a number
somebody chose. A distance cut-off in streaming clustering. The penalty that the small-variance
limit leaves behind in DP-means and BP-means. The sampling temperature of a language-model
pipeline. XGBoost adds a leaf only when the gain beats a per-leaf charge whose default is zero and
for which its paper reports no value.

The constant is not an accident of engineering. Before the small-variance limit is taken, the
new-node penalty is a codelength; the limit erases every part of it that could have been computed
and keeps only the part a person inserted. We do not take the limit. We read the price off a code
that predicts each record before it sees it.

## What is here

```
escrow/codes.py        codelength primitives: KT codes, the membership column, the mint price
escrow/engine.py       the insertion process: parse, retrieve, attach, accrue, release
escrow/batch.py        the batch objective and the repair operators: merge, reassign, delete
escrow/fastrepair.py   a faster reassign pass with the same result as batch.py
escrow/protocol.py     the one run protocol every experiment uses
escrow/provenance.py   stamps each results file with the engine checksums and the date
tests/                 24 tests, including the Kraft identities and the cell ownership invariant
experiments/           the mechanism experiments, the demonstrations and the benchmark runners
experiments/theorem1/  the measurements behind the paper's theorem, kept separate
paper/escrow.pdf       the paper
DATA.md                every external dataset: what it is, where it comes from, where to put it
requirements.txt       what each runner needs; the engine and the tests need only the standard library
```

The engine and the tests are pure Python with no dependencies. `numpy` is used when it is installed
and is not required; the two paths are asserted to give identical results.

## Run it

```bash
python3 tests/test_codes.py      # code-level gates, including three Kraft identities
python3 tests/test_engine.py     # engine gates: positive control, null control, degenerate data
python3 tests/test_batch.py      # the repair operators, exactness, determinism, cell ownership

python3 experiments/e7_immediate_vs_deferred.py   # why immediate minting cannot work
python3 experiments/e8_false_mint_null.py         # nothing is born on noise
python3 experiments/e9_support_curve.py           # the release fires exactly at the price
python3 experiments/e13_creation_bias.py          # creation bias over 270 runs
python3 experiments/e5_order_dependence.py        # what the arrival order costs
python3 experiments/wikipedia_demo.py             # raw Wikipedia infoboxes to typed structure
```

The Wikipedia demonstration fetches a few hundred pages from the public API on first run and caches
them. The benchmark runners need data and extra packages; `DATA.md` says where every input comes
from and `requirements.txt` says what each runner needs.

## What it does, and what it does not

Reported honestly, because the paper reports it that way.

**It works where structure is discrete and identifiable.** Planted structure is recovered exactly,
with the right number of nodes, on every one of twenty arrival orders and under every one of eleven
valid encodings, with not one record in three thousand changing node across codes.

**Nothing is born on noise.** Across 48 pure-noise streams from two thousand to twenty thousand
records, and 300 wider ones, no node is ever created. The streaming decision trees people use
invent 15 to 111 false nodes on the same data at their usual setting.

**It does not absorb noise as structure.** Over a 270-run creation-bias grid, no run ever returned
more nodes than were planted. Past a breakdown noise level the method writes structure off rather
than inventing it, which is the opposite of the failure the classical analysis predicts for
penalty-based rules.

**It is deterministic and it explains itself.** The same records give the same graph every run, and
every node carries a receipt in bits. A 14B language model given the identical records returns 24,
78 and 28 nodes across three seeds, forms a catch-all node holding up to 85 percent of the records,
and would spend millions of tokens on the full stream.

**Where identity lives in free text, version 1 is weak, and that is the honest limit.** On a
noun-phrase benchmark its macro F1 is 0.016 and its pairwise score sits just below the benchmark's
own no-merge floor. On a music benchmark bridged through title tokens it reaches 0.0011 where tuned
embedding baselines reach 0.61 to 0.78. Identity there is a title one token apart, which an
embedding sees at once and a categorical facet cannot see at all. Numeric and text facets are the
next version.

**One theorem is stated more carefully than it was, and we know exactly why.** The lifetime
false-mint bound holds in the idealised setting. For the shipped engine we measured the exceedance
instead of claiming it, because the released statistic is computed on a subsequence the data chose.
We then tried the repair the theorem itself proposes, charging for the choice of seed on the price
side, in three codes, and it does not work: a price is constant in the stream length while the
deficit grows, from 97 bits at five hundred records to 1,205 at four thousand. Remove the seed key's
own term and the requirement drops to 3.5 bits and stops growing, and the same charge covers it, so
the charge is a correct multiplicity correction being asked to pay for a drift. That arm is not
shippable because removing the term destroys the mint. The practical result, that nothing is born on
noise, never depended on the bound: the gate fires on the computed price, which outgrows the drift.
All of it is in `experiments/theorem1/`.

**The candidate budget is a resource bound, not a hidden knob.** A fair objection to any method like
this is that the parameter has moved from the creation penalty to the machinery around it. Swept over
sixteen values from 1 to 32,768 on four streams, the output stops moving at 128, 256, 2,048 and
3,072 and is identical at every larger budget, so the shipped default of 4,096 is on the flat part of
all four curves. Each plateau is where the pool stops evicting, fixed by the peak number of live
candidates the stream itself produces, so the budget is set above a property of the data rather than
searched (`experiments/e19_budget_curve.py`).

## License

PolyForm Noncommercial 1.0.0. You can use, modify and share this software for research, education
and any other noncommercial purpose. Commercial use of any kind requires a commercial license from
Shunya Labs. See [LICENSE.md](LICENSE.md).

## Citation

Sourav Banerjee. *ESCROW: When a Record Stream has Earned a New Node.* Indian Institute of
Technology Kharagpur and Shunya Labs. The paper is in this repository at
[paper/escrow.pdf](paper/escrow.pdf); a citation entry will appear here when it is published.
