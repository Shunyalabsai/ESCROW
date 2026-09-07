# ESCROW

**When has a record stream earned a new node?**

A stream of records arrives and does not stop, and each record is a small set of attribute key and
value pairs. ESCROW builds a graph over that stream while it is still running, deciding at every
arrival which existing nodes the record joins and whether the stream has now earned a new one. One
description length makes both decisions, and no quantity in the rule is calibrated on data.

The price of a new node is read off a code that predicts each record before it sees it, so it is
computed rather than chosen. No fresh node can pay that price on its own first record under any
valid code, which is why the evidence has to wait: it accrues in an account named by the (key,
value) pair that opened it, and the node is created at the first moment the account covers the
price. Every node that is created carries a receipt in bits naming the key that paid for it.

This is the working repository. It holds the paper source, the engine, every experiment, the
measured results and the reasoning behind them. The public release, which carries the code and the
paper alone, is at [github.com/Shunyalabsai/ESCROW](https://github.com/Shunyalabsai/ESCROW).

## Where the work stands

The paper is written and builds to ten pages of main text plus its appendices, with the argument in
the register the author asked for and every printed number read from a file under `results/`. The
engine is corrected, and the three defects an audit of our own implementation found are each guarded
by a test.

Four things are settled and should not be relitigated. Immediate minting cannot work, and both of
its forms measure at zero mints. Nothing is created on pure noise across 48 streams and 300 wider
ones, where streaming decision trees grow to a mean of 91 false nodes. Planted structure is
recovered exactly on every one of twenty arrival orders. The creation bias runs the opposite way
from the one the classical analysis predicts, since no run in a 270-run grid ever returned more
nodes than were planted.

One thing is settled and uncomfortable, so it is stated carefully rather than quietly. The lifetime
false-mint bound holds in the idealised setting and does not hold for the statistic the engine
ships, because that statistic is read off the records that carry the candidate's pattern, which is a
subsequence the data chose. We measured the exceedance instead of claiming the bound, then tried the
repair the theorem itself proposes, charging for the choice of seed on the price side, in three
codes. It does not work, and the reason is now understood: the null accumulator drifts, so the slack
the bound needs grows with the stream while a naming charge is constant in it. The practical result
never depended on the bound, because the gate fires on the computed price, which outgrows the drift.
`findings/THEOREM1.md` and `findings/THEOREM1_CORRECTIONS.md` hold the decision and the evidence.

Two things are open. The comparison against the language-model system on its own product data is
begun and not collected. The strengthening experiments in item 7 of `findings/TODO.md` are
deliberately not started, on the reviewer's own advice that they would add less than the items above.

## Where to look

| Path | What it holds |
| --- | --- |
| `paper/` | the LaTeX source and the built PDF. `main.tex` pulls in `sections/*.tex`; figures are built by `code/experiments/fig_*.py` |
| `code/escrow/` | the engine: the codelength primitives, the insertion process, the batch objective and its repair operators |
| `code/experiments/` | every experiment and demonstration, one file each, writing into `results/` |
| `code/experiments/theorem1/` | the measurements behind the theorem finding, kept apart because the paper prints their answer and not their tables |
| `code/tests/` | 24 tests, including the Kraft identities and the invariant that every cell is coded exactly once |
| `code/tools/check_numbers.py` | fails if a superseded number is still printed anywhere in the paper or the talk |
| `results/` | 67 result files, each stamped with the engine checksums and the date it was produced |
| `findings/` | the decision record: what was measured, what it refuted, and what was decided as a result |
| `related_work/` | 54 papers, downloaded and extracted, each read in full rather than from its abstract |
| `talk/` | the reveal.js deck. It is never published |
| `DATA.md` | every external input a benchmark run needs, where it comes from and where to put it |

`findings/TODO.md` is the current list of what is left, with the state of each item.

## Running things

```bash
python3 code/tests/test_codes.py      # code-level gates, including three Kraft identities
python3 code/tests/test_engine.py     # engine gates: positive control, null control, degenerate data
python3 code/tests/test_batch.py      # the repair operators, exactness, determinism, cell ownership

python3 code/experiments/e7_immediate_vs_deferred.py   # why immediate minting cannot work
python3 code/experiments/e8_false_mint_null.py         # no node is created on noise
python3 code/experiments/e9_support_curve.py           # the release fires exactly at the price
python3 code/experiments/wikipedia_demo.py             # raw infoboxes to typed structure

# these three answer "what is the comparison evidence of?" and need more than the standard library:
python3 code/experiments/e41_symmetric.py     # the same language model on the tasks the price is for
python3 code/experiments/e43_traditional_on_semantic.py            # what the encyclopedia stream ranks
python3 code/experiments/e44_string_algorithms_on_key_identity.py  # what the key-identity gold ranks

python3 code/tools/check_numbers.py                    # run before every paper build
```

The engine and the tests are pure Python over the standard library. The benchmark runners need data,
and `DATA.md` says where each input comes from.

To build the paper, run `tectonic -X compile main.tex` inside `paper/`, then check the rendered PDF
rather than the log: references must start on page eleven, with no overfull boxes, no unresolved
references and no dashes of any kind.

## Standing rules

These were each decided once, against evidence, and reverting them silently would undo that work.

The claim is that no parameter is calibrated. It is never phrased as no free parameter, because a
free parameter is exactly what the choice of code still is. The rule the method applies is a price,
and it is never called a threshold. The work is framed as parameter-free structure learning and
never as a system for building knowledge graphs, because that framing was rejected across the board
at the venue we are aiming for. Em dashes and en dashes appear nowhere, in the paper, the talk, the
code or these files.

The public repository carries the code and the paper PDF. It never carries the talk, the LaTeX
source, the findings, the private catalogue or any path, host or name from inside the lab.
