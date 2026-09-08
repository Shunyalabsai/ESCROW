# The Theorem 1 measurements

These are the scripts behind `results/e16_supermartingale_check.json`, `findings/THEOREM1.md` and
`findings/THEOREM1_CORRECTIONS.md`, and the arm outputs they produced are in
`results/theorem1_arms/`. They are kept apart from the numbered experiments because they do not
produce a result the paper prints; they answer one question about the engine, and the paper cites
their answer.

The question. The shipped value block is a sub-probability, so the martingale step of Theorem 1 is
not exact as written. Two corrections are principled: charging the escape only for the values it can
reach (`ESCROW_TIGHT_NAMING=1`), and excluding from the released statistic the seed key that created
the candidate (`ESCROW_UNSELECTED_STATISTIC=1`). Both are implemented in `code/escrow/` and both
default to off, so the shipped behaviour is unchanged and every committed result reproduces.

| script | what it measures |
| --- | --- |
| `mass_check.py` | the block's total predictive mass by brute-force summation over the alphabet, against the closed form, under both naming charges |
| `e16_stepwise.py` | the one-step expectation of the ratio under the incumbent, which is the martingale step itself: 1.000000 under the tight charge, 0.862862 as shipped, over 955,200 null key-steps |
| `e16_arms.py` | the exceedance on E16's own 48-stream design, under each of the four flag combinations |
| `e16_wide.py` | the same on 300 independent null streams and 12,000 candidate trajectories, which is the number to quote, because E16's four length blocks are prefix-nested and count the same trajectories four times |
| `quick_controls.py` | the positive controls under each arm: both synthetic streams, the null, and the deferred arm |
| `planted8_margin.py` | how far the best leftover candidate falls short of its price when the seed key's evidence is excluded, which is what shows the drift is load-bearing |
| `e13_subset.py`, `e13_k20_tail.py` | the creation-bias grid under the tight charge, including the three high-noise cells that refuted the expectation that it was free |
| `compare.py`, `drive.sh` | the harness: run an arm over the experiments in a scratch tree and diff its results against the committed ones |
| `e18_seed_charge_null.py` | E18: the null accumulator's supremum and drift as the stream grows, and the uniform slack Gamma each level would need, for the shipped statistic and for the one with the seed key's term removed |
| `e18_charge_cost.py` | E18: what the price-side seed naming charge costs the mint, over the two codes, the tight naming charge and a sweep of the uniform slack, on the canonical controls, E5's twenty orders, E7's deferred arm and the Wikipedia demonstration |
| `e18_report.py` | E18: collects every arm's exceedance and every control into `results/e18_seed_naming_charge.json` |
| `_run.py` | runs any script in this directory. The files here still carry the `sys.path` line from when they sat in `code/experiments/`, so run directly they raise ModuleNotFoundError; this puts `code` on the path and leaves the scripts unmodified |

How to run one arm without touching the repository's results:

```bash
ESCROW_TIGHT_NAMING=1 python3 code/experiments/theorem1/_run.py e16_wide.py
ESCROW_SEED_NAMING_CHARGE=1 ESCROW_SEED_NAMING_MODE=av \
    python3 code/experiments/theorem1/_run.py e16_wide.py
```

`drive.sh` copies the tree to a scratch directory first, which is the safe way to run the flags on
across every experiment at once. Read it before using it; it hardcodes a scratch path.

The finding, in one line: the tight charge makes the martingale step exactly true and does not
restore the bound, because the released statistic is computed on a subsequence the data chose.

E18 answers the question that file left open. A third flag, `ESCROW_SEED_NAMING_CHARGE=1`, keeps
the seed key's evidence and charges for the selection on the price side instead, adding a prefix
codeword that names the seed pair (`ESCROW_SEED_NAMING_MODE=av` for log2|A_n| + log2 d_a,
`ln` for the universal code on the two intern ranks, plus an optional uniform slack
`ESCROW_SEED_NAMING_GAMMA`). It does not restore the bound and it is close to free. The reason is
that the null accumulator drifts, so the slack the bound needs grows with the stream while a naming
charge is constant in it; with the seed key's term removed from the statistic the slack needed is
3.4 to 3.6 bits and is flat in the stream length, which is what a multiplicity correction should
look like. `results/e18_seed_naming_charge.json` holds the table and the verdict.
