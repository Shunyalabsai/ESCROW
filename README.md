# ESCROW

A stream of records arrives and does not stop. A record might be a product listing, a page with an
infobox, a song, or a person, and whatever it is, it arrives as a small set of named parts, one key
and one value at a time. ESCROW builds a graph over that stream while it is still running, deciding
at every arrival which existing nodes the record joins and whether the stream has now earned a new
one. One description length makes both decisions, and no quantity in the rule is calibrated on data.

A new node has a price in bits, and that price is computed from the stream rather than chosen. No
fresh node can pay its price on its own first record under any valid code, so the evidence has to
wait: it accrues in an account named by the (key, value) pair that opened it, and every later record
that goes unexplained in the same way pays into that account. The node is created at the first
moment the account covers the price, and never before. It arrives with a receipt in bits naming the
key that paid for it.

The paper is in this repository: [paper/escrow.pdf](paper/escrow.pdf).

## Why this exists

The word worth defending is *earned*. If a node exists because the data paid for it then there is a
price, and something has to say what that price is. Almost every system that builds a graph from a
stream answers with a number somebody chose. A streaming clustering rule opens a node when the
distance from a record to the nearest one exceeds a chosen value. XGBoost adds a leaf only when the
gain beats a per-leaf charge whose default is zero and for which its own paper reports no value.
DP-means and BP-means come closer, because they begin from a model of how many groups a stream
contains, but they obtain their new-node penalty by sending the noise that model assumes to zero.
Language-model pipelines state no price at all; they ask a model, record by record, and keep what it
samples.

The Bayesian rules are the interesting failure, because their constant was not there at the start.
Before the limit is taken, the new-node penalty is a codelength that the model works out for itself.
The limit erases every part of it that could have been computed and keeps only the part a person
inserted, so the constant is what a deletion left behind. We do not take the limit.

What we do instead is read the price off a code. A probability model of a stream is also a code for
it, because a prediction that gives probability P to what happens can write it down in minus log P
bits. The model here is the Indian buffet process, in which a new feature may appear at any time,
and its creation rate is given a prior and integrated out rather than fixed. What comes out is a
price that grows with the logarithm of the stream, with the constant on that logarithm equal to one.
Both sides of the comparison are bits, so there is no exchange rate to set.

One assumption is declared rather than removed. The starting prior on the stream's own minting rate
is a choice, and it is worth 0.19 bits per mint at a hundred thousand records on a declared
trajectory, and 0.34 on the trajectory our own run walks. That is the honest residue, and the paper
states it in the same breath as the claim.

## What is here

```
code/escrow/codes.py        the codelength primitives, including the KT codes, the membership
                            column and the mint price
code/escrow/engine.py       the insertion process: parse, retrieve, attach, accrue, release
code/escrow/batch.py        the batch objective and its repair operators, merge, reassign and delete
code/escrow/fastrepair.py   a faster reassign pass with the same result as batch.py
code/escrow/protocol.py     the one run protocol every experiment uses
code/escrow/provenance.py   returns the engine checksums and the date; most runners stamp their
                            results file with it
code/tests/                 24 tests, including the Kraft identities and the invariant that every
                            cell is coded exactly once
code/experiments/           the mechanism experiments, the demonstrations, the benchmark runners
                            and the figure scripts
code/experiments/theorem1/  the measurements behind the paper's theorem, with their own README
results/                    every result file the paper reads its numbers from
results/wiki_large/         4,635 Wikipedia infobox records over eight types, the cache behind the
                            scale ladder, so it replays with no network
results/wikidata_cover/     3,362 Wikidata people with their occupations, the one real multi-label
                            benchmark here, and the raw query response it was built from
paper/escrow.pdf            the paper
DATA.md                     every external dataset: what it is, where it comes from, where to put it
requirements.txt            what each runner needs
```

The engine and the tests are pure Python over the standard library. `numpy` is used when it is
installed and is not required. On the two test streams the vector path and the pure-Python path are
asserted to give the same node count, the same mints with the same members, and the same objective
to within 1e-6, by `test_numpy_path_equals_pure_path`.

## Run it

```bash
python3 code/tests/test_codes.py      # code-level gates, including the Kraft identities
python3 code/tests/test_engine.py     # engine gates: positive control, null control, degenerate data
python3 code/tests/test_batch.py      # repair operators, exactness, determinism, cell ownership

python3 code/experiments/e7_immediate_vs_deferred.py   # why immediate minting cannot work
python3 code/experiments/e8_false_mint_null.py         # no node is created on noise
python3 code/experiments/e9_support_curve.py           # the release fires exactly at the price
python3 code/experiments/e13_creation_bias.py          # creation bias over 270 runs (about 10 min)
python3 code/experiments/e5_order_dependence.py        # what the arrival order costs
python3 code/experiments/wikipedia_demo.py             # raw infoboxes to typed structure
```

Every one of these runs on the standard library alone, and each writes its result into `results/`.
The three Wikipedia caches are committed, so the Wikipedia numbers replay offline; delete a cache
file and the demo rebuilds it from the public MediaWiki API. The benchmark runners need data and
extra packages, and `DATA.md` says where every input comes from.

## What it does, and what it does not

**It recovers structure that is discrete and identifiable, and it does so without a knob.** On a
planted eight-group stream of 3,000 records it returns the right eight nodes with an adjusted Rand
index of 1.0. It does so on every one of twenty arrival orders. It also does so under every one of
the nine valid prefix codes among the eleven encodings we tried, with not one record in three
thousand changing node between them.

**Where the structure is not identifiable, the choice of code does move the answer, and we say so.**
On the two harder streams the encoding sweep changes which nodes exist and not merely when they are
born. The comparison that matters is against what a change of arrival order alone already does to
the same code: the spread tracks that band on one stream and exceeds it on the other. That is
reported in the paper rather than buried.

**No node is created on noise, across sixteen null families, with one exception we found by
widening the test.** Across 48 pure-noise streams from two thousand to twenty thousand records, and
300 further streams of the same shape, no node is ever created. Widening that to twenty-five
families, sixteen give nothing at all, over alphabets from 2 to 500, key counts from 2 to 32 and
Zipf exponents from 0 to 2. The families that do mint are those where key presence is itself random
and high: there a single node appears on some seeds, describing what the records share. The claim
that survives is narrower than the one we first printed, and the paper states it that way. The streaming decision
trees grow on that same noise at a loose confidence level, which is not their library's default:
EFDT reaches five-seed means of 15, 73, 73 and 91 nodes as the stream lengthens, with 111 in its
worst single run, and VFDT reaches a mean of 19. At the library's own default both stay at the root
and invent nothing, and that silence costs them up to 20,800 records of delay before they notice
real structure.

**It does not absorb noise as structure.** Over a 270-run creation-bias grid, no run ever returned
more nodes than were planted. Past a breakdown noise level the method writes structure off rather
than inventing it, which is the opposite of the failure the classical analysis predicts for
penalty-based rules.

**Against the incumbent on its own data we lose comprehensively, and that is the headline negative.**
AutoPKG builds product knowledge graphs by prompting a model, and its key-identity decision is the
same question this criterion claims to answer. Run on their Lazada data with their metric, 3,033 raw
keys and 328 gold pairs that should merge, their agent reaches F1 0.6355 at precision 0.6497 and
ESCROW reaches 0.0008, one true pair against 2,105 false. Two things are true and neither rescues it.
The read-out scored us by whether two keys share a node's support, which is a co-occurrence relation
and not an identity claim, and rescoring with the rule this project's own counting prescribes, that
two names for one role never share a record and draw from one value pool, is perfectly precise and
almost silent: 2 pairs of 328. And version 1 ships no operator that prices two keys as one, so the
criterion cannot make the decision at all. Key identity is an open problem here, not a contribution.
See `results/e20_full_withllm.json` and `results/e20_rescored.json`.

**Against tuned baselines it does not hold its own, and our own test said so.** We wrote before the
experiment that if an untuned distance rule, given our own categorical input rather than an
embedding, matched us, then the criterion is not what does the work. One does. On both synthetic
streams ESCROW recovers the truth exactly and so does a k-means whose k is chosen by silhouette; on
raw Wikipedia infoboxes that same baseline reaches 0.9898 against our 0.8691, consulting no label
and having nothing left to set. We report that the test fired rather than rewording it.

What survives is narrower and is the claim this work can defend. Transferred knobs, which is what a
practitioner actually carries to a new stream, collapse: on a cover task a matrix factorisation
given its knob from another fixture falls to 0.1953 and 0.1483 where ESCROW holds 0.9332 and 0.7452.
On the one real multi-label benchmark here, 3,362 Wikidata people labelled by occupation, ESCROW
reaches an omega index of 0.2044 with nothing set, level with k-means handed the true number of
labels at 0.2053, below an oracle-tuned factorisation at 0.3137, and twenty times above the best
that factorisation manages under a label-free rule fixed in advance, which is 0.0099. It is not the
most accurate method. It is the accurate one with nothing to carry over.

**It is deterministic, and every node explains itself.** The same records in the same order give the
same graph on every run, and every node carries a receipt in bits. A different arrival order gives a
different graph, which is the cost of a one-pass greedy sequence and is measured rather than hidden:
on Wikipedia the index runs from 0.64 to 0.99 with 3 to 6 nodes. For contrast, a 14B language model
given the first 1,000 raw Lazada listings in the identical order returns 24, 78 and 28 nodes across
three seeds, agreeing with itself at a mean pairwise index of 0.1647, with a catch-all node holding
23 to 85 percent of the records. It spends 107,512 to 157,821 tokens on those thousand listings,
which extrapolates to 2.3 to 3.4 million on the full 21,365-record stream, and that is a lower bound
because the prompt grows with the node list. ESCROW returns the same six nodes on every run.

**Where identity lives in free text, version 1 is weak, and that is the honest limit.** Free text is
quarantined because any code for it must first choose a tokenisation, which we measure at 5 to 75
bits per field, more than any constant the method removes. On a music benchmark whose five
categorical fields carry no identity signal for anyone, every method scores null; bridged through
title tokens, three of four oracle-tuned embedding baselines reach 0.61 to 0.78 pairwise F1 and
ESCROW reaches 0.0011. Identity there is a title one token apart, which an embedding sees at once
and a categorical facet cannot see at all. On a noun-phrase benchmark its macro F1 is 0.016, because
7,009 subjects end as singletons and most gold clusters are never formed; its micro and pairwise F1
of 0.7777 and 0.7160 sit just below that benchmark's no-merge floor, which is what the embedding
baselines reach by returning close to one cluster per surface string. Numeric and text facets, and
the operator that would price two keys as one, are the next version.

**One theorem is stated more carefully than it was, and we know exactly why.** The lifetime
false-mint bound holds in the idealised setting. For the shipped engine we measured the exceedance
instead of claiming it, because the released statistic is computed on the records that carry the
candidate's pattern, which is a subsequence the data chose. We then tried the repair the theorem
itself proposes, charging for the choice of seed on the price side, in three codes. It does not
work. The reason is that the null accumulator drifts, so the slack the bound would need grows with
the stream, from 97 bits at five hundred records to 1,205 at four thousand, while a naming charge is
constant in the stream length by construction. Remove the seed key's own term from the statistic and
the requirement becomes 3.4 to 3.6 bits and stays flat, and the same charge then covers it at every
level, which is what a multiplicity correction should look like; that arm is not shippable, because
removing the term destroys the mint. The practical result never depended on the bound, because the
gate fires on the computed price, which outgrows the drift. All of it is in
`code/experiments/theorem1/`.

**The candidate budget is a resource bound, not a hidden knob.** A fair objection to any method like
this is that the parameter has moved from the creation penalty into the machinery around it. Swept
over sixteen values from 1 to 32,768 on four streams at five seeds each, the output stops moving at
128, 256, 2,048 and 3,072 and is identical at every larger budget, so the shipped default of 4,096
sits on the flat part of all four curves. On three of the four streams the plateau is exactly where
the pool stops evicting, which is a property of the stream and not a value we searched for; on the
fourth the output settles at 256 while eviction continues to 512. Below the plateau the budget does
change the output, and not monotonically, which is why it is a resource bound rather than a quality
knob. Those four streams run 320 to 3,000 records, and on the full Lazada stream and on MusicBrainz
20K the cap does bind, so the claim is made at that scale and not beyond it.

**The price says which streams the rule is for, in closed form, and that was checkable before any
run.** Naming the t-th member of a node costs exactly log2((n - t + 1/2) / (t - 1/2)) bits, which
agrees with the shipped price routine to four parts in a billion. A member must save more bits than
the log of the stream it has to be picked out of. Separately, a candidate is named by one key and
value and accrues only on records carrying that pair, so a group is reachable only if one of its
characteristic pairs recurs often enough, which we measure at three to twenty-four records.

One number then orders every result here: the records per gold label a benchmark asks for. It is
1,000 and 375 on the planted streams, 107 on the encyclopedia stream, 3.85 for the Lazada silver
types, 1.94 on MusicBrainz 20K and 2.03 on ReVerb45K. We are at or near 1.0 on the first three and at
zero on the last three. Deduplication benchmarks ask for groups of two, and neither condition
contains a quantity that could be set differently, so this is not a task the method does badly. It is
a task its price excludes. Both conditions are necessary and neither is sufficient: `e34` gives a
stream that meets both with room to spare and is still not solved.

**The same counting gives the rest of the graph, and this is the part we did not expect.** A parent
is not a node and never could be: we prove that minting one costs a membership column the decoder can
already derive, growing with the stream, against a saving that grows only with its logarithm. The
relation lives in the attributes instead, where it costs nothing. If every record carrying key b also
carries key a and a is commoner, then b specialises a. On a planted two-level stream that recovers
the tree at precision 1.000, unchanged whether the rule demands every record or ninety-five percent
of them; at three levels all 180 implications point from the deeper key to the shallower one. Three
further relations come from the same counting: siblings, two names for one role, and value-to-value
typed edges with their cardinality. On the planted stream the specialisation and the typed edges
reach precision and recall 1.000 and the other two reach 0.750, the single error being the case the
fixture plants on purpose, two siblings drawing from one vocabulary. On 3,362 real Wikidata people it
returns 538 specialisations, 1,189 sibling pairs and 2,076 typed edges. Membership of a political
party implies gender, citizenship, instance of, and date of birth. Nothing supervised that. The
implication reading needs no engine at all, since it is a property of the counts; what the nodes add
is the same relation in compressed form.

**The repair schedule is a quantity we cannot choose without labels, and it changes the answer.** The
criterion is parameter-free and that part is exact. The protocol is not. The repair pass runs every
100 records by declaration, and sweeping that over 50, 100, 200, 500, 1,000 and a single final pass
moves the agreement from 0.3998 to 0.8669 on a planted two-level stream and from -0.0162 to 0.2078 on
the Wikidata cover benchmark. For arrival order the same problem dissolves, because the batch
codelength ranks the orders correctly and selecting on it adds nothing calibrated. That answer is not
available here: the shortest description picks the best cadence on three of six streams and on the
Wikidata benchmark picks a state scoring 0.1222 where 0.2078 was available. Treat every number here
as conditional on the shipped cadence. Closing this is the first thing a version 2 owes.

**What it costs.** The insertion step is flat in the node count when supports are disjoint, at 48.6
microseconds per record for 8 nodes and 45.0 for 128, and it is not flat when the supports all
overlap. End-to-end throughput falls from 7,892 to 909 records per second between a thousand and a
hundred thousand records, because at the top the repair pass takes 95.8 percent of the time.

## License

PolyForm Noncommercial 1.0.0. You can use, modify and share this software for research, education
and any other noncommercial purpose. Commercial use of any kind requires a commercial license from
Shunya Labs. See [LICENSE.md](LICENSE.md).

## Citation

Sourav Banerjee. *ESCROW: When a Record Stream has Earned a New Node.* Indian Institute of
Technology Kharagpur and Shunya Labs. The paper is in this repository at
[paper/escrow.pdf](paper/escrow.pdf); a citation entry will appear here when it is published.
