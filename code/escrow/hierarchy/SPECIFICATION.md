# ESCROW categorical hierarchy reference

This is the executable research specification for the EHR03 hierarchical engine.
It is not a claim that hierarchy recovery or lifetime false-creation control has been proved.
Historical results describe the historical flat engine. They are not results for this code.
The subsequent EHR05 branch code is specified separately in
[BRANCH_SPECIFICATION.md](BRANCH_SPECIFICATION.md); EHR03 remains executable unchanged.

## Problem and state

The input at time n is an ordered prefix of n finite dictionaries from Unicode key strings
to Unicode value strings. Missing a key differs from publishing an empty string. Values of
other Python types must be explicitly converted by the caller. No reference group labels,
reference edges, embeddings, or future records enter this engine.

A state is a directed acyclic graph of nodes, support sets of keys, membership sets of
record indices, and one owner for each record-key slot. Owner zero is the background.
Absent slots also have owners. Every nonzero owner must support the key and contain the
record. A child's membership is a subset of every established parent's membership.
Multiple parents mean joint specialisation. Intersecting membership alone does not imply
a parent link. Direct links have no transitive redundancy. Empty membership is forbidden;
empty support is allowed for a node that explains membership organisation only.

Node identities persist during processing, deletion and revision. The binary snapshot
reconstructs the graph up to node renaming. Run events and JSON snapshots retain the
persistent identities; they are protocol metadata, not extra predictive evidence.
There is no configured discovery depth. Every existing node is included as a refinement
scope at subsequent repairs. Finite resource and integer representation limits still apply.

## Complete observed-prefix description

Let gamma(x) be the Elias gamma code for the positive integer x+1, for x >= 0.
Its length is 2 floor(log2(x+1))+1. A string costs gamma(number of UTF-8 bytes) followed
by those bytes. Distinct keys and each key's distinct value strings are sorted.

For a categorical sequence with d possible actions and action counts c_j, define

```
KT(c_1,...,c_d) = -log2 [ Gamma(d/2) / Gamma(N+d/2)
                         * product_j Gamma(c_j+1/2) / Gamma(1/2) ]
```

Here N is the sum of the counts, including zero counts for possible unobserved actions.
KT abbreviates the Krichevsky-Trofimov, or Dirichlet-half, sequence code. There is no
multinomial coefficient: the description transmits the ordered sequence, not its histogram.
For d=1 the cost is zero. Write B(t,m)=KT(m-t,t).

The header sends n, the number of keys, every key string, each key's vocabulary size,
every value string, and each value's whole-prefix frequency. These statistics are redundant
but explicitly transmitted. They are common to every graph of this prefix. New keys and
values therefore incur actual literal and count costs, never a free dictionary lookup.

The graph is serialised in a deterministic topological order. Its cost is:

```
gamma(K)
+ B(number of direct edges, K*(K-1)/2)
+ sum_v B(number of supported keys, number of observed keys)
+ sum_v B(number of members, number of eligible records).
```

Parent indicators are one pooled Bernoulli sequence over the triangular set of possible
earlier-parent/later-child pairs. Pooling makes the ideal link cost independent of the
arbitrary ordering of equal-cost parent alternatives. The decoder reads each node's parents
and support before its membership. Eligible records are all records for a root, or the
intersection of the already decoded parent memberships for a child. Every membership
choice not implied by this eligibility is transmitted.

For each key, in arrival order, the eligible owner alphabet is background plus every
supporting member node. Owner sequences use a distinct KT block for each pair of key and
eligible-owner set. This charges selective retention in the background, ordinary overlap,
and competition between parents and children. There is exactly one ownership action per
slot. Membership and ownership are distinct; a member need not own a published value.

After an owner is decoded, key presence is transmitted. For each nonzero owner-key pair,
presence uses a binary KT block over its routed slots. Published values use a KT block
over the complete transmitted vocabulary of that key. Absence emits no value.

The background uses fixed probabilities reconstructible from the header. If key k appears
P_k times in n records, its presence probability is (P_k+1/2)/(n+1). If symbol s appears
h_ks times, its value probability is (h_ks+1/2)/(P_k+d_k/2). These probabilities do not
refit the residual background after owners have been chosen. This deliberate modelling
choice prevents a residual background from becoming an unpriced final group. It has not
been selected by benchmark-label agreement and needs empirical evaluation.

The complete decomposition is header, structure, membership, ownership, presence, and
values. Every proposed move describes the identical record prefix with this decomposition.
There is no extra proposal-count charge: the selected supports, memberships, owners and
links have already been transmitted. Search metadata does not replace any of these terms.

## Independent implementation and numerical details

`coding.py` uses closed-form log-gamma lengths. `codec.py` independently uses sequential
integer probabilities, with frequencies 2*c+1 for KT actions, and a 64-bit arithmetic
coder. It never calls the scorer. An eight-byte body length, five-byte format marker and
32-byte SHA-256 checksum are counted in the header. Explicit byte framing makes the
complete file self-delimiting. EHR03 is this format; EHR02 lacked this framing.
Arithmetic termination and byte padding produce a small integer-length
difference from the ideal score. Tests check reconstruction and this difference for
missingness, new values, overlaps, multiple parents and every move class.

Acceptance first compares complete ideal lengths and then requires a strictly shorter
actual serialisation. A 32-unit floating-point tolerance only detects numerical ties;
it is not an evidence calibration. If byte rounding prevents an improvement, the engine
reports that fact. This conservative reference search can miss an alternative serialised
improvement when its best ideal candidate does not improve at byte precision.

All elementary finite categorical decisions are normalised. The set of valid graph and
record encodings can have probability mass less than one because invalid structures,
invalid UTF-8 strings and inconsistent transmitted histograms are rejected. Such rejection
does not invalidate the code. Reconstruction and normalisation are separate gates.

## Search, structural escrow and revisions

`process(record)` evaluates a causal observation code before storing the record. The
reference implementation initially leaves each new record in the background, and repairs
the prefix every 100 records. This is a disclosed assignment latency, not an immediate
assignment claim. `snapshot()` returns direct edges, all memberships, owners, events and
unresolved proposals. `final_repair()` is explicitly separate and never implicit at EOF.

The repair budget is 4,096 distinct state evaluations across the whole repair. Invalid
and duplicate generated proposals are counted separately. The budget does not cap all
proposal-generation work, so runtime and memory must also be measured. A repair can
accept several moves; every acceptance is rescored before committing. Candidate gains
are replaced when their prefix or incumbent changes. Gains on overlapping prefixes are
never added. Competing evaluated arrangements and exact ties remain in the escrow log.
Unevaluated search space is not described as evidence against a relationship.

Proposals include key presence, key signatures and value carriers, one categorical
growth step, refinement in existing memberships, shared-parent pooling, alternative
parents, link removal, support changes, membership growth, routing changes, merge and
delete. Growth and pooling statistics only propose states. Acceptance always uses the
complete description. No planted label enters proposal generation.

Events report the operation, node and edge edits, exact past membership and owner edits,
all signed bit changes, serialised bit change, and time since the proposal first appeared.
The JSON revision payload also has a disclosed transport length, with an eight-byte
length frame, kept separate from the snapshot score. Stale candidate evaluations are
marked as needing re-evaluation, not presented as current evidence.
That last delay is proposal delay. Benchmark detection delay from signal onset is a
separate evaluation and must not be inferred from proposal delay.

## Streaming prediction is a different quantity

`predictive.py` evaluates each arriving record under distributions fitted to past records
only. It uses a mixture of current member distributions and a background expert, with
weights proportional to past membership plus one half. Known symbols plus an escape
action have normalised probabilities. Escapes and new keys transmit integer and literal
codes. The literal code is a subprobability code over valid UTF-8 strings.

These weights are rebuilt from past memberships. This is not an online posterior over
a fixed collection of whole-sequence experts, and has no claimed fixed-expert mixture
regret bound. Predictive loss is accumulated once per record and never recomputed after
a repair. It is reported separately from the description of the current graph and prefix.
The current predictor is a reference observation code, not a semantics experiment.

## Scope of claims

The formal target is a reconstructible categorical graph description with no calibrated
parameter. Coding conventions, search cadence, candidate budget and proposal families
remain declared choices. No global search optimality is asserted outside an exhaustive
audit. No recovery theorem, consistency theorem, or lifetime false-creation bound is
asserted for this adaptive search.

The historical engine's actual mint price is `codes.price`: an announcement cost,
two operation bits, membership and support lengths, a node-count change, and a node
identifier choice, with optional seed naming. It is not the idealised paper expression
for Indian buffet process posterior odds. The new engine uses complete state deltas.

The earlier immediate-mint argument is an expectation statement under its stated null,
not a pointwise impossibility theorem. A consistency argument must include membership
cost that is linear in n at fixed membership fraction. A no-hierarchy argument for a
narrow flat ownership code does not rule out conditional hierarchical membership codes.
Finally, an e-process proof for a fixed, predictably specified candidate does not prove
validity for retrospectively fitted candidates, graph revisions or shortest replay
selection. These claims require new proofs for the actual selection process.

Key-name merging remains an audited historical experiment outside this hierarchical
code. Its spelling term now aligns actual strings across keys. The historical minimum
of seven records and proposal-naming charge remain active in that historical operator;
neither is used as the new hierarchy acceptance rule. Compression of spellings alone
does not prove semantic equivalence of key names.
