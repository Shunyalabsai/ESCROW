# ESCROW branch code EHR05

This revision addresses a specific failure of the EHR03 code: a parent could fit only
one child's owned cells while retaining the membership of the whole parent. Removing
that child saved a separate membership column without worsening its data description.
The EHR03 implementation, specification and negative results remain available.

EHR05 uses the same records, graph invariants, explicit cell owners, literal dictionaries,
background model, structural moves and acceptance checks as
[EHR03](SPECIFICATION.md). It changes membership coding and the history used to predict
a node's values. These are declared coding conventions, with no calibrated parameter.
They do not establish a general hierarchy recovery guarantee.

## Transmitted format and cost

The complete description has these terms:

```
header + structure + membership_patterns + membership + ownership + presence + values
```

The frame is eight bytes of body length, the body, and a 32-byte checksum. The body starts
with `EHR05` and one option byte. Bits zero, one and two respectively select joint
membership coding, all-member predictive training and balanced data exposure. The
reference uses all three. The other settings are diagnostic ablations, not settings
selected using reference labels. The whole byte costs eight bits in every setting.

The header otherwise follows EHR03: it transmits record count, key and value strings and
whole-prefix marginal value counts. All graph parents and supports are then transmitted
in topological order, before any membership or data is decoded. The same pooled binary
Dirichlet-half code covers the triangular set of possible direct parent links. Each
node's support has its own binary Dirichlet-half code. The node count uses the Elias
gamma code of count plus one. Stable run identities are retained in events; the binary
snapshot reconstructs the state up to node renaming.

## Joint membership families

Nodes with the same set of direct parents form a family. A family is processed only
after all its parent memberships have been decoded. Among ready families, use the one
whose earliest node is first in the transmitted topological order. Roots form the
family with no parents. Eligible records are the intersection of parent memberships,
or every record for the root family.

For a family of m nodes, each eligible record has an m-bit membership pattern. A pattern
may contain zero, one or several active nodes. Thus the code permits overlap and does
not impose exclusive siblings. Transmit the number U of observed patterns using gamma,
then their distinct bit vectors in lexicographic order. Each vector is a binary
Dirichlet-half sequence with counts reset at the start of that vector. Finally transmit
the ordered sequence of pattern indices using a U-category Dirichlet-half code.

Using the definitions of gamma, B and KT in the EHR03 specification, a family costs

```
membership_patterns = gamma(U) + sum_pattern B(number of active bits, m)
membership          = KT(pattern frequencies over eligible records)
```

The dictionary is transmitted, so its data-dependent selection is paid for. There is
no extra proposal-count charge. A complementary pair of children no longer requires
two independently coded membership columns. Equal-parent families are a code factorisation,
not supplied category levels. Multiple-parent families use the intersection of their
established parents and follow the same rule. Every emitted pattern must be used.

## Data visit order

After all memberships have been decoded, group record indices by their full membership
bit vector over the transmitted topological node order. Sort the distinct vectors
lexicographically. Visit one record from each nonempty group in that order, repeatedly,
until all groups are exhausted. Keep original arrival order within each group. Decode
every key in this visit order and place its value back at its original record index.

This order uses only memberships already transmitted. It requires no untransmitted
permutation or value lookup. It is fixed by the format; the learner does not choose the
best data permutation. This is a repaired-prefix description. The causal predictive
loss still visits each record once in its actual arrival order and is never replaced
by this retrospective description.

The order exposes a parent to different membership patterns before it repeatedly codes
one branch. The earlier EHR04 revision used arrival order and failed when a whole child
arrived before its sibling. EHR04 results and their source archives are preserved. EHR05
does not assert complete arrival-order invariance: within-pattern order still affects
partially used predictors, and arrival order affects the streaming search.

## Ownership and all-member prediction

For each key and each eligible-owner set, ownership is a Dirichlet-half categorical
sequence including background. Every record-key slot has exactly one owner, including
absent slots. This is unchanged from EHR03.

For each supporting node and key, start a binary presence history and a categorical
value history at zero. Immediately before decoding a slot, its nonzero owner's presence
probability is `(previous member outcomes of this kind + 1/2)/(previous members + 1)`.
Conditional on presence, its value probability is
`(previous member occurrences + 1/2)/(previous published member values + vocabulary size/2)`.
After decoding the slot, update the histories of every eligible member node supporting
the key, including nodes that did not own it. The update uses only already decoded data.
An absent slot updates presence but emits no value. Background uses the fixed
whole-prefix probabilities transmitted in the header and has no residual refitting.

The node's own routed subsequence is therefore not a standalone exchangeable KT block.
Its cost must be evaluated in the specified visit order, conditional on all earlier
member observations. Training on those observations is free because they have already
been decoded; omitting their influence is not allowed in the reference setting.

## Correctness and scope

`branch_coding.py` scores the probability products directly. `branch_codec.py` independently
constructs and decodes integer-frequency arithmetic actions and does not import the
scorer. Both use the graph state invariants. Low-level arithmetic and bit framing
primitives are shared with EHR03. The score includes all literal and structural terms;
arithmetic termination and byte padding explain the small fractional-length difference.

Tests independently reconstruct states with missing keys, new symbols, overlapping
supports, multiple parents and background, and verify move deltas against serialised
descriptions. A finite enumeration verifies normalisation of the conditional data
component with alphabet, graph and ownership held fixed. The exact small-instance audit
enumerates all 715 multisets of at most four records, two optional binary-valued keys
and at most three nodes, including legal graphs and ownership assignments. Each multiset
uses its stated representative arrival order. Unlike EHR03, this does not cover all
ordered inputs by exchangeability; order stress tests are separate. Lower bounds
prune only nonnegative remaining costs; there is no search resource cutoff in that audit.

All tiny optima are empty graphs, so this gate supplies no positive hierarchy-recovery
evidence. The residual-branch audit compares fully encoded alternatives over separate
arrival orders. Planted groups are used only to construct evaluation counterfactuals.
Search recovery is a subsequent, independent gate. Neither a favourable counterfactual
nor a successful constructed stream proves consistency, semantic correctness, global
search optimality or lifetime false-creation control.

Use `BranchEscrowGraph` for EHR05. The original `HierarchicalEscrowGraph` continues to
use EHR03. Both retain cadence 100, budget 4,096, explicit unresolved candidates and a
separate `final_repair()` operation. Historical numerical tables are not EHR05 results.

## Conditional support proposals

The initial EHR05 search reused the historical proposals unchanged. It found the exact
leaf memberships globally on the depth-two diagnostic, but proposed all keys because
all were useful relative to background. Those proposals repeated shared parent keys
and failed the code comparison. Local growth within a parent did not supply the same
complete leaf memberships. This search failure is preserved separately from the coding
counterexample.

`branch_proposals.py` additionally tries globally proposed memberships beneath each
established parent that strictly contains them. It nominates keys by comparing a
child's presence and value KT blocks with a fixed categorical description based on
that parent's member counts. Those whole-prefix counts are proposal statistics only;
they do not substitute for the all-member sequential data code used in acceptance.
The resulting support, membership, parent and ownership choices are all transmitted
and charged by the unchanged full EHR05 scorer and encoder.

It also proposes removing support keys for which a node owns no slot. These changes
can lower redundant ownership-alphabet costs when its children take over specialisation
keys. They are proposals, not mandatory cleanups, and must improve the complete code.
Ordinary overlap does not create a link automatically. The EHR03 proposal source remains
unchanged, and fixed-ancestry and flat restrictions still apply to their EHR05 modes.

The depth-three audit additionally finds sibling pairs whose individual additions lose
bits while their joint addition wins. A compound proposal therefore allows both children
to be added together, retaining the parent, with each child's support nominated by the
same conditional rule. It can release parent support taken over by both children.
Every intermediate state is valid, but only the complete arrangement is compared for
acceptance. A compound has no extra model-choice charge because the resulting graph,
memberships, supports and owners are already encoded. Its revision payload records all
operations and their aggregate edits. Compounds cannot nest.

A joint topological membership-refresh proposal also applies the existing categorical
growth step to established nodes. This reduces the need to spend many complete candidate
evaluations on repairing separate parts of the same prefix. Both compound families are
counted as candidate states under the unchanged 4,096-state budget. The number of internal
operations is visible in the proposal and events; the budget is not a bound on all CPU
work. These are search changes, with no change to the EHR05 score or binary data format.

Shared-parent proposals also evaluate reusing an existing node with the same proposed
parent membership. They can move shared keys back to that node and release them from
children, preserving its stable identity when this complete arrangement wins.

Multiple-parent insertion can require simultaneous withdrawal of older relationships.
The paired-parent proposal family combines two shared-parent proposals that have a child
in common and disjoint nominated key supports. It enumerates subsets of surviving old
links to withdraw, in increasing subset size. These are fully scored alternative
arrangements, not accepted intermediate moves. This limited proposal family is not an
exhaustive search over all forms of joint specialisation.

Local repair and paired-parent arrangements are interleaved one proposal at a time;
neither proposal family gets exclusive use of the search budget. Invalid and duplicate
states retain the original handling. The budget still counts 4,096 distinct evaluated
states across the complete repair. Generating paired structures and enumerating link
subsets can require substantial additional work, which runtime and memory must expose.
