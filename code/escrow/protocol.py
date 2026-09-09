"""The one run protocol every experiment uses.

Every ESCROW result in the paper is produced by run_stream: the engine with the batch objective
installed (so a fresh mint can be absorbed into an existing node when its support is contained),
a repair pass every REPAIR_EVERY records, and one full repair pass at the end of the stream.
Scripts that need progress output or per-record hooks call new_run and drive the loop themselves,
but must keep the same three choices.
"""
from .engine import EscrowGraph
from .batch import BatchObjective

REPAIR_EVERY = 100


def new_run(objective_cls=BatchObjective, cand_pool_cap=None):
    """The shipped run. `cand_pool_cap` is the candidate-pool budget: None keeps the
    engine's own default (4096), so every existing caller is unchanged. It is exposed
    here only so the budget sweep (experiments/e19_budget_curve.py) can move the one
    operational parameter without leaving the protocol."""
    g = EscrowGraph() if cand_pool_cap is None else EscrowGraph(cand_pool_cap=cand_pool_cap)
    b = objective_cls(g).install()
    return g, b


def run_stream(records, every=REPAIR_EVERY, objective_cls=BatchObjective, on_progress=None,
               cand_pool_cap=None):
    g, b = new_run(objective_cls, cand_pool_cap=cand_pool_cap)
    for i, rec in enumerate(records):
        g.process(rec)
        if (i + 1) % every == 0:
            b.repair()
            if on_progress is not None:
                on_progress(i + 1, g)
    b.repair(full=True, final=True)
    return g, b


def describe(every=REPAIR_EVERY, objective_cls=BatchObjective):
    return (f"EscrowGraph + {objective_cls.__name__}(g).install(); repair() every {every} records; "
            f"final repair(full=True)")


def record_labels(g, n):
    """One label per record, for scoring the cover against a single-label gold.

    The output is a cover, so a record can sit in several nodes, while ARI and the rest of the
    single-label metrics want exactly one label each. The rule is the one the language model
    comparison has always used: the record takes the node with the most members that contains it,
    and the lowest node id breaks a tie. A record in no node takes -1, so the background is one
    cluster rather than a cluster each.

    This lives here because it was written out by hand in more than one experiment and the copies
    disagreed. Iterating `g.nodes.values()` unsorted leaves the label of a multi-node record to
    dictionary order, which on the 320 record encyclopedia stream moves 9 of them and the reported
    agreement by 0.04. Call this instead of writing the loop again.
    """
    lab = [-1] * n
    for v in sorted(g.nodes.values(), key=lambda x: (x.t, -x.nid)):
        for m in v.members:
            lab[m - 1] = v.nid
    return lab


# --------------------------------------------------------------------------- #
# Consensus over arrival orders.
#
# The batch objective is a function of the state and not of the arrival order, but the greedy
# sequence that reaches a state is not, so the same data in a different order can land somewhere
# else. E5 measures that spread on the encyclopedia stream as ARI 0.6389 to 0.986. Running several
# orders and looking at what they agree on separates the part of the answer that is the criterion
# from the part that is the search.
#
# WHAT THIS DOES NOT DO, DELIBERATELY. It does not build a synthetic state out of the votes. Doing
# that means constructing a graph by hand, which is where E47 and E48 previously went wrong twice:
# a hand-built state set a node's presence from cell ownership rather than membership and claimed
# cells the engine leaves in the background, costing 1,900 to 2,400 bits and inverting a published
# conclusion. Every state returned here is one the engine actually produced. The votes are used to
# report agreement, and the shortest description picks which real state to return, which is the
# selector the protocol already uses and adds no new rule.
# --------------------------------------------------------------------------- #

def run_consensus(records, R=3, every=REPAIR_EVERY, objective_cls=BatchObjective,
                  cand_pool_cap=None, seed=0):
    """Run R arrival orders of the same records and report what they agree on.

    Returns a dict with the state that has the shortest description, and the agreement each record
    and each pair of records commanded across the R runs. Agreement is measured on co-placement
    rather than on node identity, because node 4 in one order is not node 4 in another, and
    co-placement is also what the reported metrics score.

    `R` is a protocol quantity, like the repair cadence. It is declared, not calibrated: it does not
    appear in the criterion, and E59 measures what it costs.
    """
    import collections
    import random

    n = len(records)
    runs, labels = [], []
    for r in range(R):
        order = list(range(n))
        random.Random(seed * 1_000_003 + r).shuffle(order)
        g, b = run_stream([records[i] for i in order], every=every,
                          objective_cls=objective_cls, cand_pool_cap=cand_pool_cap)
        lab_in_run_order = record_labels(g, n)
        lab = [None] * n
        for pos, orig in enumerate(order):
            lab[orig] = lab_in_run_order[pos]
        runs.append({"order_index": r, "graph": g, "objective": b,
                     "bits": b.total(), "K": int(g.K), "labels": lab})
        labels.append(lab)

    together = collections.Counter()
    apart = collections.Counter()
    for lab in labels:
        by = collections.defaultdict(list)
        for i, l in enumerate(lab):
            if l is not None and l != -1:
                by[l].append(i)
        seen_together = set()
        for members in by.values():
            for a in range(len(members)):
                for b_ in range(a + 1, len(members)):
                    p = (members[a], members[b_])
                    together[p] += 1
                    seen_together.add(p)
        for p in list(together):
            if p not in seen_together:
                apart[p] += 1

    # per record: how often it was placed in a node at all, rather than left in the background
    placed = [sum(1 for lab in labels if lab[i] not in (None, -1)) for i in range(n)]
    best = min(runs, key=lambda r: r["bits"])
    return {
        "R": R,
        "runs": runs,
        "best": best,
        "bits_spread": (min(r["bits"] for r in runs), max(r["bits"] for r in runs)),
        "K_by_order": [r["K"] for r in runs],
        "pair_agreement": together,          # (i, j) -> orders that co-placed them
        "records_placed_in_a_node": placed,  # index -> orders that gave it a node
        "protocol": describe(every, objective_cls) + f"; {R} arrival orders, shortest kept",
    }


def consensus_confidence(cons):
    """Per-record confidence, with nothing to set.

    A record placed in a node by every order is one the criterion is sure about; a record placed by
    some orders and not others is one where the search, not the criterion, is deciding. The split is
    the confidence, and it needs no level because it is a count out of R.
    """
    R = cons["R"]
    placed = cons["records_placed_in_a_node"]
    return {
        "unanimously_placed": sum(1 for p in placed if p == R),
        "unanimously_background": sum(1 for p in placed if p == 0),
        "split": sum(1 for p in placed if 0 < p < R),
        "of": len(placed),
        "fraction_decided": (sum(1 for p in placed if p in (0, R)) / len(placed)) if placed else 1.0,
    }
