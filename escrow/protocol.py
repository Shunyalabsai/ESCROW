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
    b.repair(full=True)
    return g, b


def describe(every=REPAIR_EVERY, objective_cls=BatchObjective):
    return (f"EscrowGraph + {objective_cls.__name__}(g).install(); repair() every {every} records; "
            f"final repair(full=True)")
