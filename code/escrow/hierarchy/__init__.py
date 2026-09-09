"""Reference categorical hierarchy learner with an explicitly decodable score."""

from .model import HierarchyState, HierarchyNode

__all__ = ["HierarchyState", "HierarchyNode", "HierarchicalEscrowGraph", "BranchEscrowGraph"]


def __getattr__(name):
    if name == "HierarchicalEscrowGraph":
        from .engine import HierarchicalEscrowGraph
        return HierarchicalEscrowGraph
    if name == "BranchEscrowGraph":
        from .branch_engine import BranchEscrowGraph
        return BranchEscrowGraph
    raise AttributeError(name)
