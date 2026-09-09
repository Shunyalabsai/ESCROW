"""EHR05 reference learner. Historical EHR03 remains available by its original class."""
from .engine import HierarchicalEscrowGraph
from .branch_coding import BranchDescription
from .branch_codec import encode
from .branch_proposals import candidates


class BranchEscrowGraph(HierarchicalEscrowGraph):
    description_type = BranchDescription
    encoder = staticmethod(encode)
    code_name = "EHR05"
    proposal_source = staticmethod(candidates)
