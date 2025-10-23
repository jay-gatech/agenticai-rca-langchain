"""
PC Algorithm baseline implementation
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
except Exception:  # pragma: no cover
    pc = None
    fisherz = None

class PCAlgorithm:
    """Standard PC algorithm for causal discovery"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def discover_graph(self, data: pd.DataFrame):
        """Discover causal graph using PC algorithm"""
        if pc is None:
            raise ImportError("causallearn is not installed")
        cg = pc(
            data.values,
            alpha=self.alpha,
            indep_test=fisherz,
            stable=True
        )
        return cg.G
