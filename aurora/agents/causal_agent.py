"""
Causal Inference Agent
Constructs causal dependency graphs
"""
from typing import Dict, List
import pandas as pd
import structlog

from aurora.causal_inference.rcd_algorithm import HierarchicalRCD
from aurora.causal_inference.neural_granger import NeuralGrangerCausality

logger = structlog.get_logger()

class CausalAgent:
    """Specialist agent for causal inference"""

    def __init__(self):
        self.rcd = HierarchicalRCD()
        self.neural_granger = NeuralGrangerCausality()
        logger.info("causal_agent_initialized")

    async def infer_graph(
        self,
        normal_data: pd.DataFrame,
        failure_data: pd.DataFrame,
        failure_node: str
    ) -> Dict:
        """Construct causal graph"""

        logger.info("inferring_causal_graph", 
                   failure_node=failure_node)

        # Use RCD for discovery
        root_causes = self.rcd.discover_root_causes(
            normal_data,
            failure_data,
            failure_node,
            top_k=5
        )

        # Build graph structure
        graph = {
            "nodes": list(normal_data.columns),
            "edges": self._extract_edges(root_causes, failure_node),
            "root_causes": root_causes
        }

        return graph

    def _extract_edges(self, root_causes: List, failure_node: str) -> List:
        """Extract edges from root causes"""
        edges = []
        for cause, score in root_causes:
            edges.append({
                "source": cause,
                "target": failure_node,
                "weight": score
            })
        return edges
