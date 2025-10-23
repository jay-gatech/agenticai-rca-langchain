"""
Graph manipulation utilities
"""
import networkx as nx
import numpy as np
from typing import List, Dict

def adjacency_to_networkx(adj_matrix: np.ndarray, node_names: List[str]) -> nx.DiGraph:
    """Convert adjacency matrix to NetworkX graph"""
    G = nx.DiGraph()
    G.add_nodes_from(node_names)

    for i, source in enumerate(node_names):
        for j, target in enumerate(node_names):
            if adj_matrix[i, j] == 1:
                G.add_edge(source, target)

    return G

def compute_graph_metrics(G: nx.DiGraph) -> Dict:
    """Compute various graph metrics"""
    return {{
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
    }}
