"""
generate.py - Controlled Network Generation Module
===================================================
Generates Barabasi-Albert (BA) scale-free network and Erdos-Renyi (ER) random network
under strictly matched size (N=100) and edge count (M=196) constraints.
Handles and reports component structure for downstream centrality robustness.
"""

import networkx as nx
from typing import Tuple, Dict, Any


def generate_networks(n: int = 100, m_ba: int = 2, seed: int = 42) -> Tuple[nx.Graph, nx.Graph, Dict[str, Any]]:
    """
    Generates matched BA and ER networks.

    Parameters:
    -----------
    n : int
        Number of nodes (default: 100).
    m_ba : int
        Number of edges each new node attaches in BA model (default: 2).
    seed : int
        Random seed for full reproducibility (default: 42).

    Returns:
    --------
    G_ba : nx.Graph
        Barabasi-Albert network with N nodes and m*(N - m) edges.
    G_er : nx.Graph
        Erdos-Renyi G(n, M) network with identical node and edge counts.
    diagnostics : Dict[str, Any]
        Structural validation diagnostics including edge counts and component breakdowns.
    """
    # 1. Generate Barabasi-Albert Network (Growth + Preferential Attachment)
    G_ba = nx.barabasi_albert_graph(n=n, m=m_ba, seed=seed)
    num_edges_ba = G_ba.number_of_edges()

    # 2. Generate Erdos-Renyi Network with EXACT edge match G(n, M)
    # Expected p = 2 * M / (n * (n - 1))
    p_equivalent = (2.0 * num_edges_ba) / (n * (n - 1))
    G_er = nx.gnm_random_graph(n=n, m=num_edges_ba, seed=seed)
    num_edges_er = G_er.number_of_edges()

    assert num_edges_ba == num_edges_er, f"Edge mismatch! BA: {num_edges_ba}, ER: {num_edges_er}"

    # 3. Analyze Component Structure & Connectivity
    ba_connected = nx.is_connected(G_ba)
    er_connected = nx.is_connected(G_er)

    ba_components = sorted([len(c) for c in nx.connected_components(G_ba)], reverse=True)
    er_components = sorted([len(c) for c in nx.connected_components(G_er)], reverse=True)

    diagnostics = {
        "n": n,
        "m_edges": num_edges_ba,
        "avg_degree": (2.0 * num_edges_ba) / n,
        "p_equivalent_er": p_equivalent,
        "seed": seed,
        "ba": {
            "is_connected": ba_connected,
            "num_components": len(ba_components),
            "component_sizes": ba_components,
            "lcc_size": ba_components[0] if ba_components else 0,
            "lcc_fraction": (ba_components[0] / n) if ba_components else 0.0,
        },
        "er": {
            "is_connected": er_connected,
            "num_components": len(er_components),
            "component_sizes": er_components,
            "lcc_size": er_components[0] if er_components else 0,
            "lcc_fraction": (er_components[0] / n) if er_components else 0.0,
        }
    }

    return G_ba, G_er, diagnostics


if __name__ == "__main__":
    G_ba, G_er, diag = generate_networks()
    print("Network Generation Diagnostics:")
    print(f"Nodes: {diag['n']}, Edges: {diag['m_edges']}, Mean Degree: {diag['avg_degree']:.3f}")
    print(f"ER Equivalent p: {diag['p_equivalent_er']:.4f}")
    print(f"BA Connected: {diag['ba']['is_connected']} | Components: {diag['ba']['num_components']} | LCC: {diag['ba']['lcc_size']}")
    print(f"ER Connected: {diag['er']['is_connected']} | Components: {diag['er']['num_components']} | Sizes: {diag['er']['component_sizes']}")
