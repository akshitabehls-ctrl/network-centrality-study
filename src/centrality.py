"""
centrality.py - Comprehensive Centrality and Metric Computation Module
=======================================================================
Computes standard and spectral centrality measures for network nodes:
- Degree, Degree Centrality
- Closeness (with Wasserman-Faust correction)
- Harmonic Centrality (disconnected-robust)
- Betweenness Centrality
- Eigenvector Centrality (spectral decomposition)
- Katz Centrality (with safe attenuation factor alpha < 1 / lambda_max)
- PageRank Centrality
- Load Centrality
- Subgraph Centrality
- Supporting: Local Clustering Coefficient, K-Core Number

Also includes Gini coefficient calculation for centrality inequality quantification
and Spearman rank correlation analysis.
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy import linalg
from typing import Tuple, Dict, Any


def compute_spectral_radius(G: nx.Graph) -> float:
    """
    Computes the largest eigenvalue (spectral radius) lambda_max of the adjacency matrix.
    Required to safely parameterize Katz centrality attenuation parameter alpha.
    """
    A = nx.to_numpy_array(G)
    eigenvalues = linalg.eigvalsh(A)
    return float(np.max(eigenvalues))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Computes the Gini coefficient of inequality for a distribution of values.
    Returns value in [0, 1], where 0 = perfect equality, 1 = maximum concentration.
    """
    x = np.asarray(values, dtype=float).flatten()
    if len(x) == 0 or np.all(x == 0):
        return 0.0
    # Shift if non-negative requirement
    if np.any(x < 0):
        x = x - np.min(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    gini = float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))
    return max(0.0, min(1.0, gini))


def compute_centralities(G: nx.Graph, alpha_katz_ratio: float = 0.85) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes all specified centrality measures and structural descriptors for graph G.

    Parameters:
    -----------
    G : nx.Graph
        Input network graph.
    alpha_katz_ratio : float
        Ratio of alpha to 1 / lambda_max (must be < 1.0, default 0.85).

    Returns:
    --------
    df : pd.DataFrame
        DataFrame indexed by node ID with all centrality metrics as columns.
    metadata : Dict[str, Any]
        Spectral properties, Katz alpha, Gini coefficients, and summary statistics.
    """
    nodes = sorted(G.nodes())
    df = pd.DataFrame(index=nodes)

    # 1. Degree & Normalized Degree Centrality
    raw_degree = dict(G.degree())
    df["degree"] = pd.Series(raw_degree)
    df["degree_c"] = pd.Series(nx.degree_centrality(G))

    # 2. Closeness (handles components via Wasserman-Faust) & Harmonic (direct sum of 1/d)
    df["closeness"] = pd.Series(nx.closeness_centrality(G))
    df["harmonic"] = pd.Series(nx.harmonic_centrality(G))

    # 3. Betweenness & Load Centrality
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G, normalized=True))
    df["load"] = pd.Series(nx.load_centrality(G, normalized=True))

    # 4. Eigenvector Centrality
    try:
        df["eigenvector"] = pd.Series(nx.eigenvector_centrality_numpy(G))
    except Exception:
        df["eigenvector"] = pd.Series(nx.eigenvector_centrality(G, max_iter=2000, tol=1e-6))

    # 5. Katz Centrality with mathematically validated alpha
    lambda_max = compute_spectral_radius(G)
    alpha_upper_bound = 1.0 / lambda_max if lambda_max > 0 else 0.1
    # Use conservative alpha satisfying alpha < 1/lambda_max
    alpha_katz = float(min(0.05, alpha_upper_bound * alpha_katz_ratio))
    
    try:
        df["katz"] = pd.Series(nx.katz_centrality_numpy(G, alpha=alpha_katz))
    except Exception:
        df["katz"] = pd.Series(nx.katz_centrality(G, alpha=alpha_katz, max_iter=2000))

    # 6. PageRank
    df["pagerank"] = pd.Series(nx.pagerank(G, alpha=0.85))

    # 7. Subgraph Centrality
    df["subgraph"] = pd.Series(nx.subgraph_centrality(G))

    # 8. Structural and topological attributes
    df["clustering"] = pd.Series(nx.clustering(G))
    df["core_number"] = pd.Series(nx.core_number(G))

    # Calculate Gini coefficient for each measure
    gini_dict = {}
    for col in df.columns:
        gini_dict[col] = gini_coefficient(df[col].values)

    # Calculate Spearman correlation matrix
    spearman_corr = df.corr(method="spearman")

    metadata = {
        "lambda_max": lambda_max,
        "alpha_katz_upper_bound": alpha_upper_bound,
        "alpha_katz_chosen": alpha_katz,
        "gini_coefficients": gini_dict,
        "spearman_correlation": spearman_corr,
        "summary_statistics": df.describe().to_dict()
    }

    return df, metadata


def get_top_k_nodes(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Returns a consolidated ranking table showing top-k nodes for key centralities.
    """
    key_metrics = ["degree", "closeness", "harmonic", "betweenness", "eigenvector", "katz", "pagerank", "clustering", "core_number"]
    top_dict = {}
    for metric in key_metrics:
        if metric in df.columns:
            sorted_nodes = df[metric].sort_values(ascending=False).head(k)
            top_dict[f"{metric}_node"] = sorted_nodes.index.tolist()
            top_dict[f"{metric}_val"] = [round(v, 4) for v in sorted_nodes.values]
    return pd.DataFrame(top_dict)
