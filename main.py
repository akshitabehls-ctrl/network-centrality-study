"""
main.py - End-to-End Orchestration of Controlled Network Centrality Experiment
=============================================================================
Runs the complete analysis pipeline:
- Phase 2: Fair network generation with matched N=100, M=196, seed=42
- Phase 3: Comprehensive centrality calculations, spectral analysis, Katz alpha verification, Gini inequality
- Phase 4: Degree distribution, Poisson fit, power-law MLE, 100-ensemble averaging, finite-size scaling
- Phase 5: Global structure comparison, Spearman correlations, first-mover advantage, topology visualization, robustness percolation
- Exports all figures to figures/ and structured CSVs to results/
"""

import os
import sys
import json
import networkx as nx
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate import generate_networks
from src.centrality import compute_centralities, get_top_k_nodes, gini_coefficient
from src.degree_dist import (
    compute_degree_distribution,
    theoretical_poisson_pmf,
    fit_power_law,
    ensemble_degree_distributions,
    scaling_study,
)
from src.plots import (
    plot_degree_histograms,
    plot_er_poisson_overlay,
    plot_loglog_degree_dist,
    plot_ccdf_loglog,
    plot_ensemble_degree_dist,
    plot_scaling_study,
    plot_centrality_distributions,
    plot_correlation_heatmaps,
    plot_first_mover_advantage,
    plot_network_visualizations,
    plot_robustness_experiment,
)


def compute_global_structure_table(G_ba: nx.Graph, G_er: nx.Graph) -> pd.DataFrame:
    """
    Computes global topological metrics for BA and ER networks.
    Correctly handles connected components for path lengths and diameter.
    """
    def get_metrics(G: nx.Graph, name: str) -> dict:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        avg_deg = float(np.mean(degrees))
        max_deg = int(np.max(degrees))
        density = float(nx.density(G))
        components = list(nx.connected_components(G))
        num_components = len(components)
        
        # Giant Connected Component
        giant = G.subgraph(max(components, key=len)).copy()
        lcc_size = giant.number_of_nodes()
        
        avg_path = float(nx.average_shortest_path_length(giant))
        diameter = int(nx.diameter(giant))
        avg_clustering = float(nx.average_clustering(G))
        assortativity = float(nx.degree_assortativity_coefficient(G))
        
        return {
            "Network Model": name,
            "Nodes (N)": n,
            "Edges (M)": m,
            "Average Degree <k>": round(avg_deg, 3),
            "Max Degree k_max": max_deg,
            "Density": round(density, 4),
            "Connected Components": num_components,
            "Giant Component Size (N_LCC)": lcc_size,
            "Avg Shortest Path (LCC)": round(avg_path, 3),
            "Diameter (LCC)": diameter,
            "Avg Clustering Coefficient <C>": round(avg_clustering, 4),
            "Degree Assortativity (r)": round(assortativity, 4)
        }
        
    row_ba = get_metrics(G_ba, "Barabási–Albert (BA)")
    row_er = get_metrics(G_er, "Erdős–Rényi (ER)")
    
    return pd.DataFrame([row_ba, row_er])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    print("=" * 80)
    print("  CONTROLLED EXPERIMENT: BARABASI-ALBERT VS. ERDOS-RENYI NETWORK STUDY")
    print("=" * 80)
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------
    # PHASE 2: FAIR NETWORK GENERATION
    # -------------------------------------------------------------
    print("\n[Phase 2] Generating strictly matched networks (N=100, M=196, seed=42)...")
    G_ba, G_er, diag = generate_networks(n=100, m_ba=2, seed=42)
    
    print(f"  * Barabasi-Albert: N={G_ba.number_of_nodes()}, M={G_ba.number_of_edges()}, Connected={diag['ba']['is_connected']}")
    print(f"  * Erdos-Renyi:     N={G_er.number_of_nodes()}, M={G_er.number_of_edges()}, Connected={diag['er']['is_connected']}")
    print(f"    - ER Components breakdown: {diag['er']['component_sizes']}")
    print(f"    - ER Theoretical Equivalent p: {diag['p_equivalent_er']:.4f}")
    
    # -------------------------------------------------------------
    # PHASE 3: COMPREHENSIVE CENTRALITY COMPUTATIONS
    # -------------------------------------------------------------
    print("\n[Phase 3] Computing 9 centralities + 2 structural metrics...")
    df_ba, meta_ba = compute_centralities(G_ba)
    df_er, meta_er = compute_centralities(G_er)
    
    # Save full CSVs
    path_ba_csv = os.path.join(results_dir, "ba_centrality.csv")
    path_er_csv = os.path.join(results_dir, "er_centrality.csv")
    df_ba.to_csv(path_ba_csv, index_label="node_id")
    df_er.to_csv(path_er_csv, index_label="node_id")
    print(f"  * Saved full node centralities: {path_ba_csv} & {path_er_csv}")
    
    # Katz spectral parameter check
    print(f"  * Spectral Radius lambda_max: BA = {meta_ba['lambda_max']:.3f}, ER = {meta_er['lambda_max']:.3f}")
    print(f"  * Katz 1/lambda_max threshold: BA = {meta_ba['alpha_katz_upper_bound']:.4f}, ER = {meta_er['alpha_katz_upper_bound']:.4f}")
    print(f"  * Chosen alpha: BA = {meta_ba['alpha_katz_chosen']:.4f}, ER = {meta_er['alpha_katz_chosen']:.4f}")
    
    # Save Gini coefficients
    gini_df = pd.DataFrame({
        "Centrality Measure": list(meta_ba["gini_coefficients"].keys()),
        "BA Gini Inequality": list(meta_ba["gini_coefficients"].values()),
        "ER Gini Inequality": list(meta_er["gini_coefficients"].values())
    })
    gini_df["Inequality Ratio (BA / ER)"] = (gini_df["BA Gini Inequality"] / np.maximum(1e-5, gini_df["ER Gini Inequality"])).round(2)
    path_gini = os.path.join(results_dir, "centrality_inequality_gini.csv")
    gini_df.to_csv(path_gini, index=False)
    print(f"  * Saved Gini inequality analysis: {path_gini}")
    
    # Save Spearman correlation matrices
    path_corr_ba = os.path.join(results_dir, "ba_spearman_corr.csv")
    path_corr_er = os.path.join(results_dir, "er_spearman_corr.csv")
    meta_ba["spearman_correlation"].to_csv(path_corr_ba)
    meta_er["spearman_correlation"].to_csv(path_corr_er)
    
    # Top-10 / Top-20 ranked nodes table
    top10_ba = get_top_k_nodes(df_ba, k=10)
    top10_er = get_top_k_nodes(df_er, k=10)
    top10_ba.to_csv(os.path.join(results_dir, "ba_top10_nodes.csv"), index=False)
    top10_er.to_csv(os.path.join(results_dir, "er_top10_nodes.csv"), index=False)
    
    # Global structure summary table
    global_table = compute_global_structure_table(G_ba, G_er)
    path_global = os.path.join(results_dir, "global_structure_table.csv")
    global_table.to_csv(path_global, index=False)
    print(f"  * Saved global structure table: {path_global}")
    print("\nGlobal Network Structure Summary:")
    print(global_table.to_string(index=False))

    # -------------------------------------------------------------
    # PHASE 4: DEGREE DISTRIBUTION ANALYSIS & FITTING
    # -------------------------------------------------------------
    print("\n[Phase 4] Degree distribution analysis, power-law fitting, ensemble & scaling...")
    deg_ba, k_ba, p_ba, ccdf_ba = compute_degree_distribution(G_ba)
    deg_er, k_er, p_er, ccdf_er = compute_degree_distribution(G_er)
    
    # Theoretical Poisson
    lam = float(np.mean(deg_er))
    k_range = np.arange(0, max(np.max(deg_er), np.max(deg_ba)) + 2)
    poisson_pmf = theoretical_poisson_pmf(k_range, lam)
    
    # Power-law fitting on BA
    fit_ba = fit_power_law(deg_ba)
    print(f"  * Clauset-Shalizi-Newman Power-Law Fit for BA:")
    print(f"    - Estimated Exponent gamma: {fit_ba['gamma']:.3f} (Theoretical asymptotic = 3.0)")
    print(f"    - Estimated x_min: {fit_ba['xmin']}")
    print(f"    - Log-likelihood Ratio vs Exponential R: {fit_ba['loglikelihood_ratio']:.3f} (p-value: {fit_ba['p_value']:.4f})")
    
    with open(os.path.join(results_dir, "powerlaw_fit_summary.json"), "w") as f:
        json.dump(fit_ba, f, indent=4)
        
    # Ensemble averaging (100 realizations)
    print("  * Computing 100-realization ensemble averaging...")
    ensemble_data = ensemble_degree_distributions(n=100, m_ba=2, num_realizations=100, base_seed=42)
    
    # Finite-size scaling study (N in {100, 1000, 10000})
    print("  * Computing finite-size scaling study (N = 100, 1,000, 10,000)...")
    scaling_data = scaling_study(n_list=[100, 1000, 10000], m_ba=2, seed=42)

    # -------------------------------------------------------------
    # PHASE 5: PUBLICATION FIGURES GENERATION
    # -------------------------------------------------------------
    print("\n[Phase 5] Generating all 11 publication-grade figures...")
    
    # 1. Degree Histograms
    plot_degree_histograms(deg_ba, deg_er, figures_dir)
    
    # 2. ER Poisson Overlay
    plot_er_poisson_overlay(deg_er, poisson_pmf, k_range, figures_dir)
    
    # 3. Log-Log Degree Distribution
    mask_ba = p_ba > 0
    mask_er = p_er > 0
    plot_loglog_degree_dist(k_ba[mask_ba], p_ba[mask_ba], k_er[mask_er], p_er[mask_er], fit_ba["gamma"], figures_dir)
    
    # 4. CCDF Log-Log
    plot_ccdf_loglog(deg_ba, deg_er, figures_dir)
    
    # 5. Ensemble Averaging
    plot_ensemble_degree_dist(ensemble_data, figures_dir)
    
    # 6. Scaling Study
    plot_scaling_study(scaling_data, figures_dir)
    
    # 7. Centrality Distributions (Boxplots + Gini)
    plot_centrality_distributions(df_ba, df_er, meta_ba, meta_er, figures_dir)
    
    # 8. Correlation Heatmaps
    plot_correlation_heatmaps(meta_ba["spearman_correlation"], meta_er["spearman_correlation"], figures_dir)
    
    # 9. First-Mover Advantage
    plot_first_mover_advantage(df_ba, figures_dir)
    
    # 10. Spring Layout Topology Visualizations
    plot_network_visualizations(G_ba, G_er, df_ba, df_er, figures_dir)
    
    # 11. Robustness Percolation Experiment
    plot_robustness_experiment(G_ba, G_er, figures_dir)

    print("\n" + "=" * 80)
    print("  EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  All results saved to: {results_dir}")
    print(f"  All figures saved to: {figures_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
