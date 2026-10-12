"""
plots.py - Publication-Quality Figure Generation Module
=======================================================
Generates all 11 scientific figures for the BA vs ER comparative study:
1. Fig 1: Side-by-side Degree Histograms
2. Fig 2: ER Degree Distribution vs Theoretical Poisson Overlay
3. Fig 3: Empirical Degree Distribution Log-Log Plot (BA vs ER)
4. Fig 4: Log-Log Complementary Cumulative Distribution Function (CCDF)
5. Fig 5: 100-Realization Ensemble Average with Confidence Envelope
6. Fig 6: Finite-Size Scaling Study (N in {100, 1000, 10000})
7. Fig 7: Comparative Centrality Boxplots with Gini Coefficients
8. Fig 8: Spearman Centrality Correlation Heatmaps (BA & ER)
9. Fig 9: First-Mover Advantage (Node Creation Index vs Degree & Betweenness)
10. Fig 10: Spring Layout Network Topologies (Degree Size + Betweenness Color)
11. Fig 11: Network Robustness Under Targeted Attack vs Random Failure
"""

import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple


# Set scientific styling defaults
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.1,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

COLOR_BA = "#D95F02"   # Distinct Vermillion / Coral
COLOR_ER = "#1B9E77"   # Distinct Deep Teal / Green
COLOR_THEORY = "#7570B3" # Purple for theory/overlays


def save_fig(fig: plt.Figure, filename: str, figures_dir: str = "figures"):
    """Saves figure in both PNG (300 DPI) and PDF for publication and report integration."""
    os.makedirs(figures_dir, exist_ok=True)
    png_path = os.path.join(figures_dir, f"{filename}.png")
    pdf_path = os.path.join(figures_dir, f"{filename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path} & {pdf_path}")


def plot_degree_histograms(deg_ba: np.ndarray, deg_er: np.ndarray, figures_dir: str = "figures"):
    """Fig 1: Side-by-side Linear Degree Histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    
    # ER plot
    bins_er = np.arange(np.min(deg_er) - 0.5, np.max(deg_er) + 1.5, 1)
    ax1.hist(deg_er, bins=bins_er, color=COLOR_ER, edgecolor="black", alpha=0.85, rwidth=0.85)
    ax1.axvline(np.mean(deg_er), color="#004D40", linestyle="--", linewidth=1.8, label=f"Mean = {np.mean(deg_er):.2f}")
    ax1.set_title("Erdős–Rényi Random Network (G(n, M))\nBinomial/Poisson Bell-Shaped Distribution", fontweight="bold")
    ax1.set_xlabel("Degree $k$")
    ax1.set_ylabel("Node Frequency")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    # BA plot
    bins_ba = np.arange(np.min(deg_ba) - 0.5, np.max(deg_ba) + 1.5, 1)
    ax2.hist(deg_ba, bins=bins_ba, color=COLOR_BA, edgecolor="black", alpha=0.85, rwidth=0.85)
    ax2.axvline(np.mean(deg_ba), color="#802000", linestyle="--", linewidth=1.8, label=f"Mean = {np.mean(deg_ba):.2f}")
    ax2.axvline(np.max(deg_ba), color="#B71C1C", linestyle=":", linewidth=1.8, label=f"Max Hub = {np.max(deg_ba)}")
    ax2.set_title("Barabási–Albert Network (BA)\nHeavy-Tailed Scale-Free Distribution", fontweight="bold")
    ax2.set_xlabel("Degree $k$")
    ax2.grid(True)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    save_fig(fig, "fig1_degree_histograms", figures_dir)


def plot_er_poisson_overlay(deg_er: np.ndarray, theoretical_pmf: np.ndarray, k_vals: np.ndarray, figures_dir: str = "figures"):
    """Fig 2: ER Distribution with Theoretical Poisson Overlay."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    counts = np.bincount(deg_er, minlength=len(k_vals))[:len(k_vals)]
    empirical_p = counts / len(deg_er)
    lam = np.mean(deg_er)
    
    ax.bar(k_vals[:len(empirical_p)], empirical_p, width=0.6, color=COLOR_ER, alpha=0.7, edgecolor="black", label=f"Empirical ER ($N={len(deg_er)}, M=196$)")
    ax.plot(k_vals, theoretical_pmf, color=COLOR_THEORY, marker="o", linewidth=2.2, markersize=6, label=rf"Theoretical Poisson ($\lambda = \langle k \rangle = {lam:.2f}$)")
    
    ax.set_title("Erdos-Renyi Degree Distribution vs. Theoretical Poisson Model", fontweight="bold")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Probability $P(k)$")
    ax.set_xlim(-0.5, max(12, np.max(deg_er) + 2))
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    save_fig(fig, "fig2_er_poisson_overlay", figures_dir)


def plot_loglog_degree_dist(k_nz_ba: np.ndarray, p_ba: np.ndarray, k_nz_er: np.ndarray, p_er: np.ndarray, gamma_est: float, figures_dir: str = "figures"):
    """Fig 3: Log-log Degree Distribution P(k) vs k."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    ax.scatter(k_nz_ba, p_ba, color=COLOR_BA, s=45, label="Barabasi-Albert (BA) Empirical", zorder=3)
    ax.scatter(k_nz_er, p_er, color=COLOR_ER, s=45, marker="s", label="Erdos-Renyi (ER) Empirical", zorder=3)
    
    # Plot reference power law guide line slope = -3 (and fitted gamma)
    k_line = np.linspace(np.min(k_nz_ba), np.max(k_nz_ba), 50)
    c_fit = p_ba[0] * (k_nz_ba[0] ** (gamma_est if gamma_est else 3.0))
    p_fit = c_fit * (k_line ** (-(gamma_est if gamma_est else 3.0)))
    gamma_label = rf"BA Fitted Power Law $\gamma \approx {gamma_est:.2f}$" if gamma_est else r"BA Slope $\sim -3$"
    ax.plot(k_line, p_fit, color=COLOR_BA, linestyle="--", linewidth=1.8, label=gamma_label)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Degree Distribution $P(k)$ on Log–Log Scale", fontweight="bold")
    ax.set_xlabel("Degree $k$ (log scale)")
    ax.set_ylabel("Probability $P(k)$ (log scale)")
    ax.grid(True, which="both", ls="--")
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    save_fig(fig, "fig3_loglog_degree_dist", figures_dir)


def plot_ccdf_loglog(deg_ba: np.ndarray, deg_er: np.ndarray, figures_dir: str = "figures"):
    """Fig 4: Complementary Cumulative Distribution Function (CCDF) on Log-Log Axes."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    # BA CCDF
    sorted_ba = np.sort(deg_ba)
    ccdf_ba = 1.0 - np.arange(len(sorted_ba)) / len(sorted_ba)
    ax.plot(sorted_ba, ccdf_ba, drawstyle="steps-post", color=COLOR_BA, linewidth=2.2, label="Barabasi-Albert CCDF (Heavy Tail)")
    
    # ER CCDF
    sorted_er = np.sort(deg_er)
    ccdf_er = 1.0 - np.arange(len(sorted_er)) / len(sorted_er)
    ax.plot(sorted_er, ccdf_er, drawstyle="steps-post", color=COLOR_ER, linewidth=2.2, label="Erdos-Renyi CCDF (Exponential Decay)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Complementary CDF $P(K \geq k)$ on Log-Log Scale", fontweight="bold")
    ax.set_xlabel("Degree $k$ (log scale)")
    ax.set_ylabel(r"Pr($K \geq k$) (log scale)")
    ax.grid(True, which="both", ls="--")
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    save_fig(fig, "fig4_ccdf_loglog", figures_dir)


def plot_ensemble_degree_dist(ensemble_data: Dict[str, Any], figures_dir: str = "figures"):
    """Fig 5: 100-Realization Ensemble Average with Standard Deviation Envelope."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    k = ensemble_data["k_grid"]
    mask = k >= 1
    k_eval = k[mask]
    
    # 1. PMF Ensemble
    ba_mean = ensemble_data["ba_mean_p"][mask]
    ba_std = ensemble_data["ba_std_p"][mask]
    er_mean = ensemble_data["er_mean_p"][mask]
    er_std = ensemble_data["er_std_p"][mask]
    
    ax1.plot(k_eval, ba_mean, color=COLOR_BA, linewidth=2, label="BA Ensemble Mean")
    ax1.fill_between(k_eval, np.maximum(1e-5, ba_mean - ba_std), ba_mean + ba_std, color=COLOR_BA, alpha=0.25)
    
    ax1.plot(k_eval, er_mean, color=COLOR_ER, linewidth=2, label="ER Ensemble Mean")
    ax1.fill_between(k_eval, np.maximum(1e-5, er_mean - er_std), er_mean + er_std, color=COLOR_ER, alpha=0.25)
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 0.5)
    ax1.set_title("Ensemble $P(k)$ (100 Realizations, $N=100$)", fontweight="bold")
    ax1.set_xlabel("Degree $k$ (log scale)")
    ax1.set_ylabel("$P(k)$ (log scale)")
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    
    # 2. CCDF Ensemble
    ba_ccdf_m = ensemble_data["ba_mean_ccdf"][mask]
    ba_ccdf_s = ensemble_data["ba_std_ccdf"][mask]
    er_ccdf_m = ensemble_data["er_mean_ccdf"][mask]
    er_ccdf_s = ensemble_data["er_std_ccdf"][mask]
    
    ax2.plot(k_eval, ba_ccdf_m, color=COLOR_BA, linewidth=2, label="BA Ensemble CCDF")
    ax2.fill_between(k_eval, np.maximum(1e-5, ba_ccdf_m - ba_ccdf_s), ba_ccdf_m + ba_ccdf_s, color=COLOR_BA, alpha=0.25)
    
    ax2.plot(k_eval, er_ccdf_m, color=COLOR_ER, linewidth=2, label="ER Ensemble CCDF")
    ax2.fill_between(k_eval, np.maximum(1e-5, er_ccdf_m - er_ccdf_s), er_ccdf_m + er_ccdf_s, color=COLOR_ER, alpha=0.25)
    
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 1.1)
    ax2.set_title(r"Ensemble CCDF $P(K \geq k)$ ($\pm 1\sigma$ Band)", fontweight="bold")
    ax2.set_xlabel("Degree $k$ (log scale)")
    ax2.set_ylabel(r"Pr($K \geq k$) (log scale)")
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    plt.tight_layout()
    save_fig(fig, "fig5_ensemble_averaging", figures_dir)


def plot_scaling_study(scaling_data: Dict[int, Dict[str, Any]], figures_dir: str = "figures"):
    """Fig 6: Finite-Size Scaling Study for N in {100, 1000, 10000}."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    palette = [COLOR_BA, "#E7298A", "#7570B3"]
    
    for (n, res), color in zip(scaling_data.items(), palette):
        deg = res["degrees"]
        sorted_d = np.sort(deg)
        ccdf = 1.0 - np.arange(len(sorted_d)) / len(sorted_d)
        gamma = res["fit"]["gamma"]
        max_k = res["max_degree"]
        gamma_val = f"{gamma:.2f}" if gamma is not None else "N/A"
        label = rf"$N={n:,}$ (Max $k={max_k}$, $\hat{{\gamma}}={gamma_val}$)"
        ax.plot(sorted_d, ccdf, drawstyle="steps-post", color=color, linewidth=2.2, label=label)
    
    # Plot asymptotic theoretical slope P(K >= k) ~ k^-(gamma - 1) = k^-2
    k_ref = np.linspace(3, 100, 50)
    ccdf_ref = 5.0 * (k_ref ** -2.0)
    ax.plot(k_ref, ccdf_ref, "k--", linewidth=1.5, label=r"Asymptotic Theory $\sim k^{-2}$ (Master Eq.)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.1)
    ax.set_title("Finite-Size Scaling of BA Network Degree Distributions", fontweight="bold")
    ax.set_xlabel("Degree $k$ (log scale)")
    ax.set_ylabel(r"Complementary CDF $P(K \geq k)$ (log scale)")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    
    plt.tight_layout()
    save_fig(fig, "fig6_scaling_study", figures_dir)


def plot_centrality_distributions(df_ba: pd.DataFrame, df_er: pd.DataFrame, meta_ba: Dict[str, Any], meta_er: Dict[str, Any], figures_dir: str = "figures"):
    """Fig 7: Boxplots of Centrality Distributions comparing BA vs ER with Gini Inequality."""
    metrics = ["degree_c", "closeness", "harmonic", "betweenness", "eigenvector", "katz", "pagerank", "load", "subgraph"]
    titles = ["Degree", "Closeness", "Harmonic", "Betweenness", "Eigenvector", "Katz", "PageRank", "Load", "Subgraph (log)"]
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    
    for idx, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        v_ba = df_ba[m].values
        v_er = df_er[m].values
        
        # Log scale for subgraph due to extreme exponential span
        if m == "subgraph":
            v_ba = np.log10(np.maximum(1.0, v_ba))
            v_er = np.log10(np.maximum(1.0, v_er))
            
        plot_data = pd.DataFrame({
            "Value": np.concatenate([v_ba, v_er]),
            "Model": ["BA"] * len(v_ba) + ["ER"] * len(v_er)
        })
        
        sns.boxplot(x="Model", y="Value", data=plot_data, ax=ax, palette=[COLOR_BA, COLOR_ER], width=0.45, fliersize=3)
        
        g_ba = meta_ba["gini_coefficients"].get(m, 0.0)
        g_er = meta_er["gini_coefficients"].get(m, 0.0)
        
        ax.set_title(f"{title}\n$G_{{BA}}={g_ba:.3f}$ | $G_{{ER}}={g_er:.3f}$", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Centrality Value" if idx % 3 == 0 else "")
        ax.grid(True, axis="y")
        
    plt.suptitle("Centrality Distributions & Inequality (Gini Index): Barabási–Albert vs. Erdős–Rényi", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_fig(fig, "fig7_centrality_boxplots", figures_dir)


def plot_correlation_heatmaps(corr_ba: pd.DataFrame, corr_er: pd.DataFrame, figures_dir: str = "figures"):
    """Fig 8: Side-by-side Spearman Rank Correlation Heatmaps."""
    cols = ["degree", "closeness", "harmonic", "betweenness", "eigenvector", "katz", "pagerank", "clustering", "core_number"]
    c_ba = corr_ba.loc[cols, cols]
    c_er = corr_er.loc[cols, cols]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(c_ba, ax=ax1, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.2, vmax=1.0, cbar_kws={"label": "Spearman Correlation $\\rho$"}, square=True, annot_kws={"size": 8})
    ax1.set_title("Barabási–Albert Centrality Correlations\n(Hubs Dominate Shortest Paths & Walks)", fontweight="bold")
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    sns.heatmap(c_er, ax=ax2, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.2, vmax=1.0, cbar_kws={"label": "Spearman Correlation $\\rho$"}, square=True, annot_kws={"size": 8})
    ax2.set_title("Erdős–Rényi Centrality Correlations\n(Decoupled Topology & Homogeneous Roles)", fontweight="bold")
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    save_fig(fig, "fig8_correlation_heatmaps", figures_dir)


def plot_first_mover_advantage(df_ba: pd.DataFrame, figures_dir: str = "figures"):
    """Fig 9: Demonstrates Preferential Attachment First-Mover Advantage (Node Index vs Degree & Betweenness)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    node_idx = df_ba.index.values
    degree = df_ba["degree"].values
    betweenness = df_ba["betweenness"].values
    
    # 1. Degree vs Index
    ax1.scatter(node_idx, degree, color=COLOR_BA, edgecolors="black", s=50, alpha=0.85)
    # Fit linear/log trend line
    p_fit = np.polyfit(np.log(node_idx + 1), degree, 1)
    x_grid = np.linspace(0, 99, 100)
    ax1.plot(x_grid, p_fit[0] * np.log(x_grid + 1) + p_fit[1], color="#802000", linestyle="--", linewidth=2, label=r"Log Trend: $k(t) \propto (t/t_i)^{-1/2}$")
    
    ax1.set_title("First-Mover Advantage: Node Degree vs. Arrival Order", fontweight="bold")
    ax1.set_xlabel("Node Index $i$ (Arrival Timestamp in BA Growth)")
    ax1.set_ylabel("Degree $k_i$")
    ax1.grid(True)
    ax1.legend()
    
    # 2. Betweenness vs Index
    ax2.scatter(node_idx, betweenness, color="#7570B3", edgecolors="black", s=50, alpha=0.85)
    ax2.set_title("First-Mover Advantage: Betweenness vs. Arrival Order", fontweight="bold")
    ax2.set_xlabel("Node Index $i$ (Arrival Timestamp in BA Growth)")
    ax2.set_ylabel("Betweenness Centrality $C_B(i)$")
    ax2.grid(True)
    
    plt.tight_layout()
    save_fig(fig, "fig9_first_mover_advantage", figures_dir)


def plot_network_visualizations(G_ba: nx.Graph, G_er: nx.Graph, df_ba: pd.DataFrame, df_er: pd.DataFrame, figures_dir: str = "figures"):
    """Fig 10: Spring Layout Network Topologies with Degree Size & Betweenness Heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # Spring layout with fixed seeds for fair reproducibility
    pos_ba = nx.spring_layout(G_ba, seed=42, k=0.35)
    pos_er = nx.spring_layout(G_er, seed=42, k=0.35)
    
    # BA Visualization
    deg_ba = np.array([df_ba.loc[n, "degree"] for n in G_ba.nodes()])
    bet_ba = np.array([df_ba.loc[n, "betweenness"] for n in G_ba.nodes()])
    node_sizes_ba = 30 + 35 * deg_ba
    
    nx.draw_networkx_edges(G_ba, pos_ba, ax=ax1, alpha=0.25, edge_color="gray", width=0.8)
    nodes1 = nx.draw_networkx_nodes(G_ba, pos_ba, ax=ax1, node_size=node_sizes_ba, node_color=bet_ba, cmap="plasma", alpha=0.92, edgecolors="black", linewidths=0.5)
    cbar1 = plt.colorbar(nodes1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Betweenness Centrality", fontsize=9.5)
    ax1.set_title("Barabási–Albert Scale-Free Topology\n(Clear Core Hubs & Periphery)", fontweight="bold")
    ax1.axis("off")
    
    # ER Visualization
    deg_er = np.array([df_er.loc[n, "degree"] for n in G_er.nodes()])
    bet_er = np.array([df_er.loc[n, "betweenness"] for n in G_er.nodes()])
    node_sizes_er = 30 + 35 * deg_er
    
    nx.draw_networkx_edges(G_er, pos_er, ax=ax2, alpha=0.25, edge_color="gray", width=0.8)
    nodes2 = nx.draw_networkx_nodes(G_er, pos_er, ax=ax2, node_size=node_sizes_er, node_color=bet_er, cmap="plasma", alpha=0.92, edgecolors="black", linewidths=0.5)
    cbar2 = plt.colorbar(nodes2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Betweenness Centrality", fontsize=9.5)
    ax2.set_title("Erdős–Rényi Random Topology\n(Homogeneous Distributed Connectivity)", fontweight="bold")
    ax2.axis("off")
    
    plt.tight_layout()
    save_fig(fig, "fig10_network_topology", figures_dir)


def simulate_network_robustness(G: nx.Graph, attack_type: str = "targeted", n_trials: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs node percolation removal, tracking Giant Connected Component (GCC) size fraction.
    'targeted': remove in descending order of node degree.
    'random': remove in random order (averaged over n_trials).
    """
    n_total = G.number_of_nodes()
    fractions = np.linspace(0.0, 0.99, n_total)
    
    if attack_type == "targeted":
        # Sort nodes by degree descending
        ordered_nodes = [node for node, _ in sorted(G.degree(), key=lambda item: item[1], reverse=True)]
        H = G.copy()
        gcc_fractions = [len(max(nx.connected_components(H), key=len)) / n_total if H.number_of_nodes() > 0 else 0.0]
        
        for node in ordered_nodes[:-1]:
            H.remove_node(node)
            if H.number_of_nodes() > 0 and H.number_of_edges() > 0:
                gcc = len(max(nx.connected_components(H), key=len)) / n_total
            elif H.number_of_nodes() > 0:
                gcc = 1.0 / n_total
            else:
                gcc = 0.0
            gcc_fractions.append(gcc)
        return fractions, np.array(gcc_fractions)
        
    elif attack_type == "random":
        all_trials = []
        for trial in range(n_trials):
            np.random.seed(100 + trial)
            rand_nodes = list(np.random.permutation(list(G.nodes())))
            H = G.copy()
            trial_gcc = [len(max(nx.connected_components(H), key=len)) / n_total if H.number_of_nodes() > 0 else 0.0]
            
            for node in rand_nodes[:-1]:
                H.remove_node(node)
                if H.number_of_nodes() > 0 and H.number_of_edges() > 0:
                    gcc = len(max(nx.connected_components(H), key=len)) / n_total
                elif H.number_of_nodes() > 0:
                    gcc = 1.0 / n_total
                else:
                    gcc = 0.0
                trial_gcc.append(gcc)
            all_trials.append(trial_gcc)
        return fractions, np.mean(all_trials, axis=0)


def plot_robustness_experiment(G_ba: nx.Graph, G_er: nx.Graph, figures_dir: str = "figures"):
    """Fig 11: Network Robustness Under Targeted Attack vs Random Failure (Percolation)."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    
    f_ba_t, s_ba_t = simulate_network_robustness(G_ba, attack_type="targeted")
    f_ba_r, s_ba_r = simulate_network_robustness(G_ba, attack_type="random", n_trials=30)
    f_er_t, s_er_t = simulate_network_robustness(G_er, attack_type="targeted")
    f_er_r, s_er_r = simulate_network_robustness(G_er, attack_type="random", n_trials=30)
    
    ax.plot(f_ba_t, s_ba_t, color=COLOR_BA, linestyle="-", linewidth=2.4, label="BA: Targeted Hub Attack (Fragile)")
    ax.plot(f_ba_r, s_ba_r, color=COLOR_BA, linestyle="--", linewidth=2.0, label="BA: Random Node Failure (Robust)")
    ax.plot(f_er_t, s_er_t, color=COLOR_ER, linestyle="-", linewidth=2.4, label="ER: Targeted Node Attack")
    ax.plot(f_er_r, s_er_r, color=COLOR_ER, linestyle="--", linewidth=2.0, label="ER: Random Node Failure")
    
    ax.set_title("Network Robustness: Targeted Attack vs. Random Failure\n('Robust yet Fragile' Scale-Free Phenomenon)", fontweight="bold")
    ax.set_xlabel("Fraction of Removed Nodes $f$")
    ax.set_ylabel("Relative Giant Component Size $S(f) / N_0$")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    save_fig(fig, "fig11_robustness_percolation", figures_dir)
