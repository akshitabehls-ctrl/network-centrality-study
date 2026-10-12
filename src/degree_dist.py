"""
degree_dist.py - Degree Distribution Analysis and Power-Law Fitting Module
===========================================================================
Implements empirical degree distribution extraction, complementary cumulative
distribution function (CCDF), theoretical Poisson comparison, Clauset-Shalizi-Newman
power-law maximum likelihood estimation (MLE), 100-realization ensemble averaging,
and N in {100, 1000, 10000} finite-size scaling analysis.
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import factorial
from typing import Dict, Any, List, Tuple


def compute_degree_distribution(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes degree sequence, unique degrees, empirical PMF P(k), and CCDF P(K >= k).
    """
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    max_k = np.max(degrees)
    k_vals = np.arange(0, max_k + 1)
    
    # Counts and probabilities
    counts = np.bincount(degrees, minlength=max_k + 1)
    p_k = counts / len(degrees)
    
    # CCDF: P(K >= k) = sum_{j >= k} p_k[j]
    ccdf = np.cumsum(p_k[::-1])[::-1]
    
    # Mask to non-zero probabilities for log-log plotting
    nz_mask = p_k > 0
    k_nz = k_vals[nz_mask]
    p_k_nz = p_k[nz_mask]
    
    return degrees, k_vals, p_k, ccdf


def theoretical_poisson_pmf(k_vals: np.ndarray, lam: float) -> np.ndarray:
    """
    Computes theoretical Poisson PMF: P(k) = (lam^k * exp(-lam)) / k!
    """
    k_float = np.asarray(k_vals, dtype=float)
    # Using exp(k * ln(lam) - lam - ln(gamma(k+1))) for numerical stability
    from scipy.special import gammaln
    log_pmf = k_float * np.log(lam) - lam - gammaln(k_float + 1.0)
    return np.exp(log_pmf)


def analytical_ba_pmf(k_vals: np.ndarray, m: int = 2) -> np.ndarray:
    """
    Computes exact stationary master-equation solution for BA model:
    P(k) = 2 * m * (m + 1) / (k * (k + 1) * (k + 2)) for k >= m.
    """
    pmf = np.zeros_like(k_vals, dtype=float)
    valid = k_vals >= m
    k_v = k_vals[valid].astype(float)
    pmf[valid] = (2.0 * m * (m + 1.0)) / (k_v * (k_v + 1.0) * (k_v + 2.0))
    return pmf


def fit_power_law(degrees: np.ndarray) -> Dict[str, Any]:
    """
    Fits power-law distribution to degree sequence using Clauset-Shalizi-Newman (CSN)
    maximum likelihood methodology via `powerlaw` package (with fallback).
    """
    degrees_clean = degrees[degrees > 0]
    result = {
        "gamma": None,
        "xmin": None,
        "loglikelihood_ratio": None,
        "p_value": None,
        "method": "powerlaw_package"
    }
    
    try:
        import powerlaw
        fit = powerlaw.Fit(degrees_clean, discrete=True, verbose=False)
        result["gamma"] = float(fit.power_law.alpha)
        result["xmin"] = float(fit.power_law.xmin)
        
        # Compare power law vs exponential distribution
        R, p = fit.distribution_compare("power_law", "exponential")
        result["loglikelihood_ratio"] = float(R)
        result["p_value"] = float(p)
    except Exception as e:
        # Clauset et al. analytic discrete approximation fallback
        k_min = int(np.min(degrees_clean))
        n_sample = len(degrees_clean)
        gamma_approx = 1.0 + n_sample / np.sum(np.log(degrees_clean / (k_min - 0.5)))
        result["gamma"] = float(gamma_approx)
        result["xmin"] = float(k_min)
        result["method"] = f"analytic_mle_fallback: {e}"
        result["loglikelihood_ratio"] = 0.0
        result["p_value"] = 1.0
        
    return result


def ensemble_degree_distributions(n: int = 100, m_ba: int = 2, num_realizations: int = 100, base_seed: int = 42) -> Dict[str, Any]:
    """
    Runs an ensemble of 100 realizations of both BA and ER networks to produce
    statistically smooth degree distributions with confidence intervals.
    """
    num_edges = m_ba * (n - m_ba)
    max_k_eval = 60
    k_grid = np.arange(0, max_k_eval + 1)
    
    ba_histograms = np.zeros((num_realizations, len(k_grid)))
    er_histograms = np.zeros((num_realizations, len(k_grid)))
    
    ba_ccdfs = np.zeros((num_realizations, len(k_grid)))
    er_ccdfs = np.zeros((num_realizations, len(k_grid)))
    
    for i in range(num_realizations):
        seed = base_seed + i * 101
        G_ba = nx.barabasi_albert_graph(n=n, m=m_ba, seed=seed)
        G_er = nx.gnm_random_graph(n=n, m=num_edges, seed=seed)
        
        # BA
        deg_ba = np.array([d for _, d in G_ba.degree()], dtype=int)
        counts_ba = np.bincount(deg_ba, minlength=len(k_grid))[:len(k_grid)]
        p_ba = counts_ba / n
        ba_histograms[i, :] = p_ba
        ba_ccdfs[i, :] = np.cumsum(p_ba[::-1])[::-1]
        
        # ER
        deg_er = np.array([d for _, d in G_er.degree()], dtype=int)
        counts_er = np.bincount(deg_er, minlength=len(k_grid))[:len(k_grid)]
        p_er = counts_er / n
        er_histograms[i, :] = p_er
        er_ccdfs[i, :] = np.cumsum(p_er[::-1])[::-1]
        
    return {
        "k_grid": k_grid,
        "ba_mean_p": np.mean(ba_histograms, axis=0),
        "ba_std_p": np.std(ba_histograms, axis=0),
        "ba_mean_ccdf": np.mean(ba_ccdfs, axis=0),
        "ba_std_ccdf": np.std(ba_ccdfs, axis=0),
        "er_mean_p": np.mean(er_histograms, axis=0),
        "er_std_p": np.std(er_histograms, axis=0),
        "er_mean_ccdf": np.mean(er_ccdfs, axis=0),
        "er_std_ccdf": np.std(er_ccdfs, axis=0),
        "num_realizations": num_realizations
    }


def scaling_study(n_list: List[int] = [100, 1000, 10000], m_ba: int = 2, seed: int = 42) -> Dict[int, Dict[str, Any]]:
    """
    Evaluates degree distributions across multiple network orders N in {100, 1000, 10000}
    to observe finite-size scaling and the emergence of the asymptotic power law.
    """
    scaling_results = {}
    for n in n_list:
        G_ba = nx.barabasi_albert_graph(n=n, m=m_ba, seed=seed)
        degrees, k_vals, p_k, ccdf = compute_degree_distribution(G_ba)
        fit_res = fit_power_law(degrees)
        
        scaling_results[n] = {
            "n": n,
            "max_degree": int(np.max(degrees)),
            "degrees": degrees,
            "k_vals": k_vals,
            "p_k": p_k,
            "ccdf": ccdf,
            "fit": fit_res
        }
    return scaling_results
