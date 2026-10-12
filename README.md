# Controlled Network Centrality & Degree Distribution Study: BA vs. Random Networks

A controlled experimental study comparing **Barabási–Albert (Scale-Free)** and **Erdős–Rényi (Random)** networks under strictly matched size ($N=100$) and density ($M=196$, $\langle k \rangle = 3.92$).

---

## 🔬 Core Experimental Methodology

Most comparative network analyses suffer from parameter mismatch, introducing density confounding variables. This project eliminates confounding factors through **controlled experimentation**:

| Network Model | Generator | Nodes ($N$) | Edges ($M$) | Mean Degree $\langle k \rangle$ | Density ($\rho$) | Random Seed |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Barabási–Albert** | `nx.barabasi_albert_graph(100, 2)` | 100 | 196 | 3.920 | 0.0396 | 42 |
| **Erdős–Rényi** | `nx.gnm_random_graph(100, 196)` | 100 | 196 | 3.920 | 0.0396 | 42 |

Because $M$ and $N$ are identical, every topological and spectral difference reflects generative mechanics rather than edge-count artifacts.

---

## 📂 Repository Architecture

```
network-centrality-project/
├── src/
│   ├── generate.py        # Fair network synthesis & component analysis
│   ├── centrality.py      # 9 centrality measures, spectral radius, Gini inequality
│   ├── degree_dist.py     # PMF, CCDF, Poisson overlay, powerlaw MLE, ensemble & scaling
│   └── plots.py           # 11 publication-grade figures (PNG & PDF)
├── results/               # CSV datasets, Gini comparisons, top-10 node rankings
├── figures/               # 11 figures saved in 300 DPI PNG and vector PDF
├── notebooks/             # Interactive Jupyter exploration notebook
├── main.py                # End-to-end execution script
├── REPORT.md              # Comprehensive scientific technical report
├── report.tex             # Complete LaTeX source paper
├── requirements.txt       # Project dependencies
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Packages listed in `requirements.txt`

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. End-to-End Execution
Run the entire experimental pipeline with a single command:
```bash
python main.py
```

All 11 publication figures will be generated in `figures/` and full tabular datasets in `results/`.

---

## 📊 Summary of Findings

1. **Max Degree Divergence:** Despite identical average degree ($\langle k \rangle = 3.92$), the BA network hub reaches $k_{\max}=36$, whereas the ER maximum degree is only $k_{\max}=8$.
2. **Centrality Concentration:** Betweenness centrality in the BA network exhibits extreme inequality ($Gini = 0.794$), compared to $0.474$ in ER. The top 5 BA nodes intercept over 85% of shortest paths.
3. **Katz Parameter Bound:** The largest eigenvalue $\lambda_{\max}$ is $7.623$ in BA and $4.917$ in ER, establishing exact convergence bounds $\alpha < 1/\lambda_{\max}$. We selected $\alpha = 0.0500$.
4. **First-Mover Advantage:** Earliest-born nodes ($0, 1, 4, 5, 6$) monopolize all top centrality rankings in BA, verifying $k_i(t) \propto (t/t_i)^{1/2}$.
5. **Percolation ("Robust yet Fragile"):** Targeted hub removal fragments the BA network at $f_c \approx 0.18$, whereas it withstands random failures up to $f \approx 0.75$.

---

## 📜 Full Scientific Report
For the complete mathematical formulations, detailed correlation heatmaps, and formal citations, see:
- [REPORT.md](file:///d:/AI/SNA_Project/REPORT.md)
- [report.tex](file:///d:/AI/SNA_Project/report.tex)
