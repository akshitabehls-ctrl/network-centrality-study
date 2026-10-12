# Controlled Comparative Study of Barabási–Albert and Erdős–Rényi Networks: Centrality Distributions, Spectral Analysis, and Structural Robustness

**Author:** Antigravity Research Lab & Pair Programming Fellow  
**Date:** October 2026  
**Repository:** `network-centrality-project`  
**Artifacts Generated:** 11 Publication Figures (`figures/`), 9 Structured Datasets (`results/`)  

---

## 1. Abstract

This study presents a rigorously controlled topological and spectral comparison between scale-free and random networks. To eliminate density and edge-count confounding factors, we evaluate a Barabási–Albert (BA) preferential-attachment network ($N=100, m=2$) and an Erdős–Rényi (ER) random graph $G(n, M)$ matched to the exact identical size ($N=100$) and edge count ($M=196$, mean degree $\langle k \rangle = 3.92$, density $\rho \approx 0.0396$). We evaluate nine distinct centrality measures—degree, closeness, harmonic, betweenness, load, eigenvector, Katz ($\alpha < 1/\lambda_{\max}$), PageRank, and subgraph centrality—alongside clustering coefficients, $k$-core decompositions, and Gini inequality coefficients. The BA network exhibits marked structural inequality ($Gini_{betweenness} = 0.794$ vs. $0.474$ in ER; maximum hub degree $k_{\max}=36$ vs. $8$ in ER), driven by a pronounced first-mover advantage where early nodes monopolize network brokerage. Maximum likelihood estimation via the Clauset–Shalizi–Newman method yields a power-law exponent $\hat{\gamma} = 2.55 \pm 0.18$ ($x_{\min}=2$), with finite-size scaling ($N \in \{100, 1000, 10000\}$) validating asymptotic convergence toward $\gamma = 3.0$. Targeted hub removal precipitates catastrophic fragmentation of the BA giant component at $f_c \approx 0.18$, whereas it withstands random failures up to $f \approx 0.75$, demonstrating empirical verification of the classical "robust yet fragile" paradigm.

---

## 2. Introduction

Complex networks serve as the mathematical foundation for modeling relational structures across diverse domains, including the World Wide Web, citation networks, electrical power grids, metabolic pathways, and social interactions. In network science, two foundational paradigms represent contrasting generative mechanisms:
1. **The Erdős–Rényi (ER) Random Graph Model (1959):** Assumes edges are placed independently and uniformly at random across vertex pairs.
2. **The Barabási–Albert (BA) Scale-Free Model (1999):** Incorporates continual network growth coupled with preferential attachment ("the rich get richer"), producing power-law degree distributions.

While standard classroom analyses frequently compare arbitrary instances of these models, arbitrary parameterization often results in disparate densities and edge counts. Under such uncontrolled conditions, observed differences in path length, clustering, or centrality cannot be reliably attributed to generative mechanisms rather than simple edge density artifacts.

This investigation constructs a **strictly controlled laboratory experiment**: two networks of identical order ($N=100$) and exact edge count ($M=196$) generated under a fixed random seed (`seed=42`). Every observed variance in centrality distributions, spectral radii, correlation structures, and percolation thresholds directly reflects the fundamental difference in generative topology: homogeneous random bonding versus growth with preferential attachment.

---

## 3. Theoretical Background & Mathematical Formulations

### 3.1 Network Models

#### Erdős–Rényi $G(n, M)$ Model
The $G(n, M)$ model selects a graph uniformly at random from the ensemble of all $\binom{\binom{n}{2}}{M}$ possible simple undirected graphs with $n$ nodes and $M$ edges. In expectation, it corresponds to the binomial $G(n, p)$ model where the edge placement probability is:
$$p = \frac{2M}{n(n-1)}$$
For $N=100$ and $M=196$, $p = \frac{392}{9900} \approx 0.039596$. As $n \to \infty$ with $\lambda = n p$ held constant, the degree distribution converges to a Poisson distribution:
$$P(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad \text{where } \lambda = \langle k \rangle = 3.92$$

#### Barabási–Albert (BA) Preferential Attachment Model
The BA model starts with $m_0$ seed nodes ($m_0 = m = 2$). At each discrete time step $t$, a new vertex arrives and connects to $m$ existing vertices with probability proportional to their current degree:
$$\Pi(i) = \frac{k_i}{\sum_j k_j}$$
For an undirected network grown to $n$ vertices, the total number of edges is $M = m(n - m) = 2 \times 98 = 196$. The stationary master equation yields the asymptotic degree distribution:
$$P(k) = \frac{2 m (m + 1)}{k (k + 1) (k + 2)} \approx 2 m^2 k^{-3} \quad (k \ge m)$$
exhibiting a scale-free power law $P(k) \propto k^{-\gamma}$ with $\gamma = 3$.

---

### 3.2 Formal Centrality Definitions

Let $G = (V, E)$ be an undirected simple graph with adjacency matrix $A$, where $A_{ij} = 1$ if $(i,j) \in E$ and $0$ otherwise. Let $N = |V|$.

| Centrality Measure | Mathematical Formulation | Structural & Intuitive Interpretation |
|---|---|---|
| **Degree Centrality** | $C_D(i) = \frac{k_i}{N - 1} = \frac{\sum_j A_{ij}}{N - 1}$ | Direct local connectivity and immediate communication capacity. |
| **Closeness Centrality** | $C_C(i) = \frac{N - 1}{\sum_{j \neq i} d(i, j)}$ *(Wasserman–Faust corrected)* | Mean proximity to all other nodes; speed of information diffusion. |
| **Harmonic Centrality** | $C_H(i) = \sum_{j \neq i} \frac{1}{d(i, j)}$ | Disconnected-robust geodesic proximity ($1/\infty = 0$). |
| **Betweenness Centrality** | $C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$ | Fraction of all shortest paths traversing node $i$ (brokerage & gatekeeping). |
| **Load Centrality** | $C_L(i) = \sum_{s \neq t} \mu_{st}(i)$ | Geodesic traffic load variant under uniform routing. |
| **Eigenvector Centrality** | $x_i = \frac{1}{\lambda} \sum_j A_{ij} x_j \iff A x = \lambda x$ | Recursive prestige: importance derived from connections to well-connected neighbors. |
| **Katz Centrality** | $C_{\text{Katz}} = \left( (I - \alpha A)^{-1} - I \right) \mathbf{1}, \quad \alpha < \frac{1}{\lambda_{\max}}$ | Attenuated path-length prestige summing all walk lengths with decay factor $\alpha$. |
| **PageRank Centrality** | $p_i = \frac{1 - d}{N} + d \sum_{j \in \mathcal{N}(i)} \frac{p_j}{k_j}$ | Stationary distribution of random surfer with teleportation ($d=0.85$). |
| **Subgraph Centrality** | $C_S(i) = \sum_{k=0}^{\infty} \frac{(A^k)_{ii}}{k!} = (e^A)_{ii}$ | Participation in closed walks of all lengths, weighted by loop brevity. |
| **Local Clustering** | $C_i = \frac{2 e_i}{k_i(k_i - 1)} = \frac{(A^3)_{ii}}{k_i(k_i - 1)}$ | Triadic closure: fraction of node's neighbors that are connected. |
| **$k$-Core Number** | $K(i) = \max \{ k : i \in H_k \}$ | Maximum subgraph core where all vertices have degree $\ge k$. |

---

## 4. Controlled Experimental Methodology

### 4.1 Parameter Matching and Connectivity Handling
- **Order & Edge Count:** Both networks possess $N = 100$ and $M = 196$.
- **Seed Management:** All stochastic processes are initialized with `seed=42`.
- **Component Handling:** 
  - The BA graph is guaranteed connected by construction ($m \ge 1$), yielding a single component of size $100$.
  - In $G(n, M)$ with $N=100, M=196$, the average degree is $\langle k \rangle = 3.92$. The connectivity threshold for random graphs is $c = \ln(N) = \ln(100) \approx 4.605$. Because $\langle k \rangle < \ln(N)$, the random graph is below the full connectivity threshold and contains isolated nodes ($e^{-\langle k \rangle} \approx 0.0198$). At `seed=42`, $G_{\text{ER}}$ decomposes into two components: a giant component of $99$ nodes and $1$ isolated node ($N=1$).
  - **Handling:** Closeness centrality applies the Wasserman–Faust correction ($C_C(u) = \frac{n-1}{N-1} \frac{n-1}{\sum_{v} d(u,v)}$ where $n$ is the component size). Harmonic centrality natively computes geodesic reciprocals ($1/d(u,v) = 0$ for unreachable pairs). For spectral metrics, `scipy.linalg.eigvalsh` and `nx.eigenvector_centrality_numpy` provide robust numeric solutions.

### 4.2 Spectral Radius and Katz Attenuation Bound
For Katz centrality, the matrix $(I - \alpha A)$ is invertible via convergent Neumann series $\sum_{k=0}^{\infty} (\alpha A)^k$ if and only if $\alpha < 1/\lambda_{\max}$, where $\lambda_{\max}$ is the Perron–Frobenius eigenvalue (spectral radius) of $A$:
- **BA Network:** $\lambda_{\max} = 7.623 \implies \frac{1}{\lambda_{\max}} \approx 0.1312$.
- **ER Network:** $\lambda_{\max} = 4.917 \implies \frac{1}{\lambda_{\max}} \approx 0.2034$.
- **Selected Parameter:** We enforce $\alpha = 0.0500$, which strictly satisfies $\alpha < 1/\lambda_{\max}$ in both topologies while preserving strong attenuation across multi-hop paths.

### 4.3 Clauset–Shalizi–Newman Power-Law Estimation
We fit the BA degree sequence using discrete Maximum Likelihood Estimation (MLE) via the Kolmogorov–Smirnov (KS) distance minimization protocol. We subsequently execute a Vuong log-likelihood ratio test comparing the discrete power-law against an exponential candidate distribution:
$$\mathcal{R} = \sum_{i=1}^n \left[ \ln p_{\text{power}}(k_i) - \ln p_{\text{exp}}(k_i) \right]$$

---

## 5. Experimental Results

### 5.1 Global Structural Comparison

| Topological Metric | Barabási–Albert (BA) | Erdős–Rényi (ER) | Physical Interpretation |
|---|:---:|:---:|---|
| **Nodes ($N$)** | 100 | 100 | Matched |
| **Edges ($M$)** | 196 | 196 | Matched |
| **Average Degree $\langle k \rangle$** | 3.920 | 3.920 | Matched |
| **Maximum Degree ($k_{\max}$)** | **36** | **8** | Hub emergence: BA max degree is $4.5\times$ higher |
| **Network Density ($\rho$)** | 0.0396 | 0.0396 | Matched |
| **Connected Components** | 1 | 2 | ER possesses isolated node ($k=0$) |
| **Giant Component Size ($N_{\text{LCC}}$)** | 100 (100%) | 99 (99%) | Subcritical connectivity in ER ($\langle k \rangle < \ln N$) |
| **Average Path Length (LCC)** | **2.853** | **3.461** | Ultra-small-world effect in BA driven by hubs |
| **Network Diameter (LCC)** | **5** | **7** | Hubs reduce longest geodesic path by 28.6% |
| **Clustering Coefficient $\langle C \rangle$** | **0.1634** | **0.0462** | BA clustering is $3.5\times$ higher due to early-node cliques |
| **Degree Assortativity ($r$)** | **-0.2603** | **-0.0735** | Disassortative hub-periphery structure in BA |

---

### 5.2 Centrality Inequality Analysis: Gini Coefficients

To rigorously quantify the disparity of structural influence, we computed the Gini coefficient across all evaluated centrality distributions:

| Centrality Measure | BA Gini Inequality ($G_{\text{BA}}$) | ER Gini Inequality ($G_{\text{ER}}$) | Inequality Ratio ($G_{\text{BA}} / G_{\text{ER}}$) |
|---|:---:|:---:|:---:|
| **Betweenness** | **0.794** | 0.474 | **1.68×** |
| **Load Centrality** | **0.789** | 0.470 | **1.68×** |
| **Subgraph Centrality** | **0.624** | 0.307 | **2.04×** |
| **Eigenvector Centrality**| **0.394** | 0.344 | **1.14×** |
| **Degree Centrality** | **0.380** | 0.263 | **1.45×** |
| **PageRank Centrality** | **0.339** | 0.221 | **1.54×** |
| **Katz Centrality** | **0.088** | 0.055 | **1.61×** |
| **Harmonic Centrality** | 0.079 | 0.080 | 0.98× |
| **Closeness Centrality** | 0.075 | 0.074 | 1.02× |
| **Local Clustering** | 0.798 | 0.868 | 0.92× |

**Key Finding:** Betweenness and Subgraph centralities exhibit the highest structural concentration in the BA network. Because betweenness measures shortest-path bottleneck participation, the core hubs ($k \in \{36, 25, 20\}$) intercept over $85\%$ of all non-local geodesic traffic. In contrast, in the ER network, shortest paths diffuse across parallel paths, yielding a much more egalitarian distribution.

---

### 5.3 Top-10 Ranked Nodes Comparison

#### Barabási–Albert Top Nodes
In the BA network, node **0** is the uncontested global hub, ranked first across Degree ($36$), Closeness ($0.547$), Harmonic ($64.33$), Betweenness ($0.4088$), Eigenvector ($0.4987$), Katz ($0.2547$), and PageRank ($0.0818$). The top 5 nodes—IDs $\{0, 4, 1, 5, 6\}$—are all early nodes added during initial graph generation.

```
+-----------+--------+-------------+-------------+-------------+----------+
| Node ID   | Degree | Betweenness | Eigenvector | Closeness   | PageRank |
+-----------+--------+-------------+-------------+-------------+----------+
| 0         | 36     | 0.4088      | 0.4987      | 0.5470      | 0.0818   |
| 4         | 25     | 0.2537      | 0.3306      | 0.4975      | 0.0565   |
| 1         | 20     | 0.2182      | 0.3211      | 0.5103      | 0.0448   |
| 5         | 13     | 0.1168      | 0.2047      | 0.4670      | 0.0297   |
| 6         | 10     | 0.0876      | 0.1412      | 0.4267      | 0.0235   |
+-----------+--------+-------------+-------------+-------------+----------+
```

#### Erdős–Rényi Top Nodes
In the ER network, node importance is dispersed and decoupled:
```
+-----------+--------+-------------+-------------+-------------+----------+
| Node ID   | Degree | Betweenness | Eigenvector | Closeness   | PageRank |
+-----------+--------+-------------+-------------+-------------+----------+
| 27        | 8      | 0.0913      | 0.1753      | 0.3392      | 0.0187   |
| 31        | 8      | 0.0842      | 0.2001      | 0.3553      | 0.0181   |
| 54        | 8      | 0.0912      | 0.1797      | 0.3392      | 0.0199   |
| 7         | 8      | 0.0747      | 0.2173      | 0.3322      | 0.0184   |
| 33        | 8      | 0.0872      | 0.1673      | 0.3300      | 0.0201   |
+-----------+--------+-------------+-------------+-------------+----------+
```
No single node dominates multiple metrics in ER; maximum betweenness is only $0.0913$, over $4.4\times$ smaller than Node 0 in the BA graph.

---

### 5.4 Overview of Experimental Figures

The study generated 11 high-resolution figures in `figures/`:

1. **Figure 1 (`fig1_degree_histograms.png`):** Shows the stark divergence in degree distributions. The ER distribution is unimodal and bell-shaped around $\langle k \rangle = 3.92$, whereas the BA distribution is monotonic and heavily right-skewed, stretching out to $k=36$.
2. **Figure 2 (`fig2_er_poisson_overlay.png`):** Demonstrates close agreement between empirical ER frequencies and the theoretical Poisson PMF $P(k; \lambda = 3.92)$, confirming proper random graph realization.
3. **Figure 3 (`fig3_loglog_degree_dist.png`):** On log–log axes, the BA distribution follows a linear power-law decay, whereas the ER distribution drops off rapidly with exponential curvature.
4. **Figure 4 (`fig4_ccdf_loglog.png`):** The Complementary Cumulative Distribution Function (CCDF) avoids binning noise, illustrating clear power-law scaling $P(K \ge k) \propto k^{-(\gamma - 1)}$ for BA.
5. **Figure 5 (`fig5_ensemble_averaging.png`):** Ensemble averaging across 100 independent realizations removes stochastic noise, revealing a smooth scale-free curve with standard-deviation error bands.
6. **Figure 6 (`fig6_scaling_study.png`):** Finite-size scaling across $N \in \{100, 1000, 10000\}$ illustrates how the degree distribution straightens and extends across three decades in degree, approaching the master-equation asymptotic limit $\gamma = 3.0$.
7. **Figure 7 (`fig7_centrality_boxplots.png`):** Comparative boxplots of all nine centralities display extreme upper outliers for BA across Betweenness, Load, Eigenvector, and Subgraph centrality, alongside Gini index labels.
8. **Figure 8 (`fig8_correlation_heatmaps.png`):** Spearman rank correlation matrices highlight that BA centralities (Degree, Betweenness, Closeness, Eigenvector) are strongly correlated ($\rho \ge 0.88$), whereas in ER, Betweenness correlates weakly with Degree ($\rho \approx 0.65$).
9. **Figure 9 (`fig9_first_mover_advantage.png`):** Degree and betweenness plotted against node arrival timestamp reveal logarithmic decay: nodes introduced early accumulate connections over time, validating the cumulative advantage mechanism.
10. **Figure 10 (`fig10_network_topology.png`):** Force-directed spring layouts visually contrast the centralized hub-and-spoke layout of BA against the uniform, mesh-like topology of ER.
11. **Figure 11 (`fig11_robustness_percolation.png`):** Node percolation simulation demonstrates the "robust yet fragile" property: BA retains its giant component under random node failure up to $f \approx 0.75$, but collapses rapidly ($f_c \approx 0.18$) when hubs are targeted in descending order of degree.

---

## 6. Discussion and Mechanistic Insights

### 6.1 Preferential Attachment and First-Mover Advantage
In the Barabási–Albert model, the degree of node $i$ introduced at time $t_i$ scales dynamically according to the continuum equation:
$$k_i(t) = m \left( \frac{t}{t_i} \right)^{1/2}$$
Nodes with lower indices $i$ arrive early, providing them with more opportunities to attract incoming edges. Once a node establishes a degree lead, preferential attachment $\Pi(i) \propto k_i$ accelerates its acquisition of subsequent connections. This mechanism explains the prominence of nodes $\{0, 1, 4, 5, 6\}$ across all centrality metrics. In contrast, the ER model places edges uniformly at random, eliminating positional or temporal advantages.

### 6.2 Structural Correlation and Brokerage Monopolies
In scale-free topologies, high-degree hubs inevitably serve as network bridges. Consequently, Degree, Betweenness, and Eigenvector centralities correlate strongly ($\rho > 0.88$). In ER networks, shortest paths are distributed across multiple alternative routes, decoupling betweenness from local degree.

### 6.3 Percolation, Robustness, and Real-World Implications
The percolation curves in Figure 11 confirm the classical result established by Albert, Jeong, and Barabási (2000):
- **Random Failures:** Because most nodes in a scale-free network are low-degree peripheral nodes ($k=2$), random removals predominantly delete redundant leaves, leaving the giant component intact up to high removal fractions ($f > 0.70$).
- **Targeted Attacks:** Targeted attacks remove high-degree hubs sequentially. Removing the top 15 hubs in the BA network reduces the giant component size by more than 80%, fragmenting the network into isolated subgraphs.
- **Applications:**
  - *Epidemiology:* Super-spreader hubs in scale-free contact networks drive low epidemic thresholds; targeting vaccination to high-degree individuals suppresses transmission more effectively than uniform vaccination.
  - *Critical Infrastructure:* Internet routing backbones and power grids exhibit scale-free characteristics that make them resilient to random component failures but vulnerable to coordinated attacks on central nodes.

### 6.4 Methodological Limitations
1. **Sample Size Constraints ($N=100$):** At $N=100$, the maximum possible degree is bounded by $N$, creating finite-size cutoff effects. Our 100-realization ensemble and scaling study up to $N=10,000$ address this by demonstrating the asymptotic behavior.
2. **Binary Edge Weights:** Real-world networks frequently exhibit weighted and directed relationships, which can alter shortest paths and diffusion dynamics.

---

## 7. Conclusion

By strictly matching the network order ($N=100$) and edge count ($M=196$) across both models, this study confirms that the observed differences in centrality distributions, path lengths, clustering coefficients, and percolation thresholds are direct consequences of generative mechanics:
1. **Centrality Disparity:** Scale-free dynamics concentrate betweenness and subgraph centralities onto a small core of early hubs ($Gini_{betweenness} = 0.794$).
2. **Small-World Effect:** Hubs shorten average path lengths ($\langle d \rangle = 2.853$ in BA vs. $3.461$ in ER) and reduce diameter by nearly 30%.
3. **Robustness Trade-Off:** The scale-free network exhibits high resilience to random node deletion, but undergoes catastrophic fragmentation under targeted hub attacks ($f_c \approx 0.18$).

---

## 8. References

1. **Barabási, A.-L., & Albert, R. (1999).** Emergence of scaling in random networks. *Science*, 286(5439), 509–512.
2. **Erdős, P., & Rényi, A. (1959).** On random graphs I. *Publicationes Mathematicae Debrecen*, 6, 290–297.
3. **Freeman, L. C. (1977).** A set of measures of centrality based on betweenness. *Sociometry*, 35–41.
4. **Freeman, L. C. (1978/1979).** Centrality in social networks conceptual clarification. *Social Networks*, 1(3), 215–239.
5. **Bonacich, P. (1987).** Power and centrality: A family of measures. *American Journal of Sociology*, 92(5), 1170–1182.
6. **Katz, L. (1953).** A new status index derived from sociometric analysis. *Psychometrika*, 18(1), 39–43.
7. **Brin, S., & Page, L. (1998).** The anatomy of a large-scale hypertextual Web search engine. *Computer Networks and ISDN Systems*, 30(1–7), 107–117.
8. **Estrada, E., & Rodríguez-Velázquez, J. A. (2005).** Subgraph centrality in complex networks. *Physical Review E*, 71(5), 056103.
9. **Clauset, A., Shalizi, C. R., & Newman, M. E. (2009).** Power-law distributions in empirical data. *SIAM Review*, 51(4), 661–703.
10. **Albert, R., Jeong, H., & Barabási, A.-L. (2000).** Error and attack tolerance of complex networks. *Nature*, 406(6794), 378–382.
11. **Newman, M. E. J. (2018).** *Networks: An Introduction* (2nd ed.). Oxford University Press.

---

## 9. Appendix: Reproduction Guide & Data Dictionary

### File Structure
- `src/generate.py`: Matched network generation ($G(n, M)$ & BA).
- `src/centrality.py`: 9 centralities, spectral bounds, Gini coefficients.
- `src/degree_dist.py`: Degree distributions, Poisson fitting, Clauset–Shalizi–Newman MLE, scaling.
- `src/plots.py`: 11 publication-grade figures.
- `main.py`: End-to-end execution script.
- `results/`: Processed CSVs and JSON summary files.
- `figures/`: PNG and PDF versions of all 11 plots.

### Execution Command
```powershell
py -3.14 main.py
```
*(All seeds are fixed to 42 for reproducible results.)*
