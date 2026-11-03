# Response to Reviewers: Methodological Defense and Empirical Justifications

**Manuscript Title:** Predictive, Workload-Aware, and Energy-Proportional Virtual Machine Consolidation in Cloud Data Centres: An LSTM, K-Means++, and Ant Colony Optimization Framework  
**Authors:** Research Team  

We express our sincere gratitude to the reviewers and editors for their constructive evaluation. Below we provide detailed, point-by-point responses addressing all methodology and experimentation inquiries.

---

### Reviewer 1: Methodological Scope and Ground Truth
> **Comment 1.1:** *The reference paper by Alelyani et al. (2026) claimed to utilize 4-dimensional scheduling (CPU, Memory, Disk, Network). How does your work address these dimensions on the Google Cluster Data 2019 trace?*

**Response:**  
We thank the reviewer for highlighting this critical point. During our comprehensive read-only audit of 120,298 storage objects (10.28 TiB) across all eight Google 2019 cells (`a` through `h`), we conducted schema inspection of `instance_usage` and `instance_events`.  
We established that:
1. **CPU, Memory, CPI, and MAI** are genuinely present with 0.00% missingness.
2. **Network Bandwidth and Disk I/O** were omitted by Google in the public 2019 release to prevent topological inference attacks.

In strict accordance with scientific integrity, we do not fabricate or synthesize non-existent trace telemetry. Instead:
- Empirical trace features (CPU, RAM, CPI, MAI) drive the LSTM demand forecasting and K-Means++ clustering.
- Network bandwidth during live migration is modeled deterministically using enterprise top-of-rack link parameters ($B_w = 4.0\text{ Gbps}$ / $0.5\text{ GB/s}$) based on standard data center network specifications.  
This distinction is made transparent throughout Section 2.2 and Section 7 of the manuscript.

---

### Reviewer 2: Clustering Parameter Selection ($K=3$ vs. $K=6$)
> **Comment 2.1:** *Alelyani et al. used $K=3$ clusters. Why does your proposed framework adopt $K=6$?*

**Response:**  
Alelyani et al. adopted $K=3$ as a fixed heuristic without reporting mathematical cluster validity indices. In this work, we conducted a systematic evaluation across $K \in \{3, 4, 5, 6\}$ using three independent validity metrics:
- **Silhouette Coefficient:** $K=6$ achieved **0.3486**, outperforming $K=3$ (**0.2930**).
- **Calinski-Harabasz Index:** $K=6$ achieved **942.42**, outperforming $K=3$ (**773.17**).
- **Davies-Bouldin Index:** $K=6$ achieved **0.9130**, outperforming $K=3$ (**1.2196**).

As illustrated in Figure 5 and Table 7, $K=6$ successfully differentiates bursty analytics (high peak-to-average ratio) and memory-intensive in-memory databases that are conflated when restricted to $K=3$.

---

### Reviewer 3: Algorithmic Realism & Migration Cost Accounting
> **Comment 3.1:** *Does your simulation account for the energy and SLA overhead of live VM migration?*

**Response:**  
Yes. In contrast to naive bin-packing models that treat VM migration as an instantaneous zero-cost operation, our framework implements:
1. **Transfer Duration Modeling:** $T_{\text{mig}} = \frac{D_{\text{RAM}}}{B_w \cdot \gamma}$, where $\gamma = 0.85$ accounts for link protocol overhead.
2. **Dynamic Energy Surcharge:** $E_{\text{mig}} = P_{\text{NIC\_overhead}} \cdot T_{\text{mig}}$, charging 15 W of active network interface power throughout the transfer.
3. **Net-Benefit Gating:** A candidate migration is executed *if and only if* the cumulative lifetime energy savings exceed the migration energy overhead plus an operational safety threshold ($\delta_{\min} = 0.5\text{ Wh}$).  
As documented in Table 10 and Table 12, this economic gating prevents migration thrashing, ensuring zero SLA violations ($\text{SLAv} = 0.0$).
