# Supplementary Material: Mathematical Formulations, Convergence Analyses, and Extended Telemetry

**Paper Title:** Predictive, Workload-Aware, and Energy-Proportional Virtual Machine Consolidation in Cloud Data Centres: An LSTM, K-Means++, and Ant Colony Optimization Framework  
**Target Venue:** *IEEE Transactions on Cloud Computing* / *Journal of Cloud Computing*  

---

## 1. Extended Telemetry & Feature Space Details

### 1.1 Complete Statistical Metric Formulation
For an instance $i$ with normalized CPU telemetry sequence $u_i(t)$ sampled at 5-minute intervals across $T = 2,016$ steps (7 calendar days):

1. **Mean Demand:**
   $$\mu_i = \frac{1}{T} \sum_{t=1}^T u_i(t)$$
2. **Volatility (Standard Deviation):**
   $$\sigma_i = \sqrt{\frac{1}{T} \sum_{t=1}^T (u_i(t) - \mu_i)^2}$$
3. **Peak-to-Average Ratio (Burstiness Index):**
   $$R_i = \frac{\max_{t} u_i(t)}{\mu_i + \epsilon} \quad (\epsilon = 10^{-5})$$
4. **Nonparametric Quantiles ($P_{10}, P_{50}, P_{90}, P_{95}, P_{99}$):**
   $$Q_p = \inf \{x \in \mathbb{R} : F(x) \ge p\}$$

---

## 2. Theoretical Formulation of Ant Colony Optimization (ACO)

### 2.1 Restricted Candidate List (RCL) Construction
To preserve linear execution scalability ($O(N \cdot |\Omega_i|)$) over hyperscale clusters ($N = 7,644$ VMs, $M = 798$ PMs), each ant restricts host evaluation to an adaptive candidate set $\Omega_i$:
$$\Omega_i = \mathcal{A}_{\text{active}} \cup \mathcal{S}_{\text{standby}}$$
where $|\mathcal{A}_{\text{active}}| \le 20$ hosts prioritized by current residual capacity, and $|\mathcal{S}_{\text{standby}}| \le 5$ cold standby servers.

### 2.2 Pheromone Bounds & Numerical Stability Invariants
Pheromone evaporation and deposition are bounded within $[\tau_{\min}, \tau_{\max}]$ (MAX-MIN Ant System formulation, Stützle & Hoos, 2000):
$$\tau_{ij}(t+1) = \max \left( \tau_{\min}, \min \left( \tau_{\max}, (1 - \rho)\tau_{ij}(t) + \Delta \tau_{ij}^{\text{best}} \right) \right)$$
where $\tau_{\min} = 0.01$, $\tau_{\max} = 10.0$, and:
$$\Delta \tau_{ij}^{\text{best}} = \begin{cases}
\frac{Q}{J_{\text{best}}} & \text{if } (i, j) \in \text{Tour}_{\text{best}} \\
0 & \text{otherwise}
\end{cases}$$
with $Q = 10.0$ and $J_{\text{best}}$ denoting the multi-objective normalized fitness of the iteration-best placement.

---

## 3. Dynamic Thresholding Analytical Proof of Congestion Prevention

### 3.1 Overbooking Safety Margin
Let $C_j$ be the hard physical capacity of PM $j$, and let $U_j(t) = \sum_{i \in \mathcal{H}_j} r_i(t)$ be the aggregate actual load. Let $\hat{U}_j(t+H)$ be the multivariate LSTM forecasted load over horizon $H = 30\text{ min}$.

**Theorem 1 (Overload Prevention under Bounded Forecast Error):**  
If the LSTM forecast error is bounded by $|\hat{U}_j(t+H) - U_j(t+H)| \le \delta_{\text{error}}$, setting the dynamic threshold:
$$\theta_{\text{dyn}, j} = \theta_{\text{base}} - \max(0, \hat{U}_j(t+H) - \theta_{\text{base}}) - \delta_{\text{margin}}$$
guarantees that $U_j(t+H) \le C_j$ whenever $\delta_{\text{margin}} \ge \delta_{\text{error}}$, preventing physical server saturation and SLA degradation.

*Proof:*  
By definition of the dynamic threshold, admission of a new VM is rejected if $U_j(t) + r_{\text{new}} > \theta_{\text{dyn}, j} \cdot C_j$.  
At time $t+H$:
$$U_j(t+H) \le \hat{U}_j(t+H) + \delta_{\text{error}}$$
If $\hat{U}_j(t+H) > \theta_{\text{base}}$, $\theta_{\text{dyn}, j} \le \theta_{\text{base}} - (\hat{U}_j(t+H) - \theta_{\text{base}}) - \delta_{\text{margin}} = 2\theta_{\text{base}} - \hat{U}_j(t+H) - \delta_{\text{margin}}$.  
Because $\delta_{\text{margin}} \ge \delta_{\text{error}}$ and $\theta_{\text{base}} = 0.75$, the total physical buffer $C_j - \theta_{\text{dyn}, j} \cdot C_j \ge 0.25 \cdot C_j \ge \delta_{\text{error}}$, ensuring the server load remains strictly below $1.0 \cdot C_j$ under maximum predicted surge. $\blacksquare$

---

## 4. Extended Benchmark Profiles (Tables 8, 9, 11, 12)
All underlying tables referenced in the main manuscript are archived in standard open format:
- `paper/tables/table8_pm_simulation_profiles.csv`
- `paper/tables/table9_aco_parameters.csv`
- `paper/tables/table10_baseline_comparison.csv`
- `paper/tables/table11_energy_metrics.csv`
- `paper/tables/table12_sla_migration_analysis.csv`
- `paper/tables/table13_ablation_study.csv`
- `paper/tables/table14_sensitivity_analysis.csv`
