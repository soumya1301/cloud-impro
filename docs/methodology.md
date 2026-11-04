# Research Methodology

## 1. Architectural Pipeline
The framework follows a 6-stage operational pipeline:

1. **VM Arrival & Telemetry Ingestion:** Incoming VM lifecycle events and historical resource utilization telemetry are ingested.
2. **Workload Demand Forecasting (Multivariate LSTM):**
   - Inputs: Multivariate history $[\mathbf{u}_{\text{CPU}}(t), \mathbf{u}_{\text{RAM}}(t)]$ of length $L=12$ (60 minutes).
   - Architecture: 2-layer LSTM ($H=64$, dropout $0.2$, Sigmoid projection head).
   - Horizons: 10m ($h=2$), 30m ($h=6$), 60m ($h=12$).
3. **VM Profiling (K-Means++):**
   - 16 extracted statistical and microarchitectural features (mean, max, min, std, percentiles, CPI, MAI, duration, peak-to-average ratio).
   - Systematic evaluation across $K \in \{3, 4, 5, 6\}$ using Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics ($K=6$ selected).
4. **Dynamic Load-Aware Host Thresholds:**
   - Upper threshold dynamically contracts from 75% to 65% when the LSTM forecasts an imminent surge ($\hat{U}_j(t+h) > 0.65$), creating safety headroom.
5. **Multi-Objective Placement (Ant Colony Optimization):**
   - $N_{\text{ants}} = 15$, $\alpha = 1.0$, $\beta = 2.0$, $\rho = 0.1$, $q_0 = 0.8$.
   - Fitness function:
     $$J = 0.40 \cdot E_{\text{norm}} + 0.25 \cdot A_{\text{norm}} + 0.15 \cdot \text{Imbalance} + 0.20 \cdot \text{SLA}_{\text{risk}}$$
6. **Net-Benefit Live Migration:**
   - Migration transfer time: $T = \frac{\text{VM}_{\text{memory}}}{B_w}$ ($B_w = 4\text{ Gbps}$).
   - Evaluates:
     $$\text{NetBenefit} = \Delta E_{\text{saved}} - E_{\text{mig}} - \text{Penalty}_{\text{SLA}} > 0.5\text{ Wh}$$

