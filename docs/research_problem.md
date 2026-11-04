# Research Problem Statement

## 1. Context and Problem Definition
Cloud data centres currently consume over 200 Terawatt-hours (TWh) of electricity annually, projected to expand to 8,000 TWh by 2030. In typical enterprise clusters, physical machines (PMs) often operate at low average utilization levels (often 15%–35%), leading to severe energy inefficiency because idle or lightly loaded servers consume substantial baseline idle power (between 50% and 70% of peak power).

## 2. Fundamental Limitations of Existing Approaches
1. **Single-Resource Bias:** Most VM placement and consolidation algorithms optimize purely around CPU load (or occasionally memory), neglecting the multi-resource dynamics of cloud workloads.
2. **Workload Type Blindness:** VMs exhibit heterogeneous profiles (CPU-intensive, memory-intensive, balanced). Consolidating multiple CPU-intensive VMs on the same physical host leads to severe CPU contention, queueing latency, and Service Level Agreement violations (SLAv).
3. **Reactive vs. Predictive Scheduling:** Existing consolidation algorithms act reactively only after hosts become overloaded ($>75\%$) or underloaded ($<35\%$). This late reaction triggers emergency migrations during load surges, compounding performance degradation.
4. **Local Greedy Optima:** Heuristics such as First Fit or Best Fit cannot search the global combinatorial search space, frequently resulting in fragmented resource allocation.

## 3. The Research Objective
To develop, implement, and validate an intelligent, predictive, and multi-objective VM scheduling and consolidation framework combining:
- Multivariate Long Short-Term Memory (LSTM) networks for multi-step resource demand forecasting (10, 30, and 60 minutes).
- K-Means++ clustering for VM workload characterization and profiling ($K \in \{3,4,5,6\}$).
- Dynamic, predicted-load-aware PM thresholds providing adaptive congestion headroom.
- A calibrated nonlinear SPECpower energy model ($P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) u^{1.4}$).
- Ant Colony Optimization (ACO) for global multi-objective placement.
- Net-benefit live migration analysis to guarantee that energy saved after migration exceeds transfer overhead and SLA degradation.

