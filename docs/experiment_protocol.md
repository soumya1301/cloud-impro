# Experimental Protocol: Rigorous Benchmarking Methodology

## 1. Trace Extraction & Ground-Truth Invariants
1. **Source Data Integrity:**
   - Raw traces are retrieved directly from Google Cloud Storage (`gs://clusterdata_2019_a/` through `h`).
   - No data extrapolation or artificial variance is injected into empirical traces.
   - Exact physical timestamps are preserved to evaluate realistic temporal workload dynamics.

## 2. LSTM Forecasting Protocol
1. **Temporal Non-Leakage:**
   - Chronological training, validation, and test splits are strictly respected.
   - Normalization statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed strictly on the training partition ($T \le 1,152$) and applied forward to test sets.
2. **Multi-Horizon Validation:**
   - Lookback window $L = 12$ steps (60 minutes of historical telemetry).
   - Forecast targets: $H \in \{2, 6, 12\}$ corresponding to 10m, 30m, and 60m horizons.
   - Evaluated against standard time-series baselines:
     - **Persistence (Naïve):** $\hat{y}(t+H) = y(t)$
     - **Moving Average:** $\hat{y}(t+H) = \frac{1}{L}\sum_{i=0}^{L-1} y(t-i)$
     - **ARIMA(1,1,1)**

## 3. K-Means++ Clustering Protocol
1. **Feature Standardisation:**
   - Continuous feature matrix $X \in \mathbb{R}^{1200 \times 11}$ standard-scaled to zero mean and unit variance.
2. **Systematic Hyperparameter Search:**
   - $K \in \{3, 4, 5, 6\}$ evaluated under identical random seeding ($S = 42$).
   - Cluster validation metrics evaluated:
     - Silhouette Coefficient (internal cohesion and separation)
     - Davies-Bouldin Index (ratio of within-cluster scatter to between-cluster separation)
     - Calinski-Harabasz Index (variance ratio criterion)

## 4. Cluster Simulation & Scheduling Evaluation
1. **Cluster Setup:**
   - 7,644 VMs placed onto 798 heterogeneous physical machines.
   - Initial state: Zero load, clean cluster clones across all algorithms for deterministic, apples-to-apples comparison.
2. **Evaluated Scheduling Frameworks:**
   - **Google Trace (Simulated):** Default Borg hash-partitioned placement without dynamic power awareness.
   - **Modified Best Fit Allocation (MBFA):** Resource packing with static $75\%$ operating threshold.
   - **Modified First Fit Allocation (MFFA):** First-fit search with static $75\%$ operating threshold.
   - **Round Robin Allocation (RRA):** Uniform spread balancing across cluster nodes.
   - **Sercon Algorithm:** Server consolidation framework using static dual thresholds ($35\%$ lower, $75\%$ upper).
   - **Proposed (LSTM + K-Means++ + ACO):** Predictive dynamic thresholding, multi-objective ACO placement, and net-benefit live migration.

