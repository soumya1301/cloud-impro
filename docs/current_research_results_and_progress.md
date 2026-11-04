# Current Research Results and Project Progress Report

**Document Reference:** `docs/current_research_results_and_progress.md`
**Project Workspace:** `/Users/soumyaprasad/Desktop/cloud improvements`
**Git Branch:** `scientific_validation_correction`
**Commit Reference:** `6b1c95b98ee271084ba845dc401be437d5f641ad`
**Execution Timestamp:** 2026-11-04
**Primary Evaluated Dataset:** Google Cluster Data 2019 (Borg Workload Telemetry)
**Cluster Simulation Scale:** 7,644 Virtual Machines, 798 Physical Machines

---

## 2.1 Research Overview

### Motivation and Problem Formulation
Hyperscale cloud data centres dissipate hundreds of terawatt-hours of electrical energy annually, driven largely by low average server utilization (typically 15% to 35% in enterprise production deployments) coupled with high static idle power dissipation (frequently 50% to 70% of peak rated power). Dynamic Virtual Machine (VM) consolidation seeks to pack active workloads onto a minimum number of physical machines (PMs), transitioning unneeded hosts into low-power sleep or standby states.

However, aggressive consolidation introduces severe Quality of Service (QoS) degradation: packing workloads too tightly induces CPU queuing latency, memory contention, and Service Level Agreement violations (SLAv). Conversely, conservative packing wastes electrical energy. Conventional consolidation heuristics suffer from:
1. **Single-resource bias:** Optimizing CPU while ignoring multi-dimensional memory or microarchitectural bottlenecks.
2. **Reactive static thresholding:** Fixed upper thresholds ($T_{\text{upper}} = 75\%$) trigger consolidation migrations only after congestion occurs, often too late to avoid SLA penalties during sudden traffic bursts.
3. **Local greedy bin-packing:** First-Fit and Best-Fit heuristics lack global combinatorial foresight and easily trap in suboptimal local minima.

### Integrated Framework Architecture
To address these challenges, the research project designs and evaluates an integrated six-stage scheduling framework:
* **Workload Forecasting (LSTM & GRU):** Multivariate sequence models predict future joint CPU and memory utilization ($\hat{u}_{\text{CPU}}, \hat{u}_{\text{RAM}}$) across 10-minute, 30-minute, and 60-minute horizons. Anticipating resource surges provides proactive lead time before hosts breach capacity.
* **K-Means++ Workload Profiling:** Unsupervised clustering partitions VMs into behavioral classes based on empirical CPU intensity, memory requirements, and microarchitectural metrics (Cycles Per Instruction, Memory Accesses Per Instruction), enforcing host diversity invariants.
* **Dynamic Predicted-Load-Aware PM Thresholds:** Host operating thresholds adaptively contract from baseline $T_{\text{base}} = 0.75$ down to $0.65$ whenever predicted aggregate VM demands indicate impending host congestion.
* **Nonlinear Energy Modelling:** Power dissipation is calibrated against published SPECpower benchmarks for HP ProLiant G5 and G4 enterprise servers using a nonlinear exponent $P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) u^k$ ($k = 1.4$).
* **Ant Colony Optimization (ACO) Placement:** A global metaheuristic explores the multi-dimensional combinatorial search space, minimizing a multi-objective cost function that balances energy dissipation, active PM count, resource imbalance, and SLA risk.
* **Migration-Benefit Evaluation:** An economic decision engine guarantees that candidate migrations occur if and only if anticipated energy savings exceed the NIC transfer energy overhead and SLA degradation.

---

## 2.2 Original Research Objectives vs. Implemented Pipeline

The following table evaluates each intended research objective against actual implementation, existing experimental artifacts, verified status, and remaining work.

### Status Label Definitions:
* `IMPLEMENTED`: Code exists and runs without crashing. (Not a synonym for experimentally validated).
* `TESTED`: Automated unit/integration tests verify component mechanics and boundary assertions.
* `EXPERIMENTALLY EVALUATED`: Benchmarked on simulation traces with quantitative metric logs.
* `PARTIALLY VALIDATED`: Validated under specific simulation conditions, but subject to known data or scenario limitations.
* `NOT VALIDATED`: Empirical evidence does not confirm the hypothesized architectural advantage.
* `NOT TESTED`: Not executed or verified in the current environment.

### Status Matrix:
| Original Research Objective | Implemented Component | Existing Experimental Evidence | Current Status | Remaining Work Required |
| :--- | :--- | :--- | :---: | :--- |
| **1. Multivariate Deep Workload Forecasting** | `MultivariateLSTM` & `MultivariateGRU` (`src/vm_scheduler/prediction/`) | `table_gru_vs_lstm_prediction.csv`, `gru_{10m,30m,60m}.pt`, `lstm_{10m,30m,60m}.pt` | `EXPERIMENTALLY EVALUATED` | Address why baseline Moving Average achieves lower error; evaluate on non-stationary traces. |
| **2. K-Means++ Workload Profiling** | `KMeansPlusPlusEvaluator` (`src/vm_scheduler/clustering/kmeanspp.py`) | `table6_kmeanspp_evaluation.csv`, `kmeanspp_bundle.joblib` | `EXPERIMENTALLY EVALUATED` | Expand clustering features beyond static request dimensions to multi-hour dynamic profiles. |
| **3. Proactive Dynamic Thresholding** | Dynamic threshold logic in `AntColonyPlacer` and `PhysicalMachine` | `table13_ablation_study.csv` | `PARTIALLY VALIDATED` | Test under high-utilization/congestion scenarios where dynamic threshold contraction actually triggers. |
| **4. Calibrated SPECpower Energy Model** | `ServerPowerModel` (`src/vm_scheduler/energy/power_model.py`) | `table14_sensitivity_analysis.csv`, unit tests | `TESTED` & `EXPERIMENTALLY EVALUATED` | Validate against physical server power meters rather than benchmark curves alone. |
| **5. Global Multi-Objective Placement via ACO** | `AntColonyPlacer` (`src/vm_scheduler/scheduling/aco.py`) | `table8_cluster_performance_comparison.csv` (152 PMs vs. 798 PMs) | `EXPERIMENTALLY EVALUATED` | Perform multiple random-seed runs; conduct runtime scalability profiling across ant counts. |
| **6. Net-Benefit Migration Gating** | `MigrationManager` (`src/vm_scheduler/scheduling/migration.py`) | Table 13 ablation (1 migration executed without gating; 0 with gating) | `PARTIALLY VALIDATED` | Evaluate on multi-hour dynamic time-series traces where workload churn forces live migration. |
| **7. Predictive Scheduling Superiority** | Integration of LSTM/GRU forecasts into ACO placement | `table_scheduling_model_comparison.csv` | `NOT VALIDATED` | Reconcile VM ID namespaces so empirical forecasts can be supplied to scheduled VMs; test under surge. |
| **8. Multi-Resource Balance Enforcement** | 2D CPU & Memory residual imbalance penalty in ACO | `resource_imbalance` in Table 8 (0.1752 vs. 0.1867 in MFFA) | `PARTIALLY VALIDATED` | Incorporate disk I/O and network bandwidth if and when synthetic/extended traces are introduced. |

---

## 2.3 Forecasting Results and Critical Statistical Evaluation

The multi-horizon demand prediction pipeline was evaluated across 10-minute, 30-minute, and 60-minute horizons on 42,199 held-out test windows extracted from 600 representative workload entities from Google Borg 2019 Shard 170.

### Metric Definitions:
* **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$. Lower values indicate smaller average prediction error.
* **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$. Heavily penalizes large outlier errors; lower is better.
* **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Measures proportion of variance explained; higher is better (1.0 is perfect).

### Multi-Horizon Benchmark Table:
All values are verified against `results/tables/table_gru_vs_lstm_prediction.csv` and `experiments/prediction/gru_vs_lstm_experiment_meta.json`.

| Horizon | Model Architecture | Overall MAE | Overall RMSE | $R^2$ Score | NMAE | Bounded MAPE | CPU MAE | Memory MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10m** | **LSTM** | 0.00455 | 0.01328 | 0.5924 | 0.01495 | 31.23% | 0.00643 | 0.00267 |
| **10m** | **GRU** | 0.00470 | 0.01366 | 0.5690 | 0.01542 | 32.35% | 0.00679 | 0.00260 |
| **10m** | Persistence | 0.00422 | 0.01387 | 0.5559 | 0.01385 | 36.53% | 0.00648 | 0.00196 |
| **10m** | **Moving Average** | **0.00363** | **0.01085** | **0.7279** | **0.01191** | **28.76%** | **0.00552** | **0.00174** |
| **30m** | **LSTM** | 0.00497 | 0.01371 | 0.5750 | 0.01630 | 35.13% | 0.00710 | 0.00283 |
| **30m** | **GRU** | 0.00510 | 0.01433 | 0.5361 | 0.01675 | 35.14% | 0.00740 | 0.00281 |
| **30m** | Persistence | 0.00459 | 0.01413 | 0.5488 | 0.01505 | 37.15% | 0.00697 | 0.00220 |
| **30m** | **Moving Average** | **0.00391** | **0.01110** | **0.7213** | **0.01285** | **29.26%** | **0.00583** | **0.00200** |
| **60m** | **LSTM** | 0.00531 | 0.01429 | 0.5444 | 0.01956 | 39.78% | 0.00781 | 0.00281 |
| **60m** | **GRU** | 0.00571 | 0.01478 | 0.5124 | 0.02102 | 36.44% | 0.00820 | 0.00321 |
| **60m** | Persistence | 0.00514 | 0.01556 | 0.4593 | 0.01894 | 39.68% | 0.00775 | 0.00253 |
| **60m** | **Moving Average** | **0.00432** | **0.01200** | **0.6788** | **0.01589** | **31.45%** | **0.00643** | **0.00220** |

### Critical Scientific Findings:
1. **The Moving Average Baseline Outperforms Deep Recurrent Models:**
   * In strict contrast to conventional literature claims, the rolling **Moving Average** baseline achieves the lowest MAE (0.00363 at 10m; 0.00391 at 30m; 0.00432 at 60m) and highest $R^2$ (0.7279 at 10m; 0.7213 at 30m; 0.6788 at 60m) across all three forecast horizons.
   * Neither LSTM nor GRU is the top-performing forecaster under mean-metric rankings.
   * This finding demonstrates that in smoothly fluctuating, low-variance cluster traces, simple rolling averages filter noise more effectively than parameterized neural networks that risk slight overfitting on stationary sequences.
2. **LSTM Marginally Outperforms GRU:**
   * Comparing the two recurrent networks directly, LSTM achieves consistently lower MAE and higher $R^2$ than GRU at all horizons (e.g., at 30m: LSTM $R^2 = 0.5750$ vs. GRU $R^2 = 0.5361$).
   * However, LSTM requires 52,834 parameters compared to GRU's 40,162 parameters (23.98% more parameters) and incurs longer training times (48.0s vs. 38.97s at 30m).
3. **Multi-Resource Disaggregation:**
   * Across all models, Memory MAE is consistently lower ($\approx 0.0017 - 0.0032$) than CPU MAE ($\approx 0.0055 - 0.0082$). Memory allocations in cloud containers are static or slowly expanding, whereas CPU exhibits high-frequency burstiness.

---

## 2.4 Full-Scale Scheduling Benchmark Results

The scheduling benchmark was executed on the full simulation cluster comprising **7,644 VMs** and **798 PMs** over a 1-hour observation window.

### Benchmark Table:
All values verified against `results/tables/table8_cluster_performance_comparison.csv` and `results/tables/table_scheduling_model_comparison.csv`.

| Scheduler Algorithm | Total Energy (Wh) | Active PMs | Sleeping PMs | SLA Violations ($\text{SLAv}$) | Placement Rate (%) | AUR (%) | Resource Imbalance | Migration Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Trace (Simulated)** | 77,032.51 | 798 | 0 | 0.000000 | 100.0% (7,644/7,644) | 13.37% | 0.0576 | 0 |
| **Modified Best Fit (MBFA)** | 27,054.43 | 184 | 614 | 0.000000 | 100.0% (7,644/7,644) | 66.44% | 0.1666 | 0 |
| **Modified First Fit (MFFA)** | 24,626.13 | 154 | 644 | 0.000000 | 100.0% (7,644/7,644) | 65.62% | 0.1867 | 0 |
| **Round Robin (RRA)** | 77,032.51 | 798 | 0 | 0.000000 | 100.0% (7,644/7,644) | 13.37% | 0.0576 | 0 |
| **Sercon** | 24,626.13 | 154 | 644 | 0.000000 | 100.0% (7,644/7,644) | 65.62% | 0.1867 | 0 |
| **Proposed (LSTM + ACO)** | **24,487.22** | **152** | **646** | **0.000000** | **100.0% (7,644/7,644)** | **65.98%** | **0.1752** | **0** |
| **Proposed (GRU + ACO)** | **24,487.22** | **152** | **646** | **0.000000** | **100.0% (7,644/7,644)** | **65.98%** | **0.1752** | **0** |
| **Reactive ACO (No Forecast)** | **24,487.22** | **152** | **646** | **0.000000** | **100.0% (7,644/7,644)** | **65.98%** | **0.1752** | **0** |

### Energy Reduction Calculations:
* Versus **Google Trace Baseline (77,032.51 Wh):**
  $$\Delta E = \frac{77,032.51 - 24,487.22}{77,032.51} \times 100\% = \mathbf{68.21\% \text{ reduction}}$$
* Versus **Modified Best Fit (27,054.43 Wh):**
  $$\Delta E = \frac{27,054.43 - 24,487.22}{27,054.43} \times 100\% = \mathbf{9.49\% \text{ reduction}}$$
* Versus **Modified First Fit (24,626.13 Wh):**
  $$\Delta E = \frac{24,626.13 - 24,487.22}{24,626.13} \times 100\% = \mathbf{0.56\% \text{ reduction}}$$

### Consolidation Metrics:
* Active physical machines reduced from **798 to 152 PMs** (an **80.95% reduction in active hosts**).
* **646 PMs** transitioned to 10 W sleep state.
* Multi-resource imbalance: ACO achieves **0.1752**, improving upon MFFA (0.1867) by balancing CPU and memory consumption.
* Zero SLA violations ($\text{SLAv} = 0.0$) and 100% placement completeness across all configurations.

---

### CRITICAL SCIENTIFIC QUALIFICATION: Identical Scheduling Results
A vital finding is that **LSTM + ACO**, **GRU + ACO**, and **Reactive ACO (without forecasting)** yield **exactly identical scheduling outputs** (24,487.22 Wh, 152 active PMs, 646 sleeping PMs, 0.1752 imbalance, 0 migrations).

**Why do the predictive and reactive variants produce identical results?**
1. **Operating Load Below Congestion Threshold:** Packing 7,644 VMs onto 152 hosts yields an Average Utilization Ratio (AUR) of **65.98%**. Because 65.98% is below the base upper threshold $T_{\text{base}} = 0.75$, the dynamic threshold trigger condition ($\hat{U} > 0.65$ under projected surge) does not contract the threshold on healthy hosts.
2. **Initial Placement Completeness:** ACO's global multi-objective heuristic places all 7,644 VMs feasibly on its initial pass without overloading any host ($N_{\text{overloaded}} = 0$). With zero hosts breaching capacity, reactive migrations are never initiated.
3. **Category C Fallback Prediction Uniformity:** Because all 7,644 scheduled VMs lack historical usage time series (due to disjoint identifier schemas between Subset B and Subset A), both LSTM and GRU fallback to profile-based request modulation. Consequently, the input demand matrices provided to the scheduler are mathematically equivalent.

**Scientific Conclusion:**
> **The current experimental benchmark DOES NOT isolate or demonstrate a downstream energy or SLA advantage for learned predictive scheduling over reactive ACO.** Any claim that LSTM or GRU forecasting improves downstream cluster consolidation over reactive placement is currently unsubstantiated by the experimental evidence.

---

## 2.5 Component Ablation Study

An ablation study was executed on a controlled subset of **1,500 VMs across 150 PMs** to isolate the independent contributions of individual framework components.

### Ablation Study Table:
Verified against `results/tables/table13_ablation_study.csv`.

| Configuration Variant | Total Energy (Wh) | Active PMs | AUR (%) | SLAv | Migrated VMs | Resource Imbalance | Placement Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Full Framework** | **4,479.54** | **27** | **66.29%** | **0.0** | **0** | **0.1359** | **100.0%** |
| **w/o Prediction (Reactive Only)** | 4,479.54 | 27 | 66.29% | 0.0 | 0 | 0.1359 | 100.0% |
| **w/o Dynamic Thresholds (Static 75%)** | 4,479.54 | 27 | 66.29% | 0.0 | 0 | 0.1359 | 100.0% |
| **w/o ACO (Greedy First Fit)** | 4,544.20 | 28 | 65.96% | 0.0 | 0 | 0.1716 | 100.0% |
| **w/o Migration Benefit Gating** | 4,479.57 | 27 | 66.29% | 0.0 | 1 | 0.1360 | 100.0% |
| **Linear Power Model (Ablation)** | 4,584.61 | 27 | 66.27% | 0.0 | 0 | 0.1329 | 100.0% |

### Analysis of Component Impact:
* **Components that change the outcome:**
  1. **ACO Metaheuristic:** Removing ACO and replacing it with Greedy First Fit increases active PMs from 27 to 28, increases energy from 4,479.54 Wh to 4,544.20 Wh (+64.66 Wh), and degrades resource imbalance from 0.1359 to 0.1716. This confirms the value of combinatorial search over single-pass greedy packing.
  2. **Migration Benefit Gating:** Disabling migration gating allows an unproductive migration (1 migrated VM), slightly increasing energy to 4,479.57 Wh due to transfer overhead without consolidation benefit.
  3. **Linear Power Model:** Evaluating placement under a linear energy model changes total computed energy to 4,584.61 Wh (+105.07 Wh), highlighting that linear models underestimate consolidation gains under low-to-medium utilization.
* **Components that do NOT change the outcome:**
  * **Prediction:** Disabling prediction yields 4,479.54 Wh and 27 PMs (identical to Full Framework).
  * **Dynamic Thresholds:** Disabling dynamic thresholds yields 4,479.54 Wh and 27 PMs (identical to Full Framework).

**Scientific Conclusion:**
> **Neither workload prediction nor dynamic thresholding has demonstrated an independent energy or SLA reduction in this ablation benchmark.** Their measured impact is neutral because the consolidated cluster operates below the congestion trigger point.

---

## 2.6 Sensitivity Analysis: SPECpower Non-Linearity Parameter $k$

The SPECpower energy exponent $k$ was varied across $k \in [1.0, 2.0]$ on the 1,500-VM benchmark cluster.

### Sensitivity Table:
Verified against `results/tables/table14_sensitivity_analysis.csv`.

| Exponent $k$ | Total Energy (Wh) | Active PMs | AUR (%) | Resource Imbalance | Physical Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1.0** | 4,584.61 | 27 | 66.27% | 0.1329 | Strictly linear power scaling above idle ($P(u) = P_{\text{idle}} + \Delta P \cdot u$) |
| **1.2** | 4,529.79 | 27 | 66.37% | 0.1296 | Mild sub-linear/super-linear power curvature |
| **1.4** | 4,479.54 | 27 | 66.29% | 0.1359 | **Calibrated baseline for HP ProLiant G5/G4 enterprise servers** |
| **1.6** | 4,432.43 | 27 | 66.37% | 0.1281 | Moderate non-linear power efficiency scaling |
| **1.8** | 4,391.28 | 27 | 66.22% | 0.1381 | Strong power curvature |
| **2.0** | 4,349.42 | 27 | 66.65% | 0.1420 | Quadratic power scaling (common theoretical approximation) |

### Explanation of the Mathematical Relationship:
In the model $P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) u^k$, when utilization $u \in [0, 1]$, raising $u$ to a higher power $k > 1$ yields a smaller fraction: $u^{2.0} < u^{1.4} < u^{1.0}$ for all $u < 1$. Consequently, for hosts operating at moderate utilization ($u \approx 0.66$), higher values of $k$ mathematically yield lower computed dynamic power dissipation, reducing aggregate cluster energy from 4,584.61 Wh ($k=1.0$) down to 4,349.42 Wh ($k=2.0$). This behavior is mathematically expected and reflects model formulation rather than emergent scheduler optimization.

---

## 2.7 Automated Test Suite Verification

The project includes an automated test suite executed via PyTest:

```bash
/opt/anaconda3/bin/pytest tests/ -v
```

### Execution Summary:
* **Total Tests Executed:** 32
* **Tests Passed:** 32 (100.0%)
* **Tests Failed:** 0 (0.0%)
* **Tests Skipped:** 0 (0.0%)
* **Execution Duration:** 5.73 seconds (verified on macOS Darwin, Apple Silicon)

### Breakdown by Test Module:
1. **`tests/test_gru_and_extensions.py` (10 tests):**
   * `test_gru_architecture_and_output_dimensions`: Verified forward pass $[B, 12, 2] \to [B, 2]$ bounded within $[0, 1]$.
   * `test_gru_parameter_efficiency`: Verified exact counts (40,162 GRU vs. 52,834 LSTM; 23.98% reduction).
   * `test_gru_checkpoint_save_and_load`: Verified weight tensor equivalence across serialization.
   * `test_gru_inference_parity`: Verified inferencer output mapping and provenance categorization.
   * `test_identical_evaluation_sample_indices`: Verified identical test splits across 10m, 30m, 60m horizons.
   * `test_leakage_free_scaling`: Verified min/max values remain unaltered during validation transforms.
   * `test_entity_and_chronological_isolation`: Verified windowing strictly within entity boundaries.
   * `test_multi_resource_forecast_propagation`: Verified joint CPU and Memory updates in PM objects.
   * `test_prediction_provenance_accounting`: Verified exact sum of Category A, B, and C counts (7,644).
   * `test_decoupled_ablation_parameters`: Verified independent parameter invocation in ACO.
2. **`tests/test_suite.py` (22 tests):**
   * Verified schema column verification, multivariate dimensions, window shapes, chronological sorting, resource scaler mechanics, LSTM dimensions, prediction metrics, bounded MAPE zero handling, SPECpower curves, PM hard capacity enforcement, soft vs. hard threshold checks, dynamic threshold adaptation, ACO feasibility, SLA heuristic penalty, net-benefit positive/negative migration gating, duplicate allocation prevention, release load integrity, monotonic power scaling, and cluster cloning isolation.

### Critical Qualification on Test Scope:
> **Passing automated software tests confirms code correctness, API contracts, and absence of runtime regressions. Passing tests DO NOT establish empirical scientific validity, predictive superiority, or statistical significance of research claims.**

---

## 2.8 Current Limitations and Future Work

1. **Extreme Sparsity of Extracted Usage Telemetry:**
   * Only 28 out of 92,468 Parquet shards of `instance_usage` were extracted (~0.03% coverage). The remaining 92,440 shards remain in Google Cloud Storage.
2. **Workload Identifier Mismatch:**
   * The 7,644 VMs in Subset B were synthesized with identifiers `vm_00000` through `vm_07643`. In contrast, Subset A time-series data retained composite Borg identifiers (`collection_id_instance_index`).
   * Zero exact identifier matches exist between the two datasets.
3. **Category C Fallback Prediction Uniformity:**
   * 100% of the 7,644 scheduled VMs received Category C fallback estimates based on static resource requests modulated by cluster profile means. No scheduled VM actually utilized an autoregressive LSTM or GRU sequence forecast.
4. **Downstream Scheduling Invariance:**
   * Under 65.98% cluster utilization, LSTM+ACO, GRU+ACO, and Reactive ACO produce identical decisions. The current benchmark cannot validate the downstream utility of deep sequence models.
5. **Absence of Multi-Seed Statistical Analysis:**
   * Results represent single-run deterministic seeds ($S=42$). Confidence intervals across multiple random initializations have not yet been established.
6. **Absence of Host Congestion and Migration Triggers:**
   * Due to high cluster capacity (798 PMs for 7,644 small VMs), zero hosts experienced congestion, and zero live migrations were triggered during steady-state placement.
7. **Omission of Network and Disk Telemetry by Google:**
   * Google Cluster Data 2019 contains no instance-level disk I/O or network throughput counters. Multi-resource scheduling is limited to CPU, RAM, CPI, and MAI.

---

## 2.9 Final Research Status

The project has achieved a high degree of architectural completion, robust engineering implementation, and comprehensive automated test coverage. However, scientifically, the core hypothesis—that deep learned workload forecasting improves energy-aware VM scheduling over reactive heuristics—remains **unproven by the current experimental evidence**.

Before publication in an IEEE or ACM transactions venue, the research requires:
1. Resolving the identifier mapping between usage time series and scheduled instances.
2. Evaluating high-load stress scenarios ($U > 85\%$) where host congestion actually triggers dynamic thresholding.
3. Testing dynamic multi-hour traces with bursty job arrivals that force live migration.
4. Reporting multi-seed mean and variance statistics across all quantitative benchmarks.

---

## 2.10 Source-Artifact Reference Table

| Metric / Result Area | Primary Artifact File | Artifact Type | Verified Status |
| :--- | :--- | :---: | :---: |
| **Forecasting Accuracy (10m, 30m, 60m)** | `results/tables/table_gru_vs_lstm_prediction.csv` | CSV | Verified |
| **Forecasting Metadata & Disaggregation** | `experiments/prediction/gru_vs_lstm_experiment_meta.json` | JSON | Verified |
| **Scheduling Model Comparison** | `results/tables/table_scheduling_model_comparison.csv` | CSV | Verified |
| **Full Cluster Baseline Benchmark** | `results/tables/table8_cluster_performance_comparison.csv` | CSV | Verified |
| **Ablation Study Benchmark** | `results/tables/table13_ablation_study.csv` | CSV | Verified |
| **Sensitivity Analysis Benchmark** | `results/tables/table14_sensitivity_analysis.csv` | CSV | Verified |
| **Prediction Provenance Accounting** | `results/tables/prediction_provenance_audit.csv` | CSV | Verified |
| **Prediction Provenance Metadata** | `experiments/prediction/prediction_provenance_audit.json` | JSON | Verified |
| **K-Means++ Cluster Evaluation** | `results/tables/table6_kmeanspp_evaluation.csv` | CSV | Verified |
| **Trained LSTM Models** | `models/checkpoints/lstm_{10m,30m,60m}.pt` | PyTorch | Verified |
| **Trained GRU Models** | `models/checkpoints/gru_{10m,30m,60m}.pt` | PyTorch | Verified |
| **Automated Test Suite** | `tests/test_gru_and_extensions.py`, `tests/test_suite.py` | Python / PyTest | 32/32 Passed |
