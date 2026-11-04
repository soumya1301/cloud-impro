# Research Results Executive Summary: Energy-Aware Virtual Machine Scheduling Using Google Cluster Data 2019

**Document Reference:** `docs/research_results_executive_summary.md`
**Target Audience:** Research Supervisors, Journal Reviewers, and Research Collaborators
**Project Title:** Predictive, Workload-Aware, and Energy-Proportional Virtual Machine Consolidation in Cloud Data Centres
**Workspace:** `/Users/soumyaprasad/Desktop/cloud improvements`
**Git Branch:** `scientific_validation_correction`
**Commit Reference:** `6b1c95b98ee271084ba845dc401be437d5f641ad`
**Date:** November 2026

---

## 1. Executive Briefing

This document summarizes the current status, empirical findings, architectural trade-offs, and open scientific questions of our energy-aware Virtual Machine (VM) scheduling research project.

The project investigates whether deep recurrent workload demand forecasting (comparing **LSTM** and **GRU**), combined with **K-Means++** workload profiling, **dynamic load-aware physical machine (PM) thresholds**, calibrated **nonlinear SPECpower energy modeling**, and **Ant Colony Optimization (ACO)**, can reduce data centre electrical energy consumption while preventing Service Level Agreement violations (SLAv).

The evaluation is grounded in empirical telemetry from the **Google Cluster Data 2019** repository, simulating an enterprise heterogeneous cluster of **798 PMs** hosting **7,644 VMs**.

---

## 2. Core Quantitative Findings at a Glance

### 1. Workload Forecasting Benchmark (10m, 30m, 60m Horizons)
Evaluated on 42,199 held-out sequence windows from 600 representative Google Borg workload entities:
* **LSTM vs. GRU:** LSTM achieves marginally lower error than GRU across all horizons (at 30m: LSTM $\text{MAE} = 0.00497, R^2 = 0.5750$ vs. GRU $\text{MAE} = 0.00510, R^2 = 0.5361$).
* **The Baseline Paradox:** Simple rolling **Moving Average** achieves lower MAE and higher $R^2$ than both deep recurrent networks across all three horizons (at 30m: $\text{MAE} = 0.00391, R^2 = 0.7213$). Deep networks do not hold an empirical accuracy advantage on this stationary trace.
* **Model Efficiency:** GRU requires **40,162 trainable parameters** vs. LSTM's **52,834 parameters**—an exact **23.98% parameter reduction**—while training **18.8% to 29.9% faster** and cutting 30m inference latency from 5.51 ms to 4.41 ms.

### 2. Full-Scale Cluster Scheduling Benchmark (7,644 VMs on 798 PMs)
* **Energy Reduction:** The proposed ACO framework achieves **24,487.22 Wh**, reducing energy by **68.21%** compared to Google's uncoordinated trace baseline (77,032.51 Wh) and by **9.49%** compared to Modified Best Fit (27,054.43 Wh).
* **Consolidation Gain:** Active PMs decrease from 798 to **152 hosts** (an **80.95% reduction**), transitioning 646 PMs into 10 W sleep state.
* **SLA and Placement:** 100% placement completeness (7,644/7,644 VMs placed), zero overloaded PMs, and zero SLA violations ($\text{SLAv} = 0.0$).
* **Scheduling Invariance:** **Proposed (LSTM + ACO)**, **Proposed (GRU + ACO)**, and **Reactive ACO (without forecasting)** yield **exactly identical scheduling metrics** (24,487.22 Wh, 152 active PMs). Under healthy cluster utilization (65.98%), dynamic threshold contraction is not triggered during initial placement.

### 3. Component Ablation Study (1,500 VMs across 150 PMs)
* Replacing ACO with Greedy First Fit increases active PMs from 27 to 28, increases energy by +64.66 Wh (4,544.20 Wh vs. 4,479.54 Wh), and degrades resource imbalance from 0.1359 to 0.1716.
* Disabling migration gating allows an unproductive migration, slightly increasing energy to 4,479.57 Wh.
* In contrast, disabling prediction or dynamic thresholds leaves energy and active PM count unchanged (4,479.54 Wh, 27 PMs), because the cluster operates below congestion levels.

---

## 3. Claims Supported by Current Evidence vs. Claims Requiring Further Validation

To maintain strict scientific integrity, the table below partitions project findings into verified empirical results versus unproven hypotheses.

| Research Claim | Current Evidentiary Status | Supporting Artifact / Verification Proof | Limitations / Caveats |
| :--- | :---: | :--- | :--- |
| **1. ACO Consolidates Better than First-Fit Heuristics** | **SUPPORTED BY EVIDENCE** | `table8_cluster_performance_comparison.csv`, `table13_ablation_study.csv` | ACO achieves 152 PMs vs. 154 PMs in MFFA and 184 PMs in MBFA; 27 PMs vs. 28 PMs in ablation. |
| **2. GRU is 23.98% More Parameter-Efficient Than LSTM** | **SUPPORTED BY EVIDENCE** | `src/vm_scheduler/prediction/gru_model.py`, `tests/test_gru_and_extensions.py` | Exact counts verified: 40,162 GRU vs. 52,834 LSTM parameters. |
| **3. GRU Trains 18.8% to 29.9% Faster Than LSTM** | **SUPPORTED BY EVIDENCE** | `experiments/prediction/gru_vs_lstm_experiment_meta.json` | 38.97s vs. 48.00s at 30m; 33.66s vs. 48.00s at 60m across 30 epochs. |
| **4. Moving Average Outperforms LSTM/GRU on Borg Trace** | **SUPPORTED BY EVIDENCE** | `results/tables/table_gru_vs_lstm_prediction.csv` | Moving Average achieves lowest MAE (0.00391) and highest $R^2$ (0.7213) at 30m. |
| **5. SPECpower Non-Linearity Influences Energy Accounting** | **SUPPORTED BY EVIDENCE** | `results/tables/table14_sensitivity_analysis.csv` | Total cluster energy scales monotonically from 4,584.61 Wh ($k=1.0$) to 4,349.42 Wh ($k=2.0$). |
| **6. Predictive Scheduling Outperforms Reactive Scheduling** | **NOT SUPPORTED BY EVIDENCE** | `results/tables/table_scheduling_model_comparison.csv` | LSTM+ACO, GRU+ACO, and Reactive ACO produce identical metrics (24,487.22 Wh, 152 PMs). |
| **7. Dynamic Thresholding Reduces SLA Violations Under Bursts** | **NOT SUPPORTED BY EVIDENCE** | `results/tables/table13_ablation_study.csv` | Dynamic and static thresholds yield identical outcomes because cluster AUR is 65.98% ($< 75\%$). |
| **8. Scheduled VMs Utilize Learned Neural Forecasts** | **CONTRADICTED BY EVIDENCE** | `results/tables/prediction_provenance_audit.csv` | 100% of the 7,644 scheduled VMs received Category C fallback demand estimates due to ID mismatch. |
| **9. Four-Resource Scheduling (CPU, RAM, Disk, Network)** | **CONTRADICTED BY EVIDENCE** | `docs/initial_cloud_data_audit.md` | Google Cluster Data 2019 omits disk I/O and network bandwidth; scheduling is strictly 2-dimensional. |

---

## 4. Key Architectural and Engineering Deliverables

1. **Forecasting Models:**
   * Multivariate LSTM (`src/vm_scheduler/prediction/lstm.py`): 52,834 parameters. Checkpoints: `models/checkpoints/lstm_{10m,30m,60m}.pt`.
   * Multivariate GRU (`src/vm_scheduler/prediction/gru_model.py`): 40,162 parameters. Checkpoints: `models/checkpoints/gru_{10m,30m,60m}.pt`.
2. **Cluster Scheduling Pipeline:**
   * Ant Colony Optimization (`src/vm_scheduler/scheduling/aco.py`): Multi-resource fitness, candidate host filtering.
   * Net-Benefit Migration Manager (`src/vm_scheduler/scheduling/migration.py`): Economic cost-benefit gating.
   * Workload Profiler (`src/vm_scheduler/clustering/kmeanspp.py`): $K=6$ profile clustering.
3. **Publication-Quality Research Figures (`docs/research_figures/`):**
   * `fig1_lstm_vs_gru_mae.png`: MAE across forecasting horizons with baselines.
   * `fig2_lstm_vs_gru_r2.png`: $R^2$ scores highlighting Moving Average performance.
   * `fig3_model_parameter_comparison.png`: Visualizing the 23.98% GRU parameter reduction.
   * `fig4_model_training_time_comparison.png`: Training duration across 10m, 30m, 60m horizons.
   * `fig5_scheduling_energy_comparison.png`: Cluster energy comparison across all schedulers.
   * `fig6_ablation_study_comparison.png`: Isolation of framework component contributions.
   * `fig7_dataset_extraction_coverage.png`: Extraction coverage audit across Borg tables.
4. **Automated Test Suite:**
   * `tests/test_gru_and_extensions.py` (10 tests) and `tests/test_suite.py` (22 tests).
   * **32/32 tests passing (100% pass rate)** in 5.73s.

---

## 5. Major Limitations and Necessary Steps Before Publication

Before submitting these findings to a high-impact peer-reviewed journal (*IEEE TCC* / *ACM TOLS*):

1. **Reconcile Workload Identifiers:**
   * Current scheduled VMs in Subset B (`vm_00000`..`vm_07643`) do not share identifiers with usage time-series in Subset A (`collection_id_instance_index`).
   * A unified extraction pass must link tasks present in both event and usage logs to enable genuine Category A neural forecasting in the scheduler.
2. **Evaluate Under High-Utilization Congestion Scenarios:**
   * The current cluster consolidation ratio yields an average host utilization of 65.98%, which does not trigger dynamic thresholding ($T_{\text{base}} = 0.75$).
   * Stress-testing the cluster under high background loads ($U > 85\%$) or injecting sudden synthetic traffic bursts is required to demonstrate the SLA-protection benefit of proactive forecasting.
3. **Multi-Seed Statistical Significance:**
   * Run experiments across multiple random seeds ($N \ge 10$) and report confidence intervals for all energy, parameter, and latency metrics.
4. **Transparent Manuscript Alignment:**
   * Update the paper manuscript to explicitly report that Moving Average is competitive with deep neural models on stationary sequences, and that predictive and reactive ACO perform identically under non-congested cluster states.

---

## 6. Source-Artifact Reference Table

| Result / Claim Area | Primary Project Artifact | Format | Empirical Status |
| :--- | :--- | :---: | :---: |
| **Forecasting Benchmark (10m, 30m, 60m)** | `results/tables/table_gru_vs_lstm_prediction.csv` | CSV | Verified |
| **Detailed Forecasting Metadata** | `experiments/prediction/gru_vs_lstm_experiment_meta.json` | JSON | Verified |
| **Scheduling Model Comparison** | `results/tables/table_scheduling_model_comparison.csv` | CSV | Verified |
| **Full Cluster Baseline Benchmark** | `results/tables/table8_cluster_performance_comparison.csv` | CSV | Verified |
| **Component Ablation Benchmark** | `results/tables/table13_ablation_study.csv` | CSV | Verified |
| **SPECpower Sensitivity Analysis** | `results/tables/table14_sensitivity_analysis.csv` | CSV | Verified |
| **Prediction Provenance Accounting** | `results/tables/prediction_provenance_audit.csv` | CSV | Verified |
| **GCS Inventory & Extraction Audit** | `docs/initial_cloud_data_audit.md` | Markdown | Verified |
| **Automated Test Results** | `tests/test_gru_and_extensions.py`, `tests/test_suite.py` | PyTest | 32/32 Passed (100%) |
