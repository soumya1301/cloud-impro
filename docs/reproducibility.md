# Reproducibility Guide: Step-by-Step Reproduction Instructions

This guide provides instructions to reproduce the complete empirical pipeline and all experimental results presented in the paper.

## 1. System Requirements & Environment
- **Operating System:** macOS (Apple Silicon or Intel x86_64) or Linux (Ubuntu 20.04+).
- **Python Version:** Python 3.10+ (Tested on Python 3.12.7).
- **Core Dependencies:**
  - `torch >= 2.0.0`
  - `pandas >= 2.0.0`
  - `numpy >= 1.24.0`
  - `scipy >= 1.10.0`
  - `scikit-learn >= 1.2.0`
  - `pyarrow >= 14.0.0`
  - `matplotlib >= 3.7.0`
  - `seaborn >= 0.12.0`
  - `pytest >= 7.0.0`

## 2. Directory Layout
Ensure the following directories exist prior to script execution:
```bash
mkdir -p data/raw data/samples data/interim data/processed
mkdir -p models/checkpoints
mkdir -p experiments/predictions experiments/clustering experiments/scheduling experiments/ablation experiments/sensitivity
mkdir -p results/figures results/tables results/logs results/placements results/summaries
mkdir -p paper/figures paper/tables
```

## 3. End-to-End Pipeline Execution

### Step 1: Preprocessing and Feature Extraction
Extracts raw 5-minute sampling intervals and computes the 11 statistical features:
```bash
python scripts/run_preprocessing.py
```
*Outputs:* `data/processed/subset_a_prediction_timeseries.parquet`, `data/processed/subset_a_vm_features.parquet`.

### Step 2: Multivariate LSTM Workload Forecasting
Trains PyTorch 2-layer LSTM models across 10m, 30m, and 60m horizons:
```bash
python scripts/train_lstm.py
```
*Outputs:* `models/checkpoints/lstm_10m.pt`, `lstm_30m.pt`, `lstm_60m.pt`, `results/tables/table3_prediction_results.csv`.

### Step 3: Unsupervised K-Means++ Clustering & Profiling
Evaluates cluster validity ($K \in \{3,4,5,6\}$):
```bash
python scripts/run_clustering.py
```
*Outputs:* `results/tables/table6_kmeanspp_evaluation.csv`, `data/processed/subset_a_vm_features_labeled.parquet`.

### Step 4: Multi-Objective Cluster Scheduling Simulation
Executes Proposed framework and baselines (Google Trace, MBFA, MFFA, Round Robin, Sercon):
```bash
python scripts/run_scheduling.py
```
*Outputs:* `results/tables/table10_baseline_comparison.csv`, placement JSON files in `results/placements/`.

### Step 5: Ablation Studies and Sensitivity Analysis
Validates component necessity and SPECpower exponent sensitivity:
```bash
python scripts/run_ablation.py
```
*Outputs:* `results/tables/table13_ablation_study.csv`, `results/tables/table14_sensitivity_analysis.csv`.

### Step 6: Publication Figures & Tables Generation
Renders all publication-quality vector plots and formatted CSV tables:
```bash
python scripts/generate_report.py
```

### Step 7: Automated Test Suite Verification
Validates software assertions, boundary conditions, and invariant tests:
```bash
pytest tests/test_suite.py -v
```

