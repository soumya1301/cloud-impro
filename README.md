# LSTM-Assisted, K-Means++-Driven, and ACO-Based Energy-Aware VM Scheduling

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research implementation, experimental evaluation framework, and academic manuscript extending the VM placement methodology described in **Alelyani et al. (2026)** (*Journal of Cloud Computing*, 15:37). Grounded in the **Google Cluster Data 2019** dataset and calibrated against the **SPECpower_ssj2008** benchmark for HP ProLiant G5 and G4 enterprise servers.

---

## Key Framework Innovations
- **Multivariate LSTM Workload Forecasting:** Predicts future CPU and RAM demands at 10m, 30m, and 60m horizons.
- **K-Means++ Workload Profiling:** Systematically evaluated across $K \in \{3, 4, 5, 6\}$ with Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics ($K=6$ selected as optimal).
- **Dynamic Predicted-Load-Aware PM Thresholds:** Backs off host threshold during predicted traffic surges to prevent SLA violations.
- **Calibrated SPECpower Nonlinear Energy Model:** $P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) u^k$ ($k=1.4$).
- **Ant Colony Optimization (ACO) VM Placement:** Multi-objective metaheuristic optimizing energy, active hosts, resource balance, and SLA risk.
- **Net-Benefit-Gated Live Migration:** Migrates VMs only when energy saved exceeds migration cost and degradation penalties.

---

## Repository Structure
```text
.
├── configs/               # Modular YAML configuration files (data, lstm, clustering, aco, etc.)
├── data/
│   ├── manifests/         # Google Cluster Data 2019 manifest and inventory summaries
│   └── processed/         # Extracted experimental Subsets A (ML/Profiling) and B (Scheduling)
├── docs/                  # Detailed documentation & audits
│   ├── initial_cloud_data_audit.md
│   └── reference_paper_analysis.md
├── models/
│   └── checkpoints/       # Trained PyTorch LSTM checkpoints (10m, 30m, 60m)
├── paper/                 # Academic publication assets
│   ├── manuscript.md      # Full academic research paper
│   ├── references.bib     # Complete BibTeX bibliography
│   └── figures/           # Publication-quality figures (300 DPI)
├── results/
│   ├── tables/            # Benchmark results, evaluation metrics, ablation tables (CSV)
│   ├── figures/           # Generated PNG visualisations
│   └── predictions/       # Compressed test set predictions (.npz)
├── scripts/               # End-to-end execution scripts
│   ├── extract_subset.py
│   ├── train_lstm.py
│   ├── run_clustering.py
│   ├── run_scheduling.py
│   ├── run_ablation.py
│   └── generate_report.py
├── src/vm_scheduler/      # Modular Python package
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── prediction/
│   ├── clustering/
│   ├── simulation/
│   ├── scheduling/
│   ├── energy/
│   ├── baselines/
│   ├── evaluation/
│   └── utils/
└── tests/                 # Automated pytest regression suite
```

---

## Quickstart & Reproducibility Guide

### 1. Installation
Clone this repository and install dependencies in a Python 3.11+ environment:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Automated Test Suite
```bash
pytest tests/test_suite.py
```

### 3. Step-by-Step Phase Execution Commands

#### Step 1: Targeted Dataset Extraction
Extract Subsets A and B from the audited Google Cluster Data 2019 trace:
```bash
python scripts/extract_subset.py
```

#### Step 2: Train Multivariate LSTM Forecaster
Train the 2-layer multivariate LSTM across 10m, 30m, and 60m forecast horizons:
```bash
python scripts/train_lstm.py
```

#### Step 3: Run K-Means++ Clustering & Profiling
Evaluate cluster validity across $K \in \{3, 4, 5, 6\}$ and assign interpretable workload labels:
```bash
python scripts/run_clustering.py
```

#### Step 4: Run Comparative Scheduling Benchmarks
Simulate 7,644 VMs across 798 PMs and benchmark against Google Trace, MBFA, MFFA, Sercon, and Round Robin:
```bash
python scripts/run_scheduling.py
```

#### Step 5: Execute Ablation Studies and Sensitivity Analysis
Quantify component contributions and power model non-linearity:
```bash
python scripts/run_ablation.py
```

#### Step 6: Generate Publication Figures and Final Report
Render 300 DPI publication plots to `results/figures/` and `paper/figures/`:
```bash
python scripts/generate_report.py
```

---

## Key Experimental Results Summary

| Metric | Google Trace | MBFA | MFFA | Sercon | Round Robin | **Proposed Framework** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Energy (Wh)** | 77,032.51 | 78,826.33 | 77,414.69 | 77,167.28 | 77,105.51 | **63,431.12** (**-17.65%**) |
| **Active PMs (of 798)** | 798 | 798 | 788 | 785 | 798 | **619** (**-22.43%**) |
| **Idle PMs in Sleep** | 0 | 0 | 10 | 13 | 0 | **179** |
| **Overloaded PMs** | 0 | 91 | 0 | 0 | 3 | **0** |
| **SLA Violations (SLAv)** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| **Migrated VMs** | 0 | 0 | 0 | 0 | 0 | **8** |

---

## Citation
```bibtex
@article{vm_scheduling_2026,
  title={Predictive, Workload-Aware, and Energy-Proportional Virtual Machine Consolidation in Cloud Data Centres: An LSTM, K-Means++, and Ant Colony Optimization Framework},
  author={Research Team},
  journal={IEEE Transactions on Cloud Computing},
  year={2026}
}
```

