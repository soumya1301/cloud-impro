# Energy-Aware Virtual Machine Scheduling Using Google Cluster Data 2019

A research project exploring **predictive workload analysis and energy-aware Virtual Machine (VM) scheduling** in cloud data centres using Google Cluster Data 2019.

The project combines workload preprocessing, K-Means++ clustering, deep-learning-based CPU and memory forecasting, physical-machine (PM) energy modelling, and Ant Colony Optimization (ACO) for VM placement. It also evaluates GRU and LSTM forecasting architectures and compares scheduling outcomes against baseline approaches.

---

## 1. Project Overview

Cloud data centres consume substantial energy while managing workloads with changing resource requirements. Efficient VM placement can improve resource utilization and reduce unnecessary physical-machine power consumption.

This project investigates an integrated framework that uses historical workload information to support VM scheduling decisions.

The main research objectives are:

- Analyse real-world cloud workload characteristics.
- Extract and preprocess CPU and memory utilization data.
- Identify workload patterns using K-Means++ clustering.
- Forecast future CPU and memory demand using LSTM and GRU models.
- Model physical-machine energy consumption using a nonlinear power function.
- Explore energy-aware VM placement using Ant Colony Optimization.
- Evaluate migration costs, SLA-related metrics and energy consumption.
- Compare forecasting architectures and scheduling baselines.

## 2. Research Framework

The implemented research pipeline consists of the following stages:

1. **Dataset acquisition and audit:** Examine Google Cluster Data 2019 and its available workload tables.
2. **Data preprocessing:** Prepare workload records, handle missing values and construct normalized features.
3. **Feature engineering:** Extract CPU, memory and workload characteristics.
4. **Workload profiling:** Apply K-Means++ clustering to identify workload groups.
5. **Workload forecasting:** Train and evaluate LSTM and GRU models for CPU and memory prediction.
6. **Energy modelling:** Estimate PM power consumption using a nonlinear utilization-based model.
7. **VM scheduling:** Evaluate resource-feasible VM placement using ACO and scheduling baselines.
8. **Migration analysis:** Examine migration costs and potential energy benefits.
9. **Experimental evaluation:** Compare forecasting accuracy, energy consumption, utilization, SLA violations and other scheduling metrics.

## 3. Dataset: Google Cluster Data 2019

The project uses **Google Cluster Data 2019**, a publicly available trace of Borg cluster workload activity.

The dataset contains information about machine characteristics, workload events, collection events and periodic instance resource usage.

### Dataset tables

| Dataset table | Purpose |
|---|---|
| `instance_usage` | Periodic CPU and memory utilization measurements |
| `instance_events` | Instance lifecycle events |
| `collection_events` | Collection and workload lifecycle information |
| `machine_events` | Physical-machine events and characteristics |
| `machine_attributes` | Machine attribute information |

### Local dataset organization

```text
data/
├── raw/
│   └── Sampled source Parquet files
├── samples/
│   └── Small dataset samples
├── interim/
│   └── Intermediate processed features
├── processed/
│   ├── subset_a_prediction_timeseries.parquet
│   ├── subset_a_vm_features.parquet
│   ├── subset_a_vm_features_labeled.parquet
│   ├── subset_b_vms.parquet
│   └── subset_b_pms.parquet
├── manifests/
│   ├── google_cluster_2019_full_manifest.csv
│   └── google_cluster_2019_inventory_summary.csv
└── parquet_previews/
    └── CSV previews of Parquet datasets
```

**Important dataset note:** The repository contains sampled and processed data, not a complete local copy of the approximately 10.28 TiB source dataset. Extraction coverage and data limitations are documented in the dataset audit.

## 4. Workload Profiling and Clustering

K-Means++ is used to group workloads according to their resource characteristics.

The clustering experiments examine different cluster configurations, including:

- K = 3
- K = 4
- K = 5
- K = 6

Cluster evaluation considers standard measures such as:

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index

The processed VM feature dataset also contains K-Means labels for K = 3 and K = 6.

## 5. Workload Forecasting: LSTM and GRU

Two recurrent neural-network architectures are implemented and evaluated.

### LSTM

The Long Short-Term Memory model is used to learn temporal patterns in CPU and memory utilization.

### GRU

The Gated Recurrent Unit model provides an alternative recurrent architecture with fewer trainable parameters in the evaluated configuration.

### Forecasting configuration

| Parameter | Configuration |
|---|---|
| Input features | CPU and memory utilization |
| Forecast horizons | 10, 30 and 60 minutes |
| LSTM architecture | Two-layer recurrent model |
| GRU architecture | Two-layer recurrent model |
| Hidden dimension | 64 |
| Output | CPU and memory demand |

### Forecasting evaluation

The models are compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score
- Normalized MAE
- Bounded MAPE
- Training time
- Inference latency
- Trainable parameter count

Moving-average and persistence forecasting methods are also considered as reference baselines.

**Current finding:** LSTM performs slightly better than GRU on overall MAE and R² in the reported evaluation. However, the moving-average baseline achieves lower MAE at all three forecast horizons. GRU has approximately 23.98% fewer trainable parameters than LSTM in the evaluated configuration.

## 6. Energy Modelling and VM Scheduling

The scheduling framework represents physical machines using resource capacities and power characteristics.

The nonlinear power model is:

\[
P(u)=P_{\mathrm{idle}}+
(P_{\mathrm{max}}-P_{\mathrm{idle}})u^k
\]

Where:

- \(P(u)\): Estimated power at utilization \(u\)
- \(P_{\mathrm{idle}}\): Idle power
- \(P_{\mathrm{max}}\): Maximum power
- \(k\): Nonlinearity parameter

### Ant Colony Optimization

ACO is used to explore VM-to-PM placement decisions while considering resource feasibility and energy-related objectives.

The scheduling experiments also consider:

- CPU and memory capacity constraints
- Active and sleeping physical machines
- PM utilization
- Energy consumption
- SLA violations
- VM migration count

### Migration analysis

Migration decisions are evaluated in terms of potential energy savings and migration cost. The objective is to avoid migrations that do not provide a sufficient benefit.

## 7. Experimental Evaluation and Results

The project includes forecasting benchmarks, scheduling comparisons, component ablations and sensitivity analysis.

### Forecasting comparison

The reported LSTM and GRU experiments cover 10-minute, 30-minute and 60-minute prediction horizons.

The current results indicate:

- LSTM achieves slightly better overall forecasting accuracy than GRU.
- GRU requires fewer trainable parameters.
- GRU offers a computational-efficiency trade-off.
- Moving-average forecasting remains a strong baseline.

### Scheduling experiments

The reported full-scale scheduling experiment contains:

| Parameter | Evaluated scale |
|---|---:|
| Virtual machines | 7,644 |
| Physical machines | 798 |

The reported energy consumption for the proposed scheduling configurations is 24,487.22 Wh.

A significant qualification is that the LSTM-assisted ACO, GRU-assisted ACO and reactive ACO configurations produce identical reported scheduling outcomes in this experiment. Consequently, the current evidence does not establish that forecasting improves placement decisions over reactive ACO.

### Ablation and sensitivity studies

The repository includes experiments examining:

- Forecasting contribution
- Dynamic threshold contribution
- ACO placement contribution
- Migration gating
- Nonlinear energy modelling
- Sensitivity to the energy-model nonlinearity parameter

The results and their limitations are described in the research reports.

## 8. Repository Structure

```text
cloud-impro/
│
├── archive/
├── configs/
│
├── data/
│   ├── raw/
│   ├── samples/
│   ├── interim/
│   ├── processed/
│   ├── manifests/
│   └── parquet_previews/
│
├── docs/
│   ├── initial_cloud_data_audit.md
│   ├── current_research_results_and_progress.md
│   ├── google_cluster_data_2019_dataset_and_preprocessing.md
│   ├── gru_lstm_comparative_validation_report.md
│   ├── gru_vs_lstm_comparative_analysis.md
│   ├── research_results_executive_summary.md
│   └── research_figures/
│
├── experiments/
├── models/
├── paper/
├── results/
│
├── scripts/
│   ├── generate_docs_figures.py
│   ├── generate_gru_figures.py
│   └── run_scheduling_gru_vs_lstm.py
│
├── src/
│   └── vm_scheduler/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── prediction/
│       ├── clustering/
│       ├── simulation/
│       ├── scheduling/
│       ├── energy/
│       ├── baselines/
│       ├── evaluation/
│       └── utils/
│
├── tests/
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 9. Installation and Reproducibility

### Requirements

- Python 3.11 or later
- Git
- pip
- Dependencies listed in `requirements.txt`

### Clone the repository

```bash
git clone https://github.com/soumya1301/cloud-impro.git
cd cloud-impro
```

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run automated tests

```bash
PYTHONPATH=src python -m pytest tests/test_suite.py -v
```

The repository includes scripts for forecasting, scheduling, comparative evaluation and figure generation. Consult each script and its configuration before running experiments. Some scripts require previously generated datasets or model checkpoints.

## 10. Research Documentation

Detailed methodology, results and limitations are available in the `docs/` directory.

| Document | Description |
|---|---|
| `current_research_results_and_progress.md` | Overall implementation and experimental progress |
| `gru_vs_lstm_comparative_analysis.md` | Technical comparison of GRU and LSTM |
| `google_cluster_data_2019_dataset_and_preprocessing.md` | Dataset audit and preprocessing methodology |
| `research_results_executive_summary.md` | Summary of research findings |
| `gru_lstm_comparative_validation_report.md` | Detailed GRU/LSTM validation report |
| `initial_cloud_data_audit.md` | Initial cloud dataset inventory and extraction audit |

## 11. Limitations and Future Work

The current implementation and experiments have several limitations that must be considered when interpreting the results.

1. **Dataset coverage:** Only a subset of the available Google Cluster Data 2019 Parquet shards has been extracted and processed.
2. **Workload identity:** The forecasting and scheduling datasets have an identifier-provenance limitation; the workload entities used in scheduling cannot all be directly mapped to the original forecasting sequences.
3. **Forecasting baselines:** Moving-average forecasting currently outperforms the recurrent models on MAE.
4. **Scheduling integration:** Forecast-assisted scheduling has not yet demonstrated a measurable advantage over reactive ACO in the reported full-scale experiment.
5. **Migration behaviour:** The reported full-scale scheduling experiment produces zero migrations.
6. **Resource dimensions:** The available telemetry and implemented experiments do not establish complete network and disk-I/O forecasting.

Future work should improve workload-to-scheduler identity mapping, validate forecasting-driven placement decisions, expand targeted trace extraction, and evaluate the framework under additional workload conditions.

## 12. License

This project is distributed under the MIT License. See [LICENSE](LICENSE).

## 13. Acknowledgements

The project uses Google Cluster Data 2019 for workload analysis and references SPECpower-related server power characteristics for energy modelling.

Dataset: [Google Cluster Data 2019](https://github.com/google/cluster-data)

---

**Project:** Energy-Aware Virtual Machine Scheduling  
**Dataset:** Google Cluster Data 2019  
**Forecasting Models:** LSTM and GRU  
**Clustering:** K-Means++  
**Scheduling:** Ant Colony Optimization  
**Language:** Python  
**Repository:** https://github.com/soumya1301/cloud-impro

