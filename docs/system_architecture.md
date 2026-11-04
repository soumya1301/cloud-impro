# System Architecture Specification

## 1. Modular Architecture Overview

```mermaid
graph TD
    subgraph Data Layer
        A1[Google Cluster Data 2019] --> A2[Targeted Parquet Extractor]
        A2 --> A3[Subset A: ML & Profiling]
        A2 --> A4[Subset B: Simulation]
    end

    subgraph Intelligence Layer
        A3 --> B1[Multivariate LSTM Forecaster]
        A3 --> B2[K-Means++ Profiler]
        B1 --> B3[10m, 30m, 60m Demand Forecasts]
        B2 --> B4[Workload Profiles K=6]
    end

    subgraph Simulation & Scheduling Layer
        A4 --> C1[798 Heterogeneous PMs]
        B3 --> C2[Dynamic PM Threshold Engine]
        B4 --> C3[Diversity Invariant Filter <= 25%]
        C1 & C2 & C3 --> C4[ACO Placement Engine]
        C4 --> C5[Net-Benefit Migration Manager]
        C5 --> C6[Final VM Placement & Sleep Mode PMs]
    end

    subgraph Evaluation Layer
        C6 --> D1[SPECpower Nonlinear Energy Estimator]
        C6 --> D2[SLATAH, PDM, SLAv Metrics]
        C6 --> D3[Baseline Benchmark Suite]
    end
```

## 2. Package Structure
- `src/vm_scheduler/data/`: Manifest querying, schema inspection, and targeted extraction.
- `src/vm_scheduler/preprocessing/`: Normalization, chronological splitting, sliding-window generation.
- `src/vm_scheduler/features/`: Feature extraction, variance filtering, and collinearity pruning.
- `src/vm_scheduler/prediction/`: PyTorch LSTM model, trainer, persistence/moving average baselines, regression metrics.
- `src/vm_scheduler/clustering/`: K-Means++ evaluation suite across $K \in \{3,4,5,6\}$ and cluster profiler.
- `src/vm_scheduler/simulation/`: Heterogeneous physical machines (HP G5/G4), virtual machines, and capacity constraints.
- `src/vm_scheduler/scheduling/`: ACO multi-objective placement, dynamic thresholds, and net-benefit live migration engine.
- `src/vm_scheduler/energy/`: SPECpower nonlinear energy model ($P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) u^{1.4}$) and sleep power integration.
- `src/vm_scheduler/baselines/`: Modified First Fit (MFFA), Modified Best Fit (MBFA), Sercon, Round Robin (RRA).
- `src/vm_scheduler/evaluation/`: Standard research metric calculators matching Alelyani et al. Table 6.
- `src/vm_scheduler/utils/`: Deterministic seed setting, logging, and YAML configuration loading.

