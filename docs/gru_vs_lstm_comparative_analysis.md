# Standalone Comparative Technical and Scientific Analysis: GRU vs. LSTM Demand Forecasting Architectures

**Document Reference:** `docs/gru_vs_lstm_comparative_analysis.md`
**Project Workspace:** `/Users/soumyaprasad/Desktop/cloud improvements`
**Git Branch:** `scientific_validation_correction`
**Commit Reference:** `6b1c95b98ee271084ba845dc401be437d5f641ad`
**Evaluation Platform:** Apple Silicon / macOS Darwin (Darwin 24.3.0, arm64), Python 3.12.7, PyTorch 2.4.1
**Dataset Reference:** Google Cluster Data 2019 (Shard 170 Time Series)
**Held-Out Evaluation Split:** 42,199 sliding windows across 600 representative workload entities

---

## 3.1 Architectural Specification and Formulation

Both deep recurrent forecasting models are implemented as modular PyTorch neural network classes designed with identical input conventions, layer counts, and output projections to ensure a strictly controlled architectural comparison:

* **Multivariate LSTM:** Implemented in [`src/vm_scheduler/prediction/lstm.py`](file:///Users/soumyaprasad/Desktop/cloud%20improvements/src/vm_scheduler/prediction/lstm.py) as `MultivariateLSTM`.
* **Multivariate GRU:** Implemented in [`src/vm_scheduler/prediction/gru_model.py`](file:///Users/soumyaprasad/Desktop/cloud%20improvements/src/vm_scheduler/prediction/gru_model.py) as `MultivariateGRU`.

### Common Hyperparameters:
* **Input Dimension ($D_{\text{in}}$):** 2 (Normalized CPU Utilization $u_{\text{CPU}} \in [0, 1]$ and Normalized Memory Utilization $u_{\text{RAM}} \in [0, 1]$).
* **Historical Sequence Length ($L$):** 12 timesteps (corresponding to 60 minutes of operational telemetry at 5-minute sampling intervals).
* **Recurrent Hidden Dimension ($H$):** 64 units per recurrent layer.
* **Recurrent Depth ($N_{\text{layers}}$):** 2 stacked recurrent layers.
* **Recurrent Dropout ($p$):** 0.20 between recurrent layers.
* **Output Projection Head:** A two-stage feedforward network with non-linear activation:
  $$\text{Hidden} (64) \xrightarrow{\text{Linear}} 32 \xrightarrow{\text{ReLU}} \text{Dropout}(0.2) \xrightarrow{\text{Linear}} \text{Output} (2) \xrightarrow{\text{Sigmoid}}$$
  The terminal Sigmoid activation guarantees that predicted resource utilization vectors $\hat{\mathbf{y}} = [\hat{u}_{\text{CPU}}, \hat{u}_{\text{RAM}}]$ strictly respect the physical machine capacity bounds $[0, 1]$.

### Recurrent Mathematical Formulations:

#### 1. Long Short-Term Memory (LSTM) Formulation
LSTM regulates temporal information flow through three multiplicative gates (forget gate $\mathbf{f}_t$, input gate $\mathbf{i}_t$, output gate $\mathbf{o}_t$) and an internal additive memory cell $\mathbf{c}_t$:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

#### 2. Gated Recurrent Unit (GRU) Formulation
GRU streamlines recurrence by merging the cell state into the hidden state and coupling the input and forget mechanisms into an update gate $\mathbf{z}_t$, regulated by a reset gate $\mathbf{r}_t$:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z)$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r)$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h)$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

---

## 3.2 Verified Parameter Counts and Architectural Footprint

Parameter counts were verified by direct inspection of instantiated PyTorch models using `count_parameters()`.

### Parameter Breakdown Table:
| Structural Component | GRU Parameters | LSTM Parameters | Difference | Parameter Reduction in GRU |
| :--- | :---: | :---: | :---: | :---: |
| **Layer 1 Recurrent Weights** | 13,056 | 17,408 | -4,352 | -25.00% |
| **Layer 2 Recurrent Weights** | 24,960 | 33,280 | -8,320 | -25.00% |
| **Total Recurrent Layers** | **38,016** | **50,688** | **-12,672** | **-25.00%** |
| **Linear Projection 1 (64 $\to$ 32)** | 2,080 | 2,080 | 0 | 0.00% |
| **Linear Projection 2 (32 $\to$ 2)** | 66 | 66 | 0 | 0.00% |
| **Total Output Projection Head** | **2,146** | **2,146** | **0** | **0.00%** |
| **Total Trainable Parameters** | **40,162** | **52,834** | **-12,672** | **-23.98%** |

### Mathematical Parameter Derivations:
* **GRU Layer 1:** 3 sets of gates $\times [(D_{\text{in}} + H) \times H + 2H] = 3 \times [(2 + 64) \times 64 + 128] = 3 \times 4,352 = \mathbf{13,056}$.
* **GRU Layer 2:** 3 sets of gates $\times [(H + H) \times H + 2H] = 3 \times [(64 + 64) \times 64 + 128] = 3 \times 8,320 = \mathbf{24,960}$.
* **LSTM Layer 1:** 4 sets of gates $\times [(D_{\text{in}} + H) \times H + 2H] = 4 \times [(2 + 64) \times 64 + 128] = 4 \times 4,352 = \mathbf{17,408}$.
* **LSTM Layer 2:** 4 sets of gates $\times [(H + H) \times H + 2H] = 4 \times [(64 + 64) \times 64 + 128] = 4 \times 8,320 = \mathbf{33,280}$.
* **Projection Head:** $(64 \times 32 + 32) + (32 \times 2 + 2) = 2,080 + 66 = \mathbf{2,146}$.

$$\text{Exact GRU Parameter Reduction} = \frac{52,834 - 40,162}{52,834} \times 100\% = \mathbf{23.98\%}$$

---

## 3.3 Multi-Horizon Forecasting Accuracy Comparison

Both models and baselines were trained and tested on identical splits of 42,199 held-out sequence windows across three forecast horizons:
* **10m Horizon ($h = 2$ steps):** 10-minute ahead prediction.
* **30m Horizon ($h = 6$ steps):** 30-minute ahead prediction (primary operational scheduling window).
* **60m Horizon ($h = 12$ steps):** 60-minute ahead long-range prediction.

### Comprehensive Benchmark Table:
All values verified from `results/tables/table_gru_vs_lstm_prediction.csv` and `experiments/prediction/gru_vs_lstm_experiment_meta.json`.

| Horizon | Model | Overall MAE | Overall RMSE | $R^2$ Score | NMAE | Bounded MAPE (%) | Raw MAPE (%) | CPU MAE | Memory MAE | Train Time (s) | Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10m** | **LSTM** | 0.00455 | 0.01328 | 0.5924 | 0.01495 | 31.23% | 292.70% | 0.00643 | 0.00267 | 48.00 | 3.41 |
| | **GRU** | 0.00470 | 0.01366 | 0.5690 | 0.01542 | 32.35% | 287.09% | 0.00679 | 0.00260 | 40.66 | 3.57 |
| | Persistence | 0.00422 | 0.01387 | 0.5559 | 0.01385 | 36.53% | 197.68% | 0.00648 | 0.00196 | — | — |
| | **Moving Average** | **0.00363** | **0.01085** | **0.7279** | **0.01191** | **28.76%** | **238.03%** | **0.00552** | **0.00174** | — | — |
| **30m** | **LSTM** | 0.00497 | 0.01371 | 0.5750 | 0.01630 | 35.13% | 334.92% | 0.00710 | 0.00283 | 48.00 | 5.51 |
| | **GRU** | 0.00510 | 0.01433 | 0.5361 | 0.01675 | 35.14% | 370.41% | 0.00740 | 0.00281 | 38.97 | 4.41 |
| | Persistence | 0.00459 | 0.01413 | 0.5488 | 0.01505 | 37.15% | 275.99% | 0.00697 | 0.00220 | — | — |
| | **Moving Average** | **0.00391** | **0.01110** | **0.7213** | **0.01285** | **29.26%** | **314.56%** | **0.00583** | **0.00200** | — | — |
| **60m** | **LSTM** | 0.00531 | 0.01429 | 0.5444 | 0.01956 | 39.78% | 372.11% | 0.00781 | 0.00281 | 48.00 | 5.38 |
| | **GRU** | 0.00571 | 0.01478 | 0.5124 | 0.02102 | 36.44% | 457.86% | 0.00820 | 0.00321 | 33.66 | 3.57 |
| | Persistence | 0.00514 | 0.01556 | 0.4593 | 0.01894 | 39.68% | 384.53% | 0.00775 | 0.00253 | — | — |
| | **Moving Average** | **0.00432** | **0.01200** | **0.6788** | **0.01589** | **31.45%** | **398.64%** | **0.00643** | **0.00220** | — | — |

---

## 3.4 Computational Efficiency and Resource Consumption

### Training Configuration:
* Epochs: 30
* Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
* Learning Rate: $\eta = 0.001$ with ReduceLROnPlateau scheduling
* Batch Size: 64
* Loss Function: Mean Squared Error (MSE)
* Platform: Apple Silicon (M-series, Darwin 24.3.0), Single-threaded / MPS acceleration

### Computational Performance Metrics:
| Metric | Multivariate GRU | Multivariate LSTM | Absolute Advantage | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Trainable Parameters** | **40,162** | 52,834 | -12,672 params | **23.98% lighter footprint** |
| **Training Duration (10m Horizon)** | **40.66 s** | 48.00 s | -7.34 s | **15.29% faster training** |
| **Training Duration (30m Horizon)** | **38.97 s** | 48.00 s | -9.03 s | **18.81% faster training** |
| **Training Duration (60m Horizon)** | **33.66 s** | 48.00 s | -14.34 s | **29.88% faster training** |
| **Inference Latency (10m Horizon)** | 3.57 ms | **3.41 ms** | +0.16 ms | LSTM 4.69% faster |
| **Inference Latency (30m Horizon)** | **4.41 ms** | 5.51 ms | -1.10 ms | **GRU 19.96% faster** |
| **Inference Latency (60m Horizon)** | **3.57 ms** | 5.38 ms | -1.81 ms | **GRU 33.64% faster** |

*Methodological Note:* The reported training durations and inference latencies represent recorded single-run executions under identical environment conditions. While statistically indicative, repeated-run multi-seed benchmarks ($N \ge 10$) with confidence intervals should be conducted in future work.

---

## 3.5 Scientific Interpretation and Trade-Off Analysis

### 1. The Moving Average Baseline Paradox
The most striking empirical result is that the rolling **Moving Average** baseline achieves substantially lower error ($\text{MAE} = 0.00391, R^2 = 0.7213$ at 30m) than both deep recurrent networks ($\text{MAE} = 0.00497, R^2 = 0.5750$ for LSTM; $\text{MAE} = 0.00510, R^2 = 0.5361$ for GRU).
* **Cause:** The Google Borg usage telemetry in Shard 170 consists predominantly of long-running, stationary, low-utilization tasks. For stationary sequences, a simple moving window averages out measurement noise without the gradient variance, parameter estimation error, or regularization penalties inherent in training deep neural networks.
* **Implication:** In real-world data centre forecasting pipelines, complex deep recurrent networks must be validated against rolling statistical baselines before being deployed. For stationary workloads, classical filters may suffice.

### 2. Recurrent Gate Mechanics: Why LSTM Leads GRU in Accuracy
When comparing LSTM and GRU directly:
* LSTM maintains a consistent accuracy advantage across all horizons ($R^2$ is $+0.0234$ higher at 10m, $+0.0389$ higher at 30m, and $+0.0320$ higher at 60m).
* **Structural Explanation:** In Borg traces, workload changes often manifest as sudden step-function transitions (e.g., job completion, garbage collection phases). In GRU, the reset gate $\mathbf{r}_t$ directly couples the past hidden state into the new candidate state $\tilde{\mathbf{h}}_t$. In contrast, LSTM retains an additive cell state $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ that isolates internal historical memory from the hidden output $\mathbf{h}_t$. This enables LSTM to preserve underlying baseline load levels while capturing high-frequency bursts without corrupting the historical state.

### 3. Asymmetric Resource Predictability (CPU vs. Memory)
* Memory utilization is markedly more predictable ($\text{MAE} \approx 0.0026 - 0.0032$, $R^2 \approx 0.67 - 0.72$) than CPU utilization ($\text{MAE} \approx 0.0064 - 0.0082$, $R^2 \approx 0.45 - 0.55$).
* Containers in Borg allocate fixed memory footprints that vary slowly, whereas CPU scheduling cycles fluctuate rapidly with user traffic.

### 4. Accuracy vs. Efficiency Trade-Off
* **LSTM:** Delivers higher forecasting accuracy ($+7.2\%$ relative $R^2$ gain at 30m), making it preferable when cloud operators prioritize minimizing prediction error to avoid SLA violations.
* **GRU:** Delivers $23.98\%$ fewer parameters and up to $29.9\%$ faster training, making it attractive for edge devices, constrained hypervisors, or distributed continuous re-training architectures where memory and compute budgets are strictly constrained.

---

## 3.6 Downstream Scheduling Integration Comparison

To evaluate whether the observed differences between LSTM and GRU translate into downstream cluster placement benefits, both models were integrated into the Ant Colony Optimization (ACO) scheduling engine alongside a reactive baseline.

### Experimental Setup:
* Cluster: 7,644 VMs placed onto 798 PMs (HP ProLiant G5/G4 SPECpower cluster).
* Random Seed: $S = 42$ (identical cluster configuration across all variants).

### Scheduling Results Table:
Verified against `results/tables/table_scheduling_model_comparison.csv`.

| Metric | Proposed (LSTM + ACO) | Proposed (GRU + ACO) | Reactive ACO (No Forecast) | Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Total Energy Consumption** | **24,487.22 Wh** | **24,487.22 Wh** | **24,487.22 Wh** | **0.00 Wh (0.00%)** |
| **Active Physical Machines** | **152 PMs** | **152 PMs** | **152 PMs** | **0 PMs** |
| **Sleeping Physical Machines** | **646 PMs** | **646 PMs** | **646 PMs** | **0 PMs** |
| **SLA Violations ($\text{SLAv}$)** | **0.000000** | **0.000000** | **0.000000** | **0.0** |
| **Average Utilization Ratio (AUR)** | **65.98%** | **65.98%** | **65.98%** | **0.00%** |
| **Multi-Resource Imbalance** | **0.1752** | **0.1752** | **0.1752** | **0.0000** |
| **Executed Migrations** | **0** | **0** | **0** | **0** |
| **Placement Completeness** | **100.0% (7,644/7,644)** | **100.0% (7,644/7,644)** | **100.0% (7,644/7,644)** | **0.0%** |

### Scientific Evaluation of Downstream Equivalence:
1. **Operating Load Below Congestion Threshold:**
   When 7,644 VMs are consolidated onto 152 PMs, the cluster average utilization is **65.98%**. The dynamic threshold logic contracts host capacity from 75% to 65% only when predicted host load exceeds 65% under surge. In steady-state operation at 65.98% without active surges, the threshold does not contract on healthy hosts.
2. **Initial Placement Feasibility:**
   ACO's global combinatorial search successfully packs all 7,644 VMs without overloading any physical machine ($N_{\text{overloaded}} = 0$). Because no hosts breach threshold, zero migrations are initiated in the 1-hour window.
3. **Category C Fallback Homogeneity:**
   Because 100% of the 7,644 scheduled VMs lack empirical usage sequences in the extracted shard, both inferencers fall back to profile-modulated requests, providing identical demand vectors to the placement engine.

**Scientific Conclusion:**
> **The experimental evidence DOES NOT demonstrate an advantage for LSTM over GRU—or for either forecasting model over reactive scheduling—in downstream cluster energy or SLA performance.** The downstream scheduling results are invariant across all three models in the current benchmark.

---

## 3.7 Conclusion

1. **Forecasting Accuracy:** LSTM marginally outperforms GRU across all horizons (30m MAE 0.00497 vs. 0.00510; $R^2$ 0.5750 vs. 0.5361).
2. **Computational Cost:** GRU achieves a decisive computational advantage, reducing model parameters by 23.98% (40,162 vs. 52,834) and training time by 18.8% to 29.9%.
3. **Baseline Hierarchy:** Moving Average outperforms both recurrent deep learning models on stationary Borg sequences ($R^2 = 0.7213$ at 30m).
4. **Downstream Scheduling Impact:** Under current cluster operating loads (65.98% AUR), predictive and reactive models achieve identical consolidation (24,487.22 Wh, 152 active PMs).

**Deployment Guidance:**
* Where hypervisor memory, lightweight binaries, or rapid online adaptation is required: **GRU is the pragmatically superior choice**.
* Where maximum predictive fidelity on bursty workloads is required regardless of parameter overhead: **LSTM is preferred**.
* Where downstream scheduling benefit is claimed: **Further validation under high-load congestion scenarios is mandatory**.

---

## 3.8 Source-Artifact Reference Table

| Result / Analysis Dimension | Primary Project Artifact | Format | Verification Status |
| :--- | :--- | :---: | :---: |
| **GRU Model Architecture** | `src/vm_scheduler/prediction/gru_model.py` | Python / PyTorch | Source-verified |
| **LSTM Model Architecture** | `src/vm_scheduler/prediction/lstm.py` | Python / PyTorch | Source-verified |
| **Parameter Counts & Structure** | `tests/test_gru_and_extensions.py` (test 2) | PyTest | Verified (40,162 vs 52,834) |
| **Multi-Horizon Metric Table** | `results/tables/table_gru_vs_lstm_prediction.csv` | CSV | Verified |
| **Detailed Forecasting Metadata** | `experiments/prediction/gru_vs_lstm_experiment_meta.json` | JSON | Verified |
| **Scheduling Model Comparison** | `results/tables/table_scheduling_model_comparison.csv` | CSV | Verified |
| **Prediction Provenance Accounting** | `results/tables/prediction_provenance_audit.csv` | CSV | Verified |
| **Trained GRU Checkpoints** | `models/checkpoints/gru_{10m,30m,60m}.pt` | PyTorch Binary | Verified |
| **Trained LSTM Checkpoints** | `models/checkpoints/lstm_{10m,30m,60m}.pt` | PyTorch Binary | Verified |
