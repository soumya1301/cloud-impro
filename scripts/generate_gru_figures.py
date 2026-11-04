"""
Publication Figures Generation Script for GRU vs LSTM Extension:
Figures 11 through 15:
- Fig 11: Multi-Horizon Forecasting Accuracy (MAE, RMSE, R2) for LSTM vs GRU
- Fig 12: CPU vs Memory Demand Prediction Disaggregation
- Fig 13: Model Computational Efficiency, Parameter Count, and Inference Latency
- Fig 14: Downstream Cluster Scheduling Energy and Consolidation Comparison
- Fig 15: Empirical Prediction Provenance Distribution and Trace Topology
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

os.makedirs("results/figures", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)


def plot_fig11_accuracy():
    """Fig 11: Accuracy across 10m, 30m, 60m horizons for LSTM vs GRU."""
    df = pd.read_csv("results/tables/table_gru_vs_lstm_prediction.csv")
    horizons = df["Horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.20

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # 1. MAE
    ax1.bar(x - width * 1.5, df["LSTM_MAE"], width, label="LSTM", color="#1976d2", edgecolor="#0d47a1")
    ax1.bar(x - width * 0.5, df["GRU_MAE"], width, label="GRU", color="#388e3c", edgecolor="#1b5e20")
    ax1.bar(x + width * 0.5, df["MA_MAE"], width, label="Moving Avg", color="#f57c00", edgecolor="#e65100")
    ax1.bar(x + width * 1.5, df["Pers_MAE"], width, label="Persistence", color="#7b1fa2", edgecolor="#4a148c")
    ax1.set_ylabel("Mean Absolute Error (MAE)")
    ax1.set_title("(a) Multi-Horizon MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. RMSE
    ax2.bar(x - width * 0.6, df["LSTM_RMSE"], width * 1.2, label="LSTM", color="#1976d2", edgecolor="#0d47a1")
    ax2.bar(x + width * 0.6, df["GRU_RMSE"], width * 1.2, label="GRU", color="#388e3c", edgecolor="#1b5e20")
    ax2.set_ylabel("Root Mean Squared Error (RMSE)")
    ax2.set_title("(b) Multi-Horizon RMSE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. R2 Score
    ax3.plot(horizons, df["LSTM_R2"], marker="o", linewidth=2.5, markersize=8, label="LSTM ($R^2$)", color="#1976d2")
    ax3.plot(horizons, df["GRU_R2"], marker="s", linewidth=2.5, markersize=8, label="GRU ($R^2$)", color="#388e3c")
    ax3.plot(horizons, df["MA_R2"], marker="^", linewidth=2.0, linestyle="--", markersize=7, label="Moving Avg", color="#f57c00")
    ax3.plot(horizons, df["Pers_R2"], marker="d", linewidth=2.0, linestyle=":", markersize=7, label="Persistence", color="#7b1fa2")
    ax3.set_ylabel("Coefficient of Determination ($R^2$)")
    ax3.set_title("(c) Multi-Horizon $R^2$ Score")
    ax3.set_ylim([0.4, 0.8])
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Figure 11: Multi-Horizon Forecasting Accuracy Benchmark (LSTM vs. GRU vs. Baselines)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/fig11_gru_vs_lstm_accuracy.png")
    plt.savefig("paper/figures/fig11_gru_vs_lstm_accuracy.png")
    plt.close()
    print("Saved Figure 11.")


def plot_fig12_cpu_mem():
    """Fig 12: CPU vs Memory MAE across horizons."""
    df = pd.read_csv("results/tables/table_gru_vs_lstm_prediction.csv")
    horizons = df["Horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # CPU MAE
    ax1.bar(x - width / 2, df["LSTM_CPU_MAE"], width, label="LSTM CPU MAE", color="#1565c0", edgecolor="#0d47a1")
    ax1.bar(x + width / 2, df["GRU_CPU_MAE"], width, label="GRU CPU MAE", color="#2e7d32", edgecolor="#1b5e20")
    ax1.set_ylabel("CPU MAE (Normalized Units)")
    ax1.set_title("(a) CPU Utilization Prediction Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Memory MAE
    ax2.bar(x - width / 2, df["LSTM_MEM_MAE"], width, label="LSTM Memory MAE", color="#42a5f5", edgecolor="#1565c0")
    ax2.bar(x + width / 2, df["GRU_MEM_MAE"], width, label="GRU Memory MAE", color="#66bb6a", edgecolor="#2e7d32")
    ax2.set_ylabel("Memory MAE (Normalized Units)")
    ax2.set_title("(b) Memory Utilization Prediction Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Figure 12: Disaggregated Resource Dimension Forecast Errors (CPU vs. Memory)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/fig12_cpu_mem_prediction_comparison.png")
    plt.savefig("paper/figures/fig12_cpu_mem_prediction_comparison.png")
    plt.close()
    print("Saved Figure 12.")


def plot_fig13_efficiency():
    """Fig 13: Model Complexity, Parameter Efficiency, and Latency."""
    df = pd.read_csv("results/tables/table_gru_vs_lstm_prediction.csv")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1. Parameter count
    models = ["LSTM", "GRU"]
    params = [52834, 40162]
    colors = ["#1976d2", "#388e3c"]
    bars1 = ax1.bar(models, params, color=colors, width=0.5, edgecolor="#212121")
    ax1.set_ylabel("Trainable Parameters")
    ax1.set_title("(a) Model Parameter Footprint")
    ax1.set_ylim([0, 65000])
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1500, f"{int(yval):,}", ha="center", weight="bold")
    ax1.text(0.5, 0.25, "-23.98%\nParameters in GRU", transform=ax1.transAxes, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#388e3c"))
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Training Duration (30m Horizon)
    lstm_train = float(df.loc[df["Horizon"] == "30m", "LSTM_Train_Time_s"].values[0])
    gru_train = float(df.loc[df["Horizon"] == "30m", "GRU_Train_Time_s"].values[0])
    bars2 = ax2.bar(models, [lstm_train, gru_train], color=colors, width=0.5, edgecolor="#212121")
    ax2.set_ylabel("Training Duration (seconds, 30 epochs)")
    ax2.set_title("(b) Empirical Training Time (30m)")
    ax2.set_ylim([0, 60])
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}s", ha="center", weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Inference Latency
    horizons = df["Horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.35
    ax3.bar(x - width / 2, df["LSTM_Latency_ms"], width, label="LSTM Latency", color="#1976d2", edgecolor="#0d47a1")
    ax3.bar(x + width / 2, df["GRU_Latency_ms"], width, label="GRU Latency", color="#388e3c", edgecolor="#1b5e20")
    ax3.set_ylabel("Per-Batch Inference Latency (ms)")
    ax3.set_title("(c) Multi-Horizon Inference Speed")
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons)
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Figure 13: Model Parameter Efficiency, Training Time, and Inference Latency", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/fig13_model_efficiency_latency.png")
    plt.savefig("paper/figures/fig13_model_efficiency_latency.png")
    plt.close()
    print("Saved Figure 13.")


def plot_fig14_scheduling_energy():
    """Fig 14: Downstream scheduling energy and active PMs."""
    csv_path = "results/tables/table_scheduling_model_comparison.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping Figure 14: {csv_path} not ready yet.")
        return
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["Proposed\n(LSTM+ACO)", "Proposed\n(GRU+ACO)", "Reactive ACO\n(No Forecast)"]
    colors = ["#1976d2", "#388e3c", "#f57c00"]

    # 1. Total Energy
    bars1 = ax1.bar(labels, df["total_energy_wh"], color=colors, width=0.55, edgecolor="#212121")
    ax1.set_ylabel("Total Cluster Energy (Wh)")
    ax1.set_title("(a) Total Energy Consumption (1h)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.01), f"{yval:,.1f}", ha="center", weight="bold")

    # 2. Active PMs
    bars2 = ax2.bar(labels, df["active_pms"], color=colors, width=0.55, edgecolor="#212121")
    ax2.set_ylabel("Active Physical Machines")
    ax2.set_title("(b) Active Host Consolidation")
    ax2.set_ylim([0, 800])
    ax2.grid(True, linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 15, f"{int(yval)}", ha="center", weight="bold")

    # 3. SLA Violation Rate
    bars3 = ax3.bar(labels, df["SLAv"] * 100, color=colors, width=0.55, edgecolor="#212121")
    ax3.set_ylabel("SLA Violation Rate (%)")
    ax3.set_title("(c) Multi-Resource SLA Violation")
    ax3.grid(True, linestyle="--", alpha=0.6)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 0.0001, f"{yval:.4f}%", ha="center", weight="bold")

    plt.suptitle("Figure 14: Downstream Scheduling Performance: LSTM vs. GRU vs. Reactive ACO", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/fig14_scheduling_energy_comparison.png")
    plt.savefig("paper/figures/fig14_scheduling_energy_comparison.png")
    plt.close()
    print("Saved Figure 14.")


def plot_fig15_provenance():
    """Fig 15: Prediction Provenance Distribution and Topology."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Pie chart of Categories
    categories = [
        "Category A\n(Empirical History)",
        "Category B\n(Insufficient History)",
        "Category C\n(Request-Based Fallback)"
    ]
    sizes = [0, 0, 7644]
    colors = ["#43a047", "#fdd835", "#e53935"]
    explode = (0, 0, 0.08)

    ax1.pie(sizes, explode=explode, labels=categories, colors=colors, autopct="%1.1f%%",
            startangle=140, textprops={"fontsize": 11, "weight": "bold"})
    ax1.set_title("(a) VM Prediction Provenance Accounting\n(Total Population = 7,644 VMs)")

    # Data Architecture and Audit Traceability Explanation
    ax2.axis("off")
    audit_text = (
        "EMPIRICAL PROVENANCE AUDIT & TRACE RECONCILIATION\n"
        "────────────────────────────────────────────────────────\n"
        "• Total Evaluated VMs: 7,644 (100.0%)\n"
        "• Category A (Exact Empirical Time Series Match): 0 (0.0%)\n"
        "• Category B (Insufficient Observations < 12 steps): 0 (0.0%)\n"
        "• Category C (Telemetry-Grounded Request Fallback): 7,644 (100.0%)\n\n"
        "SCIENTIFIC ROOT CAUSE IDENTIFICATION:\n"
        "1. Disjoint Identifier Namespaces:\n"
        "   - Subset B VMs: Extracted from instance events with synthetic keys ('vm_00000').\n"
        "   - Subset A Time Series: Indexed by composite keys ('collection_id_instance_index').\n\n"
        "2. Single-Shard Google Borg 2019 Sampling:\n"
        "   - Google Borg 2019 instance_events and usage tables are sampled across\n"
        "     independent random hash shards (e.g., shard 170).\n"
        "   - Tasks sampled in the instance event shard were not co-sampled in the usage shard.\n\n"
        "3. Scientific Integrity Protocol:\n"
        "   - Model predictions in scheduling are explicitly documented as telemetry-grounded\n"
        "     demand estimates rather than pure autoregressive empirical forecasts."
    )
    ax2.text(0.02, 0.95, audit_text, fontsize=10.2, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#b0bec5", linewidth=1.5))

    plt.suptitle("Figure 15: Rigorous Prediction Provenance Accounting and Borg Trace Topology", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/fig15_prediction_provenance.png")
    plt.savefig("paper/figures/fig15_prediction_provenance.png")
    plt.close()
    print("Saved Figure 15.")


def main():
    plot_fig11_accuracy()
    plot_fig12_cpu_mem()
    plot_fig13_efficiency()
    plot_fig14_scheduling_energy()
    plot_fig15_provenance()


if __name__ == "__main__":
    main()
