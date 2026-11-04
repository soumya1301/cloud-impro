"""
Script to generate research figures for documentation in docs/research_figures/.
Generates 7 publication-quality figures:
1. LSTM vs GRU MAE across forecasting horizons (with baselines)
2. LSTM vs GRU R² across forecasting horizons (with baselines)
3. Model parameter comparison (recurrent layers vs output head)
4. Model training-time comparison across horizons
5. Scheduling energy comparison across all schedulers
6. Ablation study comparison
7. Dataset processing/extraction coverage
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set clean styling
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

out_dir = Path("docs/research_figures")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Figure 1: LSTM vs GRU MAE across forecasting horizons
def plot_fig1_mae():
    df = pd.read_csv("results/tables/table_gru_vs_lstm_prediction.csv")
    horizons = df["Horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width * 1.5, df["LSTM_MAE"], width, label="LSTM", color="#1976d2", edgecolor="#0d47a1")
    ax.bar(x - width * 0.5, df["GRU_MAE"], width, label="GRU", color="#388e3c", edgecolor="#1b5e20")
    ax.bar(x + width * 0.5, df["Pers_MAE"], width, label="Persistence", color="#7b1fa2", edgecolor="#4a148c")
    ax.bar(x + width * 1.5, df["MA_MAE"], width, label="Moving Average", color="#f57c00", edgecolor="#e65100")

    ax.set_ylabel("Mean Absolute Error (MAE, Normalized [0, 1])")
    ax.set_title("Figure 1: Mean Absolute Error (MAE) Across Forecasting Horizons (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc="upper left")
    ax.set_ylim([0, 0.007])
    ax.grid(True, linestyle="--", alpha=0.6)

    # Note moving average baseline
    ax.text(0.5, -0.15, "*Note: Moving Average exhibits lower overall MAE than both recurrent models across all horizons.",
            transform=ax.transAxes, ha="center", fontsize=9.5, style="italic")

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_lstm_vs_gru_mae.png")
    plt.close()
    print("Saved fig1_lstm_vs_gru_mae.png")

# 2. Figure 2: LSTM vs GRU R² across forecasting horizons
def plot_fig2_r2():
    df = pd.read_csv("results/tables/table_gru_vs_lstm_prediction.csv")
    horizons = df["Horizon"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(horizons, df["LSTM_R2"], marker="o", linewidth=2.5, markersize=8, label="LSTM ($R^2$)", color="#1976d2")
    ax.plot(horizons, df["GRU_R2"], marker="s", linewidth=2.5, markersize=8, label="GRU ($R^2$)", color="#388e3c")
    ax.plot(horizons, df["Pers_R2"], marker="d", linewidth=2.0, linestyle=":", markersize=7, label="Persistence ($R^2$)", color="#7b1fa2")
    ax.plot(horizons, df["MA_R2"], marker="^", linewidth=2.0, linestyle="--", markersize=7, label="Moving Average ($R^2$)", color="#f57c00")

    ax.set_ylabel("Coefficient of Determination ($R^2$, Higher is Better)")
    ax.set_title("Figure 2: Coefficient of Determination ($R^2$) Across Forecasting Horizons")
    ax.set_ylim([0.40, 0.80])
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Value labels
    for i, h in enumerate(horizons):
        ax.text(i, df.loc[df["Horizon"] == h, "LSTM_R2"].values[0] + 0.012, f"{df.loc[df['Horizon'] == h, 'LSTM_R2'].values[0]:.4f}", ha="center", fontsize=9, color="#1976d2", weight="bold")
        ax.text(i, df.loc[df["Horizon"] == h, "GRU_R2"].values[0] - 0.022, f"{df.loc[df['Horizon'] == h, 'GRU_R2'].values[0]:.4f}", ha="center", fontsize=9, color="#388e3c", weight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_lstm_vs_gru_r2.png")
    plt.close()
    print("Saved fig2_lstm_vs_gru_r2.png")

# 3. Figure 3: Model parameter comparison
def plot_fig3_params():
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["MultivariateGRU", "MultivariateLSTM"]
    recurrent_params = [38016, 50688]
    fc_params = [2146, 2146]

    x = np.arange(len(models))
    width = 0.45

    b1 = ax.bar(x, recurrent_params, width, label="Recurrent Layers (2-layer, H=64)", color="#42a5f5", edgecolor="#1565c0")
    b2 = ax.bar(x, fc_params, width, bottom=recurrent_params, label="Projection Head (64->32->2)", color="#ffb74d", edgecolor="#e65100")

    ax.set_ylabel("Trainable Parameter Count")
    ax.set_title("Figure 3: Trainable Parameter Footprint: GRU vs. LSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim([0, 65000])
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotate total counts
    totals = [40162, 52834]
    for i, tot in enumerate(totals):
        ax.text(i, tot + 1500, f"{tot:,} params", ha="center", fontweight="bold", fontsize=11)

    # Reduction callout
    ax.annotate("-23.98% Parameters\nin GRU (12,672 fewer)",
                xy=(0, 40162), xytext=(0.25, 48000),
                arrowprops=dict(facecolor="#2e7d32", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10.5, fontweight="bold", color="#1b5e20",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#81c784"))

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_model_parameter_comparison.png")
    plt.close()
    print("Saved fig3_model_parameter_comparison.png")

# 4. Figure 4: Model training time comparison
def plot_fig4_training_time():
    df = pd.read_csv("results/tables/table_gru_vs_lstm_prediction.csv")
    horizons = df["Horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - width / 2, df["LSTM_Train_Time_s"], width, label="LSTM Training Time", color="#1976d2", edgecolor="#0d47a1")
    b2 = ax.bar(x + width / 2, df["GRU_Train_Time_s"], width, label="GRU Training Time", color="#388e3c", edgecolor="#1b5e20")

    ax.set_ylabel("Training Duration (seconds, 30 epochs)")
    ax.set_title("Figure 4: Empirical Training Time Comparison (Single Run, 30 Epochs)")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylim([0, 60])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    for i, h in enumerate(horizons):
        t_lstm = df.loc[df["Horizon"] == h, "LSTM_Train_Time_s"].values[0]
        t_gru = df.loc[df["Horizon"] == h, "GRU_Train_Time_s"].values[0]
        ax.text(i - width / 2, t_lstm + 1.2, f"{t_lstm:.1f}s", ha="center", fontsize=10, weight="bold")
        ax.text(i + width / 2, t_gru + 1.2, f"{t_gru:.1f}s", ha="center", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_model_training_time_comparison.png")
    plt.close()
    print("Saved fig4_model_training_time_comparison.png")

# 5. Figure 5: Scheduling energy comparison
def plot_fig5_energy():
    # Schedulers from Table 8 and Table_scheduling_model_comparison
    labels = [
        "Google Trace\n(Simulated)",
        "Modified\nBest Fit (MBFA)",
        "Modified\nFirst Fit (MFFA)",
        "Proposed\n(LSTM + ACO)",
        "Proposed\n(GRU + ACO)",
        "Reactive ACO\n(No Forecast)"
    ]
    energies = [77032.51, 27054.43, 24626.13, 24487.22, 24487.22, 24487.22]
    active_pms = [798, 184, 154, 152, 152, 152]
    colors = ["#78909c", "#42a5f5", "#26a69a", "#1976d2", "#388e3c", "#f57c00"]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    bars = ax1.bar(labels, energies, color=colors, width=0.55, edgecolor="#263238")
    ax1.set_ylabel("Total Cluster Energy Consumption (Wh, 1-Hour Observation)", fontsize=11)
    ax1.set_title("Figure 5: Cluster Energy Consumption Comparison Across Schedulers")
    ax1.set_ylim([0, 90000])
    ax1.grid(True, linestyle="--", alpha=0.6)

    for bar, pms in zip(bars, active_pms):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1500, f"{yval:,.2f} Wh\n({pms} PMs)", ha="center", fontsize=9.5, weight="bold")

    # Critical callout for identical ACO variants
    ax1.text(0.78, 0.45, "CRITICAL SCIENTIFIC NOTE:\nLSTM+ACO, GRU+ACO, and Reactive ACO\nachieve identical 24,487.22 Wh & 152 PMs.\nForecast benefit is not isolated\nunder healthy cluster utilization (65.98%).",
             transform=ax1.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7", edgecolor="#fbc02d", linewidth=1.5))

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_scheduling_energy_comparison.png")
    plt.close()
    print("Saved fig5_scheduling_energy_comparison.png")

# 6. Figure 6: Ablation study comparison
def plot_fig6_ablation():
    df = pd.read_csv("results/tables/table13_ablation_study.csv")
    variants = [
        "Full Framework",
        "w/o Prediction",
        "w/o Dynamic Thresh",
        "w/o ACO (First Fit)",
        "w/o Migration Gate",
        "Linear Power"
    ]
    energies = df["total_energy_wph"].tolist()
    active_pms = df["active_pms"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#1976d2", "#78909c", "#78909c", "#e53935", "#ffb300", "#8e24aa"]

    # (a) Energy
    bars1 = ax1.bar(variants, energies, color=colors, width=0.55, edgecolor="#212121")
    ax1.set_ylabel("Total Cluster Energy (Wh)")
    ax1.set_title("(a) Total Energy Dissipation (1,500 VMs, 150 PMs)")
    ax1.set_ylim([4400, 4650])
    ax1.set_xticklabels(variants, rotation=25, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{yval:.2f}", ha="center", fontsize=9, weight="bold")

    # (b) Active PMs
    bars2 = ax2.bar(variants, active_pms, color=colors, width=0.55, edgecolor="#212121")
    ax2.set_ylabel("Active Physical Machines")
    ax2.set_title("(b) Active Consolidated PM Count")
    ax2.set_ylim([24, 30])
    ax2.set_xticklabels(variants, rotation=25, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.15, f"{int(yval)}", ha="center", fontsize=10, weight="bold")

    plt.suptitle("Figure 6: Component Ablation Benchmark (1,500 VMs Across 150 PMs)", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_ablation_study_comparison.png")
    plt.close()
    print("Saved fig6_ablation_study_comparison.png")

# 7. Figure 7: Dataset processing/extraction coverage
def plot_fig7_coverage():
    tables = [
        "collection_events",
        "instance_events",
        "instance_usage",
        "Total Parquet"
    ]
    total_shards = [116, 14126, 92468, 106726]
    completed_shards = [116, 8654, 28, 8798]
    percentages = [100.0, 61.26, 0.03, 8.24]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    colors = ["#43a047", "#1e88e5", "#e53935", "#8e24aa"]

    # (a) Percentage completion
    bars = ax1.bar(tables, percentages, color=colors, width=0.5, edgecolor="#212121")
    ax1.set_ylabel("Extraction Coverage (% of Parquet Shards)")
    ax1.set_title("(a) Cloud Extraction Coverage by Borg Table")
    ax1.set_ylim([0, 115])
    ax1.grid(True, linestyle="--", alpha=0.6)
    for bar, pct in zip(bars, percentages):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{pct:.2f}%", ha="center", fontsize=10.5, weight="bold")

    # (b) Shard counts (log scale)
    x = np.arange(len(tables))
    width = 0.35
    ax2.bar(x - width/2, total_shards, width, label="Total Inventoried Shards", color="#b0bec5", edgecolor="#37474f")
    ax2.bar(x + width/2, completed_shards, width, label="Completed Shards", color=colors, edgecolor="#212121")
    ax2.set_yscale("log")
    ax2.set_ylabel("Shard Count (Logarithmic Scale)")
    ax2.set_title("(b) Shard Volume: Completed vs. Total Inventoried")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tables)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Note
    ax2.text(0.5, -0.15, "*Note: instance_usage has only 28 of 92,468 shards extracted (~0.03%), representing extreme sparsity.",
             transform=ax2.transAxes, ha="center", fontsize=9.5, style="italic")

    plt.suptitle("Figure 7: Google Cluster Data 2019 Extraction Coverage Audit", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_dataset_extraction_coverage.png")
    plt.close()
    print("Saved fig7_dataset_extraction_coverage.png")


def main():
    plot_fig1_mae()
    plot_fig2_r2()
    plot_fig3_params()
    plot_fig4_training_time()
    plot_fig5_energy()
    plot_fig6_ablation()
    plot_fig7_coverage()

if __name__ == "__main__":
    main()
