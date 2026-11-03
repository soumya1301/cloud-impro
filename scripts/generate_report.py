"""
Publication-quality figures and visualization generation script.
Saves figures to results/figures/ and paper/figures/.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
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


def plot_workload_distribution():
    print("Generating Figure: Workload Distribution...")
    feat_df = pd.read_parquet("data/processed/subset_a_vm_features.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.histplot(feat_df["cpu_util_mean"], bins=40, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Average CPU Utilization Distribution (GCUs)")
    axes[0].set_xlabel("Normalized CPU Demand")
    axes[0].set_ylabel("VM Frequency")

    sns.histplot(feat_df["mem_util_mean"], bins=40, kde=True, ax=axes[1], color="#d95f02")
    axes[1].set_title("Average Memory Utilization Distribution (Ratio)")
    axes[1].set_xlabel("Normalized Memory Demand")
    axes[1].set_ylabel("VM Frequency")

    plt.tight_layout()
    fig.savefig("results/figures/fig2_workload_distribution.png")
    fig.savefig("paper/figures/fig2_workload_distribution.png")
    plt.close()


def plot_lstm_prediction():
    print("Generating Figure: LSTM Prediction vs Actual...")
    preds_data = np.load("results/predictions/preds_30m.npz")
    y_true = preds_data["y_true"][:120, 0] # First 120 steps of CPU
    y_lstm = preds_data["y_pred_lstm"][:120, 0]
    y_pers = preds_data["y_pred_pers"][:120, 0]
    y_ma = preds_data["y_pred_ma"][:120, 0]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    steps = np.arange(len(y_true))

    ax.plot(steps, y_true, label="Ground Truth (Trace)", color="black", linewidth=2.0)
    ax.plot(steps, y_lstm, label="Proposed LSTM (30m ahead)", color="#e41a1c", linestyle="--", linewidth=1.8)
    ax.plot(steps, y_pers, label="Persistence Baseline", color="#377eb8", linestyle=":", linewidth=1.5)
    ax.plot(steps, y_ma, label="Moving Average Baseline", color="#4daf4a", linestyle="-.", linewidth=1.5)

    ax.set_title("30-Minute Ahead Resource Demand Forecasting on Held-Out Test Set")
    ax.set_xlabel("Chronological Time Steps (5-min intervals)")
    ax.set_ylabel("Normalized CPU Demand")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig("results/figures/fig3_lstm_prediction.png")
    fig.savefig("paper/figures/fig3_lstm_prediction.png")
    plt.close()


def plot_clustering_evaluation():
    print("Generating Figure: K-Means++ Clustering Metrics...")
    df_eval = pd.read_csv("results/tables/table6_kmeanspp_evaluation.csv")

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = "#1f77b4"
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Silhouette Score (Higher is better)", color=color)
    line1 = ax1.plot(df_eval["K"], df_eval["silhouette_score"], marker="o", color=color, linewidth=2, label="Silhouette Score")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#d62728"
    ax2.set_ylabel("Davies-Bouldin Index (Lower is better)", color=color)
    line2 = ax2.plot(df_eval["K"], df_eval["davies_bouldin_index"], marker="s", color=color, linewidth=2, linestyle="--", label="Davies-Bouldin Index")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    plt.title("Cluster Validity Metrics Across K in {3, 4, 5, 6}")

    plt.tight_layout()
    fig.savefig("results/figures/fig4_clustering_evaluation.png")
    fig.savefig("paper/figures/fig4_clustering_evaluation.png")
    plt.close()


def plot_scheduler_comparison():
    print("Generating Figure: Energy and Utilization Comparison...")
    df_sched = pd.read_csv("results/tables/table10_baseline_comparison.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Shorten names for clean plotting
    names = [name.replace(" (Simulated)", "").replace(" (LSTM + K-Means++ + ACO)", "") for name in df_sched["Algorithm"]]
    colors = ["#999999", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#e41a1c"]

    # Energy comparison bar chart
    bars1 = ax1.bar(names, df_sched["total_energy_wph"], color=colors, edgecolor="black", width=0.6)
    ax1.set_title("Total Energy Consumption (TEC, Watt-hours)")
    ax1.set_ylabel("Watt-hours (Wph)")
    ax1.set_xticklabels(names, rotation=35, ha="right")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f"{int(yval):,}", ha="center", va="bottom", fontsize=8.5)

    # Active PMs comparison bar chart
    bars2 = ax2.bar(names, df_sched["active_pms"], color=colors, edgecolor="black", width=0.6)
    ax2.set_title("Number of Active Physical Machines (Out of 798 PMs)")
    ax2.set_ylabel("Active PMs Count")
    ax2.set_xticklabels(names, rotation=35, ha="right")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig("results/figures/fig6_scheduler_comparison.png")
    fig.savefig("paper/figures/fig6_scheduler_comparison.png")
    plt.close()


def plot_ablation_study():
    print("Generating Figure: Ablation Study Comparison...")
    df_abl = pd.read_csv("results/tables/table13_ablation_study.csv")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    variants = df_abl["Ablation_Variant"]
    energies = df_abl["total_energy_wph"]

    y_pos = np.arange(len(variants))
    colors = ["#2ca02c" if "Full" in v else "#1f77b4" for v in variants]

    ax.barh(y_pos, energies, color=colors, edgecolor="black", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    ax.set_xlabel("Total Energy Consumption (Watt-hours)")
    ax.set_title("Ablation Study: Contribution of Core Framework Components")

    for i, v in enumerate(energies):
        ax.text(v + 50, i, f"{v:.1f} Wh", va="center", fontsize=9.5)

    plt.tight_layout()
    fig.savefig("results/figures/fig8_ablation_study.png")
    fig.savefig("paper/figures/fig8_ablation_study.png")
    plt.close()


def main():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)
    plot_workload_distribution()
    plot_lstm_prediction()
    plot_clustering_evaluation()
    plot_scheduler_comparison()
    plot_ablation_study()
    print("All research figures successfully generated and saved.")


if __name__ == "__main__":
    main()

