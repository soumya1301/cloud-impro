"""
K-Means++ clustering and evaluation across K in {3, 4, 5, 6}.
Generates Table 6 (Cluster Evaluation) and Table 7 (Cluster Profiles).
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.vm_scheduler.clustering.kmeanspp import KMeansPlusPlusEvaluator
from src.vm_scheduler.clustering.profiling import WorkloadProfiler
from src.vm_scheduler.utils.config import setup_logger, set_seed, load_yaml_config

os.makedirs("results/logs", exist_ok=True)
os.makedirs("experiments/clustering", exist_ok=True)
os.makedirs("paper/tables", exist_ok=True)
logger = setup_logger("run_clustering", log_file="results/logs/clustering.log")


def main():
    set_seed(42)
    cfg = load_yaml_config("configs/clustering.yaml")
    features_path = "data/processed/subset_a_vm_features.parquet"

    logger.info("Loading VM features from %s", features_path)
    df = pd.read_parquet(features_path)

    feature_cols = cfg["features_for_clustering"]
    logger.info("Clustering features: %s", feature_cols)

    # Impute any remaining NaNs with column medians
    df_imputed = df[feature_cols].copy()
    for col in feature_cols:
        med = df_imputed[col].median()
        if pd.isna(med):
            med = 0.0
        df_imputed[col] = df_imputed[col].fillna(med)

    X = df_imputed.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    evaluator = KMeansPlusPlusEvaluator(
        k_values=cfg["evaluation_k"],
        random_state=cfg["random_state"],
        n_init=cfg["n_init"]
    )

    logger.info("Evaluating K-Means++ across K in [3, 4, 5, 6]...")
    eval_df = evaluator.evaluate(X_scaled)

    os.makedirs("results/tables", exist_ok=True)
    out_table6 = "results/tables/table6_kmeanspp_evaluation.csv"
    eval_df.to_csv(out_table6, index=False)
    eval_df.to_csv("paper/tables/table6_kmeanspp_evaluation.csv", index=False)
    logger.info("Table 6 saved to %s", out_table6)

    print("\n" + "="*80)
    print("TABLE 6: K-MEANS++ CLUSTER EVALUATION ACROSS K in {3, 4, 5, 6}")
    print("="*80)
    print(eval_df[["K", "silhouette_score", "davies_bouldin_index", "calinski_harabasz_score", "inertia"]].to_string(index=False))
    print("="*80 + "\n")

    best_k = evaluator.select_best_k(metric=cfg["best_k_criteria"])
    logger.info(f"Selected Best K based on {cfg['best_k_criteria']}: K={best_k}")

    # Generate Profile Table for both K=3 (Alelyani baseline comparison) and Selected Best K
    for k in [3, best_k]:
        model = evaluator.models[k]
        labels = model.labels_
        df[f"cluster_k{k}"] = labels

        # Unscale centroids back to feature scale
        raw_centroids = scaler.inverse_transform(model.cluster_centers_)
        profiles = WorkloadProfiler.profile_clusters(raw_centroids, feature_cols)

        profile_rows = []
        for c_id, prof in profiles.items():
            count = int(np.sum(labels == c_id))
            pct = round(count / len(labels) * 100.0, 2)
            profile_rows.append({
                "K": k,
                "cluster_id": c_id,
                "label": prof["label"],
                "vm_count": count,
                "percentage": pct,
                "cpu_centroid": prof["cpu_centroid"],
                "mem_centroid": prof["mem_centroid"]
            })

        prof_df = pd.DataFrame(profile_rows)
        out_table7 = f"results/tables/table7_cluster_profiles_k{k}.csv"
        prof_df.to_csv(out_table7, index=False)
        prof_df.to_csv(f"paper/tables/table7_cluster_profiles_k{k}.csv", index=False)
        logger.info("Table 7 (K=%d) saved to %s", k, out_table7)

        print(f"\n--- Cluster Workload Characteristics (K={k}) ---")
        print(prof_df.to_string(index=False))

    # Save VM feature dataset augmented with cluster labels
    out_labeled = "data/processed/subset_a_vm_features_labeled.parquet"
    df.to_parquet(out_labeled, index=False)
    logger.info("Labeled VM dataset saved to %s", out_labeled)

    # Save experiment run metadata
    with open("experiments/clustering/clustering_run_meta.json", "w") as fp:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "k_values": cfg["evaluation_k"],
            "best_k": best_k,
            "metrics": eval_df.to_dict(orient="records")
        }, fp, indent=2)


if __name__ == "__main__":
    main()
