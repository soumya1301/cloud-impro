"""
Workload profiling and cluster interpretability engine.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class WorkloadProfiler:
    """Interprets cluster centroids and assigns domain-grounded workload categories."""

    @staticmethod
    def profile_clusters(centroids: np.ndarray, feature_names: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Analyzes centroid coordinates along CPU and Memory axes and assigns profiles:
        - CPU-Intensive
        - Memory-Intensive
        - Balanced
        - Low Utilization / Light
        """
        df_cent = pd.DataFrame(centroids, columns=feature_names)
        profiles = {}

        # Locate CPU mean and Mem mean indices
        cpu_col = [c for c in feature_names if "cpu" in c and "mean" in c][0]
        mem_col = [c for c in feature_names if "mem" in c and "mean" in c][0]

        cpu_med = df_cent[cpu_col].median()
        mem_med = df_cent[mem_col].median()

        for idx, row in df_cent.iterrows():
            cpu_val = row[cpu_col]
            mem_val = row[mem_col]

            if cpu_val > cpu_med * 1.3 and mem_val <= mem_med * 1.3:
                category = "CPU-Intensive"
            elif mem_val > mem_med * 1.3 and cpu_val <= cpu_med * 1.3:
                category = "Memory-Intensive"
            elif cpu_val > cpu_med and mem_val > mem_med:
                category = "Balanced High"
            elif cpu_val < cpu_med * 0.7 and mem_val < mem_med * 0.7:
                category = "Low Utilization"
            else:
                category = "Balanced Moderate"

            profiles[int(idx)] = {
                "label": category,
                "cpu_centroid": round(float(cpu_val), 4),
                "mem_centroid": round(float(mem_val), 4)
            }

        return profiles

