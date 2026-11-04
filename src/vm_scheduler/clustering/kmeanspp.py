"""
K-Means++ clustering and multi-K evaluation module.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class KMeansPlusPlusEvaluator:
    """Evaluates K-Means++ across candidate cluster counts K in {3, 4, 5, 6}."""

    def __init__(self, k_values: List[int] = None, random_state: int = 42, n_init: int = 20):
        self.k_values = k_values or [3, 4, 5, 6]
        self.random_state = random_state
        self.n_init = n_init
        self.models: Dict[int, KMeans] = {}
        self.results: Dict[int, Dict[str, Any]] = {}

    def evaluate(self, X: np.ndarray) -> pd.DataFrame:
        """Fits KMeans for each K and calculates evaluation metrics."""
        summary = []
        for k in self.k_values:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=self.n_init,
                max_iter=300,
                random_state=self.random_state
            )
            labels = km.fit_predict(X)
            self.models[k] = km

            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)

            # Cluster size distribution
            unique, counts = np.unique(labels, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(unique, counts)}

            res = {
                "K": k,
                "silhouette_score": round(float(sil), 4),
                "davies_bouldin_index": round(float(db), 4),
                "calinski_harabasz_score": round(float(ch), 2),
                "inertia": round(float(km.inertia_), 2),
                "cluster_sizes": dist
            }
            self.results[k] = res
            summary.append(res)

        return pd.DataFrame(summary)

    def select_best_k(self, metric: str = "silhouette_score") -> int:
        """Selects the best K based on the specified metric."""
        if "silhouette" in metric:
            key = "silhouette_score"
            return max(self.results.keys(), key=lambda k: self.results[k][key])
        elif "davies" in metric:
            key = "davies_bouldin_index"
            return min(self.results.keys(), key=lambda k: self.results[k][key])
        elif "calinski" in metric:
            key = "calinski_harabasz_score"
            return max(self.results.keys(), key=lambda k: self.results[k][key])
        return max(self.results.keys(), key=lambda k: self.results[k].get(metric, 0))

