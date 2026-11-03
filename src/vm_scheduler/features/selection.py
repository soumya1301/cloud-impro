"""
Feature selection via variance thresholding and correlation filtering.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np


class FeatureSelector:
    """Selects predictive, non-redundant features for VM workload profiling."""

    def __init__(self, variance_threshold: float = 1e-4, correlation_threshold: float = 0.90):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def filter_low_variance(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """Removes features with near-zero variance."""
        variances = df[feature_cols].var()
        return variances[variances >= self.variance_threshold].index.tolist()

    def filter_collinear(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
        """Removes highly correlated redundant features based on pairwise Pearson correlation."""
        corr_matrix = df[feature_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.correlation_threshold)]
        selected = [col for col in feature_cols if col not in to_drop]
        return selected, corr_matrix

