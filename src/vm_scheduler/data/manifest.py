"""
Manifest parser and shard metadata catalog.
"""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class DatasetManifest:
    """Provides structured querying over the Google Cluster Data 2019 manifest."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self.df = pd.read_csv(self.manifest_path)

    def get_shards_by_table(self, table_name: str, format_type: str = "parquet.gz") -> pd.DataFrame:
        """Returns all shards for a specific table and format."""
        return self.df[(self.df["table"] == table_name) & (self.df["format"] == format_type)]

    def get_shards_by_cell(self, cell: str, format_type: str = "parquet.gz") -> pd.DataFrame:
        """Returns all shards for a specific cell."""
        return self.df[(self.df["cell"] == cell) & (self.df["format"] == format_type)]

    def get_table_summary(self) -> pd.DataFrame:
        """Computes summary statistics across tables and formats."""
        return self.df.groupby(["table", "format"]).agg(
            shard_count=("shard", "count"),
            total_size_gib=("size_GiB", "sum")
        ).reset_index()

