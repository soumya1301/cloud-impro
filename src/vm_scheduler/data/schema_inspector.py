"""
Schema inspector and resource dimension availability validator.
"""
from pathlib import Path
from typing import Dict, Any, List
import pyarrow.parquet as pq
import pandas as pd


class SchemaInspector:
    """Inspects Parquet schemas, metadata, and column types."""

    @staticmethod
    def inspect_file(file_path: str) -> Dict[str, Any]:
        """Reads Parquet metadata, schema, and row counts."""
        pf = pq.ParquetFile(file_path)
        schema = pf.schema_arrow
        fields = {f.name: str(f.type) for f in schema}
        return {
            "num_rows": pf.metadata.num_rows,
            "num_row_groups": pf.metadata.num_row_groups,
            "fields": fields
        }

    @staticmethod
    def verify_resource_dimensions(fields: Dict[str, str]) -> Dict[str, bool]:
        """Verifies the presence of CPU, RAM, Network, and Disk dimensions."""
        combined = " ".join([f"{k} {v}".lower() for k, v in fields.items()])
        has_cpu = "cpu" in combined
        has_ram = "memory" in combined or "mem" in combined
        has_net = "network" in combined or "net" in combined
        has_disk = "disk" in combined or "disk_io" in combined or "read_bytes" in combined or "write_bytes" in combined
        return {
            "CPU": has_cpu,
            "RAM": has_ram,
            "Network": has_net,
            "Disk_IO": has_disk
        }

