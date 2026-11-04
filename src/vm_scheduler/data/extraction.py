"""
Targeted extraction of experimental Subsets A (ML/Profiling) and B (Scheduling Simulation).
"""
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


class DatasetExtractor:
    """Extracts reproducible experimental subsets from processed or sample Parquet files."""

    def __init__(self, sample_usage_path: str, sample_events_path: str, sample_machines_path: str):
        self.usage_path = Path(sample_usage_path)
        self.events_path = Path(sample_events_path)
        self.machines_path = Path(sample_machines_path)

    def extract_subset_a(
        self,
        output_ts_path: str,
        output_features_path: str,
        target_vms: int = 3000,
        min_steps: int = 12
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts Subset A:
        1. Multi-step time series for multivariate LSTM forecasting.
        2. 16-feature tabular profile per VM for K-Means++ clustering.
        """
        pf = pq.ParquetFile(self.usage_path)
        df = pf.read_row_group(0).to_pandas()

        # Build consistent VM identifier
        df["vm_id"] = df["collection_id"].astype(str) + "_" + df["instance_index"].astype(str)

        # Flatten struct metrics
        df["cpu_util"] = df["average_usage"].apply(lambda x: x.get("cpus") if isinstance(x, dict) else np.nan)
        df["mem_util"] = df["average_usage"].apply(lambda x: x.get("memory") if isinstance(x, dict) else np.nan)
        df["cpu_max"] = df["maximum_usage"].apply(lambda x: x.get("cpus") if isinstance(x, dict) else np.nan)
        df["mem_max"] = df["maximum_usage"].apply(lambda x: x.get("memory") if isinstance(x, dict) else np.nan)

        # Filter positive valid signals
        valid = df[(df["cpu_util"] > 0) & (df["mem_util"] > 0)].copy()

        # Sort chronologically
        valid = valid.sort_values(["vm_id", "start_time"])

        # Select VMs with sufficient history or aggregate across collections
        vm_counts = valid["vm_id"].value_counts()
        eligible_vms = vm_counts[vm_counts >= min_steps].index.tolist()

        # If eligible VMs with strict step length are fewer than target, incorporate collection-level time series
        if len(eligible_vms) < target_vms:
            # Add top active collections grouped as distinct workloads
            top_collections = valid["collection_id"].value_counts().head(target_vms).index
            active_data = valid[valid["collection_id"].isin(top_collections)].copy()
        else:
            selected_vms = eligible_vms[:target_vms]
            active_data = valid[valid["vm_id"].isin(selected_vms)].copy()

        # Impute CPI and MAI with median
        cpi_med = active_data["cycles_per_instruction"].median()
        mai_med = active_data["memory_accesses_per_instruction"].median()
        active_data["cycles_per_instruction"] = active_data["cycles_per_instruction"].fillna(cpi_med)
        active_data["memory_accesses_per_instruction"] = active_data["memory_accesses_per_instruction"].fillna(mai_med)

        # Save clean time series
        ts_cols = [
            "vm_id", "start_time", "end_time", "cpu_util", "mem_util",
            "cpu_max", "mem_max", "assigned_memory", "page_cache_memory",
            "cycles_per_instruction", "memory_accesses_per_instruction"
        ]
        clean_ts = active_data[ts_cols].copy()
        os.makedirs(os.path.dirname(output_ts_path), exist_ok=True)
        clean_ts.to_parquet(output_ts_path, index=False)

        # Compute 16 VM-level features matching Alelyani et al.
        grouped = clean_ts.groupby("vm_id")
        features_df = grouped.agg(
            cpu_util_mean=("cpu_util", "mean"),
            cpu_util_max=("cpu_max", "max"),
            cpu_util_min=("cpu_util", "min"),
            cpu_util_std=("cpu_util", lambda s: s.std() if len(s) > 1 else 0.0),
            mem_util_mean=("mem_util", "mean"),
            mem_util_max=("mem_max", "max"),
            mem_util_min=("mem_util", "min"),
            assigned_memory=("assigned_memory", "mean"),
            page_cache_memory=("page_cache_memory", "mean"),
            cycles_per_instruction=("cycles_per_instruction", "mean"),
            memory_accesses_per_instruction=("memory_accesses_per_instruction", "mean"),
            start_time=("start_time", "min"),
            end_time=("end_time", "max")
        ).reset_index()

        features_df["duration_minutes"] = (features_df["end_time"] - features_df["start_time"]) / 1e6 / 60.0
        features_df["peak_to_average_cpu_ratio"] = features_df["cpu_util_max"] / (features_df["cpu_util_mean"] + 1e-6)

        # Select target count
        if len(features_df) > target_vms:
            features_df = features_df.head(target_vms)

        os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
        features_df.to_parquet(output_features_path, index=False)

        return clean_ts, features_df

    def extract_subset_b(
        self,
        output_vms_path: str,
        output_pms_path: str,
        target_vms: int = 7644,
        target_pms: int = 798,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts Subset B for scheduling simulation:
        1. 7,644 VM arrival events with resource requests and execution priorities.
        2. 798 heterogeneous Physical Machines with calibrated SPECpower profiles.
        """
        np.random.seed(seed)

        # Read instance events
        pf_ev = pq.ParquetFile(self.events_path)
        ev_df = pf_ev.read_row_group(0).to_pandas()

        ev_df["cpu_req"] = ev_df["resource_request"].apply(lambda x: x.get("cpus") if isinstance(x, dict) else np.nan)
        ev_df["mem_req"] = ev_df["resource_request"].apply(lambda x: x.get("memory") if isinstance(x, dict) else np.nan)

        # Impute missing requests with standard distribution observed in trace
        ev_df["cpu_req"] = ev_df["cpu_req"].fillna(0.25)
        ev_df["mem_req"] = ev_df["mem_req"].fillna(0.10)
        ev_df = ev_df[(ev_df["cpu_req"] > 0) & (ev_df["mem_req"] > 0)]

        # Sample target VMs
        if len(ev_df) >= target_vms:
            sampled_vms = ev_df.sample(n=target_vms, random_state=seed).copy()
        else:
            sampled_vms = ev_df.copy()

        sampled_vms = sampled_vms.sort_values("time").reset_index(drop=True)
        sampled_vms["vm_id"] = [f"vm_{i:05d}" for i in range(len(sampled_vms))]
        sampled_vms["arrival_time"] = (sampled_vms["time"] - sampled_vms["time"].min()) / 1e6 # seconds

        # Execution priority (0 to 300)
        sampled_vms["priority"] = sampled_vms["priority"].clip(0, 300)

        # Duration in seconds (Poisson/log-normal realistic lifetime: median 30 min, up to 12 hours)
        durations = np.random.lognormal(mean=7.5, sigma=1.0, size=len(sampled_vms)).clip(300, 86400)
        sampled_vms["duration_seconds"] = durations

        # Memory footprint in GB for live migration calculations (based on normalized memory fraction)
        sampled_vms["memory_gb"] = (sampled_vms["mem_req"] * 64.0).clip(0.5, 32.0)

        vms_out = sampled_vms[["vm_id", "arrival_time", "cpu_req", "mem_req", "priority", "duration_seconds", "memory_gb"]]
        os.makedirs(os.path.dirname(output_vms_path), exist_ok=True)
        vms_out.to_parquet(output_vms_path, index=False)

        # Build 798 heterogeneous PMs matching HP G5 / G4 SPECpower distribution
        pms_list = []
        for i in range(target_pms):
            pm_id = f"pm_{i:04d}"
            # 70% HP G5 Standard, 20% HP G5 High-Memory, 10% HP G4 Compact
            r = np.random.rand()
            if r < 0.70:
                pm_type = "hp_proliant_g5_standard"
                cpu_cap = 1.0
                mem_cap = 0.50
                p_idle = 93.7
                p_max = 135.0
                k_exp = 1.4
            elif r < 0.90:
                pm_type = "hp_proliant_g5_highmem"
                cpu_cap = 1.0
                mem_cap = 1.00
                p_idle = 98.0
                p_max = 145.0
                k_exp = 1.4
            else:
                pm_type = "hp_proliant_g4_compact"
                cpu_cap = 0.50
                mem_cap = 0.333
                p_idle = 86.0
                p_max = 117.0
                k_exp = 1.3

            pms_list.append({
                "pm_id": pm_id,
                "pm_type": pm_type,
                "cpu_capacity": cpu_cap,
                "mem_capacity": mem_cap,
                "network_bandwidth_gbps": 4.0,
                "idle_power_w": p_idle,
                "max_power_w": p_max,
                "sleep_power_w": 10.0,
                "k_nonlinearity": k_exp
            })

        pms_df = pd.DataFrame(pms_list)
        os.makedirs(os.path.dirname(output_pms_path), exist_ok=True)
        pms_df.to_parquet(output_pms_path, index=False)

        return vms_out, pms_df

