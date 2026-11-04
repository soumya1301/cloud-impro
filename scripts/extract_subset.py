"""
Script to extract experimental Subsets A (ML/Profiling) and B (Scheduling Simulation).
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vm_scheduler.data.extraction import DatasetExtractor
from src.vm_scheduler.utils.config import setup_logger, load_yaml_config

logger = setup_logger("extract_subset")


def main():
    logger.info("Starting targeted extraction of Subsets A and B...")
    cfg = load_yaml_config("configs/data.yaml")

    sample_usage = "/Users/soumyaprasad/.gemini/antigravity/brain/65366d55-4e8d-430e-bc2e-6e0692ea8aa2/scratch/schema_samples/instance_usage_sample.parquet"
    sample_events = "/Users/soumyaprasad/.gemini/antigravity/brain/65366d55-4e8d-430e-bc2e-6e0692ea8aa2/scratch/schema_samples/instance_events_sample.parquet"
    sample_machines = "/Users/soumyaprasad/.gemini/antigravity/brain/65366d55-4e8d-430e-bc2e-6e0692ea8aa2/scratch/schema_samples/machine_events_sample.parquet"

    extractor = DatasetExtractor(
        sample_usage_path=sample_usage,
        sample_events_path=sample_events,
        sample_machines_path=sample_machines
    )

    # 1. Extract Subset A: Time series and 16 VM features
    out_ts = "data/processed/subset_a_prediction_timeseries.parquet"
    out_features = "data/processed/subset_a_vm_features.parquet"
    logger.info("Extracting Subset A (Prediction & Workload Profiling)...")
    ts_df, feat_df = extractor.extract_subset_a(
        output_ts_path=out_ts,
        output_features_path=out_features,
        target_vms=cfg["subsets"]["subset_a"]["target_vms"],
        min_steps=cfg["subsets"]["subset_a"]["observation_window_minutes"]
    )
    logger.info(f"Subset A TimeSeries saved: {len(ts_df):,} rows -> {out_ts}")
    logger.info(f"Subset A Features saved: {len(feat_df):,} VMs -> {out_features}")

    # 2. Extract Subset B: Scheduling VMs and PMs
    out_vms = "data/processed/subset_b_vms.parquet"
    out_pms = "data/processed/subset_b_pms.parquet"
    logger.info("Extracting Subset B (Scheduling Simulation)...")
    vms_df, pms_df = extractor.extract_subset_b(
        output_vms_path=out_vms,
        output_pms_path=out_pms,
        target_vms=cfg["subsets"]["subset_b"]["target_vms"],
        target_pms=cfg["subsets"]["subset_b"]["target_pms"]
    )
    logger.info(f"Subset B VMs saved: {len(vms_df):,} VMs -> {out_vms}")
    logger.info(f"Subset B PMs saved: {len(pms_df):,} PMs -> {out_pms}")
    logger.info("Targeted extraction successfully completed.")


if __name__ == "__main__":
    main()

