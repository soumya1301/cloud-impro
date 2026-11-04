"""
Controlled Scheduling Integration Comparison:
Proposed (LSTM + ACO) vs. Proposed (GRU + ACO) vs. Reactive ACO (without forecasting).
Evaluated on the identical 7,644 VMs and 798 PMs.
Generates results/tables/table_scheduling_model_comparison.csv and paper/tables/table_scheduling_model_comparison.csv.
"""
import os
import sys
import copy
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.vm_scheduler.simulation.pm import PhysicalMachine, VirtualMachine
from src.vm_scheduler.scheduling.placement import LSTMKMeansACOPlacer
from src.vm_scheduler.scheduling.aco import AntColonyPlacer
from src.vm_scheduler.scheduling.migration import MigrationManager
from src.vm_scheduler.prediction.inference import LSTMInferencer
from src.vm_scheduler.prediction.gru_inference import GRUInferencer
from src.vm_scheduler.evaluation.scheduling_metrics import evaluate_cluster_performance
from src.vm_scheduler.utils.config import setup_logger, set_seed

os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("paper/tables", exist_ok=True)
os.makedirs("results/placements", exist_ok=True)
logger = setup_logger("scheduling_gru_vs_lstm", log_file="results/logs/scheduling_gru_vs_lstm.log")


def load_simulation_entities(vms_path: str, pms_path: str, kmeans_bundle_path: str):
    df_vms = pd.read_parquet(vms_path)
    df_pms = pd.read_parquet(pms_path)

    if os.path.exists(kmeans_bundle_path):
        bundle = joblib.load(kmeans_bundle_path)
        k = bundle["best_k"]
        profiles = bundle["cluster_profiles"].get(k, {})
    else:
        profiles = {}

    vms = []
    cpu_int_classes = {1, 2}

    for idx, row in df_vms.iterrows():
        c_req = float(row["cpu_req"])
        m_req = float(row["mem_req"])

        if profiles:
            best_cid = 0
            min_dist = float("inf")
            for cid, prof in profiles.items():
                dist = (c_req - prof["cpu_centroid"]) ** 2 + (m_req - prof["mem_centroid"]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_cid = cid
            c_id = best_cid
        else:
            c_id = 0

        is_cpu = (c_req > 0.35) or (c_id in cpu_int_classes)

        vm = VirtualMachine(
            vm_id=str(row["vm_id"]),
            arrival_time=float(row["arrival_time"]),
            cpu_req=c_req,
            mem_req=m_req,
            priority=int(row["priority"]),
            duration_seconds=float(row["duration_seconds"]),
            memory_gb=float(row["memory_gb"]),
            profile_class=c_id,
            is_cpu_intensive=is_cpu
        )
        vms.append(vm)

    pms = []
    for idx, row in df_pms.iterrows():
        pm = PhysicalMachine(
            pm_id=str(row["pm_id"]),
            pm_type=str(row["pm_type"]),
            cpu_capacity=float(row["cpu_capacity"]),
            mem_capacity=float(row["mem_capacity"]),
            network_bandwidth_gbps=float(row["network_bandwidth_gbps"]),
            idle_power_w=float(row["idle_power_w"]),
            max_power_w=float(row["max_power_w"]),
            sleep_power_w=float(row["sleep_power_w"]),
            k_nonlinearity=float(row["k_nonlinearity"])
        )
        pms.append(pm)

    return vms, pms, df_vms


def clone_cluster(pms):
    cloned = []
    for p in pms:
        new_p = copy.copy(p)
        new_p.hosted_vms = {}
        new_p.current_cpu_load = 0.0
        new_p.current_mem_load = 0.0
        new_p.predicted_cpu_load = 0.0
        new_p.predicted_mem_load = 0.0
        new_p.upper_threshold = new_p.base_upper_threshold
        cloned.append(new_p)
    return cloned


def main():
    set_seed(42)
    vms_path = "data/processed/subset_b_vms.parquet"
    pms_path = "data/processed/subset_b_pms.parquet"
    kmeans_bundle = "models/checkpoints/kmeanspp_bundle.joblib"
    ts_path = "data/processed/subset_a_prediction_timeseries.parquet"

    logger.info("Loading simulation cluster entities (7,644 VMs and 798 PMs)...")
    base_vms, base_pms, df_vms = load_simulation_entities(vms_path, pms_path, kmeans_bundle)
    df_ts = pd.read_parquet(ts_path)

    # 1. Generate LSTM predictions (30m horizon)
    logger.info("Generating LSTM predictions (30m horizon)...")
    lstm_infer = LSTMInferencer(checkpoint_path="models/checkpoints/lstm_30m.pt")
    lstm_preds, lstm_provenance = lstm_infer.generate_vm_predictions(df_vms, df_ts)

    # 2. Generate GRU predictions (30m horizon)
    logger.info("Generating GRU predictions (30m horizon)...")
    gru_infer = GRUInferencer(checkpoint_path="models/checkpoints/gru_30m.pt")
    gru_preds, gru_provenance = gru_infer.generate_vm_predictions(df_vms, df_ts)

    # Configurations to evaluate
    configurations = [
        {
            "name": "Proposed (LSTM + K-Means++ + ACO)",
            "predictions": lstm_preds,
            "use_pred": True,
            "dyn_thresh": True,
        },
        {
            "name": "Proposed (GRU + K-Means++ + ACO)",
            "predictions": gru_preds,
            "use_pred": True,
            "dyn_thresh": True,
        },
        {
            "name": "Reactive ACO (without forecasting)",
            "predictions": None,
            "use_pred": False,
            "dyn_thresh": True,
        },
    ]

    results = []

    for cfg in configurations:
        name = cfg["name"]
        logger.info(f"--- Running Scheduling Configuration: {name} ---")
        pms = clone_cluster(base_pms)
        vms = copy.deepcopy(base_vms)

        aco = AntColonyPlacer(num_ants=10, max_iterations=10, random_seed=42)
        mig_mgr = MigrationManager(network_bandwidth_gbps=4.0, min_net_benefit_wh=0.5)
        placer = LSTMKMeansACOPlacer(aco_placer=aco, migration_manager=mig_mgr, dynamic_threshold=cfg["dyn_thresh"])

        pred_dict = cfg["predictions"] if cfg["use_pred"] else None
        placement, info = placer.schedule(vms, pms, predicted_vm_demands=pred_dict)

        mig_count = info["migration_count"]
        mig_time_s = info["migration_time_seconds"]
        mig_energy_wh = info["migration_energy_wh"]
        placed_vms = info["placed_vms"]

        metrics = evaluate_cluster_performance(
            pms=pms,
            migration_count=mig_count,
            migration_time_seconds=mig_time_s,
            migration_energy_wh=mig_energy_wh,
            observation_hours=1.0,
            upper_threshold=0.75,
            requested_vms=len(vms),
            placed_vms=placed_vms
        )

        metrics["Configuration"] = name
        metrics["sleeping_pms"] = metrics.get("idle_pms", 0)  # Sleeping/turned-off PMs in power-saving mode
        results.append(metrics)

        # Save placement mapping
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("__", "_").strip("_")
        with open(f"results/placements/placement_{slug}.json", "w") as fp:
            json.dump(placement, fp, indent=2)

        logger.info(
            f"[{name}] Total Energy: {metrics['total_energy_wph']:.2f} Wh | "
            f"Active PMs: {metrics['active_pms']} | Sleeping PMs: {metrics['idle_pms']} | "
            f"Resource Imbalance: {metrics['resource_imbalance']:.4f} | "
            f"SLAv: {metrics['SLAv']:.6f} | Migrations: {mig_count} | "
            f"Placed: {placed_vms}/{len(vms)} ({metrics['placement_rate_percent']}%)"
        )

    df_comp = pd.DataFrame(results)

    cols = [
        "Configuration",
        "total_energy_wph",
        "active_pms",
        "idle_pms",
        "resource_imbalance",
        "SLAv",
        "migrated_vms",
        "migration_energy_wh",
        "migration_time_seconds",
        "placement_rate_percent",
        "AUR_percent",
    ]
    df_comp = df_comp[cols]
    df_comp.rename(columns={"idle_pms": "sleeping_pms", "total_energy_wph": "total_energy_wh"}, inplace=True)

    out_csv = "results/tables/table_scheduling_model_comparison.csv"
    paper_csv = "paper/tables/table_scheduling_model_comparison.csv"
    df_comp.to_csv(out_csv, index=False)
    df_comp.to_csv(paper_csv, index=False)
    logger.info(f"Saved comparison to {out_csv} and {paper_csv}")

    print("\n" + "=" * 110)
    print("SCHEDULING INTEGRATION COMPARISON: LSTM vs. GRU vs. REACTIVE ACO")
    print("=" * 110)
    print(df_comp.to_string(index=False))
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
