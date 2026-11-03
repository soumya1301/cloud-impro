"""
Comprehensive ablation study and sensitivity analysis script (CORRECTED).
Generates Table 13 (Ablation Study) and Table 14 (Sensitivity Analysis).
Saves logs to results/logs/ and run metadata to experiments/.
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

from src.vm_scheduler.simulation.pm import PhysicalMachine, VirtualMachine
from src.vm_scheduler.scheduling.placement import LSTMKMeansACOPlacer
from src.vm_scheduler.scheduling.aco import AntColonyPlacer
from src.vm_scheduler.scheduling.migration import MigrationManager
from src.vm_scheduler.prediction.inference import LSTMInferencer
from src.vm_scheduler.evaluation.scheduling_metrics import evaluate_cluster_performance
from src.vm_scheduler.utils.config import setup_logger, set_seed

os.makedirs("results/logs", exist_ok=True)
os.makedirs("experiments/ablation", exist_ok=True)
os.makedirs("experiments/sensitivity", exist_ok=True)
logger = setup_logger("run_ablation", log_file="results/logs/ablation.log")


def load_simulation_entities():
    df_vms = pd.read_parquet("data/processed/subset_b_vms.parquet").head(1500)
    df_pms = pd.read_parquet("data/processed/subset_b_pms.parquet").head(150)

    vms = []
    for idx, row in df_vms.iterrows():
        vm = VirtualMachine(
            vm_id=str(row["vm_id"]),
            arrival_time=float(row["arrival_time"]),
            cpu_req=float(row["cpu_req"]),
            mem_req=float(row["mem_req"]),
            priority=int(row["priority"]),
            duration_seconds=float(row["duration_seconds"]),
            memory_gb=float(row["memory_gb"]),
            is_cpu_intensive=(row["cpu_req"] > 0.40)
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

    return vms, pms


def clone_cluster(pms):
    cloned = []
    for p in pms:
        new_p = copy.copy(p)
        new_p.hosted_vms = {}
        new_p.current_cpu_load = 0.0
        new_p.current_mem_load = 0.0
        cloned.append(new_p)
    return cloned


def run_configuration(
    vms,
    pms,
    predictions=None,
    use_prediction=True,
    dynamic_thresh=True,
    nonlinear_power=True,
    use_aco=True,
    use_migration=True,
    custom_k_exp=None
):
    cloned_pms = clone_cluster(pms)

    # Set power model non-linearity
    for p in cloned_pms:
        if custom_k_exp is not None:
            p.k_nonlinearity = custom_k_exp
        else:
            p.k_nonlinearity = 1.4 if nonlinear_power else 1.0
        p.is_dynamic_threshold_enabled = dynamic_thresh
        if not dynamic_thresh:
            p.upper_threshold = 0.75

    if use_aco:
        aco = AntColonyPlacer(num_ants=10, max_iterations=10, random_seed=42)
        # If migration benefit gating disabled, set min_benefit to 0 to permit unconstrained migration
        mig_mgr = MigrationManager(min_net_benefit_wh=0.5 if use_migration else -999.0)
        placer = LSTMKMeansACOPlacer(aco_placer=aco, migration_manager=mig_mgr, dynamic_threshold=dynamic_thresh)

        pred_dict = predictions if use_prediction else None
        placement, info = placer.schedule(vms, cloned_pms, predicted_vm_demands=pred_dict)
        mig_count = info["migration_count"]
        mig_time_s = info["migration_time_seconds"]
        mig_energy_wh = info["migration_energy_wh"]
        placed_count = info["placed_vms"]
    else:
        # Greedy heuristic baseline
        placement = {}
        mig_count = 0
        mig_time_s = 0.0
        mig_energy_wh = 0.0
        for vm in vms:
            for pm in cloned_pms:
                if pm.can_host(vm, enforce_threshold=True):
                    pm.allocate(vm)
                    placement[vm.vm_id] = pm.pm_id
                    break
        placed_count = len(placement)

    metrics = evaluate_cluster_performance(
        pms=cloned_pms,
        migration_count=mig_count,
        migration_time_seconds=mig_time_s,
        migration_energy_wh=mig_energy_wh,
        observation_hours=1.0,
        upper_threshold=0.75,
        requested_vms=len(vms),
        placed_vms=placed_count
    )
    return metrics


def main():
    set_seed(42)
    vms, pms = load_simulation_entities()
    logger.info(f"Loaded {len(vms)} VMs and {len(pms)} PMs for ablation suite.")

    # Generate genuine model predictions for the 1,500 VMs
    inferencer = LSTMInferencer(checkpoint_path="models/checkpoints/lstm_30m.pt")
    df_vms = pd.read_parquet("data/processed/subset_b_vms.parquet").head(1500)
    df_ts = pd.read_parquet("data/processed/subset_a_prediction_timeseries.parquet")
    genuine_preds = inferencer.generate_vm_predictions(df_vms, df_ts)

    ablation_experiments = [
        {"id": "Proposed Full Framework", "pred": True, "dyn": True, "pwr": True, "aco": True, "mig": True},
        {"id": "w/o Prediction (Reactive Only)", "pred": False, "dyn": True, "pwr": True, "aco": True, "mig": True},
        {"id": "w/o Dynamic Thresholds (Static 75%)", "pred": True, "dyn": False, "pwr": True, "aco": True, "mig": True},
        {"id": "w/o ACO (Greedy First Fit)", "pred": True, "dyn": True, "pwr": True, "aco": False, "mig": True},
        {"id": "w/o Migration Benefit Gating", "pred": True, "dyn": True, "pwr": True, "aco": True, "mig": False},
        {"id": "Linear Power Model (Ablation)", "pred": True, "dyn": True, "pwr": False, "aco": True, "mig": True},
    ]

    ablation_results = []
    for exp in ablation_experiments:
        logger.info(f"Running ablation: {exp['id']}...")
        res = run_configuration(
            vms, pms,
            predictions=genuine_preds,
            use_prediction=exp["pred"],
            dynamic_thresh=exp["dyn"],
            nonlinear_power=exp["pwr"],
            use_aco=exp["aco"],
            use_migration=exp["mig"]
        )
        res["Ablation_Variant"] = exp["id"]
        ablation_results.append(res)

    df_abl = pd.DataFrame(ablation_results)
    cols = [
        "Ablation_Variant", "total_energy_wph", "active_pms", "AUR_percent",
        "SLAv", "migrated_vms", "resource_imbalance", "placement_rate_percent"
    ]
    df_abl = df_abl[cols]

    out_table13 = "results/tables/table13_ablation_study.csv"
    df_abl.to_csv(out_table13, index=False)
    df_abl.to_csv("paper/tables/table13_ablation_study.csv", index=False)
    logger.info(f"Table 13 saved to {out_table13}")

    with open("experiments/ablation/ablation_run_meta.json", "w") as fp:
        json.dump({"timestamp": datetime.now().isoformat(), "runs": df_abl.to_dict(orient="records")}, fp, indent=2)

    print("\n" + "="*100)
    print("TABLE 13: ABLATION STUDY RESULTS (CORRECTED)")
    print("="*100)
    print(df_abl.to_string(index=False))
    print("="*100 + "\n")

    # SENSITIVITY ANALYSIS (CORRECTED: Propagate custom_k_exp properly)
    logger.info("Executing corrected sensitivity analysis on power exponent k...")
    sensitivity_results = []
    k_test_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    for k_val in k_test_values:
        # Pass custom_k_exp explicitly
        res = run_configuration(
            vms, pms,
            predictions=genuine_preds,
            nonlinear_power=True,
            custom_k_exp=k_val
        )
        sensitivity_results.append({
            "k_nonlinearity": k_val,
            "total_energy_wph": res["total_energy_wph"],
            "active_pms": res["active_pms"],
            "AUR_percent": res["AUR_percent"],
            "resource_imbalance": res["resource_imbalance"]
        })
        logger.info(f"Sensitivity k={k_val}: Energy={res['total_energy_wph']} Wh")

    df_sens = pd.DataFrame(sensitivity_results)
    out_table14 = "results/tables/table14_sensitivity_analysis.csv"
    df_sens.to_csv(out_table14, index=False)
    df_sens.to_csv("paper/tables/table14_sensitivity_analysis.csv", index=False)
    logger.info(f"Table 14 saved to {out_table14}")

    with open("experiments/sensitivity/sensitivity_run_meta.json", "w") as fp:
        json.dump({"timestamp": datetime.now().isoformat(), "runs": df_sens.to_dict(orient="records")}, fp, indent=2)

    print("\n" + "="*80)
    print("TABLE 14: SENSITIVITY ANALYSIS (NONLINEARITY PARAMETER k - CORRECTED)")
    print("="*80)
    print(df_sens.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
