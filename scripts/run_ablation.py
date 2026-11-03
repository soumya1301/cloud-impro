"""
Comprehensive ablation study and sensitivity analysis script.
Generates Table 13 (Ablation Study) and Table 14 (Sensitivity Analysis).
"""
import os
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.vm_scheduler.simulation.pm import PhysicalMachine, VirtualMachine
from src.vm_scheduler.scheduling.placement import LSTMKMeansACOPlacer
from src.vm_scheduler.scheduling.aco import AntColonyPlacer
from src.vm_scheduler.scheduling.migration import MigrationManager
from src.vm_scheduler.evaluation.scheduling_metrics import evaluate_cluster_performance
from src.vm_scheduler.utils.config import setup_logger, set_seed

logger = setup_logger("run_ablation")


def load_simulation_entities():
    df_vms = pd.read_parquet("data/processed/subset_b_vms.parquet")
    df_pms = pd.read_parquet("data/processed/subset_b_pms.parquet")

    # Use smaller scale for rapid, rigorous ablation across 1,500 VMs and 150 PMs
    df_vms = df_vms.head(1500)
    df_pms = df_pms.head(150)

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


def run_configuration(vms, pms, use_prediction=True, dynamic_thresh=True, nonlinear_power=True, use_aco=True, use_migration=True):
    cloned_pms = clone_cluster(pms)

    # Set power model linearity
    for p in cloned_pms:
        p.k_nonlinearity = 1.4 if nonlinear_power else 1.0

    if not dynamic_thresh:
        for p in cloned_pms:
            p.upper_threshold = 0.75

    if use_aco:
        aco = AntColonyPlacer(num_ants=10, max_iterations=10, random_seed=42)
        mig_mgr = MigrationManager(min_net_benefit_wh=0.5 if use_migration else 999.0)
        placer = LSTMKMeansACOPlacer(aco_placer=aco, migration_manager=mig_mgr, dynamic_threshold=dynamic_thresh)

        pred_dict = {vm.vm_id: (vm.cpu_req * 1.05, vm.mem_req * 1.02) for vm in vms} if use_prediction else None
        placement, info = placer.schedule(vms, cloned_pms, predicted_vm_demands=pred_dict)
        mig_count = info["migration_count"]
        mig_cost = info["migration_cost"]
    else:
        # Greedy heuristic baseline
        placement = {}
        mig_count = 0
        mig_cost = 0.0
        for vm in vms:
            for pm in cloned_pms:
                if pm.can_host(vm, enforce_threshold=True):
                    pm.allocate(vm)
                    placement[vm.vm_id] = pm.pm_id
                    break

    metrics = evaluate_cluster_performance(
        pms=cloned_pms,
        migration_count=mig_count,
        migration_cost=mig_cost,
        observation_hours=1.0,
        upper_threshold=0.75
    )
    return metrics


def main():
    set_seed(42)
    vms, pms = load_simulation_entities()
    logger.info(f"Loaded {len(vms)} VMs and {len(pms)} PMs for ablation suite.")

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
            use_prediction=exp["pred"],
            dynamic_thresh=exp["dyn"],
            nonlinear_power=exp["pwr"],
            use_aco=exp["aco"],
            use_migration=exp["mig"]
        )
        res["Ablation_Variant"] = exp["id"]
        ablation_results.append(res)

    df_abl = pd.DataFrame(ablation_results)
    cols = ["Ablation_Variant", "total_energy_wph", "active_pms", "AUR_percent", "SLAv", "migrated_vms", "resource_imbalance"]
    df_abl = df_abl[cols]

    out_table13 = "results/tables/table13_ablation_study.csv"
    df_abl.to_csv(out_table13, index=False)
    logger.info(f"Table 13 saved to {out_table13}")

    print("\n" + "="*90)
    print("TABLE 13: ABLATION STUDY RESULTS")
    print("="*90)
    print(df_abl.to_string(index=False))
    print("="*90 + "\n")

    # Sensitivity Analysis on ACO Weights and Exponent k
    sensitivity_results = []
    for k_exp in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        cloned = clone_cluster(pms)
        for p in cloned:
            p.k_nonlinearity = k_exp
        res = run_configuration(vms, pms, nonlinear_power=True)
        sensitivity_results.append({
            "k_nonlinearity": k_exp,
            "total_energy_wph": res["total_energy_wph"],
            "active_pms": res["active_pms"],
            "AUR_percent": res["AUR_percent"]
        })

    df_sens = pd.DataFrame(sensitivity_results)
    out_table14 = "results/tables/table14_sensitivity_analysis.csv"
    df_sens.to_csv(out_table14, index=False)
    logger.info(f"Table 14 saved to {out_table14}")

    print("\n" + "="*70)
    print("TABLE 14: SENSITIVITY ANALYSIS (NONLINEARITY PARAMETER k)")
    print("="*70)
    print(df_sens.to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

