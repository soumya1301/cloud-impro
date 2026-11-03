"""
Proposed Framework: LSTM-Assisted, K-Means++-Driven and ACO-Based Energy-Aware Scheduler.
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from ..simulation.pm import PhysicalMachine, VirtualMachine
from .aco import AntColonyPlacer
from .migration import MigrationManager


class LSTMKMeansACOPlacer:
    """
    Integrated Framework:
    - Step 1: Workload Profiling via K-Means++ labels
    - Step 2: Predictive Dynamic PM Threshold adaptation using multivariate LSTM forecasts
    - Step 3: Ant Colony Optimization (ACO) global multi-objective placement
    - Step 4: Net-Benefit-Gated Consolidation & Live Migration
    """

    def __init__(
        self,
        aco_placer: Optional[AntColonyPlacer] = None,
        migration_manager: Optional[MigrationManager] = None,
        dynamic_threshold: bool = True
    ):
        self.aco_placer = aco_placer or AntColonyPlacer()
        self.migration_mgr = migration_manager or MigrationManager()
        self.dynamic_threshold = dynamic_threshold

    def schedule(
        self,
        vms: List[VirtualMachine],
        pms: List[PhysicalMachine],
        predicted_vm_demands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Executes complete scheduling pipeline on `vms` across `pms`.
        """
        # Step 1: Adapt dynamic operating thresholds using genuine predicted demands
        for pm in pms:
            pm.is_dynamic_threshold_enabled = self.dynamic_threshold
            if self.dynamic_threshold and predicted_vm_demands:
                pred_cpu_sum = sum(
                    predicted_vm_demands.get(vm.vm_id, (vm.cpu_req, vm.mem_req))[0]
                    for vm in pm.hosted_vms.values()
                )
                pred_mem_sum = sum(
                    predicted_vm_demands.get(vm.vm_id, (vm.cpu_req, vm.mem_req))[1]
                    for vm in pm.hosted_vms.values()
                )
                pm.update_dynamic_threshold(
                    predicted_cpu_demand=pred_cpu_sum,
                    predicted_mem_demand=pred_mem_sum
                )
            else:
                pm.update_dynamic_threshold(0.0, 0.0)

        # Step 2: Global Placement via ACO
        best_placement, fitness, conv_curve, place_info = self.aco_placer.optimize_placement(
            vms=vms,
            pms=pms,
            enforce_threshold=True
        )

        # Apply assignments to PM states
        pm_dict = {p.pm_id: p for p in pms}
        for vm in vms:
            if vm.vm_id in best_placement:
                target_pm_id = best_placement[vm.vm_id]
                pm_dict[target_pm_id].allocate(vm)

        # Step 3: Net-Benefit-Gated Live Migration for Underloaded Hosts
        active_pms = [p for p in pms if p.is_active]
        underloaded = [p for p in active_pms if p.cpu_utilization_ratio < 0.35]
        candidates = [p for p in active_pms if 0.35 <= p.cpu_utilization_ratio <= 0.70]

        migration_count = 0
        total_mig_time_s = 0.0
        total_mig_energy_wh = 0.0

        for src in underloaded:
            vms_in_src = list(src.hosted_vms.values())
            for vm in vms_in_src:
                for tgt in candidates:
                    should_migrate, details = self.migration_mgr.evaluate_migration(
                        vm=vm, source_pm=src, target_pm=tgt
                    )
                    if should_migrate:
                        src.release(vm.vm_id)
                        tgt.allocate(vm)
                        best_placement[vm.vm_id] = tgt.pm_id
                        migration_count += 1
                        total_mig_time_s += details.get("migration_time_seconds", 0.0)
                        total_mig_energy_wh += details.get("migration_energy_wh", 0.0)
                        break

        info = {
            "best_fitness": round(fitness, 5),
            "convergence_curve": conv_curve,
            "migration_count": migration_count,
            "migration_time_seconds": round(total_mig_time_s, 2),
            "migration_energy_wh": round(total_mig_energy_wh, 6),
            "requested_vms": place_info["requested_vms"],
            "placed_vms": place_info["placed_vms"],
            "unplaced_vms": place_info["unplaced_vms"],
            "placement_rate_percent": place_info["placement_rate_percent"]
        }
        return best_placement, info
