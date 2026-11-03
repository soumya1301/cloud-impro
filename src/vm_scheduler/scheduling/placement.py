"""
Proposed Framework: LSTM-Assisted, K-Means++-Driven and ACO-Based Energy-Aware Scheduler.
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import copy
from ..simulation.pm import PhysicalMachine, VirtualMachine
from .aco import AntColonyPlacer
from .migration import MigrationManager


class LSTMKMeansACOPlacer:
    """
    Integrates:
    - Step 1: Workload Profiling via K-Means++ labels
    - Step 2: Predictive Dynamic PM Threshold adaptation
    - Step 3: Ant Colony Optimization (ACO) global placement
    - Step 4: Net-Benefit-Gated Consolidation & Migration
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
        # Step 1: Adapt dynamic thresholds on PMs using predictions
        if self.dynamic_threshold and predicted_vm_demands:
            for pm in pms:
                # Estimate future load based on assigned VM predictions
                pred_cpu = sum(
                    predicted_vm_demands.get(vm.vm_id, (vm.cpu_req, vm.mem_req))[0]
                    for vm in pm.hosted_vms.values()
                )
                pm.update_dynamic_threshold(pred_cpu / max(pm.cpu_capacity, 1e-6))

        # Step 2: ACO Placement Optimization
        best_placement, fitness, conv_curve = self.aco_placer.optimize_placement(
            vms=vms,
            pms=pms,
            enforce_threshold=True
        )

        # Apply placements to actual PMs
        pm_dict = {p.pm_id: p for p in pms}
        for vm in vms:
            if vm.vm_id in best_placement:
                target_pm_id = best_placement[vm.vm_id]
                pm_dict[target_pm_id].allocate(vm)

        # Step 3: Post-placement Consolidation and Selective Live Migration
        active_pms = [p for p in pms if p.is_active]
        underloaded = [p for p in active_pms if p.cpu_utilization_ratio < 0.35]
        candidates = [p for p in active_pms if 0.35 <= p.cpu_utilization_ratio <= 0.70]

        migration_count = 0
        total_mig_cost = 0.0

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
                        total_mig_cost += details["migration_time_seconds"] / 3600.0 * 0.001
                        break

        info = {
            "best_fitness": round(fitness, 5),
            "convergence_curve": conv_curve,
            "migration_count": migration_count,
            "migration_cost": round(total_mig_cost, 6)
        }
        return best_placement, info

