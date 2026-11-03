"""
Comprehensive scheduling evaluation metrics: AUR, Energy, Active PMs, SLAv, Imbalance.
"""
from typing import List, Dict, Any
import numpy as np
from ..simulation.pm import PhysicalMachine


def evaluate_cluster_performance(
    pms: List[PhysicalMachine],
    migration_count: int = 0,
    migration_cost: float = 0.0,
    observation_hours: float = 1.0,
    upper_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Computes all standard research metrics matching Table 6 in Alelyani et al. (2026):
    - AUR: Average Utilization of Resources across active hosts
    - Active PMs / Idle PMs
    - Total Energy Consumption (TEC) in Watt-hours
    - Overloaded PMs (> upper_threshold)
    - Underloaded PMs (< 0.35)
    - SLATAH, PDM, SLAv
    - Resource Imbalance
    """
    total_pms = len(pms)
    active_pms = [p for p in pms if p.is_active]
    n_active = len(active_pms)
    n_idle = total_pms - n_active

    if n_active > 0:
        cpu_ratios = [p.cpu_utilization_ratio for p in active_pms]
        mem_ratios = [p.mem_utilization_ratio for p in active_pms]
        aur = float(np.mean([(c + m) / 2.0 for c, m in zip(cpu_ratios, mem_ratios)]) * 100.0)
        imbalance = float(np.mean([abs(c - m) for c, m in zip(cpu_ratios, mem_ratios)]))

        overloaded = sum(1 for p in active_pms if p.cpu_utilization_ratio > upper_threshold)
        underloaded = sum(1 for p in active_pms if p.cpu_utilization_ratio < 0.35)

        # SLATAH: fraction of active time hosts spent at >= 100% capacity
        slatah = sum(1 for p in active_pms if p.cpu_utilization_ratio >= 1.0) / max(1, n_active)
    else:
        aur = 0.0
        imbalance = 0.0
        overloaded = 0
        underloaded = 0
        slatah = 0.0

    # Total energy across active and sleep PMs
    total_power_w = sum(p.compute_current_power() for p in pms)
    tec_wph = total_power_w * observation_hours

    # PDM (Performance Degradation due to Migrations)
    pdm = migration_cost if migration_count > 0 else 0.0
    slav = slatah * pdm

    return {
        "AUR_percent": round(aur, 2),
        "active_pms": n_active,
        "idle_pms": n_idle,
        "total_energy_wph": round(tec_wph, 2),
        "overloaded_pms": overloaded,
        "underloaded_pms": underloaded,
        "migrated_vms": migration_count,
        "migration_cost": round(migration_cost, 6),
        "SLATAH": round(slatah, 5),
        "PDM": round(pdm, 5),
        "SLAv": round(slav, 6),
        "resource_imbalance": round(imbalance, 4)
    }

