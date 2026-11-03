"""
Comprehensive scheduling evaluation metrics: AUR, Energy, Active PMs, SLAv, Imbalance, Completeness.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from ..simulation.pm import PhysicalMachine


def evaluate_cluster_performance(
    pms: List[PhysicalMachine],
    migration_count: int = 0,
    migration_time_seconds: float = 0.0,
    migration_energy_wh: float = 0.0,
    observation_hours: float = 1.0,
    upper_threshold: float = 0.75,
    requested_vms: Optional[int] = None,
    placed_vms: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes standard research metrics matching and extending Alelyani et al. (2026):
    - AUR: Average Utilization of Resources across active hosts (%)
    - Active PMs / Idle PMs
    - Total Energy Consumption (TEC) in Watt-hours (Wh)
    - Overloaded PMs (> upper_threshold)
    - Underloaded PMs (< 0.35)
    - SLATAH, PDM, SLAv
    - Resource Imbalance
    - Placement completeness metrics
    """
    total_pms = len(pms)
    active_pms = [p for p in pms if p.is_active]
    n_active = len(active_pms)
    n_idle = total_pms - n_active

    total_hosted = sum(len(p.hosted_vms) for p in active_pms)
    req = requested_vms if requested_vms is not None else total_hosted
    placed = placed_vms if placed_vms is not None else total_hosted
    unplaced = max(0, req - placed)
    placement_rate = (placed / max(1, req)) * 100.0

    if n_active > 0:
        cpu_ratios = [p.cpu_utilization_ratio for p in active_pms]
        mem_ratios = [p.mem_utilization_ratio for p in active_pms]
        aur = float(np.mean([(c + m) / 2.0 for c, m in zip(cpu_ratios, mem_ratios)]) * 100.0)
        imbalance = float(np.mean([abs(c - m) for c, m in zip(cpu_ratios, mem_ratios)]))

        overloaded = sum(1 for p in active_pms if p.cpu_utilization_ratio > (upper_threshold + 1e-6))
        underloaded = sum(1 for p in active_pms if p.cpu_utilization_ratio < 0.35)

        # SLATAH: fraction of active hosts operating at or beyond 100% capacity
        slatah = sum(1 for p in active_pms if p.cpu_utilization_ratio >= (1.0 - 1e-6)) / max(1, n_active)
    else:
        aur = 0.0
        imbalance = 0.0
        overloaded = 0
        underloaded = 0
        slatah = 0.0

    # Total energy integrated over observation duration in Watt-hours
    total_power_w = sum(p.compute_current_power() for p in pms)
    tec_wph = (total_power_w * observation_hours) + migration_energy_wh

    # Performance Degradation due to Migrations (PDM)
    # Following Beloglazov & Buyya (2012): PDM is the cumulative migration downtime overhead ratio
    pdm = (migration_time_seconds / 3600.0) / max(1, placed) if migration_count > 0 else 0.0
    slav = slatah * pdm

    return {
        "AUR_percent": round(aur, 2),
        "active_pms": n_active,
        "idle_pms": n_idle,
        "total_energy_wph": round(tec_wph, 2),
        "overloaded_pms": overloaded,
        "underloaded_pms": underloaded,
        "migrated_vms": migration_count,
        "migration_time_seconds": round(migration_time_seconds, 2),
        "migration_energy_wh": round(migration_energy_wh, 6),
        "SLATAH": round(slatah, 6),
        "PDM": round(pdm, 6),
        "SLAv": round(slav, 8),
        "resource_imbalance": round(imbalance, 4),
        "requested_vms": req,
        "placed_vms": placed,
        "unplaced_vms": unplaced,
        "placement_rate_percent": round(placement_rate, 2)
    }
