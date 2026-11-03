"""
Live VM migration model and cost-benefit decision engine.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from ..simulation.pm import PhysicalMachine, VirtualMachine
from ..energy.power_model import ServerPowerModel


class MigrationManager:
    """
    Evaluates VM live migration feasibility and net economic benefit:
    NetBenefit = EnergySaved - MigrationEnergy - SLAPenalty
    """

    def __init__(
        self,
        network_bandwidth_gbps: float = 4.0,
        degradation_factor: float = 0.10,
        min_net_benefit_wh: float = 0.50,
        nic_power_overhead_w: float = 15.0,
        power_model: Optional[ServerPowerModel] = None
    ):
        self.bw_gbps = network_bandwidth_gbps
        self.degradation_factor = degradation_factor
        self.min_benefit = min_net_benefit_wh
        self.nic_power_w = nic_power_overhead_w
        self.power_model = power_model or ServerPowerModel()

    def compute_migration_time_seconds(self, vm: VirtualMachine) -> float:
        """
        Computes live migration duration based on memory footprint and link bandwidth:
        T = VM_memory (bits) / Bw (bps)
        """
        mem_gbit = vm.memory_gb * 8.0
        return float(mem_gbit / self.bw_gbps)

    def evaluate_migration(
        self,
        vm: VirtualMachine,
        source_pm: PhysicalMachine,
        target_pm: PhysicalMachine,
        observation_horizon_hours: float = 1.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates economic viability of migrating `vm` from `source_pm` to `target_pm`.
        Takes remaining VM lifetime into account.
        """
        # Validate that VM currently resides on source_pm
        if vm.vm_id not in source_pm.hosted_vms:
            return False, {"net_benefit": -1.0, "reason": "VM not hosted on source PM"}

        # Validate target PM capacity and operating threshold
        if not target_pm.can_host(vm, enforce_threshold=True):
            return False, {"net_benefit": -1.0, "reason": "Target host cannot accommodate VM"}

        # Transfer time in seconds and hours
        mig_time_s = self.compute_migration_time_seconds(vm)
        mig_time_h = mig_time_s / 3600.0

        # Remaining lifetime of the VM in hours
        remaining_hours = min(observation_horizon_hours, max(0.1, vm.duration_seconds / 3600.0))

        # Current power dissipation
        p_src_before = source_pm.compute_current_power()
        p_tgt_before = target_pm.compute_current_power()

        # Hypothetical power after migration
        src_new_cpu = max(0.0, source_pm.current_cpu_load - vm.cpu_req)
        src_active_after = (len(source_pm.hosted_vms) - 1) > 0

        if not src_active_after:
            p_src_after = source_pm.sleep_power_w
        else:
            u_src = src_new_cpu / source_pm.cpu_capacity
            p_src_after = source_pm.idle_power_w + (source_pm.max_power_w - source_pm.idle_power_w) * (u_src ** source_pm.k_nonlinearity)

        tgt_new_cpu = target_pm.current_cpu_load + vm.cpu_req
        u_tgt = tgt_new_cpu / target_pm.cpu_capacity
        p_tgt_after = target_pm.idle_power_w + (target_pm.max_power_w - target_pm.idle_power_w) * (u_tgt ** target_pm.k_nonlinearity)

        # Expected energy saved over remaining lifetime
        power_saved_w = (p_src_before + p_tgt_before) - (p_src_after + p_tgt_after)
        energy_saved_wh = power_saved_w * remaining_hours

        # Migration energy consumption (network NIC and CPU transmission power overhead)
        migration_energy_wh = self.nic_power_w * mig_time_h

        # Performance degradation penalty (Watt-hours equivalent penalty)
        sla_penalty_wh = self.degradation_factor * vm.cpu_req * 10.0

        net_benefit = energy_saved_wh - migration_energy_wh - sla_penalty_wh

        metrics = {
            "migration_time_seconds": round(mig_time_s, 2),
            "migration_energy_wh": round(migration_energy_wh, 6),
            "energy_saved_wh": round(energy_saved_wh, 4),
            "sla_penalty_wh": round(sla_penalty_wh, 4),
            "net_benefit_wh": round(net_benefit, 4)
        }

        should_migrate = net_benefit >= self.min_benefit
        return should_migrate, metrics
