"""
Sercon and Round Robin (RR) scheduling baselines.
"""
from typing import List, Dict, Optional
import copy
from ..simulation.pm import PhysicalMachine, VirtualMachine


class SerconScheduler:
    """
    Sercon consolidation algorithm:
    1. First Fit initial placement with upper threshold.
    2. Identifies underloaded PMs (< 35% utilization).
    3. Moves VMs from least-loaded PMs to most-loaded PMs.
    4. All-or-nothing policy: migrates all VMs from a host or none.
    """

    def __init__(self, upper_threshold: float = 0.75, lower_threshold: float = 0.35):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def schedule(self, vms: List[VirtualMachine], pms: List[PhysicalMachine]) -> Dict[str, str]:
        # Initial placement using First Fit
        placement = {}
        for vm in vms:
            for pm in pms:
                if pm.can_host(vm, enforce_threshold=True):
                    pm.allocate(vm)
                    placement[vm.vm_id] = pm.pm_id
                    break

        # Sercon Consolidation Phase
        active_pms = [p for p in pms if p.is_active]
        underloaded = [p for p in active_pms if p.cpu_utilization_ratio < self.lower_threshold]
        candidates = [p for p in active_pms if p.cpu_utilization_ratio >= self.lower_threshold]

        # Sort underloaded ascending (least loaded first), candidates descending (most loaded first)
        underloaded.sort(key=lambda p: p.cpu_utilization_ratio)
        candidates.sort(key=lambda p: p.cpu_utilization_ratio, reverse=True)

        for src_pm in underloaded:
            vms_to_move = list(src_pm.hosted_vms.values())
            plan = {}
            possible = True

            for vm in vms_to_move:
                moved = False
                for tgt_pm in candidates:
                    if tgt_pm.can_host(vm, enforce_threshold=True):
                        plan[vm.vm_id] = tgt_pm
                        moved = True
                        break
                if not moved:
                    possible = False
                    break

            # All-or-nothing check
            if possible:
                for vm in vms_to_move:
                    tgt = plan[vm.vm_id]
                    src_pm.release(vm.vm_id)
                    tgt.allocate(vm)
                    placement[vm.vm_id] = tgt.pm_id

        return placement


class RoundRobinScheduler:
    """
    Round Robin scheduling algorithm:
    Distributes arriving VMs cyclically across active hosts adhering to upper threshold.
    """

    def __init__(self, upper_threshold: float = 0.75):
        self.upper_threshold = upper_threshold
        self.current_idx = 0

    def schedule(self, vms: List[VirtualMachine], pms: List[PhysicalMachine]) -> Dict[str, str]:
        placement = {}
        n_pms = len(pms)

        for vm in vms:
            assigned = False
            for step in range(n_pms):
                idx = (self.current_idx + step) % n_pms
                pm = pms[idx]
                if pm.can_host(vm, enforce_threshold=True):
                    pm.allocate(vm)
                    placement[vm.vm_id] = pm.pm_id
                    self.current_idx = (idx + 1) % n_pms
                    assigned = True
                    break

            if not assigned:
                # Fallback
                for step in range(n_pms):
                    idx = (self.current_idx + step) % n_pms
                    pm = pms[idx]
                    if pm.can_host(vm, enforce_threshold=False):
                        pm.allocate(vm)
                        placement[vm.vm_id] = pm.pm_id
                        self.current_idx = (idx + 1) % n_pms
                        assigned = True
                        break

        return placement

