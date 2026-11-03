"""
Modified First Fit (MFFA) and Modified Best Fit (MBFA) VM placement baselines.
"""
from typing import List, Dict, Optional
from ..simulation.pm import PhysicalMachine, VirtualMachine


class ModifiedFirstFitScheduler:
    """
    Modified First Fit Algorithm (MFFA):
    Iterates through PM list and assigns VM to the first PM that satisfies capacity
    and does not exceed the upper utilization threshold (75%).
    """

    def __init__(self, upper_threshold: float = 0.75):
        self.upper_threshold = upper_threshold

    def schedule(self, vms: List[VirtualMachine], pms: List[PhysicalMachine]) -> Dict[str, str]:
        placement = {}
        for vm in vms:
            assigned = False
            for pm in pms:
                if pm.can_host(vm, enforce_threshold=True):
                    pm.allocate(vm)
                    placement[vm.vm_id] = pm.pm_id
                    assigned = True
                    break
            if not assigned:
                # Fallback to any host with physical capacity
                for pm in pms:
                    if pm.can_host(vm, enforce_threshold=False):
                        pm.allocate(vm)
                        placement[vm.vm_id] = pm.pm_id
                        assigned = True
                        break
        return placement


class ModifiedBestFitScheduler:
    """
    Modified Best Fit Algorithm (MBFA):
    Assigns VM to the PM that has the smallest remaining capacity after allocation,
    while respecting the upper utilization threshold (75%).
    """

    def __init__(self, upper_threshold: float = 0.75):
        self.upper_threshold = upper_threshold

    def schedule(self, vms: List[VirtualMachine], pms: List[PhysicalMachine]) -> Dict[str, str]:
        placement = {}
        for vm in vms:
            best_pm = None
            min_remaining_cpu = float("inf")

            for pm in pms:
                if pm.can_host(vm, enforce_threshold=True):
                    rem = pm.cpu_capacity - (pm.current_cpu_load + vm.cpu_req)
                    if rem < min_remaining_cpu:
                        min_remaining_cpu = rem
                        best_pm = pm

            if best_pm is None:
                # Fallback without threshold constraint
                for pm in pms:
                    if pm.can_host(vm, enforce_threshold=False):
                        rem = pm.cpu_capacity - (pm.current_cpu_load + vm.cpu_req)
                        if rem < min_remaining_cpu:
                            min_remaining_cpu = rem
                            best_pm = pm

            if best_pm is not None:
                best_pm.allocate(vm)
                placement[vm.vm_id] = best_pm.pm_id

        return placement

