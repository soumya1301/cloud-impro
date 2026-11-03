"""
VM and PM representations for heterogeneous cluster scheduling simulation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import numpy as np


@dataclass
class VirtualMachine:
    """Represents a VM instance in the scheduling environment."""
    vm_id: str
    arrival_time: float               # Arrival timestamp (seconds)
    cpu_req: float                   # CPU requirement (GCUs)
    mem_req: float                   # Memory requirement (ratio)
    priority: int = 100              # Execution priority [0-300]
    duration_seconds: float = 1800.0 # Lifetime duration
    memory_gb: float = 4.0           # Physical memory footprint for migration
    profile_class: int = 0           # Workload class from K-Means++
    is_cpu_intensive: bool = False
    assigned_pm_id: Optional[str] = None


@dataclass
class PhysicalMachine:
    """Represents a heterogeneous PM with capacity constraints and power model."""
    pm_id: str
    pm_type: str = "hp_g5"
    cpu_capacity: float = 1.0        # Normalized capacity
    mem_capacity: float = 0.50
    network_bandwidth_gbps: float = 4.0
    idle_power_w: float = 93.7
    max_power_w: float = 135.0
    sleep_power_w: float = 10.0
    k_nonlinearity: float = 1.4

    # Dynamic state
    current_cpu_load: float = 0.0
    current_mem_load: float = 0.0
    predicted_cpu_load: float = 0.0
    predicted_mem_load: float = 0.0
    hosted_vms: Dict[str, VirtualMachine] = field(default_factory=dict)

    # Dynamic threshold parameters
    upper_threshold: float = 0.75
    lower_threshold: float = 0.35

    @property
    def is_active(self) -> bool:
        return len(self.hosted_vms) > 0

    @property
    def cpu_utilization_ratio(self) -> float:
        return self.current_cpu_load / max(self.cpu_capacity, 1e-6)

    @property
    def mem_utilization_ratio(self) -> float:
        return self.current_mem_load / max(self.mem_capacity, 1e-6)

    @property
    def cpu_intensive_ratio(self) -> float:
        """Calculates proportion of host CPU utilized by Type 2 (CPU-intensive) VMs."""
        if self.current_cpu_load <= 0:
            return 0.0
        cpu_int_sum = sum(vm.cpu_req for vm in self.hosted_vms.values() if vm.is_cpu_intensive)
        return cpu_int_sum / max(self.current_cpu_load, 1e-6)

    def can_host(self, vm: VirtualMachine, enforce_threshold: bool = True) -> bool:
        """Checks if VM fits within hard physical limits and optional upper threshold."""
        new_cpu = self.current_cpu_load + vm.cpu_req
        new_mem = self.current_mem_load + vm.mem_req

        # Hard physical limits
        if new_cpu > self.cpu_capacity or new_mem > self.mem_capacity:
            return False

        # Threshold limit
        if enforce_threshold:
            if (new_cpu / self.cpu_capacity) > self.upper_threshold:
                return False
            if (new_mem / self.mem_capacity) > self.upper_threshold:
                return False

        # CPU-intensive diversity invariant (<= 25% of load from CPU-intensive VMs)
        if vm.is_cpu_intensive:
            curr_int = sum(v.cpu_req for v in self.hosted_vms.values() if v.is_cpu_intensive)
            new_int_ratio = (curr_int + vm.cpu_req) / new_cpu
            if new_int_ratio > 0.25 and len(self.hosted_vms) > 0:
                return False

        return True

    def allocate(self, vm: VirtualMachine) -> None:
        """Assigns VM to this PM."""
        self.hosted_vms[vm.vm_id] = vm
        self.current_cpu_load += vm.cpu_req
        self.current_mem_load += vm.mem_req
        vm.assigned_pm_id = self.pm_id

    def release(self, vm_id: str) -> Optional[VirtualMachine]:
        """Releases VM from this PM."""
        if vm_id in self.hosted_vms:
            vm = self.hosted_vms.pop(vm_id)
            self.current_cpu_load = max(0.0, self.current_cpu_load - vm.cpu_req)
            self.current_mem_load = max(0.0, self.current_mem_load - vm.mem_req)
            vm.assigned_pm_id = None
            return vm
        return None

    def compute_current_power(self) -> float:
        """Computes current power in Watts based on normalized CPU utilization."""
        if not self.is_active:
            return self.sleep_power_w
        u = np.clip(self.cpu_utilization_ratio, 0.0, 1.0)
        return float(self.idle_power_w + (self.max_power_w - self.idle_power_w) * (u ** self.k_nonlinearity))

    def update_dynamic_threshold(self, predicted_load: float, safety_margin: float = 0.10) -> None:
        """Adapts upper threshold dynamically based on predicted future load."""
        self.predicted_cpu_load = predicted_load
        if predicted_load > 0.65:
            # Back off threshold to provide headroom against anticipated surge
            self.upper_threshold = max(0.60, 0.75 - safety_margin)
        else:
            self.upper_threshold = 0.75

