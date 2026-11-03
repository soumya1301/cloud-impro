"""
VM and PM representations for heterogeneous cluster scheduling simulation.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import numpy as np
from ..energy.power_model import ServerPowerModel


@dataclass
class VirtualMachine:
    """Represents a VM instance in the scheduling environment."""
    vm_id: str
    arrival_time: float               # Arrival timestamp (seconds)
    cpu_req: float                   # CPU requirement (GCUs)
    mem_req: float                   # Memory requirement (normalized fraction)
    priority: int = 100              # Execution priority [0-300]
    duration_seconds: float = 1800.0 # Lifetime duration
    memory_gb: float = 4.0           # Physical memory footprint for migration
    profile_class: int = 0           # Workload class from K-Means++
    is_cpu_intensive: bool = False
    assigned_pm_id: Optional[str] = None

    # Predicted demands
    predicted_cpu: Optional[float] = None
    predicted_mem: Optional[float] = None

    def __post_init__(self):
        if self.cpu_req <= 0:
            raise ValueError(f"VM {self.vm_id} CPU requirement must be positive, got {self.cpu_req}")
        if self.mem_req <= 0:
            raise ValueError(f"VM {self.vm_id} memory requirement must be positive, got {self.mem_req}")
        if self.predicted_cpu is None:
            self.predicted_cpu = self.cpu_req
        if self.predicted_mem is None:
            self.predicted_mem = self.mem_req


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

    # Threshold parameters
    base_upper_threshold: float = 0.75
    min_upper_threshold: float = 0.60
    upper_threshold: float = 0.75
    lower_threshold: float = 0.35
    is_dynamic_threshold_enabled: bool = True

    # Workload diversity policy
    enforce_diversity_constraint: bool = False
    max_cpu_intensive_ratio: float = 0.25

    def __post_init__(self):
        if self.cpu_capacity <= 0 or self.mem_capacity <= 0:
            raise ValueError(f"PM {self.pm_id} capacities must be positive.")
        self.upper_threshold = self.base_upper_threshold

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
        """
        Validates whether VM fits on this host:
        1. Hard physical capacity check (CPU and Memory)
        2. Soft operating threshold check (if enforce_threshold=True)
        3. Configurable CPU-intensive diversity invariant
        """
        # Prevent duplicate allocation on same PM
        if vm.vm_id in self.hosted_vms:
            return False

        new_cpu = self.current_cpu_load + vm.cpu_req
        new_mem = self.current_mem_load + vm.mem_req

        # 1. Hard physical limits (must NEVER be exceeded)
        if new_cpu > (self.cpu_capacity + 1e-8) or new_mem > (self.mem_capacity + 1e-8):
            return False

        # 2. Operating threshold limits
        if enforce_threshold:
            if (new_cpu / self.cpu_capacity) > (self.upper_threshold + 1e-8):
                return False
            if (new_mem / self.mem_capacity) > (self.upper_threshold + 1e-8):
                return False

        # 3. CPU-intensive diversity invariant (if enabled)
        if self.enforce_diversity_constraint and vm.is_cpu_intensive:
            curr_int = sum(v.cpu_req for v in self.hosted_vms.values() if v.is_cpu_intensive)
            new_int_ratio = (curr_int + vm.cpu_req) / new_cpu
            if new_int_ratio > self.max_cpu_intensive_ratio and len(self.hosted_vms) > 0:
                return False

        return True

    def allocate(self, vm: VirtualMachine) -> None:
        """Assigns VM to this PM with assignment validation."""
        if vm.vm_id in self.hosted_vms:
            raise ValueError(f"VM {vm.vm_id} is already allocated on PM {self.pm_id}")

        self.hosted_vms[vm.vm_id] = vm
        self.current_cpu_load += vm.cpu_req
        self.current_mem_load += vm.mem_req
        vm.assigned_pm_id = self.pm_id

    def release(self, vm_id: str) -> Optional[VirtualMachine]:
        """Releases VM from this PM ensuring load never becomes negative."""
        if vm_id in self.hosted_vms:
            vm = self.hosted_vms.pop(vm_id)
            self.current_cpu_load = max(0.0, self.current_cpu_load - vm.cpu_req)
            self.current_mem_load = max(0.0, self.current_mem_load - vm.mem_req)
            vm.assigned_pm_id = None
            return vm
        return None

    def compute_current_power(self) -> float:
        """Computes current power in Watts using calibrated SPECpower non-linear model."""
        if not self.is_active:
            return self.sleep_power_w
        u = np.clip(self.cpu_utilization_ratio, 0.0, 1.0)
        return float(self.idle_power_w + (self.max_power_w - self.idle_power_w) * (u ** self.k_nonlinearity))

    def update_dynamic_threshold(
        self,
        predicted_cpu_demand: float,
        predicted_mem_demand: float = 0.0,
        safety_margin: float = 0.10
    ) -> None:
        """
        Dynamically adapts operating upper threshold based on predicted multi-dimensional demand.
        If dynamic thresholding is disabled, keeps threshold fixed at base_upper_threshold.
        """
        self.predicted_cpu_load = predicted_cpu_demand
        self.predicted_mem_load = predicted_mem_demand

        if not self.is_dynamic_threshold_enabled:
            self.upper_threshold = self.base_upper_threshold
            return

        pred_cpu_ratio = predicted_cpu_demand / max(self.cpu_capacity, 1e-6)
        pred_mem_ratio = predicted_mem_demand / max(self.mem_capacity, 1e-6)
        max_pred_ratio = max(pred_cpu_ratio, pred_mem_ratio)

        # Back off threshold when predicted surge exceeds 65% capacity
        if max_pred_ratio > 0.65:
            reduction = min(safety_margin, (max_pred_ratio - 0.65) * 0.5 + 0.05)
            self.upper_threshold = max(self.min_upper_threshold, self.base_upper_threshold - reduction)
        else:
            self.upper_threshold = self.base_upper_threshold
