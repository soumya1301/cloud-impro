"""
Comprehensive automated unit and regression test suite.
Covers:
1. Telemetry schema & dimension verification
2. Resource scaling & sliding window integrity
3. Multivariate LSTM architecture & forward pass
4. Prediction metrics computation
5. SPECpower nonlinear power model calibration
6. PM hard physical capacity constraints & dynamic thresholds
7. Dynamic threshold adaptation under predicted loads
8. ACO placement feasibility, dynamic heuristics & unplaced tracking
9. Net-benefit live migration decision & lifetime evaluation
10. VM allocation integrity (duplicate prevention, negative load guard)
11. Sensitivity parameter propagation (k-nonlinearity)
"""
import pytest
import numpy as np
import torch
import pandas as pd

from src.vm_scheduler.data.schema_inspector import SchemaInspector
from src.vm_scheduler.preprocessing.normalization import ResourceScaler
from src.vm_scheduler.preprocessing.windowing import create_sliding_windows
from src.vm_scheduler.prediction.lstm import MultivariateLSTM
from src.vm_scheduler.prediction.metrics import compute_prediction_metrics
from src.vm_scheduler.clustering.kmeanspp import KMeansPlusPlusEvaluator
from src.vm_scheduler.energy.power_model import ServerPowerModel
from src.vm_scheduler.simulation.pm import PhysicalMachine, VirtualMachine
from src.vm_scheduler.scheduling.aco import AntColonyPlacer
from src.vm_scheduler.scheduling.migration import MigrationManager
from src.vm_scheduler.scheduling.placement import LSTMKMeansACOPlacer


def test_schema_dimension_verification():
    inspector = SchemaInspector()
    fields = {
        "average_usage": "struct<cpus: double, memory: double>",
        "cycles_per_instruction": "double",
        "memory_accesses_per_instruction": "double"
    }
    dims = inspector.verify_resource_dimensions(fields)
    assert dims["CPU"] is True
    assert dims["RAM"] is True
    assert dims["Network"] is False
    assert dims["Disk_IO"] is False


def test_resource_scaler():
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    scaler = ResourceScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled[0], [0.0, 0.0])
    assert np.allclose(X_scaled[-1], [1.0, 1.0])
    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inv)


def test_sliding_window_shapes():
    data = np.random.rand(100, 2)
    history_len = 12
    horizon = 2
    X, y = create_sliding_windows(data, history_len=history_len, horizon=horizon)
    assert X.shape == (100 - history_len - horizon + 1, history_len, 2)
    assert y.shape == (100 - history_len - horizon + 1, 2)


def test_lstm_forward_pass():
    model = MultivariateLSTM(input_dim=2, hidden_dim=32, num_layers=1, output_dim=2)
    dummy_input = torch.randn(8, 12, 2)
    out = model(dummy_input)
    assert out.shape == (8, 2)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_prediction_metrics():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [2.9, 4.2]])
    metrics = compute_prediction_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0.0


def test_specpower_model():
    model = ServerPowerModel(p_idle=93.7, p_max=135.0, p_sleep=10.0, k_nonlinearity=1.4)
    # Sleep power
    assert model.compute_power(0.0, is_active=False) == 10.0
    # Idle power
    assert np.isclose(model.compute_power(0.0, is_active=True), 93.7)
    # Peak power
    assert np.isclose(model.compute_power(1.0, is_active=True), 135.0)
    # Half load
    half_power = model.compute_power(0.5, is_active=True)
    assert 93.7 < half_power < 135.0


def test_pm_hard_capacity_constraints():
    pm = PhysicalMachine(pm_id="pm_01", cpu_capacity=1.0, mem_capacity=0.5, upper_threshold=0.75)
    vm1 = VirtualMachine(vm_id="vm_01", arrival_time=0.0, cpu_req=0.5, mem_req=0.2)
    vm2 = VirtualMachine(vm_id="vm_02", arrival_time=0.0, cpu_req=0.6, mem_req=0.2)

    assert pm.can_host(vm1, enforce_threshold=True) is True
    pm.allocate(vm1)
    # Exceeds threshold: 0.5 + 0.6 = 1.1 > 1.0 capacity
    assert pm.can_host(vm2, enforce_threshold=False) is False


def test_dynamic_threshold_adaptation():
    """Verify that dynamic threshold backs off under high load and expands under low load."""
    pm = PhysicalMachine(pm_id="pm_dyn", cpu_capacity=1.0, mem_capacity=1.0, upper_threshold=0.75)
    pm.is_dynamic_threshold_enabled = True

    # High predicted future demand -> threshold backs off to provide headroom
    pm.update_dynamic_threshold(predicted_cpu_demand=0.85, predicted_mem_demand=0.85)
    thresh_high = pm.upper_threshold
    assert thresh_high < 0.75

    # Low predicted future demand -> threshold allows higher consolidation
    pm.update_dynamic_threshold(predicted_cpu_demand=0.15, predicted_mem_demand=0.15)
    thresh_low = pm.upper_threshold
    assert thresh_low > thresh_high
    # Hard bounds respected
    assert 0.60 <= pm.upper_threshold <= 0.90


def test_aco_placement_feasibility():
    """Verify ACO finds feasible placement without exceeding capacity."""
    pms = [
        PhysicalMachine(pm_id=f"pm_{i}", cpu_capacity=1.0, mem_capacity=1.0, upper_threshold=0.8)
        for i in range(5)
    ]
    vms = [
        VirtualMachine(vm_id=f"vm_{i}", arrival_time=0.0, cpu_req=0.2, mem_req=0.2)
        for i in range(10)
    ]
    aco = AntColonyPlacer(num_ants=5, max_iterations=3, random_seed=42)
    placement, fitness, conv, info = aco.optimize_placement(vms, pms, enforce_threshold=True)

    assert len(placement) == 10
    assert info["placed_vms"] == 10
    assert info["unplaced_vms"] == 0
    assert info["placement_rate_percent"] == 100.0

    # Verify no PM is overcommitted
    load_cpu = {p.pm_id: 0.0 for p in pms}
    for vm in vms:
        load_cpu[placement[vm.vm_id]] += vm.cpu_req
    for pm_id, load in load_cpu.items():
        assert load <= 1.0 + 1e-6


def test_migration_decision():
    mig_mgr = MigrationManager(network_bandwidth_gbps=4.0, min_net_benefit_wh=0.1)
    vm = VirtualMachine(vm_id="vm_01", arrival_time=0.0, cpu_req=0.1, mem_req=0.1, memory_gb=2.0)
    src = PhysicalMachine(pm_id="pm_src", cpu_capacity=1.0, mem_capacity=1.0)
    tgt = PhysicalMachine(pm_id="pm_tgt", cpu_capacity=1.0, mem_capacity=1.0)

    src.allocate(vm)
    should_migrate, details = mig_mgr.evaluate_migration(vm, src, tgt)
    assert isinstance(should_migrate, (bool, np.bool_))
    assert "migration_time_seconds" in details
    assert details["migration_time_seconds"] > 0


def test_vm_allocation_integrity():
    """Verify duplicate allocation prevention and negative load protection."""
    pm = PhysicalMachine(pm_id="pm_int", cpu_capacity=1.0, mem_capacity=1.0)
    vm = VirtualMachine(vm_id="vm_dup", arrival_time=0.0, cpu_req=0.3, mem_req=0.3)

    # Initial allocation succeeds
    pm.allocate(vm)
    assert pm.current_cpu_load == pytest.approx(0.3)
    assert len(pm.hosted_vms) == 1

    # can_host returns False for already hosted VM
    assert pm.can_host(vm) is False

    # Duplicate allocation raises ValueError
    with pytest.raises(ValueError, match="already allocated"):
        pm.allocate(vm)
    assert len(pm.hosted_vms) == 1

    # Release VM restores zero load
    released = pm.release("vm_dup")
    assert released.vm_id == "vm_dup"
    assert pm.current_cpu_load == pytest.approx(0.0)
    assert len(pm.hosted_vms) == 0

    # Releasing non-existent VM returns None and does not make load negative
    assert pm.release("vm_nonexistent") is None
    assert pm.current_cpu_load >= 0.0


def test_sensitivity_k_nonlinearity_propagation():
    """Verify that varying k-nonlinearity parameter monotonically changes calculated power."""
    pms_k1 = [PhysicalMachine(pm_id="pm_1", k_nonlinearity=1.0)]
    pms_k2 = [PhysicalMachine(pm_id="pm_2", k_nonlinearity=2.0)]
    vm = VirtualMachine(vm_id="vm_load", arrival_time=0.0, cpu_req=0.5, mem_req=0.5)

    pms_k1[0].allocate(vm)
    pms_k2[0].allocate(vm)

    power_k1 = pms_k1[0].compute_current_power()
    power_k2 = pms_k2[0].compute_current_power()

    # Under 50% load, u ** k:
    # k=1.0: 0.5^1 = 0.50 -> power = 93.7 + (135-93.7)*0.50 = 114.35 W
    # k=2.0: 0.5^2 = 0.25 -> power = 93.7 + (135-93.7)*0.25 = 104.025 W
    # Higher k results in lower power for sub-1.0 load
    assert power_k1 > power_k2
