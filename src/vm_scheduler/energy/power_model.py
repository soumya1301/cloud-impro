"""
Nonlinear SPECpower server energy model and time-integrated power estimation.
"""
from typing import Dict, Any
import numpy as np


class ServerPowerModel:
    """
    Nonlinear power model calibrated against the SPECpower benchmark:
    P(u) = P_idle + (P_max - P_idle) * (u ** k)
    If active but 0% load: P = P_idle.
    If in sleep mode: P = P_sleep (typically 10 W for HP G5).
    """

    def __init__(
        self,
        p_idle: float = 93.7,
        p_max: float = 135.0,
        p_sleep: float = 10.0,
        k_nonlinearity: float = 1.4
    ):
        self.p_idle = p_idle
        self.p_max = p_max
        self.p_sleep = p_sleep
        self.k = k_nonlinearity

    def compute_power(self, cpu_utilization: float, is_active: bool = True) -> float:
        """Returns power in Watts for a single host at normalized CPU utilization u in [0, 1]."""
        if not is_active:
            return self.p_sleep

        u = np.clip(cpu_utilization, 0.0, 1.0)
        return float(self.p_idle + (self.p_max - self.p_idle) * (u ** self.k))

    def estimate_energy_wh(self, cpu_utilizations: np.ndarray, duration_hours: float, is_active: bool = True) -> float:
        """
        Integrates power over time duration (in hours).
        Energy (Watt-hours) = sum(P(t) * dt)
        """
        if not is_active:
            return float(self.p_sleep * duration_hours)

        powers = np.array([self.compute_power(u, is_active=True) for u in cpu_utilizations])
        return float(np.mean(powers) * duration_hours)

