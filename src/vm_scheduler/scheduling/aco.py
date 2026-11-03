"""
Ant Colony Optimization (ACO) algorithm for multi-objective VM placement.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import copy
from ..simulation.pm import PhysicalMachine, VirtualMachine


class AntColonyPlacer:
    """
    Ant Colony Optimization (ACO) engine for multi-objective VM placement.
    Minimizes:
    J = w_energy * E_norm + w_active * A_norm + w_balance * B_norm + w_sla * SLA_norm
    """

    def __init__(
        self,
        num_ants: int = 20,
        max_iterations: int = 30,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.8,
        deposit_constant: float = 100.0,
        random_seed: int = 42
    ):
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.deposit_constant = deposit_constant
        self.rng = np.random.RandomState(random_seed)

    def optimize_placement(
        self,
        vms: List[VirtualMachine],
        pms: List[PhysicalMachine],
        enforce_threshold: bool = True
    ) -> Tuple[Dict[str, str], float, List[float]]:
        """
        Executes ACO to place `vms` onto `pms`.
        Returns:
            best_placement: dict mapping vm_id -> pm_id
            best_fitness: optimal fitness score
            convergence_curve: list of best fitness per iteration
        """
        n_vms = len(vms)
        n_pms = len(pms)

        if n_vms == 0 or n_pms == 0:
            return {}, 0.0, []

        # Initialize pheromone matrix: shape (n_vms, n_pms)
        pheromone = np.ones((n_vms, n_pms), dtype=np.float32)

        best_placement = None
        best_fitness = float("inf")
        convergence_curve = []

        # Precompute base heuristic desirability matrix
        heuristic = np.zeros((n_vms, n_pms), dtype=np.float32)
        for i, vm in enumerate(vms):
            for j, pm in enumerate(pms):
                if pm.can_host(vm, enforce_threshold=enforce_threshold):
                    # Desirability favors PMs that maintain balanced CPU/RAM ratio and high energy proportional efficiency
                    rem_cpu = max(0.01, pm.cpu_capacity - pm.current_cpu_load - vm.cpu_req)
                    rem_mem = max(0.01, pm.mem_capacity - pm.current_mem_load - vm.mem_req)
                    balance = 1.0 / (1.0 + abs(rem_cpu - rem_mem))
                    power_eff = 1.0 / max(10.0, pm.compute_current_power())
                    heuristic[i, j] = balance * power_eff * 100.0
                else:
                    heuristic[i, j] = 1e-4 # Infeasible baseline

        # Main ACO iterations
        for it in range(self.max_iterations):
            iter_placements = []
            iter_fitnesses = []

            for ant in range(self.num_ants):
                ant_placement = {}
                # Create local copy of PM state for feasibility tracking
                ant_pms = [copy.copy(pm) for pm in pms]
                for p in ant_pms:
                    p.hosted_vms = dict(p.hosted_vms)

                # Each ant constructs a complete solution
                for i, vm in enumerate(vms):
                    # Candidate selection: consider active hosts plus a small pool of inactive hosts (top candidates)
                    active_indices = [idx for idx, p in enumerate(ant_pms) if p.is_active]
                    inactive_indices = [idx for idx, p in enumerate(ant_pms) if not p.is_active][:5]
                    search_pool = active_indices + inactive_indices
                    if not search_pool:
                        search_pool = list(range(min(20, n_pms)))

                    feas_indices = [
                        j for j in search_pool
                        if ant_pms[j].can_host(vm, enforce_threshold=enforce_threshold)
                    ]

                    if not feas_indices:
                        # Fallback: check inactive hosts
                        feas_indices = [
                            j for j in range(n_pms)
                            if ant_pms[j].can_host(vm, enforce_threshold=enforce_threshold)
                        ][:10]

                    if not feas_indices:
                        # Fallback: relax threshold but respect hard capacity
                        feas_indices = [
                            j for j in range(n_pms)
                            if ant_pms[j].can_host(vm, enforce_threshold=False)
                        ][:5]

                    if not feas_indices:
                        continue

                    probs = (pheromone[i, feas_indices] ** self.alpha) * (heuristic[i, feas_indices] ** self.beta)
                    total_prob = probs.sum()

                    if total_prob > 0 and self.rng.rand() > self.q0:
                        probs /= total_prob
                        chosen_j = self.rng.choice(feas_indices, p=probs)
                    else:
                        chosen_j = feas_indices[np.argmax(probs)]

                    ant_pms[chosen_j].allocate(vm)
                    ant_placement[vm.vm_id] = ant_pms[chosen_j].pm_id

                fitness = self._calculate_fitness(ant_placement, ant_pms)
                iter_placements.append(ant_placement)
                iter_fitnesses.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_placement = ant_placement

            convergence_curve.append(best_fitness)

            # Global pheromone evaporation
            pheromone = (1.0 - self.rho) * pheromone

            # Pheromone deposit by the best ant
            if best_placement:
                pm_id_to_j = {pm.pm_id: j for j, pm in enumerate(pms)}
                deposit = self.deposit_constant / max(best_fitness, 1.0)
                for i, vm in enumerate(vms):
                    if vm.vm_id in best_placement:
                        j = pm_id_to_j[best_placement[vm.vm_id]]
                        pheromone[i, j] += deposit

        return best_placement, best_fitness, convergence_curve

    def _calculate_fitness(self, placement: Dict[str, str], pms: List[PhysicalMachine]) -> float:
        """
        Multi-objective normalized fitness:
        J = 0.40 * Energy + 0.25 * Active_PMs + 0.15 * Imbalance + 0.20 * SLA_Risk
        """
        total_energy = sum(pm.compute_current_power() for pm in pms)
        active_pms = sum(1 for pm in pms if pm.is_active)

        # Resource imbalance across active hosts: mean(|cpu_ratio - mem_ratio|)
        imbalance = 0.0
        sla_risk = 0.0
        if active_pms > 0:
            active_list = [p for p in pms if p.is_active]
            imbalance = np.mean([abs(p.cpu_utilization_ratio - p.mem_utilization_ratio) for p in active_list])
            # SLA risk penalty for hosts exceeding 80% utilization
            sla_risk = sum(max(0.0, p.cpu_utilization_ratio - 0.80) for p in active_list)

        # Normalization factors
        e_norm = total_energy / (len(pms) * 135.0)
        a_norm = active_pms / len(pms)

        fitness = (0.40 * e_norm) + (0.25 * a_norm) + (0.15 * imbalance) + (0.20 * sla_risk)
        return float(fitness)

