"""
Ant Colony Optimization (ACO) algorithm for multi-objective VM placement.
Incorporates predicted multi-dimensional demands into dynamic heuristic desirability,
adaptive operating thresholds, and SLA risk penalization.
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import copy
from ..simulation.pm import PhysicalMachine, VirtualMachine
from ..energy.power_model import ServerPowerModel


class AntColonyPlacer:
    """
    Ant Colony Optimization (ACO) engine for multi-objective VM placement.
    Minimizes:
    J = w_energy * E_norm + w_active * A_norm + w_balance * B_norm + w_sla * SLA_norm + w_unplaced * U_norm
    """

    def __init__(
        self,
        num_ants: int = 15,
        max_iterations: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.8,
        deposit_constant: float = 100.0,
        tau_min: float = 0.01,
        tau_max: float = 100.0,
        random_seed: int = 42,
        power_model: Optional[ServerPowerModel] = None
    ):
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.deposit_constant = deposit_constant
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rng = np.random.RandomState(random_seed)
        self.power_model = power_model or ServerPowerModel()

    def optimize_placement(
        self,
        vms: List[VirtualMachine],
        pms: List[PhysicalMachine],
        enforce_threshold: bool = True,
        predicted_vm_demands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[Dict[str, str], float, List[float], Dict[str, Any]]:
        """
        Executes ACO to place `vms` onto `pms`.
        Returns:
            best_placement: dict mapping vm_id -> pm_id
            best_fitness: optimal fitness score
            convergence_curve: list of best fitness per iteration
            placement_info: metadata on placed vs unplaced VMs
        """
        n_vms = len(vms)
        n_pms = len(pms)

        if n_vms == 0 or n_pms == 0:
            return {}, 0.0, [], {"requested": n_vms, "placed": 0, "unplaced": n_vms, "completion_rate": 0.0}

        # Pre-extract static PM hardware parameters into contiguous arrays
        cpu_caps = np.array([p.cpu_capacity for p in pms], dtype=np.float32)
        mem_caps = np.array([p.mem_capacity for p in pms], dtype=np.float32)
        idle_powers = np.array([p.idle_power_w for p in pms], dtype=np.float32)
        max_powers = np.array([p.max_power_w for p in pms], dtype=np.float32)
        k_exp = np.array([p.k_nonlinearity for p in pms], dtype=np.float32)
        base_thresh = np.array([p.base_upper_threshold for p in pms], dtype=np.float32)

        # Pre-extract VM requested demands (hard physical constraints)
        vm_cpus = np.array([vm.cpu_req for vm in vms], dtype=np.float32)
        vm_mems = np.array([vm.mem_req for vm in vms], dtype=np.float32)
        vm_ids = [vm.vm_id for vm in vms]
        pm_ids = [pm.pm_id for pm in pms]

        # Extract predicted demands (for dynamic thresholding and SLA risk modulation)
        pred_cpus = np.zeros(n_vms, dtype=np.float32)
        pred_mems = np.zeros(n_vms, dtype=np.float32)
        for i, vm in enumerate(vms):
            if predicted_vm_demands and vm.vm_id in predicted_vm_demands:
                p_c, p_m = predicted_vm_demands[vm.vm_id]
                pred_cpus[i] = p_c
                pred_mems[i] = p_m
            else:
                pred_cpus[i] = vm.predicted_cpu or vm.cpu_req
                pred_mems[i] = vm.predicted_mem or vm.mem_req

        # Initialize bounded pheromone matrix: shape (n_vms, n_pms)
        pheromone = np.ones((n_vms, n_pms), dtype=np.float32)

        best_placement = None
        best_fitness = float("inf")
        best_unplaced_count = n_vms
        convergence_curve = []

        # Main ACO iterations
        for it in range(self.max_iterations):
            iter_best_placement = None
            iter_best_fitness = float("inf")
            iter_unplaced_count = n_vms

            for ant in range(self.num_ants):
                ant_placement = {}
                unplaced_vms = []
                cpu_loads = np.zeros(n_pms, dtype=np.float32)
                mem_loads = np.zeros(n_pms, dtype=np.float32)
                pred_cpu_loads = np.zeros(n_pms, dtype=np.float32)
                active_mask = np.zeros(n_pms, dtype=bool)
                active_indices = []

                # Sequential placement with dynamic heuristic updating
                for i in range(n_vms):
                    v_cpu = vm_cpus[i]
                    v_mem = vm_mems[i]
                    p_cpu = pred_cpus[i]
                    p_mem = pred_mems[i]
                    v_id = vm_ids[i]

                    feas_cand = []
                    if active_indices:
                        act_arr = np.array(active_indices, dtype=np.int32)
                        # Projected load including this VM's predicted demand
                        proj_u_c = (pred_cpu_loads[act_arr] + p_cpu) / cpu_caps[act_arr]
                        # Dynamic threshold for active candidates
                        cand_thresh = np.where(
                            proj_u_c > 0.65,
                            np.maximum(0.60, base_thresh[act_arr] - np.minimum(0.10, (proj_u_c - 0.65) * 0.5 + 0.05)),
                            base_thresh[act_arr]
                        )
                        # Candidate satisfies operating threshold if enforce_threshold
                        if enforce_threshold:
                            feas = ((cpu_loads[act_arr] + v_cpu) <= cand_thresh * cpu_caps[act_arr]) & \
                                   ((mem_loads[act_arr] + v_mem) <= cand_thresh * mem_caps[act_arr])
                        else:
                            feas = ((cpu_loads[act_arr] + v_cpu) <= cpu_caps[act_arr]) & \
                                   ((mem_loads[act_arr] + v_mem) <= mem_caps[act_arr])
                        feas_cand = act_arr[feas].tolist()

                    # Restricted candidate list (RCL) bound
                    if len(feas_cand) > 20:
                        feas_cand = feas_cand[:20]

                    # Fallback 1: Cold standby inactive hosts
                    if not feas_cand:
                        inact = np.where(~active_mask)[0]
                        feas_inact = [
                            j for j in inact[:10]
                            if (v_cpu <= base_thresh[j] * cpu_caps[j] and v_mem <= base_thresh[j] * mem_caps[j])
                        ]
                        feas_cand = feas_inact[:5]

                    # Fallback 2: Relax threshold to prevent VM rejection while strictly respecting hard capacity
                    if not feas_cand:
                        if active_indices:
                            act_arr = np.array(active_indices, dtype=np.int32)
                            feas = (cpu_loads[act_arr] + v_cpu <= cpu_caps[act_arr]) & \
                                   (mem_loads[act_arr] + v_mem <= mem_caps[act_arr])
                            feas_cand = act_arr[feas].tolist()[:5]

                    # Fallback 3: Inactive host without threshold limit
                    if not feas_cand:
                        inact = np.where(~active_mask)[0]
                        feas_inact = [j for j in inact[:10] if (v_cpu <= cpu_caps[j] and v_mem <= mem_caps[j])]
                        feas_cand = feas_inact[:5]

                    if not feas_cand:
                        unplaced_vms.append(v_id)
                        continue

                    cand_arr = np.array(feas_cand, dtype=np.int32)

                    # Dynamic heuristic calculation based on instantaneous host state & predicted demand
                    rem_cpu = np.maximum(0.001, cpu_caps[cand_arr] - (cpu_loads[cand_arr] + v_cpu))
                    rem_mem = np.maximum(0.001, mem_caps[cand_arr] - (mem_loads[cand_arr] + v_mem))
                    balance = 1.0 / (1.0 + np.abs(rem_cpu - rem_mem))

                    u_hypo = np.clip((cpu_loads[cand_arr] + v_cpu) / cpu_caps[cand_arr], 0.0, 1.0)
                    # SPECpower power: P_idle + (P_max - P_idle) * (u ** k)
                    p_hypo = idle_powers[cand_arr] + (max_powers[cand_arr] - idle_powers[cand_arr]) * (u_hypo ** k_exp[cand_arr])

                    # Heuristic penalization for projected overload
                    proj_ratio = (pred_cpu_loads[cand_arr] + p_cpu) / cpu_caps[cand_arr]
                    sla_factor = np.where(proj_ratio > base_thresh[cand_arr], 0.35, 1.0)

                    heuristics = (balance / np.maximum(10.0, p_hypo)) * 100.0 * sla_factor

                    pher = pheromone[i, cand_arr]
                    probs = (pher ** self.alpha) * (heuristics ** self.beta)
                    total_prob = probs.sum()

                    if total_prob > 0 and self.rng.rand() > self.q0:
                        probs /= total_prob
                        chosen_local = self.rng.choice(len(cand_arr), p=probs)
                        chosen_j = cand_arr[chosen_local]
                    else:
                        chosen_j = cand_arr[np.argmax(probs)]

                    cpu_loads[chosen_j] += v_cpu
                    mem_loads[chosen_j] += v_mem
                    pred_cpu_loads[chosen_j] += p_cpu
                    if not active_mask[chosen_j]:
                        active_mask[chosen_j] = True
                        active_indices.append(chosen_j)

                    ant_placement[v_id] = pm_ids[chosen_j]

                # Compute normalized fitness for this ant
                act_arr = np.array(active_indices, dtype=np.int32)
                n_act = len(act_arr)
                if n_act > 0:
                    u_c = cpu_loads[act_arr] / cpu_caps[act_arr]
                    u_m = mem_loads[act_arr] / mem_caps[act_arr]
                    # SPECpower: P_idle + (P_max - P_idle) * (u ** k)
                    p_tot = idle_powers[act_arr] + (max_powers[act_arr] - idle_powers[act_arr]) * (u_c ** k_exp[act_arr])
                    total_energy = float(np.sum(p_tot))
                    imbalance = float(np.mean(np.abs(u_c - u_m)))
                    # SLA risk evaluated on projected load
                    u_proj = pred_cpu_loads[act_arr] / cpu_caps[act_arr]
                    sla_risk = float(np.sum(np.maximum(0.0, u_proj - 0.75)))
                else:
                    total_energy, imbalance, sla_risk = 0.0, 0.0, 0.0

                e_norm = total_energy / (n_pms * 135.0)
                a_norm = n_act / n_pms
                u_norm = len(unplaced_vms) / max(1, n_vms)
                fitness = float((0.35 * e_norm) + (0.20 * a_norm) + (0.15 * imbalance) + (0.10 * sla_risk) + (500.0 * u_norm))

                if fitness < iter_best_fitness:
                    iter_best_fitness = fitness
                    iter_best_placement = ant_placement
                    iter_unplaced_count = len(unplaced_vms)

            convergence_curve.append(iter_best_fitness)

            if iter_best_fitness < best_fitness:
                best_fitness = iter_best_fitness
                best_placement = iter_best_placement
                best_unplaced_count = iter_unplaced_count

            # Evaporation
            pheromone = (1.0 - self.rho) * pheromone

            # Pheromone deposit by iteration-best ant
            if iter_best_placement:
                pm_id_to_j = {pm.pm_id: j for j, pm in enumerate(pms)}
                deposit = self.deposit_constant / max(iter_best_fitness, 1.0)
                for i, vm in enumerate(vms):
                    if vm.vm_id in iter_best_placement:
                        j = pm_id_to_j[iter_best_placement[vm.vm_id]]
                        pheromone[i, j] += deposit

            # Enforce pheromone stability bounds
            pheromone = np.clip(pheromone, self.tau_min, self.tau_max)

        placed_count = len(best_placement) if best_placement else 0
        info = {
            "requested_vms": n_vms,
            "placed_vms": placed_count,
            "unplaced_vms": n_vms - placed_count,
            "placement_rate_percent": round((placed_count / max(1, n_vms)) * 100.0, 2)
        }
        return best_placement, best_fitness, convergence_curve, info
