# Implementation Correction and Validation Plan

**Document:** `docs/correction_plan.md`  
**Date:** 2026-11-03  
**Status:** In Progress — Planned Execution  

---

## 1. Summary of Identified Implementation Gaps
Following an in-depth audit of the repository, the following specific implementation gaps and bugs were identified for correction:

1. **Prediction Integration:**
   - In `scripts/run_scheduling.py`, a placeholder dictionary (`pred_dict = {vm.vm_id: (vm.cpu_req * 1.05, vm.mem_req * 1.02)}`) was used rather than actual model predictions from the trained PyTorch checkpoints.
   - *Fix:* Implement `src/vm_scheduler/prediction/inference.py` to load trained LSTM models (`models/checkpoints/lstm_30m.pt`) and generate true empirical predictions for VMs from their observation history.
2. **Dynamic PM Thresholds:**
   - `PhysicalMachine.update_dynamic_threshold()` only inspected a scalar CPU load without multi-dimensional memory consideration or configurable minimum/maximum operating bounds.
   - *Fix:* Refactor threshold update logic to adapt based on predicted multi-resource demand, provide configurable safety bounds, and ensure static ablation remains strictly static at 75%.
3. **ACO Placement Heuristic Updating & Completeness:**
   - Heuristic desirability $\eta_{ij}$ was precomputed once before the VM placement loop. As VMs are assigned, host loads change, but desirability was not dynamically updated.
   - Infeasible placements were not explicitly counted or penalized in the fitness function.
   - *Fix:* Dynamically evaluate heuristic desirability during ant solution construction based on instantaneous host load, clamp pheromones to $[\tau_{\text{min}}, \tau_{\text{max}}]$, and explicitly record and penalize unplaced VMs.
4. **PM & VM Simulation Robustness:**
   - Allocation and release did not check for duplicate VM assignments or validate input demands against negative values.
   - The 25% CPU-intensive diversity constraint was hardcoded rather than being a configurable experimental parameter.
   - *Fix:* Implement allocation integrity checks and expose `enforce_diversity_constraint` as a configurable parameter in `pm_simulation.yaml`.
5. **Unified Energy Modeling:**
   - Power calculations were instantiated separately in various scripts.
   - *Fix:* Centralize on a unified `ServerPowerModel` across simulation, ACO heuristics, migration evaluation, and final metric generation.
6. **Live Migration Economics:**
   - Migration benefit calculation did not account for remaining VM lifetime or separate migration energy overhead from SLA penalties.
   - *Fix:* Refactor `MigrationManager` to calculate transfer time ($T = \text{VM}_{\text{mem}} / B_w$), migration energy overhead (NIC + CPU transfer power), and post-migration energy savings over expected remaining lifetime.
7. **Sensitivity Analysis Bug:**
   - In `scripts/run_ablation.py`, the loop modified `k_nonlinearity` on a cloned PM list, but then invoked `run_configuration(vms, pms)` passing the un-modified original PM list.
   - *Fix:* Correct parameter propagation so each sensitivity run executes with the intended exponent.
8. **Population of Empty Folders & Research Artifacts:**
   - Explain why `data/raw`, `data/interim`, `data/samples`, `experiments/*`, `results/{logs,placements,summaries}` were initially empty and populate them with real samples, experiment logs, JSON placements, and metadata summaries.
   - Add the suggested research figures (Figures 1, 5, 7, 9, 10) and tables (Tables 8, 9, 11, 12).
9. **Automated Test Suite Expansion:**
   - Expand unit and regression tests from 9 to 20+ tests covering prediction inference, dynamic threshold bounds, ACO completeness, migration consistency, and sensitivity parameter propagation.

