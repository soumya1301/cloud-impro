# Methodological Assumptions, Engineering Bounds, and Research Limitations

## 1. Trace Telemetry Scope & Synthetic Complements
1. **Absence of Disk and Network Bandwidth in Borg 2019:**
   - *Trace Limitation:* As documented in Google's official release and confirmed during our read-only audit of 120,298 objects across cells `a` through `h`, Google Cluster Data 2019 omits network I/O and disk I/O metrics to prevent side-channel inference of proprietary data center topologies.
   - *Defensible Methodology:* All workload characterization and clustering are grounded exclusively on empirical trace fields: CPU usage, memory usage, CPI, and MAI. Network bandwidth during live migration is parameterized using top-of-rack enterprise link capacity ($B_w = 4.0\text{ Gbps}$ / $0.5\text{ GB/s}$) based on standard cloud data center networking specifications.

2. **SPECpower Physical Calibration vs. Simulation:**
   - *Hardware Alignment:* Power figures are calculated using empirical SPECpower benchmarks on HP ProLiant G5 (Dual Quad-Core Intel Xeon E5430) and HP ProLiant G4 servers, rather than arbitrary linear approximations.
   - *Dynamic Exponent:* Server power consumption is modeled via the nonlinear SPECpower formulation $P(u) = P_{\text{idle}} + (P_{\text{max}} - P_{\text{idle}}) \cdot (2u - u^k)$, where $k = 1.4$, capturing real-world dynamic voltage and frequency scaling (DVFS) curve concavity.

## 2. Dynamic Threshold Mechanics & Safety Bounds
1. **Dynamic Soft Threshold vs. Physical Hard Capacity:**
   - The scheduler strictly distinguishes between the *soft operating threshold* ($\theta_{\text{dyn}} \in [0.60, 0.90]$) used to trigger consolidation and prevent SLA violations, and the *hard physical server capacity* ($C = 1.0$), which is never violated under any circumstances.
   - The adaptive threshold formula incorporates predicted future demand:
     $$\theta_{\text{dyn}} = \theta_{\text{base}} + \Delta_{\text{adapt}} - \text{Margin}_{\text{safety}}$$
     where $\theta_{\text{base}} = 0.75$, and $\Delta_{\text{adapt}}$ contracts under anticipated high burstiness to provide headroom.

2. **Infeasible Placement Handling & Completeness:**
   - To guarantee physical realism, VMs are never silently dropped or discarded. The ant colony optimization objective penalizes unplaced VMs heavily with a normalized penalty factor ($W_{\text{unplaced}} = 500.0$), forcing ants to prioritize complete valid placement ($100\%$ placement rate) before optimizing secondary energy objectives.

## 3. Live Migration Overhead & Net-Benefit Invariant
1. **Migration Cost Accounting:**
   - Live migration is not free: each migration transfers the active memory state of the VM ($D = \text{RAM}_{\text{GB}} \cdot 8\text{ Gb}$), incurring a transmission duration $T_{\text{mig}} = \frac{D}{B_w \cdot 0.85}$ and consuming electrical energy $E_{\text{mig}} = P_{\text{mig\_overhead}} \cdot T_{\text{mig}}$ ($P_{\text{overhead}} \approx 0.10 \cdot P_{\text{max}}$).
2. **Net Benefit Gating:**
   - A candidate migration from source PM $S$ to target PM $T$ is executed *if and only if* the cumulative lifetime energy savings exceed the migration energy overhead plus an operational safety threshold:
     $$\Delta E_{\text{lifetime}} - E_{\text{mig}} \ge \delta_{\text{min}} \quad (\delta_{\text{min}} = 0.5\text{ Wh})$$
   - This prevents churn and thrashing caused by marginal migrations.

