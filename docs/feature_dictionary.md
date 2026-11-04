# Feature Dictionary: Telemetry Metrics & Derived Workload Profiles

## 1. Raw Telemetry Features (Extracted from Borg Trace)

| Feature Name | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `instance_id` | String | Unique ID | Identifier of the containerized instance / VM task. |
| `time_step` | Integer | $[0, 2015]$ | Chronological index representing 5-minute sampling intervals over 7 days. |
| `cpu_usage` | Float | $[0.0, 1.0]$ | Mean CPU consumption over interval, normalized to instance limit. |
| `mem_usage` | Float | $[0.0, 1.0]$ | Ratio of memory byte-seconds to allocation limit. |
| `cycles_per_instruction` | Float | $[0.1, 15.0]$ | Hardware CPI performance counter. |
| `memory_accesses_per_instruction` | Float | $[0.0, 0.5]$ | Hardware MAI counter (L3/DRAM traffic). |

## 2. Statistical Aggregations & Profiling Features (VM Level)

To enable unsupervised clustering ($K\text{-Means++}$) across 7,644 instances, statistical features are derived over observed windows:

| Feature Name | Formula / Definition | Semantic Interpretation |
| :--- | :--- | :--- |
| `cpu_mean` | $\mu_{\text{cpu}} = \frac{1}{T}\sum_{t=1}^T u_{\text{cpu}}(t)$ | Baseline compute demand of the instance. |
| `cpu_std` | $\sigma_{\text{cpu}} = \sqrt{\frac{1}{T}\sum_{t=1}^T (u_{\text{cpu}}(t) - \mu_{\text{cpu}})^2}$ | Volatility and unpredictability of compute load. |
| `cpu_p95` | 95th Percentile of $u_{\text{cpu}}$ | Peak operating demand excluding outlier anomalies. |
| `cpu_p99` | 99th Percentile of $u_{\text{cpu}}$ | Safety margin threshold for SLA provisioning. |
| `cpu_peak_to_avg` | $R_{\text{cpu}} = \frac{\max(u_{\text{cpu}})}{\mu_{\text{cpu}} + \epsilon}$ | Burstiness factor. Values $>3.0$ indicate spiky batch workloads. |
| `mem_mean` | $\mu_{\text{mem}} = \frac{1}{T}\sum_{t=1}^T u_{\text{mem}}(t)$ | Persistent RAM footprint requirement. |
| `mem_std` | $\sigma_{\text{mem}} = \sqrt{\frac{1}{T}\sum_{t=1}^T (u_{\text{mem}}(t) - \mu_{\text{mem}})^2}$ | Dynamic memory churn and working set fluctuation. |
| `mem_p95` | 95th Percentile of $u_{\text{mem}}$ | Upper allocation ceiling to prevent Out-Of-Memory (OOM) evictions. |
| `mem_peak_to_avg` | $R_{\text{mem}} = \frac{\max(u_{\text{mem}})}{\mu_{\text{mem}} + \epsilon}$ | Memory expansion ratio. |
| `cpi_mean` | $\frac{1}{T}\sum_{t=1}^T \text{CPI}(t)$ | Instruction-level parallelism and microarchitectural throughput. |
| `mai_mean` | $\frac{1}{T}\sum_{t=1}^T \text{MAI}(t)$ | Memory subsystem bandwidth contention risk. |

## 3. Workload Classes Identified by K-Means++ ($K=6$)

Clustering on the 11-dimensional feature matrix revealed six distinct behavioral archetypes:

1. **Cluster 0: Steady Light Utility Services:** Low CPU ($\sim 0.08$), low memory ($\sim 0.12$), very low variance. Highly consolidatable.
2. **Cluster 1: Heavy Compute-Bound Workers:** High sustained CPU ($>0.65$), moderate memory ($\sim 0.35$), low CPI ($<1.2$). Placed on HP ProLiant G5 high-performance nodes.
3. **Cluster 2: Memory-Intensive In-Memory Stores:** Moderate CPU ($\sim 0.25$), high persistent memory ($>0.70$), high MAI ($>0.08$). Co-located with low-memory CPU workers to balance host resource dimensions.
4. **Cluster 3: Bursty Periodic Analytics:** Low average load ($\sim 0.15$) but extreme peak-to-average ratio ($>4.5$). Dynamic thresholds are lowered automatically on hosts accommodating these VMs to buffer anticipated bursts.
5. **Cluster 4: High-Throughput Web Frontends:** Balanced CPU ($\sim 0.40$) and memory ($\sim 0.45$), medium volatility, moderate CPI.
6. **Cluster 5: Idle / Zombie Daemons:** Negligible resource consumption ($<0.02$). Prime candidates for aggressive packing onto consolidated standby servers.

