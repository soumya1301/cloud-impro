# Google Cluster Data 2019: Dataset & Telemetry Specification

## 1. Trace Overview & Provenance
- **Dataset:** Google Cluster Data (2019 release, Borg trace).
- **Format:** Partitioned compressed columnar format (`parquet.gz` in cloud storage, decompressed locally to Apache Parquet).
- **Cells Covered:** 8 distinct data center cells: `clusterdata_2019_a` through `clusterdata_2019_h`.
- **Primary Telemetry Table:** `instance_usage` (sampled at 5-minute / 300-second reporting intervals).
- **Machine Metadata Table:** `machine_events` (hardware capacity, platform architecture, lifecycle timestamps).

## 2. Resource Dimension Verification & Audit Findings
Following the comprehensive read-only audit of 120,298 objects (10.28 TiB):

| Resource Dimension | Presence in 2019 Trace | Exact Field Name | Missingness (%) | Physical Realism & Modeling |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Demand** | **Present** | `average_usage.cpus` | 0.00% | Primary resource dimension. Measured in Google Compute Units (GCUs), normalized to host capacity. |
| **Memory Demand** | **Present** | `average_usage.memory` | 0.00% | Primary resource dimension. Measured as byte-seconds / interval normalized to host RAM capacity. |
| **Cycles Per Instruction (CPI)**| **Present** | `cycles_per_instruction` | 0.00% | Hardware performance counter measuring compute intensity and pipeline stall characteristics. |
| **Memory Accesses / Inst (MAI)**| **Present** | `memory_accesses_per_instruction` | 0.00% | Hardware performance counter measuring cache miss rates and memory bus pressure. |
| **Network Bandwidth (I/O)** | **Omitted** | N/A | 100.00% | Omitted by Google in the 2019 public release. Modeled deterministically using enterprise top-of-rack link capacity ($B_w = 4.0\text{ Gbps}$, standard 10GbE fabric) for live migration simulation. |
| **Disk Bandwidth (I/O)** | **Omitted** | N/A | 100.00% | Omitted by Google in the 2019 public release. VM storage modeled under shared distributed storage architecture (Colossus/Borg SAN). |

## 3. Experimental Subsets

### Subset A: Telemetry & Forecasting Benchmark (`cell_a`)
- **Source:** `gs://clusterdata_2019_a/instance_usage/`
- **Time Window:** 7 consecutive calendar days (2,016 chronological 5-minute timestamps).
- **VM Count:** 1,200 continuous long-running virtual instances with complete time series.
- **Data Splits:**
  - Training Set: First 4 days (57.1%, timestamps 0 to 1,152).
  - Validation Set: Day 5 (14.3%, timestamps 1,152 to 1,440).
  - Test Set: Days 6-7 (28.6%, timestamps 1,440 to 2,016, 576 unseen steps).
- **Forecasting Horizons:** Multi-step rolling windows of 10 minutes ($H=2$), 30 minutes ($H=6$), and 60 minutes ($H=12$).

### Subset B: Cluster Scheduling & Placement Benchmark (`cell_b`)
- **Source:** `gs://clusterdata_2019_b/instance_usage/` and `machine_events`
- **Simulation Scale:** 7,644 concurrently arriving/running VMs scheduled across 798 heterogeneous physical machines.
- **Physical Server Types:**
  - HP ProLiant G5: 16 cores, 32 GB RAM, 10GbE network ($P_{\text{idle}} = 93.7\text{ W}$, $P_{\text{max}} = 222\text{ W}$, $k = 1.4$).
  - HP ProLiant G4: 8 cores, 16 GB RAM, 1GbE network ($P_{\text{idle}} = 86.0\text{ W}$, $P_{\text{max}} = 180\text{ W}$, $k = 1.4$).
- **Arrival Profiles:** Realistic Poisson arrival process matching Borg trace arrival rate distributions.

