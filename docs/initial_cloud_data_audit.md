# Initial Cloud Data and Environment Audit Report

**Project Title:** LSTM-Assisted, K-Means++-Driven and Ant Colony Optimization-Based Energy-Aware VM Scheduling  
**Document:** `docs/initial_cloud_data_audit.md`  
**Execution Timestamp:** 2026-11-03T08:45:00+05:30  
**Audit Status:** Complete — Read-Only Verification Satisfied  
**Auditor:** Antigravity AI Research Agent  

---

## Executive Summary

A comprehensive, non-destructive, read-only audit was conducted on the Google Cloud storage environment, dataset inventory manifests, and existing Colab preprocessing pipelines (Worker A and Worker B) for the research project.

Key findings:
1. **Authentication & Environment:** Active account `soumyaprasad0902@gmail.com` and project `project-192d863e-4e9a-4a5d-96f` are authenticated via Application Default Credentials (ADC) and Google Cloud CLI.
2. **Bucket Accessibility:** All eight source buckets (`gs://clusterdata_2019_a/` through `gs://clusterdata_2019_h/`) and the processed-data bucket (`gs://clusterdata-2019-processed-1002024056565/`) are 100% accessible with zero read failures.
3. **Dataset Scale:** The full Google Cluster Data 2019 inventory comprises **120,298 objects** totaling **10,523.15 GiB (~10.28 TiB)** across both `parquet.gz` (106,726 objects, 3,328.80 GiB / 3.25 TiB) and `json.gz` (13,572 objects, 7,194.35 GiB / 7.03 TiB).
4. **Existing Preprocessing Progress:**
   - Worker A and Worker B have completed **8,798 shards** (8.24% of the 106,726 Parquet shards), generating **8,764,239,878 rows (8.76 Billion rows)** of cleaned analytical Parquet data.
   - Every single one of the 8,798 checkpoints strictly matches an existing output object (0 missing outputs, 0 orphaned outputs, 0 duplicate shards).
   - Zero overlap exists between Worker A (cells `a–d`) and Worker B (cells `e–h`).
5. **Critical Resource Dimension Findings:**
   - **CPU & RAM:** Fully verified across `instance_usage`, `instance_events`, and `machine_events`. Normalized Google Compute Units (GCUs) and memory ratios are well-populated (0.00% null for primary usage metrics).
   - **Network & Disk I/O:** **Not present in Google Cluster Data 2019 trace schemas.** The source tables contain no network bandwidth/packet counters or disk I/O metrics. In accordance with the project guidelines, these cannot be invented or misrepresented; fallback experimental protocols and simulated trace enhancements are proposed.

---

## 1. Authentication and Antigravity Environment

### 1.1 Environment Details
- **Host OS:** macOS (Darwin 24.3.0, arm64)
- **Development Tool:** Google Antigravity IDE (integrated Zsh environment)
- **Active Workspace:** `/Users/soumyaprasad/Desktop/cloud improvements`
- **Python Environment:** Python 3.12.7 (Anaconda distribution, `/opt/anaconda3/bin/python3`)
- **Key Installed Libraries:** PyArrow 17.0.0, Pandas 2.2.2, PyTorch 2.4.1, Scikit-learn 1.5.1, NumPy 1.26.4
- **Local Dependency Note:** The standalone `google-cloud-storage` Python package is currently not present in the base conda environment, but Google Cloud CLI (`gcloud storage`) is fully configured and operational with active ADC.

### 1.2 Google Cloud Credentials Verification
The following read-only commands verified the authenticated identity and project:
```bash
$ gcloud auth list
ACTIVE  ACCOUNT
*       soumyaprasad0902@gmail.com

$ gcloud config get-value project
project-192d863e-4e9a-4a5d-96f
```
No tokens, passwords, or service account keys were printed, exposed, or modified.

---

## 2. Bucket Access Audit

All eight source buckets and the processed-data bucket were queried individually:

| Bucket URI | Region / Class | Read Access Status | Objects Detected |
| :--- | :--- | :---: | :--- |
| `gs://clusterdata_2019_a/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_b/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_c/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_d/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_e/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_f/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_g/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata_2019_h/` | US Multi-region (Standard) | **ACCESS OK** | Verified collection, instance & machine tables |
| `gs://clusterdata-2019-processed-1002024056565/` | US Multi-region (Standard) | **ACCESS OK** | `_access_test/`, `manifest/`, `worker_a/`, `worker_b/` |

**Result:** Zero access failures. All buckets are fully readable using the existing credentials.

---

## 3. Dataset Inventory and Manifest Analysis

Analysis based on `gs://clusterdata-2019-processed-1002024056565/manifest/`:
- `google_cluster_2019_inventory_summary.csv` (4,122 bytes)
- `google_cluster_2019_full_manifest.csv` (15,983,796 bytes, 120,298 rows)

### 3.1 Inventory Summary by Format
The dataset exists in two formats: `parquet.gz` (primary analytical format) and `json.gz` (raw trace export).
- The exact format string in the manifest is **`parquet.gz`** (not `parquet`).

| Format | Shard Count | Total Size (GiB) | Total Size (TiB) | Share of Data |
| :--- | :---: | :---: | :---: | :---: |
| `parquet.gz` | 106,726 | 3,328.80 GiB | 3.25 TiB | 31.6% (Compressed Columnar) |
| `json.gz` | 13,572 | 7,194.35 GiB | 7.03 TiB | 68.4% (Raw Text Export) |
| **Total Inventory** | **120,298** | **10,523.15 GiB** | **10.28 TiB** | **100.0%** |

### 3.2 Table Inventory Breakdown (Parquet Objects Only)

| Table Name | Parquet Shards | Total Size (GiB) | Description & Role |
| :--- | :---: | :---: | :--- |
| `instance_usage` | 92,468 | 3,161.81 GiB | Periodic 5-minute resource usage records (CPU, RAM, CPI, MAI) |
| `instance_events` | 14,126 | 163.65 GiB | VM/Task lifecycle events (SUBMIT, SCHEDULE, EVICT, KILL, FINISH) |
| `collection_events` | 116 | 3.28 GiB | Job/Allocation group lifecycle events & scheduling priority |
| `machine_attributes` | 8 | 0.06 GiB | Host key-value pairs (kernel, rack, features) — 1 shard per cell |
| `machine_events` | 8 | 0.003 GiB | Host physical capacities (CPUs, Memory) & lifecycle events — 1 shard per cell |
| **Total Parquet** | **106,726** | **3,328.80 GiB** | |

### 3.3 Inventory Breakdown by Source Cell (Parquet Objects Only)

| Cell | Total Parquet Shards | Total Parquet Size (GiB) | Worker Assigned |
| :---: | :---: | :---: | :---: |
| **a** | 13,830 | 428.07 GiB | Worker A |
| **b** | 12,868 | 403.11 GiB | Worker A |
| **c** | 16,032 | 492.62 GiB | Worker A |
| **d** | 14,378 | 447.03 GiB | Worker A |
| **e** | 11,313 | 356.10 GiB | Worker B |
| **f** | 13,446 | 429.88 GiB | Worker B |
| **g** | 12,451 | 377.57 GiB | Worker B |
| **h** | 12,408 | 394.41 GiB | Worker B |
| **Total** | **106,726** | **3,328.80 GiB** | |

---

## 4. Existing Preprocessing: Worker A and Worker B Status

The research pipeline previously deployed two Colab preprocessing workers:
- **Worker A:** Partitioned to process source cells `a`, `b`, `c`, `d`.
- **Worker B:** Partitioned to process source cells `e`, `f`, `g`, `h`.

Output and checkpoint paths audited:
```text
gs://clusterdata-2019-processed-1002024056565/worker_a/phase1/
gs://clusterdata-2019-processed-1002024056565/worker_a/phase1_checkpoints/
gs://clusterdata-2019-processed-1002024056565/worker_b/phase1/
gs://clusterdata-2019-processed-1002024056565/worker_b/phase1_checkpoints/
```

Auxiliary directories inspected:
- `worker_a/pilot/` & `worker_a/pilot_checkpoints/`: 3 pilot shards processed during pipeline development.
- `worker_b/test/`: 1 test shard (`machine_events_cell_h_shard_000000000000.parquet`).

### 4.1 Checkpoint Audit & Output Verification
Every individual checkpoint JSON file from both workers was downloaded and validated:
- Total checkpoint files in `worker_a/phase1_checkpoints/`: **4,232**
- Total checkpoint files in `worker_b/phase1_checkpoints/`: **4,566**
- Total valid JSON checkpoints: **8,798** (0 corrupted JSON files).
- Status field: **100% "completed"** (8,798 / 8,798).
- Verified matching output parquet files: **8,798 / 8,798 (100.0%)**.
- Missing or orphaned output files: **0**.

### 4.2 Overlap and Concurrency Check
- Worker A processed shards: strictly from cells `a`, `b`, `c`, and `d`.
- Worker B processed shards: strictly from cells `e`, `f`, `g`, and `h`.
- **Overlap:** Exactly **0** shared shard IDs or conflicting outputs between Worker A and Worker B.
- **Duplicate records:** Exactly **0** duplicate `(cell, table, shard)` tuples within Worker A, within Worker B, or across workers.
- **Worker activity:** The latest checkpoint timestamps date to `2026-11-02T06:42Z` (Worker A) and `2026-11-02T06:56Z` (Worker B), indicating neither worker is currently running an active uncontrolled loop.

---

## 5. Detailed Breakdown: Completed vs Remaining Shards

The calculation method compares the unique completed checkpoints against the total Parquet (`parquet.gz`) shards listed in `manifest/google_cluster_2019_full_manifest.csv`.

### 5.1 Comprehensive Table-by-Cell Breakdown

| Worker | Cell | Table | Assigned Shards | Completed Shards | Remaining Shards | Rows Written | Elapsed (s) | % Completed |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Worker A** | a | `collection_events` | 25 | 25 | 0 | 20,807,444 | 170.7 | **100.00%** |
| Worker A | a | `instance_events` | 1,718 | 1,129 | 589 | 1,127,911,178 | 7,422.5 | 65.72% |
| Worker A | a | `instance_usage` | 12,085 | 9 | 12,076 | 5,617,251 | 65.7 | 0.07% |
| Worker A | a | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker A | a | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker A** | b | `collection_events` | 11 | 11 | 0 | 7,975,436 | 73.7 | **100.00%** |
| Worker A | b | `instance_events` | 1,427 | 887 | 540 | 886,486,346 | 5,442.5 | 62.16% |
| Worker A | b | `instance_usage` | 11,428 | 0 | 11,428 | 0 | 0.0 | 0.00% |
| Worker A | b | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker A | b | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker A** | c | `collection_events` | 36 | 36 | 0 | 30,860,774 | 247.2 | **100.00%** |
| Worker A | c | `instance_events` | 3,414 | 0 | 3,414 | 0 | 0.0 | 0.00% |
| Worker A | c | `instance_usage` | 12,580 | 0 | 12,580 | 0 | 0.0 | 0.00% |
| Worker A | c | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker A | c | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker A** | d | `collection_events` | 12 | 12 | 0 | 11,242,818 | 83.2 | **100.00%** |
| **Worker A** | d | `instance_events` | 2,123 | 2,123 | 0 | 2,122,287,970 | 15,091.2 | **100.00%** |
| Worker A | d | `instance_usage` | 12,241 | 0 | 12,241 | 0 | 0.0 | 0.00% |
| Worker A | d | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker A | d | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker B** | e | `collection_events` | 6 | 6 | 0 | 5,244,061 | 60.1 | **100.00%** |
| Worker B | e | `instance_events` | 1,524 | 1,359 | 165 | 1,358,248,891 | 8,602.2 | 89.17% |
| Worker B | e | `instance_usage` | 9,782 | 0 | 9,782 | 0 | 0.0 | 0.00% |
| Worker B | e | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker B | e | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker B** | f | `collection_events` | 6 | 6 | 0 | 5,117,269 | 56.9 | **100.00%** |
| Worker B | f | `instance_events` | 1,180 | 875 | 305 | 874,056,238 | 4,652.8 | 74.15% |
| Worker B | f | `instance_usage` | 12,259 | 0 | 12,259 | 0 | 0.0 | 0.00% |
| Worker B | f | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker B | f | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker B** | g | `collection_events` | 6 | 6 | 0 | 4,096,710 | 58.0 | **100.00%** |
| Worker B | g | `instance_events` | 1,530 | 1,071 | 459 | 1,069,881,399 | 6,543.2 | 70.00% |
| Worker B | g | `instance_usage` | 10,914 | 19 | 10,895 | 11,662,559 | 192.9 | 0.17% |
| Worker B | g | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker B | g | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| **Worker B** | h | `collection_events` | 14 | 14 | 0 | 13,141,659 | 127.9 | **100.00%** |
| **Worker B** | h | `instance_events` | 1,210 | 1,210 | 0 | 1,209,603,097 | 9,034.5 | **100.00%** |
| Worker B | h | `instance_usage` | 11,179 | 0 | 11,179 | 0 | 0.0 | 0.00% |
| Worker B | h | `machine_attributes` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |
| Worker B | h | `machine_events` | 1 | 0 | 1 | 0 | 0.0 | 0.00% |

### 5.2 Worker Totals Summary

| Worker | Assigned Parquet Shards | Completed Shards | Remaining Shards | Completed % | Total Rows Written | Compute Time Logged |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Worker A (a–d)** | 57,108 | 4,232 | 52,876 | 7.41% | 4,186,634,013 | 7.94 hours |
| **Worker B (e–h)** | 49,618 | 4,566 | 45,052 | 9.20% | 4,577,605,865 | 8.15 hours |
| **Grand Total** | **106,726** | **8,798** | **97,928** | **8.24%** | **8,764,239,878** | **16.09 hours** |

### 5.3 Key Observations on Preprocessing Status
1. **`collection_events` is 100% COMPLETE:** All 116 shards across all 8 cells (`a` through `h`) have been fully processed and written to Parquet.
2. **`instance_events` is 61.26% COMPLETE:** 8,654 of 14,126 shards are completed.
   - **Cell `d`** (Worker A) is **100% complete** (2,123 / 2,123 shards).
   - **Cell `h`** (Worker B) is **100% complete** (1,210 / 1,210 shards).
   - Other cells range from 62.16% to 89.17%, with only Cell `c` at 0%.
3. **`instance_usage` is 99.97% UNPROCESSED:** Only 28 shards (9 in cell `a`, 19 in cell `g`) have been processed. The remaining 92,440 shards remain in the source buckets as `parquet.gz`.
4. **`machine_events` & `machine_attributes`:** None of the 8 shards in Phase 1 directories have been processed by the workers, though Worker B wrote 1 test shard (`machine_events_cell_h_shard_000000000000.parquet`). These files are small (under 10 MB each) and can be accessed or ingested directly in seconds.

---

## 6. Data Integrity and Checkpoint Anomalies

1. **Output Existence:** 100% verification rate. All 8,798 checkpoint records point to valid Parquet output objects in Cloud Storage.
2. **Row Count Verification:**
   - 8,786 checkpoints record exact equality between `source_rows` and `rows_written`.
   - Exactly **12 checkpoints** (6 in Worker A, 6 in Worker B) recorded `"row_groups": 1` instead of `"source_rows"`, resulting in a `NaN` in automated comparisons. However, their `rows_written` field is populated with valid positive integers and `status: "completed"`.
   - The 12 affected shards are:
     * Worker A: `b/instance_events/1376`, `b/collection_events/0009`, `a/instance_usage/0170`, `b/collection_events/0000`, `a/instance_usage/5740`, `b/instance_events/1031`.
     * Worker B: `h/instance_events/0187`, `h/collection_events/0006`, `h/instance_events/0856`, `h/collection_events/0007`, `g/instance_usage/7462`, `g/instance_usage/2525`.
3. **No Duplicate Processing:** No shard was processed more than once within either worker or across workers.

---

## 7. Schema and Resource-Field Availability Matrix

Direct inspection of actual Parquet files across all five tables established the following resource availability.

### 7.1 Verified Table Schemas

```text
Table: collection_events (832,410 rows/shard)
├── time: int64 (microseconds since trace start)
├── type: int64 (0: SUBMIT, 1: QUEUE, 2: ENABLE, ...)
├── collection_id: int64
├── scheduling_class: int64 [0-3]
├── missing_type: int64
├── collection_type: int64 (0: Job, 1: Alloc)
├── priority: int64 [0-450]
├── alloc_collection_id: int64
├── user: string (salted hash)
├── collection_name: string (salted hash)
├── collection_logical_name: string (salted hash)
├── parent_collection_id: int64
├── start_after_collection_ids: list<int64>
├── max_per_machine: int64
├── max_per_switch: int64
├── vertical_scaling: int64
└── scheduler: int64

Table: instance_events (998,903 rows/shard)
├── time: int64 (microseconds since trace start)
├── type: int64 (0: SUBMIT, 1: QUEUE, 2: ENABLE, 3: SCHEDULE, 4: EVICT, 5: FAIL, 6: FINISH, 7: KILL, 8: LOST, 9: UPDATE_PENDING, 10: UPDATE_RUNNING)
├── collection_id: int64
├── instance_index: int64
├── machine_id: int64
├── alloc_collection_id: int64
├── alloc_instance_index: int64
├── scheduling_class: int64
├── priority: int64
├── resource_request: struct
│   ├── cpus: double (normalized GCUs)
│   └── memory: double (normalized memory ratio)
└── constraint: list<struct<name: string, value: string, relation: int64>>

Table: instance_usage (624,087 rows/shard)
├── start_time: int64 (microseconds)
├── end_time: int64 (microseconds; typical window = 300s = 5 min)
├── collection_id: int64
├── instance_index: int64
├── machine_id: int64
├── alloc_collection_id: int64
├── alloc_instance_index: int64
├── collection_type: int64
├── average_usage: struct
│   ├── cpus: double (rate of CPU-core seconds per second)
│   └── memory: double (fraction of machine memory)
├── maximum_usage: struct
│   ├── cpus: double
│   └── memory: double
├── random_sample_usage: struct
│   ├── cpus: double
│   └── memory: double
├── assigned_memory: double
├── page_cache_memory: double
├── cycles_per_instruction: double (CPI)
├── memory_accesses_per_instruction: double (MAI)
├── sample_rate: double
├── cpu_usage_distribution: list<double> (11 percentiles: min, 10th..90th, max)
└── tail_cpu_usage_distribution: list<double> (9 percentiles: 91st..99th)

Table: machine_events (46,374 rows/shard)
├── time: int64
├── machine_id: int64
├── type: int64 (0: ADD, 1: REMOVE, 2: UPDATE)
├── switch_id: string (salted hash)
├── capacity: struct
│   ├── cpus: double (normalized capacity: 1.0, 0.59, 0.71, etc.)
│   └── memory: double (normalized capacity: 0.5, 0.33, 1.0, 0.25, etc.)
├── platform_id: string (salted hash)
└── missing_data_reason: int64

Table: machine_attributes (1,702,926 rows/shard)
├── time: int64
├── machine_id: int64
├── name: string (salted hash)
├── value: string (salted hash)
└── deleted: bool
```

### 7.2 Resource-Field Availability Matrix (Mandatory Deliverable)

| Resource Dimension | Candidate Source Fields | Actual Field Verified? | Storage Unit / Representation | Missingness in Sample | Usable for Research? | Notes & Ground Truth |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **CPU** | `instance_usage.average_usage.cpus`<br>`instance_usage.maximum_usage.cpus`<br>`instance_usage.cpu_usage_distribution`<br>`instance_usage.tail_cpu_usage_distribution`<br>`instance_events.resource_request.cpus`<br>`machine_events.capacity.cpus` | **YES** | CPU-core s/s normalized to GCU (0.0 to 1.0+) | **0.00%** | **YES (Primary)** | Direct continuous signal. 11 distribution percentiles available per 5-min window. |
| **RAM** | `instance_usage.average_usage.memory`<br>`instance_usage.maximum_usage.memory`<br>`instance_usage.assigned_memory`<br>`instance_usage.page_cache_memory`<br>`instance_events.resource_request.memory`<br>`machine_events.capacity.memory` | **YES** | Normalized ratio to host maximum memory | **0.00%** | **YES (Primary)** | Direct continuous signal. Assigned and page cache memory available. |
| **Network** | `network_utilization`<br>`network_bandwidth`<br>`rx_bytes` / `tx_bytes`<br>`network_usage` | **NO** | *None* | **100.0% (Field does not exist in schema)** | **NO (Trace Limitation)** | Google omitted network I/O from the 2019 public release. Must not be fabricated. |
| **Disk I/O** | `disk_io_rate`<br>`disk_read_rate`<br>`disk_write_rate`<br>`disk_usage` | **NO** | *None* | **100.0% (Field does not exist in schema)** | **NO (Trace Limitation)** | Google omitted disk I/O from the 2019 public release. Must not be fabricated. |
| **Microarchitectural** | `instance_usage.cycles_per_instruction` (CPI)<br>`instance_usage.memory_accesses_per_instruction` (MAI) | **YES** | Ratio / dimensionless | **18.51%** | **YES (Auxiliary Feature)** | Matches the 16 features extracted in the reference paper (Alelyani et al. 2026). |

---

## 8. Limitations and Unresolved Questions

1. **Absence of Network and Disk I/O Dimensions:**
   - The master prompt envisioned a 4-dimensional resource vector $[CPU, RAM, NET, DISK]$.
   - However, schema verification proves conclusively that Google Cluster Data 2019 **does not provide network or disk I/O metrics**.
   - The reference paper (Alelyani et al. 2026) itself faced this exact constraint and extracted **16 features spanning CPU, Memory, Instruction metrics (CPI, MAI), and Temporal windows**, omitting network and disk utilization from the trace.
   - *Resolution:* In accordance with the guidelines ("Never invent measurements, infer unavailable values without justification, or represent simulated values as actual Google measurements"), we must:
     a) Implement the primary trace-driven model using the verified empirical dimensions: **CPU and Memory** (augmented by microarchitectural features CPI and MAI).
     b) Provide an explicit, clearly documented **4-dimensional synthetic/simulated extension benchmark** (or network migration model based on VM memory and network link capacity $B_w = 4\text{ Gbps}$ as described in Equation 5 of Alelyani et al.) to validate the ACO placement multi-objective framework without misrepresenting simulated values as Google measurements.
2. **Bulk Processing vs. Targeted Experimental Extraction:**
   - The full Parquet dataset contains 92,468 `instance_usage` shards totaling 3.16 TiB.
   - Bulk preprocessing of all 92,468 shards would require over 180 compute hours and significant bandwidth/cost, which is unnecessary for training the LSTM and running the scheduling simulation.
   - The reference study used **2,293 VMs** for ML training and **7,644 VMs across 798 PMs** for scheduling evaluation.
   - A targeted extraction strategy filtering specific cells (e.g., Cell `d` or `h` where `instance_events` is already 100% complete) is substantially faster, zero-risk, and cost-efficient.
3. **Local Environment Dependencies:**
   - Python library `google-cloud-storage` is absent from the Anaconda base environment.
   - For subsequent Python extraction scripts, installing `google-cloud-storage` or utilizing PyArrow's built-in GCS filesystem interface via ADC is recommended.

---

## 9. Recommended Next Research Steps

To transition safely from the completed read-only audit to the research implementation:

1. **Approve Audit Findings and Dimensional Strategy:**
   - Formally accept the 2-dimensional empirical ground truth (CPU + RAM + CPI/MAI) for trace evaluation, complemented by the 4-dimensional placement formulation with network migration modeling ($B_w$ bandwidth constraint) as defined in Alelyani et al. (2026).
2. **Targeted Extraction of Experimental Subsets:**
   - Target Cell `d` or Cell `h` (where `instance_events` is 100% preprocessed and `collection_events` is 100% preprocessed).
   - Extract a self-contained, reproducible experimental dataset:
     - **Subset A (ML Prediction & Profiling):** ~2,500–5,000 distinct VM instance timeseries (5-minute windows over a 24-hour observation horizon).
     - **Subset B (Scheduling Simulation):** ~7,500–10,000 VM arrival and lifecycle events mapped to a cluster of ~800 PMs.
3. **Preserve Running Colab Workers:**
   - Keep Worker A and Worker B storage artifacts intact. Do not delete, overwrite, or interfere with them.
4. **Scaffold the Modular Repository Structure:**
   - Establish the repository directories and configuration templates specified in Master Prompt Part E.
5. **Proceed Sequentially through Phases 1–9:**
   - Phase 1: Experimental subset extraction & feature dictionary.
   - Phase 2: Feature engineering & sliding-window datasets.
   - Phase 3: Multivariate LSTM prediction (10, 30, 60 min) with baseline comparisons.
   - Phase 4 & 5: K-Means++ clustering ($K \in \{3,4,5,6\}$) and workload profiling.
   - Phase 6 & 7: Heterogeneous PM simulation and nonlinear SPECpower energy model.
   - Phase 8: Multi-objective Ant Colony Optimization (ACO) placement.
   - Phase 9: Migration cost-benefit decision engine, baseline benchmark comparisons (MFFA, MBFA, Sercon, RR), ablation studies, and paper manuscript generation.

---
*End of docs/initial_cloud_data_audit.md*

