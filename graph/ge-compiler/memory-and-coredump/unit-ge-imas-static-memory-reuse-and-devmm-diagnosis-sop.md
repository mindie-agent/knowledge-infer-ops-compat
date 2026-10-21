---
schema: mindie-graded-unit/1
id: unit-ge-imas-static-memory-reuse-and-devmm-diagnosis-sop
title: Ascend GE IMAS Graph Memory Reuse Ratio, Topological Liveness, and DevMM Allocation
  Diagnosis SOP
domain_path: graph/ge-compiler/memory-and-coredump
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: ge
  diagnosis_domain: imas-memory-and-devmm
  runtime:
  - ge
  - torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
supersedes: []
triggers:
  error_signatures:
  - IMAS reach_theory_rate below threshold
  - theory_min_memory_size exceeds HBM budget
  - DEVMM_MAX_ALLOW_ALLOC_SIZE physical page exhaustion
  - GraphScopeMem SessionScopeMem OOM during GE Build
  keywords:
  - IMAS memory solver
  - reach_theory_rate ratio formula
  - theory_min_memory_size
  - no_reuse_size
  - svm_mem_stats_show devmm table
  - GraphScopeMem
  - topological liveness interval
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/ge-memory-analysis
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-memory-analysis/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-memory-analysis/SKILL.md
  source_type: curated_repo
  source_digest: cd5fb79328f05064447d882b14dd10881def5ac0fbb1020e566ea99d478666d0
---

# Ascend GE IMAS Graph Memory Reuse Ratio, Topological Liveness, and DevMM Allocation Diagnosis SOP

## 1. Three-Layer Ascend GE Memory Architecture

When a GE graph fails with `EL0004` / `207001` device memory exhaustion during `Build` or execution, diagnose across three layers:

1. **Driver SVM / DevMM Layer (`svm_mem_stats_show`)**:
   - Tracks physical HBM page allocation (`DEVMM_MAX_ALLOW_ALLOC_SIZE`), hugepage vs normal page fragmentation, and external allocations outside GE (PyTorch `NPUCachingAllocator`, HCCL communicator buffers `HCCL_BUFFSIZE`, and KV cache pools).
2. **GE Runtime Memory Pools (`SessionScopeMem`, `GraphScopeMem`, `VarMem`)**:
   - `VarMem` holds persistent weights/parameters (`CONSTANT` / `VARIABLE` nodes).
   - `GraphScopeMem` holds static feature-map activation and operator workspace memory solved per graph.
3. **IMAS Static Offset Solver (Intra-graph Memory Allocation Solver)**:
   - Computes static HBM byte offsets for every intermediate tensor and workspace based on topological execution order and stream/event liveness intervals.

## 2. Quantitative IMAS Metrics Triage Matrix

Extract the IMAS summary lines from CANN plog (`grep -E "total_malloc_size|theory_min_memory_size|reach_theory_rate|no_reuse_size" ~/ascend/log/debug/plog/*`):

$$
\text{reach\_theory\_rate} = \frac{\text{theory\_min\_memory\_size}}{\text{total\_malloc\_size}}, \qquad
\text{reuse\_rate} = 1 - \frac{\text{total\_malloc\_size}}{\text{no\_reuse\_size}}
$$

| Diagnostic Condition | Meaning | Root Cause & Remediation SOP |
|---|---|---|
| `theory_min_memory_size` itself exceeds available HBM | Peak simultaneous live tensor volume at the worst topological cut is too high | **(1)** Long-range residual/skip connections or prefetched weights kept live across the entire graph (`first_use` near node 0, `last_use` near final node); **(2)** Parallel multi-stream branches whose lifespans overlap completely without event serialization; **(3)** Excessive operator `atomic_clean` workspace sizes. Tighten topological liveness by delaying producer execution closer to consumer or bounding batch/sequence gear. |
| `theory_min_memory_size` fits HBM, but `reach_theory_rate < 0.70` (`total_malloc_size` bloated) | IMAS static offset solver suffers severe block fragmentation | Large and small tensors alternate in allocation/free order, splitting contiguous virtual blocks. Enable memory defragmentation / topological memory priority sorting or share static memory pools across multi-gear graphs (`GE_USE_STATIC_MEMORY=1` / `ge.exec.memoryOptimizationPolicy`). |
| Both `total_malloc_size` and `VarMem` are small, yet `svm_mem_stats_show` fails | HBM is consumed outside GE `GraphScopeMem` | Audit PyTorch eager `NPUCachingAllocator` reserved pool (`torch.npu.memory_summary()`), HCCL communicator count $\times$ `HCCL_BUFFSIZE`, or leaked dynamic-shape `MergedUnknownShapeSubGraph` runtime allocations. |

## 3. Auditing Long-Lived Tensors in `ge_proto_*_Build.txt`

To identify which tensors inflate `theory_min_memory_size`:
1. Parse node topological indices (`id: <int>`) and output tensor byte sizes (`size` attr in `output_desc`) from `ge_proto_*_Build.txt`.
2. For each tensor $t$ produced at node $u$ and last consumed at node $v$, compute lifespan $\Delta(t) = v - u$ and byte-lifespan product $\text{size}(t) \times \Delta(t)$.
3. Tensors with large $\text{size}(t) \times \Delta(t)$ (such as unfreed pre-attention full-sequence activations or duplicated cast buffers) should be recomputed locally or scheduled immediately before their consumer.
