---
schema: mindie-graded-unit/1
id: unit-ge-graph-compile-stage-and-dynamic-shape-partition-sop
title: Ascend GE Graph Compile Stage Timing, Dynamic Shape Unknown-Flag Propagation,
  and Partition SOP
domain_path: graph/ge-compiler/fusion-passes-and-compile
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: ge
  diagnosis_domain: compile-stage-and-dynamic-shape
  runtime:
  - ge
  - torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
supersedes: []
triggers:
  error_signatures:
  - '[EVENT] GE Trace:CompileGraph stage耗时过长'
  - MergedUnknownShapeSubGraph partition split static graph
  - ge.graph_compiler_cache_dir cache miss recompilation
  - SubgraphPartitionAndOptimization dynamic shape fallback
  keywords:
  - GE CompileGraph stages
  - SubgraphPartitionAndOptimization
  - MergedUnknownShapeSubGraph
  - ge.graph_compiler_cache_dir
  - ASCEND_CACHE_PATH warmup
  - InferShapeAndType unknown flag
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/ge-compile-analysis+ge-dynamic-shape-analysis
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-compile-analysis/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-compile-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-dynamic-shape-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/ge_graph_compile_cache_and_warmup.md
  source_type: curated_repo
  source_digest: c25b64dec260be9f49edb060c6e71a2454887ebbd0121e1adfd71813235c8399
---

# Ascend GE Graph Compile Stage Timing, Dynamic Shape Unknown-Flag Propagation, and Partition SOP

## 1. Auditing GE Graph Compilation Stage Latency

When GE graph mode (`torchair` GE backend or MindIE GE execution) exhibits long startup or first-request latency, extract the stage breakdown from CANN plog (`ASCEND_GLOBAL_EVENT_ENABLE=1`):

```bash
grep -E "\[EVENT\] GE.*(CompileGraph|Initialize|Optimize|Subgraph|Build)" ~/ascend/log/debug/plog/plog-*.log
```

| GE Compilation Stage | Primary Responsibility | Dominant Bottleneck Cause & Fix |
|---|---|---|
| `InferShapeAndType` | Propagates tensor shapes, dtypes, and `unknown_shape` flags (`-1` dim, `-2` rank) | Unbounded `shape_range` or host-value-dependent (`ValueDepend`) ops forcing iterative symbolic inference. |
| `OptimizeStage1` | Graph-level IR passes (`PatternFusionBasePass`, constant folding, dead node removal) | Excessive unrolled loops or quadratic subgraph matching on >10,000-node graphs. |
| `SubgraphPartitionAndOptimization` | Engine partitioning (`AIcoreEngine`, `DNN_VM_AICPU`, `DNN_VM_RTS`) + FE/TE operator compilation | **Most common bottleneck (>80% of compile time)**: TBE/AscendC parallel kernel compilation when operator cache misses or dynamic shape splits graphs into dozens of subgraphs. |
| `Build` (`MemoryAllocation`) | Static IMAS topological memory solving, stream/event assignment, task generation | High stream fragmentation or thousands of partitioned known/unknown subgraphs. |

## 2. Dynamic Shape `unknown_shape` Contamination and Subgraph Partitioning SOP

In GE, a graph is classified as **static known-shape (`is_unknown_shape == false`)** only if every node has deterministic dimensions. Once a single operator introduces an unknown output shape:

1. **Sources of `unknown_shape` Contamination**:
   - Data-dependent output shape operators (`NonZero`, `Unique`, `Where`, masked `Gather`) whose output dimension is inherently `-1` at compile time.
   - Operators marked `force_unknown_shape` in their OP prototype or lacking a dynamic tiling implementation (`cannot support no tiling`).
   - Unmarked dynamic inputs propagating `-1` downstream through `InferShape`.
2. **Consequence (`MergedUnknownShapeSubGraph` Split)**:
   - `SetGraphUnknownFlag` marks the contaminated cone as `unknown`.
   - `SubgraphPartitionAndOptimization` partitions the graph into alternating **Known-Shape Subgraphs** and **Unknown-Shape Subgraphs** (`MergedUnknownShapeSubGraph`), inserting host-driven `PartitionedCall` boundaries.
   - Cross-partition operator fusion and static whole-graph IMAS memory reuse are broken across every `PartitionedCall` boundary.
3. **Diagnosis & Remediation**:
   - Grep `SetGraphUnknownFlag`, `force unknown as dynamic`, and `marked force unknown node forcibly` in plog to locate the **root contaminating node** (the earliest topological node where `unknown` flips to `true`).
   - Replace data-dependent indexing (`NonZero`/`masked_select`) with fixed-shape padded masks (`torch.where`) or bucketed static shapes so the entire graph remains known-shape.

## 3. Persistent GE Graph Compiler Cache and Multi-Gear Warmup SOP

To eliminate 3–15 minute cold compilation delays across service restarts:

```python
config = torchair.CompilerConfig()
config.ge_config.graph_compiler_cache_dir = "/var/cache/mindie/ge_cache"
# Bind deterministic graph_key per model + parallel configuration
```

1. Set `export ASCEND_CACHE_PATH=/var/cache/mindie/ge_cache` and ensure all ranks on the same node either read a pre-warmed cache or write to rank-isolated cache subdirectories during initial warmup to prevent concurrent file overwrite corruption.
2. Execute offline warmup across all configured sequence-length and batch-size gears (`max_prefill_tokens`, decode batch buckets) before opening the inference service health-check endpoint.
