---
schema: mindie-graded-unit/1
id: unit-ge-pattern-fusion-pass-and-buffer-fusion-triage-workflow
title: Ascend GE Graph PatternFusionBasePass, UB Buffer Fusion, and Pass Miscompilation
  Triage Workflow
domain_path: graph/ge-compiler/fusion-passes-and-compile
tier: L3
category: workflow
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: ge
  diagnosis_domain: ge-fusion-pass-and-ub-fusion
  runtime:
  - ge
  - torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-ge-graph-compile-stage-and-dynamic-shape-partition-sop
supersedes: []
triggers:
  error_signatures:
  - PatternFusionBasePass Fusion failed
  - BufferFusionPassBase GetFusionNodes topological cycle
  - ge_proto_Build.txt pass_name mutation error
  - fusionSwitchFile GraphFusion UBFusion bisect
  keywords:
  - PatternFusionBasePass
  - BufferFusionPassBase
  - DUMP_GE_GRAPH
  - DUMP_GRAPH_LEVEL
  - fusionSwitchFile
  - UBFusion GraphFusion
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/ge-pass-analysis+ge-overview
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-pass-analysis/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-pass-analysis/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-overview/SKILL.md
  source_type: curated_repo
  source_digest: 01ef57a1e7f423e760558107f954d7061fb2d418790a771a60ca435d79001f0c
---

# Ascend GE Graph PatternFusionBasePass, UB Buffer Fusion, and Pass Miscompilation Triage Workflow

## 1. Two-Tier Fusion Architecture in GE and FE

Ascend Graph Engine performs operator fusion at two distinct compiler tiers:

| Fusion Tier | Base Class & Stage | Transformation Scope | Typical Patterns |
|---|---|---|---|
| **Graph Fusion (`GraphFusion`)** | `PatternFusionBasePass` (`DefinePatterns` -> `Fusion`) during GE/FE `OptimizeGraph` | Replaces a multi-node high-level IR subgraph (`ge::ComputeGraph`) with a single macro-operator (`FusedInferAttentionScore`, `AddRmsNorm`, `MatMulAllReduce`) before kernel compilation. | Attention QKV+FA, Matmul+Bias+Cast, LayerNorm/RMSNorm+Residual Add. |
| **UB Buffer Fusion (`UBFusion`)** | `BufferFusionPassBase` (`DefineBufferFusionPattern` -> `GetFusionNodes`) inside FE/TBE | Keeps intermediate tensors between adjacent AI Core Cube (`AIC`) and Vector (`AIV`) or Vector+Vector operators inside on-chip Unified Buffer (`UB` / `L1`) without writing back to HBM (`GM`). | Matmul+Elemwise, Dequant+Activation, Reduce+Elemwise. |

## 2. Step-by-Step Workflow to Trace GE Pass Transformations

### Step 1: Dump Intermediate GE IR Graphs
Enable minimal-overhead IR text dumping (level 2 omits weight tensor binary blobs while preserving full node topology, shapes, formats, and `pass_name` provenance):

```bash
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=2
export DUMP_GRAPH_PATH=/tmp/ge_dump_ir
```

### Step 2: Diff Pre-Optimization vs Post-Build IR (`ge_proto_*`)
1. Locate `ge_proto_*_PreRunAfterOptimizeStage1.txt` and `ge_proto_*_Build.txt` in `/tmp/ge_dump_ir`.
2. Grep for the target operator or layer name and inspect the `attr { key: "pass_name" ... }` attribute attached to newly created fused nodes:
```bash
grep -n -C 15 "pass_name" /tmp/ge_dump_ir/ge_proto_*_Build.txt
```
3. Verify three structural invariants on every fused replacement node:
   - **Data & Control Anchor Wiring**: All incoming/outgoing data edges (`input_desc` / `output_desc`) and control dependencies from the original subgraph are preserved without introducing a topological cycle (`TopologicalSorting failed: graph has cycle`).
   - **Format & DataType Consistency**: Private tensor formats (`FRACTAL_NZ`, `NC1HWC0`) and accumulation dtypes (`DT_FLOAT16` vs `DT_BF16` vs `DT_FLOAT`) match between producer output and fused consumer input.
   - **Shape Range Propagation**: In dynamic-shape graphs, the fused node's `OpDesc` carries valid `shape_range` bounds so `InferShapeAndType` does not fail downstream.

## 3. Step 3: Bisecting Faulty Passes via `fusionSwitchFile`

When a `GraphFusion` or `UBFusion` pass causes a compile error, `InferShape` failure, or numerical accuracy regression, isolate it without recompiling CANN by passing a `fusion_switch.cfg` JSON file (`ge.fusionSwitchFile`):

```json
{
  "Switch": {
    "GraphFusion": {
      "AddRmsNormFusionPass": "off",
      "MatMulBiasAddFusionPass": "off"
    },
    "UBFusion": {
      "TbeMatmulElemwiseFusionPass": "off"
    }
  }
}
```

Bisect the active `pass_name` list recorded in `ge_proto_*_Build.txt` until the exact pass responsible for the failure is isolated.
