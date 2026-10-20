---
schema: mindie-graded-unit/1
id: unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
title: TorchAir npugraph_ex Capture, FX Lowering, and Pass Pipeline Ordering Axioms
domain_path: graph/npugraph-ex/architecture-and-passes
tier: L1
category: axiom
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: npugraph_ex
  pipeline_stage: fx-lowering-and-pass-pipeline
  runtime: torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
supersedes: []
triggers:
  error_signatures:
  - _optimize_fx pass ordering violation
  - AclConcreteGraph capture stream mismatch
  - npu_stream_switch scope exited without npu_wait_tensor
  - SuperKernel BUILD_TILING FUSE_SCOPE failure
  keywords:
  - AclConcreteGraph capture
  - _optimize_fx pipeline
  - npu_stream_switch
  - npu_wait_tensor
  - SuperKernel FUSE_SCOPE
  - decompose_auto_functionalized order
  - replace_dynamic_workspace_ops
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/npugraph-ex-architecture+npugraph-ex-overview
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-architecture/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-overview/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-architecture/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/graph_mode_torchair_aclgraph.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/superkernel_tiling_schedule.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/concepts/techniques/multi_stream_execution.md
  source_type: curated_repo
  source_digest: 4c265986bc07bcaa178a44e2e88ff18d5fabbf311274db832b2388c4d91594a0
---

# TorchAir npugraph_ex Capture, FX Lowering, and Pass Pipeline Ordering Axioms

## 1. End-to-End Capture and Lowering Boundary

`torch.npu.npugraph_ex` (and `torchair.get_npu_backend()`) lowers PyTorch models into Ascend device execution graphs through four strictly ordered stages:

1. **Bytecode Interception & Guard Synthesis (`TorchDynamo`)**: Traces Python bytecode into an FX graph while emitting input shape, stride, dtype, device, and global-state Guards. Any untraceable Python construct (`.item()`, `.cpu()`, `.tolist()`, data-dependent control flow) splits the graph into multiple subgraphs separated by eager fallback regions.
2. **AOT Dispatch & Functionalization (`AOTAutograd`)**: Traces joint forward/backward or inference-only ATen FX graphs (`aot_eager_decomp_partition`), converting in-place ATen mutations into functional `torch.ops.higher_order.auto_functionalized` wrappers to preserve SSA dataflow invariants during pass transformations.
3. **FX Optimization Pipeline (`npu_fx_compiler` / `_optimize_fx`)**: Executes deterministic FX graph passes in a fixed dependency order before target lowering.
4. **Target Concrete Graph Lowering (`AclConcreteGraph` vs `GeConcreteGraph`)**:
   - **ACLGraph Mode (`mode="reduce-overhead"` / `aclgraph`)**: Captures `aclnn` operator launches onto an Ascend Runtime task stream (`aclmdlRI` / `torch.npu.NPUGraph`) and replays the captured task ring with near-zero host dispatch overhead.
   - **GE Graph Mode (`mode="max-autotune"` / `ge`)**: Lowers FX nodes via `@register_fx_node_ge_converter` into Ascend Graph Engine IR (`ge::Graph`) for whole-graph Cube/Vector/UB fusion and static IMAS memory planning.

## 2. Pass Ordering Invariant in `_optimize_fx`

The FX optimization pipeline inside `npugraph_ex` enforces a strict topological ordering because downstream passes rely on invariants established by upstream passes:

```text
_prune_dead_code_before
  -> decompose_auto_functionalized
  -> reinplace_inplaceable_ops_pass
  -> reinplace_input_mutated_ops
  -> replace_dynamic_workspace_ops
  -> remove_noop_ops
  -> eliminate_self_copy
  -> apply_pattern_passes (_register_pattern_passes)
  -> _dead_code_elimination_pass
```

- **Functionalization Before Reinplace**: `decompose_auto_functionalized` must inline functional subgraphs and explicit `.copy_()` epilogues before `reinplace_inplaceable_ops_pass` and `reinplace_input_mutated_ops` attempt to restore in-place operator overloads and eliminate redundant buffer allocations.
- **Dynamic Workspace Replacement Before Pattern Fusion**: `replace_dynamic_workspace_ops` rewrites operators whose workspace size depends on runtime metadata prior to `apply_pattern_passes` so fused macro-operators do not hide dynamic workspace recalculation hooks from `AclConcreteGraph`.
- **No-Op & Self-Copy Cleanup**: `remove_noop_ops` (eliminating identity `slice`/`view`/`cast` nodes) and `eliminate_self_copy` (removing `copy_(x, x)` left behind by reinplacing) must run before pattern matching so trivial view/copy nodes do not break multi-node fusion patterns.

## 3. Decode vs Prefill Execution Mode Selection Axioms

| Workload Phase | Primary Bottleneck | Recommended Execution Mode | Architectural Rationale |
|---|---|---|---|
| **Prefill (Large $B 	imes S$)** | AI Core Cube/Vector compute & HBM bandwidth | **Eager (`force_eager=True`)** or **Chunked Prefill** | Single-op execution time ($> 500\,\mu\text{s}$) dominates host dispatch latency ($\sim 5\text{--}10\,\mu\text{s}$); graph capture across variable prompt lengths causes frequent Guard recompilation and static memory duplication. |
| **Decode (Small $B$, $S = 1$)** | Host Python/C++ `aclnn` dispatch & task launch | **ACLGraph (`npugraph_ex` / `piecewise`)** | Decode step operators execute in $2\text{--}20\,\mu\text{s}$, leaving NPU streams starved behind host launch overhead; `ACLGraph` replays the entire step in a single runtime submission. |
| **Decode Ultra-Low Latency** | Task Scheduler (TS) inter-kernel launch bubbles | **ACLGraph + SuperKernel** | Fuses adjacent AICore kernels inside a `FUSE_SCOPE` into one mega-kernel via `BUILD_TILING` -> `ADD_BINARY` -> `LAUNCH_SUPER_KERNEL`, eliminating TS scheduling gaps between small operators. |

## 4. Multi-Stream Scope and Data-Dependency Edge Axiom

When overlapping independent subgraphs (such as shared experts with routed expert `AllToAll` dispatch/combine, or Q/KV latent projection branches in MLA):

- Secondary streams are declared in FX graph mode via `with torchair.scope.npu_stream_switch(stream_tag, stream_priority):`.
- **Cross-Stream Ordering Invariant**: Ascend Runtime streams execute asynchronously; any tensor produced on stream $A$ and consumed on stream $B$ (or vice versa, including memory reuse safety) must be explicitly bridged by `torchair.ops.npu_wait_tensor(consumer_input, producer_output)`.
- `npu_wait_tensor` synthesizes an `aclrtEvent` record/wait edge inside `AclConcreteGraph` (or a GE `ControlAnchor` in `GeConcreteGraph`) and is preserved as an explicit side-effecting node against dead-code elimination.
