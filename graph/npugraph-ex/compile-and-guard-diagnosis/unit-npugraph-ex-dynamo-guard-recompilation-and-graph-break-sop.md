---
schema: mindie-graded-unit/1
id: unit-npugraph-ex-dynamo-guard-recompilation-and-graph-break-sop
title: TorchAir npugraph_ex Dynamo Guard Recompilation, Graph Break, and Symbolic
  Shape Diagnosis SOP
domain_path: graph/npugraph-ex/compile-and-guard-diagnosis
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: npugraph_ex
  diagnosis_domain: dynamo-guard-and-graph-break
  runtime: torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
supersedes: []
triggers:
  error_signatures:
  - torch._dynamo hit config.cache_size_limit
  - Guard failed on tensor.size
  - ACLGraph capture OOM across batch buckets
  - 'aclmdlRICaptureBegin failed: stream count exceeds hardware limit'
  keywords:
  - TORCH_LOGS guards recompiles
  - cache_size_limit recompilation
  - mark_dynamic batch seq
  - ACLGraph batch bucket OOM
  - piecewise compilation
  - stream limit capture
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/npugraph-ex-guard+npugraph-ex-compile-fail
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-guard/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-guard/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-compile-fail/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/torchair_aclgraph_capture_and_custom_op_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/aclgraph_capture_oom_and_stream_limit.md
  source_type: curated_repo
  source_digest: 586a3b5c3fda4455378ada67bbef793329aba64012d2082c279267f056aa3daa
---

# TorchAir npugraph_ex Dynamo Guard Recompilation, Graph Break, and Symbolic Shape Diagnosis SOP

## 1. Capturing Guard Failures and Graph Breaks

Enable structured Dynamo and TorchAir compile diagnostics before running the target workload:

```bash
export TORCH_LOGS="+dynamo,guards,recompiles,graph_breaks"
export TORCH_COMPILE_DEBUG=1
```

Inspect the emitted `[recompiles]` and `[graph_breaks]` entries to classify the failure into one of three categories:

| Symptom | Root Cause | Remediation SOP |
|---|---|---|
| `hit config.cache_size_limit (8)` on `tensor.size(0)` or `actual_seq_lengths` | Unmarked dynamic batch/sequence dimension or Python integer length passed by value | Call `torch._dynamo.mark_dynamic(x, 0)` before `model(x)` or pad Decode batch sizes into fixed power-of-two/stepped buckets (`[1, 2, 4, 8, 16, 32, 64]`). |
| Guard fails on `tensor.stride()` across steps | Upstream slice or transpose produces non-contiguous views on some steps and contiguous tensors on others | Enforce `.contiguous()` before the compiled module entry boundary so input stride layout remains invariant. |
| `torch._dynamo.exc.Unsupported: call_method Tensor.item` or `.cpu()` | Host-device synchronization inside `forward()` (e.g., reading MoE token counts or dynamic sequence bounds on CPU) | Move host metadata computation outside the compiled region (`@torch._dynamo.disable` helper or runner pre-step) or pass device tensors to fused NPU ops. |
| Guard fails on mutated module attribute (`self.step += 1`) | Python integer or list attribute mutated inside `forward()` invalidates ID/value guard every step | Store step counters or KV indices in pre-allocated 1D `torch.int32`/`int64` NPU tensors updated via `.fill_()` or `.add_()`. |

## 2. Resolving Custom Operator and Third-Party Library Graph Breaks

1. **`einops.rearrange` / `einops.repeat` Graph Breaks in Older PyTorch**: Replace inside hot Decode loops with native `tensor.view(...)`, `tensor.permute(...)`, and `tensor.reshape(...)` operations.
2. **Custom `torch_npu` C++ Operators Triggering Graph Breaks**:
   - Verify the operator is registered via `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)` rather than a raw `pybind11` `m.def(...)` C++ function (Dynamo cannot trace raw `pybind11` functions without a `torch.ops` schema).
   - Verify `@torch.library.register_fake("namespace::op")` is registered and does not touch `.data_ptr()` or call `EXEC_NPU_CMD`.

## 3. Bounding ACLGraph Multi-Bucket Capture Memory and Stream Exhaustion

In `ACLGraph` mode (`npugraph_ex` with static kernel capture), each distinct captured batch-size bucket allocates its own private intermediate memory pool and holds bound runtime stream descriptors:

1. **Diagnosing Capture-Phase HBM Exhaustion (`NPU out of memory during capture`)**:
   - If capturing $K$ batch buckets (e.g., $K = 32$ buckets from $B=1$ to $B=256$), intermediate activation memory scales with the number of unshared graph pools.
   - **Fix**: Enable shared graph memory pool reuse across buckets (passing a shared `pool=graph_pool_handle` across `torch.npu.NPUGraph` instances) and coarsen bucket steps (`[1, 2, 4, 8, 16, 24, 32, 48, 64]`).
2. **Diagnosing Stream/Event Limit Exhaustion (`stream count exceeds hardware limit`)**:
   - Every `npu_stream_switch` scope inside a captured graph consumes additional hardware stream and event quotas per bucket.
   - **Fix**: Audit `npu_stream_switch` tags so all layers reuse the same logical auxiliary stream tag (e.g., `"moe_shared_stream"`) rather than generating a unique tag per layer (`f"layer_{i}_stream"`), or use `piecewise` compilation to keep variable-topology communication outside the captured piece.
