---
schema: mindie-graded-unit/1
id: unit-npugraph-ex-aclgraph-replay-failure-and-address-refresh-sop
title: TorchAir npugraph_ex ACLGraph Replay Failure, Dynamic Workspace, and Frozen
  Address Refresh SOP
domain_path: graph/npugraph-ex/runtime-and-accuracy-dfx
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: npugraph_ex
  diagnosis_domain: aclgraph-replay-and-address-refresh
  runtime: torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
supersedes: []
triggers:
  error_signatures:
  - EZ9999 HBM address access fault during ACLGraph replay
  - 507015 aclmdlRIExecuteAsync failed
  - 507035 task exception during graph replay
  - ACLGraph replay output is stale or identical across decode steps
  keywords:
  - aclmdlRIExecuteAsync
  - frozen data_ptr ACLGraph
  - update_node_params tiling
  - actual_seq_lengths refresh
  - static input buffer copy_
  - ACLGraph replay stale output
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/npugraph-ex-runtime-fail+aclgraph_static_address
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-runtime-fail/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-runtime-fail/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/aclgraph_static_address_and_tiling_update.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/npugraph_ex_inplace_mutation_and_cache_update_pitfalls.md
  source_type: curated_repo
  source_digest: e808477b0aa77d371a1181eabd02f0b8ad8c361c4ed876502b442cf039b6104f
---

# TorchAir npugraph_ex ACLGraph Replay Failure, Dynamic Workspace, and Frozen Address Refresh SOP

## 1. Frozen Device Pointer and Tiling Invariant in `ACLGraph`

When `AclConcreteGraph` captures an execution stream (`aclmdlRICaptureBegin` -> `aclmdlRICaptureEnd`), the Ascend Runtime records:
1. The **exact physical HBM virtual addresses (`tensor.data_ptr()`)** of all inputs, weights, KV caches, workspaces, and outputs bound during capture.
2. The **host-computed tiling structs** emitted by `aclnnXxxGetWorkspaceSize` at capture time.

Consequently, during `aclmdlRIExecuteAsync` replay:
- Passing a newly allocated input tensor with a different `data_ptr()` without copying into the captured input buffer (or without TorchAir input-address rebinding) causes the replayed kernel to read **stale capture-time memory** or **freed HBM pages (`507015` / `507035` / `EZ9999` AI Core data exception)**.
- Changing host-dependent tiling metadata (such as `actual_seq_lengths` in FlashAttention/FIA or `block_table` active lengths) without registering a runtime parameter update hook causes the kernel to execute with capture-time sequence lengths.

## 2. Diagnosing and Fixing Stale Outputs or Replay Exceptions

### Case A: Reallocated Input or Output Buffers Across Steps
- **Symptom**: First step (capture) produces valid logits, but step 2+ outputs identical tokens or NaNs, or crashes with HBM access fault after `NPUCachingAllocator.empty_cache()`.
- **SOP**:
  1. Pre-allocate static persistent buffers for `input_ids`, `positions`, `slot_mapping`, `block_table`, and `actual_seq_lengths` up to `max_num_batched_tokens` / `max_batch_size` during runner warmup.
  2. On every decode step, slice the static buffer to the current bucket size $B$ and copy new data in-place (`static_input_ids[:B].copy_(step_input_ids)`).
  3. Never call `torch.npu.empty_cache()` between ACLGraph capture and replay on the default allocator pool if custom ops hold raw untracked pointers.

### Case B: Host-Dependent Tiling Metadata (`actual_seq_lengths_kv`) Not Refreshing
- **Symptom**: Attention output in Decode step $t+1$ ignores newly appended KV cache tokens from step $t$.
- **SOP**:
  1. Pass `actual_seq_lengths_kv` as a device tensor (`actual_seq_lengths_tensor`) where supported by `npu_fused_infer_attention_score` (`actual_seq_lengths_kv` tensor input mode), OR
  2. Register the operator in `replace_dynamic_workspace_ops` / `torchair.scope.limit_core_num` update callbacks so `AclConcreteGraph` invokes `aclmdlRIUpdate` / `update_node_params` to refresh the host tiling descriptor before `aclmdlRIExecuteAsync`.

### Case C: Isolating Compile vs Capture vs Replay Crashes
1. Run with `force_eager=True` in `CompilerConfig`: if it still fails, the bug is in the single operator's `aclnn` call or input tensor shapes/dtypes, not `npugraph_ex`.
2. Run with `ASCEND_LAUNCH_BLOCKING=1` during the first step (capture): if it fails during step 1, inspect the failing `aclnnXxxGetWorkspaceSize` / `aclnnXxx` call in plog (`grep -n "\[ERROR\]" ~/ascend/log/debug/plog/*`).
3. If step 1 succeeds and step 2 fails inside `aclmdlRIExecuteAsync`, log `[tensor.data_ptr() for tensor in graph_inputs]` across step 1 and step 2 to identify which buffer changed address or was freed prematurely.
