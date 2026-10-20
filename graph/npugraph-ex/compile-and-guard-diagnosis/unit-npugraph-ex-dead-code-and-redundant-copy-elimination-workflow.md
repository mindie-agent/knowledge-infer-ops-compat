---
schema: mindie-graded-unit/1
id: unit-npugraph-ex-dead-code-and-redundant-copy-elimination-workflow
title: TorchAir npugraph_ex Dead Code Pruning, Inplace Mutation, and Redundant Tensor
  Move Elimination Workflow
domain_path: graph/npugraph-ex/compile-and-guard-diagnosis
tier: L3
category: workflow
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: npugraph_ex
  diagnosis_domain: dead-code-and-tensor-move
  runtime: torch_npu
inherits_from:
- unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
- unit-torch-npugraph-ex-reinplace-and-tensor-move-performance-audit-sop
supersedes: []
triggers:
  error_signatures:
  - ViewCopy d2d kernel overhead in npugraph_ex profile
  - index_copy_ kv_cache_update failed to reinplace
  - slice_scatter clone epilogue survived decompose_auto_functionalized
  - RoPE cos sin cache gather emits redundant TensorMove
  keywords:
  - View2Out pass
  - ViewCopy elimination
  - index_copy_ kv cache
  - _prune_dead_code_before
  - _dead_code_elimination_pass
  - slice_scatter reinplace
  - RoPE cache indexing
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/npugraph-ex-dead-code
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-tensor-move/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-dead-code/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-tensor-move/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-View2Out/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/npugraph_ex_inplace_mutation_and_cache_update_pitfalls.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/index_copy_kv_cache_update.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/rope_cos_sin_cache_indexing.md
  source_type: curated_repo
  source_digest: 530c25ffaf8a08b79f4154f37efa7383e6e91a343c17d06a8780e185dc421e83
---

# TorchAir npugraph_ex Dead Code Pruning, Inplace Mutation, and Redundant Tensor Move Elimination Workflow

## 1. End-to-End Audit Workflow for Redundant `TensorMove` / `ViewCopy` Kernels

When Ascend Profiler (`msprof` / `torch_npu.profiler`) shows unexpected `TensorMove`, `ViewCopy`, or `aclnnInplaceCopy` kernels during `npugraph_ex` Decode replay:

### Step 1: Dump FX Graph Before and After `_optimize_fx`
Enable FX pass dumps and inspect the post-`decompose_auto_functionalized` and post-`eliminate_self_copy` graphs:
- Search for surviving `aten.clone.default`, `aten.slice_scatter.default`, `aten.select_scatter.default`, and `aten.copy_.default` nodes.
- Any `auto_functionalized` wrapper whose mutated base tensor is a **view** (`aten.slice`, `aten.view`, `aten.transpose`) rather than a storage-owning tensor cannot be directly reinplaced if the view chain has multiple aliases or non-contiguous strides.

### Step 2: Eliminate View-Chain Mutations (`View2Out` / `slice_scatter` Trap)
**Anti-Pattern**: Updating KV cache or state buffers through chained Python slice assignments inside `forward()`:
```python
# BAD: Lowers to aten.slice -> aten.slice_scatter -> full-buffer clone + copy_ in AOTAutograd
kv_cache[layer_idx, :, slot_idx, :] = new_kv
```
**Remediation**:
1. Pass per-layer KV cache tensors `[num_blocks, block_size, num_kv_heads, head_dim]` directly as top-level graph inputs rather than indexing a 5D mega-tensor inside the graph.
2. Perform updates using dedicated in-place NPU cache operators or `index_copy_` on the storage-owning tensor:
```python
# GOOD: Single in-place scatter/index_copy without full-buffer clone
kv_cache.index_copy_(0, slot_mapping, new_kv)
# Or fused paged cache update:
torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping)
```

### Step 3: Optimize RoPE `cos`/`sin` Cache Indexing
**Anti-Pattern**: Slicing and expanding `[1, max_seq, 1, head_dim]` `cos`/`sin` tables with multi-axis fancy indexing (`cos[:, position_ids, :, :]`), which emits non-contiguous `ViewCopy` + `BroadcastTo` kernels every layer.
**Remediation**:
- Pre-flatten `cos_sin_cache` to 2D `[max_seq_len, head_dim]` at model init.
- Index once at the model top-level before the decoder layer loop using `torch.index_select(cos_cache, 0, position_ids.view(-1))` (or pass `cos`, `sin` directly into `torch_npu.npu_rotary_mul` / `npu_apply_rotary_pos_emb`), sharing the gathered tensor across all $L$ decoder layers.

### Step 4: Verify Dead Code Elimination (`_prune_dead_code_before` & `_dead_code_elimination_pass`)
- `_prune_dead_code_before` strips unused `getitem` outputs from multi-output operators (e.g., unused attention softmax LSE or dropout masks) before reinplace analysis so phantom users(`len(node.users) > 1`) do not falsely block `can_inplace()`.
- `_dead_code_elimination_pass` runs after `apply_pattern_passes` to garbage-collect the original unfused operator chains replaced by `PatternPassManager`.
- **Side-Effect Invariant**: Custom operators that mutate inputs in-place without returning the mutated tensor MUST declare `Tensor(a!)` in their `TORCH_LIBRARY` schema; otherwise FX marks the node as pure (`has_side_effects() == False`) and `_dead_code_elimination_pass` silently deletes the custom op from the graph!
