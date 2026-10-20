---
schema: mindie-graded-unit/1
id: unit-torch-npugraph-ex-reinplace-and-tensor-move-performance-audit-sop
revision: 1
tier: L2
category: sop
domain_path: runtime/torch-npu
evidence_strength: software_contract
title: Static FX graph audit SOP for npugraph_ex reinplace pass misses and redundant tensor moves
conditions:
  accelerator: ascend
  runtime: torch_npu
  compiler_backend: npugraph_ex
  repository: cann/cannbot-skills
  revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-torch-npu-custom-op-registration-and-faketensor-meta-sop
provenance:
  source_type: cannbot_skill_and_compiler_contract
  source_locator: docs/evidence/cannbot-graph-and-ge-provenance.json
  source_digest: 32d60b6077316531910a673236e8f9d7a6f90909474f8fd71e51a55cea6eae44
  evidence_refs:
    - docs/evidence/cannbot-graph-and-ge-provenance.json
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/torch-npugraph-ex-performance-diagnosis/SKILL.md#L21-L160"
  supersedes: []
triggers:
  error_signatures:
    - "can_inplace return False, will skip reinplacing for node"
    - "possible missed reinplacing opportunities"
    - "Total size of missed opportunities"
    - "reinplace failed, all mutated args(get_item) must have copy epilogues"
    - "cannot find an inplace op for node"
  keywords:
    - npugraph_ex
    - reinplace
    - reinplace_inplaceable_ops_pass
    - reinplace_input_mutated_ops
    - decompose_auto_functionalized
    - eliminate_self_copy
    - tensor move
    - auto_functionalized
---
When `torch.compile(..., backend="npugraph_ex")` completes without errors (Step 5 passes) but exhibits higher latency than Eager or unexpected `TensorMove` kernels in profiling, audit the `TORCH_COMPILE_DEBUG=1` FX graph sequence (`npugraph_ex/model__N/forward/*_after_*.txt`) and `npugraph_ex/debug.log` to isolate incomplete reinplace lowering (`graph/torch-npugraph-ex-performance-diagnosis/SKILL.md:21-160`).

## 1. Functionalize vs Reinplace Four-Pass Pipeline

AOTAutograd's `Functionalize` pass rewrites in-place mutations into out-of-place SSA nodes and appends `aten.copy_(input, new)` write-back epilogues for mutated graph inputs/attributes (such as KV caches). `npugraph_ex` (`torchair/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py`) restores in-place execution across four sequential passes:

1. `reinplace_inplaceable_ops_pass` (`inplace_pass`): Converts intermediate out-of-place ops (`b = foo(a)`) into in-place ops (`a.foo_()`). Does not touch `auto_functionalized`.
2. `reinplace_input_mutated_ops` (`input_inplace_pass`): Folds `aten.copy_` write-back epilogues on `placeholder` / `get_attr` inputs and makes base-tensor reinplace decisions for `auto_functionalized` nodes (recording un-reinplaced bases in `node.meta["only_clone_these_tensors"]`).
3. `decompose_auto_functionalized`: Expands `auto_functionalized` into underlying mutable custom operators; materializes explicit `aten.clone` + write-back `aten.copy_` for any base tensor that missed reinplace.
4. `eliminate_self_copy`: Removes trivial `copy_(x, x)` self-copies left after successful in-place folding.

## 2. Three Root-Cause Categories of Redundant Tensor Moves

First inspect the final FX graph (`*_after_eliminate_self_copy.txt`) with:
```bash
grep -n "aten\.copy_\.default(\|aten\.clone\.default(" *_after_eliminate_self_copy.txt
```
Before attributing a `copy_`/`clone` to compiler reinplace failure, verify whether the model's Python wrapper (e.g., a quantization compressor or custom op wrapper) explicitly invoked `.clone()`, `._to_copy()`, or `.copy_()`. If not, classify into one of three compiler causes:

| Category | Visible Signal in `*_after_eliminate_self_copy.txt` | Confirmation in `npugraph_ex/debug.log` | Underlying Mechanism & Remediation |
|---|---|---|---|
| **Cause 1: Unfolded Input `copy_` Epilogue** | `aten.copy_.default(<placeholder\|get_attr>, src)` remains at graph tail | `[_reinplace_input_mutated_ops]` logs `can_inplace return False, will skip reinplacing for node`, `cannot find an inplace op for node`, or `all mutated args(get_item) must have copy epilogues` | Another view of the input buffer is read before `copy_`, multi-stream validation failed, or the op lacks an in-place `_` variant. Reorder view reads before mutation or register an in-place custom op variant. |
| **Cause 2: Un-reinplaced `auto_functionalized` Base** | Sequence `as_strided(input) -> clone -> as_strided -> custom.<op> -> copy_(input, as_strided_clone)` (two tensor moves per miss: one `clone` + one non-self `copy_`) | `possible missed reinplacing opportunities` and `Total size of missed opportunities` (with byte-level loss estimate), or `multi_stream_auto_functionalize` check failure | The base tensor or an aliasing view is read *after* the mutable custom op (`used-later`), preventing in-place mutation of the original buffer. Eliminate post-mutation reads of stale views or split the stream scope. |
| **Cause 3: Missed Intermediate In-Place Conversion** | Out-of-place / `*_functional` op remains in `*_after_reinplace_inplaceable_ops_pass.txt` (**no** `copy_` or `clone` node emitted, but incurs extra HBM allocation + write) | Check `reinplace_with_multi_stream_check` logs in `debug.log` | Multi-stream lifetime overlap or aliasing prevents converting `foo(a)` to `a.foo_()`. |

- **Scope Boundary**: If `torchdynamo/debug.log` contains `graph break`, `Restarting analysis`, or `guard failure`, stop the static tensor-move audit immediately and route to `unit-torch-npugraph-ex-five-step-triage-and-compile-runtime-diagnosis-workflow`.
