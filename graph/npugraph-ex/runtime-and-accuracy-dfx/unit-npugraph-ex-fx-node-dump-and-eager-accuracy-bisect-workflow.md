---
schema: mindie-graded-unit/1
id: unit-npugraph-ex-fx-node-dump-and-eager-accuracy-bisect-workflow
title: TorchAir npugraph_ex FX Node Tensor Dump, Pass Bisect, and Eager-vs-Graph Accuracy
  Diagnosis Workflow
domain_path: graph/npugraph-ex/runtime-and-accuracy-dfx
tier: L3
category: workflow
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: npugraph_ex
  diagnosis_domain: fx-dump-and-accuracy-bisect
  runtime: torch_npu
inherits_from:
- unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
- unit-torch-npugraph-ex-five-step-triage-and-compile-runtime-diagnosis-workflow
supersedes: []
triggers:
  error_signatures:
  - npugraph_ex output diverges from eager mode
  - cosine similarity drop after pattern_fusion_pass
  - non-contiguous transpose view feeds custom op in graph mode
  - FP16 intermediate overflow in fused RMSNorm or Softmax
  keywords:
  - npugraph-ex-accuracy bisect
  - CLOSE_MATMUL_K_SHIFT
  - LCCL_DETERMINISTIC
  - dump_fx_node tensor comparison
  - pattern_fusion_pass ablation
  - non-contiguous view accuracy drift
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/npugraph-ex-accuracy+npugraph-ex-debug+npugraph-ex-dfx-triage
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-accuracy/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-accuracy/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-dfx-triage/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/view_transpose_non_contiguous_accuracy_drift.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/fp16_overflow_and_bf16_cast_in_graph_mode.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/optimizations/experience/rules/deterministic_computation_switches.md
  source_type: curated_repo
  source_digest: 2ef9d311872cdde0209f1e9bb4814348cf527609b64217b02ab37c5793107441
---

# TorchAir npugraph_ex FX Node Tensor Dump, Pass Bisect, and Eager-vs-Graph Accuracy Diagnosis Workflow

## 1. Step 1: Lock Deterministic Execution Baseline

Before comparing Eager (`force_eager=True`) and Graph (`npugraph_ex`) outputs, eliminate non-deterministic reduction and matmul K-shift variance on both runs:

```bash
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=1
export CLOSE_MATMUL_K_SHIFT=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
```

```python
import torch, torch_npu
torch.manual_seed(1234)
torch.npu.manual_seed_all(1234)
torch.use_deterministic_algorithms(True)
```

If Eager vs `npugraph_ex` difference disappears after enabling deterministic flags, the discrepancy is floating-point reduction associativity rather than a graph compilation bug.

## 2. Step 2: Binary Bisect Across `_optimize_fx` Passes

If divergence persists under deterministic flags, isolate which optimization pass introduced the numerical drift by toggling pass switches on `torchair.CompilerConfig`:

1. **Disable Pattern Fusion Passes (`pattern_fusion_pass=False`)**:
   - If accuracy recovers, a fused kernel pattern (`AddRMSNorm`, `RmsNormQuant`, `FusedInferAttentionScore`, or custom `PatternPassManager` rule) has mismatched epsilon, dtype casting, or stride handling.
2. **Disable Reinplace Passes (`reinplace_inplaceable_ops_pass=False` / `reinplace_input_mutated_ops=False`)**:
   - If accuracy recovers, an in-place rewrite mutated a tensor buffer that is still read by a later node (aliasing hazard across branches or multi-stream readers).
3. **Compare First-Run Capture vs Second-Run Replay**:
   - If Run 1 (capture) matches Eager (`cosine_sim > 0.9999`) while Run 2 (replay) diverges, the bug is a **frozen address or unrefreshed host tiling metadata** issue during ACLGraph replay, not an FX pass bug.

## 3. Step 3: Node-by-Node FX Intermediate Tensor Dumping

Attach an FX node post-execution dump hook or enable TorchAir FX tensor dumping to capture every node's output in both `aot_eager` (or `force_eager=True`) and `npugraph_ex`:

```python
def register_fx_tensor_dumper(gm: torch.fx.GraphModule, dump_dir: str):
    for idx, node in enumerate(gm.graph.nodes):
        if node.op == "call_function":
            # Log node.name, node.target, max_abs_diff, cosine_similarity, and NaN/Inf count
            ...
```

Locate the **first topological FX node** where `cosine_similarity < 0.999` (for BF16/FP16) or `isfinite()` fails, and inspect its immediate inputs.

## 4. Step 4: Remediate the Three Canonical Graph-Mode Accuracy Traps

1. **Non-Contiguous `transpose` / `permute` / `slice` Feeding Custom or Fused Ops**:
   - Standard ATen ops auto-materialize contiguous copies when needed, whereas custom AscendC kernels or fused `PatternPass` targets read raw GM pointers assuming row-major contiguous layout (`stride(-1) == 1`).
   - **Fix**: Insert `.contiguous()` on non-contiguous views in `replace_fn` or require `val.is_contiguous()` inside `extra_check(match)`.
2. **FP16 Intermediate Accumulation Overflow in Graph Fusion**:
   - In FP16 models, `x.pow(2).mean(-1)` in RMSNorm or large-logit Softmax overflows FP16 max (`65504`) unless accumulated in FP32. Ensure fused `npu_rms_norm` / `npu_add_rms_norm` receives `float32` epsilon and preserves FP32 internal reduction.
3. **Cross-Stream Read-Before-Write Race**:
   - If the first diverging node sits at the merge point of a `torchair.scope.npu_stream_switch` branch, verify that `torchair.ops.npu_wait_tensor` bridges both input entry and output merge points.
