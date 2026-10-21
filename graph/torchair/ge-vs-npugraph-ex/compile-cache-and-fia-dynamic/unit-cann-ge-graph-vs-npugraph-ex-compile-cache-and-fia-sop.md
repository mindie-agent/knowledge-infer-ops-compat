---
schema: mindie-graded-unit/1
id: unit-cann-ge-graph-vs-npugraph-ex-compile-cache-and-fia-sop
revision: 1
title: "CANN Graph Modes: GE Mode vs. npugraph_ex, force_eager, Compile Cache, and Dynamic FIA Selection SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: graph/torchair/ge-vs-npugraph-ex/compile-cache-and-fia-dynamic
inherits_from:
  - unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
  - unit-ge-graph-compile-stage-and-dynamic-shape-partition-sop
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
  domain: graph
triggers:
  error_signatures:
    - "CompilerConfig"
    - "npugraph_ex"
    - "torchair.ops.npu_CompileCache"
    - "actual_seq_lengths_kv"
  keywords:
    - "torchair"
    - "ge-mode"
    - "npugraph-ex"
    - "force-eager"
    - "compile-cache"
    - "npu-fused-infer-attention-score"
provenance:
  source_type: source_contract
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/cann/zh/npu_graph_optimization.md
  evidence_refs:
    - https://github.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/cann/zh/npu_graph_optimization.md#L1-L260
  supersedes: []
---

## Falsifiable Claim

CANN TorchAir provides two distinct graph execution backends (`max-autotune` GE Graph mode vs. `npugraph_ex` ACLGraph/FX-pass mode). Selecting the wrong backend, failing to convert `actual_seq_lengths` tensor inputs into Python lists on `torch_npu.npu_fused_infer_attention_score` under dynamic shapes, or omitting `torchair.inference.cache_compile` causes either multi-minute cold-start recompilation or graph capture failures.

## Architecture Comparison: GE Mode vs. `npugraph_ex` Mode

| Dimension | GE Mode (`config.mode = "max-autotune"`) | `npugraph_ex` Mode (`config.mode = "npugraph_ex"` / `backend="npugraph_ex"`) |
| :--- | :--- | :--- |
| **IR Lowering Path** | FX Graph $\rightarrow$ TorchAir GEIR converter $\rightarrow$ GE global graph optimizer & runtime | FX Graph $\rightarrow$ Python-level FX passes (`reinplace`, `stream_assign`, fusion) $\rightarrow$ ACLGraph / `aclnn` capture |
| **Cold-Start Compilation Time** | High (3–15 minutes for large MoE/LLM graphs due to global C++ GE passes) | Low (seconds; lightweight Python FX transformations + hardware task capture) |
| **Dynamic Shape Flexibility** | Native symbolic shape support in GE (`dynamic=True`) without per-bucket task replay | Requires static shape bucketing (`compile_shapes`) for ACLGraph task capture or `dynamic=True` FX execution |
| **Debugging & Single-Step Inspection** | Supports `config.debug.run_eagerly = True` to bypass GE execution while testing FX conversion | Supports `force_eager=True` in `options` to execute FX passes in eager mode and inspect intermediate tensors |

## Standard Operating Procedure (SOP)

### 1. Configure `npugraph_ex` with Eager Fallback Debugging

When developing custom model blocks or diagnosing precision mismatches inside `npugraph_ex`, toggle `force_eager` to execute the transformed FX graph op-by-op without hardware ACLGraph capture:

```python
import torch

options = {
    "inplace_pass": True,
    "input_mutate_pass": True,
    "force_eager": False,  # Set True to debug FX pass precision op-by-op
}

compiled_model = torch.compile(
    model,
    backend="npugraph_ex",
    options=options,
    dynamic=True,
    fullgraph=True,
)
```

### 2. Configure GE Graph Mode and Persistent `cache_compile`

When deploying GE mode (`max-autotune`), wrap the target module method with `torchair.inference.cache_compile` so compiled GE artifacts persist across restarts and eliminate 5–15 minute cold-start compilation:

```python
import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.mode = "max-autotune"
config.experimental_config.frozen_parameter = True
config.experimental_config.tiling_schedule_optimize = True

# Optional: persist GE graph cache to disk
# Applied to a non-dynamically-patched forward helper method
model.cached_decode = torchair.inference.cache_compile(
    model._decode_step,
    config=config,
    dynamic=True,
    cache_dir="/workspace/ge_compile_cache",
)
```

> **Constraint**: `torchair.inference.cache_compile` validates that the graph inputs and module hierarchy remain deterministic; do not mutate `nn.Module` attributes inside the cached function during warmup.

### 3. Dynamic-Shape `npu_fused_infer_attention_score` (FIA) Contract

When compiling `torch_npu.npu_fused_infer_attention_score` under dynamic batch/sequence lengths (`dynamic=True`):
- The `actual_seq_lengths` and `actual_seq_lengths_kv` parameters accept either a Python `list[int]` or a 1D `torch.Tensor` in Eager mode, **but in Graph mode (`dynamic=True`), passing a 1D `torch.Tensor` on NPU/CPU without proper host-SymInt binding can stall or fail graph lowering**.
- **Rule**: Convert sequence lengths to a `list` of symbolic integers (`SymInt`) or pass a CPU tensor marked via `torch._dynamo.mark_dynamic`, and verify that `input_layout` (`"BNSD"`, `"BSND"`, or `"TND"`) matches the rank of `query`.

```python
# Recommended dynamic graph pattern for FIA:
actual_seq_q = [q_len] * batch_size  # SymInt list tracked by Dynamo
attn_out, softmax_lse = torch_npu.npu_fused_infer_attention_score(
    query,
    key,
    value,
    num_heads=num_heads,
    num_key_value_heads=num_kv_heads,
    input_layout="BNSD",
    scale=scale,
    actual_seq_lengths=actual_seq_q,
    actual_seq_lengths_kv=actual_seq_kv_list,
)
```
