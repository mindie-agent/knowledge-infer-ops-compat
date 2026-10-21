---
schema: mindie-graded-unit/1
id: unit-cann-autofuse-inductor-loopir-schedule-and-att-tiling-sop
revision: 1
title: "AutoFuse PyTorch Inductor NPU Extension: LoopIR Schedule Space, ATT Dynamic Tiling, and Triton-Ascend Codegen SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: graph/autofuse/inductor-npu-ext/loopir-schedule-and-att-tiling
inherits_from:
  - unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
  domain: graph
triggers:
  error_signatures:
    - "AutoFuse"
    - "torch_npu._inductor"
    - "config.enable_AutoFuse"
    - "ASCEND_LAUNCH_BLOCKING"
  keywords:
    - "autofuse"
    - "torch-inductor"
    - "loopir"
    - "att-tiling"
    - "triton-ascend"
    - "deepseek-v4"
provenance:
  source_type: source_contract
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/models/deepseek_v4/deepseek_v4_autofuse_guide.md
  evidence_refs:
    - https://github.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/models/deepseek_v4/deepseek_v4_autofuse_guide.md#L1-L240
  supersedes: []
---

## Falsifiable Claim

When compiling memory-bound Vector and Reduction subgraphs on Ascend 910B/910C using `AutoFuse` (`torch-npu-inductor-ext`), importing `torch_npu._inductor` and setting `config.enable_AutoFuse = True` prior to `torch.compile(model, backend="inductor", dynamic=False)` lowers FX/ATen subgraphs through LoopIR schedule construction, ATT (`Autofuse Tiling Tensor`) dynamic shape generation, and `triton-ascend` kernel codegen. This fuses consecutive elementwise and reduction operators (`cat + slice + norm + add`, `QKV + RoPE`, `SwiGLU`, and DSA lightning indexer tails) to eliminate HBM round-trips and operator dispatch overhead without requiring manual AscendC kernels.

## Mechanism & Four-Stage Compilation Pipeline

`AutoFuse` registers NPU-specific lowering and codegen hooks into `torch._inductor`:

1. **FX Graph Capture (`Torch Dynamo`)**:
   - Captures Python bytecode into an FX Graph and decomposes composite operators into ATen primitives.
   - Partitioning splits the graph at Cube/collective boundaries (`mm`, `bmm`, `all_reduce`, `all_to_all`) and groups contiguous Pointwise and Reduction chains into `FusedSchedulerNode` regions.
2. **LoopIR & Schedule Space Construction (`autofuse.py`)**:
   - Converts `SchedulerNode` / `FusedSchedulerNode` compute expressions into `LoopIR` with explicit loop axes (`axis_type`: Parallel vs. Reduction, `axis_length`, and stride order).
   - Evaluates candidate loop splits, axis merges, and tiling factors against Ascend Unified Buffer (UB) capacity and Vector core counts, selecting the optimal schedule via heuristic cost modeling or autotuning.
3. **ATT (`Autofuse Tiling Tensor`) Dynamic Tiling**:
   - For dynamic batch/sequence dimensions (`dynamic=True`), generates a host-side C++ tiling wrapper (`att_tiling`) alongside the device kernel so runtime tensor shapes compute tiling parameters on CPU without recompiling the device binary.
4. **Triton-Ascend Codegen (`triton_Autofuse.py`)**:
   - Translates scheduled `LoopIR` into Ascend-adapted Triton Python kernels (`tl.load`, `tl.store`, `tl.sum`, `tl.max`) and compiles them into NPU executable binaries via `triton-ascend` (`bishengir-compile`).

## High-Yield Subgraph Fusion Patterns

| Subgraph Pattern | Unfused Bottleneck | AutoFuse Schedule Optimization |
| :--- | :--- | :--- |
| **`QKV Split + RMSNorm + RoPE`** | 6–8 separate Vector launches and repeated intermediate HBM reads/writes | Fuses slice, square-mean reduction, reciprocal-sqrt, and rotary sin/cos multiply-add into 1–2 kernels within UB |
| **DSA Lightning Indexer Tail (`relu + mul + sum + mask`)** | Multi-head FP32/BF16 score reduction across `[B, S, H, K]` materializes large intermediate tensors | Fuses ReLU activation, weight scaling, head reduction, and causal mask fill in a single Reduction loop |
| **`SwiGLU + Residual Add + RMSNorm`** | Separate `silu(gate) * up` writeback followed by residual add and norm | Keeps post-MLP hidden states in UB across residual addition and variance reduction |

## Standard Operating Procedure (SOP)

### Step 1: Enable AutoFuse in PyTorch NPU

```python
import torch
import torch_npu
import torch_npu._inductor
from torch._inductor import config

# Enable AutoFuse schedule and Triton-Ascend codegen
config.enable_AutoFuse = True

compiled_fn = torch.compile(
    model_or_subm,
    backend="inductor",
    dynamic=False,
)
```

### Step 2: Inspect Generated FX, LoopIR, and Triton Kernels

Use environment flags to dump intermediate representations and verify that target operators were fused rather than falling back to eager ATen calls:

```bash
export TORCH_LOGS="+inductor,output_code"
export AUTOFUSE_PRINT_IR=1
export TRITON_CACHE_DIR="/workspace/triton_cache"
```

- Check `output_code` logs for `triton_autofuse_*` kernel definitions.
- Verify that the number of `tl.load` and `tl.store` calls matches only the external inputs and outputs of the subgraph (no intermediate global memory stores).

### Step 3: Guard Against Known Pitfalls

1. **Avoid Graph Breaks Inside Target Subgraphs**:
   - Tensor-dependent control flow (`if x.item() > 0:`), inplace views across custom C++ autograd functions, or logging calls inside `forward` split FX graphs and prevent `FusedSchedulerNode` formation.
2. **Coordinate with TorchAir (`npugraph_ex`)**:
   - Do not wrap the exact same submodule simultaneously in outer `torch.compile(backend="npugraph_ex")` and inner `torch.compile(backend="inductor")` unless explicitly partitioning leaf helper functions; otherwise Dynamo re-traces across backend boundaries.
3. **Verify Numerical Accuracy Before Benchmarking**:
   - Compare compiled subgraph outputs against eager FP32/BF16 baselines (cosine similarity $\ge 0.999$) because fused reductions change accumulation order in UB.
