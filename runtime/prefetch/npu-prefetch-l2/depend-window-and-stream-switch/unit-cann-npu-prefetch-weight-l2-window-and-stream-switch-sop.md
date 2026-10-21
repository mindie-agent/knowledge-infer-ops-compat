---
schema: mindie-graded-unit/1
id: unit-cann-npu-prefetch-weight-l2-window-and-stream-switch-sop
revision: 1
title: "CANN Weight L2 Prefetch (npu_prefetch): Execution Window, max_size Tuning, and Multi-Stream Cmo SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: runtime/prefetch/npu-prefetch-l2/depend-window-and-stream-switch
inherits_from:
  - unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
  domain: runtime
triggers:
  error_signatures:
    - "torch_npu.npu_prefetch"
    - "torchair.ops.npu_prefetch"
    - "Cmo"
  keywords:
    - "npu-prefetch"
    - "l2-cache-prefetch"
    - "cmo"
    - "weight-prefetch"
    - "mte2-bound"
provenance:
  source_type: source_contract
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/cann/zh/prefetch_principles.md
  evidence_refs:
    - https://github.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/cann/zh/prefetch_principles.md#L1-L215
  supersedes: []
---

## Falsifiable Claim

During small-batch LLM decode where `MatMul` and `QuantBatchMatMul` operators are MTE2-bound waiting on weight reads from HBM, issuing `torch_npu.npu_prefetch(target_weight, dependency_tensor, max_size)` emits a hardware Cache Maintenance Operation (`Cmo`) task that preloads `target_weight` into the SoC L2 Cache while an upstream Vector or Communication operator (`dependency_tensor`) is executing. If `npu_prefetch` is scheduled too early (before an intermediate HBM-heavy operator that evicts L2) or too late (immediately adjacent to the consumer `MatMul`), L2 hit rate drops back to baseline or causes MTE bandwidth contention.

## Hardware Mechanism & Prefetch Window Invariants

`torch_npu.npu_prefetch(input, dependency, max_size, offset=0)` lowers to a STARS `Cmo` (Cache Maintenance Operation) hardware task:
1. **Execution Ordering via `dependency`**:
   - Because `npu_prefetch` has no output tensor consumed by downstream compute, the compiler uses the `dependency` tensor argument to anchor the `Cmo` task in the execution DAG: the `Cmo` task launches **immediately after** the operator that produces `dependency` starts/completes, and **before** the consumer operator of `input` (the target weight) executes.
2. **Ideal Prefetch Window**:
   - The `dependency` operator and any operators between `dependency` and the consumer `MatMul` should be **compute-bound or communication-bound with low HBM traffic** (e.g., `RMSNorm`, `Add`, `RoPE`, or `AllReduce`/`AllGather` waiting on network latency).
   - **Too Early (Eviction Fault)**: If another large `MatMul` or KV-cache scan executes between the `Cmo` task and the consumer `MatMul`, its HBM traffic evicts the prefetched weights from L2 cache (typically 192 MB per die on 910B/910C).
   - **Too Late (Contention Fault)**: If `dependency` is the immediate input to the consumer `MatMul` with no intervening compute duration, the `Cmo` prefetch runs concurrently with the consumer `MatMul`'s own MTE2 weight loads, competing for HBM bandwidth.

## Two Scheduling Patterns

### Pattern A: Same-Stream Dependency Anchoring (`npu_prefetch`)

Anchor the weight prefetch of the **next** projection (`down_proj.weight` or `qkv_proj.weight`) to the output of an upstream Vector operator (`input_layernorm` or `act_fn`) on the same stream:

```python
import torch_npu

def decoder_mlp_with_prefetch(x, residual, up_gate_weight, down_weight):
    # 1. Upstream Vector op: RMSNorm
    norm_out, residual = rms_norm(x, residual)

    # Prefetch up_gate_weight during norm_out / activation prep (max 16 MB)
    torch_npu.npu_prefetch(up_gate_weight, norm_out, max_size=16 * 1024 * 1024)

    # 2. Consumer MatMul (reads up_gate_weight from L2 Cache)
    gate_up = torch.matmul(norm_out, up_gate_weight)

    # Prefetch down_weight while SwiGLU Vector kernel executes
    act_out = torch_npu.npu_swiglu(gate_up)
    torch_npu.npu_prefetch(down_weight, gate_up, max_size=16 * 1024 * 1024)

    # 3. Consumer MatMul (reads down_weight from L2 Cache)
    return torch.matmul(act_out, down_weight), residual
```

### Pattern B: Dedicated Auxiliary Prefetch Stream (`npu_stream_switch` + `Cmo`)

When the main stream executes back-to-back operators where single-stream `Cmo` insertion would delay task launch, place `npu_prefetch` on a dedicated auxiliary stream synchronized via `npu_wait_tensor`:

```python
import torchair
import torch_npu

with torchair.scope.npu_stream_switch("prefetch_stream", 0):
    wait_dep = torchair.scope.npu_wait_tensor( weight_tensor, trigger_tensor )
    torch_npu.npu_prefetch(weight_tensor, wait_dep, max_size=24 * 1024 * 1024)
```

## Tuning `max_size` & Profiling Verification

1. **Sizing `max_size`**:
   - Compute weight tensor byte size: $\text{bytes} = N \times K \times \text{dtype\_bytes}$.
   - Never set `max_size` above 50–60% of the physical L2 cache capacity per die (typically cap single-op prefetch at `16 MB – 48 MB`); over-prefetching evicts active KV cache and activation tiles in L2.
2. **Profiling Verification (`msprof`)**:
   - Inspect `trace_view.json` for `Cmo` tasks on the STARS hardware track and verify they overlap with Vector/Communication kernels immediately preceding the target `MatMul`.
   - Compare `op_summary_*.csv` for the target `MatMul`: `mte2_ratio` and total `Task Duration(us)` must decrease while `L2 Cache Hit Rate` increases.
