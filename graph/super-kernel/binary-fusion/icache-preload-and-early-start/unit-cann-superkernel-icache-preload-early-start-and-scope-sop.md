---
schema: mindie-graded-unit/1
id: unit-cann-superkernel-icache-preload-early-start-and-scope-sop
revision: 1
title: "CANN SuperKernel Binary Fusion: Scope Decoration, I-Cache Preload, and Operator Early-Start SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: graph/super-kernel/binary-fusion/icache-preload-and-early-start
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
    - "super_kernel"
    - "super_kernel_optimize"
    - "super_kernel_optimize_options"
    - "early_start"
  keywords:
    - "superkernel"
    - "super-kernel"
    - "icache-preload"
    - "early-start"
    - "stars-dispatch"
    - "binary-fusion"
provenance:
  source_type: source_contract
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/cann/zh/super_kernel.md
  evidence_refs:
    - https://github.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/cann/zh/super_kernel.md#L1-L185
  supersedes: []
---

## Falsifiable Claim

On Ascend 910B/910C decode graphs where per-operator STARS task scheduling, AICore context initialization, and Instruction Cache (I-Cache) cold misses add 1–3 $\mu\text{s}$ of inter-operator free time, wrapping target subgraphs in `torchair.scope.super_kernel(scope_name, options)` links the compiled AICore operator binaries into a single SuperKernel binary. Enabling `icache_preload` and `early_start` overlaps next-operator instruction prefetching and scalar setup with current-operator execution, eliminating STARS task-launch bubbles across fused operators.

## Hardware & Compiler Mechanism

In standard GE or ACLGraph execution, each operator corresponds to a separate STARS hardware task:
1. STARS scheduler dispatches the task to AIC/AIV cores.
2. Each core initializes scalar registers and loads instructions into its local I-Cache (causing instruction fetch stalls on first execution).
3. Each core drains its pipelines and signals completion back to STARS before the next dependent task launches.

**SuperKernel Binary Fusion** eliminates these boundaries at the binary level without altering each operator's internal AscendC/TBE compute logic:
- **Single STARS Task Dispatch**: A sequence of operators inside the same `super_kernel` scope is linked into one overarching entry function (`super_kernel_entry`), invoked via a single STARS task launch.
- **In-Core Barrier Synchronization**: Inter-operator dependencies inside the scope are lowered to fast hardware on-chip sync instructions (`ffts_cross_core_sync` / internal barrier) rather than returning control to the STARS hardware scheduler.
- **I-Cache Preload (`icache_preload`)**: While operator $i$ executes on Cube/Vector pipelines, the Scalar unit issues instruction prefetch instructions (`preload`) for operator $i+1$'s binary entry so instructions reside in L1 I-Cache before operator $i+1$ starts.
- **Operator Early Start (`early_start`)**: Allows the Scalar unit of AIC/AIV cores to begin executing the prologue (tiling parameter parsing and descriptor setup) of operator $i+1$ before operator $i$'s MTE3/Vector pipelines finish draining, synchronizing only immediately before the first conflicting hardware pipe instruction.

## Standard Operating Procedure (SOP)

### Step 1: Mark Target Subgraphs with `torchair.scope.super_kernel`

In GE Mode (`torchair`), wrap contiguous compute blocks (such as an entire Decoder layer's attention + MLP or post-communication compute chain) using `torchair.scope.super_kernel`:

```python
import torchair

def decoder_layer_forward(hidden_states, residual, cos, sin):
    # Enable SuperKernel scope with icache_preload and early_start options
    with torchair.scope.super_kernel(
        "decoder_layer_sk",
        "icache_preload=1;early_start=1",
    ):
        normed, residual = rms_norm(hidden_states, residual)
        qkv = qkv_proj(normed)
        attn_out = rope_and_attention(qkv, cos, sin)
        mlp_out = ffn_swiglu(attn_out)
    return mlp_out, residual
```

In `npugraph_ex` mode, pass SuperKernel optimization switches via `options` when calling `torch.compile`:

```python
options = {
    "super_kernel_optimize": True,
    "super_kernel_optimize_options": {
        "icache_preload": 1,
        "early_start": 1,
    },
}
```

### Step 2: Enforce Scope Boundary Invariants

SuperKernel binary linking applies only to pure AIC/AIV operators that can execute inside a unified STARS task context. The compiler automatically splits a `super_kernel` scope into multiple sub-SuperKernels whenever it encounters:
1. **AICPU Operators** or host-callback operators (`print`, host tensor copy).
2. **HCCL Collective Communication Operators** (`AllReduce`, `AllGather`, `Dispatch`, `Combine`) running on AIV/SDMA communication streams unless explicitly co-scheduled under supported MC2/AIV-comm fusion templates.
3. **Explicit Multi-Stream Switches** (`torchair.scope.npu_stream_switch`) with cross-stream event wait boundaries.

### Step 3: Profiling Verification

When inspecting `msprof` / `ascend_pt` timelines (`trace_view.json`) after enabling SuperKernel:
- Multiple individual operator kernels (`MatMul`, `RmsNorm`, `Add`, `SwiGLU`) disappear from the top-level Task Scheduler track and are replaced by a single fused `SuperKernel_*` task bar.
- Total kernel duration is strictly less than the sum of the individual unfused kernels plus their inter-kernel gaps (`Free Time`).
