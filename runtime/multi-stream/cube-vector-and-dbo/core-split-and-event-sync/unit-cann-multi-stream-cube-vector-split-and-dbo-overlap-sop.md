---
schema: mindie-graded-unit/1
id: unit-cann-multi-stream-cube-vector-split-and-dbo-overlap-sop
revision: 1
title: "CANN Multi-Stream Concurrent Execution: Cube-Vector Parallelism, Dual-Batch Overlap (DBO), and Core Limit SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: runtime/multi-stream/cube-vector-and-dbo/core-split-and-event-sync
inherits_from:
  - unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
  domain: runtime
triggers:
  error_signatures:
    - "npu_stream_switch"
    - "npu_wait_tensor"
    - "limit_cores"
    - "set_stream_limit"
  keywords:
    - "multi-stream"
    - "npu-stream-switch"
    - "npu-wait-tensor"
    - "dual-batch-overlap"
    - "dbo"
    - "cube-vector-parallel"
provenance:
  source_type: source_contract
  source_locator: cann/cann-recipes-infer@3c9266727c2a25bd17d503ad37d189a134ff364b:docs/cann/zh/multi_stream_principles.md
  evidence_refs:
    - https://github.com/cann/cann-recipes-infer/blob/3c9266727c2a25bd17d503ad37d189a134ff364b/docs/cann/zh/multi_stream_principles.md#L1-L235
  supersedes: []
---

## Falsifiable Claim

On Ascend 910B/910C (where AIC Cube cores and AIV Vector cores are physically decoupled in a 1:2 ratio), executing independent branches on separate streams via `torchair.scope.npu_stream_switch` and `torchair.scope.npu_wait_tensor` overlaps compute and communication only when the concurrent streams do not contend for the exact same hardware engine (e.g., Cube on Stream A + Vector/HCCL on Stream B). When two streams both launch operators targeting the same engine (e.g., two AIV Vector streams or AIV-based `MoeDistributeDispatch` alongside Vector operators), you must partition physical AIC/AIV cores per stream via `torchair.scope.limit_cores`; otherwise hardware wave quantization serializes the streams and degrades latency.

## Hardware Concurrency Model & Three Canonical Scenarios

Ascend 910B/910C separates hardware execution engines into:
- **AIC (Cube Cores)**: Matrix multiplication (`MatMul`, `GroupedMatMul`, `QuantBatchMatMul`, MLA Cube stages).
- **AIV (Vector Cores)**: Elementwise/reduction (`RMSNorm`, `RoPE`, `SwiGLU`, `TopK`) and AIV-driven HCCL collectives (`MoeDistributeDispatch`, `MoeDistributeCombine`).
- **SDMA / RoCE Engines**: Asynchronous DMA and RDMA network transfers.

### Scenario 1: Single-Batch Inter-Branch Parallelism (Cube + Vector Overlap)
In models with parallel branches (such as DeepSeek MLA Q-projection + RoPE vs. KV-projection + Cache update, or Shared Expert vs. Routed Expert gating):
- Assign the Cube-heavy branch to the default stream (`stream 0`) and the Vector-heavy branch to an auxiliary stream (`stream 1`).
- Because AIC and AIV pipelines execute independently, `stream 0` Cube ops and `stream 1` Vector ops run simultaneously with zero core partitioning needed.

### Scenario 2: Dual-Batch Overlap (DBO / Micro-Batch Ping-Pong)
In MoE Decode or Prefill with large all-to-all communication (`Dispatch` / `Combine`):
- Split the input batch into Micro-Batch 0 (`mb0`) and Micro-Batch 1 (`mb1`).
- Overlap `mb0`'s HCCL `Dispatch` / `Combine` communication stream with `mb1`'s Attention / Routed Expert `GroupedMatMul` compute stream.

### Scenario 3: Same-Engine Concurrent Streams (`limit_cores`)
When two concurrent streams both require AIV cores (for example, AIV-mode `MoeDistributeDispatch` consuming 8–16 AIV cores on the communication stream while the compute stream runs Vector operators like `RMSNorm`/`RoPE`), leaving both streams unrestricted (48 AIV cores each) causes STARS scheduling contention. Partition the cores explicitly using `torchair.scope.limit_cores(aic_num, aiv_num)`.

## Standard Operating Procedure (SOP)

### 1. Graph-Mode Multi-Stream Switching & Data-Dependency Synchronization

Never use eager `torch.npu.Stream.wait_stream()` inside a TorchAir / `npugraph_ex` traced graph; instead, use `npu_stream_switch` and `npu_wait_tensor` so the compiler inserts exact `EventRecord` / `EventWait` nodes tied to tensor data dependencies:

```python
import torchair

def parallel_branch_forward(x, shared_w, gate_w):
    # Main stream (stream 0): Cube MatMul for shared expert
    shared_out = torch.matmul(x, shared_w)

    # Switch to auxiliary stream "aux_vec_stream" (stream_priority=0)
    with torchair.scope.npu_stream_switch("aux_vec_stream", 0):
        # Wait until input tensor x is ready on main stream before reading on aux stream
        x_ready = torchair.scope.npu_wait_tensor(x, x)
        gate_logits = custom_vector_gate_op(x_ready, gate_w)

    # Main stream waits for gate_logits produced on "aux_vec_stream" before combining
    gate_ready = torchair.scope.npu_wait_tensor(shared_out, gate_logits)
    return shared_out + gate_ready
```

### 2. Core Partitioning via `torchair.scope.limit_cores`

When overlapping an AIV-based communication stream with a compute stream on a 24-AIC / 48-AIV die:

```python
with torchair.scope.npu_stream_switch("comm_stream", 0):
    # Restrict comm_stream to 8 AIC / 16 AIV cores
    with torchair.scope.limit_cores(8, 16):
        recv_x = dispatch_op(x)

with torchair.scope.limit_cores(16, 32):
    # Compute stream uses the remaining 16 AIC / 32 AIV cores concurrently
    attn_out = compute_branch(other_x)
```

### 3. Verification Rules

1. **Event Deadlock Prevention**: Ensure the dependency graph between `npu_wait_tensor(curr_stream_op, other_stream_op)` calls forms a strict DAG; never create circular waits across streams.
2. **Memory Footprint Audit**: Concurrent streams extend tensor lifetimes across the overlap window, preventing the caching allocator from reusing `mb0` activation buffers for `mb1`. Verify peak HBM usage via `torch.npu.max_memory_allocated()`.
