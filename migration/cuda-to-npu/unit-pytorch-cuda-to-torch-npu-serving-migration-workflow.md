---
schema: mindie-graded-unit/1
id: unit-pytorch-cuda-to-torch-npu-serving-migration-workflow
revision: 1
tier: L3
category: workflow
domain_path: migration/cuda-to-npu
content_digest: sha256:f57337fb64b868cdfc2dbc7635483a2c1f547984dc9a067632958c981c073800
evidence_strength: reproduced_empirical
title: Migrate CUDA-oriented PyTorch inference models and kernels to torch_npu and vllm-ascend
conditions:
  migration_direction: cuda-to-npu
  target_runtime: torch_npu
  serving_engine: vllm-ascend
  source_stack: pytorch-cuda
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-torch-npu-custom-op-registration-and-faketensor-meta-sop
  - unit-vllm-ascend-build-entrypoint-and-soc-probe
provenance:
  source_type: multi_run_empirical_synthesis
  source_locator: docs/evidence/cannbot-ops-compat-provenance.json
  source_digest: fe3cc021d0d3848cca592eb8d67f8d8fa70a25d7896841d59496572955386b71
  evidence_refs:
    - docs/evidence/cannbot-ops-compat-provenance.json
    - "receipt://vllm-ascend-evidence/k3-gqa-no-prefix-20260908"
    - "receipt://vllm-ascend-evidence/k3-gqa-no-prefix-20260908/kda-full-operator-regression-fixed.log"
    - "receipt://vllm-ascend-evidence/dsv4-qli-v2-mixed-20260829"
    - "archive://session-material/fla-kda-validation-report"
    - "archive://session-material/flashmla-c8-kernel-results"
    - "repo://vllm-ascend-causal-conv1d/csrc/torch_binding.cpp"
    - "repo://vllm-ascend-causal-conv1d/csrc/torch_binding_meta.cpp"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/model/model-infer-fusion/references/torch_npu_API/torch_npu_list.md#L28-L30"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/torch-ascendc-op-extension/routes/direct-invoke.md#L245-L263"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/model/model-infer-runtime-debug/SKILL.md#L83-L96"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/model/model-infer-graph-mode/SKILL.md#L1-L40"
  experiment_status: executed
  reproduction_scope: "Four-stage workflow exercised across live Ascend910_93 service runs, a live Ascend950 ACLGraph operator qualification and live A2/A5 AscendC kernel validation. Not one single-model end-to-end migration."
  supersedes: []
triggers:
  error_signatures:
    - "ModuleNotFoundError: No module named 'flash_attn'"
    - "ModuleNotFoundError: No module named 'causal_conv1d'"
    - "ModuleNotFoundError: No module named 'fla'"
    - "ImportError: libcudart.so"
    - "torch.cuda.CUDAGraph is not available"
    - "NCCL backend is not available"
    - "HCCL_CONNECT_TIMEOUT"
    - "HcclAllReduce error"
    - "triton CompilationError on Ascend"
    - "ACLGraph capture failed or replayed stale metadata"
    - "Meta backend not registered"
    - "custom op absent from the captured graph"
    - "NPU out of memory during capture"
  keywords:
    - cuda-to-npu
    - torch_npu
    - vllm-ascend
    - flash_attn
    - npu_fusion_attention
    - npu_fused_infer_attention_score
    - causal_conv1d
    - fla
    - triton
    - triton-ascend
    - nccl
    - hccl
    - CUDAGraph
    - NPUGraph
    - npugraph_ex
    - TorchAir
    - ACLGraph
    - piecewise
    - AscendC
    - register_fake
    - Meta
---

# What this workflow produces

A CUDA-oriented model becomes a `torch_npu` / `vllm-ascend` deployment only after four different migrations agree:

1. the dependency surface is inventoried and the target stack is frozen;
2. CUDA kernels and collectives are replaced by NPU-native implementations;
3. custom operators have Meta kernels and eager numerical parity;
4. the decode path survives ACLGraph capture and the served endpoint passes a request gate.

Skipping stage 2 produces import errors. Skipping stage 3 produces graph breaks or a graph that silently omits the custom operator. Skipping stage 4 produces a service that runs eagerly but never uses the intended graph path, or replays stale metadata.

The workflow is empirical in the sense that every stage has been exercised on live Ascend hardware in the receipts recorded in the provenance ledger. It is not one controlled end-to-end migration of a single CUDA model, and it does not establish a universal speedup.

# Stage 1 - Inventory the CUDA surface and freeze the target matrix

Do not start by editing imports. Start by making the migration finite.

## 1.1 Inventory every CUDA-only dependency

Search the model, the serving plugin and the custom-extension build:

```bash
rg -n "flash_attn|causal_conv1d|fla|triton|torch\.cuda|CUDAGraph|nccl|NCCL|xformers|apex|bitsandbytes" .
rg -n "cpp_extension|CUDAExtension|load_inline|cuda" setup.py CMakeLists.txt pyproject.toml
```

Classify each hit:

| Class | Meaning | Action |
|---|---|---|
| Native `torch` op | Runs on `torch_npu` without a CUDA kernel | Keep, but verify dtype/layout behavior |
| Vendor NPU op | Has a documented `torch_npu` or `vllm-ascend` equivalent | Replace the call, preserve semantics |
| Custom CUDA kernel | Compiled CUDA/C++ extension | Port to AscendC or Triton-Ascend; add torch binding and Meta |
| Collective | NCCL process group or CUDA-specific communication | Replace with HCCL and re-test rank symmetry |
| Graph API | `torch.cuda.CUDAGraph`, CUDA stream capture | Replace with `torch.npu.NPUGraph` / `npugraph_ex` / ACLGraph |
| Removable | Training-only, CUDA-only diagnostics, unused imports | Remove before porting |

Do not treat "imports successfully" as "supported". A Python module can import on CPU while its extension has no NPU kernel.

## 1.2 Freeze the target stack

Record exact values, not "latest":

- `torch` and `torch_npu` package versions from the vendor compatibility table;
- CANN toolkit and kernels/ops versions;
- the active SoC from `acl.get_soc_name()` and the mapping to the build target;
- the `vllm-ascend` commit and the `vllm` commit;
- the driver/firmware pair;
- Python minor, host architecture and C++ ABI.

The existing build-entrypoint unit in this repository covers the `SOC_VERSION` / `npu-smi` packaging boundary. Use it before building any custom kernel.

## 1.3 Capture a CUDA-side reference

Before replacing anything, record on a fixed prompt set:

- tokenized input ids and generation parameters;
- output token ids for greedy decoding;
- per-layer or per-operator tensors for the operators being ported;
- dtype, shape, layout and tolerance;
- baseline latency separately for prefill and decode.

The reference must be reproducible without the CUDA machine. Store the inputs and expected outputs, not only a screenshot or a summary.

## Stage 1 exit criteria

- Every CUDA-only hit has an owner and a disposition.
- The target stack is pinned to exact versions and SoC.
- A CPU or stored CUDA reference exists for each ported operator.
- The build target and kernel package plan are known.

# Stage 2 - Replace kernels and collectives with NPU-native paths

Replace one surface at a time and keep the eager path runnable after each change.

## 2.1 `flash_attn`

Map CUDA FlashAttention to the NPU fused-attention family:

| CUDA surface | NPU path | Contract to preserve |
|---|---|---|
| `flash_attn_func` / `flash_attn_varlen_func` | `torch_npu.npu_fusion_attention` or `torch_npu.npu_fused_infer_attention_score` | Q/K/V layout, causal mask, softmax scale, dropout, varlen sequence lengths |
| Decode attention | `torch_npu.npu_incre_flash_attention` or the fused-infer decode path | Query sequence length 1, KV length, block table, actual sequence lengths |
| Sparse / sliding-window attention | `torch_npu.npu_sparse_flash_attention` or the model's documented sparse path | Sparse mode, mask shape, block size, KV cache layout |

Do not copy the CUDA function signature into the NPU call. The NPU operators use different layout names, mask conventions and sequence-length types. Check the installed operator schema for:

- `input_layout` (`BSH`, `SBH`, `BSND`, `BNSD`, `TND`, and the fused-infer composite layouts);
- `sparse_mode` and the corresponding `atten_mask` shape;
- whether `actual_seq_lengths` is a `List[int]` or a `Tensor`, and its dtype;
- whether the decode path expects a block table, a slot mapping, or both.

A shape-valid call with the wrong mask mode can produce plausible but wrong logits. Validate prefill, decode, varlen boundaries and the first/last KV block separately.

## 2.2 `causal_conv1d`

The common vLLM Mamba/linear-attention path uses a fused causal convolution with a recurrent cache. On Ascend it becomes either:

- a vLLM-Ascend AscendC custom operator exposed as `torch.ops._C_ascend.npu_causal_conv1d_custom`, or
- a Triton-Ascend fallback under the model's mamba package.

The port must preserve:

- the convolution state/cache layout and the cache-index dtype;
- the distinction between prefill (`run_mode`) and decode/update;
- `activation_mode` (`none` or `silu`);
- `pad_slot_id` and `null_block_id` semantics;
- `max_query_len` and the query-start-location table.

The binding and Meta implementation are part of the port, not an afterthought. In the recorded causal-conv1d port, the change touched `csrc/torch_binding.cpp`, `csrc/torch_binding_meta.cpp` and an NPU-versus-CPU regression test in the same commit. That is the correct shape of the change: kernel, schema, Meta and test together.

## 2.3 `fla` (flash-linear-attention)

`fla` kernels are often Triton or CUDA. On Ascend:

1. look for an existing AscendC port under the `fla` tree or the model's `ops/ascendc` directory;
2. if none exists, port the kernel to AscendC, keeping the CPU/FP32 reference as the numerical specification;
3. if the Triton source is simple and supported, compile it with Triton-Ascend before writing a new AscendC kernel;
4. add the torch binding and Meta implementation.

The recorded FLA KDA validation is the acceptance pattern:

- 16 C128/V256 forward cases produced finite output;
- absolute max error versus CPU FP32 was `5.21e-6` to `1.55e-5`;
- cosine similarity was `0.9999845` to `0.9999936`;
- A2 and A5 four-stage runs matched the fla-org Triton reference with zero bad-ratio elements.

Do not accept "kernel runs" as parity. Use `rel_p99`, absolute error, cosine and a finite check, and inspect the worst points rather than only the mean.

## 2.4 Triton operators

Triton on Ascend is not CUDA Triton with a different backend. Block sizes, shared-memory assumptions, warp-level primitives and autotune spaces are hardware-specific.

- Compile with the Triton-Ascend toolchain supported by the installed CANN/torch_npu stack.
- Re-tune block sizes and num-warps for the Ascend core and UB budget.
- If a primitive is unsupported, port that operator to AscendC rather than forcing a fallback.
- Keep the Triton path and the AscendC path behind one dispatch point, and test both if both ship.

A Triton kernel that compiles but silently falls back to a slow path is a performance bug, not a correctness success.

## 2.5 `nccl` collectives

Replace NCCL with HCCL:

```python
import torch.distributed as dist
dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
```

Requirements:

- all ranks create the same groups in the same order with the same sizes;
- `torchrun` or the serving launcher supplies `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT`;
- for multi-node, set the HCCL interface/address variables required by the image (for example the HCCL socket interface and buffer size);
- do not let a conditional branch skip a collective on some ranks; an idle rank still participates or the peers block until timeout;
- test a small all-reduce/reduce-scatter before the model forward.

The recorded 16-rank service run used TP8xDP2 and TP16 and matched greedy outputs token-by-token across the two topologies. That is the kind of collective-parity check the port needs.

## 2.6 `torch.cuda.CUDAGraph`

Replace CUDA graph capture with the NPU graph path:

| CUDA | NPU |
|---|---|
| `torch.cuda.CUDAGraph()` | `torch.npu.NPUGraph()` |
| `torch.cuda.graph(g)` | `with torch.npu.graph(g):` |
| `torch.cuda.make_graphed_callables` | `torch.npu.make_graphed_callables` |
| CUDA graph capture in a serving engine | `torch.compile(..., backend="npugraph_ex")` or the engine's ACLGraph mode (`cudagraph_mode`) |

The capture rules are stricter than eager execution:

- fixed shapes and strides for every captured bucket;
- persistent input and metadata buffers, updated in place before replay;
- no allocation, no `.item()`, no `.cpu()`, no host synchronize inside the captured region;
- padded tokens must not address live KV blocks;
- every custom operator in the captured region must have a Meta kernel.

## Stage 2 exit criteria

- The model imports and runs eagerly with no CUDA-only dependency.
- Each replaced operator has a CPU/FP32 or stored CUDA reference.
- Collectives complete on the target topology.
- A single-device decode step runs under the NPU graph path before multi-rank capture is attempted.

# Stage 3 - Register Meta kernels and establish eager parity

Stage 3 is where most "it runs" migrations become correct.

## 3.1 Register a Meta / FakeTensor kernel for every custom operator

Follow the custom-op SOP in this repository. The minimum:

```python
@torch.library.custom_op("mylib::npu_x", mutates_args=())
def npu_x(x: torch.Tensor) -> torch.Tensor:
    return torch.ops._C_ascend.npu_x(x)

@npu_x.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

For a C++ operator, add `TORCH_LIBRARY_IMPL(<ns>, Meta, m)`. For a TorchAir GE path, add the FX-to-GE converter as well. The Meta kernel must not call the device, read a host scalar or allocate NPU memory.

## 3.2 Prove graph membership, not just compilation

Compile a small module with `fullgraph=True` and assert that the custom operator appears in the graph. A successful compile with the operator left in eager is a silent migration failure.

```bash
TORCH_LOGS=graph_breaks,recompiles python -m pytest <target-test>
```

For `vllm-ascend`, the service log records `enable_npugraph_ex` and the compiled graph range. Check that the operator is in the captured graph rather than only in the eager path.

## 3.3 Numerical parity

For each ported operator:

- run the CPU/FP32 reference and the NPU output on the same inputs;
- check finite values before comparing;
- assert shape, dtype and layout;
- use `torch.testing.assert_close` with an operator-specific tolerance;
- report absolute max, relative p99 and cosine for attention/linear-attention kernels;
- test prefill and decode separately;
- test varlen boundaries and the first/last KV block;
- test at least one non-contiguous or nonzero-offset input.

The FLA KDA work is a good example of reporting the distribution rather than a single average: 16 cases, finite output, absolute max in the `1e-5` range, cosine above `0.99998`, and a separate zero-bad-ratio comparison against the upstream Triton reference.

## Stage 3 exit criteria

- Every custom operator has a Meta kernel.
- `torch.compile(fullgraph=True)` includes the operator.
- Eager output matches the reference within the declared tolerance.
- No NaN/Inf in the tested dtypes and shapes.
- The test covers decode, not only prefill.

# Stage 4 - Capture, serve and gate

## 4.1 Capture the decode path

Use the serving engine's production graph mode and capture sizes. For `vllm-ascend` this is typically `FULL_DECODE_ONLY` with an explicit capture-size list:

```json
{
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [8, 16, 32, 64, 128],
  "cudagraph_num_of_warmups": 1
}
```

Or, for a standalone model, `torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)`.

Capture invariants:

- every steady-state decode step maps to a captured bucket;
- no recompilation after warmup;
- input `data_ptr()` values are stable across replays;
- metadata tensors are copied in place before replay;
- padded tokens use a sentinel that cannot touch a live cache block;
- the capture pool is sized for the real peak, not the steady-state average.

If capture fails with `Alloc sq cq fail` or an equivalent stream-resource error, reduce the capture-size set or the number of communication groups. Do not treat it as HBM OOM and start freeing memory.

## 4.2 Serve and probe

Start the service with the exact pinned stack and a fixed request matrix:

```bash
curl -k https://localhost:<port>/health
curl -k https://localhost:<port>/v1/models
```

Run:

- short prompts, long prompts and boundary lengths around the capture buckets;
- greedy decoding for token-for-token comparison;
- sampled decoding separately, with finite-logprob checks;
- concurrent bursts at the target batch size;
- a topology comparison (for example TP8xDP2 versus TP16) where the model supports it;
- a health check after every burst.

The recorded A3 run completed 896/896 requests at concurrency 128 with 131072 output tokens and a healthy endpoint. The recorded DSV4 service completed 107 GPQA-diamond cases with 98 correct and zero errors under `enable_npugraph_ex`. These are the kinds of aggregate gates to record.

## 4.3 Gate and rollback

A migration is not accepted until:

- correctness: reference parity for each ported operator and a task-level accuracy check;
- graph: no unexpected recompilation, custom ops present in the captured graph, replay matches eager;
- communication: all ranks complete the same collectives and produce matching greedy outputs where applicable;
- stability: sustained concurrency does not leak HBM or trigger preemption storms;
- performance: prefill and decode latency are reported separately, with the baseline and the measurement scope;
- rollback: the previous eager path or the previous kernel can be restored without a code rewrite.

Keep the failed capture log, the pre-change config and the exact stack versions. A clean retry without them is not evidence.

## Stage 4 exit criteria

- The service answers `/health` and `/v1/models`.
- The request matrix passes with no crash and no non-finite output.
- ACLGraph replay matches eager at the same inputs.
- The stack, capture sizes and measured scope are recorded.
- The limits (model, dtype, topology, sequence length) are written down.

# Evidence and limits

The workflow stages are supported by these live receipts:

| Receipt | Stack | Observed |
|---|---|---|
| 16-rank A3 service run | `torch_npu` 2.10.0.post4.dev20260715, CANN 9.1.0, `vllm-ascend` 0.27.1rc | FULL_DECODE_ONLY ACLGraph; 896/896 requests at concurrency 128; 131072 output tokens; TP8 and TP16 greedy outputs matched |
| A3 ACLNN custom-op regression | same A3 stack | 21 passed, 5 skipped for the KDA custom-op path |
| A5 FlashMLA C8 qualification | CANN 9.2.0 B035, Ascend950 | 65 BF16 gates passed; C8 kernel and LSE gates passed; ACLGraph replay across `[0,127]`, `[0,0]`, `[128,1024]`, `[129,1025]` |
| A2/A5 FLA KDA validation | A2 and A5 | 16 C128/V256 cases finite; cosine `0.9999845` to `0.9999936`; zero bad-ratio versus the Triton reference |
| DSV4 service run | `torch_npu` 2.10.0.post4.dev20260715, CANN 9.1.0, `vllm` 0.27.1 | `enable_npugraph_ex`; 107 GPQA-diamond cases, 98 correct, zero errors |

Limits:

- This is a synthesis of several runs, not a controlled single-model migration.
- The receipts cover selected models, dtypes and topologies. They do not prove coverage for an arbitrary CUDA model.
- Operator-level timings are not end-to-end serving throughput.
- A successful graph capture does not prove numerical parity; the eager/reference gate remains mandatory.
- The causal-conv1d port is a source-and-test receipt, not a separate live performance claim.
