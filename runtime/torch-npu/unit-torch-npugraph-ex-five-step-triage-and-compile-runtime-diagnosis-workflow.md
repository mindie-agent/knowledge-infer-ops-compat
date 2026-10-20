---
schema: mindie-graded-unit/1
id: unit-torch-npugraph-ex-five-step-triage-and-compile-runtime-diagnosis-workflow
revision: 1
tier: L3
category: workflow
domain_path: runtime/torch-npu
evidence_strength: software_contract
title: Five-step progressive backend triage and compile-vs-runtime root-cause diagnosis workflow for torch.compile npugraph_ex
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
  source_digest: 1af3cb6c1ba18e011ee1e06952d49c071b4db7fb0985928bbd518ed84ce71a80
  evidence_refs:
    - docs/evidence/cannbot-graph-and-ge-provenance.json
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/torch-npugraph-ex-dfx-triage/SKILL.md#L15-L223"
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/torch-npugraph-ex-compile-error-diagnosis/SKILL.md#L11-L83"
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/torch-npugraph-ex-runtime-error-diagnosis/SKILL.md#L11-L91"
  supersedes: []
triggers:
  error_signatures:
    - "torch._dynamo.exc.Unsupported"
    - "torch._dynamo.exc.BackendCompilerFailed"
    - "npu_fx_compiler failed"
    - "ACL graph capture failed"
    - "force_eager=True succeeds but npugraph_ex fails"
    - "aclnnXxx failed during npugraph_ex replay"
  keywords:
    - npugraph_ex
    - torch.compile
    - torchair
    - aot_eager
    - force_eager
    - TORCH_COMPILE_DEBUG
    - dfx-triage
    - graph break
---
When diagnosing crashes, graph breaks, or post-capture replay failures under `torch.compile(..., backend="npugraph_ex")` on Ascend NPU, execute the deterministic 5-step progressive backend isolation workflow before inspecting source repositories (`graph/torch-npugraph-ex-dfx-triage/SKILL.md:26-223`).

## 1. Five-Step Progressive Backend Triage Workflow

Never guess the root cause from a single `npugraph_ex` wrapper traceback. Generate 5 minimally modified entry scripts that preserve model initialization, seeds, and non-backend compile kwargs (`fullgraph`, `dynamic`, `mode`), and run them sequentially until the first failing step stops execution:

| Step | Execution Mode | Debug Switches | Layer Isolated by First Failure |
|---|---|---|---|
| **Step 1** | Pure Eager (strip outer `torch.compile`) | `TORCH_LOGS="+all"` | Model / input / `torch_npu` Eager operator or environment defect (not a graph compiler bug) |
| **Step 2** | `torch.compile(model, backend="eager", ...)` | `TORCH_LOGS="+all"` | TorchDynamo Python bytecode tracing / `graph break` / `Unsupported` |
| **Step 3** | `torch.compile(model, backend="aot_eager", ...)` | `TORCH_LOGS="+all"` | FX / AOTAutograd / Functorch decomposition or custom op `Meta` (`FakeTensor`) shape/dtype inference |
| **Step 4** | `torch.compile(model, backend="npugraph_ex", options={**opts, "force_eager": True})` | `TORCH_COMPILE_DEBUG=1` + `TORCH_COMPILE_DEBUG_DIR` | `npugraph_ex` FX entry lowering (`npu_fx_compiler`) prior to full graph optimization and capture |
| **Step 5** | `torch.compile(model, backend="npugraph_ex", options=opts)` | `TORCH_COMPILE_DEBUG=1` + `TORCH_COMPILE_DEBUG_DIR` | Full `npugraph_ex` optimization passes (`reinplace`, `replace_stream_event`), ACL graph capture, or post-capture replay |

- **Artifact Completeness Invariant**: Each executed step directory under `torch-npugraph-ex-triage-logs/<YYYYMMDD-HHMMSS>-pid<PID>/` must contain both `stdout_stderr.log` and `ascend_plog/` (`ASCEND_PROCESS_LOG_PATH`), plus `torch_compile_debug/` for Steps 4 and 5.

## 2. First-Causal-Error Anchoring & Compile-vs-Runtime Routing

Always anchor on the **earliest causal exception** that explains subsequent cascade failures, ignoring final wrapper rethrows or secondary stream-sync/task-abort errors:

1. **Route to Compile-Time Diagnosis (`torch-npugraph-ex-compile-error-diagnosis/SKILL.md:11-83`)**:
   - Triggered when Steps 1–4 fail, or Step 5 fails during initial compilation/capture with `torch._dynamo`, `Unsupported`, `BackendCompilerFailed`, `Meta`, `FakeTensor`, `npu_fx_compiler`, or ACL graph capture errors.
   - **Source Localization Order**: Read local installed packages (`site-packages/torchair/`, `site-packages/torch_npu/`, `site-packages/torch/`) at the exact frame reported in the traceback first; fall back by phase to upstream PyTorch (`torch/_dynamo`, `torch/fx`, `torch/_functorch`), TorchAir (`Ascend/torchair` `npugraph_ex`), or `Ascend/pytorch` (`torch_npu` Meta registrations).
2. **Route to Post-Capture Runtime Diagnosis (`torch-npugraph-ex-runtime-error-diagnosis/SKILL.md:11-91`)**:
   - Triggered when ACL graph capture succeeds (or the model runs $N$ steps) before failing with `aclnnXxx` / `aclop`, `HCCL` / `hccl error`, `stream` / `event` synchronization failure, `device side assert`, `segfault`, or device `OOM`.
   - **Timestamp Alignment & Subsystem Triage**: Align `stdout_stderr.log`, CANN `ascend_plog`, and HCCL logs by timestamp to isolate the first failing subsystem:
     - **Operator (`aclnn`/`aclop`)**: Inspect CANN `plog` error codes, operator input shapes/formats, and workspace sizes before checking `torch_npu` bindings.
     - **Stream/Event**: Check cross-stream dependency lowering and TorchAir `replace_stream_event` passes.
     - **Memory (`OOM`)**: Inspect ACL workspace allocation logs and `PYTORCH_NPU_ALLOC_CONF`.
3. **Route When All 5 Steps Complete Without Exception**:
   - Numerical mismatch between Step 1/4 and Step 5 → route to accuracy diagnosis (comparing `step4-npugraph_ex-force_eager/torch_compile_debug/` vs `step5-npugraph_ex/torch_compile_debug/`).
   - Latency regression or low device utilization → route to static FX tensor-move performance audit (`unit-torch-npugraph-ex-reinplace-and-tensor-move-performance-audit-sop`).
