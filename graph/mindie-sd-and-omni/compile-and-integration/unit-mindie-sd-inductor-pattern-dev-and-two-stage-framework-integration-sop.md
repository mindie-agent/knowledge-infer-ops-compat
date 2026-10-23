---
schema: mindie-graded-unit/1
id: unit-mindie-sd-inductor-pattern-dev-and-two-stage-framework-integration-sop
title: MindIE-SD MindieSDBackend Inductor pattern-dev lifecycle and two-stage third-party framework-integration SOP
tier: L2
category: sop
evidence_strength: software_contract
domain_path: graph/mindie-sd-and-omni/compile-and-integration
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
conditions:
  cann_versions:
    - ">=8.5.0"
  hardware:
    - Ascend 910B (Atlas A2)
    - Ascend 910C (Atlas A3)
    - Ascend 950PR/DT
  workloads:
    - PyTorch Inductor pattern matcher development, ViewCopy/InplaceCopy elimination, and MindIE-SD integration into vLLM-Omni, LightX2V, DiffSynth-Engine, and Cache-DiT
triggers:
  error_signatures:
    - "torch.compile(submodule, backend=MindieSDBackend()) has 0 pattern hits when return value is not assigned"
    - "BACKEND_MATCH failure triggers repeated Dynamo recompilation until recompile_limit fallback"
    - "hcom_alltoallv variable-split collective inside compiled graph degrades latency by >10x"
  keywords:
    - pattern-dev
    - framework-integration
    - MindieSDBackend
    - GraphPatternEntry
    - register_replacement
    - _compiled_call_impl
    - hccl_eager
    - compare_profiles
    - analyze_copy_kernels
provenance:
  source_type: architecture_and_runtime_guide
  source_locator: Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/pattern-dev/SKILL.md
  source_digest: 6de10998437e68775d01a16c3ce025f75c5a744b81acc96e6b4ffc663b7e982c
  evidence_refs:
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/pattern-dev/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:.agents/skills/framework-integration/SKILL.md
    - Ascend/MindIE-SD@9fe772dfb90696e21b5e5d4e92ca07dffab84cf4:docs/zh/features/compilation.md
  supersedes: []
revision: 1
---
## Standard Operating Procedure

1. **Enforce the Two-Stage Framework Integration Discipline (`API-First -> Compile`)**:
   - **Stage 1 — Runtime API Injection**: Inject `mindiesd` single operators (`rms_norm`, `rotary_position_embedding`, `layernorm_scale_shift`, `attention_forward`, `sparse_attention`) at the smallest possible seam—either via the framework's operator registry (`rms_type` / `rope_type` in `LightX2V`), CustomOp dispatch (`vLLM-Omni`), or direct model-layer replacement (`DiffSynth-Engine`). Verify signature/layout compatibility, same-seed numerical parity, and standalone Stage 1 speedup before attempting graph compilation.
   - **Stage 2 — `MindieSDBackend` Graph Compilation**: Only after Stage 1 exits cleanly, integrate `torch.compile(..., backend=MindieSDBackend())` to capture multi-operator fusion chains and eliminate functionalization copies.
   - **When to Stop at Stage 1 (Evidence-Driven Rollback)**: If eager single operators already cover the hot path and adding `torch.compile` yields negative or sub-noise ($<3\%$) wall-clock change, or alters numerical/seed semantics (non-bitwise output on `vllm-omni` image/BF16 paths), **roll back Stage 2 and keep compile default-OFF**.

2. **Avoid Cross-Framework `torch.compile` & Collective Integration Traps**:
   - **Assign `_compiled_call_impl`**: Calling `torch.compile(submodule, backend=...)` without assigning the return value or writing `submodule._compiled_call_impl` (or calling `submodule.compile()`) is a silent no-op with 0 pattern hits.
   - **Reuse a Singleton `MindieSDBackend` Instance**: Instantiating a new `MindieSDBackend()` per block or per step triggers Dynamo `BACKEND_MATCH failure`, recompiling until `recompile_limit` is hit and silently falling back to eager mode.
   - **Keep Sequence-Parallel Collectives in Eager (`@torch.compiler.disable` / `hccl_eager`)**: Compiling Ulysses/Ring collectives into the Inductor FX graph can cause Dynamo to infer dynamic `split_sizes` and lower uniform `AlltoAll` onto the variable-split `hcom_alltoallv` path (degrading collective latency by $>10\times$). Keep collectives outside the compiled region via `@torch.compiler.disable` or `seq_p_a2a_backend="hccl_eager"`.

3. **Author Inductor Fusion Patterns via the Phase 1–7 `pattern-dev` Lifecycle**:
   - **Choose `register_replacement` vs `GraphPatternEntry` (Never Hand-Write Custom FX Graph Traversal Passes)**:
     - **Path A (`register_replacement`)**: Use for functional APIs and `nn.Module` parameters. Pass `self.weight` / `self.bias` as explicit **input arguments** to `inputs()`, `pattern()`, and `replacement()` (in the pre-freeze Inductor window, module weights appear as placeholders rather than `get_attr`).
     - **Path B (`GraphPatternEntry` + Manual Handler)**: Use when the pattern spans **dynamic shape nodes** (e.g., `view([1, S, 2*F])` where sequence length $S$ varies across batch/branches in the same graph, causing trace-based `register_replacement` to bake in a constant $S$ and miss). Construct a `CallFunction` tree with `Arg()` leaves, `Ignored()` view dimensions, shared child node instances with `_users=MULTIPLE`, and a handler that walks the producer chain from `match.output_node()`.
   - **Three-File Registration (`Phase 3`)**: Always register in (1) `mindiesd/compilation/patterns/__init__.py` (`__all__`), (2) `mindiesd/compilation/passes/__init__.py` (`pattern_registry`), and (3) `mindiesd/compilation/compiliation_config.py` (`FusionPatterns` `enable_{model}_{op}: bool = True`).

4. **Verify Real Graph Hits, Kernel Diffs, and Copy Elimination (`Phase 4–7`)**:
   - **Unit Test $\ne$ Full-Model Hit**: Validate pattern hits directly on the full model's `dummy-run` traced FX graph (never rely solely on standalone `symbolic_trace` / `make_fx` toy graphs).
   - **Three-Layer Exit Evidence (`Phase 6`)**: Require (1) `PatternMatchPass replace N` count $> 0$, (2) `compare_profiles.py --eager eager/kernel_details.csv --compile compile/kernel_details.csv` showing the fused kernel appearing and replaced eager kernels disappearing, and (3) same-seed eager-vs-compile numerical parity (bitwise `0.0` for lossless algebraic rewrites; quantized dispersion tolerance for FP8/MXFP8 fusions such as `mm_swiglu_mxquant`).
   - **Copy Elimination (`Phase 7`)**: Audit `InplaceCopy` / `ViewCopy` counts with `analyze_copy_kernels.py` to ensure functionalization does not introduce extra HBM copy kernels that offset fusion gains.
