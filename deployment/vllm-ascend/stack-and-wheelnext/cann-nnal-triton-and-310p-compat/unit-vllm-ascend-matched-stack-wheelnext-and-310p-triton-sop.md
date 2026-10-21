---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-matched-stack-wheelnext-and-310p-triton-sop
revision: 1
title: "vLLM-Ascend Stack Compatibility: CANN/NNAL/torch-npu Version Matrix, WheelNext Install, and 310P Triton SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: deployment/vllm-ascend/stack-and-wheelnext/cann-nnal-triton-and-310p-compat
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-vllm-ascend-build-entrypoint-and-soc-probe
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
    - Ascend 310P
  domain: deployment
triggers:
  error_signatures:
    - "COMPILED_WITH_CXX11_ABI"
    - "undef symbol: _ZN"
    - "No module named 'triton'"
    - "Ascend 310P does not support triton-ascend"
  keywords:
    - "vllm-ascend"
    - "wheelnext"
    - "triton-ascend"
    - "cann-nnal"
    - "310p-compatibility"
    - "cxx11-abi"
provenance:
  source_type: source_contract
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/installation.md
  evidence_refs:
    - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/installation.md#L1-L260
  supersedes: []
---

## Falsifiable Claim

Running `vllm-ascend` requires strict lockstep across five components: HDK Driver/Firmware ($\ge 24.1.0$), CANN Toolkit + Kernels + NNAL ($\ge 8.5.0$), PyTorch + `torch-npu` ($\ge 2.7.1$ / $2.9.0$), `triton-ascend` (on Atlas A2/A3 only), and matched `vllm` + `vllm-ascend` releases. On Atlas 300I (`Ascend 310P`), `triton-ascend` is unsupported and must not be enabled; on x86_64 vs. aarch64 hosts, mismatched C++11 ABI (`_GLIBCXX_USE_CXX11_ABI`) between `torch`, `vllm_ascend_C`, and `nnal/atb` triggers undefined symbol errors (`_ZN...`) during custom operator loading.

## Software & Hardware Compatibility Matrix

| Series / SoC | Supported `vllm-ascend` Path | `triton-ascend` Support | Mandatory CANN Packages |
| :--- | :--- | :--- | :--- |
| **Atlas A2 (`Ascend 910B1–B4`)** | Full support (Eager, ACLGraph `npugraph_ex`, FIA, MLA, MoE MC2) | Supported (`triton-ascend` installed by default) | `toolkit` + `kernels-910b` + `nnal` |
| **Atlas A3 (`Ascend 910C` / `910_93`)** | Full support (SuperPod HCCS, dual-Die 16-rank per 8-card node, MoE EP) | Supported (`triton-ascend` installed by default) | `toolkit` + `kernels-910_93` (or `a3`) + `nnal` |
| **Atlas 300I (`Ascend 310P`)** | Supported for dense/quantized inference with FP16/INT8 | **Unsupported** (do not install or invoke `triton-ascend` kernels; use `dtype="float16"` as BF16 is unsupported on 310P) | `toolkit` + `kernels-310p` + `nnal` |

## Standard Operating Procedure (SOP)

### 1. Pre-Flight CANN + NNAL Environment Setup

Before installing or running `vllm-ascend`, ensure `toolkit`, `kernels`, and `nnal` are installed and both environment scripts are sourced:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 2. Fast Precompiled Installation via `pip` / WheelNext Index

To avoid compiling `vllm` and `vllm_ascend_C` C++/AscendC extensions from scratch (or to resolve architecture-specific wheel variants automatically):

```bash
# Option A: Standard PyPI release install (installs matched vllm + vllm-ascend + torch-npu)
pip install vllm-ascend

# Option B: WheelNext / prebuilt wheel index for exact CANN/PyTorch variant resolution
pip install --extra-index-url https://wheels.vllm.ai/ascend/ vllm-ascend
```

### 3. Source Compilation Contract (When Developing Custom C++/AscendC Ops)

When building `vllm-ascend` from source (`pip install -v -e . --no-build-isolation`):
1. Ensure `ccec` (`which ccec`) from CANN Toolkit is on `PATH` so `CMake` compiles `vllm_ascend_C` AscendC kernels (`mla`, `moe`, `pos_encoding`).
2. Match `_GLIBCXX_USE_CXX11_ABI` between PyTorch (`python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`) and `ATB_HOME_PATH` (`cxx_abi_0` vs. `cxx_abi_1`):
   - If `torch._C._GLIBCXX_USE_CXX11_ABI == False` (typical on `x86_64` manylinux wheels), `set_env.sh` in `nnal/atb` must select `cxx_abi_0`, and `COMPILE_CUSTOM_KERNELS=1` will pass `-D_GLIBCXX_USE_CXX11_ABI=0`.
3. On **Ascend 310P**:
   - Explicitly pass `--dtype float16` (never `bfloat16`) and disable Triton-dependent features (such as Triton-based FLA/Lightning Indexer or AutoFuse Triton codegen).
