---
schema: mindie-graded-unit/1
id: unit-cann-toolkit-ops-nnal-install-order-and-acl-verify-sop
revision: 1
title: "CANN Toolchain Installation Order: Toolkit -> Kernels -> NNAL (ATB) Layering and Environment Verification SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: deployment/cann-toolchain/toolkit-ops-and-nnal/install-order-and-acl-verify
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-ascendc-opdef-geir-proto-and-aclnn-two-stage-abi-axioms
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
    - Ascend 310P
  domain: deployment
triggers:
  error_signatures:
    - "EZ1001"
    - "Cannot find opp"
    - "libatb.so: cannot open shared object file"
    - "acl.init()"
  keywords:
    - "cann-toolkit"
    - "cann-kernels"
    - "cann-nnal"
    - "set-env-sh"
    - "ascend-opp-path"
provenance:
  source_type: source_contract
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/cann-operator-env-config/SKILL.md
  evidence_refs:
    - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/cann-operator-env-config/SKILL.md#L1-L165
  supersedes: []
---

## Falsifiable Claim

Deploying CANN for LLM inference or AscendC operator development requires installing packages in a strict three-stage dependency order: (1) `Ascend-cann-toolkit`, (2) SoC-matched `Ascend-cann-kernels-<soc>`, and (3) `Ascend-cann-nnal` (ATB neural network acceleration library), followed by sourcing **both** `/usr/local/Ascend/ascend-toolkit/set_env.sh` and `/usr/local/Ascend/nnal/atb/set_env.sh`. Installing `kernels` before `toolkit`, installing a mismatched `<soc>` kernel package (`910b` vs `910_93` vs `310p`), or omitting `nnal/atb/set_env.sh` causes missing `ASCEND_OPP_PATH` (`EZ1001` operator binary not found) or `libatb.so` dynamic linker failures.

## Three-Stage Package Dependency & SoC Mapping

| Stage | Package Pattern | Role & Mandatory Precondition |
| :--- | :--- | :--- |
| **1. Base Toolkit** | `Ascend-cann-toolkit_<ver>_linux-<arch>.run` | Installs ACL runtime (`libascendcl.so`), GE compiler, `bisheng`/`ccec` compiler toolchain, `msprof`, and creates the base `/usr/local/Ascend/ascend-toolkit/latest/opp` directory structure. |
| **2. SoC Precompiled Kernels** | `Ascend-cann-kernels-<soc>_<ver>_linux-<arch>.run` | Populates precompiled `aclnn` / TBE / AscendC binary kernels into `opp/built-in/op_impl/ai_core/tbe/kernel/<soc>`. **Must match `npu-smi info` Chip Name**: `910b` for Ascend 910B1–B4, `910_93` (or `a3`) for Ascend 910C, `310p` for Ascend 310P. |
| **3. NNAL (ATB Library)** | `Ascend-cann-nnal_<ver>_linux-<arch>.run` | Installs Ascend Transformer Boost (`atb` and `asdops`) under `/usr/local/Ascend/nnal/atb`. Required by MindIE LLM, `vllm-ascend`, and fused attention/MoE kernels. |

## Standard Operating Procedure (SOP)

### Step 1: Execute Sequential Silent Installation

Verify that all three `.run` packages share the exact same CANN release version (`<ver>`) and target CPU architecture (`aarch64` or `x86_64`):

```bash
chmod +x Ascend-cann-*.run

# 1. Install CANN Toolkit first
./Ascend-cann-toolkit_*_linux-*.run --install --quiet

# 2. Source toolkit environment before installing Kernels and NNAL
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. Install SoC-specific Kernels package (e.g., 910b or 910_93)
./Ascend-cann-kernels-*_linux-*.run --install --quiet

# 4. Install NNAL (ATB) package
./Ascend-cann-nnal_*_linux-*.run --install --quiet
```

### Step 2: Load Both Toolkit and NNAL Environments

In every shell session or container entrypoint script, source `ascend-toolkit/set_env.sh` **first** and `nnal/atb/set_env.sh` **second**:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Verify that the environment variables point to valid directories:
- `ASCEND_TOOLKIT_HOME` $\rightarrow$ `/usr/local/Ascend/ascend-toolkit/latest`
- `ASCEND_OPP_PATH` $\rightarrow$ `/usr/local/Ascend/ascend-toolkit/latest/opp`
- `ASCEND_AICPU_PATH` $\rightarrow$ `/usr/local/Ascend/ascend-toolkit/latest`
- `ATB_HOME_PATH` $\rightarrow$ `/usr/local/Ascend/nnal/atb/latest/atb/<cxx_abi_dir>` (selecting ABI 0 or ABI 1 matching PyTorch `_GLIBCXX_USE_CXX11_ABI`)

### Step 3: Three-Point Smoke Verification

Run the following three checks to confirm the toolchain, ACL runtime, and operator binary repository are healthy:

1. **Compiler & Tool Presence**:
   ```bash
   which ccec atc msprof
   ```
2. **ACL Runtime & Device Context Initialization**:
   ```bash
   python3 -c "import acl; ret = acl.init(); assert ret == 0, f'acl.init failed: {ret}'; print('ACL_OK')"
   ```
3. **PyTorch NPU & `aclnn` Operator Dispatch**:
   ```bash
   python3 -c "import torch, torch_npu; x = torch.randn(4, 4, device='npu:0', dtype=torch.float16); print((x @ x).shape)"
   ```
