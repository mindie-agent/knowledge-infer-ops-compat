---
schema: mindie-graded-unit/1
id: unit-torch-npu-cpp-extension-aclnn-binding-and-autograd-sop
title: PyTorch NPU C++ Extension Build, EXEC_NPU_CMD aclnn Binding, and Custom Autograd
  Function SOP
domain_path: registration/custom-ops/torch-cpp-extensions
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  extension_toolchain: torch_npu.utils.cpp_extension
  registration_layer: cpp-extension-and-autograd
  runtime: torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-torch-npu-custom-op-registration-and-faketensor-meta-sop
supersedes: []
triggers:
  error_signatures:
  - NpuExtension BuildExtension cannot find -lcust_opapi
  - c10_npu::getCurrentNPUStream stream mismatch in C++ extension
  - torch.autograd.Function backward missing PrivateUse1 registration
  - Tensor(a!) schema alias violation in TORCH_LIBRARY
  keywords:
  - NpuExtension BuildExtension
  - c10_npu::getCurrentNPUStream
  - ACLRT_LAUNCH_KERNEL
  - torch::autograd::Function
  - Tensor(a!) inplace schema
  - rpath cust_opapi
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:ops/torch-ascendc-op-extension+cannbot-knowledge:torch_extensions
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/torch-ascendc-op-extension/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/torch-ascendc-op-extension/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/torch_extensions/overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/torch_extensions/extension_build_setup.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/torch_extensions/autograd_function.md
  source_type: curated_repo
  source_digest: 3c09aeabd35c0a9990d5a4a4a7127866b1b9bb62b88dbfdf23cd81ac30223059
---

# PyTorch NPU C++ Extension Build, EXEC_NPU_CMD aclnn Binding, and Custom Autograd Function SOP

## 1. Selecting the Right AscendC PyTorch Integration Mode

`cannbot-skills/ops/torch-ascendc-op-extension` defines four canonical integration modes for binding AscendC kernels into `torch_npu`:

| Mode | Dispatch Mechanism | Compilation Toolchain | Target Use Case |
|---|---|---|---|
| **Mode 1: Direct Kernel Launch (`<<<...>>>` / `ACLRT_LAUNCH_KERNEL`)** | C++ wrapper extracts `c10_npu::getCurrentNPUStream().stream()` and launches the `.cce` kernel directly without OPP installation | `bisheng` + `torch_npu.utils.cpp_extension` | Rapid single-file AscendC kernel prototyping without `msopgen` OPP packaging. |
| **Mode 2: `aclnn` C++ Extension (`EXEC_NPU_CMD`)** | C++ wrapper invokes `EXEC_NPU_CMD(aclnn<Op>, ...)` against an installed OPP package (`libcust_opapi.so`) | `msopgen` OPP build + `NpuExtension` | Production inference/training ops needing full `OpDef`, tiling key dispatch, and multi-dtype support. |
| **Mode 3: `torch.library` Dispatcher (`TORCH_LIBRARY` + `Meta`)** | Registers schema under `torch.ops.<ns>.<op>` with `PrivateUse1` and `Meta` backends | C++ `TORCH_LIBRARY_IMPL` or Python `@torch.library.custom_op` | **Mandatory for `torch.compile`, `npugraph_ex`, and `vllm-ascend` piecewise/ACLGraph capture.** |
| **Mode 4: Custom Autograd (`torch::autograd::Function<T>`)** | Binds forward `aclnn<Op>` and backward `aclnn<Op>Grad` with `ctx->save_for_backward` | C++ `torch::autograd::Function` + `AutogradPrivateUse1` | Training or fine-tuning operators requiring differentiable backward passes. |

## 2. `setup.py` Build and RPATH Configuration SOP (`NpuExtension`)

When linking against a custom OPP package (`vendors/<vendor>/op_api`), hardcode the runtime library search path (`-Wl,-rpath`) in `setup.py` so importing the extension does not fail with `libcust_opapi.so: cannot open shared object file`:

```python
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

ascend_home = os.environ["ASCEND_HOME_PATH"]
vendor_api = os.path.join(ascend_home, "opp", "vendors", "customize", "op_api")

setup(
    name="custom_npu_ops",
    ext_modules=[
        NpuExtension(
            name="custom_npu_ops._C",
            sources=["csrc/custom_ops_binding.cpp"],
            include_dirs=[os.path.join(vendor_api, "include")],
            library_dirs=[os.path.join(vendor_api, "lib")],
            libraries=["cust_opapi", "nnopbase"],
            extra_link_args=[f"-Wl,-rpath,{os.path.join(vendor_api, 'lib')}"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

## 3. Stream, Contiguity, and In-Place Schema Rules

1. **Always Acquire the Active `torch_npu` Stream**: Inside Mode 1 direct kernel wrappers, pass `c10_npu::getCurrentNPUStream().stream(false)` (never pass `nullptr` default stream or call `aclrtSynchronizeStream` inside the op wrapper, which breaks async pipelining and `ACLGraph` capture).
2. **Enforce Input Contiguity Before Raw Pointer Extraction**: When passing `tensor.data_ptr()` to Mode 1 kernels, call `at::Tensor x_contig = x.contiguous();` first unless the kernel explicitly accepts runtime strides.
3. **Declare Aliasing Annotations (`Tensor(a!)`) on In-Place Ops**: In `TORCH_LIBRARY` schemas, any in-place mutation must annotate its mutated parameter and return value as `my_inplace_op_(Tensor(a!) x, Tensor y) -> Tensor(a!)` so AOTAutograd and `npugraph_ex` `reinplace` passes track storage mutation accurately.
