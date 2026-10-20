---
schema: mindie-graded-unit/1
id: unit-torch-npu-custom-op-ascendc-direct-kernel-and-aclnn-mutator
title: PyTorch NPU Custom Op AscendC Direct Kernel Launch, aclnn Two-Stage Wrapper,
  and Meta Registration Mutator
domain_path: registration/custom-ops/torch-cpp-extensions
tier: L4
category: mutator
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  extension_toolchain: torch_npu.utils.cpp_extension
  registration_layer: custom-op-code-mutator
  runtime: torch_npu
inherits_from:
- unit-torch-npu-custom-op-registration-and-faketensor-meta-sop
- unit-torch-npu-cpp-extension-aclnn-binding-and-autograd-sop
supersedes: []
triggers:
  error_signatures:
  - OpCommand Run failed in custom C++ extension
  - EXEC_NPU_CMD_EXT workspace allocation failure
  - TORCH_LIBRARY_IMPL AutogradPrivateUse1 missing
  - custom op C++ template for aclnn and Meta registration
  keywords:
  - EXEC_NPU_CMD template
  - TORCH_LIBRARY_IMPL PrivateUse1 Meta
  - AutogradPrivateUse1 mutator
  - at_npu::native::OpCommand
  - custom_npu_ops C++ boilerplate
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:ops/torch-ascendc-op-extension/references/mode-2+mode-3+mode-4
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/torch-ascendc-op-extension/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/torch-ascendc-op-extension/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/torch_extensions/custom_ops_torch_library.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/torch_extensions/aclnn_cpp_extension.md
  source_type: curated_repo
  source_digest: 1f4c4516f8162f6b07d621d290d6f3219834cde74ed9bb1a6813b3a0f1bee790
---

# PyTorch NPU Custom Op AscendC Direct Kernel Launch, aclnn Two-Stage Wrapper, and Meta Registration Mutator

## 1. Complete C++ `TORCH_LIBRARY` + `PrivateUse1` (`EXEC_NPU_CMD`) + `Meta` + `Autograd` Mutator Template

Use the following canonical C++ implementation (`csrc/custom_ops_binding.cpp`) to expose a custom AscendC `aclnn` operator (`aclnnAddRmsNormCustom` and `aclnnAddRmsNormCustomGrad`) to PyTorch Eager, `torch.compile`, `npugraph_ex` (`ACLGraph` & `GE`), and `torch.autograd`:

```cpp
#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "op_ nursing_cmd.h" // Or project cmd_helper.h defining EXEC_NPU_CMD
#include "aclnn_add_rms_norm_custom.h"
#include "aclnn_add_rms_norm_custom_grad.h"

namespace custom_npu_ops {

// 1. PrivateUse1 NPU Forward Implementation (Two-stage aclnn via EXEC_NPU_CMD)
std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_npu(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    double epsilon) {
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have identical shapes");
    at::Tensor y = at::empty_like(x1);
    at::Tensor x_out = at::empty_like(x1);
    auto rstd_sizes = x1.sizes().vec();
    rstd_sizes.back() = 1;
    at::Tensor rstd = at::empty(rstd_sizes, x1.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnAddRmsNormCustom, x1, x2, gamma, epsilon, y, rstd, x_out);
    return std::make_tuple(y, rstd, x_out);
}

// 2. Meta (FakeTensor) Implementation — Mandatory for Dynamo / npugraph_ex / ACLGraph
std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_meta(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    double epsilon) {
    (void)gamma;
    (void)epsilon;
    at::Tensor y = at::empty_like(x1);
    at::Tensor x_out = at::empty_like(x2);
    auto rstd_sizes = x1.sym_sizes().vec();
    rstd_sizes.back() = 1;
    at::Tensor rstd = at::empty_symint(rstd_sizes, x1.options().dtype(at::kFloat));
    return std::make_tuple(y, rstd, x_out);
}

// 3. Differentiable Autograd Function Binding (Optional for Training)
class AddRmsNormAutogradFunction : public torch::autograd::Function<AddRmsNormAutogradFunction> {
public:
    static std::tuple<at::Tensor, at::Tensor, at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x1,
        const at::Tensor& x2,
        const at::Tensor& gamma,
        double epsilon) {
        at::AutoDispatchBelowADInplaceOrView guard;
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("custom_npu::add_rms_norm", "")
            .typed<decltype(add_rms_norm_npu)>();
        auto result = op.call(x1, x2, gamma, epsilon);
        ctx->save_for_backward({std::get<2>(result), gamma, std::get<1>(result)});
        return result;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const at::Tensor& x_out = saved[0];
        const at::Tensor& gamma = saved[1];
        const at::Tensor& rstd = saved[2];
        const at::Tensor& dy = grad_outputs[0];

        at::Tensor dx = at::empty_like(x_out);
        at::Tensor dgamma = at::empty(gamma.sizes(), gamma.options().dtype(at::kFloat));
        EXEC_NPU_CMD(aclnnAddRmsNormCustomGrad, dy, x_out, rstd, gamma, dx, dgamma);
        return {dx, dx, dgamma.to(gamma.scalar_type()), at::Tensor()};
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> add_rms_norm_autograd(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    double epsilon) {
    return AddRmsNormAutogradFunction::apply(x1, x2, gamma, epsilon);
}

} // namespace custom_npu_ops

TORCH_LIBRARY(custom_npu, m) {
    m.def("add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon=1e-6) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(custom_npu, PrivateUse1, m) {
    m.impl("add_rms_norm", TORCH_FN(custom_npu_ops::add_rms_norm_npu));
}

TORCH_LIBRARY_IMPL(custom_npu, Meta, m) {
    m.impl("add_rms_norm", TORCH_FN(custom_npu_ops::add_rms_norm_meta));
}

TORCH_LIBRARY_IMPL(custom_npu, AutogradPrivateUse1, m) {
    m.impl("add_rms_norm", TORCH_FN(custom_npu_ops::add_rms_norm_autograd));
}
```

## 2. Explicit Two-Stage `aclnn` Invocation Without `EXEC_NPU_CMD`

When building an external C++ library that links only `libascendcl.so` and `libcust_opapi.so` without internal `torch_npu` `EXEC_NPU_CMD` headers, execute the two-stage `aclnn` protocol explicitly using the current NPU stream and `at::empty` workspace allocation:

1. Construct `aclTensor*` descriptors from `at::Tensor::sizes()`, `strides()`, `data_ptr()`, and `ACL_FORMAT_ND`.
2. Call `aclnn<Op>GetWorkspaceSize(..., &workspace_size, &executor)` and verify return code `== ACL_SUCCESS`.
3. If `workspace_size > 0`, allocate `at::Tensor workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, x1.options().dtype(at::kByte));` so workspace memory is managed by `NPUCachingAllocator` and safe under `ACLGraph` capture.
4. Call `aclnn<Op>(workspace_ptr, workspace_size, executor, c10_npu::getCurrentNPUStream().stream(false))`.
