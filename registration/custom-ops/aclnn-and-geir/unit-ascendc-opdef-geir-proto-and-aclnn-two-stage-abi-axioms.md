---
schema: mindie-graded-unit/1
id: unit-ascendc-opdef-geir-proto-and-aclnn-two-stage-abi-axioms
title: AscendC OpDef Host Registration, GE IR Prototype Mapping, and Two-Stage aclnn
  ABI Contract Axioms
domain_path: registration/custom-ops/aclnn-and-geir
tier: L1
category: axiom
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  operator_framework: ascendc-opdef
  registration_layer: opdef-geir-aclnn-abi
  runtime:
  - aclnn
  - ge
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
supersedes: []
triggers:
  error_signatures:
  - EZ1001 OpPrototype not registered for OpType
  - aclnnXxxGetWorkspaceSize returned 161001 ACL_ERROR_INVALID_PARAM
  - OpAICoreConfig DataType or Format mismatch with input aclTensor
  - OpMC2Def HcclGroup context missing in TilingData
  keywords:
  - ops::OpDef OP_ADD
  - OpParamDef ValueDepend
  - OpAICoreConfig AddConfig
  - OpMC2Def HcclGroup
  - aclnnGetWorkspaceSize two-stage ABI
  - aclOpExecutor lifecycle
provenance:
  source_kind: external_import
  source_ref: cannbot-knowledge:ops/ascendc/guides/registration/op_def+op_param_def+op_aicore_config
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/reg_ascendc_op_overview.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/reg_ascendc_op_overview.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/op_def.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/op_param_def.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/op_aicore_config.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/op_mc2_def.md
  source_type: curated_repo
  source_digest: 1474d6ebf6cd211151b96e4b09da4229b34e64e48fd05d600967e0d2d4d6d46f
---

# AscendC OpDef Host Registration, GE IR Prototype Mapping, and Two-Stage aclnn ABI Contract Axioms

## 1. Dual-Facet `ops::OpDef` Registration Axiom

In CANN AscendC operator engineering (`op_host/<op>_def.cpp`), a single `class <Op> : public ops::OpDef` registered via `OP_ADD(<Op>)` simultaneously synthesizes two runtime contracts:

1. **GE IR Operator Prototype (`OpProto` / `REG_OP` Equivalent)**:
   - Declares ordered `.Input("<name>")`, `.Output("<name>")`, and `.Attr("<name>")` schemas together with `this->SetInferShape(ge::InferShape)` and `this->SetInferDataType(ge::InferDataType)`.
   - Consumed by GE graph compilation (`InferShapeAndType`) and TorchAir `@register_fx_node_ge_converter` lowering.
2. **RT2 / `aclnn` Kernel Execution Descriptor (`OpAICoreDef` & `OpMC2Def`)**:
   - Declares `this->AICore().SetTiling(optiling::TilingFunc)` and per-SoC capability tables via `this->AICore().AddConfig("ascend910b", aicoreConfig)` / `AddConfig("ascend910_93", aicoreConfig)`.
   - For compute-communication fused (MC2) operators, `this->MC2().HcclGroup({"group_0"})` binds host communicator context handles so kernel-side `AscendC::GetHcclContext<0>()` resolves the device message window.

## 2. `OpParamDef` and `OpAICoreConfig` Combinatorial Matching Invariant

```cpp
ops::OpAICoreConfig aicore_config;
aicore_config.DynamicCompileStaticFlag(true)
    .DynamicFormatFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true);
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
```

- **Positional Tuple Pairing Rule**: The `.DataType({...})` and `.Format({...})` vectors across all `Input` and `Output` declarations must have **identical vector length $M$**. Entry $k \in [0, M)$ across all inputs and outputs defines the $k$-th valid `(dtype, format)` combination supported by the kernel. Mismatched vector lengths fail OPP compilation or reject valid `aclTensor` combinations with `161001` (`ACL_ERROR_INVALID_PARAM`).
- **Host Value Dependency (`ValueDepend`)**: If `TilingFunc` or `InferShape` must read the **element values** (not just shape/dtype) of an input tensor (such as `actual_seq_lengths` or slice bounds), the input MUST declare `.ValueDepend( ops::Option::REQUIRED )` (or be declared as an `aclnn` `aclIntArray*` / `aclScalar*` host argument); otherwise RT2 passes a device-only tensor whose host data pointer is `nullptr`.

## 3. Two-Stage `aclnn` C ABI Execution Invariant

Every AscendC operator compiled into `libcust_opapi.so` (or built-in `libopapi.so`) exposes a strict two-stage C ABI:

```cpp
// Stage 1: Host Preparation (InferShape + TilingFunc + Workspace Size Calculation)
aclnnStatus aclnn<Op>GetWorkspaceSize(
    const aclTensor* x, ..., aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// Stage 2: Device Workspace Binding & Asynchronous Stream Launch
aclnnStatus aclnn<Op>(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

- **`aclOpExecutor` Single-Use Ownership**: The `aclOpExecutor*` returned by Stage 1 is bound to the exact tensor metadata and allocated `workspaceSize` of that invocation and is consumed/managed by Stage 2 (`EXEC_NPU_CMD` handles Stage 1 -> `NPUCachingAllocator` workspace allocation -> Stage 2 automatically).
- **Zero-Allocation Fast Path**: When `workspaceSize == 0`, Stage 2 passes `workspace = nullptr` and launches directly onto `stream`.
