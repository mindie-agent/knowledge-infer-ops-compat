---
schema: mindie-graded-unit/1
id: unit-ascendc-infershape-tilingdata-and-workspace-registration-sop
title: AscendC InferShape, InferDataType, TilingData Struct Alignment, and Workspace
  Registration SOP
domain_path: registration/custom-ops/aclnn-and-geir
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  operator_framework: ascendc-opdef
  registration_layer: infershape-tiling-and-workspace
  runtime:
  - aclnn
  - ge
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-ascendc-opdef-geir-proto-and-aclnn-two-stage-abi-axioms
supersedes: []
triggers:
  error_signatures:
  - TilingFunc GetWorkspaceSizes missing GetLibApiWorkSpaceSize
  - BEGIN_TILING_DATA_DEF struct size alignment mismatch
  - gert::InferShapeContext GetInputShape null pointer
  - REGISTER_TILING_DATA_CLASS missing for OpType
  keywords:
  - BEGIN_TILING_DATA_DEF
  - TILING_DATA_FIELD_DEF
  - REGISTER_TILING_DATA_CLASS
  - GetLibApiWorkSpaceSize
  - gert::TilingContext
  - gert::InferShapeContext
  - SetTilingKey SetBlockDim
provenance:
  source_kind: external_import
  source_ref: cannbot-knowledge:ops/ascendc/guides/registration/tiling_func_impl+workspace_tiling+infer_shape_impl
  source_locator: https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/tiling_func_impl.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/tiling_func_impl.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/tiling_data_struct_def.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/workspace_tiling.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/infer_shape_impl.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/ascendc/guides/registration/tiling_key_design.md
  source_type: curated_repo
  source_digest: 332532405ad05edbfa4d41ea424c9adbb27ba27aff66319bedd4ecfd9b2ffb2e
---

# AscendC InferShape, InferDataType, TilingData Struct Alignment, and Workspace Registration SOP

## 1. Host `TilingData` Definition and Serialization SOP

Host-to-device tiling parameters (`op_host/<op>_tiling.h`) MUST be declared using the AscendC reflection macros so `SaveToBuffer` packs identical byte offsets as device-side `GET_TILING_DATA`:

```cpp
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MyCustomOpTilingData)
  // If using MC2 HCCL in-kernel APIs, Mc2InitTiling MUST be the first field!
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, coreSplitCounts);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MyCustomOp, MyCustomOpTilingData)
} // namespace optiling
```

- **Forbidden Constructs in `TilingData`**: Never place raw pointers, `std::vector`, `std::string`, or `bool` fields inside `BEGIN_TILING_DATA_DEF`; use fixed-width integer/float scalars (`uint32_t`, `uint64_t`, `int64_t`, `float`), fixed-length arrays (`TILING_DATA_FIELD_DEF_ARR`), or nested registered structs (`TILING_DATA_FIELD_DEF_STRUCT`).
- **MC2 First-Field Rule**: For any MC2 operator using `AscendC::Hccl`, `Mc2InitTiling` MUST be the very first field in the `TilingData` struct (`TILING_DATA_FIELD_DEF_STRUCT(Mc2InitTiling, mc2InitTiling)`), while `Mc2CcTiling` may appear at any subsequent field offset passed to `SetCcTilingV2(offsetof(...))`.

## 2. Implementing `TilingFunc(gert::TilingContext* context)` and System Workspace Accounting

Every `TilingFunc` must set four mandatory outputs on `gert::TilingContext*`:

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MyCustomOpTilingData tiling;
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = platformInfo.GetCoreNumAiv();
    uint32_t aicNum = platformInfo.GetCoreNumAic();

    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t totalElements = xShape->GetStorageShape().GetShapeSize();
    tiling.set_totalLength(static_cast<uint32_t>(totalElements));
    tiling.set_tileNum(8U);

    // 1. Set BlockDim (AICore launch count: AIV count for vector ops, AIC count for cube/mix ops)
    context->SetBlockDim(aivNum);
    // 2. Set TilingKey to select the compiled kernel template instantiation
    context->SetTilingKey(1UL);
    // 3. Serialize TilingData into RawTilingData buffer
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 4. Compute User Workspace + Mandatory AscendC High-Level API System Workspace
    uint32_t sysWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    uint64_t usrWorkspaceSize = 0ULL; // Add 32B-aligned scratch bytes if kernel uses GM workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + static_cast<uint64_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}
```

- **Critical Pitfall**: Omitting `platformInfo.GetLibApiWorkSpaceSize()` when the kernel invokes high-level `Matmul`, `SoftMax`, `Reduce`, or `SyncAll` APIs causes silent GM workspace buffer overrun into adjacent tensors!

## 3. RT2 `InferShape` and `InferDataType` Implementation SOP

```cpp
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* inShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (inShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *inShape; // Zero-allocation in-place shape copy
    return ge::GRAPH_SUCCESS;
}
```

- Never allocate heap objects inside `InferShape` (it runs on the critical host dispatch path of every eager/dynamic operator launch).
- Propagate unknown rank (`IsUnknownDimNum()`) and unknown dimensions (`-1`) explicitly when supporting GE dynamic shapes.
