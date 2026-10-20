---
schema: mindie-graded-unit/1
id: unit-torch-npu-custom-op-registration-and-faketensor-meta-sop
revision: 1
tier: L2
category: sop
domain_path: runtime/torch-npu
content_digest: sha256:f569a8c026196ab79ea4bc5c359601dfb9adbcd46fb4bc9d522e43b2b07e05dd
evidence_strength: software_contract
title: Bind AscendC and aclnn custom ops into torch.library with a device-free FakeTensor Meta kernel
conditions:
  accelerator: ascend
  runtime: torch_npu
  operator_interface: torch.library
  custom_op_source:
    - ascendc-direct-invoke
    - aclnn-registry
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
provenance:
  source_type: cannbot_skill_and_operator_contract
  source_locator: docs/evidence/cannbot-ops-compat-provenance.json
  source_digest: fe3cc021d0d3848cca592eb8d67f8d8fa70a25d7896841d59496572955386b71
  evidence_refs:
    - docs/evidence/cannbot-ops-compat-provenance.json
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/torch-ascendc-op-extension/SKILL.md#L15-L36"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/torch-ascendc-op-extension/routes/aclnn-registry.md#L92-L134"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/torch-ascendc-op-extension/routes/aclnn-registry.md#L155-L192"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/torch-ascendc-op-extension/routes/direct-invoke.md#L59-L88"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/torch-ascendc-op-extension/routes/direct-invoke.md#L147-L175"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/graph/torch-custom-ops-guide/SKILL.md#L30-L58"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/ops/ascendc/guides/torch_extensions/compilation_and_launch_boundary.md#L22-L38"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/ops/ascendc/guides/torch_extensions/aclnn_symbol_resolution_and_lifetime.md#L22-L30"
    - "receipt://vllm-ascend-evidence/dsv4-qli-v2-mixed-20260829/vendor-symbols.log"
    - "receipt://vllm-ascend-evidence/k3-gqa-no-prefix-20260908/kda-full-operator-regression-fixed.log"
  supersedes: []
triggers:
  error_signatures:
    - "undefined symbol: aclnn"
    - "aclnnXxx or aclnnXxxGetWorkspaceSize not in libopapi.so"
    - "Meta backend not registered"
    - "Meta function not implemented"
    - "torch.ops.<namespace>.<op> is missing"
    - "FakeTensor tried to execute a real NPU kernel"
    - "torch.compile graph break on custom operator"
    - "NPU arch is not supported"
    - "libhccl.so: cannot open shared object file"
    - "no converter registered for torch.ops"
    - "expected dtype"
  keywords:
    - ascend
    - torch_npu
    - torch.library
    - TORCH_LIBRARY
    - TORCH_LIBRARY_IMPL
    - register_fake
    - Meta
    - FakeTensor
    - aclnn
    - AscendC
    - EXEC_NPU_CMD
    - EXEC_NPU_CMD_V1
    - PrivateUse1
    - torch.compile
    - TorchAir
    - npugraph_ex
    - ACLGraph
    - piecewise
    - libcust_opapi
    - custom_opp
---

# When to use

Use this SOP when an AscendC or `aclnn` operator already builds and runs outside PyTorch, and it must become callable as `torch.ops.<namespace>.<op>` under eager execution, `torch.compile`, TorchAir and a `vllm-ascend` piecewise ACLGraph path.

The SOP assumes the operator package is buildable, the CANN environment is live, and `torch` / `torch_npu` match the vendor table. It does not cover operator numerics, tiling design or performance tuning.

The failure this SOP prevents is specific: the operator calls the device during tracing, or the FakeTensor path has no implementation, so `torch.compile` graph-breaks or the captured graph silently omits the operator. The second failure is the more dangerous one because the service can still run.

# Step 1 - Freeze the operator contract and choose exactly one route

Do not start from a template. Start from the operator project and prove which integration route it supports.

| Evidence in the project | Route | What the PyTorch layer does |
|---|---|---|
| `<<<...>>>` launch syntax, `.asc` / `.cce` kernel files, CMake compiles the kernel | Direct invoke | Call the kernel function from C++, manage the stream, register `TORCH_LIBRARY_FRAGMENT` + `Meta` |
| `op_host/` OpDef (`OP_ADD(...)`), `op_host/op_api/aclnn_<op>.h`, `build.sh` produces `custom_opp_<arch>.run`, `libcust_opapi.so` exports `aclnnXxxGetWorkspaceSize` / `aclnnXxx` | aclnn registry | Load the two-stage `aclnn` symbol through the vendor helper, register schema + `PrivateUse1` + `Meta` |
| Both patterns are present | Ambiguous | Stop and ask which artifact is the deliverable; do not mix the two |

Extract the contract before writing registration code:

- the exact `aclnnXxxGetWorkspaceSize` parameter order, optional parameters and dtype constraints;
- the OpDef input, output and attribute names, and their order;
- the infershape rule that determines output shape and dtype;
- the exported symbol names and the GE `op_type`, if the operator must enter a TorchAir GE graph.

The `aclnn` parameter names and the OpDef names are often not identical. The converter uses OpDef names; the `aclnn` call uses `aclnn` order. Record both mappings.

Give every custom artifact a unique prefix. A bare `aclnnXxx` or GE `Xxx` name can be resolved by a built-in CANN implementation before the custom one. The result is a successful call with the built-in kernel executing and the custom kernel never running.

# Step 2 - Build and install the SoC-matched package, then make its symbols resolvable

Build the operator package for the SoC that `acl.get_soc_name()` reports, not for the marketing name on the server:

```bash
bash build.sh -n <operator-project>
# produces build/custom_opp_<arch>.run
```

Install the package and expose both the kernel metadata and the host `aclnn` library:

```bash
export ASCEND_CUSTOM_OPP_PATH=<vendors-root>
export LD_LIBRARY_PATH=<vendors-root>/<vendor>/op_api/lib:${LD_LIBRARY_PATH:-}
```

Verify the symbol contract before writing PyTorch code:

```bash
nm -D <vendors-root>/<vendor>/op_api/lib/libcust_opapi.so | grep -E 'aclnnXxx(GetWorkspaceSize)?$'
```

Both the workspace-size and execution symbols must be present. If they are absent, the package build did not produce the `aclnn` interface; the PyTorch layer cannot repair that.

There are two legitimate symbol-resolution designs:

- **Vendor helper / dlopen:** `EXEC_NPU_CMD` or `EXEC_NPU_CMD_V1` looks up `aclnnXxxGetWorkspaceSize` and `aclnnXxx` at runtime through a documented search order. The extension does not link `libcust_opapi.so`. `nm -D` on the extension will not show `aclnn` symbols, which is normal. The runtime search path must still contain the vendor `op_api/lib`.
- **Direct link:** the extension links the vendor library and relies on `DT_NEEDED` plus RPATH/RUNPATH or `LD_LIBRARY_PATH`. A missing search entry appears as `undefined symbol: aclnnXxx` at load time.

Do not mix the two designs. A direct call to `aclnnXxx` without either a link dependency or an explicit `dlsym` is an unresolved symbol, not a missing device kernel.

Build the extension with the same C++ ABI as `torch_npu` (normally `-D_GLIBCXX_USE_CXX11_ABI=1`), link `torch_npu` and `dl`, and do not link `libcust_opapi.so` when using the dlopen helper. Source `set_env.sh` before the build and before the Python import; otherwise `import torch_npu` can fail on `libhccl.so` before the extension is even loaded.

The helper layer caches symbol and path lookups in the process. After changing `ASCEND_CUSTOM_OPP_PATH`, `LD_LIBRARY_PATH` or the vendor package, validate in a fresh process. Replacing the file under a running process does not re-resolve the old symbol.

# Step 3 - Register the schema and the PrivateUse1 implementation

Use one `TORCH_LIBRARY` block for the namespace and one `TORCH_LIBRARY_IMPL` block for the NPU backend:

```cpp
TORCH_LIBRARY(custom, m) {
    m.def("npu_x_custom_op(Tensor x1, Tensor x2, Tensor! out) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_x_custom_op", TORCH_FN(npu_x_custom_op));
}
```

The implementation calls the vendor helper with the `aclnn` parameter order:

```cpp
at::Tensor npu_x_custom_op(const at::Tensor& x1,
                           const at::Tensor& x2,
                           at::Tensor& out)
{
    EXEC_NPU_CMD_V1(aclnnXCustomOp, x1, x2, out);
    return out;
}
```

Schema rules that change graph behavior:

- **Aliasing is a contract.** An output that the caller preallocates and the kernel mutates must be declared `Tensor!`. Without the alias marker, `torch.compile` may treat the operator as a pure function and remove it or reorder it.
- **Optional inputs** use `Tensor?`; defaults are only on trailing parameters.
- **Multiple outputs** use `-> (Tensor a, Tensor? b)` and return a matching tuple.
- **Output dtype and shape** must match the Meta implementation exactly. A dynamic-quant operator that returns int8 must not inherit the input's bf16 dtype in the Meta path.
- **Namespace discipline.** Use one `TORCH_LIBRARY` per namespace. Do not define the same namespace in multiple shared objects with `_FRAGMENT` unless you have a deliberate fragment design.

Stream semantics differ by route:

- **aclnn registry:** the helper obtains the current NPU stream and enqueues the operation through the framework command path. Do not add `stream(true)` or a manual queue flush; that changes ordering and can drop the enqueue semantics.
- **Direct invoke:** a kernel called as a plain C function from a C++ extension is outside the framework queue. Use `c10_npu::getCurrentNPUStream().stream(true)` to flush the pending queue before the call, and use `at::empty_like` for outputs that the kernel fully overwrites. `stream(false)` plus a direct launch can run the kernel before earlier framework work.

For a direct-invoke route, the registration shape is the same but the implementation launches the kernel:

```cpp
at::Tensor npu_direct_op(const at::Tensor& x1, const at::Tensor& x2)
{
    auto out = at::empty_like(x1);
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    direct_kernel(block_num, nullptr, stream, x1, x2, out);
    return out;
}
```

# Step 4 - Register a Meta / FakeTensor kernel that cannot execute on the device

`torch.compile` traces with FakeTensors. It does not run the NPU kernel. The Meta kernel must compute only shape, dtype, device and alias information from the fake inputs. It must not allocate NPU memory, call `aclnn`, synchronize, or extract a host scalar.

For an output-preallocated operator, the Meta kernel returns the same fake output object:

```cpp
at::Tensor npu_x_custom_op_meta(const at::Tensor& x1,
                                const at::Tensor& x2,
                                at::Tensor& out)
{
    return out;
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_x_custom_op", &npu_x_custom_op_meta);
}
```

For an out-of-place operator, allocate a meta tensor with the real output dtype and shape:

```cpp
at::Tensor npu_x_custom_op_meta(const at::Tensor& x1, const at::Tensor& x2)
{
    auto sizes = infer_output_sizes(x1.sizes(), x2.sizes());
    return at::empty(sizes, x1.options().dtype(at::kInt));
}
```

For a Python `torch.library.custom_op`, use `register_fake` next to the definition:

```python
@torch.library.custom_op("mylib::npu_x", mutates_args=())
def npu_x(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.ops._C_ascend.npu_x(x, scale)

@npu_x.register_fake
def _(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Symbolic-shape safe: no .item(), no concrete numel arithmetic.
    return torch.empty_like(x, dtype=x.dtype, device=x.device)
```

Meta rules:

- **No device work.** Even a small `torch.empty(..., device="npu")` inside a Meta kernel defeats FakeTensor tracing and can allocate or synchronize on the real device.
- **No host scalar extraction.** `.item()`, `.cpu()`, `int(tensor)` and data-dependent Python control flow force a real execution or a graph break. Use symbolic sizes and dtype rules.
- **Alias correctness.** An in-place operator returns `None` or the same output object, matching the schema's `Tensor!` contract. An out-of-place operator returns a fresh meta tensor.
- **Dtype correctness.** `empty_like(x)` inherits `x.dtype`. If the kernel returns int8 from bf16 input, say `dtype=torch.int8` explicitly.
- **Import order.** Load the extension that defines the schema before importing the module that registers Meta. Registering Meta for a namespace that has not been defined yet leaves the operator partially registered or raises. In `vllm-ascend`, the meta module is imported from the custom-op enable path; importing it before the extension is a known partial-registration state.
- **TorchAir GE is a separate lowering.** A Meta kernel is sufficient for `npugraph_ex`/FakeTensor tracing. The TorchAir GE backend additionally needs `@register_fx_node_ge_converter` (or the equivalent C++ converter) that maps the torch op to `torchair.ge.custom_op("XCustomOp", ...)`, with names and attributes taken from the OpDef. Meta without a converter can still graph-break or fail at GE lowering.

Do not register Meta once for a whole family of operators with different output rules. Each operator's Meta function is part of its contract.

# Step 5 - Prove eager, FakeTensor, torch.compile, TorchAir and ACLGraph in that order

Do not jump to graph capture. Each layer has a distinct failure mode.

**5.1 Eager and reference parity.** Call the operator through `torch.ops.<namespace>.<op>` with NPU tensors and compare with a CPU or framework reference. Verify shape, dtype, strides and, for in-place outputs, that the preallocated tensor was updated. A schema registration alone is not execution evidence.

**5.2 FakeTensor only.** Trace the operator under `FakeTensorMode` without a real device call:

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

with FakeTensorMode():
    x = torch.empty(8, 16, device="meta")
    out = torch.empty(8, 16, device="meta")
    y = torch.ops.custom.npu_x_custom_op(x, x, out)
    print(y.shape, y.dtype, y.device)
```

If this tries to execute on the NPU, or fails with a missing Meta implementation, stop and fix Step 4.

**5.3 `torch.compile` and graph membership.** Compile a small module with `fullgraph=True` and assert that the operator appears in the graph:

```python
compiled = torch.compile(model, fullgraph=True, backend="npugraph_ex")
# Use TORCH_LOGS=graph_breaks,recompiles for diagnosis.
```

A successful run is not enough. Check that the custom operator is in the captured graph. A graph break can silently move the operator back to eager while the service still produces correct output.

**5.4 TorchAir GE.** If the service uses `torchair.get_npu_backend()`, compile the same module with the TorchAir backend and confirm the converter fires. Missing converter, mismatched attribute names or an unsupported output alias fails at GE lowering, not at the Python schema.

**5.5 `vllm-ascend` piecewise ACLGraph.** Capture the decode path with the production capture sizes and static input buffers. Inside the captured region:

- no `torch.empty` or other allocation;
- no `.item()`, `.cpu()` or `torch.npu.synchronize()`;
- no change of input `data_ptr()`;
- padded tokens use the model's sentinel (`slot_mapping = -1` or equivalent), never a live cache block;
- all metadata tensors (`positions`, `slot_mapping`, `block_tables`, `seq_lens`) are copied in place into persistent buffers before replay.

Confirm that the operator is present in the captured graph and that replay matches eager at the same inputs. A custom operator that was never captured is a tracing failure, not a performance result.

# Failure map

| Symptom | First check | Fix |
|---|---|---|
| `undefined symbol: aclnnXxx` at load | Is the extension linking the vendor library, or using `dlsym`? | Link the vendor `op_api/lib` with an RPATH, or switch to the helper and add the runtime path |
| `aclnnXxx ... not in libopapi.so` at call | `LD_LIBRARY_PATH`, `ASCEND_CUSTOM_OPP_PATH`, `nm -D` | Add the vendor `op_api/lib`; install the package; use a fresh process |
| `torch.ops.<ns>.<op>` missing | Extension loaded? Namespace/`m.def` executed? | Load the library before the call; check static initialization and symbol visibility |
| `Meta backend not registered` / `Meta function not implemented` | Step 4 registration; import order | Register `TORCH_LIBRARY_IMPL(..., Meta)` or `@op.register_fake` after the schema exists |
| FakeTensor path executes the device or hangs | Meta function calls `aclnn`, allocates NPU memory or reads `.item()` | Remove all device work from Meta; use symbolic shape rules |
| Graph break on the custom op | No Meta, no converter, or a data-dependent branch | Add Meta; add TorchAir converter for GE; remove host scalar extraction |
| Compiled graph produces the wrong dtype | Meta `empty_like` inherited the input dtype | Declare the real output dtype explicitly |
| `NPU arch is not supported` | Build target does not match the active SoC | Rebuild for the resolved `SOC_VERSION` |
| `libhccl.so: cannot open shared object file` during build/import | CANN environment not sourced | Source `set_env.sh` in the same shell; fix `LD_LIBRARY_PATH` |
| Custom kernel never executes, but the call succeeds | A built-in op with the same name is shadowing the custom op | Add a unique prefix to `aclnn`, GE op_type and torch op names |
| ACLGraph replay diverges from eager | Inputs are not in persistent buffers, or the op allocates/synchronizes inside capture | Fix buffer aliasing and remove allocation/host sync from the captured region |

# Boundaries

This SOP establishes route, symbol, schema, Meta and tracing preconditions. It does not prove the operator's numerical correctness or performance. A custom op that passes every step can still be wrong for a dtype, layout or shape that the test did not cover. Keep the CPU/FP32 reference comparison and the operator's own precision gate as separate acceptance evidence.
