---
schema: mindie-graded-unit/1
id: unit-ge-compiler-and-gert-host-coredump-gdb-triage-workflow
title: Ascend GE Compiler and GERT Runtime Host Coredump GDB Stack and Symbol Disassembly
  Triage Workflow
domain_path: graph/ge-compiler/memory-and-coredump
tier: L3
category: workflow
evidence_strength: reproduced_empirical
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: ge
  diagnosis_domain: ge-gert-host-coredump
  runtime:
  - ge
  - torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-ge-dynamic-shape-partition-memory-imas-and-coredump-triage-workflow
supersedes: []
triggers:
  error_signatures:
  - SIGSEGV in libge_compiler.so
  - SIGSEGV in libgert.so KernelRunContext
  - SIGABRT double free or corruption during aclFinalize
  - c++filt objdump disassembly GeTensorDesc null dereference
  keywords:
  - libge_compiler.so coredump
  - libgert.so crash
  - KernelRunContext GetInputPointer
  - gdb bt full disassembly
  - aclFinalize teardown crash
  - GeTensorDesc null pointer
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/ge-coredump-analysis
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-coredump-analysis/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-coredump-analysis/SKILL.md
  source_type: curated_repo
  source_digest: f7d07e3798e5929f24288734b922a8a870897414d27a731fb6ff1cf5232b98f9
---

# Ascend GE Compiler and GERT Runtime Host Coredump GDB Stack and Symbol Disassembly Triage Workflow

## 1. Extracting Stack Traces and Instruction Context from Host Core Files

When a host process crashes (`SIGSEGV` / `SIGABRT`) inside `libge_compiler.so`, `libgert.so`, `libgraph.so`, or `libcust_opapi.so`:

```bash
gdb -batch \
  -ex "set pagination off" \
  -ex "info sharedlibrary" \
  -ex "thread apply all bt" \
  -ex "frame 0" \
  -ex "info registers" \
  -ex "x/24i \$pc-32" \
  "$(which python3)" "<core-file>"
```

If CANN shared libraries are stripped of DWARF debug info (`??` line numbers), resolve the crashing function and relative instruction offset using `addr2line` / `nm -D -C` / `objdump`:

```bash
# Compute file offset = crashing_pc - shared_lib_load_base
objdump -d -C --start-address=0x<offset_hex> --stop-address=0x<end_hex> \
  "${ASCEND_HOME_PATH}/lib64/libge_compiler.so"
```

## 2. Four Canonical GE / GERT Crash Signatures and Fixes

| Crashing Library & Symbol Pattern | Register / Assembly Evidence | Root Cause | Concrete Fix |
|---|---|---|---|
| `libgraph.so` / `libge_compiler.so` inside `InferShape` or `PatternFusionBasePass::Fusion` | `x0` / `rdi` is `0x0` or small struct field offset (`0x18`, `0x20`) dereferenced on `OpDesc::MutableInputDesc(i)` | Custom pass or `InferShape` accesses an optional input (`OPTIONAL` parameter not provided by caller) without null-checking `GetInputDescPtr(i)`. | Guard optional inputs via `if (context->GetOptionalInputShape(i) != nullptr)` before reading dimensions. |
| `libgert.so` inside `gert::KernelRegistry` / `TilingContext::GetInputTensor` | Crash inside `gert::KernelRunContext::GetInputPointer` or `TilingFunc` | Custom operator `TilingFunc` calls `context->GetInputTensor(i)->GetData<T>()` on a device-resident input that was NOT declared `ValueDepend(Option::REQUIRED)` in `OpDef`. | Add `.ValueDepend(Option::REQUIRED)` to the `OpParamDef` in host `OpDef` so GERT copies the small host-dependent tensor prior to `TilingFunc`, or compute tiling purely from `GetInputShape(i)`. |
| `libcust_opapi.so` / `libopproto.so` inside `std::basic_string` or `std::vector` destructor | Crash on `free()` with corrupted pointer or `_ZNSt7__cxx1112basic_string...` vs `_ZNSs...` | `_GLIBCXX_USE_CXX11_ABI` mismatch between custom OPP/extension build and installed `libgraph.so` / `libtorch_npu.so`. | Compile custom operators and C++ extensions with `-D_GLIBCXX_USE_CXX11_ABI=$(python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")`. |
| `libascendcl.so` / `libgert.so` during Python process exit (`__run_exit_handlers` / `atexit`) | Crash on `aclrtDestroyStream` or GE `Session` teardown while another thread is in `rtStreamSynchronize` | Python daemon threads or unjoined `ProcessGroupHCCL` / `NPUGraph` objects issue NPU runtime calls after `aclFinalize()` has destroyed global driver contexts. | Explicitly join worker threads, call `dist.destroy_process_group()`, and release captured `NPUGraph` references before process exit. |
