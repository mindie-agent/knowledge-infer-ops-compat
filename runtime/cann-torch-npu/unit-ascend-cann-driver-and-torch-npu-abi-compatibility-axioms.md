---
schema: mindie-graded-unit/1
id: unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
revision: 1
tier: L1
category: axiom
domain_path: runtime/cann-torch-npu
content_digest: sha256:02ffce60d3989bf35a387638c454dc020fe2dced3bc7a4de5b83525c832c3ba3
evidence_strength: software_contract
title: Ascend driver, CANN, torch_npu ABI and SOC_VERSION form one compatibility stack
conditions:
  accelerator: ascend
  runtime: torch_npu
  compatibility_scope: driver-toolkit-kernels-abi-soc
inherits_from: []
provenance:
  source_type: cannbot_knowledge_and_vendor_contract
  source_locator: docs/evidence/cannbot-ops-compat-provenance.json
  source_digest: fe3cc021d0d3848cca592eb8d67f8d8fa70a25d7896841d59496572955386b71
  evidence_refs:
    - docs/evidence/cannbot-ops-compat-provenance.json
    - "https://raw.githubusercontent.com/Ascend/pytorch/master/COMPATIBILITY.md"
    - "https://www.hiascend.com/hardware/firmware-drivers/commercial"
    - "https://www.hiascend.com/cann/document"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/common/platforms/concepts/target_ascend910b.md#L33-L43"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/common/platforms/concepts/target_ascend950pr.md#L27-L40"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/ops/ascendc/guides/environments/build_and_runtime_dependency_boundaries.md#L24-L42"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/ops/ascendc/guides/environments/registry_package_architecture_and_runtime_identity.md#L59-L69"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/ops/ascendc/runbooks/compilation/ol_180_cann_env_init_required_for_so_load.md#L24-L34"
    - "https://gitcode.com/cann/cannbot-knowledge/blob/7ea0efbeb79f2559f121fa8514a9fe5c82245807/knowledge/ops/ascendc/optimizations/ol_284_cann_set_env_strict_mode_wrapper.md#L21-L40"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/ascendc-env-check/references/env_config_guide.md#L62-L87"
    - "https://gitcode.com/cann/cannbot-skills/blob/1fa2cbbc9fdddb4377ca26734e9b94d91c42b975/ops/ascendc-env-check/references/npu_commands.md#L27-L39"
  supersedes: []
triggers:
  error_signatures:
    - "npu-smi: command not found"
    - "aclInit error code 107001"
    - "Invalid device ID"
    - "invalid drv devId"
    - "ImportError: libtorch_npu.so: cannot open shared object file"
    - "ImportError: libhccl.so: cannot open shared object file"
    - "undefined symbol: _ZN"
    - "GLIBCXX_"
    - "AclSetCompileopt(ACL_PRECISION_MODE) error code 500001"
    - "EC0010: Failed to import Python module"
    - "np.float_ was removed in the NumPy 2.0 release"
    - "NPU arch is not supported"
    - "561003"
    - "561107"
    - "HcclReduceScatter error code 15"
    - "EE1001 The value exceeds the total number of cores"
  keywords:
    - ascend
    - cann
    - hdk
    - driver
    - firmware
    - npu-smi
    - torch_npu
    - torch
    - abi
    - glibcxx
    - ASCEND_HOME_PATH
    - ASCEND_TOOLKIT_HOME
    - ASCEND_OPP_PATH
    - LD_LIBRARY_PATH
    - SOC_VERSION
    - Ascend910B1
    - Ascend910B4
    - Ascend910_93
    - Ascend950DT
    - Ascend950PR
    - arch22
    - arch35
    - custom_opp
---

# Scope

This axiom applies whenever PyTorch code is expected to run on an Ascend NPU through `torch_npu`, including a container that only mounts a subset of the devices. It defines the compatibility premises that must hold before a build, a wheel install, a custom-operator load or a serving start can be trusted. It is a compatibility contract, not a performance or model-correctness claim.

The stack is one composite artifact:

```text
Ascend HDK driver + firmware
  -> CANN toolkit + CANN kernels/ops
  -> Python ABI + torch wheel + torch_npu wheel
  -> SOC_VERSION-selected kernel binaries
  -> host C++ ABI + loader search path
```

A mismatch at any arrow can surface at a different arrow. A healthy `npu-smi` does not prove that CANN matches the driver; a successful `import torch_npu` does not prove that the loaded kernel package matches the SoC; a successful compile does not prove that the device can execute the kernel.

## A1. Driver, CANN, torch and torch_npu are a matched set, not a best-effort assembly

The Ascend HDK driver and firmware must be a vendor-supported pair for the installed CANN release. The CANN toolkit, runtime and kernels/ops must be a supported combination for that release. `torch_npu` must be the package built for the installed `torch` major.minor, Python minor and host architecture, and that package must be listed for the installed CANN release.

The authoritative sources are:

- the CANN release notes and firmware/driver compatibility page for the driver-to-CANN pair;
- `$ASCEND_HOME_PATH/compiler/version.info` for the toolkit version and its `required_package_*` / `required_package_runtime_version` fields;
- the TorchNPU compatibility table for the `torch_npu` package, `torch`, CANN and Python combination.

Examples from the vendor table show why the package version, not the marketing product version, is the install identity:

| TorchNPU product | `torch_npu` package | `torch` | CANN |
|---|---|---|---|
| 26.1.0 | 2.10.0.post4 | 2.10.0 | 9.1.0 |
| 7.3.0 | 2.6.0.post5 | 2.6.0 | 8.5.0 |
| 7.1.0 | 2.6.0 | 2.6.0 | 8.2.RC1 |

A newer driver is not automatically compatible with an older CANN, and a newer CANN is not automatically compatible with an older driver. "Latest on both sides" is not a compatibility argument.

## A2. `npu-smi` is a visibility and health probe, not a compatibility proof

`npu-smi info` proves that the driver interface is reachable from the current container or host namespace and that at least one device is visible. It does not prove that the CANN toolkit matches the driver, that the device is fully passed through, or that the kernel package contains an image for the device.

Use structured probes for automation:

```bash
npu-smi info
npu-smi info -m
npu-smi info -t health -i <device-id>
npu-smi info -t memory -i <device-id>
```

Do not parse the human-readable main table in scripts. Its columns and line wrapping vary by driver version. Prefer `-t <type>` key/value output and treat `--help` as the command inventory for that driver.

`npu-smi info -m` is also not a reliable SoC identifier. An A3 device can report a generic `Ascend910` chip name. Resolve the active SoC with `acl.get_soc_name()` (or the platform probe used by the image, such as an asys/DSMI query) before choosing build flags.

A missing `npu-smi`, an empty device list, `aclInit error code 107001` or `Invalid device ID` is a driver/device-visibility failure. It is not a CANN Python bug and not a reason to change model code. In a container, check `/dev/davinci*` mappings, the driver library mount and the runtime's device-visibility variable before touching the Python environment.

## A3. `set_env.sh` is process-local and defines the library search path

CANN installs are not relocatable by environment variable alone. The supported way to enter an installation is to source its `set_env.sh` in the same shell or process that will build or import:

```bash
set +eu
set +o pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /usr/local/Ascend/cann/set_env.sh
set -eu
set -o pipefail
test -n "${ASCEND_HOME_PATH:-}"
```

The script sets the toolkit root (`ASCEND_HOME_PATH`; some installations also export `ASCEND_TOOLKIT_HOME`), the ops path (`ASCEND_OPP_PATH`) and the runtime library path. A custom operator package may add `ASCEND_CUSTOM_OPP_PATH` and a vendor `op_api/lib` entry.

`ASCEND_HOME_PATH` and `ASCEND_TOOLKIT_HOME` are consistent only when they resolve to the same toolkit root. If both are set, compare them before building. One pointing at the current toolkit and the other at an older package is enough to compile against one set of headers and load another set of libraries.

Three failure modes are common:

- `source set_env.sh | tail -1` runs the source in a pipeline subshell. The parent shell never receives the exports.
- A caller running `set -euo pipefail` can exit inside a benign branch of the vendor script. Relax strict mode only around `source`, restore it immediately and assert `ASCEND_HOME_PATH` afterwards.
- A fresh `bash -c`, a new container exec or a new CI step starts with an empty environment. Source again; do not assume the previous step's exports survived.

Without a live CANN environment, loading an Ascend extension can fail with `undefined symbol` errors that belong to the CANN runtime, not to the extension's own kernel code. A `libhccl.so: cannot open shared object file` during `import torch_npu` is the same class of error: the loader cannot see the runtime libraries.

## A4. `torch` and `torch_npu` are ABI-locked, not merely API-compatible

`torch_npu` is a compiled extension over `libtorch` and the CANN runtime. The wheel must match:

- the installed `torch` major.minor and the wheel's build configuration;
- the Python interpreter minor and the wheel's `cpXX` / architecture tag;
- the host C++11 ABI used by `torch` and by any custom extension;
- the glibc baseline of the runtime image.

`import torch_npu` failing with `libtorch_npu.so: cannot open shared object file`, a `_ZN...` undefined symbol or a `GLIBCXX_*` version error is an ABI/loader failure. It is not fixed by reinstalling the model package.

For a custom C++ extension, use the same C++11 ABI as the installed `torch_npu` (normally `-D_GLIBCXX_USE_CXX11_ABI=1`). Do not apply that flag blindly to an environment whose `torch` was built with the old ABI; the requirement is "same as the installed pair", not "always 1".

Python-level checks are necessary but not sufficient:

```bash
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import torch_npu; print(torch_npu.npu.device_count())"
```

A successful import proves that the Python module and its immediate shared-library dependencies loaded. It does not prove that a later custom extension or kernel package is ABI-compatible.

## A5. `SOC_VERSION` is a build target; kernel binaries are per-SoC

AscendC and `aclnn` kernel packages are compiled for a specific SoC/architecture. The binary package is not a portable "Ascend" blob.

| Family | SoC strings / variants | Architecture | Build consequence |
|---|---|---|---|
| A2 | `Ascend910B1`, `Ascend910B2` / `Ascend910B2C`, `Ascend910B3`, `Ascend910B4` / `Ascend910B4-1` | arch22 / `__NPU_ARCH__` 2201 | Kernel and tiling must be built for the matching B-variant |
| A3 | `Ascend910_93` (910C family) | arch22 / V220 dual-die | A3 is not an A2 binary; core counts and package mapping differ |
| A5 | `Ascend950PR` (prefill) and `Ascend950DT` (decode), same `ASCEND950` family | arch35 / `__NPU_ARCH__` 3510 | Must build with the A5/arch35 toolchain; A2/A3 binaries do not run |

`SOC_VERSION` (and the package's `ASCEND_COMPUTE_UNIT` / `--soc` equivalent) selects the kernel image and the compile-time architecture. A package built for the wrong target can fail in several ways:

- the kernel image is absent for the running SoC, producing a kernel/operator lookup failure such as `561003` or `561107`;
- a generic host library loads, but the device-side operator is missing a dtype or mode instance;
- the compiler rejects the target with `NPU arch is not supported`;
- a multi-SoC build silently takes one branch. A mixed list is not proof that every listed SoC was compiled; verify the generated `custom_opp_<arch>.run`, its kernel metadata and the installed vendor directory.

The A5 runtime string can carry a silicon-bin suffix, for example `Ascend950PR_9589` or `Ascend950DT_9581`. The suffix identifies the bin, not a new programming model. Do not hardcode a suffix from one server into a portable build; resolve the active string and map it to the supported `ASCEND950` target.

## A6. The loaded library is what the loader resolves, not what the environment variable says

`LD_LIBRARY_PATH` is a search order, not a selection. The first matching library wins. Already-loaded same-name libraries, `DT_NEEDED` entries, RPATH/RUNPATH and the dynamic linker cache can all override the directory you just prepended.

When a host has multiple CANN installations, verify the actual library identity:

```bash
ldd -r <extension>.so
readelf -d <extension>.so | grep -E 'NEEDED|RPATH|RUNPATH'
nm -D <library>.so | grep ' aclnn'
```

`ldd -r` run outside the target interpreter can report Python C-API symbols as unresolved even when the extension imports correctly. Confirm the final import in a fresh process that uses the same interpreter, environment and working directory as the service.

The same rule applies to custom operator packages. Two `libcust_opapi.so` copies with the same soname can coexist. Prepending the new vendor path in one shell does not change a process that already resolved the old copy. Validate symbol provenance in a fresh process and record the resolved path.

## A7. Build, load, registration, dispatch and execution are different evidence layers

A compatibility claim must name the layer it proves:

| Layer | Evidence | What it proves |
|---|---|---|
| Build | compiler/linker exit status, `nm -D`, wheel contents | Headers and link inputs were sufficient for this build |
| Load | `import torch_npu`, `torch.ops.load_library`, `ldd -r` | The dynamic loader resolved the declared dependencies |
| Registration | `torch.ops.<ns>.<op>` schema, `TORCH_LIBRARY` registration log | The dispatcher knows the operator name and schema |
| Dispatch | call reaches the `PrivateUse1` implementation | The backend implementation was selected |
| Execution | device kernel runs, output matches a reference | The SoC binary and runtime actually executed |

A green build does not prove device execution. A `torch.ops` attribute does not prove that a Meta/FakeTensor kernel exists. A registered Meta kernel does not prove that the NPU kernel is present. A kernel that runs eagerly does not prove that the same kernel survives ACLGraph capture or a TorchAir/GE lowering.

## A8. Host Python dependencies and container policy are part of the compatibility envelope

CANN releases have host-side Python adapters. A known example is CANN 8.2.1's TBE/GE adapter using `np.float_`, which NumPy 2.0 removed. The top-level symptom can be `AclSetCompileopt(ACL_PRECISION_MODE) error code 500001`, while the real line is `EC0010: Failed to import Python module` with `np.float_ was removed`. Pin the host `numpy` to the range supported by the installed CANN release; do not chase the ACL compile option first.

Container policy is also part of the stack: device mappings, driver mounts, visible-device variables and the service user's permissions determine whether a nominally correct install can initialize. A container that can run `npu-smi` but cannot open the device nodes is not a usable runtime.

# Compatibility decision table

| Question | Authoritative evidence | Pass condition |
|---|---|---|
| Does the driver match CANN? | CANN release notes and firmware/driver compatibility page | The exact driver/firmware pair is listed for the installed CANN |
| Are toolkit and kernels/ops matched? | `version.info` `required_package_*` fields | Installed subpackages satisfy the declared versions |
| Is `torch_npu` built for this `torch`? | TorchNPU compatibility table and wheel metadata | Package version, `torch` version, Python tag and architecture all match |
| Is the device visible? | `npu-smi info -m`, device nodes | At least one healthy device is visible in this namespace |
| Which SoC is active? | `acl.get_soc_name()` / platform probe | The resolved SoC maps to the build target, not merely to a marketing name |
| Does the kernel package contain this SoC? | `custom_opp_<arch>.run` contents, vendor directory, `nm -D` | A kernel image and the required operator symbols exist for the resolved SoC |
| Which library is actually loaded? | `ldd -r`, `readelf -d`, fresh-process import | The resolved path is the intended installation and the symbols resolve |
| Can the device execute? | A real operator call with a reference comparison | The device kernel runs and the output passes the operator's acceptance test |

# Minimal probe sequence

```bash
# 1. Driver and device visibility
npu-smi info
npu-smi info -m

# 2. CANN environment in this shell
set +eu; set +o pipefail
source /usr/local/Ascend/cann/set_env.sh
set -eu; set -o pipefail
test -n "${ASCEND_HOME_PATH:-}"
cat "${ASCEND_HOME_PATH}/compiler/version.info"

# 3. Python and ABI identity
python -c "import sys, torch, torch_npu; print(sys.version); print(torch.__version__, torch_npu.__version__)"
python -c "import torch_npu; print(torch_npu.npu.device_count())"

# 4. Active SoC (name is version-dependent; use the platform's documented probe)
python -c "import acl; print(acl.get_soc_name())"

# 5. Extension/package load and symbol provenance
ldd -r <extension>.so
nm -D <custom-op-library>.so | grep -E 'aclnn.*(GetWorkspaceSize)?$' | head
```

Run the probe in the same interpreter, environment and working directory as the service. A probe in an interactive shell can succeed while the service starts from a different environment.

# Failure attribution

| First observable | Most likely layer | Next probe |
|---|---|---|
| `npu-smi: command not found` | Driver tools absent or `PATH` | Check the driver install and container mount |
| `aclInit error code 107001` / `Invalid device ID` | Device not passed through or not visible | Check `/dev/davinci*`, device mappings and visibility variables |
| `libtorch_npu.so: cannot open shared object file` | Wrong wheel, architecture or loader path | `file` the wheel, `ldd`, check Python tag and architecture |
| `undefined symbol: _ZN...` / `GLIBCXX_*` | Host C++ ABI or library mismatch | Rebuild the extension with the `torch_npu` ABI; inspect `nm -D` |
| `libhccl.so: cannot open shared object file` | CANN environment not sourced | Source `set_env.sh` and check `LD_LIBRARY_PATH` |
| `AclSetCompileopt(...) 500001` + `EC0010` | Host Python dependency/version | Read the EC0010 detail line; check `numpy` and missing modules |
| `561003` / `561107` | Ops package, custom package or kernel image missing | Check `ASCEND_OPP_PATH`, `ASCEND_CUSTOM_OPP_PATH`, vendor contents and SoC |
| `NPU arch is not supported` | Wrong compiler target | Rebuild with the resolved SoC/arch toolchain |
| `HcclReduceScatter error code 15` + AICPU error | Driver too old or collective/network problem | Check driver compatibility, then HCCL config and rank symmetry |
| `EE1001 ... total number of cores` | Hardcoded core count on a different A3 variant | Disable the hard limit or use the probed core count |

# Boundaries

This unit establishes compatibility premises and evidence layers. It does not state that any particular host, container, wheel or kernel package passed a live test. It does not replace the vendor compatibility table for a specific release, and it does not authorize promoting a source reading to a hardware or reproduced-empirical claim. A service-level acceptance still needs a real operator or model run on the named SoC and stack.
