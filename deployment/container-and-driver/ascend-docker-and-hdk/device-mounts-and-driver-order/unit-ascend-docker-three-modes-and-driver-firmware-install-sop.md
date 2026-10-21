---
schema: mindie-graded-unit/1
id: unit-ascend-docker-three-modes-and-driver-firmware-install-sop
revision: 1
title: "Ascend Container & HDK Setup: Driver-Before-Firmware Order and Three Docker Device-Mount Modes SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: deployment/container-and-driver/ascend-docker-and-hdk/device-mounts-and-driver-order
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
    - Ascend 310P
  domain: deployment
triggers:
  error_signatures:
    - "drvError"
    - "libascend_hal.so: cannot open shared object file"
    - "/dev/davinci_manager"
    - "ASCEND_VISIBLE_DEVICES"
  keywords:
    - "ascend-docker-runtime"
    - "driver-firmware-install"
    - "davinci-manager"
    - "devmm-svm"
    - "hisi-hdc"
provenance:
  source_type: source_contract
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/ascend-docker/SKILL.md
  evidence_refs:
    - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/ascend-docker/SKILL.md#L1-L160
    - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/ascend-npu-driver-install/SKILL.md#L1-L180
  supersedes: []
---

## Falsifiable Claim

On bare-metal Ascend servers, fresh HDK installations MUST install the NPU Driver package (`Ascend-hdk-*-npu-driver_*.run`) **before** the Firmware package (`Ascend-hdk-*-npu-firmware_*.run`), whereas in-place upgrades MUST upgrade Firmware **before** Driver. When launching containers on the host, omitting any of the three mandatory control devices (`/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`) or host driver library mounts (`/usr/local/Ascend/driver`, `/usr/local/dcmi`, `/usr/local/bin/npu-smi`) under manual `--device` pass-through causes immediate HAL initialization failures (`drvError` or missing `libascend_hal.so`).

## Part 1: Host HDK Driver & Firmware Installation Order Invariants

1. **Hardware & OS Architecture Discovery**:
   - Verify PCIe accelerator presence (`lspci | grep -i d801` for 910A, `d802` for 910B, `d803` for 910C, `d500` for 310P) and CPU architecture (`uname -m`: `aarch64` vs. `x86_64`).
   - Ensure kernel headers (`kernel-devel` / `linux-headers-$(uname -r)`), `dkms`, `gcc`, and `make` are installed and the `HwHiAiUser` group/user exists (`groupadd HwHiAiUser && useradd -g HwHiAiUser -d /home/<user> -m HwHiAiUser -s /bin/bash`).
2. **Fresh Install Order (Driver $\rightarrow$ Firmware)**:
   ```bash
   chmod +x Ascend-hdk-*-npu-driver_*.run Ascend-hdk-*-npu-firmware_*.run
   ./Ascend-hdk-*-npu-driver_*.run --full --install-for-all
   ./Ascend-hdk-*-npu-firmware_*.run --full
   reboot
   ```
3. **Upgrade Order (Firmware $\rightarrow$ Driver)**:
   ```bash
   ./Ascend-hdk-*-npu-firmware_*.run --upgrade
   ./Ascend-hdk-*-npu-driver_*.run --upgrade
   reboot
   ```

## Part 2: Three Ascend Docker Container Launch Modes

Select the container launch mode based on whether `ascend-docker-runtime` is registered in `/etc/docker/daemon.json` and whether the workload requires whole-node privileges:

### Mode 1: `ascend-docker-runtime` Mode (Recommended Production Isolation)
When `ascend-docker-runtime` is installed, pass only `ASCEND_VISIBLE_DEVICES` (physical NPU IDs or chip ranges); the runtime automatically injects `/dev/davinci*`, `/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`, and host driver libraries:

```bash
docker run -itd \
  --name <container> \
  --net=host \
  --shm-size=512g \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /workspace:/workspace \
  <image_tag> /bin/bash
```

### Mode 2: Explicit `--device` + `-v` Non-Privileged Mode (No Runtime Installed)
When running on standard Docker without `ascend-docker-runtime`, you MUST explicitly mount every target `/dev/davinci<N>` device, the three shared control character devices, and the host driver/DCMI directories:

```bash
docker run -itd \
  --name <container> \
  --net=host \
  --shm-size=512g \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /workspace:/workspace \
  <image_tag> /bin/bash
```

### Mode 3: Full `--privileged=true` Whole-Node Debug Mode
For multi-node RDMA/RoCE benchmarking (`hccl-test`, `msprof` system-wide collection) where full `/dev` and InfiniBand/RoCE device access is needed:

```bash
docker run -itd \
  --name <container> \
  --privileged=true \
  --net=host \
  --ipc=host \
  -- pid=host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /workspace:/workspace \
  <image_tag> /bin/bash
```

## Verification Contract

Inside the container, verify that both HAL and CANN runtime see the expected NPU count before starting MindIE or vLLM-Ascend:
1. Run `npu-smi info` and verify `Health: OK` for all visible chips.
2. Run `python3 -c "import torch, torch_npu; print(torch.npu.device_count())"` and confirm the returned count equals the number of mounted `/dev/davinci<N>` devices.
