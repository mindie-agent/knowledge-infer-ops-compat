---
schema: mindie-graded-unit/1
id: unit-npu-adapter-reviewer-and-npu-smi-health-triage-sop
revision: 1
title: "GPU-to-Ascend Migration Audit & npu-smi Device Health/vNPU Management SOP"
tier: L2
category: sop
evidence_strength: software_contract
status: active
domain_path: migration/npu-adapter/code-review-and-smi/cuda-api-audit-and-device-health
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-ascendc-opdef-geir-proto-and-aclnn-two-stage-abi-axioms
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
    - Ascend 310P
  domain: migration
triggers:
  error_signatures:
    - "torch.cuda"
    - "cudaStreamCreate"
    - "flash_attn_varlen_func"
    - "npu-smi info -t health"
  keywords:
    - "npu-adapter-reviewer"
    - "gpu-to-npu-migration"
    - "npu-smi"
    - "ecc-errors"
    - "vnpu-mode"
    - "dmidecode"
provenance:
  source_type: source_contract
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/npu-adapter-reviewer/SKILL.md
  evidence_refs:
    - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/npu-adapter-reviewer/SKILL.md#L1-L250
    - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/npu-smi/SKILL.md#L1-L213
  supersedes: []
---

## Falsifiable Claim

When migrating a CUDA/PyTorch inference repository to Ascend NPU, systematically auditing seven GPU dependency classes (`__global__` kernels, `cudaMallocHost`, `cudaStream_t`, `cublasGemmEx`, `flash_attn_varlen_func`, C++ extensions, and `torch.cuda.amp`) and pre-validating hardware state via `npu-smi` (`health`, `ecc`, `usages`, `proc`, and `product` platform identification) prevents runtime fallback failures and distinguishes software migration bugs from hardware ECC/thermal faults.

## Part 1: Seven-Category GPU-to-Ascend API Migration Matrix

| Blocker Category | Typical GPU Pattern | Ascend NPU Replacement | Migration Complexity |
| :--- | :--- | :--- | :--- |
| **1. Custom Device Kernels** | CUDA `__global__` / `__device__` `.cu` kernels | `AscendC` kernel (`aclnn` two-stage API) or `triton-ascend` kernel | High |
| **2. Host/Device Memory APIs** | `cudaMalloc`, `cudaMallocHost`, `cudaMemcpyAsync` | `aclrtMalloc`, `aclrtMallocHost`, `aclrtMemcpyAsync` (or `torch.npu` caching allocator) | Medium |
| **3. Streams & Events** | `cudaStream_t`, `cudaEvent_t`, `torch.cuda.Stream` | `aclrtStream`, `aclrtEvent`, `torch.npu.Stream` / `torchair.scope.npu_stream_switch` | Medium |
| **4. Vendor Math Libraries** | `cublasGemmEx`, `cudnnConvolutionForward` | `aclnnMatmul` / `aclnnQuantMatmulV4` / ATB (`libatb.so`) fused ops | High |
| **5. FlashAttention Kernels** | `flash_attn_varlen_func`, `flash_attn_with_kvcache` | `torch_npu.npu_fused_infer_attention_score` (FIA) or `npu_flash_attention` | Medium |
| **6. Custom C++ Extensions** | `torch.utils.cpp_extension.CUDAExtension` | `torch_npu.utils.cpp_extension.NpuExtension` + `TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)` | High |
| **7. Mixed Precision & Device Dispatch** | `torch.cuda.amp.autocast`, `.cuda()`, `nccl` backend | `torch.npu.amp.autocast`, `.npu()`, `hccl` distributed backend | Low |

## Part 2: `npu-smi` Device Health, Platform ID & vNPU Triage SOP

### 1. Distinguish A2 vs. A3 Server Platform (`product` vs. Chip Name)
- **Trap**: `npu-smi info` may display `910B3` or `910_93` under Chip Name without revealing whether the chassis is an Atlas 800I A2 PCIe/HCCS server or an Atlas 800I/900 A3 SuperPod node.
- **Verification**: Always query board/product metadata alongside `info -m`:
  ```bash
  npu-smi info -l
  npu-smi info -m
  npu-smi info -t product -i 0 -c 0
  npu-smi info -t board -i 0
  ```

### 2. Five-Point Hardware Health & Zombie Process Audit
Before executing precision or performance benchmarks on device `<id>` chip `<chip>`, run:

```bash
# 1. Health status (must be OK; Warning/Error blocks execution)
npu-smi info -t health -i <id>

# 2. Check HBM ECC multi-bit errors (isolated-pages / double-bit errors)
npu-smi info -t ecc -i <id> -c <chip>

# 3. Check temperature and power throttling
npu-smi info -t temp -i <id> -c <chip>
npu-smi info -t power -i <id> -c <chip>

# 4. Check real-time AI Core / AICPU / HBM bandwidth utilization
npu-smi info -t usages -i <id> -c <chip>

# 5. Inspect active processes holding HBM on the chip
npu-smi info proc -i <id> -c <chip>
```

### 3. Virtualization (vNPU) Inspection & Lifecycle
When running in virtualized container (`vnpu-mode=0`) or VM (`vnpu-mode=1`) environments where a physical NPU (`phy_id`) is sliced via `vir01`–`vir16` templates:
- vNPU IDs always fall in the deterministic range `[phy_id * 16 + 100, phy_id * 16 + 115]`.
- Inspect active vNPU slices before launching containers:
  ```bash
  npu-smi info -t vnpu-mode
  npu-smi info -t info-vnpu -i <id> -c <chip>
  ```
