---
schema: mindie-graded-unit/1
id: unit-ge-dynamic-shape-partition-memory-imas-and-coredump-triage-workflow
revision: 1
tier: L3
category: workflow
domain_path: runtime/ge
evidence_strength: software_contract
title: GE dynamic-shape unknown-graph partition, devmm/imas HBM reuse analysis, coredump boundary, and custom fusion pass gate workflow
conditions:
  accelerator: ascend
  runtime:
    - ge
    - torch_npu
  repository: cann/cannbot-skills
  revision: f305e200aa4b1e1fed73cd21ff1f083b8df25a1e
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
provenance:
  source_type: cannbot_skill_and_compiler_contract
  source_locator: docs/evidence/cannbot-graph-and-ge-provenance.json
  source_digest: 46102b3b47b7491fed74fd306f3161c1f7bb84076aac2f2f6b218d3378136005
  evidence_refs:
    - docs/evidence/cannbot-graph-and-ge-provenance.json
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-dynamic-shape-diagnose/SKILL.md#L15-L84"
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-memory-analysis/SKILL.md#L20-L160"
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-coredump-diagnose/SKILL.md#L17-L160"
    - "https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/ge-fusion-pass-skill/SKILL.md#L12-L160"
  supersedes: []
triggers:
  error_signatures:
    - "SetGraphUnknownFlag"
    - "cannot support no tiling"
    - "force unknown as dynamic tilingDependent"
    - "marked force unknown node forcibly"
    - "svm_mem_stats_show"
    - "libge_compiler.so"
    - "libgert.so"
  keywords:
    - ge
    - unknown graph
    - dynamic shape
    - no tiling
    - devmm
    - imas
    - reuse_rate
    - reach_theory_rate
    - PatternFusionPass
    - coredump
---
When diagnosing Ascend Graph Engine (`GE`) dynamic-shape fallback, HBM memory bloat or stomping, process coredumps, or custom graph fusion passes, execute the four-subsystem workflow below (`graph/ge-dynamic-shape-diagnose`, `graph/ge-memory-analysis`, `graph/ge-coredump-diagnose`, `graph/ge-fusion-pass-skill`).

## 1. GE Dynamic-Shape (`Unknown Graph`) Three-Step Localization (`ge-dynamic-shape-diagnose/SKILL.md:15-84`)

When a graph is marked `ATTR_NAME_GRAPH_UNKNOWN_FLAG=true`, GE partitions it into static known subgraphs and dynamically scheduled unknown subgraphs (unless unknown-shape nodes satisfy the compile-time **No Tiling** contract):

1. **Step 1 — Confirm Dynamic Scheduling**:
   ```bash
   grep -E "SetGraphUnknownFlag|mark graph.*unknown|Graph.*do not need dynamic shape partition" ge_compiler.log
   ```
2. **Step 2 — Hierarchical Scenario Attribution (Stop at First Match)**:
   - **Priority 1 (GE Judgment)**: `grep -E "cannot support no tiling|unknown as host engine|unknown as subgraph unknown" ge_compiler.log`
   - **Priority 2 (GE Setting)**: `grep -E "force unknown as dynamic tilingDependent|force unknown as it does not support address refresh|Stage|DataFlow" ge_compiler.log`
   - **Priority 3 (FE Setting)**: `grep "marked force unknown node forcibly" ge_compiler.log`, then trace FE sub-reasons (`DT_STRING`, `AllToAll`, `memory size exceeds CCL buffer`, `taskNum`, `DVPP`, `ACLNN`, `TfOptimizer`).
3. **Step 3 — First Unknown-Shape Root Operator**:
   - Search `grep -E "before_infer|after_infer" ge_compiler.log` to locate the earliest operator whose output shape transitions from fixed dimensions to `-1` (`UNKNOWN_DIM`) or `-2` (`UNKNOWN_DIM_NUM`). Downstream cascade-marked operators are effects, not the root cause.

## 2. GE Device HBM Attribution & Memory Stomping Audit (`ge-memory-analysis/SKILL.md:20-160`)

1. **Module-Level Attribution (`devmm`)**:
   - Check run-stage plog (`grep -i 'svm_mem_stats_show' plog-*.log`) for `allocated_peak_size` across `GE`, `HCCL`, `RUNTIME`, and `APP`. At process exit, `current_alloced_size=0` and `alloc_cnt=free_cnt` confirm zero driver-level leak.
2. **GE Internal Breakdown (`imas total` & `imas report`)**:
   - Run `./imas total <debug_plog_dir>/plog/plog*` to separate `GE-StaticGraph` (static graph feature maps) from `GE-MemoryPool` (RT2.0 dynamic-shape page caching) and `GE-Variable` (weights/constants).
   - Run `./imas report <debug_plog_dir>/plog/plog*`:
     - If `imas_mem_graph.csv` is non-empty (**Static Graph**): Read the final summary row first. Low `reuse_rate` indicates suboptimal topological sort (`topo`) or missing control edges; `reuse_rate` normal but `reach_theory_rate < 100%` indicates long-lived contiguous blocks, best-fit mismatch, 512B alignment expansion, or multi-stream lifetime extension.
     - If `imas_mem_graph.csv` is empty (**Dynamic Shape RT2.0**): Analyze `imas_mem_pool.csv` by `allocator_id` (offline dump-graph replay tuning does not apply; mitigate pool fragmentation via `ge.exec.staticMemoryPolicy="3"` or `"4"`).
3. **Memory Stomping & VA/PA Conflict Check**:
   - Distinguish 5 stomping classes: (1) static reuse bug, (2) kernel out-of-bounds write, (3) kernel over-read of dirty reused memory, (4) kernel under-write leaving dirty tail bytes, and (5) **VA/PA mapping conflict** (under dynamic shape + `staticMemoryPolicy` 3/4 where distinct virtual addresses `Va1`/`Va2` bind to the same physical page `Pa`, requiring INFO-level GE VA/PA verification logs).

## 3. GE Coredump Frame-Ownership & Custom Fusion Pass Gates (`ge-coredump-diagnose/SKILL.md:17-160`, `ge-fusion-pass-skill/SKILL.md:12-160`)

- **Coredump Inclusion Rule**: Diagnose as a GE crash **only** if frame `#0` of the crashing thread—or the nearest non-standard-library frame after skipping `libc`/`libstdc++`/`libpthread`/`libm`/`libdl`—belongs to a GE shared library (`libge_common`, `libge_compiler`, `libge_runner`, `libge_executor`, `libgert`, `libgraph*`, `libmetadef`, `tf_adapter`). If the nearest non-stdlib frame belongs to `libruntime`/`libascendcl`, `libhccl`, or an AscendC kernel, route immediately to the owning subsystem rather than speculating on deeper GE frames.
- **Custom Fusion Pass `G1 -> G2 -> G3` Gates**:
  - Choose `PatternFusionPass` (or `PatternFusionPassV2` on CANN $\ge 9.1.0$ when reading options via `CustomPassContext`) for fixed-topology subgraph replacements, `DecomposePass` for single-op expansion, and `graph_base_pass` when control edges, nested subgraphs, or dynamic input/output arities are present. Verify SHA-256 handoff digests (`write_handoff_digest.py --check`) and GE/ES API signatures (`G2`) before compiling `.so` or running `pyatc`.
