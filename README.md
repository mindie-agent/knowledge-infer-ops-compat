# Inference Operations, Build & Compatibility Knowledge (`infer-ops-compat`)

This independent local repository (`/Volumes/data/code/knowledge-infer-ops-compat`,
canonical domain `infer-ops-compat`) holds source-verified `L2`
procedures and mechanisms for inference build/packaging entrypoints, hardware
probe vs. CPU test-shim boundaries, ABI/version compatibility, and deployment
operations. Runtime indexing, query, distillation, and synchronization code
remains in `mindie-agent/knowledge`. Authoritative validator pinning and program
execution status are recorded in
[`./publication-contract.json`](./publication-contract.json) and
`/Volumes/data/temp/mindie-003-execution/execution-status.json`.

## Directory Structure & Admitted Units

- `deployment/vllm-ascend/unit-vllm-ascend-build-entrypoint-and-soc-probe.md`
  (`L2`, `category: sop`, `evidence_strength: single_observation`,
  `revision: 1`,
  `content_digest: sha256:530bcda3284440cd5c9cd132200bf662382f11c651c3f8934fb6091312596056`)
- `docs/evidence/reviewed-source-admission-v1.json` — Admission provenance from
  `/Volumes/data/temp/mindie-003-execution/wave4/research/editorial-review-packets/root-observability-ops-editorial-review.json`,
  plus `docs/evidence/unit-vllm-ascend-build-entrypoint-and-soc-probe-review.json`.
- `index/routing-manifest.json` & `index/shard-topology.json` — Derived routing
  and shard indexes rebuilt from the Markdown units.
- `publication-contract.json`, `.github/workflows/check.yml`, & `scripts/`
  (`validate_units.py`, `rebuild_index.py`, `check_publication_contract.py`,
  `apply_feedback.py`, `run_all_checks.py`) — Local `mindie-publication-contract/2`
  declaration and thin CLI wrappers delegating to `mindie_knowledge.graded`.

## Unit Protocol, Evidence & Revision Rules

- **Schema**: Every unit conforms to `mindie-graded-unit/1` (`id`, `revision`,
  `content_digest`, `tier`, `category`, `evidence_strength`, `title`,
  `domain_path`, `inherits_from`, `conditions`, `triggers`, `provenance`).
- **Exact Source Conditions**: The initial unit is bound to
  `conditions.repository: "vllm-project/vllm-ascend"` and
  `conditions.revision: "175ef4d49c2072f9233619636f0309f4f88b4e2c"`. Default
  `Store` queries match only when the exact `repository` and `revision` are
  supplied; mismatched revisions are rejected and omitted revisions remain
  `condition_unknown` (visible only when `include_unknown=True`).
- **Evidence Scope & Revision Discipline**: This `L2` unit documents static
  build/test source boundaries (`setup.py` entry-point metadata and `SOC_VERSION`
  `npu-smi` probing vs. `tests/ut/conftest.py` CPU mock fallback and unconditional
  `_build_info` `setdefault`) and does not claim live NPU build or runtime
  execution. `id` remains stable across directory moves; any substantive edit
  must increment `revision`, recompute `content_digest`, and rebuild
  `index/*.json`.

## Publication & Strict `Feed` Boundary

Validator revision and contract metadata are tracked in
[`./publication-contract.json`](./publication-contract.json), and live release
readiness is tracked in `/Volumes/data/temp/mindie-003-execution/execution-status.json`.
Although the public GitHub remote
`https://github.com/mindie-agent/knowledge-infer-ops-compat` has been created,
new content has not yet been pushed (`new_content_pushed: false`) and the paired
formal runtime/plugin release cutover remains pending.
