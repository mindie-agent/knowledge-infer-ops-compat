# Inference Operations & Compatibility Domain Boundaries (`domain: "infer-ops-compat"`)

This repository (`knowledge-infer-ops-compat`) stores
deployment operations, build/packaging entrypoints, hardware-probe boundaries,
and version/ABI compatibility knowledge units under the canonical domain
`infer-ops-compat`. Runtime engine code lives exclusively in
`mindie-agent/knowledge`.

## Operating Rules

1. **Domain Scope & Isolation**: Keep build/packaging metadata, SoC probe vs.
   CPU test-shim behavior, wheel/ABI compatibility, container/environment
   setup, and operational deployment rules in `infer-ops-compat`. Do not mix
   operator kernels (`operators`), general scheduler/KV state machines
   (`infer-core`), profiler attribution formulas (`infer-observability`), or
   legacy `vllm-ascend` corpus snapshots (`vllm-ascend`)
   into this repository. Reject foreign-domain `mindie://<other-domain>/...`
   URIs on `explain` and `feedback`.
2. **Unit Protocol & Index Rebuild**: Every unit must validate as
   `mindie-graded-unit/1` with a path-independent `id`, monotonic `revision`,
   exact `content_digest`, explicit `conditions`, and line-level
   `provenance.source_locator` / `evidence_refs`. Rebuild
   `index/routing-manifest.json` and `index/shard-topology.json` via the graded
   indexer after any unit change.
3. **Evidence Honesty & Three-Valued Conditions**: Bind source-derived units to
   the exact inspected `repository` and `revision` (e.g.,
   `vllm-project/vllm-ascend` @ `175ef4d49c2072f9233619636f0309f4f88b4e2c`).
   Keep packaging metadata and CPU unit-test shims distinct from real device
   acceptance, never promote `single_observation` source readings to
   `reproduced_empirical` or `L4` without real execution/ablation receipts, and
   never treat `condition_unknown` as a match.
4. **Publication Gate**: [`./publication-contract.json`](./publication-contract.json)
   (`mindie-publication-contract/2`) and thin maintenance entry points under
   `scripts/` define the validator pin and local checks; run
   `python3 scripts/build_routing_index.py --root . --check` after any unit or
   index edit.
