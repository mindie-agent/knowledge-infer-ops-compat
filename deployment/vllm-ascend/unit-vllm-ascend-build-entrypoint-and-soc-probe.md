---
category: sop
conditions:
  repository: vllm-project/vllm-ascend
  revision: 175ef4d49c2072f9233619636f0309f4f88b4e2c
content_digest: sha256:530bcda3284440cd5c9cd132200bf662382f11c651c3f8934fb6091312596056
domain_path: deployment/vllm-ascend
evidence_strength: single_observation
id: unit-vllm-ascend-build-entrypoint-and-soc-probe
inherits_from: []
provenance:
  evidence_refs:
  - docs/evidence/unit-vllm-ascend-build-entrypoint-and-soc-probe-review.json
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/setup.py#L74-L154
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/setup.py#L476-L529
  - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/tests/ut/conftest.py#L39-L132
  source_digest: c3a5a3ecd9be596300fb39a115d062a889ba66eb544f7a6828885834c53b76b3
  source_locator: docs/evidence/unit-vllm-ascend-build-entrypoint-and-soc-probe-review.json
  source_type: ecosystem_repo
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L2
title: vLLM-Ascend packaging metadata and CPU test shims prove different things
triggers:
  error_signatures: []
  keywords:
  - vllm-ascend
  - packaging
  - metadata
  - and
  - cpu
  - test
  - shims
  - prove
  - different
  - things
---
At vLLM-Ascend revision 175ef4d49c2072f9233619636f0309f4f88b4e2c, setup.py declares platform, general-plugin and service-metric entry points. These strings are distribution metadata; the loader and registration function bodies are outside this source unit. When SOC_VERSION is empty, setup.py queries npu-smi to determine the build target. A missing executable returns an empty chip string and triggers the CPU-only guidance, whereas a non-zero command becomes a separate RuntimeError. tests/ut/conftest.py uses a different npu-smi info probe. Failure installs CPU mocks, including torch.npu.is_available=False. Independently of that probe result, setdefault inserts synthetic A2 build information when vllm_ascend._build_info is absent, and adapt_patch plus custom-op registration subsequently run. Successful unit-test setup therefore does not by itself prove a real device was selected, that native extensions built, or that an entry-point loader executed.

Reviewed limits and corrections:

- Do not merge npu-smi command failure with executable absence.
- The build-info setdefault is outside the failed-probe branch.
- Entry-point metadata and test imports are not device acceptance evidence.

Adoption boundary: Use to diagnose packaging and test-environment differences. Native build, loader execution and device behavior require their own observed results.

Evidence scope: Root accepted only the rewritten mechanism. Independent Cursor reading and its broader findings remain in the immutable author packet. Root source reading is enumerated separately.

This unit is a source-derived inspection procedure at the exact repository revision in conditions. It does not establish a reproduced hardware outcome or measured performance benefit.

Pinned source ranges:

- [setup.py:74-154](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/setup.py#L74-L154)
- [setup.py:476-529](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/setup.py#L476-L529)
- [tests/ut/conftest.py:39-132](https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/tests/ut/conftest.py#L39-L132)
