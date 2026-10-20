---
schema: mindie-graded-unit/1
id: unit-mindie-service-config-security-and-npu-memory-budget-sop
revision: 1
tier: L2
category: sop
domain_path: deployment/mindie-service
content_digest: sha256:9a7f2eac1d347654fa468278d906391a06d8e011351093ca420e4b0083fc77b4
evidence_strength: software_contract
title: MindIE Service config, TLS permissions and NPU memory budget are one startup contract
conditions:
  service: mindie-service
  accelerator: ascend
  config_schema: config.json
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
provenance:
  source_type: mindie_service_contract_and_public_configuration_guide
  source_locator: docs/evidence/cannbot-ops-compat-provenance.json
  source_digest: fe3cc021d0d3848cca592eb8d67f8d8fa70a25d7896841d59496572955386b71
  evidence_refs:
    - docs/evidence/cannbot-ops-compat-provenance.json
    - "https://www.hiascend.com/document/detail/en/mindie/300"
    - "https://www.hiascend.com/document/detail/zh/mindie"
    - "https://www.hiascend.com/document/detail/zh/mindie/230"
    - "https://gitee.com/mindspore/docs/raw/8e9b99c3f76201898cb3750b924c5aa7379a3a30/docs/mindformers/docs/source_zh_cn/guide/deployment.md"
  supersedes: []
triggers:
  error_signatures:
    - "Check_path: config.json failed"
    - "config.json parse error"
    - "worldSize is inconsistent with npuDeviceIds"
    - "npuMemSize exceeds the available device memory"
    - "npuMemSize caused the service to hang during startup"
    - "httpsEnabled is true but tlsCert or tlsPk is not configured"
    - "tlsPk permission denied"
    - "certificate and private key do not match"
    - "Address already in use"
    - "aclInit error code 107001"
    - "Invalid device ID"
    - "HCCL_CONNECT_TIMEOUT"
    - "maxSeqLen must be greater than maxInputTokenLen"
  keywords:
    - mindie-service
    - mindieservice_daemon
    - config.json
    - worldSize
    - npuDeviceIds
    - npuMemSize
    - maxSeqLen
    - maxInputTokenLen
    - maxPrefillTokens
    - maxPrefillBatchSize
    - maxIterTimes
    - httpsEnabled
    - tlsCert
    - tlsPk
    - tlsCaFile
    - tlsPkPwd
    - chmod 640
    - certificate
    - KV cache
    - HBM
    - startup exit
---

# When to use

Use this SOP before starting `mindieservice_daemon`, and when the daemon exits during startup or hangs before serving. It covers four coupled contracts:

1. the JSON config is syntactically valid and internally consistent;
2. the declared NPU set and model-parallel size fit the visible devices;
3. the KV-cache budget fits the free HBM after weights and runtime reservations;
4. the service user can read the config and the TLS material, and HTTPS is either correctly configured or deliberately disabled.

A service can pass JSON validation and still fail because one of the other contracts is broken. Diagnose the first failing layer, not the last printed error.

# Config contract

Field names and valid ranges vary by MindIE release. Read the schema shipped with the installed service before applying the checks below. The semantics that matter for startup are stable across the versions in scope.

| Field | Meaning | Startup invariant |
|---|---|---|
| `ServerConfig.httpsEnabled` | Enable HTTPS on the business interface | If true, all TLS fields must resolve; if false, the endpoint is plaintext and exposes a documented security risk |
| `ModelDeployConfig.modelConfig[].worldSize` | Number of NPUs used by that model instance | Must equal the number of devices assigned to that instance, and the union must fit the visible devices |
| `ModelDeployConfig.modelConfig[].npuDeviceIds` | Device-ID groups for the instance | When the installed schema honors this field, the IDs must be disjoint, in range and equal in count to `worldSize`; some releases drive placement from visible devices instead, so confirm which source is authoritative |
| `ModelDeployConfig.modelConfig[].npuMemSize` | Per-NPU KV-cache budget in GB | Must fit the free HBM after weights and non-KV reservations; `-1` has release- and model-type-specific caveats |
| `ModelDeployConfig.maxSeqLen` | Maximum total sequence length | Must be greater than or equal to `maxInputTokenLen`; the intended input plus output must fit |
| `ModelDeployConfig.maxInputTokenLen` | Maximum input length | Must not exceed `maxSeqLen` |
| `ModelDeployConfig.maxIterTimes` | Maximum decode iterations | Must be at least 1 and at most `maxSeqLen - 1`; the worst-case input plus output should fit `maxSeqLen` |
| `ModelDeployConfig.maxPrefillTokens` | Maximum total input tokens in one prefill batch | A batch budget, not a per-request cap; it may exceed `maxSeqLen` when the batch holds multiple requests. It must be large enough for the intended prefill chunking and must sit within the installed release's range. It competes with `maxPrefillBatchSize` to end a batch |
| `ModelDeployConfig.maxPrefillBatchSize` | Maximum requests in one prefill batch | Must be within the installed release's range; the first of this limit and `maxPrefillTokens` ends the batch |

Two cautions:

- `npuMemSize` is a KV-cache cap, not a promise that the allocator will obtain that much. Setting it above the free HBM after weights causes startup failure or a hang. Setting it too low causes preemption/recompute rather than a clean error.
- For multimodal models, `-1` is not a safe universal default because the vision tower needs its own reservation. Use an explicit budget derived from the model and device.

# Step 1 - Freeze the deployment identity and the authority for placement

Record, in the service user's environment:

```bash
id -un
npu-smi info -m
echo "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch_npu.npu.device_count())"
```

Confirm:

- the `mindie-service` root and the config path the daemon will read;
- the service user and the user that owns the config, logs and TLS files;
- the visible-device variable and the actual device count;
- whether the installed release places models by `npuDeviceIds`, by visible devices plus `worldSize`, or by a ranktable;
- the business, management and metrics ports;
- the installed MindIE version and its config schema.

Do not mix a shell that has one CANN installation with a service that starts under a different unit file, container entrypoint or user. The environment that starts the daemon is the environment that matters.

# Step 2 - Validate JSON syntax and cross-field arithmetic

Start with syntax and strict parsing:

```bash
python -m json.tool conf/config.json >/dev/null
```

JSON does not allow comments or trailing commas. A file that loads in a permissive parser can still fail the service's strict parser.

Then check the arithmetic:

```text
total_devices = sum(len(group) for group in npuDeviceIds)
total_devices == worldSize
device IDs are unique across model instances and in the visible range

maxInputTokenLen <= maxSeqLen
1 <= maxIterTimes <= maxSeqLen - 1
maxInputTokenLen + maxIterTimes <= maxSeqLen        # conservative request budget
maxPrefillTokens within the installed range
maxPrefillTokens >= maxInputTokenLen                # if a request must prefill in one shot
maxPrefillBatchSize within the installed range
```

`maxPrefillTokens` is a batch-level budget. It is not required to be less than `maxSeqLen`: a batch of several short requests can legitimately exceed one request's maximum sequence length. The `maxInputTokenLen + maxIterTimes` line is a conservative check, not a schema rule; the service can accept a request whose own input plus requested output fits `maxSeqLen` even when the two independent maxima do not add up.

A configuration can be schema-valid and still be a bad deployment because the declared sequence or prefill budget cannot fit the workload.

For multiple model instances, verify that their device groups are disjoint. Overlapping groups can start and then fail when two workers initialize the same device, or can silently serialize behind each other.

# Step 3 - Prove the HBM and KV-cache budget on every target device

Compute the budget per device:

```text
weights_per_device ≈ model_weight_bytes / model_parallel_size
kv_budget_gb ≈ (free_hbm_gb - weights_per_device_gb - runtime_reserve_gb) * 0.8
npuMemSize <= kv_budget_gb
```

The `0.8` factor is a headroom convention, not a hardware constant. Keep room for activations, communication buffers, the graph capture pool and fragmentation.

A KV-cache sanity check:

```text
kv_bytes_per_token ≈ 2 * layers * kv_heads * head_dim * bytes_per_element
kv_bytes_max ≈ kv_bytes_per_token * maxSeqLen * peak_concurrent_sequences
```

Use the model's real KV geometry. MLA, hybrid recurrent/attention and quantized-cache models do not follow the dense MHA formula.

Probe the devices before start:

```bash
npu-smi info -t memory -i <device-id>
npu-smi info -t proc-mem -i <device-id>
```

Requirements:

- every target device is healthy and has no unrelated process holding HBM;
- the free HBM covers weights, `npuMemSize` and the runtime reserve;
- the devices in `npuDeviceIds` are the devices that are actually visible;
- the service user has permission to open the device nodes.

If `npuMemSize` is too high, expect a startup failure or a hang rather than a clean validation message. Lower it and retry. If it is too low, expect preemption or recompute under load; that is a capacity problem, not a startup error.

# Step 4 - Lock file permissions and prove the TLS chain

The config can contain model paths, internal endpoints and TLS locations. It must not be world-readable.

```bash
chmod 640 conf/config.json
chmod 600 <tls-key>
chmod 600 <tls-password-file>
chmod 644 <tls-cert>
chmod 750 <config-directory>
```

The exact numeric mode can be stricter. The invariants are:

- the service user owns or can read the config; users outside the service user/group cannot;
- the private key and password file are readable only by the service user;
- the certificate chain is readable by the service user;
- no parent directory is world-writable;
- the daemon's log directory is writable by the service user.

`Check_path: config.json failed` is a path/permission/ownership failure, not a JSON schema failure. Check the absolute path, the service user's traverse permission on every parent directory and the file mode before editing the JSON.

If `httpsEnabled` is true, validate the TLS fields used by the installed schema. A typical block uses:

```json
{
  "httpsEnabled": true,
  "tlsCaPath": "security/ca/",
  "tlsCaFile": ["ca.pem"],
  "tlsCert": "security/certs/server.pem",
  "tlsPk": "security/keys/server.key.pem",
  "tlsPkPwd": "security/pass/key_pwd.txt"
}
```

Checks:

```bash
openssl x509 -in server.pem -noout -subject -dates -ext subjectAltName
openssl pkey -in server.key.pem -noout -check
openssl verify -CAfile ca.pem server.pem
# Compare the certificate public key with the private key public key.
openssl x509 -in server.pem -noout -pubkey > cert.pub
openssl pkey -in server.key.pem -pubout > key.pub
cmp cert.pub key.pub
```

Verify that the certificate has not expired, that its SAN covers the hostname clients use, and that the CA file is the chain the clients trust. For an encrypted key, verify that `tlsPkPwd` points to a readable password file and that the password actually decrypts the key.

If `httpsEnabled` is false, record why. The vendor documentation explicitly treats plaintext service exposure as a network-security risk. Do not disable HTTPS merely to get past a certificate error; fix the chain or isolate the network.

# Step 5 - Launch with observable logs and triage the first failing layer

Start the daemon in the foreground first, with logs on stdout:

```bash
cd <mindie-service-root>
MINDIE_LOG_TO_STDOUT=1 MINDIE_LOG_LEVEL=INFO ./bin/mindieservice_daemon
```

Keep the daemon's own log under `logs/` as well; the exact filename follows the installed release. Do not start with `nohup` until the foreground start succeeds.

After start:

```bash
npu-smi info -t proc-mem -i <device-id>
ss -lntp | grep -E ':<business-port>|:<management-port>|:<metrics-port>'
curl -k https://localhost:<business-port>/health
curl -k https://localhost:<business-port>/v1/models
```

A process that exists but never answers `/health` is not a successful start. Check HBM growth and the first error in the log.

Triage in this order:

| First observable | Layer | Action |
|---|---|---|
| `Check_path: config.json failed` | Path, ownership or traverse permission | Fix the path and mode; do not edit JSON yet |
| JSON parse error | Config syntax | Remove comments/trailing commas; validate with `python -m json.tool` |
| `worldSize` / device-count error | Placement arithmetic | Reconcile `worldSize`, `npuDeviceIds` and visible devices |
| `npuMemSize` error, OOM or startup hang | HBM/KV budget | Lower `npuMemSize`; free the device; recheck weights and reserve |
| Certificate/key/password error | TLS | Fix paths, permissions, chain, SAN, expiry and key match |
| `Address already in use` | Port conflict | Find the listener with `ss -lntp`; choose a free port or stop the old process |
| `aclInit error code 107001` / `Invalid device ID` | Device visibility | Check device mappings, driver mount and visible-device variables |
| `HCCL_CONNECT_TIMEOUT` or ranktable error | Multi-node communication | Check ranktable, network, driver and group symmetry |
| `maxSeqLen` / `maxInputTokenLen` validation | Sequence budget | Fix the inequality; recheck the worst-case output budget |
| Daemon exits with no log | Log path, permissions or stdout redirection | Check the log directory and run in the foreground |

Change one layer at a time. If you change the config, the certificate and the device visibility together, a successful restart does not tell you which fix mattered.

# Boundaries

This SOP is a configuration and startup contract. It does not claim that a particular MindIE build was started in this repository, and it does not replace a live service acceptance run. Field names, valid ranges, default values and log filenames vary by MindIE release; the installed release's schema and release notes are authoritative. Performance, model accuracy and multi-node capacity require separate measured evidence.
