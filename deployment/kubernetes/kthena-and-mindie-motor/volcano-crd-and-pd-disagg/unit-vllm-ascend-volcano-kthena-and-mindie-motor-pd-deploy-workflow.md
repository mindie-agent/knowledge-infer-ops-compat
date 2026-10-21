---
schema: mindie-graded-unit/1
id: unit-vllm-ascend-volcano-kthena-and-mindie-motor-pd-deploy-workflow
revision: 1
title: "Cloud-Native vLLM-Ascend Orchestration: Volcano Kthena ModelServing CRD and MindIE Motor PD Deployment Workflow"
tier: L3
category: workflow
evidence_strength: software_contract
status: active
domain_path: deployment/kubernetes/kthena-and-mindie-motor/volcano-crd-and-pd-disagg
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-mindie-service-config-security-and-npu-memory-budget-sop
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
  domain: deployment
triggers:
  error_signatures:
    - "workload.serving.volcano.sh/v1alpha1"
    - "ModelServing"
    - "huawei.com/Ascend910"
    - "mindie-motor"
  keywords:
    - "volcano"
    - "kthena"
    - "modelserving-crd"
    - "mindie-motor"
    - "gang-scheduling"
    - "topology-aware-scheduling"
provenance:
  source_type: source_contract
  source_locator: vllm-project/vllm-ascend@175ef4d49c2072f9233619636f0309f4f88b4e2c:docs/source/user_guide/deployment_guide/using_volcano_kthena.md
  evidence_refs:
    - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/user_guide/deployment_guide/using_volcano_kthena.md#L1-L195
    - https://github.com/vllm-project/vllm-ascend/blob/175ef4d49c2072f9233619636f0309f4f88b4e2c/docs/source/user_guide/deployment_guide/using_mindie_motor.md#L1-L150
  supersedes: []
---

## Falsifiable Claim

Deploying multi-node or Prefill-Decode (PD) disaggregated `vllm-ascend` services on Kubernetes with standard `Deployment` controllers causes partial-pod scheduling deadlocks (where Leader/Prefill pods hold NPUs while Worker/Decode pods stay `Pending`) and cross-HCCS/cross-ToR placement penalties. Production Ascend K8s clusters must use either **Volcano + Kthena (`ModelServing` CRD)** for gang scheduling and HCCS/network topology-aware placement, or **MindIE Motor** as the industrial control plane (Coordinator + Controller + NodeManager) managing `vllm-ascend` execution engines.

## Architecture Option A: Kubernetes Deployment with Volcano & Kthena (`ModelServing` CRD)

### 1. Cluster Prerequisites
- **Ascend Device Plugin & Volcano Scheduler**: Exposes `huawei.com/Ascend910` (or `huawei.com/Ascend910B`) extended resources and injects HCCS ring/superpod topology annotations onto K8s nodes.
- **Kthena (`workload.serving.volcano.sh/v1alpha1`)**: Provides the `ModelServing` CRD with `role` groups (e.g., `prefill` and `decode`, each containing an `entryPod` leader and `workerPods`), atomic gang scheduling via Volcano `PodGroup`, and role-level rolling updates/autoscaling.

### 2. `ModelServing` CRD Configuration Contract
- Set `schedulerName: volcano` and `gangPolicy` so an entire multi-node TP/EP instance (or matched Prefill + Decode role group) is scheduled atomically.
- Configure `networkTopology` under `gangPolicy` (`mode: hard` or `soft`, `highestTierAllowed: 1` or `2`) to co-locate pods within the same Atlas 800T A2 HCCS ring or Atlas 900 A3 SuperPod chassis.
- Mount `/dev/shm` (`emptyDir: {medium: Memory, sizeLimit: 64Gi}`) and request identical `huawei.com/Ascend910` limits and requests in both `entryTemplate` and `workerTemplate`.

```yaml
apiVersion: workload.serving.volcano.sh/v1alpha1
kind: ModelServing
metadata:
  name: vllm-ascend-deepseek
spec:
  schedulerName: volcano
  replicas: 1
  template:
    gangPolicy:
      minRoleReplicas:
        prefill: 1
        decode: 1
    roles:
      - name: prefill
        replicas: 1
        workerReplicas: 1
        entryTemplate:
          spec:
            containers:
              - name: vllm-prefill-leader
                image: quay.io/ascend/vllm-ascend:latest
                resources:
                  limits:
                    huawei.com/Ascend910: "8"
                  requests:
                    huawei.com/Ascend910: "8"
```

## Architecture Option B: MindIE Motor Control Plane + `vllm-ascend` Engine

When integrating `vllm-ascend` into the **MindIE Motor** distributed serving stack:
1. **Control-Plane Separation**:
   - **MindIE Motor Coordinator**: Receives external OpenAI-compatible HTTP/gRPC inference requests, runs global KV-cache prefix-aware and load-aware PD routing, and streams token chunks back to clients.
   - **MindIE Motor Controller & NodeManager**: Manages instance lifecycle, global rank-table synthesis, health heartbeats, and fault recovery across Prefill (`P`) and Decode (`D`) groups.
   - **`vllm-ascend` Engine Backend**: Runs as the underlying execution engine inside each P and D pod, communicating KV blocks over Mooncake / HIXL (`AscendDirect`).
2. **Deployment Verification**:
   - Verify that `ascend-device-plugin` has allocated disjoint physical NPUs and populated the rank table (`RANK_TABLE_FILE`) inside each pod before Motor NodeManager spawns the `vllm-ascend` worker processes.
   - Query the Coordinator health endpoint and confirm all P and D instances transition to `Ready` before sending traffic.
