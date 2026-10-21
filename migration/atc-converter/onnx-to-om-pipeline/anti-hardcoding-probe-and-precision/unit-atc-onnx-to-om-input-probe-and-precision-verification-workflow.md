---
schema: mindie-graded-unit/1
id: unit-atc-onnx-to-om-input-probe-and-precision-verification-workflow
revision: 1
title: "ATC ONNX-to-OM Model Conversion: Dynamic Input Probing, Anti-Hardcoding Rules, and Precision Triage Workflow"
tier: L3
category: workflow
evidence_strength: software_contract
status: active
domain_path: migration/atc-converter/onnx-to-om-pipeline/anti-hardcoding-probe-and-precision
inherits_from:
  - unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
  - unit-ge-graph-compile-stage-and-dynamic-shape-partition-sop
conditions:
  hardware:
    - Ascend 910B
    - Ascend 910C
    - Ascend 310P
  domain: migration
triggers:
  error_signatures:
    - "E10001"
    - "E19010"
    - "E40021"
    - "atc --model"
  keywords:
    - "atc"
    - "onnx-to-om"
    - "soc-version"
    - "input-shape"
    - "dynamic-batch-size"
    - "precision-mode-v2"
    - "ais-bench"
provenance:
  source_type: source_contract
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/atc-model-converter/SKILL.md
  evidence_refs:
    - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/atc-model-converter/SKILL.md#L1-L210
  supersedes: []
---

## Falsifiable Claim

When converting ONNX models (`.onnx`) to Ascend Offline Models (`.om`) with `atc`, hardcoding `--soc_version` or guessing `--input_shape` without programmatic inspection causes either `E10001` input node mismatch or silent precision degradation (`cosine_similarity < 0.99`). A deterministic conversion pipeline must dynamically probe `npu-smi info` for exact `soc_version`, parse the ONNX graph with `onnx` to extract exact input names/dtypes/symbolic axes, simplify redundant subgraphs via `onnxsim`, compile with explicit `--precision_mode_v2`, and verify end-to-end cosine similarity ($\ge 0.99$) against ONNXRuntime CPU outputs.

## Five-Stage Conversion & Verification Workflow

### Stage 1: Programmatic SoC & ONNX Input Signature Probe (Anti-Hardcoding)

1. **Query Exact `soc_version`**:
   - Run `npu-smi info` and combine the `Name` field (`910B1`, `910B2`, `910B3`, `910B4`, `310P3`) with the `"Ascend"` prefix (e.g., `--soc_version=Ascend910B2` or `--soc_version=Ascend310P3`). Never hardcode `Ascend910B` without the trailing chip variant digit.
2. **Inspect ONNX Graph Inputs & Symbolic Axes**:
   ```python
   import onnx

   model = onnx.load("model.onnx")
   init_names = {init.name for init in model.graph.initializer}
   for inp in model.graph.input:
       if inp.name in init_names:
           continue
       dims = [
           d.dim_value if d.dim_value > 0 else d.dim_param or -1
           for d in inp.type.tensor_type.shape.dim
       ]
       elem_type = inp.type.tensor_type.elem_type
       print(f"Input: {inp.name}, shape={dims}, elem_type={elem_type}")
   ```

### Stage 2: ONNX Graph Simplification (`onnxsim`)

Before invoking `atc`, fold constant shape-manipulation chains (`Shape -> Gather -> Unsqueeze -> Concat`) that otherwise trigger AICPU fallback or dynamic shape partition splits in GE:

```bash
python3 -m onnxsim model.onnx model_sim.onnx
```

### Stage 3: Execute `atc` with Static or Gear-Based Dynamic Shapes

- **Fixed Input Shapes**:
  ```bash
  atc --framework=5 \
      --model=model_sim.onnx \
      --output=model_static \
      --input_format=NCHW \
      --input_shape="input_ids:1,512;attention_mask:1,512" \
      --soc_version=Ascend910B2 \
      --precision_mode_v2=origin
  ```
- **Multi-Gear Dynamic Batch / Resolution (`--dynamic_batch_size` / `--dynamic_dims`)**:
  Set the dynamic axis to `-1` in `--input_shape` and declare up to 100 discrete gears in `--dynamic_batch_size` or `--dynamic_dims`:
  ```bash
  atc --framework=5 \
      --model=model_sim.onnx \
      --output=model_dyn \
      --input_format=ND \
      --input_shape="input_ids:-1,512;attention_mask:-1,512" \
      --dynamic_batch_size="1,2,4,8,16" \
      --soc_version=Ascend910B2
  ```
  *(Constraint: `--dynamic_batch_size`, `--dynamic_image_size`, and `--dynamic_dims` are mutually exclusive in a single `atc` invocation).*

### Stage 4: End-to-End Numerical Precision Audit (`onnxruntime` vs. `ais_bench`)

1. Generate deterministic random inputs (`np.random.seed(42)`) matching each input's exact dtype (`float32`, `int64`, `float16`) and save them as `.bin` / `.npy` files.
2. Run CPU reference inference using `onnxruntime.InferenceSession("model_sim.onnx")`.
3. Run NPU inference on `model_static.om` using `ais_bench` (`python3 -m ais_bench --model model_static.om --input input_0.bin,input_1.bin --output ./out/`).
4. Compute **Cosine Similarity**, **Max Absolute Error**, and **Mean Relative Error**:
   - **Pass Gate**: $\text{Cosine Similarity} \ge 0.99$ and $\text{Relative Error} < 1\%$.

### Stage 5: Precision Loss Escalation Ladder

If $\text{Cosine Similarity} < 0.99$:
1. Re-compile with `--precision_mode_v2=origin` (or `--precision_mode=force_fp32` / `must_keep_origin_dtype`) to prevent Cube FP16/HF32 downcasting on sensitive norms or Softmax layers.
2. Enable mixed-precision blacklisting via `--modify_mixlist=ops_info.json` to keep specific overflowing layers (`LayerNorm`, `ReduceMean`, `Softmax`, `Pow`) in FP32 while leaving `MatMul`/`Conv` in FP16.
