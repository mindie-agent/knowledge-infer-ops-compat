---
schema: mindie-graded-unit/1
id: unit-npugraph-ex-custom-pattern-fusion-pass-and-converter-sop
title: TorchAir npugraph_ex PatternPass Registration, FX Node Splicing, and Custom
  Op Converter SOP
domain_path: graph/npugraph-ex/architecture-and-passes
tier: L2
category: sop
evidence_strength: software_contract
status: active
revision: 1
conditions:
  accelerator: ascend
  compiler_backend: npugraph_ex
  pipeline_stage: pattern-pass-and-converter
  runtime: torch_npu
inherits_from:
- unit-ascend-cann-driver-and-torch-npu-abi-compatibility-axioms
- unit-npugraph-ex-capture-lowering-and-pass-pipeline-axioms
supersedes: []
triggers:
  error_signatures:
  - PatternPassManager register_pattern match failed
  - fx.Node meta['val'] missing after replace_all_uses_with
  - register_fx_node_ge_converter missing for custom op
  - OpOverloadPacket passed instead of OpOverload in pattern search_fn
  keywords:
  - PatternPassManager
  - _register_pattern_passes
  - register_fx_node_ge_converter
  - extra_check Match
  - fx.Graph inserting_before
  - node.meta val propagation
provenance:
  source_kind: external_import
  source_ref: cannbot-skills:graph/npugraph-ex-pattern-pass+npugraph-ex-code-playbook
  source_locator: https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-pattern-pass/SKILL.md
  evidence_refs:
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-pattern-pass/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/graph/npugraph-ex-code-playbook/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/model/model-infer-multi-stream/SKILL.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/model/inference/runbooks/compilation/torch_compile_graph_breaks_and_unsupported_ops.md
  source_type: curated_repo
  source_digest: 8a71a073a5d43c26c5cdee05d9cbc619ab61ca89fe0ff902ad976292759a9962
---

# TorchAir npugraph_ex PatternPass Registration, FX Node Splicing, and Custom Op Converter SOP

## 1. Registering Subgraph Fusion Patterns via `PatternPassManager`

To fuse a multi-operator ATen/NPU sequence (e.g., `Add + RMSNorm`, `RoPE + Cast`, or `Quant + Matmul`) inside `npugraph_ex`, register a pattern pass in `_register_pattern_passes`:

```python
import torch
from torch._inductor.pattern_matcher import Match, CallFunction, KeywordArg, Arg
from torchair.patterns.pattern_pass_manager import _ pattern_manager

def _ rmsnorm_quant_extra_check(match: Match) -> bool:
    x_node = match.kwargs.get("x") or match.args[0]
    val = x_node.meta.get("val")
    if val is None or not isinstance(val, torch.Tensor):
        return False
    # Enforce hardware kernel constraints: last dim 32B aligned, bf16/fp16 only
    return val.dtype in (torch.float16, torch.bfloat16) and (val.shape[-1] % 16 == 0)

def register_rmsnorm_quant_pattern(pm):
    # Always use specific OpOverload (e.g. .default), never OpOverloadPacket
    search_pattern = CallFunction(
        torch.ops.npu.npu_quantize.default,
        CallFunction(
            torch.ops.npu.npu_rms_norm.default,
            KeywordArg("x"),
            KeywordArg("gamma"),
            KeywordArg("epsilon"),
            _users=1,
        ),
        KeywordArg("scales"),
        KeywordArg("zero_points"),
        KeywordArg("dtype"),
        KeywordArg("axis"),
    )
    pm.register_replacement(
        search_fn=search_pattern,
        replace_fn= _replace_with_fused_rmsnorm_quant,
        extra_check=_rmsnorm_quant_extra_check,
    )
```

### Mandatory Pattern Registration Rules
1. **Use `OpOverload`, Never `OpOverloadPacket`**: FX traced graphs record exact overloads (`torch.ops.aten.add.Tensor`, `torch.ops.npu.npu_rms_norm.default`). Using `torch.ops.aten.add` without `.Tensor` silently fails to match.
2. **Validate Intermediate Fan-Out (`_users`)**: If an intermediate node (such as the unquantized `rms_norm` output) is also consumed by a residual branch, fusing it without preserving the unquantized output or setting `_users=2` with multi-output replacement corrupts the graph.
3. **Guard Hardware Preconditions in `extra_check(match)`**: Check `node.meta["val"]` (`FakeTensor`) for dtype, rank, contiguity (`val.is_contiguous()`), and last-dimension alignment before permitting replacement.

## 2. Direct FX Graph Splicing and Metadata Preservation SOP

When writing a custom `fx.GraphModule` pass that splices nodes directly:

```python
def splice_fused_op(gm: torch.fx.GraphModule) -> bool:
    changed = False
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.npu.my_source_op.default:
            continue
        with graph.inserting_before(node):
            fused_node = graph.call_function(
                torch.ops.npu.my_fused_op.default,
                args=node.args,
                kwargs=node.kwargs,
            )
            # CRITICAL: Copy FakeTensor metadata so downstream passes and ACLGraph capture succeed
            fused_node.meta["val"] = node.meta.get("val")
            fused_node.meta["tensor_meta"] = node.meta.get("tensor_meta")
        node.replace_all_uses_with(fused_node)
        graph.erase_node(node)
        changed = True
    if changed:
        graph.lint()
        gm.recompile()
    return changed
```

- **Metadata Propagation Invariant**: Every newly inserted `fx.Node` MUST populate `node.meta["val"]` (a `FakeTensor` or tuple of `FakeTensor`s matching the target schema's `Meta` implementation). Missing `meta["val"]` causes downstream `reinplace`, `replace_dynamic_workspace_ops`, or `extra_check` functions to crash or skip optimization.
- **Always Run `graph.lint()` and `gm.recompile()`**: Erasing a node that still has surviving users raises an immediate topological error during `graph.lint()`.

## 3. Custom Op Lowering: ACLGraph vs GE Converter Contract

1. **Both ACLGraph and GE Modes Require `Meta` Registration**: Register a `@torch.library.register_fake("namespace::op")` (or `TORCH_LIBRARY_IMPL(namespace, Meta, m)`) returning empty tensors with exact output shapes, dtypes, and strides.
2. **ACLGraph Mode**: Directly records the `PrivateUse1` C++ kernel dispatch (`EXEC_NPU_CMD` / `ACLRT_LAUNCH_KERNEL`) onto the capture stream; no GE converter is invoked.
3. **GE Graph Mode**: Additionally requires `@torchair.register_fx_node_ge_converter(torch.ops.namespace.op.default)` to emit the corresponding `ge.custom_op` IR node and bind input/output `TensorSpec` descriptors.

## 4. Multi-Stream Overlap Splicing Checklist

1. Wrap only the secondary branch inside `with torchair.scope.npu_stream_switch("aux_stream", 0):`.
2. Insert `wait_in = torchair.ops.npu_wait_tensor(branch_input, upstream_producer)` at the entry of the secondary stream so the secondary stream does not read upstream buffer memory before the main stream finishes writing it.
3. Insert `merged = torchair.ops.npu_wait_tensor(main_branch_out, aux_branch_out)` on the main stream before combining outputs (`add` / `cat`) so the main stream waits for secondary stream completion.
