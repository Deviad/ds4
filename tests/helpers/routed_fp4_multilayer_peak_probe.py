#!/usr/bin/env python3
"""Story 13.3b-5g synthetic multi-layer peak diagnostics.

No real model/shard/dataset/config is opened.  The parent process runs cells
serially; every measurement cell runs in a fresh subprocess with an 8 GB MLX
memory limit and a bounded timeout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MLX_SRC = ROOT / "python-envs" / "mlx" / "src"
REPORT = ROOT / "agent-output" / "cmux-13-3b" / "multilayer-peak-report.json"
INTERACTION_REPORT = ROOT / "agent-output" / "cmux-13-3b" / "interaction-ablation-report.json"
FIXTURE = ROOT / "tests" / "fixtures" / "deepseek_v4_nn_multilayer_peak_topology.json"
MEMORY_LIMIT = 8_000_000_000
TIMEOUT_S = 180
INTERACTION_THRESHOLDS = {"collapsed_max_delta": 227_479_736, "persistent_min_delta": 941_632_822}
LORA_KEYS = ("self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj")
LORA_CONFIG = {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": set(LORA_KEYS)}
SCHEMA = [
    "case", "depth", "experts", "nonempty_experts", "tokens", "hidden", "intermediate", "top_k",
    "hc_mult", "compression_ratio", "checkpoint", "component_mask", "active_baseline",
    "active_after_graph", "peak_forward", "peak_backward", "active_final", "cache_final",
    "custom_kernel_nodes", "duration_s", "finite_loss", "finite_gradients", "exit_code", "signal", "error",
]
EVIDENCE_SCHEMA = [
    "ordinal", "cell_identity_sha256", "child_pid", "child_nonce", "started_monotonic_ns",
    "finished_monotonic_ns", "trainable_key_count", "trainable_keys_sha256", "input_gradient_shape",
    "input_gradient_nonzero", "measured_custom_kernel_nodes",
]
INTERACTION_SCHEMA = SCHEMA[:16] + [
    "pre_boundary_active", "post_boundary_active",
] + SCHEMA[16:] + [
    "loss", "forward_exact", "forward_max_abs_diff", "materialized_loss_abs_diff",
    "materialized_lora_grad_max_abs_diff", "materialized_lora_grad_max_rel_diff",
    "materialized_input_grad_max_abs_diff", "materialized_input_grad_max_rel_diff",
    "materialized_allclose_atol_rtol_1e_5",
]
PROVENANCE_SOURCES = (
    "docs/technical-spec.md",
    "docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md",
    "agent-output/cmux-13-3b/architecture-13-3b-5g-first-backward-oom.md",
    "agent-output/cmux-13-3b/requirements-13-3b-5g-synthetic-peak.md",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_row(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: None for k in SCHEMA} | {
        "case": cell["case"],
        "depth": cell.get("depth"),
        "experts": cell.get("experts"),
        "nonempty_experts": cell.get("nonempty_experts"),
        "tokens": cell.get("tokens"),
        "hidden": cell.get("hidden"),
        "intermediate": cell.get("intermediate"),
        "top_k": cell.get("top_k"),
        "hc_mult": cell.get("hc_mult"),
        "compression_ratio": cell.get("compression_ratio"),
        "checkpoint": cell.get("checkpoint"),
        "component_mask": cell.get("component_mask"),
        "exit_code": 0,
        "signal": None,
        "error": None,
    }


def base_interaction_row(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: None for k in INTERACTION_SCHEMA} | {
        "case": cell["case"],
        "depth": cell.get("depth"),
        "experts": cell.get("experts"),
        "nonempty_experts": cell.get("nonempty_experts"),
        "tokens": cell.get("tokens"),
        "hidden": cell.get("hidden"),
        "intermediate": cell.get("intermediate"),
        "top_k": cell.get("top_k"),
        "hc_mult": cell.get("hc_mult"),
        "compression_ratio": cell.get("compression_ratio"),
        "checkpoint": cell.get("checkpoint"),
        "component_mask": cell.get("component_mask"),
        "exit_code": 0,
        "signal": None,
        "error": None,
    }


def plan_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for depth in (1, 2, 4, 8, 16, 43):
        cells.append({"case": "probe_b_depth", "depth": depth, "experts": 8, "nonempty_experts": 8, "tokens": 64, "hidden": 128, "intermediate": 64, "top_k": 2, "hc_mult": 4, "compression_ratio": 0, "checkpoint": True, "component_mask": "full"})
    for depth in (1, 4, 8):
        cells.append({"case": "probe_b_depth", "depth": depth, "experts": 8, "nonempty_experts": 8, "tokens": 64, "hidden": 128, "intermediate": 64, "top_k": 2, "hc_mult": 4, "compression_ratio": 0, "checkpoint": False, "component_mask": "full"})
    for depth in (1, 8, 16, 43):
        for experts in (2, 8, 32, 128, 256):
            top_k = min(6, experts)
            nonempty = min(experts, 64 * top_k)
            common = {"depth": depth, "experts": experts, "nonempty_experts": nonempty, "tokens": 64, "hidden": 64, "intermediate": 32, "top_k": top_k, "hc_mult": None, "compression_ratio": 0, "checkpoint": None, "component_mask": "routed_only"}
            cells.append({"case": "probe_c_routed_one_graph", **common})
            cells.append({"case": "probe_c_routed_sequential", **common})
    for component in ("full", "no_routed", "no_shared", "no_attention", "routed_only"):
        for cr in (0, 4, 128):
            cells.append({"case": "probe_d_ablation", "depth": 43, "experts": 8, "nonempty_experts": 8, "tokens": 64, "hidden": 128, "intermediate": 64, "top_k": 2, "hc_mult": 4, "compression_ratio": cr, "checkpoint": True, "component_mask": component})
    return cells


def cell_identity(cell: dict[str, Any]) -> str:
    payload = {k: cell.get(k) for k in SCHEMA[:12]}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def plan_identity_hash(cells: list[dict[str, Any]] | None = None) -> str:
    ids = [cell_identity(c) for c in (cells or plan_cells())]
    return sha256_text("\n".join(ids))


def plan() -> dict[str, Any]:
    cells = plan_cells()
    return {
        "story": "13.3b-5g",
        "memory_limit": MEMORY_LIMIT,
        "timeout_s": TIMEOUT_S,
        "real_assets_accessed": False,
        "cells": cells,
        "cell_identity_sha256": [cell_identity(c) for c in cells],
        "planned_cell_list_sha256": plan_identity_hash(cells),
    }


def interaction_plan_cells() -> list[dict[str, Any]]:
    common = {
        "experts": 8,
        "nonempty_experts": 8,
        "tokens": 64,
        "hidden": 128,
        "intermediate": 64,
        "top_k": 2,
        "hc_mult": 4,
        "compression_ratio": 4,
        "component_mask": "no_shared",
    }
    cells: list[dict[str, Any]] = []
    cells.extend({"case": "composed_checkpoint_on", "depth": d, "checkpoint": True, **common} for d in (1, 2, 4, 8, 16, 43))
    cells.extend({"case": "composed_checkpoint_off", "depth": d, "checkpoint": False, **common} for d in (1, 4, 8))
    cells.extend({"case": "attention_forward_routed_vjp", "depth": d, "checkpoint": True, **common} for d in (1, 8, 16, 43))
    cells.extend({"case": "routed_forward_attention_vjp", "depth": d, "checkpoint": True, **common} for d in (1, 8, 16, 43))
    cells.extend({"case": "materialized_attention_routed_boundary", "depth": d, "checkpoint": True, **common} for d in (1, 8, 16, 43))
    cells.extend({"case": "composed_layer_sequential", "depth": d, "checkpoint": True, **common} for d in (1, 8, 16, 43))
    return cells


def interaction_cell_identity(cell: dict[str, Any]) -> str:
    payload = {k: cell.get(k) for k in SCHEMA[:12]}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def interaction_plan_identity_hash(cells: list[dict[str, Any]] | None = None) -> str:
    ids = [interaction_cell_identity(c) for c in (cells or interaction_plan_cells())]
    return sha256_text("\n".join(ids))


def interaction_plan() -> dict[str, Any]:
    cells = interaction_plan_cells()
    return {
        "story": "13.3b-5h",
        "memory_limit": MEMORY_LIMIT,
        "timeout_s": TIMEOUT_S,
        "real_assets_accessed": False,
        "classification_thresholds": dict(INTERACTION_THRESHOLDS),
        "cells": cells,
        "cell_identity_sha256": [interaction_cell_identity(c) for c in cells],
        "planned_cell_list_sha256": interaction_plan_identity_hash(cells),
    }


def _import_mlx():
    if str(MLX_SRC) not in sys.path:
        sys.path.insert(0, str(MLX_SRC))
    import mlx.core as mx
    return mx


def _setup_mlx():
    mx = _import_mlx()
    mx.disable_compile()
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(0)
    mx.clear_cache()
    return mx


def _config(depth: int, *, T: int, H: int, I: int, E: int, K: int, hc: int, cr: int) -> dict[str, Any]:
    return dict(
        model_type="deepseek_v4_nn", vocab_size=max(128, T + 1), hidden_size=H,
        num_hidden_layers=depth, num_hash_layers=0, mlp_layer_types=["moe"] * depth,
        hc_mult=hc, hc_sinkhorn_iters=3, n_routed_experts=E, num_experts_per_tok=K,
        n_shared_experts=1, moe_intermediate_size=I, expert_dtype="fp4",
        num_attention_heads=4 if H >= 128 else 2, num_key_value_heads=1, head_dim=16,
        q_lora_rank=16, o_lora_rank=16, o_groups=2, qk_rope_head_dim=4,
        compression_ratio=cr, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        swiglu_limit=10.0, rms_norm_eps=1e-6,
    )


def _fixture_config(derived: dict[str, Any], *, T: int, cr: int) -> dict[str, Any]:
    cfg = dict(derived["model_defaults"])
    cfg.update(
        vocab_size=max(int(cfg["vocab_size"]), T + 1),
        num_hidden_layers=derived["num_hidden_layers"],
        mlp_layer_types=derived["mlp_layer_types"],
        hidden_size=derived["hidden_size"],
        moe_intermediate_size=derived["moe_intermediate_size"],
        n_routed_experts=derived["n_routed_experts"],
        num_experts_per_tok=derived["num_experts_per_tok"],
        hc_mult=derived["hc_mult"],
        compression_ratio=cr,
    )
    return cfg


def _fill_model_fp4(model: Any, mx: Any) -> None:
    for idx, layer in enumerate(model.model.layers):
        ex = layer.mlp.experts
        byte = 0x11 + (idx % 7)
        ex.w1_weight = mx.full(ex.w1_weight.shape, byte, dtype=mx.uint8)
        ex.w2_weight = mx.full(ex.w2_weight.shape, byte + 1, dtype=mx.uint8)
        ex.w3_weight = mx.full(ex.w3_weight.shape, byte + 2, dtype=mx.uint8)
        scale = 0.01 + (idx % 5) * 0.001
        ex.w1_scale = mx.full(ex.w1_scale.shape, scale, dtype=mx.bfloat16)
        ex.w2_scale = mx.full(ex.w2_scale.shape, scale, dtype=mx.bfloat16)
        ex.w3_scale = mx.full(ex.w3_scale.shape, scale, dtype=mx.bfloat16)
        layer.mlp.gate_weight = mx.full(layer.mlp.gate_weight.shape, 0.001 + idx * 0.00001, dtype=mx.float32)
        layer.mlp.e_score_correction_bias = mx.arange(layer.mlp.n_routed_experts, dtype=mx.float32) * 1e-5


def _tree_flat(tree: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten
    return list(tree_flatten(tree))


def _tree_values(tree: Any) -> list[Any]:
    return [v for _, v in _tree_flat(tree)]


def _tree_keys(tree: Any) -> list[str]:
    return sorted(k for k, _ in _tree_flat(tree))


def _hash_keys(keys: list[str]) -> str:
    return sha256_text("\n".join(sorted(keys)))


def _finite_array(value: Any, mx: Any) -> bool:
    return bool(mx.all(mx.isfinite(value)).item())


def _finite_tree(tree: Any, mx: Any) -> bool:
    vals = _tree_values(tree)
    if not vals:
        return False
    mx.eval(*vals)
    return all(_finite_array(v, mx) for v in vals)


def _nonzero_array(value: Any, mx: Any) -> bool:
    return bool(mx.any(mx.abs(value.astype(mx.float32)) > 0).item())


def _max_array_diff(a: Any, b: Any, mx: Any) -> tuple[float, float, bool]:
    aa = a.astype(mx.float32); bb = b.astype(mx.float32)
    abs_diff = mx.abs(aa - bb)
    max_abs = float(mx.max(abs_diff).item())
    rel_diff = abs_diff / mx.maximum(mx.abs(bb), mx.array(1e-12, dtype=mx.float32))
    max_rel = float(mx.max(rel_diff).item())
    tolerance = mx.array(1e-5, dtype=mx.float32) + mx.array(1e-5, dtype=mx.float32) * mx.abs(bb)
    allclose = bool(mx.all(abs_diff <= tolerance).item())
    return max_abs, max_rel, allclose


def _max_grad_diff(a: Any, b: Any, mx: Any) -> tuple[float, float, bool, bool, list[str], str, list[str]]:
    aa = dict(_tree_flat(a)); bb = dict(_tree_flat(b))
    keys_a = sorted(aa); keys_b = sorted(bb)
    keys_equal = bool(keys_a) and keys_a == keys_b
    shape_mismatches = [k for k in sorted(set(keys_a) & set(keys_b)) if tuple(aa[k].shape) != tuple(bb[k].shape)]
    max_abs = 0.0; max_rel = 0.0; allclose = keys_equal and not shape_mismatches
    for k in keys_a if keys_equal else sorted(set(keys_a) & set(keys_b)):
        da = aa[k].astype(mx.float32); db = bb[k].astype(mx.float32)
        abs_diff = mx.abs(da - db)
        diff = float(mx.max(abs_diff).item())
        rel = float(mx.max(abs_diff / mx.maximum(mx.abs(db), mx.array(1e-12, dtype=mx.float32))).item())
        max_abs = max(max_abs, diff); max_rel = max(max_rel, rel)
        tolerance = mx.array(1e-5, dtype=mx.float32) + mx.array(1e-5, dtype=mx.float32) * mx.abs(db)
        allclose = allclose and bool(mx.all(abs_diff <= tolerance).item())
    return max_abs, max_rel, keys_equal, allclose, keys_a, _hash_keys(keys_a), shape_mismatches


def _count_custom_kernel_nodes(mx: Any, outputs: list[Any]) -> int:
    with tempfile.NamedTemporaryFile(prefix="ds4-13-3b-5g-", suffix=".dot", delete=False) as f:
        path = f.name
    try:
        mx.export_to_dot(path, *outputs)
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.lstrip()
                if "CustomKernel" in stripped and "->" not in stripped:
                    count += 1
        return count
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _zero_identity_like(x: Any) -> Any:
    mx = _import_mlx()
    return x - mx.stop_gradient(x)


def _zero_residual_identity_like(residual: Any, like: Any) -> Any:
    mx = _import_mlx()
    return mx.mean(residual - mx.stop_gradient(residual), axis=-2).astype(like.dtype)


def _routed_only(block: Any, x: Any, input_ids: Any | None) -> Any:
    mx = _import_mlx()
    from ds4_ft_mlx.routed_fp4_metal import routed_fp4
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import _host_unique_rows_per_expert

    scores, indices, _denom = block._route(x, input_ids)
    orig_shape = x.shape
    T = int(__import__("math").prod(orig_shape[:-1]))
    H = int(orig_shape[-1])
    x_flat = x.reshape(T, H)
    scores_flat = scores.reshape(T, block.n_routed_experts)
    indices_flat = indices.reshape(T, block.top_k)
    rows_by_expert = _host_unique_rows_per_expert(mx.stop_gradient(indices_flat), block.n_routed_experts)
    routed_flat = routed_fp4(x_flat, scores_flat, rows_by_expert, block.experts, routed_scaling_factor=block.routed_scaling_factor)
    return routed_flat.reshape(orig_shape)


def _masked_layer_forward(layer: Any, hidden_states: Any, input_ids: Any, component: str) -> Any:
    mx = _import_mlx()
    dtype = hidden_states.dtype
    post, comb, collapsed = layer.attn_hc(hidden_states)
    attn_input = layer.input_layernorm(collapsed)
    if component in ("no_attention", "routed_only"):
        attn_output = _zero_residual_identity_like(hidden_states, attn_input)
    else:
        attn_output = layer.self_attn(attn_input, cache=None)
    hidden_states = mx.expand_dims(post.astype(dtype), -1) * mx.expand_dims(attn_output, -2) + (
        comb.astype(dtype).transpose(0, 1, 3, 2) @ hidden_states
    )

    post, comb, collapsed = layer.ffn_hc(hidden_states)
    mlp_input = layer.post_attention_layernorm(collapsed)
    if component in ("full", "no_attention"):
        mlp_output = layer.mlp(mlp_input, input_ids=input_ids)
    elif component == "no_routed":
        mlp_output = layer.mlp.shared_experts(mlp_input) + _zero_residual_identity_like(hidden_states, mlp_input)
    elif component == "no_shared":
        mlp_output = _routed_only(layer.mlp, mlp_input, input_ids) + _zero_residual_identity_like(hidden_states, mlp_input)
    elif component == "routed_only":
        mlp_output = _routed_only(layer.mlp, mlp_input, input_ids) + _zero_residual_identity_like(hidden_states, mlp_input)
    else:
        raise ValueError(f"unknown component mask {component!r}")
    return mx.expand_dims(post.astype(dtype), -1) * mx.expand_dims(mlp_output, -2) + (
        comb.astype(dtype).transpose(0, 1, 3, 2) @ hidden_states
    )


def model_forward_once(model: Any, ids: Any, input_probe: Any | None = None, component_mask: str = "full") -> Any:
    mx = _import_mlx()
    h = model.model.embed_tokens(ids)
    h = mx.broadcast_to(mx.expand_dims(h, -2), (*h.shape[:-1], model.args.hc_mult, model.args.hidden_size))
    if input_probe is not None:
        h = h + (input_probe - mx.stop_gradient(input_probe))
    for layer in model.model.pipeline_layers:
        h = _masked_layer_forward(layer, h, ids, component_mask)
    return model.lm_head(model.model.norm(model.model.hc_head(h)))


def _boundary_mark(telemetry: dict[str, int] | None, key: str, value: int) -> None:
    if telemetry is not None:
        telemetry[key] = max(int(value), int(telemetry.get(key, 0)))


def _interaction_layer_forward(layer: Any, hidden_states: Any, input_ids: Any, mode: str, telemetry: dict[str, int] | None = None) -> Any:
    mx = _import_mlx()
    dtype = hidden_states.dtype
    post, comb, collapsed = layer.attn_hc(hidden_states)
    attn_input = layer.input_layernorm(collapsed)
    attn_output = layer.self_attn(attn_input, cache=None)
    after_attention = mx.expand_dims(post.astype(dtype), -1) * mx.expand_dims(attn_output, -2) + (
        comb.astype(dtype).transpose(0, 1, 3, 2) @ hidden_states
    )
    _boundary_mark(telemetry, "pre_boundary_active", mx.get_active_memory())
    if mode == "attention_forward_routed_vjp":
        ffn_hidden = mx.stop_gradient(after_attention) + _zero_identity_like(hidden_states)
    else:
        ffn_hidden = after_attention
    if mode == "materialized_attention_routed_boundary":
        _materialize_resident(mx, ffn_hidden)
    _boundary_mark(telemetry, "post_boundary_active", mx.get_active_memory())

    post, comb, collapsed = layer.ffn_hc(ffn_hidden)
    mlp_input = layer.post_attention_layernorm(collapsed)
    routed = _routed_only(layer.mlp, mlp_input, input_ids)
    if mode == "routed_forward_attention_vjp":
        mlp_output = mx.stop_gradient(routed) + _zero_residual_identity_like(ffn_hidden, mlp_input)
    else:
        mlp_output = routed + _zero_residual_identity_like(ffn_hidden, mlp_input)
    return mx.expand_dims(post.astype(dtype), -1) * mx.expand_dims(mlp_output, -2) + (
        comb.astype(dtype).transpose(0, 1, 3, 2) @ ffn_hidden
    )


def interaction_forward_once(model: Any, ids: Any, input_probe: Any | None = None, mode: str = "composed_checkpoint_on", telemetry: dict[str, int] | None = None) -> Any:
    mx = _import_mlx()
    h = model.model.embed_tokens(ids)
    h = mx.broadcast_to(mx.expand_dims(h, -2), (*h.shape[:-1], model.args.hc_mult, model.args.hidden_size))
    if input_probe is not None:
        h = h + (input_probe - mx.stop_gradient(input_probe))
    layer_mode = "composed" if mode in ("composed_checkpoint_on", "composed_checkpoint_off", "composed_layer_sequential") else mode
    for layer in model.model.pipeline_layers:
        h = _interaction_layer_forward(layer, h, ids, layer_mode, telemetry)
    return model.lm_head(model.model.norm(model.model.hc_head(h)))


def _direct_routed_branch_output(layer: Any, hidden_states: Any, ids: Any) -> tuple[Any, Any, Any]:
    mlp_input = layer.post_attention_layernorm(layer.ffn_hc(hidden_states)[2])
    routed = _routed_only(layer.mlp, mlp_input, ids)
    shared = layer.mlp.shared_experts(mlp_input)
    full = layer.mlp(mlp_input, input_ids=ids)
    return routed, shared, full


def _quantize_applicable_base(model: Any) -> None:
    import mlx.nn as nn
    def predicate(_path: str, module: Any) -> bool:
        weight = getattr(module, "weight", None)
        return isinstance(module, nn.Linear) and weight is not None and tuple(weight.shape)[-1] % 32 == 0
    nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)


def _expected_lora_keys(depth: int) -> list[str]:
    first = depth - min(depth, 16)
    out: list[str] = []
    for idx in range(first, depth):
        for proj in LORA_KEYS:
            for leaf in ("lora_a", "lora_b"):
                out.append(f"model.layers.{idx}.{proj}.{leaf}")
    return sorted(out)


def _assert_lora_contract(model: Any, depth: int) -> tuple[list[str], str, dict[str, list[int]]]:
    params = model.trainable_parameters()
    keys = _tree_keys(params)
    expected = _expected_lora_keys(depth)
    if keys != expected:
        missing = sorted(set(expected) - set(keys))[:10]
        extra = sorted(set(keys) - set(expected))[:10]
        raise AssertionError(f"LoRA trainable key mismatch: missing={missing} extra={extra} count={len(keys)} expected={len(expected)}")
    shapes: dict[str, list[int]] = {}
    modules = dict(model.named_modules())
    for key in expected:
        module_key, leaf = key.rsplit(".", 1)
        module = modules[module_key]
        arr = getattr(module, leaf)
        shapes[key] = list(arr.shape)
        if leaf == "lora_a" and arr.shape[-1] != 8:
            raise AssertionError(f"{key} rank mismatch {arr.shape}")
        if leaf == "lora_b" and arr.shape[0] != 8:
            raise AssertionError(f"{key} rank mismatch {arr.shape}")
    return keys, _hash_keys(keys), shapes


def _prepare_lora_model(model: Any, depth: int) -> tuple[list[str], str, dict[str, list[int]]]:
    from mlx_lm.tuner.utils import linear_to_lora_layers
    _quantize_applicable_base(model)
    model.freeze()
    linear_to_lora_layers(model, min(depth, 16), dict(LORA_CONFIG))
    return _assert_lora_contract(model, depth)


def _materialize_resident(mx: Any, *values: Any) -> None:
    vals: list[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            vals.extend(value)
        else:
            vals.append(value)
    if vals:
        mx.eval(*vals)
    mx.clear_cache()


def run_probe_a() -> dict[str, Any]:
    mx = _setup_mlx()
    from mlx_lm.tuner.trainer import grad_checkpoint
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4_nn
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs, DecoderLayerNN

    original = DecoderLayerNN.__call__
    cfg = _config(43, T=4, H=16, I=16, E=2, K=2, hc=2, cr=0)
    ids = mx.arange(4, dtype=mx.int32).reshape(1, 4)

    def make_model():
        mx.random.seed(1337)
        model = Model(ModelArgs.from_dict(cfg))
        _fill_model_fp4(model, mx)
        return model

    def loss_for(model, params):
        model.update(params)
        logits = model(ids)
        return mx.mean(logits * logits)

    try:
        off_model = make_model()
        off_loss, off_grads = mx.value_and_grad(lambda p: loss_for(off_model, p))(off_model.trainable_parameters())
        mx.eval(off_loss, *_tree_values(off_grads))
        finite_loss_off = _finite_array(off_loss, mx)
        finite_gradients_off = _finite_tree(off_grads, mx)

        on_model = make_model()
        layer_count = len(on_model.layers)
        decoder_count = sum(isinstance(layer, DecoderLayerNN) for layer in on_model.layers)
        grad_checkpoint(on_model.layers[0])
        checkpointed = DecoderLayerNN.__call__
        wrapper_name = getattr(checkpointed, "__name__", "")
        calls = {"n": 0}

        def counted(self, *args, **kwargs):
            calls["n"] += 1
            return checkpointed(self, *args, **kwargs)

        DecoderLayerNN.__call__ = counted
        y = on_model(ids)
        mx.eval(y)
        forward_calls = calls["n"]
        DecoderLayerNN.__call__ = checkpointed
        on_loss, on_grads = mx.value_and_grad(lambda p: loss_for(on_model, p))(on_model.trainable_parameters())
        mx.eval(on_loss, *_tree_values(on_grads))
        finite_loss_on = _finite_array(on_loss, mx)
        finite_gradients_on = _finite_tree(on_grads, mx)
        max_abs, max_rel, keys_equal, allclose, keys, key_hash, shape_mismatches = _max_grad_diff(on_grads, off_grads, mx)
        return {
            "layer_count": layer_count,
            "pipeline_layer_count": len(on_model.model.pipeline_layers),
            "decoder_layer_count": decoder_count,
            "checkpointed_forward_calls": forward_calls,
            "checkpoint_wrapper_name": wrapper_name,
            "finite_loss_off": finite_loss_off,
            "finite_loss_on": finite_loss_on,
            "finite_gradients_off": finite_gradients_off,
            "finite_gradients_on": finite_gradients_on,
            "gradient_key_count_off": len(keys),
            "gradient_key_count_on": len(_tree_keys(on_grads)),
            "gradient_keys_sha256_off": key_hash,
            "gradient_keys_sha256_on": _hash_keys(_tree_keys(on_grads)),
            "gradient_key_sets_equal": keys_equal,
            "gradient_shapes_equal": not shape_mismatches,
            "gradient_shape_mismatches": shape_mismatches,
            "gradient_allclose_atol_rtol_1e_5": allclose,
            "max_gradient_abs_diff": max_abs,
            "max_gradient_rel_diff": max_rel,
            "monkeypatch_subprocess_only": True,
        }
    finally:
        deepseek_v4_nn.DecoderLayerNN.__call__ = original


def _expert_payload(mx: Any, E: int, H: int, I: int):
    class Expert: pass
    hp = ((H + 31) // 32) * 32; ip = ((I + 31) // 32) * 32
    ex = Expert(); ex.n_routed_experts = E; ex.hidden_size = H; ex.intermediate_size = I; ex.limit = 10.0
    ex.w1_weight = mx.full((E, I, hp // 2), 0x11, dtype=mx.uint8)
    ex.w3_weight = mx.full((E, I, hp // 2), 0x21, dtype=mx.uint8)
    ex.w2_weight = mx.full((E, H, ip // 2), 0x31, dtype=mx.uint8)
    ex.w1_scale = mx.full((E, I, hp // 32), 0.001, dtype=mx.bfloat16)
    ex.w3_scale = mx.full((E, I, hp // 32), 0.001, dtype=mx.bfloat16)
    ex.w2_scale = mx.full((E, H, ip // 32), 0.001, dtype=mx.bfloat16)
    return ex


def _rows_by_expert(T: int, E: int, K: int) -> list[list[int]]:
    rows = [[] for _ in range(E)]
    for t in range(T):
        for slot in range(K):
            eid = (t * K + slot) % E
            rows[eid].append(t)
    return rows


def _probe_c_comparison_contract() -> dict[str, Any]:
    return {
        "route_formula": "eid = (token_index * K + slot_index) mod E",
        "tokens": 64,
        "top_k_rule": "K=min(6,E)",
        "low_e_control": {"experts": 2, "top_k": 2, "assignments": 128, "comparable_to_expert_count_axis": False},
        "normalized_expert_count_axis": {"experts": [8, 32, 128, 256], "top_k": 6, "assignments": 384},
        "allowed_grouping_keys": ["case", "depth", "experts", "top_k", "tokens"],
        "forbidden_expert_count_trend_members": [2],
    }


def _success_evidence(row: dict[str, Any], keys: list[str] | None, key_hash: str | None, input_grad: Any | None, mx: Any) -> dict[str, Any]:
    if input_grad is None:
        input_shape = None
        input_nonzero = None
    else:
        mx.eval(input_grad)
        input_shape = list(input_grad.shape)
        input_nonzero = bool(mx.any(mx.abs(input_grad.astype(mx.float32)) > 0).item())
    return {
        "trainable_key_count": None if keys is None else len(keys),
        "trainable_keys_sha256": key_hash,
        "input_gradient_shape": input_shape,
        "input_gradient_nonzero": input_nonzero,
        "measured_custom_kernel_nodes": row["custom_kernel_nodes"],
    }


def _zero_param_anchor(params: Any, mx: Any) -> Any:
    total = mx.array(0.0, dtype=mx.float32)
    for value in _tree_values(params):
        total = total + mx.sum(value.astype(mx.float32) * 0.0)
    return total


def run_model_cell(cell: dict[str, Any]) -> dict[str, Any]:
    mx = _setup_mlx()
    from mlx_lm.tuner.trainer import grad_checkpoint
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs
    row = base_row(cell)
    t0 = time.perf_counter()
    try:
        if cell["case"] == "probe_d_ablation":
            derived = validate_topology_fixture(FIXTURE)
            cfg = _fixture_config(derived, T=cell["tokens"], cr=cell["compression_ratio"])
            for row_key, cfg_key in (("depth", "num_hidden_layers"), ("experts", "n_routed_experts"), ("hidden", "hidden_size"), ("intermediate", "moe_intermediate_size"), ("top_k", "num_experts_per_tok"), ("hc_mult", "hc_mult")):
                if int(cell[row_key]) != int(cfg[cfg_key]):
                    raise AssertionError(f"Probe D cell {row_key}={cell[row_key]} differs from fixture-derived {cfg_key}={cfg[cfg_key]}")
        else:
            cfg = _config(cell["depth"], T=cell["tokens"], H=cell["hidden"], I=cell["intermediate"], E=cell["experts"], K=cell["top_k"], hc=cell["hc_mult"], cr=cell["compression_ratio"])
        mx.random.seed(13 + int(cell["depth"]) + int(cell.get("compression_ratio") or 0))
        model = Model(ModelArgs.from_dict(cfg))
        _fill_model_fp4(model, mx)
        keys, key_hash, _shapes = _prepare_lora_model(model, cell["depth"])
        if cell.get("checkpoint"):
            grad_checkpoint(model.layers[0])
        ids = (mx.arange(cell["tokens"], dtype=mx.int32) % max(2, cell["tokens"])).reshape(1, cell["tokens"])
        h0 = model.model.embed_tokens(ids)
        h0 = mx.broadcast_to(mx.expand_dims(h0, -2), (*h0.shape[:-1], model.args.hc_mult, model.args.hidden_size))
        input_probe = mx.zeros_like(h0)
        _materialize_resident(mx, _tree_values(model.parameters()), ids, input_probe)
        row["active_baseline"] = mx.get_active_memory()

        mx.reset_peak_memory()
        forward = model_forward_once(model, ids, None, cell["component_mask"])
        forward_loss = mx.mean(forward * forward)
        mx.eval(forward_loss)
        row["peak_forward"] = mx.get_peak_memory()

        def loss_for(params, probe):
            model.update(params)
            y = model_forward_once(model, ids, probe, cell["component_mask"])
            return mx.mean(y * y) + _zero_param_anchor(params, mx)

        mx.reset_peak_memory()
        value, grads = mx.value_and_grad(loss_for, argnums=(0, 1))(model.trainable_parameters(), input_probe)
        lora_grads, input_grad = grads
        grad_keys = _tree_keys(lora_grads)
        if grad_keys != keys:
            raise AssertionError("LoRA gradient key set differs from trainable key set")
        row["active_after_graph"] = mx.get_active_memory()
        grad_vals = _tree_values(lora_grads)
        row["custom_kernel_nodes"] = _count_custom_kernel_nodes(mx, [value, *grad_vals, input_grad])
        mx.eval(value, *grad_vals, input_grad)
        row["peak_backward"] = mx.get_peak_memory()
        row["finite_loss"] = _finite_array(value, mx)
        row["finite_gradients"] = _finite_tree(lora_grads, mx) and _finite_array(input_grad, mx)
        if not row["finite_gradients"]:
            nonfinite_input = int(mx.sum(mx.logical_not(mx.isfinite(input_grad))).item())
            nonfinite_lora = sum(int(mx.sum(mx.logical_not(mx.isfinite(v))).item()) for v in grad_vals)
            raise FloatingPointError(f"raw non-finite gradient values: input={nonfinite_input} lora={nonfinite_lora}")
        row["active_final"] = mx.get_active_memory(); row["cache_final"] = mx.get_cache_memory()
        row["_cell_evidence"] = _success_evidence(row, keys, key_hash, input_grad, mx)
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        row["exit_code"] = 1
        row["_traceback"] = traceback.format_exc()[-2000:]
    row["duration_s"] = time.perf_counter() - t0
    return row


def _make_interaction_model(cell: dict[str, Any], mx: Any) -> tuple[Any, list[str], str, dict[str, list[int]], Any, Any]:
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs
    derived = validate_topology_fixture(FIXTURE)
    cfg = _fixture_config(derived, T=cell["tokens"], cr=cell["compression_ratio"])
    for row_key, cfg_key in (("depth", "num_hidden_layers"), ("experts", "n_routed_experts"), ("hidden", "hidden_size"), ("intermediate", "moe_intermediate_size"), ("top_k", "num_experts_per_tok"), ("hc_mult", "hc_mult")):
        expected = int(cell[row_key])
        if row_key == "depth":
            cfg[cfg_key] = expected
        elif expected != int(cfg[cfg_key]):
            raise AssertionError(f"interaction cell {row_key}={cell[row_key]} differs from fixture-derived {cfg_key}={cfg[cfg_key]}")
    cfg["mlp_layer_types"] = ["moe"] * int(cell["depth"])
    mx.random.seed(1701 + int(cell["depth"]))
    model = Model(ModelArgs.from_dict(cfg))
    _fill_model_fp4(model, mx)
    keys, key_hash, shapes = _prepare_lora_model(model, cell["depth"])
    ids = (mx.arange(cell["tokens"], dtype=mx.int32) % max(2, cell["tokens"])).reshape(1, cell["tokens"])
    h0 = model.model.embed_tokens(ids)
    h0 = mx.broadcast_to(mx.expand_dims(h0, -2), (*h0.shape[:-1], model.args.hc_mult, model.args.hidden_size))
    input_probe = mx.zeros_like(h0)
    return model, keys, key_hash, shapes, ids, input_probe


def _finalize_interaction_success(row: dict[str, Any], keys: list[str], key_hash: str, input_grad: Any, mx: Any) -> None:
    row["finite_loss"] = bool(row["finite_loss"])
    row["finite_gradients"] = bool(row["finite_gradients"])
    if not row["finite_gradients"]:
        raise FloatingPointError("raw non-finite interaction gradient values")
    row["active_final"] = mx.get_active_memory()
    row["cache_final"] = mx.get_cache_memory()
    row["_cell_evidence"] = _success_evidence(row, keys, key_hash, input_grad, mx)


def _run_interaction_one_graph(cell: dict[str, Any], row: dict[str, Any], mx: Any) -> dict[str, Any]:
    from mlx_lm.tuner.trainer import grad_checkpoint
    model, keys, key_hash, _shapes, ids, input_probe = _make_interaction_model(cell, mx)
    if cell.get("checkpoint"):
        grad_checkpoint(model.layers[0])
    _materialize_resident(mx, _tree_values(model.parameters()), ids, input_probe)
    mode = cell["case"]

    telemetry_f: dict[str, int] = {}
    mx.reset_peak_memory()
    forward = interaction_forward_once(model, ids, None, mode, telemetry_f)
    forward_loss = mx.mean(forward * forward)
    mx.eval(forward_loss)
    row["peak_forward"] = mx.get_peak_memory()
    row["pre_boundary_active"] = telemetry_f.get("pre_boundary_active")
    row["post_boundary_active"] = telemetry_f.get("post_boundary_active")

    if mode in ("attention_forward_routed_vjp", "routed_forward_attention_vjp"):
        composed = interaction_forward_once(model, ids, None, "composed_checkpoint_on", {})
        mx.eval(composed, forward)
        diff = float(mx.max(mx.abs(composed.astype(mx.float32) - forward.astype(mx.float32))).item())
        row["forward_exact"] = diff == 0.0
        row["forward_max_abs_diff"] = diff
        del composed
    del forward, forward_loss
    mx.clear_cache()

    def loss_for(params, probe):
        model.update(params)
        telemetry: dict[str, int] = {}
        y = interaction_forward_once(model, ids, probe, mode, telemetry)
        row["pre_boundary_active"] = max(row.get("pre_boundary_active") or 0, telemetry.get("pre_boundary_active", 0)) or row.get("pre_boundary_active")
        row["post_boundary_active"] = max(row.get("post_boundary_active") or 0, telemetry.get("post_boundary_active", 0)) or row.get("post_boundary_active")
        return mx.mean(y * y) + _zero_param_anchor(params, mx)

    mx.reset_peak_memory()
    row["active_baseline"] = mx.get_active_memory()
    value, grads = mx.value_and_grad(loss_for, argnums=(0, 1))(model.trainable_parameters(), input_probe)
    lora_grads, input_grad = grads
    if _tree_keys(lora_grads) != keys:
        raise AssertionError("LoRA gradient key set differs from trainable key set")
    row["active_after_graph"] = mx.get_active_memory()
    grad_vals = _tree_values(lora_grads)
    row["custom_kernel_nodes"] = _count_custom_kernel_nodes(mx, [value, *grad_vals, input_grad])
    mx.eval(value, *grad_vals, input_grad)
    row["loss"] = float(value.item())
    row["peak_backward"] = mx.get_peak_memory()
    row["finite_loss"] = _finite_array(value, mx)
    row["finite_gradients"] = _finite_tree(lora_grads, mx) and _finite_array(input_grad, mx)
    _finalize_interaction_success(row, keys, key_hash, input_grad, mx)
    return row


def _run_interaction_materialized(cell: dict[str, Any], row: dict[str, Any], mx: Any) -> dict[str, Any]:
    from mlx_lm.tuner.trainer import grad_checkpoint
    model, keys, key_hash, _shapes, ids, input_probe = _make_interaction_model(cell, mx)
    grad_checkpoint(model.layers[0])
    _materialize_resident(mx, _tree_values(model.parameters()), ids, input_probe)

    def loss_for(mode: str, params: Any, probe: Any):
        model.update(params)
        telemetry: dict[str, int] = {}
        y = interaction_forward_once(model, ids, probe, mode, telemetry)
        if mode == "materialized_attention_routed_boundary":
            row["pre_boundary_active"] = max(row.get("pre_boundary_active") or 0, telemetry.get("pre_boundary_active", 0)) or row.get("pre_boundary_active")
            row["post_boundary_active"] = max(row.get("post_boundary_active") or 0, telemetry.get("post_boundary_active", 0)) or row.get("post_boundary_active")
        return mx.mean(y * y) + _zero_param_anchor(params, mx)

    mx.reset_peak_memory()
    forward = interaction_forward_once(model, ids, None, "materialized_attention_routed_boundary", {})
    forward_loss = mx.mean(forward * forward)
    mx.eval(forward_loss)
    row["peak_forward"] = mx.get_peak_memory()
    del forward, forward_loss
    mx.clear_cache()

    mx.reset_peak_memory()
    row["active_baseline"] = mx.get_active_memory()
    value, grads = mx.value_and_grad(lambda p, x: loss_for("materialized_attention_routed_boundary", p, x), argnums=(0, 1))(model.trainable_parameters(), input_probe)
    lora_grads, input_grad = grads
    row["active_after_graph"] = mx.get_active_memory()
    grad_vals = _tree_values(lora_grads)
    row["custom_kernel_nodes"] = _count_custom_kernel_nodes(mx, [value, *grad_vals, input_grad])
    mx.eval(value, *grad_vals, input_grad)
    row["loss"] = float(value.item())
    row["peak_backward"] = mx.get_peak_memory()

    base_value, base_grads = mx.value_and_grad(lambda p, x: loss_for("composed_checkpoint_on", p, x), argnums=(0, 1))(model.trainable_parameters(), input_probe)
    base_lora_grads, base_input_grad = base_grads
    mx.eval(base_value, *_tree_values(base_lora_grads), base_input_grad)
    lora_abs, lora_rel, keys_equal, lora_allclose, _keys, _hash, shape_mismatches = _max_grad_diff(lora_grads, base_lora_grads, mx)
    input_abs, input_rel, input_allclose = _max_array_diff(input_grad, base_input_grad, mx)
    loss_abs, _loss_rel, loss_allclose = _max_array_diff(value, base_value, mx)
    row["materialized_loss_abs_diff"] = loss_abs
    row["materialized_lora_grad_max_abs_diff"] = lora_abs
    row["materialized_lora_grad_max_rel_diff"] = lora_rel
    row["materialized_input_grad_max_abs_diff"] = input_abs
    row["materialized_input_grad_max_rel_diff"] = input_rel
    row["materialized_allclose_atol_rtol_1e_5"] = bool(loss_allclose and keys_equal and not shape_mismatches and lora_allclose and input_allclose)
    if not row["materialized_allclose_atol_rtol_1e_5"]:
        raise AssertionError("materialized boundary parity failed")
    row["finite_loss"] = _finite_array(value, mx) and _finite_array(base_value, mx)
    row["finite_gradients"] = _finite_tree(lora_grads, mx) and _finite_array(input_grad, mx) and _finite_tree(base_lora_grads, mx) and _finite_array(base_input_grad, mx)
    _finalize_interaction_success(row, keys, key_hash, input_grad, mx)
    return row


def _run_interaction_sequential(cell: dict[str, Any], row: dict[str, Any], mx: Any) -> dict[str, Any]:
    from mlx_lm.tuner.trainer import grad_checkpoint
    model, keys, key_hash, _shapes, ids, input_probe = _make_interaction_model(cell, mx)
    grad_checkpoint(model.layers[0])
    h = model.model.embed_tokens(ids)
    h = mx.broadcast_to(mx.expand_dims(h, -2), (*h.shape[:-1], model.args.hc_mult, model.args.hidden_size))
    h = h + (input_probe - mx.stop_gradient(input_probe))
    _materialize_resident(mx, _tree_values(model.parameters()), ids, input_probe, h)
    peak_f = peak_b = active_graph = nodes = 0
    stage_baselines: list[int] = []
    baseline_for_peak_b: int | None = None
    finite_loss = finite_gradients = True
    last_input_grad = input_probe
    last_value = None
    for layer in model.model.pipeline_layers:
        mx.reset_peak_memory()
        out = _interaction_layer_forward(layer, h, ids, "composed", {})
        f_loss = mx.mean(out * out)
        mx.eval(f_loss)
        peak_f = max(peak_f, mx.get_peak_memory())

        def stage_loss(params, stage_input):
            model.update(params)
            y = _interaction_layer_forward(layer, stage_input, ids, "composed", {})
            return mx.mean(y * y) + _zero_param_anchor(params, mx)

        del f_loss
        mx.clear_cache()
        stage_baseline = mx.get_active_memory()
        stage_baselines.append(stage_baseline)
        mx.reset_peak_memory()
        value, grads = mx.value_and_grad(stage_loss, argnums=(0, 1))(model.trainable_parameters(), h)
        lora_grads, input_grad = grads
        if _tree_keys(lora_grads) != keys:
            raise AssertionError("LoRA gradient key set differs from trainable key set")
        active_graph = max(active_graph, mx.get_active_memory())
        grad_vals = _tree_values(lora_grads)
        nodes += _count_custom_kernel_nodes(mx, [value, *grad_vals, input_grad])
        mx.eval(value, *grad_vals, input_grad)
        stage_peak_b = mx.get_peak_memory()
        if stage_peak_b >= peak_b:
            peak_b = stage_peak_b
            baseline_for_peak_b = stage_baseline
        finite_loss = finite_loss and _finite_array(value, mx)
        finite_gradients = finite_gradients and _finite_tree(lora_grads, mx) and _finite_array(input_grad, mx)
        last_input_grad = input_grad
        last_value = value
        h = mx.stop_gradient(out)
        del value, lora_grads, input_grad, out, grad_vals
        mx.clear_cache()
    row["active_baseline"] = baseline_for_peak_b if baseline_for_peak_b is not None else (max(stage_baselines) if stage_baselines else mx.get_active_memory())
    row["active_after_graph"] = active_graph
    row["peak_forward"] = peak_f
    row["peak_backward"] = peak_b
    row["custom_kernel_nodes"] = nodes
    row["finite_loss"] = finite_loss
    row["finite_gradients"] = finite_gradients
    row["loss"] = float(last_value.item()) if last_value is not None else None
    _finalize_interaction_success(row, keys, key_hash, last_input_grad, mx)
    return row


def run_interaction_cell(cell: dict[str, Any]) -> dict[str, Any]:
    mx = _setup_mlx()
    row = base_interaction_row(cell)
    t0 = time.perf_counter()
    try:
        if cell["case"] == "materialized_attention_routed_boundary":
            row = _run_interaction_materialized(cell, row, mx)
        elif cell["case"] == "composed_layer_sequential":
            row = _run_interaction_sequential(cell, row, mx)
        else:
            row = _run_interaction_one_graph(cell, row, mx)
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        row["exit_code"] = 1
        row["_traceback"] = traceback.format_exc()[-2000:]
    row["duration_s"] = time.perf_counter() - t0
    return row


def run_routed_cell(cell: dict[str, Any]) -> dict[str, Any]:
    mx = _setup_mlx()
    from ds4_ft_mlx.routed_fp4_metal import routed_fp4
    row = base_row(cell)
    t0 = time.perf_counter()
    try:
        T, H, I, E, K, D = cell["tokens"], cell["hidden"], cell["intermediate"], cell["experts"], cell["top_k"], cell["depth"]
        ex = _expert_payload(mx, E, H, I)
        rows = _rows_by_expert(T, E, K)
        row["nonempty_experts"] = sum(bool(r) for r in rows)
        scores = mx.ones((T, E), dtype=mx.float32)
        x0 = mx.ones((T, H), dtype=mx.float32) * 0.125
        _materialize_resident(mx, scores, x0, [ex.w1_weight, ex.w2_weight, ex.w3_weight, ex.w1_scale, ex.w2_scale, ex.w3_scale])
        row["active_baseline"] = mx.get_active_memory()

        def layer(xx):
            return xx + routed_fp4(xx, scores, rows, ex, routed_scaling_factor=1.0) * 0.01

        def chain(xx):
            yy = xx
            for _ in range(D):
                yy = layer(yy)
            return yy

        if cell["case"].endswith("sequential"):
            peak_f = peak_b = active_graph = nodes = 0
            yy = x0
            finite = True
            for _ in range(D):
                mx.reset_peak_memory()
                out = layer(yy)
                loss = mx.sum(out * out)
                mx.eval(loss)
                peak_f = max(peak_f, mx.get_peak_memory())

                def layer_loss(z, route_scores):
                    routed = routed_fp4(z, route_scores, rows, ex, routed_scaling_factor=1.0)
                    y = z + routed * 0.01
                    return mx.sum(y * y)

                mx.reset_peak_memory()
                value, grads = mx.value_and_grad(layer_loss, argnums=(0, 1))(yy, scores)
                grad_x, grad_scores = grads
                active_graph = max(active_graph, mx.get_active_memory())
                nodes += _count_custom_kernel_nodes(mx, [value, grad_x, grad_scores])
                mx.eval(value, grad_x, grad_scores)
                peak_b = max(peak_b, mx.get_peak_memory())
                finite = finite and _finite_array(value, mx) and _finite_array(grad_x, mx) and _finite_array(grad_scores, mx)
                yy = out
                del value, grad_x, grad_scores, out
                mx.clear_cache()
            row["finite_loss"] = finite; row["finite_gradients"] = finite
            row["peak_forward"] = peak_f; row["peak_backward"] = peak_b; row["active_after_graph"] = active_graph
            row["custom_kernel_nodes"] = nodes
        else:
            mx.reset_peak_memory()
            y = chain(x0)
            forward_loss = mx.sum(y * y)
            mx.eval(forward_loss)
            row["peak_forward"] = mx.get_peak_memory(); row["finite_loss"] = _finite_array(forward_loss, mx)

            def loss_for(z, route_scores):
                yy = z
                for _ in range(D):
                    routed = routed_fp4(yy, route_scores, rows, ex, routed_scaling_factor=1.0)
                    yy = yy + routed * 0.01
                return mx.sum(yy * yy)

            mx.reset_peak_memory()
            value, grads = mx.value_and_grad(loss_for, argnums=(0, 1))(x0, scores)
            grad_x, grad_scores = grads
            row["active_after_graph"] = mx.get_active_memory()
            row["custom_kernel_nodes"] = _count_custom_kernel_nodes(mx, [value, grad_x, grad_scores])
            mx.eval(value, grad_x, grad_scores)
            row["peak_backward"] = mx.get_peak_memory(); row["finite_gradients"] = _finite_array(grad_x, mx) and _finite_array(grad_scores, mx)
        expected = D * row["nonempty_experts"] * 6
        if row["custom_kernel_nodes"] != expected:
            raise AssertionError(f"Probe C CustomKernel node mismatch measured={row['custom_kernel_nodes']} expected={expected}")
        row["active_final"] = mx.get_active_memory(); row["cache_final"] = mx.get_cache_memory()
        row["_cell_evidence"] = _success_evidence(row, None, None, None, mx)
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"; row["exit_code"] = 1
        row["_traceback"] = traceback.format_exc()[-2000:]
    row["duration_s"] = time.perf_counter() - t0
    return row


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    if cell["case"].startswith("probe_c"):
        return run_routed_cell(cell)
    return run_model_cell(cell)


def child_main(cell_json: str) -> int:
    cell = json.loads(cell_json)
    nonce = cell.pop("__nonce", None)
    if nonce:
        print(json.dumps({"child_nonce": nonce, "pid": os.getpid()}, sort_keys=True), flush=True)
    row = run_cell(cell)
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if row.get("error") is None else 1


def interaction_child_main(cell_json: str) -> int:
    cell = json.loads(cell_json)
    nonce = cell.pop("__nonce", None)
    if nonce:
        print(json.dumps({"child_nonce": nonce, "pid": os.getpid()}, sort_keys=True), flush=True)
    row = run_interaction_cell(cell)
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if row.get("error") is None else 1


def _empty_evidence(cell: dict[str, Any], nonce: str, start_ns: int, finish_ns: int) -> dict[str, Any]:
    return {
        "ordinal": cell["__ordinal"],
        "cell_identity_sha256": cell_identity(cell),
        "child_pid": None,
        "child_nonce": nonce,
        "started_monotonic_ns": start_ns,
        "finished_monotonic_ns": finish_ns,
        "trainable_key_count": None,
        "trainable_keys_sha256": None,
        "input_gradient_shape": None,
        "input_gradient_nonzero": None,
        "measured_custom_kernel_nodes": None,
    }


def normalize_process_row(row: dict[str, Any], cell: dict[str, Any], returncode: int | None, stderr: str = "") -> dict[str, Any]:
    if returncode is None:
        row["exit_code"] = None
        row["signal"] = "TIMEOUT"
        row["error"] = (row.get("error") or f"timeout after {TIMEOUT_S}s: {stderr}")[:500]
    elif returncode < 0:
        signo = -returncode
        names = {s.value: s.name for s in signal.Signals}
        row["exit_code"] = None
        row["signal"] = names.get(signo, f"SIG{signo}")
        row["error"] = (row.get("error") or stderr or f"child terminated by signal {signo}")[-500:]
    else:
        row["exit_code"] = int(returncode)
        row["signal"] = None
        if returncode != 0:
            row["error"] = (row.get("error") or stderr or "non-zero child exit")[-500:]
        elif row.get("error"):
            row["exit_code"] = 1
        else:
            row["error"] = None
    if row.get("error") is not None:
        row["error"] = str(row["error"])[-500:]
    return row


def _parse_child_stdout(stdout: str, nonce: str) -> tuple[int | None, dict[str, Any] | None]:
    child_pid = None
    row = None
    for line in stdout.splitlines():
        try:
            maybe = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(maybe, dict) and maybe.get("child_nonce") == nonce:
            child_pid = maybe.get("pid")
        elif isinstance(maybe, dict) and "case" in maybe:
            row = maybe
    return child_pid, row


def _popen_capture(cmd: list[str], timeout_s: int) -> tuple[int | None, int, str, str, str]:
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        stdout, stderr = proc.communicate()
        return proc.pid, None, stdout or (exc.stdout or ""), stderr or (exc.stderr or ""), "TIMEOUT"
    return proc.pid, proc.returncode, stdout, stderr, ""


def _observed_exit_signal_error(returncode: int | None, stderr: str, timeout_error: str = "") -> dict[str, Any]:
    if returncode is None:
        return {"parent_observed_exit_code": None, "parent_observed_signal": "TIMEOUT", "parent_observed_error": (timeout_error or stderr or "timeout")[-500:]}
    if returncode < 0:
        signo = -returncode
        names = {s.value: s.name for s in signal.Signals}
        return {"parent_observed_exit_code": None, "parent_observed_signal": names.get(signo, f"SIG{signo}"), "parent_observed_error": (stderr or f"child terminated by signal {signo}")[-500:]}
    return {"parent_observed_exit_code": int(returncode), "parent_observed_signal": None, "parent_observed_error": (stderr[-500:] if returncode != 0 and stderr else None)}


def run_cell_subprocess(cell: dict[str, Any], ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    nonce = uuid.uuid4().hex
    child_cell = dict(cell, __nonce=nonce)
    start_ns = time.monotonic_ns()
    cmd = [sys.executable, str(Path(__file__).resolve()), "child", json.dumps(child_cell, sort_keys=True)]
    child_pid, returncode, stdout, stderr, timeout_error = _popen_capture(cmd, TIMEOUT_S)
    finish_ns = time.monotonic_ns()
    if returncode is None:
        row = base_row(cell); row["duration_s"] = (finish_ns - start_ns) / 1e9
        row = normalize_process_row(row, cell, None, stderr or timeout_error)
        evidence = _empty_evidence({**cell, "__ordinal": ordinal}, nonce, start_ns, finish_ns)
        evidence["child_pid"] = child_pid
        return row, evidence
    echoed_pid, row = _parse_child_stdout(stdout, nonce)
    child_pid = child_pid if child_pid is not None else echoed_pid
    if row is None:
        row = base_row(cell); row["duration_s"] = (finish_ns - start_ns) / 1e9; row["error"] = (stderr or stdout or "missing child row")[-500:]
    child_extra = row.pop("_cell_evidence", {}) if isinstance(row, dict) else {}
    row.pop("_traceback", None)
    row = normalize_process_row(row, cell, returncode, stderr)
    evidence = _empty_evidence({**cell, "__ordinal": ordinal}, nonce, start_ns, finish_ns)
    evidence["child_pid"] = child_pid
    evidence.update({k: child_extra.get(k) for k in EVIDENCE_SCHEMA if k in child_extra})
    if evidence["measured_custom_kernel_nodes"] is None:
        evidence["measured_custom_kernel_nodes"] = row.get("custom_kernel_nodes")
    return {k: row.get(k) for k in SCHEMA}, {k: evidence.get(k) for k in EVIDENCE_SCHEMA}


def run_interaction_cell_subprocess(cell: dict[str, Any], ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    nonce = uuid.uuid4().hex
    child_cell = dict(cell, __nonce=nonce)
    start_ns = time.monotonic_ns()
    cmd = [sys.executable, str(Path(__file__).resolve()), "interaction-child", json.dumps(child_cell, sort_keys=True)]
    child_pid, returncode, stdout, stderr, timeout_error = _popen_capture(cmd, TIMEOUT_S)
    finish_ns = time.monotonic_ns()
    if returncode is None:
        row = base_interaction_row(cell); row["duration_s"] = (finish_ns - start_ns) / 1e9
        row = normalize_process_row(row, cell, None, stderr or timeout_error)
        evidence = _empty_evidence({**cell, "__ordinal": ordinal}, nonce, start_ns, finish_ns)
        evidence["child_pid"] = child_pid
        return row, evidence
    echoed_pid, row = _parse_child_stdout(stdout, nonce)
    child_pid = child_pid if child_pid is not None else echoed_pid
    if row is None:
        row = base_interaction_row(cell); row["duration_s"] = (finish_ns - start_ns) / 1e9; row["error"] = (stderr or stdout or "missing child row")[-500:]
    child_extra = row.pop("_cell_evidence", {}) if isinstance(row, dict) else {}
    row.pop("_traceback", None)
    row = normalize_process_row(row, cell, returncode, stderr)
    evidence = _empty_evidence({**cell, "__ordinal": ordinal}, nonce, start_ns, finish_ns)
    evidence["child_pid"] = child_pid
    evidence.update({k: child_extra.get(k) for k in EVIDENCE_SCHEMA if k in child_extra})
    if evidence["measured_custom_kernel_nodes"] is None:
        evidence["measured_custom_kernel_nodes"] = row.get("custom_kernel_nodes")
    return {k: row.get(k) for k in INTERACTION_SCHEMA}, {k: evidence.get(k) for k in EVIDENCE_SCHEMA}


def protected_hashes() -> dict[str, str]:
    files = [
        "ds4.c",
        "ds4_cli.c",
        "ds4_server.c",
        "ds4_metal.m",
        "ds4_cuda.cu",
        "ds4_distributed.c",
        "ds4_ssd.c",
        "python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py",
        "python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py",
        "python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py",
        "python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal",
    ]
    files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "metal").glob("*.metal")))
    out = {}
    for rel in files:
        p = ROOT / rel
        if p.exists():
            out[rel] = sha256_path(p)
    return out


def _assert_repo_relative_source(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if ROOT.resolve() not in p.parents and p != ROOT.resolve():
        raise ValueError(f"provenance source escapes repository root: {rel}")
    banned = ("/Volumes/", "model", "shard", "dataset", "config.json")
    lowered = rel.lower()
    if any(term in lowered for term in banned):
        raise ValueError(f"provenance source names forbidden real-asset token: {rel}")
    if not p.is_file():
        raise ValueError(f"missing provenance source: {rel}")
    return p


def validate_topology_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = ["schema_version", "story", "purpose", "real_assets_accessed", "model_defaults", "provenance", "layers"]
    if set(data) != set(required):
        raise ValueError(f"fixture top-level keys must be exactly {required}")
    if data["schema_version"] != 1 or data["story"] != "13.3b-5g" or data["real_assets_accessed"] is not False:
        raise ValueError("fixture constants mismatch")
    layers = data["layers"]
    if not isinstance(layers, list) or len(layers) != 43:
        raise ValueError("fixture must contain exactly 43 layers")
    layer_keys = {"index", "layer_type", "hidden_size", "intermediate_size", "experts", "top_k", "hc_mult"}
    common: dict[str, int] = {}
    for idx, layer in enumerate(layers):
        if set(layer) != layer_keys:
            raise ValueError(f"layer {idx} keys mismatch")
        if layer["index"] != idx or isinstance(layer["index"], bool):
            raise ValueError("layer indices must be exactly 0..42 in order")
        if layer["layer_type"] != "moe":
            raise ValueError("all fixture layers must be moe")
        for key in ("hidden_size", "intermediate_size", "experts", "top_k", "hc_mult"):
            if isinstance(layer[key], bool) or not isinstance(layer[key], int) or layer[key] <= 0:
                raise ValueError(f"layer {idx} {key} must be a positive integer")
        if layer["top_k"] > layer["experts"]:
            raise ValueError("top_k cannot exceed experts")
        if layer["hidden_size"] % 16 or layer["intermediate_size"] % 32:
            raise ValueError("fixture dimensions violate current FP4/attention constraints")
        for key in ("hidden_size", "intermediate_size", "experts", "top_k", "hc_mult"):
            common.setdefault(key, layer[key])
            if common[key] != layer[key]:
                raise ValueError(f"layer {idx} inconsistent {key}")
    defaults = data["model_defaults"]
    needed_defaults = {
        "model_type", "vocab_size", "num_hash_layers", "n_shared_experts", "expert_dtype",
        "num_attention_heads", "num_key_value_heads", "head_dim", "q_lora_rank", "o_lora_rank",
        "o_groups", "qk_rope_head_dim", "hc_sinkhorn_iters", "scoring_func", "routed_scaling_factor",
        "swiglu_limit", "rms_norm_eps",
    }
    if set(defaults) != needed_defaults:
        raise ValueError(f"model_defaults keys mismatch: {sorted(set(defaults) ^ needed_defaults)}")
    for key in ("vocab_size", "num_hash_layers", "n_shared_experts", "num_attention_heads", "num_key_value_heads", "head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "qk_rope_head_dim", "hc_sinkhorn_iters"):
        if isinstance(defaults[key], bool) or not isinstance(defaults[key], int) or (key != "num_hash_layers" and defaults[key] <= 0) or (key == "num_hash_layers" and defaults[key] < 0):
            raise ValueError(f"model_defaults {key} invalid")
    if defaults["model_type"] != "deepseek_v4_nn" or defaults["expert_dtype"] != "fp4" or defaults["scoring_func"] != "sqrtsoftplus":
        raise ValueError("model_defaults constants mismatch")
    if common["hidden_size"] % defaults["num_attention_heads"] != 0:
        raise ValueError("attention heads must divide hidden size")
    provenance = data["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"durable_authority", "direct_shape_matrix_evidence", "derivation_map"}:
        raise ValueError("fixture provenance shape mismatch")
    observed_hashes: dict[str, str] = {}
    for group in ("durable_authority", "direct_shape_matrix_evidence"):
        entries = provenance[group]
        if not isinstance(entries, list):
            raise ValueError("provenance entries must be lists")
        for entry in entries:
            if set(entry) != {"path", "sha256"}:
                raise ValueError("provenance entry keys mismatch")
            rel = entry["path"]
            p = _assert_repo_relative_source(rel)
            actual = sha256_path(p)
            if entry["sha256"] != actual or entry["sha256"] != entry["sha256"].lower():
                raise ValueError(f"provenance hash mismatch for {rel}")
            observed_hashes[rel] = actual
    if sorted(observed_hashes) != sorted(PROVENANCE_SOURCES):
        raise ValueError("provenance sources mismatch")
    return {
        "fixture_path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "fixture_sha256": sha256_path(path),
        "schema_version": data["schema_version"],
        "layer_count": len(layers),
        "model_defaults": defaults,
        "num_hidden_layers": len(layers),
        "mlp_layer_types": [layer["layer_type"] for layer in layers],
        "hidden_size": common["hidden_size"],
        "moe_intermediate_size": common["intermediate_size"],
        "n_routed_experts": common["experts"],
        "num_experts_per_tok": common["top_k"],
        "hc_mult": common["hc_mult"],
        "provenance": provenance,
        "source_hashes": observed_hashes,
    }


def run_probe_a_subprocess() -> dict[str, Any]:
    nonce = uuid.uuid4().hex
    start_ns = time.monotonic_ns()
    cmd = [sys.executable, str(Path(__file__).resolve()), "probe-a-child", nonce]
    child_pid, returncode, stdout, stderr, timeout_error = _popen_capture(cmd, TIMEOUT_S)
    finish_ns = time.monotonic_ns()
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except Exception:
        payload = {"error": (stderr or stdout)[-1000:]}
    payload["fresh_child_execution"] = {
        "child_pid": child_pid,
        "child_nonce": nonce,
        "started_monotonic_ns": start_ns,
        "finished_monotonic_ns": finish_ns,
        **_observed_exit_signal_error(returncode, stderr, timeout_error),
    }
    return payload


def run_interaction_report() -> dict[str, Any]:
    derived = validate_topology_fixture(FIXTURE)
    rows: list[dict[str, Any]] = []
    cell_evidence: list[dict[str, Any]] = []
    cells = interaction_plan_cells()
    for ordinal, cell in enumerate(cells):
        row, evidence = run_interaction_cell_subprocess(cell, ordinal)
        rows.append(row)
        cell_evidence.append(evidence)
    return {
        "metadata": {
            "story": "13.3b-5h",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "memory_limit": MEMORY_LIMIT,
            "timeout_s": TIMEOUT_S,
            "serial": True,
            "execution_mode": "serial_fresh_subprocess_per_cell",
            "max_concurrency": 1,
            "planned_cell_count": len(cells),
            "executed_cell_count": len(rows),
            "planned_cell_identity_sha256": [interaction_cell_identity(c) for c in cells],
            "planned_cell_list_sha256": interaction_plan_identity_hash(cells),
            "executed_cell_list_sha256": sha256_text("\n".join(interaction_cell_identity(r) for r in rows)),
            "parent_pid": os.getpid(),
            "real_assets_accessed": False,
            "classification": "not-classified-by-coder",
            "classification_thresholds": dict(INTERACTION_THRESHOLDS),
            "schema": INTERACTION_SCHEMA,
            "cell_evidence_schema": EVIDENCE_SCHEMA,
            "custom_kernel_nodes_scope": "differentiated graph: value plus every evaluated gradient leaf",
            "custom_kernel_nodes_method": "mx.export_to_dot; count DOT node lines containing CustomKernel",
            "active_after_graph_boundary": "active MLX memory immediately after constructing the unevaluated differentiated graph and immediately before DOT export/evaluation",
            "materialized_boundary_parity": "loss, every LoRA gradient leaf, and input gradient checked with true elementwise atol=rtol=1e-5 against matched composed no_shared row semantics; key sets and shapes must match exactly",
            "active_baseline_boundary": "active MLX memory captured after any diagnostic forward/comparison graph is evaluated and released, immediately before differentiated graph construction",
            "sequential_aggregation": {"active_baseline": "per_stage_active_baseline matching the stage that produced max(per_stage_backward_peak)", "active_after_graph": "max(per_stage_active_after_graph)", "peak_forward": "max(per_stage_forward_peak)", "peak_backward": "max(per_stage_backward_peak)", "custom_kernel_nodes": "sum(per_stage_measured_nodes)"},
            "fixture": {k: derived[k] for k in ("fixture_path", "fixture_sha256", "schema_version", "layer_count", "provenance", "source_hashes")},
            "fixture_config_derivation": {k: derived[k] for k in ("num_hidden_layers", "mlp_layer_types", "hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "hc_mult", "model_defaults")},
            "protected_hashes": protected_hashes(),
        },
        "cell_evidence": cell_evidence,
        "rows": rows,
    }


def run_report() -> dict[str, Any]:
    derived = validate_topology_fixture(FIXTURE)
    probe_a_data = run_probe_a_subprocess()
    rows: list[dict[str, Any]] = []
    cell_evidence: list[dict[str, Any]] = []
    cells = plan_cells()
    for ordinal, cell in enumerate(cells):
        row, evidence = run_cell_subprocess(cell, ordinal)
        rows.append(row)
        cell_evidence.append(evidence)
    return {
        "metadata": {
            "story": "13.3b-5g",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "memory_limit": MEMORY_LIMIT,
            "timeout_s": TIMEOUT_S,
            "serial": True,
            "execution_mode": "serial_fresh_subprocess_per_cell",
            "max_concurrency": 1,
            "planned_cell_count": len(cells),
            "executed_cell_count": len(rows),
            "planned_cell_identity_sha256": [cell_identity(c) for c in cells],
            "planned_cell_list_sha256": plan_identity_hash(cells),
            "executed_cell_list_sha256": sha256_text("\n".join(cell_identity(r) for r in rows)),
            "parent_pid": os.getpid(),
            "real_assets_accessed": False,
            "classification": "not-authorized-by-diagnostic",
            "schema": SCHEMA,
            "cell_evidence_schema": EVIDENCE_SCHEMA,
            "custom_kernel_nodes_scope": "differentiated graph: value plus every evaluated gradient leaf",
            "custom_kernel_nodes_method": "mx.export_to_dot; count DOT node lines containing CustomKernel",
            "active_after_graph_boundary": "active MLX memory immediately after constructing the unevaluated differentiated graph and immediately before DOT export/evaluation",
            "probe_c_sequential_aggregation": {"active_after_graph": "max(per_stage_active_after_graph)", "peak_forward": "max(per_stage_forward_peak)", "peak_backward": "max(per_stage_backward_peak)", "custom_kernel_nodes": "sum(per_stage_measured_nodes)"},
            "probe_c_comparison_contract": _probe_c_comparison_contract(),
            "fixture": {k: derived[k] for k in ("fixture_path", "fixture_sha256", "schema_version", "layer_count", "provenance", "source_hashes")},
            "fixture_config_derivation": {k: derived[k] for k in ("num_hidden_layers", "mlp_layer_types", "hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "hc_mult", "model_defaults")},
            "protected_hashes": protected_hashes(),
        },
        "probe_a": probe_a_data,
        "cell_evidence": cell_evidence,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "run", "child", "probe-a-child", "validate-fixture", "interaction-plan", "interaction-run", "interaction-child"])
    ap.add_argument("cell", nargs="?")
    ap.add_argument("--output", default=None)
    ns = ap.parse_args()
    if ns.command == "plan":
        print(json.dumps(plan(), sort_keys=True)); return 0
    if ns.command == "interaction-plan":
        print(json.dumps(interaction_plan(), sort_keys=True)); return 0
    if ns.command == "child":
        return child_main(ns.cell or "{}")
    if ns.command == "interaction-child":
        return interaction_child_main(ns.cell or "{}")
    if ns.command == "probe-a-child":
        data = run_probe_a()
        if ns.cell:
            data["child_nonce"] = ns.cell
            data["pid"] = os.getpid()
        print(json.dumps(data, sort_keys=True)); return 0
    if ns.command == "validate-fixture":
        path = Path(ns.cell) if ns.cell else FIXTURE
        print(json.dumps(validate_topology_fixture(path), sort_keys=True)); return 0
    if ns.command == "interaction-run":
        report = run_interaction_report()
        out = Path(ns.output or INTERACTION_REPORT); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"output": str(out), "rows": len(report["rows"])}, sort_keys=True)); return 0
    report = run_report()
    out = Path(ns.output or REPORT); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "rows": len(report["rows"]), "probe_a": report["probe_a"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
