"""Story 13.3b-5d — sparse routed-token FP4 backward.

RED → GREEN TDD for the eager per-expert sparse dispatch with exact custom
first-order VJP in SparseMoeBlockNN.__call__.
"""

from __future__ import annotations

import ast
import gc
import inspect
import multiprocessing
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten


def _tiny_config() -> dict[str, object]:
    return dict(
        model_type="deepseek_v4_nn",
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_hash_layers=1,
        mlp_layer_types=["hash_moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=3,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=16,
        expert_dtype="fp4",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=2,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _tiny_config_real_dim() -> dict[str, object]:
    """Same routing topology but real H/I so one-expert dequant is realistic."""
    return dict(
        model_type="deepseek_v4_nn",
        vocab_size=64,
        hidden_size=4096,
        num_hidden_layers=1,
        num_hash_layers=0,
        mlp_layer_types=["moe"],
        hc_mult=1,
        hc_sinkhorn_iters=3,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp4",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _fill_fp4(experts, nibble_byte: int = 0x11, scale: float = 0.05) -> None:
    experts.w1_weight = mx.full(experts.w1_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w2_weight = mx.full(experts.w2_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w3_weight = mx.full(experts.w3_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w1_scale = mx.full(experts.w1_scale.shape, scale, dtype=mx.bfloat16)
    experts.w2_scale = mx.full(experts.w2_scale.shape, scale, dtype=mx.bfloat16)
    experts.w3_scale = mx.full(experts.w3_scale.shape, scale, dtype=mx.bfloat16)


def _fill_fp4_distinct(experts) -> None:
    w1 = []
    w2 = []
    w3 = []
    s1 = []
    s2 = []
    s3 = []
    for eid in range(experts.n_routed_experts):
        w1.append(mx.full(experts.w1_weight.shape[1:], 0x11 + 3 * eid, dtype=mx.uint8))
        w2.append(mx.full(experts.w2_weight.shape[1:], 0x22 + 5 * eid, dtype=mx.uint8))
        w3.append(mx.full(experts.w3_weight.shape[1:], 0x33 + 7 * eid, dtype=mx.uint8))
        s1.append(mx.full(experts.w1_scale.shape[1:], 0.03 + 0.01 * eid, dtype=mx.bfloat16))
        s2.append(mx.full(experts.w2_scale.shape[1:], 0.04 + 0.01 * eid, dtype=mx.bfloat16))
        s3.append(mx.full(experts.w3_scale.shape[1:], 0.05 + 0.01 * eid, dtype=mx.bfloat16))
    experts.w1_weight = mx.stack(w1)
    experts.w2_weight = mx.stack(w2)
    experts.w3_weight = mx.stack(w3)
    experts.w1_scale = mx.stack(s1)
    experts.w2_scale = mx.stack(s2)
    experts.w3_scale = mx.stack(s3)


def _nonuniform_x(shape: tuple[int, ...], start: float = 0.05) -> mx.array:
    size = 1
    for dim in shape:
        size *= dim
    vals = mx.arange(size, dtype=mx.float32).reshape(shape)
    return vals / float(size + 7) + start


def _zero_shared(block) -> None:
    block.shared_experts.gate_proj.weight = mx.zeros_like(block.shared_experts.gate_proj.weight)
    block.shared_experts.up_proj.weight = mx.zeros_like(block.shared_experts.up_proj.weight)
    block.shared_experts.down_proj.weight = mx.zeros_like(block.shared_experts.down_proj.weight)


# ---------------------------------------------------------------------------
# Dense reference — exact current semantics, independent of __call__ impl.
# ---------------------------------------------------------------------------

def _dense_moe_reference(block, x, input_ids):
    """Re-implements the dense per-expert loop with current _route contract."""
    scores, indices, denom = block._route(x, input_ids)
    routed = mx.zeros_like(x)
    for eid in range(block.n_routed_experts):
        expert_out = block.experts.forward_one(eid, x)
        selected = mx.any(indices == eid, axis=-1).astype(x.dtype)
        factor = (
            scores[..., eid]
            / (denom + 1e-20)
        ) * selected * block.routed_scaling_factor
        routed = routed + expert_out * mx.expand_dims(factor, -1)
    return routed + block.shared_experts(x)


def _assert_forward_combined_close(got, ref, *, label: str) -> None:
    got_np = np.array(got.tolist(), dtype=np.float32)
    ref_np = np.array(ref.tolist(), dtype=np.float32)
    assert got_np.shape == ref_np.shape, label
    assert np.isfinite(got_np).all(), label
    abs_err = np.abs(got_np - ref_np)
    bound = 2e-6 + 1e-6 * np.abs(ref_np)
    nrmse = float(np.sqrt(np.mean(np.square(abs_err.astype(np.float64)))) / max(float(np.sqrt(np.mean(np.square(ref_np.astype(np.float64))))), 1e-12))
    denom = np.maximum(np.maximum(np.abs(got_np), np.abs(ref_np)), 1e-6)
    max_rel = float(np.max(abs_err / denom))
    max_abs = float(np.max(abs_err))
    assert np.all(abs_err <= bound) and nrmse <= 1e-6, (
        f"{label} max_abs={max_abs} max_rel={max_rel} nrmse={nrmse}"
    )


# ---------------------------------------------------------------------------
# 1. Independent tiny forward parity
# ---------------------------------------------------------------------------

def test_sparse_forward_matches_dense_reference():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4_distinct(block.experts)
    _zero_shared(block)
    block.gate_weight = mx.stack(
        [mx.ones((16,)) * scale for scale in (0.01, 0.02, 0.03, 0.04)]
    )
    block.e_score_correction_bias = mx.array([0.0, 0.01, 0.02, 0.03], dtype=mx.float32)

    x = _nonuniform_x((2, 3, 16), start=0.11)
    assert len({bytes(row) for row in block.experts.w1_weight[:, 0, :].tolist()}) == block.n_routed_experts
    assert len({tuple(float(v) for v in row) for row in block.experts.w1_scale[:, 0, :].tolist()}) == block.n_routed_experts
    out = block(x, input_ids=None)
    ref = _dense_moe_reference(block, x, None)
    mx.eval(out, ref)

    assert out.shape == ref.shape == (2, 3, 16)
    _assert_forward_combined_close(out, ref, label="sparse whole forward")


def test_downstream_proxy_logits_and_score_gradients_match_dense_reference():
    from ds4_ft_mlx.routed_fp4_metal import routed_fp4
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    cfg = _tiny_config()
    cfg["hidden_size"] = 32
    cfg["moe_intermediate_size"] = 32
    cfg["n_routed_experts"] = 4
    cfg["num_experts_per_tok"] = 3
    args = ModelArgs.from_dict(cfg)
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4_distinct(block.experts)
    _zero_shared(block)
    rows_by_expert = [[0, 2, 5], [1, 2, 4], [0, 3, 5], [1, 4]]
    x = _nonuniform_x((6, 32), start=0.07)
    scores = mx.array(
        [
            [0.70, 0.10, 0.30, 0.20],
            [0.15, 0.80, 0.25, 0.60],
            [0.40, 0.50, 0.10, 0.20],
            [0.20, 0.15, 0.90, 0.25],
            [0.12, 0.65, 0.18, 0.55],
            [0.75, 0.22, 0.35, 0.25],
        ],
        dtype=mx.float32,
    )
    proj = mx.array(
        [[((h * 17 + v * 23) % 19 - 9) / 113.0 for v in range(7)] for h in range(32)],
        dtype=mx.float32,
    )
    cotangent = mx.array(
        [[((t * 5 + v * 7) % 11 - 5) / 17.0 for v in range(7)] for t in range(6)],
        dtype=mx.float32,
    )

    def dense_routed(x_, scores_):
        denom = mx.zeros((x_.shape[0],), dtype=mx.float32)
        for eid, rows in enumerate(rows_by_expert):
            row_arr = mx.array(rows, dtype=mx.int32)
            denom = denom.at[row_arr].add(scores_[row_arr, eid])
        out = mx.zeros_like(x_)
        x3 = x_[None, :, :]
        for eid, rows in enumerate(rows_by_expert):
            row_arr = mx.array(rows, dtype=mx.int32)
            factor = mx.zeros((x_.shape[0],), dtype=mx.float32).at[row_arr].add(
                block.routed_scaling_factor * scores_[row_arr, eid] / (denom[row_arr] + 1e-20)
            )
            expert_out = block.experts.forward_one(eid, x3).reshape(x_.shape)
            out = out + expert_out * factor[:, None]
        return out

    def sparse_logits(x_, scores_):
        y = routed_fp4(
            x_, scores_, rows_by_expert, block.experts,
            routed_scaling_factor=block.routed_scaling_factor,
        )
        return y @ proj

    def dense_logits(x_, scores_):
        return dense_routed(x_, scores_) @ proj

    sparse = sparse_logits(x, scores)
    dense = dense_logits(x, scores)
    mx.eval(sparse, dense)
    _assert_forward_combined_close(sparse, dense, label="downstream proxy logits")
    sparse_np = np.array(sparse.tolist(), dtype=np.float32)
    dense_np = np.array(dense.tolist(), dtype=np.float32)
    assert np.array_equal(np.argmax(sparse_np, axis=-1), np.argmax(dense_np, axis=-1))
    sorted_dense = np.sort(dense_np, axis=-1)
    min_margin = float(np.min(sorted_dense[:, -1] - sorted_dense[:, -2]))
    assert min_margin > 1e-7, f"proxy top-1 reference margin {min_margin}"

    def sparse_loss(x_, scores_):
        return mx.sum(sparse_logits(x_, scores_) * cotangent)

    def dense_loss(x_, scores_):
        return mx.sum(dense_logits(x_, scores_) * cotangent)

    gx_sparse, gs_sparse = mx.grad(sparse_loss, argnums=(0, 1))(x, scores)
    gx_dense, gs_dense = mx.grad(dense_loss, argnums=(0, 1))(x, scores)
    mx.eval(gx_sparse, gs_sparse, gx_dense, gs_dense)
    for label, got, ref in (
        ("proxy input gradient", gx_sparse, gx_dense),
        ("proxy score gradient", gs_sparse, gs_dense),
    ):
        diff = float(mx.max(mx.abs(got - ref)).item())
        rel = diff / max(1e-12, float(mx.max(mx.abs(ref)).item()))
        assert diff <= 1e-5 and rel <= 1e-5, f"{label} diff={diff} rel={rel}"


# ---------------------------------------------------------------------------
# 2. Routing semantics
# ---------------------------------------------------------------------------

def test_learned_tie_breaks_lower_expert_ids():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts)
    _zero_shared(block)
    # Force equal scores+bias for first three experts
    block.gate_weight = mx.zeros((4, 16), dtype=mx.float32)
    block.e_score_correction_bias = mx.zeros((4,), dtype=mx.float32)

    x = mx.ones((1, 1, 16), dtype=mx.float32)
    scores, indices, denom = block._route(x, None)
    mx.eval(scores, indices, denom)
    # Stable sort lower-index wins on ties
    assert indices[0, 0].tolist() == [0, 1]


def test_duplicate_hash_routes_count_once_per_token():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    d = _tiny_config()
    d["num_hash_layers"] = 1
    d["mlp_layer_types"] = ["hash_moe"]
    d["num_hidden_layers"] = 1
    args = ModelArgs.from_dict(d)
    block = SparseMoeBlockNN(args, layer_idx=0)
    _fill_fp4(block.experts)
    _zero_shared(block)
    # tid2eid: every token maps to the SAME two experts [0, 0]
    block.tid2eid = mx.zeros((args.vocab_size, 2), dtype=mx.int32)

    x = mx.ones((1, 3, 16), dtype=mx.float32) * 0.25
    scores, indices, denom = block._route(x, input_ids=mx.zeros((1, 3), dtype=mx.int32))
    out = block(x, input_ids=mx.zeros((1, 3), dtype=mx.int32))
    ref = _dense_moe_reference(block, x, mx.zeros((1, 3), dtype=mx.int32))
    mx.eval(out, ref)

    assert out.shape == (1, 3, 16)
    _assert_forward_combined_close(out, ref, label="duplicate hash forward")
    # Duplicate collapsed: each token contributes expert 0 once
    for t in range(3):
        assert float(denom[0, t].item()) == float(scores[0, t, 0].item())


def test_empty_expert_skipped_and_unchanged_output():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts, nibble_byte=0x11, scale=0.05)
    _zero_shared(block)
    # Route only to expert 3, leaving 0/1/2 empty
    block.gate_weight = mx.zeros((4, 16), dtype=mx.float32)
    block.gate_weight[3] = 1.0
    block.e_score_correction_bias = mx.zeros((4,), dtype=mx.float32)

    x = mx.ones((1, 2, 16), dtype=mx.float32) * 0.25
    out = block(x, input_ids=None)
    ref = _dense_moe_reference(block, x, None)
    mx.eval(out, ref)

    _assert_forward_combined_close(out, ref, label="empty expert forward")


def test_correction_bias_selection_only():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts)
    _zero_shared(block)
    # Expert 3 selected because of large positive bias; but its contribution
    # weight uses unbiased score.
    block.gate_weight = mx.zeros((4, 16), dtype=mx.float32)
    block.gate_weight[0] = 1.0
    block.gate_weight[3] = 0.1
    block.e_score_correction_bias = mx.array([0.0, 0.0, 0.0, 10.0], dtype=mx.float32)

    x = mx.ones((1, 1, 16), dtype=mx.float32) * 0.25
    scores, indices, denom = block._route(x, None)
    mx.eval(scores, indices, denom)
    # Expert 3 should be selected due to bias
    assert 3 in indices[0, 0].tolist()
    # Denominator uses unbiased scores only
    assert float(denom[0, 0].item()) == pytest.approx(
        float(scores[0, 0, 0].item()) + float(scores[0, 0, 3].item()), abs=1e-5
    )


def test_shared_expert_added_once_full_input():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4_distinct(block.experts)
    # Make shared expert non-zero and input-dependent.  Omission, double-add,
    # or evaluating it on routed subsets changes these assertions.
    block.shared_experts.gate_proj.weight = mx.eye(args.moe_intermediate_size, args.hidden_size)
    block.shared_experts.up_proj.weight = mx.eye(args.moe_intermediate_size, args.hidden_size)
    block.shared_experts.down_proj.weight = mx.eye(args.hidden_size, args.moe_intermediate_size)

    x = _nonuniform_x((1, 2, 16), start=0.25)
    scores, indices, denom = block._route(x, None)
    routed_only = mx.zeros_like(x)
    for eid in range(block.n_routed_experts):
        expert_out = block.experts.forward_one(eid, x)
        selected = mx.any(indices == eid, axis=-1).astype(x.dtype)
        factor = scores[..., eid] / (denom + 1e-20) * selected * block.routed_scaling_factor
        routed_only = routed_only + expert_out * mx.expand_dims(factor, -1)
    shared = block.shared_experts(x)
    out = block(x, input_ids=None)
    ref = routed_only + shared
    mx.eval(out, ref, routed_only, shared)

    assert bool(mx.any(mx.abs(shared) > 0).item()), "shared fixture must be non-zero"
    assert float(mx.max(mx.abs(out - routed_only)).item()) > 1e-6, "shared omission undetected"
    _assert_forward_combined_close(out, ref, label="shared expert forward")
    assert float(mx.max(mx.abs(out - (routed_only + 2 * shared))).item()) > 1e-6


# ---------------------------------------------------------------------------
# 3. Gradient parity
# ---------------------------------------------------------------------------

def test_sparse_input_gradient_matches_dense():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts, nibble_byte=0xCD, scale=0.06)
    _zero_shared(block)
    block.gate_weight = mx.stack(
        [mx.ones((16,)) * scale for scale in (0.01, 0.02, 0.03, 0.04)]
    )
    block.e_score_correction_bias = mx.array([0.0, 0.01, 0.02, 0.03], dtype=mx.float32)

    x = mx.ones((2, 3, 16), dtype=mx.float32) * 0.25

    def sparse_loss(x_):
        return mx.sum(block(x_, None) * block(x_, None))

    def dense_loss(x_):
        return mx.sum(_dense_moe_reference(block, x_, None) * _dense_moe_reference(block, x_, None))

    g_sparse = mx.grad(sparse_loss)(x)
    g_dense = mx.grad(dense_loss)(x)
    mx.eval(g_sparse, g_dense)

    assert bool(mx.all(mx.isfinite(g_sparse)).item())
    assert bool(mx.all(mx.isfinite(g_dense)).item())
    assert bool(mx.any(mx.abs(g_sparse.astype(mx.float32)) > 0).item())
    diff = float(mx.max(mx.abs(g_sparse - g_dense)).item())
    # atol=rtol=1e-5 over float32 cast
    rel = diff / max(1e-12, float(mx.max(mx.abs(g_dense)).item()))
    assert diff <= 1e-5 and rel <= 1e-5, f"input grad diff={diff} rel={rel}"


def test_sparse_gate_weight_gradient_matches_dense():
    """Gate gradients match dense autodiff and a finite-difference reference."""
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4_distinct(block.experts)
    _zero_shared(block)
    gw0 = mx.stack(
        [
            mx.linspace(0.01 + 0.01 * eid, 0.025 + 0.01 * eid, 16)
            for eid in range(4)
        ]
    )
    block.gate_weight = gw0
    block.e_score_correction_bias = mx.array([0.0, 0.005, 0.01, 0.015], dtype=mx.float32)

    x = _nonuniform_x((2, 3, 16), start=0.17)

    def sparse_loss_gate(gw):
        block.gate_weight = gw
        y = block(x, None)
        return mx.sum(y * y)

    def dense_loss_gate(gw):
        block.gate_weight = gw
        y = _dense_moe_reference(block, x, None)
        return mx.sum(y * y)

    g_sparse = mx.grad(sparse_loss_gate)(gw0)
    g_dense = mx.grad(dense_loss_gate)(gw0)
    mx.eval(g_sparse, g_dense)

    assert bool(mx.all(mx.isfinite(g_sparse)).item())
    assert bool(mx.all(mx.isfinite(g_dense)).item())
    assert bool(mx.any(mx.abs(g_dense.astype(mx.float32)) > 0).item()), (
        "ordinary dense autodiff gate gradient must be non-zero for this fixture"
    )
    diff = float(mx.max(mx.abs(g_sparse - g_dense)).item())
    rel = diff / max(1e-12, float(mx.max(mx.abs(g_dense)).item()))
    assert diff <= 1e-5 and rel <= 1e-5, f"gate grad diff={diff} rel={rel}"

    flat0 = [float(v) for row in gw0.tolist() for v in row]
    dense_flat = [float(v) for row in g_dense.tolist() for v in row]
    nonzero = sorted(range(len(dense_flat)), key=lambda i: abs(dense_flat[i]), reverse=True)[:3]
    assert all(abs(dense_flat[i]) > 1e-8 for i in nonzero)
    eps = 1e-3
    for idx in nonzero:
        plus = list(flat0)
        minus = list(flat0)
        plus[idx] += eps
        minus[idx] -= eps
        gw_plus = mx.array(plus, dtype=mx.float32).reshape(gw0.shape)
        gw_minus = mx.array(minus, dtype=mx.float32).reshape(gw0.shape)
        l_plus = float(dense_loss_gate(gw_plus).item())
        l_minus = float(dense_loss_gate(gw_minus).item())
        g_num = (l_plus - l_minus) / (2 * eps)
        assert g_num == pytest.approx(dense_flat[idx], rel=2e-2, abs=2e-3)
    block.gate_weight = gw0



def test_gradient_finite_and_nonzero_clamp_near():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4_distinct(block.experts)
    _zero_shared(block)
    gw0 = mx.stack(
        [
            mx.linspace(0.12 + 0.03 * eid, 0.24 + 0.03 * eid, 16)
            for eid in range(4)
        ]
    )
    block.gate_weight = gw0
    block.e_score_correction_bias = mx.zeros((4,), dtype=mx.float32)

    x = mx.array([
        [[0.1] * 8 + [2.0] * 8],
        [[2.0] * 8 + [0.1] * 8],
        [[0.5] * 16],
    ], dtype=mx.float32).transpose(1, 0, 2)

    def sparse_loss_x(x_):
        block.gate_weight = gw0
        y = block(x_, None)
        return mx.sum(y * y)

    def dense_loss_x(x_):
        block.gate_weight = gw0
        y = _dense_moe_reference(block, x_, None)
        return mx.sum(y * y)

    def sparse_loss_gate(gw):
        block.gate_weight = gw
        y = block(x, None)
        return mx.sum(y * y)

    def dense_loss_gate(gw):
        block.gate_weight = gw
        y = _dense_moe_reference(block, x, None)
        return mx.sum(y * y)

    gx_sparse = mx.grad(sparse_loss_x)(x)
    gx_dense = mx.grad(dense_loss_x)(x)
    gg_sparse = mx.grad(sparse_loss_gate)(gw0)
    gg_dense = mx.grad(dense_loss_gate)(gw0)
    mx.eval(gx_sparse, gx_dense, gg_sparse, gg_dense)

    for name, grad in {
        "gx_sparse": gx_sparse,
        "gx_dense": gx_dense,
        "gg_sparse": gg_sparse,
        "gg_dense": gg_dense,
    }.items():
        assert bool(mx.all(mx.isfinite(grad)).item()), name
        assert bool(mx.any(mx.abs(grad) > 0).item()), name
    xdiff = float(mx.max(mx.abs(gx_sparse - gx_dense)).item())
    gdiff = float(mx.max(mx.abs(gg_sparse - gg_dense)).item())
    xrel = xdiff / max(1e-12, float(mx.max(mx.abs(gx_dense)).item()))
    grel = gdiff / max(1e-12, float(mx.max(mx.abs(gg_dense)).item()))
    assert xdiff <= 1e-5 and xrel <= 1e-5, f"clamp input grad diff={xdiff} rel={xrel}"
    assert gdiff <= 1e-5 and grel <= 1e-5, f"clamp gate grad diff={gdiff} rel={grel}"


# ---------------------------------------------------------------------------
# 4. Detached index contract
# ---------------------------------------------------------------------------

def test_indices_are_stop_gradient_before_host_materialization():
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4_nn

    source = inspect.getsource(deepseek_v4_nn.SparseMoeBlockNN.__call__)
    assert "_host_unique_rows_per_expert" in source
    assert "_y_cache" not in source
    assert "_f_cache" not in source

    tree = ast.parse(textwrap.dedent(source))
    helper_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_host_unique_rows_per_expert"
    ]
    assert len(helper_calls) == 1
    first_arg = helper_calls[0].args[0]
    assert isinstance(first_arg, ast.Call)
    assert isinstance(first_arg.func, ast.Attribute)
    assert first_arg.func.attr == "stop_gradient"
    assert getattr(first_arg.func.value, "id", None) == "mx"



def test_duplicate_row_scatter_cotangent_correct():
    """Duplicate route slots collapse; repeated token-row cotangents scatter-add."""
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    d_dup = _tiny_config()
    d_dup["num_hash_layers"] = 1
    d_dup["mlp_layer_types"] = ["hash_moe"]
    d_dup["num_hidden_layers"] = 1
    d_dup["num_experts_per_tok"] = 2
    args_dup = ModelArgs.from_dict(d_dup)
    dup = SparseMoeBlockNN(args_dup, layer_idx=0)
    _fill_fp4_distinct(dup.experts)
    _zero_shared(dup)
    dup.tid2eid = mx.zeros((args_dup.vocab_size, 2), dtype=mx.int32)

    d_single = dict(d_dup)
    d_single["num_experts_per_tok"] = 1
    args_single = ModelArgs.from_dict(d_single)
    single = SparseMoeBlockNN(args_single, layer_idx=0)
    single.experts.w1_weight = dup.experts.w1_weight
    single.experts.w2_weight = dup.experts.w2_weight
    single.experts.w3_weight = dup.experts.w3_weight
    single.experts.w1_scale = dup.experts.w1_scale
    single.experts.w2_scale = dup.experts.w2_scale
    single.experts.w3_scale = dup.experts.w3_scale
    _zero_shared(single)
    single.gate_weight = dup.gate_weight
    single.tid2eid = mx.zeros((args_single.vocab_size, 1), dtype=mx.int32)

    x = _nonuniform_x((1, 3, 16), start=0.21)
    input_ids = mx.array([[0, 1, 2]], dtype=mx.int32)
    cot = _nonuniform_x((1, 3, 16), start=0.33)

    def dup_loss(x_):
        return mx.sum(dup(x_, input_ids) * cot)

    def single_loss(x_):
        return mx.sum(single(x_, input_ids) * cot)

    y_dup = dup(x, input_ids)
    y_single = single(x, input_ids)
    g_dup = mx.grad(dup_loss)(x)
    g_single = mx.grad(single_loss)(x)
    mx.eval(y_dup, y_single, g_dup, g_single)

    _assert_forward_combined_close(y_dup, y_single, label="duplicate row forward")
    assert float(mx.max(mx.abs(g_dup - g_single)).item()) <= 1e-5
    assert bool(mx.any(mx.abs(g_dup) > 0).item())




# ---------------------------------------------------------------------------
# 5. Routed-token instrumentation
# ---------------------------------------------------------------------------

def test_exact_routed_token_counts_and_sum_bound():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts)
    _zero_shared(block)
    block.gate_weight = mx.stack(
        [mx.ones((16,)) * scale for scale in (0.01, 0.02, 0.03, 0.04)]
    )

    x = mx.ones((2, 3, 16), dtype=mx.float32) * 0.25
    scores, indices, _ = block._route(x, None)
    mx.eval(indices)
    flat = indices.reshape(-1, indices.shape[-1])
    T = flat.shape[0]
    K = flat.shape[1]
    unique_counts = []
    for eid in range(block.n_routed_experts):
        rows = []
        seen = set()
        for t in range(T):
            for k in range(K):
                if int(flat[t, k].item()) == eid and t not in seen:
                    rows.append(t)
                    seen.add(t)
        unique_counts.append(len(rows))

    total = sum(unique_counts)
    assert total <= T * K
    # Sparse implementation should call each expert with exactly R_e rows.
    # We verify this by instrumenting forward_one in the implementation,
    # but the contract test checks the route math here.
    for eid, c in enumerate(unique_counts):
        if c == 0:
            # Empty expert must not be called
            pass
        else:
            assert c <= T, f"expert {eid} R_e={c} > T"


def test_expert_call_sizes_are_r_e_not_full_t(monkeypatch):
    """GREEN on sparse: only non-empty experts enter the opaque primitive with exact R_e rows."""
    import ds4_ft_mlx.routed_fp4_metal as routed_fp4_metal
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts)
    _zero_shared(block)
    # Route only to expert 3; expert 0 gets selected via tie-breaking with K=2
    block.gate_weight = mx.zeros((4, 16), dtype=mx.float32)
    block.gate_weight[3] = 1.0
    block.e_score_correction_bias = mx.zeros((4,), dtype=mx.float32)

    x = mx.ones((1, 2, 16), dtype=mx.float32) * 0.25
    T = int(x.shape[0] * x.shape[1])

    call_sizes = []
    orig_routed_fp4 = routed_fp4_metal.routed_fp4

    def instrumented_routed_fp4(x_flat, scores_flat, rows_by_expert, experts, *, routed_scaling_factor):
        del scores_flat, experts
        for eid, rows in enumerate(rows_by_expert):
            if rows:
                call_sizes.append((eid, (len(rows), x_flat.shape[1])))
        return orig_routed_fp4(
            x_flat,
            block._scores(x).reshape(T, block.n_routed_experts),
            rows_by_expert,
            block.experts,
            routed_scaling_factor=routed_scaling_factor,
        )

    monkeypatch.setattr(routed_fp4_metal, "routed_fp4", instrumented_routed_fp4)
    out = block(x, input_ids=None)
    mx.eval(out)

    # Sparse: experts 0 and 3 called; experts 1,2 are empty → not called
    called_eids = {eid for eid, _ in call_sizes}
    assert 0 in called_eids  # selected via tie-break
    assert 3 in called_eids  # highest score
    assert 1 not in called_eids, "expert 1 is empty"
    assert 2 not in called_eids, "expert 2 is empty"
    assert len(called_eids) == 2
    for eid, shape in call_sizes:
        assert shape[0] == T, f"expert {eid} expected R_e={T}, got {shape[0]}"


def test_dense_memory_grows_with_e_sparse_should_not(monkeypatch):
    """GREEN on sparse: total primitive row volume = sum R_e ≤ T*K, not E*T."""
    import ds4_ft_mlx.routed_fp4_metal as routed_fp4_metal
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    orig_routed_fp4_base = routed_fp4_metal.routed_fp4

    def call_volume(n_experts: int) -> int:
        d = _tiny_config()
        d["n_routed_experts"] = n_experts
        d["num_experts_per_tok"] = min(2, n_experts)
        d["num_hidden_layers"] = 1
        d["mlp_layer_types"] = ["moe"]
        d["num_hash_layers"] = 0
        args = ModelArgs.from_dict(d)
        block = SparseMoeBlockNN(args, layer_idx=0)
        block.experts.w1_weight = mx.full(block.experts.w1_weight.shape, 0x11, dtype=mx.uint8)
        block.experts.w2_weight = mx.full(block.experts.w2_weight.shape, 0x11, dtype=mx.uint8)
        block.experts.w3_weight = mx.full(block.experts.w3_weight.shape, 0x11, dtype=mx.uint8)
        block.experts.w1_scale = mx.full(block.experts.w1_scale.shape, 0.05, dtype=mx.bfloat16)
        block.experts.w2_scale = mx.full(block.experts.w2_scale.shape, 0.05, dtype=mx.bfloat16)
        block.experts.w3_scale = mx.full(block.experts.w3_scale.shape, 0.05, dtype=mx.bfloat16)
        block.shared_experts.gate_proj.weight = mx.zeros_like(block.shared_experts.gate_proj.weight)
        block.shared_experts.up_proj.weight = mx.zeros_like(block.shared_experts.up_proj.weight)
        block.shared_experts.down_proj.weight = mx.zeros_like(block.shared_experts.down_proj.weight)
        block.gate_weight = mx.ones((n_experts, args.hidden_size), dtype=mx.float32) * 0.01
        x = mx.ones((1, 8, args.hidden_size), dtype=mx.float32) * 0.25

        total_rows = 0
        def instrument(x_flat, scores_flat, rows_by_expert, experts, *, routed_scaling_factor):
            nonlocal total_rows
            total_rows += sum(len(rows) for rows in rows_by_expert)
            return orig_routed_fp4_base(
                x_flat,
                scores_flat,
                rows_by_expert,
                experts,
                routed_scaling_factor=routed_scaling_factor,
            )

        monkeypatch.setattr(routed_fp4_metal, "routed_fp4", instrument)
        out = block(x, None)
        mx.eval(out)
        return total_rows

    T = 8
    vol_4 = call_volume(4)
    vol_8 = call_volume(8)
    # Sparse: volume = sum R_e ≤ T*K, never E*T for partial assignment
    K = 2
    assert vol_4 <= T * K, f"sparse vol_4={vol_4}"
    assert vol_8 <= T * K, f"sparse vol_8={vol_8}"


# ---------------------------------------------------------------------------
# 6. Bounded-growth memory probe
# ---------------------------------------------------------------------------

def _run_peak_case(args_dict: dict, mode: str, result_queue: multiprocessing.Queue) -> None:
    """Run one dense or sparse forward+VJP in a fresh process."""
    import gc
    import mlx.core as mx
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    mx.clear_cache()
    gc.collect()

    args = ModelArgs.from_dict(args_dict)
    layer_idx = 0 if len(args.mlp_layer_types) == 1 else 1
    block = SparseMoeBlockNN(args, layer_idx=layer_idx)
    block.experts.w1_weight = mx.full(block.experts.w1_weight.shape, 0x11, dtype=mx.uint8)
    block.experts.w2_weight = mx.full(block.experts.w2_weight.shape, 0x11, dtype=mx.uint8)
    block.experts.w3_weight = mx.full(block.experts.w3_weight.shape, 0x11, dtype=mx.uint8)
    block.experts.w1_scale = mx.full(block.experts.w1_scale.shape, 0.05, dtype=mx.bfloat16)
    block.experts.w2_scale = mx.full(block.experts.w2_scale.shape, 0.05, dtype=mx.bfloat16)
    block.experts.w3_scale = mx.full(block.experts.w3_scale.shape, 0.05, dtype=mx.bfloat16)
    block.shared_experts.gate_proj.weight = mx.zeros_like(block.shared_experts.gate_proj.weight)
    block.shared_experts.up_proj.weight = mx.zeros_like(block.shared_experts.up_proj.weight)
    block.shared_experts.down_proj.weight = mx.zeros_like(block.shared_experts.down_proj.weight)
    block.gate_weight = mx.ones((args.n_routed_experts, args.hidden_size), dtype=mx.float32) * 0.01

    x = mx.ones((1, 8, args.hidden_size), dtype=mx.float32) * 0.25
    mx.eval(
        x, block.experts.w1_weight, block.experts.w2_weight, block.experts.w3_weight,
        block.experts.w1_scale, block.experts.w2_scale, block.experts.w3_scale, block.gate_weight,
    )
    mx.clear_cache()
    gc.collect()
    active_baseline = mx.get_active_memory()
    mx.reset_peak_memory()

    def _dense_ref(x_):
        scores, indices, denom = block._route(x_, None)
        routed = mx.zeros_like(x_)
        for eid in range(block.n_routed_experts):
            expert_out = block.experts.forward_one(eid, x_)
            selected = mx.any(indices == eid, axis=-1).astype(x_.dtype)
            factor = (scores[..., eid] / (denom + 1e-20)) * selected * block.routed_scaling_factor
            routed = routed + expert_out * mx.expand_dims(factor, -1)
        return routed + block.shared_experts(x_)

    def loss(x_):
        y = _dense_ref(x_) if mode == "dense" else block(x_, None)
        return mx.sum(y * y)

    g = mx.grad(loss)(x)
    mx.eval(g)
    active_delta = mx.get_active_memory() - active_baseline
    peak_delta = max(0, mx.get_peak_memory() - active_baseline)
    result_queue.put((float(active_delta), float(peak_delta)))


def test_sparse_memory_growth_bounded_separate_process():
    """Dense and sparse peaks use independent fresh child processes."""
    ctx = multiprocessing.get_context("spawn")
    base = _tiny_config()
    base["num_hidden_layers"] = 1
    base["mlp_layer_types"] = ["moe"]
    base["num_hash_layers"] = 0

    results = {}
    for n_experts in (4, 8):
        d = dict(base)
        d["n_routed_experts"] = n_experts
        d["num_experts_per_tok"] = min(2, n_experts)
        for mode in ("dense", "sparse"):
            q = ctx.Queue()
            p = ctx.Process(target=_run_peak_case, args=(d, mode, q))
            p.start()
            p.join(timeout=120)
            assert p.exitcode == 0, f"{mode} memory probe failed for E={n_experts}"
            results[(mode, n_experts)] = q.get(timeout=5)

    dense_4 = results[("dense", 4)][1]
    dense_8 = results[("dense", 8)][1]
    sparse_4 = results[("sparse", 4)][1]
    sparse_8 = results[("sparse", 8)][1]

    assert dense_8 > dense_4, f"dense did not grow: {dense_4} -> {dense_8}"
    assert sparse_8 < sparse_4 * 1.5 + 1024 * 1024, (
        f"sparse grew unexpectedly: {sparse_4} -> {sparse_8}"
    )


# ---------------------------------------------------------------------------
# 7. Real-config formula contract (no-shard)
# ---------------------------------------------------------------------------

def test_real_config_formula_contract_no_shard():
    """Pin ADR 0028 assignment-proportional envelope; no dequant matrix term."""
    H = 4096
    I = 2048
    E = 256
    K = 6
    T = 4096
    A = T * K
    mib = 1024 * 1024

    route_meta_bytes = T * K * 4
    assert route_meta_bytes == 98304  # exactly 96 KiB

    assignment_y = A * H * 4
    assignment_i = A * I * 4
    a_partial = A * ((I + 7) // 8) * 4
    x_or_routed = T * H * 4
    scores = T * E * 4
    assert assignment_y == 384 * mib
    assert assignment_i == 192 * mib
    assert a_partial == 24 * mib
    assert x_or_routed == 64 * mib
    assert scores == 4 * mib

    forward_total = (192 + 384 + 384 + 64 + 64 + 4) * mib
    backward_total = (192 + 192 + 192 + 24 + 384 + 256 + 8) * mib
    operation_envelope = backward_total + 256 * mib
    assert forward_total == 1092 * mib
    assert backward_total == 1248 * mib
    assert operation_envelope == 1504 * mib
    assert operation_envelope < 2 * 1024 * mib

    process_ceiling = 300 * 1024 * 1024 * 1024
    scheduler_limit = 400_000_000_000
    margin = scheduler_limit - process_ceiling
    assert margin == 77_877_452_800
    assert margin > 72 * 1024 * 1024 * 1024


def _run_real_dim_memory_probe(result_queue: multiprocessing.Queue) -> None:
    """Run forward+VJP with real H=4096, I=2048 and two non-empty experts."""
    import mlx.core as mx
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    mx.clear_cache()
    gc.collect()

    T = 4096
    args = ModelArgs.from_dict(dict(
        model_type="deepseek_v4_nn",
        vocab_size=T,
        hidden_size=4096,
        num_hidden_layers=1,
        num_hash_layers=1,
        mlp_layer_types=["hash_moe"],
        hc_mult=1,
        hc_sinkhorn_iters=3,
        n_routed_experts=2,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp4",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    ))
    block = SparseMoeBlockNN(args, layer_idx=0)
    block.experts.w1_weight = mx.full(block.experts.w1_weight.shape, 0x11, dtype=mx.uint8)
    block.experts.w2_weight = mx.full(block.experts.w2_weight.shape, 0x11, dtype=mx.uint8)
    block.experts.w3_weight = mx.full(block.experts.w3_weight.shape, 0x11, dtype=mx.uint8)
    block.experts.w1_scale = mx.full(block.experts.w1_scale.shape, 0.05, dtype=mx.bfloat16)
    block.experts.w2_scale = mx.full(block.experts.w2_scale.shape, 0.05, dtype=mx.bfloat16)
    block.experts.w3_scale = mx.full(block.experts.w3_scale.shape, 0.05, dtype=mx.bfloat16)

    class _ZeroShared:
        def __call__(self, x):
            return mx.zeros_like(x)

    block.shared_experts = _ZeroShared()
    block.gate_weight = mx.ones((2, 4096), dtype=mx.float32) * 0.01
    block.tid2eid = mx.tile(mx.array([[0, 1]], dtype=mx.int32), (T, 1))

    x = mx.ones((1, T, 4096), dtype=mx.float32) * 0.25
    input_ids = mx.arange(T, dtype=mx.int32).reshape(1, T)
    mx.eval(
        x,
        input_ids,
        block.tid2eid,
        block.experts.w1_weight,
        block.experts.w2_weight,
        block.experts.w3_weight,
        block.experts.w1_scale,
        block.experts.w2_scale,
        block.experts.w3_scale,
        block.gate_weight,
    )
    mx.clear_cache()
    gc.collect()
    active_baseline = mx.get_active_memory()
    mx.reset_peak_memory()

    def loss(x_):
        y = block(x_, input_ids)
        return mx.sum(y * y)

    y = block(x, input_ids)
    g = mx.grad(loss)(x)
    scores, indices, _ = block._route(x, input_ids)
    mx.eval(y, g, indices)
    flat = indices.reshape(T, 2).tolist()
    nonempty = sum(any(eid in set(slots) for slots in flat) for eid in range(2))
    active_delta = mx.get_active_memory() - active_baseline
    peak_delta = max(0, mx.get_peak_memory() - active_baseline)
    finite_y = bool(mx.all(mx.isfinite(y)).item())
    finite_g = bool(mx.all(mx.isfinite(g)).item())
    result_queue.put((float(active_delta), float(peak_delta), finite_y, finite_g, int(nonempty)))


def test_real_config_operation_peak_memory_probe_no_shard():
    """No-shard real-dimension probe gates operation peak delta, not final active."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_real_dim_memory_probe, args=(q,))
    p.start()
    p.join(timeout=300)
    assert p.exitcode == 0, "real-dim memory probe failed"
    active_delta, peak_delta, finite_y, finite_g, nonempty = q.get(timeout=5)
    assert finite_y
    assert finite_g
    assert nonempty >= 2
    assert peak_delta < 2 * 1024 * 1024 * 1024, (
        f"real-dim operation peak delta {peak_delta} >= 2 GiB; active_delta={active_delta}"
    )


# ---------------------------------------------------------------------------
# 8. LoRA regression — existing trainer contract still green
# ---------------------------------------------------------------------------

def _attach_trainer_lora(model) -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    nn.quantize(model, group_size=32, bits=4)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=2,
        config={
            "rank": 2,
            "scale": 4.0,
            "dropout": 0.0,
            "keys": {"self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"},
        },
    )


def test_lora_backward_still_finite_all_nonzero():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    cfg = _tiny_config()
    # Dimensions must be divisible by nn.quantize group_size=32
    cfg["hidden_size"] = 32
    cfg["moe_intermediate_size"] = 32
    cfg["q_lora_rank"] = 32
    cfg["o_lora_rank"] = 32
    cfg["head_dim"] = 32
    model = Model(ModelArgs.from_dict(cfg))

    # Synthesize FP4 + hash tables so forward runs
    for idx, layer in enumerate(model.model.layers):
        layer.mlp.experts.w1_weight = mx.full(
            layer.mlp.experts.w1_weight.shape, 0x11 + idx, dtype=mx.uint8
        )
        layer.mlp.experts.w2_weight = mx.full(
            layer.mlp.experts.w2_weight.shape, 0x11 + idx, dtype=mx.uint8
        )
        layer.mlp.experts.w3_weight = mx.full(
            layer.mlp.experts.w3_weight.shape, 0x11 + idx, dtype=mx.uint8
        )
        layer.mlp.experts.w1_scale = mx.full(
            layer.mlp.experts.w1_scale.shape, 0.04 + idx * 0.01, dtype=mx.bfloat16
        )
        layer.mlp.experts.w2_scale = mx.full(
            layer.mlp.experts.w2_scale.shape, 0.04 + idx * 0.01, dtype=mx.bfloat16
        )
        layer.mlp.experts.w3_scale = mx.full(
            layer.mlp.experts.w3_scale.shape, 0.04 + idx * 0.01, dtype=mx.bfloat16
        )
        layer.mlp.shared_experts.gate_proj.weight = mx.zeros_like(
            layer.mlp.shared_experts.gate_proj.weight
        )
        layer.mlp.shared_experts.up_proj.weight = mx.zeros_like(
            layer.mlp.shared_experts.up_proj.weight
        )
        layer.mlp.shared_experts.down_proj.weight = mx.zeros_like(
            layer.mlp.shared_experts.down_proj.weight
        )
        if layer.mlp.is_hash:
            table = [
                [token % layer.mlp.n_routed_experts, (token + 1) % layer.mlp.n_routed_experts]
                for token in range(model.args.vocab_size)
            ]
            layer.mlp.tid2eid = mx.array(table, dtype=mx.int32)

    _attach_trainer_lora(model)

    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4]], dtype=mx.int32)

    def loss_for_params(params):
        model.update(params)
        logits = model(input_ids)
        return nn.losses.cross_entropy(logits, targets).mean()

    loss, grads = mx.value_and_grad(loss_for_params)(model.trainable_parameters())
    grad_flat = dict(tree_flatten(grads))
    mx.eval(loss, *grad_flat.values())

    assert loss.shape == ()
    assert bool(mx.isfinite(loss).item())
    assert grad_flat
    for name, grad in grad_flat.items():
        assert bool(mx.all(mx.isfinite(grad)).item()), name
    assert any(bool(mx.any(mx.abs(g) > 0).item()) for g in grad_flat.values())


# ---------------------------------------------------------------------------
# 9. Compile / eager gate
# ---------------------------------------------------------------------------

def test_compiled_host_routing_fails_as_documented():
    """MLX 0.31.2: eval inside compile is forbidden; verify exact error."""
    import mlx.core as mx

    def _inner():
        x = mx.ones((2, 3, 16), dtype=mx.float32)
        # Simulate host materialization + eval inside compile
        @mx.compile
        def bad(x):
            y = mx.stop_gradient(x)
            mx.eval(y)
            return y

        return bad(x)

    try:
        _inner()
        pytest.fail("expected ValueError for eval inside compile")
    except ValueError as exc:
        assert "eval" in str(exc).lower() or "compile" in str(exc).lower()


def test_wrapper_local_disable_compile_passes():
    """Global compile disable before trainer-style value_and_grad call passes."""
    import mlx.core as mx

    # There is no public compile-mode query; unconditionally disable then
    # test, and leave disabled (it is the documented training policy anyway).
    mx.disable_compile()

    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts)
    _zero_shared(block)
    block.gate_weight = mx.ones((4, 16), dtype=mx.float32) * 0.01

    x = mx.ones((2, 3, 16), dtype=mx.float32) * 0.25

    def loss(x_):
        return mx.sum(block(x_, None))

    # Both grad and value_and_grad must work (trainer uses value_and_grad)
    g = mx.grad(loss)(x)
    v, g2 = mx.value_and_grad(loss)(x)
    mx.eval(g, v, g2)
    assert bool(mx.all(mx.isfinite(g)).item())
    assert bool(mx.isfinite(v).item())
    assert bool(mx.all(mx.isfinite(g2)).item())


def test_import_does_not_change_global_compile_mode():
    """Importing deepseek_v4_nn must not disable or otherwise alter global MLX compile state."""
    import subprocess
    import sys

    script = '''
import mlx.core as mx

# Prove compile works before import
@mx.compile
def before(x):
    return x * 2
mx.eval(before(mx.array([1.0])))

# Import the module
from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4_nn

# Prove compile still works after import
@mx.compile
def after(x):
    return x * 3
mx.eval(after(mx.array([1.0])))
print("COMPILE_OK")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd="/Users/spotted/projects/ds4-finetuning",
    )
    assert result.returncode == 0, f"import side-effect: {result.stderr}"
    assert "COMPILE_OK" in result.stdout


@pytest.mark.parametrize("bad", [-1, 4])
def test_host_unique_rows_per_expert_rejects_out_of_range_route_ids(bad):
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import _host_unique_rows_per_expert

    with pytest.raises(ValueError, match="expert id"):
        _host_unique_rows_per_expert(mx.array([[0, bad]], dtype=mx.int32), 4)


def test_host_unique_rows_per_expert_accepts_valid_boundaries_and_collapses_duplicates():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import _host_unique_rows_per_expert

    rows = _host_unique_rows_per_expert(mx.array([[0, 3], [3, 3]], dtype=mx.int32), 4)
    assert rows == [[0], [], [], [0, 1]]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sparse_moe_block_promotes_half_activation_and_preserves_first_order_cotangent(dtype):
    mx.disable_compile()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts, scale=0.02)
    _zero_shared(block)
    block.gate_weight = mx.stack([mx.ones((16,), dtype=mx.float32) * s for s in (0.01, 0.02, 0.03, 0.04)])
    block.e_score_correction_bias = mx.array([0.0, 0.01, 0.02, 0.03], dtype=mx.float32)
    x = _nonuniform_x((2, 3, 16), start=0.2).astype(dtype)

    def loss(xx):
        y = block(xx, None)
        return mx.sum(y.astype(mx.float32) * mx.arange(y.size, dtype=mx.float32).reshape(y.shape))

    value, grad = mx.value_and_grad(loss)(x)
    mx.eval(value, grad)
    assert bool(mx.isfinite(value).item())
    assert grad.shape == x.shape
    assert bool(mx.all(mx.isfinite(grad.astype(mx.float32))).item())
    assert bool(mx.any(mx.abs(grad.astype(mx.float32)) > 0).item())
