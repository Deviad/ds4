"""Story 13.3b-3 AC4: AttentionNN guard lift + mixed compressed layers.

The nn module owns submodule/leaf wiring only.  The math oracle remains the
functional `_attention_mlx` dispatch, whose CSA/HCA branches are separately
parity-graded against torch real-dim fixtures.  This test proves that mixed
sliding/CSA/HCA AttentionNN layers choose the right per-layer compression ratio,
construct the compressor/indexer leaves, and pass the exact synthesized weights
through the dispatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")


def _mixed_attention_config() -> dict[str, object]:
    return dict(
        model_type="deepseek_v4_nn",
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=3,
        num_hash_layers=0,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        compress_ratios=[0, 4, 128],
        index_topk=16,
        mlp_layer_types=["moe", "moe", "moe"],
        hc_mult=1,
        hc_sinkhorn_iters=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=16,
        expert_dtype="fp4",
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        qk_rope_head_dim=4,
        index_head_dim=8,
        index_n_heads=4,
        compression_ratio=0,
        compress_rope_theta=160000.0,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        sliding_window=128,
        rms_norm_eps=1e-6,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _matrix(rows: int, cols: int, scale: float) -> mx.array:
    vals = mx.arange(rows * cols, dtype=mx.float32).reshape((rows, cols))
    return ((vals % 23) - 11.0) * scale


def _vector(size: int, start: float, stop: float) -> mx.array:
    if size == 1:
        return mx.array([start], dtype=mx.float32)
    return mx.linspace(start, stop, size, dtype=mx.float32)


def _hidden(seq_len: int, hidden_size: int) -> mx.array:
    vals = mx.arange(seq_len * hidden_size, dtype=mx.float32).reshape((1, seq_len, hidden_size))
    return ((vals % 29) - 14.0) * 0.003


def _weights_for(args, ratio: int) -> dict[str, mx.array]:
    heads_per_group = args.num_attention_heads // args.o_groups
    weights = {
        "q_a_proj.weight": _matrix(args.q_lora_rank, args.hidden_size, 0.002),
        "q_norm.weight": _vector(args.q_lora_rank, 0.85, 1.15),
        "q_b_proj.weight": _matrix(args.num_attention_heads * args.head_dim, args.q_lora_rank, 0.0015),
        "kv_proj.weight": _matrix(args.head_dim, args.hidden_size, 0.0025),
        "kv_norm.weight": _vector(args.head_dim, 0.9, 1.1),
        "o_a_proj.weight": _matrix(args.o_groups * args.o_lora_rank, heads_per_group * args.head_dim, 0.002),
        "o_b_proj.weight": _matrix(args.hidden_size, args.o_groups * args.o_lora_rank, 0.002),
        "sinks": _vector(args.num_attention_heads, -0.2, 0.2),
    }
    if ratio == 4:
        weights.update(
            {
                "compressor_wkv": _matrix(2 * args.head_dim, args.hidden_size, 0.002),
                "compressor_wgate": _matrix(2 * args.head_dim, args.hidden_size, 0.0017),
                "compressor_ape": _matrix(4, 2 * args.head_dim, 0.001),
                "compressor_norm": _vector(args.head_dim, 0.85, 1.15),
                "indexer_compressor_wkv": _matrix(2 * args.index_head_dim, args.hidden_size, 0.002),
                "indexer_compressor_wgate": _matrix(2 * args.index_head_dim, args.hidden_size, 0.0017),
                "indexer_compressor_ape": _matrix(4, 2 * args.index_head_dim, 0.001),
                "indexer_compressor_norm": _vector(args.index_head_dim, 0.9, 1.1),
                "indexer_wq_b": _matrix(args.index_n_heads * args.index_head_dim, args.q_lora_rank, 0.0015),
                "indexer_proj": _matrix(args.index_n_heads, args.hidden_size, 0.002),
            }
        )
    elif ratio == 128:
        weights.update(
            {
                "compressor_wkv": _matrix(args.head_dim, args.hidden_size, 0.002),
                "compressor_wgate": _matrix(args.head_dim, args.hidden_size, 0.0017),
                "compressor_ape": _matrix(128, args.head_dim, 0.001),
                "compressor_norm": _vector(args.head_dim, 0.85, 1.15),
            }
        )
    return weights


def _set_linear(linear, weight: mx.array) -> None:
    linear.weight = weight


def _load_attention_weights(attn, weights: dict[str, mx.array], ratio: int) -> None:
    _set_linear(attn.q_a_proj, weights["q_a_proj.weight"])
    attn.q_norm.weight = weights["q_norm.weight"]
    _set_linear(attn.q_b_proj, weights["q_b_proj.weight"])
    _set_linear(attn.kv_proj, weights["kv_proj.weight"])
    attn.kv_norm.weight = weights["kv_norm.weight"]
    _set_linear(attn.o_a_proj, weights["o_a_proj.weight"])
    _set_linear(attn.o_b_proj, weights["o_b_proj.weight"])
    attn.sinks = weights["sinks"]
    if ratio != 0:
        _set_linear(attn.compressor.wkv, weights["compressor_wkv"])
        _set_linear(attn.compressor.wgate, weights["compressor_wgate"])
        attn.compressor.ape = weights["compressor_ape"]
        attn.compressor.norm.weight = weights["compressor_norm"]
    if ratio == 4:
        _set_linear(attn.indexer.compressor.wkv, weights["indexer_compressor_wkv"])
        _set_linear(attn.indexer.compressor.wgate, weights["indexer_compressor_wgate"])
        attn.indexer.compressor.ape = weights["indexer_compressor_ape"]
        attn.indexer.compressor.norm.weight = weights["indexer_compressor_norm"]
        _set_linear(attn.indexer.weights_proj, weights["indexer_proj"])
        _set_linear(attn.indexer.wq_b, weights["indexer_wq_b"])


def test_model_args_normalizes_stale_all_sliding_layer_types_from_compress_ratios():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs

    cfg = _mixed_attention_config()
    cfg["layer_types"] = ["sliding_attention", "sliding_attention", "sliding_attention"]
    args = ModelArgs.from_dict(cfg)

    assert args.layer_types == ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
    assert args.compress_ratios == [0, 4, 128]


def test_attention_nn_mixed_sliding_csa_hca_layers_match_functional_dispatch():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import AttentionNN, ModelArgs

    args = ModelArgs.from_dict(_mixed_attention_config())
    expected_ratios = [0, 4, 128]
    seq_len = 129
    x = _hidden(seq_len, args.hidden_size)

    for layer_idx, ratio in enumerate(expected_ratios):
        attn = AttentionNN(args, layer_idx)
        assert attn.config.compression_ratio == ratio
        assert hasattr(attn, "compressor") is (ratio != 0)
        assert hasattr(attn, "indexer") is (ratio == 4)

        weights = _weights_for(attn.config, ratio)
        _load_attention_weights(attn, weights, ratio)

        got = attn(x)
        direct = _attention_mlx(
            attn.config,
            x,
            weights,
            index_topk=(args.index_topk if ratio == 4 else None),
        )
        mx.eval(got, direct)

        assert tuple(got.shape) == (1, seq_len, args.hidden_size)
        assert bool(mx.all(mx.isfinite(got)).item())
        max_abs = mx.max(mx.abs(got - direct)).item()
        assert max_abs == pytest.approx(0.0, abs=0.0)
