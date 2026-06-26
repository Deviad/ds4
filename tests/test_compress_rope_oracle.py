"""AC0 RoPE oracle for Story 13.3b-1.

Pins compress-layer RoPE to the TRAILING qk_rope_head_dim=64 slice, YaRN
factor=16/original_max_position_embeddings=65536/theta=160000, and block-start
positions arange(n_win) * rate. Torch DeepseekV4RotaryEmbedding is the oracle;
MLX only supplies the additive table helper plus the frozen tail-apply primitive.
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
torch = pytest.importorskip("torch")


def real_hca_torch_config():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        model_type="deepseek_v4",
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=1024,
        o_lora_rank=1024,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=64,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp4",
        compress_rope_theta=160000,
        rope_theta=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        compress_ratios=[128],
        layer_types=["heavily_compressed_attention"],
        mlp_layer_types=["moe"],
        rms_norm_eps=1e-6,
        sliding_window=128,
        o_groups=8,
        hc_mult=4,
        attention_dropout=0.0,
    )


def real_hca_mlx_args():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

    return ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 1,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "index_n_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 1,
            "moe_intermediate_size": 2048,
            "expert_dtype": "fp4",
            "compress_rope_theta": 160000.0,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            },
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "layer_types": ["heavily_compressed_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "sliding_window": 128,
            "o_groups": 8,
            "compression_ratio": 128,
        }
    )


def _torch_compress_cos_sin(n_win: int, rate: int = 128):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding

    config = real_hca_torch_config()
    rotary = DeepseekV4RotaryEmbedding(config)
    positions = (torch.arange(n_win, dtype=torch.long) * rate).unsqueeze(0)
    x = torch.zeros((1, n_win, config.head_dim), dtype=torch.float32)
    return rotary(x, position_ids=positions, layer_type="compress")


def test_compress_rope_yarn_tail_tables_match_torch_oracle_and_rotate_trailing_64():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
        _apply_rope_tail_mlx,
        _compress_rope_yarn_tail_tables_mlx,
    )

    args = real_hca_mlx_args()
    n_win = 4
    rate = args.compression_ratio
    positions = mx.arange(n_win) * rate

    mlx_cos, mlx_sin = _compress_rope_yarn_tail_tables_mlx(
        args.head_dim,
        args.qk_rope_head_dim,
        args.compress_rope_theta,
        16,
        65536,
        n_win,
        positions=positions,
    )
    torch_cos, torch_sin = _torch_compress_cos_sin(n_win, rate)
    mlx_cos_t = torch.tensor(mlx_cos.tolist(), dtype=torch.float32).unsqueeze(0)
    mlx_sin_t = torch.tensor(mlx_sin.tolist(), dtype=torch.float32).unsqueeze(0)

    assert tuple(torch_cos.shape) == (1, n_win, 32)
    assert tuple(mlx_cos_t.shape) == (1, n_win, 32)
    torch.testing.assert_close(mlx_cos_t, torch_cos.float(), atol=1e-3, rtol=0.0)
    torch.testing.assert_close(mlx_sin_t, torch_sin.float(), atol=1e-3, rtol=0.0)

    base = (torch.arange(n_win * args.head_dim, dtype=torch.float32).reshape(1, n_win, args.head_dim) / 997.0) - 1.0
    torch_x = base.to(torch.bfloat16)
    torch_rotated = apply_rotary_pos_emb(torch_x.unsqueeze(1), torch_cos.to(torch.bfloat16), torch_sin.to(torch.bfloat16)).squeeze(1)
    mlx_rotated = _apply_rope_tail_mlx(
        mx.array(base.tolist(), dtype=mx.bfloat16),
        mlx_cos.astype(mx.bfloat16),
        mlx_sin.astype(mx.bfloat16),
        qk_rope_head_dim=args.qk_rope_head_dim,
    )
    mlx_rotated_t = torch.tensor(mlx_rotated.astype(mx.float32).tolist(), dtype=torch.float32)

    torch.testing.assert_close(
        mlx_rotated_t[..., : args.head_dim - args.qk_rope_head_dim],
        torch_x.float()[..., : args.head_dim - args.qk_rope_head_dim],
        atol=1e-2,
        rtol=0.0,
    )
    torch.testing.assert_close(mlx_rotated_t, torch_rotated.float(), atol=1e-2, rtol=0.0)
