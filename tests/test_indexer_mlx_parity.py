"""AC2 Indexer MLX parity for Story 13.3b-2a.

Compares the additive MLX `_indexer_mlx` against torch DeepseekV4Indexer.forward
at real indexer dimensions. This pins Ca/Cb overlap pooling, compress-YaRN RoPE
on the trailing qk_rope_head_dim=64 of index_head_dim=128, future masking, and
`-1` sentinel emission. Top-k tie ordering is set-based by contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")


def _real_header_shape(key: str):
    from safetensors import safe_open

    root = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
    if not root.exists():
        pytest.skip(f"real hf-f8shim checkpoint absent: {root}")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = root / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return tuple(handle.get_slice(key).get_shape())


def real_csa_torch_config():
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
        index_topk=512,
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
        compress_ratios=[4],
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        rms_norm_eps=1e-6,
        sliding_window=128,
        o_groups=8,
        hc_mult=4,
        attention_dropout=0.0,
    )


def real_csa_mlx_args():
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
            "layer_types": ["compressed_sparse_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "sliding_window": 128,
            "o_groups": 8,
            "compression_ratio": 4,
        }
    )


def _synth_indexer_fixture(seq_len: int = 257):
    g = torch.Generator(device="cpu").manual_seed(133202)
    hidden = torch.randn((1, seq_len, 4096), generator=g, dtype=torch.float32) * 0.02
    q_residual = torch.randn((1, seq_len, 1024), generator=g, dtype=torch.float32) * 0.025
    weights = {
        "indexer_compressor_wkv": torch.randn((256, 4096), generator=g, dtype=torch.float32) * 0.012,
        "indexer_compressor_wgate": torch.randn((256, 4096), generator=g, dtype=torch.float32) * 0.010,
        "indexer_compressor_ape": torch.randn((4, 256), generator=g, dtype=torch.float32) * 0.01,
        "indexer_compressor_norm": torch.linspace(0.85, 1.15, 128, dtype=torch.float32),
        "indexer_wq_b": torch.randn((64 * 128, 1024), generator=g, dtype=torch.float32) * 0.006,
        "indexer_proj": torch.randn((64, 4096), generator=g, dtype=torch.float32) * 0.008,
    }
    return hidden, q_residual, weights


def _load_torch_indexer(config, weights):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer

    module = DeepseekV4Indexer(config).eval()
    with torch.no_grad():
        module.kv_proj.weight.copy_(weights["indexer_compressor_wkv"])
        module.gate_proj.weight.copy_(weights["indexer_compressor_wgate"])
        module.position_bias.copy_(weights["indexer_compressor_ape"])
        module.kv_norm.weight.copy_(weights["indexer_compressor_norm"])
        module.q_b_proj.weight.copy_(weights["indexer_wq_b"])
        module.scorer.weights_proj.weight.copy_(weights["indexer_proj"])
    return module


def _torch_indexer_scores(module, hidden, q_residual, position_ids):
    """Use torch internals only to expose score values for AC2 pick-set scoring."""

    batch, seq_len, _ = hidden.shape
    kv = module.kv_proj(hidden)
    gate = module.gate_proj(hidden)
    usable = (kv.shape[1] // module.compress_rate) * module.compress_rate
    chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // module.compress_rate
        ratio = module.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + module.position_bias
        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, module.head_dim))
        new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, module.head_dim), float("-inf"))
        new_kv[:, :, ratio:] = chunk_kv[..., module.head_dim :]
        new_gate[:, :, ratio:] = chunk_gate[..., module.head_dim :]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : module.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : module.head_dim]
        compressed = module.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
        positions = torch.arange(n_windows, device=compressed.device) * module.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = module.rotary_emb(compressed, position_ids=positions, layer_type=module.rope_layer_type)
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, module.head_dim))

    cos_q, sin_q = module.rotary_emb(hidden, position_ids=position_ids, layer_type=module.rope_layer_type)
    q = module.q_b_proj(q_residual).view(batch, seq_len, -1, module.head_dim).transpose(1, 2)
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
    return module.scorer(q, compressed, hidden)


def _mlx_indexer_scores(args, hidden, q_residual, weights, position_ids):
    """MLX score path used only for score-value comparison against torch oracle."""

    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
        _apply_rope_tail_mlx,
        _compress_rope_yarn_tail_tables_mlx,
        _indexer_scorer_mlx,
    )

    batch, seq_len, _ = hidden.shape
    rate = args.compression_ratio
    out_dim = args.index_head_dim
    usable = (seq_len // rate) * rate
    n_windows = usable // rate
    kv = hidden[:, :usable, :] @ weights["indexer_compressor_wkv"].T
    gate = hidden[:, :usable, :] @ weights["indexer_compressor_wgate"].T
    kv = kv.reshape((batch, n_windows, rate, 2 * out_dim))
    gate = gate.reshape((batch, n_windows, rate, 2 * out_dim)) + weights["indexer_compressor_ape"].reshape((1, 1, rate, 2 * out_dim))
    zero_kv = mx.zeros((batch, 1, rate, out_dim), dtype=kv.dtype)
    masked_gate = mx.full((batch, 1, rate, out_dim), -float("inf"), dtype=gate.dtype)
    ca_kv = mx.concatenate([zero_kv, kv[:, :-1, :, :out_dim]], axis=1)
    ca_gate = mx.concatenate([masked_gate, gate[:, :-1, :, :out_dim]], axis=1)
    new_kv = mx.concatenate([ca_kv, kv[..., out_dim:]], axis=2)
    new_gate = mx.concatenate([ca_gate, gate[..., out_dim:]], axis=2)
    probs = mx.softmax(new_gate.astype(mx.float32), axis=2).astype(new_kv.dtype)
    pooled = mx.sum(new_kv * probs, axis=2)
    compressed = pooled * mx.rsqrt(mx.mean(pooled * pooled, axis=-1, keepdims=True) + args.rms_norm_eps) * weights["indexer_compressor_norm"]
    rope_scaling = args.rope_scaling or {}
    c_cos, c_sin = _compress_rope_yarn_tail_tables_mlx(
        out_dim,
        args.qk_rope_head_dim,
        args.compress_rope_theta,
        float(rope_scaling.get("factor", 16.0)),
        int(rope_scaling.get("original_max_position_embeddings", 65536)),
        n_windows,
        positions=mx.arange(n_windows) * rate,
    )
    compressed = _apply_rope_tail_mlx(compressed, c_cos, c_sin, qk_rope_head_dim=args.qk_rope_head_dim)

    q = (q_residual @ weights["indexer_wq_b"].T).reshape((batch, seq_len, args.index_n_heads, out_dim))
    q_positions = mx.broadcast_to(mx.expand_dims(position_ids, -1), (batch, seq_len, args.index_n_heads)).reshape((batch * seq_len * args.index_n_heads,))
    q_cos, q_sin = _compress_rope_yarn_tail_tables_mlx(
        out_dim,
        args.qk_rope_head_dim,
        args.compress_rope_theta,
        float(rope_scaling.get("factor", 16.0)),
        int(rope_scaling.get("original_max_position_embeddings", 65536)),
        int(q_positions.shape[0]),
        positions=q_positions,
    )
    q = _apply_rope_tail_mlx(
        q.reshape((1, batch * seq_len * args.index_n_heads, out_dim)),
        q_cos,
        q_sin,
        qk_rope_head_dim=args.qk_rope_head_dim,
    ).reshape((batch, seq_len, args.index_n_heads, out_dim))
    return _indexer_scorer_mlx(
        q,
        compressed,
        hidden,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        weights_proj=weights["indexer_proj"],
    )


def test_indexer_mlx_matches_torch_topk_sets_sentinels_and_scores_real_dims():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _indexer_mlx

    assert _real_header_shape("layers.2.attn.indexer.compressor.wkv.weight") == (256, 4096)
    assert _real_header_shape("layers.2.attn.indexer.compressor.wgate.weight") == (256, 4096)
    assert _real_header_shape("layers.2.attn.indexer.compressor.ape") == (4, 256)
    assert _real_header_shape("layers.2.attn.indexer.compressor.norm.weight") == (128,)
    assert _real_header_shape("layers.2.attn.indexer.weights_proj.weight") == (64, 4096)
    assert _real_header_shape("layers.2.attn.indexer.wq_b.weight") == (8192, 1024)

    args = real_csa_mlx_args()
    config = real_csa_torch_config()
    hidden, q_residual, weights = _synth_indexer_fixture()
    position_ids = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)
    torch_module = _load_torch_indexer(config, weights)
    with torch.no_grad():
        expected = torch_module(
            hidden,
            q_residual,
            position_ids,
            past_key_values=None,
            layer_idx=0,
        )
        expected_scores = _torch_indexer_scores(torch_module, hidden, q_residual, position_ids)

    mlx_hidden = mx.array(hidden.numpy())
    mlx_q_residual = mx.array(q_residual.numpy())
    mlx_weights = {key: mx.array(value.numpy()) for key, value in weights.items()}
    mlx_position_ids = mx.array(position_ids.numpy())
    got_scores = _mlx_indexer_scores(args, mlx_hidden, mlx_q_residual, mlx_weights, mlx_position_ids)
    got = _indexer_mlx(
        args,
        mlx_hidden,
        mlx_q_residual,
        mlx_weights,
        position_ids=mlx_position_ids,
        index_topk=512,
    )
    mx.eval(got_scores, got)
    got_scores_t = torch.tensor(got_scores.tolist(), dtype=torch.float32)
    got_t = torch.tensor(got.tolist(), dtype=torch.long)

    compressed_len = hidden.shape[1] // 4
    assert tuple(got.shape) == tuple(expected.shape) == (1, hidden.shape[1], compressed_len)
    assert compressed_len == 64
    assert torch.equal(got_t.eq(-1), expected.eq(-1))
    torch.testing.assert_close(got_scores_t, expected_scores.float(), atol=1e-3, rtol=0.0)

    for b in range(expected.shape[0]):
        for t in range(expected.shape[1]):
            expected_valid = [int(i) for i in expected[b, t].tolist() if int(i) >= 0]
            got_valid = [int(i) for i in got_t[b, t].tolist() if int(i) >= 0]
            assert set(got_valid) == set(expected_valid), (b, t, expected_valid, got_valid)
            assert len(got_valid) == len(expected_valid)
