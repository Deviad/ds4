"""Trainable MLX nn.Module DeepSeek V4 port.

Story 13.3a-1 owns the skeleton, AttentionNN, HyperConnectionNN/HyperHeadNN,
and a shared-MLP stub.  The parity-fixture ``deepseek_v4.py`` remains frozen;
this module imports its proven MLX helpers instead of editing or copying them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.pipeline import PipelineMixin

from .deepseek_v4 import (
    ModelArgs as _ParityModelArgs,
    _apply_rope_tail_mlx,
    _attention_mlx,  # imported as the frozen math contract for AttentionNN (§8)
    _hyperconnection_mlx,
    _hyperhead_mlx,
    _rms_norm_mlx,
    sanitize_weights,
)


@dataclass
class ModelArgs(_ParityModelArgs):
    """DeepSeek V4 nn.Module config; sibling model_type keeps parity fixture intact."""

    model_type: str = "deepseek_v4_nn"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelArgs":
        model_type = config.get("model_type", "deepseek_v4_nn")
        if model_type != "deepseek_v4_nn":
            raise ValueError(f"expected model_type deepseek_v4_nn, got {model_type!r}")
        field_names = {field.name for field in fields(cls)}
        values = {key: config[key] for key in field_names if key in config}
        values.setdefault("model_type", "deepseek_v4_nn")
        args = cls(**values)
        args._normalize_layer_types()
        args.validate()
        return args

    def _normalize_layer_types(self) -> None:
        if self.mlp_layer_types is None:
            n_hash = int(self.num_hash_layers)
            self.mlp_layer_types = ["hash_moe"] * n_hash + ["moe"] * (self.num_hidden_layers - n_hash)
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types length must equal num_hidden_layers")
        invalid = [kind for kind in self.mlp_layer_types if kind not in {"hash_moe", "moe"}]
        if invalid:
            raise ValueError(f"unsupported DeepSeek V4 mlp layer types: {invalid}")
        if self.layer_types is None:
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")


def _normal(shape: tuple[int, ...], *, scale: float = 0.02) -> mx.array:
    return mx.random.normal(shape=shape) * scale


def _causal_sliding_mask(seq_len: int, sliding_window: int) -> mx.array:
    positions = mx.arange(seq_len)
    causal = mx.where(positions[:, None] >= positions[None, :], 0.0, -1e9)
    if sliding_window <= 0 or sliding_window >= seq_len:
        return causal
    return mx.where((positions[:, None] - positions[None, :]) < sliding_window, causal, -1e9)


def _rope_tail_tables(head_dim: int, qk_rope_head_dim: int, rope_theta: float, seq_len: int) -> tuple[mx.array, mx.array]:
    half = qk_rope_head_dim // 2
    positions = mx.arange(seq_len)
    dim_idx = mx.arange(0, qk_rope_head_dim, 2)
    freqs = 1.0 / (rope_theta ** (dim_idx / qk_rope_head_dim))
    angles = mx.expand_dims(positions, 1) * mx.expand_dims(freqs, 0)
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    if cos.shape[-1] != half or head_dim < qk_rope_head_dim:
        raise ValueError("invalid qk_rope_head_dim for RoPE tail")
    return cos, sin


class HyperConnectionNN(nn.Module):
    """mHC residual mixer; wraps frozen ``_hyperconnection_mlx`` (§6)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = _normal((mix, self.hc_mult * config.hidden_size))
        self.base = mx.zeros((mix,))
        self.scale = mx.ones((3,))

    def __call__(self, stream: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        result = _hyperconnection_mlx(
            stream,
            fn=self.fn,
            base=self.base,
            scale=self.scale,
            hc_mult=self.hc_mult,
            eps=self.hc_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            rms_norm_eps=self.rms_norm_eps,
        )
        return result["post"], result["comb"], result["collapsed"]


class HyperHeadNN(nn.Module):
    """Final HC-stream collapse; wraps frozen ``_hyperhead_mlx`` (§6)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.fn = _normal((self.hc_mult, self.hc_mult * config.hidden_size))
        self.base = mx.zeros((self.hc_mult,))
        self.scale = mx.ones((1,))

    def __call__(self, stream: mx.array) -> mx.array:
        if self.hc_mult == 1:
            return stream[..., 0, :]
        return _hyperhead_mlx(
            stream,
            fn=self.fn,
            base=self.base,
            scale=self.scale,
            hc_mult=self.hc_mult,
            eps=self.hc_eps,
            rms_norm_eps=self.rms_norm_eps,
        )


class AttentionNN(nn.Module):
    """DeepSeek V4 MLA attention: grouped-o, sink logits, RoPE tail (§8)."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.rope_theta = config.rope_theta
        self.sliding_window = config.sliding_window
        self.rms_norm_eps = config.rms_norm_eps
        if config.compression_ratio != 0:
            raise NotImplementedError("13.3a-1 AttentionNN supports only compression_ratio=0")
        if self.num_key_value_heads != 1:
            raise NotImplementedError("13.3a-1 AttentionNN supports only num_key_value_heads=1")
        if self.num_heads % self.o_groups != 0:
            raise ValueError("num_attention_heads must be divisible by o_groups")

        heads_per_group = self.num_heads // self.o_groups
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.o_a_proj = nn.Linear(heads_per_group * self.head_dim, self.o_groups * self.o_lora_rank, bias=False)
        self.o_b_proj = nn.Linear(self.o_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.sinks = mx.zeros((self.num_heads,))

    def __call__(self, x: mx.array, cache: Any | None = None) -> mx.array:
        del cache  # cache/CSA are 13.3b concerns; this slice is cache-less sliding attention.
        q_a = self.q_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_a).reshape((*x.shape[:-1], self.num_heads, self.head_dim))
        q = _rms_norm_mlx(q, mx.ones((self.head_dim,)), self.rms_norm_eps)
        kv = self.kv_norm(self.kv_proj(x))

        seq_len = x.shape[1]
        cos, sin = _rope_tail_tables(self.head_dim, self.qk_rope_head_dim, self.rope_theta, seq_len)
        q = _apply_rope_tail_mlx(q, cos, sin, qk_rope_head_dim=self.qk_rope_head_dim)
        kv = _apply_rope_tail_mlx(kv, cos, sin, qk_rope_head_dim=self.qk_rope_head_dim)

        scale = self.head_dim ** -0.5
        q_by_head = q.transpose(0, 2, 1, 3)
        kv_by_head = mx.expand_dims(kv, 1)
        scores = (q_by_head @ kv_by_head.transpose(0, 1, 3, 2)) * scale
        scores = scores + _causal_sliding_mask(seq_len, self.sliding_window)
        sink_logits = self.sinks.reshape((1, self.num_heads, 1, 1)) + mx.zeros_like(scores[..., :1])
        scores_with_sink = mx.concatenate([scores, sink_logits], axis=-1)
        probs = mx.softmax(scores_with_sink, axis=-1)[..., :seq_len]
        attended = probs @ kv_by_head
        attended = attended.transpose(0, 2, 1, 3)
        attended = _apply_rope_tail_mlx(attended, cos, -sin, qk_rope_head_dim=self.qk_rope_head_dim)

        heads_per_group = self.num_heads // self.o_groups
        low_rank_chunks = []
        for group in range(self.o_groups):
            head_start = group * heads_per_group
            head_end = head_start + heads_per_group
            row_start = group * self.o_lora_rank
            row_end = row_start + self.o_lora_rank
            group_heads = attended[..., head_start:head_end, :]
            flat_group = group_heads.reshape((*group_heads.shape[:-2], heads_per_group * self.head_dim))
            low_rank_chunks.append(self.o_a_proj(flat_group)[..., row_start:row_end])
        low_rank = mx.concatenate(low_rank_chunks, axis=-1)
        return self.o_b_proj(low_rank)


class MLPNN(nn.Module):
    """Shared SwiGLU MLP stub for 13.3a-1; SparseMoeBlockNN lands in 13.3a-2."""

    def __init__(self, config: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size * max(1, config.n_shared_experts or 1)
        self.limit = config.swiglu_limit
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array, input_ids: mx.array | None = None) -> mx.array:
        del input_ids  # hash routing is 13.3a-2; this stub keeps the signature.
        gate = mx.clip(self.gate_proj(x), None, self.limit)
        up = mx.clip(self.up_proj(x), -self.limit, self.limit)
        return self.down_proj(mx.sigmoid(gate) * gate * up)


class DecoderLayerNN(nn.Module):
    """Decoder layer over the [B, S, hc_mult, hidden] residual stream."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnectionNN(config)
        self.ffn_hc = HyperConnectionNN(config)
        self.self_attn = AttentionNN(config, layer_idx)
        self.mlp = MLPNN(config, config.mlp_layer_types[layer_idx])

    def __call__(self, hidden_states: mx.array, input_ids: mx.array | None = None, cache: Any | None = None) -> mx.array:
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(self.input_layernorm(collapsed), cache=cache)
        hidden_states = mx.expand_dims(post.astype(dtype), -1) * mx.expand_dims(attn_output, -2) + (
            comb.astype(dtype).transpose(0, 1, 3, 2) @ hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return mx.expand_dims(post.astype(dtype), -1) * mx.expand_dims(mlp_output, -2) + (
            comb.astype(dtype).transpose(0, 1, 3, 2) @ hidden_states
        )


class DeepseekV4ModelNN(PipelineMixin, nn.Module):
    """Backbone model: embed → HC stream decoder → HyperHead → RMSNorm."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayerNN(config, idx) for idx in range(config.num_hidden_layers)]
        self.hc_head = HyperHeadNN(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(mx.expand_dims(h, -2), (*h.shape[:-1], self.hc_mult, self.hidden_size))

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, input_ids=inputs, cache=layer_cache)

        return self.norm(self.hc_head(h))


class Model(nn.Module):
    """mlx-lm entry point for ``model_type='deepseek_v4_nn'``."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        if not isinstance(config, ModelArgs):
            if isinstance(config, dict):
                config = ModelArgs.from_dict(config)
            else:
                config = ModelArgs.from_dict(vars(config))
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4ModelNN(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        sanitized, _ = sanitize_weights(weights)
        return sanitized

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(key: str) -> bool:
            return "e_score_correction_bias" not in key

        return predicate


__all__ = [
    "ModelArgs",
    "Model",
    "DeepseekV4ModelNN",
    "DecoderLayerNN",
    "AttentionNN",
    "HyperConnectionNN",
    "HyperHeadNN",
    "MLPNN",
]
