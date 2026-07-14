"""Trainable MLX nn.Module DeepSeek V4 port.

Story 13.3a-1 owns the skeleton, AttentionNN, HyperConnectionNN/HyperHeadNN,
and a shared-MLP stub.  The parity-fixture ``deepseek_v4.py`` remains frozen;
this module imports its proven MLX helpers instead of editing or copying them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.pipeline import PipelineMixin


def _register_transformers_tokenizer_config() -> None:
    """Let AutoTokenizer tolerate the MLX-only ``deepseek_v4_nn`` model_type.

    ``mlx_lm.load`` imports this model before loading the tokenizer. Transformers
    itself has no DeepSeek-V4-nn config class, so register a narrow tokenizer-only
    config that discards DS4 layer/rope fields which generic validation rejects.
    """

    try:
        from transformers import AutoConfig, PretrainedConfig
    except Exception:  # pragma: no cover - transformers is optional outside mlx_lm.convert.
        return

    class _DeepseekV4NNTokenizerConfig(PretrainedConfig):
        model_type = "deepseek_v4_nn"

        def __init__(self, **kwargs: Any):
            for key in (
                "rope_scaling",
                "rope_theta",
                "compress_rope_theta",
                "layer_types",
                "mlp_layer_types",
                "compress_ratios",
                "compress_rates",
                "index_topk",
            ):
                kwargs.pop(key, None)
            super().__init__(**kwargs)

    try:
        AutoConfig.register("deepseek_v4_nn", _DeepseekV4NNTokenizerConfig)
    except ValueError as exc:
        if "already used" not in str(exc):
            raise


_register_transformers_tokenizer_config()


_DEFAULT_COMPRESS_RATES = {
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _layer_type_from_ratio(ratio: int) -> str:
    if ratio == 0:
        return "sliding_attention"
    if ratio == 4:
        return "compressed_sparse_attention"
    if ratio == 128:
        return "heavily_compressed_attention"
    raise ValueError(f"unsupported DeepSeek V4 attention compression ratio {ratio}")


def _layer_compression_ratio(config: Any, layer_idx: int) -> int:
    ratios = getattr(config, "compress_ratios", None)
    if ratios is not None:
        if layer_idx >= len(ratios):
            raise ValueError("compress_ratios length must cover num_hidden_layers")
        return int(ratios[layer_idx])
    layer_types = getattr(config, "layer_types", None)
    layer_type = "sliding_attention" if layer_types is None else layer_types[layer_idx]
    if layer_type == "sliding_attention":
        return int(getattr(config, "compression_ratio", 0)) if getattr(config, "compression_ratio", 0) else 0
    rates = getattr(config, "compress_rates", None) or _DEFAULT_COMPRESS_RATES
    if layer_type not in rates:
        raise NotImplementedError(f"unsupported DeepSeek V4 attention layer_type {layer_type!r}")
    return int(rates[layer_type])


from .deepseek_v4 import (
    ModelArgs as _ParityModelArgs,
    _attention_mlx,  # imported as the frozen math contract for AttentionNN (§8)
    _dequantize_fp4_block_scale_mlx,
    _hyperconnection_mlx,
    _hyperhead_mlx,
    _softplus_mlx,
    sanitize_weights,
)


@dataclass
class ModelArgs(_ParityModelArgs):
    """DeepSeek V4 nn.Module config; sibling model_type keeps parity fixture intact."""

    model_type: str = "deepseek_v4_nn"
    compress_rates: dict[str, int] | None = None
    compress_ratios: list[int] | None = None
    index_topk: int = 512

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
        if self.compress_ratios is not None:
            if len(self.compress_ratios) < self.num_hidden_layers:
                raise ValueError("compress_ratios length must cover num_hidden_layers")
            self.compress_ratios = [int(value) for value in self.compress_ratios[: self.num_hidden_layers]]
            # The shimmed real config currently carries stale all-sliding layer_types;
            # the per-layer compression ratios are the truthful attention layout.
            if all(kind == "sliding_attention" for kind in self.layer_types) and any(value != 0 for value in self.compress_ratios):
                self.layer_types = [_layer_type_from_ratio(value) for value in self.compress_ratios]


def _normal(shape: tuple[int, ...], *, scale: float = 0.02) -> mx.array:
    return mx.random.normal(shape=shape) * scale


def _ceil_to_block(value: int, block: int) -> int:
    return ((int(value) + block - 1) // block) * block


def _pad_last_dim(x: mx.array, target: int) -> mx.array:
    current = x.shape[-1]
    if current == target:
        return x
    if current > target:
        raise ValueError(f"cannot project {current} features through FP4 weight with {target} inputs")
    pad_width = tuple((0, 0) for _ in range(x.ndim - 1)) + ((0, target - current),)
    return mx.pad(x, pad_width)


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


class AttentionCompressorNN(nn.Module):
    """Trainable compressor leaves; AttentionNN exports them into functional weights."""

    def __init__(self, *, hidden_size: int, out_dim: int, rate: int, overlap: bool, eps: float):
        super().__init__()
        projection_dim = (2 if overlap else 1) * out_dim
        self.wkv = nn.Linear(hidden_size, projection_dim, bias=False)
        self.wgate = nn.Linear(hidden_size, projection_dim, bias=False)
        self.ape = _normal((rate, projection_dim))
        self.norm = nn.RMSNorm(out_dim, eps=eps)


class AttentionIndexerNN(nn.Module):
    """CSA indexer leaves mirroring torch DeepseekV4Indexer."""

    def __init__(self, config: ModelArgs, rate: int):
        super().__init__()
        self.compressor = AttentionCompressorNN(
            hidden_size=config.hidden_size,
            out_dim=config.index_head_dim,
            rate=rate,
            overlap=True,
            eps=config.rms_norm_eps,
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.wq_b = nn.Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False)


class AttentionNN(nn.Module):
    """DeepSeek V4 MLA attention: grouped-o, sink logits, RoPE tail (§8)."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if config.layer_types is None or layer_idx >= len(config.layer_types):
            raise ValueError("AttentionNN requires layer_types covering layer_idx")
        self.layer_type = config.layer_types[layer_idx]
        self.compression_ratio = _layer_compression_ratio(config, layer_idx)
        self.config = replace(config, compression_ratio=self.compression_ratio)
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
        self.index_topk = int(getattr(config, "index_topk", 512))
        if self.num_key_value_heads != 1:
            raise NotImplementedError("13.3a-1 AttentionNN supports only num_key_value_heads=1")
        if self.num_heads % self.o_groups != 0:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if self.compression_ratio not in {0, 4, 128}:
            raise NotImplementedError(f"AttentionNN does not support compression_ratio={self.compression_ratio}")

        heads_per_group = self.num_heads // self.o_groups
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.o_a_proj = nn.Linear(heads_per_group * self.head_dim, self.o_groups * self.o_lora_rank, bias=False)
        self.o_b_proj = nn.Linear(self.o_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.sinks = mx.zeros((self.num_heads,))
        if self.compression_ratio == 4:
            self.compressor = AttentionCompressorNN(
                hidden_size=self.hidden_size,
                out_dim=self.head_dim,
                rate=4,
                overlap=True,
                eps=self.rms_norm_eps,
            )
            self.indexer = AttentionIndexerNN(config, rate=4)
        elif self.compression_ratio == 128:
            self.compressor = AttentionCompressorNN(
                hidden_size=self.hidden_size,
                out_dim=self.head_dim,
                rate=128,
                overlap=False,
                eps=self.rms_norm_eps,
            )

    def __call__(self, x: mx.array, cache: Any | None = None) -> mx.array:
        del cache  # cache/stateful KV remain out of scope; 13.3b real path is cache-less.

        def linear_weight(linear: Any) -> mx.array:
            if isinstance(linear, nn.QuantizedLinear):
                return mx.dequantize(
                    linear.weight,
                    linear.scales,
                    linear.biases,
                    group_size=linear.group_size,
                    bits=linear.bits,
                    mode=linear.mode,
                )
            weight = getattr(linear, "weight", None)
            if weight is not None:
                return weight
            base = getattr(linear, "linear", None)
            if base is not None and hasattr(linear, "lora_a") and hasattr(linear, "lora_b"):
                base_weight = linear_weight(base)
                delta = ((linear.scale * linear.lora_b.T) @ linear.lora_a.T).astype(base_weight.dtype)
                return base_weight + delta
            raise AttributeError(f"{type(linear).__name__} does not expose a frozen-compatible weight")

        weights = {
            "q_a_proj.weight": linear_weight(self.q_a_proj),
            "q_norm.weight": self.q_norm.weight,
            "q_b_proj.weight": linear_weight(self.q_b_proj),
            "kv_proj.weight": linear_weight(self.kv_proj),
            "kv_norm.weight": self.kv_norm.weight,
            "o_a_proj.weight": linear_weight(self.o_a_proj),
            "o_b_proj.weight": linear_weight(self.o_b_proj),
            "sinks": self.sinks,
        }
        if self.compression_ratio == 0:
            return _attention_mlx(self.config, x, weights)

        weights.update(
            {
                "compressor_wkv": linear_weight(self.compressor.wkv),
                "compressor_wgate": linear_weight(self.compressor.wgate),
                "compressor_ape": self.compressor.ape,
                "compressor_norm": self.compressor.norm.weight,
            }
        )
        if self.compression_ratio == 4:
            weights.update(
                {
                    "indexer_compressor_wkv": linear_weight(self.indexer.compressor.wkv),
                    "indexer_compressor_wgate": linear_weight(self.indexer.compressor.wgate),
                    "indexer_compressor_ape": self.indexer.compressor.ape,
                    "indexer_compressor_norm": self.indexer.compressor.norm.weight,
                    "indexer_proj": linear_weight(self.indexer.weights_proj),
                    "indexer_wq_b": linear_weight(self.indexer.wq_b),
                }
            )
            return _attention_mlx(self.config, x, weights, index_topk=self.index_topk)
        return _attention_mlx(self.config, x, weights)


class MLPNN(nn.Module):
    """Shared SwiGLU expert reused by SparseMoeBlockNN (§A Q4-Q5)."""

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
        del input_ids
        gate = mx.clip(self.gate_proj(x), None, self.limit)
        up = mx.clip(self.up_proj(x), -self.limit, self.limit)
        return self.down_proj(mx.sigmoid(gate) * gate * up)


class DeepseekV4FP4Experts(nn.Module):
    """Frozen FP4 routed experts; dequantizes on the fly via ADR 0024 primitive."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        if str(config.expert_dtype).lower() != "fp4":
            raise NotImplementedError("DeepseekV4FP4Experts supports only expert_dtype='fp4'")
        self.n_routed_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.limit = config.swiglu_limit

        # OCP MXFP4 packs two logical input features per byte and scales each
        # 32-logical-feature block.  Tiny 16-wide tests are padded only on the
        # FP4 input axes; real 4096/2048 shapes stay byte-identical to ckpt layout.
        hidden_logical = _ceil_to_block(self.hidden_size, 32)
        intermediate_logical = _ceil_to_block(self.intermediate_size, 32)
        hidden_packed = hidden_logical // 2
        hidden_scale = hidden_logical // 32
        intermediate_packed = intermediate_logical // 2
        intermediate_scale = intermediate_logical // 32

        self.w1_weight = mx.zeros((self.n_routed_experts, self.intermediate_size, hidden_packed), dtype=mx.uint8)
        self.w1_scale = mx.zeros((self.n_routed_experts, self.intermediate_size, hidden_scale), dtype=mx.bfloat16)
        self.w3_weight = mx.zeros((self.n_routed_experts, self.intermediate_size, hidden_packed), dtype=mx.uint8)
        self.w3_scale = mx.zeros((self.n_routed_experts, self.intermediate_size, hidden_scale), dtype=mx.bfloat16)
        self.w2_weight = mx.zeros((self.n_routed_experts, self.hidden_size, intermediate_packed), dtype=mx.uint8)
        self.w2_scale = mx.zeros((self.n_routed_experts, self.hidden_size, intermediate_scale), dtype=mx.bfloat16)
        self.freeze()

    def _dequant(self, weight: mx.array, scale: mx.array) -> mx.array:
        return _dequantize_fp4_block_scale_mlx(weight, scale)

    def forward_one(self, eid: int, x: mx.array) -> mx.array:
        w1 = self._dequant(self.w1_weight[eid], self.w1_scale[eid])
        w2 = self._dequant(self.w2_weight[eid], self.w2_scale[eid])
        w3 = self._dequant(self.w3_weight[eid], self.w3_scale[eid])
        x_for_w1 = _pad_last_dim(x, w1.shape[1])
        gate = mx.clip(x_for_w1 @ w1.T, None, self.limit)
        up = mx.clip(x_for_w1 @ w3.T, -self.limit, self.limit)
        hidden = mx.sigmoid(gate) * gate * up
        hidden_for_w2 = _pad_last_dim(hidden, w2.shape[1])
        return hidden_for_w2 @ w2.T

    def __call__(self, x: mx.array, eids: mx.array) -> mx.array:
        slot_outputs = []
        for slot in range(eids.shape[-1]):
            out = mx.zeros_like(x)
            slot_eids = eids[..., slot]
            for eid in range(self.n_routed_experts):
                selected = (slot_eids == eid).astype(x.dtype)
                out = out + self.forward_one(eid, x) * mx.expand_dims(selected, -1)
            slot_outputs.append(out)
        return mx.stack(slot_outputs, axis=-2)


def _host_unique_rows_per_expert(
    indices_flat: mx.array, n_experts: int
) -> list[list[int]]:
    """Host-materialize unique flat token rows per expert from stopped indices.

    ``mx.stop_gradient`` must be applied by the caller before materialization.
    Duplicate expert ids within one token are collapsed exactly once.
    Token order within each expert’s row list is preserved.
    """
    indices_np = indices_flat.tolist()  # [T, K]
    rows_by_expert: list[list[int]] = [[] for _ in range(n_experts)]
    for t, slots in enumerate(indices_np):
        seen: set[int] = set()
        for raw_eid in slots:
            eid = int(raw_eid)
            if eid < 0 or eid >= n_experts:
                raise ValueError(f"route expert id {eid} out of range [0, {n_experts})")
            if eid not in seen:
                rows_by_expert[eid].append(t)
                seen.add(eid)
    return rows_by_expert


class SparseMoeBlockNN(nn.Module):
    """DeepSeek V4 sparse MoE with learned top-k or tid2eid hash routing."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.mlp_layer_types[layer_idx]
        self.is_hash = self.layer_type == "hash_moe"
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        if self.top_k > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")

        self.gate_weight = _normal((self.n_routed_experts, self.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        if self.is_hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        self.experts = DeepseekV4FP4Experts(config)
        self.shared_experts = MLPNN(config, "shared_experts")

        frozen_keys = ["e_score_correction_bias"]
        if self.is_hash:
            frozen_keys.append("tid2eid")
        self.freeze(recurse=False, keys=frozen_keys)

    @property
    def gate_proj(self):
        return self.shared_experts.gate_proj

    @property
    def up_proj(self):
        return self.shared_experts.up_proj

    @property
    def down_proj(self):
        return self.shared_experts.down_proj

    def _scores(self, x: mx.array) -> mx.array:
        if self.scoring_func != "sqrtsoftplus":
            raise NotImplementedError("SparseMoeBlockNN supports only scoring_func='sqrtsoftplus'")
        logits = x @ self.gate_weight.T
        return mx.sqrt(_softplus_mlx(logits))

    def _select_indices(self, scores: mx.array, x_shape: tuple[int, ...], input_ids: mx.array | None) -> mx.array:
        if self.is_hash:
            if input_ids is None:
                raise ValueError("hash_moe SparseMoeBlockNN requires input_ids")
            return self.tid2eid[input_ids.reshape(-1)].reshape(*x_shape[:-1], self.top_k)
        selection_scores = scores + self.e_score_correction_bias
        return mx.argsort(-selection_scores, axis=-1)[..., : self.top_k]

    def routing_indices(self, x: mx.array, input_ids: mx.array | None = None) -> mx.array:
        return self._select_indices(self._scores(x), x.shape, input_ids)

    def _route(self, x: mx.array, input_ids: mx.array | None) -> tuple[mx.array, mx.array, mx.array]:
        scores = self._scores(x)
        indices = self._select_indices(scores, x.shape, input_ids)
        denom = mx.zeros_like(scores[..., 0])
        for eid in range(self.n_routed_experts):
            selected = mx.any(indices == eid, axis=-1).astype(scores.dtype)
            denom = denom + scores[..., eid] * selected
        return scores, indices, denom

    def __call__(self, x: mx.array, input_ids: mx.array | None = None) -> mx.array:
        scores, indices, _denom = self._route(x, input_ids)

        # Flatten leading token axes for per-expert gather/scatter.  Only the
        # stopped integer route metadata crosses the host boundary; packed expert
        # payloads, activations, scores, outputs, and cotangents stay in MLX.
        orig_shape = x.shape
        T = int(math.prod(orig_shape[:-1]))
        H = int(orig_shape[-1])
        x_flat = x.reshape(T, H)
        scores_flat = scores.reshape(T, self.n_routed_experts)
        indices_flat = indices.reshape(T, self.top_k)
        rows_by_expert = _host_unique_rows_per_expert(
            mx.stop_gradient(indices_flat), self.n_routed_experts
        )

        from ds4_ft_mlx.routed_fp4_metal import routed_fp4

        routed_flat = routed_fp4(
            x_flat,
            scores_flat,
            rows_by_expert,
            self.experts,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        routed = routed_flat.reshape(orig_shape)
        return routed + self.shared_experts(x)


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
        self.mlp = SparseMoeBlockNN(config, layer_idx)

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

    def load_weights(self, weights: dict[str, Any] | list[tuple[str, Any]], strict: bool = True) -> None:
        # Story 13.3b-4b, ADR 0025/0027: shimmed ckpt-native keys -> nn param tree.
        # Mirrors frozen deepseek_v4.Model.load_weights remap-pre-pass shape.
        from ds4_ft_mlx.deepseek_v4_nn_remap import remap_weight_dict

        mapping = dict(weights)
        # IDEMPOTENT GATE: model-4bit reloads already carry nn keys (model.* / lm_head.*,
        # including quant .scales/.biases). Remapping those would trip orphan-key guards.
        is_native = any(not key.startswith(("model.", "lm_head.")) for key in mapping)
        if is_native:
            cfg = {
                "n_routed_experts": self.args.n_routed_experts,
                "num_hash_layers": self.args.num_hash_layers,
            }
            mapping, _report = remap_weight_dict(mapping, config=cfg)
        super().load_weights(list(mapping.items()), strict=strict)

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
    "AttentionCompressorNN",
    "AttentionIndexerNN",
    "HyperConnectionNN",
    "HyperHeadNN",
    "MLPNN",
    "DeepseekV4FP4Experts",
    "SparseMoeBlockNN",
]
