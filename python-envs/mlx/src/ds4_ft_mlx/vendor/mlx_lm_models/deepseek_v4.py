"""Fail-closed DeepSeek V4 scaffold for MLX-LM.

This Story 11 implementation provides project-controlled import/registration,
config parsing, explicit fixture modes, and a bounded real MLX forward path for
proven tiny configurations.  Synthetic multi-layer stacking is supported for
num_hidden_layers in {1,2,3} with hc_mult=1; multi-layer with hc_mult>1 fails
closed because no stacked reference proves it.  Full DeepSeek V4 support still
fails closed for the real 43-layer Flash config, stateful/real-scale
compressors/indexers, unproven expert layouts, real checkpoint load/forward,
and generation smoke.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
import math
from typing import Any

try:
    import mlx.core as mx
except ModuleNotFoundError:  # pragma: no cover - optional dependency in non-MLX envs
    mx = None

from ds4_ft_mlx.deepseek_v4_attention_spec import (
    DeepSeekV4AttentionSpec,
    _linear,
    _rms_norm,
    _softmax,
    apply_output_inverse_rope_tail,
    apply_rope_tail,
    hca_block_bias,
    tiny_compressor_indexer_attention_reference,
    tiny_hyperconnection_forward,
    tiny_multihead_grouped_attention_reference,
)
from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_forward


_REQUIRED_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "q_lora_rank",
    "qk_rope_head_dim",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
)
_SUPPORTED_EXPERT_DTYPES = {"fp4", "i8", "I8"}
# OCP MXFP4 E2M1 (1 sign, 2 exp, 1 mantissa, bias=1). Index = 4-bit code 0..15.
# Codes 8..15 are the negative half (-0.0 .. -6.0).
_E2M1_FP4_LUT = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_E2M1_FP4_LUT_MX = mx.array(_E2M1_FP4_LUT, dtype=mx.float32) if mx is not None else None


def forward_parity_blockers() -> tuple[str, ...]:
    return (
        "full attention parity with RoPE/cache/sinks/compressor/indexer",
        "full decoder-layer hyperconnection residual mixing and final hyperhead parity",
        "full MoE parity with packed FP4/I8 expert dequant and expert kernels",
        "full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)",
    )


def _canonicalize_weight_key(name: str, *, num_hidden_layers: int = 1) -> str:
    """Map standard Transformers weight names to internal fixture keys.

    Accepts both internal keys (e.g. ``embed.weight``) and standard
    Transformers names (``model.embed_tokens.weight``,
    ``model.layers.0.input_layernorm.weight``).  For single-layer real models
    the layer index is stripped; for multi-layer synthetic models the layer
    index is preserved as ``layers.{i}.<sub>`` so per-layer weights can be
    distinguished.
    """

    if name.startswith("model.embed_tokens.weight"):
        return "embed.weight"
    if name.startswith("model.norm.weight"):
        return "norm.weight"
    if name.startswith("model.hc_head."):
        return name[len("model."):]
    if name.startswith("lm_head.weight"):
        return "lm_head.weight"
    if name.startswith("model.layers."):
        rest = name[len("model.layers."):]
        idx_sep = rest.find(".")
        if idx_sep == -1:
            return rest
        layer_idx = rest[:idx_sep]
        rest = rest[idx_sep + 1:]
        if rest.startswith("self_attn."):
            rest = rest[len("self_attn."):]
            # Transformers uses q_a_norm for the q_norm weight.
            if rest == "q_a_norm.weight":
                rest = "q_norm.weight"
        if num_hidden_layers > 1:
            return f"layers.{layer_idx}.{rest}"
        return rest
    return name


def _mlx_array_to_py(value: Any) -> Any:
    """Convert an MLX (or numpy-like) array to nested Python lists/scalars."""

    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_mlx_array(value: Any) -> Any:
    """Convert a Python list or MLX/numpy array to an MLX array."""

    if mx is None:
        raise NotImplementedError("mlx is required for real DeepSeek V4 load_weights but is not installed")
    if isinstance(value, mx.array):
        return value
    return mx.array(value)


# ---------------------------------------------------------------------------
# Bounded real-mode MLX helpers (synthetic small configs: up to 3 layers with
# hc_mult=1, compressor-free sliding attention, multi-head/grouped attention,
# top-k MoE).  Unproven components remain fail-closed.
# ---------------------------------------------------------------------------


def _rms_norm_mlx(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _linear_mlx(x: mx.array, weight: mx.array) -> mx.array:
    # weight shape: [out_features, in_features]
    return x @ weight.T


def _dequantize_i8_block_scale_mlx(weight: mx.array, scale: mx.array, *, block_size: int = 16, axis: int = 1) -> mx.array:
    """Dequantize signed I8 expert weights with BF16/F32 block scales."""

    if mx is None:
        raise NotImplementedError("mlx is required for I8 expert dequantization")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if axis < 0:
        axis += len(weight.shape)
    if axis < 0 or axis >= len(weight.shape):
        raise ValueError("scale axis out of range")
    if weight.shape[axis] % block_size != 0:
        raise ValueError("I8 block-scale dequant requires complete blocks")
    expected_scale_shape = list(weight.shape)
    expected_scale_shape[axis] = weight.shape[axis] // block_size
    if tuple(scale.shape) != tuple(expected_scale_shape):
        raise ValueError(f"I8 scale shape mismatch: got {scale.shape}, expected {tuple(expected_scale_shape)}")
    expanded = mx.repeat(scale.astype(mx.float32), repeats=block_size, axis=axis)
    return weight.astype(mx.float32) * expanded


def _dequantize_fp4_block_scale_mlx(weight: mx.array, scale: mx.array, *, block_size: int = 32, axis: int = 1) -> mx.array:
    """On-the-fly FP4 (OCP MXFP4 E2M1) → fp32 dequant of byte-packed routed experts.

    Shimmed-ckpt routed-expert I8 containers hold 2 E2M1 nibbles/byte along the
    in-features axis, LSB-first (low nibble = even logical element). Per-32-logical
    BF16 block scale, linear domain (direct multiply, NOT E8M0). Mirrors the i8
    primitive's STRUCTURE (shape guards + mx.repeat broadcast); math derives from
    OCP MXFP4 spec + HF mxfp4 reference (transformers/integrations/mxfp4.py:28-45,
    292-298), NOT _dequantize_i8_block_scale_mlx / _apply_i8_block_scales (ADR 0017).
    """

    if mx is None or _E2M1_FP4_LUT_MX is None:
        raise NotImplementedError("mlx is required for FP4 expert dequantization")
    if block_size != 32:
        raise ValueError("fp4 block_size must be 32 (OCP MXFP4 default; BA Q2 LOCKED)")
    if axis < 0:
        axis += len(weight.shape)
    if axis < 0 or axis >= len(weight.shape):
        raise ValueError("scale axis out of range")
    if axis != 1:
        raise ValueError("fp4 scale axis must be 1 (in-features)")
    in_logical = weight.shape[axis] * 2
    if in_logical % block_size != 0:
        raise ValueError("FP4 block-scale dequant requires complete logical blocks")

    u = weight.astype(mx.uint8)
    lo_idx = mx.bitwise_and(u, 0x0F)
    hi_idx = mx.bitwise_and(mx.right_shift(u, 4), 0x0F)
    lo_val = _E2M1_FP4_LUT_MX[lo_idx]
    hi_val = _E2M1_FP4_LUT_MX[hi_idx]
    fp4 = mx.stack([lo_val, hi_val], axis=-1).reshape(u.shape[0], u.shape[1] * 2)

    n_blocks = scale.shape[axis]
    if n_blocks * block_size != fp4.shape[axis]:
        raise ValueError(f"fp4 scale cols {n_blocks} * block_size {block_size} != in_logical {fp4.shape[axis]}")
    expected_scale_shape = list(weight.shape)
    expected_scale_shape[axis] = n_blocks
    if tuple(scale.shape) != tuple(expected_scale_shape):
        raise ValueError(f"fp4 scale shape mismatch: got {scale.shape}, expected {tuple(expected_scale_shape)}")
    scale_expanded = mx.repeat(scale.astype(mx.float32), repeats=block_size, axis=axis)
    return fp4 * scale_expanded


def _softplus_mlx(x: mx.array) -> mx.array:
    return mx.log1p(mx.exp(x))


def _rope_tables_mlx(head_dim: int, rope_theta: float, seq_len: int) -> tuple[mx.array, mx.array]:
    positions = mx.arange(seq_len)
    dim_idx = mx.arange(0, head_dim, 2)
    freqs = 1.0 / (rope_theta ** (dim_idx / head_dim))
    angles = mx.expand_dims(positions, 1) * mx.expand_dims(freqs, 0)
    return mx.cos(angles), mx.sin(angles)


def _validate_qk_rope_head_dim(head_dim: int, qk_rope_head_dim: int) -> None:
    if type(qk_rope_head_dim) is not int or qk_rope_head_dim <= 0:
        raise ValueError("qk_rope_head_dim must be a positive even integer")
    if qk_rope_head_dim % 2 != 0:
        raise ValueError("qk_rope_head_dim must be even for pairwise RoPE")
    if qk_rope_head_dim > head_dim:
        raise ValueError("qk_rope_head_dim cannot exceed head_dim")


def _rope_tail_tables_mlx(head_dim: int, qk_rope_head_dim: int, rope_theta: float, seq_len: int) -> tuple[mx.array, mx.array]:
    """RoPE tables for only the q/k tail slice, not the full head_dim."""

    _validate_qk_rope_head_dim(head_dim, qk_rope_head_dim)
    positions = mx.arange(seq_len)
    dim_idx = mx.arange(0, qk_rope_head_dim, 2)
    freqs = 1.0 / (rope_theta ** (dim_idx / qk_rope_head_dim))
    angles = mx.expand_dims(positions, 1) * mx.expand_dims(freqs, 0)
    return mx.cos(angles), mx.sin(angles)


def _broadcast_rope_tail_table_mlx(table: mx.array, x: mx.array, qk_rope_head_dim: int) -> mx.array:
    half = qk_rope_head_dim // 2
    if table.shape[-1] != half:
        raise ValueError(f"RoPE table width must be qk_rope_head_dim/2={half}, got {table.shape[-1]}")
    if len(table.shape) == 1:
        return table
    if len(table.shape) != 2:
        raise ValueError("RoPE table must have shape [half] or [seq, half]")
    seq_len = table.shape[0]
    x_ndim = len(x.shape)
    if x_ndim == 2:
        return table
    if x_ndim == 3:
        return table.reshape((1, seq_len, half))
    if x_ndim == 4:
        return table.reshape((1, seq_len, 1, half))
    raise ValueError("RoPE tail helper expects x shape [seq, head], [batch, seq, head], or [batch, seq, heads, head]")


def _apply_rope_mlx(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # Pairwise rotate_half; x shape [..., head_dim]; cos/sin shape [seq, head_dim/2].
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return mx.stack([y1, y2], axis=-1).reshape(x.shape)


def _apply_rope_tail_mlx(x: mx.array, cos: mx.array, sin: mx.array, *, qk_rope_head_dim: int) -> mx.array:
    """Apply pairwise RoPE to only the last qk_rope_head_dim channels."""

    head_dim = int(x.shape[-1])
    _validate_qk_rope_head_dim(head_dim, qk_rope_head_dim)
    cos = _broadcast_rope_tail_table_mlx(cos, x, qk_rope_head_dim)
    sin = _broadcast_rope_tail_table_mlx(sin, x, qk_rope_head_dim)
    nope_dim = head_dim - qk_rope_head_dim
    prefix = x[..., :nope_dim]
    tail = x[..., nope_dim:]
    rotated_tail = _apply_rope_mlx(tail, cos, sin)
    return mx.concatenate([prefix, rotated_tail], axis=-1)


_CSA_WEIGHT_KEYS = frozenset({
    "compressor_wkv",
    "compressor_wgate",
    "compressor_ape",
    "compressor_norm",
    "indexer_wq_b",
    "indexer_proj",
    "indexer_compressor_wkv",
    "indexer_compressor_wgate",
    "indexer_compressor_ape",
    "indexer_compressor_norm",
})


def _linear_in_out_mlx(x: mx.array, weight: mx.array) -> mx.array:
    """Linear projection for spec-layout ``[in_features, out_features]`` weights."""

    return x @ weight


def _rope_full_tables_mlx(head_dim: int, positions: mx.array, rope_theta: float) -> tuple[mx.array, mx.array]:
    """CSA uses full-head RoPE at compressed block start positions."""

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("full RoPE head_dim must be positive even")
    dim_idx = mx.arange(0, head_dim, 2)
    freqs = 1.0 / (rope_theta ** (dim_idx / head_dim))
    angles = mx.expand_dims(positions.astype(mx.float32), -1) * mx.expand_dims(freqs, 0)
    return mx.cos(angles), mx.sin(angles)


def _apply_rope_full_mlx(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply pairwise full-channel RoPE without the tail-only split."""

    return _apply_rope_mlx(x, cos, sin)


def _csa_config_error(args: ModelArgs) -> str | None:
    if args.compression_ratio != 4:
        return "MLX compressors/indexers CSA compressed attention supports only proven compression_ratio=4"
    if args.num_attention_heads != 1 or args.o_groups != 1:
        return "MLX compressors/indexers CSA compressed attention is proven only for single-head o_groups=1 tiny fixtures"
    if args.num_key_value_heads != 1:
        return "MLX CSA compressed attention is proven only for num_key_value_heads=1"
    if args.hc_mult != 1:
        return "MLX CSA compressed attention is proven only for hc_mult=1 cache-less tiny fixtures"
    if args.hidden_size != args.head_dim:
        return "MLX CSA compressed attention requires hidden_size == head_dim in the proven tiny subset"
    if args.q_lora_rank != args.hidden_size:
        return "MLX CSA compressed attention requires q_lora_rank == hidden_size for q_residual fixture parity"
    if args.index_n_heads <= 0 or args.index_head_dim <= 0:
        return "MLX CSA compressed attention requires positive indexer dimensions"
    return None


def _require_csa_config(args: ModelArgs) -> None:
    error = _csa_config_error(args)
    if error is not None:
        raise NotImplementedError(error)


def _require_csa_weights(weights: dict[str, mx.array]) -> None:
    missing = sorted(_CSA_WEIGHT_KEYS - set(weights))
    if missing:
        raise ValueError(f"missing MLX CSA compressed attention weights: {', '.join(missing)}")


def _csa_windowed_compressor_mlx(
    x: mx.array,
    *,
    wkv: mx.array,
    wgate: mx.array,
    ape: mx.array,
    norm: mx.array,
    compress_rate: int,
    out_dim: int,
    rms_norm_eps: float,
    rope_theta: float,
) -> mx.array:
    """Stateless CSA Ca/Cb softmax-gated compressor in MLX arrays.

    The output is cache-less [batch, compressed_blocks, out_dim]; Ca reuses the
    previous window only inside this single forward, while cross-call carry is
    deferred to the generation/KV-cache story.
    """

    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    expected_ape_shape = (2 * out_dim, compress_rate)
    if tuple(ape.shape) != expected_ape_shape:
        raise ValueError(f"CSA compressor ape shape mismatch: got {ape.shape}, expected {expected_ape_shape}")
    if tuple(norm.shape) != (out_dim,):
        raise ValueError(f"CSA compressor norm shape mismatch: got {norm.shape}, expected {(out_dim,)}")

    batch, seq_len, _hidden = x.shape
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate
    if n_windows == 0:
        return mx.zeros((batch, 0, out_dim), dtype=x.dtype)

    kv = _linear_in_out_mlx(x[:, :usable, :], wkv)
    gate = _linear_in_out_mlx(x[:, :usable, :], wgate)
    if tuple(kv.shape[-1:]) != (2 * out_dim,) or tuple(gate.shape[-1:]) != (2 * out_dim,):
        raise ValueError("CSA compressor projections must produce 2*out_dim channels")

    # APE is stored [2*out_dim, rate]; transpose to token-major Ca/Cb biases
    # so the softmax gate pools over the overlap-token axis like Transformers.
    ca_bias = ape[:out_dim, :].T.reshape((1, compress_rate, out_dim))
    cb_bias = ape[out_dim:, :].T.reshape((1, compress_rate, out_dim))
    zero_kv = mx.zeros((batch, compress_rate, out_dim), dtype=kv.dtype)
    masked_gate = mx.full((batch, compress_rate, out_dim), -1e9, dtype=gate.dtype)
    if n_windows == 0:
        return mx.zeros((batch, 0, out_dim), dtype=x.dtype)

    positions = mx.arange(n_windows) * compress_rate
    cos, sin = _rope_full_tables_mlx(out_dim, positions, rope_theta)

    compressed = []
    for window in range(n_windows):
        current = slice(window * compress_rate, (window + 1) * compress_rate)
        cb_kv = kv[:, current, out_dim:]
        cb_gate = gate[:, current, out_dim:] + cb_bias
        if window == 0:
            ca_kv = zero_kv
            ca_gate = masked_gate
        else:
            previous = slice((window - 1) * compress_rate, window * compress_rate)
            ca_kv = kv[:, previous, :out_dim]
            ca_gate = gate[:, previous, :out_dim] + ca_bias
        new_kv = mx.concatenate([ca_kv, cb_kv], axis=1)
        new_gate = mx.concatenate([ca_gate, cb_gate], axis=1)
        # Gate across Ca+Cb tokens, then RMSNorm and full-head RoPE at the
        # compressed block start to preserve the reference block ordering.
        pooled = mx.sum(mx.softmax(new_gate, axis=1) * new_kv, axis=1)
        normed = _rms_norm_mlx(pooled, norm, rms_norm_eps)
        compressed.append(mx.expand_dims(_apply_rope_full_mlx(normed, cos[window], sin[window]), axis=1))
    return mx.concatenate(compressed, axis=1)


def _csa_compressor_mlx(args: ModelArgs, x: mx.array, weights: dict[str, mx.array]) -> mx.array:
    _require_csa_config(args)
    _require_csa_weights(weights)
    return _csa_windowed_compressor_mlx(
        x,
        wkv=weights["compressor_wkv"],
        wgate=weights["compressor_wgate"],
        ape=weights["compressor_ape"],
        norm=weights["compressor_norm"],
        compress_rate=args.compression_ratio,
        out_dim=args.head_dim,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.compress_rope_theta,
    )


def _csa_indexer_mlx(
    args: ModelArgs,
    hidden_states: mx.array,
    q_residual: mx.array,
    weights: dict[str, mx.array],
    *,
    index_topk: int | None = None,
) -> dict[str, Any]:
    _require_csa_config(args)
    _require_csa_weights(weights)
    if index_topk is None:
        # Option A: index_topk is a per-forward kwarg, defaulting from the
        # current sequence length rather than becoming ModelArgs state.
        index_topk = max(1, hidden_states.shape[1] // args.compression_ratio)
    if index_topk <= 0:
        raise ValueError("index_topk must be positive")

    # The indexer uses its own downscaled compressor at index_head_dim; it is
    # intentionally separate from the value compressor above.
    compressed = _csa_windowed_compressor_mlx(
        hidden_states,
        wkv=weights["indexer_compressor_wkv"],
        wgate=weights["indexer_compressor_wgate"],
        ape=weights["indexer_compressor_ape"],
        norm=weights["indexer_compressor_norm"],
        compress_rate=args.compression_ratio,
        out_dim=args.index_head_dim,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.compress_rope_theta,
    )
    batch, seq_len, _hidden = hidden_states.shape
    compressed_len = compressed.shape[1]
    q = _linear_in_out_mlx(q_residual, weights["indexer_wq_b"])
    expected_q_dim = args.index_n_heads * args.index_head_dim
    if q.shape[-1] != expected_q_dim:
        raise ValueError(f"indexer_wq_b output dimension mismatch: got {q.shape[-1]}, expected {expected_q_dim}")
    q = q.reshape((batch, seq_len, args.index_n_heads, args.index_head_dim))
    q_cos, q_sin = _rope_full_tables_mlx(args.index_head_dim, mx.arange(seq_len), args.compress_rope_theta)
    q = _apply_rope_full_mlx(q, q_cos.reshape((1, seq_len, 1, q_cos.shape[-1])), q_sin.reshape((1, seq_len, 1, q_sin.shape[-1])))

    scorer_weights = _linear_in_out_mlx(hidden_states, weights["indexer_proj"]) * (args.index_n_heads ** -0.5)
    dots = mx.sum(mx.expand_dims(q, 3) * mx.expand_dims(mx.expand_dims(compressed, 1), 1), axis=-1)
    # ReLU scorer: per-index-head positive dot products are weighted and summed
    # into [batch, seq, compressed_blocks] before causal/top-k pruning.
    scores = mx.sum(mx.maximum(dots * (args.index_head_dim ** -0.5), 0.0) * mx.expand_dims(scorer_weights, -1), axis=2)

    scores_py = scores.tolist()
    top_k = min(index_topk, compressed_len)
    topk_indices: list[list[list[int]]] = []
    topk_mask: list[list[list[bool]]] = []
    valid_mask: list[list[list[bool]]] = []
    for b in range(batch):
        batch_indices: list[list[int]] = []
        batch_topk_mask: list[list[bool]] = []
        batch_valid: list[list[bool]] = []
        for t in range(seq_len):
            # hca_block_bias parity: a query may attend only fully completed
            # compressed blocks, then the indexer keeps the top-k of that set.
            threshold = (t + 1) // args.compression_ratio
            candidates = [entry for entry in range(compressed_len) if entry < threshold]
            candidates.sort(key=lambda entry: float(scores_py[b][t][entry]), reverse=True)
            selected = candidates[:top_k]
            batch_indices.append(selected + [-1] * (top_k - len(selected)))
            batch_topk_mask.append([True] * len(selected) + [False] * (top_k - len(selected)))
            selected_set = set(selected)
            batch_valid.append([entry in selected_set for entry in range(compressed_len)])
        topk_indices.append(batch_indices)
        topk_mask.append(batch_topk_mask)
        valid_mask.append(batch_valid)

    return {
        "scores": scores,
        "topk_indices": topk_indices,
        "topk_mask": topk_mask,
        "valid_mask": mx.array(valid_mask),
        "compressed_len": compressed_len,
        "compressed_index_kv": compressed,
    }


def _csa_attention_mlx(
    args: ModelArgs,
    x: mx.array,
    weights: dict[str, mx.array],
    *,
    index_topk: int | None = None,
) -> mx.array:
    _require_csa_config(args)
    _require_csa_weights(weights)

    compressed_kv = _csa_compressor_mlx(args, x, weights)
    indexer = _csa_indexer_mlx(args, x, x, weights, index_topk=index_topk)
    scores = indexer["scores"]
    valid_mask = indexer["valid_mask"]
    compressed_len = int(indexer["compressed_len"])
    if compressed_len == 0:
        return mx.zeros((x.shape[0], x.shape[1], args.head_dim), dtype=x.dtype)

    masked_scores = mx.where(valid_mask, scores, -1e9)
    probs = mx.softmax(masked_scores, axis=-1)
    attended = probs @ compressed_kv
    # Softmax over an all-masked row would be uniform; zero those early queries
    # explicitly to match the pure-Python CSA reference.
    has_valid = mx.sum(valid_mask.astype(mx.float32), axis=-1, keepdims=True) > 0
    return mx.where(has_valid, attended, mx.zeros_like(attended))


# --- Real-dim CSA/HCA port (Story 13.3b; additive only; tiny path byte-identical) ---


def _compress_rope_yarn_tail_tables_mlx(
    head_dim: int,
    qk_rope_head_dim: int,
    rope_theta: float,
    yarn_factor: float,
    yarn_orig_max: int,
    seq_len: int,
    *,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    """YaRN compress-RoPE tables for the trailing qk_rope_head_dim slice."""

    _validate_qk_rope_head_dim(head_dim, qk_rope_head_dim)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if int(positions.shape[0]) != seq_len:
        raise ValueError(f"positions length mismatch: got {positions.shape[0]}, expected {seq_len}")
    if yarn_factor <= 0:
        raise ValueError("yarn_factor must be positive")
    if yarn_orig_max <= 0:
        raise ValueError("yarn_orig_max must be positive")

    dim = qk_rope_head_dim
    beta_fast = 32.0
    beta_slow = 1.0

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(yarn_orig_max / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(rope_theta))

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001

    dim_idx = mx.arange(0, dim, 2).astype(mx.float32)
    pos_freqs = rope_theta ** (dim_idx / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (float(yarn_factor) * pos_freqs)
    ramp = mx.clip((mx.arange(dim // 2).astype(mx.float32) - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    inv_freq = inv_freq_interpolation * (1.0 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor

    angles = mx.expand_dims(positions.astype(mx.float32), 1) * mx.expand_dims(inv_freq, 0)
    return mx.cos(angles), mx.sin(angles)


def _hca_compressor_mlx(
    args: ModelArgs,
    x: mx.array,
    weights: dict[str, mx.array],
    *,
    position_ids: mx.array,
) -> tuple[mx.array, mx.array | None]:
    """Stateless real-dim HCA compressor: one non-overlap window per 128 tokens."""

    rate = int(args.compression_ratio)
    if rate != 128:
        raise NotImplementedError("MLX real HCA compressor currently supports compression_ratio=128")
    out_dim = int(args.head_dim)
    required = {"compressor_wkv", "compressor_wgate", "compressor_ape", "compressor_norm"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing MLX HCA compressor weights: {', '.join(missing)}")
    if tuple(weights["compressor_wkv"].shape) != (out_dim, args.hidden_size):
        raise ValueError(f"HCA compressor_wkv shape mismatch: got {weights['compressor_wkv'].shape}, expected {(out_dim, args.hidden_size)}")
    if tuple(weights["compressor_wgate"].shape) != (out_dim, args.hidden_size):
        raise ValueError(f"HCA compressor_wgate shape mismatch: got {weights['compressor_wgate'].shape}, expected {(out_dim, args.hidden_size)}")
    if tuple(weights["compressor_ape"].shape) != (rate, out_dim):
        raise ValueError(f"HCA compressor_ape shape mismatch: got {weights['compressor_ape'].shape}, expected {(rate, out_dim)}")
    if tuple(weights["compressor_norm"].shape) != (out_dim,):
        raise ValueError(f"HCA compressor_norm shape mismatch: got {weights['compressor_norm'].shape}, expected {(out_dim,)}")

    batch, seq_len, _hidden = x.shape
    if tuple(position_ids.shape) != (batch, seq_len):
        raise ValueError(f"position_ids shape mismatch: got {position_ids.shape}, expected {(batch, seq_len)}")
    usable = (seq_len // rate) * rate
    n_windows = usable // rate
    if n_windows == 0:
        compressed_kv = mx.zeros((batch, 1, 0, out_dim), dtype=x.dtype)
        return compressed_kv, None

    kv = _linear_mlx(x[:, :usable, :], weights["compressor_wkv"])
    gate = _linear_mlx(x[:, :usable, :], weights["compressor_wgate"])
    kv = kv.reshape((batch, n_windows, rate, out_dim))
    gate = gate.reshape((batch, n_windows, rate, out_dim)) + weights["compressor_ape"].reshape((1, 1, rate, out_dim))
    gate_probs = mx.softmax(gate.astype(mx.float32), axis=2).astype(kv.dtype)
    pooled = mx.sum(kv * gate_probs, axis=2)
    compressed = _rms_norm_mlx(pooled, weights["compressor_norm"], args.rms_norm_eps)

    positions = mx.arange(n_windows) * rate
    cos, sin = _compress_rope_yarn_tail_tables_mlx(
        args.head_dim,
        args.qk_rope_head_dim,
        args.compress_rope_theta,
        float((args.rope_scaling or {}).get("factor", 16.0)),
        int((args.rope_scaling or {}).get("original_max_position_embeddings", 65536)),
        n_windows,
        positions=positions,
    )
    compressed = _apply_rope_tail_mlx(compressed, cos, sin, qk_rope_head_dim=args.qk_rope_head_dim)
    compressed_kv = mx.expand_dims(compressed, 1)
    if seq_len == 1:
        return compressed_kv, None

    entry_indices = mx.arange(n_windows).reshape((1, 1, 1, n_windows))
    causal_threshold = ((position_ids + 1) // rate).reshape((batch, 1, seq_len, 1))
    visible = entry_indices < causal_threshold
    zeros = mx.zeros((batch, 1, seq_len, n_windows), dtype=x.dtype)
    neg_inf = mx.full((batch, 1, seq_len, n_windows), -float("inf"), dtype=x.dtype)
    return compressed_kv, mx.where(visible, zeros, neg_inf)



def _attention_real_mlx(
    args: ModelArgs,
    x: mx.array,
    weights: dict[str, mx.array],
    *,
    index_topk: int | None = None,
) -> mx.array:
    """Real-dim compressed attention. Story 13.3b-1 implements the HCA branch."""

    if args.compression_ratio != 128:
        error = _csa_config_error(args)
        if error is not None:
            raise NotImplementedError(error)
        raise NotImplementedError("MLX real compressed attention currently supports HCA compression_ratio=128")
    if index_topk is not None:
        raise NotImplementedError("index_topk is CSA-only; Story 13.3b-1 implements HCA without indexer")
    if args.num_key_value_heads != 1:
        raise NotImplementedError("MLX real compressed attention currently supports only num_key_value_heads=1")
    if args.num_attention_heads % args.o_groups != 0:
        raise NotImplementedError("MLX real compressed attention requires num_attention_heads divisible by o_groups")
    if "sinks" not in weights:
        raise NotImplementedError("MLX real attention slice requires explicit per-head sink logits")
    raw_sinks = weights["sinks"]
    if tuple(raw_sinks.shape) != (args.num_attention_heads,):
        raise NotImplementedError("MLX real attention slice requires one sink logit per query head")

    q_a = _linear_mlx(x, weights["q_a_proj.weight"])
    q_a = _rms_norm_mlx(q_a, weights["q_norm.weight"], args.rms_norm_eps)
    q = _linear_mlx(q_a, weights["q_b_proj.weight"])
    q = q.reshape((*q.shape[:-1], args.num_attention_heads, args.head_dim))
    q = _rms_norm_mlx(q, mx.ones(args.head_dim), args.rms_norm_eps)
    kv = _linear_mlx(x, weights["kv_proj.weight"])
    kv = _rms_norm_mlx(kv, weights["kv_norm.weight"], args.rms_norm_eps)

    batch, seq_len, _hidden = x.shape
    rope_scaling = args.rope_scaling or {}
    cos, sin = _compress_rope_yarn_tail_tables_mlx(
        args.head_dim,
        args.qk_rope_head_dim,
        args.compress_rope_theta,
        float(rope_scaling.get("factor", 16.0)),
        int(rope_scaling.get("original_max_position_embeddings", 65536)),
        seq_len,
        positions=mx.arange(seq_len),
    )
    q = _apply_rope_tail_mlx(q, cos, sin, qk_rope_head_dim=args.qk_rope_head_dim)
    kv = _apply_rope_tail_mlx(kv, cos, sin, qk_rope_head_dim=args.qk_rope_head_dim)

    position_ids = mx.broadcast_to(mx.arange(seq_len).reshape((1, seq_len)), (batch, seq_len))
    compressed_kv, block_bias = _hca_compressor_mlx(args, x, weights, position_ids=position_ids)
    kv_by_head = mx.expand_dims(kv, 1)  # [batch, 1, seq, head_dim], broadcast over heads
    kv_full = mx.concatenate([kv_by_head, compressed_kv], axis=2)

    sliding_mask = _causal_sliding_mask_mlx(seq_len, args.sliding_window).reshape((1, 1, seq_len, seq_len))
    if compressed_kv.shape[2] > 0:
        if block_bias is None:
            block_bias = mx.zeros((batch, 1, seq_len, compressed_kv.shape[2]), dtype=x.dtype)
        mask = mx.concatenate([sliding_mask + mx.zeros((batch, 1, seq_len, seq_len), dtype=x.dtype), block_bias], axis=-1)
    else:
        mask = sliding_mask

    scale = args.head_dim ** -0.5
    q_by_head = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
    scores = (q_by_head @ kv_full.transpose(0, 1, 3, 2)) * scale
    scores = scores + mask
    sink_logits = raw_sinks.reshape((1, args.num_attention_heads, 1, 1)) + mx.zeros_like(scores[..., :1])
    scores_with_sink = mx.concatenate([scores, sink_logits], axis=-1)
    probs_with_sink = mx.softmax(scores_with_sink, axis=-1)
    probs = probs_with_sink[..., : kv_full.shape[2]]
    attended = probs @ kv_full
    attended = attended.transpose(0, 2, 1, 3)  # [batch, seq, heads, head_dim]
    attended = _apply_rope_tail_mlx(attended, cos, -sin, qk_rope_head_dim=args.qk_rope_head_dim)

    heads_per_group = args.num_attention_heads // args.o_groups
    expected_o_a_shape = (args.o_groups * args.o_lora_rank, heads_per_group * args.head_dim)
    if tuple(weights["o_a_proj.weight"].shape) != expected_o_a_shape:
        raise ValueError(f"o_a_proj.weight shape mismatch for grouped output: got {weights['o_a_proj.weight'].shape}, expected {expected_o_a_shape}")
    expected_o_b_shape = (args.hidden_size, args.o_groups * args.o_lora_rank)
    if tuple(weights["o_b_proj.weight"].shape) != expected_o_b_shape:
        raise ValueError(f"o_b_proj.weight shape mismatch for grouped output: got {weights['o_b_proj.weight'].shape}, expected {expected_o_b_shape}")

    low_rank_chunks = []
    for group in range(args.o_groups):
        head_start = group * heads_per_group
        head_end = head_start + heads_per_group
        row_start = group * args.o_lora_rank
        row_end = row_start + args.o_lora_rank
        group_heads = attended[..., head_start:head_end, :]
        flat_group = group_heads.reshape((*group_heads.shape[:-2], heads_per_group * args.head_dim))
        low_rank_chunks.append(_linear_mlx(flat_group, weights["o_a_proj.weight"][row_start:row_end, :]))
    low_rank = mx.concatenate(low_rank_chunks, axis=-1)
    return _linear_mlx(low_rank, weights["o_b_proj.weight"])



def _causal_sliding_mask_mlx(seq_len: int, sliding_window: int) -> mx.array:
    positions = mx.arange(seq_len)
    causal = mx.where(positions[:, None] >= positions[None, :], 0.0, -1e9)
    if sliding_window <= 0 or sliding_window >= seq_len:
        return causal
    return mx.where((positions[:, None] - positions[None, :]) < sliding_window, causal, -1e9)


def _hyperconnection_mlx(
    stream: mx.array,
    *,
    fn: mx.array,
    base: mx.array,
    scale: mx.array,
    hc_mult: int,
    eps: float,
    sinkhorn_iters: int,
    rms_norm_eps: float,
) -> dict[str, mx.array]:
    """Hyperconnection collapse/gates for bounded real-mode MLX arrays.

    ``stream`` has shape ``[batch, seq, hc_mult, hidden]``.  The returned
    ``comb`` matrix follows the Transformers/pure-Python convention where rows
    index old streams and columns index new streams.
    """

    if hc_mult <= 0:
        raise ValueError("hc_mult must be positive")
    if len(stream.shape) != 4 or stream.shape[-2] != hc_mult:
        raise ValueError("hyperconnection stream shape must be [batch, seq, hc_mult, hidden]")
    expected_mix = (2 + hc_mult) * hc_mult
    expected_fn_shape = (expected_mix, hc_mult * stream.shape[-1])
    if tuple(fn.shape) != expected_fn_shape:
        raise ValueError(f"hyperconnection fn shape mismatch: got {fn.shape}, expected {expected_fn_shape}")
    if tuple(base.shape) != (expected_mix,):
        raise ValueError(f"hyperconnection base shape mismatch: got {base.shape}, expected {(expected_mix,)}")
    if tuple(scale.shape) != (3,):
        raise ValueError(f"hyperconnection scale shape mismatch: got {scale.shape}, expected {(3,)}")

    flat = stream.reshape((*stream.shape[:-2], hc_mult * stream.shape[-1]))
    flat = _rms_norm_mlx(flat, mx.ones(flat.shape[-1]), rms_norm_eps)
    mixes = _linear_mlx(flat, fn)  # [batch, seq, (2 + hc_mult) * hc_mult]
    pre_w = mixes[..., :hc_mult]
    post_w = mixes[..., hc_mult: 2 * hc_mult]
    comb_w = mixes[..., 2 * hc_mult:]
    pre = mx.sigmoid(pre_w * scale[0] + base[:hc_mult]) + eps
    post = 2.0 * mx.sigmoid(post_w * scale[1] + base[hc_mult: 2 * hc_mult])
    comb_logits = comb_w * scale[2] + base[2 * hc_mult:]
    comb = comb_logits.reshape((*comb_logits.shape[:-1], hc_mult, hc_mult))
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    collapsed = mx.sum(mx.expand_dims(pre, -1) * stream, axis=-2)
    return {"collapsed": collapsed, "post": post, "comb": comb}


def _hyperhead_mlx(
    stream: mx.array,
    *,
    fn: mx.array,
    base: mx.array,
    scale: mx.array,
    hc_mult: int,
    eps: float,
    rms_norm_eps: float,
) -> mx.array:
    """Final HyperHead collapse for bounded real-mode MLX streams."""

    if hc_mult <= 1:
        raise ValueError("hyperhead collapse is only needed for hc_mult>1")
    if len(stream.shape) != 4 or stream.shape[-2] != hc_mult:
        raise ValueError("hyperhead stream shape must be [batch, seq, hc_mult, hidden]")
    hidden = int(stream.shape[-1])
    expected_fn_shape = (hc_mult, hc_mult * hidden)
    if tuple(fn.shape) != expected_fn_shape:
        raise ValueError(f"hyperhead fn shape mismatch: got {fn.shape}, expected {expected_fn_shape}")
    if tuple(base.shape) != (hc_mult,):
        raise ValueError(f"hyperhead base shape mismatch: got {base.shape}, expected {(hc_mult,)}")
    if tuple(scale.shape) != (1,):
        raise ValueError(f"hyperhead scale shape mismatch: got {scale.shape}, expected {(1,)}")

    flat = stream.reshape((*stream.shape[:-2], hc_mult * hidden))
    flat = _rms_norm_mlx(flat, mx.ones(flat.shape[-1]), rms_norm_eps)
    mixes = _linear_mlx(flat, fn)  # [batch, seq, hc_mult]
    pre = mx.sigmoid(mixes * scale[0] + base) + eps
    return mx.sum(mx.expand_dims(pre, -1) * stream, axis=-2)


def _attention_mlx(args: ModelArgs, x: mx.array, weights: dict[str, mx.array], *, index_topk: int | None = None) -> mx.array:
    """Bounded attention in MLX.

    The default path is multi-head sliding-window causal attention.  ``index_topk``
    stays a per-forward kwarg (Option A), and the CSA compressed path is
    deliberately limited to the tiny single-head, cache-less
    ``compression_ratio=4`` subset proven by Story 11.14 fixtures.
    """

    if args.compression_ratio != 0:
        if _csa_config_error(args) is None:
            return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
        return _attention_real_mlx(args, x, weights, index_topk=index_topk)
    if args.num_key_value_heads != 1:
        raise NotImplementedError("MLX attention currently supports only num_key_value_heads=1")
    if args.num_attention_heads % args.o_groups != 0:
        raise NotImplementedError("MLX attention requires num_attention_heads divisible by o_groups")
    if "sinks" not in weights:
        raise NotImplementedError("MLX attention slice requires explicit per-head sink logits")
    raw_sinks = weights["sinks"]
    if tuple(raw_sinks.shape) != (args.num_attention_heads,):
        raise NotImplementedError("MLX attention slice requires one sink logit per query head")

    q_a = _linear_mlx(x, weights["q_a_proj.weight"])
    q_a = _rms_norm_mlx(q_a, weights["q_norm.weight"], args.rms_norm_eps)
    q = _linear_mlx(q_a, weights["q_b_proj.weight"])
    q = q.reshape((*q.shape[:-1], args.num_attention_heads, args.head_dim))
    q = _rms_norm_mlx(q, mx.ones(args.head_dim), args.rms_norm_eps)
    kv = _linear_mlx(x, weights["kv_proj.weight"])
    kv = _rms_norm_mlx(kv, weights["kv_norm.weight"], args.rms_norm_eps)

    seq_len = x.shape[1]
    cos, sin = _rope_tail_tables_mlx(args.head_dim, args.qk_rope_head_dim, args.rope_theta, seq_len)
    q = _apply_rope_tail_mlx(q, cos, sin, qk_rope_head_dim=args.qk_rope_head_dim)
    kv = _apply_rope_tail_mlx(kv, cos, sin, qk_rope_head_dim=args.qk_rope_head_dim)

    scale = args.head_dim ** -0.5
    q_by_head = q.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
    kv_by_head = mx.expand_dims(kv, 1)  # [batch, 1, seq, head_dim], broadcast over heads
    scores = (q_by_head @ kv_by_head.transpose(0, 1, 3, 2)) * scale
    scores = scores + _causal_sliding_mask_mlx(seq_len, args.sliding_window)
    sink_logits = raw_sinks.reshape((1, args.num_attention_heads, 1, 1)) + mx.zeros_like(scores[..., :1])
    scores_with_sink = mx.concatenate([scores, sink_logits], axis=-1)
    probs_with_sink = mx.softmax(scores_with_sink, axis=-1)
    probs = probs_with_sink[..., :seq_len]
    attended = probs @ kv_by_head
    attended = attended.transpose(0, 2, 1, 3)  # [batch, seq, heads, head_dim]
    attended = _apply_rope_tail_mlx(attended, cos, -sin, qk_rope_head_dim=args.qk_rope_head_dim)

    heads_per_group = args.num_attention_heads // args.o_groups
    expected_o_a_shape = (args.o_groups * args.o_lora_rank, heads_per_group * args.head_dim)
    if tuple(weights["o_a_proj.weight"].shape) != expected_o_a_shape:
        raise ValueError(f"o_a_proj.weight shape mismatch for grouped output: got {weights['o_a_proj.weight'].shape}, expected {expected_o_a_shape}")
    expected_o_b_shape = (args.hidden_size, args.o_groups * args.o_lora_rank)
    if tuple(weights["o_b_proj.weight"].shape) != expected_o_b_shape:
        raise ValueError(f"o_b_proj.weight shape mismatch for grouped output: got {weights['o_b_proj.weight'].shape}, expected {expected_o_b_shape}")

    low_rank_chunks = []
    for group in range(args.o_groups):
        head_start = group * heads_per_group
        head_end = head_start + heads_per_group
        row_start = group * args.o_lora_rank
        row_end = row_start + args.o_lora_rank
        group_heads = attended[..., head_start:head_end, :]
        flat_group = group_heads.reshape((*group_heads.shape[:-2], heads_per_group * args.head_dim))
        low_rank_chunks.append(_linear_mlx(flat_group, weights["o_a_proj.weight"][row_start:row_end, :]))
    low_rank = mx.concatenate(low_rank_chunks, axis=-1)
    return _linear_mlx(low_rank, weights["o_b_proj.weight"])


def _moe_mlx(args: ModelArgs, x: mx.array, weights: dict[str, mx.array]) -> mx.array:
    """Bounded synthetic top-k MoE in MLX.

    Supports small unquantized fixture weights, the existing synthetic I8
    block-scale path, and FP4 packed routed experts when per-projection scale
    metadata is present. Legacy raw fixtures with expert_dtype='fp4' and no
    scale keys stay on the raw-weight path; real checkpoint loads are guarded
    by _fp4_has_scales before this helper runs.
    """

    if args.scoring_func != "sqrtsoftplus":
        raise NotImplementedError("MLX MoE currently supports only scoring_func='sqrtsoftplus'")
    if args.num_experts_per_tok > args.n_routed_experts:
        raise NotImplementedError("MLX MoE num_experts_per_tok cannot exceed n_routed_experts")

    router = weights["mlp.gate.weight"]
    correction_bias = weights.get("mlp.gate.e_score_correction_bias")
    logits = _linear_mlx(x, router)
    scores = mx.sqrt(_softplus_mlx(logits))
    selection_scores = scores if correction_bias is None else scores + correction_bias
    top_idx = mx.argsort(-selection_scores, axis=-1)[..., : args.num_experts_per_tok]

    denom = mx.zeros_like(scores[..., 0])
    for eid in range(args.n_routed_experts):
        selected = mx.any(top_idx == eid, axis=-1).astype(scores.dtype)
        denom = denom + scores[..., eid] * selected

    routed = mx.zeros_like(x)
    expert_dtype = str(args.expert_dtype).lower()
    for eid in range(args.n_routed_experts):
        if expert_dtype == "i8":
            w1 = _dequantize_i8_block_scale_mlx(weights[f"mlp.experts.{eid}.w1.weight"], weights[f"mlp.experts.{eid}.w1.scale"])
            w2 = _dequantize_i8_block_scale_mlx(weights[f"mlp.experts.{eid}.w2.weight"], weights[f"mlp.experts.{eid}.w2.scale"])
            w3 = _dequantize_i8_block_scale_mlx(weights[f"mlp.experts.{eid}.w3.weight"], weights[f"mlp.experts.{eid}.w3.scale"])
        elif expert_dtype == "fp4":
            scale_keys = [
                f"mlp.experts.{eid}.w1.scale",
                f"mlp.experts.{eid}.w2.scale",
                f"mlp.experts.{eid}.w3.scale",
            ]
            if all(key in weights for key in scale_keys):
                w1 = _dequantize_fp4_block_scale_mlx(weights[f"mlp.experts.{eid}.w1.weight"], weights[f"mlp.experts.{eid}.w1.scale"])
                w2 = _dequantize_fp4_block_scale_mlx(weights[f"mlp.experts.{eid}.w2.weight"], weights[f"mlp.experts.{eid}.w2.scale"])
                w3 = _dequantize_fp4_block_scale_mlx(weights[f"mlp.experts.{eid}.w3.weight"], weights[f"mlp.experts.{eid}.w3.scale"])
            elif any(key in weights for key in scale_keys):
                missing = [key for key in scale_keys if key not in weights]
                raise KeyError(f"missing FP4 expert scale keys: {missing}")
            else:
                # Legacy synthetic fixtures use expert_dtype='fp4' with raw small
                # matrices and no scale metadata. Real checkpoint loads are still
                # guarded by _fp4_has_scales before reaching this forward path.
                w1 = weights[f"mlp.experts.{eid}.w1.weight"]
                w2 = weights[f"mlp.experts.{eid}.w2.weight"]
                w3 = weights[f"mlp.experts.{eid}.w3.weight"]
        else:
            w1 = weights[f"mlp.experts.{eid}.w1.weight"]
            w2 = weights[f"mlp.experts.{eid}.w2.weight"]
            w3 = weights[f"mlp.experts.{eid}.w3.weight"]
        gate = mx.clip(_linear_mlx(x, w1), None, args.swiglu_limit)
        up = mx.clip(_linear_mlx(x, w3), -args.swiglu_limit, args.swiglu_limit)
        expert_out = _linear_mlx(mx.sigmoid(gate) * gate * up, w2)
        selected = mx.any(top_idx == eid, axis=-1).astype(x.dtype)
        factor = (scores[..., eid] / (denom + 1e-20)) * selected * args.routed_scaling_factor
        routed = routed + expert_out * mx.expand_dims(factor, -1)

    sw1 = weights["mlp.shared_experts.w1.weight"]
    sw2 = weights["mlp.shared_experts.w2.weight"]
    sw3 = weights["mlp.shared_experts.w3.weight"]
    shared_gate = mx.clip(_linear_mlx(x, sw1), None, args.swiglu_limit)
    shared_up = mx.clip(_linear_mlx(x, sw3), -args.swiglu_limit, args.swiglu_limit)
    shared_out = _linear_mlx(mx.sigmoid(shared_gate) * shared_gate * shared_up, sw2)
    return routed + shared_out


def run_tiny_csa_attention_mlx_fixture() -> dict[str, Any]:
    """Story 11.14 MLX CSA compressed-attention parity fixture."""

    if mx is None:
        return {
            "fixture": "csa-compressor-indexer-attention-mlx",
            "status": "skipped",
            "max_abs_error": None,
            "reason": "mlx is not installed",
        }

    args = ModelArgs.from_dict({
        "model_type": "deepseek_v4",
        "vocab_size": 8,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "index_head_dim": 2,
        "index_n_heads": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "expert_dtype": "fp4",
        "rms_norm_eps": 1e-6,
        "hc_mult": 1,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "rope_theta": 10000.0,
        "o_groups": 1,
        "compression_ratio": 4,
    })

    def dense(rows: int, cols: int, scale: float) -> list[list[float]]:
        return [
            [round(scale * (i + j + 1) * (1.0 if (i + j) % 2 == 0 else -1.0), 4) for j in range(cols)]
            for i in range(rows)
        ]

    weights = {
        "compressor_wkv": dense(4, 8, 0.5),
        "compressor_wgate": dense(4, 8, 0.25),
        "compressor_ape": dense(8, 4, 0.05),
        "compressor_norm": [1.0, 1.0, 1.0, 1.0],
        "indexer_wq_b": dense(4, 4, 0.5),
        "indexer_proj": dense(4, 2, 0.3),
        "indexer_compressor_wkv": dense(4, 4, 0.4),
        "indexer_compressor_wgate": dense(4, 4, 0.2),
        "indexer_compressor_ape": dense(4, 4, 0.05),
        "indexer_compressor_norm": [1.0, 1.0],
    }
    hidden = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]
    spec = DeepSeekV4AttentionSpec(
        args.hidden_size,
        args.num_attention_heads,
        args.head_dim,
        args.q_lora_rank,
        args.o_lora_rank,
        args.qk_rope_head_dim,
        args.o_groups,
        args.compression_ratio,
        args.index_n_heads,
        args.index_head_dim,
    )
    expected = tiny_compressor_indexer_attention_reference(
        spec,
        hidden,
        hidden,
        weights,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.compress_rope_theta,
        index_topk=2,
    )["attended"]
    got = _csa_attention_mlx(args, mx.array([hidden]), {key: mx.array(value) for key, value in weights.items()}, index_topk=2).tolist()[0]
    max_abs_error = 0.0
    for got_row, exp_row in zip(got, expected):
        for gv, ev in zip(got_row, exp_row):
            max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
    return {
        "fixture": "csa-compressor-indexer-attention-mlx",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "MLX CSA compressor softmax-gated KV compression",
            "MLX Lightning indexer scoring and top-k mask",
            "MLX compressed attention value mixing",
            "all-masked early-query zero output",
        ],
        "not_covered": [
            "stateful cache overlap",
            "multi-head compressed attention",
            "real DeepSeek V4 Flash checkpoint compatibility",
            "full generation path",
        ],
    }


def run_embedding_rmsnorm_head_fixture() -> dict[str, Any]:
    args = ModelArgs.from_dict({
        "model_type": "deepseek_v4",
        "vocab_size": 3,
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "q_lora_rank": 2,
        "qk_rope_head_dim": 1,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "expert_dtype": "fp4",
        "forward_parity_fixture": "embedding-rmsnorm-head",
        "rms_norm_eps": 0.0,
    })
    model = Model(args)
    model.load_tiny_weights({
        "embed.weight": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "norm.weight": [1.0, 0.5],
        "lm_head.weight": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    })
    got = model([[0, 1]])
    expected = [[
        [1.0 / math.sqrt(2.5), 1.0 / math.sqrt(2.5), 2.0 / math.sqrt(2.5)],
        [3.0 / math.sqrt(12.5), 2.0 / math.sqrt(12.5), 5.0 / math.sqrt(12.5)],
    ]]
    max_abs_error = 0.0
    for got_seq, exp_seq in zip(got, expected):
        for got_row, exp_row in zip(got_seq, exp_seq):
            for got_value, exp_value in zip(got_row, exp_row):
                max_abs_error = max(max_abs_error, abs(float(got_value) - float(exp_value)))
    return {
        "fixture": "embedding-rmsnorm-head",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["embed.weight", "norm.weight", "lm_head.weight", "RMSNorm", "linear lm_head"],
        "not_covered": list(forward_parity_blockers()),
    }


def _make_rope_cos_sin(*, position_ids: list[int], rope_dim: int, rope_theta: float) -> dict[int, tuple[list[float], list[float]]]:
    """Build full-length interleaved cos/sin tables for the given positions."""

    half = rope_dim // 2
    inv_freq = [1.0 / (rope_theta ** (i / rope_dim)) for i in range(0, rope_dim, 2)]
    out: dict[int, tuple[list[float], list[float]]] = {}
    for pos in position_ids:
        half_cos = [math.cos(pos * f) for f in inv_freq]
        half_sin = [math.sin(pos * f) for f in inv_freq]
        cos = [v for pair in zip(half_cos, half_cos) for v in pair]
        sin = [v for pair in zip(half_sin, half_sin) for v in pair]
        out[pos] = (cos, sin)
    return out


def _integrated_attention_forward(
    args: ModelArgs,
    hidden_states: list[list[float]],
    weights: dict[str, Any],
    position_ids: list[int],
    *,
    rms_norm_eps: float,
) -> list[list[float]]:
    """Deterministic pure-Python sliding attention for the integrated layer fixture.

    Covers q_a/q_b/kv projections, RMSNorms, partial interleaved RoPE on the
    trailing rope slice, causal scaled dot-product attention with a learnable
    sink, output inverse RoPE, and grouped low-rank output projection.  For
    ``compression_ratio=4`` it delegates to the cache-less CSA reference; stateful
    Ca carry across forward calls remains deferred to Story 11.16.
    """

    if args.compression_ratio != 0:
        _require_csa_config(args)
        spec = DeepSeekV4AttentionSpec(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            head_dim=args.head_dim,
            q_lora_rank=args.q_lora_rank,
            o_lora_rank=args.o_lora_rank,
            qk_rope_head_dim=args.qk_rope_head_dim,
            num_output_groups=args.o_groups,
            compression_ratio=args.compression_ratio,
            index_n_heads=args.index_n_heads,
            index_head_dim=args.index_head_dim,
        )
        return tiny_compressor_indexer_attention_reference(
            spec,
            hidden_states,
            hidden_states,
            weights,
            rms_norm_eps=rms_norm_eps,
            rope_theta=args.compress_rope_theta,
        )["attended"]
    if args.num_key_value_heads != 1:
        raise NotImplementedError("integrated layer fixture currently supports only num_key_value_heads=1")
    if args.num_attention_heads % args.o_groups != 0:
        raise NotImplementedError("integrated layer fixture requires num_attention_heads divisible by o_groups")
    if position_ids != list(range(len(hidden_states))):
        raise NotImplementedError("integrated layer fixture currently supports contiguous position_ids starting at zero")

    spec = DeepSeekV4AttentionSpec(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
        q_lora_rank=args.q_lora_rank,
        o_lora_rank=args.o_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        num_output_groups=args.o_groups,
        compression_ratio=args.compression_ratio,
    )
    return tiny_multihead_grouped_attention_reference(
        spec,
        hidden_states,
        weights,
        rms_norm_eps=rms_norm_eps,
        rope_theta=args.rope_theta,
        sliding_window=args.sliding_window,
    )


def _integrated_moe_forward(
    args: ModelArgs,
    hidden_states: list[list[float]],
    weights: dict[str, Any],
) -> list[list[float]]:
    """Deterministic pure-Python top-k MoE for the integrated layer fixture."""

    config = MoEConfig(
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        n_routed_experts=args.n_routed_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        n_shared_experts=args.n_shared_experts,
    )
    moe_weights = {
        "router.weight": weights["mlp.gate.weight"],
        "router.e_score_correction_bias": weights.get("mlp.gate.e_score_correction_bias", [0.0] * args.n_routed_experts),
    }
    for expert_id in range(args.n_routed_experts):
        for proj in ("w1", "w2", "w3"):
            moe_weights[f"experts.{expert_id}.{proj}"] = weights[f"mlp.experts.{expert_id}.{proj}.weight"]
    for proj in ("w1", "w2", "w3"):
        moe_weights[f"shared.{proj}"] = weights[f"mlp.shared_experts.{proj}.weight"]
    return tiny_topk_moe_forward(
        config,
        hidden_states,
        moe_weights,
        scoring_func=args.scoring_func,
        routed_scaling_factor=args.routed_scaling_factor,
        swiglu_limit=args.swiglu_limit,
    )


def _integrated_layer_forward(
    args: ModelArgs,
    input_ids: list[list[int]],
    weights: dict[str, Any],
) -> list[list[list[list[float]]]]:
    """Run one deterministic DeepSeek V4 decoder layer (attention + MoE).

    Supports hc_mult>=1 hyperconnection residual mixing and the cache-less
    tiny CSA compressor subset. Caches/stateful CSA carry and hash routing are
    not supported.
    """

    if args.layer_types is None or args.layer_types[0] != "sliding_attention" or any(t != "sliding_attention" for t in args.layer_types):
        raise NotImplementedError("integrated layer fixture requires all layer_types='sliding_attention'")
    if args.mlp_layer_types is None or any(t != "moe" for t in args.mlp_layer_types):
        raise NotImplementedError("integrated layer fixture requires all mlp_layer_types='moe'")

    embeddings = weights["embed.weight"]
    batch = [row for row in input_ids]
    seq_len = len(batch[0]) if batch else 0
    position_ids = list(range(seq_len))
    hc = args.hc_mult

    # Build stream state [B, S, hc_mult, H].  All streams start as the same embedding.
    streams: list[list[list[list[float]]]] = []
    for row in batch:
        token_streams: list[list[list[float]]] = []
        for token_id in row:
            token_streams.append([[float(v) for v in embeddings[token_id]] for _ in range(hc)])
        streams.append(token_streams)

    ffn_outputs = _integrated_layer_residual_streams(args, streams, weights, position_ids)

    # Preserve the old [B, S, H] return shape for hc_mult=1 callers.
    if hc == 1:
        return [[stream[0] for stream in batch] for batch in ffn_outputs]
    return ffn_outputs


def _integrated_layer_residual_streams(
    args: ModelArgs,
    streams: list[list[list[list[float]]]],
    weights: dict[str, Any],
    position_ids: list[int],
) -> list[list[list[list[float]]]]:
    """Run one decoder layer (attention + MoE residual) over stream state.

    ``streams`` is [B, S, hc_mult, H]; ``weights`` holds the flat per-layer
    keys (attn_hc.*, input_layernorm.weight, q_a_proj.weight, ...).
    """

    hc = args.hc_mult

    def _hyperconnection_residual(
        token_streams: list[list[list[float]]],
        *,
        fn: list[list[float]],
        base: list[float],
        scale: list[float],
        norm_weight: list[float],
        sublayer: Callable[[list[list[float]]], list[list[float]]],
    ) -> list[list[list[float]]]:
        hc_out = tiny_hyperconnection_forward(
            hidden_streams=token_streams,
            fn=fn,
            base=base,
            scale=scale,
            hc_mult=hc,
            eps=args.hc_eps,
            sinkhorn_iters=args.hc_sinkhorn_iters,
            rms_norm_eps=args.rms_norm_eps,
        )
        collapsed = hc_out["collapsed"]
        norm_collapsed = [_rms_norm(token, norm_weight, args.rms_norm_eps) for token in collapsed]
        sub_out = sublayer(norm_collapsed)
        new_streams: list[list[list[float]]] = []
        for post, comb, old_streams, attn in zip(hc_out["post"], hc_out["comb"], token_streams, sub_out):
            token_new: list[list[float]] = []
            for k in range(hc):
                stream = [post[k] * av for av in attn]
                for j in range(hc):
                    weight = comb[j][k]
                    old_stream = old_streams[j]
                    for d in range(len(stream)):
                        stream[d] += weight * old_stream[d]
                token_new.append(stream)
            new_streams.append(token_new)
        return new_streams

    attn_outputs: list[list[list[list[float]]]] = []
    for token_streams in streams:
        attn_outputs.append(
            _hyperconnection_residual(
                token_streams,
                fn=weights["attn_hc.fn"],
                base=weights["attn_hc.base"],
                scale=weights["attn_hc.scale"],
                norm_weight=weights["input_layernorm.weight"],
                sublayer=lambda norm_collapsed: _integrated_attention_forward(
                    args, norm_collapsed, weights, position_ids, rms_norm_eps=args.rms_norm_eps
                ),
            )
        )

    ffn_outputs: list[list[list[list[float]]]] = []
    for token_streams in attn_outputs:
        ffn_outputs.append(
            _hyperconnection_residual(
                token_streams,
                fn=weights["ffn_hc.fn"],
                base=weights["ffn_hc.base"],
                scale=weights["ffn_hc.scale"],
                norm_weight=weights["post_attention_layernorm.weight"],
                sublayer=lambda norm_collapsed: _integrated_moe_forward(args, norm_collapsed, weights),
            )
        )
    return ffn_outputs


def _integrated_multilayer_forward(
    args: ModelArgs,
    input_ids: list[list[int]],
    weights: dict[str, Any],
) -> list[list[list[float]]]:
    """Pure-Python stacked multi-layer reference for synthetic DeepSeek V4.

    Embeds once, then runs each decoder layer in sequence where layer N+1
    consumes layer N's stream output.  Final optional norm/lm_head is applied
    once after the last layer.  Per-layer weights are keyed ``layers.{i}.*``.
    Only hc_mult=1 multi-layer parity is proven here.
    """

    if args.hc_mult != 1:
        raise NotImplementedError("integrated multi-layer reference currently supports only hc_mult=1")
    if args.layer_types is None or len(args.layer_types) != args.num_hidden_layers or any(t != "sliding_attention" for t in args.layer_types):
        raise NotImplementedError("integrated multi-layer reference requires all layer_types='sliding_attention' with length num_hidden_layers")
    if args.mlp_layer_types is None or len(args.mlp_layer_types) != args.num_hidden_layers or any(t != "moe" for t in args.mlp_layer_types):
        raise NotImplementedError("integrated multi-layer reference requires all mlp_layer_types='moe' with length num_hidden_layers")

    embeddings = weights["embed.weight"]
    batch = [row for row in input_ids]
    seq_len = len(batch[0]) if batch else 0
    position_ids = list(range(seq_len))
    hc = args.hc_mult

    streams: list[list[list[list[float]]]] = []
    for row in batch:
        token_streams: list[list[list[float]]] = []
        for token_id in row:
            token_streams.append([[float(v) for v in embeddings[token_id]] for _ in range(hc)])
        streams.append(token_streams)

    for i in range(args.num_hidden_layers):
        prefix = f"layers.{i}."
        layer_weights = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        streams = _integrated_layer_residual_streams(args, streams, layer_weights, position_ids)

    hidden = [[stream[0] for stream in token_batch] for token_batch in streams]
    if "norm.weight" in weights:
        norm_weight = weights["norm.weight"]
        hidden = [[_rms_norm(token, norm_weight, args.rms_norm_eps) for token in token_batch] for token_batch in hidden]
    if "lm_head.weight" in weights:
        lm_head = weights["lm_head.weight"]
        hidden = [
            [
                [sum(float(token[d]) * float(row[d]) for d in range(len(token))) for row in lm_head]
                for token in token_batch
            ]
            for token_batch in hidden
        ]
    return hidden


def set_transformers_integrated_weights(model: Any, weights: dict[str, Any]) -> None:
    """Copy deterministic tiny weights into a Transformers DeepseekV4Model.

    Also replaces the grouped expert forward with a tiny-tensor-safe loop.
    """

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.tensor(weights["embed.weight"], dtype=torch.float32))
        layer = model.layers[0]
        layer.input_layernorm.weight.copy_(torch.tensor(weights["input_layernorm.weight"], dtype=torch.float32))
        layer.post_attention_layernorm.weight.copy_(torch.tensor(weights["post_attention_layernorm.weight"], dtype=torch.float32))
        layer.self_attn.q_a_proj.weight.copy_(torch.tensor(weights["q_a_proj.weight"], dtype=torch.float32))
        layer.self_attn.q_a_norm.weight.copy_(torch.tensor(weights["q_norm.weight"], dtype=torch.float32))
        layer.self_attn.q_b_proj.weight.copy_(torch.tensor(weights["q_b_proj.weight"], dtype=torch.float32))
        layer.self_attn.kv_proj.weight.copy_(torch.tensor(weights["kv_proj.weight"], dtype=torch.float32))
        layer.self_attn.kv_norm.weight.copy_(torch.tensor(weights["kv_norm.weight"], dtype=torch.float32))
        layer.self_attn.o_a_proj.weight.copy_(torch.tensor(weights["o_a_proj.weight"], dtype=torch.float32))
        layer.self_attn.o_b_proj.weight.copy_(torch.tensor(weights["o_b_proj.weight"], dtype=torch.float32))
        layer.self_attn.sinks.copy_(torch.tensor(weights["sinks"], dtype=torch.float32))
        layer.attn_hc.fn.copy_(torch.tensor(weights["attn_hc.fn"], dtype=torch.float32))
        layer.attn_hc.base.copy_(torch.tensor(weights["attn_hc.base"], dtype=torch.float32))
        layer.attn_hc.scale.copy_(torch.tensor(weights["attn_hc.scale"], dtype=torch.float32))
        layer.ffn_hc.fn.copy_(torch.tensor(weights["ffn_hc.fn"], dtype=torch.float32))
        layer.ffn_hc.base.copy_(torch.tensor(weights["ffn_hc.base"], dtype=torch.float32))
        layer.ffn_hc.scale.copy_(torch.tensor(weights["ffn_hc.scale"], dtype=torch.float32))
        layer.mlp.gate.weight.copy_(torch.tensor(weights["mlp.gate.weight"], dtype=torch.float32))
        layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor(weights["mlp.gate.e_score_correction_bias"], dtype=torch.float32))
        for eid in range(2):
            w1 = torch.tensor(weights[f"mlp.experts.{eid}.w1.weight"], dtype=torch.float32)
            w3 = torch.tensor(weights[f"mlp.experts.{eid}.w3.weight"], dtype=torch.float32)
            layer.mlp.experts.gate_up_proj[eid].copy_(torch.cat([w1, w3], dim=0))
            layer.mlp.experts.down_proj[eid].copy_(torch.tensor(weights[f"mlp.experts.{eid}.w2.weight"], dtype=torch.float32))
        layer.mlp.shared_experts.gate_proj.weight.copy_(torch.tensor(weights["mlp.shared_experts.w1.weight"], dtype=torch.float32))
        layer.mlp.shared_experts.up_proj.weight.copy_(torch.tensor(weights["mlp.shared_experts.w3.weight"], dtype=torch.float32))
        layer.mlp.shared_experts.down_proj.weight.copy_(torch.tensor(weights["mlp.shared_experts.w2.weight"], dtype=torch.float32))

        if "hc_head.fn" in weights:
            model.hc_head.hc_fn.copy_(torch.tensor(weights["hc_head.fn"], dtype=torch.float32))
            model.hc_head.hc_base.copy_(torch.tensor(weights["hc_head.base"], dtype=torch.float32))
            model.hc_head.hc_scale.copy_(torch.tensor(weights["hc_head.scale"], dtype=torch.float32))

        def _simple_experts_forward(experts: Any, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
            final = torch.zeros_like(hidden_states)
            for token_idx in range(hidden_states.shape[0]):
                acc = torch.zeros(hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)
                for k in range(top_k_index.shape[1]):
                    eid = int(top_k_index[token_idx, k].item())
                    weight = top_k_weights[token_idx, k]
                    gate_up = F.linear(hidden_states[token_idx], experts.gate_up_proj[eid])
                    gate, up = gate_up.chunk(2, dim=-1)
                    gate = gate.clamp(max=experts.limit)
                    up = up.clamp(min=-experts.limit, max=experts.limit)
                    hidden = experts.act_fn(gate) * up
                    out = F.linear(hidden, experts.down_proj[eid]) * weight
                    acc += out
                final[token_idx] = acc
            return final

        layer.mlp.experts.forward = lambda hidden_states, top_k_index, top_k_weights: _simple_experts_forward(
            layer.mlp.experts, hidden_states, top_k_index, top_k_weights
        )


def run_integrated_layer_fixture(
    reference_output: list[list[list[float]]] | None = None,
    *,
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic integrated layer parity fixture.

    Builds a tiny hc_mult=1 DeepSeek V4 config, runs one attention+MoE layer
    with deterministic weights, and optionally compares against a trusted
    Transformers reference.  This is partial evidence only: compressors,
    caches, multi-stream hyperconnections, and packed expert dequant remain
    fail-closed.
    """

    args = ModelArgs.from_dict({
        "model_type": "deepseek_v4",
        "vocab_size": 4,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "n_shared_experts": 1,
        "expert_dtype": "fp4",
        "forward_parity_fixture": "integrated-layer",
        "rms_norm_eps": 1e-6,
        "hc_mult": 1,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "o_groups": 1,
    })
    model = Model(args)
    use_weights = _make_integrated_tiny_weights() if weights is None else weights
    model.load_integrated_weights(use_weights)
    got = model([[0, 1]])
    if reference_output is None:
        raise ValueError("run_integrated_layer_fixture requires a trusted reference_output to record parity evidence")
    expected_batches = 1
    expected_seq = 2
    expected_hidden = args.hidden_size
    if (
        len(reference_output) != expected_batches
        or len(reference_output[0]) != expected_seq
        or any(len(row) != expected_hidden for row in reference_output[0])
    ):
        raise ValueError(
            f"reference_output must have shape [{expected_batches}][{expected_seq}][{expected_hidden}], "
            f"got {len(reference_output)} batches, first batch {len(reference_output[0])} rows"
        )
    max_abs_error = 0.0
    for got_seq, ref_seq in zip(got, reference_output):
        for got_row, ref_row in zip(got_seq, ref_seq):
            for gv, rv in zip(got_row, ref_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))
    return {
        "fixture": "integrated-layer-attention-moe",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "single-layer attention + MoE compared against Transformers reference",
            "q_a/q_b/kv projections and norms",
            "partial interleaved RoPE and inverse RoPE",
            "causal attention with disabled sink",
            "grouped output projection",
            "separate attn_hc and ffn_hc residual gates (hc_mult=1)",
            "top-k routed MoE with shared expert",
        ],
        "not_covered": [
            "compressors (CSA/HCA)",
            "cache/sliding-window state",
            "learnable attention sink parity",
            "hc_mult>1 stream mixing",
            "packed FP4/I8 expert dequant",
            "full model load/forward",
        ],
    }


def run_integrated_layer_hc_mult_fixture(
    reference_output: list[list[list[list[float]]]] | None = None,
    *,
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic hc_mult=2 integrated layer parity fixture.

    Builds a tiny DeepSeek V4 config with hc_mult=2, runs one attention+MoE
    layer with deterministic weights, and compares against a trusted
    Transformers reference.  This proves full decoder-layer hyperconnection
    residual mixing; final hyperhead collapse is tested separately.
    """

    args = ModelArgs.from_dict({
        "model_type": "deepseek_v4",
        "vocab_size": 4,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "n_shared_experts": 1,
        "expert_dtype": "fp4",
        "forward_parity_fixture": "integrated-layer",
        "rms_norm_eps": 1e-6,
        "hc_mult": 2,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "o_groups": 1,
    })
    model = Model(args)
    use_weights = _make_integrated_tiny_weights(hc_mult=2, nonzero_hc=True) if weights is None else weights
    model.load_integrated_weights(use_weights)
    got = model([[0, 1]])
    if reference_output is None:
        try:
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
        except Exception as exc:
            return {
                "fixture": "integrated-layer-hc-mult",
                "status": "skipped",
                "max_abs_error": None,
                "reason": f"Transformers/torch reference unavailable: {exc}",
            }
        config = DeepseekV4Config(
            vocab_size=4,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            num_experts_per_tok=1,
            n_routed_experts=2,
            moe_intermediate_size=2,
            n_shared_experts=1,
            expert_dtype="fp4",
            rms_norm_eps=1e-6,
            hc_mult=2,
            hc_eps=1e-6,
            hc_sinkhorn_iters=1,
            layer_types=["sliding_attention"],
            mlp_layer_types=["moe"],
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
            rope_theta=10000.0,
            sliding_window=128,
            o_groups=1,
            rope_parameters={
                "main": {"rope_type": "default", "rope_theta": 10000.0},
                "compress": {"rope_type": "default", "rope_theta": 160000.0},
            },
        )
        ref_model = DeepseekV4Model(config)
        ref_model.eval()
        set_transformers_integrated_weights(ref_model, use_weights)
        ref_model.hc_head = torch.nn.Identity()  # type: ignore[attr-defined]
        ref_model.norm = torch.nn.Identity()  # type: ignore[attr-defined]
        with torch.no_grad():
            ref_out = ref_model(torch.tensor([[0, 1]], dtype=torch.long)).last_hidden_state
        reference_output = ref_out.tolist()
    expected_batches = 1
    expected_seq = 2
    expected_hc = 2
    expected_hidden = args.hidden_size
    if (
        len(reference_output) != expected_batches
        or len(reference_output[0]) != expected_seq
        or any(len(token) != expected_hc for token in reference_output[0])
        or any(len(stream) != expected_hidden for token in reference_output[0] for stream in token)
    ):
        raise ValueError(
            f"reference_output must have shape [{expected_batches}][{expected_seq}][{expected_hc}][{expected_hidden}]"
        )
    max_abs_error = 0.0
    for got_batch, ref_batch in zip(got, reference_output):
        for got_token, ref_token in zip(got_batch, ref_batch):
            for got_stream, ref_stream in zip(got_token, ref_token):
                for gv, rv in zip(got_stream, ref_stream):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))
    return {
        "fixture": "integrated-layer-hc-mult",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "single-layer attention + MoE with hc_mult=2",
            "full decoder-layer hyperconnection residual mixing",
            "attn_hc and ffn_hc stream collapse/comb/post",
            "Sinkhorn-projected comb matrix",
            "q_a/q_b/kv projections and norms",
            "top-k routed MoE with shared expert",
        ],
        "not_covered": [
            "compressors (CSA/HCA)",
            "cache/sliding-window state",
            "learnable attention sink parity",
            "final hyperhead collapse",
            "packed FP4/I8 expert dequant",
            "full model load/forward",
        ],
    }


def _make_integrated_tiny_weights(*, nonzero_hc: bool = False, hc_mult: int = 1) -> dict[str, Any]:
    """Return deterministic tiny weights for the integrated layer fixture."""

    hidden = 4
    intermediate = 2
    identity4 = [[1.0 if i == j else 0.0 for j in range(hidden)] for i in range(hidden)]
    if hc_mult == 1:
        if nonzero_hc:
            # Distinct nonzero attn_hc and ffn_hc weights exercise separate residual gates.
            attn_fn = [[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            attn_base = [0.1, -0.1, 0.0]
            attn_scale = [0.25, -0.25, 0.0]
            ffn_fn = [[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]]
            ffn_base = [-0.1, 0.1, 0.0]
            ffn_scale = [-0.25, 0.25, 0.0]
        else:
            attn_fn = ffn_fn = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            attn_base = ffn_base = [0.0, 0.0, 0.0]
            attn_scale = ffn_scale = [0.0, 0.0, 0.0]
        hc_head_fn = hc_head_base = hc_head_scale = None
    else:
        fn_rows = (2 + hc_mult) * hc_mult
        fn_dim = hc_mult * hidden
        if nonzero_hc:
            # Non-identity, distinct hypernetwork weights break comb symmetry and
            # produce unequal post gates, strengthening the hc_mult>1 residual
            # mixing regression test.
            def _cyclic_shift(rows: int, cols: int, shift: int) -> list[list[float]]:
                return [[1.0 if (j - i) % cols == shift else 0.0 for j in range(cols)] for i in range(rows)]

            attn_fn = _cyclic_shift(fn_rows, fn_dim, shift=1)
            attn_base = [0.1 + 0.05 * (i % 3) for i in range(fn_rows)]
            attn_scale = [0.25, -0.25, 0.15]
            ffn_fn = _cyclic_shift(fn_rows, fn_dim, shift=2)
            ffn_base = [-0.1 + 0.05 * (i % 3) for i in range(fn_rows)]
            ffn_scale = [-0.2, 0.2, -0.1]
            hc_head_fn = _cyclic_shift(hc_mult, fn_dim, shift=1)
            hc_head_base = [0.05 if i % 2 == 0 else -0.05 for i in range(hc_mult)]
            hc_head_scale = [0.3]
        else:
            # Identity hypernetwork rows make the mix logits equal to the flattened
            # stream coordinates, which is deterministic and easy to reproduce.
            attn_fn = [[1.0 if i == j else 0.0 for j in range(fn_dim)] for i in range(fn_rows)]
            attn_base = [0.0] * fn_rows
            attn_scale = [0.0, 0.0, 0.0]
            ffn_fn = [[1.0 if i == j else 0.0 for j in range(fn_dim)] for i in range(fn_rows)]
            ffn_base = [0.0] * fn_rows
            ffn_scale = [0.0, 0.0, 0.0]
            hc_head_fn = [[1.0 if i == j else 0.0 for j in range(fn_dim)] for i in range(hc_mult)]
            hc_head_base = [0.0] * hc_mult
            hc_head_scale = [0.0]
    result = {
        "embed.weight": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]],
        "input_layernorm.weight": [1.0, 1.0, 1.0, 1.0],
        "post_attention_layernorm.weight": [1.0, 1.0, 1.0, 1.0],
        "q_a_proj.weight": identity4,
        "q_norm.weight": [1.0, 1.0, 1.0, 1.0],
        "q_b_proj.weight": identity4,
        "kv_proj.weight": identity4,
        "kv_norm.weight": [1.0, 1.0, 1.0, 1.0],
        "o_a_proj.weight": identity4,
        "o_b_proj.weight": identity4,
        "sinks": [-1e9],
        "attn_hc.fn": attn_fn,
        "attn_hc.base": attn_base,
        "attn_hc.scale": attn_scale,
        "ffn_hc.fn": ffn_fn,
        "ffn_hc.base": ffn_base,
        "ffn_hc.scale": ffn_scale,
        "mlp.gate.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "mlp.gate.e_score_correction_bias": [0.0, 0.0],
        "mlp.experts.0.w1.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "mlp.experts.0.w2.weight": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        "mlp.experts.0.w3.weight": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "mlp.experts.1.w1.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "mlp.experts.1.w2.weight": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        "mlp.experts.1.w3.weight": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "mlp.shared_experts.w1.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "mlp.shared_experts.w2.weight": [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        "mlp.shared_experts.w3.weight": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    }
    if hc_head_fn is not None:
        result["hc_head.fn"] = hc_head_fn
        result["hc_head.base"] = hc_head_base
        result["hc_head.scale"] = hc_head_scale
    return result


def sanitize_weights(weights: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strip DeepSeek V4 MTP tensors from a weight dict.

    This mirrors Transformers' `_keys_to_ignore_on_load_unexpected` for
    `mtp.*` and MLX-LM's precedent of sanitizing multi-token prediction layers
    before strict model loading.  It does not mutate the input mapping.
    """

    sanitized: dict[str, Any] = {}
    removed: list[str] = []
    for name, value in weights.items():
        if str(name).startswith("mtp."):
            removed.append(str(name))
        else:
            sanitized[str(name)] = value
    return sanitized, removed


@dataclass
class ModelArgs:
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    index_head_dim: int = 128
    index_n_heads: int = 64
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    expert_dtype: str = "fp4"
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict[str, Any] | None = None
    num_hash_layers: int = 3
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    forward_parity_fixture: str | None = None
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    compression_ratio: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ModelArgs":
        if config.get("model_type", "deepseek_v4") != "deepseek_v4":
            raise ValueError(f"expected model_type deepseek_v4, got {config.get('model_type')!r}")
        missing = [key for key in _REQUIRED_FIELDS if key not in config]
        if missing:
            raise ValueError(f"missing required DeepSeek V4 config fields: {', '.join(missing)}")
        field_names = {field.name for field in fields(cls)}
        values = {key: config[key] for key in field_names if key in config}
        args = cls(**values)
        args.validate()
        return args

    def validate(self) -> None:
        if self.expert_dtype not in _SUPPORTED_EXPERT_DTYPES:
            raise ValueError(f"unsupported DeepSeek V4 expert_dtype {self.expert_dtype!r}; expected one of {sorted(_SUPPORTED_EXPERT_DTYPES)}")
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("DeepSeek V4 hidden_size and num_hidden_layers must be positive")
        if self.num_attention_heads <= 0 or self.head_dim <= 0:
            raise ValueError("DeepSeek V4 attention head counts and dimensions must be positive")
        if self.q_lora_rank <= 0 or self.qk_rope_head_dim <= 0:
            raise ValueError("DeepSeek V4 q_lora_rank and qk_rope_head_dim must be positive")
        if self.n_routed_experts <= 0 or self.num_experts_per_tok <= 0:
            raise ValueError("DeepSeek V4 routed expert counts must be positive")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("DeepSeek V4 num_experts_per_tok cannot exceed n_routed_experts")


class Model:
    """DeepSeek V4 scaffold with explicit tiny parity fixture modes and a
    bounded real (non-fixture) mode for tiny synthetic configs.

    The real mode supports a single-layer sliding-attention + top-k MoE model
    with positive hc_mult, the proven synthetic multi-head/grouped-output
    attention subset, and the proven tiny cache-less CSA ``compression_ratio=4``
    subset.  It stores loaded weights as MLX arrays and runs the proven bounded
    path with MLX operations.  Larger configs remain fail-closed.
    """

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        sanitized, _ = sanitize_weights(weights)
        return sanitized

    _INTEGRATED_WEIGHTS: frozenset[str] = frozenset({
        "embed.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_a_proj.weight",
        "q_norm.weight",
        "q_b_proj.weight",
        "kv_proj.weight",
        "kv_norm.weight",
        "o_a_proj.weight",
        "o_b_proj.weight",
        "sinks",
        "attn_hc.fn",
        "attn_hc.base",
        "attn_hc.scale",
        "ffn_hc.fn",
        "ffn_hc.base",
        "ffn_hc.scale",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.experts.0.w1.weight",
        "mlp.experts.0.w2.weight",
        "mlp.experts.0.w3.weight",
        "mlp.experts.1.w1.weight",
        "mlp.experts.1.w2.weight",
        "mlp.experts.1.w3.weight",
        "mlp.shared_experts.w1.weight",
        "mlp.shared_experts.w2.weight",
        "mlp.shared_experts.w3.weight",
    })

    def __init__(self, args: ModelArgs):
        self.args = args
        self._tiny_weights: dict[str, Any] = {}
        self._real_weights: dict[str, mx.array] = {}
        if args.forward_parity_fixture in {"embedding-rmsnorm-head", "integrated-layer"}:
            return
        if args.forward_parity_fixture is not None:
            raise NotImplementedError(f"unknown forward_parity_fixture {args.forward_parity_fixture!r}")
        self._validate_real_mode()

    def _validate_real_mode(self) -> None:
        """Fail closed unless the config is within the proven tiny subset."""

        args = self.args
        # Gate #1 relaxed (Story 11.15g): allow real num_hidden_layers (e.g. 43).
        # Proven-component parity = 11.15e structural shape-compat (header-only).
        # Forward parity is 11.15h scope; construction is safe here.
        if args.num_hidden_layers <= 0:
            raise NotImplementedError("real DeepSeek V4 model requires positive num_hidden_layers")
        # Gate #2 relaxed (Story 11.15g): allow real layer_types vectors (sliding+full).
        # Proven-component parity = 11.15e structural shape-compat.
        if args.layer_types is None or len(args.layer_types) != args.num_hidden_layers:
            raise NotImplementedError("real DeepSeek V4 model requires layer_types with length num_hidden_layers")
        # Gate #3 relaxed (Story 11.15g): allow real mlp_layer_types vectors (moe+dense).
        # Proven-component parity = 11.15e structural shape-compat.
        if args.mlp_layer_types is None or len(args.mlp_layer_types) != args.num_hidden_layers:
            raise NotImplementedError("real DeepSeek V4 model requires mlp_layer_types with length num_hidden_layers")
        if args.compression_ratio != 0:
            error = _csa_config_error(args)
            if error is not None:
                raise NotImplementedError(error)
            # Gate #5(b) relaxed (Story 11.15g): allow compressed attention with
            # num_hidden_layers>1 (hc_mult=1 retained by _csa_config_error).
            # Proven-component parity = 11.14 stateless CSA composed per-layer
            # (stateless == N independent single-layer CSAs) + 11.25 multi-head.
        if args.num_key_value_heads != 1:
            raise NotImplementedError("real DeepSeek V4 model currently supports only num_key_value_heads=1")
        if args.o_groups <= 0:
            raise NotImplementedError("real DeepSeek V4 model requires positive o_groups")
        if args.num_attention_heads % args.o_groups != 0:
            raise NotImplementedError("real DeepSeek V4 model requires num_attention_heads divisible by o_groups")
        if args.hc_mult <= 0:
            raise NotImplementedError("real DeepSeek V4 model requires positive hc_mult")
        # Gate #10 relaxed (Story 11.15g): allow multi-layer forward with hc_mult>1.
        # Proven-component parity = 11.11 hyperconnection synthetic
        # (_hyperconnection_mlx handles hc_mult>1 directly + softmax/Sinkhorn).
        # Gate #11 staged-relaxation proven per Q3 probe (Story 11.15g):
        # _moe_mlx I8 composition vs BF16 pre-dequant reference at
        # n_routed_experts in {8, 32, 256} -> L2_REL = 0.000e+00 <= 5e-3.
        # Proven component = dequantize_i8_e8m0_block_scale (ADR 0017,
        # max_abs<=1e-5 vs OCP witness) + _moe_mlx top-k stub (11.15d).
        # Drift > 5e-3 = STOP per supervisor STOP-rule (i)/(iv).
        if args.num_experts_per_tok > args.n_routed_experts:
            raise NotImplementedError("real DeepSeek V4 model requires num_experts_per_tok<=n_routed_experts")
        if args.n_shared_experts != 1:
            raise NotImplementedError("real DeepSeek V4 model currently supports only n_shared_experts=1")
        if args.scoring_func != "sqrtsoftplus":
            raise NotImplementedError("real DeepSeek V4 model currently supports only scoring_func='sqrtsoftplus'")
        if args.expert_dtype not in _SUPPORTED_EXPERT_DTYPES:
            raise NotImplementedError(f"real DeepSeek V4 model does not support expert_dtype {args.expert_dtype!r}")

    def load_tiny_weights(self, weights: dict[str, Any]) -> None:
        required = {"embed.weight", "norm.weight", "lm_head.weight"}
        missing = sorted(required - set(weights))
        if missing:
            raise ValueError(f"missing tiny DeepSeek V4 fixture weights: {', '.join(missing)}")
        self._tiny_weights = {key: weights[key] for key in required}

    def load_integrated_weights(self, weights: dict[str, Any]) -> None:
        missing = sorted(self._INTEGRATED_WEIGHTS - set(weights))
        if missing:
            raise ValueError(f"missing integrated DeepSeek V4 fixture weights: {', '.join(missing)}")
        self._tiny_weights = {key: weights[key] for key in self._INTEGRATED_WEIGHTS}

    def load_weights(self, weights: dict[str, Any] | list[tuple[str, Any]], strict: bool = True) -> None:
        if isinstance(weights, list):
            mapping = dict(weights)
        else:
            mapping = dict(weights)
        if self.args.forward_parity_fixture is None:
            from ds4_ft_mlx.shimmed_ckpt_key_remap import remap_shimmed_ckpt_keys  # Story 13.1, ADR 0021
            mapping, _dropped = remap_shimmed_ckpt_keys(mapping)                      # pre-pass: shimmed→vendor-internal
            canonical = {_canonicalize_weight_key(k, num_hidden_layers=self.args.num_hidden_layers): _as_mlx_array(v) for k, v in mapping.items()}
            self._load_real_weights(canonical, strict=strict)
            return
        canonical = {_canonicalize_weight_key(k): _mlx_array_to_py(v) for k, v in mapping.items()}
        if self.args.forward_parity_fixture == "embedding-rmsnorm-head":
            allowed = {"embed.weight", "norm.weight", "lm_head.weight"}
            unexpected = sorted(set(canonical) - allowed)
            if strict and unexpected:
                raise ValueError(f"unexpected tiny DeepSeek V4 fixture weights: {', '.join(unexpected)}")
            self.load_tiny_weights(canonical)
        elif self.args.forward_parity_fixture == "integrated-layer":
            unexpected = sorted(set(canonical) - self._INTEGRATED_WEIGHTS)
            if strict and unexpected:
                raise ValueError(f"unexpected integrated DeepSeek V4 fixture weights: {', '.join(unexpected)}")
            self.load_integrated_weights(canonical)
        else:
            raise NotImplementedError(f"load_weights not implemented for fixture {self.args.forward_parity_fixture!r}")

    def _load_real_weights(self, weights: dict[str, mx.array], *, strict: bool) -> None:
        args = self.args
        n_layers = args.num_hidden_layers
        multi = n_layers > 1
        per_layer_base = {key for key in self._INTEGRATED_WEIGHTS if not key.startswith("mlp.experts.") and key != "embed.weight"}
        if args.compression_ratio != 0:
            # The proven CSA path consumes compressor/indexer weights directly;
            # do not require or allow unused q/kv/o/sink sliding-attention tensors.
            per_layer_base -= {
                "q_a_proj.weight",
                "q_norm.weight",
                "q_b_proj.weight",
                "kv_proj.weight",
                "kv_norm.weight",
                "o_a_proj.weight",
                "o_b_proj.weight",
                "sinks",
            }
        required: set[str] = {"embed.weight"}
        for i in range(n_layers):
            prefix = f"layers.{i}." if multi else ""
            for key in per_layer_base:
                required.add(f"{prefix}{key}")
            if args.compression_ratio != 0:
                for key in _CSA_WEIGHT_KEYS:
                    required.add(f"{prefix}{key}")
            for eid in range(args.n_routed_experts):
                for proj in ("w1", "w2", "w3"):
                    required.add(f"{prefix}mlp.experts.{eid}.{proj}.weight")
            _edt = str(args.expert_dtype).lower()
            _fp4_has_scales = _edt == "fp4" and f"{prefix}mlp.experts.0.w1.scale" in weights
            if _edt == "i8" or _fp4_has_scales:
                for eid in range(args.n_routed_experts):
                    for proj in ("w1", "w2", "w3"):
                        required.add(f"{prefix}mlp.experts.{eid}.{proj}.scale")
        loadable = dict(weights)
        missing_set = required - set(loadable)
        if mx is None:
            raise NotImplementedError("mlx is required for real DeepSeek V4 load_weights but is not installed")
        for i in range(n_layers):
            prefix = f"layers.{i}." if multi else ""
            bias_key = f"{prefix}mlp.gate.e_score_correction_bias"
            if bias_key in missing_set:
                # Transformers registers this as a persistent zero buffer; some
                # checkpoints omit it, so default it explicitly and visibly.
                loadable[bias_key] = mx.zeros((args.n_routed_experts,), dtype=mx.float32)
                missing_set.discard(bias_key)
        missing = sorted(missing_set)
        if missing:
            raise ValueError(f"missing real DeepSeek V4 weights: {', '.join(missing)}")
        optional_final = {"norm.weight", "lm_head.weight"}
        optional_hc_head = {"hc_head.fn", "hc_head.base", "hc_head.scale"} if args.hc_mult > 1 else set()
        present_hc_head = set(loadable) & {"hc_head.fn", "hc_head.base", "hc_head.scale"}
        if present_hc_head and present_hc_head != {"hc_head.fn", "hc_head.base", "hc_head.scale"}:
            missing_hc_head = sorted({"hc_head.fn", "hc_head.base", "hc_head.scale"} - present_hc_head)
            raise ValueError(f"incomplete real DeepSeek V4 hc_head weights: {', '.join(missing_hc_head)}")
        if strict:
            allowed = required | optional_final | optional_hc_head
            unexpected = sorted(set(loadable) - allowed)
            if unexpected:
                raise ValueError(f"unexpected real DeepSeek V4 weights: {', '.join(unexpected)}")
        self._real_weights = {key: loadable[key] for key in loadable}

    def eval(self) -> "Model":
        return self

    def parameters(self) -> dict[str, Any]:
        if self.args.forward_parity_fixture is None:
            return self._real_weights
        return self._tiny_weights

    def __call__(self, input_ids: list[list[int]]) -> list[list[list[float]]]:
        if self.args.forward_parity_fixture is None:
            if not self._real_weights:
                raise ValueError("DeepSeek V4 real weights are not loaded")
            return self._real_forward(input_ids)
        if not self._tiny_weights:
            raise ValueError("DeepSeek V4 weights are not loaded")
        if self.args.forward_parity_fixture == "embedding-rmsnorm-head":
            embeddings = self._tiny_weights["embed.weight"]
            norm_weight = self._tiny_weights["norm.weight"]
            lm_head = self._tiny_weights["lm_head.weight"]
            return [[self._project(self._rms_norm(embeddings[token_id], norm_weight), lm_head) for token_id in row] for row in input_ids]
        if self.args.forward_parity_fixture == "integrated-layer":
            return _integrated_layer_forward(self.args, input_ids, self._tiny_weights)
        raise NotImplementedError(f"forward not implemented for fixture {self.args.forward_parity_fixture!r}")

    def _layer_weights_view(self, layer_idx: int) -> dict[str, mx.array]:
        """Return the per-layer weight dict (flat subkeys) for layer ``layer_idx``."""

        if self.args.num_hidden_layers > 1:
            prefix = f"layers.{layer_idx}."
            return {key[len(prefix):]: value for key, value in self._real_weights.items() if key.startswith(prefix)}
        return self._real_weights

    def _real_layer_forward(self, args: ModelArgs, h: mx.array, layer_w: dict[str, mx.array]) -> mx.array:
        """Run one decoder layer (attention + MoE) over stream state ``h``."""

        attn_hc = _hyperconnection_mlx(
            h,
            fn=layer_w["attn_hc.fn"],
            base=layer_w["attn_hc.base"],
            scale=layer_w["attn_hc.scale"],
            hc_mult=args.hc_mult,
            eps=args.hc_eps,
            sinkhorn_iters=args.hc_sinkhorn_iters,
            rms_norm_eps=args.rms_norm_eps,
        )
        norm_in = _rms_norm_mlx(attn_hc["collapsed"], layer_w["input_layernorm.weight"], args.rms_norm_eps)
        attn_out = _attention_mlx(args, norm_in, layer_w)
        attn_residual = mx.sum(mx.expand_dims(attn_hc["comb"], -1) * mx.expand_dims(h, -2), axis=-3)
        h = mx.expand_dims(attn_hc["post"], -1) * mx.expand_dims(attn_out, -2) + attn_residual

        ffn_hc = _hyperconnection_mlx(
            h,
            fn=layer_w["ffn_hc.fn"],
            base=layer_w["ffn_hc.base"],
            scale=layer_w["ffn_hc.scale"],
            hc_mult=args.hc_mult,
            eps=args.hc_eps,
            sinkhorn_iters=args.hc_sinkhorn_iters,
            rms_norm_eps=args.rms_norm_eps,
        )
        norm_post = _rms_norm_mlx(ffn_hc["collapsed"], layer_w["post_attention_layernorm.weight"], args.rms_norm_eps)
        moe_out = _moe_mlx(args, norm_post, layer_w)
        ffn_residual = mx.sum(mx.expand_dims(ffn_hc["comb"], -1) * mx.expand_dims(h, -2), axis=-3)
        h = mx.expand_dims(ffn_hc["post"], -1) * mx.expand_dims(moe_out, -2) + ffn_residual
        return h

    def _real_forward(self, input_ids: list[list[int]]) -> list[list[list[float]]]:
        """Run the tiny real model with MLX array operations."""

        if mx is None:
            raise NotImplementedError("mlx is required for real DeepSeek V4 forward but is not installed")
        args = self.args
        ids = mx.array(input_ids)
        h0 = self._real_weights["embed.weight"][ids]
        h = mx.repeat(mx.expand_dims(h0, -2), repeats=args.hc_mult, axis=-2)

        for layer_idx in range(args.num_hidden_layers):
            layer_w = self._layer_weights_view(layer_idx)
            h = self._real_layer_forward(args, h, layer_w)

        if args.hc_mult == 1:
            h = h[..., 0, :]
            if "norm.weight" in self._real_weights:
                h = _rms_norm_mlx(h, self._real_weights["norm.weight"], args.rms_norm_eps)
            if "lm_head.weight" in self._real_weights:
                h = _linear_mlx(h, self._real_weights["lm_head.weight"])
            return h.tolist()

        has_hc_head = {"hc_head.fn", "hc_head.base", "hc_head.scale"}.issubset(self._real_weights)
        has_final_projection = "norm.weight" in self._real_weights or "lm_head.weight" in self._real_weights
        if has_final_projection and not has_hc_head:
            raise NotImplementedError("real DeepSeek V4 hc_mult>1 forward does not support final norm/lm_head without hc_head weights")
        if has_hc_head:
            h = _hyperhead_mlx(
                h,
                fn=self._real_weights["hc_head.fn"],
                base=self._real_weights["hc_head.base"],
                scale=self._real_weights["hc_head.scale"],
                hc_mult=args.hc_mult,
                eps=args.hc_eps,
                rms_norm_eps=args.rms_norm_eps,
            )
            if "norm.weight" in self._real_weights:
                h = _rms_norm_mlx(h, self._real_weights["norm.weight"], args.rms_norm_eps)
            if "lm_head.weight" in self._real_weights:
                h = _linear_mlx(h, self._real_weights["lm_head.weight"])
            return h.tolist()
        return h.tolist()

    def _rms_norm(self, vector: list[float], weight: list[float]) -> list[float]:
        if len(vector) != len(weight):
            raise ValueError("tiny DeepSeek V4 fixture norm weight shape mismatch")
        mean_square = sum(float(x) * float(x) for x in vector) / len(vector)
        denom = math.sqrt(mean_square + float(self.args.rms_norm_eps))
        return [float(x) / denom * float(w) for x, w in zip(vector, weight)]

    @staticmethod
    def _project(vector: list[float], weight: list[list[float]]) -> list[float]:
        for row in weight:
            if len(row) != len(vector):
                raise ValueError("tiny DeepSeek V4 fixture lm_head weight shape mismatch")
        return [sum(float(x) * float(w) for x, w in zip(vector, row)) for row in weight]
