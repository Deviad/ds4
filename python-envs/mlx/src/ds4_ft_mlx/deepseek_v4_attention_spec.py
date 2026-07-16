"""Tiny DeepSeek V4 attention shape-spec helpers.

This module is a deterministic parity scaffold only.  It records the DS4 tensor
layout semantics for q_a/q_b/wkv/wo_a/wo_b and rope/nope split rules used by MLX
implementation work.  It now includes tiny pure-Python forward references for
the proven compressor/indexer CSA, MoE, and hyperconnection subsets; real-scale
stateful generation remains outside this scaffold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

Shape = tuple[int, ...]


@dataclass(frozen=True)
class RopeNopeSplit:
    """Per-head split with RoPE applied to the tail of each attention head."""

    nope_dim: int
    rope_dim: int

    @property
    def nope_slice(self) -> tuple[int, int]:
        return (0, self.nope_dim)

    @property
    def rope_slice(self) -> tuple[int, int]:
        return (self.nope_dim, self.nope_dim + self.rope_dim)

    def describe(self) -> str:
        return f"nope[{self.nope_slice[0]}:{self.nope_slice[1]}] rope[{self.rope_slice[0]}:{self.rope_slice[1]}]"


@dataclass(frozen=True)
class DeepSeekV4AttentionSpec:
    """Shape-only DS4/DeepSeek V4 attention specification.

    Tensor shapes are expressed in DS4 matvec layout, i.e. `(input_dim,
    output_dim)` for 2-D projection tensors.  This matches the existing DS4
    runtime validation (`tensor_expect_layout(..., dim0=input, dim1=output)`) and
    avoids implying MLX module implementation is complete.
    """

    hidden_size: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    qk_rope_head_dim: int
    num_output_groups: int = 1
    compression_ratio: int = 0
    index_n_heads: int | None = None
    index_head_dim: int | None = None

    def __post_init__(self) -> None:
        positive_fields = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "num_output_groups": self.num_output_groups,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.qk_rope_head_dim < 0:
            raise ValueError("qk_rope_head_dim must be non-negative")
        if self.qk_rope_head_dim > self.head_dim:
            raise ValueError("qk_rope_head_dim cannot exceed head_dim")
        if self.num_attention_heads % self.num_output_groups != 0:
            raise ValueError("num_attention_heads must be divisible by num_output_groups")
        if self.compression_ratio < 0:
            raise ValueError("compression_ratio must be non-negative")
        if (self.index_n_heads is None) != (self.index_head_dim is None):
            raise ValueError("index_n_heads and index_head_dim must be provided together")
        if self.index_n_heads is not None:
            if self.index_n_heads <= 0 or self.index_head_dim <= 0:
                raise ValueError("indexer dimensions must be positive")
            if self.compression_ratio != 4:
                raise ValueError("indexer dimensions require compression_ratio=4")

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def heads_per_output_group(self) -> int:
        return self.num_attention_heads // self.num_output_groups

    @property
    def out_low_dim(self) -> int:
        return self.num_output_groups * self.o_lora_rank

    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    def rope_nope_split(self) -> RopeNopeSplit:
        return RopeNopeSplit(nope_dim=self.qk_nope_head_dim, rope_dim=self.qk_rope_head_dim)

    def ds4_tensor_shapes(self) -> dict[str, Shape]:
        """Return core attention projection tensor shapes in DS4 layout."""

        return {
            "wq_a": (self.hidden_size, self.q_lora_rank),
            "wq_b": (self.q_lora_rank, self.q_dim),
            "wkv": (self.hidden_size, self.head_dim),
            "wo_a": (self.head_dim * self.heads_per_output_group, self.out_low_dim),
            "wo_b": (self.out_low_dim, self.hidden_size),
        }

    def flash_mlx_safetensors_shapes(self) -> dict[str, Shape]:
        """Return Flash checkpoint/MLX attention shapes in `[out, in]` layout.

        This is intentionally distinct from :meth:`ds4_tensor_shapes`, whose
        projection matrices use DS4 matvec layout `(input_dim, output_dim)`.
        Safetensors/Transformers/MLX linear weights are stored as
        `[out_features, in_features]`.
        """

        return {
            "q_a_proj.weight": (self.q_lora_rank, self.hidden_size),
            "q_norm.weight": (self.q_lora_rank,),
            "q_b_proj.weight": (self.q_dim, self.q_lora_rank),
            "kv_proj.weight": (self.head_dim, self.hidden_size),
            "kv_norm.weight": (self.head_dim,),
            "o_a_proj.weight": (self.out_low_dim, self.head_dim * self.heads_per_output_group),
            "o_b_proj.weight": (self.hidden_size, self.out_low_dim),
            "sinks": (self.num_attention_heads,),
        }

    def activation_shapes(self, batch: int, seq_len: int) -> dict[str, Shape]:
        if batch <= 0 or seq_len <= 0:
            raise ValueError("batch and seq_len must be positive")
        return {
            "q_a_out": (batch, seq_len, self.q_lora_rank),
            "q_b_out_flat": (batch, seq_len, self.q_dim),
            "q_b_out_heads": (batch, seq_len, self.num_attention_heads, self.head_dim),
            "wkv_out": (batch, seq_len, self.head_dim),
            "wo_a_grouped_in": (
                batch,
                seq_len,
                self.num_output_groups,
                self.head_dim * self.heads_per_output_group,
            ),
            "wo_a_out_flat": (batch, seq_len, self.out_low_dim),
            "wo_b_out": (batch, seq_len, self.hidden_size),
        }

    def norm_and_sink_shapes(self) -> dict[str, Shape]:
        """Return attention norm/sink tensor shapes required by V4/DS4."""

        return {
            "q_norm": (self.q_lora_rank,),
            "kv_norm": (self.head_dim,),
            "attn_sink": (self.num_attention_heads,),
            "attn_norm": (self.hidden_size,),
        }

    def hyperconnection_shapes(self, *, hc_mult: int) -> dict[str, Shape]:
        """Return per-layer Hyper-Connection metadata shapes.

        This is shape metadata only.  Sinkhorn/pre/post/combination behavior is
        intentionally not implemented here.
        """

        if hc_mult <= 0:
            raise ValueError("hc_mult must be positive")
        mix = (2 + hc_mult) * hc_mult
        flat_hidden = hc_mult * self.hidden_size
        return {
            "hc_attn_base": (mix,),
            "hc_attn_fn": (mix, flat_hidden),
            "hc_attn_scale": (3,),
            "hc_ffn_base": (mix,),
            "hc_ffn_fn": (mix, flat_hidden),
            "hc_ffn_scale": (3,),
        }

    def compressor_shapes(self) -> dict[str, Shape]:
        """Return compressor tensor shapes, or an empty mapping when disabled."""

        if self.compression_ratio == 0:
            return {}
        coeff = 2 if self.compression_ratio == 4 else 1
        width = coeff * self.head_dim
        return {
            "compressor_ape": (width, self.compression_ratio),
            "compressor_wkv": (self.hidden_size, width),
            "compressor_wgate": (self.hidden_size, width),
            "compressor_norm": (self.head_dim,),
        }

    def indexer_shapes(self) -> dict[str, Shape]:
        """Return indexer tensor shapes, or an empty mapping when absent."""

        if self.index_n_heads is None or self.index_head_dim is None:
            return {}
        index_q_dim = self.index_n_heads * self.index_head_dim
        index_width = 2 * self.index_head_dim
        return {
            "indexer_wq_b": (self.q_lora_rank, index_q_dim),
            "indexer_proj": (self.hidden_size, self.index_n_heads),
            "indexer_compressor_ape": (index_width, self.compression_ratio),
            "indexer_compressor_wkv": (self.hidden_size, index_width),
            "indexer_compressor_wgate": (self.hidden_size, index_width),
            "indexer_compressor_norm": (self.index_head_dim,),
        }


def _linear(vector: list[float], weight: list[list[float]]) -> list[float]:
    if not weight:
        raise ValueError("linear weight must be non-empty")
    if len(vector) != len(weight):
        raise ValueError("linear input dimension mismatch")
    out_dim = len(weight[0])
    if any(len(row) != out_dim for row in weight):
        raise ValueError("linear weight rows must have consistent output dimension")
    return [sum(float(vector[i]) * float(weight[i][j]) for i in range(len(vector))) for j in range(out_dim)]


def _linear_out_in(vector: list[float], weight: list[list[float]]) -> list[float]:
    """Linear projection for safetensors/MLX `[out_features, in_features]` weights."""

    if not weight:
        raise ValueError("linear weight must be non-empty")
    in_dim = len(weight[0])
    if len(vector) != in_dim:
        raise ValueError("linear input dimension mismatch")
    if any(len(row) != in_dim for row in weight):
        raise ValueError("linear weight rows must have consistent input dimension")
    return [sum(float(row[i]) * float(vector[i]) for i in range(in_dim)) for row in weight]


def _rms_norm(vector: list[float], weight: list[float] | None, eps: float) -> list[float]:
    if weight is not None and len(weight) != len(vector):
        raise ValueError("RMSNorm weight dimension mismatch")
    denom = math.sqrt(sum(float(x) * float(x) for x in vector) / len(vector) + eps)
    if weight is None:
        return [float(x) / denom for x in vector]
    return [float(x) / denom * float(w) for x, w in zip(vector, weight)]


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def attention_scores_with_sink(logits: list[float], *, sink: float) -> list[float]:
    """Return attention scores after applying the DeepSeek V4 sink logit.

    Transformers concatenates the per-head sink as one extra logit, softmaxes
    over keys+sink, then drops the sink probability before multiplying by V.
    Consequently returned key scores can sum to less than one.
    """

    if not logits:
        raise ValueError("attention logits must be non-empty")
    probs = _softmax([float(value) for value in logits] + [float(sink)])
    return probs[:-1]


def sliding_window_cache_update(
    past_kv: list[list[float]],
    incoming_kv: list[list[float]],
    *,
    sliding_window: int,
) -> dict[str, list[list[float]]]:
    """Mirror Transformers V4 HCA cache update return vs persisted state.

    The current attention call receives the full `past + incoming` K=V rows.
    The cache stored for the next step keeps only `sliding_window - 1` rows,
    because the next incoming token supplies the current row.
    """

    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    combined = [list(row) for row in past_kv] + [list(row) for row in incoming_kv]
    if not combined:
        return {"attention_kv": [], "persisted_kv": []}
    width = len(combined[0])
    if any(len(row) != width for row in combined):
        raise ValueError("KV cache entries must have consistent width")
    persisted_len = max(0, sliding_window - 1)
    persisted = combined[-persisted_len:] if persisted_len else []
    return {"attention_kv": combined, "persisted_kv": persisted}


def hca_block_bias(*, position_ids: list[int], compressed_len: int, compression_ratio: int) -> list[list[float]] | None:
    """Build HCA compressor causal block-bias metadata.

    Mirrors the reference no-bias cases for single-token decode or no compressed
    entries, otherwise masks entries where
    `entry_index >= (position_id + 1) // hca_compression_ratio`.  This covers
    HCA mask metadata only, not CSA indexer scatter or compressor forward values.
    """

    if compressed_len < 0:
        raise ValueError("compressed_len must be non-negative")
    if compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive")
    for position in position_ids:
        if position < 0:
            raise ValueError("position_ids must be non-negative")
    if len(position_ids) == 1 or compressed_len == 0:
        return None
    out: list[list[float]] = []
    for position in position_ids:
        causal_threshold = (int(position) + 1) // compression_ratio
        out.append([0.0 if entry_idx < causal_threshold else float("-inf") for entry_idx in range(compressed_len)])
    return out


def csa_topk_indexer_gather(
    block_scores: list[list[float]],
    *,
    block_bias: list[list[float]] | None,
    top_k: int,
) -> dict[str, object]:
    """Return deterministic CSA top-k gather indices and masks.

    This helper covers indexer gather metadata only: row-wise top-k compressed
    block ids after applying a causal block-bias mask.  It pads under-filled rows
    with `-1` and a false gather mask.  It does not gather compressed KV values
    or claim compressor/indexer forward parity.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_count = len(block_scores)
    if block_bias is not None and len(block_bias) != query_count:
        raise ValueError("block_bias query dimension mismatch")
    block_count = len(block_scores[0]) if block_scores else 0
    if any(len(row) != block_count for row in block_scores):
        raise ValueError("block_scores rows must have consistent compressed block dimension")
    if block_bias is not None and any(len(row) != block_count for row in block_bias):
        raise ValueError("block_bias rows must match block_scores compressed block dimension")

    gather_indices: list[list[int]] = []
    gather_mask: list[list[bool]] = []
    eligible_counts: list[int] = []
    for row_idx, score_row in enumerate(block_scores):
        candidates: list[tuple[float, int]] = []
        bias_row = [0.0] * block_count if block_bias is None else block_bias[row_idx]
        for block_idx, (score, bias) in enumerate(zip(score_row, bias_row)):
            effective = float(score) + float(bias)
            if math.isfinite(effective):
                candidates.append((effective, block_idx))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = [block_idx for _score, block_idx in candidates[:top_k]]
        eligible_counts.append(len(candidates))
        padded = selected + [-1] * (top_k - len(selected))
        mask = [True] * len(selected) + [False] * (top_k - len(selected))
        gather_indices.append(padded)
        gather_mask.append(mask)
    return {
        "gather_shape": (query_count, top_k),
        "gather_indices": gather_indices,
        "gather_mask": gather_mask,
        "eligible_counts": eligible_counts,
    }


def run_tiny_csa_topk_indexer_fixture() -> dict[str, object]:
    block_bias = hca_block_bias(position_ids=[1, 3, 5], compressed_len=3, compression_ratio=2)
    got = csa_topk_indexer_gather(
        [[0.1, 0.9, 0.8], [0.2, 0.7, 0.3], [0.2, 0.7, 0.9]],
        block_bias=block_bias,
        top_k=2,
    )
    expected = {
        "gather_shape": (3, 2),
        "gather_indices": [[0, -1], [1, 0], [2, 1]],
        "gather_mask": [[True, False], [True, True], [True, True]],
        "eligible_counts": [1, 2, 3],
    }
    max_abs_error = 0.0 if got == expected else 1.0
    return {
        "fixture": "csa-topk-indexer-gather-mask",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "CSA top-k indexer gather shape",
            "causal block-bias mask application",
            "stable score tie ordering by block id",
            "padded gather indices and masks",
        ],
        "not_covered": [
            "compressor/indexer forward",
            "compressed KV value gather",
            "full CSA attention parity",
            "trusted Transformers numeric forward reference",
        ],
    }


def _validate_tail_rope_dim(spec: DeepSeekV4AttentionSpec) -> None:
    if spec.qk_rope_head_dim <= 0:
        raise ValueError("qk_rope_head_dim must be positive for tail-RoPE attention")
    if spec.qk_rope_head_dim > spec.head_dim:
        raise ValueError("qk_rope_head_dim cannot exceed head_dim")
    if spec.qk_rope_head_dim % 2 != 0:
        raise ValueError("qk_rope_head_dim must be even for pairwise RoPE")


def _rope_cos_sin_for_position(*, position: int, rope_dim: int, theta: float) -> tuple[list[float], list[float]]:
    if position < 0:
        raise ValueError("position must be non-negative")
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError("rope_dim must be a positive even integer")
    cos: list[float] = []
    sin: list[float] = []
    for dim in range(0, rope_dim, 2):
        freq = 1.0 / (theta ** (dim / rope_dim))
        angle = float(position) * freq
        c = math.cos(angle)
        s = math.sin(angle)
        cos.extend([c, c])
        sin.extend([s, s])
    return cos, sin


def _split_heads(vector: list[float], *, num_heads: int, head_dim: int) -> list[list[float]]:
    if len(vector) != num_heads * head_dim:
        raise ValueError("flat head vector dimension mismatch")
    return [vector[i * head_dim: (i + 1) * head_dim] for i in range(num_heads)]


def _grouped_output_projection(
    attended_heads: list[list[float]],
    *,
    spec: DeepSeekV4AttentionSpec,
    o_a_weight: list[list[float]],
    o_b_weight: list[list[float]],
) -> list[float]:
    expected_o_a = spec.flash_mlx_safetensors_shapes()["o_a_proj.weight"]
    expected_o_b = spec.flash_mlx_safetensors_shapes()["o_b_proj.weight"]
    if (len(o_a_weight), len(o_a_weight[0]) if o_a_weight else 0) != expected_o_a:
        raise ValueError("o_a_proj.weight shape mismatch for grouped output")
    if (len(o_b_weight), len(o_b_weight[0]) if o_b_weight else 0) != expected_o_b:
        raise ValueError("o_b_proj.weight shape mismatch for grouped output")
    if len(attended_heads) != spec.num_attention_heads:
        raise ValueError("attended head count mismatch")

    low_rank: list[float] = []
    for group in range(spec.num_output_groups):
        head_start = group * spec.heads_per_output_group
        group_heads = attended_heads[head_start: head_start + spec.heads_per_output_group]
        flat_group = [value for head in group_heads for value in head]
        row_start = group * spec.o_lora_rank
        rows = o_a_weight[row_start: row_start + spec.o_lora_rank]
        low_rank.extend(_linear_out_in(flat_group, rows))
    return _linear_out_in(low_rank, o_b_weight)


_INCREMENTAL_CSA_DEFERRED = (
    "incremental KV cache covers only compression_ratio=0 (compressor-free) sliding-window attention; "
    "CSA stateful Ca-carry across steps is deferred (Story 11.16 follow-up)"
)


def _validate_multihead_attention_inputs(
    spec: DeepSeekV4AttentionSpec,
    weights: dict[str, list[list[float]] | list[float]],
    *,
    sliding_window: int,
) -> None:
    if spec.compression_ratio != 0:
        raise ValueError("multi-head reference requires compression_ratio=0")
    _validate_tail_rope_dim(spec)
    if sliding_window < 0:
        raise ValueError("sliding_window must be non-negative")
    required = {"q_a_proj.weight", "q_norm.weight", "q_b_proj.weight", "kv_proj.weight", "kv_norm.weight", "o_a_proj.weight", "o_b_proj.weight", "sinks"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing multi-head attention fixture weights: {', '.join(missing)}")
    shapes = spec.flash_mlx_safetensors_shapes()
    matrix_keys = ("q_a_proj.weight", "q_b_proj.weight", "kv_proj.weight", "o_a_proj.weight", "o_b_proj.weight")
    vector_keys = ("q_norm.weight", "kv_norm.weight", "sinks")
    for key in matrix_keys:
        matrix = weights[key]
        expected = shapes[key]
        if not isinstance(matrix, list) or (len(matrix), len(matrix[0]) if matrix else 0) != expected:
            raise ValueError(f"{key} shape mismatch for Flash attention reference")
    for key in vector_keys:
        vector = weights[key]
        if not isinstance(vector, list) or len(vector) != shapes[key][0]:
            raise ValueError(f"{key} shape mismatch for Flash attention reference")


def _project_token(
    spec: DeepSeekV4AttentionSpec,
    hidden: list[float],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    pos: int,
    rms_norm_eps: float,
    rope_theta: float,
) -> tuple[list[list[float]], list[float]]:
    """Project one absolute-position token into RoPE-baked query heads and KV."""

    nope_dim = spec.qk_nope_head_dim
    q_a = _linear_out_in(hidden, weights["q_a_proj.weight"])  # type: ignore[arg-type]
    q_residual = _rms_norm(q_a, weights["q_norm.weight"], rms_norm_eps)  # type: ignore[arg-type]
    q_flat = _linear_out_in(q_residual, weights["q_b_proj.weight"])  # type: ignore[arg-type]
    q_heads = [_rms_norm(head, None, rms_norm_eps) for head in _split_heads(q_flat, num_heads=spec.num_attention_heads, head_dim=spec.head_dim)]
    kv = _rms_norm(_linear_out_in(hidden, weights["kv_proj.weight"]), weights["kv_norm.weight"], rms_norm_eps)  # type: ignore[arg-type]
    cos, sin = _rope_cos_sin_for_position(position=pos, rope_dim=spec.qk_rope_head_dim, theta=rope_theta)
    return (
        [apply_rope_tail(head, nope_dim=nope_dim, cos=cos, sin=sin) for head in q_heads],
        apply_rope_tail(kv, nope_dim=nope_dim, cos=cos, sin=sin),
    )


def _attend_at_position(
    spec: DeepSeekV4AttentionSpec,
    q_heads_roped: list[list[float]],
    key_kv_rows: list[list[float]],
    sinks: list[float],
    *,
    pos: int,
    rope_theta: float,
    o_a_weight: list[list[float]],
    o_b_weight: list[list[float]],
) -> list[float]:
    """Apply one query position to already-position-baked KV rows."""

    if not key_kv_rows:
        raise ValueError("incremental attention requires at least one KV row")
    if any(len(row) != spec.head_dim for row in key_kv_rows):
        raise ValueError("KV cache row width must match head_dim")
    if len(q_heads_roped) != spec.num_attention_heads:
        raise ValueError("query head count mismatch")
    if len(sinks) != spec.num_attention_heads:
        raise ValueError("sink count mismatch")

    scale = spec.head_dim ** -0.5
    nope_dim = spec.qk_nope_head_dim
    cos, sin = _rope_cos_sin_for_position(position=pos, rope_dim=spec.qk_rope_head_dim, theta=rope_theta)
    attended_heads: list[list[float]] = []
    for head_idx, query in enumerate(q_heads_roped):
        logits = [sum(q * k for q, k in zip(query, kv_row)) * scale for kv_row in key_kv_rows]
        probs = attention_scores_with_sink(logits, sink=float(sinks[head_idx]))
        attended = [sum(prob * kv_row[dim] for prob, kv_row in zip(probs, key_kv_rows)) for dim in range(spec.head_dim)]
        attended_heads.append(apply_output_inverse_rope_tail(attended, nope_dim=nope_dim, cos=cos, sin=sin))
    return _grouped_output_projection(attended_heads, spec=spec, o_a_weight=o_a_weight, o_b_weight=o_b_weight)


def tiny_multihead_grouped_attention_reference(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    sliding_window: int = 0,
) -> list[list[float]]:
    """Synthetic Flash-style multi-head attention reference.

    This dependency-free fixture mirrors the bounded non-compressor path with
    safetensors/MLX `[out, in]` weights: q_a/q_b, weighted q_a RMSNorm,
    unweighted per-head q RMSNorm, one shared KV head broadcast to all query
    heads, tail-only RoPE on the last `qk_rope_head_dim` channels, per-head sink
    as an extra stable-softmax bucket that is dropped before value mixing,
    inverse output RoPE, independent grouped `o_a` blocks, and final `o_b`.
    """

    _validate_multihead_attention_inputs(spec, weights, sliding_window=sliding_window)
    q_heads_by_pos: list[list[list[float]]] = []
    kv_by_pos: list[list[float]] = []
    for pos, hidden in enumerate(hidden_states):
        q_heads, kv = _project_token(spec, hidden, weights, pos=pos, rms_norm_eps=rms_norm_eps, rope_theta=rope_theta)
        q_heads_by_pos.append(q_heads)
        kv_by_pos.append(kv)

    outputs: list[list[float]] = []
    sinks = weights["sinks"]  # type: ignore[assignment]
    for pos, q_heads in enumerate(q_heads_by_pos):
        start = 0 if sliding_window == 0 else max(0, pos - sliding_window + 1)
        outputs.append(_attend_at_position(
            spec,
            q_heads,
            kv_by_pos[start: pos + 1],
            sinks,  # type: ignore[arg-type]
            pos=pos,
            rope_theta=rope_theta,
            o_a_weight=weights["o_a_proj.weight"],  # type: ignore[arg-type]
            o_b_weight=weights["o_b_proj.weight"],  # type: ignore[arg-type]
        ))
    return outputs


class IncrementalSlidingKVCache:
    """Tiny compressor-free incremental cache storing position-baked RoPE'd KV rows."""

    def __init__(
        self,
        spec: DeepSeekV4AttentionSpec,
        weights: dict[str, list[list[float]] | list[float]],
        *,
        sliding_window: int = 0,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
    ) -> None:
        if spec.compression_ratio != 0:
            raise NotImplementedError(_INCREMENTAL_CSA_DEFERRED)
        _validate_multihead_attention_inputs(spec, weights, sliding_window=sliding_window)
        self._spec = spec
        self._weights = weights
        self._sliding_window = int(sliding_window)
        self._rms_norm_eps = float(rms_norm_eps)
        self._rope_theta = float(rope_theta)
        self._kv_rows: list[list[float]] = []
        self._next_pos = 0

    @property
    def kv_rows(self) -> list[list[float]]:
        return [list(row) for row in self._kv_rows]

    @property
    def next_pos(self) -> int:
        return self._next_pos

    def step(self, hidden_token: list[float]) -> list[float]:
        q_heads, new_kv = _project_token(
            self._spec,
            list(hidden_token),
            self._weights,
            pos=self._next_pos,
            rms_norm_eps=self._rms_norm_eps,
            rope_theta=self._rope_theta,
        )
        if self._sliding_window == 0:
            attention_kv = self._kv_rows + [new_kv]
            persisted_kv = attention_kv
        else:
            update = sliding_window_cache_update(self._kv_rows, [new_kv], sliding_window=self._sliding_window)
            attention_kv = update["attention_kv"]
            persisted_kv = update["persisted_kv"]
        out = _attend_at_position(
            self._spec,
            q_heads,
            attention_kv,
            self._weights["sinks"],  # type: ignore[arg-type]
            pos=self._next_pos,
            rope_theta=self._rope_theta,
            o_a_weight=self._weights["o_a_proj.weight"],  # type: ignore[arg-type]
            o_b_weight=self._weights["o_b_proj.weight"],  # type: ignore[arg-type]
        )
        self._kv_rows = [list(row) for row in persisted_kv]
        self._next_pos += 1
        return out


def tiny_incremental_attention(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    sliding_window: int = 0,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
) -> list[list[float]]:
    """Run the tiny attention reference token-by-token through a KV cache."""

    cache = IncrementalSlidingKVCache(spec, weights, sliding_window=sliding_window, rms_norm_eps=rms_norm_eps, rope_theta=rope_theta)
    return [cache.step(list(hidden)) for hidden in hidden_states]


def tiny_greedy_decode(
    spec: DeepSeekV4AttentionSpec,
    prompt_hidden: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    embed_table: list[list[float]],
    vocab_proj: list[list[float]],
    steps: int,
    sliding_window: int = 0,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
) -> dict[str, list[object]]:
    """Weight-free deterministic greedy decode over the tiny incremental cache."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not prompt_hidden:
        raise ValueError("prompt_hidden must be non-empty")
    if not embed_table:
        raise ValueError("embed_table must be non-empty")
    if len(vocab_proj) != len(embed_table):
        raise ValueError("vocab_proj row count must match embed_table")

    cache = IncrementalSlidingKVCache(spec, weights, sliding_window=sliding_window, rms_norm_eps=rms_norm_eps, rope_theta=rope_theta)
    hidden_states = [list(row) for row in prompt_hidden[:-1]]
    for hidden in hidden_states:
        cache.step(hidden)

    current_hidden = list(prompt_hidden[-1])
    token_ids: list[int] = []
    step_outputs: list[list[float]] = []
    for _ in range(steps):
        hidden_states.append(list(current_hidden))
        out = cache.step(current_hidden)
        step_outputs.append(out)
        logits = _linear_out_in(out, vocab_proj)
        token = max(range(len(logits)), key=lambda idx: logits[idx])
        token_ids.append(int(token))
        current_hidden = list(embed_table[token])
    return {"token_ids": token_ids, "step_outputs": step_outputs, "hidden_states": hidden_states}


def tiny_sliding_attention_no_compressor(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    rms_norm_eps: float = 1e-6,
) -> list[list[float]]:
    """Deterministic V4 sliding-attention fixture without RoPE/cache/compressor.

    This is a numeric parity slice for the non-compressor path only.  It follows
    the reference projection order: q_a -> q_norm -> q_b -> unweighted q norm,
    kv -> kv_norm, causal scaled dot-product attention with K=V, grouped output
    low-rank projection (`wo_a`) and final output projection (`wo_b`).
    """

    if spec.compression_ratio != 0:
        raise ValueError("tiny sliding fixture requires compression_ratio=0")
    if spec.qk_rope_head_dim != 0:
        raise ValueError("tiny sliding fixture does not cover RoPE; use qk_rope_head_dim=0")
    if spec.num_attention_heads != 1 or spec.num_output_groups != 1:
        raise ValueError("tiny sliding fixture currently covers one head and one output group")
    required = {"wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing tiny attention fixture weights: {', '.join(missing)}")

    queries: list[list[float]] = []
    kvs: list[list[float]] = []
    for hidden in hidden_states:
        q_a = _linear(hidden, weights["wq_a"])  # type: ignore[arg-type]
        q_residual = _rms_norm(q_a, weights["q_norm"], rms_norm_eps)  # type: ignore[arg-type]
        q = _rms_norm(_linear(q_residual, weights["wq_b"]), None, rms_norm_eps)  # type: ignore[arg-type]
        kv = _rms_norm(_linear(hidden, weights["wkv"]), weights["kv_norm"], rms_norm_eps)  # type: ignore[arg-type]
        queries.append(q)
        kvs.append(kv)

    outputs: list[list[float]] = []
    scale = spec.head_dim ** -0.5
    for pos, query in enumerate(queries):
        logits = [sum(q * k for q, k in zip(query, key)) * scale for key in kvs[: pos + 1]]
        probs = _softmax(logits)
        attended = [sum(prob * kvs[i][dim] for i, prob in enumerate(probs)) for dim in range(spec.head_dim)]
        low_rank = _linear(attended, weights["wo_a"])  # type: ignore[arg-type]
        outputs.append(_linear(low_rank, weights["wo_b"]))  # type: ignore[arg-type]
    return outputs


def apply_rope_tail(vector: list[float], *, nope_dim: int, cos: list[float], sin: list[float]) -> list[float]:
    """Apply a standard pairwise RoPE rotation to the tail slice only."""

    rope = vector[nope_dim:]
    if len(rope) % 2 != 0:
        raise ValueError("RoPE tail dimension must be even")
    if len(cos) != len(rope) or len(sin) != len(rope):
        raise ValueError("RoPE cos/sin length must match rope tail")
    out = list(vector[:nope_dim])
    for i in range(0, len(rope), 2):
        x0 = rope[i]
        x1 = rope[i + 1]
        # Pairwise formulation equivalent to HF rotate_half for one pair.
        out.append(x0 * cos[i] - x1 * sin[i])
        out.append(x1 * cos[i + 1] + x0 * sin[i + 1])
    return out


def run_tiny_rope_tail_fixture() -> dict[str, object]:
    got = apply_rope_tail([10.0, 20.0, 1.0, 2.0], nope_dim=2, cos=[0.0, 0.0], sin=[1.0, 1.0])
    expected = [10.0, 20.0, -2.0, 1.0]
    max_abs_error = max(abs(a - b) for a, b in zip(got, expected))
    return {
        "fixture": "rope-tail-pairwise",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["tail-only RoPE", "pairwise rotate_half semantics"],
        "not_covered": ["YARN scaling", "position id cache", "attention output inverse RoPE"],
    }


def apply_output_inverse_rope_tail(vector: list[float], *, nope_dim: int, cos: list[float], sin: list[float]) -> list[float]:
    """Apply the conjugate output RoPE rotation used after K=V attention."""

    return apply_rope_tail(vector, nope_dim=nope_dim, cos=cos, sin=[-value for value in sin])


def run_tiny_sink_cache_inverse_rope_fixture() -> dict[str, object]:
    sink_scores = attention_scores_with_sink([0.0, 0.0], sink=math.log(2.0))
    cache = sliding_window_cache_update([[1.0], [2.0]], [[3.0]], sliding_window=3)
    value_with_rope = apply_rope_tail([7.0, 1.0, 2.0], nope_dim=1, cos=[0.0, 0.0], sin=[1.0, 1.0])
    inverse_rope = apply_output_inverse_rope_tail(value_with_rope, nope_dim=1, cos=[0.0, 0.0], sin=[1.0, 1.0])
    bias = hca_block_bias(position_ids=[0, 1, 2, 3], compressed_len=2, compression_ratio=2)
    expected_bias = [[float("-inf"), float("-inf")], [0.0, float("-inf")], [0.0, float("-inf")], [0.0, 0.0]]
    checks = [
        abs(sink_scores[0] - 0.25),
        abs(sink_scores[1] - 0.25),
        0.0 if cache == {"attention_kv": [[1.0], [2.0], [3.0]], "persisted_kv": [[2.0], [3.0]]} else 1.0,
        max(abs(a - b) for a, b in zip(inverse_rope, [7.0, 1.0, 2.0])),
        0.0 if bias == expected_bias else 1.0,
    ]
    max_abs_error = max(checks)
    return {
        "fixture": "sink-cache-inverse-rope-hca-bias",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "attention sink extra logit",
            "sink probability dropped from value scores",
            "sliding K=V cache returns full attention KV and persists window-minus-one state",
            "output inverse RoPE tail",
            "HCA compressor causal block-bias metadata with no-bias cases",
        ],
        "not_covered": ["compressor/indexer forward", "CSA top-k indexer gather", "full cache object integration"],
    }


def tiny_hyperconnection_forward(
    *,
    hidden_streams: list[list[list[float]]],
    fn: list[list[float]],
    base: list[float],
    scale: list[float],
    hc_mult: int,
    eps: float,
    sinkhorn_iters: int,
    rms_norm_eps: float,
) -> dict[str, list[object]]:
    if hc_mult <= 0:
        raise ValueError("hc_mult must be positive")
    if len(scale) != 3:
        raise ValueError("hyperconnection scale must have 3 entries")
    expected_mix = (2 + hc_mult) * hc_mult
    if len(base) != expected_mix or len(fn) != expected_mix:
        raise ValueError("hyperconnection fn/base shape mismatch")
    pre_batch: list[list[float]] = []
    collapsed_batch: list[list[float]] = []
    post_batch: list[list[float]] = []
    comb_batch: list[list[list[float]]] = []
    for token_streams in hidden_streams:
        if len(token_streams) != hc_mult:
            raise ValueError("hidden stream count mismatch")
        flat = [value for stream in token_streams for value in stream]
        flat = _rms_norm(flat, None, rms_norm_eps)
        if any(len(row) != len(flat) for row in fn):
            raise ValueError("hyperconnection fn input dimension mismatch")
        mixes = [sum(float(x) * float(w) for x, w in zip(flat, row)) for row in fn]
        pre_w = mixes[:hc_mult]
        post_w = mixes[hc_mult: 2 * hc_mult]
        comb_w = mixes[2 * hc_mult:]
        pre_b = base[:hc_mult]
        post_b = base[hc_mult: 2 * hc_mult]
        comb_b = base[2 * hc_mult:]
        pre = [1.0 / (1.0 + math.exp(-(w * scale[0] + b))) + eps for w, b in zip(pre_w, pre_b)]
        post = [2.0 / (1.0 + math.exp(-(w * scale[1] + b))) for w, b in zip(post_w, post_b)]
        comb_logits = [comb_w[i] * scale[2] + comb_b[i] for i in range(hc_mult * hc_mult)]
        comb_rows = [comb_logits[i * hc_mult: (i + 1) * hc_mult] for i in range(hc_mult)]
        comb = [_softmax(row) for row in comb_rows]
        comb = [[value + eps for value in row] for row in comb]
        for col in range(hc_mult):
            col_sum = sum(comb[row][col] for row in range(hc_mult)) + eps
            for row in range(hc_mult):
                comb[row][col] /= col_sum
        for _ in range(max(0, sinkhorn_iters - 1)):
            for row in range(hc_mult):
                row_sum = sum(comb[row]) + eps
                comb[row] = [value / row_sum for value in comb[row]]
            for col in range(hc_mult):
                col_sum = sum(comb[row][col] for row in range(hc_mult)) + eps
                for row in range(hc_mult):
                    comb[row][col] /= col_sum
        hidden_dim = len(token_streams[0])
        collapsed = [sum(pre[stream_idx] * token_streams[stream_idx][dim] for stream_idx in range(hc_mult)) for dim in range(hidden_dim)]
        pre_batch.append(pre)
        collapsed_batch.append(collapsed)
        post_batch.append(post)
        comb_batch.append(comb)
    return {"pre": pre_batch, "collapsed": collapsed_batch, "post": post_batch, "comb": comb_batch}


def _max_abs_error_nested(actual: object, expected: object) -> float:
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return float("inf")
        err = 0.0
        for got_item, expected_item in zip(actual, expected):
            err = max(err, _max_abs_error_nested(got_item, expected_item))
        return err
    return abs(float(actual) - float(expected))


def run_tiny_hyperconnection_fixture() -> dict[str, object]:
    got = tiny_hyperconnection_forward(
        hidden_streams=[[[2.0]]],
        fn=[[0.0], [0.0], [0.0]],
        base=[0.0, 0.0, 0.0],
        scale=[0.0, 0.0, 0.0],
        hc_mult=1,
        eps=0.0,
        sinkhorn_iters=1,
        rms_norm_eps=0.0,
    )
    expected = {"pre": [[0.5]], "collapsed": [[1.0]], "post": [[1.0]], "comb": [[[1.0]]]}
    max_abs_error = max(_max_abs_error_nested(got[key], expected[key]) for key in expected)
    return {
        "fixture": "hyperconnection-hc1-collapse",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["pre", "post", "comb", "sinkhorn hc=1", "stream collapse"],
        "not_covered": ["multi-stream hc_mult>1", "full layer residual mixing"],
    }


def run_tiny_hyperconnection_hc2_fixture() -> dict[str, object]:
    """Deterministic partial evidence for non-uniform hc_mult=2 math.

    This fixture uses nonzero hypernetwork rows, base offsets, and scale values
    so pre/post/comb depend on RMSNorm input sensitivity, split positions, and
    Sinkhorn column normalization.  It is not an integrated decoder-layer
    residual placement test.
    """

    got = tiny_hyperconnection_forward(
        hidden_streams=[[[3.0, 4.0], [0.0, 5.0]]],
        fn=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        base=[0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.05, -0.05],
        scale=[0.5, -0.25, 1.25],
        hc_mult=2,
        eps=0.0,
        sinkhorn_iters=1,
        rms_norm_eps=0.0,
    )
    expected = {
        "pre": [[0.628144306391142, 0.6241279919178186]],
        "collapsed": [[1.8844329191734261, 5.633217185153661]],
        "post": [[1.0085784333258068, 0.802624679775096]],
        "comb": [[[0.3597055166182544, 0.7932581563947655], [0.6402944833817457, 0.20674184360523457]]],
    }
    max_abs_error = max(_max_abs_error_nested(got[key], expected[key]) for key in expected)
    return {
        "fixture": "hyperconnection-hc2-pre-post-comb",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "hc_mult=2 pre weights",
            "nonzero hypernetwork projection",
            "base and scale splits",
            "multi-stream collapse",
            "hc_mult=2 post gates",
            "non-uniform 2x2 comb matrix",
            "one-step Sinkhorn column normalization",
        ],
        "not_covered": [
            "full decoder-layer residual placement",
            "final hyperhead collapse",
            "integrated Transformers layer parity",
        ],
    }


def run_tiny_hyperconnection_hc2_transformers_fixture() -> dict[str, object]:
    """hc_mult=2 hyperconnection parity fixture against Transformers reference.

    Compares `tiny_hyperconnection_forward` outputs (`post`, `comb`, `collapsed`)
    against the Transformers `DeepseekV4HyperConnection` module on deterministic
    tiny weights.  This is partial evidence only: full decoder-layer residual
    placement and final hyperhead collapse remain fail-closed.
    """

    try:
        import torch
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "fixture": "hyperconnection-hc2-transformers",
            "status": "skipped",
            "max_abs_error": None,
            "reason": f"Transformers/torch reference unavailable: {exc}",
        }

    hidden_size = 4
    hc_mult = 2
    sinkhorn_iters = 2
    rms_norm_eps = 1e-6
    hc_eps = 1e-6
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        rms_norm_eps=rms_norm_eps,
    )
    module = DeepseekV4HyperConnection(config)
    module.eval()

    # Deterministic tiny weights.
    mix = (2 + hc_mult) * hc_mult
    fn_init = torch.arange(mix * hc_mult * hidden_size, dtype=torch.float32).reshape(mix, hc_mult * hidden_size) * 0.01
    base_init = torch.arange(mix, dtype=torch.float32) * 0.05
    scale_init = torch.tensor([0.5, -0.25, 1.25], dtype=torch.float32)
    with torch.no_grad():
        module.fn.copy_(fn_init)
        module.base.copy_(base_init)
        module.scale.copy_(scale_init)

    # [batch=1, seq_len=2, hc_mult=2, hidden_size=4]
    hidden_streams = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        ref_post, ref_comb, ref_collapsed = module(hidden_streams)
    ref_post = ref_post.squeeze(0).tolist()  # [seq_len, hc_mult]
    ref_comb = ref_comb.squeeze(0).tolist()  # [seq_len, hc_mult, hc_mult]
    ref_collapsed = ref_collapsed.squeeze(0).tolist()  # [seq_len, hidden_size]

    weights = {
        "fn": module.fn.tolist(),
        "base": module.base.tolist(),
        "scale": module.scale.tolist(),
    }
    got = tiny_hyperconnection_forward(
        hidden_streams=hidden_streams.squeeze(0).tolist(),
        fn=weights["fn"],
        base=weights["base"],
        scale=weights["scale"],
        hc_mult=hc_mult,
        eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        rms_norm_eps=rms_norm_eps,
    )

    max_abs_error = max(
        _max_abs_error_nested(got["post"], ref_post),
        _max_abs_error_nested(got["comb"], ref_comb),
        _max_abs_error_nested(got["collapsed"], ref_collapsed),
    )
    return {
        "fixture": "hyperconnection-hc2-transformers",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "hc_mult=2 hypernetwork fn/base/scale gating",
            "stream flatten + unweighted RMSNorm",
            "pre collapse weights",
            "post placement weights",
            "comb 2x2 doubly-stochastic mixer",
            "Sinkhorn row/column normalization",
            "Transformers DeepseekV4HyperConnection module reference",
        ],
        "not_covered": [
            "full decoder-layer residual placement",
            "final hyperhead collapse",
            "integrated Transformers layer parity",
        ],
    }


def tiny_hyperhead_collapse(
    *,
    hidden_streams: list[list[list[float]]],
    fn: list[list[float]],
    base: list[float],
    scale: float,
    hc_mult: int,
    eps: float,
    rms_norm_eps: float,
) -> dict[str, list[list[float]]]:
    """Final DeepSeek V4 HyperHead collapse before the shared final RMSNorm.

    Mirrors Transformers `DeepseekV4HyperHead.forward`: flatten stream state,
    unweighted RMSNorm, linear projection through `hc_fn`, sigmoid/base/scale
    pre-weights, then weighted sum over the stream axis.
    """

    if hc_mult <= 0:
        raise ValueError("hc_mult must be positive")
    if len(base) != hc_mult or len(fn) != hc_mult:
        raise ValueError("hyperhead fn/base shape mismatch")
    pre_batch: list[list[float]] = []
    collapsed_batch: list[list[float]] = []
    for token_streams in hidden_streams:
        if len(token_streams) != hc_mult:
            raise ValueError("hidden stream count mismatch")
        if not token_streams:
            raise ValueError("hidden streams must be non-empty")
        hidden_dim = len(token_streams[0])
        if hidden_dim <= 0 or any(len(stream) != hidden_dim for stream in token_streams):
            raise ValueError("hidden streams must have consistent non-empty hidden dimension")
        flat = [value for stream in token_streams for value in stream]
        flat = _rms_norm(flat, None, rms_norm_eps)
        if any(len(row) != len(flat) for row in fn):
            raise ValueError("hyperhead fn input dimension mismatch")
        mixes = [sum(float(x) * float(w) for x, w in zip(flat, row)) for row in fn]
        pre = [1.0 / (1.0 + math.exp(-(w * float(scale) + b))) + eps for w, b in zip(mixes, base)]
        collapsed = [sum(pre[stream_idx] * token_streams[stream_idx][dim] for stream_idx in range(hc_mult)) for dim in range(hidden_dim)]
        pre_batch.append(pre)
        collapsed_batch.append(collapsed)
    return {"pre": pre_batch, "collapsed": collapsed_batch}


def run_tiny_hyperhead_fixture() -> dict[str, object]:
    got = tiny_hyperhead_collapse(
        hidden_streams=[[[3.0, 4.0], [0.0, 5.0]]],
        fn=[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        base=[0.1, -0.2],
        scale=0.5,
        hc_mult=2,
        eps=0.0,
        rms_norm_eps=0.0,
    )
    expected = {
        "pre": [[0.628144306391142, 0.6241279919178186]],
        "collapsed": [[1.8844329191734261, 5.633217185153661]],
    }
    max_abs_error = max(_max_abs_error_nested(got[key], expected[key]) for key in expected)
    return {
        "fixture": "final-hyperhead-hc2-collapse",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "final hyperhead weighted stream collapse",
            "hyperhead RMSNorm-flatten projection",
            "hyperhead base/scale sigmoid weights",
        ],
        "not_covered": [
            "shared final RMSNorm after hyperhead",
            "integrated model output head parity",
            "real checkpoint hyperhead tensor loading",
        ],
    }


def run_tiny_hyperhead_transformers_fixture() -> dict[str, object]:
    """Final HyperHead collapse parity fixture against Transformers reference."""

    try:
        import torch
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "fixture": "hyperhead-transformers-reference",
            "status": "skipped",
            "max_abs_error": None,
            "reason": f"Transformers/torch reference unavailable: {exc}",
        }

    config = DeepseekV4Config(hidden_size=4, hc_mult=2, rms_norm_eps=1e-6, hc_eps=1e-6)
    module = DeepseekV4HyperHead(config)
    module.eval()
    with torch.no_grad():
        module.hc_fn.copy_(torch.eye(8, dtype=torch.float32)[:2])
        module.hc_base.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
        module.hc_scale.copy_(torch.tensor([0.5], dtype=torch.float32))

    hidden = torch.tensor([[[[3.0, 4.0, 0.0, 5.0], [1.0, 2.0, -1.0, 0.0]]]], dtype=torch.float32)
    with torch.no_grad():
        ref = module(hidden).squeeze(0).squeeze(0).tolist()

    got = tiny_hyperhead_collapse(
        hidden_streams=hidden.squeeze(0).tolist(),
        fn=module.hc_fn.tolist(),
        base=module.hc_base.tolist(),
        scale=float(module.hc_scale.item()),
        hc_mult=2,
        eps=1e-6,
        rms_norm_eps=1e-6,
    )
    collapsed = got["collapsed"][0]
    if len(collapsed) != len(ref):
        return {
            "fixture": "hyperhead-transformers-reference",
            "status": "failed",
            "max_abs_error": None,
            "reason": "shape mismatch",
            "covered": [],
            "not_covered": ["shared final RMSNorm", "full model output head"],
        }
    max_abs_error = max(abs(a - b) for a, b in zip(collapsed, ref))
    return {
        "fixture": "hyperhead-transformers-reference",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "final hyperhead weighted stream collapse",
            "Transformers DeepseekV4HyperHead module reference",
        ],
        "not_covered": [
            "shared final RMSNorm after hyperhead",
            "integrated model output head parity",
            "real checkpoint hyperhead tensor loading",
        ],
    }


def run_tiny_sliding_attention_fixture() -> dict[str, object]:
    spec = DeepSeekV4AttentionSpec(
        hidden_size=2,
        num_attention_heads=1,
        head_dim=2,
        q_lora_rank=2,
        o_lora_rank=2,
        qk_rope_head_dim=0,
        num_output_groups=1,
    )
    got = tiny_sliding_attention_no_compressor(spec, [[1.0, 0.0], [0.0, 1.0]], {
        "wq_a": [[1.0, 0.0], [0.0, 1.0]],
        "q_norm": [1.0, 1.0],
        "wq_b": [[1.0, 0.0], [0.0, 1.0]],
        "wkv": [[1.0, 0.0], [0.0, 1.0]],
        "kv_norm": [1.0, 1.0],
        "wo_a": [[1.0, 0.0], [0.0, 1.0]],
        "wo_b": [[1.0, 0.0], [0.0, 1.0]],
    }, rms_norm_eps=0.0)
    exp_weight_1 = math.exp(math.sqrt(2.0)) / (1.0 + math.exp(math.sqrt(2.0)))
    exp_weight_0 = 1.0 - exp_weight_1
    expected = [
        [math.sqrt(2.0), 0.0],
        [exp_weight_0 * math.sqrt(2.0), exp_weight_1 * math.sqrt(2.0)],
    ]
    max_abs_error = 0.0
    for got_row, expected_row in zip(got, expected):
        for got_value, expected_value in zip(got_row, expected_row):
            max_abs_error = max(max_abs_error, abs(float(got_value) - float(expected_value)))
    return {
        "fixture": "sliding-attention-no-compressor-no-rope",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["q_a", "q_norm", "q_b", "q_b_norm", "kv", "kv_norm", "causal attention", "wo_a", "wo_b"],
        "not_covered": ["RoPE", "cache", "attention sinks", "compressor/indexer", "hyperconnection"],
    }


def tiny_hca_compressor_forward(
    hidden_states: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    compress_rate: int,
    rms_norm_eps: float,
    rope_cos: list[list[float]],
    rope_sin: list[list[float]],
    first_window_position: int = 0,
) -> list[list[float]]:
    """Deterministic HCA compressor forward for a single stateless chunk.

    Mirrors Transformers `DeepseekV4HCACompressor.forward` when no cache is
    supplied: project KV and gate, close complete windows of `compress_rate`
    tokens, softmax-gate each window into one compressed entry, RMSNorm, and
    apply interleaved RoPE at deterministic absolute positions.
    """

    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    required = {"kv_proj", "gate_proj", "position_bias", "kv_norm"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing HCA compressor weights: {', '.join(missing)}")
    kv_proj = weights["kv_proj"]  # type: ignore[assignment]
    gate_proj = weights["gate_proj"]  # type: ignore[assignment]
    position_bias = weights["position_bias"]  # type: ignore[assignment]
    kv_norm = weights["kv_norm"]  # type: ignore[assignment]

    seq_len = len(hidden_states)
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate
    if n_windows == 0:
        return []
    if len(rope_cos) < n_windows or len(rope_sin) < n_windows:
        raise ValueError("rope_cos/sin must cover n_windows")

    kvs = [_linear(token, kv_proj) for token in hidden_states[:usable]]  # type: ignore[arg-type]
    gates = [_linear(token, gate_proj) for token in hidden_states[:usable]]  # type: ignore[arg-type]
    head_dim = len(kvs[0])
    if any(len(row) != head_dim for row in kvs) or any(len(row) != head_dim for row in gates):
        raise ValueError("kv/gate projection dimensions must be consistent")
    if len(position_bias) != compress_rate or any(len(row) != head_dim for row in position_bias):
        raise ValueError("position_bias shape must be [compress_rate, head_dim]")

    compressed: list[list[float]] = []
    for w in range(n_windows):
        window_kv = kvs[w * compress_rate : (w + 1) * compress_rate]
        window_gate = gates[w * compress_rate : (w + 1) * compress_rate]
        # Add position bias and softmax independently per head dimension.
        probs_per_dim: list[list[float]] = []
        for d in range(head_dim):
            biased = [window_gate[t][d] + position_bias[t][d] for t in range(compress_rate)]
            probs_per_dim.append(_softmax(biased))
        pooled = [sum(probs_per_dim[d][t] * window_kv[t][d] for t in range(compress_rate)) for d in range(head_dim)]
        normed = _rms_norm(pooled, kv_norm, rms_norm_eps)
        position = first_window_position + w * compress_rate
        # HCA applies RoPE to the full head (no nope split) at absolute position.
        roped = apply_rope_tail(normed, nope_dim=0, cos=rope_cos[w], sin=rope_sin[w])
        _ = position  # position used implicitly by caller-provided cos/sin
        compressed.append(roped)
    return compressed


def run_tiny_hca_compressor_fixture() -> dict[str, object]:
    """HCA compressor parity fixture against Transformers reference.

    Builds a tiny config with `heavily_compressed_attention` rate 2, runs the
    Transformers `DeepseekV4HCACompressor` module on deterministic weights, and
    compares the pure-Python helper output.  This is partial evidence only:
    CSA compressor overlap, indexer scoring, cache state, and full layer
    integration remain fail-closed.
    """

    try:
        import torch
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "fixture": "hca-compressor-forward",
            "status": "skipped",
            "max_abs_error": None,
            "reason": f"Transformers/torch reference unavailable: {exc}",
        }

    hidden_size = 4
    head_dim = 4
    compress_rate = 2
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=1,
        compress_rates={"heavily_compressed_attention": compress_rate, "compressed_sparse_attention": compress_rate},
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0},
            "compress": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    module = DeepseekV4HCACompressor(config)
    module.eval()

    # Deterministic tiny weights.
    with torch.no_grad():
        module.kv_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32))
        module.gate_proj.weight.copy_(torch.eye(hidden_size, dtype=torch.float32))
        module.position_bias.copy_(torch.zeros(compress_rate, head_dim, dtype=torch.float32))
        module.kv_norm.weight.copy_(torch.ones(head_dim, dtype=torch.float32))

    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    q_residual = torch.zeros(1, 4, 1, dtype=torch.float32)  # not used by HCA compressor

    with torch.no_grad():
        ref_compressed_kv, _ = module(hidden, q_residual, position_ids, past_key_values=None, layer_idx=0)
    ref = ref_compressed_kv.squeeze(0).squeeze(0).tolist()  # [n_windows, head_dim]

    n_windows = len(ref)
    positions = torch.arange(n_windows, dtype=torch.long) * compress_rate
    with torch.no_grad():
        cos, sin = module.rotary_emb(hidden, position_ids=positions.unsqueeze(0), layer_type=module.rope_layer_type)
    # cos/sin from rotary_emb are half-length; expand to full head dim.
    cos_full = cos.repeat_interleave(2, dim=-1).squeeze(0).tolist()
    sin_full = sin.repeat_interleave(2, dim=-1).squeeze(0).tolist()

    weights = {
        # DS4 layout: [input_dim, output_dim]
        "kv_proj": module.kv_proj.weight.T.tolist(),
        "gate_proj": module.gate_proj.weight.T.tolist(),
        "position_bias": module.position_bias.tolist(),
        "kv_norm": module.kv_norm.weight.tolist(),
    }
    got = tiny_hca_compressor_forward(
        hidden.squeeze(0).tolist(),
        weights,
        compress_rate=compress_rate,
        rms_norm_eps=1e-6,
        rope_cos=cos_full,
        rope_sin=sin_full,
    )

    if len(got) != len(ref) or any(len(got_row) != len(ref_row) for got_row, ref_row in zip(got, ref)):
        return {
            "fixture": "hca-compressor-forward",
            "status": "failed",
            "max_abs_error": None,
            "reason": f"shape mismatch: got {len(got)}x{len(got[0]) if got else 0}, expected {len(ref)}x{len(ref[0]) if ref else 0}",
            "covered": [],
            "not_covered": ["CSA compressor overlap", "CSA indexer scoring", "stateful cache update", "full layer integration"],
        }
    max_abs_error = 0.0
    for got_row, ref_row in zip(got, ref):
        for gv, rv in zip(got_row, ref_row):
            max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))

    return {
        "fixture": "hca-compressor-forward",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "HCA compressor kv/gate projection",
            "windowed softmax-gated pooling",
            "RMSNorm on compressed entries",
            "interleaved RoPE at absolute window positions",
            "Transformers DeepseekV4HCACompressor module reference",
        ],
        "not_covered": [
            "CSA compressor Ca/Cb overlap",
            "CSA indexer scoring and top-k",
            "stateful cache update",
            "full DeepseekV4Model layer integration",
        ],
    }


def tiny_csa_compressor_forward(
    hidden_states: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    compress_rate: int,
    rms_norm_eps: float,
    rope_cos: list[list[float]],
    rope_sin: list[list[float]],
    first_window_position: int = 0,
) -> list[list[float]]:
    """Deterministic CSA compressor forward for a single stateless chunk.

    Mirrors Transformers `DeepseekV4CSACompressor.forward` when no cache is
    supplied and `past_key_values is None`: project KV and gate to
    `2 * head_dim`, split into Ca (first head_dim) and Cb (last head_dim),
    assemble overlapping windows of width `2 * compress_rate`, softmax-gate
    each window into one compressed entry, RMSNorm, and apply interleaved RoPE
    at deterministic absolute positions.
    """

    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    required = {"kv_proj", "gate_proj", "position_bias", "kv_norm"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing CSA compressor weights: {', '.join(missing)}")
    kv_proj = weights["kv_proj"]  # type: ignore[assignment]
    gate_proj = weights["gate_proj"]  # type: ignore[assignment]
    position_bias = weights["position_bias"]  # type: ignore[assignment]
    kv_norm = weights["kv_norm"]  # type: ignore[assignment]

    seq_len = len(hidden_states)
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate
    if n_windows == 0:
        return []
    if len(rope_cos) < n_windows or len(rope_sin) < n_windows:
        raise ValueError("rope_cos/sin must cover n_windows")

    kvs = [_linear(token, kv_proj) for token in hidden_states[:usable]]
    gates = [_linear(token, gate_proj) for token in hidden_states[:usable]]
    out_dim = len(kvs[0])
    if out_dim % 2 != 0:
        raise ValueError("CSA compressor projection must be divisible by 2")
    head_dim = out_dim // 2
    if any(len(row) != out_dim for row in kvs) or any(len(row) != out_dim for row in gates):
        raise ValueError("kv/gate projection dimensions must be consistent")
    if len(position_bias) != compress_rate or any(len(row) != out_dim for row in position_bias):
        raise ValueError("position_bias shape must be [compress_rate, 2*head_dim]")

    # Add position bias to every token's gate in row-major layout.
    biased_gates: list[list[float]] = []
    for idx, gate in enumerate(gates):
        pos = idx % compress_rate
        biased_gates.append([float(gate[d]) + float(position_bias[pos][d]) for d in range(out_dim)])

    compressed: list[list[float]] = []
    for w in range(n_windows):
        # Assemble new_kv / new_gate of shape [2*compress_rate, head_dim].
        new_kv: list[list[float]] = [[0.0] * head_dim for _ in range(2 * compress_rate)]
        new_gate: list[list[float]] = [[float("-inf")] * head_dim for _ in range(2 * compress_rate)]

        # Second half: current window Cb.
        for t in range(compress_rate):
            token_kv = kvs[w * compress_rate + t]
            token_gate = biased_gates[w * compress_rate + t]
            for d in range(head_dim):
                new_kv[compress_rate + t][d] = float(token_kv[head_dim + d])
                new_gate[compress_rate + t][d] = float(token_gate[head_dim + d])

        # First half: previous window Ca.
        if w > 0:
            for t in range(compress_rate):
                token_kv = kvs[(w - 1) * compress_rate + t]
                token_gate = biased_gates[(w - 1) * compress_rate + t]
                for d in range(head_dim):
                    new_kv[t][d] = float(token_kv[d])
                    new_gate[t][d] = float(token_gate[d])

        # Softmax-gate per head dimension over 2*compress_rate entries.
        pooled = []
        for d in range(head_dim):
            probs = _softmax([new_gate[i][d] for i in range(2 * compress_rate)])
            pooled.append(sum(probs[i] * new_kv[i][d] for i in range(2 * compress_rate)))
        normed = _rms_norm(pooled, kv_norm, rms_norm_eps)
        roped = apply_rope_tail(normed, nope_dim=0, cos=rope_cos[w], sin=rope_sin[w])
        compressed.append(roped)
    return compressed


def run_tiny_csa_compressor_fixture() -> dict[str, object]:
    """CSA compressor parity fixture against Transformers reference.

    Builds a tiny config with `compressed_sparse_attention` rate 2, runs the
    Transformers `DeepseekV4CSACompressor` module on deterministic weights, and
    compares the pure-Python helper output for the compressor's compressed KV.
    This is partial evidence only: CSA indexer scoring/top-k, cached overlap,
    and full layer integration remain fail-closed.
    """

    try:
        import torch
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "fixture": "csa-compressor-forward",
            "status": "skipped",
            "max_abs_error": None,
            "reason": f"Transformers/torch reference unavailable: {exc}",
        }

    hidden_size = 4
    head_dim = 4
    compress_rate = 2
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=1,
        q_lora_rank=hidden_size,
        qk_rope_head_dim=head_dim,
        compress_rates={"compressed_sparse_attention": compress_rate},
        compress_rope_theta=10000.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        index_n_heads=1,
        index_head_dim=head_dim,
        index_topk=1,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0},
            "compress": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    module = DeepseekV4CSACompressor(config)
    module.eval()

    with torch.no_grad():
        eye = torch.eye(hidden_size, dtype=torch.float32)
        module.kv_proj.weight.copy_(torch.cat([eye, eye], dim=0))
        module.gate_proj.weight.copy_(torch.cat([eye, eye], dim=0))
        module.position_bias.copy_(torch.zeros(compress_rate, 2 * head_dim, dtype=torch.float32))
        module.kv_norm.weight.copy_(torch.ones(head_dim, dtype=torch.float32))

    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    q_residual = torch.zeros(1, 4, hidden_size, dtype=torch.float32)

    with torch.no_grad():
        ref_compressed_kv, _ = module(hidden, q_residual, position_ids, past_key_values=None, layer_idx=0)
    ref = ref_compressed_kv.squeeze(0).squeeze(0).tolist()  # [n_windows, head_dim]

    n_windows = len(ref)
    positions = torch.arange(n_windows, dtype=torch.long) * compress_rate
    with torch.no_grad():
        cos, sin = module.rotary_emb(hidden, position_ids=positions.unsqueeze(0), layer_type=module.rope_layer_type)
    cos_full = cos.repeat_interleave(2, dim=-1).squeeze(0).tolist()
    sin_full = sin.repeat_interleave(2, dim=-1).squeeze(0).tolist()

    weights = {
        "kv_proj": module.kv_proj.weight.T.tolist(),
        "gate_proj": module.gate_proj.weight.T.tolist(),
        "position_bias": module.position_bias.tolist(),
        "kv_norm": module.kv_norm.weight.tolist(),
    }
    got = tiny_csa_compressor_forward(
        hidden.squeeze(0).tolist(),
        weights,
        compress_rate=compress_rate,
        rms_norm_eps=1e-6,
        rope_cos=cos_full,
        rope_sin=sin_full,
    )

    if len(got) != len(ref) or any(len(got_row) != len(ref_row) for got_row, ref_row in zip(got, ref)):
        return {
            "fixture": "csa-compressor-forward",
            "status": "failed",
            "max_abs_error": None,
            "reason": f"shape mismatch: got {len(got)}x{len(got[0]) if got else 0}, expected {len(ref)}x{len(ref[0]) if ref else 0}",
            "covered": [],
            "not_covered": ["CSA indexer scoring", "cached overlap state", "full layer integration"],
        }
    max_abs_error = 0.0
    for got_row, ref_row in zip(got, ref):
        for gv, rv in zip(got_row, ref_row):
            max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))

    return {
        "fixture": "csa-compressor-forward",
        "status": "ok" if max_abs_error <= 1e-5 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "CSA compressor kv/gate projection",
            "CSA compressor Ca/Cb overlap",
            "windowed softmax-gated pooling",
            "RMSNorm on compressed entries",
            "interleaved RoPE at absolute window positions",
            "Transformers DeepseekV4CSACompressor module reference",
        ],
        "not_covered": [
            "CSA indexer scoring and top-k",
            "cached overlap state",
            "full DeepseekV4Model layer integration",
        ],
    }


def _rope_cos_sin(positions: Sequence[int], head_dim: int, theta: float) -> tuple[list[list[float]], list[list[float]]]:
    """Compute repeated full-length RoPE cos/sin for a list of positions."""

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    inv_freq = [1.0 / (theta ** ((2 * i) / head_dim)) for i in range(half)]
    cos: list[list[float]] = []
    sin: list[list[float]] = []
    for pos in positions:
        freqs = [pos * freq for freq in inv_freq]
        cos_row: list[float] = []
        sin_row: list[float] = []
        for c, s in zip((math.cos(f) for f in freqs), (math.sin(f) for f in freqs)):
            cos_row.extend([c, c])
            sin_row.extend([s, s])
        cos.append(cos_row)
        sin.append(sin_row)
    return cos, sin


def _apply_rotary_full(vector: list[float], cos: list[float], sin: list[float]) -> list[float]:
    """Apply pairwise RoPE rotation to a full head vector."""

    if len(vector) % 2 != 0:
        raise ValueError("vector length must be even")
    if len(cos) != len(vector) or len(sin) != len(vector):
        raise ValueError("cos/sin length must match vector length")
    out: list[float] = []
    for i in range(0, len(vector), 2):
        x0 = vector[i]
        x1 = vector[i + 1]
        out.append(x0 * cos[i] - x1 * sin[i])
        out.append(x1 * cos[i + 1] + x0 * sin[i + 1])
    return out


def tiny_csa_indexer_forward(
    hidden_states: list[list[float]],
    q_residual: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
    rope_theta: float,
    position_ids: list[int],
) -> dict[str, object]:
    """Deterministic CSA Lightning indexer scoring/top-k forward.

    Mirrors Transformers `DeepseekV4Indexer.forward` with no cache:
    compress hidden states into indexer KV entries, apply RoPE to compressed
    keys and to queries, score with the Lightning scorer, apply causal masking,
    and return top-k indices.  This is partial evidence only: cache overlap,
    full CSA attention, and layer integration remain fail-closed.
    """

    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    if index_n_heads <= 0 or index_head_dim <= 0 or index_topk <= 0:
        raise ValueError("indexer dimensions must be positive")
    required = {"kv_proj", "gate_proj", "position_bias", "kv_norm", "q_b_proj", "weights_proj"}
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing CSA indexer weights: {', '.join(missing)}")

    kv_proj = weights["kv_proj"]  # type: ignore[assignment]
    gate_proj = weights["gate_proj"]  # type: ignore[assignment]
    position_bias = weights["position_bias"]  # type: ignore[assignment]
    kv_norm = weights["kv_norm"]  # type: ignore[assignment]
    q_b_proj = weights["q_b_proj"]  # type: ignore[assignment]
    weights_proj = weights["weights_proj"]  # type: ignore[assignment]

    seq_len = len(hidden_states)
    if len(q_residual) != seq_len or len(position_ids) != seq_len:
        raise ValueError("hidden_states, q_residual, and position_ids must have the same sequence length")

    kv = [_linear(token, kv_proj) for token in hidden_states]
    gate = [_linear(token, gate_proj) for token in hidden_states]
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate

    compressed: list[list[float]] = []
    for w in range(n_windows):
        new_kv: list[list[float]] = [[0.0] * index_head_dim for _ in range(2 * compress_rate)]
        new_gate: list[list[float]] = [[float("-inf")] * index_head_dim for _ in range(2 * compress_rate)]
        for t in range(compress_rate):
            slot_cb = compress_rate + t
            for d in range(index_head_dim):
                new_kv[slot_cb][d] = kv[w * compress_rate + t][index_head_dim + d]
                new_gate[slot_cb][d] = gate[w * compress_rate + t][index_head_dim + d] + position_bias[t][index_head_dim + d]
            if w > 0:
                slot_ca = t
                for d in range(index_head_dim):
                    new_kv[slot_ca][d] = kv[(w - 1) * compress_rate + t][d]
                    new_gate[slot_ca][d] = gate[(w - 1) * compress_rate + t][d] + position_bias[t][d]
        pooled: list[float] = []
        for d in range(index_head_dim):
            probs = _softmax([new_gate[i][d] for i in range(2 * compress_rate)])
            pooled.append(sum(probs[i] * new_kv[i][d] for i in range(2 * compress_rate)))
        normed = _rms_norm(pooled, kv_norm, rms_norm_eps)
        cos, sin = _rope_cos_sin([w * compress_rate], index_head_dim, rope_theta)
        compressed.append(_apply_rotary_full(normed, cos[0], sin[0]))

    compressed_len = len(compressed)

    q = [_linear(token, q_b_proj) for token in q_residual]
    q_heads: list[list[list[float]]] = []
    for t in range(seq_len):
        expected_q_dim = index_n_heads * index_head_dim
        if len(q[t]) != expected_q_dim:
            raise ValueError(f"q_b_proj output dimension mismatch at token {t}: {len(q[t])} != {expected_q_dim}")
        heads: list[list[float]] = []
        for h in range(index_n_heads):
            head = q[t][h * index_head_dim : (h + 1) * index_head_dim]
            cos, sin = _rope_cos_sin([position_ids[t]], index_head_dim, rope_theta)
            heads.append(_apply_rotary_full(head, cos[0], sin[0]))
        q_heads.append(heads)

    weights_scaling = index_n_heads ** -0.5
    scorer_weights = [[v * weights_scaling for v in _linear(token, weights_proj)] for token in hidden_states]
    softmax_scale = index_head_dim ** -0.5

    scores: list[list[float]] = []
    for t in range(seq_len):
        row_scores: list[float] = []
        for c_idx in range(compressed_len):
            score = 0.0
            for h in range(index_n_heads):
                dot = sum(q * k for q, k in zip(q_heads[t][h], compressed[c_idx])) * softmax_scale
                score += max(0.0, dot) * scorer_weights[t][h]
            row_scores.append(score)
        scores.append(row_scores)

    top_k = min(index_topk, compressed_len)
    topk_indices: list[list[int]] = []
    topk_mask: list[list[bool]] = []
    for t in range(seq_len):
        threshold = (position_ids[t] + 1) // compress_rate
        candidates = [(scores[t][c_idx], c_idx) for c_idx in range(compressed_len) if c_idx < threshold]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = [c_idx for _, c_idx in candidates[:top_k]]
        padded = selected + [-1] * (top_k - len(selected))
        mask = [True] * len(selected) + [False] * (top_k - len(selected))
        topk_indices.append(padded)
        topk_mask.append(mask)

    return {
        "scores": scores,
        "topk_indices": topk_indices,
        "topk_mask": topk_mask,
        "compressed_len": compressed_len,
    }


def run_tiny_csa_indexer_scorer_fixture() -> dict[str, object]:
    """CSA indexer scoring/top-k parity fixture against Transformers reference.

    Builds a tiny config with `compressed_sparse_attention` rate 2 and a small
    Lightning indexer, runs the Transformers `DeepseekV4Indexer` module on
    deterministic weights, and compares the pure-Python helper's scores and
    top-k indices.  This is partial evidence only: cache overlap and full CSA
    attention remain fail-closed.
    """

    try:
        import torch
        import torch.nn.functional as F
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "fixture": "csa-indexer-scorer-topk",
            "status": "skipped",
            "max_abs_error": None,
            "reason": f"Transformers/torch reference unavailable: {exc}",
        }

    hidden_size = 4
    q_lora_rank = hidden_size
    compress_rate = 2
    index_n_heads = 2
    index_head_dim = 2
    index_topk = 2
    rope_theta = 10000.0
    rms_norm_eps = 1e-6

    # Transformers DeepseekV4Indexer creates its own rotary embedding using
    # config.head_dim as the rotary dimension.  For the indexer module to match
    # index_head_dim we set head_dim == index_head_dim and leave the default
    # partial_rotary_factor (1.0) so the RoPE is applied to the full head.
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        head_dim=index_head_dim,
        num_attention_heads=1,
        q_lora_rank=q_lora_rank,
        qk_rope_head_dim=index_head_dim,
        compress_rates={"compressed_sparse_attention": compress_rate},
        compress_rope_theta=rope_theta,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": rope_theta},
            "compress": {"rope_type": "default", "rope_theta": rope_theta},
        },
    )
    module = DeepseekV4Indexer(config)
    module.eval()

    eye = torch.eye(hidden_size, dtype=torch.float32)
    with torch.no_grad():
        module.kv_proj.weight.copy_(torch.cat([eye[:index_head_dim], eye[:index_head_dim]], dim=0))
        module.gate_proj.weight.copy_(torch.cat([eye[:index_head_dim], eye[:index_head_dim]], dim=0))
        module.position_bias.copy_(torch.zeros(compress_rate, 2 * index_head_dim, dtype=torch.float32))
        module.kv_norm.weight.copy_(torch.ones(index_head_dim, dtype=torch.float32))
        module.q_b_proj.weight.copy_(torch.eye(index_n_heads * index_head_dim, q_lora_rank, dtype=torch.float32))
        module.scorer.weights_proj.weight.copy_(eye[:index_n_heads])

    # Capture reference scores via a thin wrapper around the scorer.
    original_scorer_forward = module.scorer.forward
    captured_scores: list[torch.Tensor] = []

    def _scorer_wrapper(q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = original_scorer_forward(q, compressed_kv, hidden_states)
        captured_scores.append(scores.detach().clone())
        return scores

    module.scorer.forward = _scorer_wrapper  # type: ignore[method-assign]

    hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
    position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    q_residual = hidden.clone()

    with torch.no_grad():
        ref_topk = module(hidden, q_residual, position_ids, past_key_values=None, layer_idx=0)
    ref_topk_list = ref_topk.squeeze(0).tolist()  # [seq_len, top_k]

    if not captured_scores:
        return {
            "fixture": "csa-indexer-scorer-topk",
            "status": "failed",
            "max_abs_error": None,
            "reason": "could not capture reference scorer scores",
            "covered": [],
            "not_covered": ["stateful cache overlap", "full CSA attention", "full layer integration"],
        }
    ref_scores = captured_scores[0].squeeze(0).tolist()  # [seq_len, compressed_len]

    weights = {
        "kv_proj": module.kv_proj.weight.T.tolist(),
        "gate_proj": module.gate_proj.weight.T.tolist(),
        "position_bias": module.position_bias.tolist(),
        "kv_norm": module.kv_norm.weight.tolist(),
        "q_b_proj": module.q_b_proj.weight.T.tolist(),
        "weights_proj": module.scorer.weights_proj.weight.T.tolist(),
    }
    got = tiny_csa_indexer_forward(
        hidden.squeeze(0).tolist(),
        q_residual.squeeze(0).tolist(),
        weights,
        compress_rate=compress_rate,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        position_ids=[0, 1, 2, 3],
    )

    if (
        len(got["scores"]) != len(ref_scores)
        or any(len(got_row) != len(ref_row) for got_row, ref_row in zip(got["scores"], ref_scores))
    ):
        return {
            "fixture": "csa-indexer-scorer-topk",
            "status": "failed",
            "max_abs_error": None,
            "reason": f"score matrix shape mismatch: got {len(got['scores'])}x{len(got['scores'][0]) if got['scores'] else 0}, expected {len(ref_scores)}x{len(ref_scores[0]) if ref_scores else 0}",
            "covered": [],
            "not_covered": ["stateful cache overlap", "full CSA attention", "full layer integration"],
        }
    max_score_error = 0.0
    for got_row, ref_row in zip(got["scores"], ref_scores):
        for gv, rv in zip(got_row, ref_row):
            max_score_error = max(max_score_error, abs(float(gv) - float(rv)))
    indices_match = got["topk_indices"] == ref_topk_list

    return {
        "fixture": "csa-indexer-scorer-topk",
        "status": "ok" if (max_score_error <= 1e-5 and indices_match) else "failed",
        "max_abs_error": max_score_error if indices_match else float("inf"),
        "covered": [
            "CSA indexer kv/gate projection",
            "CSA indexer Ca/Cb overlap compression",
            "RMSNorm on compressed indexer entries",
            "interleaved RoPE on compressed keys and queries",
            "Lightning indexer scoring",
            "causal top-k selection",
            "Transformers DeepseekV4Indexer module reference",
        ],
        "not_covered": [
            "stateful cache overlap",
            "full CSA attention",
            "full DeepseekV4Model layer integration",
        ],
    }


def tiny_compressor_indexer_attention_reference(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    q_residual: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    position_ids: list[int] | None = None,
    index_topk: int | None = None,
) -> dict[str, object]:
    """Integrated CSA compressor + Lightning indexer + causal block-bias ref.

    This is an end-to-end *synthetic* compressed-attention reference that
    composes the Transformers-validated component references and produces an
    attention output per query.  It exercises, with the spec's own weight names
    from :meth:`compressor_shapes` / :meth:`indexer_shapes`:

    1. **CSA compressor** -- project hidden states to ``2 * head_dim`` KV and
       gate, assemble Ca/Cb overlapping windows of width ``2 * compress_rate``
       (Ca = previous window's ``[:head_dim]``, Cb = current window's
       ``[head_dim:]``), **softmax-gate** each window into one compressed
       entry, RMSNorm, and apply interleaved RoPE at absolute window
       positions.  Output ``compressed_kv`` is ``[n_windows, head_dim]``.
    2. **Lightning indexer** -- a downscaled compressor at ``index_head_dim``
       produces compressed indexer keys; the index query comes from
       ``indexer_wq_b`` (per-head ``index_head_dim``), RoPE-applied; the scorer
       is ``sum_h w_{t,h} * ReLU(q . k) * softmax_scale`` with per-token head
       weights from ``indexer_proj``.  Output ``index_scores`` is
       ``[seq_len, n_windows]``.
    3. **Causal block-bias** -- ``hca_block_bias`` masks compressed entries
       ``entry_idx >= (position + 1) // compress_rate`` (the causal compressed
       block threshold), combined with the indexer top-k gather mask.
    4. **Compressed attention output** -- per query, softmax over the valid
       (un-masked) entries of ``index_scores`` and weighted-sum ``compressed_kv``
       into ``attended`` of shape ``[seq_len, head_dim]``.  Rows whose entries
       are all masked (early queries) return zeros.

    Gating note (deviation from the slice brief): the brief said "sigmoid
    gating", but the trusted reference -- Transformers
    ``DeepseekV4CSACompressor.forward`` (``new_gate.softmax(dim=2)``) and the
    already-validated ``tiny_csa_compressor_forward`` / ``tiny_hca_compressor_forward``
    fixtures -- uses per-dimension **softmax** windowed gating.  ``sigmoid``
    appears only in the HyperConnection/HyperHead path in Transformers, never in
    the compressor.  This reference therefore reuses the softmax-gated validated
    compressor for consistency and verifiability; see
    ``agent-output/story-11-compressor-indexer.md``.

    This is partial evidence only: stateful cache overlap, full core attention
    (main ``q_a``/``q_b`` query heads against compressed KV, grouped output),
    HCA rate-128, and full layer integration remain fail-closed.
    """

    if spec.compression_ratio != 4:
        raise ValueError("integrated CSA attention reference requires compression_ratio=4 (CSA)")
    if spec.index_n_heads is None or spec.index_head_dim is None:
        raise ValueError("integrated CSA attention reference requires index_n_heads/index_head_dim")
    compress_rate = spec.compression_ratio
    index_n_heads = spec.index_n_heads
    index_head_dim = spec.index_head_dim

    required = {
        "compressor_wkv", "compressor_wgate", "compressor_ape", "compressor_norm",
        "indexer_wq_b", "indexer_proj",
        "indexer_compressor_wkv", "indexer_compressor_wgate", "indexer_compressor_ape", "indexer_compressor_norm",
    }
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing CSA attention fixture weights: {', '.join(missing)}")

    seq_len = len(hidden_states)
    if len(q_residual) != seq_len:
        raise ValueError("hidden_states and q_residual must have the same sequence length")
    if position_ids is None:
        position_ids = list(range(seq_len))
    if len(position_ids) != seq_len:
        raise ValueError("position_ids length must match sequence length")

    # --- 1. CSA compressor: translate spec-name weights -> internal names ---
    head_dim = spec.head_dim
    width = 2 * head_dim
    compressor_internal = {
        "kv_proj": weights["compressor_wkv"],  # type: ignore[dict-item]
        "gate_proj": weights["compressor_wgate"],  # type: ignore[dict-item]
        # compressor_ape is (width, ratio); internal position_bias is [ratio, width].
        "position_bias": [
            [weights["compressor_ape"][d][t] for d in range(width)]  # type: ignore[index]
            for t in range(compress_rate)
        ],
        "kv_norm": weights["compressor_norm"],  # type: ignore[dict-item]
    }
    n_windows = (seq_len // compress_rate)
    compressor_positions = [w * compress_rate for w in range(n_windows)]
    cos_c, sin_c = _rope_cos_sin(compressor_positions, head_dim, rope_theta)
    compressed_kv = tiny_csa_compressor_forward(
        hidden_states,
        compressor_internal,
        compress_rate=compress_rate,
        rms_norm_eps=rms_norm_eps,
        rope_cos=cos_c,
        rope_sin=sin_c,
    )
    compressed_len = len(compressed_kv)

    # --- 2. Lightning indexer: translate spec-name weights -> internal names ---
    index_width = 2 * index_head_dim
    index_q_dim = index_n_heads * index_head_dim
    indexer_internal = {
        "kv_proj": weights["indexer_compressor_wkv"],  # type: ignore[dict-item]
        "gate_proj": weights["indexer_compressor_wgate"],  # type: ignore[dict-item]
        "position_bias": [
            [weights["indexer_compressor_ape"][d][t] for d in range(index_width)]  # type: ignore[index]
            for t in range(compress_rate)
        ],
        "kv_norm": weights["indexer_compressor_norm"],  # type: ignore[dict-item]
        "q_b_proj": weights["indexer_wq_b"],  # type: ignore[dict-item]
        "weights_proj": weights["indexer_proj"],  # type: ignore[dict-item]
    }
    indexer_result = tiny_csa_indexer_forward(
        hidden_states,
        q_residual,
        indexer_internal,
        compress_rate=compress_rate,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk if index_topk is not None else max(1, compressed_len),
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        position_ids=position_ids,
    )
    index_scores = [list(map(float, row)) for row in indexer_result["scores"]]  # type: ignore[index]
    topk_indices = indexer_result["topk_indices"]  # type: ignore[index]
    topk_mask = indexer_result["topk_mask"]  # type: ignore[index]

    # --- 3. Causal block-bias (hca_block_bias) combined with indexer top-k ---
    causal_bias = hca_block_bias(
        position_ids=position_ids, compressed_len=compressed_len, compression_ratio=compress_rate
    )
    block_bias: list[list[float]] = []
    for t in range(seq_len):
        if compressed_len == 0:
            block_bias.append([])
            continue
        causal_row = [float("-inf")] * compressed_len if causal_bias is None else [float(v) for v in causal_bias[t]]
        # Gather mask: an entry is indexer-valid iff it appears among this row's
        # top-k indices (mask True) and is a real (>= 0) index.
        valid_entries = {
            int(topk_indices[t][k])  # type: ignore[index]
            for k in range(len(topk_indices[t]))  # type: ignore[index]
            if bool(topk_mask[t][k]) and int(topk_indices[t][k]) >= 0  # type: ignore[index]
        }
        row = []
        for entry in range(compressed_len):
            if entry not in valid_entries:
                row.append(float("-inf"))
            else:
                row.append(causal_row[entry])
        block_bias.append(row)

    # --- 4. Compressed attention output: softmax(scores + bias) @ compressed_kv ---
    attended: list[list[float]] = []
    for t in range(seq_len):
        if compressed_len == 0:
            attended.append([0.0] * head_dim)
            continue
        masked = [float(index_scores[t][entry]) + block_bias[t][entry] for entry in range(compressed_len)]
        if all(math.isinf(v) and v < 0 for v in masked):
            # Early queries with no causally-ready compressed entry.
            attended.append([0.0] * head_dim)
            continue
        probs = _softmax(masked)
        out = [
            sum(probs[entry] * float(compressed_kv[entry][d]) for entry in range(compressed_len))
            for d in range(head_dim)
        ]
        attended.append(out)

    return {
        "compressed_kv": compressed_kv,
        "index_scores": index_scores,
        "block_bias": block_bias,
        "topk_indices": topk_indices,
        "topk_mask": topk_mask,
        "attended": attended,
        "compressed_len": compressed_len,
    }


def tiny_multihead_csa_fusion_reference(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    sliding_window: int = 0,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    position_ids: list[int] | None = None,
    index_topk: int | None = None,
) -> dict[str, object]:
    """Stateless single-chunk multi-head CSA fusion reference.

    This mirrors the real ``DeepseekV4Attention`` CSA forward at the attention
    module boundary for a tiny, cache-less chunk: project multi-head queries and
    one shared MQA KV head, run the already-proven CSA compressor/indexer to get
    compressed KV plus a per-query ``0/-inf`` block-bias mask, concatenate
    ``[sliding_kv | compressed_kv]`` on the KV axis, apply per-head sink softmax
    to masked logits, undo output RoPE on each head, then use the grouped output
    projection.  The boundary is deliberately stateless: no ``StatefulCSACache``
    gate or vendor runtime gate is lifted here.
    """

    if spec.compression_ratio != 4:
        raise ValueError("multi-head CSA fusion reference requires compression_ratio=4")
    if spec.index_n_heads is None or spec.index_head_dim is None:
        raise ValueError("multi-head CSA fusion reference requires index_n_heads/index_head_dim")
    _validate_tail_rope_dim(spec)
    if sliding_window < 0:
        raise ValueError("sliding_window must be non-negative")

    seq_len = len(hidden_states)
    if position_ids is None:
        position_ids = list(range(seq_len))
    if len(position_ids) != seq_len:
        raise ValueError("position_ids length must match sequence length")
    if any(position < 0 for position in position_ids):
        raise ValueError("position_ids must be non-negative")
    if any(len(row) != spec.hidden_size for row in hidden_states):
        raise ValueError("hidden_states width must match hidden_size")

    core_required = {
        "q_a_proj.weight", "q_norm.weight", "q_b_proj.weight", "kv_proj.weight",
        "kv_norm.weight", "o_a_proj.weight", "o_b_proj.weight", "sinks",
    }
    csa_required = {
        "compressor_wkv", "compressor_wgate", "compressor_ape", "compressor_norm",
        "indexer_wq_b", "indexer_proj", "indexer_compressor_wkv", "indexer_compressor_wgate",
        "indexer_compressor_ape", "indexer_compressor_norm",
    }
    missing = sorted((core_required | csa_required) - set(weights))
    if missing:
        raise ValueError(f"missing multi-head CSA fusion weights: {', '.join(missing)}")

    core_shapes = spec.flash_mlx_safetensors_shapes()
    matrix_shapes = {
        "q_a_proj.weight": core_shapes["q_a_proj.weight"],
        "q_b_proj.weight": core_shapes["q_b_proj.weight"],
        "kv_proj.weight": core_shapes["kv_proj.weight"],
        "o_a_proj.weight": core_shapes["o_a_proj.weight"],
        "o_b_proj.weight": core_shapes["o_b_proj.weight"],
        "compressor_wkv": spec.compressor_shapes()["compressor_wkv"],
        "compressor_wgate": spec.compressor_shapes()["compressor_wgate"],
        "compressor_ape": spec.compressor_shapes()["compressor_ape"],
        "indexer_wq_b": spec.indexer_shapes()["indexer_wq_b"],
        "indexer_proj": spec.indexer_shapes()["indexer_proj"],
        "indexer_compressor_wkv": spec.indexer_shapes()["indexer_compressor_wkv"],
        "indexer_compressor_wgate": spec.indexer_shapes()["indexer_compressor_wgate"],
        "indexer_compressor_ape": spec.indexer_shapes()["indexer_compressor_ape"],
    }
    vector_shapes = {
        "q_norm.weight": core_shapes["q_norm.weight"],
        "kv_norm.weight": core_shapes["kv_norm.weight"],
        "sinks": core_shapes["sinks"],
        "compressor_norm": spec.compressor_shapes()["compressor_norm"],
        "indexer_compressor_norm": spec.indexer_shapes()["indexer_compressor_norm"],
    }
    for key, expected in matrix_shapes.items():
        matrix = weights[key]
        if not isinstance(matrix, list) or (len(matrix), len(matrix[0]) if matrix else 0) != expected:
            raise ValueError(f"{key} shape mismatch for multi-head CSA fusion reference")
    for key, expected in vector_shapes.items():
        vector = weights[key]
        if not isinstance(vector, list) or len(vector) != expected[0]:
            raise ValueError(f"{key} shape mismatch for multi-head CSA fusion reference")

    q_residual_by_pos: list[list[float]] = []
    q_heads_by_pos: list[list[list[float]]] = []
    sliding_kv_by_pos: list[list[float]] = []
    for hidden, position in zip(hidden_states, position_ids):
        q_a = _linear_out_in(hidden, weights["q_a_proj.weight"])  # type: ignore[arg-type]
        q_residual = _rms_norm(q_a, weights["q_norm.weight"], rms_norm_eps)  # type: ignore[arg-type]
        q_flat = _linear_out_in(q_residual, weights["q_b_proj.weight"])  # type: ignore[arg-type]
        q_heads = [
            _rms_norm(head, None, rms_norm_eps)
            for head in _split_heads(q_flat, num_heads=spec.num_attention_heads, head_dim=spec.head_dim)
        ]
        kv = _rms_norm(
            _linear_out_in(hidden, weights["kv_proj.weight"]),  # type: ignore[arg-type]
            weights["kv_norm.weight"],  # type: ignore[arg-type]
            rms_norm_eps,
        )
        cos, sin = _rope_cos_sin_for_position(position=int(position), rope_dim=spec.qk_rope_head_dim, theta=rope_theta)
        q_residual_by_pos.append(q_residual)
        q_heads_by_pos.append([apply_rope_tail(head, nope_dim=spec.qk_nope_head_dim, cos=cos, sin=sin) for head in q_heads])
        sliding_kv_by_pos.append(apply_rope_tail(kv, nope_dim=spec.qk_nope_head_dim, cos=cos, sin=sin))

    csa = tiny_compressor_indexer_attention_reference(
        spec,
        hidden_states,
        q_residual_by_pos,
        weights,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        position_ids=position_ids,
        index_topk=index_topk,
    )
    compressed_kv = [[float(value) for value in row] for row in csa["compressed_kv"]]  # type: ignore[index]
    block_bias = [[float(value) for value in row] for row in csa["block_bias"]]  # type: ignore[index]
    for row in block_bias:
        for value in row:
            if value != 0.0 and not (math.isinf(value) and value < 0.0):
                raise ValueError("CSA block_bias must be a 0/-inf additive mask")

    sinks = weights["sinks"]  # type: ignore[assignment]
    scale = spec.head_dim ** -0.5
    outputs: list[list[float]] = []
    for row_idx, (position, q_heads) in enumerate(zip(position_ids, q_heads_by_pos)):
        start = 0 if sliding_window == 0 else max(0, row_idx - sliding_window + 1)
        sliding_rows = sliding_kv_by_pos[start: row_idx + 1]
        kv_rows = sliding_rows + compressed_kv
        additive_mask = [0.0] * len(sliding_rows) + block_bias[row_idx]
        if len(kv_rows) != len(additive_mask):
            raise ValueError("KV rows and additive mask length mismatch")
        cos, sin = _rope_cos_sin_for_position(position=int(position), rope_dim=spec.qk_rope_head_dim, theta=rope_theta)
        attended_heads: list[list[float]] = []
        for head_idx, query in enumerate(q_heads):
            logits = [
                sum(q * k for q, k in zip(query, kv_row)) * scale + float(mask)
                for kv_row, mask in zip(kv_rows, additive_mask)
            ]
            probs = attention_scores_with_sink(logits, sink=float(sinks[head_idx]))
            attended = [
                sum(prob * kv_row[dim] for prob, kv_row in zip(probs, kv_rows))
                for dim in range(spec.head_dim)
            ]
            attended_heads.append(apply_output_inverse_rope_tail(attended, nope_dim=spec.qk_nope_head_dim, cos=cos, sin=sin))
        outputs.append(_grouped_output_projection(
            attended_heads,
            spec=spec,
            o_a_weight=weights["o_a_proj.weight"],  # type: ignore[arg-type]
            o_b_weight=weights["o_b_proj.weight"],  # type: ignore[arg-type]
        ))

    return {
        "output": outputs,
        "sliding_kv": sliding_kv_by_pos,
        "compressed_kv": compressed_kv,
        "block_bias": block_bias,
        "topk_indices": csa["topk_indices"],
        "topk_mask": csa["topk_mask"],
        "compressed_len": csa["compressed_len"],
        "covered": [
            "stateless multi-head CSA fusion",
            "[sliding|compressed] KV concatenation",
            "per-query 0/-inf CSA block-bias mask",
            "per-head attention sinks",
            "grouped output projection",
        ],
        "not_covered": [
            "stateful multi-head CSA cache",
            "vendor MLX runtime gate lift",
            "Track-B forward parity unblock",
        ],
    }


_STATEFUL_CSA_SUBSET_ERROR = "stateful CSA carry is proven only for the tiny single-head compression_ratio=4 subset (hc_mult>=1 supported)"


def _csa_internal_weights(
    spec: DeepSeekV4AttentionSpec,
    weights: dict[str, list[list[float]] | list[float]],
) -> tuple[dict[str, list[list[float]] | list[float]], dict[str, list[list[float]] | list[float]]]:
    required = {
        "compressor_wkv", "compressor_wgate", "compressor_ape", "compressor_norm",
        "indexer_wq_b", "indexer_proj",
        "indexer_compressor_wkv", "indexer_compressor_wgate", "indexer_compressor_ape", "indexer_compressor_norm",
    }
    missing = sorted(required - set(weights))
    if missing:
        raise ValueError(f"missing CSA attention fixture weights: {', '.join(missing)}")
    compress_rate = spec.compression_ratio
    head_dim = spec.head_dim
    width = 2 * head_dim
    index_head_dim = int(spec.index_head_dim or 0)
    index_width = 2 * index_head_dim
    compressor_internal = {
        "kv_proj": weights["compressor_wkv"],
        "gate_proj": weights["compressor_wgate"],
        "position_bias": [
            [weights["compressor_ape"][d][t] for d in range(width)]  # type: ignore[index]
            for t in range(compress_rate)
        ],
        "kv_norm": weights["compressor_norm"],
    }
    indexer_internal = {
        "kv_proj": weights["indexer_compressor_wkv"],
        "gate_proj": weights["indexer_compressor_wgate"],
        "position_bias": [
            [weights["indexer_compressor_ape"][d][t] for d in range(index_width)]  # type: ignore[index]
            for t in range(compress_rate)
        ],
        "kv_norm": weights["indexer_compressor_norm"],
        "q_b_proj": weights["indexer_wq_b"],
        "weights_proj": weights["indexer_proj"],
    }
    return compressor_internal, indexer_internal


def _compress_stateful_csa_window(
    current_window: list[list[float]],
    previous_window: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    compress_rate: int,
    head_dim: int,
    rms_norm_eps: float,
    rope_theta: float,
    window_index: int,
) -> list[float]:
    if len(current_window) != compress_rate:
        raise ValueError("current CSA window must be complete")
    if previous_window and len(previous_window) != compress_rate:
        raise ValueError("previous CSA window must be complete when present")
    # Reuse the cache-less compressor helper on the one-window or two-window
    # slice.  The last emitted block is exactly the current window with the
    # previous completed window as Ca overlap.
    local_hidden = [list(row) for row in previous_window] + [list(row) for row in current_window]
    local_windows = len(local_hidden) // compress_rate
    first_local_window = window_index - (local_windows - 1)
    cos, sin = _rope_cos_sin([(first_local_window + i) * compress_rate for i in range(local_windows)], head_dim, rope_theta)
    compressed = tiny_csa_compressor_forward(
        local_hidden,
        weights,
        compress_rate=compress_rate,
        rms_norm_eps=rms_norm_eps,
        rope_cos=cos,
        rope_sin=sin,
    )
    if not compressed:
        raise ValueError("complete CSA window did not emit a compressed block")
    return list(compressed[-1])


def _stateful_csa_score_token(
    *,
    hidden_token: list[float],
    q_residual_token: list[float],
    indexer_internal: dict[str, list[list[float]] | list[float]],
    compressed_index_keys: list[list[float]],
    emitted_compressed_kv: list[list[float]],
    position_id: int,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_theta: float,
    head_dim: int,
) -> dict[str, object]:
    q_b_proj = indexer_internal["q_b_proj"]
    weights_proj = indexer_internal["weights_proj"]
    q = _linear(q_residual_token, q_b_proj)  # type: ignore[arg-type]
    expected_q_dim = index_n_heads * index_head_dim
    if len(q) != expected_q_dim:
        raise ValueError(f"q_b_proj output dimension mismatch: {len(q)} != {expected_q_dim}")
    q_heads: list[list[float]] = []
    cos_q, sin_q = _rope_cos_sin([position_id], index_head_dim, rope_theta)
    for h in range(index_n_heads):
        head = q[h * index_head_dim : (h + 1) * index_head_dim]
        q_heads.append(_apply_rotary_full(head, cos_q[0], sin_q[0]))

    weights_scaling = index_n_heads ** -0.5
    scorer_weights = [v * weights_scaling for v in _linear(hidden_token, weights_proj)]  # type: ignore[arg-type]
    softmax_scale = index_head_dim ** -0.5
    scores: list[float] = []
    for compressed in compressed_index_keys:
        score = 0.0
        for h in range(index_n_heads):
            dot = sum(qv * kv for qv, kv in zip(q_heads[h], compressed)) * softmax_scale
            score += max(0.0, dot) * scorer_weights[h]
        scores.append(score)

    compressed_len = len(compressed_index_keys)
    threshold = (position_id + 1) // compress_rate
    candidates = [(scores[c_idx], c_idx) for c_idx in range(compressed_len) if c_idx < threshold]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [c_idx for _, c_idx in candidates[:index_topk]]
    topk_indices = selected + [-1] * (index_topk - len(selected))
    topk_mask = [True] * len(selected) + [False] * (index_topk - len(selected))
    valid_entries = set(selected)
    block_bias = [0.0 if entry in valid_entries and entry < threshold else float("-inf") for entry in range(compressed_len)]

    if emitted_compressed_kv and len(emitted_compressed_kv[0]) != head_dim:
        raise ValueError("emitted compressed KV width must match head_dim")
    if compressed_len == 0 or all(math.isinf(v) and v < 0 for v in block_bias):
        attended = [0.0] * head_dim
    else:
        masked = [float(scores[entry]) + block_bias[entry] for entry in range(compressed_len)]
        probs = _softmax(masked)
        attended = [
            sum(probs[entry] * float(emitted_compressed_kv[entry][d]) for entry in range(compressed_len))
            for d in range(head_dim)
        ]
    return {
        "index_scores": scores,
        "block_bias": block_bias,
        "topk_indices": topk_indices,
        "topk_mask": topk_mask,
        "attended": attended,
    }


class StatefulCSACache:
    """Tiny single-head ratio-4 CSA cache carrying one Ca window plus pending Cb.

    Accepts any hc_mult>=1 (hc_mult is orthogonal to the cached KV windows: the
    cache compresses the post-collapse hidden stream and never reads hc_mult).
    """

    def __init__(
        self,
        spec: DeepSeekV4AttentionSpec,
        weights: dict[str, list[list[float]] | list[float]],
        *,
        index_topk: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        num_key_value_heads: int = 1,
        hc_mult: int = 1,
        sliding_window: int = 0,
    ) -> None:
        if spec.compression_ratio != 4:
            raise NotImplementedError(f"{_STATEFUL_CSA_SUBSET_ERROR}; compression_ratio=4 required")
        if num_key_value_heads != 1:
            raise NotImplementedError(f"{_STATEFUL_CSA_SUBSET_ERROR}; num_key_value_heads=1 required")
        if spec.index_n_heads is None or spec.index_head_dim is None:
            raise NotImplementedError(f"{_STATEFUL_CSA_SUBSET_ERROR}; index_n_heads/index_head_dim required")
        if index_topk is not None and index_topk <= 0:
            raise ValueError("index_topk must be positive")
        if sliding_window < 0:
            raise ValueError("sliding_window must be non-negative")
        self._spec = spec
        self._weights = weights
        self._compress_rate = spec.compression_ratio
        self._index_n_heads = spec.index_n_heads
        self._index_head_dim = spec.index_head_dim
        self._index_topk = int(index_topk if index_topk is not None else 1)
        self._rms_norm_eps = float(rms_norm_eps)
        self._rope_theta = float(rope_theta)
        self._compressor_internal, self._indexer_internal = _csa_internal_weights(spec, weights)
        self._pending_hidden: list[list[float]] = []
        self._prev_window_hidden: list[list[float]] = []
        self._abs_pos = 0
        self._sliding_window = int(sliding_window)
        self._sliding_kv_rows: list[list[float]] = []
        self._emitted_compressed_kv: list[list[float]] = []
        self._emitted_index_keys: list[list[float]] = []

    @property
    def pending_hidden(self) -> list[list[float]]:
        return [list(row) for row in self._pending_hidden]

    @property
    def prev_window_hidden(self) -> list[list[float]]:
        return [list(row) for row in self._prev_window_hidden]

    @property
    def emitted_compressed_kv(self) -> list[list[float]]:
        return [list(row) for row in self._emitted_compressed_kv]

    @property
    def sliding_kv_rows(self) -> list[list[float]]:
        return [list(row) for row in self._sliding_kv_rows]

    @property
    def abs_pos(self) -> int:
        return self._abs_pos

    @property
    def index_topk(self) -> int:
        return self._index_topk

    def _complete_pending_window_if_ready(self) -> None:
        if len(self._pending_hidden) != self._compress_rate:
            return
        window_index = len(self._emitted_compressed_kv)
        self._emitted_compressed_kv.append(_compress_stateful_csa_window(
            self._pending_hidden,
            self._prev_window_hidden,
            self._compressor_internal,
            compress_rate=self._compress_rate,
            head_dim=self._spec.head_dim,
            rms_norm_eps=self._rms_norm_eps,
            rope_theta=self._rope_theta,
            window_index=window_index,
        ))
        self._emitted_index_keys.append(_compress_stateful_csa_window(
            self._pending_hidden,
            self._prev_window_hidden,
            self._indexer_internal,
            compress_rate=self._compress_rate,
            head_dim=self._index_head_dim,
            rms_norm_eps=self._rms_norm_eps,
            rope_theta=self._rope_theta,
            window_index=window_index,
        ))
        self._prev_window_hidden = [list(row) for row in self._pending_hidden]
        self._pending_hidden = []

    def step(self, hidden_chunk: list[list[float]], q_residual_chunk: list[list[float]]) -> dict[str, object]:
        if len(hidden_chunk) != len(q_residual_chunk):
            raise ValueError("hidden_chunk and q_residual_chunk must have the same length")
        rows_scores: list[list[float]] = []
        rows_bias: list[list[float]] = []
        rows_topk: list[list[int]] = []
        rows_mask: list[list[bool]] = []
        rows_attended: list[list[float]] = []
        for hidden, q_residual in zip(hidden_chunk, q_residual_chunk, strict=True):
            self._pending_hidden.append(list(hidden))
            self._complete_pending_window_if_ready()
            scored = _stateful_csa_score_token(
                hidden_token=list(hidden),
                q_residual_token=list(q_residual),
                indexer_internal=self._indexer_internal,
                compressed_index_keys=self._emitted_index_keys,
                emitted_compressed_kv=self._emitted_compressed_kv,
                position_id=self._abs_pos,
                compress_rate=self._compress_rate,
                index_n_heads=self._index_n_heads,
                index_head_dim=self._index_head_dim,
                index_topk=self._index_topk,
                rope_theta=self._rope_theta,
                head_dim=self._spec.head_dim,
            )
            rows_scores.append(scored["index_scores"])  # type: ignore[arg-type]
            rows_bias.append(scored["block_bias"])  # type: ignore[arg-type]
            rows_topk.append(scored["topk_indices"])  # type: ignore[arg-type]
            rows_mask.append(scored["topk_mask"])  # type: ignore[arg-type]
            rows_attended.append(scored["attended"])  # type: ignore[arg-type]
            self._abs_pos += 1
        return {
            "compressed_kv": self.emitted_compressed_kv,
            "index_scores": rows_scores,
            "block_bias": rows_bias,
            "topk_indices": rows_topk,
            "topk_mask": rows_mask,
            "attended": rows_attended,
            "compressed_len": len(self._emitted_compressed_kv),
        }

    def _project_stateful_fusion_token(
        self,
        hidden_token: list[float],
        q_residual_token: list[float],
        *,
        position_id: int,
    ) -> tuple[list[list[float]], list[float], list[float], list[float]]:
        required = {"q_b_proj.weight", "kv_proj.weight", "kv_norm.weight"}
        missing = sorted(required - set(self._weights))
        if missing:
            raise ValueError(f"missing stateful CSA fusion weights: {', '.join(missing)}")
        q_flat = _linear_out_in(q_residual_token, self._weights["q_b_proj.weight"])  # type: ignore[arg-type]
        q_heads = [
            _rms_norm(head, None, self._rms_norm_eps)
            for head in _split_heads(q_flat, num_heads=self._spec.num_attention_heads, head_dim=self._spec.head_dim)
        ]
        kv = _rms_norm(
            _linear_out_in(hidden_token, self._weights["kv_proj.weight"]),  # type: ignore[arg-type]
            self._weights["kv_norm.weight"],  # type: ignore[arg-type]
            self._rms_norm_eps,
        )
        cos, sin = _rope_cos_sin_for_position(
            position=int(position_id),
            rope_dim=self._spec.qk_rope_head_dim,
            theta=self._rope_theta,
        )
        q_heads = [apply_rope_tail(head, nope_dim=self._spec.qk_nope_head_dim, cos=cos, sin=sin) for head in q_heads]
        sliding_kv = apply_rope_tail(kv, nope_dim=self._spec.qk_nope_head_dim, cos=cos, sin=sin)
        return q_heads, sliding_kv, cos, sin

    def step_fusion(
        self,
        hidden_chunk: list[list[float]],
        q_residual_chunk: list[list[float]],
        *,
        position_ids_chunk: list[int] | None = None,
    ) -> dict[str, object]:
        """Advance compressed + sliding carries and emit per-token CSA fusion output.

        This is additive to the 11.17 compressed-only ``step``.  The sliding KV
        carry reuses ``sliding_window_cache_update``'s ``sliding_window-1``
        persisted-state rule, while the compressed carry continues to use the
        existing Ca/Cb helpers and per-token ``0/-inf`` block-bias scorer.
        """

        if len(hidden_chunk) != len(q_residual_chunk):
            raise ValueError("hidden_chunk and q_residual_chunk must have the same length")
        if position_ids_chunk is None:
            position_ids_chunk = list(range(self._abs_pos, self._abs_pos + len(hidden_chunk)))
        if len(position_ids_chunk) != len(hidden_chunk):
            raise ValueError("position_ids_chunk length must match hidden_chunk length")
        if any(len(row) != self._spec.hidden_size for row in hidden_chunk):
            raise ValueError("hidden_chunk width must match hidden_size")
        if any(len(row) != self._spec.q_lora_rank for row in q_residual_chunk):
            raise ValueError("q_residual_chunk width must match q_lora_rank")
        required = {"o_a_proj.weight", "o_b_proj.weight", "sinks"}
        missing = sorted(required - set(self._weights))
        if missing:
            raise ValueError(f"missing stateful CSA fusion weights: {', '.join(missing)}")

        rows_scores: list[list[float]] = []
        rows_bias: list[list[float]] = []
        rows_topk: list[list[int]] = []
        rows_mask: list[list[bool]] = []
        rows_output: list[list[float]] = []
        rows_sliding_len: list[int] = []
        scale = self._spec.head_dim ** -0.5
        sinks = self._weights["sinks"]  # type: ignore[assignment]

        for hidden, q_residual, position_id in zip(hidden_chunk, q_residual_chunk, position_ids_chunk, strict=True):
            if int(position_id) != self._abs_pos:
                raise ValueError("stateful CSA fusion supports contiguous position_ids only")
            q_heads, new_sliding_kv, cos, sin = self._project_stateful_fusion_token(
                list(hidden),
                list(q_residual),
                position_id=int(position_id),
            )
            if self._sliding_window == 0:
                sliding_rows = self._sliding_kv_rows + [new_sliding_kv]
                self._sliding_kv_rows = [list(row) for row in sliding_rows]
            else:
                update = sliding_window_cache_update(
                    self._sliding_kv_rows,
                    [new_sliding_kv],
                    sliding_window=self._sliding_window,
                )
                sliding_rows = update["attention_kv"]
                self._sliding_kv_rows = update["persisted_kv"]

            self._pending_hidden.append(list(hidden))
            self._complete_pending_window_if_ready()
            scored = _stateful_csa_score_token(
                hidden_token=list(hidden),
                q_residual_token=list(q_residual),
                indexer_internal=self._indexer_internal,
                compressed_index_keys=self._emitted_index_keys,
                emitted_compressed_kv=self._emitted_compressed_kv,
                position_id=int(position_id),
                compress_rate=self._compress_rate,
                index_n_heads=self._index_n_heads,
                index_head_dim=self._index_head_dim,
                index_topk=self._index_topk,
                rope_theta=self._rope_theta,
                head_dim=self._spec.head_dim,
            )
            block_bias = [float(value) for value in scored["block_bias"]]  # type: ignore[index]
            for value in block_bias:
                if value != 0.0 and not (math.isinf(value) and value < 0.0):
                    raise ValueError("CSA block_bias must be a 0/-inf additive mask")
            kv_rows = [list(row) for row in sliding_rows] + [list(row) for row in self._emitted_compressed_kv]
            additive_mask = [0.0] * len(sliding_rows) + block_bias
            if len(kv_rows) != len(additive_mask):
                raise ValueError("KV rows and additive mask length mismatch")

            attended_heads: list[list[float]] = []
            for head_idx, query in enumerate(q_heads):
                logits = [
                    sum(q * k for q, k in zip(query, kv_row)) * scale + float(mask)
                    for kv_row, mask in zip(kv_rows, additive_mask)
                ]
                probs = attention_scores_with_sink(logits, sink=float(sinks[head_idx]))
                attended = [
                    sum(prob * kv_row[dim] for prob, kv_row in zip(probs, kv_rows))
                    for dim in range(self._spec.head_dim)
                ]
                attended_heads.append(apply_output_inverse_rope_tail(
                    attended,
                    nope_dim=self._spec.qk_nope_head_dim,
                    cos=cos,
                    sin=sin,
                ))
            rows_output.append(_grouped_output_projection(
                attended_heads,
                spec=self._spec,
                o_a_weight=self._weights["o_a_proj.weight"],  # type: ignore[arg-type]
                o_b_weight=self._weights["o_b_proj.weight"],  # type: ignore[arg-type]
            ))
            rows_scores.append(scored["index_scores"])  # type: ignore[arg-type]
            rows_bias.append(block_bias)
            rows_topk.append(scored["topk_indices"])  # type: ignore[arg-type]
            rows_mask.append(scored["topk_mask"])  # type: ignore[arg-type]
            rows_sliding_len.append(len(sliding_rows))
            self._abs_pos += 1

        return {
            "output": rows_output,
            "sliding_kv": self.sliding_kv_rows,
            "sliding_lengths": rows_sliding_len,
            "compressed_kv": self.emitted_compressed_kv,
            "index_scores": rows_scores,
            "block_bias": rows_bias,
            "topk_indices": rows_topk,
            "topk_mask": rows_mask,
            "compressed_len": len(self._emitted_compressed_kv),
        }


def tiny_stateful_csa_attention(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    q_residual: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    call_splits: list[int],
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    index_topk: int | None = None,
    num_key_value_heads: int = 1,
    hc_mult: int = 1,
) -> dict[str, object]:
    if len(hidden_states) != len(q_residual):
        raise ValueError("hidden_states and q_residual must have the same sequence length")
    if sum(call_splits) != len(hidden_states) or any(split <= 0 for split in call_splits):
        raise ValueError("call_splits must be positive and sum to the sequence length")
    cache = StatefulCSACache(
        spec,
        weights,
        index_topk=index_topk,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        num_key_value_heads=num_key_value_heads,
        hc_mult=hc_mult,
    )
    index_scores: list[list[float]] = []
    block_bias: list[list[float]] = []
    topk_indices: list[list[int]] = []
    topk_mask: list[list[bool]] = []
    attended: list[list[float]] = []
    offset = 0
    for split in call_splits:
        chunk = cache.step(hidden_states[offset: offset + split], q_residual[offset: offset + split])
        index_scores.extend(chunk["index_scores"])  # type: ignore[arg-type]
        block_bias.extend(chunk["block_bias"])  # type: ignore[arg-type]
        topk_indices.extend(chunk["topk_indices"])  # type: ignore[arg-type]
        topk_mask.extend(chunk["topk_mask"])  # type: ignore[arg-type]
        attended.extend(chunk["attended"])  # type: ignore[arg-type]
        offset += split

    compressed_kv = cache.emitted_compressed_kv
    compressed_len = len(compressed_kv)
    final_topk = min(cache.index_topk, compressed_len)

    def _pad_float_rows(rows: list[list[float]]) -> list[list[float]]:
        return [list(row) + [float("-inf")] * (compressed_len - len(row)) for row in rows]

    return {
        "compressed_kv": compressed_kv,
        "index_scores": _pad_float_rows(index_scores),
        "block_bias": _pad_float_rows(block_bias),
        "topk_indices": [list(row[:final_topk]) + [-1] * max(0, final_topk - len(row)) for row in topk_indices],
        "topk_mask": [list(row[:final_topk]) + [False] * max(0, final_topk - len(row)) for row in topk_mask],
        "attended": attended,
        "compressed_len": compressed_len,
    }


def tiny_stateful_csa_fusion_reference(
    spec: DeepSeekV4AttentionSpec,
    hidden_states: list[list[float]],
    q_residual: list[list[float]],
    weights: dict[str, list[list[float]] | list[float]],
    *,
    call_splits: list[int],
    sliding_window: int,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    position_ids: list[int] | None = None,
    index_topk: int | None = None,
    num_key_value_heads: int = 1,
    hc_mult: int = 1,
) -> dict[str, object]:
    """Stateful single-head CSA fusion reference over arbitrary call splits.

    This fills the missing stateful fusion cell: the cache carries the existing
    compressed CSA stream and a sliding-window per-token KV stream, then fuses
    ``[sliding | compressed]`` per token with the shared ``0/-inf`` CSA
    ``block_bias``.  It intentionally stays behind the existing single-head /
    o_groups=1 stateful cache gates.
    """

    if len(hidden_states) != len(q_residual):
        raise ValueError("hidden_states and q_residual must have the same sequence length")
    if sum(call_splits) != len(hidden_states) or any(split <= 0 for split in call_splits):
        raise ValueError("call_splits must be positive and sum to the sequence length")
    if sliding_window < 0:
        raise ValueError("sliding_window must be non-negative")
    seq_len = len(hidden_states)
    if position_ids is None:
        position_ids = list(range(seq_len))
    if len(position_ids) != seq_len:
        raise ValueError("position_ids length must match sequence length")
    if any(int(pos) != idx for idx, pos in enumerate(position_ids)):
        raise ValueError("stateful CSA fusion reference supports contiguous position_ids only")

    cache = StatefulCSACache(
        spec,
        weights,
        index_topk=index_topk,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        num_key_value_heads=num_key_value_heads,
        hc_mult=hc_mult,
        sliding_window=sliding_window,
    )
    outputs: list[list[float]] = []
    index_scores: list[list[float]] = []
    block_bias: list[list[float]] = []
    topk_indices: list[list[int]] = []
    topk_mask: list[list[bool]] = []
    sliding_lengths: list[int] = []
    offset = 0
    for split in call_splits:
        chunk = cache.step_fusion(
            hidden_states[offset: offset + split],
            q_residual[offset: offset + split],
            position_ids_chunk=position_ids[offset: offset + split],
        )
        outputs.extend(chunk["output"])  # type: ignore[arg-type]
        index_scores.extend(chunk["index_scores"])  # type: ignore[arg-type]
        block_bias.extend(chunk["block_bias"])  # type: ignore[arg-type]
        topk_indices.extend(chunk["topk_indices"])  # type: ignore[arg-type]
        topk_mask.extend(chunk["topk_mask"])  # type: ignore[arg-type]
        sliding_lengths.extend(chunk["sliding_lengths"])  # type: ignore[arg-type]
        offset += split

    compressed_kv = cache.emitted_compressed_kv
    compressed_len = len(compressed_kv)
    final_topk = min(cache.index_topk, compressed_len)

    def _pad_float_rows(rows: list[list[float]]) -> list[list[float]]:
        return [list(row) + [float("-inf")] * (compressed_len - len(row)) for row in rows]

    return {
        "output": outputs,
        "sliding_kv": cache.sliding_kv_rows,
        "sliding_lengths": sliding_lengths,
        "compressed_kv": compressed_kv,
        "index_scores": _pad_float_rows(index_scores),
        "block_bias": _pad_float_rows(block_bias),
        "topk_indices": [list(row[:final_topk]) + [-1] * max(0, final_topk - len(row)) for row in topk_indices],
        "topk_mask": [list(row[:final_topk]) + [False] * max(0, final_topk - len(row)) for row in topk_mask],
        "compressed_len": compressed_len,
        "covered": [
            "stateful single-head CSA fusion",
            "sliding KV carry",
            "per-chunk [sliding|compressed] KV concatenation",
            "per-query 0/-inf CSA block-bias mask",
            "per-head attention sinks",
            "grouped output projection",
        ],
        "not_covered": [
            "stateful multi-head CSA cache",
            "stateful multi-group CSA cache",
            "vendor MLX runtime gate lift",
            "Track-B forward parity unblock",
        ],
    }


def run_tiny_compressor_indexer_attention_fixture() -> dict[str, object]:
    """Self-check for the integrated CSA compressor/indexer attention reference.

    Verifies output shapes, causal block-bias thresholding, all-masked rows
    yield zero output, and that the compressor path agrees with the validated
    ``tiny_csa_compressor_forward`` helper.  This is structural self-evidence
    only (no Transformers end-to-end numeric reference); full layer integration
    remains fail-closed.
    """

    spec = DeepSeekV4AttentionSpec(
        hidden_size=4,
        num_attention_heads=1,
        head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        qk_rope_head_dim=4,
        num_output_groups=1,
        compression_ratio=4,
        index_n_heads=2,
        index_head_dim=2,
    )
    ratio = spec.compression_ratio

    def _dense(rows: int, cols: int, scale: float) -> list[list[float]]:
        return [
            [round(scale * (i + j + 1) * (1.0 if (i + j) % 2 == 0 else -1.0), 4) for j in range(cols)]
            for i in range(rows)
        ]

    weights = {
        "compressor_wkv": _dense(4, 8, 0.5),
        "compressor_wgate": _dense(4, 8, 0.25),
        "compressor_ape": _dense(8, 4, 0.05),
        "compressor_norm": [1.0, 1.0, 1.0, 1.0],
        "indexer_wq_b": _dense(4, 4, 0.5),
        "indexer_proj": _dense(4, 2, 0.3),
        "indexer_compressor_wkv": _dense(4, 4, 0.4),
        "indexer_compressor_wgate": _dense(4, 4, 0.2),
        "indexer_compressor_ape": _dense(4, 4, 0.05),
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

    got = tiny_compressor_indexer_attention_reference(spec, hidden, hidden, weights, index_topk=2)
    max_abs_error = 0.0

    # Shape checks.
    if len(got["compressed_kv"]) != 2 or len(got["compressed_kv"][0]) != spec.head_dim:
        max_abs_error = float("inf")
    if len(got["index_scores"]) != 8 or len(got["index_scores"][0]) != 2:
        max_abs_error = float("inf")
    if len(got["attended"]) != 8 or len(got["attended"][0]) != spec.head_dim:
        max_abs_error = float("inf")

    # Causal threshold + all-masked zero-output checks.
    for pos in range(8):
        threshold = (pos + 1) // ratio
        for entry in range(2):
            expected_masked = entry >= threshold
            got_masked = math.isinf(got["block_bias"][pos][entry]) and got["block_bias"][pos][entry] < 0
            if got_masked != expected_masked:
                max_abs_error = float("inf")
        if threshold == 0:
            # All entries masked (threshold 0 for pos 0,1,2) -> zero output.
            if any(float(v) != 0.0 for v in got["attended"][pos]):
                max_abs_error = float("inf")

    # Compressor agreement with the validated helper.
    positions = [w * ratio for w in range(2)]
    cos_c, sin_c = _rope_cos_sin(positions, spec.head_dim, 10000.0)
    internal = {
        "kv_proj": weights["compressor_wkv"],
        "gate_proj": weights["compressor_wgate"],
        "position_bias": [
            [weights["compressor_ape"][d][t] for d in range(2 * spec.head_dim)]
            for t in range(ratio)
        ],
        "kv_norm": weights["compressor_norm"],
    }
    ref_compressed = tiny_csa_compressor_forward(
        hidden, internal, compress_rate=ratio, rms_norm_eps=1e-6, rope_cos=cos_c, rope_sin=sin_c
    )
    for got_row, ref_row in zip(got["compressed_kv"], ref_compressed):
        for gv, rv in zip(got_row, ref_row):
            max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))

    return {
        "fixture": "compressor-indexer-attention",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "CSA compressor softmax-gated KV compression",
            "CSA compressor Ca/Cb overlap",
            "CSA Lightning indexer routing",
            "causal block-bias threshold",
            "indexer top-k gather mask",
            "compressed attention value mixing",
            "all-masked early-query zero output",
        ],
        "not_covered": [
            "MLX compressor/indexer port",
            "stateful cache overlap",
            "main q_a/q_b core attention against compressed KV",
            "grouped output projection o_a/o_b for compressed path",
            "HCA rate-128 non-overlapping compression",
            "real DeepSeek V4 Flash checkpoint compatibility",
            "full DeepSeekV4Model layer integration",
        ],
    }


def compressor_forward_placeholder(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedError("DeepSeek V4 compressor forward parity is not implemented in this scaffold")


def hyperconnection_forward_placeholder(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedError("DeepSeek V4 hyperconnection forward parity is not implemented in this scaffold")


# ---------------------------------------------------------------------------
# Story 11.53 -- real_config_* numpy reference variants.
#
# These are genuine numpy re-implementations of the OCP-witness spec math
# (the same math validated by the co-located `tiny_*` pure-python references,
# which are themselves validated against HF Transformers `DeepseekV4Model`).
# They are NOT transliterations of vendor `_real_forward` (ADR 0007 §4).
# Each takes dequantized numpy weight arrays (BF16-shimmed tensors are read
# as BF16 and viewed to float64; FP8+UE8M0 tensors go through the ADR 0017
# `deepseek_v4_dequant` helper before being passed in). The independent
# witness for each GREEN primitive is the `tiny_*` pure-python reference run
# on identical real-config-shaped input.
#
# Per-primitive STOP (iv): the full multihead-grouped attention path and the
# stateful CSA / compositional attention primitives at real config
# (num_attention_heads=64, head_dim=512, q_lora_rank=1024 -> q_dim=32768,
# o_groups=8) have a CPU-infeasible pure-python `tiny_*` witness, and an
# independent second numpy attention implementation at real MQA dims would
# transliterate `_real_forward`. Their `real_config_*` stubs raise
# NotImplementedError("STOP (iv): ...") and composition is deferred to
# Story 11.54 (`numpy_real_forward_reference.py`).
# ---------------------------------------------------------------------------


def _np_softmax_last(axis_last: "np.ndarray") -> "np.ndarray":
    m = axis_last.max(axis=-1, keepdims=True)
    e = np.exp(axis_last - m)
    return e / e.sum(axis=-1, keepdims=True)


def _np_rms_norm(vector: "np.ndarray", weight: "np.ndarray | None", eps: float) -> "np.ndarray":
    vector = np.asarray(vector, dtype=np.float64)
    denom = np.sqrt(np.mean(vector * vector, axis=-1, keepdims=True) + eps)
    out = vector / denom
    if weight is not None:
        out = out * np.asarray(weight, dtype=np.float64)
    return out


def _np_apply_rope_tail(vector: "np.ndarray", nope_dim: int, cos: "np.ndarray", sin: "np.ndarray") -> "np.ndarray":
    """Numpy port of `apply_rope_tail` (pairwise rotate_half on the tail slice)."""
    vector = np.asarray(vector, dtype=np.float64)
    cos = np.asarray(cos, dtype=np.float64)
    sin = np.asarray(sin, dtype=np.float64)
    head = vector[..., :nope_dim]
    rope = vector[..., nope_dim:]
    x0 = rope[..., 0::2]
    x1 = rope[..., 1::2]
    c0 = cos[..., 0::2]
    c1 = cos[..., 1::2]
    s0 = sin[..., 0::2]
    s1 = sin[..., 1::2]
    r0 = x0 * c0 - x1 * s0
    r1 = x1 * c1 + x0 * s1
    rope_out = np.empty_like(rope)
    rope_out[..., 0::2] = r0
    rope_out[..., 1::2] = r1
    if nope_dim == 0:
        return rope_out
    return np.concatenate([head, rope_out], axis=-1)


def real_config_embed_tokens(input_ids: "np.ndarray | Sequence[int]", embed_weight: "np.ndarray") -> "np.ndarray":
    """Group D -- `model.embed_tokens` lookup at real config (vocab=129280, hidden=4096)."""
    embed_weight = np.asarray(embed_weight, dtype=np.float64)
    ids = np.asarray(list(input_ids), dtype=np.int64)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    return embed_weight[ids]


def real_config_rms_norm(x: "np.ndarray", weight: "np.ndarray | None", eps: float) -> "np.ndarray":
    """Group D -- RMSNorm for input_layernorm / post_attention_layernorm / model.norm.

    `weight` is the per-channel scale (`layers.N.attn_norm.weight`,
    `layers.N.ffn_norm.weight`, `model.norm.weight`). Pass None for an
    unweighted norm (the hyperconnection flatten path uses unweighted RMSNorm
    internally; see `real_config_hyperconnection_forward`).
    """
    return _np_rms_norm(x, weight, eps)


def real_config_lm_head(
    hidden: "np.ndarray",
    head_weight: "np.ndarray",
    top_k: int = 0,
) -> tuple["np.ndarray", "np.ndarray | None"]:
    """Group D -- `lm_head` projection at real config (hidden=4096 -> vocab=129280).

    Returns `(logits, cols)`. When `top_k > 0`, only the top-`top_k` vocab
    columns (by max logit over the sequence) are materialized to keep the CPU
    witness floor small; `cols` holds the selected vocab ids.
    """
    W = np.asarray(head_weight, dtype=np.float64)  # [vocab, hidden]
    h = np.asarray(hidden, dtype=np.float64)        # [seq, hidden]
    logits = h @ W.T
    if top_k and top_k > 0 and top_k < logits.shape[-1]:
        cols = np.argsort(-logits.max(axis=0))[:top_k]
        return logits[:, cols], cols
    return logits, None


def real_config_hyperconnection_forward(
    *,
    hidden_streams: "np.ndarray",
    fn: "np.ndarray",
    base: "np.ndarray",
    scale: "np.ndarray",
    hc_mult: int,
    eps: float,
    sinkhorn_iters: int,
    rms_norm_eps: float,
) -> dict[str, "np.ndarray"]:
    """Group A -- numpy port of `tiny_hyperconnection_forward` at real config.

    `hidden_streams`: [seq, hc_mult, hidden]. Real layer-0 `hc_attn_fn` is
    F32 shape [24, 16384] = [(2+hc_mult)*hc_mult, hc_mult*hidden] with
    hc_mult=4, hidden=4096. Returns `pre`/`post`/`collapsed`/`comb` as numpy
    arrays matching the tiny reference's batched layout.
    """
    streams = np.asarray(hidden_streams, dtype=np.float64)
    seq, hm, hidden = streams.shape
    if hm != hc_mult:
        raise ValueError("hidden stream count mismatch")
    fn = np.asarray(fn, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    expected_mix = (2 + hc_mult) * hc_mult
    if fn.shape != (expected_mix, hc_mult * hidden):
        raise ValueError("hyperconnection fn shape mismatch")
    if len(scale) != 3:
        raise ValueError("hyperconnection scale must have 3 entries")
    if base.shape[0] != expected_mix:
        raise ValueError("hyperconnection base shape mismatch")
    flat = streams.reshape(seq, hc_mult * hidden)
    flat = _np_rms_norm(flat, None, rms_norm_eps)
    mixes = flat @ fn.T  # [seq, expected_mix]
    pre_w = mixes[:, :hc_mult]
    post_w = mixes[:, hc_mult:2 * hc_mult]
    comb_w = mixes[:, 2 * hc_mult:]
    pre_b = base[:hc_mult]
    post_b = base[hc_mult:2 * hc_mult]
    comb_b = base[2 * hc_mult:]
    pre = 1.0 / (1.0 + np.exp(-(pre_w * scale[0] + pre_b))) + eps
    post = 2.0 / (1.0 + np.exp(-(post_w * scale[1] + post_b)))
    comb_logits = comb_w * scale[2] + comb_b
    comb = comb_logits.reshape(seq, hc_mult, hc_mult)
    comb = _np_softmax_last(comb)
    comb = comb + eps
    # Initial column normalization (mirrors tiny reference's pre-loop col pass).
    comb = comb / (comb.sum(axis=1, keepdims=True) + eps)
    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (comb.sum(axis=2, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=1, keepdims=True) + eps)
    collapsed = np.einsum("sh,shd->sd", pre, streams)
    return {"pre": pre, "collapsed": collapsed, "post": post, "comb": comb}


def real_config_hyperhead_collapse(
    *,
    hidden_streams: "np.ndarray",
    fn: "np.ndarray",
    base: "np.ndarray",
    scale: "np.ndarray",
    hc_mult: int,
    eps: float,
    rms_norm_eps: float,
) -> dict[str, "np.ndarray"]:
    """Group A -- numpy port of `tiny_hyperhead_collapse` at real config.

    `hidden_streams`: [seq, hc_mult, hidden]. Real `hc_head_fn` is F32 shape
    [4, 16384] = [hc_mult, hc_mult*hidden]. Returns `pre`/`collapsed`.
    """
    streams = np.asarray(hidden_streams, dtype=np.float64)
    seq, hm, hidden = streams.shape
    if hm != hc_mult:
        raise ValueError("hidden stream count mismatch")
    fn = np.asarray(fn, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    scale_v = float(np.asarray(scale).reshape(-1)[0])
    if fn.shape != (hc_mult, hc_mult * hidden):
        raise ValueError("hyperhead fn shape mismatch")
    if base.shape[0] != hc_mult:
        raise ValueError("hyperhead base shape mismatch")
    flat = streams.reshape(seq, hc_mult * hidden)
    flat = _np_rms_norm(flat, None, rms_norm_eps)
    mixes = flat @ fn.T  # [seq, hc_mult]
    pre = 1.0 / (1.0 + np.exp(-(mixes * scale_v + base))) + eps
    collapsed = np.einsum("sh,shd->sd", pre, streams)
    return {"pre": pre, "collapsed": collapsed}


def real_config_hca_compressor_forward(
    hidden_states: "np.ndarray",
    weights: dict[str, "np.ndarray"],
    *,
    compress_rate: int,
    rms_norm_eps: float,
    rope_cos: "np.ndarray",
    rope_sin: "np.ndarray",
    first_window_position: int = 0,
) -> "np.ndarray":
    """Group A -- numpy port of `tiny_hca_compressor_forward` (synthetic,
    compression_ratio=4 CSA branch; real config has compression_ratio=0 and no
    compressor tensors, so this is exercised on real-config-shaped synthetic
    input per Architect §1 Q6).
    """
    hidden = np.asarray(hidden_states, dtype=np.float64)  # [seq, hidden]
    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    kv_proj = np.asarray(weights["kv_proj"], dtype=np.float64)      # [head_dim, hidden]
    gate_proj = np.asarray(weights["gate_proj"], dtype=np.float64)  # [head_dim, hidden]
    position_bias = np.asarray(weights["position_bias"], dtype=np.float64)  # [compress_rate, head_dim]
    kv_norm = np.asarray(weights["kv_norm"], dtype=np.float64)      # [head_dim]
    seq_len = hidden.shape[0]
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate
    if n_windows == 0:
        return np.zeros((0, kv_proj.shape[0]), dtype=np.float64)
    cos = np.asarray(rope_cos, dtype=np.float64)
    sin = np.asarray(rope_sin, dtype=np.float64)
    if cos.shape[0] < n_windows or sin.shape[0] < n_windows:
        raise ValueError("rope_cos/sin must cover n_windows")
    kvs = hidden[:usable] @ kv_proj      # [usable, head_dim]  (kv_proj is [hidden, head_dim])
    gates = hidden[:usable] @ gate_proj # [usable, head_dim]
    head_dim = kvs.shape[1]
    if position_bias.shape != (compress_rate, head_dim):
        raise ValueError("position_bias shape must be [compress_rate, head_dim]")
    compressed = np.zeros((n_windows, head_dim), dtype=np.float64)
    for w in range(n_windows):
        wk = kvs[w * compress_rate:(w + 1) * compress_rate]      # [cr, head_dim]
        wg = gates[w * compress_rate:(w + 1) * compress_rate]    # [cr, head_dim]
        biased = wg + position_bias                               # [cr, head_dim]
        probs = _np_softmax_last(biased.T).T                       # softmax over cr per dim
        pooled = (probs * wk).sum(axis=0)                          # [head_dim]
        normed = _np_rms_norm(pooled, kv_norm, rms_norm_eps)
        roped = _np_apply_rope_tail(normed, nope_dim=0, cos=cos[w], sin=sin[w])
        compressed[w] = roped
    return compressed


def real_config_csa_compressor_forward(
    hidden_states: "np.ndarray",
    weights: dict[str, "np.ndarray"],
    *,
    compress_rate: int,
    rms_norm_eps: float,
    rope_cos: "np.ndarray",
    rope_sin: "np.ndarray",
    first_window_position: int = 0,
) -> "np.ndarray":
    """Group B -- numpy port of `tiny_csa_compressor_forward` (synthetic,
    compression_ratio=4 CSA branch; real config has compression_ratio=0).
    """
    hidden = np.asarray(hidden_states, dtype=np.float64)
    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    kv_proj = np.asarray(weights["kv_proj"], dtype=np.float64)
    gate_proj = np.asarray(weights["gate_proj"], dtype=np.float64)
    position_bias = np.asarray(weights["position_bias"], dtype=np.float64)
    kv_norm = np.asarray(weights["kv_norm"], dtype=np.float64)
    seq_len = hidden.shape[0]
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate
    if n_windows == 0:
        return np.zeros((0, kv_proj.shape[0] // 2), dtype=np.float64)
    cos = np.asarray(rope_cos, dtype=np.float64)
    sin = np.asarray(rope_sin, dtype=np.float64)
    if cos.shape[0] < n_windows or sin.shape[0] < n_windows:
        raise ValueError("rope_cos/sin must cover n_windows")
    kvs = hidden[:usable] @ kv_proj    # [usable, 2*head_dim]  (kv_proj is [hidden, 2*head_dim])
    gates = hidden[:usable] @ gate_proj
    out_dim = kvs.shape[1]
    if out_dim % 2 != 0:
        raise ValueError("CSA compressor projection must be divisible by 2")
    head_dim = out_dim // 2
    if position_bias.shape != (compress_rate, out_dim):
        raise ValueError("position_bias shape must be [compress_rate, 2*head_dim]")
    # Add per-token position bias by window slot.
    pos_ids = np.arange(usable) % compress_rate
    biased_gates = gates + position_bias[pos_ids]   # [usable, 2*head_dim]
    compressed = np.zeros((n_windows, head_dim), dtype=np.float64)
    for w in range(n_windows):
        new_kv = np.zeros((2 * compress_rate, head_dim), dtype=np.float64)
        new_gate = np.full((2 * compress_rate, head_dim), -np.inf)
        cur = slice(w * compress_rate, (w + 1) * compress_rate)
        # Cb slot (last compress_rate rows): current window's [head_dim:]
        new_kv[compress_rate:] = kvs[cur, head_dim:]
        new_gate[compress_rate:] = biased_gates[cur, head_dim:]
        if w > 0:
            prev = slice((w - 1) * compress_rate, w * compress_rate)
            # Ca slot (first compress_rate rows): previous window's [:head_dim]
            new_kv[:compress_rate] = kvs[prev, :head_dim]
            new_gate[:compress_rate] = biased_gates[prev, :head_dim]
        probs = _np_softmax_last(new_gate.T).T   # softmax over rows per dim
        pooled = (probs * new_kv).sum(axis=0)
        normed = _np_rms_norm(pooled, kv_norm, rms_norm_eps)
        cos_w = cos[w]
        sin_w = sin[w]
        roped = _np_apply_rope_tail(normed, nope_dim=0, cos=cos_w, sin=sin_w)
        compressed[w] = roped
    return compressed


def real_config_csa_indexer_forward(
    hidden_states: "np.ndarray",
    q_residual: "np.ndarray",
    weights: dict[str, "np.ndarray"],
    *,
    compress_rate: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rms_norm_eps: float,
    rope_theta: float,
    position_ids: "Sequence[int]",
) -> dict[str, "np.ndarray"]:
    """Group B -- numpy port of `tiny_csa_indexer_forward` (synthetic,
    compression_ratio=4 CSA branch; real config has compression_ratio=0).
    """
    hidden = np.asarray(hidden_states, dtype=np.float64)
    q_res = np.asarray(q_residual, dtype=np.float64)
    if compress_rate <= 0:
        raise ValueError("compress_rate must be positive")
    kv_proj = np.asarray(weights["kv_proj"], dtype=np.float64)
    gate_proj = np.asarray(weights["gate_proj"], dtype=np.float64)
    position_bias = np.asarray(weights["position_bias"], dtype=np.float64)
    kv_norm = np.asarray(weights["kv_norm"], dtype=np.float64)
    q_b_proj = np.asarray(weights["q_b_proj"], dtype=np.float64)
    weights_proj = np.asarray(weights["weights_proj"], dtype=np.float64)
    seq_len = hidden.shape[0]
    if q_res.shape[0] != seq_len or len(position_ids) != seq_len:
        raise ValueError("hidden/q_residual/position_ids length mismatch")
    usable = (seq_len // compress_rate) * compress_rate
    n_windows = usable // compress_rate
    kv = hidden @ kv_proj    # [seq, 2*head_dim]
    gate = hidden @ gate_proj
    out_dim = kv.shape[1]
    if out_dim % 2 != 0:
        raise ValueError("CSA indexer kv projection must be divisible by 2")
    head_dim = out_dim // 2
    if head_dim != index_head_dim:
        raise ValueError("index kv head_dim must match index_head_dim")
    if position_bias.shape[0] != compress_rate:
        raise ValueError("position_bias shape mismatch")
    pos_ids = np.arange(usable) % compress_rate
    biased_gate = gate[:usable] + position_bias[pos_ids]   # [usable, 2*head_dim]
    # Compressed indexer keys (mirror csa_compressor Ca/Cb window logic).
    compressed = np.zeros((n_windows, head_dim), dtype=np.float64)
    for w in range(n_windows):
        new_kv = np.zeros((2 * compress_rate, head_dim), dtype=np.float64)
        new_gate = np.full((2 * compress_rate, head_dim), -np.inf)
        cur = slice(w * compress_rate, (w + 1) * compress_rate)
        new_kv[compress_rate:] = kv[cur, head_dim:]
        new_gate[compress_rate:] = biased_gate[cur, head_dim:]
        if w > 0:
            prev = slice((w - 1) * compress_rate, w * compress_rate)
            new_kv[:compress_rate] = kv[prev, :head_dim]
            new_gate[:compress_rate] = biased_gate[prev, :head_dim]
        probs = _np_softmax_last(new_gate.T).T
        pooled = (probs * new_kv).sum(axis=0)
        normed = _np_rms_norm(pooled, kv_norm, rms_norm_eps)
        pos = w * compress_rate
        cos_pp, sin_pp = _np_rope_cos_sin(np.array([pos]), head_dim, rope_theta)
        compressed[w] = _np_apply_rope_tail(normed, 0, cos_pp[0], sin_pp[0])
    compressed_len = n_windows
    # Query heads.
    q_proj = q_res @ q_b_proj   # [seq, index_n_heads*index_head_dim]
    expected_q_dim = index_n_heads * index_head_dim
    if q_proj.shape[1] != expected_q_dim:
        raise ValueError("q_b_proj output dimension mismatch")
    cos_q, sin_q = _np_rope_cos_sin(np.asarray(list(position_ids), dtype=np.int64), index_head_dim, rope_theta)
    q_heads = q_proj.reshape(seq_len, index_n_heads, index_head_dim)
    q_heads = _np_apply_rope_tail(q_heads, 0, cos_q[:, None, :], sin_q[:, None, :])
    weights_scaling = index_n_heads ** -0.5
    scorer_weights = (hidden @ weights_proj) * weights_scaling  # [seq, index_n_heads]
    softmax_scale = index_head_dim ** -0.5
    # dots[t, h, c] = sum_d q[t,h,d]*compressed[c,d] * softmax_scale
    dots = np.einsum("thd,cd->thc", q_heads, compressed) * softmax_scale   # [seq, n_heads, compressed_len]
    relu = np.maximum(0.0, dots)
    scores = (relu * scorer_weights[:, :, None]).sum(axis=1)              # [seq, compressed_len]
    top_k = min(index_topk, compressed_len)
    topk_indices = np.full((seq_len, top_k), -1, dtype=np.int64)
    topk_mask = np.zeros((seq_len, top_k), dtype=bool)
    pos_arr = np.asarray(list(position_ids), dtype=np.int64)
    for t in range(seq_len):
        threshold = (int(pos_arr[t]) + 1) // compress_rate
        valid = np.arange(compressed_len) < threshold
        cand_idx = np.where(valid)[0]
        order = np.argsort(-scores[t, cand_idx], kind="stable")
        sel = cand_idx[order][:top_k]
        topk_indices[t, :len(sel)] = sel
        topk_mask[t, :len(sel)] = True
    return {
        "scores": scores,
        "topk_indices": topk_indices,
        "topk_mask": topk_mask,
        "compressed_len": np.array(compressed_len),
    }


def _np_rope_cos_sin(positions: "np.ndarray", rope_dim: int, theta: float) -> tuple["np.ndarray", "np.ndarray"]:
    """Numpy port of `_rope_cos_sin_for_position` returning [len(positions), rope_dim]."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    inv_freq = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim))
    freqs = np.outer(positions, inv_freq)            # [n, rope_dim//2]
    cos = np.repeat(np.cos(freqs), 2, axis=-1)        # [n, rope_dim]
    sin = np.repeat(np.sin(freqs), 2, axis=-1)
    return cos, sin


_GROUP_C_STOP_PRIMITIVES = (
    "multihead_grouped_attention_reference",
    "sliding_attention_no_compressor",
    "compressor_indexer_attention_reference",
    "stateful_csa_attention",
    "stateful_csa_fusion_reference",
    "multihead_csa_fusion_reference",
    "incremental_attention",
    "greedy_decode",
)


def _real_config_stop_iv(name: str) -> None:
    raise NotImplementedError(
        f"STOP (iv): real_config_{name} cannot be scaled to an independent "
        f"real-config witness within Story 11.53 without transliterating "
        f"_real_forward. The primitive composes the full multihead-grouped "
        f"attention / stateful CSA path at real MQA dims "
        f"(num_attention_heads=64, head_dim=512, q_lora_rank=1024 -> "
        f"q_dim=32768, o_groups=8) whose pure-python tiny_* witness is "
        f"CPU-infeasible and whose independent second numpy implementation "
        f"would transliterate the vendor _real_forward attention path; full "
        f"attention composition is deferred to Story 11.54 "
        f"(numpy_real_forward_reference.py)."
    )


def real_config_multihead_grouped_attention_reference(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("multihead_grouped_attention_reference")


def real_config_sliding_attention_no_compressor(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("sliding_attention_no_compressor")


def real_config_compressor_indexer_attention_reference(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("compressor_indexer_attention_reference")


def real_config_stateful_csa_attention(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("stateful_csa_attention")


def real_config_stateful_csa_fusion_reference(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("stateful_csa_fusion_reference")


def real_config_multihead_csa_fusion_reference(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("multihead_csa_fusion_reference")


def real_config_incremental_attention(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("incremental_attention")


def real_config_greedy_decode(*_args: object, **_kwargs: object) -> None:
    _real_config_stop_iv("greedy_decode")
