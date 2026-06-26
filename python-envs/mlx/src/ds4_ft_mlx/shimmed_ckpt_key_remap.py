"""Story 13.1 shimmed-checkpoint key remap adapter.

Pure key-string translation only.  The adapter never reads or mutates tensor
payload bytes, never changes dtype, never reshapes layouts, and never
re-quantizes.  It maps the existing shimmed DeepSeek-V4 checkpoint vocabulary
to vendor-internal key vocabulary expected by
``python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py``.
"""

from __future__ import annotations

import re as _re
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import Any as _Any

_RENAME_RULES: tuple[tuple[_re.Pattern[str], str], ...] = (
    # §2.1 Cluster 1 — embed / head / norm / hc_head
    (_re.compile(r"^embed\.weight$"), "embed.weight"),
    (_re.compile(r"^head\.weight$"), "lm_head.weight"),
    (_re.compile(r"^norm\.weight$"), "norm.weight"),
    (_re.compile(r"^hc_head_base$"), "hc_head.base"),
    (_re.compile(r"^hc_head_fn$"), "hc_head.fn"),
    (_re.compile(r"^hc_head_scale$"), "hc_head.scale"),
    # §2.2 Cluster 2 — layers.N.attn.*
    (_re.compile(r"^layers\.(\d+)\.attn_norm\.weight$"), r"layers.\1.input_layernorm.weight"),
    (_re.compile(r"^layers\.(\d+)\.ffn_norm\.weight$"), r"layers.\1.post_attention_layernorm.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.kv_norm\.weight$"), r"layers.\1.kv_norm.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.q_norm\.weight$"), r"layers.\1.q_norm.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.wkv\.weight$"), r"layers.\1.kv_proj.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.wq_a\.weight$"), r"layers.\1.q_a_proj.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.wq_b\.weight$"), r"layers.\1.q_b_proj.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.wo_a\.weight$"), r"layers.\1.o_a_proj.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.wo_b\.weight$"), r"layers.\1.o_b_proj.weight"),
    (_re.compile(r"^layers\.(\d+)\.attn\.attn_sink$"), r"layers.\1.sinks"),
    # §2.3 Cluster 3 — shared_experts
    (_re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w([123])\.weight$"), r"layers.\1.mlp.shared_experts.w\2.weight"),
    # §2.4 Cluster 4 — experts weight
    (_re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$"), r"layers.\1.mlp.experts.\2.w\3.weight"),
    # §2.5 Cluster 5 — experts scale + gate + hc_attn + hc_ffn
    (_re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.scale$"), r"layers.\1.mlp.experts.\2.w\3.scale"),
    (_re.compile(r"^layers\.(\d+)\.ffn\.gate\.weight$"), r"layers.\1.mlp.gate.weight"),
    (_re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$"), r"layers.\1.mlp.gate.e_score_correction_bias"),
    (_re.compile(r"^layers\.(\d+)\.hc_attn_(base|fn|scale)$"), r"layers.\1.attn_hc.\2"),
    (_re.compile(r"^layers\.(\d+)\.hc_ffn_(base|fn|scale)$"), r"layers.\1.ffn_hc.\2"),
)

_DROP_RULES: tuple[_re.Pattern[str], ...] = (
    # §2.6 Cluster 6 — DROP (no vendor slot)
    _re.compile(r"^layers\.\d+\.attn\.w(kv|q_a|q_b|o_a|o_b)\.scale$"),
    _re.compile(r"^layers\.\d+\.attn\.compressor\..+$"),
    _re.compile(r"^layers\.\d+\.attn\.indexer\..+$"),
    _re.compile(r"^layers\.\d+\.ffn\.shared_experts\.w[123]\.scale$"),
    _re.compile(r"^layers\.\d+\.ffn\.gate\.tid2eid$"),
    _re.compile(r"^mtp\.\d+\..+$"),
)

# Mirrors vendor deepseek_v4._CSA_WEIGHT_KEYS without importing the vendor
# module during its load-time remap pre-pass.
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


def remap_shimmed_ckpt_keys(
    weights: _Iterable[tuple[str, _Any]] | _Mapping[str, _Any],
) -> tuple[dict[str, _Any], list[str]]:
    """Story 13.1 pure key-string shimmed→vendor-internal remap.

    Touches key strings + dict membership ONLY.  Never reads/mutates tensor
    bytes, dtype, layout, or quantization.

    Returns ``(remapped_mapping, dropped_keys)`` where ``dropped_keys`` is the
    recorded inverse (for §2.6 DROP policy audit).  Raises ``KeyError`` on any
    unmapped residual key (STOP-rule (xiii) — total-coverage invariant).
    """
    if isinstance(weights, _Mapping):
        items = list(weights.items())
    else:
        items = []
        for item in weights:
            if isinstance(item, str):
                items.append((item, None))
            else:
                k, v = item
                items.append((k, v))

    remapped: dict[str, _Any] = {}
    dropped: list[str] = []

    for raw_key, value in items:
        key = str(raw_key)

        # Cluster-6 DROP check first
        if any(pat.match(key) for pat in _DROP_RULES):
            dropped.append(key)
            continue

        # Identity passthrough: already vendor-internal
        if _is_vendor_internal(key):
            remapped[key] = value
            continue

        # Cluster 1-5 RENAME
        renamed = _try_rename(key)
        if renamed is not None:
            remapped[renamed] = value
            continue

        # Unknown residual — STOP-rule (xiii)
        raise KeyError(key)

    return remapped, dropped


def _is_vendor_internal(key: str) -> bool:
    """Return True if *key* is already vendor-internal or HF-canonical.

    HF-canonical ``model.*`` keys are passed through so the vendor
    ``_canonicalize_weight_key`` can strip/translate them.
    """
    if key in {"embed.weight", "norm.weight", "lm_head.weight", "hc_head.base", "hc_head.fn", "hc_head.scale"}:
        return True
    csa_key = key
    if key.startswith("layers."):
        parts = key.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            csa_key = parts[2]
    if csa_key in _CSA_WEIGHT_KEYS:
        return True
    if key.startswith("model.") or key == "lm_head.weight":
        return True
    layer_prefixed = (
        r"^layers\.\d+\."
        r"(input_layernorm\.weight|post_attention_layernorm\.weight"
        r"|q_norm\.weight|kv_norm\.weight|q_a_proj\.weight|q_b_proj\.weight"
        r"|kv_proj\.weight|o_a_proj\.weight|o_b_proj\.weight|sinks"
        r"|attn_hc\.(base|fn|scale)|ffn_hc\.(base|fn|scale)"
        r"|mlp\.gate\.(weight|e_score_correction_bias)"
        r"|mlp\.shared_experts\.w[123]\.weight"
        r"|mlp\.experts\.\d+\.w[123]\.(weight|scale))$"
    )
    bare = (
        r"^(input_layernorm\.weight|post_attention_layernorm\.weight"
        r"|q_norm\.weight|kv_norm\.weight|q_a_proj\.weight|q_b_proj\.weight"
        r"|kv_proj\.weight|o_a_proj\.weight|o_b_proj\.weight|sinks"
        r"|attn_hc\.(base|fn|scale)|ffn_hc\.(base|fn|scale)"
        r"|mlp\.gate\.(weight|e_score_correction_bias)"
        r"|mlp\.shared_experts\.w[123]\.weight"
        r"|mlp\.experts\.\d+\.w[123]\.(weight|scale))$"
    )
    if _re.match(layer_prefixed, key) or _re.match(bare, key):
        return True
    return False


def _try_rename(key: str) -> str | None:
    """Apply _RENAME_RULES; return renamed key or None if no rule matched."""
    for pattern, repl in _RENAME_RULES:
        renamed, count = _re.subn(pattern, repl, key)
        if count:
            return renamed
    return None
