"""Shared ckpt-native -> ``deepseek_v4_nn`` remap core.

This module is the single source of truth for Story 13.3b remapping used by
both ``scripts/remap_ds4_nn_weights.py`` header planning and the nn
``Model.load_weights`` materialized load path.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KeyRemap:
    """Single raw-key remap decision."""

    raw: str
    target: str | None
    action: str = "rename"
    transform: str = "identity"
    expert_slot: tuple[int, int, str] | None = None  # (layer, expert, w1_weight/w1_scale/...)


@dataclass
class RemapReport:
    """Audit information emitted by header and materialized remaps."""

    dropped_keys: list[str] = field(default_factory=list)
    synthetic_zero_keys: set[str] = field(default_factory=set)
    stacked_targets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    transforms: dict[str, str] = field(default_factory=dict)
    payload_bytes_read: int = 0

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "dropped_keys": sorted(self.dropped_keys),
            "synthetic_zero_keys": sorted(self.synthetic_zero_keys),
            "stacked_targets": {key: list(value) for key, value in sorted(self.stacked_targets.items())},
            "transforms": dict(sorted(self.transforms.items())),
            "payload_bytes_read": self.payload_bytes_read,
        }


_LAYER = r"layers\.(\d+)"
_EXPERT = r"experts\.(\d+)"

_DROP_PATTERNS = (
    re.compile(r"^mtp\.\d+\..+$"),
    re.compile(r"^layers\.\d+\.attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)\.scale$"),
    re.compile(r"^layers\.\d+\.attn\.indexer\.wq_b\.scale$"),
    re.compile(r"^layers\.\d+\.ffn\.shared_experts\.w[123]\.scale$"),
)

_DIRECT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^embed\.weight$"), "model.embed_tokens.weight"),
    (re.compile(r"^head\.weight$"), "lm_head.weight"),
    (re.compile(r"^norm\.weight$"), "model.norm.weight"),
    (re.compile(r"^hc_head_fn$"), "model.hc_head.fn"),
    (re.compile(r"^hc_head_base$"), "model.hc_head.base"),
    (re.compile(r"^hc_head_scale$"), "model.hc_head.scale"),
    (re.compile(rf"^{_LAYER}\.attn_norm\.weight$"), r"model.layers.\1.input_layernorm.weight"),
    (re.compile(rf"^{_LAYER}\.ffn_norm\.weight$"), r"model.layers.\1.post_attention_layernorm.weight"),
    (re.compile(rf"^{_LAYER}\.hc_attn_(fn|base|scale)$"), r"model.layers.\1.attn_hc.\2"),
    (re.compile(rf"^{_LAYER}\.hc_ffn_(fn|base|scale)$"), r"model.layers.\1.ffn_hc.\2"),
    (re.compile(rf"^{_LAYER}\.attn\.wq_a\.weight$"), r"model.layers.\1.self_attn.q_a_proj.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.wq_b\.weight$"), r"model.layers.\1.self_attn.q_b_proj.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.wkv\.weight$"), r"model.layers.\1.self_attn.kv_proj.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.wo_a\.weight$"), r"model.layers.\1.self_attn.o_a_proj.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.wo_b\.weight$"), r"model.layers.\1.self_attn.o_b_proj.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.q_norm\.weight$"), r"model.layers.\1.self_attn.q_norm.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.kv_norm\.weight$"), r"model.layers.\1.self_attn.kv_norm.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.attn_sink$"), r"model.layers.\1.self_attn.sinks"),
    (re.compile(rf"^{_LAYER}\.attn\.compressor\.wkv\.weight$"), r"model.layers.\1.self_attn.compressor.wkv.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.compressor\.wgate\.weight$"), r"model.layers.\1.self_attn.compressor.wgate.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.compressor\.ape$"), r"model.layers.\1.self_attn.compressor.ape"),
    (re.compile(rf"^{_LAYER}\.attn\.compressor\.norm\.weight$"), r"model.layers.\1.self_attn.compressor.norm.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.indexer\.compressor\.wkv\.weight$"), r"model.layers.\1.self_attn.indexer.compressor.wkv.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.indexer\.compressor\.wgate\.weight$"), r"model.layers.\1.self_attn.indexer.compressor.wgate.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.indexer\.compressor\.ape$"), r"model.layers.\1.self_attn.indexer.compressor.ape"),
    (re.compile(rf"^{_LAYER}\.attn\.indexer\.compressor\.norm\.weight$"), r"model.layers.\1.self_attn.indexer.compressor.norm.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.indexer\.weights_proj\.weight$"), r"model.layers.\1.self_attn.indexer.weights_proj.weight"),
    (re.compile(rf"^{_LAYER}\.attn\.indexer\.wq_b\.weight$"), r"model.layers.\1.self_attn.indexer.wq_b.weight"),
    (re.compile(rf"^{_LAYER}\.ffn\.gate\.weight$"), r"model.layers.\1.mlp.gate_weight"),
    (re.compile(rf"^{_LAYER}\.ffn\.gate\.bias$"), r"model.layers.\1.mlp.e_score_correction_bias"),
    (re.compile(rf"^{_LAYER}\.ffn\.gate\.tid2eid$"), r"model.layers.\1.mlp.tid2eid"),
    (re.compile(rf"^{_LAYER}\.ffn\.shared_experts\.w1\.weight$"), r"model.layers.\1.mlp.shared_experts.gate_proj.weight"),
    (re.compile(rf"^{_LAYER}\.ffn\.shared_experts\.w2\.weight$"), r"model.layers.\1.mlp.shared_experts.down_proj.weight"),
    (re.compile(rf"^{_LAYER}\.ffn\.shared_experts\.w3\.weight$"), r"model.layers.\1.mlp.shared_experts.up_proj.weight"),
)

_EXPERT_RULE = re.compile(rf"^{_LAYER}\.ffn\.{_EXPERT}\.w([123])\.(weight|scale)$")


def _get_mx():
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without MLX installed.
        raise RuntimeError("mlx is required for materialized remap_weight_dict") from exc
    return mx


def _config_int(config: Mapping[str, Any], name: str, default: int | None = None) -> int:
    if name in config and config[name] is not None:
        return int(config[name])
    if default is not None:
        return default
    raise KeyError(f"missing required config field {name!r}")


def remap_key(raw_key: str) -> KeyRemap:
    """Map one shimmed checkpoint key to a ``deepseek_v4_nn`` key decision.

    Raises ``KeyError`` for unknown non-dropped keys; callers treat that as the
    Story 13.3b-4 orphan-tensor STOP detector.
    """

    key = str(raw_key)
    if any(pattern.match(key) for pattern in _DROP_PATTERNS):
        return KeyRemap(raw=key, target=None, action="drop", transform="drop")

    expert_match = _EXPERT_RULE.match(key)
    if expert_match:
        layer = int(expert_match.group(1))
        expert = int(expert_match.group(2))
        wnum = expert_match.group(3)
        kind = expert_match.group(4)
        leaf = f"w{wnum}_{kind}"
        target = f"model.layers.{layer}.mlp.experts.{leaf}"
        transform = "stack_uint8" if kind == "weight" else "stack"
        return KeyRemap(raw=key, target=target, action="stack", transform=transform, expert_slot=(layer, expert, leaf))

    for pattern, replacement in _DIRECT_RULES:
        renamed, count = pattern.subn(replacement, key)
        if count:
            transform = "identity"
            if renamed.endswith(".tid2eid"):
                transform = "cast_int32"
            return KeyRemap(raw=key, target=renamed, action="rename", transform=transform)

    raise KeyError(key)


def _zero_bias_keys(num_hash_layers: int) -> set[str]:
    return {f"model.layers.{layer}.mlp.e_score_correction_bias" for layer in range(num_hash_layers)}


def _validate_expert_sources(groups: Mapping[str, Mapping[int, str]], n_routed_experts: int) -> None:
    expected = set(range(n_routed_experts))
    for target, by_expert in groups.items():
        got = set(by_expert)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise ValueError(f"{target}: incomplete expert stack missing={missing[:10]} extra={extra[:10]}")


def remap_weight_dict(
    weights: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], RemapReport]:
    """Remap materialized tensors to the ``deepseek_v4_nn`` parameter tree."""

    mx = _get_mx()
    n_routed_experts = _config_int(config, "n_routed_experts")
    num_hash_layers = _config_int(config, "num_hash_layers", 0)
    items = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    out: dict[str, Any] = {}
    direct_sources: dict[str, str] = {}
    expert_groups: dict[str, dict[int, tuple[str, Any, str]]] = defaultdict(dict)
    report = RemapReport(payload_bytes_read=0)

    for raw_key, value in sorted(((str(k), v) for k, v in items), key=lambda item: item[0]):
        decision = remap_key(raw_key)
        if decision.action == "drop":
            report.dropped_keys.append(raw_key)
            continue
        if decision.action == "stack":
            if decision.target is None or decision.expert_slot is None:
                raise AssertionError(f"invalid expert remap decision for {raw_key}")
            _layer, expert, _leaf = decision.expert_slot
            if expert in expert_groups[decision.target]:
                raise ValueError(f"duplicate expert {expert} for {decision.target}")
            expert_groups[decision.target][expert] = (raw_key, value, decision.transform)
            report.transforms[decision.target] = decision.transform
            continue
        if decision.target is None:
            raise AssertionError(f"rename without target for {raw_key}")
        if decision.target in out:
            raise ValueError(f"duplicate target {decision.target}: {direct_sources[decision.target]} and {raw_key}")
        out[decision.target] = _convert_value(value, decision.transform, mx)
        direct_sources[decision.target] = raw_key
        report.transforms[decision.target] = decision.transform

    _validate_expert_sources({k: {eid: raw for eid, (raw, _value, _transform) in v.items()} for k, v in expert_groups.items()}, n_routed_experts)
    for target, by_expert in expert_groups.items():
        ordered = [by_expert[eid] for eid in range(n_routed_experts)]
        values = [_convert_value(value, transform, mx) for _raw, value, transform in ordered]
        out[target] = mx.stack(values, axis=0)
        report.stacked_targets[target] = tuple(raw for raw, _value, _transform in ordered)

    for key in _zero_bias_keys(num_hash_layers):
        if key not in out:
            out[key] = mx.zeros((_config_int(config, "n_routed_experts"),), dtype=mx.float32)
            report.synthetic_zero_keys.add(key)
            report.transforms[key] = "zero"

    return out, report


def _convert_value(value: Any, transform: str, mx: Any) -> Any:
    if transform == "cast_int32":
        return value.astype(mx.int32) if hasattr(value, "astype") else mx.array(value, dtype=mx.int32)
    if transform == "stack_uint8":
        return value.astype(mx.uint8) if hasattr(value, "astype") else mx.array(value, dtype=mx.uint8)
    return value


__all__ = [
    "KeyRemap",
    "RemapReport",
    "remap_key",
    "remap_weight_dict",
]
