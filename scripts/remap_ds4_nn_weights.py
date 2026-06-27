#!/usr/bin/env python3
"""Remap shimmed DeepSeek V4 Flash checkpoint weights to ``deepseek_v4_nn``.

The input is the already-shimmed HF snapshot produced by
``scripts/shim_ds4_safetensors.py``: dense attention tensors are BF16-readable
and routed experts remain packed FP4 ``*.weight`` plus BF16 ``*.scale`` sidecars.
This script owns the higher-level key-space conversion only:

* add the ``model.`` prefixes used by the MLX ``nn.Module`` tree,
* drop redundant dense BF16 ``*.scale`` sidecars,
* stack per-expert FP4 tensors into the frozen nn expert leaves,
* synthesize the three hash-layer ``e_score_correction_bias`` zero leaves, and
* drop ``mtp.*`` tensors that the nn model does not expose.

Important Story 13.3b-4 invariant: CSA, indexer, and HCA ``ape`` tensors are
copied verbatim in checkpoint token-major layout.  No transpose is performed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class KeyRemap:
    """Single raw-key remap decision."""

    raw: str
    target: str | None
    action: str = "rename"
    transform: str = "identity"
    expert_slot: tuple[int, int, str] | None = None  # (layer, expert, w1_weight/w1_scale/...)


@dataclass(frozen=True)
class TensorSpec:
    """Header-only tensor description for the remapped nn checkpoint."""

    dtype: str
    shape: tuple[int, ...]
    sources: tuple[str, ...] = ()
    transform: str = "identity"


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


def read_safetensors_header(path: pathlib.Path) -> dict[str, Any]:
    """Read only the safetensors JSON header."""

    with pathlib.Path(path).open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: shorter than safetensors header length")
        header_len = int.from_bytes(raw_len, "little")
        header_bytes = fp.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be a JSON object")
    return header


def _index_weight_map(checkpoint_dir: pathlib.Path) -> dict[str, str]:
    index_path = pathlib.Path(checkpoint_dir) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path}: expected object weight_map")
    return {str(key): str(value) for key, value in weight_map.items()}


def _read_checkpoint_header_specs(checkpoint_dir: pathlib.Path) -> dict[str, TensorSpec]:
    root = pathlib.Path(checkpoint_dir)
    weight_map = _index_weight_map(root)
    headers: dict[str, dict[str, Any]] = {}
    specs: dict[str, TensorSpec] = {}
    for raw_key, shard_name in sorted(weight_map.items()):
        if shard_name not in headers:
            headers[shard_name] = read_safetensors_header(root / shard_name)
        meta = headers[shard_name].get(raw_key)
        if not isinstance(meta, dict):
            raise ValueError(f"{root / shard_name}:{raw_key}: missing tensor header")
        dtype = str(meta.get("dtype"))
        shape = meta.get("shape")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            raise ValueError(f"{root / shard_name}:{raw_key}: invalid shape {shape!r}")
        specs[raw_key] = TensorSpec(dtype=dtype, shape=tuple(int(x) for x in shape), sources=(raw_key,))
    return specs


def remap_header_specs(
    raw_specs: Mapping[str, TensorSpec | Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, TensorSpec], RemapReport]:
    """Remap safetensors header specs without reading tensor payload bytes."""

    n_routed_experts = _config_int(config, "n_routed_experts")
    num_hash_layers = _config_int(config, "num_hash_layers", 0)
    out: dict[str, TensorSpec] = {}
    direct_sources: dict[str, str] = {}
    expert_groups: dict[str, dict[int, tuple[str, TensorSpec]]] = defaultdict(dict)
    report = RemapReport(payload_bytes_read=0)

    for raw_key, raw_meta in sorted(raw_specs.items()):
        spec = _coerce_spec(raw_key, raw_meta)
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
            expert_groups[decision.target][expert] = (raw_key, spec)
            report.transforms[decision.target] = decision.transform
            continue
        if decision.target is None:
            raise AssertionError(f"rename without target for {raw_key}")
        if decision.target in out:
            raise ValueError(f"duplicate target {decision.target}: {direct_sources[decision.target]} and {raw_key}")
        out[decision.target] = TensorSpec(
            dtype=_target_dtype(decision.target, spec.dtype),
            shape=spec.shape,
            sources=(raw_key,),
            transform=decision.transform,
        )
        direct_sources[decision.target] = raw_key
        report.transforms[decision.target] = decision.transform

    _validate_expert_sources({k: {eid: raw for eid, (raw, _spec) in v.items()} for k, v in expert_groups.items()}, n_routed_experts)
    for target, by_expert in expert_groups.items():
        ordered = [by_expert[eid] for eid in range(n_routed_experts)]
        first_shape = ordered[0][1].shape
        first_dtype = ordered[0][1].dtype
        for raw_key, spec in ordered[1:]:
            if spec.shape != first_shape:
                raise ValueError(f"{target}: expert shape mismatch at {raw_key}: {spec.shape} != {first_shape}")
            if spec.dtype != first_dtype:
                raise ValueError(f"{target}: expert dtype mismatch at {raw_key}: {spec.dtype} != {first_dtype}")
        dtype = "U8" if target.endswith("_weight") and first_dtype in {"I8", "U8"} else first_dtype
        out[target] = TensorSpec(
            dtype=dtype,
            shape=(n_routed_experts, *first_shape),
            sources=tuple(raw for raw, _spec in ordered),
            transform=report.transforms.get(target, "stack"),
        )
        report.stacked_targets[target] = tuple(raw for raw, _spec in ordered)

    for key in _zero_bias_keys(num_hash_layers):
        if key not in out:
            out[key] = TensorSpec(dtype="F32", shape=(n_routed_experts,), sources=(), transform="zero")
            report.synthetic_zero_keys.add(key)
            report.transforms[key] = "zero"

    return out, report


def remap_header_specs_from_checkpoint(
    checkpoint_dir: str | pathlib.Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, TensorSpec], RemapReport]:
    """Read a checkpoint index + safetensors headers and return remapped specs."""

    root = pathlib.Path(checkpoint_dir)
    cfg = _load_config(root, config)
    return remap_header_specs(_read_checkpoint_header_specs(root), config=cfg)


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


def _coerce_spec(raw_key: str, raw_meta: TensorSpec | Mapping[str, Any]) -> TensorSpec:
    if isinstance(raw_meta, TensorSpec):
        return raw_meta
    dtype = raw_meta.get("dtype")
    shape = raw_meta.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, (list, tuple)):
        raise ValueError(f"{raw_key}: expected dtype+shape metadata")
    return TensorSpec(dtype=dtype, shape=tuple(int(x) for x in shape), sources=(raw_key,))


def _target_dtype(target: str, dtype: str) -> str:
    if target.endswith(".tid2eid"):
        return "I32"
    if target.endswith("_weight") and dtype == "I8":
        return "U8"
    return dtype


def _convert_value(value: Any, transform: str, mx: Any) -> Any:
    if transform == "cast_int32":
        return value.astype(mx.int32) if hasattr(value, "astype") else mx.array(value, dtype=mx.int32)
    if transform == "stack_uint8":
        return value.astype(mx.uint8) if hasattr(value, "astype") else mx.array(value, dtype=mx.uint8)
    return value


def _load_config(root: pathlib.Path, config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is not None:
        return dict(config)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "deepseek_v4_nn"
    return cfg


def _count_dropped(dropped: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key in dropped:
        if key.startswith("mtp."):
            counts["mtp"] += 1
        elif key.endswith(".scale"):
            counts["scale_sidecar"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def _cmd_plan(args: argparse.Namespace) -> int:
    config = None
    if args.config is not None:
        config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
        config["model_type"] = "deepseek_v4_nn"
    specs, report = remap_header_specs_from_checkpoint(args.src, config=config)
    summary = {
        "output_keys": len(specs),
        "dropped_keys": len(report.dropped_keys),
        "dropped_counts": _count_dropped(report.dropped_keys),
        "synthetic_zero_keys": sorted(report.synthetic_zero_keys),
        "stacked_targets": len(report.stacked_targets),
        "payload_bytes_read": report.payload_bytes_read,
    }
    if args.verbose:
        summary["keys"] = sorted(specs)
        summary["report"] = report.as_jsonable()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="header/index-only remap audit; reads zero tensor payload bytes")
    plan.add_argument("--src", required=True, type=pathlib.Path, help="already-shimmed checkpoint directory")
    plan.add_argument("--config", type=pathlib.Path, default=None, help="optional config.json override")
    plan.add_argument("--verbose", action="store_true")
    plan.set_defaults(func=_cmd_plan)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
