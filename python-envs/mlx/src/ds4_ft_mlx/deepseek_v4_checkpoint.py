"""Bounded DeepSeek V4 checkpoint validation and selected tensor loading.

Header validation and dry-run load planning inspect only the Hugging Face
safetensors index plus safetensors JSON headers.  Execute-mode selected loading
is budget-gated and reads only explicitly selected tensor byte ranges.  This
module deliberately does not perform model conversion or claim full load/forward
parity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ds4_ft_mlx.deepseek_v4_dequant import read_safetensors_header
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs


DEFAULT_SHIMMED_INDEX = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/model.safetensors.index.json")


_LAYER_RE = re.compile(r"^layers\.(?P<layer>\d+)\.(?P<rest>.+)$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(?P<expert>\d+)\.(?P<rest>w[123]\.(?:weight|scale))$")
_DEFAULTABLE_ZERO_KEYS = {
    "mlp.gate.e_score_correction_bias": "Transformers DeepseekV4MoEGate registers e_score_correction_bias as a persistent zero buffer",
}


_CHECKPOINT_LAYER0_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.wq_a.weight": "q_a_proj.weight",
    "attn.wq_a.scale": "q_a_proj.scale",
    "attn.q_norm.weight": "q_norm.weight",
    "attn.wq_b.weight": "q_b_proj.weight",
    "attn.wq_b.scale": "q_b_proj.scale",
    "attn.wkv.weight": "kv_proj.weight",
    "attn.wkv.scale": "kv_proj.scale",
    "attn.kv_norm.weight": "kv_norm.weight",
    "attn.wo_a.weight": "o_a_proj.weight",
    "attn.wo_a.scale": "o_a_proj.scale",
    "attn.wo_b.weight": "o_b_proj.weight",
    "attn.wo_b.scale": "o_b_proj.scale",
    "attn.attn_sink": "sinks",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.w1.weight",
    "ffn.shared_experts.w1.scale": "mlp.shared_experts.w1.scale",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale": "mlp.shared_experts.w2.scale",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.w3.weight",
    "ffn.shared_experts.w3.scale": "mlp.shared_experts.w3.scale",
}


def canonicalize_checkpoint_key(name: str, *, layer: int = 0) -> str | None:
    """Map shimmed-checkpoint tensor names to bounded real-model keys.

    Only the selected decoder layer is canonicalized.  Other layers and MTP
    tensors return ``None`` because the bounded real model currently covers a
    single synthetic layer only.
    """

    if name == "embed.weight":
        return "embed.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "norm.weight"
    if name.startswith("mtp."):
        return None

    m = _LAYER_RE.match(name)
    if not m:
        return None
    if int(m.group("layer")) != int(layer):
        return None
    rest = m.group("rest")
    if rest in _CHECKPOINT_LAYER0_MAP:
        return _CHECKPOINT_LAYER0_MAP[rest]

    routed_prefix = "ffn.experts."
    if rest.startswith(routed_prefix):
        tail = rest[len(routed_prefix) :]
        parts = tail.split(".")
        if len(parts) == 3 and parts[0].isdigit() and parts[1] in {"w1", "w2", "w3"} and parts[2] in {"weight", "scale"}:
            return f"mlp.experts.{parts[0]}.{parts[1]}.{parts[2]}"
    return None


def bounded_real_required_keys(*, expert_dtype: str = "i8", n_routed_experts: int = 2) -> tuple[str, ...]:
    if n_routed_experts != 2:
        raise ValueError("bounded real checkpoint validation currently supports only n_routed_experts=2")
    required = set(Model._INTEGRATED_WEIGHTS)
    if expert_dtype.lower() == "i8":
        required |= {
            f"mlp.experts.{eid}.{proj}.scale"
            for eid in range(n_routed_experts)
            for proj in ("w1", "w2", "w3")
        }
    return tuple(sorted(required))


def bounded_real_expected_metadata(*, expert_dtype: str = "i8", n_routed_experts: int = 2) -> dict[str, dict[str, object]]:
    """Expected header dtype/shape metadata for the real V4 Flash bounded layer-0 subset."""

    if n_routed_experts != 2:
        raise ValueError("bounded real checkpoint validation currently supports only n_routed_experts=2")
    expected: dict[str, dict[str, object]] = {
        "embed.weight": {"dtype": "BF16", "shape": [129280, 4096]},
        "input_layernorm.weight": {"dtype": "BF16", "shape": [4096]},
        "post_attention_layernorm.weight": {"dtype": "BF16", "shape": [4096]},
        "q_a_proj.weight": {"dtype": "BF16", "shape": [1024, 4096]},
        "q_norm.weight": {"dtype": "BF16", "shape": [1024]},
        "q_b_proj.weight": {"dtype": "BF16", "shape": [32768, 1024]},
        "kv_proj.weight": {"dtype": "BF16", "shape": [512, 4096]},
        "kv_norm.weight": {"dtype": "BF16", "shape": [512]},
        "o_a_proj.weight": {"dtype": "BF16", "shape": [8192, 4096]},
        "o_b_proj.weight": {"dtype": "BF16", "shape": [4096, 8192]},
        "sinks": {"dtype": "F32", "shape": [64]},
        "attn_hc.fn": {"dtype": "F32", "shape": [24, 16384]},
        "attn_hc.base": {"dtype": "F32", "shape": [24]},
        "attn_hc.scale": {"dtype": "F32", "shape": [3]},
        "ffn_hc.fn": {"dtype": "F32", "shape": [24, 16384]},
        "ffn_hc.base": {"dtype": "F32", "shape": [24]},
        "ffn_hc.scale": {"dtype": "F32", "shape": [3]},
        "mlp.gate.weight": {"dtype": "BF16", "shape": [256, 4096]},
        "mlp.gate.e_score_correction_bias": {"dtype": "F32", "shape": [256]},
        "mlp.shared_experts.w1.weight": {"dtype": "BF16", "shape": [2048, 4096]},
        "mlp.shared_experts.w2.weight": {"dtype": "BF16", "shape": [4096, 2048]},
        "mlp.shared_experts.w3.weight": {"dtype": "BF16", "shape": [2048, 4096]},
    }
    if expert_dtype.lower() == "i8":
        for eid in range(n_routed_experts):
            expected[f"mlp.experts.{eid}.w1.weight"] = {"dtype": "I8", "shape": [2048, 2048]}
            expected[f"mlp.experts.{eid}.w1.scale"] = {"dtype": "BF16", "shape": [2048, 128]}
            expected[f"mlp.experts.{eid}.w2.weight"] = {"dtype": "I8", "shape": [4096, 1024]}
            expected[f"mlp.experts.{eid}.w2.scale"] = {"dtype": "BF16", "shape": [4096, 64]}
            expected[f"mlp.experts.{eid}.w3.weight"] = {"dtype": "I8", "shape": [2048, 2048]}
            expected[f"mlp.experts.{eid}.w3.scale"] = {"dtype": "BF16", "shape": [2048, 128]}
    return expected


def _index_hash_and_data(index_path: Path) -> tuple[str, dict[str, Any]]:
    raw = index_path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{index_path}: malformed safetensors index JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict):
        raise ValueError(f"{index_path}: safetensors index must contain an object weight_map")
    return hashlib.sha256(raw).hexdigest(), data


def _family(key: str) -> str:
    m = _EXPERT_RE.match(key)
    if m:
        return f"mlp.experts.*.{m.group('rest')}"
    return key


def _header_for_file(headers: dict[str, dict[str, dict[str, object]]], shard_path: Path) -> dict[str, dict[str, object]]:
    cache_key = str(shard_path)
    if cache_key not in headers:
        if not shard_path.exists():
            raise FileNotFoundError(f"missing safetensors shard referenced by index: {shard_path}")
        headers[cache_key] = read_safetensors_header(shard_path)
    return headers[cache_key]


_STACKED_EXPERT_RE = re.compile(r"^ffn\.experts\.(?P<expert>\d+)\.(?P<rest>w[123]\.(?:weight|scale))$")

_STACKED_CORE_SHAPES: dict[str, list[int]] = {
    "attn.attn_sink": [64],
    "attn.kv_norm.weight": [512],
    "attn.q_norm.weight": [1024],
    "attn.wkv.scale": [4, 32],
    "attn.wkv.weight": [512, 4096],
    "attn.wo_a.scale": [64, 32],
    "attn.wo_a.weight": [8192, 4096],
    "attn.wo_b.scale": [32, 64],
    "attn.wo_b.weight": [4096, 8192],
    "attn.wq_a.scale": [8, 32],
    "attn.wq_a.weight": [1024, 4096],
    "attn.wq_b.scale": [256, 8],
    "attn.wq_b.weight": [32768, 1024],
    "attn_norm.weight": [4096],
    "ffn.gate.weight": [256, 4096],
    "ffn.shared_experts.w1.scale": [16, 32],
    "ffn.shared_experts.w1.weight": [2048, 4096],
    "ffn.shared_experts.w2.scale": [32, 16],
    "ffn.shared_experts.w2.weight": [4096, 2048],
    "ffn.shared_experts.w3.scale": [16, 32],
    "ffn.shared_experts.w3.weight": [2048, 4096],
    "ffn_norm.weight": [4096],
    "hc_attn_base": [24],
    "hc_attn_fn": [24, 16384],
    "hc_attn_scale": [3],
    "hc_ffn_base": [24],
    "hc_ffn_fn": [24, 16384],
    "hc_ffn_scale": [3],
}

_STACKED_ROUTER_AUX_SHAPES = {
    "ffn.gate.bias": [256],
    "ffn.gate.e_score_correction_bias": [256],
    "ffn.gate.tid2eid": [129280, 6],
}

_STACKED_EXPERT_SHAPES = {
    "w1.weight": [2048, 2048],
    "w1.scale": [2048, 128],
    "w2.weight": [4096, 1024],
    "w2.scale": [4096, 64],
    "w3.weight": [2048, 2048],
    "w3.scale": [2048, 128],
}

_STACKED_COMPRESSOR_FAMILIES = frozenset({
    "attn.compressor.ape",
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
})

_STACKED_INDEXER_FAMILIES = frozenset({
    "attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight",
    "attn.indexer.compressor.wgate.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wq_b.scale",
    "attn.indexer.wq_b.weight",
})

_STACKED_STRUCTURAL_FAMILIES = _STACKED_COMPRESSOR_FAMILIES | _STACKED_INDEXER_FAMILIES

_STACKED_TOP_LEVEL_SHAPES = {
    "embed.weight": [129280, 4096],
    "head.weight": [129280, 4096],
    "norm.weight": [4096],
    "hc_head_fn": [4, 16384],
    "hc_head_base": [4],
    "hc_head_scale": [1],
}


def _meta_shape(meta: Mapping[str, object] | None) -> list[int] | None:
    if meta is None:
        return None
    shape = meta.get("shape")
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes, bytearray)):
        return [int(dim) for dim in shape]
    return None


def _stacked_family(rest: str) -> str:
    match = _STACKED_EXPERT_RE.match(rest)
    if match:
        return f"ffn.experts.N.{match.group('rest')}"
    return rest


def _stacked_known_family(family: str) -> bool:
    if family in _STACKED_CORE_SHAPES or family in _STACKED_ROUTER_AUX_SHAPES or family in _STACKED_STRUCTURAL_FAMILIES:
        return True
    if family.startswith("ffn.experts.N.") and family[len("ffn.experts.N.") :] in _STACKED_EXPERT_SHAPES:
        return True
    return False


def _expected_stacked_structural_families(layer: int) -> set[str]:
    """Real V4 Flash structural families expected by layer id, names only."""

    families: set[str] = set()
    if layer >= 2:
        families.update(_STACKED_COMPRESSOR_FAMILIES)
    if layer == 2 or (layer >= 4 and layer % 2 == 0):
        families.update(_STACKED_INDEXER_FAMILIES)
    return families


def _issue(layer: int | None, key: str, expected_shape: list[int] | None, observed_shape: list[int] | None) -> dict[str, object]:
    return {"layer": layer, "key": key, "expected_shape": expected_shape, "observed_shape": observed_shape}


def stacked_shape_compatibility_report(
    header_meta: Mapping[str, Mapping[str, object]],
    *,
    expected_num_layers: int,
    n_routed_experts: int,
    expert_dtype: str = "i8",
) -> dict[str, object]:
    """Header-only structural shape-compatibility report for stacked V4 layers."""

    if expected_num_layers <= 0:
        raise StackedShapeCompatError("expected_num_layers must be positive")
    if n_routed_experts <= 0:
        raise StackedShapeCompatError("n_routed_experts must be positive")
    if str(expert_dtype).lower() != "i8":
        raise StackedShapeCompatError("stacked shape compatibility currently supports expert_dtype='i8'")

    layer_meta: dict[int, dict[str, Mapping[str, object]]] = {}
    layer_families: dict[int, set[str]] = {}
    expert_indices: dict[int, set[int]] = {}
    unclassified: set[str] = set()
    core_issues: list[dict[str, object]] = []
    structural_issues: list[dict[str, object]] = []
    blockers: list[str] = []

    for name in sorted(header_meta):
        if name.startswith("mtp."):
            continue
        match = _LAYER_RE.match(name)
        if not match:
            continue
        layer = int(match.group("layer"))
        rest = match.group("rest")
        layer_meta.setdefault(layer, {})[rest] = header_meta[name]
        family = _stacked_family(rest)
        layer_families.setdefault(layer, set()).add(family)
        expert_match = _STACKED_EXPERT_RE.match(rest)
        if expert_match:
            expert_indices.setdefault(layer, set()).add(int(expert_match.group("expert")))
        if not _stacked_known_family(family):
            unclassified.add(family)

    observed_layers = sorted(layer_meta)
    expected_layers = set(range(expected_num_layers))
    missing_layers = [layer for layer in range(expected_num_layers) if layer not in layer_meta]
    extra_layers = [layer for layer in observed_layers if layer not in expected_layers]
    if missing_layers:
        blockers.append(f"missing layer(s): {missing_layers}")
    if extra_layers:
        blockers.append(f"extra layer(s): {extra_layers}")

    for key, expected_shape in sorted(_STACKED_TOP_LEVEL_SHAPES.items()):
        observed_shape = _meta_shape(header_meta.get(key))
        if observed_shape != expected_shape:
            core_issues.append(_issue(None, key, expected_shape, observed_shape))

    expected_experts = set(range(n_routed_experts))
    for layer in sorted(expected_layers & set(observed_layers)):
        per_layer = layer_meta[layer]
        for family in sorted(_expected_stacked_structural_families(layer)):
            if family not in per_layer:
                structural_issues.append(_issue(layer, f"layers.{layer}.{family}", None, None))
        for rest, expected_shape in sorted(_STACKED_CORE_SHAPES.items()):
            observed_shape = _meta_shape(per_layer.get(rest))
            if observed_shape != expected_shape:
                core_issues.append(_issue(layer, f"layers.{layer}.{rest}", expected_shape, observed_shape))
        aux_present = [rest for rest in sorted(_STACKED_ROUTER_AUX_SHAPES) if rest in per_layer]
        if not aux_present:
            core_issues.append(_issue(layer, f"layers.{layer}.ffn.gate.bias|tid2eid", [256], None))
        for rest in aux_present:
            observed_shape = _meta_shape(per_layer.get(rest))
            expected_shape = _STACKED_ROUTER_AUX_SHAPES[rest]
            if observed_shape != expected_shape:
                core_issues.append(_issue(layer, f"layers.{layer}.{rest}", expected_shape, observed_shape))
        actual_experts = expert_indices.get(layer, set())
        if actual_experts != expected_experts:
            missing = sorted(expected_experts - actual_experts)
            extra = sorted(actual_experts - expected_experts)
            blockers.append(f"layers.{layer} expert index gap: missing={missing} extra={extra}")
        for eid in range(n_routed_experts):
            for rest, expected_shape in sorted(_STACKED_EXPERT_SHAPES.items()):
                key = f"ffn.experts.{eid}.{rest}"
                observed_shape = _meta_shape(per_layer.get(key))
                if observed_shape != expected_shape:
                    core_issues.append(_issue(layer, f"layers.{layer}.{key}", expected_shape, observed_shape))

    signature_groups: dict[tuple[str, ...], list[int]] = {}
    for layer in observed_layers:
        signature = tuple(sorted(layer_families.get(layer, set())))
        signature_groups.setdefault(signature, []).append(layer)
    templates = [
        {"signature": list(signature), "layers": layers}
        for signature, layers in sorted(signature_groups.items(), key=lambda item: item[1][0])
    ]

    conflicts: list[dict[str, object]] = []
    for signature, layers in sorted(signature_groups.items(), key=lambda item: item[1][0]):
        for family in signature:
            by_shape: dict[tuple[int, ...], list[int]] = {}
            for layer in layers:
                shapes = {
                    tuple(_meta_shape(meta) or [])
                    for rest, meta in layer_meta[layer].items()
                    if _stacked_family(rest) == family
                }
                for shape in shapes:
                    by_shape.setdefault(shape, []).append(layer)
            if len(by_shape) > 1:
                conflicts.append({
                    "family": family,
                    "layers": layers,
                    "shapes": [list(shape) for shape in sorted(by_shape)],
                })

    if core_issues:
        blockers.append(f"core family shape/presence issue(s): {len(core_issues)}")
    if structural_issues:
        missing_keys = ", ".join(str(issue["key"]) for issue in structural_issues[:5])
        blockers.append(f"structural family presence issue(s): {len(structural_issues)} missing {missing_keys}")
    if conflicts:
        blockers.append(f"intra-template shape conflict(s): {len(conflicts)}")
    if unclassified:
        blockers.append("unclassified families: " + ", ".join(sorted(unclassified)))

    return {
        "ok": not blockers,
        "num_layers_observed": len(observed_layers),
        "expected_num_layers": int(expected_num_layers),
        "templates": templates,
        "missing_layers": missing_layers,
        "extra_layers": extra_layers,
        "core_family_issues": core_issues,
        "structural_family_issues": structural_issues,
        "intra_template_shape_conflicts": conflicts,
        "unclassified_families": sorted(unclassified),
        "blockers": sorted(blockers),
    }


def assert_stacked_shape_compatible(
    header_meta: Mapping[str, Mapping[str, object]],
    *,
    expected_num_layers: int,
    n_routed_experts: int,
    expert_dtype: str = "i8",
) -> dict[str, object]:
    report = stacked_shape_compatibility_report(
        header_meta,
        expected_num_layers=expected_num_layers,
        n_routed_experts=n_routed_experts,
        expert_dtype=expert_dtype,
    )
    if not report["ok"]:
        blockers = report.get("blockers", [])
        message = str(blockers[0] if isinstance(blockers, list) and blockers else "stacked shape compatibility failed")
        for issue_key in ("core_family_issues", "structural_family_issues"):
            for issue in report.get(issue_key, []):
                if isinstance(issue, Mapping):
                    message += f"; {issue.get('key')} expected={issue.get('expected_shape')} observed={issue.get('observed_shape')}"
                    break
            if ";" in message:
                break
        raise StackedShapeCompatError(message)
    return report


def stacked_shape_compatibility_from_index(
    index_path: str | Path = DEFAULT_SHIMMED_INDEX,
    *,
    checkpoint_dir: str | Path | None = None,
    expected_num_layers: int,
    n_routed_experts: int,
    expert_dtype: str = "i8",
    max_header_bytes: int = 128 * 1024 * 1024,
) -> dict[str, object]:
    """Run stacked shape-compatibility from a safetensors index, header-only."""

    index = Path(index_path)
    base_dir = Path(checkpoint_dir) if checkpoint_dir is not None else index.parent
    index_hash, data = _index_hash_and_data(index)
    weight_map = {str(k): str(v) for k, v in data["weight_map"].items()}
    headers: dict[str, dict[str, dict[str, object]]] = {}
    header_meta: dict[str, dict[str, object]] = {}
    for name, shard in sorted(weight_map.items()):
        if name.startswith("mtp."):
            continue
        if not (name.startswith("layers.") or name in _STACKED_TOP_LEVEL_SHAPES):
            continue
        shard_path = base_dir / shard
        cache_key = str(shard_path)
        if cache_key not in headers:
            if not shard_path.exists():
                raise FileNotFoundError(f"missing safetensors shard referenced by index: {shard_path}")
            headers[cache_key] = read_safetensors_header(shard_path, max_bytes=max_header_bytes)
        meta = headers[cache_key].get(name)
        if meta is not None:
            header_meta[name] = {"dtype": meta.get("dtype"), "shape": meta.get("shape")}
    report = stacked_shape_compatibility_report(
        header_meta,
        expected_num_layers=expected_num_layers,
        n_routed_experts=n_routed_experts,
        expert_dtype=expert_dtype,
    )
    return {**report, "index_sha256": index_hash, "headers_read": sorted(headers)}


def validate_bounded_real_checkpoint_headers(
    index_path: str | Path = DEFAULT_SHIMMED_INDEX,
    *,
    checkpoint_dir: str | Path | None = None,
    layer: int = 0,
    expert_dtype: str = "i8",
    n_routed_experts: int = 2,
    expected_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate bounded real-model tensor coverage using index/header data only."""

    index = Path(index_path)
    base_dir = Path(checkpoint_dir) if checkpoint_dir is not None else index.parent
    index_hash, data = _index_hash_and_data(index)
    weight_map = {str(k): str(v) for k, v in data["weight_map"].items()}
    required = set(bounded_real_required_keys(expert_dtype=expert_dtype, n_routed_experts=n_routed_experts))
    expected = dict(expected_metadata) if expected_metadata is not None else bounded_real_expected_metadata(expert_dtype=expert_dtype, n_routed_experts=n_routed_experts)

    canonical_sources: dict[str, str] = {}
    unexpected_counter: Counter[str] = Counter()
    ignored_counter: Counter[str] = Counter()
    for source_name in weight_map:
        canonical = canonicalize_checkpoint_key(source_name, layer=layer)
        if canonical is None:
            if source_name.startswith("layers."):
                ignored_counter["other_layer_or_unmapped_layer_tensor"] += 1
            elif source_name.startswith("mtp."):
                ignored_counter["mtp_tensor"] += 1
            else:
                ignored_counter["unmapped_top_level_tensor"] += 1
            continue
        if canonical in required and canonical not in canonical_sources:
            canonical_sources[canonical] = source_name
        elif canonical not in required:
            unexpected_counter[_family(canonical)] += 1

    headers: dict[str, dict[str, dict[str, object]]] = {}
    matched: dict[str, dict[str, Any]] = {}
    header_missing: list[str] = []
    for canonical, source_name in sorted(canonical_sources.items()):
        shard = weight_map[source_name]
        header = _header_for_file(headers, base_dir / shard)
        meta = header.get(source_name)
        if meta is None:
            header_missing.append(source_name)
            matched[canonical] = {"source_name": source_name, "file": shard, "dtype": None, "shape": None}
            continue
        matched[canonical] = {
            "source_name": source_name,
            "file": shard,
            "dtype": meta.get("dtype"),
            "shape": list(meta.get("shape", [])) if isinstance(meta.get("shape"), list) else meta.get("shape"),
        }

    unresolved_missing = set(required - set(matched))
    synthetic_defaults: dict[str, dict[str, object]] = {}
    if "mlp.gate.e_score_correction_bias" in unresolved_missing and "mlp.gate.weight" in matched:
        gate_shape = matched["mlp.gate.weight"].get("shape")
        default_shape = [gate_shape[0]] if isinstance(gate_shape, list) and gate_shape else None
        synthetic_defaults["mlp.gate.e_score_correction_bias"] = {
            "dtype": "F32",
            "shape": default_shape,
            "value": "zeros",
            "proof": _DEFAULTABLE_ZERO_KEYS["mlp.gate.e_score_correction_bias"],
        }
        unresolved_missing.remove("mlp.gate.e_score_correction_bias")
    missing = sorted(unresolved_missing)
    dtype_mismatches: list[dict[str, object]] = []
    shape_mismatches: list[dict[str, object]] = []
    for key, actual in sorted(synthetic_defaults.items()):
        exp = expected.get(key)
        if not exp:
            continue
        if exp.get("dtype") is not None and actual.get("dtype") != exp.get("dtype"):
            dtype_mismatches.append({"key": key, "source_name": None, "expected": exp.get("dtype"), "actual": actual.get("dtype")})
        if exp.get("shape") is not None and actual.get("shape") != exp.get("shape"):
            shape_mismatches.append({"key": key, "source_name": None, "expected": exp.get("shape"), "actual": actual.get("shape")})
    for key, actual in sorted(matched.items()):
        exp = expected.get(key)
        if not exp:
            continue
        expected_dtype = exp.get("dtype")
        expected_shape = exp.get("shape")
        if expected_dtype is not None and actual.get("dtype") != expected_dtype:
            dtype_mismatches.append({
                "key": key,
                "source_name": actual.get("source_name"),
                "expected": expected_dtype,
                "actual": actual.get("dtype"),
            })
        if expected_shape is not None and actual.get("shape") != expected_shape:
            shape_mismatches.append({
                "key": key,
                "source_name": actual.get("source_name"),
                "expected": list(expected_shape) if isinstance(expected_shape, tuple) else expected_shape,
                "actual": actual.get("shape"),
            })
    blockers = [
        "header-only validation does not read tensor payloads or construct MLX weights",
        "dense FP8 weight/scale dequantization is not integrated into real Model.load_weights",
        "full checkpoint has multi-layer/hc_mult/compressor/indexer/expert families beyond the bounded real subset",
        "generation smoke and DS4 inspect remain required before training/conversion is unblocked",
    ]
    if missing:
        blockers.append("required bounded-real tensor families are missing from the checkpoint index")
    if header_missing:
        blockers.append("index/header mismatch: one or more indexed tensors are absent from shard headers")
    if dtype_mismatches:
        blockers.append("required bounded-real tensor families have dtype mismatches")
    if shape_mismatches:
        blockers.append("required bounded-real tensor families have shape mismatches")

    return {
        "schema": 1,
        "kind": "deepseek-v4-checkpoint-header-report",
        "index_path": str(index),
        "index_sha256": index_hash,
        "weight_map_count": len(weight_map),
        "layer": layer,
        "expert_dtype": expert_dtype,
        "n_routed_experts": n_routed_experts,
        "required_keys": sorted(required),
        "coverage_ok": not missing and not header_missing and not dtype_mismatches and not shape_mismatches,
        "matched": matched,
        "synthetic_defaults": synthetic_defaults,
        "expected_metadata_checked": sorted(expected),
        "dtype_mismatches": dtype_mismatches,
        "shape_mismatches": shape_mismatches,
        "missing": missing,
        "header_missing": sorted(header_missing),
        "unexpected_families": sorted(unexpected_counter),
        "unexpected_family_counts": dict(sorted(unexpected_counter.items())),
        "ignored_family_counts": dict(sorted(ignored_counter.items())),
        "headers_read": sorted(headers),
        "blockers": blockers,
    }


_DTYPE_NBYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
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


class CheckpointTensorLoadError(ValueError):
    """Fail-closed selected tensor load-plan/execute error."""


class StackedShapeCompatError(CheckpointTensorLoadError):
    """Fail-closed stacked checkpoint shape-compatibility error."""


def _shape_numel(shape: object) -> int:
    if not isinstance(shape, list) or not shape:
        raise CheckpointTensorLoadError(f"invalid tensor shape metadata: {shape!r}")
    n = 1
    for dim in shape:
        if type(dim) is not int or dim < 0:
            raise CheckpointTensorLoadError(f"invalid tensor shape dimension: {shape!r}")
        n *= dim
    return n


def _tensor_nbytes(meta: Mapping[str, object]) -> int:
    shape_numel = _shape_numel(meta.get("shape"))
    offsets = meta.get("data_offsets")
    if offsets is not None:
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets)):
            raise CheckpointTensorLoadError(f"invalid safetensors data_offsets: {offsets!r}")
        start = int(offsets[0])
        end = int(offsets[1])
        if start < 0 or end < start:
            raise CheckpointTensorLoadError(f"invalid safetensors data_offsets: {offsets!r}")
        return end - start
    dtype = str(meta.get("dtype"))
    if dtype not in _DTYPE_NBYTES:
        raise CheckpointTensorLoadError(f"unsupported safetensors dtype for byte estimate: {dtype!r}")
    return shape_numel * _DTYPE_NBYTES[dtype]


def _decode_selected_tensor_payload(raw: bytes, *, dtype: str, shape: list[int]) -> Any:
    try:
        import mlx.core as mx
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional MLX/numpy env
        raise CheckpointTensorLoadError(f"mlx and numpy are required for execute=True selected tensor loading: {exc}") from exc

    numel = 1
    for dim in shape:
        if type(dim) is not int or dim < 0:
            raise CheckpointTensorLoadError(f"invalid tensor shape for selected payload decode: {shape!r}")
        numel *= dim
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4")
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2")
    elif dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (words << 16).view("<f4")
    elif dtype == "I8":
        arr = np.frombuffer(raw, dtype=np.int8)
    else:
        raise CheckpointTensorLoadError(f"unsupported selected tensor dtype for payload decode: {dtype!r}")
    if int(arr.size) != int(numel):
        raise CheckpointTensorLoadError(f"selected tensor payload size mismatch for dtype={dtype} shape={shape}: got {arr.size}, expected {numel}")
    return mx.array(arr.reshape(shape))


def _load_selected_tensors_as_mlx(base_dir: Path, by_file: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    import struct

    for shard, entries in by_file.items():
        shard_path = base_dir / shard
        with shard_path.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                raise CheckpointTensorLoadError(f"{shard_path}: file too short for safetensors header")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            data_start = 8 + header_len
            for entry in entries:
                offsets = entry.get("data_offsets")
                if not (isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets)):
                    raise CheckpointTensorLoadError(f"cannot load selected tensor without valid data_offsets: {entry['source_name']}")
                start, end = int(offsets[0]), int(offsets[1])
                if start < 0 or end < start:
                    raise CheckpointTensorLoadError(f"invalid safetensors data_offsets: {offsets!r}")
                f.seek(data_start + start)
                raw = f.read(end - start)
                if len(raw) != end - start:
                    raise CheckpointTensorLoadError(f"{shard_path}: truncated selected tensor payload for {entry['source_name']}")
                arrays[entry["canonical_key"]] = _decode_selected_tensor_payload(raw, dtype=str(entry["dtype"]), shape=list(entry["shape"]))
    return arrays


def _default_bounded_model_args(*, expert_dtype: str = "i8", n_routed_experts: int = 2, num_experts_per_tok: int = 1) -> ModelArgs:
    """Return the smallest config accepted by bounded real-mode shape checks."""

    return ModelArgs(
        vocab_size=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        o_lora_rank=16,
        qk_rope_head_dim=8,
        index_head_dim=1,
        index_n_heads=1,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_shared_experts=1,
        moe_intermediate_size=16,
        expert_dtype=expert_dtype,
        hc_mult=1,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        o_groups=1,
        compression_ratio=0,
    )


def _coerce_bounded_model_args(
    model_config: ModelArgs | Mapping[str, Any] | None,
    *,
    expert_dtype: str,
    n_routed_experts: int,
) -> ModelArgs:
    if model_config is None:
        return _default_bounded_model_args(expert_dtype=expert_dtype, n_routed_experts=n_routed_experts)
    if isinstance(model_config, ModelArgs):
        return model_config
    return ModelArgs.from_dict(dict(model_config))


def _coerce_shape_model_args(
    model_config: ModelArgs | Mapping[str, Any] | None,
    model: Any | None,
    *,
    expert_dtype: str,
    n_routed_experts: int,
) -> ModelArgs:
    if model_config is not None:
        return _coerce_bounded_model_args(model_config, expert_dtype=expert_dtype, n_routed_experts=n_routed_experts)
    model_args = getattr(model, "args", None)
    if isinstance(model_args, ModelArgs):
        return model_args
    if isinstance(model_args, Mapping):
        return ModelArgs.from_dict(dict(model_args))
    return _default_bounded_model_args(expert_dtype=expert_dtype, n_routed_experts=n_routed_experts)


def bounded_model_expected_shapes(args: ModelArgs) -> dict[str, list[int]]:
    """Expected canonical tensor shapes for the proven bounded real-mode subset."""

    if args.num_hidden_layers not in (1, 2, 3):
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility supports only num_hidden_layers in {1,2,3}")
    if args.compression_ratio != 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility does not support compressors")
    if args.layer_types is None or len(args.layer_types) != args.num_hidden_layers or any(t != "sliding_attention" for t in args.layer_types):
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires layer_types='sliding_attention' with length num_hidden_layers")
    if args.mlp_layer_types is None or len(args.mlp_layer_types) != args.num_hidden_layers or any(t != "moe" for t in args.mlp_layer_types):
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires mlp_layer_types='moe' with length num_hidden_layers")
    if args.num_key_value_heads != 1:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility currently supports num_key_value_heads=1")
    if args.num_attention_heads <= 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires positive num_attention_heads")
    if args.o_groups <= 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires positive o_groups")
    if args.num_attention_heads % args.o_groups != 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires num_attention_heads divisible by o_groups")
    if args.hc_mult <= 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires positive hc_mult")
    if args.num_hidden_layers > 1 and int(args.hc_mult) != 1:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility supports multi-layer only for hc_mult=1; hc_mult>1 multi-layer parity is not proven")
    if args.n_routed_experts <= 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires positive n_routed_experts")
    if args.n_routed_experts > 4:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility currently supports only synthetic n_routed_experts<=4")
    if args.num_experts_per_tok <= 0:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires positive num_experts_per_tok")
    if args.num_experts_per_tok > args.n_routed_experts:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility requires num_experts_per_tok<=n_routed_experts")
    if args.n_shared_experts != 1:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility currently supports n_shared_experts=1")
    if args.scoring_func != "sqrtsoftplus":
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility currently supports only scoring_func='sqrtsoftplus'")
    expert_dtype = str(args.expert_dtype).lower()
    if expert_dtype not in {"fp4", "i8"}:
        raise CheckpointTensorLoadError("bounded ModelArgs shape compatibility currently supports synthetic expert_dtype='fp4' or expert_dtype='i8'")
    if expert_dtype == "i8" and (args.hidden_size % 16 != 0 or args.moe_intermediate_size % 16 != 0):
        raise CheckpointTensorLoadError("I8 bounded ModelArgs require hidden_size and moe_intermediate_size divisible by 16")

    hidden = int(args.hidden_size)
    vocab = int(args.vocab_size)
    head_dim = int(args.head_dim)
    num_heads = int(args.num_attention_heads)
    heads_per_group = num_heads // int(args.o_groups)
    q_rank = int(args.q_lora_rank)
    o_rank = int(args.o_lora_rank)
    experts = int(args.n_routed_experts)
    intermediate = int(args.moe_intermediate_size)
    hc_rows = (2 + int(args.hc_mult)) * int(args.hc_mult)
    hc_cols = int(args.hc_mult) * hidden

    shapes: dict[str, list[int]] = {
        "embed.weight": [vocab, hidden],
    }
    multi = args.num_hidden_layers > 1
    per_layer_keys = {
        "input_layernorm.weight": [hidden],
        "post_attention_layernorm.weight": [hidden],
        "q_a_proj.weight": [q_rank, hidden],
        "q_norm.weight": [q_rank],
        "q_b_proj.weight": [num_heads * head_dim, q_rank],
        "kv_proj.weight": [head_dim, hidden],
        "kv_norm.weight": [head_dim],
        "o_a_proj.weight": [int(args.o_groups) * o_rank, heads_per_group * head_dim],
        "o_b_proj.weight": [hidden, int(args.o_groups) * o_rank],
        "sinks": [num_heads],
        "attn_hc.fn": [hc_rows, hc_cols],
        "attn_hc.base": [hc_rows],
        "attn_hc.scale": [3],
        "ffn_hc.fn": [hc_rows, hc_cols],
        "ffn_hc.base": [hc_rows],
        "ffn_hc.scale": [3],
        "mlp.gate.weight": [experts, hidden],
        "mlp.gate.e_score_correction_bias": [experts],
        "mlp.shared_experts.w1.weight": [intermediate, hidden],
        "mlp.shared_experts.w2.weight": [hidden, intermediate],
        "mlp.shared_experts.w3.weight": [intermediate, hidden],
    }
    for i in range(args.num_hidden_layers):
        prefix = f"layers.{i}." if multi else ""
        for key, shape in per_layer_keys.items():
            shapes[f"{prefix}{key}"] = list(shape)
        for eid in range(experts):
            shapes[f"{prefix}mlp.experts.{eid}.w1.weight"] = [intermediate, hidden]
            shapes[f"{prefix}mlp.experts.{eid}.w2.weight"] = [hidden, intermediate]
            shapes[f"{prefix}mlp.experts.{eid}.w3.weight"] = [intermediate, hidden]
            if expert_dtype == "i8":
                shapes[f"{prefix}mlp.experts.{eid}.w1.scale"] = [intermediate, hidden // 16]
                shapes[f"{prefix}mlp.experts.{eid}.w2.scale"] = [hidden, intermediate // 16]
                shapes[f"{prefix}mlp.experts.{eid}.w3.scale"] = [intermediate, hidden // 16]
    shapes["norm.weight"] = [hidden]
    shapes["lm_head.weight"] = [vocab, hidden]
    shapes["hc_head.fn"] = [int(args.hc_mult), int(args.hc_mult) * hidden]
    shapes["hc_head.base"] = [int(args.hc_mult)]
    shapes["hc_head.scale"] = [1]
    return shapes


def validate_selected_plan_model_shapes(plan: Mapping[str, Any], args: ModelArgs) -> dict[str, Any]:
    """Compare a selected load plan's canonical tensor shapes with ModelArgs."""

    expected_shapes = bounded_model_expected_shapes(args)
    actual_shapes: dict[str, object] = {}
    for entry in plan.get("tensors", []):
        if isinstance(entry, Mapping):
            actual_shapes[str(entry.get("canonical_key"))] = entry.get("shape")
    synthetic = plan.get("synthetic_defaults", {})
    if isinstance(synthetic, Mapping):
        for key, entry in synthetic.items():
            if isinstance(entry, Mapping):
                actual_shapes[str(key)] = entry.get("shape")

    selected = [str(k) for k in plan.get("selected_keys", [])]
    missing = sorted(k for k in selected if k not in actual_shapes)
    unexpected = sorted(k for k in actual_shapes if k not in expected_shapes)
    mismatches = []
    for key in sorted(k for k in selected if k in actual_shapes and k in expected_shapes):
        expected = expected_shapes[key]
        actual = actual_shapes[key]
        if actual != expected:
            mismatches.append({"key": key, "expected": expected, "actual": actual})
    return {
        "model_shape_compatible": not missing and not unexpected and not mismatches,
        "model_shape_mismatches": mismatches,
        "model_shape_missing": missing,
        "model_shape_unexpected": unexpected,
        "model_shape_expected_keys": sorted(expected_shapes),
    }


def _attach_model_shape_compatibility(plan: dict[str, Any], args: ModelArgs) -> dict[str, Any]:
    plan.update(validate_selected_plan_model_shapes(plan, args))
    return plan


def plan_selected_tensor_load(
    index_path: str | Path = DEFAULT_SHIMMED_INDEX,
    canonical_keys: Sequence[str] | set[str] | tuple[str, ...] = (),
    *,
    checkpoint_dir: str | Path | None = None,
    layer: int = 0,
    expected_metadata: Mapping[str, Mapping[str, object]] | None = None,
    max_bytes: int | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Dry-run or execute a strict selected-tensor safetensors load plan.

    Dry-run reads only the safetensors index and JSON headers.  Execute mode
    requires an explicit byte budget and loads only the selected source tensor
    byte ranges through a bounded safetensors ``data_offsets`` reader.
    """

    selected = tuple(sorted(set(canonical_keys)))
    if not selected:
        raise CheckpointTensorLoadError("canonical_keys must select at least one tensor")
    if execute and max_bytes is None:
        raise CheckpointTensorLoadError("execute=True requires an explicit max_bytes budget")
    if max_bytes is not None and max_bytes < 0:
        raise CheckpointTensorLoadError("max_bytes must be non-negative")

    index = Path(index_path)
    base_dir = Path(checkpoint_dir) if checkpoint_dir is not None else index.parent
    index_hash, data = _index_hash_and_data(index)
    weight_map = {str(k): str(v) for k, v in data["weight_map"].items()}
    expected = dict(expected_metadata) if expected_metadata is not None else bounded_real_expected_metadata()

    canonical_sources: dict[str, str] = {}
    for source_name in weight_map:
        canonical = canonicalize_checkpoint_key(source_name, layer=layer)
        if canonical in selected and canonical not in canonical_sources:
            canonical_sources[canonical] = source_name

    headers: dict[str, dict[str, dict[str, object]]] = {}
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    dtype_mismatches: list[dict[str, object]] = []
    shape_mismatches: list[dict[str, object]] = []
    synthetic_defaults: dict[str, dict[str, object]] = {}

    for key in selected:
        source_name = canonical_sources.get(key)
        if source_name is None:
            if key == "mlp.gate.e_score_correction_bias":
                gate_source = canonical_sources.get("mlp.gate.weight")
                if gate_source is None:
                    for candidate_source in weight_map:
                        if canonicalize_checkpoint_key(candidate_source, layer=layer) == "mlp.gate.weight":
                            gate_source = candidate_source
                            break
                if gate_source is not None:
                    shard = weight_map[gate_source]
                    header = _header_for_file(headers, base_dir / shard)
                    gate_meta = header.get(gate_source)
                    gate_shape = gate_meta.get("shape") if isinstance(gate_meta, dict) else None
                    if not (isinstance(gate_shape, list) and gate_shape and type(gate_shape[0]) is int and gate_shape[0] >= 0):
                        raise CheckpointTensorLoadError(f"cannot synthesize {key}: invalid gate shape {gate_shape!r}")
                    default_shape = [gate_shape[0]]
                    synthetic_defaults[key] = {
                        "dtype": "F32",
                        "shape": default_shape,
                        "value": "zeros",
                        "nbytes": default_shape[0] * 4,
                        "proof": _DEFAULTABLE_ZERO_KEYS[key],
                    }
                    continue
            missing.append(key)
            continue

        shard = weight_map[source_name]
        header = _header_for_file(headers, base_dir / shard)
        meta = header.get(source_name)
        if not isinstance(meta, dict):
            missing.append(key)
            continue
        dtype = meta.get("dtype")
        shape = list(meta.get("shape", [])) if isinstance(meta.get("shape"), list) else meta.get("shape")
        exp = expected.get(key)
        if exp is not None:
            if exp.get("dtype") is not None and dtype != exp.get("dtype"):
                dtype_mismatches.append({"key": key, "source_name": source_name, "expected": exp.get("dtype"), "actual": dtype})
            if exp.get("shape") is not None and shape != exp.get("shape"):
                shape_mismatches.append({"key": key, "source_name": source_name, "expected": exp.get("shape"), "actual": shape})
        entries.append({
            "canonical_key": key,
            "source_name": source_name,
            "file": shard,
            "dtype": dtype,
            "shape": shape,
            "nbytes": _tensor_nbytes(meta),
            "data_offsets": list(meta["data_offsets"]) if isinstance(meta.get("data_offsets"), list) else None,
        })

    for key, actual in sorted(synthetic_defaults.items()):
        exp = expected.get(key)
        if not exp:
            continue
        if exp.get("dtype") is not None and actual.get("dtype") != exp.get("dtype"):
            dtype_mismatches.append({"key": key, "source_name": None, "expected": exp.get("dtype"), "actual": actual.get("dtype")})
        if exp.get("shape") is not None and actual.get("shape") != exp.get("shape"):
            shape_mismatches.append({"key": key, "source_name": None, "expected": exp.get("shape"), "actual": actual.get("shape")})

    synthetic_nbytes = sum(int(v.get("nbytes", 0)) for v in synthetic_defaults.values())
    total_selected_bytes = sum(int(e["nbytes"]) for e in entries) + synthetic_nbytes
    budget_ok = max_bytes is None or total_selected_bytes <= max_bytes
    plan = {
        "schema": 1,
        "kind": "deepseek-v4-selected-tensor-load-plan",
        "index_path": str(index),
        "index_sha256": index_hash,
        "layer": layer,
        "execute": execute,
        "selected_keys": list(selected),
        "tensor_count": len(entries),
        "synthetic_defaults": synthetic_defaults,
        "total_selected_bytes": total_selected_bytes,
        "max_bytes": max_bytes,
        "budget_ok": budget_ok,
        "missing": sorted(missing),
        "dtype_mismatches": dtype_mismatches,
        "shape_mismatches": shape_mismatches,
        "tensors": entries,
        "headers_read": sorted(headers),
        "payload_reader": "bounded direct safetensors selected data_offsets reader" if execute else None,
    }
    if missing or dtype_mismatches or shape_mismatches:
        raise CheckpointTensorLoadError(json.dumps({"error": "selected tensor load plan validation failed", "plan": plan}, sort_keys=True))
    if not budget_ok:
        raise CheckpointTensorLoadError(json.dumps({"error": "selected tensor byte budget exceeded", "plan": plan}, sort_keys=True))
    if not execute:
        return {"plan": plan, "arrays": {}}

    by_file: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_file.setdefault(entry["file"], []).append(entry)
    arrays = _load_selected_tensors_as_mlx(base_dir, by_file)
    if synthetic_defaults:
        try:
            import mlx.core as mx
        except Exception as exc:  # pragma: no cover - depends on optional MLX env
            raise CheckpointTensorLoadError(f"mlx is required for synthetic defaults: {exc}") from exc
        for key, default in synthetic_defaults.items():
            shape = default.get("shape")
            if not isinstance(shape, list):
                raise CheckpointTensorLoadError(f"cannot synthesize {key}: missing default shape")
            arrays[key] = mx.zeros(tuple(shape), dtype=mx.float32)
    return {"plan": plan, "arrays": arrays}


def load_bounded_real_model_from_checkpoint(
    index_path: str | Path = DEFAULT_SHIMMED_INDEX,
    *,
    checkpoint_dir: str | Path | None = None,
    layer: int = 0,
    expert_dtype: str = "i8",
    n_routed_experts: int = 2,
    expected_metadata: Mapping[str, Mapping[str, object]] | None = None,
    max_bytes: int | None = None,
    execute: bool = False,
    model_config: ModelArgs | Mapping[str, Any] | None = None,
    model: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Dry-run or execute-load the complete bounded real-mode key set.

    This is the checkpoint-to-bounded-``Model`` integration helper: it always
    selects ``bounded_real_required_keys()`` and delegates validation, byte
    accounting, synthetic zero-bias handling, and selected payload loading to
    ``plan_selected_tensor_load()``.  In dry-run mode no payload bytes are read
    and no model is constructed.  In execute mode the caller must provide an
    explicit byte budget; the selected canonical arrays are passed directly to
    ``Model.load_weights()`` on either the supplied model or a freshly
    constructed bounded tiny real-mode ``Model``.
    """

    if execute and max_bytes is None:
        raise CheckpointTensorLoadError("execute=True requires an explicit max_bytes budget")

    required_keys = bounded_real_required_keys(expert_dtype=expert_dtype, n_routed_experts=n_routed_experts)
    args = _coerce_shape_model_args(model_config, model, expert_dtype=expert_dtype, n_routed_experts=n_routed_experts)
    dry_selected = plan_selected_tensor_load(
        index_path,
        required_keys,
        checkpoint_dir=checkpoint_dir,
        layer=layer,
        expected_metadata=expected_metadata,
        max_bytes=max_bytes,
        execute=False,
    )
    plan = dict(dry_selected["plan"])
    plan["bounded_model_required_keys"] = list(required_keys)
    plan["bounded_model_required_key_count"] = len(required_keys)
    _attach_model_shape_compatibility(plan, args)
    if not execute:
        return {"plan": plan, "arrays": {}, "model": None, "model_loaded": False}
    if not plan["model_shape_compatible"]:
        raise CheckpointTensorLoadError(json.dumps({"error": "selected tensor load plan is not compatible with ModelArgs", "plan": plan}, sort_keys=True))

    selected = plan_selected_tensor_load(
        index_path,
        required_keys,
        checkpoint_dir=checkpoint_dir,
        layer=layer,
        expected_metadata=expected_metadata,
        max_bytes=max_bytes,
        execute=True,
    )
    execute_plan = dict(selected["plan"])
    execute_plan["bounded_model_required_keys"] = list(required_keys)
    execute_plan["bounded_model_required_key_count"] = len(required_keys)
    _attach_model_shape_compatibility(execute_plan, args)
    if not execute_plan["model_shape_compatible"]:
        raise CheckpointTensorLoadError(json.dumps({"error": "selected tensor load plan is not compatible with ModelArgs", "plan": execute_plan}, sort_keys=True))

    arrays = selected["arrays"]
    target_model = model
    if target_model is None:
        target_model = Model(args)
    target_model.load_weights(arrays, strict=strict)
    return {"plan": execute_plan, "arrays": arrays, "model": target_model, "model_loaded": True}


__all__ = [
    "CheckpointTensorLoadError",
    "StackedShapeCompatError",
    "DEFAULT_SHIMMED_INDEX",
    "bounded_model_expected_shapes",
    "bounded_real_expected_metadata",
    "bounded_real_required_keys",
    "canonicalize_checkpoint_key",
    "load_bounded_real_model_from_checkpoint",
    "plan_selected_tensor_load",
    "stacked_shape_compatibility_from_index",
    "stacked_shape_compatibility_report",
    "assert_stacked_shape_compatible",
    "validate_bounded_real_checkpoint_headers",
    "validate_selected_plan_model_shapes",
]
