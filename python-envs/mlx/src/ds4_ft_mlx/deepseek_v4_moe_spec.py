"""Tiny fail-closed MoE semantics helpers for DeepSeek V4 Flash.

This module is intentionally metadata-only: it does not import MLX, open
safetensors shards, or perform dequantization.  It captures the tensor-family
and shape expectations needed before a future DeepSeek V4 MLX implementation can
safely route/expert-load MoE weights.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

_ROUTED = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<projection>w[123])\.(?P<kind>weight|scale)$")
_SHARED = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.shared_experts\.(?P<projection>w[123])\.(?P<kind>weight|scale)$")
_ROUTER = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.(?:gate|router)\.(?P<kind>weight|bias|scale|tid2eid)$")

_ALLOWED_FLOAT_DTYPES = {"F16", "FLOAT16", "BF16", "BFLOAT16", "F32", "FLOAT32"}
_ALLOWED_INTEGER_METADATA_DTYPES = {"I32", "INT32", "I64", "INT64", "U32", "UINT32", "U64", "UINT64"}
_BLOCKED_EXPERT_DTYPES = {"FP4", "I4", "INT4", "I8", "INT8"}


class MoESpecError(ValueError):
    """Raised when deterministic MoE metadata does not match the spec."""


class MoEQuantizationBlocked(MoESpecError):
    """Raised for packed expert formats without trusted dequant parity."""


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int = 1

    def __post_init__(self) -> None:
        for name in ("hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "n_shared_experts"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise MoESpecError(f"{name} must be a positive integer, got {value!r}")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise MoESpecError("num_experts_per_tok cannot exceed n_routed_experts")


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: str

    def normalized_dtype(self) -> str:
        return self.dtype.upper().replace(" ", "")


@dataclass(frozen=True)
class MoETensorName:
    name: str
    family: str
    layer: int
    kind: str
    projection: str | None = None
    expert_id: int | None = None


@dataclass(frozen=True)
class RouterLayerSpec:
    layer: int
    weight_shape: tuple[int, ...] | None = None
    bias_shape: tuple[int, ...] | None = None
    tid2eid_shape: tuple[int, ...] | None = None

    @property
    def has_bias(self) -> bool:
        return self.bias_shape is not None

    @property
    def has_tid2eid(self) -> bool:
        return self.tid2eid_shape is not None


@dataclass(frozen=True)
class MoEManifestReport:
    router_layers: dict[int, RouterLayerSpec] = field(default_factory=dict)
    routed_experts_by_layer: dict[int, list[int]] = field(default_factory=dict)
    shared_layers: list[int] = field(default_factory=list)
    quantized_expert_tensors: list[str] = field(default_factory=list)
    missing_required_tensors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required_tensors


def classify_moe_tensor_name(name: str) -> MoETensorName | None:
    match = _ROUTED.match(name)
    if match:
        return MoETensorName(
            name=name,
            family="routed_expert",
            layer=int(match.group("layer")),
            expert_id=int(match.group("expert")),
            projection=match.group("projection"),
            kind=match.group("kind"),
        )
    match = _SHARED.match(name)
    if match:
        return MoETensorName(
            name=name,
            family="shared_expert",
            layer=int(match.group("layer")),
            projection=match.group("projection"),
            kind=match.group("kind"),
        )
    match = _ROUTER.match(name)
    if match:
        return MoETensorName(
            name=name,
            family="router",
            layer=int(match.group("layer")),
            kind=match.group("kind"),
        )
    return None


def expected_expert_shape(config: MoEConfig, family: str, projection: str) -> tuple[int, int]:
    if projection not in {"w1", "w2", "w3"}:
        raise MoESpecError(f"unsupported expert projection {projection!r}; expected w1/w2/w3")
    if family == "routed_expert":
        intermediate = config.moe_intermediate_size
    elif family == "shared_expert":
        intermediate = config.moe_intermediate_size * config.n_shared_experts
    else:
        raise MoESpecError(f"unsupported expert family {family!r}; expected routed_expert/shared_expert")
    if projection in {"w1", "w3"}:
        return (intermediate, config.hidden_size)
    return (config.hidden_size, intermediate)


def projection_semantics(projection: str) -> str:
    if projection in {"w1", "w3"}:
        return "gate/up projection: expected <intermediate, hidden>"
    if projection == "w2":
        return "down projection: expected <hidden, intermediate>"
    return "unknown projection"


def expected_router_shape(config: MoEConfig, kind: str) -> tuple[int, ...]:
    if kind in {"weight", "scale"}:
        return (config.n_routed_experts, config.hidden_size)
    if kind in {"bias", "tid2eid"}:
        return (config.n_routed_experts,)
    raise MoESpecError(f"unsupported router tensor kind {kind!r}")


def _validate_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...], detail: str) -> None:
    if actual != expected:
        raise MoESpecError(f"{name}: shape {actual} does not match expected {expected}; {detail}")


def _validate_dtype(name: str, parsed: MoETensorName, meta: TensorMeta, trusted_dequant: bool, quantized: list[str]) -> None:
    dtype = meta.normalized_dtype()
    if parsed.family in {"routed_expert", "shared_expert"} and parsed.kind == "weight":
        if dtype in _BLOCKED_EXPERT_DTYPES and not trusted_dequant:
            raise MoEQuantizationBlocked(
                f"{meta.dtype} expert tensor is fail-closed until trusted dequant parity exists: {name}; "
                "do not route packed FP4/I8 experts into MLX without a parity fixture"
            )
        if dtype in _BLOCKED_EXPERT_DTYPES:
            quantized.append(name)
            return
    if dtype in _ALLOWED_FLOAT_DTYPES:
        return
    if parsed.family == "router" and parsed.kind == "tid2eid" and dtype in _ALLOWED_INTEGER_METADATA_DTYPES:
        return
    if parsed.kind == "scale" and dtype in _ALLOWED_FLOAT_DTYPES:
        return
    raise MoESpecError(f"{name}: unsupported dtype {meta.dtype!r} for {parsed.family}.{parsed.kind}")


def expected_moe_weight_names_for_layer(layer: int, config: MoEConfig) -> list[str]:
    names = [f"layers.{layer}.ffn.gate.weight"]
    for expert_id in range(config.n_routed_experts):
        for projection in ("w1", "w2", "w3"):
            names.append(f"layers.{layer}.ffn.experts.{expert_id}.{projection}.weight")
    for projection in ("w1", "w2", "w3"):
        names.append(f"layers.{layer}.ffn.shared_experts.{projection}.weight")
    return names


def validate_moe_name_completeness(names: Iterable[str], config: MoEConfig) -> list[str]:
    present = set(names)
    moe_layers: set[int] = set()
    for name in present:
        parsed = classify_moe_tensor_name(name)
        if parsed is not None:
            moe_layers.add(parsed.layer)
    missing: list[str] = []
    for layer in sorted(moe_layers):
        for required in expected_moe_weight_names_for_layer(layer, config):
            if required not in present:
                missing.append(required)
    return missing


def _linear(vector: list[float], weight: list[list[float]]) -> list[float]:
    if len(weight) == 0 or len(vector) != len(weight[0]):
        raise MoESpecError("MoE linear weight shape mismatch")
    return [sum(float(x) * float(w) for x, w in zip(vector, row)) for row in weight]


def _silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def _softplus(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _router_score(x: float, scoring_func: str) -> float:
    if scoring_func == "sqrtsoftplus":
        return math.sqrt(_softplus(x))
    if scoring_func in {"silu", "swiglu"}:
        return _silu(x)
    raise MoESpecError(f"unsupported tiny MoE scoring_func {scoring_func!r}")


def _expert_forward(
    vector: list[float],
    w1: list[list[float]],
    w2: list[list[float]],
    w3: list[list[float]],
    *,
    swiglu_limit: float,
) -> list[float]:
    gate = [min(value, swiglu_limit) for value in _linear(vector, w1)]
    up = [min(max(value, -swiglu_limit), swiglu_limit) for value in _linear(vector, w3)]
    hidden = [_silu(g) * u for g, u in zip(gate, up)]
    return _linear(hidden, w2)


def _reshape_flat(flat: Sequence[float], shape: tuple[int, int]) -> list[list[float]]:
    rows, cols = shape
    if len(flat) != rows * cols:
        raise MoESpecError(f"cannot reshape flat list of length {len(flat)} to {shape}")
    return [list(flat[i * cols : (i + 1) * cols]) for i in range(rows)]


def _expert_forward_i8(
    vector: list[float],
    weights: Mapping[str, object],
    expert_id: int,
    shape_w1: tuple[int, int],
    shape_w2: tuple[int, int],
    shape_w3: tuple[int, int],
    *,
    swiglu_limit: float,
    block_size: int,
    scale_axis: int,
) -> list[float]:
    from ds4_ft_mlx.deepseek_v4_dequant import dequantize_i8_block_scale

    prefix = f"experts.{expert_id}."
    w1 = _reshape_flat(
        dequantize_i8_block_scale(
            weights[f"{prefix}w1.weight"],  # type: ignore[arg-type]
            weights[f"{prefix}w1.scale"],  # type: ignore[arg-type]
            shape=shape_w1,
            block_size=block_size,
            scale_axis=scale_axis,
        ),
        shape_w1,
    )
    w2 = _reshape_flat(
        dequantize_i8_block_scale(
            weights[f"{prefix}w2.weight"],  # type: ignore[arg-type]
            weights[f"{prefix}w2.scale"],  # type: ignore[arg-type]
            shape=shape_w2,
            block_size=block_size,
            scale_axis=scale_axis,
        ),
        shape_w2,
    )
    w3 = _reshape_flat(
        dequantize_i8_block_scale(
            weights[f"{prefix}w3.weight"],  # type: ignore[arg-type]
            weights[f"{prefix}w3.scale"],  # type: ignore[arg-type]
            shape=shape_w3,
            block_size=block_size,
            scale_axis=scale_axis,
        ),
        shape_w3,
    )
    return _expert_forward(vector, w1, w2, w3, swiglu_limit=swiglu_limit)


def tiny_topk_moe_routing(
    config: MoEConfig,
    hidden_states: list[list[float]],
    weights: Mapping[str, list[list[float]] | list[float]],
    *,
    scoring_func: str = "sqrtsoftplus",
    routed_scaling_factor: float = 1.0,
) -> list[dict[str, object]]:
    """Pure, decode-free DeepSeek V4 top-k routing decisions per token.

    Selection uses biased scores (`score + e_score_correction_bias`) only for
    ranking.  Contribution weights are normalized from the un-biased selected
    scores and multiplied by `routed_scaling_factor`.  Ties are deterministic:
    lower expert index wins for equal biased scores.
    """

    router = weights.get("router.weight")
    if router is None or not isinstance(router, list):
        raise MoESpecError("missing router.weight")
    if len(router) != config.n_routed_experts:
        raise MoESpecError("router.weight expert count mismatch")
    bias_raw = weights.get("router.e_score_correction_bias", [0.0 for _ in range(config.n_routed_experts)])
    if not isinstance(bias_raw, list) or len(bias_raw) != config.n_routed_experts:
        raise MoESpecError("router.e_score_correction_bias shape mismatch")
    correction_bias = [float(value) for value in bias_raw]

    reports: list[dict[str, object]] = []
    for vector in hidden_states:
        logits = [sum(float(x) * float(w) for x, w in zip(vector, row)) for row in router]
        scores = [_router_score(logit, scoring_func) for logit in logits]
        # Explicit two-part key documents the stable-sort contract: descending
        # biased score, then lower index for ties.
        selected = sorted(range(len(scores)), key=lambda idx: (-(scores[idx] + correction_bias[idx]), idx))[: config.num_experts_per_tok]
        selected_scores = [scores[idx] for idx in selected]
        denom = sum(selected_scores) + 1e-20
        contribution_weights = [(score / denom) * routed_scaling_factor for score in selected_scores]
        reports.append({
            "selected": selected,
            "scores": scores,
            "selected_scores": selected_scores,
            "contribution_weights": contribution_weights,
        })
    return reports


def tiny_topk_moe_i8_forward(
    config: MoEConfig,
    hidden_states: list[list[float]],
    weights: Mapping[str, object],
    *,
    scoring_func: str = "sqrtsoftplus",
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float = 10.0,
    block_size: int = 16,
    scale_axis: int = 1,
) -> list[list[float]]:
    """Deterministic top-k MoE fixture for I8 block-scale routed experts.

    Routed expert weights are provided as signed I8 bytes with paired BF16
    block scales; shared experts remain unquantized floats.  The fixture
    dequantizes selected experts on demand and then applies the same V4 top-k
    routing, score normalisation, routed scaling factor, and SwiGLU clamp
    semantics as `tiny_topk_moe_forward`.  This is partial evidence only:
    real checkpoint payload decode, expert parallel kernels, and FP4 remain
    fail-closed.
    """

    shape_w1 = (config.moe_intermediate_size, config.hidden_size)
    shape_w2 = (config.hidden_size, config.moe_intermediate_size)
    shape_w3 = (config.moe_intermediate_size, config.hidden_size)

    router = weights.get("router.weight")
    if router is None or not isinstance(router, list):
        raise MoESpecError("missing router.weight")
    if len(router) != config.n_routed_experts:
        raise MoESpecError("router.weight expert count mismatch")
    bias_raw = weights.get("router.e_score_correction_bias", [0.0 for _ in range(config.n_routed_experts)])
    if not isinstance(bias_raw, list) or len(bias_raw) != config.n_routed_experts:
        raise MoESpecError("router.e_score_correction_bias shape mismatch")
    correction_bias = [float(value) for value in bias_raw]

    outputs: list[list[float]] = []
    for vector in hidden_states:
        logits = [sum(float(x) * float(w) for x, w in zip(vector, row)) for row in router]
        scores = [_router_score(logit, scoring_func) for logit in logits]
        top = sorted(range(len(scores)), key=lambda idx: scores[idx] + correction_bias[idx], reverse=True)[: config.num_experts_per_tok]
        denom = sum(scores[idx] for idx in top) + 1e-20
        routed = [0.0 for _ in range(config.hidden_size)]
        for idx in top:
            expert = _expert_forward_i8(
                vector,
                weights,
                idx,
                shape_w1,
                shape_w2,
                shape_w3,
                swiglu_limit=swiglu_limit,
                block_size=block_size,
                scale_axis=scale_axis,
            )
            factor = (scores[idx] / denom) * routed_scaling_factor
            routed = [value + factor * expert_value for value, expert_value in zip(routed, expert)]
        shared = _expert_forward(
            vector,
            weights["shared.w1"],  # type: ignore[arg-type]
            weights["shared.w2"],  # type: ignore[arg-type]
            weights["shared.w3"],  # type: ignore[arg-type]
            swiglu_limit=swiglu_limit,
        )
        outputs.append([value + shared_value for value, shared_value in zip(routed, shared)])
    return outputs


def tiny_topk_moe_forward(
    config: MoEConfig,
    hidden_states: list[list[float]],
    weights: Mapping[str, list[list[float]] | list[float]],
    *,
    scoring_func: str = "sqrtsoftplus",
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float = 10.0,
) -> list[list[float]]:
    """Deterministic top-k MoE fixture for unquantized DeepSeek V4 experts.

    Covers V4 TopK router dot-product scores, `sqrtsoftplus` scoring,
    `e_score_correction_bias` before top-k, selected-weight normalisation,
    routed scaling factor, expert SwiGLU clamp semantics, and shared expert
    addition.  It intentionally excludes hash routing and packed FP4/I8 expert
    dequant.
    """

    routing = tiny_topk_moe_routing(
        config,
        hidden_states,
        weights,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
    )
    outputs: list[list[float]] = []
    for vector, route in zip(hidden_states, routing, strict=True):
        routed = [0.0 for _ in range(config.hidden_size)]
        selected = route["selected"]  # type: ignore[assignment]
        contribution_weights = route["contribution_weights"]  # type: ignore[assignment]
        for idx, factor in zip(selected, contribution_weights, strict=True):
            expert = _expert_forward(
                vector,
                weights[f"experts.{idx}.w1"],  # type: ignore[arg-type]
                weights[f"experts.{idx}.w2"],  # type: ignore[arg-type]
                weights[f"experts.{idx}.w3"],  # type: ignore[arg-type]
                swiglu_limit=swiglu_limit,
            )
            routed = [value + float(factor) * expert_value for value, expert_value in zip(routed, expert)]
        shared = _expert_forward(
            vector,
            weights["shared.w1"],  # type: ignore[arg-type]
            weights["shared.w2"],  # type: ignore[arg-type]
            weights["shared.w3"],  # type: ignore[arg-type]
            swiglu_limit=swiglu_limit,
        )
        outputs.append([value + shared_value for value, shared_value in zip(routed, shared)])
    return outputs


def tiny_hash_moe_forward(
    config: MoEConfig,
    hidden_states: list[list[float]],
    *,
    input_ids: list[int],
    weights: Mapping[str, list[list[float]] | list[float]],
    scoring_func: str = "sqrtsoftplus",
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float = 10.0,
) -> list[list[float]]:
    """Deterministic DeepSeek V4 hash-router fixture using `tid2eid` selection."""

    if len(input_ids) != len(hidden_states):
        raise MoESpecError("input_ids length must match hidden_states length")
    router = weights.get("router.weight")
    tid2eid = weights.get("router.tid2eid")
    if router is None or not isinstance(router, list):
        raise MoESpecError("missing router.weight")
    if tid2eid is None or not isinstance(tid2eid, list):
        raise MoESpecError("missing router.tid2eid")
    outputs: list[list[float]] = []
    for vector, token_id in zip(hidden_states, input_ids):
        if token_id < 0 or token_id >= len(tid2eid):
            raise MoESpecError(f"token id {token_id} outside tid2eid table")
        selected = [int(idx) for idx in tid2eid[token_id]]  # type: ignore[index]
        if len(selected) != config.num_experts_per_tok:
            raise MoESpecError("tid2eid top-k width mismatch")
        logits = [sum(float(x) * float(w) for x, w in zip(vector, row)) for row in router]
        scores = [_router_score(logit, scoring_func) for logit in logits]
        denom = sum(scores[idx] for idx in selected) + 1e-20
        routed = [0.0 for _ in range(config.hidden_size)]
        for idx in selected:
            expert = _expert_forward(
                vector,
                weights[f"experts.{idx}.w1"],  # type: ignore[arg-type]
                weights[f"experts.{idx}.w2"],  # type: ignore[arg-type]
                weights[f"experts.{idx}.w3"],  # type: ignore[arg-type]
                swiglu_limit=swiglu_limit,
            )
            factor = (scores[idx] / denom) * routed_scaling_factor
            routed = [value + factor * expert_value for value, expert_value in zip(routed, expert)]
        shared = _expert_forward(
            vector,
            weights["shared.w1"],  # type: ignore[arg-type]
            weights["shared.w2"],  # type: ignore[arg-type]
            weights["shared.w3"],  # type: ignore[arg-type]
            swiglu_limit=swiglu_limit,
        )
        outputs.append([value + shared_value for value, shared_value in zip(routed, shared)])
    return outputs


def run_tiny_hash_moe_fixture() -> dict[str, object]:
    cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
    got = tiny_hash_moe_forward(cfg, [[1.0], [1.0]], input_ids=[0, 1], weights={
        "router.weight": [[1.0], [1.0]],
        "router.tid2eid": [[1], [0]],
        "experts.0.w1": [[1.0]],
        "experts.0.w2": [[1.0]],
        "experts.0.w3": [[1.0]],
        "experts.1.w1": [[2.0]],
        "experts.1.w2": [[3.0]],
        "experts.1.w3": [[4.0]],
        "shared.w1": [[0.0]],
        "shared.w2": [[0.0]],
        "shared.w3": [[0.0]],
    }, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=10.0)
    expected = [
        [1.5 * 3.0 * (2.0 / (1.0 + math.exp(-2.0))) * 4.0],
        [1.5 * (1.0 / (1.0 + math.exp(-1.0)))],
    ]
    max_abs_error = 0.0
    for got_row, expected_row in zip(got, expected):
        for got_value, expected_value in zip(got_row, expected_row):
            max_abs_error = max(max_abs_error, abs(float(got_value) - float(expected_value)))
    return {
        "fixture": "hash-moe-tid2eid-unquantized",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["router.weight", "sqrtsoftplus scoring", "tid2eid selection", "score normalisation", "routed_scaling_factor", "SwiGLU clamp", "shared expert addition"],
        "not_covered": ["packed FP4/I8 expert dequant", "expert parallel kernels"],
    }


def run_tiny_topk_moe_i8_fixture() -> dict[str, object]:
    """Deterministic I8 block-scale top-k MoE parity fixture.

    Compares `tiny_topk_moe_i8_forward` against a PyTorch reference that
    applies the same per-block BF16 scale dequantization.  This proves the
    integrated I8 expert dequant path, not full DS4 runtime parity.
    """

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "fixture": "topk-moe-i8-block-scale",
            "status": "skipped",
            "max_abs_error": None,
            "reason": f"torch reference unavailable: {exc}",
        }

    torch.manual_seed(11)
    cfg = MoEConfig(hidden_size=16, moe_intermediate_size=16, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
    block_size = 16
    scale_axis = 1

    def _make_i8_weight(shape):
        weight_f32 = torch.randn(shape, dtype=torch.float32) * 0.1
        scale = weight_f32.abs().amax(dim=scale_axis, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale) * 1e-6, scale)
        i8 = (weight_f32 / scale).round().clamp(-128, 127).to(torch.int8)
        return i8, scale.to(torch.bfloat16)

    shapes = {
        "w1": (cfg.moe_intermediate_size, cfg.hidden_size),
        "w2": (cfg.hidden_size, cfg.moe_intermediate_size),
        "w3": (cfg.moe_intermediate_size, cfg.hidden_size),
    }
    weights: dict[str, object] = {
        "router.weight": [[0.1] * cfg.hidden_size, [-0.1] * cfg.hidden_size],
        "router.e_score_correction_bias": [0.0, 0.0],
        "shared.w1": [[0.01] * cfg.hidden_size for _ in range(cfg.moe_intermediate_size)],
        "shared.w2": [[0.01] * cfg.moe_intermediate_size for _ in range(cfg.hidden_size)],
        "shared.w3": [[0.01] * cfg.hidden_size for _ in range(cfg.moe_intermediate_size)],
    }
    reference_float: dict[str, dict[str, torch.Tensor]] = {}
    for eid in range(cfg.n_routed_experts):
        reference_float[eid] = {}
        for proj, shape in shapes.items():
            i8, scale_bf16 = _make_i8_weight(shape)
            weights[f"experts.{eid}.{proj}.weight"] = bytes(i8.numpy().tobytes())
            weights[f"experts.{eid}.{proj}.scale"] = bytes(scale_bf16.contiguous().view(torch.uint8).numpy().tobytes())
            scale_repeated = scale_bf16.float().repeat_interleave(block_size, dim=scale_axis)
            reference_float[eid][proj] = i8.float() * scale_repeated

    hidden = torch.randn(2, cfg.hidden_size, dtype=torch.float32) * 0.1
    hidden_list = hidden.tolist()

    got = tiny_topk_moe_i8_forward(
        cfg,
        hidden_list,
        weights,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        block_size=block_size,
        scale_axis=scale_axis,
    )

    def _silu_ref(x):
        return x / (1.0 + torch.exp(-x))

    expected: list[list[float]] = []
    for token_idx, vector in enumerate(hidden):
        logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
        for eid in range(cfg.n_routed_experts):
            logits[eid] = sum(vector[d].item() * weights["router.weight"][eid][d] for d in range(cfg.hidden_size))
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
        top_idx = int(torch.argmax(scores).item())
        w1 = reference_float[top_idx]["w1"]
        w2 = reference_float[top_idx]["w2"]
        w3 = reference_float[top_idx]["w3"]
        gate = (vector @ w1.T).clamp(max=10.0)
        up = (vector @ w3.T).clamp(min=-10.0, max=10.0)
        expert_out = (_silu_ref(gate) * up) @ w2.T
        shared_w1 = torch.tensor(weights["shared.w1"], dtype=torch.float32)
        shared_w2 = torch.tensor(weights["shared.w2"], dtype=torch.float32)
        shared_w3 = torch.tensor(weights["shared.w3"], dtype=torch.float32)
        shared_gate = (vector @ shared_w1.T).clamp(max=10.0)
        shared_up = (vector @ shared_w3.T).clamp(min=-10.0, max=10.0)
        shared_out = (_silu_ref(shared_gate) * shared_up) @ shared_w2.T
        expected.append((expert_out + shared_out).tolist())

    max_abs_error = 0.0
    for got_row, ref_row in zip(got, expected):
        for gv, rv in zip(got_row, ref_row):
            max_abs_error = max(max_abs_error, abs(gv - rv))
    return {
        "fixture": "topk-moe-i8-block-scale",
        "status": "ok" if max_abs_error <= 1e-3 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "I8 signed integer decode",
            "BF16 block-scale decode",
            "block size 16 along axis 1",
            "top-k routed MoE routing",
            "sqrtsoftplus scoring",
            "score normalisation",
            "routed_scaling_factor",
            "SwiGLU clamp",
            "shared expert addition",
            "PyTorch reference parity",
        ],
        "not_covered": [
            "real checkpoint payload decode",
            "packed FP4 expert dequant",
            "expert parallel kernels",
            "DS4 runtime verification",
        ],
    }


def run_tiny_topk_moe_fixture() -> dict[str, object]:
    cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
    got = tiny_topk_moe_forward(cfg, [[1.0], [-1.0]], {
        "router.weight": [[1.0], [-1.0]],
        "router.e_score_correction_bias": [0.0, 3.0],
        "experts.0.w1": [[12.0]],
        "experts.0.w2": [[1.0]],
        "experts.0.w3": [[20.0]],
        "experts.1.w1": [[1.0]],
        "experts.1.w2": [[2.0]],
        "experts.1.w3": [[3.0]],
        "shared.w1": [[0.0]],
        "shared.w2": [[0.0]],
        "shared.w3": [[0.0]],
    }, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=10.0)
    expected = [
        [1.5 * 2.0 * (1.0 / (1.0 + math.exp(-1.0))) * 3.0],
        [1.5 * 2.0 * (-1.0 / (1.0 + math.exp(1.0))) * -3.0],
    ]
    max_abs_error = 0.0
    for got_row, expected_row in zip(got, expected):
        for got_value, expected_value in zip(got_row, expected_row):
            max_abs_error = max(max_abs_error, abs(float(got_value) - float(expected_value)))
    return {
        "fixture": "topk-moe-unquantized",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["router.weight", "sqrtsoftplus scoring", "e_score_correction_bias", "topk", "score normalisation", "routed_scaling_factor", "SwiGLU clamp", "shared expert addition"],
        "not_covered": ["hash router tid2eid", "packed FP4/I8 expert dequant", "expert parallel kernels"],
    }


def validate_moe_manifest(manifest: Mapping[str, TensorMeta], config: MoEConfig, *, trusted_dequant: bool = False) -> MoEManifestReport:
    router_acc: dict[int, dict[str, tuple[int, ...]]] = {}
    routed: dict[int, set[int]] = {}
    shared_layers: set[int] = set()
    present_weights: set[str] = set()
    quantized: list[str] = []

    for name in sorted(manifest):
        meta = manifest[name]
        parsed = classify_moe_tensor_name(name)
        if parsed is None:
            raise MoESpecError(f"{name}: not a recognized DeepSeek V4 MoE/router tensor name")
        if parsed.family == "router":
            expected = expected_router_shape(config, parsed.kind)
            _validate_shape(name, tuple(meta.shape), expected, "router gate metadata over routed experts")
            _validate_dtype(name, parsed, meta, trusted_dequant, quantized)
            router_acc.setdefault(parsed.layer, {})[parsed.kind] = tuple(meta.shape)
            if parsed.kind == "weight":
                present_weights.add(name)
            continue

        assert parsed.projection is not None
        if parsed.kind == "weight":
            expected = expected_expert_shape(config, parsed.family, parsed.projection)
            _validate_shape(name, tuple(meta.shape), expected, projection_semantics(parsed.projection))
            _validate_dtype(name, parsed, meta, trusted_dequant, quantized)
        elif parsed.kind == "scale":
            _validate_dtype(name, parsed, meta, trusted_dequant, quantized)
        else:
            raise MoESpecError(f"{name}: unsupported expert tensor kind {parsed.kind!r}")

        if parsed.kind == "weight":
            present_weights.add(name)
        if parsed.family == "routed_expert":
            assert parsed.expert_id is not None
            if parsed.expert_id >= config.n_routed_experts:
                raise MoESpecError(f"{name}: expert_id {parsed.expert_id} exceeds n_routed_experts={config.n_routed_experts}")
            routed.setdefault(parsed.layer, set()).add(parsed.expert_id)
        else:
            shared_layers.add(parsed.layer)

    router_layers = {
        layer: RouterLayerSpec(
            layer=layer,
            weight_shape=kinds.get("weight"),
            bias_shape=kinds.get("bias"),
            tid2eid_shape=kinds.get("tid2eid"),
        )
        for layer, kinds in sorted(router_acc.items())
    }
    routed_sorted = {layer: sorted(experts) for layer, experts in sorted(routed.items())}
    layers = sorted(set(router_acc) | set(routed) | set(shared_layers))
    missing: list[str] = []
    for layer in layers:
        for required in expected_moe_weight_names_for_layer(layer, config):
            if required not in present_weights:
                missing.append(required)
    return MoEManifestReport(
        router_layers=router_layers,
        routed_experts_by_layer=routed_sorted,
        shared_layers=sorted(shared_layers),
        quantized_expert_tensors=sorted(quantized),
        missing_required_tensors=missing,
    )
