"""Opaque packed-FP4 Metal primitive for DeepSeek V4 training routed experts.

Story 13.3b-5f / ADR 0028 owns this package-local Apple Metal boundary.  The
primitive is training-only, first-order only, and fail-closed outside the exact
MLX/Apple Metal runtime used by the acceptance tests.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import math
import platform
import sys
from functools import lru_cache
from typing import Any, Iterable, Sequence

import mlx.core as mx

KERNEL_NAMES = (
    "ds4_fp4_pair_swiglu_forward",
    "ds4_fp4_down_forward",
    "ds4_fp4_down_input_vjp",
    "ds4_fp4_pair_swiglu_vjp_terms",
    "ds4_fp4_pair_input_vjp",
    "ds4_fp4_reduce_a",
)

_SOURCE_NAME = "metal/ds4_routed_fp4_train.metal"
_COMMON_BEGIN = "// DS4_ROUTED_FP4_TRAIN_COMMON_BEGIN"
_COMMON_END = "// DS4_ROUTED_FP4_TRAIN_COMMON_END"
_KERNEL_BEGIN = "// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN "
_KERNEL_END = "// DS4_ROUTED_FP4_TRAIN_KERNEL_END "


def _runtime_error(reason: str) -> RuntimeError:
    return RuntimeError(f"Story 13.3b-5f ADR 0028 packed-FP4 Metal primitive unavailable: {reason}")


def _ensure_supported() -> None:
    if sys.platform != "darwin":
        raise _runtime_error(f"requires darwin, got {sys.platform}")
    if platform.machine() != "arm64":
        raise _runtime_error(f"requires Apple arm64, got {platform.machine()}")
    version = importlib.metadata.version("mlx")
    if version != "0.31.2":
        raise _runtime_error(f"requires mlx==0.31.2, got {version}")
    if not mx.metal.is_available():
        raise _runtime_error("mx.metal.is_available() is false")
    if "gpu" not in str(mx.default_device()).lower():
        raise _runtime_error(f"default MLX device must be GPU, got {mx.default_device()}")


def _require_unique_section(source: str, begin: str, end: str, label: str) -> str:
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError(f"expected exactly one Metal section for {label}")
    return source.split(begin, 1)[1].split(end, 1)[0]


@lru_cache(maxsize=1)
def _source_sections() -> tuple[str, dict[str, str]]:
    source_path = importlib.resources.files("ds4_ft_mlx").joinpath(_SOURCE_NAME)
    source = source_path.read_text()
    header = _require_unique_section(source, _COMMON_BEGIN, _COMMON_END, "common header")
    bodies: dict[str, str] = {}
    for name in KERNEL_NAMES:
        begin = f"{_KERNEL_BEGIN}{name}"
        end = f"{_KERNEL_END}{name}"
        bodies[name] = _require_unique_section(source, begin, end, name)
    return header, bodies


@lru_cache(maxsize=1)
def _kernels() -> dict[str, Any]:
    _ensure_supported()
    header, bodies = _source_sections()
    specs = {
        "ds4_fp4_pair_swiglu_forward": (
            ["x_flat", "rows", "w1", "s1", "w3", "s3", "limit"],
            ["hidden"],
        ),
        "ds4_fp4_down_forward": (["hidden", "w2", "s2"], ["y"]),
        "ds4_fp4_down_input_vjp": (["g_flat", "rows", "w2", "s2"], ["gu"]),
        "ds4_fp4_pair_swiglu_vjp_terms": (
            ["x_flat", "rows", "gu", "f", "w1", "s1", "w3", "s3", "limit"],
            ["dgate", "dup", "a_partial"],
        ),
        "ds4_fp4_pair_input_vjp": (["dgate", "dup", "w1", "s1", "w3", "s3"], ["dx"]),
        "ds4_fp4_reduce_a": (["a_partial"], ["a"]),
    }
    return {
        name: mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=bodies[name],
            header=header,
            ensure_row_contiguous=False,
        )
        for name, (input_names, output_names) in specs.items()
    }


def _as_limit(limit: float) -> mx.array:
    value = float(limit)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("limit must be finite and positive")
    return mx.array([value], dtype=mx.float32)


def _require_finite_positive(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def _check_uint_grid(name: str, *dims: int) -> None:
    limit = (1 << 32) - 1
    for dim in dims:
        if int(dim) < 0 or int(dim) > limit:
            raise ValueError(f"{name} grid dimension {dim} exceeds uint32")


def _check_rank_dtype(name: str, arr: mx.array, rank: int, dtype: mx.Dtype) -> None:
    if arr.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {arr.shape}")
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {arr.dtype}")


def _check_rank(name: str, arr: mx.array, rank: int) -> None:
    if arr.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {arr.shape}")


def _activation_to_float32(name: str, arr: mx.array) -> mx.array:
    _check_rank(name, arr, 2)
    if arr.dtype not in (mx.float32, mx.float16, mx.bfloat16):
        raise ValueError(f"{name} must have float32/float16/bfloat16 dtype, got {arr.dtype}")
    return arr.astype(mx.float32)


def _rows_list(rows: mx.array, T: int) -> list[int]:
    _check_rank_dtype("rows", rows, 1, mx.int32)
    values = [int(v) for v in rows.tolist()]
    if any(v < 0 or v >= T for v in values):
        raise ValueError("rows must be in [0,T)")
    if len(set(values)) != len(values):
        raise ValueError("rows must be unique within an expert")
    if values != sorted(values):
        raise ValueError("rows must preserve ascending token order")
    return values


def _validate_one(
    x_flat: mx.array,
    rows: mx.array,
    w1: mx.array,
    s1: mx.array,
    w3: mx.array,
    s3: mx.array,
    w2: mx.array,
    s2: mx.array,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[int, int, int]:
    _check_rank_dtype("x_flat", x_flat, 2, mx.float32)
    T, Hx = x_flat.shape
    H = int(hidden_size)
    I = int(intermediate_size)
    if H <= 0 or I <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive")
    if Hx != H:
        raise ValueError(f"x_flat second dim {Hx} != hidden_size {H}")
    rows_values = _rows_list(rows, int(T))
    for name, arr in (("w1", w1), ("w3", w3), ("w2", w2)):
        _check_rank_dtype(name, arr, 2, mx.uint8)
    for name, arr in (("s1", s1), ("s3", s3), ("s2", s2)):
        _check_rank_dtype(name, arr, 2, mx.bfloat16)
    hp = ((H + 31) // 32) * 32
    ip = ((I + 31) // 32) * 32
    if w1.shape != (I, hp // 2) or w3.shape != (I, hp // 2):
        raise ValueError(f"w1/w3 must have shape {(I, hp // 2)}, got {w1.shape}/{w3.shape}")
    if s1.shape != (I, hp // 32) or s3.shape != (I, hp // 32):
        raise ValueError(f"s1/s3 must have shape {(I, hp // 32)}, got {s1.shape}/{s3.shape}")
    if w2.shape != (H, ip // 2):
        raise ValueError(f"w2 must have shape {(H, ip // 2)}, got {w2.shape}")
    if s2.shape != (H, ip // 32):
        raise ValueError(f"s2 must have shape {(H, ip // 32)}, got {s2.shape}")
    return len(rows_values), H, I


def packed_fp4_forward_one(
    x_flat: mx.array,
    rows: mx.array,
    w1: mx.array,
    s1: mx.array,
    w3: mx.array,
    s3: mx.array,
    w2: mx.array,
    s2: mx.array,
    *,
    hidden_size: int,
    intermediate_size: int,
    limit: float,
) -> mx.array:
    x32 = _activation_to_float32("x_flat", x_flat)
    R, H, I = _validate_one(
        x32, rows, w1, s1, w3, s3, w2, s2,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
    )
    if R == 0:
        return mx.zeros((0, H), dtype=mx.float32)
    _check_uint_grid("packed_fp4_forward_one", 256, (R + 7) // 8, (I + 7) // 8, (H + 7) // 8)
    limit_arr = _as_limit(limit)
    kernels = _kernels()
    hidden = kernels["ds4_fp4_pair_swiglu_forward"](
        inputs=[x32, rows, w1, s1, w3, s3, limit_arr],
        template=[("hidden_size", H), ("intermediate_size", I)],
        grid=(256, (R + 7) // 8, (I + 7) // 8),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, I)],
        output_dtypes=[mx.float32],
    )[0]
    return kernels["ds4_fp4_down_forward"](
        inputs=[hidden, w2, s2],
        template=[("hidden_size", H), ("intermediate_size", I)],
        grid=(256, (R + 7) // 8, (H + 7) // 8),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, H)],
        output_dtypes=[mx.float32],
    )[0]


def packed_fp4_input_vjp_one(
    x_flat: mx.array,
    g_flat: mx.array,
    rows: mx.array,
    factor: mx.array,
    w1: mx.array,
    s1: mx.array,
    w3: mx.array,
    s3: mx.array,
    w2: mx.array,
    s2: mx.array,
    *,
    hidden_size: int,
    intermediate_size: int,
    limit: float,
) -> tuple[mx.array, mx.array]:
    R, H, I = _validate_one(
        x_flat, rows, w1, s1, w3, s3, w2, s2,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
    )
    _check_rank_dtype("g_flat", g_flat, 2, mx.float32)
    _check_rank_dtype("factor", factor, 1, mx.float32)
    if g_flat.shape != x_flat.shape:
        raise ValueError("g_flat shape must equal x_flat shape")
    if factor.shape != (R,):
        raise ValueError(f"factor shape must be {(R,)}, got {factor.shape}")
    if R == 0:
        return mx.zeros((0, H), dtype=mx.float32), mx.zeros((0,), dtype=mx.float32)
    _check_uint_grid(
        "packed_fp4_input_vjp_one", 256, (R + 7) // 8, (I + 7) // 8, (H + 7) // 8, R
    )
    limit_arr = _as_limit(limit)
    kernels = _kernels()
    gu = kernels["ds4_fp4_down_input_vjp"](
        inputs=[g_flat, rows, w2, s2],
        template=[("hidden_size", H), ("intermediate_size", I)],
        grid=(256, (R + 7) // 8, (I + 7) // 8),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, I)],
        output_dtypes=[mx.float32],
    )[0]
    a_tiles = (I + 7) // 8
    dgate, dup, a_partial = kernels["ds4_fp4_pair_swiglu_vjp_terms"](
        inputs=[x_flat, rows, gu, factor, w1, s1, w3, s3, limit_arr],
        template=[("hidden_size", H), ("intermediate_size", I), ("a_tiles", a_tiles)],
        grid=(256, (R + 7) // 8, a_tiles),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, I), (R, I), (R, a_tiles)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    dx = kernels["ds4_fp4_pair_input_vjp"](
        inputs=[dgate, dup, w1, s1, w3, s3],
        template=[("hidden_size", H), ("intermediate_size", I)],
        grid=(256, (R + 7) // 8, (H + 7) // 8),
        threadgroup=(256, 1, 1),
        output_shapes=[(R, H)],
        output_dtypes=[mx.float32],
    )[0]
    a = kernels["ds4_fp4_reduce_a"](
        inputs=[a_partial],
        template=[],
        grid=(R * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(R,)],
        output_dtypes=[mx.float32],
    )[0]
    return dx, a


def _expert_arrays(experts: Any, eid: int) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    return (
        experts.w1_weight[eid],
        experts.w1_scale[eid],
        experts.w3_weight[eid],
        experts.w3_scale[eid],
        experts.w2_weight[eid],
        experts.w2_scale[eid],
    )


def _validate_expert_payload_bounds(experts: Any, n_experts: int) -> None:
    for name in ("w1_weight", "w1_scale", "w3_weight", "w3_scale", "w2_weight", "w2_scale"):
        arr = getattr(experts, name)
        if arr.ndim < 1 or int(arr.shape[0]) != n_experts:
            raise ValueError(f"{name} first dimension must match experts.n_routed_experts")


def _row_array(rows: Sequence[int]) -> mx.array:
    return mx.array([int(r) for r in rows], dtype=mx.int32)


def routed_fp4(
    x_flat: mx.array,
    scores_flat: mx.array,
    rows_by_expert: Sequence[Sequence[int]],
    experts: Any,
    *,
    routed_scaling_factor: float,
) -> mx.array:
    x32 = _activation_to_float32("x_flat", x_flat)
    _check_rank_dtype("scores_flat", scores_flat, 2, mx.float32)
    T, H = x32.shape
    if scores_flat.shape[0] != T:
        raise ValueError("scores_flat first dimension must match x_flat")
    H_expert = int(experts.hidden_size)
    I_expert = int(experts.intermediate_size)
    n_experts = int(experts.n_routed_experts)
    if scores_flat.shape[1] != n_experts:
        raise ValueError("scores_flat expert dimension must match experts.n_routed_experts")
    if len(rows_by_expert) != n_experts:
        raise ValueError("rows_by_expert length must match experts.n_routed_experts")
    if H != H_expert:
        raise ValueError("x_flat hidden dimension must match experts.hidden_size")
    _validate_expert_payload_bounds(experts, n_experts)
    rsf = _require_finite_positive("routed_scaling_factor", routed_scaling_factor)

    nonempty: list[tuple[int, mx.array, slice]] = []
    assignment_rows_py: list[int] = []
    assignment_eids_py: list[int] = []
    cursor = 0
    for eid, rows in enumerate(rows_by_expert):
        if not rows:
            continue
        rows_arr = _row_array(rows)
        _rows_list(rows_arr, T)
        n = len(rows)
        nonempty.append((eid, rows_arr, slice(cursor, cursor + n)))
        assignment_rows_py.extend(int(r) for r in rows)
        assignment_eids_py.extend([eid] * n)
        cursor += n

    if cursor == 0:
        return mx.zeros_like(x32)

    assignment_rows = mx.array(assignment_rows_py, dtype=mx.int32)
    assignment_eids = mx.array(assignment_eids_py, dtype=mx.int32)
    _check_uint_grid("routed_fp4", len(assignment_rows_py), T, n_experts, H_expert, I_expert)

    @mx.custom_function
    def _routed(_x32: mx.array, _scores: mx.array) -> mx.array:
        selected = _scores[assignment_rows, assignment_eids]
        denom = mx.zeros((T,), dtype=_scores.dtype).at[assignment_rows].add(selected)
        dprime = denom[assignment_rows] + 1e-20
        factor = (rsf * selected / dprime).astype(mx.float32)
        y_parts = []
        for eid, rows_arr, _segment in nonempty:
            w1, s1, w3, s3, w2, s2 = _expert_arrays(experts, eid)
            y_parts.append(
                packed_fp4_forward_one(
                    _x32, rows_arr, w1, s1, w3, s3, w2, s2,
                    hidden_size=H_expert, intermediate_size=I_expert, limit=experts.limit,
                )
            )
        y_assignment = mx.concatenate(y_parts, axis=0)
        contribution = y_assignment * factor[:, None]
        return mx.zeros((_x32.shape[0], _x32.shape[1]), dtype=mx.float32).at[assignment_rows].add(contribution)

    @_routed.vjp
    def _routed_vjp(primals: tuple[mx.array, mx.array], g: mx.array, routed: mx.array) -> tuple[mx.array, mx.array]:
        _x32, _scores = primals
        g32 = g.astype(mx.float32)
        selected = _scores[assignment_rows, assignment_eids]
        denom = mx.zeros((T,), dtype=_scores.dtype).at[assignment_rows].add(selected)
        dprime = denom[assignment_rows] + 1e-20
        factor = (rsf * selected / dprime).astype(mx.float32)
        dx_parts = []
        a_parts = []
        for eid, rows_arr, segment in nonempty:
            w1, s1, w3, s3, w2, s2 = _expert_arrays(experts, eid)
            dx_e, a_e = packed_fp4_input_vjp_one(
                _x32, g32, rows_arr, factor[segment], w1, s1, w3, s3, w2, s2,
                hidden_size=H_expert, intermediate_size=I_expert, limit=experts.limit,
            )
            dx_parts.append(dx_e)
            a_parts.append(a_e)
        dx_assignment = mx.concatenate(dx_parts, axis=0)
        a_assignment = mx.concatenate(a_parts, axis=0)
        dx = mx.zeros((_x32.shape[0], _x32.shape[1]), dtype=mx.float32).at[assignment_rows].add(dx_assignment)
        common = mx.sum(g32 * routed, axis=-1)
        dscore_assignment = rsf * a_assignment / dprime - common[assignment_rows] / dprime
        dscores = mx.zeros_like(_scores).at[assignment_rows, assignment_eids].add(dscore_assignment.astype(_scores.dtype))
        return dx, dscores

    return _routed(x32, scores_flat)
