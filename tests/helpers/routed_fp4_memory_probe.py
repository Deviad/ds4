#!/usr/bin/env python3
"""Fresh-process memory probe for Story 13.3b-5f routed packed-FP4 primitive."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MLX_SRC = ROOT / "python-envs" / "mlx" / "src"
if str(MLX_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_SRC))

import mlx.core as mx  # noqa: E402

from ds4_ft_mlx.routed_fp4_metal import packed_fp4_forward_one, packed_fp4_input_vjp_one, routed_fp4  # noqa: E402


class Experts:
    pass


def _experts(e: int, h: int, i: int) -> Experts:
    hp = ((h + 31) // 32) * 32
    ip = ((i + 31) // 32) * 32
    ex = Experts()
    ex.n_routed_experts = e
    ex.hidden_size = h
    ex.intermediate_size = i
    ex.limit = 10.0
    ex.w1_weight = mx.stack([mx.full((i, hp // 2), 0x11 + (eid % 7), dtype=mx.uint8) for eid in range(e)])
    ex.w3_weight = mx.stack([mx.full((i, hp // 2), 0x21 + (eid % 7), dtype=mx.uint8) for eid in range(e)])
    ex.w2_weight = mx.stack([mx.full((h, ip // 2), 0x31 + (eid % 7), dtype=mx.uint8) for eid in range(e)])
    ex.w1_scale = mx.stack([mx.full((i, hp // 32), 0.0005 * (1.0 + eid / max(1, e)), dtype=mx.bfloat16) for eid in range(e)])
    ex.w3_scale = mx.stack([mx.full((i, hp // 32), 0.0005 * (1.0 + eid / max(1, e)), dtype=mx.bfloat16) for eid in range(e)])
    ex.w2_scale = mx.stack([mx.full((h, ip // 32), 0.0005 * (1.0 + eid / max(1, e)), dtype=mx.bfloat16) for eid in range(e)])
    return ex


def _rows_by_expert(t: int, e: int, k: int) -> list[list[int]]:
    rows = [[] for _ in range(e)]
    for tok in range(t):
        for slot in range(k):
            rows[(tok + slot) % e].append(tok)
    return rows


def _scores(t: int, e: int) -> mx.array:
    return mx.ones((t, e), dtype=mx.float32)


def _mem() -> dict[str, int]:
    return {
        "active": int(mx.get_active_memory()),
        "cache": int(mx.get_cache_memory()),
        "peak": int(mx.get_peak_memory()),
    }


def _collect() -> None:
    mx.clear_cache()
    gc.collect()


def _finite_nonzero(arr: mx.array) -> tuple[bool, bool]:
    arr32 = arr.astype(mx.float32)
    mx.eval(arr32)
    return bool(mx.all(mx.isfinite(arr32)).item()), bool(mx.any(mx.abs(arr32) > 0).item())


def fixed_e(e: int) -> dict[str, object]:
    t, h, i, k = 1024, 1024, 512, 2
    ex = _experts(e, h, i)
    x = mx.ones((t, h), dtype=mx.float32) * 0.125
    scores = _scores(t, e)
    rows = _rows_by_expert(t, e, k)
    mx.eval(x, scores, ex.w1_weight, ex.w2_weight, ex.w3_weight, ex.w1_scale, ex.w2_scale, ex.w3_scale)
    _collect()
    baseline = _mem()
    mx.reset_peak_memory()

    def loss(args: dict[str, mx.array]) -> mx.array:
        y = routed_fp4(args["x"], args["scores"], rows, ex, routed_scaling_factor=1.0)
        return mx.sum(y * y)

    value, grads = mx.value_and_grad(loss)({"x": x, "scores": scores})
    mx.eval(value, grads["x"], grads["scores"])
    x_grad_finite, x_grad_nonzero = _finite_nonzero(grads["x"])
    score_grad_finite, score_grad_nonzero = _finite_nonzero(grads["scores"])
    operation = _mem()

    boundary = []
    _collect()
    mx.reset_peak_memory()
    g = mx.ones_like(x)
    for eid, erows in enumerate(rows):
        row_arr = mx.array(erows, dtype=mx.int32)
        factor = mx.ones((len(erows),), dtype=mx.float32)
        before = _mem()
        hidden_y = packed_fp4_forward_one(
            x, row_arr,
            ex.w1_weight[eid], ex.w1_scale[eid], ex.w3_weight[eid], ex.w3_scale[eid], ex.w2_weight[eid], ex.w2_scale[eid],
            hidden_size=h, intermediate_size=i, limit=ex.limit,
        )
        dx, a = packed_fp4_input_vjp_one(
            x, g, row_arr, factor,
            ex.w1_weight[eid], ex.w1_scale[eid], ex.w3_weight[eid], ex.w3_scale[eid], ex.w2_weight[eid], ex.w2_scale[eid],
            hidden_size=h, intermediate_size=i, limit=ex.limit,
        )
        mx.eval(hidden_y, dx, a)
        after = _mem()
        boundary.append({"expert": eid, "rows": len(erows), "before": before, "after": after})
    return {
        "case": "fixed-e",
        "T": t,
        "H": h,
        "I": i,
        "K": k,
        "E": e,
        "A": sum(len(r) for r in rows),
        "baseline": baseline,
        "operation": operation,
        "operation_peak_delta": max(0, operation["peak"] - baseline["active"]),
        "operation_active_delta": operation["active"] - baseline["active"],
        "x_grad_finite": x_grad_finite,
        "x_grad_nonzero": x_grad_nonzero,
        "score_grad_finite": score_grad_finite,
        "score_grad_nonzero": score_grad_nonzero,
        "boundary": boundary,
    }


def real_dim() -> dict[str, object]:
    t, h, i, e, k = 512, 4096, 2048, 2, 2
    ex = _experts(e, h, i)
    x = mx.ones((t, h), dtype=mx.float32) * 0.125
    scores = _scores(t, e)
    rows = _rows_by_expert(t, e, k)
    mx.eval(x, scores, ex.w1_weight, ex.w2_weight, ex.w3_weight, ex.w1_scale, ex.w2_scale, ex.w3_scale)
    _collect()
    baseline = _mem()
    mx.reset_peak_memory()

    def loss(args: dict[str, mx.array]) -> mx.array:
        y = routed_fp4(args["x"], args["scores"], rows, ex, routed_scaling_factor=1.0)
        return mx.sum(y * y)

    value, grads = mx.value_and_grad(loss)({"x": x, "scores": scores})
    mx.eval(value, grads["x"], grads["scores"])
    x_grad_finite, x_grad_nonzero = _finite_nonzero(grads["x"])
    score_grad_finite, score_grad_nonzero = _finite_nonzero(grads["scores"])
    operation = _mem()

    row_arr = mx.array(list(range(8)), dtype=mx.int32)
    factor = mx.ones((8,), dtype=mx.float32)
    g = mx.ones_like(x)
    y_e = packed_fp4_forward_one(
        x, row_arr,
        ex.w1_weight[0], ex.w1_scale[0], ex.w3_weight[0], ex.w3_scale[0], ex.w2_weight[0], ex.w2_scale[0],
        hidden_size=h, intermediate_size=i, limit=ex.limit,
    )
    dx_e, a_e = packed_fp4_input_vjp_one(
        x, g, row_arr, factor,
        ex.w1_weight[0], ex.w1_scale[0], ex.w3_weight[0], ex.w3_scale[0], ex.w2_weight[0], ex.w2_scale[0],
        hidden_size=h, intermediate_size=i, limit=ex.limit,
    )
    mx.eval(y_e, dx_e, a_e)
    direct_finite = all(bool(mx.all(mx.isfinite(arr.astype(mx.float32))).item()) for arr in (y_e, dx_e, a_e))

    return {
        "case": "real-dim",
        "T": t,
        "H": h,
        "I": i,
        "K": k,
        "E": e,
        "A": sum(len(r) for r in rows),
        "baseline": baseline,
        "operation": operation,
        "operation_peak_delta": max(0, operation["peak"] - baseline["active"]),
        "operation_active_delta": operation["active"] - baseline["active"],
        "x_grad_finite": x_grad_finite,
        "x_grad_nonzero": x_grad_nonzero,
        "score_grad_finite": score_grad_finite,
        "score_grad_nonzero": score_grad_nonzero,
        "direct_one_expert": {
            "y_shape": list(y_e.shape),
            "dx_shape": list(dx_e.shape),
            "a_shape": list(a_e.shape),
            "finite": direct_finite,
        },
    }


def formula() -> dict[str, int]:
    t, h, i, k = 4096, 4096, 2048, 6
    a = t * k
    mib = 1024 * 1024
    parts = {
        "assignment_y": a * h * 4,
        "assignment_i": a * i * 4,
        "a_partial": a * ((i + 7) // 8) * 4,
        "x_or_routed": t * h * 4,
        "scores": t * 256 * 4,
    }
    parts["forward_total"] = (192 + 384 + 384 + 64 + 64 + 4) * mib
    parts["backward_total"] = (192 + 192 + 192 + 24 + 384 + 256 + 8) * mib
    parts["envelope"] = parts["backward_total"] + 256 * mib
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["fixed-e", "real-dim", "formula"])
    ap.add_argument("--experts", type=int, default=2)
    ns = ap.parse_args()
    if ns.case == "fixed-e":
        out = fixed_e(ns.experts)
    elif ns.case == "real-dim":
        out = real_dim()
    else:
        out = formula()
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
