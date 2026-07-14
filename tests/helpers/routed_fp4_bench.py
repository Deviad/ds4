#!/usr/bin/env python3
"""Tracked routed FP4 micro-benchmark for Story 13.3b-5f.

Runs no full model/shards. Measures packed one-expert forward and input-VJP at
R=1/8/32/96 with H=4096, I=2048, then reports 256x43x20 extrapolation from
R=96 p50/p95. Use --dry-run for the chain-of-custody contract without timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MLX_SRC = ROOT / "python-envs" / "mlx" / "src"
if str(MLX_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_SRC))

import mlx.core as mx  # noqa: E402

from ds4_ft_mlx.routed_fp4_metal import packed_fp4_forward_one, packed_fp4_input_vjp_one  # noqa: E402

ROWS = [1, 8, 32, 96]
H = 4096
I = 2048
DEFAULT_REPEATS = 25
DEFAULT_WARMUP = 3


class Expert:
    pass


def _expert() -> Expert:
    hp = ((H + 31) // 32) * 32
    ip = ((I + 31) // 32) * 32
    ex = Expert()
    ex.w1 = mx.full((I, hp // 2), 0x11, dtype=mx.uint8)
    ex.w3 = mx.full((I, hp // 2), 0x21, dtype=mx.uint8)
    ex.w2 = mx.full((H, ip // 2), 0x31, dtype=mx.uint8)
    ex.s1 = mx.full((I, hp // 32), 0.0005, dtype=mx.bfloat16)
    ex.s3 = mx.full((I, hp // 32), 0.0005, dtype=mx.bfloat16)
    ex.s2 = mx.full((H, ip // 32), 0.0005, dtype=mx.bfloat16)
    mx.eval(ex.w1, ex.w3, ex.w2, ex.s1, ex.s3, ex.s2)
    return ex


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _eval_call(fn) -> None:
    out = fn()
    if isinstance(out, tuple):
        mx.eval(*out)
    else:
        mx.eval(out)


def _time_call(fn, repeats: int, warmup: int = DEFAULT_WARMUP) -> dict[str, float]:
    for _ in range(warmup):
        _eval_call(fn)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _eval_call(fn)
        timings.append(time.perf_counter() - start)
    return {
        "p50": statistics.median(timings),
        "p95": _percentile(timings, 0.95),
        "samples": len(timings),
        "warmup": warmup,
    }


def run(repeats: int) -> dict[str, object]:
    ex = _expert()
    results = []
    for r in ROWS:
        x = mx.ones((r, H), dtype=mx.float32) * 0.125
        g = mx.ones_like(x)
        rows = mx.arange(r, dtype=mx.int32)
        factor = mx.ones((r,), dtype=mx.float32)
        mx.eval(x, g, rows, factor)
        forward = _time_call(
            lambda: packed_fp4_forward_one(
                x, rows, ex.w1, ex.s1, ex.w3, ex.s3, ex.w2, ex.s2,
                hidden_size=H, intermediate_size=I, limit=10.0,
            ),
            repeats,
        )
        input_vjp = _time_call(
            lambda: packed_fp4_input_vjp_one(
                x, g, rows, factor, ex.w1, ex.s1, ex.w3, ex.s3, ex.w2, ex.s2,
                hidden_size=H, intermediate_size=I, limit=10.0,
            ),
            repeats,
        )
        results.append({"R": r, "forward": forward, "input_vjp": input_vjp})
    r96 = next(item for item in results if item["R"] == 96)
    scale = 256 * 43 * 20 / 3600.0
    return {
        "rows": ROWS,
        "H": H,
        "I": I,
        "repeats": repeats,
        "warmup": DEFAULT_WARMUP,
        "results": results,
        "extrapolated_hours_256x43x20": {
            "p50": r96["input_vjp"]["p50"] * scale,
            "p95": r96["input_vjp"]["p95"] * scale,
        },
    }


def dry_run(repeats: int) -> dict[str, object]:
    return {
        "rows": ROWS,
        "H": H,
        "I": I,
        "repeats": repeats,
        "warmup": DEFAULT_WARMUP,
        "command": "python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_bench.py --repeats 25",
        "extrapolation": "256x43x20 uses R=96 input-VJP p50/p95 seconds * 256 * 43 * 20 / 3600 hours",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ns = ap.parse_args()
    data = dry_run(ns.repeats) if ns.dry_run else run(ns.repeats)
    print(json.dumps(data, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
