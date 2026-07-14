"""Story 13.3b-5f routed packed-FP4 Metal primitive tests."""

from __future__ import annotations

import importlib.resources
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")

_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _decode_np(packed: np.ndarray, scale: np.ndarray, logical_cols: int) -> np.ndarray:
    out = np.zeros((packed.shape[0], logical_cols), dtype=np.float32)
    for r in range(packed.shape[0]):
        for k in range(logical_cols):
            byte = int(packed[r, k // 2])
            code = (byte & 0x0F) if k % 2 == 0 else ((byte >> 4) & 0x0F)
            out[r, k] = _LUT[code] * np.float32(scale[r, k // 32])
    return out


def _fixture(H: int = 64, I: int = 40, T: int = 5):
    hp = ((H + 31) // 32) * 32
    ip = ((I + 31) // 32) * 32
    x = (np.arange(T * H, dtype=np.float32).reshape(T, H) / 997.0) - 0.09
    rows = np.array([0, 3, 4], dtype=np.int32)
    w1 = (np.arange(I * (hp // 2), dtype=np.uint16).reshape(I, hp // 2) % 256).astype(np.uint8)
    w3 = ((np.arange(I * (hp // 2), dtype=np.uint16).reshape(I, hp // 2) * 3 + 5) % 256).astype(np.uint8)
    w2 = ((np.arange(H * (ip // 2), dtype=np.uint16).reshape(H, ip // 2) * 7 + 11) % 256).astype(np.uint8)
    s1 = (np.arange(I * (hp // 32), dtype=np.float32).reshape(I, hp // 32) % 5 + 1.0) / 4096.0
    s3 = (np.arange(I * (hp // 32), dtype=np.float32).reshape(I, hp // 32) % 7 + 1.0) / 4096.0
    s2 = (np.arange(H * (ip // 32), dtype=np.float32).reshape(H, ip // 32) % 3 + 1.0) / 4096.0
    return x, rows, w1, s1.astype(np.float32), w3, s3.astype(np.float32), w2, s2.astype(np.float32)


def _forward_np(x, rows, w1, s1, w3, s3, w2, s2, *, H: int, I: int, limit: float):
    W1 = _decode_np(w1, s1, ((H + 31) // 32) * 32)[:, :H]
    W3 = _decode_np(w3, s3, ((H + 31) // 32) * 32)[:, :H]
    W2 = _decode_np(w2, s2, ((I + 31) // 32) * 32)[:, :I]
    xe = x[rows]
    u1 = xe @ W1.T
    u3 = xe @ W3.T
    gate = np.minimum(u1, limit)
    up = np.clip(u3, -limit, limit)
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    y = hidden @ W2.T
    return y.astype(np.float32), hidden.astype(np.float32), W1, W3, W2, u1, u3


def _assert_forward_combined_close(got: np.ndarray, ref: np.ndarray, *, label: str) -> None:
    assert got.shape == ref.shape, label
    assert got.dtype == np.float32, label
    assert ref.dtype == np.float32, label
    assert np.isfinite(got).all(), label
    abs_err = np.abs(got - ref)
    bound = 2e-6 + 1e-6 * np.abs(ref)
    nrmse = float(np.sqrt(np.mean(np.square(abs_err.astype(np.float64)))) / max(float(np.sqrt(np.mean(np.square(ref.astype(np.float64))))), 1e-12))
    denom = np.maximum(np.maximum(np.abs(got), np.abs(ref)), 1e-6)
    max_rel = float(np.max(abs_err / denom))
    max_abs = float(np.max(abs_err))
    assert np.all(abs_err <= bound) and nrmse <= 1e-6, (
        f"{label} max_abs={max_abs} max_rel={max_rel} nrmse={nrmse}"
    )


def _kernel_body(source: str, name: str) -> str:
    begin = f"// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN {name}"
    end = f"// DS4_ROUTED_FP4_TRAIN_KERNEL_END {name}"
    return source.split(begin, 1)[1].split(end, 1)[0]


def test_package_local_metal_primitive_exists_with_six_kernel_markers():
    import ds4_ft_mlx.routed_fp4_metal as routed_fp4_metal

    source_path = importlib.resources.files("ds4_ft_mlx").joinpath(
        "metal/ds4_routed_fp4_train.metal"
    )
    source = source_path.read_text()
    assert "DS4_ROUTED_FP4_TRAIN_COMMON_BEGIN" in source
    assert "_dequantize_fp4_block_scale_mlx" not in source
    assert "gather_qmm" not in source
    for name in (
        "ds4_fp4_pair_swiglu_forward",
        "ds4_fp4_down_forward",
        "ds4_fp4_down_input_vjp",
        "ds4_fp4_pair_swiglu_vjp_terms",
        "ds4_fp4_pair_input_vjp",
        "ds4_fp4_reduce_a",
    ):
        assert name in source
        assert name in routed_fp4_metal.KERNEL_NAMES


def test_forward_one_matches_independent_packed_fp4_reference():
    from ds4_ft_mlx.routed_fp4_metal import packed_fp4_forward_one

    H, I = 64, 40
    x, rows, w1, s1, w3, s3, w2, s2 = _fixture(H, I)
    ref, *_ = _forward_np(x, rows, w1, s1, w3, s3, w2, s2, H=H, I=I, limit=1.7)
    out = packed_fp4_forward_one(
        mx.array(x, dtype=mx.float32),
        mx.array(rows, dtype=mx.int32),
        mx.array(w1, dtype=mx.uint8),
        mx.array(s1, dtype=mx.bfloat16),
        mx.array(w3, dtype=mx.uint8),
        mx.array(s3, dtype=mx.bfloat16),
        mx.array(w2, dtype=mx.uint8),
        mx.array(s2, dtype=mx.bfloat16),
        hidden_size=H,
        intermediate_size=I,
        limit=1.7,
    )
    mx.eval(out)
    got = np.array(out.tolist(), dtype=np.float32)
    assert out.shape == (len(rows), H)
    assert out.dtype == mx.float32
    _assert_forward_combined_close(got, ref, label="primitive forward")


def test_input_vjp_one_matches_independent_dense_reference():
    from ds4_ft_mlx.routed_fp4_metal import packed_fp4_input_vjp_one

    H, I = 64, 40
    x, rows, w1, s1, w3, s3, w2, s2 = _fixture(H, I)
    g = (np.arange(x.size, dtype=np.float32).reshape(x.shape) / 1231.0) + 0.02
    factor = np.array([0.125, 0.5, 1.25], dtype=np.float32)
    y, hidden, W1, W3, W2, u1, u3 = _forward_np(x, rows, w1, s1, w3, s3, w2, s2, H=H, I=I, limit=1.7)
    ge = g[rows]
    gu = ge @ W2
    gate = np.minimum(u1, 1.7)
    up = np.clip(u3, -1.7, 1.7)
    sig = 1.0 / (1.0 + np.exp(-gate))
    silu = gate * sig
    dhidden = factor[:, None] * gu
    dup = dhidden * silu * ((u3 > -1.7) & (u3 < 1.7))
    dgate = dhidden * up * (sig + gate * sig * (1.0 - sig)) * (u1 < 1.7)
    dx_ref = dgate @ W1 + dup @ W3
    a_ref = np.sum(hidden * gu, axis=-1)

    dx, a = packed_fp4_input_vjp_one(
        mx.array(x, dtype=mx.float32),
        mx.array(g, dtype=mx.float32),
        mx.array(rows, dtype=mx.int32),
        mx.array(factor, dtype=mx.float32),
        mx.array(w1, dtype=mx.uint8),
        mx.array(s1, dtype=mx.bfloat16),
        mx.array(w3, dtype=mx.uint8),
        mx.array(s3, dtype=mx.bfloat16),
        mx.array(w2, dtype=mx.uint8),
        mx.array(s2, dtype=mx.bfloat16),
        hidden_size=H,
        intermediate_size=I,
        limit=1.7,
    )
    mx.eval(dx, a)
    dx_np = np.array(dx.tolist(), dtype=np.float32)
    a_np = np.array(a.tolist(), dtype=np.float32)
    assert float(np.max(np.abs(dx_np - dx_ref))) <= 1e-5
    assert float(np.max(np.abs(a_np - a_ref))) <= 1e-5


def test_fail_closed_shape_guard_prevents_silent_fallback():
    from ds4_ft_mlx.routed_fp4_metal import packed_fp4_forward_one

    with pytest.raises(ValueError, match="w1/w3"):
        packed_fp4_forward_one(
            mx.zeros((1, 8), dtype=mx.float32),
            mx.array([0], dtype=mx.int32),
            mx.zeros((1, 4), dtype=mx.uint8),
            mx.zeros((1, 1), dtype=mx.bfloat16),
            mx.zeros((1, 4), dtype=mx.uint8),
            mx.zeros((1, 1), dtype=mx.bfloat16),
            mx.zeros((8, 16), dtype=mx.uint8),
            mx.zeros((8, 1), dtype=mx.bfloat16),
            hidden_size=8,
            intermediate_size=1,
            limit=1.0,
        )


def test_forward_one_promotes_half_activation_without_inconsistent_reject():
    from ds4_ft_mlx.routed_fp4_metal import packed_fp4_forward_one

    H, I = 64, 40
    x, rows, w1, s1, w3, s3, w2, s2 = _fixture(H, I)
    out = packed_fp4_forward_one(
        mx.array(x, dtype=mx.float16), mx.array(rows, dtype=mx.int32),
        mx.array(w1, dtype=mx.uint8), mx.array(s1, dtype=mx.bfloat16), mx.array(w3, dtype=mx.uint8), mx.array(s3, dtype=mx.bfloat16),
        mx.array(w2, dtype=mx.uint8), mx.array(s2, dtype=mx.bfloat16), hidden_size=H, intermediate_size=I, limit=1.7,
    )
    mx.eval(out)
    assert out.shape == (len(rows), H)
    assert out.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(out)).item())


# ---------------------------------------------------------------------------
# Reviewer 5f remediation: structural opacity, guard, derivative, memory probes.
# ---------------------------------------------------------------------------

class _FakeExperts:
    pass


def _stacked_experts(H: int = 32, I: int = 8, E: int = 2):
    ex = _FakeExperts()
    hp = ((H + 31) // 32) * 32
    ip = ((I + 31) // 32) * 32
    ex.n_routed_experts = E
    ex.hidden_size = H
    ex.intermediate_size = I
    ex.limit = 2.0
    ex.w1_weight = mx.full((E, I, hp // 2), 0x11, dtype=mx.uint8)
    ex.w3_weight = mx.full((E, I, hp // 2), 0x11, dtype=mx.uint8)
    ex.w2_weight = mx.full((E, H, ip // 2), 0x11, dtype=mx.uint8)
    ex.w1_scale = mx.full((E, I, hp // 32), 0.01, dtype=mx.bfloat16)
    ex.w3_scale = mx.full((E, I, hp // 32), 0.01, dtype=mx.bfloat16)
    ex.w2_scale = mx.full((E, H, ip // 32), 0.01, dtype=mx.bfloat16)
    return ex


def _one_fp4_row(code: int, rows: int = 1, logical_cols: int = 32, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    packed = np.zeros((rows, ((logical_cols + 31) // 32) * 16), dtype=np.uint8)
    packed[:, 0] = np.uint8(code & 0x0F)
    scales = np.ones((rows, ((logical_cols + 31) // 32)), dtype=np.float32) * scale
    return packed, scales


def test_matrix_kernels_structurally_use_8x8x32_tiles_and_simd_reductions():
    import ds4_ft_mlx.routed_fp4_metal as routed_fp4_metal

    source = importlib.resources.files("ds4_ft_mlx").joinpath("metal/ds4_routed_fp4_train.metal").read_text()
    wrapper = Path(routed_fp4_metal.__file__).read_text()
    assert "threadgroup=(256, 1, 1)" in wrapper
    assert "grid=(256, (R + 7) // 8" in wrapper
    assert "uint(hidden_size) <= 32u" not in source
    assert "uint(intermediate_size) <= 32u" not in source
    assert "for (uint c = 0u; c < 32u; ++c)" not in source
    assert "threadgroup float x_tile[8][32]" not in source
    assert "threadgroup float hv_tile[8][32]" not in source
    assert "threadgroup float gv_tile[8][32]" not in source
    assert "threadgroup float dg_tile[8][32]" not in source
    assert "threadgroup float du_tile[8][32]" not in source
    assert "(lane & 31u) == 0u && r" not in source
    for name in (
        "ds4_fp4_pair_swiglu_forward",
        "ds4_fp4_down_forward",
        "ds4_fp4_down_input_vjp",
        "ds4_fp4_pair_swiglu_vjp_terms",
        "ds4_fp4_pair_input_vjp",
    ):
        body = _kernel_body(source, name)
        assert "k0 += 32u" in body, name
        assert "simd_row == 0u" in body, name
        assert "simd_sum" in body, name
        assert "[lane]" in body, name
        assert "lane == 0u)" in body, name
    k1 = _kernel_body(source, "ds4_fp4_pair_swiglu_forward")
    k4 = _kernel_body(source, "ds4_fp4_pair_swiglu_vjp_terms")
    skeleton = [
        "w1_tile[j][lane]",
        "w3_tile[j][lane]",
        "u1[j] = metal::fma(xv, w1_tile[j][lane], u1[j]);",
        "u3[j] = metal::fma(xv, w3_tile[j][lane], u3[j]);",
        "u1[j] = simd_sum(u1[j]);",
        "u3[j] = simd_sum(u3[j]);",
    ]
    for snippet in skeleton:
        assert snippet in k1
        assert snippet in k4


@pytest.mark.parametrize(
    "case,w1_scale,w3_code,w3_scale",
    [
        ("u1 == +limit", 1.0, 0x01, 1.0),
        ("u3 == +limit", 0.5, 0x02, 1.0),
        ("u3 == -limit", 0.5, 0x0A, 1.0),
    ],
)
def test_input_vjp_exact_clip_boundaries_are_sensitivity_proven(case, w1_scale, w3_code, w3_scale):
    from ds4_ft_mlx.routed_fp4_metal import packed_fp4_input_vjp_one

    H, I, limit = 32, 40, 1.0
    x = np.zeros((1, H), dtype=np.float32)
    x[0, 0] = 1.0
    g = np.zeros_like(x)
    g[0, 0] = 1.0
    rows = np.array([0], dtype=np.int32)
    factor = np.array([1.0], dtype=np.float32)
    w1, s1 = _one_fp4_row(0x02, rows=I, logical_cols=H, scale=w1_scale)
    w3, s3 = _one_fp4_row(w3_code, rows=I, logical_cols=H, scale=w3_scale)
    w2, s2 = _one_fp4_row(0x02, rows=H, logical_cols=I, scale=1.0)

    _, hidden, W1, W3, W2, u1, u3 = _forward_np(x, rows, w1, s1, w3, s3, w2, s2, H=H, I=I, limit=limit)
    ge = g[rows]
    gu = ge @ W2
    gate = np.minimum(u1, limit)
    up = np.clip(u3, -limit, limit)
    sig = 1.0 / (1.0 + np.exp(-gate))
    silu = gate * sig
    dhidden = factor[:, None] * gu
    raw_dup = dhidden * silu
    raw_dgate = dhidden * up * (sig + gate * sig * (1.0 - sig))
    dup = raw_dup * ((u3 > -limit) & (u3 < limit))
    dgate = raw_dgate * (u1 < limit)
    dx_ref = dgate @ W1 + dup @ W3
    a_ref = np.sum(hidden * gu, axis=-1)

    assert np.isclose(u1[0, 0], limit) if case == "u1 == +limit" else True
    assert np.isclose(u3[0, 0], limit) if case == "u3 == +limit" else True
    assert np.isclose(u3[0, 0], -limit) if case == "u3 == -limit" else True
    assert abs(float(gu[0, 0])) > 0.0
    assert abs(float(up[0, 0])) > 0.0
    wrong_dup = raw_dup * ((u3 >= -limit) & (u3 <= limit))
    wrong_dgate = raw_dgate * (u1 <= limit)
    wrong_dx = wrong_dgate @ W1 + wrong_dup @ W3
    assert abs(float(raw_dgate[0, 0])) > 0.0
    if case == "u1 == +limit":
        assert abs(float(wrong_dgate[0, 0] - dgate[0, 0])) > 0.0
    else:
        assert abs(float(raw_dup[0, 0])) > 0.0
        assert abs(float(wrong_dup[0, 0] - dup[0, 0])) > 0.0
    assert float(np.max(np.abs(wrong_dx - dx_ref))) > 1e-3

    dx, a = packed_fp4_input_vjp_one(
        mx.array(x, dtype=mx.float32), mx.array(g, dtype=mx.float32), mx.array(rows, dtype=mx.int32), mx.array(factor, dtype=mx.float32),
        mx.array(w1, dtype=mx.uint8), mx.array(s1, dtype=mx.bfloat16), mx.array(w3, dtype=mx.uint8), mx.array(s3, dtype=mx.bfloat16),
        mx.array(w2, dtype=mx.uint8), mx.array(s2, dtype=mx.bfloat16), hidden_size=H, intermediate_size=I, limit=limit,
    )
    mx.eval(dx, a)
    assert float(np.max(np.abs(np.array(dx.tolist(), dtype=np.float32) - dx_ref))) <= 1e-6
    assert float(np.max(np.abs(np.array(a.tolist(), dtype=np.float32) - a_ref))) <= 1e-6


def test_simplified_score_identity_matches_independent_q_form_on_actual_assignments():
    scores = np.array([[0.2, 0.7, 0.4], [0.5, 0.1, 0.3]], dtype=np.float32)
    duplicate_indices = [[1, 1, 2], [0, 2]]
    assignments: list[tuple[int, int]] = []
    for t, slots in enumerate(duplicate_indices):
        seen: set[int] = set()
        for eid in slots:
            if eid not in seen:
                assignments.append((t, eid))
                seen.add(eid)
    assert assignments == [(0, 1), (0, 2), (1, 0), (1, 2)]

    y_assignment = np.array(
        [[1.0, -2.0, 0.5], [0.5, 1.0, -1.0], [-1.5, 0.25, 2.0], [0.75, 1.25, -0.5]],
        dtype=np.float32,
    )
    upstream = np.array([[0.25, -0.5, 1.5], [1.0, 0.5, -0.25]], dtype=np.float32)
    rsf = 1.5
    denom = np.zeros((scores.shape[0],), dtype=np.float32)
    for t, eid in assignments:
        denom[t] += scores[t, eid]
    routed = np.zeros_like(upstream)
    a = np.zeros((len(assignments),), dtype=np.float32)
    for k, (t, eid) in enumerate(assignments):
        factor = rsf * scores[t, eid] / denom[t]
        routed[t] += factor * y_assignment[k]
        a[k] = float(np.dot(upstream[t], y_assignment[k]))
    common = np.sum(upstream * routed, axis=-1)

    simplified = np.zeros_like(scores)
    prior_q = np.zeros_like(scores)
    for k, (t, eid) in enumerate(assignments):
        simplified[t, eid] += rsf * a[k] / denom[t] - common[t] / denom[t]
        support = [j for j, (tj, _ej) in enumerate(assignments) if tj == t]
        old = 0.0
        for j in support:
            _tj, ej = assignments[j]
            derivative = (1.0 if ej == eid else 0.0) / denom[t] - scores[t, ej] / (denom[t] ** 2)
            old += rsf * a[j] * derivative
        prior_q[t, eid] = old
    assert np.count_nonzero(simplified[0]) == 2
    assert np.count_nonzero(simplified[1]) == 2
    assert simplified[0, 0] == simplified[1, 1] == 0.0
    assert np.allclose(simplified, prior_q, rtol=0.0, atol=1e-5)


def test_routed_outer_transform_freezes_payloads_and_returns_x_score_cotangents():
    from ds4_ft_mlx.routed_fp4_metal import routed_fp4

    ex = _stacked_experts(H=32, I=8, E=2)
    x = (mx.arange(96, dtype=mx.float32).reshape(3, 32) / 97.0 + 0.1).astype(mx.bfloat16)
    scores = mx.array([[0.7, 0.2], [0.4, 0.6], [0.1, 0.9]], dtype=mx.float32)
    rows = [[0, 1], [1, 2]]

    def loss(args):
        xx, ss = args["x"], args["scores"]
        y = routed_fp4(xx, ss, rows, ex, routed_scaling_factor=1.25)
        return mx.sum(y * mx.arange(96, dtype=mx.float32).reshape(3, 32))

    value, grads = mx.value_and_grad(loss)({"x": x, "scores": scores})
    mx.eval(value, grads["x"], grads["scores"])
    assert bool(mx.isfinite(value).item())
    assert bool(mx.all(mx.isfinite(grads["x"])).item())
    assert bool(mx.all(mx.isfinite(grads["scores"])).item())
    assert bool(mx.any(mx.abs(grads["x"].astype(mx.float32)) > 0).item())
    assert bool(mx.any(mx.abs(grads["scores"].astype(mx.float32)) > 0).item())
    for name in ("w1_weight", "w1_scale", "w3_weight", "w3_scale", "w2_weight", "w2_scale"):
        assert name not in grads


def test_exported_outer_graph_contains_routed_custom_kernels_not_dequant_arithmetic(tmp_path):
    from ds4_ft_mlx.routed_fp4_metal import routed_fp4

    ex = _stacked_experts(H=32, I=8, E=2)
    x = mx.ones((3, 32), dtype=mx.float32) * 0.25
    scores = mx.array([[0.7, 0.2], [0.4, 0.6], [0.1, 0.9]], dtype=mx.float32)
    y = routed_fp4(x, scores, [[0, 1], [1, 2]], ex, routed_scaling_factor=1.0)
    dot = tmp_path / "routed.dot"
    mx.export_to_dot(str(dot), y)
    text = dot.read_text()
    assert text.count("CustomKernel") >= 2
    assert "dequant" not in text.lower()
    assert "GatherQMM" not in text


def test_source_and_api_reject_dense_fallback_native_mxfp4_and_differentiable_frozen_payloads():
    source = Path(_MLX_SRC / "ds4_ft_mlx" / "routed_fp4_metal.py").read_text()
    metal = importlib.resources.files("ds4_ft_mlx").joinpath("metal/ds4_routed_fp4_train.metal").read_text()
    assert "mx.vjp(forward_one" not in source
    assert "forward_one(eid" not in source
    assert "mxfp4" not in source.lower() + metal.lower()
    assert "e8m0" not in source.lower() + metal.lower()
    assert "_routed(_x32: mx.array, _scores: mx.array)" in source
    assert "return dx, dscores" in source
    assert "w1_weight" not in source.split("def _routed(", 1)[1].split("@_routed.vjp", 1)[0]


def test_installed_package_resource_and_exact_mlx_dependency_contract():
    source_path = importlib.resources.files("ds4_ft_mlx").joinpath("metal/ds4_routed_fp4_train.metal")
    assert source_path.is_file()
    pyproject = (_ROOT / "python-envs" / "mlx" / "pyproject.toml").read_text()
    assert '"mlx==0.31.2"' in pyproject
    assert '"mlx>=0.31.2"' not in pyproject
    assert 'ds4_ft_mlx = ["metal/*.metal"]' in pyproject


def test_unsupported_platform_or_version_fails_closed(monkeypatch):
    import ds4_ft_mlx.routed_fp4_metal as routed_fp4_metal

    monkeypatch.setattr(routed_fp4_metal.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="requires darwin"):
        routed_fp4_metal._ensure_supported()
    monkeypatch.setattr(routed_fp4_metal.sys, "platform", "darwin")
    monkeypatch.setattr(routed_fp4_metal.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError, match="requires Apple arm64"):
        routed_fp4_metal._ensure_supported()
    monkeypatch.setattr(routed_fp4_metal.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(routed_fp4_metal.importlib.metadata, "version", lambda _name: "0.31.3")
    with pytest.raises(RuntimeError, match="requires mlx==0.31.2"):
        routed_fp4_metal._ensure_supported()


def test_duplicate_metal_source_markers_are_rejected():
    import ds4_ft_mlx.routed_fp4_metal as routed_fp4_metal

    dup = "A // DS4_ROUTED_FP4_TRAIN_COMMON_BEGIN x // DS4_ROUTED_FP4_TRAIN_COMMON_END B // DS4_ROUTED_FP4_TRAIN_COMMON_BEGIN y // DS4_ROUTED_FP4_TRAIN_COMMON_END"
    with pytest.raises(RuntimeError, match="exactly one"):
        routed_fp4_metal._require_unique_section(dup, routed_fp4_metal._COMMON_BEGIN, routed_fp4_metal._COMMON_END, "common")


def test_routed_wrapper_deterministic_guards_before_kernel_launch():
    from ds4_ft_mlx.routed_fp4_metal import routed_fp4

    ex = _stacked_experts(E=2)
    x = mx.ones((2, 32), dtype=mx.float32)
    scores = mx.ones((2, 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="scores_flat must have dtype"):
        routed_fp4(x, scores.astype(mx.float16), [[0], [1]], ex, routed_scaling_factor=1.0)
    with pytest.raises(ValueError, match="expert dimension"):
        routed_fp4(x, mx.ones((2, 3), dtype=mx.float32), [[0], [1]], ex, routed_scaling_factor=1.0)
    with pytest.raises(ValueError, match="rows_by_expert length"):
        routed_fp4(x, scores, [[0]], ex, routed_scaling_factor=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        routed_fp4(x, scores, [[0], [1]], ex, routed_scaling_factor=float("nan"))
    ex_bad = _stacked_experts(E=2)
    ex_bad.w1_weight = ex_bad.w1_weight[:1]
    with pytest.raises(ValueError, match="w1_weight first dimension"):
        routed_fp4(x, scores, [[0], [1]], ex_bad, routed_scaling_factor=1.0)


def test_fresh_process_memory_probe_formula_and_commands_are_tracked():
    helper = _ROOT / "tests" / "helpers" / "routed_fp4_memory_probe.py"
    assert helper.is_file()
    import subprocess, json
    tracked = subprocess.check_output(["git", "ls-files", str(helper)], text=True).strip()
    assert tracked == str(helper.relative_to(_ROOT))
    out = subprocess.check_output([sys.executable, str(helper), "formula"], text=True)
    data = json.loads(out)
    assert data["assignment_y"] == 384 * 1024 * 1024
    assert data["assignment_i"] == 192 * 1024 * 1024
    assert data["a_partial"] == 24 * 1024 * 1024
    assert data["envelope"] == 1504 * 1024 * 1024
    assert data["envelope"] < 2 * 1024 * 1024 * 1024


def test_fresh_process_fixed_e_operation_peaks_are_assignment_bounded():
    import subprocess, json
    helper = _ROOT / "tests" / "helpers" / "routed_fp4_memory_probe.py"
    peaks = []
    for experts in (2, 4, 8):
        raw = subprocess.check_output([sys.executable, str(helper), "fixed-e", "--experts", str(experts)], text=True, timeout=240)
        data = json.loads(raw)
        assert data["T"] == 1024 and data["H"] == 1024 and data["I"] == 512 and data["K"] == 2
        assert data["A"] == 2048
        assert len(data["boundary"]) == experts
        assert all(sample["rows"] > 0 for sample in data["boundary"])
        assert data["operation_peak_delta"] < 1504 * 1024 * 1024
        assert data["x_grad_finite"] and data["x_grad_nonzero"]
        assert data["score_grad_finite"] and data["score_grad_nonzero"]
        peaks.append(data["operation_peak_delta"])
    assert max(peaks) - min(peaks) <= 64 * 1024 * 1024


def test_fresh_process_real_dimension_operation_peak_under_two_gib():
    import subprocess, json
    helper = _ROOT / "tests" / "helpers" / "routed_fp4_memory_probe.py"
    raw = subprocess.check_output([sys.executable, str(helper), "real-dim"], text=True, timeout=240)
    data = json.loads(raw)
    assert data["H"] == 4096 and data["I"] == 2048
    assert data["operation_peak_delta"] < 2 * 1024 * 1024 * 1024
    assert data["x_grad_finite"] and data["x_grad_nonzero"]
    assert data["score_grad_finite"] and data["score_grad_nonzero"]
    direct = data["direct_one_expert"]
    assert direct["y_shape"] == [8, 4096]
    assert direct["dx_shape"] == [8, 4096]
    assert direct["a_shape"] == [8]
    assert direct["finite"]


def test_tracked_routed_fp4_benchmark_helper_lists_exact_r_cases_and_commands():
    import subprocess, json
    helper = _ROOT / "tests" / "helpers" / "routed_fp4_bench.py"
    assert helper.is_file()
    tracked = subprocess.check_output(["git", "ls-files", str(helper)], text=True).strip()
    assert tracked == str(helper.relative_to(_ROOT))
    raw = subprocess.check_output([sys.executable, str(helper), "--dry-run"], text=True)
    data = json.loads(raw)
    assert data["rows"] == [1, 8, 32, 96]
    assert data["repeats"] >= 25
    assert data["warmup"] >= 1
    assert "256x43x20" in data["extrapolation"]
    assert "tests/helpers/routed_fp4_bench.py" in data["command"]
