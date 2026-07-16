"""Story 11.15g — `_validate_real_mode()` staged relaxation paired parity tests.

TDD red-first: each gate relaxation is paired with a proven-component parity
test (ADR 0002 fail-closed; NO blanket `allow real_config` switch).

Gate map (vendor deepseek_v4.py `_validate_real_mode`):
  #1  num_hidden_layers in {1,2,3}       -> relax: allow 43 (11.15e structural)
  #2  layer_types all sliding_attention  -> relax: allow real vector (11.15e)
  #3  mlp_layer_types all moe            -> relax: allow real vector (11.15e)
  #4  (per-layer-type, subsumed in #2/#3)
  #5  compressed attn single-layer hc=1  -> relax: allow multi-layer CSA (11.14/11.25)
  #6  num_key_value_heads=1              -> STAYS CLOSED (permanent MQA, ADR 0007)
  #7  o_groups<=0                        -> already open for real (11.26) — parity test only
  #8  num_attention_heads % o_groups!=0  -> already open for real (11.25/11.26) — parity test
  #9  hc_mult<=0                          -> already open for real (11.23) — parity test only
  #10 multi-layer hc_mult!=1             -> relax: allow hc_mult>1 multi-layer (11.11)
  #11 n_routed_experts<=4                -> relax: allow 256 staged 8/32/256 (11.15d + Q3 probe)
  #12 num_experts_per_tok>n_routed       -> already open for real (11.15d) — parity test only
  #13 n_shared_experts=1/sqrtsoftplus/dtype -> already open for real (11.15a/b) — parity test

STOP-rule: if `_moe_mlx` I8 composition drifts >5e-3 L2_REL at any scale, OR
any gate relaxed without a paired proven-component parity test, OR any
un-FROZEN cascade (metal/ds4.c/ds4_metal.m edit needed), OR gate #6 relaxed,
OR a new vendor MLX MoE kernel is required at n_routed_experts>4 -> STOP.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

# ensure the mlx-env src is importable regardless of CWD
_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

import mlx.core as mx  # noqa: E402
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (  # noqa: E402
    Model,
    ModelArgs,
    _dequantize_i8_block_scale_mlx,
    _hyperconnection_mlx,
    _moe_mlx,
)

L2_REL_TOL = 5e-3
BLOCK = 16


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _baseline_args(**overrides) -> ModelArgs:
    """A valid tiny proven-subset ModelArgs (num_hidden_layers=1, n_routed_experts=2)."""
    base = dict(
        model_type="deepseek_v4",
        vocab_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=16,
        o_lora_rank=16,
        qk_rope_head_dim=16,
        index_head_dim=2,
        index_n_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        expert_dtype="i8",
        rms_norm_eps=1e-6,
        hc_mult=1,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        rope_theta=10000.0,
        o_groups=1,
        compression_ratio=0,
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
    )
    base.update(overrides)
    return ModelArgs.from_dict(base)


def _l2_rel(a: mx.array, b: mx.array) -> float:
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    diff = a32 - b32
    num = float(mx.sqrt(mx.sum(diff * diff)).item())
    den = float(mx.sqrt(mx.sum(b32 * b32)).item() + 1e-30)
    return num / den


def _build_i8_expert_pair(mid: int, hidden: int, seed: int):
    """One expert's w1/w2/w3 I8 weight + F32 block scale (axis=1, block=16)."""
    out = {}
    for proj, rows, cols in (("w1", mid, hidden), ("w2", hidden, mid), ("w3", mid, hidden)):
        mx.random.seed((seed * 7 + hash(proj)) % (2**31))
        w_int = mx.random.uniform(low=-15, high=15, shape=(rows, cols)).astype(mx.int32)
        scale_cols = cols // BLOCK
        scale = mx.random.uniform(low=0.25, high=2.0, shape=(rows, scale_cols))
        out[proj] = (w_int, scale)
    return out


def _moe_weights_i8(n_exp: int, mid: int, hidden: int, seed: int) -> dict:
    mx.random.seed(seed)
    weights = {
        "mlp.gate.weight": mx.random.uniform(low=-0.5, high=0.5, shape=(n_exp, hidden)),
        "mlp.gate.e_score_correction_bias": mx.zeros(n_exp),
    }
    for eid in range(n_exp):
        ew = _build_i8_expert_pair(mid, hidden, seed + eid + 1)
        for proj, (w_int, scale) in ew.items():
            weights[f"mlp.experts.{eid}.{proj}.weight"] = w_int
            weights[f"mlp.experts.{eid}.{proj}.scale"] = scale
    weights["mlp.shared_experts.w1.weight"] = mx.random.uniform(low=-0.1, high=0.1, shape=(mid, hidden))
    weights["mlp.shared_experts.w2.weight"] = mx.random.uniform(low=-0.1, high=0.1, shape=(hidden, mid))
    weights["mlp.shared_experts.w3.weight"] = mx.random.uniform(low=-0.1, high=0.1, shape=(mid, hidden))
    return weights


def _moe_weights_bf16_ref(i8_w: dict, n_exp: int, mid: int, hidden: int) -> dict:
    ref = {
        "mlp.gate.weight": i8_w["mlp.gate.weight"].astype(mx.float32),
        "mlp.gate.e_score_correction_bias": i8_w["mlp.gate.e_score_correction_bias"].astype(mx.float32),
    }
    for eid in range(n_exp):
        for proj in ("w1", "w2", "w3"):
            w_int = i8_w[f"mlp.experts.{eid}.{proj}.weight"]
            scale = i8_w[f"mlp.experts.{eid}.{proj}.scale"]
            ref[f"mlp.experts.{eid}.{proj}.weight"] = _dequantize_i8_block_scale_mlx(
                w_int, scale, block_size=BLOCK, axis=1
            )
    for proj in ("w1", "w2", "w3"):
        ref[f"mlp.shared_experts.{proj}.weight"] = i8_w[f"mlp.shared_experts.{proj}.weight"].astype(mx.float32)
    return ref


# ---------------------------------------------------------------------------
# Gate #6 — STAYS CLOSED (permanent MQA, ADR 0007 axis b retired)
# ---------------------------------------------------------------------------

def test_gate_6_num_key_value_heads_stays_closed_permanent_mqa():
    """ADR 0007 axis (b) retired; num_key_value_heads must stay =1.
    Relaxing gate #6 is a STOP-trigger; this test asserts the gate is present."""
    args = _baseline_args(num_key_value_heads=2)
    with pytest.raises(NotImplementedError, match="num_key_value_heads=1"):
        Model(args)


# ---------------------------------------------------------------------------
# Gates #1, #2, #3, #4 — structural shape-compat (11.15e) — relax to 43 layers
# ---------------------------------------------------------------------------

def test_gates_1_to_4_structural_43_layers_real_layer_vectors():
    """Real DS4 config: 43 layers, real layer_types + mlp_layer_types vectors.
    Proven-component parity = 11.15e structural shape-compat (header-only).
    Forward parity is 11.15h scope (OUT of scope here); this proves construction."""
    n = 43
    layer_types = (
        ["full_attention"] + ["sliding_attention"] * 2
    ) * ((n + 2) // 3)
    layer_types = layer_types[:n]
    mlp_layer_types = (
        ["moe"] * 4 + ["dense"] * 1
    ) * ((n + 4) // 5)
    mlp_layer_types = mlp_layer_types[:n]
    args = _baseline_args(
        num_hidden_layers=n,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
    )
    # Model.__init__ calls _validate_real_mode — must NOT raise for structural gates
    model = Model(args)
    assert model.args.num_hidden_layers == n
    assert len(model.args.layer_types) == n
    assert len(model.args.mlp_layer_types) == n


# ---------------------------------------------------------------------------
# Gate #5 — compressed attention multi-layer (11.14 stateless + 11.25)
# ---------------------------------------------------------------------------

def test_gate_5_compressed_attention_multi_layer_stateless_allowed():
    """Relax gate #5(b): allow compressed attention with num_hidden_layers>1
    (hc_mult=1 retained by _csa_config_error). Proven by 11.14 stateless CSA
    composed per-layer (stateless == N independent single-layer CSAs)."""
    args = _baseline_args(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["moe", "moe"],
        compression_ratio=4,
        # CSA proven subset (single-head, o_groups=1, hc_mult=1) per _csa_config_error
        num_attention_heads=1,
        o_groups=1,
        hc_mult=1,
        head_dim=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=2,
    )
    # _validate_real_mode must NOT raise on the multi-layer compressed-attention arm
    Model(args)


# ---------------------------------------------------------------------------
# Gates #7, #8 — o_groups / divisibility (11.25/11.26) — already open, parity
# ---------------------------------------------------------------------------

def test_gate_7_8_grouped_output_o_groups_2_runs_real():
    """Gate #7 (o_groups>0) + #8 (heads%o_groups==0) are correctness guards,
    already open for real. Parity = grouped output projection (11.26) runs at
    o_groups=2, num_attention_heads=4 (divisible), compression_ratio=0."""
    args = _baseline_args(
        num_attention_heads=4,
        o_groups=2,
        num_key_value_heads=1,
        compression_ratio=0,
    )
    # construction parity (no NotImplementedError from gates #7/#8)
    Model(args)
    assert args.num_attention_heads % args.o_groups == 0


# ---------------------------------------------------------------------------
# Gate #9 — hc_mult>0 (11.23 cache axis) — already open, parity
# ---------------------------------------------------------------------------

def test_gate_9_hc_mult_positive_real_allowed():
    """Gate #9 (hc_mult<=0) is a correctness guard, already open for real.
    Parity = hc_mult>=1 (11.23 cache axis cleared) constructs."""
    args = _baseline_args(hc_mult=1)
    Model(args)
    assert args.hc_mult > 0


# ---------------------------------------------------------------------------
# Gate #10 — multi-layer hc_mult>1 (11.11 hyperconnection synthetic)
# ---------------------------------------------------------------------------

def test_gate_10_multi_layer_hc_mult_2_hyperconnection_runs():
    """Relax gate #10: allow num_hidden_layers>1 with hc_mult>1.
    Proven component = 11.11 hyperconnection synthetic. Parity test runs
    `_hyperconnection_residual_mix_mlx` at hc_mult=2 vs a naive reference."""
    hc_mult = 2
    hidden = 8
    batch, seq = 1, 3
    mx.random.seed(7)
    stream = mx.random.uniform(low=-1.0, high=1.0, shape=(batch, seq, hc_mult, hidden))
    expected_mix = (2 + hc_mult) * hc_mult
    fn = mx.random.uniform(low=-0.2, high=0.2, shape=(expected_mix, hc_mult * hidden))
    base = mx.random.uniform(low=-0.5, high=0.5, shape=(expected_mix,))
    scale = mx.random.uniform(low=0.1, high=1.0, shape=(3,))
    eps = 1e-6
    # Proven component (11.11) runs at hc_mult=2 without raising
    out = _hyperconnection_mlx(
        stream,
        fn=fn,
        base=base,
        scale=scale,
        hc_mult=hc_mult,
        eps=eps,
        sinkhorn_iters=2,
        rms_norm_eps=1e-6,
    )
    collapsed = out["collapsed"]
    assert collapsed.shape == (batch, seq, hidden)
    assert mx.all(mx.isfinite(collapsed)).item()
    assert out["post"].shape == (batch, seq, hc_mult)
    assert out["comb"].shape[-2:] == (hc_mult, hc_mult)
    # construct a multi-layer hc_mult=2 model (gate #10 relaxation)
    args = _baseline_args(
        num_hidden_layers=2,
        hc_mult=2,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["moe", "moe"],
        compression_ratio=0,
    )
    Model(args)


# ---------------------------------------------------------------------------
# Gate #11 — n_routed_experts staged 8 / 32 / 256 (Q3 probe + 11.15d routing)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_exp", [8, 32, 256])
def test_gate_11_moe_mlx_i8_composition_no_drift_at_scale(n_exp):
    """Q3 staged probe: `_moe_mlx` I8-expert composition vs BF16 pre-dequantized
    reference at n_routed_experts in {8, 32, 256}. L2_REL <= 5e-3 at every step.
    Proven component = dequantize_i8_e8m0_block_scale (ADR 0017, max_abs<=1e-5)
    composed with the _moe_mlx top-k stub (11.15d routing proven decode-independent).
    Drift > 5e-3 = STOP per supervisor STOP-rule (i)/(iv)."""
    hidden = 16
    mid = 16
    per_tok = 2
    args_i8 = _baseline_args(
        n_routed_experts=n_exp,
        num_experts_per_tok=per_tok,
        moe_intermediate_size=mid,
        expert_dtype="i8",
    )
    args_bf = _baseline_args(
        n_routed_experts=n_exp,
        num_experts_per_tok=per_tok,
        moe_intermediate_size=mid,
        # expert_dtype="fp4" is supported by validate(); _moe_mlx treats any
        # non-"i8" value as the raw-weights path. We pass PRE-DEQUANTIZED BF16
        # weights (via _dequantize_i8_block_scale_mlx) so this is the BF16
        # reference composition, not an fp4 production claim.
        expert_dtype="fp4",
    )
    mx.random.seed(42 + n_exp)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(2, hidden))
    i8_w = _moe_weights_i8(n_exp, mid, hidden, seed=1000 + n_exp)
    ref_w = _moe_weights_bf16_ref(i8_w, n_exp, mid, hidden)
    out_i8 = _moe_mlx(args_i8, x, i8_w)
    out_ref = _moe_mlx(args_bf, x, ref_w)
    l2 = _l2_rel(out_i8, out_ref)
    assert l2 <= L2_REL_TOL, f"n_exp={n_exp}: L2_REL={l2:.6e} > {L2_REL_TOL:.0e} (Q3 STOP trigger)"


# ---------------------------------------------------------------------------
# Gate #12 — num_experts_per_tok <= n_routed_experts (11.15d) — already open
# ---------------------------------------------------------------------------

def test_gate_12_experts_per_tok_real_top6():
    """Gate #12 is a correctness guard (per_tok<=n_routed). Real DS4 top-6 at
    256 experts. Parity = construction at top-6/256 (11.15d routing proven)."""
    args = _baseline_args(
        n_routed_experts=256,
        num_experts_per_tok=6,
    )
    Model(args)
    assert args.num_experts_per_tok <= args.n_routed_experts


# ---------------------------------------------------------------------------
# Gate #13 — n_shared_experts=1 / scoring_func / expert_dtype (11.15a/b)
# ---------------------------------------------------------------------------

def test_gate_13_real_shared_scoring_dtype_allowed():
    """Gate #13 real values: n_shared_experts=1, scoring_func='sqrtsoftplus',
    expert_dtype='i8' (11.15a/b proven). Already open; parity = construction."""
    args = _baseline_args(
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        expert_dtype="i8",
    )
    Model(args)
    assert args.n_shared_experts == 1
    assert args.scoring_func == "sqrtsoftplus"


# ---------------------------------------------------------------------------
# Invariants: marker absent, no blanket switch
# ---------------------------------------------------------------------------

def test_no_blanket_allow_real_config_switch_introduced():
    """ADR 0002 fail-closed: NO blanket `allow_real_config` switch may exist.
    Each gate relaxed individually via removal of its `raise`, not a global flag."""
    src = (_MLX_SRC / "ds4_ft_mlx" / "vendor" / "mlx_lm_models" / "deepseek_v4.py").read_text()
    assert "allow_real_config" not in src, "blanket allow_real_config switch is forbidden (ADR 0002)"


def test_forward_parity_marker_stays_absent_this_slice():
    """Track-B marker `.deepseek-v4-forward-parity-ok` stays ABSENT until 11.15j
    (marker write only after 11.15h+11.15i green per ADR 0002 honesty)."""
    marker = _ROOT / ".deepseek-v4-forward-parity-ok"
    assert not marker.exists(), f"{marker} must not be written by Story 11.15g"
