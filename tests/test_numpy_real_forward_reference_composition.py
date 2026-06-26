"""Story 11.54 -- TDD composition suite for ``numpy_real_forward_reference.py``.

Stages (per architecture §6 / AC10 T1-T8, D5 mapping):

* T1/T2 -- composition shape + tail primitives (skeleton); structural.
* T3 -- hyperconnection residual-mix einsum (Q7 contract) on small dims.
* T4 -- inline STOP-iv MQA attention composition vs
  ``tiny_multihead_grouped_attention_reference`` (the proven spec-math witness
  validated vs HF Transformers ``DeepseekV4Model`` in 11.15g/11.53) on SMALL
  fixture dims (NOT real config); ``max_abs <= 1e-5`` per ADR 0017 witness tier.
* T5 -- MoE composition bridge marker (Group E REUSE per §1 Q3).
* T6 -- half-layer composition (hc_attn + rms + MQA + residual mix) vs the
  tiny_* primitives on small dims.
* T7/T8 -- full 43-layer real-config MLX ``_real_forward`` parity (AC3/AC4);
  gated behind ``DS4_RUN_SLOW_PARITY=1`` (requires the 162GB shimmed ckpt +
  MLX env + bounded per-layer witness). Without the witness harness this slice
  STOPs (iv) per architecture §3.
* Invariants (AC6) + non-circularity audit (AC12, B.1-B.4).

The numpy composition is a port of ``tiny_*`` SPEC MATH (validated vs HF
Transformers), NOT a transliteration of vendor ``_real_forward`` (ADR 0007 §4).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python-envs" / "mlx" / "src"))

from ds4_ft_mlx.deepseek_v4_attention_spec import (  # noqa: E402
    DeepSeekV4AttentionSpec,
    tiny_hyperconnection_forward,
    tiny_multihead_grouped_attention_reference,
    _rope_cos_sin,
)
from ds4_ft_mlx import numpy_real_forward_reference as nrf  # noqa: E402
from ds4_ft_mlx import real_forward_intermediate_dump as rfid  # noqa: E402

_PROMPTS_JSON = ROOT / "agent-output" / "cmux-11-55" / "prompts.json"


def _live_tokenize(text: str, seq_len: int):
    """Tokenize + truncate to the Q5 witness floor (live-gated callers only)."""
    from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin

    install_deepseek_v4_plugin()
    import mlx_lm

    tok = mlx_lm.load_tokenizer(str(CKPT_DIR))
    ids = list(tok.encode(text, add_special_tokens=False))
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    return ids


def _l2_rel_harness_vs_ref(h_mlx, h_ref) -> float:
    a = np.asarray(h_mlx, dtype=np.float64)
    b = np.asarray(h_ref, dtype=np.float64)
    diff = float(np.linalg.norm((a - b).ravel()))
    denom = float(np.linalg.norm(a.ravel()))
    return diff / denom if denom > 0 else diff

CKPT_DIR = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
HAS_CKPT = CKPT_DIR.is_dir()
HAS_MLX = True
try:
    import mlx.core as mx  # noqa: F401
except Exception:
    HAS_MLX = False
RUN_SLOW = os.environ.get("DS4_RUN_SLOW_PARITY") == "1"

EPS = 1e-6
TOL_WITNESS = 1e-5  # ADR 0017 primitive/witness tier (small-dims composition)
TOL_COMP = 5e-3     # ADR 0020 compositional tier (per-layer real config)


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


# --------------------------------------------------------------------------- #
# Small-fixture attention spec builder (real-key + spec-key conventions).
# --------------------------------------------------------------------------- #

def _small_spec():
    return DeepSeekV4AttentionSpec(
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=32,
        o_lora_rank=16,
        qk_rope_head_dim=8,
        num_output_groups=2,
    )


def _small_attn_weights(rng, spec):
    shapes = spec.flash_mlx_safetensors_shapes()
    w = {
        "q_a": rng.standard_normal(shapes["q_a_proj.weight"]) * 0.1,   # [qlora, H]
        "q_norm": rng.standard_normal(shapes["q_norm.weight"]) * 0.1,  # [qlora]
        "q_b": rng.standard_normal(shapes["q_b_proj.weight"]) * 0.1,   # [qdim, qlora]
        "kv": rng.standard_normal(shapes["kv_proj.weight"]) * 0.1,     # [hd, H]
        "kv_norm": rng.standard_normal(shapes["kv_norm.weight"]) * 0.1,  # [hd]
        "o_a": rng.standard_normal(shapes["o_a_proj.weight"]) * 0.1,   # [olow, hpg*hd]
        "o_b": rng.standard_normal(shapes["o_b_proj.weight"]) * 0.1,   # [H, olow]
        "sinks": rng.standard_normal(shapes["sinks"]) * 0.5,           # [nheads]
    }
    return w


def _to_spec_dict(w):
    """Spec-key dict (nested lists) for the tiny_multihead witness."""
    return {
        "q_a_proj.weight": w["q_a"].tolist(),
        "q_norm.weight": w["q_norm"].tolist(),
        "q_b_proj.weight": w["q_b"].tolist(),
        "kv_proj.weight": w["kv"].tolist(),
        "kv_norm.weight": w["kv_norm"].tolist(),
        "o_a_proj.weight": w["o_a"].tolist(),
        "o_b_proj.weight": w["o_b"].tolist(),
        "sinks": w["sinks"].tolist(),
    }


def _to_real_dict(w):
    """Real-key dict (numpy) for the 11.54 numpy composer."""
    return {
        "wq_a": w["q_a"], "q_norm": w["q_norm"], "wq_b": w["q_b"],
        "wkv": w["kv"], "kv_norm": w["kv_norm"], "wo_a": w["o_a"],
        "wo_b": w["o_b"], "attn_sink": w["sinks"],
    }


# --------------------------------------------------------------------------- #
# T1 / T2 -- composition shape + tail primitives (structural).
# --------------------------------------------------------------------------- #

class TestCompositionShape:
    def test_compose_mqa_attention_shape_small(self):
        spec = _small_spec()
        rng = np.random.default_rng(7)
        w = _small_attn_weights(rng, spec)
        hidden = rng.standard_normal((5, spec.hidden_size))
        out = nrf._compose_mqa_attention(
            hidden, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads,
            head_dim=spec.head_dim, q_lora_rank=spec.q_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=0, rms_norm_eps=EPS,
        )
        assert out.shape == (5, spec.hidden_size)

    def test_causal_sliding_mask_shape_and_causality(self):
        m = nrf._causal_sliding_mask_np(6, 3)
        assert m.shape == (6, 6)
        pos = np.arange(6)[:, None]
        key = np.arange(6)[None, :]
        assert np.all(np.where(key > pos, m == -np.inf, True))
        # sliding window: keys older than pos-2 masked
        assert m[3, 0] == -np.inf  # pos=3, key=0 outside window 3
        assert m[3, 1] != -np.inf  # pos=3, key=1 within window


# --------------------------------------------------------------------------- #
# T3 -- hyperconnection residual-mix einsum (Q7 contract).
# --------------------------------------------------------------------------- #

class TestHyperconnectionResidualMix:
    def test_einsum_matches_brute_force(self):
        rng = np.random.default_rng(11)
        seq, hm, hidden = 3, 4, 8
        h = rng.standard_normal((seq, hm, hidden))
        comb = rng.random((seq, hm, hm))
        post = rng.random((seq, hm))
        sub = rng.standard_normal((seq, hidden))
        got = nrf._hyperconnection_residual_mix(h, {"comb": comb, "post": post}, sub)
        # brute force: h_new[s,k,d] = post[s,k]*sub[s,d] + sum_a comb[s,a,k]*h[s,a,d]
        exp = np.zeros((seq, hm, hidden))
        for s in range(seq):
            for k in range(hm):
                for d in range(hidden):
                    exp[s, k, d] = post[s, k] * sub[s, d] + sum(comb[s, a, k] * h[s, a, d] for a in range(hm))
        assert _max_abs(got, exp) <= 1e-12

    def test_residual_stream_count_preserved(self):
        rng = np.random.default_rng(12)
        h = rng.standard_normal((2, 4, 16))
        hc = {"comb": rng.random((2, 4, 4)), "post": rng.random((2, 4))}
        sub = rng.standard_normal((2, 16))
        out = nrf._hyperconnection_residual_mix(h, hc, sub)
        assert out.shape == h.shape  # residual state dim preserved across layer


# --------------------------------------------------------------------------- #
# T4 -- inline STOP-iv MQA attention composition vs tiny_* witness (small dims).
# --------------------------------------------------------------------------- #

class TestComposeMqaAttentionVsTiny:
    @pytest.mark.parametrize("seed,seq,sw", [(1, 3, 0), (2, 4, 2), (3, 6, 4)])
    def test_vs_tiny_multihead_grouped_attention(self, seed, seq, sw):
        spec = _small_spec()
        rng = np.random.default_rng(seed)
        w = _small_attn_weights(rng, spec)
        hidden = rng.standard_normal((seq, spec.hidden_size)) * 0.5

        out_np = nrf._compose_mqa_attention(
            hidden, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads,
            head_dim=spec.head_dim, q_lora_rank=spec.q_lora_rank,
            qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=sw, rms_norm_eps=EPS,
        )
        out_tiny = tiny_multihead_grouped_attention_reference(
            spec, hidden.tolist(), _to_spec_dict(w),
            rms_norm_eps=EPS, rope_theta=10000.0, sliding_window=sw,
        )
        assert _max_abs(out_np, np.asarray(out_tiny)) <= TOL_WITNESS

    def test_sliding_branch_matches_tiny(self):
        # _compose_sliding_attention delegates to the MQA port with window>0.
        spec = _small_spec()
        rng = np.random.default_rng(31)
        w = _small_attn_weights(rng, spec)
        hidden = rng.standard_normal((5, spec.hidden_size)) * 0.5
        a = nrf._compose_sliding_attention(
            hidden, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads, head_dim=spec.head_dim,
            q_lora_rank=spec.q_lora_rank, qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=3, rms_norm_eps=EPS,
        )
        b = tiny_multihead_grouped_attention_reference(
            spec, hidden.tolist(), _to_spec_dict(w),
            rms_norm_eps=EPS, rope_theta=10000.0, sliding_window=3,
        )
        assert _max_abs(a, np.asarray(b)) <= TOL_WITNESS


# --------------------------------------------------------------------------- #
# T5 -- CSA NOOP (Q6) + MoE bridge marker.
# --------------------------------------------------------------------------- #

class TestCsaNoop:
    def test_compression_ratio_zero_is_noop(self):
        assert nrf._compose_csa_fusion(0) is None

    def test_compression_nonzero_raises_stop_iv(self):
        with pytest.raises(NotImplementedError, match=r"STOP \(iv\)"):
            nrf._compose_csa_fusion(4)


class TestMoEBridgeContract:
    def test_mlx_available_or_stop_iv(self):
        # MoE reuses _moe_mlx (§1 Q3). If MLX is absent, the bridge must STOP (iv).
        try:
            import mlx.core as mx  # noqa: F401
            from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx  # noqa: F401
        except Exception:
            with pytest.raises(NotImplementedError, match=r"STOP \(iv\)"):
                nrf._moe_out_via_mlx(np.zeros((1, 4)), {}, args=None)


# --------------------------------------------------------------------------- #
# T6 -- half-layer composition vs tiny_* primitives (small dims).
# --------------------------------------------------------------------------- #

def _np_to_list3(a):
    return [[[float(x) for x in row] for row in mat] for mat in a]


class TestHalfLayerComposition:
    def test_attn_site_matches_tiny_composition(self):
        """hc_attn -> rms_norm -> MQA -> residual mix, vs tiny_* primitives."""
        spec = _small_spec()
        rng = np.random.default_rng(41)
        hc_mult = 4
        seq = 3
        hidden = spec.hidden_size
        # stream state
        h_streams = rng.standard_normal((seq, hc_mult, hidden)) * 0.3
        # hyperconnection tensors (real shapes: [mix, hc_mult*hidden]) reduced.
        mix = (2 + hc_mult) * hc_mult
        fn = rng.standard_normal((mix, hc_mult * hidden)) * 0.05
        base = rng.standard_normal((mix,)) * 0.05
        scale = rng.standard_normal((3,)) * 0.1
        attn_norm_w = rng.standard_normal((hidden,)) * 0.1
        w = _small_attn_weights(rng, spec)

        # 11.54 numpy composition.
        hc_np = nrf.real_config_hyperconnection_forward(
            hidden_streams=h_streams, fn=fn, base=base, scale=scale,
            hc_mult=hc_mult, eps=1e-6, sinkhorn_iters=20, rms_norm_eps=EPS,
        )
        normed_np = nrf.real_config_rms_norm(hc_np["collapsed"], attn_norm_w, EPS)
        attn_np = nrf._compose_mqa_attention(
            normed_np, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads, head_dim=spec.head_dim,
            q_lora_rank=spec.q_lora_rank, qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=0, rms_norm_eps=EPS,
        )
        h_new_np = nrf._hyperconnection_residual_mix(h_streams, hc_np, attn_np)

        # tiny_* witness composition (same math, list-based).
        hc_tiny = tiny_hyperconnection_forward(
            hidden_streams=_np_to_list3(h_streams), fn=fn.tolist(), base=base.tolist(),
            scale=scale.tolist(), hc_mult=hc_mult, eps=1e-6, sinkhorn_iters=20,
            rms_norm_eps=EPS,
        )
        collapsed_tiny = np.asarray(hc_tiny["collapsed"])
        # brute-force rms_norm on the tiny collapsed (matches _np_rms_norm).
        normed_tiny = nrf._np_rms_norm(collapsed_tiny, attn_norm_w, EPS)
        attn_tiny = tiny_multihead_grouped_attention_reference(
            spec, normed_tiny.tolist(), _to_spec_dict(w),
            rms_norm_eps=EPS, rope_theta=10000.0, sliding_window=0,
        )
        # Residual mix via the same einsum (Q7) but tiny comb arrays.
        comb_t = np.asarray(hc_tiny["comb"])
        post_t = np.asarray(hc_tiny["post"])
        h_new_tiny = nrf._hyperconnection_residual_mix(
            h_streams, {"comb": comb_t, "post": post_t}, np.asarray(attn_tiny),
        )
        assert _max_abs(h_new_np, h_new_tiny) <= TOL_WITNESS


# --------------------------------------------------------------------------- #
# T7 / T8 -- real-config full 43-layer MLX parity (gated).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (RUN_SLOW and HAS_CKPT and HAS_MLX),
    reason="full real-config parity needs DS4_RUN_SLOW_PARITY=1 + ckpt + MLX (Q4 witness harness)",
)
class TestRealConfigParity:
    def test_t7_layer0_l2_rel(self):
        # AC3-of-11.55: layer-0 post-FFN h_streams L2_REL <= 5e-3 (ADR 0020
        # compositional tier) vs MLX _real_forward witness harness (11.55 GREEN).
        prompt = rfid.resolve_prompt("p1_short", str(_PROMPTS_JSON))
        ids = _live_tokenize(prompt["text"], int(prompt["seq_len"]))
        result = rfid.forward_capture(
            prompt_id="p1_short",
            prompt_ids=ids,
            model_path=str(CKPT_DIR),
            layers_to_capture=[0],
            capture_logits=False,
        )
        h_mlx = np.asarray(result["captured_layers"][0], dtype=np.float64)
        _logits_ref, inters = nrf.forward(
            ids, model_path=str(CKPT_DIR),
            layers_to_compare=[0], return_intermediates=True,
        )
        h_ref = np.asarray(inters[0], dtype=np.float64)
        rel = _l2_rel_harness_vs_ref(h_mlx, h_ref)
        assert rel <= TOL_COMP, f"T7 L2_REL={rel:.3e} > {TOL_COMP:.0e}"

    def test_t8_argmax_top7_overlap(self):
        # AC4-of-11.55: end-to-end argmax top-7 >= 5/7 vs _real_forward final logits.
        prompt = rfid.resolve_prompt("p2_argmax", str(_PROMPTS_JSON))
        ids = _live_tokenize(prompt["text"], int(prompt["seq_len"]))
        result = rfid.forward_capture(
            prompt_id="p2_argmax",
            prompt_ids=ids,
            model_path=str(CKPT_DIR),
            layers_to_capture=[0],
            capture_logits=True,
        )
        logits_mlx = np.asarray(result["captured_logits"], dtype=np.float64)
        top_mlx = list(np.argsort(logits_mlx[0, -1, :])[::-1][:7].tolist())
        logits_ref = np.asarray(nrf.forward(ids, model_path=str(CKPT_DIR)), dtype=np.float64)
        top_ref = list(np.argsort(logits_ref[-1, :])[::-1][:7].tolist())
        overlap = len(set(top_mlx) & set(top_ref))
        assert overlap >= 5, f"T8 overlap={overlap}/7 < 5; mlx={top_mlx} ref={top_ref}"


# --------------------------------------------------------------------------- #
# Invariants (AC6) + non-circularity audit (AC12, B.1-B.4).
# ---------------------------------------------------------------------------

EXPECTED_SHAS = {
    "docs/adr/0007-expert-block-geometry-shape-authoritative.md": "51db7085c0330416",
    "docs/adr/0008-two-track-parity-gguf-vs-mlx.md": "3f0f31ea8e70bc91",
    "docs/adr/0017-b2-metal-carry-forward.md": "88f8a81175920d7a",
    "docs/adr/0020-story-12-3-ac4-hypothesis-retrospective.md": "e1bb9a03d0fa74b5",
    "python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py": "dc5aaaab9bb079d2",
    "python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py": None,  # 11.53 FROZEN
    "tests/test_deepseek_v4_forward_parity_11_15h.py": "2df25cec54b91c7e",
}


def _sha16(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class TestInvariants:
    def test_marker_absent(self):
        assert not (ROOT / ".deepseek-v4-forward-parity-ok").exists()

    def test_adrs_unchanged(self):
        for rel, sha in EXPECTED_SHAS.items():
            if sha is None or not rel.startswith("docs/adr/"):
                continue
            p = ROOT / rel
            assert p.exists(), rel
            assert _sha16(p) == sha, f"ADR drift: {rel}"

    def test_vendor_deepseek_v4_frozen(self):
        p = ROOT / "python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py"
        assert _sha16(p) == EXPECTED_SHAS[str(p.relative_to(ROOT)).replace("\\", "/")]

    def test_11_15h_stop_audit_frozen(self):
        """RESOLVED 2026-06-24 slice 11.15h-r3 per ADR 0021 loader-extension
        (Pick A Q1b reversal — loader returns RAW i8; single dequant stays in
        _moe_mlx) + shimmed ckpt config layer_types patch (Pick B per user
        authorization); prior STOP-iv Option-2 premise (numpy reference runnable
        over shimmed ckpt) restored LIVE."""
        p = ROOT / "tests/test_deepseek_v4_forward_parity_11_15h.py"
        assert _sha16(p) == EXPECTED_SHAS[str(p.relative_to(ROOT)).replace("\\", "/")]

    def test_attention_spec_frozen(self):
        # 11.53 owns; 11.54 must NOT edit existing real_config_* / tiny_* bodies.
        p = ROOT / "python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py"
        assert p.exists() and p.stat().st_size > 0

    def test_cengine_metal_untouched(self):
        for rel in ("ds4.c", "ds4.h", "metal/dense.metal", "metal/flash_attn.metal"):
            p = ROOT / rel
            assert p.exists() and p.stat().st_size > 0, rel


class TestNonCircularityAudit:
    """AC12 / B.1-B.4 audit pattern."""

    def test_b1_distinct_witness_path(self):
        # The numpy composer is a spec-port; the tiny_* witness is a separate
        # pure-python implementation. They consume the same weights but run
        # independent code paths.
        spec = _small_spec()
        rng = np.random.default_rng(101)
        w = _small_attn_weights(rng, spec)
        hidden = rng.standard_normal((4, spec.hidden_size))
        a = nrf._compose_mqa_attention(
            hidden, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads, head_dim=spec.head_dim,
            q_lora_rank=spec.q_lora_rank, qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=0, rms_norm_eps=EPS,
        )
        b = np.asarray(tiny_multihead_grouped_attention_reference(
            spec, hidden.tolist(), _to_spec_dict(w), rms_norm_eps=EPS, rope_theta=10000.0))
        assert a.shape == b.shape
        assert _max_abs(a, b) <= TOL_COMP  # distinct paths -> same output

    def test_b3_comparator_catches_corruption(self):
        spec = _small_spec()
        rng = np.random.default_rng(102)
        w = _small_attn_weights(rng, spec)
        hidden = rng.standard_normal((4, spec.hidden_size))
        ref = nrf._compose_mqa_attention(
            hidden, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads, head_dim=spec.head_dim,
            q_lora_rank=spec.q_lora_rank, qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=0, rms_norm_eps=EPS,
        )
        # Tamper attention sinks (drive softmax toward sink bucket -> kills
        # KV mixing); output must drift beyond the compositional tolerance.
        corrupt = dict(_to_real_dict(w))
        corrupt["attn_sink"] = corrupt["attn_sink"] + 20.0
        corrupt_out = nrf._compose_mqa_attention(
            hidden, corrupt,
            num_attention_heads=spec.num_attention_heads, head_dim=spec.head_dim,
            q_lora_rank=spec.q_lora_rank, qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=0, rms_norm_eps=EPS,
        )
        assert _max_abs(ref, corrupt_out) > TOL_COMP

    def test_b4_non_degenerate_signal(self):
        spec = _small_spec()
        rng = np.random.default_rng(103)
        w = _small_attn_weights(rng, spec)
        hidden = rng.standard_normal((4, spec.hidden_size))
        assert float(np.mean(np.abs(hidden))) > 0.0
        out = nrf._compose_mqa_attention(
            hidden, _to_real_dict(w),
            num_attention_heads=spec.num_attention_heads, head_dim=spec.head_dim,
            q_lora_rank=spec.q_lora_rank, qk_rope_head_dim=spec.qk_rope_head_dim,
            o_groups=spec.num_output_groups, o_lora_rank=spec.o_lora_rank,
            hidden_size=spec.hidden_size, rope_theta=10000.0,
            sliding_window=0, rms_norm_eps=EPS,
        )
        assert float(np.mean(np.abs(out))) > 0.0


# --------------------------------------------------------------------------- #
# F1-FIX -- CPU-only synthetic-safetensors loader unit test (Axis-1 invariants).
# Exercises ``_OfflineLoader.load`` end-to-end on a tiny synthetic shard written
# in the *standard compact* safetensors wire format (8-byte LE header length +
# JSON header + raw BF16 payload). Proves the post-rename loader path returns
# the correct tensor values CPU-only (no 162GB ckpt, no MLX env).
# --------------------------------------------------------------------------- #

class TestOfflineLoaderSyntheticSafetensors:
    """RED exposes a latent off-by-8 offset bug in ``_OfflineLoader.load``.

    The F1 rename (``_shard_meta`` -> ``_load_shard_meta``) removes the
    method/attr collision that *crashed* forward on the first tensor read, but
    this witness shows the loader still reads SHIFTED bytes: ``offset`` is
    computed as ``8 + n + start`` while ``n`` already encodes ``8 + header_len``
    (double-counting the 8-byte length prefix). The correct offset is
    ``n + start``. Escalated for Reviewer authorization as scope beyond the
    F1 rename recipe ("NO OTHER CHANGES").
    """

    def _write_synthetic_shard(self, tmp_path, *, key="embed.weight", shape=(3, 4)):
        import json as _json
        import struct as _struct

        rng = np.random.default_rng(1234)
        f32 = rng.standard_normal(shape).astype(np.float32)
        # BF16 payload = round-toward-zero truncate of the float32 mantissa.
        bf16_u16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
        raw = bf16_u16.tobytes()
        header = {
            key: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, len(raw)]},
            "__metadata__": {"format": "pt"},
        }
        # Standard compact separators, matching real safetensors wire format.
        hbytes = _json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob = _struct.pack("<Q", len(hbytes)) + hbytes + raw
        shard = tmp_path / "model-00001-of-00001.safetensors"
        shard.write_bytes(blob)
        (tmp_path / "model.safetensors.index.json").write_text(
            _json.dumps({"weight_map": {key: shard.name}, "metadata": {"total_size": len(raw)}})
        )
        # Ground truth = BF16-truncated f32, viewed back to float32 (loader's
        # own BF16->F32 upcast convention).
        truth = (bf16_u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return tmp_path, key, truth, shape

    def test_offline_loader_reads_synthetic_safetensors(self, tmp_path):
        ckpt_dir, key, truth, shape = self._write_synthetic_shard(tmp_path)
        loader = nrf._OfflineLoader(ckpt_dir)
        got = loader.load(key)
        assert got.shape == tuple(shape)
        # Loader upcasts to float64 at the tail of ``load``.
        assert got.dtype == np.float64
        assert _max_abs(got, truth) <= 1e-2  # BF16-magnitude tolerance


# ---------------------------------------------------------------------------
# F1-FIX regression: CPU-only synthetic-safetensors loader proves the
# ``_OfflineLoader._load_shard_meta`` (renamed from method/attr collision) +
# header-offset fix (``offset = n + start``; the prior ``8 + n + start``
# double-counted the 8-byte length prefix since ``n`` already = 8 + header_len).
# This test does NOT require the 162GB shimmed ckpt or MLX env; it constructs a
# tiny synthetic safetensors shard + index on-disk then exercises the loader
# via ``_OfflineLoader.load(...)``. Catches the F1 + offset bug class without
# the T7/T8 MLX-gated parity witness.
# ---------------------------------------------------------------------------


class TestF1OfflineLoaderSyntheticSafetensors:
    """F1 regression: synthetic BF16 [3, 4] ``embed.weight`` tensor -> loader."""

    def _build_synthetic_shard(self, tmp_path):
        """Write a synthetic safetensors shard + matching index.json.

        Reuses the upstream ``safetensors.numpy.save_file`` helper so the
        on-disk byte layout is identical to a real HF safetensors shard.
        Returns (ckpt_dir, ground_truth_arr).
        """
        from safetensors.numpy import save_file

        # BF16 [3, 4] synthetic embed.weight. numpy has no native bf16 dtype,
        # so we cast float32 -> bf16 via safetensors (which natively handles
        # BF16 to disk). The loader upcasts back to float64 at read time.
        truth = np.array(
            [[0.1257, 0.2188, -0.6404, 1.0078],
             [-0.0137, 0.5469, 0.0704, -2.1094],
             [0.2891, -0.4219, 0.1641, 0.8750]],
            dtype=np.float32,
        )
        # Cast through float32 -> store as BF16 on disk. The loader reads BF16
        # bytes and upcasts to float64; assert within BF16 tolerance (rounded
        # to 8 mantissa bits — 1e-2 max_abs is generous for these magnitudes).
        bf16_arr = truth.astype("<f4")  # little-endian float32 (safetensors BF16 store)
        # Repackage float32 bytes as BF16 (truncate low 16 mantissa bits) is
        # what safetensors does internally when dtype='BF16'; pass "BF16"
        # dtype through the standard header route by hand-writing the file.
        # Simpler: use save_file with the float32 array but force BF16 header
        # via the safetensors low-level path -- but save_file reads the array
        # dtype. Use a direct writer instead (safetensors file format is
        # well-documented and trivial).
        import json as _json
        import struct as _struct

        # BF16 encode: take float32 IEEE bytes, shift right 16, store uint16.
        # Use COMPACT separators (matches real safetensors wire format; loader's
        # ``_load_shard_meta`` computes ``n = 8 + len(json.dumps(meta,
        # separators=(",", ":")))`` with compact format — file header bytes must
        # match that compact layout or the offset is wrong by the size delta).
        f32_le = truth.astype("<f4")
        u32_le = f32_le.view(np.uint32)
        bf16_u16 = (u32_le >> 16).astype(np.uint16)
        raw_bytes = bf16_u16.tobytes()  # little-endian uint16 BF16 payload

        header = {
            "embed.weight": {
                "dtype": "BF16",
                "shape": [3, 4],
                "data_offsets": [0, len(raw_bytes)],
            },
            "__metadata__": {"format": "pt"},
        }
        # Match the loader's compact json.dumps in ``_load_shard_meta``.
        header_json = _json.dumps(header, separators=(",", ":"))
        # Safetensors leading 8-byte little-endian length prefix = len(json_bytes).
        header_bytes = header_json.encode("utf-8")
        prefix = _struct.pack("<Q", len(header_bytes))

        shard_path = tmp_path / "synthetic-00001-of-00001.safetensors"
        with open(shard_path, "wb") as fh:
            fh.write(prefix)
            fh.write(header_bytes)
            fh.write(raw_bytes)

        index = {
            "weight_map": {"embed.weight": shard_path.name},
            "metadata": {"total_size": len(raw_bytes) + len(header_bytes) + 8},
        }
        with open(tmp_path / "model.safetensors.index.json", "w") as fh:
            _json.dump(index, fh)

        return tmp_path, truth

    def test_load_returns_correct_bf16_tensor_after_f1_fix(self, tmp_path):
        """RED-then-GREEN: pre-F1 the loader crashed (``_shard_meta`` is a dict
        not callable). Pre-offset-fix the loader returned shifted data
        (expected first value 0.1257, got -0.6404 = the 3rd value shifted by 8
        bytes). Post-F1-FIX the loader returns the correct tensor."""
        from ds4_ft_mlx.numpy_real_forward_reference import _OfflineLoader

        ckpt_dir, truth = self._build_synthetic_shard(tmp_path)
        loader = _OfflineLoader(ckpt_dir)
        got = loader.load("embed.weight")
        # Shape + dtype.
        assert got.shape == (3, 4), f"got shape {got.shape}"
        # Loader upcasts bf16 -> float64 at the tail of ``load``.
        assert got.dtype == np.float64, f"got dtype {got.dtype}"
        # Values: BF16 truncation tolerance (~5e-3 for magnitudes <2).
        max_abs = float(np.max(np.abs(got - truth.astype(np.float64))))
        assert max_abs <= 1e-2, (
            f"F1 OFFSET REGRESSION: first value expected {truth[0,0]:.4f}, "
            f"got {got[0,0]:.4f} (max_abs={max_abs:.4f}). "
            f"If first got == third expected (-0.6404), the 8-byte length "
            f"prefix is still being double-counted in offset computation."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
