"""Story 13.1 — AC2 synthetic + AC4 real load + AC7 exhaustive key coverage.

Anti-circularity (ADR 0007 §4):  EXPECTED keys and DROPPED lists are
hand-written literals — NOT derived by calling remap logic on itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ds4_ft_mlx.shimmed_ckpt_key_remap import remap_shimmed_ckpt_keys

CKPT = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")

# --- helpers -----------------------------------------------------------------

def _mx_array(v):
    """Return an MLX array if available, else a list scalar fallback."""
    try:
        import mlx.core as mx
        return mx.array(v)
    except Exception:
        return v


# --- AC2: synthetic fixture (covers every cluster) --------------------------

SYNTHETIC_SHIMMED = {
    # §2.1 Cluster 1
    "embed.weight": _mx_array([1.0]),
    "head.weight": _mx_array([2.0]),
    "norm.weight": _mx_array([3.0]),
    "hc_head_base": _mx_array([4.0]),
    "hc_head_fn": _mx_array([5.0]),
    "hc_head_scale": _mx_array([6.0]),
    # §2.2 Cluster 2
    "layers.0.attn_norm.weight": _mx_array([7.0]),
    "layers.0.ffn_norm.weight": _mx_array([8.0]),
    "layers.0.attn.kv_norm.weight": _mx_array([9.0]),
    "layers.0.attn.q_norm.weight": _mx_array([10.0]),
    "layers.0.attn.wkv.weight": _mx_array([11.0]),
    "layers.0.attn.wq_a.weight": _mx_array([12.0]),
    "layers.0.attn.wq_b.weight": _mx_array([13.0]),
    "layers.0.attn.wo_a.weight": _mx_array([14.0]),
    "layers.0.attn.wo_b.weight": _mx_array([15.0]),
    "layers.0.attn.attn_sink": _mx_array([16.0]),
    # §2.3 Cluster 3
    "layers.0.ffn.shared_experts.w1.weight": _mx_array([17.0]),
    "layers.0.ffn.shared_experts.w2.weight": _mx_array([18.0]),
    "layers.0.ffn.shared_experts.w3.weight": _mx_array([19.0]),
    # §2.4 Cluster 4
    "layers.0.ffn.experts.0.w1.weight": _mx_array([20.0]),
    "layers.0.ffn.experts.0.w2.weight": _mx_array([21.0]),
    "layers.0.ffn.experts.0.w3.weight": _mx_array([22.0]),
    # §2.5 Cluster 5
    "layers.0.ffn.experts.0.w1.scale": _mx_array([23.0]),
    "layers.0.ffn.experts.0.w2.scale": _mx_array([24.0]),
    "layers.0.ffn.experts.0.w3.scale": _mx_array([25.0]),
    "layers.0.ffn.gate.weight": _mx_array([26.0]),
    "layers.0.ffn.gate.bias": _mx_array([27.0]),
    "layers.0.hc_attn_base": _mx_array([28.0]),
    "layers.0.hc_attn_fn": _mx_array([29.0]),
    "layers.0.hc_attn_scale": _mx_array([30.0]),
    "layers.0.hc_ffn_base": _mx_array([31.0]),
    "layers.0.hc_ffn_fn": _mx_array([32.0]),
    "layers.0.hc_ffn_scale": _mx_array([33.0]),
    # §2.6 Cluster 6 (DROP candidates)
    "layers.0.attn.wkv.scale": _mx_array([34.0]),
    "layers.0.attn.wq_a.scale": _mx_array([35.0]),
    "layers.0.attn.compressor.wkv.weight": _mx_array([36.0]),
    "layers.0.attn.indexer.wq_b.weight": _mx_array([37.0]),
    "layers.0.ffn.shared_experts.w1.scale": _mx_array([38.0]),
    "layers.0.ffn.gate.tid2eid": _mx_array([39.0]),
    "mtp.0.foo.weight": _mx_array([40.0]),
}

EXPECTED_RENAMED_KEYS = {
    "embed.weight",
    "lm_head.weight",
    "norm.weight",
    "hc_head.base",
    "hc_head.fn",
    "hc_head.scale",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.kv_norm.weight",
    "layers.0.q_norm.weight",
    "layers.0.kv_proj.weight",
    "layers.0.q_a_proj.weight",
    "layers.0.q_b_proj.weight",
    "layers.0.o_a_proj.weight",
    "layers.0.o_b_proj.weight",
    "layers.0.sinks",
    "layers.0.mlp.shared_experts.w1.weight",
    "layers.0.mlp.shared_experts.w2.weight",
    "layers.0.mlp.shared_experts.w3.weight",
    "layers.0.mlp.experts.0.w1.weight",
    "layers.0.mlp.experts.0.w2.weight",
    "layers.0.mlp.experts.0.w3.weight",
    "layers.0.mlp.experts.0.w1.scale",
    "layers.0.mlp.experts.0.w2.scale",
    "layers.0.mlp.experts.0.w3.scale",
    "layers.0.mlp.gate.weight",
    "layers.0.mlp.gate.e_score_correction_bias",
    "layers.0.attn_hc.base",
    "layers.0.attn_hc.fn",
    "layers.0.attn_hc.scale",
    "layers.0.ffn_hc.base",
    "layers.0.ffn_hc.fn",
    "layers.0.ffn_hc.scale",
}

EXPECTED_DROPPED = {
    "layers.0.attn.wkv.scale",
    "layers.0.attn.wq_a.scale",
    "layers.0.attn.compressor.wkv.weight",
    "layers.0.attn.indexer.wq_b.weight",
    "layers.0.ffn.shared_experts.w1.scale",
    "layers.0.ffn.gate.tid2eid",
    "mtp.0.foo.weight",
}


def test_remap_synthetic_all_clusters():
    mapping, dropped = remap_shimmed_ckpt_keys(SYNTHETIC_SHIMMED)
    assert set(mapping.keys()) == EXPECTED_RENAMED_KEYS
    assert len(mapping) == len(EXPECTED_RENAMED_KEYS)
    # Values pass through unchanged
    assert mapping["embed.weight"] == SYNTHETIC_SHIMMED["embed.weight"]
    assert mapping["layers.0.mlp.experts.0.w1.scale"] == SYNTHETIC_SHIMMED["layers.0.ffn.experts.0.w1.scale"]
    assert set(dropped) == EXPECTED_DROPPED


def test_remap_identity_passthrough_vendor_internal():
    # Already-vendor-internal keys (with layer prefix) must pass through
    vendor_internal = {
        "layers.0.q_a_proj.weight": _mx_array([1.0]),
        "layers.0.mlp.experts.5.w3.scale": _mx_array([2.0]),
        "embed.weight": _mx_array([3.0]),
        "hc_head.base": _mx_array([4.0]),
    }
    mapping, dropped = remap_shimmed_ckpt_keys(vendor_internal)
    assert set(mapping.keys()) == set(vendor_internal.keys())
    assert dropped == []


def test_remap_unknown_key_raises_keyerror():
    with pytest.raises(KeyError, match="layers.0.unmapped.foo"):
        remap_shimmed_ckpt_keys({"layers.0.unmapped.foo": _mx_array([1.0])})


# --- AC2 synthetic Model load (fp4) -----------------------------------------

def test_synthetic_model_load_weights_fp4():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

    n_layers = 2
    n_routed = 2
    args = ModelArgs(
        num_hidden_layers=n_layers,
        layer_types=["sliding_attention"] * n_layers,
        mlp_layer_types=["moe"] * n_layers,
        n_routed_experts=n_routed,
        num_experts_per_tok=2,
        expert_dtype="fp4",
    )
    model = Model(args)

    # Build a complete shimmed-key dict; pre-pass renames, canonical keeps prefix.
    raw = {"embed.weight": _mx_array([1.0])}
    for i in range(n_layers):
        raw.update({
            f"layers.{i}.attn_norm.weight": _mx_array([1.0]),
            f"layers.{i}.ffn_norm.weight": _mx_array([1.0]),
            f"layers.{i}.attn.kv_norm.weight": _mx_array([1.0]),
            f"layers.{i}.attn.q_norm.weight": _mx_array([1.0]),
            f"layers.{i}.attn.wkv.weight": _mx_array([1.0]),
            f"layers.{i}.attn.wq_a.weight": _mx_array([1.0]),
            f"layers.{i}.attn.wq_b.weight": _mx_array([1.0]),
            f"layers.{i}.attn.wo_a.weight": _mx_array([1.0]),
            f"layers.{i}.attn.wo_b.weight": _mx_array([1.0]),
            f"layers.{i}.attn.attn_sink": _mx_array([1.0]),
            f"layers.{i}.hc_attn_base": _mx_array([1.0]),
            f"layers.{i}.hc_attn_fn": _mx_array([1.0]),
            f"layers.{i}.hc_attn_scale": _mx_array([1.0]),
            f"layers.{i}.hc_ffn_base": _mx_array([1.0]),
            f"layers.{i}.hc_ffn_fn": _mx_array([1.0]),
            f"layers.{i}.hc_ffn_scale": _mx_array([1.0]),
            f"layers.{i}.ffn.shared_experts.w1.weight": _mx_array([1.0]),
            f"layers.{i}.ffn.shared_experts.w2.weight": _mx_array([1.0]),
            f"layers.{i}.ffn.shared_experts.w3.weight": _mx_array([1.0]),
            f"layers.{i}.ffn.gate.weight": _mx_array([1.0]),
        })
        for eid in range(n_routed):
            for proj in ("w1", "w2", "w3"):
                raw[f"layers.{i}.ffn.experts.{eid}.{proj}.weight"] = _mx_array([1.0])
                raw[f"layers.{i}.ffn.experts.{eid}.{proj}.scale"] = _mx_array([1.0])
        # gate.bias omitted — vendor zero-defaults e_score_correction_bias
    # optional_final + hc_head
    raw.update({
        "head.weight": _mx_array([1.0]),
        "norm.weight": _mx_array([1.0]),
        "hc_head_base": _mx_array([1.0]),
        "hc_head_fn": _mx_array([1.0]),
        "hc_head_scale": _mx_array([1.0]),
    })
    model.load_weights(raw, strict=True)
    params = model.parameters()
    assert params
    # Expert scale is present (fp4 required-set)
    assert any("experts.0.w1.scale" in k for k in params)


# --- AC7: exhaustive 69,187-key coverage ------------------------------------

def test_exhaustive_live_index_no_residual_unmapped():
    if not (CKPT / "model.safetensors.index.json").exists():
        pytest.skip("shimmed ckpt index absent")
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
    raw_keys = list(idx["weight_map"].keys())
    mapping, dropped = remap_shimmed_ckpt_keys(raw_keys)
    assert len(mapping) + len(dropped) == len(raw_keys) == 69187
    # No KeyError raised → every key mapped or dropped
    # Dropped keys are all from cluster-6 patterns
    for d in dropped:
        assert (
            d.startswith("mtp.")
            or "compressor." in d
            or "indexer." in d
            or ".attn.wkv.scale" in d
            or ".attn.wq_a.scale" in d
            or ".attn.wq_b.scale" in d
            or ".attn.wo_a.scale" in d
            or ".attn.wo_b.scale" in d
            or "shared_experts.w1.scale" in d
            or "shared_experts.w2.scale" in d
            or "shared_experts.w3.scale" in d
            or d.endswith(".gate.tid2eid")
        )


# --- AC4: real 163GB shimmed ckpt load (slow-gated) -------------------------

@pytest.mark.skipif(
    os.environ.get("DS4_RUN_SLOW_PARITY") != "1",
    reason="live ckpt 163GB mmap; set DS4_RUN_SLOW_PARITY=1",
)
def test_real_shimmed_ckpt_loads_via_mlx_lm():
    from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin
    install_deepseek_v4_plugin()
    from mlx_lm import load
    model, _tokenizer = load(str(CKPT))
    assert hasattr(model, "parameters")
    params = model.parameters()
    assert params
    # Forward crashes at _moe_mlx FP4 are EXPECTED (Story 13.2 owns)


# --- AC8 anti-regression ----------------------------------------------------

def test_remap_total_and_deterministic_on_synthetic():
    m1, d1 = remap_shimmed_ckpt_keys(SYNTHETIC_SHIMMED)
    m2, d2 = remap_shimmed_ckpt_keys(SYNTHETIC_SHIMMED)
    assert m1 == m2
    assert d1 == d2
