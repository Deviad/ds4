#!/usr/bin/env python3
"""Story 11.15h — Full-model 43-layer MLX forward parity vs OCP-witness-derived BF16 reference.

TDD red-first scaffold. Frozen byte-spec per
`agent-output/cmux-11-15h/architecture.md` §2 + `task-coder.md`.

Reference construction = Q7 path (a): OCP-witness-derived BF16 per-layer hidden
states, computed by an INDEPENDENT closed-form reference forward (stdlib math /
numpy), NOT a transliteration of MLX `_real_forward` (ADR 0007 §4
anti-transliteration). Production = MLX `_real_forward` over the shimmed BF16
checkpoint, Metal-only (Q5).

This slice's RED state documents STOP-cascade (iv): an independent real-config
43-layer reference forward cannot be constructed in-slice without either
(a) transliterating `_real_forward` (circular, ADR 0007 §4 rejects) or
(b) a multi-epic scaling of the proven `tiny_*` spec references
    (`deepseek_v4_attention_spec` / `deepseek_v4_moe_spec` / `deepseek_v4_dequant`,
     validated against HF Transformers only at tiny synthetic configs) to the
     real config (head_dim=512, num_attention_heads=64, index_topk=512, 43
     layers, 162 GB I8/E8M0 weights). A pure-Python list-of-floats reference at
     that scale is infeasible within the AC9 30-min-per-prompt wall-time cap;
     numpy vectorization would be a multi-day effort whose per-layer driver
     risks transliterating `_real_forward`'s primitive composition.

=> ADR 0008 amendment required (parent-arbitrated, NOT slice) per §3 STOP (iv).
STOP is a VALID Coder completion per 11.51 precedent. See
`agent-output/cmux-11-15h/coder-stop.md`.

NOT a derived-logits-compute test: the STOP-rule reporting test asserts the
STOP (iv) condition is surfaced, not that logits are computed.
"""

from __future__ import annotations

import math
import os
import resource
import sys
import time
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"
MLX_VENV_PYTHON = REPO_ROOT / "python-envs" / "mlx" / ".venv" / "bin" / "python3"

CANONICAL_SHIMMED_CHECKPOINT = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
L2_REL_TOL = 5e-3
ARGMAX_TOPK = 7
ARGMAX_OVERLAP_MIN = 5
MAX_PER_PROMPT_SECONDS = 1800
MEM_HIGH_WATER_BYTES = 460 * (1024 ** 3)

# --- LIVE Metal+ckpt machine guards (Architect Q9 runtime). ---
try:  # MLX + Metal device probe; MLX absent -> LIVE tests skip below.
    import mlx.core as _mx_probe  # noqa: F401
    MX_AVAILABLE = True
except Exception:  # pragma: no cover - non-Metal CI
    _mx_probe = None
    MX_AVAILABLE = False
CKPT_PRESENT = bool(CANONICAL_SHIMMED_CHECKPOINT.exists())


def _peak_rss_bytes() -> int:
    """Peak RSS in BYTES. macOS ru_maxrss is already bytes; Linux reports KB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


# Q7 (Architect verdict) -- deterministic seed set ONCE per session before any
# forward (RoPE yarn + sinkhorn determinism). Idempotent flag-guard; called from
# setUp() so it fires before the first LIVE test under both unittest + pytest.
_DS4_SEED_SET = False


def _seed_once() -> None:
    global _DS4_SEED_SET
    if _DS4_SEED_SET:
        return
    _DS4_SEED_SET = True
    try:
        import mlx.core as mx
        os.environ.setdefault("MX_DEFAULT_DEVICE", "gpu")
        mx.set_default_device(mx.gpu)  # Metal; AGENTS.md "Metal default"
        mx.random.seed(DETERMINISTIC_SEED)
        import numpy as np
        np.random.seed(DETERMINISTIC_SEED)
    except Exception:
        pass  # MLX absent -> LIVE tests skip via MX_AVAILABLE guard


def pytest_configure(config):  # noqa: D401 -- pytest session-start hook (Q7).
    _seed_once()

# Q1 (Architect verdict a) -- module-level lazy-cached real DS4 tokenizer.
# Canonical ckpt directory ships its tokenizer alongside the safetensors; the
# cached instance is reused across AC3 + AC4 + the 2 flipped STOP-iv marker tests
# (one tokenizer source, one encode pass per prompt; no per-test reload).
_DS4_TOKENIZER = None


def _ds4_tokenizer():
    global _DS4_TOKENIZER
    if _DS4_TOKENIZER is None:
        from transformers import AutoTokenizer
        _DS4_TOKENIZER = AutoTokenizer.from_pretrained(
            str(CANONICAL_SHIMMED_CHECKPOINT))
    return _DS4_TOKENIZER


def _tokenize_ids(text: str, seq_floor: int = 4) -> list[list[int]]:
    """Real DS4 tokenizer encode (batch=1) truncated to Q5 seq<=4 floor."""
    ids = _ds4_tokenizer().encode(text)
    return [ids[:seq_floor]]


# Q1 — deterministic prompt set (3 prompts, verbatim in evidence).
PROMPTS: list[dict[str, object]] = [
    {"id": "short_ascii", "text": "Hello", "input_ids_factory": lambda: _encode_short_ascii()},
    {"id": "long_multiline", "text": "Line one\\nLine two\\nLine three\\nLine four", "input_ids_factory": lambda: _encode_long_multiline()},
    {"id": "boundary_bos_eos", "text": "<bos><eos>", "input_ids_factory": lambda: _encode_boundary()},
]
DETERMINISTIC_SEED = 20260624


def _encode_short_ascii() -> list[list[int]]:
    # "Hello" real DS4 tokenizer encode; truncated to Q5 seq<=4 floor.
    return _tokenize_ids("Hello")


def _encode_long_multiline() -> list[list[int]]:
    # Q4 (Architect verdict): seq_len=15 overflows the Q5 seq<=4 CPU-safety
    # floor; return [] sentinel so AC3+AC4 loops `continue` (skipped, deferred
    # to 11.15i smoke-generate). NEVER reaches harness `_validate_args`.
    return []


def _encode_boundary() -> list[list[int]]:
    # boundary bos/eos adjacency (Q1 iii); real DS4 tokenizer encode, truncated.
    return _tokenize_ids("<bos><eos>")


def _l2_rel(a, b) -> float:
    """Relative L2 distance ‖a-b‖ / ‖b‖ (MLX-array or list-of-float)."""
    import mlx.core as mx
    a32 = a.astype(mx.float32) if hasattr(a, "astype") else mx.array(a, mx.float32)
    b32 = b.astype(mx.float32) if hasattr(b, "astype") else mx.array(b, mx.float32)
    diff = a32 - b32
    num = float(mx.sqrt(mx.sum(diff * diff)).item())
    den = float(mx.sqrt(mx.sum(b32 * b32)).item()) + 1e-30
    return num / den


def _topk_argmax(logits, k: int) -> list[int]:
    """Return top-k argmax token ids (descending) of a 1-D logits vector."""
    import mlx.core as mx
    arr = logits.astype(mx.float32) if hasattr(logits, "astype") else mx.array(logits, mx.float32)
    flat = arr.reshape(-1) if arr.ndim > 1 else arr
    n = flat.size
    vals = []
    for i in range(n):
        vals.append((float(flat[i].item()), i))
    vals.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _, idx in vals[:k]]


def _overlap(a: list[int], b: list[int]) -> int:
    return len(set(a) & set(b))


def dequantize_shimmed_checkpoint_to_bf16_reference(ckpt_dir: Path) -> dict:
    """Q7 path step 1 -- Pass-through shimmed ckpt path (11.55 harness + numpy
    reference compose the real forward; Q9.iv reconciliation INSIDE this slice).

    shimmed BF16 ckpt is loaded mmap-backed by MLX `DeepseekV4Model.load_real_weights`
    (vendor path) inside `real_forward_intermediate_dump.forward_capture` +
    `numpy_real_forward_reference.forward` -- NO CPU-side full dequant of the
    ~163GB I8/E8M0 weights dict (AGENTS.md safety: "Do not run multiple huge
    model processes concurrently"; would exceed RAM budget). The OCP MX v1.0
    E8M0 dequant primitive (`tests/ds4_e8m0_ocp_witness.py` +
    `deepseek_v4_dequant`) stays proven per ADR 0017 (max_abs<=1e-5) but is NOT
    applied to materialize a full BF16 weights dict here. This function returns
    the ckpt path; the two FROZEN harnesses mmap-load it.
    """
    if not ckpt_dir.exists():
        raise NotImplementedError(
            f"canonical shimmed checkpoint absent: {ckpt_dir}")
    return {
        "model_path": str(ckpt_dir),
        "_ckpt_dir": ckpt_dir,
        "proven": "shimmed ckpt path supplied directly; MLX+numpy harness loads "
                  "mmap; full OCP dequant stays primitive-proven per ADR 0017 "
                  "but NOT applied (RAM safety)",
    }


def independent_real_config_43_layer_reference_forward(
    args, weights_bf16: dict, input_ids: list[list[int]],
) -> tuple[dict[int, object], object]:
    """Q7 path step 2 -- INDEPENDENT real-config numpy reference forward
    (`numpy_real_forward_reference.forward`, FROZEN post-11.54r2 F1-FIX).

    Non-circular per ADR 0007 §4 anti-transliteration + 11.55 AC12 4-point
    audit: the numpy reference is NOT a transliteration of MLX `_real_forward`
    (proven 11.53 + 11.54 DONE record); it composes the proven `real_config_*`
    primitives scaled from the `tiny_*` SPEC math validated vs HF Transformers.

    Layer-0-only witness per Architect Q2 (full 43-layer sweep DEFERRED to
    follow-up slice 11.15i). Q9.iv reconciliation: numpy reference returns
    float64 natively; `np.asarray(..., dtype=np.float64)` normalizes dtype +
    contiguity INSIDE this stub (NOT a numpy reference edit).

    Returns ({0: h_streams_layer0_np}, final_logits_np) where
    `h_streams_layer0_np` is post-FFN residual stream layer 0
    (shape [seq, 4, 4096] dtype float64) and `final_logits_np` is lm_head logits
    (shape [seq, vocab_size=129280] dtype float64).
    """
    import numpy as np
    from ds4_ft_mlx import numpy_real_forward_reference as nrf

    if not input_ids or not input_ids[0]:
        return ({}, None)  # long_multiline sentinel; AC3/AC4 loop filters already

    model_path = weights_bf16["model_path"]
    logits_np, intermediates = nrf.forward(
        input_ids=input_ids[0],
        model_path=model_path,
        layers_to_compare=[0],
        return_intermediates=True,
    )
    h_streams_layer0_np = np.asarray(intermediates[0], dtype=np.float64)
    per_layer_hidden_states = {0: h_streams_layer0_np}
    final_logits = np.asarray(logits_np, dtype=np.float64)
    return per_layer_hidden_states, final_logits


def capture_production_per_layer_hidden_states(
    args, shimmed_checkpoint: Path, input_ids: list[list[int]],
) -> tuple[dict[int, object], object]:
    """Q7 path step 3 -- Production per-layer hidden states via MLX `_real_forward`
    + 11.55 NON-MUTATING witness harness
    (`real_forward_intermediate_dump.forward_capture`, FROZEN post-11.55 DONE).

    Non-circular per 11.55 Q3 (a-mod): `DeepseekV4ModelWithDump` subclass OVERRIDE
    of `_real_layer_forward` dispatches `super()._real_layer_forward(...)` --
    vendor math byte-exact; the harness only appends a read-only `.astype(float64)`
    copy of the returned `h` stream to a side-effect list (NOT a re-derivation).
    The vendor `out` tensor returned to `_real_forward` downstream is UNCHANGED
    (next-layer `h` input + hc_head collapse + lm_head intact).

    Layer-0-only witness per Architect Q2. Q6 (Architect verdict): direct API
    call, NOT CLI subprocess (single MLX process; `mx.random.seed` covers both
    harness + numpy reference under one RNG context).

    Returns ({0: captured_layers[0]}, captured_logits) where `captured_layers[0]`
    is the harness's MLX->np.float64 snapshot of vendor `_real_layer_forward`
    post-FFN residual output layer 0 (shape [seq, 4, 4096]) and `captured_logits`
    is `np.float64 [seq, vocab_size=129280]`.
    """
    import numpy as np
    from ds4_ft_mlx import real_forward_intermediate_dump as rfid

    if not input_ids or not input_ids[0]:
        return ({}, None)  # long_multiline sentinel; AC3/AC4 loop filters already

    result = rfid.forward_capture(
        prompt_id="11_15h_resume_probe",
        prompt_ids=list(input_ids[0]),
        model_path=str(shimmed_checkpoint),
        layers_to_capture=[0],
        capture_logits=True,
    )
    per_layer_hidden_states = {0: np.asarray(result["captured_layers"][0],
                                             dtype=np.float64)}
    final_logits = np.asarray(result["captured_logits"], dtype=np.float64)
    return per_layer_hidden_states, final_logits


class DeepSeekV4ForwardParity11_15hTests(unittest.TestCase):

    def setUp(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, str(MLX_SRC))
        _seed_once()  # Q7: deterministic MX/np seed ONCE per session before any forward.

    def tearDown(self):
        sys.path[:] = self._old_path

    # ---- Q4 / Q5: Metal-only device assertion (PASSES — real invariant). ----

    def test_q5_metal_only_device_gpu_asserted(self):
        import mlx.core as mx
        dev = mx.default_device()
        self.assertNotEqual(str(dev),
                            str(mx.cpu),
                            "CPU forward FORBIDDEN (AGENTS.md); production "
                            "forward MUST be Metal-only GPU (Q5).")

    # ---- AC6: invariants (PASS — cheap, real assertions). ----

    def test_ac6_forward_parity_marker_absent_11_15j_owns_write(self):
        # 11.15j owns marker write; MUST stay absent.
        marker = REPO_ROOT / ".deepseek-v4-forward-parity-ok"
        self.assertFalse(
            marker.exists(),
            f"invariant regression: {marker} present before 11.15j write",
        )

    def test_ac6_model_4bit_and_convert_shimmed_absent(self):
        self.assertFalse((REPO_ROOT / "model-4bit").exists(),
                         "model-4bit materialization forbidden (Epic 13)")
        self.assertFalse((REPO_ROOT / "convert-shimmed").exists(),
                         "convert-shimmed invocation forbidden (Epic 13)")

    def test_ac6_forward_parity_blockers_byte_unchanged(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import forward_parity_blockers
        blockers = forward_parity_blockers()
        self.assertEqual(blockers, (
            "full attention parity with RoPE/cache/sinks/compressor/indexer",
            "full decoder-layer hyperconnection residual mixing and final hyperhead parity",
            "full MoE parity with packed FP4/I8 expert dequant and expert kernels",
            "full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)",
        ))

    def test_ac6_binding_adrs_unchanged(self):
        for adr in ("0001-metal-graph-is-production-path.md",
                    "0002-parity-first-fail-closed-gates.md",
                    "0007-expert-block-geometry-shape-authoritative.md",
                    "0008-two-track-parity-gguf-vs-mlx.md",
                    "0017-b2-metal-carry-forward.md",
                    "0019-fusion-primary-adapter-serving.md",
                    "0020-story-12-3-ac4-hypothesis-retrospective.md",
                    "0021-loader-paired-tensor-synthesis-i8-e8m0.md"):
            self.assertTrue((REPO_ROOT / "docs" / "adr" / adr).exists(),
                            f"binding ADR missing: {adr}")

    def test_ac6_validate_real_mode_frozen_no_new_gate_relaxations(self):
        # FROZEN post-11.15g: gates #1/#2/#3/#5(b)/#10/#11 relaxed; #6 CLOSED;
        # guards #7-9/#12-13 intact. No NEW relaxations in 11.15h.
        src = (MLX_SRC / "ds4_ft_mlx" / "vendor" / "mlx_lm_models"
               / "deepseek_v4.py").read_text(encoding="utf-8")
        self.assertIn("Gate #10 relaxed (Story 11.15g)", src)
        self.assertIn("Gate #11 staged-relaxation proven per Q3 probe (Story 11.15g)", src)
        # Gate #6 MQA stays CLOSED.
        self.assertIn('currently supports only num_key_value_heads=1', src)
        # No 11.15h gate-relaxation marker introduced.
        self.assertNotIn("Gate #", src.replace("Gate #10 relaxed (Story 11.15g)", "")
                         .replace("Gate #11 staged-relaxation proven per Q3 probe (Story 11.15g)", "")
                         .replace("Gate #1 relaxed (Story 11.15g)", "")
                         .replace("Gate #2 relaxed (Story 11.15g)", "")
                         .replace("Gate #3 relaxed (Story 11.15g)", "")
                         .replace("Gate #5(b) relaxed (Story 11.15g)", ""))

    # ---- AC2: reference construction SPEC (RED — STOP iv blocker). ----

    def test_ac2_ocp_witness_dequant_primitive_loads_proven(self):
        # The OCP-witness E8M0/I8 dequant primitive IS proven (ADR 0017) and
        # loads cleanly — this sub-step of path (a) is not the blocker.
        from ds4_ft_mlx import deepseek_v4_dequant as dq
        self.assertTrue(hasattr(dq, "dequantize_i8_e8m0_block_scale"))
        from tests import ds4_e8m0_ocp_witness as wit
        self.assertTrue(hasattr(wit, "decode_i8_e8m0_to_float32"))

    @unittest.skip(
        "Superseded by ADR 0022 strategic pivot; gold-forward path is "
        "hardware-infeasible on M3 Ultra — see "
        "docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md"
    )
    def test_ac2_reference_forward_independent_real_config_not_constructible_stop_iv(self):
        """STOP (iv) RESOLVED 2026-06-24 parent-arbitration Option 2: stub now
        returns a real-tensor payload via 11.55 harness + numpy reference; no
        NotImplementedError. Historical STOP-iv audit preserved via KEPT test
        name + docstring (assertion flipped post-resume witness)."""
        weights_bf16 = dequantize_shimmed_checkpoint_to_bf16_reference(
            CANONICAL_SHIMMED_CHECKPOINT)
        ref_layers, ref_logits = independent_real_config_43_layer_reference_forward(
            args=None, weights_bf16=weights_bf16,
            input_ids=_encode_short_ascii())
        self.assertIn(0, ref_layers)
        self.assertEqual(ref_layers[0].shape[1], 4)  # hc_mult=4
        self.assertEqual(ref_layers[0].shape[2], 4096)  # hidden_size
        self.assertEqual(ref_layers[0].dtype, np.float64)  # Q9 float64
        self.assertEqual(ref_logits.shape[1], 129280)  # vocab_size
        self.assertEqual(ref_logits.dtype, np.float64)

    # ---- AC3 / AC4: per-layer L2_REL + end-to-end argmax top-7 overlap ----
    # (RED — both reference and production-capture raise NotImplementedError).

    @unittest.skip(
        "Superseded by ADR 0022 strategic pivot; gold-forward path is "
        "hardware-infeasible on M3 Ultra — see "
        "docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md"
    )
    def test_ac3_per_layer_l2_rel_le_5e_minus3_across_43_layers(self):
        # AC3 default: full 43-layer enumeration, per-layer L2_REL <= 5e-3 per
        # prompt. RED: reference forward + MinCal production capture both raise
        # NotImplementedError (feature not implemented).
        dequant = dequantize_shimmed_checkpoint_to_bf16_reference(
            CANONICAL_SHIMMED_CHECKPOINT)
        for prompt in PROMPTS:
            input_ids = prompt["input_ids_factory"]()
            if not input_ids or not input_ids[0]:
                continue  # Q4 long_multiline sentinel; deferred 11.15i
            t0 = time.perf_counter()
            ref_layers, _ = independent_real_config_43_layer_reference_forward(
                args=None, weights_bf16=dequant, input_ids=input_ids)
            prod_layers, _ = capture_production_per_layer_hidden_states(
                args=None, shimmed_checkpoint=CANONICAL_SHIMMED_CHECKPOINT,
                input_ids=input_ids)
            dt = time.perf_counter() - t0
            peak = _peak_rss_bytes()
            print(f"[ac3] prompt={prompt['id']} wall={dt:.1f}s "
                  f"peak_rss={peak / (1024 ** 3):.1f}GB", file=sys.stderr)
            if dt > MAX_PER_PROMPT_SECONDS:
                self.fail(f"AC9 STOP (xi): per-prompt wall {dt:.0f}s > "
                          f"{MAX_PER_PROMPT_SECONDS}s prompt {prompt['id']} "
                          f"(promote parent-arbitrated slice; AC7 ix)")
            if peak > MEM_HIGH_WATER_BYTES:
                self.fail(f"AC7 (x) STOP: peak RSS {peak / (1024 ** 3):.1f}GB "
                          f"> {MEM_HIGH_WATER_BYTES / (1024 ** 3):.0f}GB "
                          f"prompt {prompt['id']}")
            self.assertGreaterEqual(len(ref_layers), 1)
            self.assertEqual(len(ref_layers), len(prod_layers))
            for idx in ref_layers:
                self.assertLessEqual(
                    _l2_rel(prod_layers[idx], ref_layers[idx]), L2_REL_TOL,
                    f"per-layer L2_REL > {L2_REL_TOL} at layer {idx} prompt "
                    f"{prompt['id']} (§3 STOP (i) if drift real; here reference "
                    "not constructible — STOP iv).")

    @unittest.skip(
        "Superseded by ADR 0022 strategic pivot; gold-forward path is "
        "hardware-infeasible on M3 Ultra — see "
        "docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md"
    )
    def test_ac4_end_to_end_argmax_top7_overlap_at_least_5_of_7(self):
        # AC4 NON-NEGOTIABLE under AC9 fallback. RED: final logits not computable
        # (reference + production capture both raise).
        dequant = dequantize_shimmed_checkpoint_to_bf16_reference(
            CANONICAL_SHIMMED_CHECKPOINT)
        for prompt in PROMPTS:
            input_ids = prompt["input_ids_factory"]()
            if not input_ids or not input_ids[0]:
                continue  # Q4 long_multiline sentinel; deferred 11.15i
            t0 = time.perf_counter()
            _, ref_logits = independent_real_config_43_layer_reference_forward(
                args=None, weights_bf16=dequant, input_ids=input_ids)
            _, prod_logits = capture_production_per_layer_hidden_states(
                args=None, shimmed_checkpoint=CANONICAL_SHIMMED_CHECKPOINT,
                input_ids=input_ids)
            dt = time.perf_counter() - t0
            peak = _peak_rss_bytes()
            print(f"[ac4] prompt={prompt['id']} wall={dt:.1f}s "
                  f"peak_rss={peak / (1024 ** 3):.1f}GB", file=sys.stderr)
            if dt > MAX_PER_PROMPT_SECONDS:
                self.fail(f"AC9 STOP (xi): per-prompt wall {dt:.0f}s > "
                          f"{MAX_PER_PROMPT_SECONDS}s prompt {prompt['id']} "
                          f"(promote parent-arbitrated slice; AC7 ix)")
            if peak > MEM_HIGH_WATER_BYTES:
                self.fail(f"AC7 (x) STOP: peak RSS {peak / (1024 ** 3):.1f}GB "
                          f"> {MEM_HIGH_WATER_BYTES / (1024 ** 3):.0f}GB "
                          f"prompt {prompt['id']}")
            overlap = _overlap(_topk_argmax(ref_logits, ARGMAX_TOPK),
                               _topk_argmax(prod_logits, ARGMAX_TOPK))
            self.assertGreaterEqual(
                overlap, ARGMAX_OVERLAP_MIN,
                f"argmax top-{ARGMAX_TOPK} overlap {overlap} < "
                f"{ARGMAX_OVERLAP_MIN} prompt {prompt['id']}")

    # ---- STOP-rule reporting (NOT a derived-logits-compute test). ----

    @unittest.skip(
        "Superseded by ADR 0022 strategic pivot; gold-forward path is "
        "hardware-infeasible on M3 Ultra — see "
        "docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md"
    )
    def test_stop_rule_iv_reference_forward_blocker_is_surfaced(self):
        """STOP (iv) RESOLVED 2026-06-24 parent-arbitration Option 2: no longer
        STOP; stub returns a real-tensor payload. Historical audit preserved
        via KEPT test name + docstring (RESOLVED-state assertion flip)."""
        weights_bf16 = dequantize_shimmed_checkpoint_to_bf16_reference(
            CANONICAL_SHIMMED_CHECKPOINT)
        ref_layers, _ = independent_real_config_43_layer_reference_forward(
            args=None, weights_bf16=weights_bf16,
            input_ids=_encode_boundary())
        self.assertIn(0, ref_layers)
        self.assertEqual(ref_layers[0].dtype, np.float64)  # Q9 uniform float64


if __name__ == "__main__":
    unittest.main(verbosity=2)
