"""Story 11.55 -- TDD suite for the NON-MUTATING MLX ``_real_forward`` dump harness.

Stages (per architecture.md / Coder task T1-T8):

* T1 -- byte-intactness guard: vendor ``deepseek_v4.py`` FROZEN (sha256 pin).
* T2 -- harness subclass importable; override is a transparent ``super()``
  dispatch (no decoration, signature matches vendor).
* T3 -- ``forward_capture`` programmatic API signature present.
* T4 -- ``dump_intermediates`` writes ``h_streams.npz`` + ``meta.json`` sidecar;
  ``np.load`` round-trips shape ``[4, 4, 4096]`` dtype ``float64``.
* T5 -- CLI argparse defaults ``layers=[0]``, ``emit_logits=False``.
* T6 (live) -- T7 layer-0 post-FFN ``h_streams`` ``L2_REL <= 5e-3`` vs
  ``numpy_real_forward_reference.forward(..., return_intermediates=True)`` (ADR 0020).
* T7 (live) -- T8 end-to-end argmax top-7 overlap ``>= 5/7`` vs numpy reference
  full logits.

CPU-only stages (T1-T5) run everywhere; live stages (T6/T7) are gated behind
``DS4_RUN_SLOW_PARITY=1`` + MLX + the shimmed ckpt (skip otherwise, per Q2/Q6).
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python-envs" / "mlx" / "src"))

from ds4_ft_mlx import real_forward_intermediate_dump as rfid  # noqa: E402
from ds4_ft_mlx import numpy_real_forward_reference as nrf  # noqa: E402
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model as _VendorModel  # noqa: E402

CKPT_DIR = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
HAS_CKPT = CKPT_DIR.is_dir()
HAS_MLX = True
try:
    import mlx.core as mx  # noqa: F401
except Exception:
    HAS_MLX = False
RUN_SLOW = os.environ.get("DS4_RUN_SLOW_PARITY") == "1"

PROMPTS_PATH = ROOT / "agent-output" / "cmux-11-55" / "prompts.json"

TOL_COMP = 5e-3  # ADR 0020 compositional tier


def _sha256_16(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _l2_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = float(np.linalg.norm((a - b).ravel()))
    denom = float(np.linalg.norm(a.ravel()))
    return diff / denom if denom > 0 else diff


# --------------------------------------------------------------------------- #
# T1 -- byte-intactness guard (vendor FROZEN + harness adds a NEW file only).
# --------------------------------------------------------------------------- #


class TestT1ByteIntactness:
    def test_t1_vendor_deepseek_v4_py_frozen_byte_identical_HEAD_221bdac(self):
        # AGENTS.md untracked-Python caveat: the vendor file is not in the git
        # tree, so we pin the sha256[:16] (mirrors 11.54 EXPECTED_SHAS) rather
        # than `git rev-parse 221bdac:<path>`. This asserts the harness does
        # NOT mutate vendor source (AC2 + AC6).
        vendor = ROOT / "python-envs" / "mlx" / "src" / "ds4_ft_mlx" / "vendor" / "mlx_lm_models" / "deepseek_v4.py"
        assert vendor.is_file(), f"vendor file missing: {vendor}"
        assert _sha256_16(vendor) == rfid.VENDOR_SHA256_16, "vendor deepseek_v4.py byte drift"

    def test_harness_module_is_new_file_not_vendor(self):
        # The harness lives in its own module; Story 13.3a adds a sanctioned
        # sibling nn.Module port in the vendor plugin directory.  The harness
        # module itself must still NOT leak into that directory.
        harness = ROOT / "python-envs" / "mlx" / "src" / "ds4_ft_mlx" / "real_forward_intermediate_dump.py"
        assert harness.is_file(), "harness module missing"
        vendor_dir = ROOT / "python-envs" / "mlx" / "src" / "ds4_ft_mlx" / "vendor" / "mlx_lm_models"
        py_files = {p.name for p in vendor_dir.iterdir() if p.suffix == ".py"}
        allowed = {"__init__.py", "deepseek_v4.py", "deepseek_v4_nn.py"}
        assert py_files == allowed, (
            f"vendor/mlx_lm_models gained unexpected .py files (AC2 violation): {py_files}"
        )


# --------------------------------------------------------------------------- #
# T2 -- harness subclass structure (CPU-only, no Metal/ckpt).
# --------------------------------------------------------------------------- #


class TestT2HarnessSubclass:
    def test_t2_harness_subclass_exists_importable(self):
        assert issubclass(rfid.DeepseekV4ModelWithDump, _VendorModel)

    def test_override_is_transparent_super_dispatch(self):
        ov = rfid.DeepseekV4ModelWithDump._real_layer_forward
        # Plain function, no functools.wraps / decorator stacking that would
        # obscure the super() dispatch.
        assert inspect.isfunction(ov), "override must be a plain function (no decorator stacking)"
        sig = list(inspect.signature(ov).parameters)
        assert sig == ["self", "args", "h", "layer_w"], f"override signature drift: {sig}"

    def test_real_forward_override_resets_counter(self):
        # _real_forward override exists + resets witness buffers for repeat calls.
        assert "reset" in (rfid.DeepseekV4ModelWithDump._real_forward.__doc__ or "").lower()


# --------------------------------------------------------------------------- #
# T3 -- forward_capture API signature (RED pre-API / GREEN post-API).
# --------------------------------------------------------------------------- #


class TestT3ForwardCaptureSignature:
    def test_t3_forward_capture_signature_accepts_layers_to_capture(self):
        sig = inspect.signature(rfid.forward_capture)
        params = sig.parameters
        assert "prompt_id" in params
        assert "prompt_ids" in params
        assert "layers_to_capture" in params
        assert "capture_logits" in params
        # default layers_to_capture -> [0] (T7 target)
        assert params["layers_to_capture"].default is None  # resolves to [0] inside
        assert params["capture_logits"].default is False

    def test_validate_args_enforces_seq_len_floor(self):
        with pytest.raises(ValueError, match="CPU-safety floor"):
            rfid._validate_args([1, 2, 3, 4, 5], [0])
        with pytest.raises(ValueError):
            rfid._validate_args([], [0])
        with pytest.raises(ValueError):
            rfid._validate_args([1, 2, 3, 4], [])  # no layers
        # in-bounds: no raise
        rfid._validate_args([1, 2, 3, 4], [0])


# --------------------------------------------------------------------------- #
# T4 -- np.savez writer + meta.json sidecar (CPU-only, mocked tensors).
# --------------------------------------------------------------------------- #


class TestT4DumpIntermediates:
    def test_t4_dump_intermediates_writes_npz_and_meta_sidecar(self, tmp_path):
        mock = {0: np.zeros((4, 4, 4096), dtype=np.float64)}
        meta = {
            "prompt_id": "p1_short",
            "seq_len": 4,
            "capture_line": rfid.VENDOR_CAPTURE_LINE,
            "vendor_head_sha": rfid.VENDOR_HEAD_SHA,
            "numpy_ref_sha256_16": rfid.NUMPY_REF_SHA256_16,
            "dtype": "float64",
            "layers_captured": [0],
        }
        written = rfid.dump_intermediates(
            out_root=str(tmp_path),
            prompt_id="p1_short",
            captured_layers_by_idx=mock,
            captured_logits=None,
            meta=meta,
        )
        npz = tmp_path / "prompt_id=p1_short" / "layer_0" / "h_streams.npz"
        side = tmp_path / "prompt_id=p1_short" / "layer_0" / "meta.json"
        assert npz in written and side in written
        arr = np.load(npz)["h_streams"]
        assert arr.shape == (4, 4, 4096), arr.shape
        assert arr.dtype == np.float64
        import json
        layer_meta = json.loads(side.read_text())
        assert layer_meta["layer_idx"] == 0
        assert layer_meta["shape"] == [4, 4, 4096]
        assert layer_meta["capture_line"] == rfid.VENDOR_CAPTURE_LINE

    def test_dump_intermediates_writes_logits_when_emitted(self, tmp_path):
        logits = np.zeros((1, 4, 129280), dtype=np.float64)
        written = rfid.dump_intermediates(
            out_root=str(tmp_path),
            prompt_id="p2_argmax",
            captured_layers_by_idx={0: np.zeros((4, 4, 4096), dtype=np.float64)},
            captured_logits=logits,
            meta={"prompt_id": "p2_argmax", "dtype": "float64", "layers_captured": [0]},
        )
        assert (tmp_path / "prompt_id=p2_argmax" / "logits" / "logits.npz") in written


# --------------------------------------------------------------------------- #
# T5 -- CLI argparse defaults (CPU-only).
# --------------------------------------------------------------------------- #


class TestT5Cli:
    def test_t5_cli_argparse_layer_default_0_emit_logits_off(self):
        ns = rfid._build_arg_parser().parse_args([])
        assert rfid._parse_layers(ns.layers) == [0]
        assert ns.emit_logits is False
        assert ns.prompt == "p1_short"
        assert ns.model_path == rfid.DEFAULT_CKPT_DIR


_LIVE_PYTEST_MARKS = [
    pytest.mark.skipif(
        not (RUN_SLOW and HAS_CKPT and HAS_MLX),
        reason="full real-config parity needs DS4_RUN_SLOW_PARITY=1 + ckpt + MLX (Q4 witness harness)",
    ),
]


def _apply_marks(marks, func):
    for m in reversed(marks):
        func = m(func)
    return func


def _tokenize(model_path: str, text: str, seq_len: int) -> list[int]:
    from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin

    install_deepseek_v4_plugin()
    import mlx_lm

    tok = mlx_lm.load_tokenizer(model_path)
    ids = list(tok.encode(text, add_special_tokens=False))
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    return ids


# --------------------------------------------------------------------------- #
# T6 (live) -- T7 layer-0 post-FFN h_streams L2_REL <= 5e-3.
# --------------------------------------------------------------------------- #


def _run_t7():
    prompt = rfid.resolve_prompt("p1_short", PROMPTS_PATH)
    ids = _tokenize(rfid.DEFAULT_CKPT_DIR, prompt["text"], int(prompt["seq_len"]))
    result = rfid.forward_capture(
        prompt_id="p1_short",
        prompt_ids=ids,
        model_path=rfid.DEFAULT_CKPT_DIR,
        layers_to_capture=[0],
        capture_logits=False,
    )
    h_mlx = result["captured_layers"][0]

    # numpy reference witness -- same prompt_ids, layer 0 intermediate.
    logits_ref, inters = nrf.forward(
        ids,
        model_path=rfid.DEFAULT_CKPT_DIR,
        layers_to_compare=[0],
        return_intermediates=True,
    )
    h_ref = np.asarray(inters[0], dtype=np.float64)
    return _l2_rel(h_mlx, h_ref)


def test_t7_layer0_l2_rel_live():
    rel = _run_t7()
    assert rel <= TOL_COMP, f"T7 L2_REL={rel:.3e} > {TOL_COMP:.0e} (ADR 0020 compositional tier)"


test_t7_layer0_l2_rel_live = _apply_marks(_LIVE_PYTEST_MARKS, test_t7_layer0_l2_rel_live)


# --------------------------------------------------------------------------- #
# T7 (live) -- T8 end-to-end argmax top-7 overlap >= 5/7.
# --------------------------------------------------------------------------- #


def _argmax_top7(logits_row: np.ndarray) -> list[int]:
    row = np.asarray(logits_row, dtype=np.float64).ravel()
    return list(np.argsort(row)[::-1][:7].tolist())


def _run_t8():
    prompt = rfid.resolve_prompt("p2_argmax", PROMPTS_PATH)
    ids = _tokenize(rfid.DEFAULT_CKPT_DIR, prompt["text"], int(prompt["seq_len"]))
    result = rfid.forward_capture(
        prompt_id="p2_argmax",
        prompt_ids=ids,
        model_path=rfid.DEFAULT_CKPT_DIR,
        layers_to_capture=[0],
        capture_logits=True,
    )
    # harness logits: [batch, seq, vocab]; take last position of batch 0.
    logits_mlx = np.asarray(result["captured_logits"], dtype=np.float64)
    top_mlx = _argmax_top7(logits_mlx[0, -1, :])

    logits_ref = np.asarray(
        nrf.forward(ids, model_path=rfid.DEFAULT_CKPT_DIR), dtype=np.float64
    )
    top_ref = _argmax_top7(logits_ref[-1, :])

    overlap = len(set(top_mlx) & set(top_ref))
    return overlap, top_mlx, top_ref


def test_t8_argmax_top7_overlap_live():
    overlap, top_mlx, top_ref = _run_t8()
    assert overlap >= 5, (
        f"T8 argmax top-7 overlap={overlap}/7 < 5; mlx={top_mlx} ref={top_ref}"
    )


test_t8_argmax_top7_overlap_live = _apply_marks(_LIVE_PYTEST_MARKS, test_t8_argmax_top7_overlap_live)
