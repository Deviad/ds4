"""Story 11.55 -- NON-MUTATING MLX ``_real_forward`` per-layer intermediate dump harness.

Delivers a witness-only subclass of the vendored DeepSeek-V4 ``Model`` that
captures the post-FFN hyperconnection-residual-mix stream ``h`` returned by
vendor ``_real_layer_forward`` (line 1882 ``return h``) once per layer, WITHOUT
editing vendor source (ADR 0007 §4 anti-transliteration honored).

Non-circularity audit (AC12, B.1-B.4 mirrors 11.53 Reviewer pattern):
    * B.1: capture source = vendor math OUTPUT tensor (witness); numpy
      reference is the parity TARGET, never fed back into the vendor forward.
    * B.2: override body dispatches to ``super()._real_layer_forward`` --
      vendor math verbatim; harness does NOT reimplement hyperconnection
      mix / MoE / attention. Only an interleaved read-only ``.astype`` copy.
    * B.3: vendor file byte-identical (FROZEN HEAD ``221bdac``); ``out`` is
      returned UNCHANGED to ``_real_forward`` so the downstream vendor
      pipeline (next-layer ``h`` input, hc_head collapse, lm_head) is intact.
    * B.4: production whole-model Metal graph path is unaffected -- the
      subclass is opt-in (only the harness constructs ``...WithDump``).

Parity contract (Architecture Q1/§Q7):
    The captured stream ``h`` (shape ``[batch, seq, hc_mult=4, hidden=4096]``
    on the vendor side, batch-squeezed to ``[seq, 4, 4096]`` when dumped) is
    semantically identical to the numpy reference intermediate
    ``h_streams[L]`` captured by ``numpy_real_forward_reference.forward(
    ..., layers_to_compare=[L], return_intermediates=True)`` AFTER the
    post-FFN ``_hyperconnection_residual_mix`` step (Q7 ground-truthed in
    11.54 §4 -- vendor line 1881 einsum == numpy reference einsum). ADR 0020
    compositional tier ``L2_REL <= 5e-3`` is the parity gate (T7).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# MLX is optional at import time: witness live stages are gated by the test
# suite (skipif not HAS_MLX). The harness module imports cleanly without MLX.
try:
    import mlx.core as mx  # type: ignore

    _HAS_MLX = True
except Exception:  # pragma: no cover -- MLX absent
    mx = None  # type: ignore
    _HAS_MLX = False

# Vendor import: absolute path matches numpy_real_forward_reference convention.
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model as _DeepseekV4Model

# --------------------------------------------------------------------------- #
# Constants (Architecture Q4 / Q5 / Q6).
# --------------------------------------------------------------------------- #

VENDOR_CAPTURE_LINE = "deepseek_v4.py:1882 return h"
VENDOR_HEAD_SHA = "221bdac"
VENDOR_SHA256_16 = "812df0f7a34c0f07"  # ADR 0026 additive deepseek_v4.py baseline.
NUMPY_REF_SHA256_16 = "08d32750fc919453"  # post-11.54r2 F1-FIX.

DEFAULT_CKPT_DIR = "/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/"
DEFAULT_OUT_ROOT = "/Volumes/Data NVME/mlx-ft/ds4/intermediates"

_REPO_ROOT = Path(__file__).resolve().parents[4]
PROMPTS_PATH = _REPO_ROOT / "agent-output" / "cmux-11-55" / "prompts.json"

# Q5 CPU-safety floor: per-layer witness prompt seq_len cap.
SEQ_LEN_FLOOR = 4


def _sha256_16(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Witness harness subclass (Architecture Q3 -- approach (a-mod)).
# --------------------------------------------------------------------------- #


class DeepseekV4ModelWithDump(_DeepseekV4Model):
    """NON-MUTATING witness harness: per-layer post-FFN ``h_streams`` snapshot.

    Overrides the per-layer seam ``_real_layer_forward`` (NOT the monolithic
    ``_real_forward``) so each layer call pipelines through ``super()``-dispatched
    vendor math; the harness only interleaves a read-only ``.astype(float64)``
    copy of the returned stream state. Returns the vendor ``out`` UNCHANGED so the
    downstream forward (next layer's ``h`` input + hc_head + lm_head) is intact.

    Layer-index tracking (Q3): ``_real_forward`` iterates
    ``for layer_idx in range(num_hidden_layers)`` calling
    ``self._real_layer_forward(args, h, layer_w)`` exactly once per layer in
    monotonic 0..n-1 order. A harness counter (reset on every ``_real_forward``
    entry, snapshotted BEFORE increment) uniquely identifies the layer index
    -- no vendor edit + no ``layer_idx`` plumbed through the signature required.
    """

    def __init__(
        self,
        *args: Any,
        layers_to_capture: list[int] | None = None,
        capture_logits: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._layers_to_capture: set[int] = set(layers_to_capture or [])
        self._capture_logits: bool = capture_logits
        self._captured_layers_by_idx: dict[int, np.ndarray] = {}
        self._captured_logits: Any = None
        self._layer_counter: int = 0

    # Witness seam: reset per-forward buffers, dispatch to vendor super()
    # byte-exact, then optionally stash the final logits return (T8). The vendor
    # result is returned UNCHANGED so ``__call__`` sees the vendor output.
    def _real_forward(self, input_ids: list[list[int]]) -> Any:
        """Reset witness state, run vendor forward byte-exact, return it unchanged."""
        self._layer_counter = 0
        self._captured_layers_by_idx = {}
        self._captured_logits = None
        out = super()._real_forward(input_ids)  # vendor math, byte-exact
        if self._capture_logits:
            self._captured_logits = out
        return out  # UNCHANGED -- caller (vendor __call__) gets vendor result verbatim

    # -- seam: capture post-FFN h stream, return vendor out UNCHANGED --
    def _real_layer_forward(self, args: Any, h: Any, layer_w: Any) -> Any:
        out = super()._real_layer_forward(args, h, layer_w)  # vendor math, byte-exact
        layer_idx = self._layer_counter
        if layer_idx in self._layers_to_capture:
            # Read-only float64 copy; `out` itself is returned UNCHANGED below.
            # Batch-squeeze leading dim of size 1 -> [seq, hc_mult, hidden]
            # to match the numpy reference h_streams[L] contract (Q4/Q7).
            snap = np.asarray(out, dtype=np.float64)
            if snap.ndim == 4 and snap.shape[0] == 1:
                snap = snap[0]
            self._captured_layers_by_idx[layer_idx] = np.ascontiguousarray(snap)
        self._layer_counter += 1
        return out  # UNCHANGED -- next layer consumes vendor `out` verbatim


# --------------------------------------------------------------------------- #
# Arg validation (Q5 CPU-safety floor).
# --------------------------------------------------------------------------- #


def _validate_args(prompt_ids: list[int], layers_to_capture: list[int]) -> None:
    """Enforce the Q5 witness floor: seq_len <= 4 + bounded sampled layers."""
    seq_len = len(prompt_ids)
    if seq_len < 1:
        raise ValueError("prompt_ids must contain at least one token (seq_len >= 1)")
    if seq_len > SEQ_LEN_FLOOR:
        raise ValueError(
            f"seq_len={seq_len} > {SEQ_LEN_FLOOR} CPU-safety floor (Q5); "
            "use a bounded witness prompt"
        )
    if not layers_to_capture:
        raise ValueError("layers_to_capture must be a non-empty list of layer indices")
    if len(layers_to_capture) > 4:
        raise ValueError(
            f"layers_to_capture len={len(layers_to_capture)} > 4 (Q4 sampled-layers cap)"
        )


# --------------------------------------------------------------------------- #
# Prompts table (deterministic; Architecture Q6 --prompts.json resolution).
# --------------------------------------------------------------------------- #


def load_prompts_table(path: Path | str = PROMPTS_PATH) -> dict[str, dict[str, Any]]:
    """Load the deterministic 3-prompt table keyed by prompt id."""
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return {p["id"]: p for p in blob["prompts"]}


def resolve_prompt(
    prompt_id: str,
    path: Path | str = PROMPTS_PATH,
) -> dict[str, Any]:
    """Resolve a prompt id against prompts.json (deterministic text + seq_len)."""
    table = load_prompts_table(path)
    if prompt_id not in table:
        raise KeyError(
            f"unknown prompt_id={prompt_id!r}; expected one of {sorted(table)}"
        )
    return table[prompt_id]


# --------------------------------------------------------------------------- #
# Vendor model loader -- reuses production mlx_lm mmap-backed path (no eager
# 163GB copy; AGENTS.md "model loading mmap-backed" honored).
# --------------------------------------------------------------------------- #


def _load_vendor_base_model(model_path: str):
    """Load the vendored DeepSeek-V4 Model via the production mlx_lm path.

    Returns the fully-constructed base ``Model`` (lazy mmap-backed weights) +
    the tokenizer. The plugin registration guarantees
    ``mlx_lm.models.deepseek_v4`` resolves to THIS repository's vendored module.
    """
    from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin

    if not install_deepseek_v4_plugin():
        raise RuntimeError(
            "ds4_ft_mlx MLX-LM plugin unavailable (vendor module missing or "
            "mlx_lm not installed); cannot load vendored DeepseekV4Model"
        )
    import mlx_lm

    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Programmatic capture API (Test Manager + Reviewer consume directly, no CLI
# subprocess overhead). Architecture stage T3.
# --------------------------------------------------------------------------- #


def forward_capture(
    prompt_id: str,
    prompt_ids: list[int],
    model_path: str = DEFAULT_CKPT_DIR,
    layers_to_capture: list[int] | None = None,
    capture_logits: bool = False,
    max_layers: int | None = None,
) -> dict[str, Any]:
    """Run the NON-MUTATING witness harness and return captured intermediates.

    Parameters
    ----------
    prompt_id : str
        Deterministic prompt id (p0_single / p1_short / p2_argmax) for meta sidecar.
    prompt_ids : list[int]
        Bounded witness token ids (Q5 floor seq_len <= 4).
    model_path : str
        Shimmed checkpoint directory (safetensors index + config.json).
    layers_to_capture : list[int] | None
        Layer indices whose post-FFN ``h`` stream is captured. Default [0]
        (T7 L2_REL target).
    capture_logits : bool
        If True, stash the final ``_real_forward`` logits return for T8 argmax.
    max_layers : int | None
        Optional debug cap on the number of layers executed (CPU-safety witness).

    Returns
    -------
    dict
        ``{"captured_layers": list[np.ndarray] (layer-idx-aligned subset),
           "captured_logits": np.ndarray | None,
           "meta": {...}}``
    """
    layers = list(layers_to_capture if layers_to_capture is not None else [0])
    _validate_args(prompt_ids, layers)

    if not _HAS_MLX:
        raise RuntimeError("mlx.core is required for forward_capture but is not installed")

    base_model, tokenizer = _load_vendor_base_model(model_path)

    if max_layers is not None:
        # Debug cap: shrink num_hidden_layers on a shallow args clone so the
        # monotonic layer counter still maps cleanly. Vendor _real_forward reads
        # args.num_hidden_layers; we pass an args-like namespace alias.
        import types as _t

        a = base_model.args
        a = _t.SimpleNamespace(**{**vars(a), "num_hidden_layers": int(max_layers)})
        base_model.args = a  # type: ignore[attr-defined]

    harness = DeepseekV4ModelWithDump(
        base_model.args,
        layers_to_capture=layers,
        capture_logits=capture_logits,
    )
    # Transfer lazy mmap-backed weights (dict of mx.array refs; NO byte copy).
    harness._real_weights = base_model._real_weights

    seq_len = len(prompt_ids)
    harness([[int(t) for t in prompt_ids]])  # dispatches overridden _real_forward

    captured = [harness._captured_layers_by_idx[L] for L in sorted(harness._captured_layers_by_idx)]
    logits_np = None
    if capture_logits and harness._captured_logits is not None:
        logits_np = np.asarray(harness._captured_logits, dtype=np.float64)

    meta = _build_meta(
        prompt_id=prompt_id,
        seq_len=seq_len,
        layers=layers,
        args=base_model.args,
        captured_shapes={
            L: list(harness._captured_layers_by_idx[L].shape)
            for L in sorted(harness._captured_layers_by_idx)
        },
        logits_shape=(list(logits_np.shape) if logits_np is not None else None),
        max_layers=max_layers,
    )

    return {
        "captured_layers": captured,
        "captured_logits": logits_np,
        "meta": meta,
    }


def _build_meta(
    *,
    prompt_id: str,
    seq_len: int,
    layers: list[int],
    args: Any,
    captured_shapes: dict[int, list[int]],
    logits_shape: list[int] | None,
    max_layers: int | None,
) -> dict[str, Any]:
    args_snapshot: dict[str, Any] = {}
    for key in (
        "num_hidden_layers",
        "vocab_size",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "hc_mult",
        "hc_eps",
        "hc_sinkhorn_iters",
        "rms_norm_eps",
        "expert_dtype",
        "n_routed_experts",
        "num_experts_per_tok",
        "n_shared_experts",
        "moe_intermediate_size",
        "sliding_window",
        "rope_theta",
        "compression_ratio",
        "o_groups",
    ):
        if hasattr(args, key):
            try:
                args_snapshot[key] = getattr(args, key)
            except Exception:
                args_snapshot[key] = "<unreadable>"
    return {
        "prompt_id": prompt_id,
        "seq_len": seq_len,
        "layers_captured": sorted(layers),
        "capture_line": VENDOR_CAPTURE_LINE,
        "vendor_head_sha": VENDOR_HEAD_SHA,
        "vendor_sha256_16": VENDOR_SHA256_16,
        "numpy_ref_sha256_16": NUMPY_REF_SHA256_16,
        "model_args": args_snapshot,
        "dtype": "float64",
        "captured_shapes": {str(L): s for L, s in captured_shapes.items()},
        "logits_shape": logits_shape,
        "max_layers": max_layers,
        "captured_at_epoch": int(time.time()),
    }


# --------------------------------------------------------------------------- #
# Disk dump (Architecture stage T4 -- np.savez + meta.json sidecar).
# --------------------------------------------------------------------------- #


def dump_intermediates(
    out_root: str,
    prompt_id: str,
    captured_layers_by_idx: dict[int, np.ndarray] | list[np.ndarray],
    captured_logits: np.ndarray | None,
    meta: dict[str, Any],
) -> list[Path]:
    """Write `h_streams.npz` + `meta.json` per captured layer under out_root.

    Layout (Q4):
        <out_root>/prompt_id=<id>/layer_<L>/h_streams.npz
        <out_root>/prompt_id=<id>/layer_<L>/meta.json
    Each ``.npz`` is an uncompressed ``np.savez`` of a single array ``h_streams``
    shape ``[seq, hc_mult=4, hidden=4096]`` dtype ``float64``.
    """
    root = Path(out_root) / f"prompt_id={prompt_id}"
    written: list[Path] = []

    # Accept either a dict keyed by layer idx or a plain list (caller meta carries order).
    if isinstance(captured_layers_by_idx, dict):
        items = sorted(captured_layers_by_idx.items())
    else:
        items = list(enumerate(captured_layers_by_idx))

    for layer_idx, arr in items:
        arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        layer_dir = root / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        npz_path = layer_dir / "h_streams.npz"
        meta_path = layer_dir / "meta.json"

        np.savez(npz_path, h_streams=arr)  # uncompressed single array (Q4)

        layer_meta = dict(meta)
        layer_meta["layer_idx"] = int(layer_idx)
        layer_meta["shape"] = list(arr.shape)
        layer_meta["dtype"] = str(arr.dtype)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(layer_meta, fh, indent=2, sort_keys=True)
        written.append(npz_path)
        written.append(meta_path)

    # Optional end-to-end logits dump (T8 --emit-logits).
    if captured_logits is not None:
        logits_np = np.ascontiguousarray(np.asarray(captured_logits, dtype=np.float64))
        logits_dir = root / "logits"
        logits_dir.mkdir(parents=True, exist_ok=True)
        np.savez(logits_dir / "logits.npz", logits=logits_np)
        written.append(logits_dir / "logits.npz")

    return written


# --------------------------------------------------------------------------- #
# CLI (Architecture stage T5).
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m ds4_ft_mlx.real_forward_intermediate_dump",
        description="NON-MUTATING MLX _real_forward per-layer intermediate dump harness (Story 11.55).",
    )
    p.add_argument("--prompt", default="p1_short", help="prompt id from prompts.json (default p1_short)")
    p.add_argument("--layers", default="0", help="comma-separated layer indices (default '0')")
    p.add_argument("--model-path", default=DEFAULT_CKPT_DIR, help="shimmed checkpoint dir")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="intermediates output root")
    p.add_argument("--emit-logits", action="store_true", default=False, help="dump final lm_head logits (T8)")
    p.add_argument("--max-layers", type=int, default=None, help="debug cap on layers executed")
    p.add_argument("--prompts-json", default=str(PROMPTS_PATH), help="prompts.json table path")
    return p


def _parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _tokenize_prompt(tokenizer, text: str, seq_len: int) -> list[int]:
    """Tokenize + truncate to the witness floor seq_len (deterministic)."""
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    return ids


def main(argv: list[str] | None = None) -> int:
    if not _HAS_MLX:
        raise SystemExit("mlx.core not installed; harness requires the MLX Metal env")
    parser = _build_arg_parser()
    ns = parser.parse_args(argv)

    prompt = resolve_prompt(ns.prompt, ns.prompts_json)
    layers = _parse_layers(ns.layers)

    # Tokenize text -> bounded witness input_ids (Q5 floor enforced inside forward_capture).
    # Load ONLY the tokenizer here (cheap, no weights); forward_capture owns the
    # mmap-backed model load so the 163GB ckpt is read exactly once.
    from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin
    install_deepseek_v4_plugin()
    import mlx_lm
    tokenizer = mlx_lm.load_tokenizer(ns.model_path)
    prompt_ids = _tokenize_prompt(tokenizer, prompt["text"], int(prompt["seq_len"]))

    result = forward_capture(
        prompt_id=ns.prompt,
        prompt_ids=prompt_ids,
        model_path=ns.model_path,
        layers_to_capture=layers,
        capture_logits=ns.emit_logits,
        max_layers=ns.max_layers,
    )

    idx_map = {L: result["captured_layers"][i] for i, L in enumerate(sorted(result["meta"]["layers_captured"]))}
    written = dump_intermediates(
        out_root=ns.out_root,
        prompt_id=ns.prompt,
        captured_layers_by_idx=idx_map,
        captured_logits=result["captured_logits"],
        meta=result["meta"],
    )
    print(json.dumps({"written": [str(p) for p in written], "meta": result["meta"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
