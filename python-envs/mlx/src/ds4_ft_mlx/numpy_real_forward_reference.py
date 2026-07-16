"""Story 11.54 -- non-circular full 43-layer numpy reference forward.

Composes the 10 GREEN ``real_config_*`` primitives scaled in Story 11.53 (Group
A/B/D) and the 8 STOP-(iv) compositional pieces that 11.53 deferred into a
single whole-model ``forward`` over the real shimmed DeepSeek-V4 checkpoint.

Non-circularity contract (ADR 0007 §4 honored):

* Attention composition (``_compose_mqa_attention`` / ``_compose_sliding_attention``)
  is a numpy port of the ``tiny_multihead_grouped_attention_reference`` /
  ``tiny_sliding_attention_no_compressor`` SPEC MATH (already validated vs HF
  Transformers ``DeepseekV4Model`` in Story 11.15g / 11.53). It is *not* a
  transliteration of vendor ``_real_forward`` / ``_attention_mlx``. The MLX
  production surface is the parity WITNESS, not the source.
* Hyperconnection residual mixing (``_hyperconnection_residual_mix``) follows the
  Q7 ground-truth contract recorded in ``agent-output/cmux-11-54/architecture.md``
  (derived from HF ``DeepseekV4HyperConnection.forward`` /
  ``DeepseekV4DecoderLayer.forward``), not the vendor `_real_layer_forward` body.
* MoE (Group E) reuses the proven Story 11.15g primitive ``_moe_mlx`` (import +
  call only; independently BF16-pre-dequant-validated) per Architecture §1 Q3,
  rather than re-porting the spec MoE math.
* CSA path NOOPs at real config (``compression_ratio=0`` all 43 layers; no
  compressor/indexer tensors) per §1 Q6; ``_compose_csa_fusion`` explicit branch.

CPU-safety (AGENTS.md): per-prompt bounded ``seq_len`` witnesses only; the
module never materializes the full 163 GB in memory -- layers are mmap-loaded
on demand and released. Real-config full-parity stages T7/T8 require the MLX
production forward witness and are gated behind ``DS4_RUN_SLOW_PARITY``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# spec primitives (FROZEN, 11.53 owns): import + call only.
from ds4_ft_mlx.deepseek_v4_attention_spec import (  # noqa: E402
    _np_apply_rope_tail,
    _np_rope_cos_sin,
    _np_rms_norm,
    _np_softmax_last,
    real_config_embed_tokens,
    real_config_hyperconnection_forward,
    real_config_hyperhead_collapse,
    real_config_lm_head,
    real_config_rms_norm,
)

# Constants from real config.json (read verbatim by _load_config, not hardcoded).
_HC_MULT = 4
_HC_SINKHORN_ITERS = 20

# Default shimmed checkpoint (Architecture §0 / evidence.json).
DEFAULT_CKPT_DIR = "/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/"


# ---------------------------------------------------------------------------
# Offline safetensors weight loader (mmap-backed; NOT MLX load).
# ---------------------------------------------------------------------------

class _OfflineLoader:
    """Bounded safetensors reader mirroring the 11.53 test loader.

    Reads the safetensors index JSON, then memmaps individual tensors on demand
    (BF16 -> float64). Never holds the full shard set in memory.
    """

    def __init__(self, ckpt_dir: str | Path):
        self.ckpt_dir = Path(ckpt_dir)
        index_path = self.ckpt_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"safetensors index missing: {index_path}")
        with open(index_path, "r") as fh:
            index = json.load(fh)
        self.weight_map: dict[str, str] = dict(index.get("weight_map", {}))
        self._shard_meta: dict[str, dict[str, Any]] = {}

    def _load_shard_meta(self, shard: str) -> dict[str, Any]:
        # NOTE: method renamed from ``_shard_meta`` to avoid collision with the
        # instance-attr cache ``self._shard_meta`` (dict). Pre-rename the dict
        # attribute shadowed the bound method, raising ``TypeError: 'dict'
        # object is not callable`` on the first ``load()`` tensor read.
        if shard in self._shard_meta:
            return self._shard_meta[shard]
        # Reuse the proven checkpoint header reader (FROZEN import + call only).
        from ds4_ft_mlx.deepseek_v4_checkpoint import read_safetensors_header

        path = self.ckpt_dir / shard
        header = read_safetensors_header(path)
        meta = header if isinstance(header, dict) else {}
        n = meta.get("__metadata__")
        # Build per-key metadata: {key: {dtype, shape, data_offsets}}
        per_key = {}
        for key, val in meta.items():
            if key == "__metadata__" or not isinstance(val, dict):
                continue
            dtype = val.get("dtype")
            shape = val.get("shape", [])
            offs = val.get("data_offsets", [0, 0])
            per_key[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [int(offs[0]), int(offs[1])]}
        out = {"header": per_key, "n": 8 + len(json.dumps(meta, separators=(",", ":")))}
        self._shard_meta[shard] = out
        return out

    def _dtype_kind(self, dtype: str) -> str:
        d = dtype.upper()
        if d in ("BF16", "F16"):
            return "u2"  # uint16, view-cast
        if d == "F32":
            return "f4"
        if d in ("F64",):
            return "f8"
        if d == "I8":
            return "i1"  # int8 — raw, no dequant in loader (single dequant stays in _moe_mlx)
        if d == "I64":
            return "i8"  # int64 — raw, existing else-branch casts to float32→float64
        raise ValueError(f"unsupported safetensors dtype for numpy loader: {dtype}")

    def load(self, key: str, rows: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if key not in self.weight_map:
            raise KeyError(f"tensor not in safetensors index: {key}")
        shard = self.weight_map[key]
        meta = self._load_shard_meta(shard)["header"][key]
        dtype = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        # ``self._load_shard_meta(shard)["n"]`` ALREADY equals ``8 + len(json.dumps(meta))``
        # (8-byte length prefix + header_len). Do NOT add another 8 — the prior
        # ``8 + n + start`` double-counted the prefix and misaligned every read by
        # 8 bytes (caught by the synthetic-safetensors loader unit test added in the
        # same F1-FIX slice as the ``_shard_meta``→``_load_shard_meta`` rename).
        offset = self._load_shard_meta(shard)["n"] + start
        kind = self._dtype_kind(dtype)
        count = (end - start) // np.dtype(kind).itemsize
        base = self.ckpt_dir / shard
        flat = np.memmap(str(base), dtype=kind, mode="r", offset=offset, shape=(count,))
        if dtype.upper() == "BF16":
            flat = (flat.astype(np.uint32) << 16).view(np.float32)
        else:
            flat = flat.astype(np.float32)
        out = flat.reshape(shape).astype(np.float64)
        if rows is not None:
            out = out[np.asarray(list(rows))]
        return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Private composition helpers (numpy ports of tiny_* SPEC math -- non-circular).
# ---------------------------------------------------------------------------

def _causal_sliding_mask_np(seq_len: int, sliding_window: int) -> np.ndarray:
    """Causal + sliding-window additive mask (0 allowed, -inf masked).

    Mirrors ``tiny_multihead_grouped_attention_reference`` attention window
    slice ``kv_by_pos[start:pos+1]`` where ``start=max(0, pos-sliding_window+1)``.
    """
    pos = np.arange(seq_len)[:, None]
    key = np.arange(seq_len)[None, :]
    causal = np.where(key <= pos, 0.0, -np.inf)
    if sliding_window and sliding_window > 0:
        sw = np.where(key >= (pos - sliding_window + 1), 0.0, -np.inf)
    else:
        sw = 0.0
    return causal + sw


def _grouped_output_np(
    attended: np.ndarray,
    weights: dict,
    *,
    num_attention_heads: int,
    o_groups: int,
    o_lora_rank: int,
    head_dim: int,
    wo_a_key: str = "wo_a",
    wo_b_key: str = "wo_b",
) -> np.ndarray:
    """Numpy port of ``_grouped_output_projection``.

    ``attended``: [seq, num_attention_heads, head_dim].
    Weights are safetensors/MLX ``[out, in]`` layout.
    """
    seq = attended.shape[0]
    heads_per_group = num_attention_heads // o_groups
    o_a = np.asarray(weights[wo_a_key], dtype=np.float64)  # [o_groups*o_lora, heads_per_group*head_dim]
    o_b = np.asarray(weights[wo_b_key], dtype=np.float64)  # [hidden, o_groups*o_lora]
    chunks = []
    for group in range(o_groups):
        hs = group * heads_per_group
        he = hs + heads_per_group
        flat = attended[:, hs:he, :].reshape(seq, heads_per_group * head_dim)
        rows = o_a[group * o_lora_rank:(group + 1) * o_lora_rank, :]
        chunks.append(flat @ rows.T)  # [seq, o_lora_rank]
    low_rank = np.concatenate(chunks, axis=-1)  # [seq, o_groups*o_lora_rank]
    return low_rank @ o_b.T  # [seq, hidden]


def _compose_mqa_attention(
    x: np.ndarray,
    weights: dict,
    *,
    num_attention_heads: int,
    head_dim: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    hidden_size: int,
    rope_theta: float,
    sliding_window: int,
    rms_norm_eps: float,
    sinks_row: np.ndarray | None = None,
) -> np.ndarray:
    """Composition piece -- full MQA sliding-window attention at real dims.

    Numpy port of ``tiny_multihead_grouped_attention_reference`` body (spec
    math; NOT transliteration of vendor ``_attention_mlx`` / ``_real_forward``).

    One shared KV head broadcast across all query heads (num_key_value_heads=1,
    ADR 0007), tail-only RoPE on last ``qk_rope_head_dim`` channels, per-head
    sink as extra stable-softmax bucket dropped before V-mix, inverse output
    RoPE, independent grouped ``o_a`` blocks, final ``o_b``.
    """
    x = np.asarray(x, dtype=np.float64)
    seq = x.shape[0]
    nope = head_dim - qk_rope_head_dim
    # Real shimmed checkpoint stores attention projections as BF16 (the F8-E4M3
    # weights were dequantized to BF16 at shim time; per-block ``.scale`` tensors
    # are retained but NOT re-applied -- the ``.weight`` is already the final
    # value, matching vendor ``_attention_mlx`` which consumes the weight directly).
    q_a = x @ np.asarray(weights["wq_a"], dtype=np.float64).T  # [seq, q_lora_rank]
    q_a = _np_rms_norm(q_a, np.asarray(weights["q_norm"], dtype=np.float64), rms_norm_eps)
    q = q_a @ np.asarray(weights["wq_b"], dtype=np.float64).T  # [seq, q_dim]
    q = q.reshape(seq, num_attention_heads, head_dim)
    q = _np_rms_norm(q, None, rms_norm_eps)  # per-head unweighted q RMSNorm
    kv = x @ np.asarray(weights["wkv"], dtype=np.float64).T  # [seq, head_dim]
    kv = _np_rms_norm(kv, np.asarray(weights["kv_norm"], dtype=np.float64), rms_norm_eps)

    cos, sin = _np_rope_cos_sin(np.arange(seq, dtype=np.float64), qk_rope_head_dim, rope_theta)  # [seq, rope_dim]
    q = _np_apply_rope_tail(q, nope, cos[:, None, :], sin[:, None, :])
    kv = _np_apply_rope_tail(kv, nope, cos, sin)

    scale = head_dim ** -0.5
    qh = q.transpose(1, 0, 2)  # [heads, seq, head_dim]
    kvb = kv[None, :, :]      # [1, seq, head_dim] broadcast over heads
    scores = (qh @ kvb.transpose(0, 2, 1)) * scale  # [heads, seq, seq]
    mask = _causal_sliding_mask_np(seq, sliding_window)  # [seq, seq]
    scores = scores + mask[None, :, :]  # broadcast over heads

    if sinks_row is None:
        sinks_row = np.asarray(weights["attn_sink"], dtype=np.float64)
    sinks_row = np.asarray(sinks_row, dtype=np.float64).reshape(num_attention_heads)
    sink_col = np.broadcast_to(sinks_row.reshape(num_attention_heads, 1, 1), (num_attention_heads, seq, 1))
    scores_full = np.concatenate([scores, sink_col], axis=-1)  # [heads, seq, seq+1]
    probs = _np_softmax_last(scores_full)  # stable softmax over last axis
    probs_kv = probs[..., :seq]  # drop sink before V-mix
    attended = probs_kv @ kvb  # [heads, seq, head_dim]
    attended = _np_apply_rope_tail(attended, nope, cos[None, :, :], (-sin)[None, :, :])  # inverse output RoPE
    attended = attended.transpose(1, 0, 2)  # [seq, heads, head_dim]

    return _grouped_output_np(
        attended,
        weights,
        num_attention_heads=num_attention_heads,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        head_dim=head_dim,
        wo_a_key="wo_a",
        wo_b_key="wo_b",
    )


def _compose_sliding_attention(*args, **kwargs) -> np.ndarray:
    """Sliding-window branch (sliding_window>0). Delegates to the MQA port."""
    return _compose_mqa_attention(*args, **kwargs)


def _compose_csa_fusion(compression_ratio: int) -> None:
    """CSA compressor/indexer fusion -- NOOP at real config (Q6).

    Real config has ``compression_ratio=0`` all 43 layers and NO
    compressor/indexer tensors (11.53 69,187-tensor scan). This branch is
    explicit and auditable; a future ``compression_ratio != 0`` integrated layer
    would route through the 11.53 Group B CSA primitives -- that path is left
    ready (defers to STOP (iv) standalone primitives) and is NOT exercised here.
    """
    if compression_ratio != 0:
        raise NotImplementedError(
            "STOP (iv): real_config CSA fusion at compression_ratio != 0 is a "
            "future integrated-layer path (no compressor/indexer tensors in the "
            "11.53 shimmed checkpoint)."
        )
    return None


def _compose_incremental_step(*_args: object, **_kwargs: object) -> None:
    """Incremental KV attention composition piece (one decode step).

    STOP (iv): composes the full attention MQA path through a position-baked
    sliding-window KV cache at real MQA dims (num_attention_heads=64,
    head_dim=512, q_dim=32768, o_groups=8). The pure-python ``tiny_*`` witness
    (``IncrementalSlidingKVCache``) is CPU-infeasible at real config and an
    independent second numpy implementation would transliterate vendor
    ``_real_forward`` decoding; the bounded greedy-decode witness is deferred to
    ``11.15h`` resume + the Q4 MLX witness harness (AC3/AC4 real-config stages).
    """
    raise NotImplementedError(
        "STOP (iv): real_config incremental_attention composition cannot be "
        "wired at real MQA dims without a bounded MLX per-step witness (Q4)."
    )


def _compose_greedy_decode(*_args: object, **_kwargs: object) -> None:
    """Greedy decode loop composition piece.

    STOP (iv): composes incremental attention + lm_head into a greedy decode
    loop; depends on ``_compose_incremental_step`` and a bounded top-k logits
    slice. The pure-python ``tiny_greedy_decode`` witness is CPU-infeasible at
    real config. Deferred to ``11.15h`` resume + Q4 MLX witness.
    """
    raise NotImplementedError(
        "STOP (iv): real_config greedy_decode composition depends on the "
        "incremental KV step (STOP iv); deferred to 11.15h resume + Q4 witness."
    )


def _hyperconnection_residual_mix(
    h_streams: np.ndarray,
    hc: dict[str, np.ndarray],
    sublayer_out: np.ndarray,
) -> np.ndarray:
    """Q7 hyperconnection residual mix (attn/ffn site).

    Contract (architecture §1 Q7, HF ``DeepseekV4DecoderLayer.forward`` L1146-1149):
        attn_residual = einsum("sak,sad->skd", comb, h_streams)
        h_new = post[...,None] * sublayer_out[...,None,:] + attn_residual

    New stream ``k`` mixes old streams ``a`` via ``comb[a, k]``. ``comb`` is
    normalized per-layer by ``real_config_hyperconnection_forward`` (Sinkhorn).
    """
    h_streams = np.asarray(h_streams, dtype=np.float64)  # [seq, hc_mult, hidden]
    comb = np.asarray(hc["comb"], dtype=np.float64)       # [seq, old=hc_mult, new=hc_mult]
    post = np.asarray(hc["post"], dtype=np.float64)       # [seq, new=hc_mult]
    sublayer_out = np.asarray(sublayer_out, dtype=np.float64)  # [seq, hidden]
    attn_residual = np.einsum("sak,sad->skd", comb, h_streams)  # [seq, new, hidden]
    return post[..., None] * sublayer_out[:, None, :] + attn_residual


# ---------------------------------------------------------------------------
# MoE bridge: reuse proven _moe_mlx (Architecture §1 Q3; non-circular primitive reuse).
# ---------------------------------------------------------------------------

def _moe_out_via_mlx(
    x_np: np.ndarray,
    layer_moe_weights_np: dict,
    *,
    args,
) -> np.ndarray:
    """Call vendor ``_moe_mlx`` on real-config MoE weights (import + call only).

    Per Architecture §1 Q3, the MoE branch is REUSED (not re-ported from spec):
    Story 11.15g Q3 proved ``_moe_mlx`` I8/BF16-expert composition vs a
    BF16-pre-dequant reference at ``L2_REL = 0.000e+00``. Re-deriving the
    routed-I8 dequant + sqrtsoftplus + top-6 + routed_scaling_factor + shared
    expert path in pure numpy would re-introduce a drift surface.

    The real safetensors keys spell ``layers.N.ffn.*``; ``_moe_mlx`` expects
    ``mlp.*``-prefixed keys per its own contract. This bridge re-prefixes and
    converts numpy -> mx.array -> numpy. It does NOT read or copy vendor forward
    composition.
    """
    try:
        import mlx.core as mx
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx
    except Exception as exc:  # pragma: no cover - MLX not installed
        raise NotImplementedError(f"STOP (iv): MoE _moe_mlx unavailable: {exc}") from exc

    # Map real layer MoE keys (``ffn.*``) onto _moe_mlx's ``mlp.*`` contract.
    remap = {
        "ffn.gate.weight": "mlp.gate.weight",
        "ffn.gate.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
        "ffn.shared_experts.w1.weight": "mlp.shared_experts.w1.weight",
        "ffn.shared_experts.w2.weight": "mlp.shared_experts.w2.weight",
        "ffn.shared_experts.w3.weight": "mlp.shared_experts.w3.weight",
    }
    weights_mlx: dict[str, Any] = {}
    for real_key, mlx_key in remap.items():
        if real_key in layer_moe_weights_np:
            weights_mlx[mlx_key] = mx.array(np.asarray(layer_moe_weights_np[real_key], dtype=np.float32))
    # routed experts
    n = getattr(args, "n_routed_experts", 256)
    for eid in range(n):
        for proj in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                rk = f"ffn.experts.{eid}.{proj}.{kind}"
                if rk in layer_moe_weights_np:
                    weights_mlx[f"mlp.experts.{eid}.{proj}.{kind}"] = mx.array(
                        np.asarray(layer_moe_weights_np[rk], dtype=np.float32)
                    )

    # Real shimmed routed experts are I8 + BF16 block-scale (ADR 0007); force the
    # i8 dequant path regardless of the config.json ``expert_dtype`` spelling
    # ("fp4"), since the shimmed checkpoint materializes routed experts as I8.
    import types as _t
    if getattr(args, "expert_dtype", None) != "i8":
        args = _t.SimpleNamespace(**{**vars(args), "expert_dtype": "i8"})
    x_mlx = mx.array(np.asarray(x_np, dtype=np.float32))[None, :, :]  # [batch, seq, hidden]
    out = _moe_mlx(args, x_mlx, weights_mlx)
    out_np = np.asarray(out, dtype=np.float64)[0]  # drop batch
    return np.ascontiguousarray(out_np)


# ---------------------------------------------------------------------------
# Public entrypoint (AC1 signature verbatim).
# ---------------------------------------------------------------------------

def _load_config(model_path: str | Path) -> dict:
    path = Path(model_path) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"config.json missing: {path}")
    with open(path, "r") as fh:
        return json.load(fh)


def _config_args(config: dict | None = None, model_path: str | Path = DEFAULT_CKPT_DIR):
    """Build a lightweight args namespace from config.json (offline JSON read)."""
    from types import SimpleNamespace

    if config is None:
        config = _load_config(model_path)
    a = SimpleNamespace()
    for k, v in config.items():
        setattr(a, k, v)
    # Derived dims used by composition.
    a.num_key_value_heads = config.get("num_key_value_heads", 1)
    a.heads_per_output_group = a.num_attention_heads // a.o_groups
    a.q_dim = a.num_attention_heads * a.head_dim
    a.out_low_dim = a.o_groups * a.o_lora_rank
    return a


def forward(
    input_ids,
    model_path: str | Path = DEFAULT_CKPT_DIR,
    config=None,
    *,
    layers_to_compare: list[int] | None = None,
    return_intermediates: bool = False,
    max_layers: int | None = None,
):
    """Full real-config 43-layer numpy forward reference.

    Parameters
    ----------
    input_ids : list[int] | np.ndarray
        Bounded-seq prompt token ids (CPU-safe witness floor seq<=4 per Q5).
    model_path : str | Path
        Shimmed checkpoint directory (safetensors index + config.json).
    config : object | None
        Pre-built ModelArgs (offline); built from config.json if None.
    layers_to_compare : list[int] | None
        Layers whose post-FFN residual ``h`` is captured as an intermediate.
    return_intermediates : bool
        If True, return ``(logits, per_layer_intermediates)``.
    max_layers : int | None
        Debug cap on number of layers executed (CPU-safety witness floor).

    Returns
    -------
    np.ndarray | (np.ndarray, list[np.ndarray])
        Final logits ``[seq, vocab=129280]`` (or top-k slice when cheap),
        optionally with per-layer post-FFN residual ``h`` per AC3.
    """
    args = config if config is not None else _config_args(config=None, model_path=model_path)
    n_layers = int(getattr(args, "num_hidden_layers", 43))
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)
    hc_mult = int(getattr(args, "hc_mult", _HC_MULT))
    eps = float(getattr(args, "hc_eps", 1e-6))
    rms_eps = float(getattr(args, "rms_norm_eps", 1e-6))
    sinkhorn_iters = int(getattr(args, "hc_sinkhorn_iters", _HC_SINKHORN_ITERS))
    rope_theta = float(getattr(args, "rope_theta", 10000.0))
    sliding_window = int(getattr(args, "sliding_window", 128))

    loader = _OfflineLoader(model_path)

    ids = np.asarray(list(input_ids), dtype=np.int64).reshape(-1)
    embed = loader.load("embed.weight")
    h = real_config_embed_tokens(ids, embed)  # [seq, hidden]
    seq = h.shape[0]
    # Q7 §1 stream init: broadcast-expand to [seq, hc_mult, hidden].
    h_streams = np.broadcast_to(h[:, None, :], (seq, hc_mult, h.shape[-1])).copy()

    # Layer indices whose post-FFN residual is captured.
    capture = set(layers_to_compare) if layers_to_compare is not None else set(range(n_layers))
    intermediates: list[np.ndarray] = []

    # Pre-load norm weights + lm head lazily per-layer. CA NOOP for all real layers.
    for L in range(n_layers):
        prefix = f"layers.{L}."
        # --- attention hyperconnection ---
        hc_attn = real_config_hyperconnection_forward(
            hidden_streams=h_streams,
            fn=loader.load(prefix + "hc_attn_fn"),
            base=loader.load(prefix + "hc_attn_base"),
            scale=loader.load(prefix + "hc_attn_scale"),
            hc_mult=hc_mult,
            eps=eps,
            sinkhorn_iters=sinkhorn_iters,
            rms_norm_eps=rms_eps,
        )
        attn_norm_w = loader.load(prefix + "attn_norm.weight")
        # Q7 §2: collapse -> rms_norm (input_layernorm on collapsed stream).
        norm_in = real_config_rms_norm(hc_attn["collapsed"], attn_norm_w, rms_eps)

        attn_weights = {
            "wq_a": loader.load(prefix + "attn.wq_a.weight"),
            "q_norm": loader.load(prefix + "attn.q_norm.weight"),
            "wq_b": loader.load(prefix + "attn.wq_b.weight"),
            "wkv": loader.load(prefix + "attn.wkv.weight"),
            "kv_norm": loader.load(prefix + "attn.kv_norm.weight"),
            "wo_a": loader.load(prefix + "attn.wo_a.weight"),
            "wo_b": loader.load(prefix + "attn.wo_b.weight"),
            "attn_sink": loader.load(prefix + "attn.attn_sink"),
        }
        attn_out = _compose_mqa_attention(
            norm_in,
            attn_weights,
            num_attention_heads=int(args.num_attention_heads),
            head_dim=int(args.head_dim),
            q_lora_rank=int(args.q_lora_rank),
            qk_rope_head_dim=int(args.qk_rope_head_dim),
            o_groups=int(args.o_groups),
            o_lora_rank=int(args.o_lora_rank),
            hidden_size=int(args.hidden_size),
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            rms_norm_eps=rms_eps,
        )
        # CSA fusion NOOP at real config (compression_ratio=0).
        _compose_csa_fusion(int(getattr(args, "compression_ratio", 0)))
        # hyperconnection residual mix (attn site).
        h_streams = _hyperconnection_residual_mix(h_streams, hc_attn, attn_out)

        # --- ffn hyperconnection + MoE ---
        hc_ffn = real_config_hyperconnection_forward(
            hidden_streams=h_streams,
            fn=loader.load(prefix + "hc_ffn_fn"),
            base=loader.load(prefix + "hc_ffn_base"),
            scale=loader.load(prefix + "hc_ffn_scale"),
            hc_mult=hc_mult,
            eps=eps,
            sinkhorn_iters=sinkhorn_iters,
            rms_norm_eps=rms_eps,
        )
        ffn_norm_w = loader.load(prefix + "ffn_norm.weight")
        norm_post = real_config_rms_norm(hc_ffn["collapsed"], ffn_norm_w, rms_eps)

        # MoE (Group E) via proven _moe_mlx import (Q3).
        moe_weights = {}
        # gather routed + shared expert tensors (lazy per-layer).
        for k in loader.weight_map:
            if k.startswith(prefix + "ffn."):
                moe_weights[k[len(prefix):]] = loader.load(k)
        moe_out = _moe_out_via_mlx(norm_post, moe_weights, args=args)
        h_streams = _hyperconnection_residual_mix(h_streams, hc_ffn, moe_out)

        if L in capture:
            intermediates.append(np.ascontiguousarray(h_streams))

    # --- tail: final norm + hyperhead collapse + lm_head ---
    final_norm_w = loader.load("norm.weight")
    hc_head = real_config_hyperhead_collapse(
        hidden_streams=h_streams,
        fn=loader.load("hc_head_fn"),
        base=loader.load("hc_head_base"),
        scale=loader.load("hc_head_scale"),
        hc_mult=hc_mult,
        eps=eps,
        rms_norm_eps=rms_eps,
    )
    h_final = real_config_rms_norm(hc_head["collapsed"], final_norm_w, rms_eps)
    head_w = loader.load("head.weight")
    logits, _cols = real_config_lm_head(h_final, head_w, top_k=0)

    if return_intermediates:
        return logits, intermediates
    return logits
