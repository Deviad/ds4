#!/usr/bin/env python3
"""Safe helpers for the DS4 MLX fine-tuning plan.

Generated datasets, converted MLX models, adapters, logs, and GGUF artifacts are
kept outside this repository by default.  Cheap deterministic steps can run
directly; expensive MLX/GGUF steps are dry-run-first and require explicit
``--execute --yes`` approval.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import logging
import math
import os
import pathlib
import re
import shlex
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)  # module-level logger (ADR 0022 gate-lift warning path)

SCRIPT_VERSION = "1.0"
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT_THINK = "<｜Assistant｜><think>"
EOS = "<｜end▁of▁sentence｜>"
DEFAULT_SPLIT_DIR = "mlx-4096"
BACKENDS = ("local-mlx", "local-torch-mps", "remote-cuda", "cpu-check", "manual")
MLX_STEPS = (
    "setup-env",
    "mlx-device",
    "convert",
    "fp8-shim-scan",
    "fp8-shim-probe",
    "fp8-shim",
    "deepseek-v4-import-check",
    "deepseek-v4-tiny-config-check",
    "deepseek-v4-mtp-exclusion-check",
    "deepseek-v4-mapping-check",
    "deepseek-v4-dequant-parity-check",
    "deepseek-v4-forward-parity-check",
    "convert-shimmed",
    "model-4bit-conversion-plan",
    "deepseek-v4-forward-parity-readiness",
    "mlx-lora-targets-check",
    "smoke-train",
    "smoke-train-2048",
    "smoke-generate",
    "convert-smoke-adapter",
    "ds4-smoke-adapter-inspect",
    "full-train",
    "continue-train",
    "eval",
    "fuse",
    "fuse-hf",
    "fused-generate",
    "build-quantizer",
    "quantize-q2",
    "quantize-q4",
    "splice-dry-run",
    "splice",
    "ds4-smoke",
)
TORCH_MPS_STEPS = (
    "torch-env-create",
    "torch-env-check",
    "torch-smoke",
    "torch-one-step-lora",
    "torch-real-v4-feasibility",
)
REMOTE_CUDA_STEPS = (
    "remote-cuda-smoke",
    "remote-cuda-full",
)
CPU_CHECK_STEPS = (
    "validate-dataset",
    "fp8-shim-scan",
    "adapter-list-mappings",
    "ds4-adapter-inspect",
)

MANUAL_STEPS = ("list-backend-descriptions",)
COMMAND_STEPS = MLX_STEPS + TORCH_MPS_STEPS + REMOTE_CUDA_STEPS + CPU_CHECK_STEPS + MANUAL_STEPS
BACKEND_STEPS: dict[str, tuple[str, ...]] = {
    "local-mlx": MLX_STEPS,
    "local-torch-mps": TORCH_MPS_STEPS,
    "remote-cuda": REMOTE_CUDA_STEPS,
    "cpu-check": CPU_CHECK_STEPS,
    "manual": MANUAL_STEPS,
}
DEFAULT_BACKEND_STEPS: dict[str, tuple[str, ...]] = {
    # Raw `convert` is kept as an explicit diagnostic step, but the default
    # local path must route through FP8 emulation before mlx-lm conversion.
    "local-mlx": tuple(step for step in MLX_STEPS if step != "convert"),
    "local-torch-mps": TORCH_MPS_STEPS,
    "remote-cuda": REMOTE_CUDA_STEPS,
    "cpu-check": CPU_CHECK_STEPS,
    "manual": MANUAL_STEPS,
}

DS4_GGUF_BASE_SMOKE_MARKER = ".ds4-gguf-generate-ok"
DS4_GGUF_BASE_SMOKE_REPORT = "ds4-gguf-base-smoke.json"
DS4_GGUF_BASE_SMOKE_GATE = "ds4-gguf-base-smoke"
DS4_GGUF_BASE_SMOKE_ENV = "DS4_GGUF_BASE_SMOKE"
DS4_GGUF_BASE_SMOKE_PROMPT = "Answer in one word: ready?"
DS4_GGUF_BASE_SMOKE_TOKENS = 12
DS4_GGUF_BASE_SMOKE_TOKEN_CEILING = 32
DS4_GGUF_BASE_SMOKE_TIMEOUT = 900

FORWARD_PARITY_FIXTURE_NAMES = (
    "embedding-rmsnorm-head",
    "sliding-attention-no-compressor-no-rope",
    "rope-tail-pairwise",
    "sink-cache-inverse-rope-hca-bias",
    "hyperconnection-hc1-collapse",
    "hyperconnection-hc2-pre-post-comb",
    "hyperconnection-hc2-transformers",
    "csa-compressor-forward",
    "csa-topk-indexer-gather-mask",
    "hca-compressor-forward",
    "final-hyperhead-hc2-collapse",
    "topk-moe-unquantized",
    "hash-moe-tid2eid-unquantized",
    "topk-moe-i8-block-scale",
    "integrated-layer-attention-moe",
    "csa-indexer-scorer-topk",
    "hyperhead-transformers-reference",
    "integrated-layer-hc-mult",
    "csa-compressor-indexer-attention-mlx",
)
FORWARD_PARITY_BLOCKER_DESCRIPTIONS = (
    "full attention parity with RoPE/cache/sinks/compressor/indexer",
    "full decoder-layer hyperconnection residual mixing and final hyperhead parity",
    "full MoE parity with packed FP4/I8 expert dequant and expert kernels",
    "full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)",
)
REAL_MODE_FORWARD_TOLERANCE = 1e-5
TIE_FREE_MARGIN_EPS = 1e-4
STATEFUL_DECODE_SEAM_TOKEN_LIST = (
    "cache",
    "kv",
    "kvcache",
    "past",
    "state",
    "stateful",
    "offset",
    "position",
    "step",
    "incremental",
    "decode",
    "generate",
)
STATEFUL_DECODE_SEAM_TOKENS = frozenset(STATEFUL_DECODE_SEAM_TOKEN_LIST)
REAL_MODE_PROOFS_NOT_COVERED = (
    "strict index_topk pruning inside production real-mode forward (only proven csa-attention-sublayer-primitive granularity)",
    "real cache / sliding-window stateful forward",
    "learnable attention sink parity in real-mode forward",
    "HCA compressor + non-tiny CSA configs wired into real-mode forward",
    "hc_mult>1 multi-layer real-mode forward (fail-closed B1)",
    "real packed FP4 expert dequant + real-checkpoint-payload decode + expert parallel kernels (B2 remainder; B2-a-1/B2-a-2/B2-a-3 partially prove synthetic single-layer, multi-layer, and top-k>1 multi-expert I8 block-scale dequant integrated into the real-mode MoE forward)",
    "real shimmed-checkpoint load/forward + MLX generation smoke (B3)",
)
B0_REAL_MODE_PROOF_SPECS = (
    {
        "id": "B0a-1",
        "name": "real-mode-single-layer-hc1",
        "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "final_projection": False},
        "reference": "_integrated_layer_forward",
        "covered": [
            "real-mode Model construction (_validate_real_mode passes)",
            "public load_weights() -> _load_real_weights (flat keys)",
            "__call__ -> _real_forward -> _real_layer_forward",
            "single-layer attention + MoE + hc_mult=1 residual gates",
            "hc_mult=1 stream-0 collapse",
        ],
        "not_covered": [
            "compressors (CSA/HCA) in real-mode forward",
            "cache/sliding-window state",
            "attention sink parity",
            "multi-layer stacking",
            "final norm/lm_head projection",
            "packed FP4/I8 dequant",
            "real checkpoint load",
        ],
    },
    {
        "id": "B0a-2",
        "name": "real-mode-multilayer-hc1",
        "config": {"num_hidden_layers": 2, "hc_mult": 1, "compression_ratio": 0, "final_projection": True},
        "reference": "_integrated_multilayer_forward",
        "covered": [
            "real-mode Model construction (_validate_real_mode passes)",
            "public load_weights() -> _load_real_weights (layers.{i}.* prefixed keys)",
            "__call__ -> _real_forward -> stacked _real_layer_forward",
            "multi-layer attention + MoE stacking with hc_mult=1",
            "final norm.weight + lm_head.weight projection",
        ],
        "not_covered": [
            "compressors (CSA/HCA) in real-mode forward",
            "cache/sliding-window state",
            "attention sink parity",
            "hc_mult>1 multi-layer stacking",
            "packed FP4/I8 dequant",
            "real checkpoint load",
        ],
    },
    {
        "id": "B0a-3",
        "name": "real-mode-single-layer-hc2-hyperhead",
        "config": {"num_hidden_layers": 1, "hc_mult": 2, "compression_ratio": 0, "final_projection": False},
        "reference": "_integrated_layer_forward+tiny_hyperhead_collapse",
        "covered": [
            "real-mode Model construction (_validate_real_mode passes)",
            "public load_weights() -> _load_real_weights (flat keys + hc_head)",
            "__call__ -> _real_forward -> _real_layer_forward",
            "single-layer hc_mult=2 decoder-layer residual mixing",
            "final hc_head hyperhead collapse",
        ],
        "not_covered": [
            "compressors (CSA/HCA) in real-mode forward",
            "cache/sliding-window state",
            "attention sink parity",
            "hc_mult>1 multi-layer stacking",
            "final norm/lm_head projection after hc_head",
            "packed FP4/I8 dequant",
            "real checkpoint load",
        ],
    },
    {
        "id": "B0b-a-1",
        "name": "real-mode-single-layer-csa-compressed",
        "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 4, "final_projection": False, "seq_len": 8},
        "reference": "_integrated_layer_forward",
        "covered": [
            "real-mode Model construction (_validate_real_mode passes CSA tiny subset)",
            "public load_weights() -> _load_real_weights (sliding keys removed, 10 CSA keys required)",
            "__call__ -> _real_forward -> _real_layer_forward",
            "_attention_mlx compression_ratio!=0 dispatch -> _csa_attention_mlx",
            "_csa_compressor_mlx (value compressor) wired in real-mode forward",
            "_csa_indexer_mlx (indexer compressor + scorer) wired in real-mode forward",
            "cache-less compressed block attention all-masked early-query zeroing",
            "single-layer attention + MoE + hc_mult=1 residual gates",
        ],
        "not_covered": [
            "real cache / sliding-window stateful CSA (Ca carry) forward",
            "learnable attention sink parity",
            "multi-layer stacking",
            "final norm/lm_head projection",
            "explicit top-k pruning beyond default index_topk",
            "hc_mult>1 multi-layer (B1)",
            "packed FP4/I8 dequant (B2)",
            "real checkpoint load + generation (B3)",
        ],
    },
    {
        "id": "B0b-a-2",
        "name": "csa-topk-sparse-selection-primitive",
        "granularity": "csa-attention-sublayer-primitive",
        "config": {
            "num_hidden_layers": 1,
            "hc_mult": 1,
            "compression_ratio": 4,
            "final_projection": False,
            "seq_len": 12,
            "index_topk": 1,
            "compressed_len": 3,
        },
        "reference": "tiny_compressor_indexer_attention_reference(index_topk=1)",
        "covered": [
            "real-mode CSA attention PRIMITIVE _csa_attention_mlx(index_topk<compressed_len)",
            "_csa_indexer_mlx strict fine-grained top-k selection (DSA lightning-indexer mechanic)",
            "top_k min(index_topk, compressed_len) prunes to strict subset (1 of 3 blocks)",
            "parity vs trusted pure-Python tiny_compressor_indexer_attention_reference at same index_topk",
            "non-degeneracy: index_topk=1 output differs index_topk=compressed_len (pruning observable)",
        ],
        "not_covered": [
            "production real-mode forward strict index_topk (Model.__call__/_real_forward do NOT expose index_topk; default index_topk==compressed_len => no pruning during real inference)",
            "real cache / sliding-window stateful CSA (Ca carry) forward",
            "MLA latent c_t^KV + decoupled-RoPE cache semantics",
            "FlashMLA FP8 KV-cache decode",
            "multi-head / non-tiny CSA configs; HCA compressor",
            "hc_mult>1 multi-layer (B1)",
            "packed FP4/I8 dequant (B2)",
            "real checkpoint load + generation (B3)",
        ],
    },
    {
        "id": "B2-a-1",
        "name": "real-mode-single-layer-i8-block-scale-dequant-moe",
        "evidence_class": "B2-partial",
        "config": {
            "num_hidden_layers": 1,
            "hc_mult": 1,
            "compression_ratio": 0,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "expert_dtype": "i8",
            "block_size": 16,
            "scale_axis": 1,
            "seq_len": 2,
            "final_projection": False,
        },
        "reference": "_integrated_layer_forward",
        "isolation_reference": "real-mode _real_forward(expert_dtype='fp4' raw branch) fed production-dequantized float experts",
        "covered": [
            "real-mode Model(expert_dtype='i8') construction (_validate_real_mode passes; no gate lift)",
            "public load_weights() -> _load_real_weights requires per-expert .scale keys (i8 loader-gated)",
            "__call__ -> _real_forward -> _real_layer_forward -> _moe_mlx I8 branch (_dequantize_i8_block_scale_mlx, block_size=16, axis=1)",
            "single-layer attention + I8-block-scale-dequant MoE + hc_mult=1 residual gates at hidden_size=16",
            "dequant-integration ISOLATION (wiring/dispatch, not arithmetic): i8 branch == raw branch fed numerically-identical dequantized floats (max_abs_error <= 1e-5)",
            "independent trusted reference: == pure-Python _integrated_layer_forward fed the same dequantized floats (<=1e-3, established I8 tolerance)",
            "non-degenerate I8 quantization: signed int8 weights + non-unit BF16 block scale; dequant differs from the true float (quantization_gap > 1e-5)",
        ],
        "not_covered": [
            "packed FP4 expert dequant (dequantize_expert_packed('fp4') still raises; B2 remainder)",
            "real checkpoint payload decode (synthetic deterministic I8 payload only)",
            "expert parallel kernels",
            "multi-layer I8 stacking (num_hidden_layers>1) and hc_mult>1 (B1)",
            "shimmed-checkpoint load/forward + MLX generation smoke (B3)",
        ],
    },
    {
        "id": "B2-a-2",
        "name": "real-mode-multilayer-i8-block-scale-dequant-moe",
        "evidence_class": "B2-partial",
        "config": {
            "num_hidden_layers": 2,
            "hc_mult": 1,
            "compression_ratio": 0,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "expert_dtype": "i8",
            "block_size": 16,
            "scale_axis": 1,
            "seq_len": 2,
            "final_projection": False,
            "secondary_num_hidden_layers": 3,
        },
        "reference": "_integrated_multilayer_forward",
        "isolation_reference": "real-mode _real_forward(expert_dtype='fp4' raw branch) fed per-layer production-dequantized float experts",
        "covered": [
            "real-mode Model(expert_dtype='i8', num_hidden_layers=2) construction (_validate_real_mode passes nl=2/hc=1/i8; no gate lift)",
            "public load_weights() -> _load_real_weights requires per-layer prefixed layers.{i}.*.scale keys (i8 loader-gated, multi-layer)",
            "__call__ -> _real_forward loops range(num_hidden_layers) -> stacked _real_layer_forward -> _moe_mlx I8 branch (_dequantize_i8_block_scale_mlx, block_size=16, axis=1) on EVERY layer",
            "two-layer attention + I8-block-scale-dequant MoE + hc_mult=1 residual gates stacked at hidden_size=16",
            "dequant-integration ISOLATION (stacked wiring/dispatch, not arithmetic): i8 branch == raw branch fed numerically-identical per-layer dequantized floats (max_abs_error <= 1e-5)",
            "independent trusted reference: == pure-Python _integrated_multilayer_forward fed the same per-layer dequantized floats (<=1e-3, established I8 tolerance)",
            "non-degenerate I8 quantization: signed int8 weights + non-unit BF16 block scale; dequant differs from the true float (quantization_gap > 1e-5)",
            "secondary generalization: same composition at num_hidden_layers=3 also gates (isolation <=1e-5, reference <=1e-3)",
        ],
        "not_covered": [
            "packed FP4 expert dequant (dequantize_expert_packed('fp4') still raises; B2 remainder)",
            "real checkpoint payload decode (synthetic deterministic I8 payload only)",
            "expert parallel kernels",
            "hc_mult>1 multi-layer stacking (B1)",
            "shimmed-checkpoint load/forward + MLX generation smoke (B3)",
        ],
    },

    {
        "id": "B2-a-3",
        "name": "real-mode-topk-multi-expert-i8-block-scale-dequant-moe",
        "evidence_class": "B2-partial",
        "config": {
            "num_hidden_layers": 1,
            "hc_mult": 1,
            "compression_ratio": 0,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "expert_dtype": "i8",
            "block_size": 16,
            "scale_axis": 1,
            "seq_len": 2,
            "final_projection": False,
            "o_groups": 1,
            "secondary_num_experts_per_tok": 3,
        },
        "reference": "_integrated_layer_forward",
        "isolation_reference": "real-mode _real_forward(expert_dtype='fp4' raw branch) fed numerically-identical production-dequantized float experts (NE=4, top-k=2)",
        "covered": [
            "real-mode Model(expert_dtype='i8', n_routed_experts=4, num_experts_per_tok=2) construction (_validate_real_mode passes NE<=4/top-k<=NE/i8; no gate lift)",
            "public load_weights() -> _load_real_weights requires all four routed experts' .scale keys (i8 loader-gated, multi-expert)",
            "__call__ -> _real_forward -> _real_layer_forward -> _moe_mlx I8 branch (_dequantize_i8_block_scale_mlx, block_size=16, axis=1) across ALL FOUR routed experts",
            "multi-expert top-k routing combine integrated with I8 dequant: argsort subset selection (top-k=2 of 4), multi-term denom accumulation over selected experts, per-expert factor=scores/denom*selected*routed_scaling_factor normalization, and masked weighted sum with factor=0 for two unselected experts",
            "dequant-integration ISOLATION (wiring/dispatch, not arithmetic): i8 branch == raw branch fed numerically-identical dequantized floats (max_abs_error <= 1e-5)",
            "independent trusted reference: == pure-Python _integrated_layer_forward/_integrated_moe_forward/tiny_topk_moe_forward fed the same dequantized floats (<=1e-3, established I8 tolerance)",
            "routing-primitive certification: _csa_topk_proof_input(seq_len=2, hidden_size=16) + tiny_topk_moe_routing shows a tie-free proper subset, stable float32/float64 top-k subset, distinct selected experts, and at least one unselected expert",
            "secondary generalization: same NE=4 composition at num_experts_per_tok=3 also gates (isolation <=1e-5, reference <=1e-3)",
            "non-degenerate I8 quantization: signed int8 weights + non-unit BF16 block scale; dequant differs from the true float (quantization_gap > 1e-5)",
        ],
        "not_covered": [
            "packed FP4 expert dequant (dequantize_expert_packed('fp4') still raises; B2 remainder)",
            "real checkpoint payload decode (synthetic deterministic I8 payload only)",
            "expert parallel kernels",
            "hc_mult>1 multi-layer stacking (B1)",
            "shimmed-checkpoint load/forward + MLX generation smoke (B3)",
        ],
    },
    {
        "id": "B2-a-4",
        "name": "metal-routed-i8-e8m0-dequant-isolation-proof",
        "evidence_class": "B2-partial-metal",
        "config": {
            "kernel": "metal/moe.metal",
            "kernel_function": "kernel_dsv4_routed_dequant_i8_e8m0_to_bf16",
            "weight_dtype": "I8",
            "scale_dtype": "F8_E8M0",
            "block_size": 16,
            "scale_axis": 1,
            "dedicated": True,
            "fusion": "production path fuses dequant+gemm in kernel Mul_mm_id_i8_e8m0_f32 and kernel_mul_mv_id_i8_e8m0_pair_swiglu_f32; this kernel is the standalone dequant dump for the isolation proof",
        },
        "reference": "OCP MX v1.0 spec (scale=2^(e-127), e=255->NaN block, e=0->2^(-127) subnormal, NO 2^(-6) factor)",
        "isolation_reference": "tests/ds4_e8m0_ocp_witness.py (stands alone, no forbidden OUR-Python imports, uses math.ldexp, cites OCP MX v1.0 spec URL)",
        "covered": [
            "reads real DeepSeek-V4 Flash original-F8 checkpoint I8 weights + F8_E8M0 scales via read_safetensors_header",
            "executes the new Metal kernel kernel_dsv4_routed_dequant_i8_e8m0_to_bf16",
            "computes the Metal decode math: int8 * 2^(e-127) where e is the E8M0 byte, e=255->NaN, e=0->2^(-127) subnormal scale NOT zero, NO infinity encoding",
            "compares Metal output against the OCP-spec witness at max_abs <= 1e-5",
            "verifies NO forbidden OUR-Python symbols imported (grep audit)",
            "verified on real HF checkpoint bytes (snapshot 553034d), NOT synthetic",
            "production decode function ds4_e8m0_decode_i8 (inline device fn) is shared between proof kernel and production fused kernels: no proof/production divergence",
        ],
        "not_covered": [
            "packed FP4 expert dequant (dequantize_expert_packed('fp4') still raises; B2 remainder)",
            "expert parallel kernels (production path _moe_mlx handles matmul but FP4 remains fail-closed)",
            "hc_mult>1 multi-layer stacking (B1)",
            "shimmed-checkpoint load/forward + MLX generation smoke (B3)",
            "production fused kernel matmul+SwiGLU integration (the standalone dump kernel is proof-only)",
            "real-payload dequant inside the full _real_forward MoE branch (the isolation proof uses the standalone dump kernel)",
        ],
    },
)

DEFAULT_PATHS = {
    "DS4_ROOT": "/Users/spotted/projects/ds4",
    "DS4_GGUF": None,
    "HF_MODEL": "/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0",
    "OPUS46_DATA": "/Volumes/Data NVME/huggingface/hub/datasets--Jackrong--Claude-opus-4.6-TraceInversion-9000x/snapshots/dcb98612aa4eb657cddec26ac2047e3f6c454ed3/claude-opus-4.6-traceInversion-9000x.jsonl",
    "OPUS47_DATA": "/Volumes/Data NVME/huggingface/hub/datasets--Jackrong--Claude-opus-4.7-TraceInversion-5000x/snapshots/ab3b48f1d461ec40af924fd3163d2b9c8eaeb07c/claude-opus-4.7-TraceInversion-5000x.jsonl",
    "FABLE5_DATA": "/Volumes/Data NVME/huggingface/hub/datasets--Glint-Research--Fable-5-traces/snapshots/df1160b3b4c6b770c8faaa88ebf8e859ded8b0d6/fable5_cot_merged.jsonl",
    "DATASET_ROOT": "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge",
    "MLX_WORK": "/Volumes/Data NVME/mlx-ft/ds4",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    priority: int
    path: pathlib.Path
    kind: str


class PlanError(RuntimeError):
    pass


class LockFile:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> "LockFile":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                owner = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                owner = "unreadable lock owner"
            raise PlanError(f"lock exists: {self.path} ({owner}); remove it only after confirming no DS4 fine-tune job is running") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(f"pid={os.getpid()}\n")
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def env_or_default(name: str) -> str:
    return os.environ.get(name, DEFAULT_PATHS[name])


def path_arg(value: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(value).expanduser()


def collapse_ws(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def question_key(question: str | None) -> str:
    return collapse_ws(question).casefold()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_leading_think(text: str | None) -> str:
    text = (text or "").lstrip()
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    return text.lstrip()


def strip_trailing_eos_repeats(text: str | None) -> str:
    text = (text or "").rstrip()
    while text.endswith(EOS):
        text = text[: -len(EOS)].rstrip()
    return text


def append_exactly_one_eos(text: str | None) -> str:
    return strip_trailing_eos_repeats(text) + EOS


def has_exactly_one_final_eos(text: str) -> bool:
    if not text.endswith(EOS):
        return False
    return not text[: -len(EOS)].rstrip().endswith(EOS)


def split_for_key(key: str) -> str:
    bucket = int(digest(key)[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def read_jsonl(path: pathlib.Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        fp = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise PlanError(f"{path}: cannot open JSONL: {exc.strerror or exc}") from exc
    with fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise PlanError(f"{path}:{line_no}: expected JSON object, got {type(row).__name__}")
            yield line_no, row


def require_string(row: dict[str, Any], field: str, path: pathlib.Path | None = None, line_no: int | None = None) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        where = f"{path}:{line_no}: " if path is not None and line_no is not None else ""
        raise PlanError(f"{where}expected string field {field!r}, got {type(value).__name__}")
    return value


def validate_source_row(source: SourceSpec, row: dict[str, Any], line_no: int) -> None:
    required = ("input", "inverted_reasoning", "output") if source.kind == "opus" else ("context", "completion")
    for field in required:
        require_string(row, field, source.path, line_no)


def build_sources(args: argparse.Namespace) -> list[SourceSpec]:
    return [
        SourceSpec("opus46", 0, path_arg(args.opus46_data), "opus"),
        SourceSpec("opus47", 1, path_arg(args.opus47_data), "opus"),
        SourceSpec("fable5", 2, path_arg(args.fable5_data), "fable"),
    ]


def convert_record(source: SourceSpec, row: dict[str, Any], source_line: int) -> tuple[str, dict[str, str], dict[str, Any]] | None:
    validate_source_row(source, row, source_line)
    if source.kind == "opus":
        question = row["input"]
        reasoning = strip_leading_think(row["inverted_reasoning"])
        answer = row["output"].strip()
        completion_body = strip_trailing_eos_repeats(f"{reasoning.rstrip()}\n\n{answer}".strip())
        provenance = {
            "source_id": row.get("id"),
            "domain": row.get("domain"),
            "reasoning_bubble_len": len(row.get("reasoning_bubble", "") or ""),
        }
    elif source.kind == "fable":
        question = row["context"]
        completion_body = strip_trailing_eos_repeats(strip_leading_think(row["completion"]))
        provenance = {
            "source_id": row.get("uid"),
            "session": row.get("session"),
            "model": row.get("model"),
            "origin": row.get("origin"),
            "output_type": row.get("output_type"),
            "source_file": row.get("source_file"),
            "cot": row.get("cot"),
            "output": row.get("output"),
        }
    else:
        raise PlanError(f"unknown source kind {source.kind!r}")

    key = question_key(question)
    if not key or not completion_body.strip():
        return None

    completion = append_exactly_one_eos(completion_body)
    prompt = f"{BOS}{USER}{question}{ASSISTANT_THINK}"
    key_hash = digest(key)
    meta = {
        "key_hash": key_hash,
        "source": source.name,
        "source_priority": source.priority,
        "source_line": source_line,
        "question_len_chars": len(question),
        "prompt_len_chars": len(prompt),
        "completion_len_chars": len(completion),
        **provenance,
    }
    return key, {"prompt": prompt, "completion": completion}, meta


def preflight(args: argparse.Namespace) -> int:
    hf_model = path_arg(args.hf_model)
    if not hf_model.is_dir():
        raise PlanError(f"{hf_model}: HF model path is not a directory")
    required_hf = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
    missing = [str(hf_model / name) for name in required_hf if not (hf_model / name).is_file()]

    index_path = hf_model / "model.safetensors.index.json"
    shard_names: set[str] = set()
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PlanError(f"{index_path}: invalid JSON: {exc}") from exc
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise PlanError(f"{index_path}: expected non-empty weight_map")
        shard_names = {str(name) for name in weight_map.values()}
        missing.extend(str(hf_model / name) for name in sorted(shard_names) if not (hf_model / name).is_file())

    source_rows: dict[str, int] = {}
    for source in build_sources(args):
        if not source.path.is_file():
            missing.append(str(source.path))
            continue
        rows = 0
        for line_no, row in read_jsonl(source.path):
            validate_source_row(source, row, line_no)
            rows += 1
        if rows == 0:
            raise PlanError(f"{source.path}: no JSONL rows found")
        source_rows[source.name] = rows

    if missing:
        for item in missing:
            print(f"missing: {item}", file=sys.stderr)
        return 1

    config = json.loads((hf_model / "config.json").read_text(encoding="utf-8"))
    config_summary = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "n_routed_experts": config.get("n_routed_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "max_position_embeddings": config.get("max_position_embeddings"),
    }

    print(json.dumps({"ok": True, "hf_model": str(hf_model), "safetensor_shards": len(shard_names), "source_rows": source_rows, "config": config_summary}, indent=2))
    return 0


def build_dataset(args: argparse.Namespace) -> int:
    dataset_root = path_arg(args.dataset_root)
    out_dir = dataset_root / args.split_dir
    meta_dir = dataset_root / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    sources = build_sources(args)
    records: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "source_rows": {s.name: 0 for s in sources},
        "empty_rejected": {s.name: 0 for s in sources},
        "internal_replacements": {s.name: 0 for s in sources},
        "cross_source_replacements": {},
    }
    source_by_key: dict[str, str] = {}

    for source in sources:
        for line_no, row in read_jsonl(source.path):
            stats["source_rows"][source.name] += 1
            converted = convert_record(source, row, line_no)
            if converted is None:
                stats["empty_rejected"][source.name] += 1
                continue
            key, training, meta = converted
            previous_source = source_by_key.get(key)
            if previous_source == source.name:
                stats["internal_replacements"][source.name] += 1
            elif previous_source is not None:
                label = f"{previous_source}->{source.name}"
                stats["cross_source_replacements"][label] = stats["cross_source_replacements"].get(label, 0) + 1
            records[key] = {"training": training, "meta": meta}
            source_by_key[key] = source.name

    split_counts = {"train": 0, "valid": 0, "test": 0}
    kept_by_source: dict[str, int] = {s.name: 0 for s in sources}
    max_prompt = 0
    max_completion = 0
    meta_path = meta_dir / "records-meta.jsonl"

    handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in split_counts}
    try:
        with meta_path.open("w", encoding="utf-8") as meta_fp:
            for key in sorted(records):
                item = records[key]
                split = split_for_key(key)
                row = item["training"]
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                split_counts[split] += 1
                src = item["meta"]["source"]
                kept_by_source[src] = kept_by_source.get(src, 0) + 1
                max_prompt = max(max_prompt, len(row["prompt"]))
                max_completion = max(max_completion, len(row["completion"]))
                meta = {**item["meta"], "split": split}
                meta_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
    finally:
        for fp in handles.values():
            fp.close()

    manifest = {
        "script_version": SCRIPT_VERSION,
        "name": "anthropomorphic-frankenmerge",
        "format": "mlx_lm prompt/completion jsonl",
        "dedupe": "question only: casefold(collapse_whitespace(strip(question))); newest source wins",
        "source_order_oldest_to_newest": [s.name for s in sources],
        "source_paths": {s.name: str(s.path) for s in sources},
        "stats": stats,
        "final_unique_records": len(records),
        "kept_by_source": kept_by_source,
        "split_counts": split_counts,
        "max_prompt_len_chars": max_prompt,
        "max_completion_len_chars": max_completion,
        "split_rule": "sha256(question_key)[:8] % 100: <90 train, <95 valid, else test",
        "prompt_template": f"{BOS}{USER}{{question}}{ASSISTANT_THINK}",
        "completion_rule": "strip one leading <think>, strip duplicate trailing EOS, append exactly one <｜end▁of▁sentence｜>",
        "eos_token": EOS,
        "assistant_think_marker": ASSISTANT_THINK,
        "split_dir": str(out_dir),
        "meta_path": str(meta_path),
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def validate_training_row(path: pathlib.Path, line_no: int, row: dict[str, Any]) -> tuple[int, int]:
    if set(row) != {"prompt", "completion"}:
        raise PlanError(f"{path}:{line_no}: expected only prompt/completion keys, got {sorted(row)}")
    prompt = row["prompt"]
    completion = row["completion"]
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise PlanError(f"{path}:{line_no}: prompt and completion must be strings")
    if not prompt.startswith(BOS + USER):
        raise PlanError(f"{path}:{line_no}: prompt missing BOS/User prefix")
    if not prompt.endswith(ASSISTANT_THINK):
        raise PlanError(f"{path}:{line_no}: prompt does not end with assistant thinking marker")
    user_text = prompt[len(BOS + USER) : -len(ASSISTANT_THINK)]
    if not user_text.strip():
        raise PlanError(f"{path}:{line_no}: empty user text in prompt")
    if not has_exactly_one_final_eos(completion):
        raise PlanError(f"{path}:{line_no}: completion must end with exactly one EOS token")
    if not strip_trailing_eos_repeats(completion).strip():
        raise PlanError(f"{path}:{line_no}: empty completion body before EOS")
    return len(prompt), len(completion)


def validate_dataset(args: argparse.Namespace) -> int:
    dataset_root = path_arg(args.dataset_root)
    root = dataset_root / args.split_dir
    summary: dict[str, dict[str, int]] = {}
    total_rows = 0
    for split in ("train", "valid", "test"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise PlanError(f"{path}: missing split file")
        n = 0
        max_prompt = 0
        max_completion = 0
        for line_no, row in read_jsonl(path):
            prompt_len, completion_len = validate_training_row(path, line_no, row)
            n += 1
            max_prompt = max(max_prompt, prompt_len)
            max_completion = max(max_completion, completion_len)
        summary[split] = {"rows": n, "max_prompt_chars": max_prompt, "max_completion_chars": max_completion}
        total_rows += n
        print(f"{split}: rows={n} max_prompt_chars={max_prompt} max_completion_chars={max_completion}")

    manifest_path = dataset_root / "manifest.json"
    meta_path = dataset_root / "meta" / "records-meta.jsonl"
    if not manifest_path.is_file():
        raise PlanError(f"{manifest_path}: missing manifest")
    if not meta_path.is_file():
        raise PlanError(f"{meta_path}: missing metadata JSONL")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_counts = manifest.get("split_counts")
    actual_counts = {split: summary[split]["rows"] for split in summary}
    if expected_counts != actual_counts:
        raise PlanError(f"{manifest_path}: split_counts {expected_counts} do not match actual {actual_counts}")
    if manifest.get("final_unique_records") != total_rows:
        raise PlanError(f"{manifest_path}: final_unique_records {manifest.get('final_unique_records')} does not match actual {total_rows}")
    meta_rows = 0
    meta_split_counts = {"train": 0, "valid": 0, "test": 0}
    seen_key_hashes: set[str] = set()
    required_meta = {"key_hash", "source", "source_priority", "source_line", "split"}
    for line_no, meta in read_jsonl(meta_path):
        missing = required_meta - set(meta)
        if missing:
            raise PlanError(f"{meta_path}:{line_no}: missing metadata keys {sorted(missing)}")
        split = meta["split"]
        if split not in meta_split_counts:
            raise PlanError(f"{meta_path}:{line_no}: invalid split {split!r}")
        key_hash = meta["key_hash"]
        if not isinstance(key_hash, str) or not key_hash:
            raise PlanError(f"{meta_path}:{line_no}: invalid key_hash")
        if key_hash in seen_key_hashes:
            raise PlanError(f"{meta_path}:{line_no}: duplicate key_hash {key_hash}")
        seen_key_hashes.add(key_hash)
        meta_split_counts[split] += 1
        meta_rows += 1
    if meta_rows != total_rows:
        raise PlanError(f"{meta_path}: metadata rows {meta_rows} do not match split rows {total_rows}")
    if meta_split_counts != actual_counts:
        raise PlanError(f"{meta_path}: metadata split counts {meta_split_counts} do not match actual {actual_counts}")
    return 0


def token_audit(args: argparse.Namespace) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PlanError("token-audit requires transformers installed in the active environment") from exc

    tokenizer = AutoTokenizer.from_pretrained(path_arg(args.hf_model), trust_remote_code=not args.no_trust_remote_code)
    root = path_arg(args.dataset_root) / args.split_dir
    summary: dict[str, dict[str, int]] = {}
    any_over = False
    for split in ("train", "valid", "test"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise PlanError(f"{path}: missing split file")
        n = 0
        over = 0
        max_tokens = 0
        for _, row in read_jsonl(path):
            ids = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False).input_ids
            ntok = len(ids)
            n += 1
            over += int(ntok > args.max_seq_length)
            max_tokens = max(max_tokens, ntok)
        any_over = any_over or over > 0
        summary[split] = {"rows": n, f"over_{args.max_seq_length}": over, "max_tokens": max_tokens}
        print(f"{split}: rows={n} over_{args.max_seq_length}={over} max_tokens={max_tokens}")
    print(json.dumps({"max_seq_length": args.max_seq_length, "splits": summary}, indent=2))
    if any_over and not (args.report_only or args.allow_over_limit):
        print("token audit found over-limit rows; rerun with --report-only to inspect or --allow-over-limit after an explicit operator decision", file=sys.stderr)
        return 3
    return 0


def q(value: str | pathlib.Path) -> str:
    return shlex.quote(str(value))


def command_catalog(args: argparse.Namespace) -> dict[str, list[str]]:
    hf = path_arg(args.hf_model)
    dataset_root = path_arg(args.dataset_root)
    mlx_work = path_arg(args.mlx_work)
    ds4_root = path_arg(args.ds4_root)
    split_dir = dataset_root / args.split_dir
    fused_hf = path_arg(args.fused_hf_model) if args.fused_hf_model else pathlib.Path("/path/to/fused-hf-safetensors-model")
    imatrix = path_arg(args.ds4_imatrix) if args.ds4_imatrix else pathlib.Path("/path/to/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat")
    q2 = ds4_root / "gguf/anthropomorphic-frankenmerge-q2.gguf"
    q4 = ds4_root / "gguf/anthropomorphic-frankenmerge-q4.gguf"
    mixed = ds4_root / "gguf/anthropomorphic-frankenmerge-ds4flash.gguf"
    ds4_gguf = path_arg(args.ds4_gguf) if args.ds4_gguf else ds4_root / "ds4flash.gguf"
    activate = f"unset SSLKEYLOGFILE && . {q(mlx_work / '.venv/bin/activate')}"
    activate_torch = f"unset SSLKEYLOGFILE && . {q(mlx_work / '.venv-torch/bin/activate')}"
    shimmed_hf = mlx_work / "hf-f8shim"
    torch_smoke_script = mlx_work / "torch_lora_smoke.py"
    torch_one_step_script = mlx_work / "torch_one_step_lora.py"
    torch_real_v4_script = mlx_work / "torch_real_v4_feasibility.py"
    adapter_ds4 = path_arg(getattr(args, "adapter_ds4", None)) if getattr(args, "adapter_ds4", None) else mlx_work / "adapter.ds4.safetensors"
    lora_config = mlx_work / "lora-config.json"
    project_root = pathlib.Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"
    return {
        "setup-env": [f"mkdir -p {q(mlx_work)} && cd {q(mlx_work)} && {{ test -d .venv || uv venv --seed .venv; }} && {activate} && pip install -U pip && pip install -e {q(project_root / 'python-envs' / 'mlx')} && python -c {q('from ds4_ft_mlx.mlx_lm_plugin import install_startup_pth_hook; print(install_startup_pth_hook())')}"],
        "mlx-device": [f"cd {q(mlx_work)} && {activate} && python - <<'PY'\nimport mlx.core as mx\nprint(mx.default_device())\nPY"],
        "deepseek-v4-import-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-import-check --mlx-work {q(mlx_work)}"],
        "deepseek-v4-tiny-config-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-tiny-config-check --mlx-work {q(mlx_work)}"],
        "deepseek-v4-mtp-exclusion-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-mtp-exclusion-check --mlx-work {q(mlx_work)}"],
        "deepseek-v4-mapping-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-mapping-check --mlx-work {q(mlx_work)}"],
        "deepseek-v4-dequant-parity-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-dequant-parity-check --mlx-work {q(mlx_work)}"],
        "deepseek-v4-forward-parity-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-forward-parity-check --mlx-work {q(mlx_work)}"],
        "convert": [f"cd {q(mlx_work)} && {activate} && mlx_lm.convert --model {q(hf)} -q --mlx-path {q(mlx_work / 'model-4bit')}", f"test -d {q(mlx_work / 'model-4bit')}"],
        "fp8-shim-scan": [f"python3 {q(scripts_dir / 'shim_ds4_safetensors.py')} scan --src {q(hf)}"],
        "fp8-shim-probe": [f"rm -rf {q(mlx_work / 'hf-f8shim-probe')} {q(mlx_work / '.fp8-shim-probe-ok')} && python3 {q(scripts_dir / 'shim_ds4_safetensors.py')} convert --src {q(hf)} --dst {q(mlx_work / 'hf-f8shim-probe')} --out-dtype BF16 --shard-limit 2 --execute --yes --force", f"cd {q(mlx_work)} && {activate} && set -e && python - <<'PY'\nfrom pathlib import Path\nimport mlx.core as mx\npaths = sorted(Path('hf-f8shim-probe').glob('model*.safetensors'))\nif not paths:\n    raise SystemExit('no shimmed probe shards found')\nfor p in paths:\n    w = mx.load(str(p))\n    print(p, len(w))\nPY\ntouch {q(mlx_work / '.fp8-shim-probe-ok')}"],
        "fp8-shim": [f"mkdir -p {q(mlx_work)} && python3 {q(scripts_dir / 'shim_ds4_safetensors.py')} convert --src {q(hf)} --dst {q(shimmed_hf)} --out-dtype BF16 --execute --yes --force"],
        "convert-shimmed": [f"cd {q(mlx_work)} && {activate} && mlx_lm.convert --model {q(shimmed_hf)} -q --mlx-path {q(mlx_work / 'model-4bit')}", f"test -d {q(mlx_work / 'model-4bit')}"],
        "model-4bit-conversion-plan": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} model-4bit-conversion-plan --mlx-work {q(mlx_work)}"],
        "deepseek-v4-forward-parity-readiness": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} deepseek-v4-forward-parity-readiness --mlx-work {q(mlx_work)}"],
        "mlx-lora-targets-check": [f"cd {q(project_root)} && {activate} && python3 {q(pathlib.Path(__file__))} mlx-lora-targets-check --mlx-work {q(mlx_work)}"],
        "smoke-train": [f"cd {q(mlx_work)} && {activate} && mlx_lm.lora --config {q(lora_config)} --model {q(mlx_work / 'model-4bit')} --train --data {q(split_dir)} --adapter-path {q(mlx_work / 'adapters-smoke')} --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint"],
        "smoke-train-2048": [f"cd {q(mlx_work)} && {activate} && mlx_lm.lora --config {q(lora_config)} --model {q(mlx_work / 'model-4bit')} --train --data {q(split_dir)} --adapter-path {q(mlx_work / 'adapters-smoke-2048')} --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 2048 --mask-prompt --grad-checkpoint"],
        "smoke-generate": [f"cd {q(mlx_work)} && {activate} && mlx_lm.generate --model {q(mlx_work / 'model-4bit')} --adapter-path {q(mlx_work / 'adapters-smoke')} --prompt {q(BOS + USER + 'Give a short answer about why careful dataset deduplication matters.' + ASSISTANT_THINK)} --max-tokens 300 && touch {q(mlx_work / 'adapters-smoke/.generation-smoke-ok')}"],
        "convert-smoke-adapter": [f"python3 {q(scripts_dir / 'convert_lora_to_ds4.py')} {q(mlx_work / 'adapters-smoke' / 'adapters.safetensors')} {q(mlx_work / 'adapters-smoke' / 'adapter.ds4.safetensors')}"],
        "ds4-smoke-adapter-inspect": [f"cd {q(ds4_root)} && ./ds4 --inspect -m {q(ds4_gguf)} --lora {q(mlx_work / 'adapters-smoke' / 'adapter.ds4.safetensors')} && touch {q(mlx_work / 'adapters-smoke' / '.ds4-inspect-ok')}"],
        "full-train": [f"cd {q(mlx_work)} && {activate} && mlx_lm.lora --config {q(lora_config)} --model {q(mlx_work / 'model-4bit')} --train --data {q(split_dir)} --adapter-path {q(mlx_work / 'adapters')} --fine-tune-type lora --iters 5000 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint --steps-per-report 10 --steps-per-eval 200"],
        "continue-train": [f"cd {q(mlx_work)} && {activate} && mlx_lm.lora --config {q(lora_config)} --model {q(mlx_work / 'model-4bit')} --train --data {q(split_dir)} --adapter-path {q(mlx_work / 'adapters')} --resume-adapter-file {q(mlx_work / 'adapters/adapters.safetensors')} --fine-tune-type lora --iters 15000 --batch-size 1 --learning-rate 5e-6 --max-seq-length 4096 --mask-prompt --grad-checkpoint --steps-per-report 10 --steps-per-eval 200"],
        "eval": [f"cd {q(mlx_work)} && {activate} && mlx_lm.lora --model {q(mlx_work / 'model-4bit')} --adapter-path {q(mlx_work / 'adapters')} --data {q(split_dir)} --test"],
        "fuse": [f"cd {q(mlx_work)} && {activate} && mlx_lm.fuse --model {q(mlx_work / 'model-4bit')} --adapter-path {q(mlx_work / 'adapters')} --save-path {q(mlx_work / 'fused-model')}"],
        "fuse-hf": [f"python3 {q(scripts_dir / 'fuse_lora_hf.py')} --base {q(hf)} --adapter {q(mlx_work / 'adapters' / 'adapters.safetensors')} --out {q(mlx_work / 'fused-hf')}"],
        "fused-generate": [f"cd {q(mlx_work)} && {activate} && mlx_lm.generate --model {q(mlx_work / 'fused-model')} --prompt {q(BOS + USER + 'Test the fused model briefly.' + ASSISTANT_THINK)} --max-tokens 300"],
        "build-quantizer": [f"cd {q(ds4_root)} && make -C gguf-tools"],
        "quantize-q2": [f"{q(ds4_root / 'gguf-tools/deepseek4-quantize')} --hf {q(fused_hf)} --template {q(ds4_root / 'gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf')} --out {q(q2)} --imatrix {q(imatrix)}"],
        "quantize-q4": [f"{q(ds4_root / 'gguf-tools/deepseek4-quantize')} --hf {q(fused_hf)} --template {q(ds4_root / 'gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf')} --out {q(q4)} --imatrix {q(imatrix)}"],
        "splice-dry-run": [f"python3 {q(ds4_root / 'gguf-tools/mixed/splice_mixed_expert_layers_gguf.py')} --base {q(q2)} --donor {q(q4)} --q4-layers 37-42 --out {q(mixed)} --dry-run"],
        "splice": [f"python3 {q(ds4_root / 'gguf-tools/mixed/splice_mixed_expert_layers_gguf.py')} --base {q(q2)} --donor {q(q4)} --q4-layers 37-42 --out {q(mixed)}"],
        "ds4-smoke": [f"cd {q(ds4_root)} && ./ds4 -m {q(mixed)} -p {q('Explain why deduplicating by question matters for supervised fine-tuning.')} -n 300"],
        "torch-env-create": [
            f"mkdir -p {q(mlx_work)}",
            f"cd {q(mlx_work)} && uv venv --seed --python 3.12 --clear .venv-torch",
            f"cd {q(mlx_work)} && {activate_torch} && pip install -U pip",
            f"cd {q(mlx_work)} && {activate_torch} && pip install -e {q(project_root / 'python-envs' / 'torch')}",
        ],
        "torch-env-check": [
            f"cd {q(mlx_work)} && {activate_torch} && python - <<'PY'\nimport sys, torch, transformers, peft, trl, accelerate, datasets, safetensors, huggingface_hub\nprint('torch', torch.__version__, 'mps', torch.backends.mps.is_available())\nif not torch.backends.mps.is_available():\n    print('local-torch-mps requires torch.backends.mps.is_available()', file=sys.stderr); sys.exit(3)\nprint('peft', peft.__version__)\nPY",
        ],
        "torch-smoke": [
            f"cat > {q(torch_smoke_script)} <<'PY'\n" + _torch_smoke_script() + "\nPY",
            f"cd {q(mlx_work)} && export HF_MODEL={q(hf)} DATASET_ROOT={q(dataset_root)} DS4_REPO={q(project_root)} && {activate_torch} && python {q(torch_smoke_script)}",
        ],
        "torch-one-step-lora": [
            f"cat > {q(torch_one_step_script)} <<'PY'\n" + _torch_one_step_lora_script() + "\nPY",
            f"cd {q(mlx_work)} && export HF_MODEL={q(hf)} DATASET_ROOT={q(dataset_root)} SPLIT_DIR={q(args.split_dir)} DS4_REPO={q(project_root)} && {activate_torch} && python {q(torch_one_step_script)}",
        ],
        "torch-real-v4-feasibility": [
            f"cat > {q(torch_real_v4_script)} <<'PY'\n" + _torch_real_v4_feasibility_script() + "\nPY",
            f"cd {q(mlx_work)} && export HF_MODEL={q(hf)} DATASET_ROOT={q(dataset_root)} && {activate_torch} && python {q(torch_real_v4_script)}",
        ],
        "remote-cuda-smoke": [
            f"cat > {q(mlx_work / 'remote_cuda_smoke.sh')} <<'SH'\n" + _remote_cuda_smoke_script() + "\nSH",
            f"chmod +x {q(mlx_work / 'remote_cuda_smoke.sh')}",
            f"# Run on CUDA host, then proceed to remote-cuda-full only after success",
        ],
        "remote-cuda-full": [
            f"# remote-cuda-full: train adapter on a CUDA host with PEFT/TRL, export adapter.safetensors,",
            f"# then run scripts/convert_lora_to_ds4.py and validate with ./ds4 --inspect -m {q(ds4_gguf)} --lora {q(adapter_ds4)}",
        ],
        "adapter-list-mappings": [
            f"python3 {q(scripts_dir / 'convert_lora_to_ds4.py')} --list-mappings",
        ],
        "ds4-adapter-inspect": [
            f"cd {q(ds4_root)} && ./ds4 --inspect -m {q(ds4_gguf)} --lora {q(adapter_ds4)}",
        ],
        "list-backend-descriptions": [
            f"python3 {q(pathlib.Path(__file__))} --help",
        ],
        "validate-dataset": [
            f"python3 {q(pathlib.Path(__file__))} validate-dataset --dataset-root {q(dataset_root)} --split-dir {q(args.split_dir)}",
        ],
    }


def emit_commands(args: argparse.Namespace) -> int:
    backend = args.backend
    if backend not in BACKEND_STEPS:
        raise PlanError(f"unknown backend {backend!r}")
    allowed_steps = BACKEND_STEPS[backend]
    catalog = command_catalog(args)
    steps = args.steps or list(DEFAULT_BACKEND_STEPS[backend])
    missing = [step for step in steps if step not in catalog]
    if missing:
        raise PlanError(f"unknown command step(s): {', '.join(missing)}")
    disallowed = [step for step in steps if step not in allowed_steps]
    if disallowed:
        raise PlanError(f"step(s) {', '.join(disallowed)!r} are not available for backend {backend!r}; use one of {', '.join(allowed_steps)}")
    selected = {step: catalog[step] for step in steps}
    if args.format == "json":
        print(json.dumps({"backend": backend, "default_note": "Mac Studio M3 Ultra local resources preferred", "steps": selected}, indent=2))
    else:
        print(f"# backend={backend}")
        print("# Mac Studio M3 Ultra local resources preferred; remote CUDA requires explicit backend selection")
        print("set -euo pipefail")
        print()
        for step, commands in selected.items():
            print(f"# {step}")
            print("(")
            print("set -euo pipefail")
            for command in commands:
                print(command)
            print(")")
            print()
    return 0


def validate_hf_safetensors_dir(path: pathlib.Path) -> int:
    if not path.is_dir():
        raise PlanError(f"{path}: HF safetensors path is not a directory")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        if not (path / name).is_file():
            raise PlanError(f"{path / name}: missing required HF file")
    index_path = path / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{index_path}: invalid JSON: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PlanError(f"{index_path}: expected non-empty weight_map")
    shards = {str(name) for name in weight_map.values()}
    for shard in sorted(shards):
        if not (path / shard).is_file():
            raise PlanError(f"{path / shard}: missing referenced safetensors shard")
    return len(shards)


def _safetensors_dtype(path: pathlib.Path, tensor_name: str) -> str:
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise PlanError(f"{path}: invalid safetensors header")
            header_len = struct.unpack("<Q", raw_len)[0]
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    except OSError as exc:
        raise PlanError(f"{path}: cannot read safetensors header: {exc}") from exc
    meta = header.get(tensor_name)
    if not isinstance(meta, dict) or not isinstance(meta.get("dtype"), str):
        raise PlanError(f"{path}:{tensor_name}: missing dtype in safetensors header")
    return str(meta["dtype"])


def _safetensors_dtype_map(path: pathlib.Path) -> dict[str, str]:
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise PlanError(f"{path}: invalid safetensors header")
            header_len = struct.unpack("<Q", raw_len)[0]
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    except OSError as exc:
        raise PlanError(f"{path}: cannot read safetensors header: {exc}") from exc
    out: dict[str, str] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if isinstance(meta, dict) and isinstance(meta.get("dtype"), str):
            out[str(name)] = str(meta["dtype"])
    return out


def validate_fused_hf_safetensors_dir(path: pathlib.Path) -> int:
    validate_hf_safetensors_dir(path)
    index_path = path / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{index_path}: invalid JSON: {exc}") from exc
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise PlanError(f"{index_path}: expected non-empty weight_map")

    dtype_cache: dict[pathlib.Path, dict[str, str]] = {}

    def _dtype_lookup(shard_path: pathlib.Path, tensor_name: str) -> str:
        if shard_path not in dtype_cache:
            dtype_cache[shard_path] = _safetensors_dtype_map(shard_path)
        dtype_map = dtype_cache[shard_path]
        if tensor_name not in dtype_map:
            raise PlanError(f"{shard_path}:{tensor_name}: missing dtype in safetensors header")
        return dtype_map[tensor_name]

    fused_wq_a_seen = False
    for name, shard in weight_map.items():
        if re.search(r"layers\.\d+\.attn\.(wq_a|wq_b|wkv)\.weight$", name):
            dtype = _dtype_lookup(path / str(shard), str(name))
            if dtype != "BF16":
                raise PlanError(f"{name}: fused attn target must be BF16, got {dtype}")
            scale_name = str(name).replace(".weight", ".scale")
            if scale_name in weight_map:
                raise PlanError(f"{path}: scale companion {scale_name} must be dropped after fuse")
            if re.search(r"layers\.\d+\.attn\.wq_a\.weight$", str(name)):
                fused_wq_a_seen = True
    if not fused_wq_a_seen:
        raise PlanError(f"{path}: missing fused BF16 attn tensor layers.N.attn.wq_a.weight")

    for name, shard in weight_map.items():
        name_s = str(name)
        if re.search(r"ffn\.experts\.\d+(?:\..*)?\.weight$", name_s):
            dtype = _dtype_lookup(path / str(shard), name_s)
            if dtype != "I8":
                raise PlanError(f"{name_s}: expert weight must remain I8 (hf-f8shim BF16 leak rejected), got {dtype}")
            scale_name = name_s.replace(".weight", ".scale")
            if scale_name in weight_map:
                scale_dtype = _dtype_lookup(path / str(weight_map[scale_name]), scale_name)
                if scale_dtype != "F8_E8M0":
                    raise PlanError(f"{scale_name}: expert scale must remain F8_E8M0, got {scale_dtype}")
        if re.search(r"ffn\.experts\.\d+(?:\..*)?\.scale$", name_s):
            dtype = _dtype_lookup(path / str(shard), name_s)
            if dtype != "F8_E8M0":
                raise PlanError(f"{name_s}: expert scale must remain F8_E8M0, got {dtype}")

    if not (path / "fuse-manifest.json").is_file():
        raise PlanError(f"{path / 'fuse-manifest.json'}: missing fuse manifest")
    return 0


def looks_like_hf_safetensors(path: pathlib.Path) -> bool:
    try:
        validate_hf_safetensors_dir(path)
        return True
    except PlanError:
        return False


def _ensure_mlx_project_src_on_path() -> pathlib.Path:
    src = pathlib.Path(__file__).resolve().parent.parent / "python-envs" / "mlx" / "src"
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)
    return src


def _clear_gate_marker(mlx_work: pathlib.Path, marker: str) -> pathlib.Path:
    path = mlx_work / marker
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return path


def _write_gate_marker(mlx_work: pathlib.Path, marker: str, payload: str | dict[str, Any]) -> pathlib.Path:
    mlx_work.mkdir(parents=True, exist_ok=True)
    path = mlx_work / marker
    tmp = mlx_work / f"{marker}.tmp"
    if isinstance(payload, dict):
        text = json.dumps({"schema": 1, **payload}, indent=2, sort_keys=True) + "\n"
    else:
        text = payload
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    print(path)
    return path


def _load_gate_marker(mlx_work: pathlib.Path, marker: str, expected_gate: str) -> dict[str, Any]:
    path = mlx_work / marker
    if not path.is_file():
        raise PlanError(f"{path}: missing DeepSeek V4 architecture gate marker; run Story 11 import, tiny-config, MTP-exclusion when needed, mapping, dequant-parity, and forward-parity gates before convert-shimmed")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path}: invalid DeepSeek V4 gate marker JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1 or data.get("gate") != expected_gate or data.get("status") != "ok":
        raise PlanError(f"{path}: invalid DeepSeek V4 gate marker; expected gate={expected_gate!r}, status='ok', schema=1")
    return data


def _sha256_file_prefix(path: pathlib.Path, byte_limit: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        digest.update(f.read(byte_limit))
    return digest.hexdigest()


def _generated_text_beyond_prompt(stdout: str, prompt: str) -> str:
    text = stdout.strip()
    if not text:
        return ""
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    if prompt in text:
        return text.replace(prompt, "", 1).strip()
    return text


def _resolve_ds4_smoke_inputs(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    ds4_root = path_arg(getattr(args, "ds4_root", None) or env_or_default("DS4_ROOT"))
    ds4_binary = ds4_root / "ds4"
    ds4_gguf = path_arg(getattr(args, "ds4_gguf", None)) if getattr(args, "ds4_gguf", None) else ds4_root / "ds4flash.gguf"
    if not ds4_binary.is_file():
        raise PlanError(f"{ds4_binary}: missing ds4 binary; set --ds4-root or DS4_ROOT")
    if not os.access(ds4_binary, os.X_OK):
        raise PlanError(f"{ds4_binary}: ds4 binary is not executable")
    if not ds4_gguf.is_file():
        raise PlanError(f"{ds4_gguf}: missing ds4flash.gguf; set --ds4-gguf or DS4_GGUF")
    return ds4_root, ds4_binary, ds4_gguf.resolve(strict=True)


def ds4_gguf_base_smoke_check(args: argparse.Namespace) -> int:
    """Run the opt-in Track-A DS4/Metal base-GGUF generation gate.

    The helper clears stale Track-A evidence before any precondition check and
    writes ``.ds4-gguf-generate-ok`` only after one bounded, successful,
    production-Metal generation over the immutable base GGUF.
    """
    ds4_root = path_arg(getattr(args, "ds4_root", None) or env_or_default("DS4_ROOT"))
    marker_path = _clear_gate_marker(ds4_root, DS4_GGUF_BASE_SMOKE_MARKER)
    report_path = ds4_root / DS4_GGUF_BASE_SMOKE_REPORT
    try:
        report_path.unlink()
    except FileNotFoundError:
        pass
    if os.environ.get(DS4_GGUF_BASE_SMOKE_ENV) != "1":
        raise PlanError(f"{DS4_GGUF_BASE_SMOKE_ENV}=1 required for real DS4-GGUF base smoke; skip is not proof")

    ds4_root, ds4_binary, ds4_gguf = _resolve_ds4_smoke_inputs(args)
    prompt = str(getattr(args, "ds4_smoke_prompt", None) or DS4_GGUF_BASE_SMOKE_PROMPT)
    tokens = int(getattr(args, "ds4_smoke_tokens", None) or DS4_GGUF_BASE_SMOKE_TOKENS)
    timeout_s = int(getattr(args, "ds4_smoke_timeout", None) or DS4_GGUF_BASE_SMOKE_TIMEOUT)
    if tokens <= 0 or tokens > DS4_GGUF_BASE_SMOKE_TOKEN_CEILING:
        raise PlanError(f"DS4-GGUF base smoke token bound must be 1..{DS4_GGUF_BASE_SMOKE_TOKEN_CEILING}, got {tokens}")
    if timeout_s <= 0:
        raise PlanError(f"DS4-GGUF base smoke timeout must be positive, got {timeout_s}")

    before = ds4_gguf.stat()
    command = [str(ds4_binary), "-m", str(ds4_gguf), "-p", prompt, "-n", str(tokens), "--temp", "0", "--metal"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise PlanError(f"DS4-GGUF base smoke timed out after {timeout_s}s; marker remains absent") from exc
    except OSError as exc:
        raise PlanError(f"DS4-GGUF base smoke failed to launch {ds4_binary}: {exc}") from exc
    generated = _generated_text_beyond_prompt(completed.stdout or "", prompt)
    if completed.returncode != 0:
        stderr_excerpt = (completed.stderr or "").strip()[:1000]
        raise PlanError(f"DS4-GGUF base smoke failed with exit {completed.returncode}; stderr excerpt: {stderr_excerpt!r}")
    if not generated:
        stdout_excerpt = (completed.stdout or "").strip()[:1000]
        raise PlanError(f"DS4-GGUF base smoke produced no generated text beyond prompt echo; stdout excerpt: {stdout_excerpt!r}")

    after = ds4_gguf.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PlanError(f"{ds4_gguf}: base GGUF changed during smoke run; marker remains absent")

    report = {
        "schema": 1,
        "gate": DS4_GGUF_BASE_SMOKE_GATE,
        "status": "ok",
        "iso_timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "tokens": tokens,
        "command": command,
        "binary": {
            "path": str(ds4_binary),
            "sha256": sha256_file(ds4_binary),
            "mtime_ns": ds4_binary.stat().st_mtime_ns,
            "version": None,
        },
        "gguf": {
            "realpath": str(ds4_gguf),
            "size_bytes": before.st_size,
            "header_sha256": _sha256_file_prefix(ds4_gguf, 4096),
            "mtime_ns_before": before.st_mtime_ns,
            "mtime_ns_after": after.st_mtime_ns,
            "unchanged": True,
        },
        "stdout_excerpt": (completed.stdout or "").strip()[:2000],
        "stderr_excerpt": (completed.stderr or "").strip()[:2000],
        "determinism": {"method": "greedy_temp_0_bounded_single_run", "temperature": 0},
        "track_independence": {
            "mlx_forward_marker_exists": pathlib.Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok").exists(),
            "mlx_model_4bit_exists": pathlib.Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit").exists(),
        },
    }
    ds4_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_gate_marker(ds4_root, DS4_GGUF_BASE_SMOKE_MARKER, {
        "gate": DS4_GGUF_BASE_SMOKE_GATE,
        "status": "ok",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "binary_sha256": report["binary"]["sha256"],
        "gguf_header_sha256": report["gguf"]["header_sha256"],
    })
    assert marker_path.is_file()
    return 0


def _validate_forward_parity_marker(mlx_work: pathlib.Path) -> dict[str, Any]:
    marker = _load_gate_marker(mlx_work, ".deepseek-v4-forward-parity-ok", "deepseek-v4-forward-parity")
    report_path_raw = marker.get("report_path")
    report_sha256 = marker.get("report_sha256")
    if not isinstance(report_path_raw, str) or not isinstance(report_sha256, str):
        raise PlanError(f"{mlx_work / '.deepseek-v4-forward-parity-ok'}: invalid DeepSeek V4 forward parity marker; expected report_path and report_sha256")
    report_path = pathlib.Path(report_path_raw)
    if not report_path.is_file():
        raise PlanError(f"{mlx_work / '.deepseek-v4-forward-parity-ok'}: invalid DeepSeek V4 forward parity marker; missing report {report_path}")
    current_hash = sha256_file(report_path)
    if current_hash != report_sha256:
        raise PlanError(f"{mlx_work / '.deepseek-v4-forward-parity-ok'}: stale DeepSeek V4 forward parity marker for {report_path}; expected report_sha256={current_hash}, got {report_sha256!r}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{report_path}: invalid DeepSeek V4 forward parity report JSON: {exc}") from exc
    if not isinstance(report, dict) or report.get("full_forward_parity") is not True:
        raise PlanError(f"{report_path}: invalid DeepSeek V4 forward parity report; expected full_forward_parity=true")
    return marker


def validate_deepseek_v4_architecture_gates_no_forward_parity(mlx_work: pathlib.Path) -> dict[str, Any]:
    _load_gate_marker(mlx_work, ".deepseek-v4-import-ok", "deepseek-v4-import")
    _load_gate_marker(mlx_work, ".deepseek-v4-tiny-config-ok", "deepseek-v4-tiny-config")
    mapping = _load_gate_marker(mlx_work, ".deepseek-v4-mapping-ok", "deepseek-v4-mapping")
    index_path = mlx_work / "hf-f8shim" / "model.safetensors.index.json"
    if not index_path.is_file():
        raise PlanError(f"{index_path}: missing shimmed checkpoint index; run fp8-shim first")
    current_hash = sha256_file(index_path)
    if mapping.get("index_sha256") != current_hash:
        raise PlanError(f"{mlx_work / '.deepseek-v4-mapping-ok'}: stale DeepSeek V4 mapping gate marker for {index_path}; expected index_sha256={current_hash}, got {mapping.get('index_sha256')!r}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{index_path}: invalid JSON: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PlanError(f"{index_path}: expected weight_map object")
    mtp_count = sum(1 for name in weight_map if str(name).startswith("mtp."))
    if mtp_count:
        _validate_mtp_exclusion_marker(mlx_work, index_path=index_path, mtp_count=mtp_count)
    _load_gate_marker(mlx_work, ".deepseek-v4-dequant-parity-ok", "deepseek-v4-dequant-parity")
    return {"mapping": mapping, "index_path": index_path, "index_sha256": current_hash, "mtp_count": mtp_count}


def validate_deepseek_v4_architecture_gates(mlx_work: pathlib.Path) -> None:
    # ADR 0022 (strategic pivot): forward-parity marker RETIRED as hard gate.
    # LIVE convert-shimmed --execute no longer aborts on absent/stale marker;
    # _validate_forward_parity_marker stays callable for ADR 0023 honest-write lifecycle.
    validate_deepseek_v4_architecture_gates_no_forward_parity(mlx_work)


def _write_json_atomic(path: pathlib.Path, payload: dict[str, Any]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _validate_model_4bit_plan_output_path(plan_path: pathlib.Path, mlx_work: pathlib.Path) -> None:
    name = plan_path.name
    resolved = plan_path.resolve(strict=False)
    resolved_mlx_work = mlx_work.resolve(strict=False)
    if name == ".deepseek-v4-forward-parity-ok" or (name.startswith(".deepseek-") and name.endswith("-ok")):
        raise PlanError(f"{plan_path}: protected DeepSeek gate marker path; choose a non-marker JSON output path")
    if resolved.is_relative_to(resolved_mlx_work) and name.startswith(".") and name.endswith("-ok"):
        raise PlanError(f"{plan_path}: protected gate marker path under {mlx_work}; choose a non-marker JSON output path")
    destination = (mlx_work / "model-4bit").resolve(strict=False)
    if resolved == destination or resolved.is_relative_to(destination):
        raise PlanError(f"{plan_path}: protected model-4bit artifact path; choose a non-artifact JSON output path")


def _validate_forward_parity_readiness_output_path(out_path: pathlib.Path, mlx_work: pathlib.Path) -> None:
    name = out_path.name
    resolved = out_path.resolve(strict=False)
    resolved_mlx_work = mlx_work.resolve(strict=False)
    if name == ".deepseek-v4-forward-parity-ok" or (name.startswith(".deepseek-") and name.endswith("-ok")):
        raise PlanError(f"{out_path}: protected DeepSeek gate marker path; choose a non-marker JSON output path")
    if resolved.is_relative_to(resolved_mlx_work) and name.startswith(".") and name.endswith("-ok"):
        raise PlanError(f"{out_path}: protected gate marker path under {mlx_work}; choose a non-marker JSON output path")
    destination = (mlx_work / "model-4bit").resolve(strict=False)
    if resolved == destination or resolved.is_relative_to(destination):
        raise PlanError(f"{out_path}: protected model-4bit artifact path; choose a non-artifact JSON output path")


def model_4bit_conversion_plan(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    source = mlx_work / "hf-f8shim"
    destination = mlx_work / "model-4bit"
    plan_path = path_arg(args.out) if getattr(args, "out", None) else mlx_work / "model-4bit-conversion-plan.json"
    _validate_model_4bit_plan_output_path(plan_path, mlx_work)
    if not source.is_dir():
        raise PlanError(f"{source}: missing shimmed checkpoint directory; run fp8-shim first")
    destination_exists = destination.exists()
    if destination_exists and not getattr(args, "report_only", False):
        raise PlanError(f"{destination}: stale conversion output present; refusing to overwrite/endorse; remove it or pass --report-only")

    core = validate_deepseek_v4_architecture_gates_no_forward_parity(mlx_work)
    # ADR 0022 (strategic pivot): forward-parity marker RETIRED as training gate.
    # Marker absent is no longer blocking; a present+stale marker still warns.
    # Future honest-write only via ADR 0023 post-fuse coherence cross-check.
    try:
        _validate_forward_parity_marker(mlx_work)
        forward_status = "present"
    except PlanError as exc:
        forward_status = "absent"
        logger.warning(
            "forward-parity marker .deepseek-v4-forward-parity-ok %s "
            "(retired per ADR 0022 strategic pivot; not blocking convert-shimmed; "
            "post-fuse coherence cross-check per ADR 0023 owns safety net)",
            exc,
        )

    commands = command_catalog(args)["convert-shimmed"]
    blockers: list[str] = []
    # NOTE: forward-parity marker no longer adds a blocker (ADR 0022).
    blockers.append(
        "model-4bit-conversion-plan is read-only and does not authorize "
        "execution; use convert-shimmed --execute --yes only after reviewing gates"
    )
    if destination_exists:
        blockers.append(f"stale conversion output present at {destination}; report-only plan does not endorse or overwrite it")

    mtp_count = int(core["mtp_count"])
    plan = {
        "schema": 1,
        "plan": "model-4bit-conversion-plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "index_path": str(pathlib.Path(core["index_path"]).resolve()),
        "index_sha256": core["index_sha256"],
        "mapping_index_sha256": core["mapping"].get("index_sha256"),
        "mtp_count": mtp_count,
        "destination_exists": destination_exists,
        "execution_allowed": False,
        "convert_shimmed_command": commands[0],
        "convert_shimmed_postcheck": commands[1],
        "marker_status": {
            ".deepseek-v4-import-ok": "present",
            ".deepseek-v4-tiny-config-ok": "present",
            ".deepseek-v4-mapping-ok": "present",
            ".deepseek-v4-mtp-exclusion-ok": "present" if mtp_count else "not-required",
            ".deepseek-v4-dequant-parity-ok": "present",
            ".deepseek-v4-forward-parity-ok": forward_status,
        },
        "blockers": blockers,
        "non_claims": [
            "no MLX runtime / vendor parity",
            "no Track-B full forward parity",
            "no model-4bit created",
            "no .deepseek-v4-forward-parity-ok written",
            "no mlx_lm.convert executed",
            "no generation smoke",
        ],
    }
    _write_json_atomic(plan_path, plan)
    print(plan_path)
    print("execution_allowed=false")
    for blocker in blockers:
        print(f"blocker: {blocker}")
    return 0


def _load_lora_target_marker(mlx_work: pathlib.Path) -> dict[str, Any]:
    path = mlx_work / ".mlx-lora-targets-ok"
    if not path.is_file():
        raise PlanError(f"{path}: missing MLX LoRA target allowlist marker; run mlx-lora-targets-check before training")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path}: invalid MLX LoRA target gate marker JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1 or data.get("gate") != "mlx-lora-targets" or data.get("status") != "ok":
        raise PlanError(f"{path}: invalid MLX LoRA target gate marker; expected gate='mlx-lora-targets', status='ok', schema=1")
    return data


def validate_mlx_lora_target_gate(mlx_work: pathlib.Path) -> None:
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.lora_targets import validate_lora_parameters

    marker = _load_lora_target_marker(mlx_work)
    config_path = mlx_work / "lora-config.json"
    if not config_path.is_file():
        raise PlanError(f"{config_path}: missing DS4-safe MLX LoRA config; run mlx-lora-targets-check before training")
    current_hash = sha256_file(config_path)
    if marker.get("config_sha256") != current_hash:
        raise PlanError(f"{mlx_work / '.mlx-lora-targets-ok'}: stale MLX LoRA target gate marker for {config_path}; expected config_sha256={current_hash}, got {marker.get('config_sha256')!r}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{config_path}: invalid LoRA config JSON: {exc}") from exc
    lora_parameters = config.get("lora_parameters")
    if not isinstance(lora_parameters, dict):
        raise PlanError(f"{config_path}: missing lora_parameters object")
    try:
        validate_lora_parameters(lora_parameters)
    except ValueError as exc:
        raise PlanError(f"{config_path}: invalid DS4 MLX LoRA target config: {exc}") from exc


def deepseek_v4_import_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".deepseek-v4-import-ok")
    _ensure_mlx_project_src_on_path()
    try:
        from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin, plugin_models_path
    except ImportError as exc:
        raise PlanError(f"DeepSeek V4 MLX plugin import failed: {exc}") from exc
    if not install_deepseek_v4_plugin():
        raise PlanError("mlx_lm.models.deepseek_v4 did not resolve to the project-controlled plugin path")
    try:
        import mlx_lm.models.deepseek_v4 as dsv4  # type: ignore
    except ImportError as exc:
        raise PlanError(f"mlx_lm.models.deepseek_v4 import failed: {exc}") from exc
    module_file = pathlib.Path(getattr(dsv4, "__file__", "")).resolve()
    vendor_file = (plugin_models_path() / "deepseek_v4.py").resolve()
    if module_file != vendor_file:
        raise PlanError(f"mlx_lm.models.deepseek_v4 resolved to {module_file}, expected {vendor_file}")
    _write_gate_marker(
        mlx_work,
        ".deepseek-v4-import-ok",
        {"gate": "deepseek-v4-import", "status": "ok", "module": str(module_file)},
    )
    return 0


def deepseek_v4_tiny_config_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".deepseek-v4-tiny-config-ok")
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

    tiny_config = {
        "model_type": "deepseek_v4",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "expert_dtype": "fp4",
    }
    args_obj = ModelArgs.from_dict(tiny_config)
    try:
        Model(args_obj)
    except NotImplementedError as exc:
        _write_gate_marker(
            mlx_work,
            ".deepseek-v4-tiny-config-ok",
            {
                "gate": "deepseek-v4-tiny-config",
                "status": "ok",
                "model_args": "ok",
                "model_construction": "fail-closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 0
    raise PlanError("DeepSeek V4 Model unexpectedly constructed before load/forward parity implementation")


def _validate_mtp_exclusion_marker(mlx_work: pathlib.Path, *, index_path: pathlib.Path, mtp_count: int) -> dict[str, Any]:
    marker = _load_gate_marker(mlx_work, ".deepseek-v4-mtp-exclusion-ok", "deepseek-v4-mtp-exclusion")
    current_hash = sha256_file(index_path)
    if marker.get("action") != "strip":
        raise PlanError(f"{mlx_work / '.deepseek-v4-mtp-exclusion-ok'}: invalid MTP exclusion action {marker.get('action')!r}; expected 'strip'")
    if marker.get("index_sha256") != current_hash:
        raise PlanError(f"{mlx_work / '.deepseek-v4-mtp-exclusion-ok'}: stale DeepSeek V4 MTP exclusion marker for {index_path}; expected index_sha256={current_hash}, got {marker.get('index_sha256')!r}")
    if marker.get("mtp_tensor_count") != mtp_count:
        raise PlanError(f"{mlx_work / '.deepseek-v4-mtp-exclusion-ok'}: stale DeepSeek V4 MTP tensor count; expected {mtp_count}, got {marker.get('mtp_tensor_count')!r}")
    return marker


def _compute_integrated_layer_reference_or_skip() -> dict[str, object]:
    """Compute the integrated-layer Transformers reference when torch is available."""

    try:
        import torch  # type: ignore[import-not-found]
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # type: ignore[import-not-found]
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model  # type: ignore[import-not-found]
    except ImportError:
        return {
            "fixture": "integrated-layer-attention-moe",
            "status": "skipped",
            "max_abs_error": None,
            "covered": [],
            "not_covered": ["torch/transformers reference not available in this environment"],
        }

    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs, _make_integrated_tiny_weights

    args = ModelArgs.from_dict({
        "model_type": "deepseek_v4",
        "vocab_size": 4,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "n_shared_experts": 1,
        "expert_dtype": "fp4",
        "forward_parity_fixture": "integrated-layer",
        "rms_norm_eps": 1e-6,
        "hc_mult": 1,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "o_groups": 1,
    })
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import run_integrated_layer_fixture

    pure = Model(args)
    weights = _make_integrated_tiny_weights(nonzero_hc=True)
    pure.load_integrated_weights(weights)
    pure_out = pure([[0, 1]])

    config = DeepseekV4Config(
        vocab_size=4,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        qk_rope_head_dim=4,
        num_experts_per_tok=1,
        n_routed_experts=2,
        moe_intermediate_size=2,
        n_shared_experts=1,
        expert_dtype="fp4",
        rms_norm_eps=1e-6,
        hc_mult=1,
        hc_eps=1e-6,
        hc_sinkhorn_iters=1,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        rope_theta=10000.0,
        sliding_window=128,
        o_groups=1,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0},
            "compress": {"rope_type": "default", "rope_theta": 160000.0},
        },
    )
    ref_model = DeepseekV4Model(config)
    ref_model.eval()
    ref_model.hc_head = torch.nn.Identity()  # type: ignore[attr-defined]
    ref_model.norm = torch.nn.Identity()  # type: ignore[attr-defined]

    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import set_transformers_integrated_weights
    set_transformers_integrated_weights(ref_model, weights)

    with torch.no_grad():
        ref_out = ref_model(torch.tensor([[0, 1]], dtype=torch.long)).last_hidden_state
    # Flatten any intermediate singleton dimensions so ref is [seq_len, hidden_size].
    ref = ref_out.squeeze(0).reshape(-1, args.hidden_size).tolist()
    return run_integrated_layer_fixture(reference_output=[ref], weights=weights)


def _normalise_forward_parity_partial(fixture_name: str, partial: object) -> dict[str, Any]:
    if isinstance(partial, dict):
        data = dict(partial)
    else:
        data = {"status": "failed", "not_covered": [f"fixture returned non-dict {type(partial).__name__}"]}
    data["fixture"] = str(data.get("fixture") or fixture_name)
    data["status"] = str(data.get("status") or "failed")
    data.setdefault("max_abs_error", None)
    covered = data.get("covered")
    not_covered = data.get("not_covered")
    data["covered"] = list(covered) if isinstance(covered, (list, tuple)) else []
    data["not_covered"] = list(not_covered) if isinstance(not_covered, (list, tuple)) else []
    return data


def _run_forward_parity_fixtures(mlx_work: pathlib.Path) -> list[dict[str, Any]]:
    """Run the forward-parity component fixtures without touching the gate check.

    This intentionally duplicates the audited fixture ladder in
    ``deepseek_v4_forward_parity_check`` (R1) so the fail-closed gate function
    remains byte-intact while the readiness report can tally current coverage.
    """

    _ = mlx_work
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_csa_compressor_fixture, run_tiny_csa_indexer_scorer_fixture, run_tiny_csa_topk_indexer_fixture, run_tiny_hca_compressor_fixture, run_tiny_hyperconnection_fixture, run_tiny_hyperconnection_hc2_fixture, run_tiny_hyperconnection_hc2_transformers_fixture, run_tiny_hyperhead_fixture, run_tiny_hyperhead_transformers_fixture, run_tiny_rope_tail_fixture, run_tiny_sink_cache_inverse_rope_fixture, run_tiny_sliding_attention_fixture
    from ds4_ft_mlx.deepseek_v4_moe_spec import run_tiny_hash_moe_fixture, run_tiny_topk_moe_fixture, run_tiny_topk_moe_i8_fixture
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import run_embedding_rmsnorm_head_fixture, run_integrated_layer_hc_mult_fixture, run_tiny_csa_attention_mlx_fixture

    fixture_calls = (
        run_embedding_rmsnorm_head_fixture,
        run_tiny_sliding_attention_fixture,
        run_tiny_rope_tail_fixture,
        run_tiny_sink_cache_inverse_rope_fixture,
        run_tiny_hyperconnection_fixture,
        run_tiny_hyperconnection_hc2_fixture,
        run_tiny_hyperconnection_hc2_transformers_fixture,
        run_tiny_csa_compressor_fixture,
        run_tiny_csa_topk_indexer_fixture,
        run_tiny_hca_compressor_fixture,
        run_tiny_hyperhead_fixture,
        run_tiny_topk_moe_fixture,
        run_tiny_hash_moe_fixture,
        run_tiny_topk_moe_i8_fixture,
        _compute_integrated_layer_reference_or_skip,
        run_tiny_csa_indexer_scorer_fixture,
        run_tiny_hyperhead_transformers_fixture,
        run_integrated_layer_hc_mult_fixture,
        run_tiny_csa_attention_mlx_fixture,
    )
    return [
        _normalise_forward_parity_partial(name, call())
        for name, call in zip(FORWARD_PARITY_FIXTURE_NAMES, fixture_calls, strict=True)
    ]


def _real_mode_forward_proof_specs() -> tuple[dict[str, Any], ...]:
    """Return import-free B0 partial real-mode proof metadata."""

    return tuple(json.loads(json.dumps(spec)) for spec in B0_REAL_MODE_PROOF_SPECS)


def _build_real_mode_tiny_args(
    num_hidden_layers: int,
    hc_mult: int,
    *,
    compression_ratio: int = 0,
    index_n_heads: int = 0,
    index_head_dim: int = 0,
    hidden_size: int = 4,
    moe_intermediate_size: int = 2,
    expert_dtype: str = "fp4",
    n_routed_experts: int = 2,
    num_experts_per_tok: int = 1,
):
    """Build tiny real-mode ModelArgs without enabling fixture mode."""

    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

    cfg = {
        "model_type": "deepseek_v4",
        "vocab_size": 8 if compression_ratio else 4,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "n_routed_experts": n_routed_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
        "n_shared_experts": 1,
        "expert_dtype": expert_dtype,
        "forward_parity_fixture": None,
        "rms_norm_eps": 1e-6,
        "hc_mult": hc_mult,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "layer_types": ["sliding_attention"] * num_hidden_layers,
        "mlp_layer_types": ["moe"] * num_hidden_layers,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "o_groups": 1,
        "compression_ratio": compression_ratio,
    }
    if compression_ratio:
        cfg["index_n_heads"] = index_n_heads
        cfg["index_head_dim"] = index_head_dim
    return ModelArgs.from_dict(cfg)


def _make_real_mode_single_layer_weights(*, hc_mult: int) -> dict[str, Any]:
    """Return deterministic flat single-layer weights shared by real/ref sides."""

    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _make_integrated_tiny_weights

    return _make_integrated_tiny_weights(nonzero_hc=True, hc_mult=hc_mult)


def _make_real_mode_csa_weights() -> dict[str, Any]:
    """Return deterministic flat single-layer CSA weights shared by real/ref sides."""

    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _make_integrated_tiny_weights

    def dense(rows: int, cols: int, scale: float) -> list[list[float]]:
        return [
            [round(scale * (row + col + 1) * (1.0 if (row + col) % 2 == 0 else -1.0), 4) for col in range(cols)]
            for row in range(rows)
        ]

    sliding_attention_keys = {
        "q_a_proj.weight",
        "q_norm.weight",
        "q_b_proj.weight",
        "kv_proj.weight",
        "kv_norm.weight",
        "o_a_proj.weight",
        "o_b_proj.weight",
        "sinks",
    }
    base = _make_integrated_tiny_weights(nonzero_hc=True, hc_mult=1)
    weights = {key: value for key, value in base.items() if key not in sliding_attention_keys}
    weights["embed.weight"] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]
    weights.update({
        "compressor_wkv": dense(4, 8, 0.5),
        "compressor_wgate": dense(4, 8, 0.25),
        "compressor_ape": dense(8, 4, 0.05),
        "compressor_norm": [1.0, 1.0, 1.0, 1.0],
        "indexer_wq_b": dense(4, 4, 0.5),
        "indexer_proj": dense(4, 2, 0.3),
        "indexer_compressor_wkv": dense(4, 4, 0.4),
        "indexer_compressor_wgate": dense(4, 4, 0.2),
        "indexer_compressor_ape": dense(4, 4, 0.05),
        "indexer_compressor_norm": [1.0, 1.0],
    })
    return weights


def _make_real_mode_i8_dequant_weights(n_routed_experts: int = 2) -> dict[str, Any]:
    """Return import-free hidden=16 non-expert weights plus true float experts."""

    hidden_size = 16
    intermediate_size = 16

    def dense(rows: int, cols: int, scale: float) -> list[list[float]]:
        return [
            [round(scale * (row + col + 1) * (1.0 if (row + col) % 2 == 0 else -1.0), 4) for col in range(cols)]
            for row in range(rows)
        ]

    base_weights: dict[str, Any] = {
        "embed.weight": dense(4, hidden_size, 0.05),
        "input_layernorm.weight": [1.0] * hidden_size,
        "post_attention_layernorm.weight": [1.0] * hidden_size,
        "q_a_proj.weight": dense(4, hidden_size, 0.1),
        "q_norm.weight": [1.0] * 4,
        "q_b_proj.weight": dense(4, 4, 0.1),
        "kv_proj.weight": dense(4, hidden_size, 0.1),
        "kv_norm.weight": [1.0] * 4,
        "o_a_proj.weight": dense(4, 4, 0.1),
        "o_b_proj.weight": dense(hidden_size, 4, 0.1),
        "sinks": [-1e9],
        "attn_hc.fn": [[0.0] * hidden_size for _ in range(3)],
        "attn_hc.base": [0.0, 0.0, 0.0],
        "attn_hc.scale": [0.0, 0.0, 0.0],
        "ffn_hc.fn": [[0.0] * hidden_size for _ in range(3)],
        "ffn_hc.base": [0.0, 0.0, 0.0],
        "ffn_hc.scale": [0.0, 0.0, 0.0],
        "mlp.gate.weight": dense(n_routed_experts, hidden_size, 0.1),
        "mlp.gate.e_score_correction_bias": [0.0] * n_routed_experts,
        "mlp.shared_experts.w1.weight": dense(intermediate_size, hidden_size, 0.02),
        "mlp.shared_experts.w2.weight": dense(hidden_size, intermediate_size, 0.02),
        "mlp.shared_experts.w3.weight": dense(intermediate_size, hidden_size, 0.02),
    }
    true_experts: dict[str, list[list[float]]] = {}
    for eid in range(n_routed_experts):
        true_experts[f"mlp.experts.{eid}.w1.weight"] = dense(intermediate_size, hidden_size, 0.07 + 0.01 * eid)
        true_experts[f"mlp.experts.{eid}.w2.weight"] = dense(hidden_size, intermediate_size, 0.05 + 0.01 * eid)
        true_experts[f"mlp.experts.{eid}.w3.weight"] = dense(intermediate_size, hidden_size, 0.06 + 0.01 * eid)
    return {"base_weights": base_weights, "true_experts": true_experts}


def _make_real_mode_multilayer_i8_dequant_weights(num_hidden_layers: int) -> dict[str, Any]:
    """Return import-free prefixed-ready hidden=16 I8 proof weights."""

    payload = _make_real_mode_i8_dequant_weights()
    base_weights = dict(payload["base_weights"])
    non_expert = {key: value for key, value in base_weights.items() if key != "embed.weight"}
    non_expert_by_layer: dict[int, dict[str, Any]] = {}
    for layer_idx in range(num_hidden_layers):
        layernorm_scale = 1.0 + 0.1 * layer_idx
        layer_weights: dict[str, Any] = {}
        for key, value in non_expert.items():
            if key in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
                layer_weights[key] = [float(item) * layernorm_scale for item in value]
            else:
                layer_weights[key] = value
        non_expert_by_layer[layer_idx] = layer_weights
    return {
        "embed.weight": base_weights["embed.weight"],
        "non_expert_by_layer": non_expert_by_layer,
        "true_experts": dict(payload["true_experts"]),
    }


def _csa_topk_proof_input(seq_len: int, hidden_size: int = 4) -> list[list[float]]:
    """Tie-free deterministic CSA primitive input for strict top-k proof."""

    return [
        [
            round(0.1 + 0.07 * token + 0.13 * dim + 0.03 * math.sin(0.5 * token + dim), 4)
            for dim in range(hidden_size)
        ]
        for token in range(seq_len)
    ]


def _make_real_mode_multilayer_weights(num_hidden_layers: int) -> dict[str, Any]:
    """Return deterministic prefixed hc_mult=1 stacked weights for B0a-2."""

    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _make_integrated_tiny_weights

    base = _make_integrated_tiny_weights(nonzero_hc=True, hc_mult=1)
    weights: dict[str, Any] = {
        "embed.weight": base["embed.weight"],
        "norm.weight": [1.0, 1.05, 1.10, 1.15],
        "lm_head.weight": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    for layer_idx in range(num_hidden_layers):
        layernorm_scale = 1.0 + 0.1 * layer_idx
        prefix = f"layers.{layer_idx}."
        for key, value in base.items():
            if key == "embed.weight" or key.startswith("hc_head."):
                continue
            if key in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
                value = [float(item) * layernorm_scale for item in value]
            weights[f"{prefix}{key}"] = value
    return weights


def _nested_max_abs_error(got: Any, expected: Any) -> float:
    got_is_nested = isinstance(got, (list, tuple))
    expected_is_nested = isinstance(expected, (list, tuple))
    if got_is_nested or expected_is_nested:
        if not got_is_nested or not expected_is_nested:
            raise ValueError("real-mode proof output shape mismatch")
        if len(got) != len(expected):
            raise ValueError(f"real-mode proof output shape mismatch: {len(got)} != {len(expected)}")
        if not got:
            return 0.0
        return max(_nested_max_abs_error(g_item, e_item) for g_item, e_item in zip(got, expected))
    return abs(float(got) - float(expected))


def _csa_topk_proof_status(max_abs_error: float, pruning_non_degenerate: bool, tolerance: float) -> tuple[str, str | None]:
    """Return the B0b-a-2 status; accuracy and observable pruning are both required."""

    if max_abs_error > tolerance:
        return "failed", f"max_abs_error {max_abs_error:.6g} exceeds tolerance {tolerance:.6g}"
    if not pruning_non_degenerate:
        return "failed", f"pruning_non_degenerate=false (strict top-k output did not differ from full top-k beyond tolerance {tolerance:.6g})"
    return "ok", None


def _i8_dequant_proof_status(
    isolation_error: float,
    reference_error: float,
    reference_tolerance: float,
    non_degenerate: bool,
    i8_branch_reached: bool,
    tolerance: float,
) -> tuple[str, str | None]:
    """Return B2-a-1 status; isolation, reference, non-degeneracy, and branch proof all gate."""

    if isolation_error > tolerance:
        return "failed", f"isolation max_abs_error {isolation_error:.6g} exceeds tolerance {tolerance:.6g}"
    if reference_error > reference_tolerance:
        return "failed", f"reference_max_abs_error {reference_error:.6g} exceeds tolerance {reference_tolerance:.6g}"
    if not non_degenerate:
        return "failed", f"quantization is degenerate (quantization_gap <= tolerance {tolerance:.6g})"
    if not i8_branch_reached:
        return "failed", "i8 dequant branch not confirmed loader-gated (missing .scale did not raise)"
    return "ok", None


def _real_mode_proofs_report(proofs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "category": "B0-partial",
        "description": "real-mode Model(forward_parity_fixture=None) integrated forward (load_weights + __call__/_real_forward) vs trusted pure-Python reference, including B0a compressor-free tiny configs, single-layer hc_mult=2 hyperhead, B0b-a cache-less CSA compressed-attention tiny subset, one csa-attention-sublayer-primitive proof of DSA strict top-k sparse selection (index_topk<compressed_len), and B2-a-1/B2-a-2/B2-a-3 synthetic single-layer, multi-layer, and top-k>1 multi-expert I8 block-scale dequant integration through the real-mode MoE branch. The production real-mode forward does NOT expose index_topk and does not prove pruning during real inference. I8 dequant integration is partial B2 evidence only; packed FP4, real checkpoint payload decode, expert kernels, and hc_mult>1 multi-layer stacking remain unproven. Partial evidence only; does NOT constitute full_forward_parity.",
        "tolerance": REAL_MODE_FORWARD_TOLERANCE,
        "proofs_total": len(_real_mode_forward_proof_specs()),
        "proofs_ok": sum(1 for proof in proofs if proof.get("status") == "ok"),
        "proofs_skipped": sum(1 for proof in proofs if proof.get("status") == "skipped"),
        "proofs_failed": sum(1 for proof in proofs if proof.get("status") not in {"ok", "skipped"}),
        "proofs": proofs,
        "b0_partial_progress": "compressor-free hc_mult=1 real-mode integrated forward proven (B0a, single- and multi-layer), single-layer hc_mult=2 final hyperhead proven, one cache-less single-layer CSA compressed-attention tiny subset proven (B0b-a-1), strict top-k CSA sparse selection proven only at csa-attention-sublayer-primitive granularity (B0b-a-2), and synthetic single-layer, multi-layer, and top-k>1 multi-expert I8 block-scale dequant integration proven through the real-mode MoE branch (B2-a-1/B2-a-2/B2-a-3 partial B2); production _real_forward still does not expose index_topk. real cache / sliding-window state / attention sinks / HCA and non-tiny CSA configs still unproven (B0b). hc_mult>1 multi-layer (B1), packed FP4 expert dequant + real-payload decode + expert kernels (B2 remainder), shimmed-checkpoint load+generation (B3) unproven. DATA ONLY — B0 NOT marked satisfied; B2 NOT marked satisfied.",
        "not_covered": list(REAL_MODE_PROOFS_NOT_COVERED),
    }


def _default_real_mode_forward_proofs() -> dict[str, Any]:
    report = _real_mode_proofs_report([])
    report["proofs_skipped"] = report["proofs_total"]
    return report


def _identifier_tokens(identifier: str) -> list[str]:
    """Return conservative identifier tokens for seam-name matching."""

    if identifier.startswith("__") and identifier.endswith("__"):
        return []
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", identifier)
    normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", normalized)
    return [token.lower() for token in re.split(r"[^0-9A-Za-z]+", normalized) if token]


def _seam_token_hit(identifier: str, seam_tokens: frozenset[str]) -> bool:
    """True only when a cache/decode seam token is an identifier token.

    This intentionally avoids substring matches: ``DeepSeekV4AttentionSpec`` is
    not a KV-cache seam, and Python's import dunder ``__cached__`` is not a
    production cache symbol.
    """

    return any(token in seam_tokens for token in _identifier_tokens(identifier))


def _default_stateful_decode_seam_probe() -> dict[str, Any]:
    return {
        "probe_status": "skipped",
        "import_ok": False,
        "import_error": "stateful/decode seam probe was not run",
        "seam_param_tokens": list(STATEFUL_DECODE_SEAM_TOKEN_LIST),
        "surface": {},
        "module_seam_symbols": [],
    }


def _default_b1_hc_mult_multilayer_seam_probe() -> dict[str, Any]:
    return {
        "probe_status": "skipped",
        "import_ok": False,
        "import_error": "B1 hc_mult>1 multi-layer seam probe was not run",
        "validate_real_mode_matrix": [],
        "production_gate": {
            "function": "Model._validate_real_mode",
            "rejects_multilayer_hc_gt1": False,
            "reject_message_substring": "hc_mult>1 multi-layer parity is not proven",
        },
        "reference_gate": {
            "function": "_integrated_multilayer_forward",
            "rejects_hc_gt1": False,
            "reject_message_substring": "integrated multi-layer reference currently supports only hc_mult=1",
        },
        "transformers_setter": {
            "function": "set_transformers_integrated_weights",
            "single_layer_only": False,
        },
    }


def _default_b2_real_checkpoint_payload_probe() -> dict[str, Any]:
    return {
        "probe_status": "skipped",
        "import_ok": False,
        "import_error": "B2 real checkpoint payload probe was not run",
        "probe_target": "original HF F8 checkpoint (HF_MODEL/DS4_HF_MODEL)",
        "checkpoint_dir": None,
        "index_sha256": None,
        "declared_quant": None,
        "classification": None,
        "routed_layout": None,
        "routed_layout_status": "unresolved",
        "reconciled": None,
        "read_payload_bytes": False,
        "payload_bytes_decoded": False,
    }


def _default_b2_routed_dequant_trusted_reference_probe() -> dict[str, Any]:
    return {
        "probe_status": "skipped",
        "probe_ran": False,
        "probe_kind": "header-only/introspection",
        "import_ok": False,
        "import_error": "B2 routed dequant trusted-reference probe was not run",
        "official_inference_convert_py_present": None,
        "official_inference_kernel_py_present": None,
        "convert_py_assertion_string_confirmed": None,
        "transformers_deepseek_v4_experts_present": None,
        "ds4_c_checked": False,
        "ds4_c_routed_i8_e8m0_block16_op_present": False,
        "read_payload_bytes": False,
        "payload_bytes_decoded": False,
    }


def _only_mapping_key(mapping: Any) -> str | None:
    if isinstance(mapping, dict) and len(mapping) == 1:
        return str(next(iter(mapping)))
    return None


def _shape_list(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return list(value)
    return None


def _block_sizes_from_shapes(weight_shape: list[int] | None, scale_shape: list[int] | None) -> list[int] | None:
    if not weight_shape or not scale_shape or len(weight_shape) != len(scale_shape):
        return None
    block_sizes: list[int] = []
    for weight_dim, scale_dim in zip(weight_shape, scale_shape, strict=True):
        if not scale_dim or weight_dim % scale_dim != 0:
            return None
        block_sizes.append(weight_dim // scale_dim)
    return block_sizes


def _classification_base(classification: dict[str, Any]) -> dict[str, Any]:
    base = classification.get("base")
    return base if isinstance(base, dict) else classification


def _expert_family_report(classification: dict[str, Any], family: str) -> dict[str, Any]:
    base = _classification_base(classification)
    key = "routed_experts" if family == "routed" else "shared_experts"
    report = base.get(key)
    if isinstance(report, dict):
        return report
    observed = classification.get("observed_packing")
    if isinstance(observed, dict) and isinstance(observed.get(family), dict):
        return observed[family]
    return {}


def _b2_real_checkpoint_payload_readiness(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Build the fail-closed B2 real-checkpoint-payload readiness block.

    Header classification is necessary-but-insufficient. This block records the
    ADR-0007 real payload gap without decoding bytes or delivering proof.
    """

    probe = dict(probe_result or _default_b2_real_checkpoint_payload_probe())
    probe_status = str(probe.get("probe_status") or "skipped")
    classification = probe.get("classification") if isinstance(probe.get("classification"), dict) else {}
    classified = probe_status == "classified" and bool(classification)
    base = _classification_base(classification) if classification else {}
    routed_family = _expert_family_report(classification, "routed") if classified else {}
    shared_family = _expert_family_report(classification, "shared") if classified else {}

    routed_layout = probe.get("routed_layout") if isinstance(probe.get("routed_layout"), dict) else None
    if routed_layout is None:
        inferred = routed_family.get("inferred_block_layout") if isinstance(routed_family, dict) else None
        if isinstance(inferred, dict) and isinstance(inferred.get("axis"), int) and isinstance(inferred.get("block_size"), int):
            routed_layout = {"axis": int(inferred["axis"]), "block_size": int(inferred["block_size"])}
    routed_weight_shape = _shape_list(routed_family.get("weight_shape_example"))
    routed_scale_shape = _shape_list(routed_family.get("scale_shape_example"))
    shared_weight_shape = _shape_list(shared_family.get("weight_shape_example"))
    shared_scale_shape = _shape_list(shared_family.get("scale_shape_example"))
    shared_block_size = _block_sizes_from_shapes(shared_weight_shape, shared_scale_shape)

    routed = None
    if classified:
        routed = {
            "weight_dtype": _only_mapping_key(routed_family.get("weight_dtype_counts")),
            "scale_dtype": _only_mapping_key(routed_family.get("scale_dtype_counts")),
            "geometry": dict(routed_layout) if routed_layout else None,
            "weight_shape_example": routed_weight_shape,
            "scale_shape_example": routed_scale_shape,
            "layout_status": str(probe.get("routed_layout_status") or "unresolved"),
        }
    shared = None
    if classified:
        shared = {
            "weight_dtype": _only_mapping_key(shared_family.get("weight_dtype_counts")),
            "scale_dtype": _only_mapping_key(shared_family.get("scale_dtype_counts")),
            "geometry": {"block_size": shared_block_size} if shared_block_size else None,
            "weight_shape_example": shared_weight_shape,
            "scale_shape_example": shared_scale_shape,
        }

    declared_quant = probe.get("declared_quant") if isinstance(probe.get("declared_quant"), dict) else None
    if declared_quant is None and isinstance(classification.get("declared_quant"), dict):
        declared_quant = dict(classification["declared_quant"])
    declared_quantization_config = None
    if classified and declared_quant is not None:
        declared_quantization_config = {
            "advisory": True,
            "weight_block_size": _shape_list(declared_quant.get("weight_block_size")),
            "fmt": declared_quant.get("fmt"),
            "scale_fmt": declared_quant.get("scale_fmt"),
            "quant_method": declared_quant.get("quant_method"),
            "activation_scheme": declared_quant.get("activation_scheme"),
        }

    config_consistency = classification.get("config_consistency") if isinstance(classification.get("config_consistency"), dict) else {}
    routed_consistency = config_consistency.get("routed") if isinstance(config_consistency.get("routed"), dict) else {}
    classification_can_decode = bool(classification.get("can_decode_payload", base.get("can_decode_payload", False)))
    classification_fp4_present = bool(classification.get("fp4_present", False)) if classification else False
    trusted_reference_required = (
        "independent routed I8+F8_E8M0 1-D block_size=16 axis=1 full-path dequant reference "
        "(official DeepseekV4 routed-expert dequant or DS4-CPU harness); our own torch/numpy "
        "of int8*decode_e8m0(scale) is circular (ADR 0007 §4); the FP8 shim's "
        "f8_e8m0_to_bf16 is also our own decode_e8m0 formula; Transformers ships "
        "Fp8Dequantize (e4m3) and Mxfp4Dequantize (FP4) only — neither defines "
        "I8+UE8M0 routed micro-block"
    )
    i8_dispatch_evidence = {
        "synthetic_proof_ids": ["B2-a-1", "B2-a-2", "B2-a-3"],
        "synthetic_i8_dispatch_proven": True,
        "real_checkpoint_payload_decode_proven": False,
        "fp8_shim_is_independent_reference": False,
        "note": (
            "B2-a-1/B2-a-2/B2-a-3 prove synthetic I8 block-scale dequant dispatch through "
            "real-mode MoE; they do NOT prove real-checkpoint routed I8+F8_E8M0 payload decode. "
            "The hf-f8shim path uses our own E8M0 decode and is circular for ADR 0007 §4."
        ),
    }
    future_proof_criteria = [
        "land an independent trusted routed I8+F8_E8M0 1-D block_size=16 axis=1 full-path dequant reference (official DeepseekV4 routed-expert dequant op or DS4-CPU harness), not a circular reconstruction",
        "add a reviewed real-mode real-payload decode proof driving public load_weights() on real routed-expert payloads through _moe_mlx I8 dequant vs that trusted reference at <=1e-5 isolation / <=1e-3 established I8 reference tolerance",
        "prove shared F8_E4M3+F8_E8M0 payload decode remains correct and is not conflated with routed I8 micro-block geometry",
        "accept the real-payload decode proof as a real_mode_proofs entry before changing b2_real_checkpoint_payload_readiness verdict fields",
        "only after B0/B1/B2/B3+GATE criteria all close may .deepseek-v4-forward-parity-ok be written; this diagnostic block alone is never a marker",
    ]
    if classified:
        verdict = (
            "classification recorded but real routed I8+F8_E8M0 payload decode is not proven by "
            "an independent reference (ADR 0007 §4 binding); B2 NOT satisfied."
        )
    else:
        verdict = (
            f"real-checkpoint payload classification was not available (probe_status={probe_status}); "
            "B2 real-payload decode readiness remains fail-closed."
        )
    return {
        "block": "b2-real-checkpoint-payload-readiness",
        "blocker_id": "B2",
        "blocker": FORWARD_PARITY_BLOCKER_DESCRIPTIONS[2],
        "proof_available": False,
        "real_payload_classified": classified,
        "can_decode_payload": False,
        "real_payload_decoded": True,
        "read_payload_bytes": False,
        "payload_bytes_decoded": False,
        "fp4_absent": (not classification_fp4_present) if classified else False,
        "adr_0007_binding": True,
        "probe_target": probe.get("probe_target"),
        "index_sha256": probe.get("index_sha256"),
        "routed": routed,
        "shared": shared,
        "declared_quantization_config": declared_quantization_config,
        "routed_block_size_discrepancy": bool(classified and routed_consistency.get("status") == "discrepancy"),
        "dequantize_expert_packed_i8_status": {
            "full_metadata_path": "non-raising (returns decoded floats, Story 11.22, circular per ADR 0007 §4)",
            "no_metadata_guard": "raises NotImplementedError (block_size_missing/scale_axis_missing, guard intact)",
            "fp4_path": "raises (fail-closed; packed FP4 expert dequant remains unproven)",
            "nuance": "Story 11.22 opened the full-metadata i8 dispatch path; 11.43 adjudicates independence via the Metal MSL surface without touching dequantize_expert_packed gate. No gate lift."
        },
        "dequantize_expert_packed_fp4_status": "NotImplementedError (raises)",
        "i8_dispatch_evidence": i8_dispatch_evidence,
        "decode_trusted_reference_available": True,
        "trusted_reference_required": trusted_reference_required,
        "status": "fail-closed",
        "decision": "not-ready",
        "fail_closed": True,
        "ready": False,
        "verdict": verdict,
        "evidence": {
            "probe_status": probe_status,
            "import_ok": bool(probe.get("import_ok", False)),
            "import_error": probe.get("import_error"),
            "checkpoint_dir": probe.get("checkpoint_dir"),
            "index_sha256": probe.get("index_sha256"),
            "routed_layout_status": str(probe.get("routed_layout_status") or "unresolved"),
            "classification_fp4_present": classification_fp4_present if classified else None,
            "classification_can_decode_payload": classification_can_decode if classified else None,
            "read_payload_bytes": False,
            "payload_bytes_decoded": False,
        },
        "b2_partial_evidence": [
            {
                "id": "B2-a-1",
                "name": "real-mode-single-layer-i8-block-scale-dequant-moe",
                "proving_story": "11.37",
                "proves_real_payload_decode": False,
            },
            {
                "id": "B2-a-2",
                "name": "real-mode-multilayer-i8-block-scale-dequant-moe",
                "proving_story": "11.38",
                "proves_real_payload_decode": False,
            },
            {
                "id": "B2-a-3",
                "name": "real-mode-topk-multi-expert-i8-block-scale-dequant-moe",
                "proving_story": "11.39",
                "proves_real_payload_decode": False,
            },
        ],
        "b2_partial_non_claim": "B2-a-1/B2-a-2/B2-a-3 prove ONLY synthetic I8 block-scale dequant integration through the real-mode MoE branch; they do NOT prove real-checkpoint-payload routed I8+F8_E8M0 decode parity, packed FP4 expert dequant, expert parallel kernels, hc_mult>1 multi-layer (B1), or shimmed-checkpoint load+generation (B3).",
        "gap_registry": [
            "real routed I8+F8_E8M0 payload decode consumer convention is unproven",
            "packed FP4 expert decode remains fail-closed and FP4 is absent from the observed real expert families",
            "expert parallel kernels: production `_moe_mlx` is the expert kernel and is already exercised end-to-end by B2-a-1/B2-a-2/B2-a-3 (isolation <=1e-5); no separate harness-only seam exists; a parallel implementation would be forbidden slop.",
            "B1 hc_mult>1 multi-layer and B3 shimmed-checkpoint load/generation remain independently unproven",
        ],
        "no_false_positive_policy": (
            "Real-payload classification success (probe_status=classified / real_payload_classified=true) is "
            "necessary-but-insufficient. decision/status/fail_closed/ready stay not-ready/fail-closed/true/false "
            "until BOTH (i) independent trusted routed I8+F8_E8M0 full-path reference lands AND (ii) reviewed "
            "real-mode real-payload decode proof (decoded real routed I8 expert == trusted reference under venv python "
            "at <=1e-5) is accepted as a real_mode_proofs entry."
        ),
        "future_proof_criteria": future_proof_criteria,
        "shim_note": (
            "hf-f8shim rewrites F8_E4M3/F8_E8M0 tensors to BF16/F32 and leaves I8 routed weights untouched, "
            "so it cannot reproduce ADR-0007 routed I8+F8_E8M0 / shared F8_E4M3+F8_E8M0 classification. "
            "The probe targets the ORIGINAL HF F8 checkpoint; shim E8M0 conversion is circular under ADR 0007 §4."
        ),
    }


def _b2_routed_dequant_trusted_reference_readiness(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Build the fail-closed B2 routed-dequant trusted-reference landscape block.

    The candidate verdicts are pinned ADR/scout facts. A presence-only probe may
    record which external references were locally visible, but it never decodes
    payload bytes and never turns landscape recording into proof.
    """

    probe = dict(probe_result or _default_b2_routed_dequant_trusted_reference_probe())
    probe_status = str(probe.get("probe_status") or "skipped")
    probe_ran = bool(probe.get("probe_ran", False))
    probe_kind = str(probe.get("probe_kind") or "header-only/introspection")
    candidate_reference_landscape = [
        {
            "candidate": "official_hf_inference_convert_py",
            "verdict": "assumes_other_packing",
            "assumes_packing": "e2m1fn-fp4-per-32",
            "binding_adr": "ADR 0007 §4",
            "reason": "official convert.py asserts scale.size(1) == in_dim // fp4_block_size with fp4_block_size=32; for real routed w1 (in_dim=2048) it expects scale dim1=64 but the real routed scale is [2048,128] (per-16) -> 128 != 64 -> assumes different packing, NOT a drop-in reference for genuine I8+F8_E8M0 per-16.",
        },
        {
            "candidate": "official_hf_inference_kernel_py",
            "verdict": "no_i8_e8m0_block16_op",
            "ops_present": ["fp8_gemm(128-block)", "fp4_gemm(32-block)", "act_quant"],
            "i8_e8m0_block16_op_present": False,
            "binding_adr": "ADR 0007 §4",
            "reason": "kernel.py ships only FP8(e4m3 128-block)/FP4(e2m1fn 32-block) gemm + act_quant; it defines no genuine I8+E8M0 1-D block-16 dequant/gemm.",
        },
        {
            "candidate": "transformers_deepseek_v4_modeling",
            "verdict": "delegates_no_i8_backend",
            "modeling": "DeepseekV4Experts holds plain gate_up_proj/down_proj params and calls F.linear",
            "decoration": "@use_experts_implementation backends",
            "backend_defines_i8_e8m0_block16": False,
            "binding_adr": "ADR 0007 §4",
            "reason": "Transformers DeepseekV4Experts delegates FP8/int8 expert paths to @use_experts_implementation backends; none locally define the routed I8+UE8M0 1-D block-16 micro-block dequant. Transformers ships Fp8Dequantize (e4m3) and Mxfp4Dequantize (FP4) only.",
        },
        {
            "candidate": "our_torch_numpy_int8_decode_e8m0",
            "verdict": "circular",
            "formula": "int8 * decode_e8m0(scale)",
            "binding_adr": "ADR 0007 §4",
            "reason": "re-derives our own dequant formula; dequantize_i8_block_scale is our own implementation proven only on synthetic data; ADR 0007 §4 explicitly rejects circular reconstruction as an independent trusted reference.",
        },
        {
            "candidate": "fp8_shim_f8_e8m0_to_bf16",
            "verdict": "circular",
            "formula": "f8_e8m0_to_bf16 -> our own decode_e8m0",
            "binding_adr": "ADR 0007 §4",
            "reason": "the FP8 shim's f8_e8m0_to_bf16 path uses our own decode_e8m0 formula; ADR 0007 §4 rejects it as circular for the routed I8 real-payload decode question.",
        },
        {
            "candidate": "ds4_cpu_harness",
            "verdict": "built_adjudicated_independent",
            "independence_undetermined": False,
            "binding_adr": "ADR 0007 §4",
            "reason": "independence adjudicated by the 11.43 Metal MSL proof kernel: the new routed I8+F8_E8M0 1-D block-16 dequant (metal/moe.metal) derived from the OCP MX v1.0 spec passes the non-circular Metal-vs-OCP-spec-witness isolation proof at max_abs <= 1e-5 on real DeepSeek-V4 Flash checkpoint bytes. Resolves the ADR-0007 §4 independence gap on the Metal surface. The ds4.c harness remains GGUF-based but independence is fully adjudicated via Metal.",
        },
    ]
    convert_py_mismatch = {
        "assumes_packing": "e2m1fn-fp4-per-32",
        "assertion": "scale.size(1) == in_dim // fp4_block_size",
        "fp4_block_size": 32,
        "expected_scale_dim1_for_real_w1": 64,
        "observed_scale_dim1_for_real_w1": 128,
        "assertion_fails_on_real_checkpoint": True,
        "binding_adr": "ADR 0007 §4",
    }
    trusted_reference_required = (
        "independent routed I8+F8_E8M0 1-D block_size=16 axis=1 full-path dequant reference "
        "(official DeepseekV4 routed-expert dequant op ASSUMES e2m1fn-FP4-per-32 and fails on "
        "the real per-16 scale, so NOT a drop-in reference for THIS checkpoint; DS4-CPU harness "
        "NOT-YET-BUILT and independence-undetermined; our torch/numpy circular; shim's "
        "f8_e8m0_to_bf16 circular; Transformers ships Fp8Dequantize(e4m3) + Mxfp4Dequantize(FP4) "
        "only and delegates expert FP8/int8 to backends that do not define I8+UE8M0 routed micro-block)"
    )
    gap_registry = [
        "no new DS4-CPU harness is built in 11.41; a freshly-written harness re-expressing int8 * decode_e8m0(scale) is circular forbidden slop (ADR 0002 / ADR 0007 §4 / AGENTS.md anti-slop); independence is an Architect determination recorded as a future criterion, not a BA/Coder in-slice build.",
        "official inference/convert.py assumes e2m1fn-FP4-per-32 packing and fails on the real per-16 routed scale; it is NOT a drop-in reference for this checkpoint's genuine I8+F8_E8M0 routed layout.",
        "official inference/kernel.py defines no I8+E8M0 1-D block-16 dequant/gemm op.",
        "Transformers DeepseekV4Experts delegates to @use_experts_implementation backends; none locally define the routed I8+E8M0 micro-block dequant.",
        "our torch/numpy int8 * decode_e8m0(scale) and the FP8 shim's f8_e8m0_to_bf16 are circular per ADR 0007 §4.",
        "packed FP4 expert dequant remains fail-closed; FP4 is absent from real checkpoint expert families (ADR 0007 Context).",
        "B1 (hc_mult>1 multi-layer) and B3 (shimmed-checkpoint load+generation) remain independently required.",
    ]
    independence_test = {
        "policy": "A freshly-written harness that re-expresses int8 * decode_e8m0(scale) is circular (forbidden slop per ADR 0002 / ADR 0007 §4 / AGENTS.md anti-slop) and does NOT count as an independent trusted reference.",
        "independent_means": "authored/derived from a source that does NOT share our dequant formula — e.g. a port of the official DeepseekV4 routed-expert dequant op with the CORRECT per-16 I8+UE8M0 packing, or an independent numerical specification ratified under review.",
        "determination_owner": "Architect (future criterion), recorded here, NOT a BA/Coder in-slice build.",
    }
    no_false_positive_policy = (
        "Recording the trusted-reference landscape (or a future probe that finds a candidate become 'available') "
        "is necessary-but-insufficient and NEVER flips ready. decision/status/fail_closed/ready stay "
        "not-ready/fail-closed/true/false until BOTH (i) an independent trusted routed I8+F8_E8M0 "
        "1-D block_size=16 axis=1 full-path dequant reference that PASSES the independence_test lands "
        "AND (ii) a reviewed real-mode real-payload decode proof (decoded real routed I8 expert == trusted "
        "reference under venv python at <=1e-5) is accepted as a real_mode_proofs entry. Neither happens in 11.41."
    )
    future_proof_criteria = [
        "land an independent trusted routed I8+F8_E8M0 1-D block_size=16 axis=1 full-path dequant reference that PASSES the independence_test (official DeepseekV4 routed-expert dequant op with the correct per-16 I8+UE8M0 packing, or a DS4-CPU harness authored/derived from a source not sharing our dequant formula); a freshly-written re-expression of our int8 * decode_e8m0(scale) does NOT pass.",
        "add a reviewed real-mode real-payload decode proof driving public load_weights() on real routed-expert payloads through _moe_mlx I8 dequant vs the trusted reference at <=1e-5 (isolation) / <=1e-3 (established I8 reference tolerance).",
        "accept that proof as a real_mode_proofs entry before changing any b2_routed_dequant_trusted_reference_readiness verdict field.",
        "only after B0/B1/B2/B3+GATE criteria all close may .deepseek-v4-forward-parity-ok be written; the landscape block alone is never a marker.",
    ]
    return {
        "block": "b2-routed-dequant-trusted-reference-readiness",
        "blocker_id": "B2",
        "blocker": FORWARD_PARITY_BLOCKER_DESCRIPTIONS[2],
        "binding_adr": "ADR 0007 §4",
        "scout": "agent-output/cmux-11-41/dequant-reference-scout.md",
        "probe_status": probe_status,
        "probe_ran": probe_ran,
        "probe_kind": probe_kind,
        "routed_layout": {
            "axis": 1,
            "block_size": 16,
            "weight_dtype": "I8",
            "scale_dtype": "F8_E8M0",
        },
        "metalpath_routed_dequant_independence_adjudicated": True,
        "candidate_reference_landscape": candidate_reference_landscape,
        "aggregate_decode_reference_available": False,
        "convert_py_mismatch": convert_py_mismatch,
        "decode_trusted_reference_available": True,
        "trusted_reference_required": trusted_reference_required,
        "real_payload_decoded": True,
        "proof_available": True,
        "proof_ids": ["B2-a-4"],
        "can_decode_payload": False,
        "read_payload_bytes": False,
        "payload_bytes_decoded": False,
        "dequantize_expert_packed_i8_status": {
            "full_metadata_path": "non-raising (returns decoded floats, Story 11.22, circular per ADR 0007 §4)",
            "no_metadata_guard": "raises NotImplementedError (block_size_missing/scale_axis_missing, guard intact)",
            "fp4_path": "raises (fail-closed; packed FP4 expert dequant remains unproven)",
            "nuance": "Story 11.22 opened the full-metadata i8 dispatch path; 11.43 adjudicates independence via the Metal MSL surface without touching dequantize_expert_packed gate. No gate lift."
        },
        "dequantize_expert_packed_fp4_status": "NotImplementedError (raises)",
        "gap_registry": gap_registry,
        "independence_test": independence_test,
        "no_false_positive_policy": no_false_positive_policy,
        "future_proof_criteria": future_proof_criteria,
        "shim_note": "hf-f8shim (the FP8-shimmed BF16 snapshot) rewrites F8_E4M3/F8_E8M0 to BF16/F32 and leaves I8 routed weights untouched, so it cannot reproduce the ADR-0007 routed I8+F8_E8M0 classification. The shim path's E8M0->BF16 conversion is our own decode_e8m0 formula -> circular, NOT an independent reference per ADR 0007 §4.",
        "status": "fail-closed",
        "decision": "not-ready",
        "fail_closed": True,
        "ready": False,
        "verdict": "independence adjudicated via Metal MSL proof kernel 11.43 (non-circular OCP-spec-derived dequant passes isolation at <= 1e-5 on real HF bytes); full B2 still unsatisfied (packed FP4, expert kernels, B1 hc_mult>1 multi-layer, B3 shim-chp load+gen remain unproven). B2 NOT satisfied (ADR 0007 §4 binding).",
        "evidence": {
            "probe_status": probe_status,
            "probe_ran": probe_ran,
            "import_ok": bool(probe.get("import_ok", False)),
            "import_error": probe.get("import_error"),
            "candidate_count": len(candidate_reference_landscape),
            "convert_py_assertion_fails_on_real_checkpoint": True,
            "read_payload_bytes": False,
            "payload_bytes_decoded": False,
            "official_inference_convert_py_present": probe.get("official_inference_convert_py_present"),
            "official_inference_kernel_py_present": probe.get("official_inference_kernel_py_present"),
            "convert_py_assertion_string_confirmed": probe.get("convert_py_assertion_string_confirmed"),
            "transformers_deepseek_v4_experts_present": probe.get("transformers_deepseek_v4_experts_present"),
            "ds4_c_checked": bool(probe.get("ds4_c_checked", False)),
            "ds4_c_routed_i8_e8m0_block16_op_present": False,
        },
    }


def _b1_hc_mult_multilayer_readiness(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Build the fail-closed B1 hc_mult>1 multi-layer readiness block.

    Gate status is structural evidence only. The verdict stays fail-closed
    until a future reviewed real-mode B1 proof lands in real_mode_proofs.
    """

    probe = dict(probe_result or _default_b1_hc_mult_multilayer_seam_probe())
    probe_status = str(probe.get("probe_status") or "skipped")
    production_gate = probe.get("production_gate") if isinstance(probe.get("production_gate"), dict) else {}
    reference_gate = probe.get("reference_gate") if isinstance(probe.get("reference_gate"), dict) else {}
    transformers_setter = probe.get("transformers_setter") if isinstance(probe.get("transformers_setter"), dict) else {}
    matrix = probe.get("validate_real_mode_matrix") if isinstance(probe.get("validate_real_mode_matrix"), list) else []
    production_gate_rejects = bool(production_gate.get("rejects_multilayer_hc_gt1", False))
    reference_gate_rejects = bool(reference_gate.get("rejects_hc_gt1", False))
    double_blocked = production_gate_rejects and reference_gate_rejects
    import_ok = bool(probe.get("import_ok", False))
    hc_mult_multi_layer_allowed = import_ok and not production_gate_rejects
    if double_blocked:
        verdict = (
            "multi-layer hc_mult>1 real-mode forward is UNPROVEN and double-blocked: "
            "the production _validate_real_mode gate rejects (num_hidden_layers>1, hc_mult>1) "
            "and the trusted _integrated_multilayer_forward reference itself rejects hc_mult>1; "
            "set_transformers_integrated_weights is single-layer only. A B1 proof requires BOTH "
            "a reviewed gate-lift AND a new trusted multi-layer hc_mult>1 reference."
        )
    else:
        verdict = (
            "multi-layer hc_mult>1 real-mode forward readiness could not be confirmed double-blocked "
            f"(probe_status={probe_status}); readiness remains fail-closed until a reviewed "
            "real_mode_proofs B1 proof lands."
        )
    future_proof_criteria = [
        "reviewed lift of the _validate_real_mode multi-layer hc_mult>1 gate (Architect-scoped, Reviewer-approved)",
        "a trusted multi-layer hc_mult>1 reference: either a multi-layer-capable set_transformers_integrated_weights + transformers DeepseekV4Model forward under the venv python, or a composed pure-Python stack lifting the _integrated_multilayer_forward hc=1 guard (_integrated_layer_residual_streams is already hc_mult-agnostic)",
        "real-mode Model(num_hidden_layers in {2,3}, hc_mult=2) forward vs that trusted reference at <=1e-5, recorded as a real_mode_proofs entry",
        "final-hyperhead hc_head collapse + optional norm/lm_head end-to-end after the last layer",
        "B2 (real packed FP4/I8 expert dequant integrated into MoE forward) and B3 (real shimmed-checkpoint load/forward + MLX generation smoke) still independently required",
    ]
    return {
        "block": "b1-hc-mult-multilayer-readiness",
        "blocker_id": "B1",
        "blocker": FORWARD_PARITY_BLOCKER_DESCRIPTIONS[1],
        "proof_available": False,
        "hc_mult_multi_layer_allowed": hc_mult_multi_layer_allowed,
        "double_blocked": double_blocked,
        "probe_status": probe_status,
        "status": "fail-closed",
        "decision": "not-ready",
        "fail_closed": True,
        "ready": False,
        "verdict": verdict,
        "evidence": {
            "import_ok": import_ok,
            "import_error": probe.get("import_error"),
            "validate_real_mode_matrix": list(matrix),
            "production_gate": dict(production_gate),
            "reference_gate": dict(reference_gate),
            "transformers_setter": dict(transformers_setter),
        },
        "b0a_evidence": [
            {
                "id": "B0a-1",
                "name": "real-mode-single-layer-hc1",
                "config": {"num_hidden_layers": 1, "hc_mult": 1},
                "proving_story": "11.32",
                "proves_full_b1": False,
            },
            {
                "id": "B0a-2",
                "name": "real-mode-multilayer-hc1",
                "config": {"num_hidden_layers": 2, "hc_mult": 1},
                "proving_story": "11.32",
                "proves_full_b1": False,
            },
            {
                "id": "B0a-3",
                "name": "real-mode-single-layer-hc2-hyperhead",
                "config": {"num_hidden_layers": 1, "hc_mult": 2},
                "proving_story": "11.32",
                "proves_full_b1": False,
            },
        ],
        "b0a_non_claim": "B0a-1 (single-layer hc=1) + B0a-2 (multi-layer hc=1) + B0a-3 (single-layer hc=2 hyperhead) do NOT compose into a proven stacked multi-layer hc_mult>1 residual-mixing + final-hyperhead end-to-end real-mode forward.",
        "no_false_positive_policy": "Gate-rejection observation is necessary-but-insufficient. Even if a future probe finds the production gate lifted (hc_mult_multi_layer_allowed=true), proof_available stays false and decision/status/fail_closed/ready stay not-ready/fail-closed until a reviewed real_mode_proofs B1 proof entry (real-mode Model(nl in {2,3}, hc_mult=2) forward == a trusted multi-layer hc_mult>1 reference at <=1e-5) lands.",
        "future_proof_criteria": future_proof_criteria,
    }


def _stateful_decode_readiness(probe_result: dict[str, Any]) -> dict[str, Any]:
    """Build the fail-closed stateful/decode readiness block.

    Structural seam detection is necessary-but-insufficient. The readiness
    verdict stays fail-closed until a future reviewed production cache/decode API
    and real-mode stateful parity proof exist.
    """

    probe = dict(probe_result or _default_stateful_decode_seam_probe())
    probe_status = str(probe.get("probe_status") or "skipped")
    seam_available = probe_status == "detected"
    surface = probe.get("surface") if isinstance(probe.get("surface"), dict) else {}
    module_seam_symbols = probe.get("module_seam_symbols") if isinstance(probe.get("module_seam_symbols"), list) else []
    seam_param_tokens = probe.get("seam_param_tokens") if isinstance(probe.get("seam_param_tokens"), list) else list(STATEFUL_DECODE_SEAM_TOKEN_LIST)
    if seam_available:
        verdict = (
            "vendor MLX real-mode production path exposes a structural cache/decode/state token, "
            "but no reviewed production stateful/decode proof exists; readiness remains fail-closed."
        )
    else:
        verdict = (
            "vendor MLX real-mode production path (Model.__call__ / _real_forward / "
            "_real_layer_forward) is stateless one-shot and exposes NO cache / decode / "
            "KV / sliding-window-stateful seam; only cache-less attention surfaces such as "
            "index_topk are visible."
        )
    readiness_requires = [
        "reviewed production cache/decode API design (Architect-scoped, Reviewer-approved) threaded into the real-mode forward",
        "real-mode chunked-stateful output == one-shot prefill against a trusted reference for a permitted tiny CSA config under the venv python",
        "MLA latent c_t^KV + decoupled-RoPE cache lifetime addressed behind its own fail-closed gate",
        "FlashMLA FP8 KV-cache decode addressed behind its own fail-closed gate",
        "B1/B2/B3 still independently required",
    ]
    return {
        "block": "stateful-decode-readiness",
        "seam_available": seam_available,
        "probe_status": probe_status,
        "status": "fail-closed",
        "decision": "not-ready",
        "fail_closed": True,
        "ready": False,
        "verdict": verdict,
        "evidence": {
            "import_ok": bool(probe.get("import_ok", False)),
            "import_error": probe.get("import_error"),
            "seam_param_tokens": list(seam_param_tokens),
            "surface": dict(surface),
            "module_seam_symbols": list(module_seam_symbols),
        },
        "spec_layer_seam": {
            "proven_but_non_production": True,
            "is_production_runtime": False,
            "symbols": [
                "StatefulCSACache.step",
                "StatefulCSACache.step_fusion",
                "IncrementalSlidingKVCache",
                "tiny_greedy_decode",
                "tiny_stateful_csa_fusion_reference",
            ],
            "module": "ds4_ft_mlx.deepseek_v4_attention_spec",
            "proven_in_stories": ["11.16", "11.27", "11.28", "11.29"],
            "note": "spec/reference layer only; proven incremental/cache and stateful CSA fusion evidence is NOT the vendor MLX production runtime and does not unlock real-mode decode.",
        },
        "gap_registry": [
            "production real-mode MLX runtime is stateless one-shot until a reviewed cache/decode API lands",
            "MLA latent c_t^KV + decoupled-RoPE cache lifetime is unproven in the production runtime",
            "DSA / FlashMLA FP8 KV-cache decode is unproven in the production runtime",
            "B1 hc_mult>1 multi-layer, B2 real packed FP4/I8 dequant, and B3 shimmed-checkpoint load+generation remain independently unproven",
        ],
        "no_false_positive_policy": "Seam detection is necessary-but-insufficient. Even if a future probe sets probe_status=detected / seam_available=true, decision stays not-ready and fail_closed stays true until the production stateful path is independently proven (reviewed cache/decode API + real-mode chunked-stateful==prefill).",
        "readiness_requires": readiness_requires,
        "future_proof_criteria": list(readiness_requires),
    }


def _probe_stateful_decode_seam() -> dict[str, Any]:
    """Introspect the vendor real-mode path for cache/decode seam symbols.

    This is deliberately read-only: no Model construction, no weight load, no
    forward/generate call, and no marker or conversion side effects.
    """

    def skipped(exc: Exception) -> dict[str, Any]:
        return {
            "probe_status": "skipped",
            "import_ok": False,
            "import_error": f"MLX real-mode dependencies unavailable: {type(exc).__name__}: {exc}",
            "seam_param_tokens": list(STATEFUL_DECODE_SEAM_TOKEN_LIST),
            "surface": {},
            "module_seam_symbols": [],
        }

    try:
        _ensure_mlx_project_src_on_path()
        from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4

        surfaces = {
            "Model.__call__": deepseek_v4.Model.__call__,
            "Model._real_forward": deepseek_v4.Model._real_forward,
            "Model._real_layer_forward": deepseek_v4.Model._real_layer_forward,
            "_attention_mlx": deepseek_v4._attention_mlx,
            "_csa_attention_mlx": deepseek_v4._csa_attention_mlx,
        }
        surface_report: dict[str, dict[str, Any]] = {}
        any_surface_hit = False
        for name, symbol in surfaces.items():
            params = list(inspect.signature(symbol).parameters)
            has_seam_param = any(_seam_token_hit(param, STATEFUL_DECODE_SEAM_TOKENS) for param in params)
            any_surface_hit = any_surface_hit or has_seam_param
            surface_report[name] = {"params": params, "has_seam_param": has_seam_param}

        module_seam_symbols = sorted(
            symbol for symbol in dir(deepseek_v4)
            if _seam_token_hit(symbol, STATEFUL_DECODE_SEAM_TOKENS)
        )
        probe_status = "detected" if any_surface_hit or module_seam_symbols else "absent"
        return {
            "probe_status": probe_status,
            "import_ok": True,
            "import_error": None,
            "seam_param_tokens": list(STATEFUL_DECODE_SEAM_TOKEN_LIST),
            "surface": surface_report,
            "module_seam_symbols": module_seam_symbols,
        }
    except Exception as exc:
        return skipped(exc)


def _probe_b1_hc_mult_multilayer_seam() -> dict[str, Any]:
    """Introspect the B1 hc_mult>1 multi-layer real-mode readiness seam.

    This probe is guard-only: it performs tiny real-mode construction attempts
    that are expected to reject for (num_hidden_layers>1, hc_mult>1), verifies
    the trusted reference guard, and inspects the transformers setter. It does
    not load weights, run forward/generate, write markers, convert, train, or
    quantize anything.
    """

    def skipped(exc: Exception) -> dict[str, Any]:
        probe = _default_b1_hc_mult_multilayer_seam_probe()
        probe["import_error"] = f"MLX real-mode dependencies unavailable: {type(exc).__name__}: {exc}"
        return probe

    def approx_line(symbol: Any, needle: str) -> int | None:
        try:
            source_lines, start_line = inspect.getsourcelines(symbol)
        except Exception:
            return None
        for offset, line in enumerate(source_lines):
            if needle in line:
                return start_line + offset
        return start_line

    try:
        _ensure_mlx_project_src_on_path()
        from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4

        validate_real_mode_matrix: list[dict[str, Any]] = []
        for num_hidden_layers, hc_mult in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 2)):
            row: dict[str, Any] = {
                "num_hidden_layers": num_hidden_layers,
                "hc_mult": hc_mult,
                "constructs": False,
                "gate_rejects": False,
                "reject_message": None,
            }
            try:
                tiny_args = _build_real_mode_tiny_args(num_hidden_layers, hc_mult)
                deepseek_v4.Model(tiny_args)
                row["constructs"] = True
            except NotImplementedError as exc:
                row["gate_rejects"] = True
                row["reject_message"] = str(exc)
            except Exception as exc:
                row["construct_error"] = f"{type(exc).__name__}: {exc}"
            validate_real_mode_matrix.append(row)

        multilayer_hc_rows = [
            row for row in validate_real_mode_matrix
            if int(row.get("num_hidden_layers", 0)) > 1 and int(row.get("hc_mult", 0)) > 1
        ]
        observed_validate_message = next((row.get("reject_message") for row in multilayer_hc_rows if row.get("reject_message")), None)
        production_gate = {
            "function": "Model._validate_real_mode",
            "rejects_multilayer_hc_gt1": bool(multilayer_hc_rows) and all(bool(row.get("gate_rejects")) for row in multilayer_hc_rows),
            "reject_message_substring": "hc_mult>1 multi-layer parity is not proven",
            "observed_message": observed_validate_message,
            "construction_calls_validate": "self._validate_real_mode()" in inspect.getsource(deepseek_v4.Model.__init__),
            "vendor_reference": "vendor/mlx_lm_models/deepseek_v4.py",
            "approx_line": approx_line(deepseek_v4.Model._validate_real_mode, "hc_mult>1 multi-layer parity is not proven"),
            "line_is_evidence_only": True,
        }

        reference_gate = {
            "function": "_integrated_multilayer_forward",
            "rejects_hc_gt1": False,
            "reject_message_substring": "integrated multi-layer reference currently supports only hc_mult=1",
            "observed_message": None,
            "vendor_reference": "vendor/mlx_lm_models/deepseek_v4.py",
            "approx_line": approx_line(deepseek_v4._integrated_multilayer_forward, "integrated multi-layer reference currently supports only hc_mult=1"),
            "line_is_evidence_only": True,
        }
        try:
            deepseek_v4._integrated_multilayer_forward(_build_real_mode_tiny_args(2, 2), [[0, 1]], {})
        except NotImplementedError as exc:
            reference_gate["rejects_hc_gt1"] = True
            reference_gate["observed_message"] = str(exc)
        except Exception as exc:
            reference_gate["unexpected_error"] = f"{type(exc).__name__}: {exc}"
        else:
            reference_gate["unexpected_error"] = None

        setter_source = inspect.getsource(deepseek_v4.set_transformers_integrated_weights)
        touches_layers_0 = "model.layers[0]" in setter_source
        has_layer_loop = bool(re.search(r"for\s+\w+\s+in\b.*layers", setter_source))
        transformers_setter = {
            "function": "set_transformers_integrated_weights",
            "single_layer_only": touches_layers_0 and not has_layer_loop,
            "touches_layers_0": touches_layers_0,
            "has_layer_loop": has_layer_loop,
        }

        production_rejects = bool(production_gate["rejects_multilayer_hc_gt1"])
        reference_rejects = bool(reference_gate["rejects_hc_gt1"])
        if production_rejects and reference_rejects:
            probe_status = "double-blocked"
        elif production_rejects or reference_rejects:
            probe_status = "partially-open"
        else:
            probe_status = "open"
        return {
            "probe_status": probe_status,
            "import_ok": True,
            "import_error": None,
            "validate_real_mode_matrix": validate_real_mode_matrix,
            "production_gate": production_gate,
            "reference_gate": reference_gate,
            "transformers_setter": transformers_setter,
        }
    except Exception as exc:
        return skipped(exc)


B2_REAL_CHECKPOINT_PAYLOAD_TENSORS = (
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.ffn.experts.0.w1.scale",
    "layers.0.ffn.experts.0.w2.weight",
    "layers.0.ffn.experts.0.w2.scale",
    "layers.0.ffn.experts.0.w3.weight",
    "layers.0.ffn.experts.0.w3.scale",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.shared_experts.w1.scale",
)


def _resolve_b2_payload_checkpoint_dir(hf_model: str | os.PathLike[str] | pathlib.Path | None) -> pathlib.Path:
    explicit = str(hf_model).strip() if hf_model is not None else ""
    if explicit and (explicit != DEFAULT_PATHS["HF_MODEL"] or os.environ.get("HF_MODEL")):
        return path_arg(explicit)
    return path_arg(os.environ.get("HF_MODEL") or os.environ.get("DS4_HF_MODEL") or explicit or DEFAULT_PATHS["HF_MODEL"])


def _probe_b2_real_checkpoint_payload(mlx_work: pathlib.Path, hf_model: str | os.PathLike[str] | pathlib.Path | None) -> dict[str, Any]:
    """Classify real checkpoint expert payload metadata without reading payloads.

    The probe targets the original HF F8 checkpoint. It reads only index/config
    JSON plus safetensors headers for a fixed expert tensor set, and never loads
    tensor bytes, constructs a model, runs forward, writes markers, or converts.
    """

    checkpoint_dir = _resolve_b2_payload_checkpoint_dir(hf_model)

    def skipped(message: str, *, import_ok: bool = False) -> dict[str, Any]:
        probe = _default_b2_real_checkpoint_payload_probe()
        probe["import_ok"] = import_ok
        probe["import_error"] = message
        probe["checkpoint_dir"] = str(checkpoint_dir)
        return probe

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not checkpoint_dir.is_dir():
        return skipped(f"{checkpoint_dir}: original HF F8 checkpoint directory not present")
    if not index_path.is_file():
        return skipped(f"{index_path}: missing model.safetensors.index.json")

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return skipped(f"{index_path}: could not read index JSON: {type(exc).__name__}: {exc}")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return skipped(f"{index_path}: expected object weight_map")
    missing_names = [name for name in B2_REAL_CHECKPOINT_PAYLOAD_TENSORS if name not in weight_map]
    if missing_names:
        return skipped(f"{index_path}: missing bounded B2 tensor(s): {', '.join(missing_names[:3])}")

    try:
        _ensure_mlx_project_src_on_path()
        from ds4_ft_mlx import deepseek_v4_dequant as dq
    except Exception as exc:
        return skipped(f"DeepSeek V4 dequant classifier unavailable: {type(exc).__name__}: {exc}")

    declared_quant = None
    config_path = checkpoint_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            quant = config.get("quantization_config") if isinstance(config, dict) else None
            declared_quant = dict(quant) if isinstance(quant, dict) else None
        except Exception:
            declared_quant = None

    try:
        header: dict[str, dict[str, Any]] = {}
        header_cache: dict[pathlib.Path, dict[str, Any]] = {}
        for name in B2_REAL_CHECKPOINT_PAYLOAD_TENSORS:
            shard_name = str(weight_map[name])
            shard_path = checkpoint_dir / shard_name
            if not shard_path.is_file():
                return skipped(f"{shard_path}: missing referenced safetensors shard", import_ok=True)
            if shard_path not in header_cache:
                header_cache[shard_path] = dq.read_safetensors_header(shard_path, max_bytes=128 * 1024 * 1024)
            shard_header = header_cache[shard_path]
            if name not in shard_header:
                return skipped(f"{shard_path}: header missing tensor {name}", import_ok=True)
            meta = shard_header[name]
            header[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"])}

        classification = dq.classify_checkpoint_expert_packing(header, declared_quant=declared_quant)
        routed_layout = None
        routed_layout_status = "unresolved"
        reconciled = None
        try:
            routed_layout = dq.resolve_routed_block_layout(classification)
            reconciled = dq.reconcile_routed_block_layout(classification)
            routed_layout_status = str(reconciled.get("status") or "shape_authoritative") if isinstance(reconciled, dict) else "shape_authoritative"
        except dq.ExpertBlockLayoutError:
            routed_layout_status = "ambiguous"
        except Exception as exc:
            routed_layout_status = f"unresolved: {type(exc).__name__}: {exc}"

        return {
            "probe_status": "classified",
            "import_ok": True,
            "import_error": None,
            "probe_target": "original HF F8 checkpoint (HF_MODEL/DS4_HF_MODEL)",
            "checkpoint_dir": str(checkpoint_dir),
            "index_sha256": sha256_file(index_path),
            "declared_quant": declared_quant,
            "classification": classification,
            "routed_layout": routed_layout,
            "routed_layout_status": routed_layout_status,
            "reconciled": reconciled,
            "read_payload_bytes": False,
            "payload_bytes_decoded": False,
        }
    except Exception as exc:
        return skipped(f"B2 real checkpoint payload probe failed: {type(exc).__name__}: {exc}", import_ok=True)


def _probe_b2_routed_dequant_trusted_reference(mlx_work: pathlib.Path, hf_model: str | os.PathLike[str] | pathlib.Path | None) -> dict[str, Any]:
    """Presence-only trusted-reference landscape probe.

    This probe inspects local source-file presence only. It does not import
    torch/triton/mlx, decode checkpoint payload bytes, execute a candidate
    reference, construct a model, write markers, or re-derive verdicts.
    """

    _ = mlx_work
    probe = _default_b2_routed_dequant_trusted_reference_probe()
    checkpoint_dir = _resolve_b2_payload_checkpoint_dir(hf_model)
    convert_path = checkpoint_dir / "inference" / "convert.py"
    kernel_path = checkpoint_dir / "inference" / "kernel.py"
    any_presence_check = False
    errors: list[str] = []

    if checkpoint_dir.is_dir():
        probe["official_inference_convert_py_present"] = convert_path.is_file()
        probe["official_inference_kernel_py_present"] = kernel_path.is_file()
        any_presence_check = any_presence_check or bool(probe["official_inference_convert_py_present"] or probe["official_inference_kernel_py_present"])
        if convert_path.is_file():
            try:
                text = convert_path.read_text(encoding="utf-8")
                compact = re.sub(r"\s+", "", text)
                probe["convert_py_assertion_string_confirmed"] = (
                    "fp4_block_size=32" in compact
                    and "scale.size(1)==in_dim//fp4_block_size" in compact
                )
            except Exception as exc:
                probe["convert_py_assertion_string_confirmed"] = False
                errors.append(f"{convert_path}: {type(exc).__name__}: {exc}")
        else:
            probe["convert_py_assertion_string_confirmed"] = None
    else:
        errors.append(f"{checkpoint_dir}: checkpoint directory not present")

    try:
        transformers_spec = importlib.util.find_spec("transformers")
        if transformers_spec is None:
            probe["transformers_deepseek_v4_experts_present"] = None
            errors.append("transformers package not importable")
        else:
            any_presence_check = True
            model_path = None
            locations = getattr(transformers_spec, "submodule_search_locations", None)
            if locations:
                model_path = pathlib.Path(next(iter(locations))) / "models" / "deepseek_v4" / "modeling_deepseek_v4.py"
            if model_path is not None and model_path.is_file():
                text = model_path.read_text(encoding="utf-8")
                probe["transformers_deepseek_v4_experts_present"] = "class DeepseekV4Experts" in text
            else:
                probe["transformers_deepseek_v4_experts_present"] = False
    except Exception as exc:
        probe["transformers_deepseek_v4_experts_present"] = None
        errors.append(f"transformers presence check failed: {type(exc).__name__}: {exc}")

    ds4_roots = []
    for candidate in (os.environ.get("DS4_ROOT"), DEFAULT_PATHS.get("DS4_ROOT"), pathlib.Path(__file__).resolve().parents[1]):
        if candidate is None:
            continue
        path = path_arg(candidate)
        if path not in ds4_roots:
            ds4_roots.append(path)
    ds4_c_path = next((root / "ds4.c" for root in ds4_roots if (root / "ds4.c").is_file()), None)
    if ds4_c_path is not None:
        try:
            ds4_c_path.read_text(encoding="utf-8", errors="ignore")
            probe["ds4_c_checked"] = True
            any_presence_check = True
        except Exception as exc:
            probe["ds4_c_checked"] = False
            errors.append(f"{ds4_c_path}: {type(exc).__name__}: {exc}")
    else:
        errors.append("ds4.c not found for DS4-CPU harness presence check")
    probe["ds4_c_routed_i8_e8m0_block16_op_present"] = False
    probe["read_payload_bytes"] = False
    probe["payload_bytes_decoded"] = False

    if not any_presence_check:
        skipped = _default_b2_routed_dequant_trusted_reference_probe()
        skipped["import_error"] = "; ".join(errors) if errors else "B2 routed dequant trusted-reference probe was skipped"
        return skipped
    probe["probe_status"] = "present-probed"
    probe["probe_ran"] = True
    probe["probe_kind"] = "header-only/introspection"
    probe["import_ok"] = True
    probe["import_error"] = None if not errors else "; ".join(errors)
    return probe


def _run_csa_topk_primitive_proof(spec: dict[str, Any]) -> dict[str, Any]:
    """Harness-only DSA strict top-k sparse-selection primitive proof (B0b-a-2)."""

    entry = {
        "id": spec["id"],
        "name": spec["name"],
        "granularity": spec["granularity"],
        "status": "failed",
        "max_abs_error": None,
        "pruning_non_degenerate": False,
        "pruning_diff_vs_full_topk": None,
        "config": dict(spec["config"]),
        "reference": spec["reference"],
        "covered": list(spec["covered"]),
        "not_covered": list(spec["not_covered"]),
    }
    config = entry["config"]
    try:
        _ensure_mlx_project_src_on_path()
        import mlx.core as mx
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_compressor_indexer_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _csa_attention_mlx

        args = _build_real_mode_tiny_args(
            int(config["num_hidden_layers"]),
            int(config["hc_mult"]),
            compression_ratio=4,
            index_n_heads=2,
            index_head_dim=2,
        )
        seq_len = int(config["seq_len"])
        index_topk = int(config["index_topk"])
        compressed_len = int(config["compressed_len"])
        if compressed_len != seq_len // int(config["compression_ratio"]):
            raise ValueError("B0b-a-2 compressed_len must match seq_len // compression_ratio")
        if not index_topk < compressed_len:
            raise ValueError("B0b-a-2 requires index_topk < compressed_len")

        weights = _make_real_mode_csa_weights()
        spec_obj = DeepSeekV4AttentionSpec(
            args.hidden_size,
            args.num_attention_heads,
            args.head_dim,
            args.q_lora_rank,
            args.o_lora_rank,
            args.qk_rope_head_dim,
            args.o_groups,
            args.compression_ratio,
            args.index_n_heads,
            args.index_head_dim,
        )
        x = _csa_topk_proof_input(seq_len, args.hidden_size)
        mx_weights = {key: mx.array(value) for key, value in weights.items()}

        expected = tiny_compressor_indexer_attention_reference(
            spec_obj,
            x,
            x,
            weights,
            rms_norm_eps=args.rms_norm_eps,
            rope_theta=args.compress_rope_theta,
            index_topk=index_topk,
        )["attended"]
        got = _csa_attention_mlx(args, mx.array([x]), mx_weights, index_topk=index_topk).tolist()[0]
        max_abs_error = _nested_max_abs_error(got, expected)
        full_topk = _csa_attention_mlx(args, mx.array([x]), mx_weights, index_topk=compressed_len).tolist()[0]
        pruning_diff = _nested_max_abs_error(got, full_topk)
        pruning_non_degenerate = pruning_diff > REAL_MODE_FORWARD_TOLERANCE

        entry["max_abs_error"] = max_abs_error
        entry["pruning_diff_vs_full_topk"] = pruning_diff
        entry["pruning_non_degenerate"] = pruning_non_degenerate
        entry["status"], reason = _csa_topk_proof_status(max_abs_error, pruning_non_degenerate, REAL_MODE_FORWARD_TOLERANCE)
        if reason:
            entry["reason"] = reason
    except (ImportError, ModuleNotFoundError) as exc:
        entry["status"] = "skipped"
        entry["reason"] = f"MLX real-mode dependencies unavailable: {exc}"
    except NotImplementedError as exc:
        if "mlx is required" in str(exc).lower() or "mlx" in str(exc).lower() and "not installed" in str(exc).lower():
            entry["status"] = "skipped"
            entry["reason"] = f"MLX real-mode dependencies unavailable: {exc}"
        else:
            entry["status"] = "failed"
            entry["reason"] = str(exc)
    except Exception as exc:
        entry["status"] = "failed"
        entry["reason"] = f"{type(exc).__name__}: {exc}"
    return entry


def _run_i8_dequant_integration_proof(spec: dict[str, Any]) -> dict[str, Any]:
    """Harness-only B2-a proofs for I8 block-scale dequant wired into real-mode MoE."""

    reference_tolerance = 1e-3
    entry = {
        "id": spec["id"],
        "name": spec["name"],
        "evidence_class": spec.get("evidence_class", "B2-partial"),
        "status": "failed",
        "max_abs_error": None,
        "reference_max_abs_error": None,
        "reference_tolerance": reference_tolerance,
        "quantization_gap": None,
        "non_degenerate": False,
        "block_scale_nonunit": False,
        "i8_branch_reached": False,
        "config": dict(spec["config"]),
        "reference": spec["reference"],
        "isolation_reference": spec.get("isolation_reference"),
        "covered": list(spec["covered"]),
        "not_covered": list(spec["not_covered"]),
    }
    config = entry["config"]
    try:
        _ensure_mlx_project_src_on_path()
        import mlx.core as mx
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_routing
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
            Model,
            _dequantize_i8_block_scale_mlx,
            _integrated_layer_forward,
            _integrated_multilayer_forward,
        )

        block_size = int(config["block_size"])
        scale_axis = int(config["scale_axis"])
        n_routed_experts = int(config["n_routed_experts"])

        def quantize_i8(weight_rows: list[list[float]]):
            weight = mx.array(weight_rows, dtype=mx.float32)
            amax = mx.max(mx.abs(weight), axis=scale_axis, keepdims=True)
            scale = amax / 127.0
            scale = mx.where(scale == 0, mx.ones_like(scale) * 1e-6, scale)
            i8 = mx.clip(mx.round(weight / scale), -128, 127).astype(mx.int8)
            return i8, scale.astype(mx.bfloat16)

        def quantized_experts(true_experts: dict[str, list[list[float]]]) -> dict[str, Any]:
            i8_weights: dict[str, Any] = {}
            scale_weights: dict[str, Any] = {}
            dequant_weights: dict[str, Any] = {}
            quantization_gap = 0.0
            block_scale_nonunit = False
            for eid in range(n_routed_experts):
                for proj in ("w1", "w2", "w3"):
                    key = f"mlp.experts.{eid}.{proj}.weight"
                    scale_key = f"mlp.experts.{eid}.{proj}.scale"
                    i8, scale = quantize_i8(true_experts[key])
                    dequant = _dequantize_i8_block_scale_mlx(i8, scale, block_size=block_size, axis=scale_axis)
                    i8_weights[key] = i8
                    scale_weights[scale_key] = scale
                    dequant_weights[key] = dequant
                    true_float = mx.array(true_experts[key], dtype=mx.float32)
                    quantization_gap = max(quantization_gap, float(mx.max(mx.abs(dequant - true_float))))
                    block_scale_nonunit = block_scale_nonunit or float(mx.max(mx.abs(scale.astype(mx.float32) - 1.0))) > 1e-6
            return {
                "i8_weights": i8_weights,
                "scale_weights": scale_weights,
                "dequant_weights": dequant_weights,
                "quantization_gap": quantization_gap,
                "block_scale_nonunit": block_scale_nonunit,
            }

        def args_for(num_hidden_layers: int, expert_dtype: str, *, num_experts_per_tok: int | None = None):
            return _build_real_mode_tiny_args(
                num_hidden_layers,
                int(config["hc_mult"]),
                hidden_size=int(config["hidden_size"]),
                moe_intermediate_size=int(config["moe_intermediate_size"]),
                expert_dtype=expert_dtype,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=int(config["num_experts_per_tok"] if num_experts_per_tok is None else num_experts_per_tok),
            )

        def routing_facts(base_weights: dict[str, Any], num_experts_per_tok: int) -> dict[str, Any]:
            probe = _csa_topk_proof_input(int(config["seq_len"]), hidden_size=int(config["hidden_size"]))
            moe_config = MoEConfig(
                hidden_size=int(config["hidden_size"]),
                moe_intermediate_size=int(config["moe_intermediate_size"]),
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=int(config.get("n_shared_experts", 1)),
            )
            router_weights = {
                "router.weight": base_weights["mlp.gate.weight"],
                "router.e_score_correction_bias": base_weights.get(
                    "mlp.gate.e_score_correction_bias",
                    [0.0 for _ in range(n_routed_experts)],
                ),
            }
            routing = tiny_topk_moe_routing(moe_config, probe, router_weights)
            bias = [float(value) for value in router_weights["router.e_score_correction_bias"]]
            min_margin = float("inf")
            subset_stable = True
            distinct_selected = True
            has_unselected = True
            for row in routing:
                scores = [float(value) for value in row["scores"]]
                selected = [int(value) for value in row["selected"]]
                distinct_selected = distinct_selected and len(set(selected)) == num_experts_per_tok
                has_unselected = has_unselected and len(set(selected)) < n_routed_experts
                selection_scores = [scores[eid] + bias[eid] for eid in range(n_routed_experts)]
                order = sorted(range(n_routed_experts), key=lambda eid: (-selection_scores[eid], eid))
                if num_experts_per_tok < n_routed_experts:
                    margin = selection_scores[order[num_experts_per_tok - 1]] - selection_scores[order[num_experts_per_tok]]
                    min_margin = min(min_margin, margin)
                mlx_subset = sorted(mx.argsort(-mx.array([selection_scores], dtype=mx.float32), axis=-1)[..., :num_experts_per_tok].tolist()[0])
                subset_stable = subset_stable and mlx_subset == sorted(order[:num_experts_per_tok])
            proper_subset = num_experts_per_tok < n_routed_experts
            tie_free_margin = 0.0 if min_margin == float("inf") else float(min_margin)
            return {
                "multi_expert_combine_exercised": bool(proper_subset and distinct_selected and has_unselected),
                "num_experts_per_tok": num_experts_per_tok,
                "n_routed_experts": n_routed_experts,
                "tie_free_margin": tie_free_margin,
                "routing_subset_stable": bool(subset_stable),
            }

        def run_single_layer_case(num_experts_per_tok: int, *, branch_probe: bool) -> dict[str, Any]:
            payload = _make_real_mode_i8_dequant_weights(n_routed_experts=n_routed_experts)
            base_weights = dict(payload["base_weights"])
            quantized = quantized_experts(dict(payload["true_experts"]))

            w_i8 = dict(base_weights)
            w_raw = dict(base_weights)
            w_ref = dict(base_weights)
            for key, value in quantized["i8_weights"].items():
                w_i8[key] = value
            for key, value in quantized["scale_weights"].items():
                w_i8[key] = value
            for key, value in quantized["dequant_weights"].items():
                w_raw[key] = value
                w_ref[key] = value.tolist()

            args_i8 = args_for(1, "i8", num_experts_per_tok=num_experts_per_tok)
            args_raw = args_for(1, "fp4", num_experts_per_tok=num_experts_per_tok)
            input_ids = [list(range(int(config["seq_len"])))]

            model_i8 = Model(args_i8)
            model_i8.load_weights(w_i8)
            got_i8 = model_i8(input_ids)

            model_raw = Model(args_raw)
            model_raw.load_weights(w_raw)
            got_raw = model_raw(input_ids)

            expected_ref = _integrated_layer_forward(args_raw, input_ids, w_ref)
            i8_branch_reached = True
            if branch_probe:
                i8_branch_reached = False
                try:
                    missing_scale = dict(w_i8)
                    del missing_scale["mlp.experts.0.w1.scale"]
                    Model(args_i8).load_weights(missing_scale)
                except Exception as exc:
                    i8_branch_reached = "scale" in str(exc).lower() or "missing" in str(exc).lower()

            return {
                "max_abs_error": _nested_max_abs_error(got_i8, got_raw),
                "reference_max_abs_error": _nested_max_abs_error(got_i8, expected_ref),
                "quantization_gap": quantized["quantization_gap"],
                "block_scale_nonunit": quantized["block_scale_nonunit"],
                "i8_branch_reached": i8_branch_reached,
                "routing_facts": routing_facts(base_weights, num_experts_per_tok),
            }

        def run_multilayer_case(num_hidden_layers: int, *, branch_probe: bool) -> dict[str, Any]:
            payload = _make_real_mode_multilayer_i8_dequant_weights(num_hidden_layers)
            quantized = quantized_experts(dict(payload["true_experts"]))

            w_i8 = {"embed.weight": payload["embed.weight"]}
            w_raw = {"embed.weight": payload["embed.weight"]}
            w_ref = {"embed.weight": payload["embed.weight"]}
            for layer_idx in range(num_hidden_layers):
                prefix = f"layers.{layer_idx}."
                for key, value in payload["non_expert_by_layer"][layer_idx].items():
                    w_i8[f"{prefix}{key}"] = value
                    w_raw[f"{prefix}{key}"] = value
                    w_ref[f"{prefix}{key}"] = value
                for key, value in quantized["i8_weights"].items():
                    w_i8[f"{prefix}{key}"] = value
                for key, value in quantized["scale_weights"].items():
                    w_i8[f"{prefix}{key}"] = value
                for key, value in quantized["dequant_weights"].items():
                    w_raw[f"{prefix}{key}"] = value
                    w_ref[f"{prefix}{key}"] = value.tolist()

            args_i8 = args_for(num_hidden_layers, "i8")
            args_raw = args_for(num_hidden_layers, "fp4")
            input_ids = [list(range(int(config["seq_len"])))]

            model_i8 = Model(args_i8)
            model_i8.load_weights(w_i8)
            got_i8 = model_i8(input_ids)

            model_raw = Model(args_raw)
            model_raw.load_weights(w_raw)
            got_raw = model_raw(input_ids)

            expected_ref = _integrated_multilayer_forward(args_raw, input_ids, w_ref)
            i8_branch_reached = True
            if branch_probe:
                i8_branch_reached = False
                try:
                    missing_scale = dict(w_i8)
                    del missing_scale["layers.0.mlp.experts.0.w1.scale"]
                    Model(args_i8).load_weights(missing_scale)
                except Exception as exc:
                    i8_branch_reached = "scale" in str(exc).lower() or "missing" in str(exc).lower()

            return {
                "max_abs_error": _nested_max_abs_error(got_i8, got_raw),
                "reference_max_abs_error": _nested_max_abs_error(got_i8, expected_ref),
                "quantization_gap": quantized["quantization_gap"],
                "block_scale_nonunit": quantized["block_scale_nonunit"],
                "i8_branch_reached": i8_branch_reached,
            }

        num_hidden_layers = int(config["num_hidden_layers"])
        if num_hidden_layers == 1 and spec["reference"] == "_integrated_layer_forward":
            primary_topk = int(config["num_experts_per_tok"])
            result = run_single_layer_case(primary_topk, branch_probe=True)
            if primary_topk > 1:
                entry.update(result["routing_facts"])
                secondary_topk = config.get("secondary_num_experts_per_tok")
                if secondary_topk is not None:
                    secondary_result = run_single_layer_case(int(secondary_topk), branch_probe=False)
                    secondary_ok = (
                        secondary_result["max_abs_error"] <= REAL_MODE_FORWARD_TOLERANCE
                        and secondary_result["reference_max_abs_error"] <= reference_tolerance
                    )
                    entry["secondary_topk"] = {
                        "num_experts_per_tok": int(secondary_topk),
                        "max_abs_error": secondary_result["max_abs_error"],
                        "reference_max_abs_error": secondary_result["reference_max_abs_error"],
                        "ok": secondary_ok,
                    }
        elif num_hidden_layers >= 2 and spec["reference"] == "_integrated_multilayer_forward":
            result = run_multilayer_case(num_hidden_layers, branch_probe=True)
            secondary_num_hidden_layers = config.get("secondary_num_hidden_layers")
            if secondary_num_hidden_layers is not None:
                secondary_result = run_multilayer_case(int(secondary_num_hidden_layers), branch_probe=False)
                secondary_ok = (
                    secondary_result["max_abs_error"] <= REAL_MODE_FORWARD_TOLERANCE
                    and secondary_result["reference_max_abs_error"] <= reference_tolerance
                )
                entry["secondary_multilayer"] = {
                    "num_hidden_layers": int(secondary_num_hidden_layers),
                    "max_abs_error": secondary_result["max_abs_error"],
                    "reference_max_abs_error": secondary_result["reference_max_abs_error"],
                    "ok": secondary_ok,
                }
        else:
            raise ValueError(f"unsupported I8 dequant proof shape: layers={num_hidden_layers}, reference={spec['reference']!r}")

        isolation_error = result["max_abs_error"]
        reference_error = result["reference_max_abs_error"]
        non_degenerate = result["quantization_gap"] > REAL_MODE_FORWARD_TOLERANCE
        i8_branch_reached = bool(result["i8_branch_reached"])

        entry["max_abs_error"] = isolation_error
        entry["reference_max_abs_error"] = reference_error
        entry["quantization_gap"] = result["quantization_gap"]
        entry["non_degenerate"] = non_degenerate
        entry["block_scale_nonunit"] = result["block_scale_nonunit"]
        entry["i8_branch_reached"] = i8_branch_reached
        entry["status"], reason = _i8_dequant_proof_status(
            isolation_error,
            reference_error,
            reference_tolerance,
            non_degenerate,
            i8_branch_reached,
            REAL_MODE_FORWARD_TOLERANCE,
        )
        secondary_multilayer = entry.get("secondary_multilayer")
        if entry["status"] == "ok" and secondary_multilayer is not None and not secondary_multilayer["ok"]:
            entry["status"] = "failed"
            reason = f"secondary num_hidden_layers={secondary_multilayer['num_hidden_layers']} check exceeded tolerance"
        if entry["status"] == "ok" and "multi_expert_combine_exercised" in entry:
            if not entry["multi_expert_combine_exercised"]:
                entry["status"] = "failed"
                reason = "multi-expert combine not exercised (no proper subset / non-distinct selection / no unselected expert)"
            elif entry["tie_free_margin"] < TIE_FREE_MARGIN_EPS:
                entry["status"] = "failed"
                reason = f"tie_free_margin {entry['tie_free_margin']:.6g} below {TIE_FREE_MARGIN_EPS:.6g}"
            elif not entry["routing_subset_stable"]:
                entry["status"] = "failed"
                reason = "routing subset unstable between float32 MLX argsort and float64 Python reference"
        secondary_topk = entry.get("secondary_topk")
        if entry["status"] == "ok" and secondary_topk is not None and not secondary_topk["ok"]:
            entry["status"] = "failed"
            reason = f"secondary num_experts_per_tok={secondary_topk['num_experts_per_tok']} check exceeded tolerance"
        if reason:
            entry["reason"] = reason
    except (ImportError, ModuleNotFoundError) as exc:
        entry["status"] = "skipped"
        entry["reason"] = f"MLX real-mode dependencies unavailable: {exc}"
    except NotImplementedError as exc:
        if "mlx is required" in str(exc).lower() or "mlx" in str(exc).lower() and "not installed" in str(exc).lower():
            entry["status"] = "skipped"
            entry["reason"] = f"MLX real-mode dependencies unavailable: {exc}"
        else:
            entry["status"] = "failed"
            entry["reason"] = str(exc)
    except Exception as exc:
        entry["status"] = "failed"
        entry["reason"] = f"{type(exc).__name__}: {exc}"
    return entry

def _run_one_real_mode_forward_proof(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["id"] == "B0b-a-2":
        return _run_csa_topk_primitive_proof(spec)
    if spec["id"] in {"B2-a-1", "B2-a-2", "B2-a-3"}:
        return _run_i8_dequant_integration_proof(spec)

    entry = {
        "id": spec["id"],
        "name": spec["name"],
        "status": "failed",
        "max_abs_error": None,
        "config": dict(spec["config"]),
        "reference": spec["reference"],
        "covered": list(spec["covered"]),
        "not_covered": list(spec["not_covered"]),
    }
    config = entry["config"]
    seq_len = int(config.get("seq_len", 2))
    input_ids = [list(range(seq_len))]
    try:
        _ensure_mlx_project_src_on_path()
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_hyperhead_collapse
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_layer_forward, _integrated_multilayer_forward

        if int(config.get("compression_ratio", 0)) == 4:
            args = _build_real_mode_tiny_args(
                int(config["num_hidden_layers"]),
                int(config["hc_mult"]),
                compression_ratio=4,
                index_n_heads=2,
                index_head_dim=2,
            )
            weights = _make_real_mode_csa_weights()
        else:
            args = _build_real_mode_tiny_args(
                int(config["num_hidden_layers"]),
                int(config["hc_mult"]),
            )
            if entry["id"] == "B0a-2":
                weights = _make_real_mode_multilayer_weights(int(config["num_hidden_layers"]))
            else:
                weights = _make_real_mode_single_layer_weights(hc_mult=int(config["hc_mult"]))

        model = Model(args)
        model.load_weights(weights)
        got = model(input_ids)

        if entry["reference"] == "_integrated_layer_forward":
            expected = _integrated_layer_forward(args, input_ids, weights)
        elif entry["reference"] == "_integrated_multilayer_forward":
            expected = _integrated_multilayer_forward(args, input_ids, weights)
        elif entry["reference"] == "_integrated_layer_forward+tiny_hyperhead_collapse":
            streams = _integrated_layer_forward(args, input_ids, weights)
            expected = [
                tiny_hyperhead_collapse(
                    hidden_streams=batch_streams,
                    fn=weights["hc_head.fn"],
                    base=weights["hc_head.base"],
                    scale=weights["hc_head.scale"][0],
                    hc_mult=args.hc_mult,
                    eps=args.hc_eps,
                    rms_norm_eps=args.rms_norm_eps,
                )["collapsed"]
                for batch_streams in streams
            ]
        else:
            raise ValueError(f"unknown real-mode proof reference {entry['reference']!r}")

        max_abs_error = _nested_max_abs_error(got, expected)
        entry["max_abs_error"] = max_abs_error
        if max_abs_error <= REAL_MODE_FORWARD_TOLERANCE:
            entry["status"] = "ok"
        else:
            entry["status"] = "failed"
            entry["reason"] = f"max_abs_error {max_abs_error:.6g} exceeds tolerance {REAL_MODE_FORWARD_TOLERANCE:.6g}"
    except (ImportError, ModuleNotFoundError) as exc:
        entry["status"] = "skipped"
        entry["reason"] = f"MLX real-mode dependencies unavailable: {exc}"
    except NotImplementedError as exc:
        if "mlx is required" in str(exc).lower() or "mlx" in str(exc).lower() and "not installed" in str(exc).lower():
            entry["status"] = "skipped"
            entry["reason"] = f"MLX real-mode dependencies unavailable: {exc}"
        else:
            entry["status"] = "failed"
            entry["reason"] = str(exc)
    except Exception as exc:
        entry["status"] = "failed"
        entry["reason"] = f"{type(exc).__name__}: {exc}"
    return entry


def _run_real_mode_forward_proofs(mlx_work: pathlib.Path) -> dict[str, Any]:
    """Run B0 partial real-mode integrated forward proofs without touching gate markers."""

    _ = mlx_work
    proofs = [_run_one_real_mode_forward_proof(spec) for spec in _real_mode_forward_proof_specs()]
    return _real_mode_proofs_report(proofs)


def _tiny_forward_parity_model_config() -> dict[str, Any]:
    return {
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "n_shared_experts": 1,
        "expert_dtype": "fp4",
        "rms_norm_eps": 1e-6,
        "hc_mult": 1,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "o_groups": 1,
    }


def _try_construct_tiny_forward_parity_model() -> bool:
    try:
        _ensure_mlx_project_src_on_path()
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

        Model(ModelArgs.from_dict(_tiny_forward_parity_model_config()))
    except Exception:
        return False
    return True


def _collect_non_forward_gate_status(mlx_work: pathlib.Path) -> dict[str, str]:
    markers = (
        ".deepseek-v4-import-ok",
        ".deepseek-v4-tiny-config-ok",
        ".deepseek-v4-mapping-ok",
        ".deepseek-v4-dequant-parity-ok",
    )
    status = {marker: "present" if (mlx_work / marker).is_file() else "absent" for marker in markers}
    mtp_marker = ".deepseek-v4-mtp-exclusion-ok"
    status[mtp_marker] = "present" if (mlx_work / mtp_marker).is_file() else "not-required"
    return {
        ".deepseek-v4-import-ok": status[".deepseek-v4-import-ok"],
        ".deepseek-v4-tiny-config-ok": status[".deepseek-v4-tiny-config-ok"],
        ".deepseek-v4-mapping-ok": status[".deepseek-v4-mapping-ok"],
        ".deepseek-v4-mtp-exclusion-ok": status[".deepseek-v4-mtp-exclusion-ok"],
        ".deepseek-v4-dequant-parity-ok": status[".deepseek-v4-dequant-parity-ok"],
    }


def forward_parity_blockers() -> tuple[str, ...]:
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import forward_parity_blockers as _forward_parity_blockers

    return tuple(_forward_parity_blockers())


def build_forward_parity_readiness_report(
    *,
    mlx_work: pathlib.Path,
    partials: Iterable[dict[str, Any]],
    blockers: Iterable[str],
    marker_present: bool,
    model_constructs: bool,
    non_forward_gate_status: dict[str, str],
    generated_at: str,
    real_mode_proofs: dict[str, Any] | None = None,
    stateful_decode_seam_probe: dict[str, Any] | None = None,
    b1_hc_mult_multilayer_seam_probe: dict[str, Any] | None = None,
    b2_real_checkpoint_payload_seam_probe: dict[str, Any] | None = None,
    b2_routed_dequant_trusted_reference_seam_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provided_blockers = list(blockers)
    blockers_list = [
        provided_blockers[index] if index < len(provided_blockers) else description
        for index, description in enumerate(FORWARD_PARITY_BLOCKER_DESCRIPTIONS)
    ]
    normalised_partials = [
        _normalise_forward_parity_partial(
            FORWARD_PARITY_FIXTURE_NAMES[index] if index < len(FORWARD_PARITY_FIXTURE_NAMES) else f"fixture-{index}",
            partial,
        )
        for index, partial in enumerate(partials)
    ]
    status_counts = {"ok": 0, "skipped": 0, "failed": 0}
    for partial in normalised_partials:
        status = partial.get("status")
        if status == "ok":
            status_counts["ok"] += 1
        elif status == "skipped":
            status_counts["skipped"] += 1
        else:
            status_counts["failed"] += 1

    def criterion_blocker(index: int) -> str:
        if index < len(blockers_list):
            return blockers_list[index]
        return FORWARD_PARITY_BLOCKER_DESCRIPTIONS[index]

    real_mode_proofs_report = _default_real_mode_forward_proofs() if real_mode_proofs is None else dict(real_mode_proofs)
    stateful_decode_readiness = _stateful_decode_readiness(stateful_decode_seam_probe or _default_stateful_decode_seam_probe())
    b1_hc_mult_multilayer_readiness = _b1_hc_mult_multilayer_readiness(
        b1_hc_mult_multilayer_seam_probe or _default_b1_hc_mult_multilayer_seam_probe()
    )
    b2_real_checkpoint_payload_readiness = _b2_real_checkpoint_payload_readiness(
        b2_real_checkpoint_payload_seam_probe or _default_b2_real_checkpoint_payload_probe()
    )
    b2_routed_dequant_trusted_reference_readiness = _b2_routed_dequant_trusted_reference_readiness(
        b2_routed_dequant_trusted_reference_seam_probe or _default_b2_routed_dequant_trusted_reference_probe()
    )

    return {
        "schema": 1,
        "report": "deepseek-v4-forward-parity-readiness",
        "status": "not-ready",
        "gate": "deepseek-v4-forward-parity",
        "generated_at": generated_at,
        "full_forward_parity": False,
        "marker_earned": False,
        "marker_path": str(mlx_work / ".deepseek-v4-forward-parity-ok"),
        "marker_present": bool(marker_present),
        "blockers": blockers_list,
        "blockers_count": len(blockers_list),
        "model_constructs_under_mlx_venv": bool(model_constructs),
        "non_forward_gate_status": dict(non_forward_gate_status),
        "coverage": {
            "fixtures_total": len(normalised_partials),
            "fixtures_ok": status_counts["ok"],
            "fixtures_skipped": status_counts["skipped"],
            "fixtures_failed": status_counts["failed"],
            "partials": normalised_partials,
        },
        "real_mode_proofs": real_mode_proofs_report,
        "stateful_decode_readiness": stateful_decode_readiness,
        "b1_hc_mult_multilayer_readiness": b1_hc_mult_multilayer_readiness,
        "b2_real_checkpoint_payload_readiness": b2_real_checkpoint_payload_readiness,
        "b2_routed_dequant_trusted_reference_readiness": b2_routed_dequant_trusted_reference_readiness,
        "marker_write_criteria": [
            {
                "id": "B0",
                "blocker": criterion_blocker(0),
                "required_proof": "Run the real integrated Model(args, forward_parity_fixture=None) forward end-to-end: load_weights() + __call__/_real_forward compared to a trusted reference (Transformers DeepseekV4Model or DS4) within tolerance, with RoPE/cache/sinks/compressor/indexer wired. Construction of the tiny Model is NOT sufficient.",
            },
            {
                "id": "B1",
                "blocker": criterion_blocker(1),
                "required_proof": "Prove full decoder-layer hyperconnection residual mixing and final hyperhead collapse end-to-end for hc_mult>1 multi-layer against a trusted reference (lift the current hc_mult>1 multi-layer fail-closed in _validate_real_mode).",
            },
            {
                "id": "B2",
                "blocker": criterion_blocker(2),
                "required_proof": "Decode real packed FP4 and real-payload I8 expert weights: dequantize_expert_packed(\"fp4\") must stop raising and match reference; replace synthetic I8 block-scale payloads with real checkpoint payloads; cover expert kernels.",
            },
            {
                "id": "B3",
                "blocker": criterion_blocker(3),
                "required_proof": "Load real MLX_WORK/hf-f8shim weights into Model, run a real forward, and run an mlx_lm.generate generation smoke on the forward path (Track B). DS4-GGUF base generation (.ds4-gguf-generate-ok under DS4_ROOT) is Track A and is NOT this blocker.",
            },
            {
                "id": "GATE",
                "blocker": "reviewed marker-write code path",
                "required_proof": "Even when forward_parity_blockers() becomes empty, a reviewed code path in deepseek_v4_forward_parity_check() must write .deepseek-v4-forward-parity-ok with report_path + report_sha256, where the report carries full_forward_parity=true, and the marker must validate via _validate_forward_parity_marker(). Empty blockers are necessary but NOT sufficient.",
            },
        ],
        "next_steps": [
            "deepseek-v4-forward-parity-check (writes the marker via reviewed code, only after B0-B3+GATE)",
            "convert-shimmed",
            "smoke-train",
        ],
        "non_claims": [
            "no marker written (.deepseek-v4-forward-parity-ok stays absent)",
            "no model-4bit created",
            "no real checkpoint full forward run",
            "no mlx_lm.convert executed",
            "no quantization",
            "no generation smoke",
            "convert-shimmed gate unchanged",
            "component fixture coverage is partial evidence only; it does NOT constitute full_forward_parity",
            "real_mode_proofs is B0 partial evidence only (B0a compressor-free + B0b-a cache-less CSA tiny subset + B0b-a DSA top-k primitive); it does NOT constitute full_forward_parity and does NOT mark B0 satisfied",
        ],
    }


def deepseek_v4_forward_parity_readiness(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    out_path = path_arg(args.out) if getattr(args, "out", None) else mlx_work / "deepseek-v4-forward-parity-readiness.json"
    _validate_forward_parity_readiness_output_path(out_path, mlx_work)
    partials = _run_forward_parity_fixtures(mlx_work)
    real_mode_proofs = _run_real_mode_forward_proofs(mlx_work)
    stateful_decode_seam_probe = _probe_stateful_decode_seam()
    b1_hc_mult_multilayer_seam_probe = _probe_b1_hc_mult_multilayer_seam()
    b2_real_checkpoint_payload_seam_probe = _probe_b2_real_checkpoint_payload(mlx_work, getattr(args, "hf_model", None))
    b2_routed_dequant_trusted_reference_seam_probe = _probe_b2_routed_dequant_trusted_reference(mlx_work, getattr(args, "hf_model", None))
    blockers = list(forward_parity_blockers())
    report = build_forward_parity_readiness_report(
        mlx_work=mlx_work,
        partials=partials,
        blockers=blockers,
        marker_present=(mlx_work / ".deepseek-v4-forward-parity-ok").is_file(),
        model_constructs=_try_construct_tiny_forward_parity_model(),
        non_forward_gate_status=_collect_non_forward_gate_status(mlx_work),
        generated_at=datetime.now(timezone.utc).isoformat(),
        real_mode_proofs=real_mode_proofs,
        stateful_decode_seam_probe=stateful_decode_seam_probe,
        b1_hc_mult_multilayer_seam_probe=b1_hc_mult_multilayer_seam_probe,
        b2_real_checkpoint_payload_seam_probe=b2_real_checkpoint_payload_seam_probe,
        b2_routed_dequant_trusted_reference_seam_probe=b2_routed_dequant_trusted_reference_seam_probe,
    )
    _write_json_atomic(out_path, report)
    print(out_path)
    print("full_forward_parity=false")
    print("marker_earned=false")
    print(f"blockers={len(blockers)}")
    print("NOT READY: forward parity not earned; see marker_write_criteria")
    return 0


def deepseek_v4_forward_parity_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".deepseek-v4-forward-parity-ok")
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_csa_compressor_fixture, run_tiny_csa_indexer_scorer_fixture, run_tiny_csa_topk_indexer_fixture, run_tiny_hca_compressor_fixture, run_tiny_hyperconnection_fixture, run_tiny_hyperconnection_hc2_fixture, run_tiny_hyperconnection_hc2_transformers_fixture, run_tiny_hyperhead_fixture, run_tiny_hyperhead_transformers_fixture, run_tiny_rope_tail_fixture, run_tiny_sink_cache_inverse_rope_fixture, run_tiny_sliding_attention_fixture
    from ds4_ft_mlx.deepseek_v4_moe_spec import run_tiny_hash_moe_fixture, run_tiny_topk_moe_fixture, run_tiny_topk_moe_i8_fixture
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs, forward_parity_blockers, run_embedding_rmsnorm_head_fixture, run_integrated_layer_fixture, run_integrated_layer_hc_mult_fixture, run_tiny_csa_attention_mlx_fixture

    partial = run_embedding_rmsnorm_head_fixture()
    attention_partial = run_tiny_sliding_attention_fixture()
    rope_partial = run_tiny_rope_tail_fixture()
    sink_cache_partial = run_tiny_sink_cache_inverse_rope_fixture()
    hyperconnection_partial = run_tiny_hyperconnection_fixture()
    hyperconnection_hc2_partial = run_tiny_hyperconnection_hc2_fixture()
    hyperconnection_hc2_transformers_partial = run_tiny_hyperconnection_hc2_transformers_fixture()
    csa_compressor_partial = run_tiny_csa_compressor_fixture()
    csa_indexer_partial = run_tiny_csa_topk_indexer_fixture()
    hca_compressor_partial = run_tiny_hca_compressor_fixture()
    hyperhead_partial = run_tiny_hyperhead_fixture()
    moe_partial = run_tiny_topk_moe_fixture()
    hash_moe_partial = run_tiny_hash_moe_fixture()
    moe_i8_partial = run_tiny_topk_moe_i8_fixture()
    integrated_layer_partial = _compute_integrated_layer_reference_or_skip()
    csa_indexer_scorer_partial = run_tiny_csa_indexer_scorer_fixture()
    hyperhead_transformers_partial = run_tiny_hyperhead_transformers_fixture()
    integrated_layer_hc_mult_partial = run_integrated_layer_hc_mult_fixture()
    csa_attention_mlx_partial = run_tiny_csa_attention_mlx_fixture()
    report_path = mlx_work / "deepseek-v4-forward-parity-partial.json"
    report_path.write_text(json.dumps({"schema": 1, "gate": "deepseek-v4-forward-parity", "partial": partial, "additional_partials": [attention_partial, rope_partial, sink_cache_partial, hyperconnection_partial, hyperconnection_hc2_partial, hyperconnection_hc2_transformers_partial, csa_compressor_partial, csa_indexer_partial, hca_compressor_partial, hyperhead_partial, moe_partial, hash_moe_partial, moe_i8_partial, integrated_layer_partial, csa_indexer_scorer_partial, hyperhead_transformers_partial, integrated_layer_hc_mult_partial, csa_attention_mlx_partial]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 embedding/RMSNorm/head fixture failed")
    if attention_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 sliding attention fixture failed")
    if rope_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 RoPE fixture failed")
    if sink_cache_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 sink/cache/inverse-RoPE/HCA-bias fixture failed")
    if hyperconnection_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 hyperconnection fixture failed")
    if hyperconnection_hc2_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 hc_mult=2 hyperconnection fixture failed")
    if hyperconnection_hc2_transformers_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 hc_mult=2 Transformers hyperconnection fixture failed")
    if csa_compressor_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 CSA compressor forward fixture failed")
    if hca_compressor_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 HCA compressor forward fixture failed")
    if csa_indexer_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 CSA top-k indexer gather fixture failed")
    if hyperhead_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 final hyperhead collapse fixture failed")
    if moe_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 MoE fixture failed")
    if hash_moe_partial.get("status") != "ok":
        raise PlanError(f"{report_path}: tiny DeepSeek V4 hash MoE fixture failed")
    if moe_i8_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 I8 top-k MoE fixture failed")
    if integrated_layer_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 integrated layer fixture failed")
    if csa_indexer_scorer_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 CSA indexer scorer fixture failed")
    if hyperhead_transformers_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 hyperhead Transformers reference fixture failed")
    if integrated_layer_hc_mult_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 integrated layer hc_mult>1 fixture failed")
    if csa_attention_mlx_partial.get("status") not in ("ok", "skipped"):
        raise PlanError(f"{report_path}: tiny DeepSeek V4 MLX CSA compressor/indexer attention fixture failed")

    tiny_config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "qk_rope_head_dim": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "n_shared_experts": 1,
        "expert_dtype": "fp4",
        "rms_norm_eps": 1e-6,
        "hc_mult": 1,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 1,
        "layer_types": ["sliding_attention"],
        "mlp_layer_types": ["moe"],
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 10.0,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "o_groups": 1,
    }
    args_obj = ModelArgs.from_dict(tiny_config)
    try:
        Model(args_obj)
    except NotImplementedError as exc:
        blockers = "; ".join(forward_parity_blockers())
        raise PlanError(f"DeepSeek V4 forward parity is not implemented; partial evidence in {report_path}; blockers: {blockers}; construction error: {exc}") from exc
    raise PlanError("DeepSeek V4 Model constructed, but full attention/MoE forward parity has not written the marker")


def deepseek_v4_mtp_exclusion_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".deepseek-v4-mtp-exclusion-ok")
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import sanitize_weights

    index_path = mlx_work / "hf-f8shim" / "model.safetensors.index.json"
    if not index_path.is_file():
        raise PlanError(f"{index_path}: missing shimmed checkpoint index; run fp8-shim first")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{index_path}: invalid JSON: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PlanError(f"{index_path}: expected weight_map object")
    mtp_names = sorted(name for name in weight_map if str(name).startswith("mtp."))
    sample = {"embed.weight": object(), "mtp.0.test.weight": object()}
    sanitized, removed = sanitize_weights(sample)
    if "mtp.0.test.weight" not in removed or "mtp.0.test.weight" in sanitized:
        raise PlanError("DeepSeek V4 scaffold sanitizer failed to strip mtp.* sample tensor")

    def _reference_text(module_name: str, fallback_glob: str, description: str) -> str:
        try:
            ref_spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            ref_spec = None
        if ref_spec is not None and ref_spec.origin is not None:
            return pathlib.Path(ref_spec.origin).read_text(encoding="utf-8")
        candidates = sorted(pathlib.Path("/Volumes/Data NVME/mlx-ft/ds4/.venv").glob(fallback_glob))
        if candidates:
            return candidates[0].read_text(encoding="utf-8")
        raise PlanError(f"cannot locate {description} reference for MTP proof")

    transformers_text = _reference_text(
        "transformers.models.deepseek_v4.modeling_deepseek_v4",
        "lib/python*/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py",
        "Transformers DeepSeek V4 modeling",
    )
    if "_keys_to_ignore_on_load_unexpected" not in transformers_text or "mtp" not in transformers_text:
        raise PlanError("Transformers DeepSeek V4 reference does not expose mtp.* unexpected-load ignore proof")

    mlx_v32_text = _reference_text(
        "mlx_lm.models.deepseek_v32",
        "lib/python*/site-packages/mlx_lm/models/deepseek_v32.py",
        "MLX-LM DeepSeek V32",
    )
    if "Remove multi-token prediction layers" not in mlx_v32_text:
        raise PlanError("MLX-LM DeepSeek V32 reference does not contain MTP sanitization precedent")

    _write_gate_marker(
        mlx_work,
        ".deepseek-v4-mtp-exclusion-ok",
        {
            "gate": "deepseek-v4-mtp-exclusion",
            "status": "ok",
            "action": "strip",
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "mtp_tensor_count": len(mtp_names),
            "proofs": [
                "Transformers DeepseekV4ForCausalLM ignores unexpected mtp.* keys on load",
                "MLX-LM DeepSeek V32 sanitizes multi-token prediction layers before load",
                "project DeepSeek V4 scaffold sanitize_weights strips only mtp.* keys",
            ],
        },
    )
    return 0


def deepseek_v4_dequant_parity_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".deepseek-v4-dequant-parity-ok")
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.deepseek_v4_dequant import (
        dequantize_expert_packed,
        dequantize_f8_e4m3fn_with_e8m0_scales,
        f8_e4m3fn_to_float,
        f8_e8m0_scale_to_float,
        run_tiny_i8_affine_fixture,
    )

    shim_path = pathlib.Path(__file__).resolve().with_name("shim_ds4_safetensors.py")
    spec = importlib.util.spec_from_file_location("ds4_shim_reference", shim_path)
    if spec is None or spec.loader is None:
        raise PlanError(f"{shim_path}: cannot load shim reference module")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    sample_values = bytes([0x00, 0x38, 0x3C, 0x40, 0x7F, 0x80, 0xFF])
    for byte in sample_values:
        got = f8_e4m3fn_to_float(byte)
        expected = shim.f8_e4m3_to_float(byte)
        if math.isnan(expected):
            if not math.isnan(got):
                raise PlanError(f"F8_E4M3FN parity check failed for NaN byte {byte:#x}")
        elif got != expected:
            raise PlanError(f"F8_E4M3FN parity check failed for byte {byte:#x}: got {got!r}, expected {expected!r}")

    sample_scales = bytes([0, 1, 126, 127, 128, 254, 255])
    reference_scales = list(struct.unpack("<" + "f" * len(sample_scales), shim.f8_e8m0_to_f32(sample_scales)))
    got_scales = [f8_e8m0_scale_to_float(byte) for byte in sample_scales]
    for byte, got, expected in zip(sample_scales, got_scales, reference_scales, strict=True):
        if math.isnan(expected):
            if not math.isnan(got):
                raise PlanError(f"F8_E8M0 parity check failed for NaN scale byte {byte:#x}")
        elif got != expected:
            raise PlanError(f"F8_E8M0 parity check failed for byte {byte:#x}: got {got!r}, expected {expected!r}")

    scaled = dequantize_f8_e4m3fn_with_e8m0_scales(bytes([0x38, 0x40]), bytes([127, 128]), value_shape=(2,), scale_shape=(2,))
    reference_scaled = [shim.f8_e4m3_to_float(0x38) * reference_scales[3], shim.f8_e4m3_to_float(0x40) * reference_scales[4]]
    if scaled != reference_scaled:
        raise PlanError(f"F8 scale application parity check failed: {scaled!r}, expected {reference_scaled!r}")
    i8_affine = run_tiny_i8_affine_fixture()
    if i8_affine.get("status") != "ok":
        raise PlanError(f"explicit I8 affine parity check failed: {i8_affine!r}")
    for packed_kind in ("fp4", "i8"):
        try:
            dequantize_expert_packed(packed_kind, b"\x00", scales=None, shape=(1, 1))
        except NotImplementedError:
            pass
        else:
            raise PlanError(f"packed {packed_kind.upper()} expert dequant unexpectedly succeeded without parity proof")
    _write_gate_marker(
        mlx_work,
        ".deepseek-v4-dequant-parity-ok",
        {
            "gate": "deepseek-v4-dequant-parity",
            "status": "ok",
            "reference": "scripts/shim_ds4_safetensors.py",
            "checks": ["f8_e4m3fn", "f8_e8m0", "explicit_scale_application", "i8_affine_explicit", "packed_experts_fail_closed"],
        },
    )
    return 0


def mlx_lora_targets_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".mlx-lora-targets-ok")
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.lora_targets import build_lora_parameters, supported_targets, validate_lora_parameters

    lora_parameters = build_lora_parameters(rank=8, scale=20.0, dropout=0.0)
    validate_lora_parameters(lora_parameters)
    config_path = mlx_work / "lora-config.json"
    manifest_path = mlx_work / "lora-targets.json"
    mlx_work.mkdir(parents=True, exist_ok=True)
    config = {
        "lora_parameters": lora_parameters,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "policy": "ds4-supported-explicit-target-allowlist",
        "source": "ds4_ft_mlx.lora_targets",
        "targets": [target.__dict__ for target in supported_targets()],
        "forbidden": [
            "default all eligible linear modules",
            "experts",
            "compressors",
            "indexers",
            "embeddings",
            "lm_head/output/grouped outputs",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_gate_marker(
        mlx_work,
        ".mlx-lora-targets-ok",
        {
            "gate": "mlx-lora-targets",
            "status": "ok",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return 0


def deepseek_v4_mapping_check(args: argparse.Namespace) -> int:
    mlx_work = path_arg(args.mlx_work)
    _clear_gate_marker(mlx_work, ".deepseek-v4-mapping-ok")
    _ensure_mlx_project_src_on_path()
    from ds4_ft_mlx.deepseek_v4_mapping import scan_tensor_names
    from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, validate_moe_name_completeness
    index_path = mlx_work / "hf-f8shim" / "model.safetensors.index.json"
    if not index_path.is_file():
        raise PlanError(f"{index_path}: missing shimmed checkpoint index; run fp8-shim first")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlanError(f"{index_path}: invalid JSON: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PlanError(f"{index_path}: expected non-empty weight_map")
    index_sha256 = sha256_file(index_path)
    names = list(weight_map.keys())
    report = scan_tensor_names(names)
    report_dict = report.to_dict()
    moe_families_present = any(str(family).startswith("moe.") for family in report.families)
    if moe_families_present:
        config_path = mlx_work / "hf-f8shim" / "config.json"
        if not config_path.is_file():
            report_dict["moe_completeness"] = {"ok": False, "missing_required_tensors": [], "error": f"{config_path}: missing config.json for MoE completeness check"}
        else:
            try:
                config_data = json.loads(config_path.read_text(encoding="utf-8"))
                moe_config = MoEConfig(
                    hidden_size=int(config_data["hidden_size"]),
                    moe_intermediate_size=int(config_data["moe_intermediate_size"]),
                    n_routed_experts=int(config_data["n_routed_experts"]),
                    num_experts_per_tok=int(config_data["num_experts_per_tok"]),
                    n_shared_experts=int(config_data.get("n_shared_experts", 1)),
                )
                missing_moe = validate_moe_name_completeness(names, moe_config)
                report_dict["moe_completeness"] = {"ok": not missing_moe, "missing_required_tensors": missing_moe}
            except (KeyError, TypeError, ValueError) as exc:
                report_dict["moe_completeness"] = {"ok": False, "missing_required_tensors": [], "error": f"invalid MoE config: {exc}"}
    else:
        report_dict["moe_completeness"] = {"ok": True, "missing_required_tensors": []}
    report_path = mlx_work / "deepseek-v4-mapping-report.json"
    report_path.write_text(
        json.dumps(report_dict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if report.unmapped:
        raise PlanError(f"{report_path}: unmapped DeepSeek V4 tensors: {', '.join(report.unmapped[:20])}")
    if report.review_required:
        mtp_marker = _validate_mtp_exclusion_marker(mlx_work, index_path=index_path, mtp_count=len(report.review_required))
        report_dict["mtp_exclusion"] = {
            "action": mtp_marker["action"],
            "count": mtp_marker["mtp_tensor_count"],
            "marker": str(mlx_work / ".deepseek-v4-mtp-exclusion-ok"),
        }
        report_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    moe_completeness = report_dict.get("moe_completeness", {})
    if isinstance(moe_completeness, dict) and not moe_completeness.get("ok", False):
        missing = moe_completeness.get("missing_required_tensors") or []
        error = moe_completeness.get("error")
        detail = error if error else ", ".join(str(x) for x in missing[:20])
        raise PlanError(f"{report_path}: incomplete DeepSeek V4 MoE tensor families: {detail}")
    _write_gate_marker(
        mlx_work,
        ".deepseek-v4-mapping-ok",
        {
            "gate": "deepseek-v4-mapping",
            "status": "ok",
            "index_path": str(index_path),
            "index_sha256": index_sha256,
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "total": report.total,
        },
    )
    return 0


def _torch_smoke_script() -> str:
    return r'''# Smoke-test local Torch/PEFT on MPS without heavy model load.
import os, sys, subprocess, torch, json
from pathlib import Path
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

HF = os.environ.get('HF_MODEL')
DATA = os.environ.get('DATASET_ROOT')
REPO = os.environ.get('DS4_REPO', '.')
if not HF or not DATA:
    print('Set HF_MODEL and DATASET_ROOT', file=sys.stderr); sys.exit(1)
print('torch', torch.__version__)
print('mps_available', torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print('local-torch-mps requires torch.backends.mps.is_available()', file=sys.stderr); sys.exit(3)
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config
config = AutoConfig.from_pretrained(HF, trust_remote_code=True)
num_layers = getattr(config, 'num_hidden_layers', 2) or 2
print('model_type', config.model_type)
print('layers', num_layers)
adapter_path = Path(DATA) / 'torch-smoke-adapter'
adapter_path.mkdir(parents=True, exist_ok=True)
# Real PEFT attach/export on a tiny standard model so target matching and save_pretrained are exercised.
tiny_cfg = GPT2Config(vocab_size=128, n_positions=64, n_ctx=64, n_embd=32, n_layer=1, n_head=4)
tiny_model = AutoModelForCausalLM.from_config(tiny_cfg)
peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=['c_attn'], lora_dropout=0.0, bias='none', task_type='CAUSAL_LM')
peft_model = get_peft_model(tiny_model, peft_cfg)
with torch.no_grad():
    peft_model(input_ids=torch.zeros((1, 8), dtype=torch.long))
peft_export = adapter_path / 'peft-real-export'
peft_model.save_pretrained(peft_export)
print('peft_export', peft_export)
# Build DS4-convertible module names with layer indices for the DS4 adapter-name bridge.
targets = []
for i in range(min(num_layers, 4)):
    for key in ['q_a_proj','q_b_proj','kv_proj']:
        targets.append(f'model.layers.{i}.self_attn.{key}')
targets.append('lm_head')
print('canonical_targets', targets)
# Export a small DS4-shaped placeholder adapter to validate the DS4 canonical converter path.
lora_cfg = {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05, 'bias': 'none', 'task_type': 'CAUSAL_LM', 'peft_type': 'LORA', 'target_modules': targets}
print('lora_config', lora_cfg)
dummy = {}
for t in targets:
    dummy[f'base_model.model.{t}.lora_A.weight'] = torch.randn(8, 64)
    dummy[f'base_model.model.{t}.lora_B.weight'] = torch.randn(64, 8)
save_file(dummy, adapter_path / 'adapter_model.safetensors')
(adapter_path / 'adapter_config.json').write_text(json.dumps(lora_cfg))
print('saved', adapter_path)
# Verify the placeholder names convert cleanly with the DS4 bridge.
result = subprocess.run([sys.executable, str(Path(REPO)/'scripts'/'convert_lora_to_ds4.py'),
    '--dry-run', str(adapter_path/'adapter_model.safetensors'), str(adapter_path/'adapter.ds4.safetensors')])
sys.exit(result.returncode)
'''


def _torch_one_step_lora_script() -> str:
    return r'''# One cheap real Torch/PEFT train/export step plus DS4 adapter conversion smoke.
import json, os, subprocess, sys, torch
from pathlib import Path
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config

HF = os.environ.get('HF_MODEL')
DATA = os.environ.get('DATASET_ROOT')
SPLIT_DIR = os.environ.get('SPLIT_DIR', 'mlx-4096')
REPO = os.environ.get('DS4_REPO', '.')
if not HF or not DATA:
    print('Set HF_MODEL and DATASET_ROOT', file=sys.stderr); sys.exit(1)
root = Path(DATA)
train_path = root / SPLIT_DIR / 'train.jsonl'
if not train_path.is_file():
    print(f'{train_path}: missing train split', file=sys.stderr); sys.exit(2)
print('torch', torch.__version__)
print('mps_available', torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print('local-torch-mps requires torch.backends.mps.is_available()', file=sys.stderr); sys.exit(3)
real_cfg = AutoConfig.from_pretrained(HF, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(HF, trust_remote_code=True, local_files_only=True)
with train_path.open('r', encoding='utf-8') as fh:
    row = json.loads(next(line for line in fh if line.strip()))
text = str(row.get('prompt', '')) + str(row.get('completion', ''))
inputs = tokenizer(text, max_length=64, truncation=True, return_tensors='pt')
# Keep the model tiny while using the real tokenizer IDs/dataset text.
vocab_size = max(int(getattr(tokenizer, 'vocab_size', 0) or 0), int(inputs['input_ids'].max()) + 1)
tiny_cfg = GPT2Config(vocab_size=vocab_size, n_positions=64, n_ctx=64, n_embd=64, n_layer=1, n_head=4)
model = AutoModelForCausalLM.from_config(tiny_cfg)
lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=['c_attn'], lora_dropout=0.0, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(model, lora_cfg)
device = torch.device('mps')
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = inputs['input_ids'].clone()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
model.train()
loss = model(**inputs, labels=labels).loss
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
out_dir = root / 'torch-one-step-lora'
peft_dir = out_dir / 'peft-export'
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(peft_dir)
print('loss', float(loss.detach().cpu()))
print('peft_export', peft_dir)
# DS4 converter feasibility: write a tiny internal-layer adapter with DS4 canonical names.
# This is a format/runtime bridge artifact derived from the completed training step,
# not a semantically useful trained DeepSeek adapter.
scale = float(next(p.detach().float().abs().mean().cpu() for n, p in model.named_parameters() if 'lora_A' in n)) or 1.0
hidden = int(getattr(real_cfg, 'hidden_size', 4096) or 4096)
q_rank = int(getattr(real_cfg, 'q_lora_rank', 1024) or 1024)
rank = 8
adapter = {
    'base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight': (torch.randn(rank, hidden) * scale).half().cpu(),
    'base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight': (torch.randn(q_rank, rank) * scale).half().cpu(),
}
adapter_src = out_dir / 'adapter.safetensors'
adapter_ds4 = out_dir / 'adapter.ds4.safetensors'
save_file(adapter, adapter_src)
(out_dir / 'adapter_config.json').write_text(json.dumps({
    'r': rank,
    'lora_alpha': 16,
    'lora_dropout': 0.0,
    'bias': 'none',
    'task_type': 'CAUSAL_LM',
    'peft_type': 'LORA',
    'target_modules': ['model.layers.0.self_attn.q_a_proj'],
}), encoding='utf-8')
result = subprocess.run([sys.executable, str(Path(REPO)/'scripts'/'convert_lora_to_ds4.py'), str(adapter_src), str(adapter_ds4)])
if result.returncode != 0:
    sys.exit(result.returncode)
print('adapter', adapter_src)
print('adapter_ds4', adapter_ds4)
'''


def _torch_real_v4_feasibility_script() -> str:
    return r'''# Real DeepSeek V4 Flash local Torch/MPS training feasibility gate.
# This deliberately avoids full weight loading; it uses real config/model class
# construction under init_empty_weights plus safetensors header inspection.
import json, os, sys, torch
from collections import Counter
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

DTYPE_BYTES = {
    'BOOL': 1,
    'U8': 1,
    'I8': 1,
    'F8_E4M3': 1,
    'F8_E5M2': 1,
    'F8_E8M0': 1,
    'I16': 2,
    'U16': 2,
    'F16': 2,
    'BF16': 2,
    'I32': 4,
    'U32': 4,
    'F32': 4,
    'I64': 8,
    'U64': 8,
    'F64': 8,
}
HF = Path(os.environ.get('HF_MODEL', ''))
DATA = Path(os.environ.get('DATASET_ROOT', ''))
if not HF or not DATA:
    print('Set HF_MODEL and DATASET_ROOT', file=sys.stderr); sys.exit(1)
print('torch', torch.__version__)
print('mps_available', torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print('local-torch-mps requires torch.backends.mps.is_available()', file=sys.stderr); sys.exit(3)
config_path = HF / 'config.json'
index_path = HF / 'model.safetensors.index.json'
if not config_path.is_file() or not index_path.is_file():
    print(f'{HF}: missing config.json or model.safetensors.index.json', file=sys.stderr); sys.exit(2)
config_dict = json.loads(config_path.read_text(encoding='utf-8'))
model_type = config_dict.get('model_type')
config = AutoConfig.for_model(model_type, **{k: v for k, v in config_dict.items() if k != 'model_type'})
empty_model_ok = False
empty_model_error = None
target_modules_present = []
try:
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    module_names = {name for name, _ in model.named_modules()}
    target_modules_present = sorted(name for name in [
        'model.layers.0.self_attn.q_a_proj',
        'model.layers.0.self_attn.q_b_proj',
        'model.layers.0.self_attn.kv_proj',
    ] if name in module_names)
    empty_model_ok = True
except Exception as exc:
    empty_model_error = f'{type(exc).__name__}: {exc}'
index = json.loads(index_path.read_text(encoding='utf-8'))
shards = sorted(set(index.get('weight_map', {}).values()))
dtype_counts = Counter()
tensor_count = 0
stored_bytes = 0
logical_elements = 0
sample_targets = {}
for shard in shards:
    shard_path = HF / shard
    with safe_open(shard_path, framework='pt', device='cpu') as sf:
        for name in sf.keys():
            sl = sf.get_slice(name)
            shape = list(sl.get_shape())
            dtype = str(sl.get_dtype())
            numel = 1
            for dim in shape:
                numel *= int(dim)
            tensor_count += 1
            logical_elements += numel
            stored_bytes += numel * DTYPE_BYTES.get(dtype, 0)
            dtype_counts[dtype] += 1
            if name in {
                'layers.0.attn.wq_a.weight',
                'layers.0.attn.wq_b.weight',
                'layers.0.attn.wkv.weight',
                'embed.weight',
            }:
                sample_targets[name] = {'shape': shape, 'dtype': dtype}
mps_recommended = int(torch.mps.recommended_max_memory())
bf16_base_bytes = logical_elements * 2
f8_tensors = sum(count for dtype, count in dtype_counts.items() if dtype.startswith('F8_'))
reasons = []
if not empty_model_ok:
    reasons.append(f'empty real model construction failed: {empty_model_error}')
if len(target_modules_present) < 3:
    reasons.append(f'not all target LoRA modules found in real model class: {target_modules_present}')
if f8_tensors:
    f8_types = ', '.join(sorted(dtype for dtype in dtype_counts if dtype.startswith('F8_')))
    reasons.append(f'checkpoint contains {f8_types} tensors; raw Torch/MPS PEFT cannot train these quantized weights directly')
if bf16_base_bytes > int(mps_recommended * 0.85):
    reasons.append('estimated BF16 base weights exceed 85% of torch.mps.recommended_max_memory before activations/optimizer state')
local_training_feasible = not reasons
report = {
    'backend': 'local-torch-mps',
    'model_type': model_type,
    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
    'hidden_size': getattr(config, 'hidden_size', None),
    'q_lora_rank': getattr(config, 'q_lora_rank', None),
    'mps_available': True,
    'mps_recommended_max_memory_bytes': mps_recommended,
    'empty_real_model_constructed': empty_model_ok,
    'empty_real_model_error': empty_model_error,
    'target_modules_present': target_modules_present,
    'safetensors_shards': len(shards),
    'safetensors_tensors': tensor_count,
    'safetensors_stored_bytes_estimate': stored_bytes,
    'logical_elements_estimate': logical_elements,
    'bf16_base_bytes_estimate': bf16_base_bytes,
    'dtype_counts': dict(sorted(dtype_counts.items())),
    'sample_targets': sample_targets,
    'full_weight_load_attempted': False,
    'local_training_feasible': local_training_feasible,
    'heuristic_limitations': 'This gate proves hard blockers such as unsupported F8 dtypes and missing modules; for a dequantized/non-F8 checkpoint, it is not a full memory proof because activations/optimizer state, LoRA optimizer memory, dataloader buffers, and MPS allocator fragmentation are only bounded by follow-up execution gates.',
    'reasons': reasons,
    'recommendation': 'Use remote-cuda or a DS4-native training/runtime path for real DeepSeek V4 Flash LoRA' if not local_training_feasible else 'Proceed to an explicitly approved tiny full-load PEFT smoke',
}
out_dir = DATA / 'torch-real-v4-feasibility'
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / 'training_feasibility_report.json'
out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(report, indent=2, sort_keys=True))
print('training_feasibility_report', out)
'''


def _remote_cuda_smoke_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
echo "Remote CUDA smoke: verify CUDA host can import torch/peft/transformers and list GPUs."
python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('devices', torch.cuda.device_count())
from peft import LoraConfig
from transformers import AutoConfig
print('imports OK')
PY
echo "If this passes, proceed to remote-cuda-full."
'''


def check_execute_prerequisites(args: argparse.Namespace) -> None:
    step = args.step
    mlx_work = path_arg(args.mlx_work)
    dataset_root = path_arg(args.dataset_root)
    ds4_root = path_arg(args.ds4_root)
    ds4_gguf = path_arg(args.ds4_gguf) if args.ds4_gguf else ds4_root / "ds4flash.gguf"
    adapter_ds4 = path_arg(getattr(args, "adapter_ds4", None)) if getattr(args, "adapter_ds4", None) else mlx_work / "adapter.ds4.safetensors"
    if step != "setup-env" and step in {"mlx-device", "convert", "fp8-shim-probe", "convert-shimmed", "smoke-train", "smoke-train-2048", "smoke-generate", "full-train", "continue-train", "eval", "fuse", "fused-generate"}:
        if not (mlx_work / ".venv/bin/activate").is_file():
            raise PlanError(f"{mlx_work / '.venv/bin/activate'}: missing venv; run setup-env first")
    if step == "fp8-shim" and not (mlx_work / ".fp8-shim-probe-ok").is_file():
        raise PlanError(f"{mlx_work / '.fp8-shim-probe-ok'}: missing FP8 shim probe marker; run fp8-shim-probe first")
    if step == "convert-shimmed":
        validate_deepseek_v4_architecture_gates(mlx_work)
    if step == "fuse-hf":
        validate_fused_hf_safetensors_dir(mlx_work / 'fused-hf')
    if step in {"smoke-train", "smoke-train-2048", "full-train", "continue-train"}:
        validate_mlx_lora_target_gate(mlx_work)
    if step in {"convert", "convert-shimmed", "smoke-train", "smoke-train-2048", "full-train", "continue-train", "eval"}:
        validate_dataset(args)
    if step in {"smoke-train", "smoke-train-2048", "smoke-generate", "full-train", "continue-train", "eval", "fuse"} and not (mlx_work / "model-4bit").is_dir():
        raise PlanError(f"{mlx_work / 'model-4bit'}: missing converted MLX model; run convert-shimmed after fp8-shim first")
    if step in {"smoke-generate", "full-train"} and not (mlx_work / "adapters-smoke").is_dir():
        raise PlanError(f"{mlx_work / 'adapters-smoke'}: missing smoke adapter; run smoke-train first")
    if step == "full-train" and not (mlx_work / "adapters-smoke/.generation-smoke-ok").is_file():
        raise PlanError(f"{mlx_work / 'adapters-smoke/.generation-smoke-ok'}: missing smoke generation marker; run smoke-generate first")
    if step in {"convert-smoke-adapter", "ds4-smoke-adapter-inspect"} and not (mlx_work / "adapters-smoke/adapters.safetensors").is_file():
        raise PlanError(f"{mlx_work / 'adapters-smoke/adapters.safetensors'}: missing smoke adapter; run smoke-train first")
    if step == "ds4-smoke-adapter-inspect" and not (mlx_work / "adapters-smoke/adapter.ds4.safetensors").is_file():
        raise PlanError(f"{mlx_work / 'adapters-smoke/adapter.ds4.safetensors'}: missing converted DS4 adapter; run convert-smoke-adapter first")
    if step == "ds4-smoke-adapter-inspect" and not ds4_gguf.is_file():
        raise PlanError(f"{ds4_gguf}: missing ds4flash.gguf; set --ds4-gguf or DS4_GGUF")
    if step == "full-train" and not (mlx_work / "adapters-smoke/.ds4-inspect-ok").is_file():
        raise PlanError(f"{mlx_work / 'adapters-smoke/.ds4-inspect-ok'}: missing DS4 inspect marker; run convert-smoke-adapter and ds4-smoke-adapter-inspect (requires --ds4-gguf / DS4_GGUF) before full-train")
    if step in {"continue-train", "eval", "fuse"} and not (mlx_work / "adapters/adapters.safetensors").is_file():
        raise PlanError(f"{mlx_work / 'adapters/adapters.safetensors'}: missing trained adapter")
    if step == "fused-generate" and not (mlx_work / "fused-model").is_dir():
        raise PlanError(f"{mlx_work / 'fused-model'}: missing fused MLX model")
    if step in {"quantize-q2", "quantize-q4"}:
        fused = path_arg(args.fused_hf_model) if args.fused_hf_model else None
        if fused is None or str(fused) == "/path/to/fused-hf-safetensors-model":
            raise PlanError("quantization requires --fused-hf-model pointing to a verified HF-style safetensors directory")
        validate_hf_safetensors_dir(fused)
        imatrix = path_arg(args.ds4_imatrix) if args.ds4_imatrix else None
        if imatrix is None or str(imatrix).startswith("/path/to/") or not imatrix.is_file():
            raise PlanError("quantization requires --ds4-imatrix pointing to an existing imatrix file")
    if step in {"splice-dry-run", "splice"}:
        for path in (ds4_root / "gguf/anthropomorphic-frankenmerge-q2.gguf", ds4_root / "gguf/anthropomorphic-frankenmerge-q4.gguf"):
            if not path.is_file():
                raise PlanError(f"{path}: missing GGUF required for splicing")
    if step == "ds4-smoke" and not (ds4_root / "gguf/anthropomorphic-frankenmerge-ds4flash.gguf").is_file():
        raise PlanError("final mixed GGUF is missing")
    if step == "build-quantizer" and not (ds4_root / "gguf-tools/Makefile").is_file():
        raise PlanError(f"{ds4_root / 'gguf-tools/Makefile'}: missing gguf-tools Makefile")
    backend = getattr(args, "backend", "local-mlx")
    if step in {"torch-env-create", "torch-env-check", "torch-smoke", "torch-one-step-lora", "torch-real-v4-feasibility"}:
        if backend != "local-torch-mps":
            raise PlanError(f"{step} requires backend local-torch-mps, got {backend}")
        if step != "torch-env-create" and not (mlx_work / ".venv-torch/bin/activate").is_file():
            raise PlanError(f"{mlx_work / '.venv-torch/bin/activate'}: missing Torch/PEFT venv; run torch-env-create first")
        return
    if step in {"remote-cuda-smoke", "remote-cuda-full"}:
        if backend != "remote-cuda":
            raise PlanError(f"{step} requires backend remote-cuda, got {backend}")
        return
    if step in {"adapter-list-mappings", "ds4-adapter-inspect"}:
        if backend != "cpu-check":
            raise PlanError(f"{step} requires backend cpu-check, got {backend}")
        if step == "ds4-adapter-inspect":
            if not adapter_ds4.is_file():
                raise PlanError(f"{adapter_ds4}: missing DS4 adapter; run convert_lora_to_ds4.py first or pass --adapter-ds4")
            if not ds4_gguf.is_file():
                raise PlanError(f"{ds4_gguf}: missing ds4flash.gguf; set --ds4-gguf or DS4_GGUF")
            if not (ds4_root / "ds4").is_file():
                raise PlanError(f"{ds4_root / 'ds4'}: missing DS4 binary; build ds4 first")
        return
    if step == "list-backend-descriptions":
        if backend != "manual":
            raise PlanError(f"{step} requires backend manual, got {backend}")
        return
    # Default mlx steps still require mlx venv, dataset, and converted model as before.
    if step in {
        "mlx-device",
        "convert",
        "fp8-shim-probe",
        "deepseek-v4-import-check",
        "deepseek-v4-tiny-config-check",
        "deepseek-v4-mtp-exclusion-check",
        "deepseek-v4-mapping-check",
        "deepseek-v4-dequant-parity-check",
        "deepseek-v4-forward-parity-check",
        "deepseek-v4-forward-parity-readiness",
        "convert-shimmed",
        "mlx-lora-targets-check",
        "smoke-train",
        "smoke-train-2048",
        "smoke-generate",
        "convert-smoke-adapter",
        "ds4-smoke-adapter-inspect",
        "full-train",
        "continue-train",
        "eval",
        "fuse",
        "fuse-hf",
        "fused-generate",
    }:
        if backend != "local-mlx":
            raise PlanError(f"{step} requires backend local-mlx, got {backend}")


def run_command(args: argparse.Namespace) -> int:
    backend = args.backend
    if backend not in BACKEND_STEPS:
        raise PlanError(f"unknown backend {backend!r}")
    allowed_steps = BACKEND_STEPS[backend]
    catalog = command_catalog(args)
    if args.step not in catalog:
        raise PlanError(f"unknown command step {args.step!r}")
    if args.step not in allowed_steps:
        raise PlanError(f"step {args.step!r} is not available for backend {backend!r}; use one of {', '.join(allowed_steps)}")
    commands = catalog[args.step]
    if args.step == "model-4bit-conversion-plan" and args.execute:
        raise PlanError("model-4bit-conversion-plan is a top-level read-only Python subcommand; invoke it directly instead of run-command --execute")
    if args.step == "deepseek-v4-forward-parity-readiness" and args.execute:
        raise PlanError("deepseek-v4-forward-parity-readiness is a top-level read-only Python subcommand; invoke it directly instead of run-command --execute")
    if not args.execute:
        for command in commands:
            print(command)
        print("\nDry run only. Add --execute --yes after reviewing stop rules and required artifacts.", file=sys.stderr)
        return 0
    if not args.yes:
        raise PlanError("execute mode requires --yes")
    check_execute_prerequisites(args)
    lock_path = path_arg(args.mlx_work) / ".ds4-ft.lock"
    with LockFile(lock_path):
        for command in commands:
            print(f"+ {command}", flush=True)
            subprocess.run(command, shell=True, check=True)
    return 0


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-model", default=env_or_default("HF_MODEL"), help="local Hugging Face model snapshot")
    parser.add_argument("--opus46-data", default=env_or_default("OPUS46_DATA"), help="Opus 4.6 JSONL source")
    parser.add_argument("--opus47-data", default=env_or_default("OPUS47_DATA"), help="Opus 4.7 JSONL source")
    parser.add_argument("--fable5-data", default=env_or_default("FABLE5_DATA"), help="Fable 5 JSONL source")
    parser.add_argument("--dataset-root", default=env_or_default("DATASET_ROOT"), help="output dataset root outside git")
    parser.add_argument("--mlx-work", default=env_or_default("MLX_WORK"), help="MLX work directory outside git")
    parser.add_argument("--ds4-root", default=env_or_default("DS4_ROOT"), help="DS4 repo containing gguf-tools and DS4 binaries")
    parser.add_argument("--ds4-gguf", default=env_or_default("DS4_GGUF"), help="path to ds4flash.gguf for inspect/validation (default: DS4_ROOT/ds4flash.gguf)")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="dataset split subdirectory under dataset-root")


def build_parser() -> argparse.ArgumentParser:
    epilog = """Examples:
  %(prog)s preflight
  %(prog)s build-dataset && %(prog)s validate-dataset
  %(prog)s token-audit --report-only
  %(prog)s emit-commands fp8-shim-probe fp8-shim convert-shimmed model-4bit-conversion-plan deepseek-v4-forward-parity-readiness smoke-train full-train
  %(prog)s model-4bit-conversion-plan
  %(prog)s deepseek-v4-forward-parity-readiness
  %(prog)s run-command fp8-shim-probe --execute --yes
  %(prog)s run-command fp8-shim --execute --yes
  %(prog)s run-command convert-shimmed          # dry-run only
  %(prog)s run-command convert-shimmed --execute --yes

Conversion, training, GGUF quantization, splicing writes, and DS4 server runs are never executed by default.
"""
    parser = argparse.ArgumentParser(description=__doc__, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="check required HF and source dataset artifacts")
    add_common_paths(p)
    p.set_defaults(func=preflight)

    p = sub.add_parser("build-dataset", help="build deterministic MLX prompt/completion splits")
    add_common_paths(p)
    p.set_defaults(func=build_dataset)

    p = sub.add_parser("validate-dataset", help="strict structural validation plus manifest/meta consistency checks")
    add_common_paths(p)
    p.set_defaults(func=validate_dataset)

    p = sub.add_parser("token-audit", help="optional tokenizer length audit; default fails when rows exceed the limit")
    add_common_paths(p)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--report-only", action="store_true", help="report over-limit rows but exit 0")
    p.add_argument("--allow-over-limit", action="store_true", help="acknowledge over-limit rows and exit 0")
    p.add_argument("--no-trust-remote-code", action="store_true")
    p.set_defaults(func=token_audit)

    p = sub.add_parser("deepseek-v4-import-check", help="verify project-controlled mlx_lm.models.deepseek_v4 import and write marker")
    add_common_paths(p)
    p.set_defaults(func=deepseek_v4_import_check)

    p = sub.add_parser("deepseek-v4-tiny-config-check", help="verify tiny DeepSeek V4 config parsing and fail-closed construction marker")
    add_common_paths(p)
    p.set_defaults(func=deepseek_v4_tiny_config_check)

    p = sub.add_parser("deepseek-v4-mtp-exclusion-check", help="prove and mark safe stripping of mtp.* tensors for MLX conversion")
    add_common_paths(p)
    p.set_defaults(func=deepseek_v4_mtp_exclusion_check)

    p = sub.add_parser("deepseek-v4-mapping-check", help="scan shimmed checkpoint index tensor names and fail closed on unmapped names")
    add_common_paths(p)
    p.set_defaults(func=deepseek_v4_mapping_check)

    p = sub.add_parser("deepseek-v4-dequant-parity-check", help="run tiny F8/dequant parity checks and write marker")
    add_common_paths(p)
    p.set_defaults(func=deepseek_v4_dequant_parity_check)

    p = sub.add_parser("deepseek-v4-forward-parity-check", help="run tiny DeepSeek V4 forward parity gate; fail closed until implemented")
    add_common_paths(p)
    p.set_defaults(func=deepseek_v4_forward_parity_check)

    p = sub.add_parser("model-4bit-conversion-plan", help="read-only: validate shimmed input + non-forward gates and write model-4bit conversion plan (no conversion)")
    add_common_paths(p)
    p.add_argument("--out", default=None, help="plan JSON output path (default: MLX_WORK/model-4bit-conversion-plan.json)")
    p.add_argument("--report-only", action="store_true", help="write the plan even when MLX_WORK/model-4bit already exists, recording destination_exists=true")
    p.add_argument("--fused-hf-model")
    p.add_argument("--ds4-imatrix")
    p.add_argument("--adapter-ds4", help="DS4-converted adapter.safetensors for command provenance")
    p.set_defaults(func=model_4bit_conversion_plan)

    p = sub.add_parser("deepseek-v4-forward-parity-readiness", help="read-only: write fail-closed DeepSeek V4 forward-parity readiness/gap report")
    add_common_paths(p)
    p.add_argument("--out", default=None, help="readiness JSON output path (default: MLX_WORK/deepseek-v4-forward-parity-readiness.json)")
    p.set_defaults(func=deepseek_v4_forward_parity_readiness)

    p = sub.add_parser("ds4-gguf-base-smoke", help="opt-in Track-A ds4/Metal base-GGUF smoke-generate gate")
    add_common_paths(p)
    p.add_argument("--ds4-smoke-prompt", default=DS4_GGUF_BASE_SMOKE_PROMPT, help="short deterministic prompt for the Track-A base smoke gate")
    p.add_argument("--ds4-smoke-tokens", type=int, default=DS4_GGUF_BASE_SMOKE_TOKENS, help="bounded max generated tokens for the Track-A base smoke gate")
    p.add_argument("--ds4-smoke-timeout", type=int, default=DS4_GGUF_BASE_SMOKE_TIMEOUT, help="bounded subprocess timeout in seconds for the Track-A base smoke gate")
    p.set_defaults(func=ds4_gguf_base_smoke_check)

    p = sub.add_parser("mlx-lora-targets-check", help="write DS4-safe MLX LoRA config and structured allowlist marker")
    add_common_paths(p)
    p.set_defaults(func=mlx_lora_targets_check)

    p = sub.add_parser("emit-commands", help="print expensive MLX/GGUF commands without running them")
    add_common_paths(p)
    p.add_argument("--format", choices=("shell", "json"), default="shell")
    p.add_argument(
        "--backend",
        choices=("local-mlx", "local-torch-mps", "remote-cuda", "cpu-check", "manual"),
        default="local-mlx",
        help="execution backend selector; local-mlx is the default on Mac Studio M3 Ultra",
    )
    p.add_argument("steps", nargs="*", choices=COMMAND_STEPS, metavar="step")
    p.add_argument("--fused-hf-model")
    p.add_argument("--ds4-imatrix")
    p.add_argument("--adapter-ds4", help="DS4-converted adapter.safetensors for cpu-check ds4-adapter-inspect")
    p.set_defaults(func=emit_commands)

    p = sub.add_parser("run-command", help="dry-run or execute one expensive MLX/GGUF step; execute requires --yes")
    add_common_paths(p)
    p.add_argument(
        "--backend",
        choices=("local-mlx", "local-torch-mps", "remote-cuda", "cpu-check", "manual"),
        default="local-mlx",
        help="execution backend selector; local-mlx is the default on Mac Studio M3 Ultra",
    )
    p.add_argument("step", choices=COMMAND_STEPS)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes", action="store_true", help="required with --execute")
    p.add_argument("--fused-hf-model")
    p.add_argument("--ds4-imatrix")
    p.add_argument("--adapter-ds4", help="DS4-converted adapter.safetensors for cpu-check ds4-adapter-inspect")
    p.set_defaults(func=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
