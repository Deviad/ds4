"""Tensor-family mapping scaffold for DeepSeek V4 Flash MLX port.

The scanner is intentionally conservative: every checkpoint tensor must map to a
known family or remain a hard blocker before a full `convert-shimmed` attempt is
accepted.  Families are semantic buckets, not final MLX parameter names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_LAYER_ATTN = re.compile(r"^layers\.\d+\.attn\.(?P<name>[^.]+)\.(?P<kind>weight|scale)$")
_EXPERT = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.(?P<name>w[123])\.(?P<kind>weight|scale)$")
_SHARED = re.compile(r"^layers\.\d+\.ffn\.shared_experts\.(?P<name>w[123])\.(?P<kind>weight|scale)$")
_ROUTER = re.compile(r"^layers\.\d+\.ffn\.(?:gate|router|.*router.*)\.(?:weight|scale|bias|tid2eid)$")
_NORM = re.compile(r"^layers\.\d+\..*(?:norm|layernorm).*\.(?:weight|bias)$")
_HYPER = re.compile(r"^(?:hc_head_(?:base|fn|scale)|(?:mtp\.\d+\.)?hc_head_(?:base|fn|scale)|layers\.\d+\.(?:hc_|.*hyper|.*hc).*)$")
_INDEXER = re.compile(r"^layers\.\d+\.(?:indexer|.*index.*|.*score.*).*(?:\.(?:weight|scale|bias)|\.ape|attn_sink)$")
_COMPRESSOR = re.compile(r"^layers\.\d+\.(?:.*compressor.*|.*hca.*|.*csa.*).*(?:\.(?:weight|scale|bias)|\.ape)$")

_ATTN_FAMILIES = {
    "wq_a": "attention.q_a",
    "wq_b": "attention.q_b",
    "wkv": "attention.kv",
    "wo_a": "attention.output_a",
    "wo_b": "attention.output_b",
}

MTP_POLICY_FAMILY = "mtp.intentional-review-required"
_MTP_POLICY = {
    "family": MTP_POLICY_FAMILY,
    "status": "review-required",
    "action": "fail-closed",
    "supported": False,
    "ignored": False,
    "stripped": False,
    "instruction": (
        "Do not make mapping pass for mtp.* tensors unless support, safe ignore, "
        "or safe stripping is proven and documented."
    ),
    "rationale": (
        "DeepSeek V4 Flash exposes mtp.* tensors for multi-token prediction. "
        "No parity proof currently shows that MLX conversion/runtime can consume "
        "them, nor that they can be silently ignored or stripped without changing "
        "model semantics."
    ),
}


def mtp_policy() -> dict[str, object]:
    """Return the explicit fail-closed policy for DeepSeek V4 MTP tensors."""

    return dict(_MTP_POLICY)


@dataclass(frozen=True)
class TensorMappingReport:
    total: int
    families: dict[str, int] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    review_required: list[str] = field(default_factory=list)
    review_required_policy: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.unmapped and not self.review_required

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "total": self.total,
            "families": self.families,
            "unmapped": self.unmapped,
            "review_required": self.review_required,
            "review_required_policy": self.review_required_policy,
        }


def _with_kind(base: str, kind: str) -> str:
    return f"{base}_scale" if kind == "scale" else base


def classify_tensor_name(name: str) -> str | None:
    if name in {"embed.weight", "model.embed_tokens.weight"}:
        return "embedding"
    if name in {"output.weight", "lm_head.weight", "head.weight"}:
        return "output"
    if name in {"norm.weight", "model.norm.weight"}:
        return "norm"
    if re.match(r"^layers\.\d+\.attn\.attn_sink$", name):
        return "attention.sink"
    if name.startswith("mtp."):
        return MTP_POLICY_FAMILY

    match = _LAYER_ATTN.match(name)
    if match:
        family = _ATTN_FAMILIES.get(match.group("name"))
        if family:
            return _with_kind(family, match.group("kind"))

    match = _EXPERT.match(name)
    if match:
        return _with_kind(f"moe.expert.{match.group('name')}", match.group("kind"))

    match = _SHARED.match(name)
    if match:
        return _with_kind(f"moe.shared.{match.group('name')}", match.group("kind"))

    if _ROUTER.match(name):
        return "moe.router"
    if _COMPRESSOR.match(name):
        return "compressor"
    if _INDEXER.match(name):
        return "indexer"
    if _HYPER.match(name):
        return "hyperconnection"
    if _NORM.match(name):
        return "norm"
    return None


def scan_tensor_names(names: Iterable[str]) -> TensorMappingReport:
    families: dict[str, int] = {}
    unmapped: list[str] = []
    review_required: list[str] = []
    review_required_policy: dict[str, dict[str, object]] = {}
    total = 0
    for name in names:
        total += 1
        family = classify_tensor_name(name)
        if family is None:
            unmapped.append(name)
        else:
            families[family] = families.get(family, 0) + 1
            if family == MTP_POLICY_FAMILY:
                review_required.append(name)
                review_required_policy[MTP_POLICY_FAMILY] = mtp_policy()
    return TensorMappingReport(
        total=total,
        families=dict(sorted(families.items())),
        unmapped=unmapped,
        review_required=review_required,
        review_required_policy=review_required_policy,
    )
