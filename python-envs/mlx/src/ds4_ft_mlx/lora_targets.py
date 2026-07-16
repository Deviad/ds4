"""DeepSeek V4 Flash MLX LoRA target allowlist for DS4 runtime adapters.

This helper intentionally models the current DS4 runtime-supported internal LoRA
surface only: q_a, q_b, and kv.  It does not start MLX training or inspect model
weights.  The MLX-LM target mechanism was checked by source inspection:
`mlx_lm.lora.train_model()` passes `args.lora_parameters` to
`mlx_lm.tuner.utils.linear_to_lora_layers()`, whose implementation reads
`config.get('keys', None)`.  If `keys` is absent, MLX-LM discovers every
eligible Linear/SwitchLinear/Embedding module and applies LoRA broadly.  The
installed CLI parser has no CLI target-module argument, so DS4-safe targeting
must be represented in config/YAML or equivalent code through
`lora_parameters.keys`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

SUPPORTED_ALIASES = ("q_a", "q_b", "kv")

MLX_LM_LORA_SOURCE_CONTRACT = (
    "Inspection of mlx_lm.lora and mlx_lm.tuner.utils: train_model calls "
    "linear_to_lora_layers(model, ..., args.lora_parameters); "
    "linear_to_lora_layers uses config.get('keys', None); absent keys trigger "
    "automatic discovery of all eligible Linear/SwitchLinear/Embedding modules; "
    "the mlx_lm.lora CLI parser exposes no CLI target-module argument."
)


@dataclass(frozen=True)
class LoraTarget:
    alias: str
    ds4_target: str
    mlx_key: str
    hf_suffix: str
    notes: str


_TARGETS = (
    LoraTarget(
        alias="q_a",
        ds4_target="attn_q_a",
        mlx_key="self_attn.q_a_proj",
        hf_suffix="self_attn.q_a_proj",
        notes="DS4 runtime attention q_a low-rank projection target.",
    ),
    LoraTarget(
        alias="q_b",
        ds4_target="attn_q_b",
        mlx_key="self_attn.q_b_proj",
        hf_suffix="self_attn.q_b_proj",
        notes="DS4 runtime attention q_b projection target.",
    ),
    LoraTarget(
        alias="kv",
        ds4_target="attn_kv",
        mlx_key="self_attn.kv_proj",
        hf_suffix="self_attn.kv_proj",
        notes="DS4 runtime combined key/value projection target.",
    ),
)

_BY_ALIAS = {target.alias: target for target in _TARGETS}
_ALLOWED_KEYS = {target.mlx_key for target in _TARGETS}
_FORBIDDEN_ALIASES = {
    "all",
    "all-linear",
    "all_linear",
    "default",
    "linear",
    "all_linears",
    "output",
    "lm_head",
    "head",
    "experts",
    "expert",
    "moe",
    "compressor",
    "indexer",
    "grouped_output",
    "grouped-output",
}


def supported_targets() -> tuple[LoraTarget, ...]:
    return _TARGETS


def supported_ds4_targets() -> tuple[str, ...]:
    return tuple(target.ds4_target for target in _TARGETS)


def supported_mlx_keys() -> tuple[str, ...]:
    return tuple(target.mlx_key for target in _TARGETS)


def _unsupported(value: str) -> ValueError:
    return ValueError(
        f"unsupported DS4 MLX LoRA target {value!r}; current DS4 runtime allowlist is "
        f"{', '.join(SUPPORTED_ALIASES)} only"
    )


def resolve_target_aliases(targets: Iterable[str] | None = None) -> list[str]:
    """Resolve explicit user-facing aliases to MLX-LM `lora_parameters.keys`.

    `None` means the full DS4-supported allowlist.  An empty iterable is an
    error because MLX-LM's absent/empty target behavior is easy to confuse with
    unsafe defaults.
    """

    aliases = list(SUPPORTED_ALIASES if targets is None else targets)
    if not aliases:
        raise _unsupported("<empty>")
    keys: list[str] = []
    seen: set[str] = set()
    for raw in aliases:
        alias = str(raw).strip()
        if alias in _FORBIDDEN_ALIASES or alias not in _BY_ALIAS:
            raise _unsupported(alias)
        key = _BY_ALIAS[alias].mlx_key
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def build_lora_parameters(
    *,
    targets: Iterable[str] | None = None,
    rank: int = 8,
    scale: float = 20.0,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Build the DS4-safe MLX-LM `lora_parameters` dict.

    The returned dict is suitable for MLX-LM YAML/config use.  It deliberately
    includes `keys`; omitting that field would allow MLX-LM's all-eligible-module
    discovery path.
    """

    if rank <= 0:
        raise ValueError("LoRA rank must be > 0")
    if scale <= 0:
        raise ValueError("LoRA scale must be > 0")
    if dropout < 0:
        raise ValueError("LoRA dropout must be >= 0")
    return {
        "rank": int(rank),
        "scale": float(scale),
        "dropout": float(dropout),
        "keys": resolve_target_aliases(targets),
    }


def validate_lora_parameters(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate that MLX-LM LoRA config cannot attach outside DS4 support."""

    keys = config.get("keys")
    if not keys:
        raise ValueError(
            "DS4 MLX LoRA config is missing lora_parameters.keys and would default to all eligible linear modules"
        )
    if isinstance(keys, str):
        raise ValueError("lora_parameters.keys must be a list of explicit MLX module keys, not a string")
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in keys:
        key = str(raw).strip()
        if key not in _ALLOWED_KEYS:
            raise _unsupported(key)
        if key not in seen:
            resolved.append(key)
            seen.add(key)
    return tuple(resolved)
