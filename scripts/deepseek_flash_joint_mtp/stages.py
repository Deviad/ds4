"""Immutable stage registry and deterministic ordinary-stage selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .canonical import digest_payload


@dataclass(frozen=True)
class StageSpec:
    name: str
    depends_on: tuple[str, ...]
    kind: str
    contract_version: int
    output_slots: tuple[str, ...]
    boundary_resume_only: bool


ORDINARY_STAGE_NAMES = (
    "preflight",
    "prepare-joint-base",
    "train-joint",
    "fuse-joint",
    "export-fused-imatrix-source",
    "collect-imatrix",
    "repair-imatrix",
    "quantize-and-embed",
    "inventory-verify",
    "no-sidecar-load-verify",
    "correctness-verify",
    "performance-verify",
    "release-candidate",
)


def _ordinary(name: str, previous: str | None) -> StageSpec:
    return StageSpec(
        name=name,
        depends_on=() if previous is None else (previous,),
        kind="ordinary",
        contract_version=1,
        output_slots=("artifact",),
        boundary_resume_only=False,
    )


ORDINARY_SPECS = tuple(
    _ordinary(name, ORDINARY_STAGE_NAMES[index - 1] if index else None)
    for index, name in enumerate(ORDINARY_STAGE_NAMES)
)
STAGE_SPECS = ORDINARY_SPECS + (
    StageSpec("promote", ("release-candidate",), "human-action", 1, (), True),
    StageSpec("cleanup", ("release-candidate",), "human-action", 1, (), True),
)
STAGE_BY_NAME = {spec.name: spec for spec in STAGE_SPECS}
REGISTRY_DIGEST = digest_payload([asdict(spec) for spec in STAGE_SPECS])


def registry_payload() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in STAGE_SPECS]


def ordinary_specs(*, from_stage: str | None = None, through_stage: str | None = None) -> tuple[StageSpec, ...]:
    if from_stage in {"promote", "cleanup"} or through_stage in {"promote", "cleanup"}:
        raise ValueError("human-action stages are separate commands")
    names = list(ORDINARY_STAGE_NAMES)
    if from_stage is not None and from_stage not in names:
        raise ValueError(f"unknown ordinary stage: {from_stage}")
    if through_stage is not None and through_stage not in names:
        raise ValueError(f"unknown ordinary stage: {through_stage}")
    start = names.index(from_stage) if from_stage else 0
    end = names.index(through_stage) if through_stage else len(names) - 1
    if start > end:
        raise ValueError("--from-stage must not follow --through-stage")
    return tuple(ORDINARY_SPECS[start:end + 1])


def transitive_dependencies(stage_name: str) -> tuple[str, ...]:
    if stage_name not in STAGE_BY_NAME:
        raise ValueError(f"unknown stage: {stage_name}")
    result: list[str] = []
    current = STAGE_BY_NAME[stage_name]
    while current.depends_on:
        dependency = current.depends_on[0]
        result.append(dependency)
        current = STAGE_BY_NAME[dependency]
    result.reverse()
    return tuple(result)
