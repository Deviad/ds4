"""Run layout, immutable records, exclusive locking, and manifest validation."""

from __future__ import annotations

import fcntl
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .canonical import atomic_write_json, digest_payload, identity_dict, load_json
from .config import ConfigError, ResolvedConfig, validate_run_id
from .stages import ORDINARY_STAGE_NAMES


class StateError(RuntimeError):
    """Raised when run state cannot be trusted or published safely."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class RunState:
    config: ResolvedConfig
    run_id: str
    root: Path
    output_root: Path
    registry_digest: str
    pipeline_identity: list[dict[str, Any]]

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifest"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def active_path(self) -> Path:
        return self.config.state_root / "active" / f"{self.config.resolved_config_digest}.json"

    def stage_dir(self, stage: str) -> Path:
        return self.stages_dir / stage

    def output_dir(self, stage: str) -> Path:
        return self.output_root / self.run_id / stage

    def output_path(self, stage: str, slot: str = "artifact") -> Path:
        return self.output_dir(stage) / slot

    def append_manifest(self, event_type: str, event_data: dict[str, Any], *, stage: str | None = None, referenced_record_digest: str | None = None) -> str:
        head_path = self.manifest_dir / "head.json"
        head = load_json(head_path)
        sequence = int(head["sequence"]) + 1
        payload = {
            "schema_version": 1,
            "sequence": sequence,
            "previous_entry_digest": head["digest"],
            "event_type": event_type,
            "run_id": self.run_id,
            "stage": stage,
            "referenced_record_digest": referenced_record_digest,
            "event_data": event_data,
            "created_at": utc_now(),
        }
        wrapper = {"payload": payload, "digest": digest_payload(payload)}
        entry_path = self.manifest_dir / f"{sequence:06d}-{event_type}.json"
        if entry_path.exists():
            raise StateError(f"manifest entry already exists: {entry_path}")
        atomic_write_json(entry_path, wrapper)
        atomic_write_json(head_path, {"sequence": sequence, "digest": wrapper["digest"]})
        return wrapper["digest"]

    def write_failure(self, stage: str, attempt_id: str, payload: dict[str, Any]) -> None:
        path = self.stage_dir(stage) / "attempts" / attempt_id / "failure.json"
        atomic_write_json(path, payload)
        self.append_manifest("failure", {"attempt_id": attempt_id, "failure_digest": digest_payload(payload)}, stage=stage)

    def completion_path(self, stage: str) -> Path:
        return self.stage_dir(stage) / "complete.json"

    def read_completion(self, stage: str) -> dict[str, Any] | None:
        path = self.completion_path(stage)
        if not path.exists():
            return None
        try:
            wrapper = load_json(path)
            if not isinstance(wrapper, dict) or set(wrapper) != {"payload", "digest"}:
                raise StateError(f"invalid completion wrapper: {path}")
            if wrapper["digest"] != digest_payload(wrapper["payload"]):
                raise StateError(f"completion digest mismatch: {path}")
            return wrapper
        except (OSError, ValueError, TypeError) as exc:
            raise StateError(str(exc)) from exc

    def write_completion(self, stage: str, payload: dict[str, Any]) -> str:
        path = self.completion_path(stage)
        if path.exists():
            raise StateError(f"completion already exists: {path}")
        wrapper = {"payload": payload, "digest": digest_payload(payload)}
        atomic_write_json(path, wrapper, mode=0o444)
        self.append_manifest("completion", {"completion_digest": wrapper["digest"]}, stage=stage, referenced_record_digest=wrapper["digest"])
        return wrapper["digest"]

    def completion_valid(self, stage: str, dependency_digests: dict[str, str], *, current_tool_identity: dict[str, Any] | None = None) -> bool:
        wrapper = self.read_completion(stage)
        if wrapper is None:
            return False
        payload = wrapper["payload"]
        if payload.get("resolved_config_digest") != self.config.resolved_config_digest:
            raise StateError(f"changed config for completed stage: {stage}")
        if payload.get("registry_digest") != self.registry_digest:
            raise StateError(f"changed registry for completed stage: {stage}")
        expected_pipeline_digest = digest_payload(self.pipeline_identity)
        if payload.get("pipeline_identity_digest") != expected_pipeline_digest:
            raise StateError(f"changed pipeline identity for completed stage: {stage}")
        if payload.get("dependency_completion_digests") != dependency_digests:
            raise StateError(f"stale dependency for completed stage: {stage}")
        if current_tool_identity is not None and payload.get("tool_identity") != current_tool_identity:
            raise StateError(f"changed tool identity for completed stage: {stage}")
        output_root = (self.output_root / self.run_id).resolve()
        for output in payload.get("output_identities", {}).values():
            path = Path(output["canonical_path"])
            if path.is_symlink() or not (path == output_root or output_root in path.parents):
                raise StateError(f"output escaped run root for stage: {stage}")
            if not path.exists() or identity_dict(path) != output:
                raise StateError(f"changed output for completed stage: {stage}")
        return payload.get("next_stage_eligible") is True

    def next_attempt_id(self, stage: str) -> str:
        attempts = self.stage_dir(stage) / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        numbers = []
        for child in attempts.iterdir():
            try:
                numbers.append(int(child.name))
            except ValueError:
                raise StateError(f"unknown attempt directory: {child}")
        return f"{(max(numbers) if numbers else 0) + 1:04d}"

    def verify_manifest(self) -> None:
        if not self.manifest_dir.is_dir():
            raise StateError("manifest directory missing")
        children = sorted(self.manifest_dir.iterdir())
        head_path = self.manifest_dir / "head.json"
        if head_path.is_symlink() or not head_path.is_file() or any(child.name == "head.json" and child != head_path for child in children):
            raise StateError("manifest head is invalid")
        entry_pattern = re.compile(r"^[0-9]{6}-[a-z0-9-]+\.json$")
        entry_paths = []
        for child in children:
            if child == head_path:
                continue
            if child.is_symlink() or not child.is_file() or not entry_pattern.fullmatch(child.name):
                raise StateError(f"unknown manifest entry: {child}")
            entry_paths.append(child)
        head = load_json(head_path)
        previous = None
        completion_refs: set[str] = set()
        for expected, path in enumerate(entry_paths):
            wrapper = load_json(path)
            if not isinstance(wrapper, dict) or set(wrapper) != {"payload", "digest"}:
                raise StateError(f"invalid manifest wrapper: {path}")
            payload = wrapper["payload"]
            if payload["sequence"] != expected:
                raise StateError(f"manifest sequence gap at {path}")
            if payload["previous_entry_digest"] != previous:
                raise StateError(f"manifest link mismatch at {path}")
            if wrapper["digest"] != digest_payload(payload):
                raise StateError(f"manifest digest mismatch at {path}")
            if payload.get("event_type") == "completion":
                reference = payload.get("referenced_record_digest")
                if not isinstance(reference, str):
                    raise StateError(f"completion event lacks record reference: {path}")
                completion_refs.add(reference)
            previous = wrapper["digest"]
        if not entry_paths or head != {"sequence": len(entry_paths) - 1, "digest": previous}:
            raise StateError("manifest head mismatch")
        self._verify_layout(completion_refs)

    def _verify_layout(self, completion_refs: set[str]) -> None:
        allowed_root = {"resolved-config.json", "lock", "manifest", "stages"}
        for child in self.root.iterdir():
            if child.is_symlink() or child.name not in allowed_root:
                raise StateError(f"unknown run-state entry: {child}")
        if not self.stages_dir.is_dir():
            raise StateError("stages directory missing")
        for stage_dir in self.stages_dir.iterdir():
            if stage_dir.is_symlink() or not stage_dir.is_dir() or stage_dir.name not in set(ORDINARY_STAGE_NAMES):
                raise StateError(f"unknown stage-state entry: {stage_dir}")
            for child in stage_dir.iterdir():
                if child.name not in {"attempts", "complete.json", "checkpoint.json"}:
                    raise StateError(f"unknown stage-state entry: {child}")
            complete = stage_dir / "complete.json"
            if complete.exists():
                wrapper = load_json(complete)
                if wrapper.get("digest") not in completion_refs:
                    raise StateError(f"unreferenced completion record: {complete}")
            attempts = stage_dir / "attempts"
            if attempts.exists():
                if not attempts.is_dir():
                    raise StateError(f"attempts is not a directory: {attempts}")
                for attempt in attempts.iterdir():
                    if attempt.is_symlink() or not attempt.is_dir() or not re.fullmatch(r"[0-9]{4}", attempt.name):
                        raise StateError(f"unknown attempt entry: {attempt}")
                    allowed_attempt = {"launch.json", "process-started.json", "stdout.log", "result.json", "validation.json", "failure.json", "temp"}
                    for child in attempt.iterdir():
                        if child.is_symlink() or child.name not in allowed_attempt:
                            raise StateError(f"unknown attempt entry: {child}")
                    temp = attempt / "temp"
                    if temp.exists():
                        if temp.is_symlink() or not temp.is_dir() or any(child.is_symlink() or child.name != "artifact.partial" or not child.is_file() for child in temp.iterdir()):
                            raise StateError(f"unknown temporary output: {temp}")
        existing_completions = []
        for stage_dir in self.stages_dir.iterdir():
            complete = stage_dir / "complete.json"
            if complete.exists():
                existing_completions.append(load_json(complete).get("digest"))
        if any(reference not in existing_completions for reference in completion_refs):
            raise StateError("manifest references missing completion record")
        output_run = self.output_root / self.run_id
        if not output_run.is_dir() or output_run.is_symlink():
            raise StateError("run output directory is missing or unsafe")
        for stage_output in output_run.iterdir():
            if stage_output.is_symlink() or not stage_output.is_dir() or stage_output.name not in set(ORDINARY_STAGE_NAMES):
                raise StateError(f"unknown run-output entry: {stage_output}")
            for child in stage_output.iterdir():
                if child.is_symlink() or child.name != "artifact" or not child.is_file():
                    raise StateError(f"unknown run-output file: {child}")
            if (stage_output / "artifact").exists() and not (self.stage_dir(stage_output.name) / "complete.json").exists():
                raise StateError(f"unknown final output without completion: {stage_output / 'artifact'}")


@contextmanager
def locked_run(state: RunState) -> Iterator[RunState]:
    lock_path = state.root / "lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield state
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def create_run(config: ResolvedConfig, run_id: str, registry_digest: str, pipeline_identity: list[dict[str, Any]]) -> RunState:
    try:
        validate_run_id(run_id)
    except ConfigError as exc:
        raise StateError(str(exc)) from exc
    root = config.state_root / run_id
    output_root = config.output_root / run_id
    if root.exists() or output_root.exists():
        raise StateError(f"run already exists: {run_id}")
    root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    state = RunState(config, run_id, root, config.output_root, registry_digest, pipeline_identity)
    state.manifest_dir.mkdir()
    state.stages_dir.mkdir()
    snapshot = config.as_snapshot(run_id, registry_digest, pipeline_identity, utc_now())
    atomic_write_json(root / "resolved-config.json", snapshot, mode=0o444)
    genesis_payload = {
        "schema_version": 1,
        "sequence": 0,
        "previous_entry_digest": None,
        "event_type": "genesis",
        "run_id": run_id,
        "stage": None,
        "referenced_record_digest": None,
        "event_data": {
            "source_config_identity": config.source_config_identity,
            "resolved_config_digest": config.resolved_config_digest,
            "registry_digest": registry_digest,
            "pipeline_identity": pipeline_identity,
        },
        "created_at": utc_now(),
    }
    genesis = {"payload": genesis_payload, "digest": digest_payload(genesis_payload)}
    atomic_write_json(state.manifest_dir / "000000-genesis.json", genesis)
    atomic_write_json(state.manifest_dir / "head.json", {"sequence": 0, "digest": genesis["digest"]})
    (root / "lock").touch()
    return state


def load_run(config: ResolvedConfig, run_id: str, registry_digest: str, pipeline_identity: list[dict[str, Any]]) -> RunState:
    try:
        validate_run_id(run_id)
    except ConfigError as exc:
        raise StateError(str(exc)) from exc
    root = config.state_root / run_id
    output_root = config.output_root / run_id
    if not root.is_dir() or not output_root.is_dir():
        raise StateError(f"run does not exist: {run_id}")
    state = RunState(config, run_id, root, config.output_root, registry_digest, pipeline_identity)
    snapshot = load_json(root / "resolved-config.json")
    if snapshot.get("run_id") != run_id or snapshot.get("resolved_config_digest") != config.resolved_config_digest:
        raise StateError("run snapshot does not match config")
    if snapshot.get("source_config_identity") != config.source_config_identity:
        raise StateError("source config identity changed")
    if snapshot.get("registry_digest") != registry_digest or snapshot.get("pipeline_identity") != pipeline_identity:
        raise StateError("run snapshot does not match current pipeline")
    state.verify_manifest()
    return state


def update_active(state: RunState) -> None:
    atomic_write_json(state.active_path, {"run_id": state.run_id, "resolved_config_digest": state.config.resolved_config_digest})


def active_run_id(config: ResolvedConfig) -> str | None:
    path = config.state_root / "active" / f"{config.resolved_config_digest}.json"
    if not path.exists():
        return None
    value = load_json(path)
    if not isinstance(value, dict) or set(value) != {"run_id", "resolved_config_digest"}:
        raise StateError("malformed active pointer")
    if value.get("resolved_config_digest") != config.resolved_config_digest:
        raise StateError("active pointer config mismatch")
    try:
        return validate_run_id(value["run_id"])
    except ConfigError as exc:
        raise StateError("active pointer run id is invalid") from exc
