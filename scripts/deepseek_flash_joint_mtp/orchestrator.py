"""Plan, dry-run, execution, resume, and state-only verification."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from .adapters import AdapterUnavailable, RunContext, adapter_for
from .canonical import atomic_write_json, digest_payload, identity_dict, load_json
from .config import ConfigError, ResolvedConfig, load_config, validate_run_id
from .runner import RunnerError, run_process
from .stages import ORDINARY_STAGE_NAMES, STAGE_SPECS, REGISTRY_DIGEST, ordinary_specs, transitive_dependencies
from .state import StateError, RunState, active_run_id, create_run, load_run, locked_run, update_active, utc_now


class PublicationError(RuntimeError):
    """Raised when an output cannot be safely published."""


class OrchestrationError(RuntimeError):
    """Mapped by the CLI into a stable exit code."""

    def __init__(self, message: str, code: int = 3):
        super().__init__(message)
        self.code = code


def pipeline_identity(entrypoint: Path) -> list[dict[str, Any]]:
    package = entrypoint.parent / "deepseek_flash_joint_mtp"
    paths = [entrypoint] + sorted(package.glob("*.py"))
    return [{"path": str(path.resolve(strict=True)), **identity_dict(path)} for path in paths]


def _context(config: ResolvedConfig, state: RunState) -> RunContext:
    return RunContext(config, state.run_id, state.registry_digest, state.pipeline_identity, state.root, state.config.output_root)


def _preview_context(config: ResolvedConfig) -> RunContext:
    fake_id = "preview-run"
    return RunContext(config, fake_id, REGISTRY_DIGEST, [], config.state_root / fake_id, config.output_root / fake_id)


def _preview_dependencies(config: ResolvedConfig, context: RunContext, stage: Any) -> dict[str, dict[str, Any]]:
    dependencies: dict[str, dict[str, Any]] = {}
    for name in stage.depends_on:
        output_path = context.output_root / name / "artifact"
        digest = digest_payload({"preview": True, "stage": name, "config": config.resolved_config_digest})
        dependencies[name] = {
            "digest": digest,
            "preview": True,
            "payload": {
                "output_identities": {
                    "artifact": {
                        "kind": "preview",
                        "stage": name,
                        "canonical_path": str(output_path),
                        "digest": digest,
                    }
                }
            },
        }
    return dependencies


def _plan_object(config: ResolvedConfig) -> dict[str, Any]:
    nodes = []
    context = _preview_context(config)
    for stage in STAGE_SPECS:
        adapter = adapter_for(config, stage.name)
        item: dict[str, Any] = {
            "name": stage.name,
            "depends_on": list(stage.depends_on),
            "kind": stage.kind,
            "contract_version": stage.contract_version,
            "output_slots": list(stage.output_slots),
            "boundary_resume_only": stage.boundary_resume_only,
            "adapter_status": "not_implemented" if config.adapter == "production-v1" and stage.name != "train-joint" else "available",
            "launchable": stage.kind == "ordinary" and (config.adapter == "fixture-v1" or stage.name == "train-joint"),
            "approved_roots": config.data["paths"]["approved_write_roots"],
            "protected_roots": config.data["paths"]["protected_roots"],
        }
        if stage.kind == "ordinary" and (config.adapter == "fixture-v1" or stage.name == "train-joint"):

            attempt = context.state_root / stage.name / "attempts" / "0001"
            temp = attempt / "temp" / "artifact.partial"
            final = context.output_root / stage.name / "artifact"
            try:
                dependencies = _preview_dependencies(config, context, stage)
                launch = adapter.render(stage, context, dependencies, attempt, temp, final)
                item["preview_dependencies"] = {name: record["digest"] for name, record in dependencies.items()}
                item["argv"] = list(launch.argv)
                item["cwd"] = str(launch.cwd)
                item["environment"] = dict(launch.environment)
                item["input_identities"] = dict(launch.input_identities)
                item["temporary_outputs"] = [str(path) for path in launch.temporary_outputs]
                item["final_outputs"] = [str(path) for path in launch.final_outputs]
                item["publication_order"] = [str(path) for path in launch.final_outputs]
            except Exception as exc:
                item["launchable"] = False
                item["reason"] = str(exc)
        else:
            item["argv"] = None
            item["reason"] = "not implemented in Slice 1" if stage.kind == "ordinary" else "human action; separate command"
        nodes.append(item)
    return {
        "schema_version": 1,
        "pipeline_version": config.data["pipeline_version"],
        "adapter": config.adapter,
        "registry_digest": REGISTRY_DIGEST,
        "resolved_config_digest": config.resolved_config_digest,
        "source_config_identity": config.source_config_identity,
        "nodes": nodes,
        "approved_read_roots": config.data["paths"]["approved_read_roots"],
        "approved_write_roots": config.data["paths"]["approved_write_roots"],
        "protected_roots": config.data["paths"]["protected_roots"],
    }


def plan(config_path: str | Path) -> dict[str, Any]:
    return _plan_object(load_config(config_path))


def dry_run(config_path: str | Path, fixture_root: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path, preview_fixture_root=fixture_root)
    result = _plan_object(config)
    result["dry_run"] = True
    result["preview_run_id"] = "preview-run"
    result["launches"] = [
        node for node in result["nodes"] if node["kind"] == "ordinary" and node.get("launchable")
    ]
    return result


def _new_run_id() -> str:
    return "run-" + time.strftime("%Y%m%dt%H%M%sz", time.gmtime()) + "-" + secrets.token_hex(4)


def _prepare_state(config: ResolvedConfig, run_id: str | None, resume: bool, entrypoint: Path) -> RunState:
    identity = pipeline_identity(entrypoint)
    if resume:
        selected = run_id or active_run_id(config)
        if selected is None:
            selected = _new_run_id()
            config.state_root.mkdir(parents=True, exist_ok=True)
            config.output_root.mkdir(parents=True, exist_ok=True)
            state = create_run(config, selected, REGISTRY_DIGEST, identity)
            update_active(state)
            return state
        try:
            validate_run_id(selected)
        except ConfigError as exc:
            raise OrchestrationError(str(exc), 3) from exc
        return load_run(config, selected, REGISTRY_DIGEST, identity)
    selected = run_id or _new_run_id()
    config.state_root.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    return create_run(config, selected, REGISTRY_DIGEST, identity)


def _dependency_records(state: RunState, stage_name: str, *, current_tool_identity: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for dependency in transitive_dependencies(stage_name):
        record = state.read_completion(dependency)
        if record is None:
            raise OrchestrationError(f"missing dependency completion: {dependency}", 3)
        dependency_digests = {
            ancestor: state.read_completion(ancestor)["digest"]
            for ancestor in transitive_dependencies(dependency)
        }
        state.completion_valid(dependency, dependency_digests, current_tool_identity=current_tool_identity)
        records[dependency] = record
    return records


def _publish(state: RunState, stage: str, temporary: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise PublicationError(f"unknown existing final output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    if not temporary.exists() or temporary.is_symlink():
        raise PublicationError(f"temporary output missing: {temporary}")
    try:
        if temporary.stat().st_dev != final.parent.stat().st_dev:
            raise PublicationError("temporary and final output are on different filesystems")
        os.replace(temporary, final)
        directory_fd = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except PublicationError:
        raise
    except OSError as exc:
        raise PublicationError(str(exc)) from exc


def _execute_stage(state: RunState, stage_spec: Any, adapter: Any) -> int:
    try:
        current_tool_identity = adapter.tool_identity() if hasattr(adapter, "tool_identity") else None
    except AdapterUnavailable as exc:
        raise OrchestrationError(str(exc), 4) from exc
    dependencies = _dependency_records(state, stage_spec.name, current_tool_identity=current_tool_identity)
    dependency_digests = {name: record["digest"] for name, record in dependencies.items()}
    existing = state.read_completion(stage_spec.name)
    if existing is not None:
        if state.completion_valid(stage_spec.name, dependency_digests, current_tool_identity=current_tool_identity):
            return 0
        raise OrchestrationError(f"invalid completed stage: {stage_spec.name}", 3)
    final = state.output_path(stage_spec.name)
    if final.exists() or final.is_symlink():
        raise OrchestrationError(f"unknown final output without completion: {final}", 3)
    attempt_id = state.next_attempt_id(stage_spec.name)
    attempt_dir = state.stage_dir(stage_spec.name) / "attempts" / attempt_id
    temporary = attempt_dir / "temp" / "artifact.partial"
    try:
        launch = adapter.render(stage_spec, _context(state.config, state), dependencies, attempt_dir, temporary, final)
    except AdapterUnavailable as exc:
        state.write_failure(stage_spec.name, attempt_id, {"failure_class": "validation", "diagnostic": str(exc)})
        raise OrchestrationError(str(exc), 4) from exc
    try:
        result = run_process(
            launch,
            attempt_dir,
            manifest_event=lambda event, data: state.append_manifest(event, data, stage=stage_spec.name),
        )
    except RunnerError as exc:
        state.write_failure(stage_spec.name, attempt_id, {"failure_class": "launch", "diagnostic": str(exc)})
        raise OrchestrationError(str(exc), 5) from exc
    if result.forwarded_signals:
        state.write_failure(stage_spec.name, attempt_id, {"failure_class": "interrupted", "process_result": result.as_dict()})
        return 128 + result.forwarded_signals[0]
    if result.return_code != 0:
        state.write_failure(stage_spec.name, attempt_id, {"failure_class": "child", "process_result": result.as_dict()})
        return result.return_code
    if result.log_error is not None:
        state.write_failure(stage_spec.name, attempt_id, {"failure_class": "logging", "diagnostic": result.log_error, "process_result": result.as_dict()})
        return 5
    try:
        validation = adapter.validate(stage_spec, _context(state.config, state), launch, result, temporary)
        validation_path = attempt_dir / "validation.json"
        atomic_write_json(validation_path, validation)
        try:
            _publish(state, stage_spec.name, temporary, final)
        except PublicationError as exc:
            state.write_failure(stage_spec.name, attempt_id, {"failure_class": "publication", "diagnostic": str(exc), "process_result": result.as_dict()})
            raise OrchestrationError(f"stage publication failed: {stage_spec.name}: {exc}", 5) from exc
        output_identities = {"artifact": identity_dict(final)}
        launch_record = load_json(attempt_dir / "launch.json")
        completion_payload = {
            "schema_version": 1,
            "run_id": state.run_id,
            "stage": stage_spec.name,
            "stage_contract_version": stage_spec.contract_version,
            "attempt_id": attempt_id,
            "resolved_config_digest": state.config.resolved_config_digest,
            "registry_digest": state.registry_digest,
            "pipeline_identity_digest": digest_payload(state.pipeline_identity),
            "dependency_completion_digests": dependency_digests,
            "launch_digest": launch_record["launch_digest"],
            "argv": list(launch.argv),
            "cwd": str(launch.cwd),
            "environment": dict(launch.environment),
            "tool_identity": dict(launch.tool_identity),
            "input_identities": dict(launch.input_identities),
            "output_identities": output_identities,
            "started_at": result.started_at,
            "ended_at": result.ended_at,
            "process_status": result.status,
            "validation_report_identity": identity_dict(validation_path),
            "next_stage_eligible": True,
        }
        state.write_completion(stage_spec.name, completion_payload)
        return 0
    except OrchestrationError:
        raise
    except Exception as exc:
        state.write_failure(stage_spec.name, attempt_id, {"failure_class": "validation", "diagnostic": str(exc), "process_result": result.as_dict()})
        raise OrchestrationError(f"stage validation failed: {stage_spec.name}: {exc}", 5) from exc


def run(config_path: str | Path, *, run_id: str | None = None, resume: bool = False, from_stage: str | None = None, through_stage: str | None = None, entrypoint: Path | None = None) -> tuple[int, str]:
    config = load_config(config_path)
    if config.adapter == "production-v1":
        raise OrchestrationError("not implemented in Slice 1", 4)
    if not resume and from_stage not in (None, "preflight"):
        raise OrchestrationError("a new run may start only at preflight", 3)
    if resume and from_stage not in (None, "preflight") and run_id is None and active_run_id(config) is None:
        raise OrchestrationError("a new run may start only at preflight", 3)
    try:
        selected = ordinary_specs(from_stage=from_stage, through_stage=through_stage)
    except ValueError as exc:
        raise OrchestrationError(str(exc), 2) from exc
    entrypoint = entrypoint or Path(__file__).resolve().parent.parent / "deepseek_flash_joint_mtp_pipeline.py"
    state = _prepare_state(config, run_id, resume, entrypoint)
    with locked_run(state):
        adapter = adapter_for(config, "train-joint")
        for stage_spec in selected:
            code = _execute_stage(state, stage_spec, adapter)
            if code != 0:
                return code, state.run_id
        return 0, state.run_id


def verify(config_path: str | Path, run_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    entrypoint = Path(__file__).resolve().parent.parent / "deepseek_flash_joint_mtp_pipeline.py"
    state = load_run(config, run_id, REGISTRY_DIGEST, pipeline_identity(entrypoint))
    adapter = adapter_for(config, "train-joint")
    try:
        current_tool_identity = adapter.tool_identity() if hasattr(adapter, "tool_identity") else None
    except AdapterUnavailable as exc:
        raise StateError(str(exc)) from exc
    with locked_run(state):
        completed: list[str] = []
        for stage in ORDINARY_STAGE_NAMES:
            record = state.read_completion(stage)
            if record is None:
                return {"valid": True, "run_id": run_id, "completed": completed, "next_stage": stage}
            dependencies = {name: state.read_completion(name)["digest"] for name in transitive_dependencies(stage)}
            if not state.completion_valid(stage, dependencies, current_tool_identity=current_tool_identity):
                raise StateError(f"invalid completion: {stage}")
            completed.append(stage)
        return {"valid": True, "run_id": run_id, "completed": completed, "next_stage": None}


def unavailable_action(config_path: str | Path, run_id: str, action: str, confirm: str) -> None:
    del config_path, run_id, action, confirm
    raise OrchestrationError("not implemented in Slice 1", 4)
