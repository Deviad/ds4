#!/usr/bin/env python3
"""Fail-closed, synthetic-first two-phase DS4 checkpoint/resume pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PILOT_VENDOR_SHA = "80fab4e419a57f9465bb9e2f4e90010d645e124c"
PILOT_PROVIDER_SHA256 = "20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518"
PILOT_SMOKE_SHA256 = "ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8"
PILOT_WORKSPACE = "/Volumes/Data NVME/mlx-ft/ds4"
PILOT_INTERPRETER = "/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python"
PILOT_MODEL = "/Volumes/Data NVME/mlx-ft/ds4/model-4bit"
PILOT_DATA = "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke"
PILOT_CONFIG = "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json"
PILOT_PROVENANCE = "agent-output/cmux-14-3/filtered-dataset-provenance.json"
PILOT_LOCK_TIMEOUT = 60
PILOT_LOCK_POLL = 2
PILOT_MEMORY_HEADROOM = 32 * 1024**3
PILOT_DISK_MIN_FREE = 1 * 1024**3
PILOT_TOTAL_BUDGET = 4200
PILOT_FINAL_REPORT = REPO_ROOT / "agent-output" / "cmux-14-5" / "pilot-report.json"
RETRY_POLICY = "none"
FALLBACK_POLICY = "none"
TERMINAL_MUTATION_POINTS = (
    "parser/config", "preflight", "start-save", "provider", "callback",
    "checkpoint-validation", "phase-report", "final-report", "phase-marker",
    "final-marker", "timeout", "watchdog-cancellation", "lock-release",
    "phase-b-success", "phase-b-failure", "final-aggregation",
)

COMMON_VALUES: dict[str, Any] = {
    "max_seq_length": 4096, "batch_size": 1, "learning_rate": 1e-5,
    "mask_prompt": True, "grad_checkpoint": True, "segment_size": 1,
    "grad_accumulation_steps": 1, "seed": 0, "fine_tune_type": "lora",
    "optimizer": "adam", "num_layers": 16, "val_batches": 25,
    "steps_per_report": 1, "save_every": 1, "report_to": None,
    "project_name": None, "trust_remote_code": False, "lr_schedule": None,
    "clear_cache_threshold": 0,
}
PHASES: dict[str, dict[str, Any]] = {
    "phase-a": {"iters": 2, "steps_per_eval": 2, "timeout": 2700, "global_offset": 0,
                "adapter_path": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a",
                "resume_adapter_file": None, "report": "agent-output/cmux-14-5/phase-a-report.json"},
    "phase-b": {"iters": 1, "steps_per_eval": 1, "timeout": 1500, "global_offset": 2,
                "adapter_path": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b",
                "resume_adapter_file": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors",
                "report": "agent-output/cmux-14-5/phase-b-report.json"},
}
NON_CLAIMS = ["optimizer_state_continuity", "rng_state_continuity", "dataset_cursor_continuity",
              "scheduler_continuity", "trainer_global_iteration", "convergence_or_quality",
              "throughput_improvement", "full-training-readiness"]


class PilotError(RuntimeError):
    """Terminal fail-closed pilot error."""


@dataclass(frozen=True)
class Watchdog:
    event: threading.Event
    thread: threading.Thread

    def cancel(self) -> None:
        self.event.set()
        signal.alarm(0)


@dataclass(frozen=True)
class _ArrayMeta:
    path: str
    shape: tuple[int, ...]
    dtype: str


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def file_sha256(path: pathlib.Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with pathlib.Path(path).open("rb") as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise PilotError(f"file hash failed: {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json_snapshot(path: pathlib.Path) -> tuple[str, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as exc:
        raise PilotError("Phase A hashed report payload is not readable JSON") from exc
    if not isinstance(parsed, dict):
        raise PilotError("Phase A hashed report payload must be a JSON object")
    return digest, parsed


def _frame(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def canonical_tensor_digest(path: pathlib.Path | str) -> dict[str, Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise PilotError(f"safetensors path is not a regular file: {path}")
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise PilotError(f"truncated safetensors header: {path}")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length > 128 * 1024 * 1024:
                raise PilotError(f"safetensors header too large: {path}")
            header_bytes = source.read(header_length)
            if len(header_bytes) != header_length:
                raise PilotError(f"truncated safetensors JSON header: {path}")
            try:
                header = json.loads(header_bytes, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, _DuplicateKeyError) as exc:
                raise PilotError(f"invalid safetensors JSON header: {path}: {exc}") from exc
            if not isinstance(header, dict):
                raise PilotError(f"invalid safetensors layout: {path}")
            data_start = 8 + header_length
            data_size = size - data_start
            if data_size < 0:
                raise PilotError(f"invalid safetensors layout: {path}")
            metadata = header.get("__metadata__", {})
            if not isinstance(metadata, dict):
                raise PilotError("safetensors __metadata__ must be an object")
            tensors: list[tuple[str, str, list[int], int, int]] = []
            spans: list[tuple[int, int, str]] = []
            for name, item in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(name, str) or not isinstance(item, dict):
                    raise PilotError(f"invalid tensor entry: {name}")
                dtype, shape, offsets = item.get("dtype"), item.get("shape"), item.get("data_offsets")
                if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                    raise PilotError(f"invalid tensor schema: {name}")
                if any(type(dim) is not int or dim < 0 for dim in shape):
                    raise PilotError(f"invalid tensor shape: {name}")
                if any(type(pos) is not int for pos in offsets):
                    raise PilotError(f"invalid tensor offsets: {name}")
                start, end = offsets
                if start < 0 or end < start or end > data_size:
                    raise PilotError(f"tensor offsets out of bounds: {name}")
                expected = _DTYPE_BYTES[dtype]
                elements = math.prod(shape)
                if end - start != elements * expected:
                    raise PilotError(f"tensor byte size does not match dtype/shape: {name}")
                spans.append((start, end, name))
                tensors.append((name, dtype, shape, start, end))
            if not tensors:
                raise PilotError(f"safetensors contains no tensors: {path}")
            spans.sort()
            if spans[0][0] != 0:
                raise PilotError("tensor offsets are not contiguous: leading gap")
            for previous, current in zip(spans, spans[1:]):
                if current[0] != previous[1]:
                    if current[0] < previous[1]:
                        raise PilotError(f"overlapping tensor offsets: {previous[2]}, {current[2]}")
                    raise PilotError("tensor offsets are not contiguous: gap")
            if spans[-1][1] != data_size:
                raise PilotError("safetensors contains trailing tensor data")
            tensors.sort(key=lambda item: item[0])
            digest = hashlib.sha256(b"DS4_CANONICAL_TENSOR_V1")
            manifest: list[dict[str, Any]] = []
            for name, dtype, shape, start, end in sorted(tensors, key=lambda item: item[0]):
                _frame(digest, name.encode())
                _frame(digest, dtype.encode())
                digest.update(struct.pack("<Q", len(shape)))
                for dimension in shape:
                    digest.update(struct.pack("<Q", dimension))
                nbytes = end - start
                digest.update(struct.pack("<Q", nbytes))
                source.seek(data_start + start)
                remaining = nbytes
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise PilotError(f"truncated tensor data: {name}")
                    digest.update(chunk)
                    remaining -= len(chunk)
                manifest.append({"name": name, "dtype": dtype, "shape": shape, "nbytes": nbytes})
    except OSError as exc:
        raise PilotError(f"safetensors read failed: {path}: {exc}") from exc
    return {"path": str(path), "file_sha256": file_sha256(path),
            "canonical_tensor_digest_v1": digest.hexdigest(), "tensor_count": len(manifest),
            "tensors": manifest}


def verify_resume_continuity(source: pathlib.Path | str, resume_start: pathlib.Path | str,
                             phase_start: pathlib.Path | str) -> dict[str, Any]:
    source_info, resume_info, start_info = (canonical_tensor_digest(item) for item in (source, resume_start, phase_start))
    equal = source_info["canonical_tensor_digest_v1"] == resume_info["canonical_tensor_digest_v1"] and source_info["tensors"] == resume_info["tensors"]
    changed = resume_info["canonical_tensor_digest_v1"] != start_info["canonical_tensor_digest_v1"]
    if not equal:
        raise PilotError("resume-start digest/schema does not equal Phase A checkpoint")
    if not changed:
        raise PilotError("resume-start digest must differ from Phase A start")
    return {"equal_source": True, "changed_from_phase_start": True, "source": source_info, "resume_start": resume_info}


def ensure_absent(path: pathlib.Path | str) -> None:
    if pathlib.Path(path).exists():
        raise PilotError(f"output or marker already exists: {path}")


def global_steps(phase: str, local_steps: Iterable[int]) -> list[int]:
    if phase not in PHASES:
        raise PilotError(f"unknown phase: {phase}")
    return [PHASES[phase]["global_offset"] + int(step) for step in local_steps]


def validate_step_evidence(phase: str, records: list[dict[str, Any]], expected: int) -> list[int]:
    if len(records) != expected:
        raise PilotError(f"{phase} evidence cardinality {len(records)} != {expected}")
    local_steps = [int(record.get("local_step", -1)) for record in records]
    if local_steps != list(range(1, expected + 1)):
        raise PilotError(f"{phase} local steps out of order: {local_steps}")
    for record in records:
        loss, tokens = record.get("loss"), record.get("n_tokens")
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)) or type(tokens) is not int or tokens <= 0:
            raise PilotError(f"{phase} nonfinite or invalid step evidence")
        if record.get("gradients_finite") is not True:
            raise PilotError(f"{phase} gradient finiteness evidence missing")
        schema = record.get("gradient_schema")
        if not isinstance(schema, list) or not schema or record.get("gradient_leaf_count") != len(schema):
            raise PilotError(f"{phase} gradient schema must be nonempty")
        if any(not isinstance(item, dict) or not item.get("path") or "shape" not in item or "dtype" not in item for item in schema):
            raise PilotError(f"{phase} gradient schema evidence incomplete")
        if record.get("expected_mask_tokens") is None or record.get("mask_tokens_match") is not True:
            raise PilotError(f"{phase} mask token evidence mismatch")
    return local_steps


def validate_phase_b_dependency(report: dict[str, Any], marker: dict[str, Any], resume_source: pathlib.Path | str) -> None:
    if report.get("status") != "ok" or report.get("phase") != "phase-a" or marker.get("status") != "ok" or marker.get("phase") != "phase-a":
        raise PilotError("Phase A report and OK marker required")
    if report.get("provider_calls") != 2 or report.get("optimizer_updates") != 2:
        raise PilotError("Phase A report cardinality binding missing")
    report_path = marker.get("report_path")
    expected_report_sha = marker.get("report_sha256")
    canonical_report_path = (REPO_ROOT / PHASES["phase-a"]["report"]).resolve()
    if not report_path or pathlib.Path(report_path).resolve() != canonical_report_path:
        raise PilotError("Phase A marker report path is not the canonical report")
    if not expected_report_sha:
        raise PilotError("Phase A report hash binding mismatch")
    observed_report_sha, hashed_report = _read_json_snapshot(canonical_report_path)
    if observed_report_sha != expected_report_sha:
        raise PilotError("Phase A report hash binding mismatch")
    if hashed_report != report:
        raise PilotError("Phase A report object does not equal hashed payload")
    if not marker.get("contract_digest") or marker.get("contract_digest") != report.get("contract_digest"):
        raise PilotError("Phase A contract binding missing")
    source = pathlib.Path(resume_source)
    source_info = report.get("resume_source")
    if not isinstance(source_info, dict) or source_info.get("path") != str(source):
        raise PilotError("Phase A resume source path binding missing")
    if marker.get("resume_source") not in (str(source), source_info):
        raise PilotError("Phase A resume source path mismatch")
    if marker.get("resume_source_file_sha256") != source_info.get("file_sha256") or marker.get("resume_source_canonical_tensor_digest_v1") != source_info.get("canonical_tensor_digest_v1"):
        raise PilotError("Phase A resume source marker binding mismatch")
    if not source.is_file() or file_sha256(source) != source_info.get("file_sha256"):
        raise PilotError("Phase A resume source hash mismatch")
    if not source_info.get("canonical_tensor_digest_v1"):
        raise PilotError("Phase A resume source canonical digest binding missing")


def _effective(args: argparse.Namespace) -> dict[str, Any]:
    keys = set(COMMON_VALUES) | {"phase", "model", "data", "config", "adapter_path", "resume_adapter_file", "train", "test", "hf_dataset"}
    effective = {key: getattr(args, key, None) for key in sorted(keys)}
    command = list(getattr(args, "_command_argv", ()) or sys.argv)
    effective["command"] = command or [PILOT_INTERPRETER, str(REPO_ROOT / "scripts/ds4_segmented_pilot.py")]
    return effective


def _config_value(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    nested = config.get("lora_parameters")
    return nested.get(key) if isinstance(nested, dict) else None


_OPTIONAL_PIN_KEYS = {"report_to", "project_name", "trust_remote_code", "lr_schedule", "clear_cache_threshold"}


def validate_pins(args: argparse.Namespace, phase: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    expected = dict(COMMON_VALUES)
    expected.update({"iters": phase["iters"], "steps_per_eval": phase["steps_per_eval"], "adapter_path": phase["adapter_path"], "resume_adapter_file": phase["resume_adapter_file"]})
    if args.phase not in PHASES:
        raise PilotError("unknown phase")
    if args.train is not None and args.train is not True:
        raise PilotError("pinned-args: --train is required")
    for key, expected_value in expected.items():
        actual = getattr(args, key, None)
        if actual is not None and actual != expected_value:
            raise PilotError(f"pinned-args: {key}={actual!r}, expected {expected_value!r}")
    if args.train is not True:
        raise PilotError("required argument --train")
    for key, expected_value in expected.items():
        actual = getattr(args, key, None)
        if expected_value is None:
            if actual is not None:
                raise PilotError(f"pinned-args: {key} must be omitted for {args.phase}")
            continue
        if actual is None and key not in _OPTIONAL_PIN_KEYS:
            raise PilotError(f"required argument: --{key.replace('_', '-')}")
        if actual is None and key in _OPTIONAL_PIN_KEYS:
            actual = expected_value
        config_value = _config_value(config, key)
        if config_value is not None and config_value != expected_value:
            raise PilotError(f"config-conflict: {key}={config_value!r}, expected {expected_value!r}")
    for key, expected_path in {"model": PILOT_MODEL, "data": PILOT_DATA, "config": PILOT_CONFIG, "adapter_path": phase["adapter_path"], "resume_adapter_file": phase["resume_adapter_file"]}.items():
        actual = getattr(args, key, None)
        if expected_path is None:
            if actual is not None:
                raise PilotError(f"pinned-paths: {key} must be omitted for {args.phase}")
            continue
        if actual is None:
            raise PilotError(f"required argument: --{key.replace('_', '-')}")
        if str(actual) != expected_path:
            raise PilotError(f"pinned-paths: {key}={actual!r}, expected {expected_path!r}")
    if args.phase == "phase-a" and args.resume_adapter_file is not None:
        raise PilotError("pinned-args: Phase A cannot resume")
    if args.phase == "phase-b" and not args.resume_adapter_file:
        raise PilotError("pinned-args: Phase B requires resume source")
    effective = _effective(args)
    effective.update(expected)
    return effective


def _read_config(path: str | None) -> dict[str, Any]:
    if not path:
        raise PilotError("required config path")
    try:
        with pathlib.Path(path).open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"config file read/parse failed: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotError("config must be a JSON object")
    return value


def _validate_lora_config(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("lora_parameters", config)
    if not isinstance(values, dict):
        raise PilotError("LoRA config must contain an object")
    expected = {"rank": 8, "scale": 20.0, "dropout": 0.0}
    for key, value in expected.items():
        if values.get(key) != value:
            raise PilotError(f"LoRA config pin mismatch: {key}")
    keys = values.get("keys")
    if not isinstance(keys, list) or len(keys) != len(set(keys)) or not keys:
        raise PilotError("LoRA config canonical key set missing")
    try:
        from ds4_ft_mlx.lora_targets import build_lora_parameters
        canonical = build_lora_parameters(rank=8, scale=20.0, dropout=0.0)["keys"]
    except ImportError as exc:
        raise PilotError(f"LoRA target helper unavailable: {exc}") from exc
    if keys != canonical:
        raise PilotError("LoRA config canonical key set mismatch")
    return {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": list(keys)}


def atomic_write_json(path: pathlib.Path | str, value: dict[str, Any]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_text_fsync(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def aggregate_reports(phase_a: dict[str, Any], phase_b: dict[str, Any]) -> dict[str, Any]:
    if phase_a.get("status") != "ok" or phase_b.get("status") != "ok":
        raise PilotError("final report requires both phases to pass")
    steps = list(phase_a.get("steps", [])) + list(phase_b.get("steps", []))
    if [step.get("global_step") for step in steps] != [1, 2, 3]:
        raise PilotError("final global progression must be 1,2,3")
    total = float(phase_a.get("wall_seconds", 0.0)) + float(phase_b.get("wall_seconds", 0.0))
    if total > PILOT_TOTAL_BUDGET:
        raise PilotError("active wall-clock budget exceeded")
    identity_a = phase_a.get("identity_manifest") or {}
    identity_b = phase_b.get("identity_manifest") or {}
    immutable_a = identity_a.get("immutable", identity_a) if isinstance(identity_a, dict) else identity_a
    immutable_b = identity_b.get("immutable", identity_b) if isinstance(identity_b, dict) else identity_b
    if immutable_a and immutable_b and immutable_a != immutable_b:
        raise PilotError("Phase A/B immutable identity mismatch")
    final_contract = hashlib.sha256(json.dumps(
        {"phase_a": phase_a.get("contract_digest", ""), "phase_b": phase_b.get("contract_digest", "")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"status": "ok", "provider_calls": phase_a.get("provider_calls", 0) + phase_b.get("provider_calls", 0),
            "optimizer_updates": phase_a.get("optimizer_updates", 0) + phase_b.get("optimizer_updates", 0),
            "steps": steps, "global_progression": [0, 1, 2, 3], "total_active_wall_seconds": total,
            "phase_a": phase_a, "phase_b": phase_b, "output_path": phase_b.get("output_path", PHASES["phase-b"]["adapter_path"]),
            "contract_digest": final_contract, "non_claims": list(NON_CLAIMS)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    for name in ("model", "data", "adapter_path", "config", "resume_adapter_file"):
        parser.add_argument(f"--{name.replace('_', '-')}")
    parser.add_argument("--train", action="store_true", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--hf-dataset", default=False)
    parser.add_argument("--fine-tune-type")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--mask-prompt", action="store_true", default=None)
    parser.add_argument("--grad-checkpoint", action="store_true", default=None)
    parser.add_argument("--grad-accumulation-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--val-batches", type=int)
    parser.add_argument("--steps-per-report", type=int)
    parser.add_argument("--steps-per-eval", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--segment-size", type=int)
    return parser


_LOCK_OWNED_PATH: pathlib.Path | None = None
_LOCK_OWNED_TOKEN: str | None = None
_LOCK_OWNED_PARTIAL = False


def _ft_lock_path(workspace: pathlib.Path | str | None = None) -> pathlib.Path:
    if workspace is None:
        workspace = PILOT_WORKSPACE
    return pathlib.Path(workspace) / ".ds4-ft.lock"


def _acquire_ft_lock(workspace: pathlib.Path | str | None = None) -> pathlib.Path:
    global _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN, _LOCK_OWNED_PARTIAL
    if workspace is None:
        workspace = PILOT_WORKSPACE
    workspace = pathlib.Path(workspace)
    if str(workspace) != PILOT_WORKSPACE and not workspace.exists():
        raise PilotError(f"lock workspace must already exist: {workspace}")
    lock = _ft_lock_path(workspace)
    if not lock.parent.is_dir():
        raise PilotError(f"pinned lock workspace missing: {lock.parent}")
    token = str(os.getpid())
    deadline = time.monotonic() + PILOT_LOCK_TIMEOUT
    while time.monotonic() < deadline:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(PILOT_LOCK_POLL)
            continue
        except OSError as exc:
            raise PilotError(f"lock acquisition failed: {exc}") from exc
        _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN, _LOCK_OWNED_PARTIAL = lock, token, True
        owner = None
        try:
            owner = os.fdopen(fd, "w", encoding="utf-8")
            owner.write(f"pid={token}\n")
            owner.flush()
            os.fsync(owner.fileno())
        except Exception as exc:
            try:
                if owner is None:
                    os.close(fd)
                else:
                    owner.close()
            except OSError:
                pass
            cleanup_error = None
            try:
                lock.unlink()
            except OSError as unlink_error:
                cleanup_error = unlink_error
            if cleanup_error is None:
                _LOCK_OWNED_PATH = _LOCK_OWNED_TOKEN = None
                _LOCK_OWNED_PARTIAL = False
            detail = f"; partial-lock cleanup failed: {cleanup_error}" if cleanup_error else ""
            raise PilotError(f"lock acquisition failed: {exc}{detail}") from exc
        finally:
            if owner is not None and not owner.closed:
                try:
                    owner.close()
                except OSError:
                    pass
        _LOCK_OWNED_PARTIAL = False
        return lock
    raise PilotError(f"lock acquisition timeout after {PILOT_LOCK_TIMEOUT}s: {lock}")


def _release_ft_lock() -> None:
    global _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN, _LOCK_OWNED_PARTIAL
    lock, token = _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN
    if lock is None or token is None:
        return
    if not _LOCK_OWNED_PARTIAL and (not lock.is_file() or lock.read_text(encoding="utf-8") != f"pid={token}\n"):
        raise PilotError("refusing to release lock owned by another process")
    lock.unlink()
    _LOCK_OWNED_PATH = _LOCK_OWNED_TOKEN = None
    _LOCK_OWNED_PARTIAL = False


def _quarantine_ft_lock() -> pathlib.Path | None:
    """Move a retained partial lock out of the blocking lock pathname."""
    global _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN, _LOCK_OWNED_PARTIAL
    lock, token = _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN
    if lock is None or token is None:
        return None
    if not lock.exists():
        _LOCK_OWNED_PATH = _LOCK_OWNED_TOKEN = None
        _LOCK_OWNED_PARTIAL = False
        return None
    quarantine = lock.with_name(f"{lock.name}.quarantine.{token}")
    os.replace(lock, quarantine)
    _LOCK_OWNED_PATH = _LOCK_OWNED_TOKEN = None
    _LOCK_OWNED_PARTIAL = False
    return quarantine


def _install_timeout_watchdog(seconds: int) -> Watchdog:
    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise PilotError(f"phase timeout after {seconds}s")
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(seconds)
    event = None
    try:
        event = threading.Event()
        def backup() -> None:
            if not event.wait(seconds + 5):
                os.kill(os.getpid(), signal.SIGALRM)
        thread = threading.Thread(target=backup, name="ds4-pilot-watchdog", daemon=True)
        thread.start()
        return Watchdog(event, thread)
    except Exception:
        signal.alarm(0)
        if event is not None:
            event.set()
        raise


def _flatten_gradients(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(tree, dict):
        output: list[tuple[str, Any]] = []
        for key in sorted(tree):
            output.extend(_flatten_gradients(tree[key], f"{prefix}.{key}" if prefix else str(key)))
        return output
    if isinstance(tree, (list, tuple)):
        output = []
        for index, value in enumerate(tree):
            output.extend(_flatten_gradients(value, f"{prefix}.{index}" if prefix else str(index)))
        return output
    return [(prefix, tree)]


def _pinned_flatten(tree: Any, tree_flatten: Any) -> list[tuple[str, Any]]:
    if tree_flatten is None:
        raise PilotError("MLX tree_flatten unavailable")
    flattened = tree_flatten(tree)
    items = list(flattened.items()) if isinstance(flattened, dict) else list(flattened)
    if not items or any(not isinstance(path, str) or not path for path, _ in items):
        raise PilotError("trainable tree must contain nonempty named leaves")
    return items


def _schema_from_tree(tree: Any, tree_flatten: Any) -> list[dict[str, Any]]:
    leaves = [_array_meta(path, value) for path, value in _pinned_flatten(tree, tree_flatten)]
    schema = [{"path": item.path, "shape": list(item.shape), "dtype": item.dtype} for item in leaves]
    if not schema:
        raise PilotError("trainable schema must not be empty")
    return schema


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _array_meta(path: str, value: Any) -> _ArrayMeta:
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    if dtype.startswith("mlx.core."):
        dtype = dtype.removeprefix("mlx.core.")
    return _ArrayMeta(path, shape, dtype)


def _finite(value: Any, mx: Any = None) -> bool:
    try:
        if mx is not None and hasattr(mx, "isfinite"):
            check = mx.all(mx.isfinite(value))
            if hasattr(mx, "eval"):
                mx.eval(check)
            return bool(_scalar(check))
        result = value if isinstance(value, (int, float)) else bool(value)
        return bool(result) and (not isinstance(result, float) or math.isfinite(result))
    except (TypeError, ValueError):
        return False


def _expected_mask_tokens(batch: tuple[Any, ...]) -> int | None:
    if len(batch) < 2:
        return None
    tokens, lengths = batch[0], batch[1]
    token_shape = tuple(int(item) for item in getattr(tokens, "shape", ()))
    if len(token_shape) != 2:
        try:
            token_shape = (len(tokens), len(tokens[0]))
        except (TypeError, IndexError):
            return None
    target_length = token_shape[1] - 1
    if target_length < 1:
        return None
    if hasattr(lengths, "tolist"):
        lengths = lengths.tolist()
    try:
        rows = lengths if isinstance(lengths, list) and lengths and isinstance(lengths[0], (list, tuple)) else [lengths]
        total = 0
        for row in rows:
            lower = max(1, int(row[0]))
            upper = min(target_length, int(row[1]))
            total += max(0, upper - lower + 1)
        return total
    except (TypeError, IndexError, ValueError):
        return None


def _require_scalar_dtype(value: Any, mx: Any, expected: str, label: str) -> Any:
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    if shape != ():
        raise PilotError(f"provider {label} must be a scalar {expected}")
    actual_dtype = getattr(value, "dtype", None)
    expected_dtype = getattr(mx, expected, None)
    if expected_dtype is None or (actual_dtype != expected_dtype and str(actual_dtype) != str(expected_dtype)):
        raise PilotError(f"provider {label} must have dtype {expected}")
    return value


class _StepObservingProvider:
    def __init__(self, delegate: Any, phase: str, mx_module: Any = None,
                 tree_flatten: Any = None, expected_schema: list[dict[str, Any]] | None = None) -> None:
        self.delegate, self.phase, self.mx = delegate, phase, mx_module
        self.tree_flatten = tree_flatten
        self.records: list[dict[str, Any]] = []
        self.expected_schema = expected_schema
        if self.expected_schema is not None and not self.expected_schema:
            raise PilotError("model trainable schema must not be empty")

    def __call__(self, model: Any, *batch: Any) -> Any:
        started = time.perf_counter()
        result = self.delegate(model, *batch)
        elapsed = time.perf_counter() - started
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise PilotError("provider must return ((loss, token_count), gradients)")
        loss_tokens, gradients = result
        if not isinstance(loss_tokens, (tuple, list)) or len(loss_tokens) != 2:
            raise PilotError("provider must return ((loss, token_count), gradients)")
        loss, n_tokens = loss_tokens
        loss = _require_scalar_dtype(loss, self.mx, "float32", "loss")
        n_tokens = _require_scalar_dtype(n_tokens, self.mx, "int32", "token count")
        loss_value, token_value = _scalar(loss), _scalar(n_tokens)
        try:
            token_value = int(token_value)
        except (TypeError, ValueError) as exc:
            raise PilotError("provider token count is not an integer") from exc
        if self.tree_flatten is None:
            raise PilotError("provider gradient flattening semantics unavailable")
        leaves = [_array_meta(path, leaf) for path, leaf in _pinned_flatten(gradients, self.tree_flatten)]
        schema = [{"path": item.path, "shape": list(item.shape), "dtype": item.dtype} for item in leaves]
        if not schema:
            raise PilotError("provider gradient schema must not be empty")
        if self.expected_schema is None:
            self.expected_schema = schema
        if schema != self.expected_schema:
            raise PilotError("provider gradient schema drift")
        gradient_leaves = [leaf for _, leaf in _pinned_flatten(gradients, self.tree_flatten)]
        finite = _finite(loss, self.mx) and all(_finite(leaf, self.mx) for leaf in gradient_leaves)
        expected_tokens = _expected_mask_tokens(batch)
        if not isinstance(loss_value, (int, float)) or not math.isfinite(float(loss_value)) or token_value <= 0 or not finite:
            raise PilotError("provider returned nonfinite loss/gradient or nonpositive token count")
        if expected_tokens is None or token_value != expected_tokens:
            raise PilotError("provider token count does not match expected mask")
        local = len(self.records) + 1
        self.records.append({"phase": self.phase, "provider_call": local, "local_step": local,
                             "global_step": global_steps(self.phase, [local])[0], "loss": float(loss_value),
                             "loss_dtype": str(getattr(loss, "dtype", "")),
                             "token_dtype": str(getattr(n_tokens, "dtype", "")), "n_tokens": token_value,
                             "expected_mask_tokens": expected_tokens, "mask_tokens_match": True,
                             "gradient_schema": schema, "gradient_leaf_count": len(schema),
                             "gradient_paths": [item["path"] for item in schema],
                             "gradient_shapes": {item["path"]: item["shape"] for item in schema},
                             "gradient_dtypes": {item["path"]: item["dtype"] for item in schema},
                             "gradients_finite": True, "provider_elapsed_seconds": elapsed})
        return result


class _PilotTrainingCallback:
    def __init__(self, phase: str) -> None:
        self.phase, self.records, self.validation_records = phase, [], []

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        if not isinstance(val_info, dict):
            raise PilotError("validation callback requires one dict")
        loss = _scalar(val_info.get("val_loss"))
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise PilotError("validation callback reported nonfinite loss")
        self.validation_records.append({"iteration": int(val_info.get("iteration", -1)), "val_loss": float(loss), "val_time": float(val_info.get("val_time", 0.0))})

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        if not isinstance(train_info, dict):
            raise PilotError("training callback requires one dict")
        loss = _scalar(train_info.get("train_loss"))
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise PilotError("training callback reported nonfinite loss")
        iteration = int(train_info.get("iteration", -1))
        if iteration < 1:
            raise PilotError("training callback iteration invalid")
        speed = float(train_info.get("iterations_per_second", 0.0))
        self.records.append({"phase": self.phase, "local_step": iteration, "global_step": global_steps(self.phase, [iteration])[0],
                             "loss": float(loss), "learning_rate": float(_scalar(train_info.get("learning_rate", 0.0))),
                             "tokens_per_second": float(_scalar(train_info.get("tokens_per_second", 0.0))),
                             "iterations_per_second": speed, "train_step_wall_seconds": 1.0 / speed if speed > 0 else None,
                             "optimizer_update_ordinal": iteration})


def _immutable_identity(identity: dict[str, Any] | Any) -> Any:
    return identity.get("immutable", identity) if isinstance(identity, dict) else identity


def compare_immutable_identity(previous: dict[str, Any], current: dict[str, Any]) -> None:
    if _immutable_identity(previous) != _immutable_identity(current):
        raise PilotError("Phase B immutable identity mismatch before model loading")


def contract_digest(phase: str, effective: dict[str, Any], identity: dict[str, Any] | None = None) -> str:
    identity_value = _immutable_identity(identity or {})
    payload = {"phase": phase, "common": COMMON_VALUES, "phase_spec": PHASES[phase],
               "effective": effective, "identity": identity_value}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_training_api() -> dict[str, Any]:
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx_lm.tuner.datasets import CacheDataset, load_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx.utils import tree_flatten
        from ds4_ft_mlx.segmented_loss_and_grad import make_ds4_segmented_loss_and_grad
    except ImportError as exc:
        raise PilotError(f"MLX pilot dependencies unavailable: {exc}") from exc
    return {"mx": mx, "optim": optim, "load": load, "load_dataset": load_dataset, "CacheDataset": CacheDataset,
            "TrainingArgs": TrainingArgs, "train": train, "linear_to_lora_layers": linear_to_lora_layers,
            "tree_flatten": tree_flatten, "make_provider": make_ds4_segmented_loss_and_grad}


def _api_get(api: Any, key: str) -> Any:
    return api[key] if isinstance(api, dict) else getattr(api, key)


def _apply_phase_seeds(mx: Any, seed: int) -> list[str]:
    try:
        import numpy as np
        np.random.seed(seed)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise PilotError(f"NumPy seed application failed: {exc}") from exc
    random = getattr(mx, "random", None)
    if random is None or not hasattr(random, "seed"):
        raise PilotError("MLX random.seed unavailable")
    random.seed(seed)
    return ["numpy.random.seed", "mx.random.seed"]


def _save_trainable(api: Any, model: Any, path: pathlib.Path) -> list[dict[str, Any]]:
    mx = _api_get(api, "mx")
    flat = _pinned_flatten(model.trainable_parameters(), _api_get(api, "tree_flatten"))
    values = dict(flat)
    schema = _schema_from_tree(values, _api_get(api, "tree_flatten"))
    if hasattr(mx, "eval"):
        mx.eval(*values.values())
    if hasattr(mx, "save_safetensors"):
        mx.save_safetensors(str(path), values)
    else:
        raise PilotError("MLX save_safetensors unavailable")
    return schema


def _execute_training(args: argparse.Namespace, phase: dict[str, Any], api: Any, output: pathlib.Path,
                      provider: _StepObservingProvider, callback: _PilotTrainingCallback,
                      config: dict[str, Any] | None = None) -> None:
    """Execute the exact pinned vendor API seam; never call vendor source with old arguments."""
    config = config or {}
    seed_order = _apply_phase_seeds(_api_get(api, "mx"), args.seed)
    model, tokenizer = _api_get(api, "load")(args.model, tokenizer_config={"trust_remote_code": False}, trust_remote_code=False)
    train_set, valid_set, _ = _api_get(api, "load_dataset")(args, tokenizer)
    model.freeze()
    lora_values = config.get("lora_parameters", config)
    _api_get(api, "linear_to_lora_layers")(model, args.num_layers, config=lora_values)
    trainable_schema = _schema_from_tree(model.trainable_parameters(), _api_get(api, "tree_flatten"))
    provider.expected_schema = trainable_schema
    if args.phase == "phase-b":
        model.load_weights(args.resume_adapter_file, strict=False)
    start_name = "resume-start.safetensors" if args.phase == "phase-b" else "phase-a-start.safetensors"
    start_path = output / start_name
    saved_schema = _save_trainable(api, model, start_path)
    if saved_schema != trainable_schema:
        raise PilotError("trainable schema changed before training")
    if args.phase == "phase-b":
        phase_start = pathlib.Path(PHASES["phase-a"]["adapter_path"]) / "phase-a-start.safetensors"
        if not phase_start.is_file():
            raise PilotError("Phase A start checkpoint missing for resume proof")
        verify_resume_continuity(args.resume_adapter_file, start_path, phase_start)
    make_provider = api.get("make_provider") if isinstance(api, dict) else getattr(api, "make_provider", None)
    if make_provider is not None:
        provider.delegate = make_provider(segment_size=1)
    optimizer_factory = api.get("make_optimizer") if isinstance(api, dict) else getattr(api, "make_optimizer", None)
    optimizer = optimizer_factory(args.learning_rate) if optimizer_factory else _api_get(api, "optim").Adam(learning_rate=args.learning_rate)
    adapter_file = output / "adapters.safetensors"
    training_args = _api_get(api, "TrainingArgs")(
        batch_size=1, iters=phase["iters"], val_batches=25, steps_per_report=1,
        steps_per_eval=phase["steps_per_eval"], steps_per_save=1, adapter_file=str(adapter_file),
        max_seq_length=4096, grad_checkpoint=True, grad_accumulation_steps=1)
    atomic_write_json(output / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": lora_values, "seed": 0, "seed_order": seed_order, "phase": args.phase})
    cache_dataset = api.get("CacheDataset") if isinstance(api, dict) else getattr(api, "CacheDataset", None)
    train_dataset = cache_dataset(train_set) if cache_dataset else train_set
    val_dataset = cache_dataset(valid_set) if cache_dataset else valid_set
    train_fn = _api_get(api, "train")
    train_fn(model, optimizer, train_dataset, val_dataset, args=training_args,
             loss_and_grad=provider, training_callback=callback)


def _artifact_info(path: pathlib.Path) -> dict[str, Any]:
    return canonical_tensor_digest(path)


def _validate_artifacts(phase_name: str, output: pathlib.Path) -> dict[str, Any]:
    phase = PHASES[phase_name]
    start = output / ("phase-a-start.safetensors" if phase_name == "phase-a" else "resume-start.safetensors")
    checkpoints = [output / f"{step:07d}_adapters.safetensors" for step in range(1, phase["iters"] + 1)]
    final = output / "adapters.safetensors"
    for path in [start, *checkpoints, final, output / "adapter_config.json"]:
        if not path.is_file():
            raise PilotError(f"missing required pilot artifact: {path}")
    numbered = {path.name for path in output.glob("*_adapters.safetensors")}
    expected_numbered = {path.name for path in checkpoints}
    if numbered != expected_numbered:
        raise PilotError("checkpoint cardinality or unexpected numbered checkpoint")
    infos = {"start": _artifact_info(start), "checkpoints": [_artifact_info(path) for path in checkpoints], "final": _artifact_info(final)}
    all_infos = [infos["start"], *infos["checkpoints"], infos["final"]]
    schema = all_infos[0]["tensors"]
    if any(item["tensors"] != schema for item in all_infos[1:]):
        raise PilotError("checkpoint trainable schema drift")
    digests = [item["canonical_tensor_digest_v1"] for item in infos["checkpoints"]]
    if phase_name == "phase-a":
        if infos["start"]["canonical_tensor_digest_v1"] == digests[0] or digests[0] == digests[1] or digests[1] != infos["final"]["canonical_tensor_digest_v1"]:
            raise PilotError("Phase A checkpoint progression proof failed")
    else:
        if infos["start"]["canonical_tensor_digest_v1"] == infos["final"]["canonical_tensor_digest_v1"] or digests[0] != infos["final"]["canonical_tensor_digest_v1"]:
            raise PilotError("Phase B checkpoint progression proof failed")
    return infos


def _marker_path(phase: str, status: str) -> pathlib.Path:
    return pathlib.Path(PILOT_WORKSPACE) / f".ds4-segmented-pilot-{phase}-{status}"


def _all_attempt_paths(phase: str) -> list[pathlib.Path]:
    paths = [PILOT_FINAL_REPORT, pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-ok", pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-fail"]
    if phase == "phase-a":
        paths.extend(REPO_ROOT / spec["report"] for spec in PHASES.values())
        paths.extend(_marker_path(name, status) for name in PHASES for status in ("ok", "fail"))
    else:
        paths.extend([REPO_ROOT / PHASES[phase]["report"], _marker_path(phase, "ok"), _marker_path(phase, "fail")])
    return paths


def _check_attempt_gates(phase: str) -> None:
    for path in _all_attempt_paths(phase):
        ensure_absent(path)
    ensure_absent(PHASES[phase]["adapter_path"])


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError(f"identity command failed: {' '.join(args)}: {exc}") from exc


def _resource_gate() -> dict[str, Any]:
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
        rss = [(int(proc.pid), int(proc.memory_info().rss)) for proc in psutil.process_iter(["pid", "memory_info"]) if proc.pid != os.getpid()]
    except (ImportError, OSError) as exc:
        raise PilotError(f"resource observer unavailable: {exc}") from exc
    competing = [item for item in rss if item[1] > 50 * 1024**3]
    if competing:
        raise PilotError(f"competing process exceeds RSS gate: {competing}")
    if available < PILOT_MEMORY_HEADROOM:
        raise PilotError("memory headroom below 32 GiB")
    free = shutil_disk_free(PILOT_WORKSPACE)
    if free < PILOT_DISK_MIN_FREE:
        raise PilotError("disk free below 1 GiB")
    return {"available_memory": available, "disk_free": free, "competing_processes": rss}


def shutil_disk_free(path: str | pathlib.Path) -> int:
    import shutil
    try:
        return int(shutil.disk_usage(path).free)
    except OSError as exc:
        raise PilotError(f"disk resource check failed: {path}: {exc}") from exc


def _file_manifest(root: pathlib.Path) -> list[dict[str, Any]]:
    if root.is_file():
        return [{"path": str(root), "size": root.stat().st_size, "sha256": file_sha256(root)}]
    if not root.is_dir():
        raise PilotError(f"identity path missing: {root}")
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": file_sha256(path)})
    if not entries:
        raise PilotError(f"identity directory is empty: {root}")
    return entries


def _validate_provenance_splits(provenance: pathlib.Path, dataset_manifest: list[dict[str, Any]]) -> dict[str, str]:
    try:
        value = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"dataset provenance read failed: {provenance}: {exc}") from exc
    splits = value.get("splits") if isinstance(value, dict) else None
    if not isinstance(splits, dict):
        raise PilotError("dataset provenance split hashes missing")
    actual = {item["path"]: item["sha256"] for item in dataset_manifest}
    result: dict[str, str] = {}
    for name, entry in splits.items():
        expected = entry.get("filtered_sha256") if isinstance(entry, dict) else None
        if expected is None:
            continue
        if actual.get(name) != expected:
            raise PilotError(f"dataset provenance split hash mismatch: {name}")
        result[name] = expected
    required = {"train.jsonl", "valid.jsonl", "test.jsonl"}
    if set(result) != required:
        raise PilotError("dataset provenance must bind train/valid/test split hashes")
    return result


def _runtime_preflight(args: argparse.Namespace, phase: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if sys.executable != PILOT_INTERPRETER:
        raise PilotError(f"interpreter identity mismatch: {sys.executable}")
    if not pathlib.Path(PILOT_MODEL).is_dir() or not pathlib.Path(PILOT_DATA).is_dir() or not pathlib.Path(PILOT_CONFIG).is_file():
        raise PilotError("pinned model, dataset, or config identity unavailable")
    provider_path = REPO_ROOT / "python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py"
    smoke_path = REPO_ROOT / "scripts/ds4_segmented_smoke.py"
    if file_sha256(provider_path) != PILOT_PROVIDER_SHA256 or file_sha256(smoke_path) != PILOT_SMOKE_SHA256:
        raise PilotError("frozen provider or smoke identity mismatch")
    vendor_head = _git_output("git", "-C", "vendor/mlx-lm", "rev-parse", "HEAD")
    if vendor_head != PILOT_VENDOR_SHA:
        raise PilotError(f"vendor identity mismatch: {vendor_head}")
    vendor_gitlink = _git_output("git", "ls-files", "--stage", "--", "vendor/mlx-lm")
    expected_gitlink = f"160000 {PILOT_VENDOR_SHA} 0"
    if not vendor_gitlink.startswith(expected_gitlink) or not vendor_gitlink.endswith("vendor/mlx-lm"):
        raise PilotError("outer vendor gitlink identity mismatch")
    vendor_status = _git_output("git", "-C", "vendor/mlx-lm", "status", "--porcelain")
    if vendor_status:
        raise PilotError("vendor checkout is not clean")
    lora = _validate_lora_config(config)
    resources = _resource_gate()
    try:
        import mlx
        import mlx_lm
        mlx_version = getattr(mlx, "__version__", None)
        if mlx_version != "0.31.2":
            raise PilotError(f"MLX version mismatch: {mlx_version}")
        mlx_lm_path = pathlib.Path(mlx_lm.__file__).resolve()
        vendor_root = (REPO_ROOT / "vendor/mlx-lm").resolve()
        try:
            mlx_lm_path.relative_to(vendor_root)
        except ValueError as exc:
            raise PilotError(f"resolved mlx_lm path is outside pinned vendor: {mlx_lm_path}") from exc
    except ImportError as exc:
        raise PilotError(f"runtime source identity unavailable: {exc}") from exc
    provenance = REPO_ROOT / PILOT_PROVENANCE
    if not provenance.is_file():
        raise PilotError(f"dataset provenance missing: {provenance}")
    model_manifest = _file_manifest(pathlib.Path(PILOT_MODEL))
    dataset_manifest = _file_manifest(pathlib.Path(PILOT_DATA))
    split_hashes = _validate_provenance_splits(provenance, dataset_manifest)
    immutable = {"interpreter": sys.executable, "python_version": list(sys.version_info[:3]),
                 "mlx_version": mlx_version, "mlx_lm_resolved_module": str(mlx_lm_path),
                 "vendor_head": vendor_head, "vendor_gitlink": vendor_gitlink,
                 "vendor_inner_clean": True, "provider_sha256": PILOT_PROVIDER_SHA256,
                 "smoke_sha256": PILOT_SMOKE_SHA256, "pilot_source_sha256": file_sha256(pathlib.Path(__file__)),
                 "config_sha256": file_sha256(PILOT_CONFIG), "provenance_sha256": file_sha256(provenance),
                 "provenance_split_hashes": split_hashes, "model_manifest": model_manifest,
                 "dataset_manifest": dataset_manifest, "lora_parameters": lora,
                 "git_head": _git_output("git", "rev-parse", "HEAD")}
    return {"immutable": immutable, "dynamic_resources": resources,
            "repository_status": _git_output("git", "status", "--short")}


def _phase_failure_report(phase_name: str, message: str, started: float, *, effective: dict[str, Any] | None = None,
                          identity: dict[str, Any] | None = None, contract: str | None = None,
                          output: pathlib.Path | None = None, lock_lifecycle: dict[str, Any] | None = None,
                          watchdog_cancelled: bool = False) -> dict[str, Any]:
    return {"status": "fail", "phase": phase_name, "error": message, "wall_seconds": time.monotonic() - started,
            "effective": effective or {}, "commands": (effective or {}).get("command", []),
            "identity_manifest": identity or {}, "contract_digest": contract or "",
            "output_path": str(output or PHASES[phase_name]["adapter_path"]),
            "lock_lifecycle": lock_lifecycle or {}, "watchdog_cancelled": watchdog_cancelled,
            "exit_code": 1, "retry": RETRY_POLICY, "fallback": FALLBACK_POLICY, "non_claims": list(NON_CLAIMS)}


def _ok_marker_paths() -> list[pathlib.Path]:
    workspace = pathlib.Path(PILOT_WORKSPACE)
    return [_marker_path(phase, "ok") for phase in PHASES] + [workspace / ".ds4-segmented-pilot-ok"]


def _remove_ok_markers() -> None:
    errors: list[Exception] = []
    for path in _ok_marker_paths():
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            # A cleanup injection or transient unlink failure must not leave an
            # OK pathname published. Rename it out of the evidence namespace.
            rollback = path.with_name(path.name + ".rollback")
            try:
                os.replace(path, rollback)
                try:
                    rollback.unlink()
                except OSError:
                    pass
            except OSError as rename_error:
                errors.append(rename_error)
    if errors:
        raise PilotError("cannot roll back success markers: " + "; ".join(str(exc) for exc in errors)) from errors[0]


def _marker_fields(phase: str, status: str, report_path: pathlib.Path, report: dict[str, Any]) -> dict[str, Any]:
    return {"phase": phase, "status": status, "report_path": str(report_path),
            "report_sha256": file_sha256(report_path), "output_path": report.get("output_path", PHASES[phase]["adapter_path"]),
            "timestamp": time.time(), "exit_code": 0 if status == "ok" else 1,
            "contract_digest": report.get("contract_digest", "")}


def _write_failure_evidence(phase: str, report: dict[str, Any]) -> None:
    # Roll back published success evidence before replacing any report it names.
    _remove_ok_markers()
    report_path = REPO_ROOT / PHASES[phase]["report"]
    atomic_write_json(report_path, report)
    final = dict(report)
    final["phase_report"] = str(report_path)
    atomic_write_json(PILOT_FINAL_REPORT, final)
    errors: list[Exception] = []
    for marker_path, marker in (
        (_marker_path(phase, "fail"), _marker_fields(phase, "fail", report_path, report)),
        (pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-fail",
         _marker_fields(phase, "fail", PILOT_FINAL_REPORT, final)),
    ):
        try:
            atomic_write_json(marker_path, marker)
        except Exception as exc:
            errors.append(exc)
    if errors:
        _remove_ok_markers()
        raise PilotError("failure evidence write failed: " + "; ".join(str(exc) for exc in errors)) from errors[0]


def _write_success_evidence(phase: str, report: dict[str, Any]) -> None:
    try:
        report_path = REPO_ROOT / PHASES[phase]["report"]
        atomic_write_json(report_path, report)
        marker = _marker_fields(phase, "ok", report_path, report)
        if phase == "phase-a":
            source = report.get("resume_source")
            if source:
                marker["resume_source"] = source["path"]
                marker["resume_source_file_sha256"] = source["file_sha256"]
                marker["resume_source_canonical_tensor_digest_v1"] = source["canonical_tensor_digest_v1"]
        else:
            phase_a = json.loads((REPO_ROOT / PHASES["phase-a"]["report"]).read_text(encoding="utf-8"))
            final = aggregate_reports(phase_a, report)
            atomic_write_json(PILOT_FINAL_REPORT, final)
        atomic_write_json(_marker_path(phase, "ok"), marker)
        if phase == "phase-b":
            atomic_write_json(pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-ok",
                              _marker_fields(phase, "ok", PILOT_FINAL_REPORT, final))
    except Exception:
        # Multi-file publication cannot be a filesystem transaction. Remove all
        # success markers before the caller writes bound failure evidence.
        _remove_ok_markers()
        raise


def _expected_validation_steps(phase: dict[str, Any]) -> list[int]:
    return [iteration - 1 for iteration in range(1, phase["iters"] + 1)
            if iteration == 1 or iteration % phase["steps_per_eval"] == 0 or iteration == phase["iters"]]


def run_phase(args: argparse.Namespace, *, api: Any = None, preflight: Any = None) -> dict[str, Any]:
    phase_name, phase = args.phase, PHASES[args.phase]
    started = time.monotonic()
    watchdog: Watchdog | None = None
    watchdog_cancelled = False
    lock_acquired = False
    lock_release_attempts = 0
    workspace = pathlib.Path(PILOT_WORKSPACE)
    lock_lifecycle = {"path": str(_ft_lock_path(workspace)), "acquired": False, "released": False, "release_attempts": 0}
    report: dict[str, Any] | None = None
    error: Exception | None = None
    attempt_gate_rejected = False
    effective: dict[str, Any] = {}
    identity: dict[str, Any] = {}
    contract = ""
    output = pathlib.Path(phase["adapter_path"])
    try:
        # Occupied evidence is a terminal one-attempt record. Check it before
        # any watchdog, config, or identity work can fail and overwrite it.
        try:
            _check_attempt_gates(phase_name)
        except Exception:
            attempt_gate_rejected = True
            raise
        watchdog = _install_timeout_watchdog(phase["timeout"])
        config = _read_config(args.config)
        effective = validate_pins(args, phase, config)
        contract = contract_digest(phase_name, effective)
        _acquire_ft_lock(workspace)
        lock_acquired = True
        lock_lifecycle["acquired"] = True
        try:
            _check_attempt_gates(phase_name)
        except Exception:
            attempt_gate_rejected = True
            raise
        phase_a_report: dict[str, Any] | None = None
        if phase_name == "phase-b":
            phase_a_report_path = REPO_ROOT / PHASES["phase-a"]["report"]
            if not phase_a_report_path.is_file():
                raise PilotError("Phase A report missing")
            phase_a_report = json.loads(phase_a_report_path.read_text(encoding="utf-8"))
            marker_path = _marker_path("phase-a", "ok")
            if not marker_path.is_file():
                raise PilotError("Phase A OK marker missing")
            validate_phase_b_dependency(phase_a_report, json.loads(marker_path.read_text(encoding="utf-8")), phase["resume_adapter_file"])
        identity = (preflight or _runtime_preflight)(args, phase, config)
        if phase_name == "phase-b" and phase_a_report is not None:
            compare_immutable_identity(phase_a_report.get("identity_manifest", {}), identity)
        contract = contract_digest(phase_name, effective, identity)
        output.mkdir(parents=True)
        runtime_api = api or _load_training_api()
        tree_flatten = _api_get(runtime_api, "tree_flatten")
        provider = _StepObservingProvider(lambda *_: (_ for _ in ()).throw(PilotError("provider not initialized")), phase_name,
                                          _api_get(runtime_api, "mx"), tree_flatten=tree_flatten)
        callback = _PilotTrainingCallback(phase_name)
        _execute_training(args, phase, runtime_api, output, provider, callback, config=config)
        artifacts = _validate_artifacts(phase_name, output)
        validate_step_evidence(phase_name, provider.records, phase["iters"])
        callback_steps = [item["local_step"] for item in callback.records]
        if callback_steps != list(range(1, phase["iters"] + 1)):
            raise PilotError("completed-update callback cardinality/order mismatch")
        validation_steps = [item["iteration"] for item in callback.validation_records]
        if validation_steps != _expected_validation_steps(phase):
            raise PilotError("validation callback cardinality/order mismatch")
        if len(callback.records) != len(provider.records):
            raise PilotError("provider/callback cardinality mismatch")
        for index, (provider_record, callback_record) in enumerate(zip(provider.records, callback.records)):
            if provider_record["local_step"] != callback_record["local_step"]:
                raise PilotError("provider/callback step pairing mismatch")
            callback_record["provider_call"] = provider_record["provider_call"]
            callback_record["checkpoint"] = artifacts["checkpoints"][index]
        report = {"status": "ok", "phase": phase_name, "effective": effective, "contract_digest": contract,
                  "identity_manifest": identity, "provider_calls": len(provider.records), "optimizer_updates": len(callback.records),
                  "steps": callback.records, "provider_evidence": provider.records, "validation_evidence": callback.validation_records,
                  "artifacts": artifacts, "output_path": str(output), "commands": effective.get("command", []),
                  "wall_seconds": time.monotonic() - started, "retry": RETRY_POLICY, "fallback": FALLBACK_POLICY,
                  "non_claims": list(NON_CLAIMS), "global_mapping": [0] + global_steps(phase_name, callback_steps)}
        if phase_name == "phase-a":
            report["resume_source"] = artifacts["checkpoints"][-1]
        else:
            source_info = canonical_tensor_digest(phase["resume_adapter_file"])
            if artifacts["start"]["canonical_tensor_digest_v1"] != source_info["canonical_tensor_digest_v1"]:
                raise PilotError("Phase B resume-start does not equal Phase A source")
            report["resume_source"] = source_info
        if report["wall_seconds"] > phase["timeout"]:
            raise PilotError("phase exceeded hard timeout")
    except Exception as exc:
        error = exc
    finally:
        if watchdog is not None:
            try:
                watchdog.cancel()
                watchdog_cancelled = True
            except Exception as exc:
                cancellation_error = PilotError(f"watchdog cancellation failed: {exc}")
                error = cancellation_error if error is None else PilotError(f"{error}; {cancellation_error}")
        partial_lock_owned = _LOCK_OWNED_PARTIAL
        if lock_acquired or _LOCK_OWNED_PATH is not None:
            lock_release_attempts += 1
            lock_lifecycle["release_attempts"] = lock_release_attempts
            try:
                _release_ft_lock()
                lock_lifecycle["released"] = True
            except Exception as exc:
                release_error = PilotError(f"lock release failed: {exc}")
                error = release_error if error is None else PilotError(f"{error}; {release_error}")
                if partial_lock_owned:
                    try:
                        quarantine = _quarantine_ft_lock()
                        lock_lifecycle["quarantined"] = True
                        if quarantine is not None:
                            lock_lifecycle["quarantine_path"] = str(quarantine)
                    except Exception as quarantine_error:
                        error = PilotError(f"{error}; partial lock quarantine failed: {quarantine_error}")
    lock_lifecycle["release_attempts"] = lock_release_attempts
    if error is not None:
        failed = _phase_failure_report(phase_name, str(error), started, effective=effective, identity=identity,
                                       contract=contract, output=output, lock_lifecycle=lock_lifecycle,
                                       watchdog_cancelled=watchdog_cancelled)
        if attempt_gate_rejected:
            # Existing one-attempt evidence is itself the terminal record. Never
            # delete or overwrite it with a second invocation's failure report.
            return failed
        try:
            _write_failure_evidence(phase_name, failed)
        except Exception as evidence_error:
            failed["cleanup_warning"] = f"failure evidence write failed: {evidence_error}"
        return failed
    assert report is not None
    report["lock_lifecycle"] = lock_lifecycle
    report["watchdog_cancelled"] = watchdog_cancelled
    try:
        _write_success_evidence(phase_name, report)
    except Exception as exc:
        failed = _phase_failure_report(phase_name, f"report/marker write failed: {exc}", started, effective=effective,
                                       identity=identity, contract=contract, output=output,
                                       lock_lifecycle=lock_lifecycle, watchdog_cancelled=watchdog_cancelled)
        try:
            _write_failure_evidence(phase_name, failed)
        except Exception as evidence_error:
            failed["cleanup_warning"] = f"failure evidence write failed: {evidence_error}"
        return failed
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_phase(args)
    if report.get("status") != "ok":
        print(f"pilot failed closed: {report.get('error', 'unknown failure')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
