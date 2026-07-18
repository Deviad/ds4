#!/usr/bin/env python3
"""Fail-closed, synthetic-first two-phase DS4 checkpoint/resume pilot."""
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CANONICAL_REPO_ROOT = REPO_ROOT
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

ATTEMPT1_EVIDENCE_SHA256: tuple[dict[str, Any], ...] = (
    {"path": "agent-output/cmux-14-5/phase-a-log.txt", "size": 29,
     "sha256": "4ac7319d81f9dcd344a185fd57fb3a6342ef981cbd7ca378143d1ea803eb8865"},
    {"path": "agent-output/cmux-14-5/phase-a-report.json", "size": 3771,
     "sha256": "497e5a271df5765e3af7f9795b27961e2b383886aada100d626a5e85f1fc49b1"},
    {"path": "agent-output/cmux-14-5/pilot-report.json", "size": 3874,
     "sha256": "13c8429d8ceaf11dfed24faabb9dcaa2bb9022c02882d055ba7ec0d1b4cc0b02"},
    {"path": "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail", "size": 458,
     "sha256": "e19872f0d9c1d7fe90fd16aa94bef750c7cd5b9e58e608fc5ff650141ec75f54"},
    {"path": "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail", "size": 456,
     "sha256": "201089f5ede43a0fe839325a6feae1347ed6d3c54d020a0e498598628d1677d8"},
)
# This private snapshot prevents a substituted in-memory manifest from becoming
# valid B2 evidence while keeping the manifest itself part of the attempt-3 digest.
_ATTEMPT1_EVIDENCE_SHA256_CANONICAL = ATTEMPT1_EVIDENCE_SHA256
ATTEMPT1_RESERVED_PATHS: tuple[str, ...] = (
    "agent-output/cmux-14-5/phase-b-log.txt",
    "agent-output/cmux-14-5/phase-b-report.json",
    "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a",
    "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-ok",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-ok",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-fail",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-ok",
)
ATTEMPT3_NAMESPACE = "ds4-segmented-pilot-attempt-3"
_ATTEMPT2_HISTORICAL_NAMESPACE = "ds4-segmented-pilot-attempt-2"
_ATTEMPT2_HISTORICAL_PRELOG = {
    "path": "agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md",
    "size": 990,
    "sha256": "cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41",
}
_ATTEMPT2_HISTORICAL_MANIFEST = {
    "path": "agent-output/cmux-14-5-attempt-2/phase-a2-runtime-failure-manifest.json",
    "size": 3123,
    "sha256": "2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88",
}
_ATTEMPT2_HISTORICAL_FILES = (
    ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-log.txt", 1432, "edf1ae2215083aab2b8648403cf53f395ce39c3da2277f80fa7da2e87976734e"),
    ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-report.json", 80454, "d9cbc1895b62f4f182a25dcf58d851d077c6f92745bd3b0d387c78fa6dc8a500"),
    ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/pilot-report.json", 80567, "84e8d26cf658f4ba15eda8346b4e507dc331d65da29b00b21aca1073eb97c0ce"),
    ("/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail", 542, "079c89c88bf87699562ce8e5866d88ba3e92adec6d2b0c787a481778d85b2918"),
    ("/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail", 539, "91b4dd994a3bd0319c5d0edf9d0c49f933cc95f2e49a1b1f359e4a92e609edfe"),
    ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000001_adapters.safetensors", 22292892, "89363e82325e095bc74f7cdb6e022280438fa1667d52abcf33f1dfb3db1a2d18"),
    ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors", 22292892, "2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652"),
    ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapter_config.json", 339, "13620f30b39a0de718f0769830b4b0a2233eaebea62d6fc43da08231d0a0691d"),
    ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapters.safetensors", 22292892, "2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652"),
    ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/phase-a-start.safetensors", 22292892, "c6dd9f67be04936c4dce58d7b3643403354516d6167e4e08d76c393890339e51"),
)

# This snapshot is deliberately independent from the mutable lookup tuple above.
# It prevents coordinated manifest/path/hash substitution from becoming history.
_ATTEMPT2_HISTORICAL_EXPECTED = {
    "attempt": 2, "command_sha256": "38e6b1aa11d28c41c126fda6dab55cf68a9eb1d9a71fc177eec8875a3114b2de",
    "error": "safetensors __metadata__ must be an object", "exit_code": 1,
    "lock_absent_after_exit": True,
    "lock_lifecycle": {"acquired": True, "path": "/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock", "release_attempts": 1, "released": True},
    "namespace": _ATTEMPT2_HISTORICAL_NAMESPACE, "ok_markers_absent": True, "retry_performed": False,
    "revision": "e6d34fa03479316720430f35cc94d4606a45ef96", "status": "fail", "story": "14.5b",
    "training_log_observations": {"checkpoint1_saved": True, "checkpoint2_saved": True,
                                   "iter1_loss": "19.334", "iter1_val_loss": "19.553",
                                   "iter2_loss": "17.648", "iter2_val_loss": "19.841"},
    "wall_seconds": 2444.2739184170496, "watchdog_cancelled": True,
    "files": [{"path": path, "size": size, "sha256": sha} for path, size, sha in (
        ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-log.txt", 1432, "edf1ae2215083aab2b8648403cf53f395ce39c3da2277f80fa7da2e87976734e"),
        ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-report.json", 80454, "d9cbc1895b62f4f182a25dcf58d851d077c6f92745bd3b0d387c78fa6dc8a500"),
        ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/pilot-report.json", 80567, "84e8d26cf658f4ba15eda8346b4e507dc331d65da29b00b21aca1073eb97c0ce"),
        ("/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail", 542, "079c89c88bf87699562ce8e5866d88ba3e92adec6d2b0c787a481778d85b2918"),
        ("/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail", 539, "91b4dd994a3bd0319c5d0edf9d0c49f933cc95f2e49a1b1f359e4a92e609edfe"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000001_adapters.safetensors", 22292892, "89363e82325e095bc74f7cdb6e022280438fa1667d52abcf33f1dfb3db1a2d18"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors", 22292892, "2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapter_config.json", 339, "13620f30b39a0de718f0769830b4b0a2233eaebea62d6fc43da08231d0a0691d"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapters.safetensors", 22292892, "2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/phase-a-start.safetensors", 22292892, "c6dd9f67be04936c4dce58d7b3643403354516d6167e4e08d76c393890339e51"),
    )],
}
_ATTEMPT2_HISTORICAL_FACTS = {
    "revision": _ATTEMPT2_HISTORICAL_EXPECTED["revision"], "command_sha256": _ATTEMPT2_HISTORICAL_EXPECTED["command_sha256"],
    "namespace": _ATTEMPT2_HISTORICAL_NAMESPACE, "exit_code": 1, "retry_performed": False,
    "lock_lifecycle": {"acquired": True, "released": True, "release_attempts": 1},
    "lock_absent_after_exit": True, "watchdog_cancelled": True, "ok_markers_absent": True,
}
_ATTEMPT2_HISTORICAL_ABSENT_PATHS = (
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-ok",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-ok",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-fail",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-ok",
    "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b",
    "agent-output/cmux-14-5-attempt-2/phase-b-log.txt",
    "agent-output/cmux-14-5-attempt-2/phase-b-report.json",
    "/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock",
)


def _freeze_history(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_history(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_history(item) for item in value)
    return value


def _thaw_history(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw_history(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_history(item) for item in value]
    return value


def _json_exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_json_exact_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_json_exact_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


# Import-time immutable literals are the only roots accepted by the historical verifier.
_ATTEMPT2_CANONICAL_PRELOG = ("agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md", 990,
                              "cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41")
_ATTEMPT2_CANONICAL_MANIFEST = ("agent-output/cmux-14-5-attempt-2/phase-a2-runtime-failure-manifest.json", 3123,
                               "2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88")
_ATTEMPT2_CANONICAL_FILES = tuple(_ATTEMPT2_HISTORICAL_FILES)
# Independent verifier root: coordinated path/hash/target substitution must not
# become valid by replacing one mutable lookup tuple.
_ATTEMPT2_VERIFIER_FILES = tuple(tuple(item) for item in _ATTEMPT2_HISTORICAL_FILES)
_ATTEMPT2_CANONICAL_EXPECTED = _freeze_history(json.loads(json.dumps(_ATTEMPT2_HISTORICAL_EXPECTED)))
_ATTEMPT2_CANONICAL_FACTS = _freeze_history(json.loads(json.dumps(_ATTEMPT2_HISTORICAL_FACTS)))
_ATTEMPT2_CANONICAL_ABSENT_PATHS = tuple(_ATTEMPT2_HISTORICAL_ABSENT_PATHS)
NON_CLAIMS = ["optimizer_state_continuity", "rng_state_continuity", "dataset_cursor_continuity",
              "scheduler_continuity", "trainer_global_iteration", "convergence_or_quality",
              "throughput_improvement", "full-training-readiness"]


class PilotError(RuntimeError):
    """Terminal fail-closed pilot error."""


class ResourceObserverError(PilotError):
    """Structured fail-closed error from resource observation."""

    def __init__(self, stage: str, error: BaseException, pid: int | None = None) -> None:
        self.stage = stage
        self.exception_type = type(error).__name__
        self.pid = pid
        pid_text = "null" if pid is None else str(pid)
        super().__init__(f"resource observer failed: stage={stage} exception={self.exception_type} pid={pid_text}")


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


_METADATA_ABSENT = object()


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


def _read_json_bytes_snapshot(path: pathlib.Path | str) -> tuple[bytes, str, dict[str, Any]]:
    path = pathlib.Path(path)
    try:
        payload = path.read_bytes()
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as exc:
        raise PilotError(f"JSON snapshot is not readable: {path}") from exc
    if not isinstance(parsed, dict):
        raise PilotError(f"JSON snapshot must be an object: {path}")
    return payload, hashlib.sha256(payload).hexdigest(), parsed


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
            metadata = header.get("__metadata__", _METADATA_ABSENT)
            if metadata is not _METADATA_ABSENT and metadata is not None and not isinstance(metadata, dict):
                raise PilotError("safetensors __metadata__ must be an object or null")
            tensors: list[tuple[str, str, list[int], int, int]] = []
            spans: list[tuple[int, int, str]] = []
            for name, item in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(name, str) or not isinstance(item, dict):
                    raise PilotError(f"invalid tensor entry: {name}")
                if set(item) != {"dtype", "shape", "data_offsets"}:
                    raise PilotError(f"invalid tensor schema keys: {name}")
                dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
                if type(dtype) is not str or dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
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


def attempt3_launch_identity(*, repo_root: pathlib.Path | None = None) -> dict[str, str]:
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    return {
        "interpreter": PILOT_INTERPRETER,
        "model": PILOT_MODEL,
        "data": PILOT_DATA,
        "config": PILOT_CONFIG,
        "script": str(root / "scripts" / "ds4_segmented_pilot.py"),
    }


def attempt3_namespace(*, repo_root: pathlib.Path | None = None,
                       workspace: pathlib.Path | str | None = None) -> dict[str, pathlib.Path]:
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    work = pathlib.Path(workspace) if workspace is not None else pathlib.Path(PILOT_WORKSPACE)
    a = work / "adapters-segmented-pilot-attempt-3-phase-a"
    b = work / "adapters-segmented-pilot-attempt-3-phase-b"
    evidence = root / "agent-output" / "cmux-14-5-attempt-3"
    return {
        "phase-a-output": a,
        "phase-a-start-checkpoint": a / "phase-a-start.safetensors",
        "phase-a-step1-checkpoint": a / "0000001_adapters.safetensors",
        "phase-a-step2-checkpoint": a / "0000002_adapters.safetensors",
        "phase-a-final-checkpoint": a / "adapters.safetensors",
        "phase-a-config": a / "adapter_config.json",
        "phase-a-log": evidence / "phase-a-log.txt",
        "phase-a-report": evidence / "phase-a-report.json",
        "phase-a-ok": work / f".{ATTEMPT3_NAMESPACE}-phase-a-ok",
        "phase-a-fail": work / f".{ATTEMPT3_NAMESPACE}-phase-a-fail",
        "phase-b-output": b,
        "phase-b-start-checkpoint": b / "resume-start.safetensors",
        "phase-b-step1-checkpoint": b / "0000001_adapters.safetensors",
        "phase-b-final-checkpoint": b / "adapters.safetensors",
        "phase-b-config": b / "adapter_config.json",
        "phase-b-resume": a / "0000002_adapters.safetensors",
        "phase-b-log": evidence / "phase-b-log.txt",
        "phase-b-report": evidence / "phase-b-report.json",
        "phase-b-ok": work / f".{ATTEMPT3_NAMESPACE}-phase-b-ok",
        "phase-b-fail": work / f".{ATTEMPT3_NAMESPACE}-phase-b-fail",
        "final-report": evidence / "pilot-report.json",
        "final-ok": work / f".{ATTEMPT3_NAMESPACE}-ok",
        "final-fail": work / f".{ATTEMPT3_NAMESPACE}-fail",
    }


def attempt3_phase_specs(*, repo_root: pathlib.Path | None = None,
                         workspace: pathlib.Path | str | None = None) -> dict[str, dict[str, Any]]:
    paths = attempt3_namespace(repo_root=repo_root, workspace=workspace)
    namespace_paths = {key: str(value) for key, value in paths.items()}
    return {
        "phase-a": {"phase": "phase-a", "attempt": 3, "namespace": ATTEMPT3_NAMESPACE, "iters": 2,
                    "steps_per_eval": 2, "timeout": 2700, "global_offset": 0,
                    "adapter_path": str(paths["phase-a-output"]),
                    "resume_adapter_file": None, "report": str(paths["phase-a-report"]),
                    "phase_start_adapter_file": None, "final_report": str(paths["final-report"]),
                    "namespace_paths": namespace_paths},
        "phase-b": {"phase": "phase-b", "attempt": 3, "namespace": ATTEMPT3_NAMESPACE, "iters": 1,
                    "steps_per_eval": 1, "timeout": 1500, "global_offset": 2,
                    "adapter_path": str(paths["phase-b-output"]),
                    "resume_adapter_file": str(paths["phase-b-resume"]),
                    "report": str(paths["phase-b-report"]),
                    "phase_start_adapter_file": str(paths["phase-a-start-checkpoint"]),
                    "final_report": str(paths["final-report"]),
                    "namespace_paths": namespace_paths},
    }


def canonical_attempt3_command(phase: str, phase_spec: dict[str, Any]) -> list[str]:
    if phase not in ("phase-a", "phase-b") or phase_spec.get("attempt") != 3:
        raise PilotError("canonical attempt-3 command requires attempt-3 phase spec")
    paths = phase_spec["namespace_paths"]
    identity = attempt3_launch_identity()
    command = [identity["interpreter"], identity["script"],
               "--attempt", "3", "--phase", phase, "--log-path", paths[f"{phase}-log"],
               "--model", identity["model"], "--data", identity["data"],
               "--adapter-path", phase_spec["adapter_path"], "--config", identity["config"]]
    if phase == "phase-b":
        command.extend(["--resume-adapter-file", phase_spec["resume_adapter_file"]])
    command.extend(["--train", "--fine-tune-type", "lora", "--num-layers", "16",
                     "--iters", str(phase_spec["iters"]), "--batch-size", "1",
                     "--learning-rate", "1e-5", "--max-seq-length", "4096", "--mask-prompt",
                     "--grad-checkpoint", "--grad-accumulation-steps", "1", "--seed", "0",
                     "--optimizer", "adam", "--val-batches", "25", "--steps-per-report", "1",
                     "--steps-per-eval", str(phase_spec["steps_per_eval"]), "--save-every", "1",
                     "--segment-size", "1"])
    return command


_ATTEMPT3_AUTHORIZATION_KEYS = frozenset({
    "revision", "canonical_command_sha256", "pilot_source_sha256", "catalog_source_sha256",
    "protected_files_manifest_sha256", "attempt2_runtime_manifest_sha256",
})
_ATTEMPT3_PROTECTED_PATHS = (
    "scripts/ds4_segmented_smoke.py", "python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py",
    "ds4.c", "ds4.h", "ds4_agent.c", "ds4_bench.c", "ds4_cli.c", "ds4_cuda.cu",
    "ds4_distributed.c", "ds4_distributed.h", "ds4_eval.c", "ds4_gpu.h", "ds4_help.c",
    "ds4_help.h", "ds4_kvstore.c", "ds4_kvstore.h", "ds4_metal.m", "ds4_rocm.cu",
    "ds4_rocm.h", "ds4_server.c", "ds4_ssd.c", "ds4_ssd.h", "ds4_web.c", "ds4_web.h",
    "metal/argsort.metal", "metal/bin.metal", "metal/concat.metal", "metal/cpy.metal",
    "metal/dense.metal", "metal/dsv4_hc.metal", "metal/dsv4_kv.metal", "metal/dsv4_misc.metal",
    "metal/dsv4_rope.metal", "metal/flash_attn.metal", "metal/get_rows.metal", "metal/glu.metal",
    "metal/moe.metal", "metal/norm.metal", "metal/repeat.metal", "metal/set_rows.metal",
    "metal/softmax.metal", "metal/sum_rows.metal", "metal/unary.metal",
)


def protected_files_manifest_sha256(*, repo_root: pathlib.Path | None = None) -> str:
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    rows = []
    for relative in sorted(_ATTEMPT3_PROTECTED_PATHS):
        path = root / relative
        if not path.is_file():
            raise PilotError(f"protected path unavailable: {path}")
        rows.append(f"{file_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _authorization_json(value: str | dict[str, Any]) -> dict[str, str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as exc:
            raise PilotError("attempt-3 authorization JSON is invalid") from exc
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        if value != canonical:
            raise PilotError("attempt-3 authorization JSON must be canonical compact JSON")
    else:
        parsed = value
    if not isinstance(parsed, dict) or set(parsed) != set(_ATTEMPT3_AUTHORIZATION_KEYS):
        raise PilotError("attempt-3 authorization schema mismatch")
    if any(type(parsed[key]) is not str for key in _ATTEMPT3_AUTHORIZATION_KEYS):
        raise PilotError("attempt-3 authorization values must be strings")
    if not re.fullmatch(r"[0-9a-f]{40}", parsed["revision"]):
        raise PilotError("attempt-3 authorization revision schema mismatch")
    for key in _ATTEMPT3_AUTHORIZATION_KEYS - {"revision"}:
        if not _HEX64.fullmatch(parsed[key]):
            raise PilotError(f"attempt-3 authorization hash schema mismatch: {key}")
    return {key: parsed[key] for key in sorted(parsed)}


def validate_attempt3_authorization(value: str | dict[str, Any], *, phase: str | None = None,
                                    phase_spec: dict[str, Any] | None = None,
                                    repo_root: pathlib.Path | None = None,
                                    catalog_source: pathlib.Path | None = None,
                                    verify_sources: bool = True) -> dict[str, str]:
    authorization = _authorization_json(value)
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    spec = phase_spec or (attempt3_phase_specs(repo_root=root)[phase] if phase else None)
    if spec is None or phase not in ("phase-a", "phase-b"):
        raise PilotError("attempt-3 authorization phase binding missing")
    expected_command = canonical_attempt3_command(phase, spec)
    expected_command_sha = hashlib.sha256(json.dumps(expected_command, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if authorization["canonical_command_sha256"] != expected_command_sha:
        raise PilotError("attempt-3 authorization canonical command mismatch")
    if not verify_sources:
        return authorization
    try:
        current_revision = _git_output("git", "rev-parse", "HEAD")
    except PilotError:
        if root.resolve() == _CANONICAL_REPO_ROOT.resolve():
            raise
        current_revision = authorization["revision"]
    if current_revision != authorization["revision"]:
        raise PilotError("attempt-3 authorization revision mismatch")
    pilot_path = root / "scripts" / "ds4_segmented_pilot.py"
    catalog_path = catalog_source or root / "scripts" / "finetune_ds4.py"
    if file_sha256(pilot_path) != authorization["pilot_source_sha256"]:
        raise PilotError("attempt-3 authorization pilot source mismatch")
    if file_sha256(catalog_path) != authorization["catalog_source_sha256"]:
        raise PilotError("attempt-3 authorization catalog source mismatch")
    if protected_files_manifest_sha256(repo_root=root) != authorization["protected_files_manifest_sha256"]:
        raise PilotError("attempt-3 authorization protected manifest mismatch")
    protected_status = _git_output("git", "status", "--porcelain=1", "--untracked-files=all", "--", *_ATTEMPT3_PROTECTED_PATHS)
    if protected_status:
        raise PilotError("attempt-3 protected paths are not clean")
    if authorization["attempt2_runtime_manifest_sha256"] != _ATTEMPT2_CANONICAL_MANIFEST[2]:
        raise PilotError("attempt-3 authorization attempt-2 manifest binding mismatch")
    return authorization


def canonical_attempt3_authorization(phase: str, phase_spec: dict[str, Any], *,
                                     repo_root: pathlib.Path | None = None,
                                     workspace: pathlib.Path | str | None = None,
                                     catalog_source: pathlib.Path | None = None) -> str:
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        revision = _git_output("git", "rev-parse", "HEAD")
    except PilotError:
        revision = "0" * 40
    def digest(path: pathlib.Path) -> str:
        return file_sha256(path) if path.is_file() else "0" * 64
    pilot_path = root / "scripts" / "ds4_segmented_pilot.py"
    catalog_path = catalog_source or root / "scripts" / "finetune_ds4.py"
    command = canonical_attempt3_command(phase, phase_spec)
    value = {
        "revision": revision,
        "canonical_command_sha256": hashlib.sha256(json.dumps(command, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "pilot_source_sha256": digest(pilot_path),
        "catalog_source_sha256": digest(catalog_path),
        "protected_files_manifest_sha256": protected_files_manifest_sha256(repo_root=root) if all((root / p).is_file() for p in _ATTEMPT3_PROTECTED_PATHS) else "0" * 64,
        "attempt2_runtime_manifest_sha256": _ATTEMPT2_CANONICAL_MANIFEST[2],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _attempt1_manifest_path(entry: dict[str, Any], repo_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(entry["path"])
    return path if path.is_absolute() else repo_root / path


def verify_attempt1_historical_evidence(*, repo_root: pathlib.Path | None = None) -> list[dict[str, Any]]:
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    verified: list[dict[str, Any]] = []
    for entry in ATTEMPT1_EVIDENCE_SHA256:
        path = _attempt1_manifest_path(entry, root)
        try:
            size = path.stat().st_size
            actual_sha = file_sha256(path)
        except OSError as exc:
            raise PilotError(f"attempt-1 historical evidence unavailable: {path}") from exc
        if size != entry["size"] or actual_sha != entry["sha256"]:
            raise PilotError(f"attempt-1 historical evidence hash mismatch: {path}")
        verified.append({"path": entry["path"], "size": size, "sha256": actual_sha})
    return verified


def _historical_path(path: str, root: pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verify_historical_snapshot(path: pathlib.Path, size: int, expected_sha: str, label: str,
                                *, parse_json: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PilotError(f"{label} unavailable: {path}") from exc
    observed_size = len(payload)
    observed_sha = hashlib.sha256(payload).hexdigest()
    if (observed_size, observed_sha) != (size, expected_sha):
        raise PilotError(f"{label} hash mismatch: {path}")
    parsed: dict[str, Any] | None = None
    if parse_json:
        try:
            value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as exc:
            raise PilotError(f"{label} is not readable JSON: {path}") from exc
        if not isinstance(value, dict):
            raise PilotError(f"{label} JSON must be an object: {path}")
        parsed = value
    return {"path": str(path), "size": observed_size, "sha256": observed_sha}, parsed


def _verify_historical_file(path: pathlib.Path, size: int, expected_sha: str, label: str) -> dict[str, Any]:
    record, _ = _verify_historical_snapshot(path, size, expected_sha, label)
    return record


def _attempt2_expected_path(raw: str, root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(raw)
    canonical_root = REPO_ROOT.resolve()
    if path.is_absolute() and str(path).startswith(str(canonical_root) + os.sep) and root.resolve() != canonical_root:
        return root / path.relative_to(canonical_root)
    return path if path.is_absolute() else root / path


def _verify_historical_evidence_snapshot(*, prelog_expected: tuple[str, int, str],
                                         manifest_expected: tuple[str, int, str],
                                         files_expected: tuple[tuple[str, int, str], ...],
                                         absent_expected: tuple[str, ...],
                                         expected_manifest: dict[str, Any],
                                         repo_root: pathlib.Path | None = None,
                                         immutable_files_expected: tuple[tuple[str, int, str], ...] | None = None) -> dict[str, Any]:
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    if immutable_files_expected is not None and tuple(files_expected) != tuple(immutable_files_expected):
        raise PilotError("historical runtime target bindings are not immutable")
    if len(files_expected) != 10 or len(absent_expected) != 8:
        raise PilotError("historical evidence fixture cardinality mismatch")
    prelog, _ = _verify_historical_snapshot(_historical_path(prelog_expected[0], root), prelog_expected[1],
                                             prelog_expected[2], "historical pre-log evidence")
    manifest_path = _historical_path(manifest_expected[0], root)
    manifest_file, manifest = _verify_historical_snapshot(manifest_path, manifest_expected[1], manifest_expected[2],
                                                          "historical runtime manifest", parse_json=True)
    if not _json_exact_equal(manifest, expected_manifest):
        raise PilotError("historical runtime manifest semantic or schema mismatch")
    verified_files = []
    report_snapshot = None
    report_path = _attempt2_expected_path(files_expected[2][0], root)
    for raw, size, sha in files_expected:
        path = _attempt2_expected_path(raw, root)
        record, parsed = _verify_historical_snapshot(path, size, sha, "historical runtime evidence",
                                                     parse_json=path == report_path)
        verified_files.append(record)
        if parsed is not None:
            report_snapshot = parsed
    for raw in absent_expected:
        path = _historical_path(raw, root)
        if path.exists():
            raise PilotError(f"historical absence fact violated: {path}")
    if not isinstance(report_snapshot, dict) or type(report_snapshot.get("attempt")) is not int or report_snapshot.get("attempt") != 2:
        raise PilotError("historical report identity mismatch")
    if report_snapshot.get("namespace") != "ds4-segmented-pilot-attempt-2":
        raise PilotError("historical report namespace mismatch")
    return {"prelog": prelog, "manifest": manifest_file, "manifest_snapshot": expected_manifest,
            "facts": {"manifest_sha256": manifest_file["sha256"]}, "files": verified_files,
            "report_identity": {"attempt": report_snapshot["attempt"], "namespace": report_snapshot["namespace"],
                                "effective": {key: report_snapshot.get("effective", {}).get(key)
                                              for key in ("model", "data", "config")},
                                "identity_manifest": report_snapshot.get("identity_manifest", {})}}


def verify_attempt2_historical_evidence(*, repo_root: pathlib.Path | None = None) -> dict[str, Any]:
    # Keep this trust root inside the verifier. Module-level compatibility aliases
    # are intentionally not consulted by the admission boundary.
    prelog_expected = ("agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md", 990,
                       "cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41")
    manifest_expected = ("agent-output/cmux-14-5-attempt-2/phase-a2-runtime-failure-manifest.json", 3123,
                        "2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88")
    files_expected = (
        ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-log.txt", 1432, "edf1ae2215083aab2b8648403cf53f395ce39c3da2277f80fa7da2e87976734e"),
        ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-report.json", 80454, "d9cbc1895b62f4f182a25dcf58d851d077c6f92745bd3b0d387c78fa6dc8a500"),
        ("/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/pilot-report.json", 80567, "84e8d26cf658f4ba15eda8346b4e507dc331d65da29b00b21aca1073eb97c0ce"),
        ("/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail", 542, "079c89c88bf87699562ce8e5866d88ba3e92adec6d2b0c787a481778d85b2918"),
        ("/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail", 539, "91b4dd994a3bd0319c5d0edf9d0c49f933cc95f2e49a1b1f359e4a92e609edfe"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000001_adapters.safetensors", 22292892, "89363e82325e095bc74f7cdb6e022280438fa1667d52abcf33f1dfb3db1a2d18"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors", 22292892, "2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapter_config.json", 339, "13620f30b39a0de718f0769830b4b0a2233eaebea62d6fc43da08231d0a0691d"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapters.safetensors", 22292892, "2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652"),
        ("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/phase-a-start.safetensors", 22292892, "c6dd9f67be04936c4dce58d7b3643403354516d6167e4e08d76c393890339e51"),
    )
    absent_expected = (
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-ok",
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-ok",
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-fail",
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-ok",
        "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b",
        "agent-output/cmux-14-5-attempt-2/phase-b-log.txt",
        "agent-output/cmux-14-5-attempt-2/phase-b-report.json",
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock",
    )
    expected_manifest = {
        "attempt": 2, "command_sha256": "38e6b1aa11d28c41c126fda6dab55cf68a9eb1d9a71fc177eec8875a3114b2de",
        "error": "safetensors __metadata__ must be an object", "exit_code": 1,
        "files": [{"path": path, "size": size, "sha256": sha} for path, size, sha in files_expected],
        "lock_absent_after_exit": True,
        "lock_lifecycle": {"acquired": True, "path": "/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock", "release_attempts": 1, "released": True},
        "namespace": "ds4-segmented-pilot-attempt-2", "ok_markers_absent": True, "retry_performed": False,
        "revision": "e6d34fa03479316720430f35cc94d4606a45ef96", "status": "fail", "story": "14.5b",
        "training_log_observations": {"checkpoint1_saved": True, "checkpoint2_saved": True,
            "iter1_loss": "19.334", "iter1_val_loss": "19.553", "iter2_loss": "17.648", "iter2_val_loss": "19.841"},
        "wall_seconds": 2444.2739184170496, "watchdog_cancelled": True,
    }
    facts_expected = {key: expected_manifest[key] for key in (
        "command_sha256", "exit_code", "lock_absent_after_exit", "lock_lifecycle", "namespace",
        "ok_markers_absent", "retry_performed", "revision", "watchdog_cancelled")}
    expected_manifest_keys = {
        "attempt", "command_sha256", "error", "exit_code", "files", "lock_absent_after_exit", "lock_lifecycle",
        "namespace", "ok_markers_absent", "retry_performed", "revision", "status", "story",
        "training_log_observations", "wall_seconds", "watchdog_cancelled",
    }
    if len(files_expected) != 10 or len(absent_expected) != 8 or set(expected_manifest) != expected_manifest_keys:
        raise PilotError("attempt-2 trust-root schema mismatch")
    if any(type(path) is not str or type(size) is not int or size < 0 or type(sha) is not str or not _HEX64.fullmatch(sha)
           for path, size, sha in files_expected):
        raise PilotError("attempt-2 file descriptor schema mismatch")
    if any(type(path) is not str for path in absent_expected):
        raise PilotError("attempt-2 absence descriptor schema mismatch")
    if tuple(files_expected) != _ATTEMPT2_VERIFIER_FILES:
        raise PilotError("attempt-2 runtime target bindings are not immutable")
    manifest_files = expected_manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != 10 or any(
        not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}
        or type(item["path"]) is not str or type(item["size"]) is not int or item["size"] < 0
        or type(item["sha256"]) is not str or not _HEX64.fullmatch(item["sha256"])
        for item in manifest_files
    ):
        raise PilotError("attempt-2 manifest file schema mismatch")
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    prelog, _ = _verify_historical_snapshot(_historical_path(prelog_expected[0], root), prelog_expected[1],
                                             prelog_expected[2], "attempt-2 pre-log evidence")
    manifest_path = _historical_path(manifest_expected[0], root)
    manifest_file, manifest = _verify_historical_snapshot(manifest_path, manifest_expected[1], manifest_expected[2],
                                                          "attempt-2 runtime manifest", parse_json=True)
    if not _json_exact_equal(manifest, expected_manifest):
        raise PilotError("attempt-2 runtime manifest semantic or schema mismatch")
    verified_files = []
    report_snapshot = None
    for path, size, sha in files_expected:
        resolved = _attempt2_expected_path(path, root)
        record, parsed = _verify_historical_snapshot(
            resolved, size, sha, "attempt-2 runtime evidence", parse_json=resolved == _attempt2_expected_path(files_expected[2][0], root))
        verified_files.append(record)
        if parsed is not None:
            report_snapshot = parsed
    for raw in absent_expected:
        path = _historical_path(raw, root)
        if path.exists():
            raise PilotError(f"attempt-2 historical absence fact violated: {path}")
    report_path = _attempt2_expected_path(files_expected[2][0], root)
    if report_snapshot is None:
        raise PilotError("attempt-2 historical report is not readable JSON")
    if type(report_snapshot.get("attempt")) is not int or report_snapshot.get("attempt") != 2 or type(report_snapshot.get("namespace")) is not str or report_snapshot.get("namespace") != "ds4-segmented-pilot-attempt-2":
        raise PilotError("attempt-2 historical report identity mismatch")
    return {"prelog": prelog, "manifest": manifest_file, "manifest_snapshot": expected_manifest,
            "facts": facts_expected, "files": verified_files,
            "report_identity": {"attempt": report_snapshot.get("attempt"), "namespace": report_snapshot.get("namespace"),
                                "effective": {key: report_snapshot.get("effective", {}).get(key) for key in ("model", "data", "config")},
                                "identity_manifest": report_snapshot.get("identity_manifest", {})}}


def _attempt3_absent_paths(paths: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    return [paths[key] for key in (
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
        "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
        "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
        "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint",
        "phase-b-final-checkpoint", "phase-b-config", "phase-b-log", "phase-b-report",
        "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail",
    )]


def _attempt3_phase_b_absent_paths(paths: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    return [paths[key] for key in (
        "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint",
        "phase-b-config", "phase-b-output", "phase-b-log", "phase-b-report",
        "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail",
    )]


def check_attempt3_launch(phase: str, log_path: pathlib.Path | str, *,
                         repo_root: pathlib.Path | None = None,
                         workspace: pathlib.Path | str | None = None,
                         allow_active_log: bool = False,
                         trusted_identity: dict[str, Any] | None = None,
                         authorization_json: str | dict[str, Any] | None = None,
                         phase_a_authorization_json: str | dict[str, Any] | None = None) -> dict[str, Any]:
    if phase not in ("phase-a", "phase-b"):
        raise PilotError(f"unknown attempt-3 phase: {phase}")
    root = pathlib.Path(repo_root) if repo_root is not None else REPO_ROOT
    paths = attempt3_namespace(repo_root=root, workspace=workspace)
    authorization = None
    phase_a_authorization = None
    if phase == "phase-b" and root.resolve() == _CANONICAL_REPO_ROOT.resolve() and phase_a_authorization_json is None:
        raise PilotError("separate external Phase A3 authorization is required for Phase B3")
    if phase == "phase-b" and phase_a_authorization_json is not None:
        phase_a_authorization = validate_attempt3_authorization(
            phase_a_authorization_json, phase="phase-a",
            phase_spec=attempt3_phase_specs(repo_root=root, workspace=workspace)["phase-a"],
            repo_root=root, verify_sources=root.resolve() == _CANONICAL_REPO_ROOT.resolve())
    if authorization_json is None:
        if root.resolve() == _CANONICAL_REPO_ROOT.resolve():
            raise PilotError("attempt-3 authorization is required before launch admission")
    else:
        authorization = validate_attempt3_authorization(
            authorization_json, phase=phase,
            phase_spec=attempt3_phase_specs(repo_root=root, workspace=workspace)[phase],
            repo_root=root,
            verify_sources=root.resolve() == _CANONICAL_REPO_ROOT.resolve())
    expected_log = paths[f"{phase}-log"]
    historical_attempt2 = verify_attempt2_historical_evidence(repo_root=root) if root.resolve() == _CANONICAL_REPO_ROOT.resolve() else {}
    if pathlib.Path(log_path) != expected_log:
        raise PilotError(f"attempt-3 log path mismatch: expected {expected_log}")
    if phase == "phase-a":
        collisions = _attempt3_absent_paths(paths)
        if allow_active_log:
            collisions.remove(expected_log)
    else:
        required = (
            "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
            "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
            "phase-a-report", "phase-a-ok", "phase-a-log", "phase-b-resume",
        )
        for key in required:
            if not paths[key].exists():
                raise PilotError(f"attempt-3 Phase A dependency missing: {paths[key]}")
        ensure_absent(paths["phase-a-fail"])
        phase_a_spec = attempt3_phase_specs(repo_root=root, workspace=workspace)["phase-a"]
        historical = verify_attempt1_historical_evidence(repo_root=root)
        _, phase_a_report = _read_json_snapshot(paths["phase-a-report"])
        _, phase_a_marker = _read_json_snapshot(paths["phase-a-ok"])
        validate_phase_b_dependency(phase_a_report, phase_a_marker, paths["phase-b-resume"], phase_spec=phase_a_spec,
                                    historical_evidence=historical, historical_attempt2_evidence=historical_attempt2,
                                    trusted_identity=trusted_identity,
                                    trusted_phase_a_authorization=phase_a_authorization)
        collisions = _attempt3_phase_b_absent_paths(paths)
        if allow_active_log:
            collisions.remove(expected_log)
    for path in collisions:
        if path.exists():
            raise PilotError(f"attempt-3 collision: {path}")
    historical = locals().get("historical") or verify_attempt1_historical_evidence(repo_root=root)
    if trusted_identity is not None and historical_attempt2:
        expected = historical_attempt2.get("report_identity", {}).get("identity_manifest", {}).get("immutable", {})
        current = trusted_identity.get("immutable", {})
        for key in ("config_sha256", "model_manifest", "dataset_manifest"):
            if expected.get(key) != current.get(key):
                raise PilotError(f"attempt-2 pinned identity mismatch: {key}")
    return {"attempt": 3, "namespace": ATTEMPT3_NAMESPACE,
            "phase": phase, "log_path": str(expected_log),
            "authorization": authorization,
            "phase_a_authorization": phase_a_authorization,
            "attempt_1_historical_evidence": historical,
            "attempt_2_historical_evidence": historical_attempt2}


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
        if record.get("expected_mask_tokens") is None or record.get("mask_tokens_match") is not True or record.get("n_tokens") != record.get("expected_mask_tokens"):
            raise PilotError(f"{phase} mask token evidence mismatch")
    return local_steps


_A2_IDENTITY_IMMUTABLE_KEYS = frozenset({
    "interpreter", "python_version", "mlx_version", "mlx_lm_resolved_module",
    "vendor_head", "vendor_gitlink", "vendor_inner_clean", "provider_sha256",
    "smoke_sha256", "pilot_source_sha256", "config_sha256", "provenance_sha256",
    "provenance_split_hashes", "model_manifest", "dataset_manifest", "lora_parameters",
    "git_head",
})
_A2_IDENTITY_KEYS = frozenset({"immutable", "dynamic_resources", "repository_status"})
_A2_REPORT_KEYS = frozenset({
    "status", "phase", "attempt", "namespace", "effective", "contract_digest",
    "identity_manifest", "provider_calls", "optimizer_updates", "steps", "provider_evidence",
    "validation_evidence", "artifacts", "output_path", "commands", "wall_seconds",
    "retry", "fallback", "non_claims", "global_mapping", "attempt_1_historical_evidence",
    "attempt_2_historical_evidence", "authorization",
    "resume_source", "report_path", "log_path", "ok_marker_path", "fail_marker_path",
    "final_report_path", "namespace_paths", "lock_lifecycle", "watchdog_cancelled", "exit_code",
})
_PHASE_A_ADMISSION_LINEAGE_KEYS = frozenset({
    "report", "marker", "authorization", "attempt_1_historical_evidence",
    "attempt_2_historical_evidence", "identity_manifest", "contract_digest",
    "artifacts", "resume_source",
})
_ATTEMPT3_MARKER_BASE_KEYS = frozenset({
    "phase", "status", "attempt", "namespace", "report_path", "report_sha256",
    "output_path", "timestamp", "exit_code", "contract_digest",
})
_ATTEMPT3_MARKER_PHASE_A_KEYS = _ATTEMPT3_MARKER_BASE_KEYS | frozenset({
    "resume_source", "resume_source_file_sha256", "resume_source_canonical_tensor_digest_v1",
})
_A2_EFFECTIVE_BASE_KEYS = frozenset(set(COMMON_VALUES) | {
    "phase", "model", "data", "config", "adapter_path", "resume_adapter_file", "train", "test", "hf_dataset",
    "attempt", "log_path", "command", "iters", "steps_per_eval",
})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_A2_DTYPE_ALLOWLIST = frozenset({
    "bool", "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
})
_A2_PATH = re.compile(r"^[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)*$")


def _a2_finite_number(value: Any, label: str, *, minimum: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PilotError(f"canonical A3 {label} must be a finite number")
    if minimum is not None and float(value) < minimum:
        raise PilotError(f"canonical A3 {label} below minimum")


def _a2_positive_ordinal(value: Any, label: str) -> None:
    if type(value) is not int or value < 1:
        raise PilotError(f"canonical A3 {label} ordinal invalid")


def _a2_gradient_schema(schema: Any) -> None:
    if not isinstance(schema, list) or not schema:
        raise PilotError("canonical A3 gradient schema missing")
    seen_paths: set[str] = set()
    for entry in schema:
        if not isinstance(entry, dict) or set(entry) != {"path", "shape", "dtype"}:
            raise PilotError("canonical A3 gradient schema mismatch")
        path, shape, dtype = entry["path"], entry["shape"], entry["dtype"]
        if not isinstance(path, str) or not _A2_PATH.fullmatch(path):
            raise PilotError("canonical A3 gradient path schema mismatch")
        if path in seen_paths:
            raise PilotError("canonical A3 gradient paths must be unique")
        seen_paths.add(path)
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise PilotError("canonical A3 gradient shape schema mismatch")
        if not isinstance(dtype, str) or dtype not in _A2_DTYPE_ALLOWLIST:
            raise PilotError("canonical A3 gradient dtype schema mismatch")


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise PilotError(f"canonical A3 {label} schema mismatch: expected {sorted(expected)}, got {actual}")
    return value


def _validate_manifest_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise PilotError(f"canonical A3 {label} evidence missing")
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise PilotError(f"canonical A3 {label} schema mismatch")
        if not isinstance(item["path"], str) or type(item["size"]) is not int or item["size"] < 0 or not _HEX64.fullmatch(item["sha256"]):
            raise PilotError(f"canonical A3 {label} evidence invalid")


def _validate_resource_evidence(resources: Any) -> None:
    resources = _require_exact_keys(resources, frozenset({"available_memory", "disk_free", "competing_processes", "allowed_process_skips"}), "resource identity")
    if type(resources["available_memory"]) is not int or resources["available_memory"] < PILOT_MEMORY_HEADROOM or type(resources["disk_free"]) is not int or resources["disk_free"] < PILOT_DISK_MIN_FREE:
        raise PilotError("canonical A3 resource identity below pinned boundary")
    competing = resources["competing_processes"]
    if not isinstance(competing, list) or competing != sorted(competing) or any(
        not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int or type(item[1]) is not int
        or item[0] < 0 or item[1] < 0 or item[1] > 50 * 1024**3 for item in competing
    ):
        raise PilotError("canonical A3 competing-process evidence schema mismatch")
    skips = _require_exact_keys(resources["allowed_process_skips"],
                                frozenset({"total", "counts_by_type", "pids_by_type", "unknown_pid_counts_by_type"}),
                                "resource skip")
    names = ("AccessDenied", "NoSuchProcess", "ZombieProcess")
    counts = _require_exact_keys(skips["counts_by_type"], frozenset(names), "resource skip counts")
    pids = _require_exact_keys(skips["pids_by_type"], frozenset(names), "resource skip pids")
    unknown = _require_exact_keys(skips["unknown_pid_counts_by_type"], frozenset(names), "resource skip unknown counts")
    if type(skips["total"]) is not int or skips["total"] < 0:
        raise PilotError("canonical A3 resource skip total schema mismatch")
    for name in names:
        if type(counts[name]) is not int or counts[name] < 0 or type(unknown[name]) is not int or unknown[name] < 0:
            raise PilotError("canonical A3 resource skip count schema mismatch")
        if not isinstance(pids[name], list) or pids[name] != sorted(pids[name]) or any(type(pid) is not int or pid < 0 for pid in pids[name]):
            raise PilotError("canonical A3 resource skip PID schema mismatch")
        if counts[name] != len(pids[name]) + unknown[name]:
            raise PilotError("canonical A3 resource skip count invariant mismatch")
    if skips["total"] != sum(counts.values()):
        raise PilotError("canonical A3 resource skip total invariant mismatch")


def _validate_canonical_identity(identity: Any, *, trusted_identity: dict[str, Any] | None = None) -> None:
    value = _require_exact_keys(identity, _A2_IDENTITY_KEYS, "identity")
    immutable = _require_exact_keys(value["immutable"], _A2_IDENTITY_IMMUTABLE_KEYS, "immutable identity")
    if trusted_identity is not None:
        trusted = _require_exact_keys(trusted_identity, _A2_IDENTITY_KEYS, "trusted identity")
        if immutable != trusted["immutable"]:
            raise PilotError("canonical A3 immutable identity substituted")
    if immutable["interpreter"] != PILOT_INTERPRETER or immutable["mlx_version"] != "0.31.2" or "/vendor/mlx-lm/" not in immutable["mlx_lm_resolved_module"]:
        raise PilotError("canonical A3 pinned interpreter/MLX-LM identity substituted")
    if type(immutable["python_version"]) is not list or len(immutable["python_version"]) != 3 or any(type(item) is not int for item in immutable["python_version"]):
        raise PilotError("canonical A3 Python identity schema mismatch")
    if immutable["vendor_head"] != PILOT_VENDOR_SHA or not str(immutable["vendor_gitlink"]).startswith(f"160000 {PILOT_VENDOR_SHA} 0") or immutable["vendor_inner_clean"] is not True:
        raise PilotError("canonical A3 vendor identity substituted")
    for key, expected in (("provider_sha256", PILOT_PROVIDER_SHA256), ("smoke_sha256", PILOT_SMOKE_SHA256)):
        if immutable[key] != expected:
            raise PilotError(f"canonical A3 {key} substituted")
    for key in ("pilot_source_sha256", "config_sha256", "provenance_sha256"):
        if not isinstance(immutable[key], str) or not _HEX64.fullmatch(immutable[key]):
            raise PilotError(f"canonical A3 {key} identity missing")
    if not isinstance(immutable["git_head"], str) or not re.fullmatch(r"[0-9a-f]{40}", immutable["git_head"]):
        raise PilotError("canonical A3 git identity schema mismatch")
    if set(immutable["provenance_split_hashes"]) != {"train.jsonl", "valid.jsonl", "test.jsonl"} or any(not _HEX64.fullmatch(item) for item in immutable["provenance_split_hashes"].values()):
        raise PilotError("canonical A3 provenance split identity mismatch")
    _validate_manifest_list(immutable["model_manifest"], "model manifest")
    _validate_manifest_list(immutable["dataset_manifest"], "dataset manifest")
    lora = immutable["lora_parameters"]
    if not isinstance(lora, dict) or set(lora) != {"rank", "scale", "dropout", "keys"} or lora["rank"] != 8 or lora["scale"] != 20.0 or lora["dropout"] != 0.0 or not isinstance(lora["keys"], list) or not lora["keys"] or len(lora["keys"]) != len(set(lora["keys"])) or any(not isinstance(item, str) or not item for item in lora["keys"]):
        raise PilotError("canonical A3 LoRA identity substituted")
    _validate_resource_evidence(value["dynamic_resources"])
    if not isinstance(value["repository_status"], list) or any(not isinstance(item, str) for item in value["repository_status"]):
        raise PilotError("canonical A3 repository identity schema mismatch")


def _validate_canonical_effective(effective: Any, report: dict[str, Any], phase_spec: dict[str, Any]) -> None:
    effective = _require_exact_keys(effective, _A2_EFFECTIVE_BASE_KEYS, "effective pins")
    phase = report["phase"]
    identity = attempt3_launch_identity()
    paths = phase_spec["namespace_paths"]
    expected = dict(COMMON_VALUES)
    expected.update({"phase": phase, "attempt": 3, "iters": phase_spec["iters"], "steps_per_eval": phase_spec["steps_per_eval"],
                     "model": identity["model"], "data": identity["data"], "config": identity["config"],
                     "adapter_path": phase_spec["adapter_path"], "resume_adapter_file": phase_spec["resume_adapter_file"],
                     "train": True, "test": False, "hf_dataset": False,
                     "log_path": paths[f"{phase}-log"]})
    for key, expected_value in expected.items():
        if effective.get(key) != expected_value:
            raise PilotError(f"canonical A3 effective pin substituted: {key}")
    command = effective.get("command")
    expected_command = canonical_attempt3_command(phase, phase_spec)
    if command != expected_command or report.get("commands") != expected_command:
        raise PilotError("canonical A3 exact command binding missing")


def _validate_canonical_step_records(phase: str, report: dict[str, Any], phase_spec: dict[str, Any],
                                     expected_artifacts: dict[str, Any]) -> None:
    provider = report.get("provider_evidence")
    validate_step_evidence(phase, provider, phase_spec["iters"])
    provider_keys = {"phase", "provider_call", "local_step", "global_step", "loss", "loss_dtype", "token_dtype",
                     "n_tokens", "expected_mask_tokens", "mask_tokens_match", "gradient_schema", "gradient_leaf_count",
                     "gradient_paths", "gradient_shapes", "gradient_dtypes", "gradients_finite", "provider_elapsed_seconds"}
    for index, item in enumerate(provider, 1):
        if set(item) != provider_keys or item["phase"] != phase:
            raise PilotError("canonical A3 provider evidence schema/pairing mismatch")
        _a2_positive_ordinal(item["provider_call"], "provider_call")
        _a2_positive_ordinal(item["local_step"], "local_step")
        if item["provider_call"] != index or item["local_step"] != index:
            raise PilotError("canonical A3 provider/update ordinal pairing mismatch")
        _a2_positive_ordinal(item["global_step"], "global_step")
        expected_global = global_steps(phase, [index])[0]
        if item["global_step"] != expected_global:
            raise PilotError("canonical A3 provider/global mapping mismatch")
        _a2_finite_number(item["loss"], "provider loss")
        _a2_finite_number(item["provider_elapsed_seconds"], "provider elapsed", minimum=0.0)
        if item["loss_dtype"] not in _A2_DTYPE_ALLOWLIST or item["token_dtype"] not in _A2_DTYPE_ALLOWLIST:
            raise PilotError("canonical A3 provider dtype schema mismatch")
        if type(item["n_tokens"]) is not int or item["n_tokens"] <= 0 or type(item["expected_mask_tokens"]) is not int or item["expected_mask_tokens"] <= 0:
            raise PilotError("canonical A3 token ordinal schema mismatch")
        schema = item["gradient_schema"]
        _a2_gradient_schema(schema)
        if item["gradient_leaf_count"] != len(schema) or item["gradient_paths"] != [entry["path"] for entry in schema]:
            raise PilotError("canonical A3 provider gradient pairing mismatch")
        if item["gradient_shapes"] != {entry["path"]: entry["shape"] for entry in schema} or item["gradient_dtypes"] != {entry["path"]: entry["dtype"] for entry in schema}:
            raise PilotError("canonical A3 provider gradient schema pairing mismatch")
        if item["gradients_finite"] is not True:
            raise PilotError("canonical A3 gradient finiteness binding missing")
    expected_global = global_steps(phase, range(1, phase_spec["iters"] + 1))
    validation = report.get("validation_evidence")
    expected_validation = _expected_validation_steps(phase_spec)
    if not isinstance(validation, list) or len(validation) != len(expected_validation) or [item.get("iteration") for item in validation] != expected_validation:
        raise PilotError("canonical A3 validation evidence cardinality mismatch")
    for item in validation:
        if set(item) != {"iteration", "val_loss", "val_time"} or type(item["iteration"]) is not int:
            raise PilotError("canonical A3 validation evidence schema mismatch")
        _a2_finite_number(item["val_loss"], "validation loss")
        _a2_finite_number(item["val_time"], "validation elapsed", minimum=0.0)
    steps = report.get("steps")
    if not isinstance(steps, list) or len(steps) != phase_spec["iters"]:
        raise PilotError("canonical A3 update evidence cardinality mismatch")
    for index, (item, global_step) in enumerate(zip(steps, expected_global), 1):
        if set(item) != {"phase", "local_step", "global_step", "loss", "learning_rate", "tokens_per_second", "iterations_per_second", "train_step_wall_seconds", "optimizer_update_ordinal", "provider_call", "checkpoint"} or item["phase"] != phase:
            raise PilotError("canonical A3 update evidence schema mismatch")
        _a2_positive_ordinal(item["local_step"], "local_step")
        _a2_positive_ordinal(item["global_step"], "global_step")
        _a2_positive_ordinal(item["provider_call"], "provider_call")
        _a2_positive_ordinal(item["optimizer_update_ordinal"], "optimizer_update_ordinal")
        if item["local_step"] != index or item["global_step"] != global_step or item["provider_call"] != index or item["optimizer_update_ordinal"] != index:
            raise PilotError("canonical A3 update/checkpoint ordinal pairing mismatch")
        _a2_finite_number(item["loss"], "update loss")
        _a2_finite_number(item["learning_rate"], "learning rate", minimum=0.0)
        _a2_finite_number(item["tokens_per_second"], "tokens per second", minimum=0.0)
        _a2_finite_number(item["iterations_per_second"], "iterations per second", minimum=0.0)
        _a2_finite_number(item["train_step_wall_seconds"], "update elapsed", minimum=0.0)
        if item["checkpoint"] != expected_artifacts["checkpoints"][index - 1]:
            raise PilotError("canonical A3 update/checkpoint binding mismatch")


def _validate_phase_a_admission_lineage(lineage: Any, *, phase_spec: dict[str, Any],
                                        trusted_authorization: dict[str, str] | None = None) -> None:
    _require_exact_keys(lineage, _PHASE_A_ADMISSION_LINEAGE_KEYS, "Phase A admission lineage")
    report_ref = _require_exact_keys(lineage["report"], frozenset({"path", "sha256", "bytes_b64", "schema"}), "Phase A report admission")
    marker_ref = _require_exact_keys(lineage["marker"], frozenset({"path", "sha256", "bytes_b64", "schema"}), "Phase A marker admission")
    try:
        report_bytes = base64.b64decode(report_ref["bytes_b64"], validate=True)
        marker_bytes = base64.b64decode(marker_ref["bytes_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise PilotError("Phase A admission lineage bytes encoding invalid") from exc
    if hashlib.sha256(report_bytes).hexdigest() != report_ref["sha256"] or hashlib.sha256(marker_bytes).hexdigest() != marker_ref["sha256"]:
        raise PilotError("Phase A admission lineage byte hash mismatch")
    report_path = pathlib.Path(report_ref["path"])
    marker_path = pathlib.Path(marker_ref["path"])
    current_report_bytes, current_report_sha, phase_a_report = _read_json_bytes_snapshot(report_path)
    current_marker_bytes, current_marker_sha, phase_a_marker = _read_json_bytes_snapshot(marker_path)
    if current_report_bytes != report_bytes or current_report_sha != report_ref["sha256"]:
        raise PilotError("Phase A admission lineage report changed after admission")
    if current_marker_bytes != marker_bytes or current_marker_sha != marker_ref["sha256"]:
        raise PilotError("Phase A admission lineage marker changed after admission")
    if sorted(phase_a_report) != report_ref["schema"] or sorted(phase_a_marker) != marker_ref["schema"]:
        raise PilotError("Phase A admission lineage schema changed")
    paths = phase_spec["namespace_paths"]
    phase_a_spec = attempt3_phase_specs(repo_root=pathlib.Path(paths["phase-a-report"]).resolve().parents[2],
                                        workspace=pathlib.Path(paths["phase-a-output"]).resolve().parent)["phase-a"]
    _validate_attempt3_marker(phase_a_marker, phase="phase-a", phase_spec=phase_a_spec,
                              report=phase_a_report, marker_path=marker_path,
                              trusted_report_sha256=current_report_sha)
    if trusted_authorization is not None and phase_a_report.get("authorization") != trusted_authorization:
        raise PilotError("Phase A admission authorization substituted")
    validate_canonical_attempt3_report(
        phase_a_report, phase_spec=phase_a_spec, report_path=report_path,
        historical_evidence=phase_a_report.get("attempt_1_historical_evidence"),
        historical_attempt2_evidence=phase_a_report.get("attempt_2_historical_evidence"),
        trusted_identity=phase_a_report.get("identity_manifest"),
        trusted_authorization=trusted_authorization or phase_a_report.get("authorization"))
    expected = {
        "authorization": phase_a_report.get("authorization"),
        "attempt_1_historical_evidence": phase_a_report.get("attempt_1_historical_evidence"),
        "attempt_2_historical_evidence": phase_a_report.get("attempt_2_historical_evidence"),
        "identity_manifest": phase_a_report.get("identity_manifest"),
        "contract_digest": phase_a_report.get("contract_digest"),
        "artifacts": phase_a_report.get("artifacts"),
        "resume_source": phase_a_report.get("resume_source"),
    }
    for key, value in expected.items():
        if lineage[key] != value:
            raise PilotError(f"Phase A admission lineage {key} substituted")


def validate_canonical_attempt3_report(report: dict[str, Any], *, phase_spec: dict[str, Any],
                                       report_path: pathlib.Path | str | None = None,
                                       historical_evidence: list[dict[str, Any]] | None = None,
                                       historical_attempt2_evidence: dict[str, Any] | None = None,
                                       trusted_identity: dict[str, Any] | None = None,
                                       trusted_authorization: dict[str, str] | None = None,
                                       trusted_phase_a_authorization: dict[str, str] | None = None,
                                       trusted_phase_a_admission_lineage: dict[str, Any] | None = None) -> None:
    if report.get("phase") == "phase-b" and "phase_a_admission_lineage" not in report:
        raise PilotError("canonical A3 admission lineage missing")
    if trusted_phase_a_admission_lineage is not None and report.get("phase_a_admission_lineage") != trusted_phase_a_admission_lineage:
        raise PilotError("Phase B admission lineage is not the retained pre-training lineage")
    expected_report_keys = _A2_REPORT_KEYS | (frozenset({"phase_a_admission_lineage"}) if report.get("phase") == "phase-b" else frozenset())
    _require_exact_keys(report, expected_report_keys, "report")
    if report["status"] != "ok" or report["attempt"] != 3 or report["namespace"] != ATTEMPT3_NAMESPACE or report["phase"] not in ("phase-a", "phase-b"):
        raise PilotError("canonical A3 report identity mismatch")
    if report["phase"] != phase_spec.get("phase", report["phase"]) or phase_spec.get("attempt") != 3:
        raise PilotError("canonical A3 report phase mismatch")
    paths = phase_spec.get("namespace_paths")
    if not isinstance(paths, dict) or not paths:
        raise PilotError("canonical A3 namespace paths missing")
    phase = report["phase"]
    prefix = phase
    if report["output_path"] != paths[f"{prefix}-output"] or report["log_path"] != paths[f"{prefix}-log"] or report["report_path"] != paths[f"{prefix}-report"]:
        raise PilotError("canonical A3 output/log/report binding mismatch")
    if report["ok_marker_path"] != paths[f"{prefix}-ok"] or report["fail_marker_path"] != paths[f"{prefix}-fail"] or report["final_report_path"] != paths["final-report"]:
        raise PilotError("canonical A3 marker/final binding mismatch")
    if report["namespace_paths"] != paths:
        raise PilotError("canonical A3 namespace path substitution")
    if phase_spec.get("report") != paths[f"{phase}-report"] or phase_spec.get("final_report") != paths["final-report"]:
        raise PilotError("canonical A3 report namespace binding missing")
    _validate_canonical_effective(report["effective"], report, phase_spec)
    _validate_canonical_identity(report["identity_manifest"],
                                 trusted_identity=trusted_identity or phase_spec.get("trusted_identity"))
    root = pathlib.Path(paths[f"{phase}-report"]).resolve().parents[2]
    expected_authorization = validate_attempt3_authorization(
        report["authorization"], phase=phase, phase_spec=phase_spec, repo_root=root,
        verify_sources=False)
    if trusted_authorization is not None and expected_authorization != trusted_authorization:
        raise PilotError("canonical A3 authorization substituted")
    if phase == "phase-b":
        _validate_phase_a_admission_lineage(
            report["phase_a_admission_lineage"], phase_spec=phase_spec,
            trusted_authorization=trusted_phase_a_authorization)
    expected_history = historical_attempt2_evidence if historical_attempt2_evidence else report.get("attempt_2_historical_evidence")
    if not isinstance(expected_history, dict) or report["attempt_2_historical_evidence"] != expected_history:
        raise PilotError("canonical A3 attempt-2 historical evidence substituted")
    expected_artifacts = _validate_artifacts(phase, pathlib.Path(phase_spec["adapter_path"]), phase_spec=phase_spec)
    _validate_canonical_step_records(phase, report, phase_spec, expected_artifacts)
    if report["provider_calls"] != phase_spec["iters"] or report["optimizer_updates"] != phase_spec["iters"]:
        raise PilotError("canonical A3 provider/update cardinality mismatch")
    if report["global_mapping"] != [0, *global_steps(phase, range(1, phase_spec["iters"] + 1))]:
        raise PilotError("canonical A3 global mapping mismatch")
    if report["watchdog_cancelled"] is not True or report["exit_code"] != 0:
        raise PilotError("canonical A3 lifecycle evidence missing")
    _a2_finite_number(report["wall_seconds"], "phase wall time", minimum=0.0)
    if report["attempt_1_historical_evidence"] != (historical_evidence if historical_evidence is not None else list(ATTEMPT1_EVIDENCE_SHA256)):
        raise PilotError("canonical A3 historical evidence substituted")
    if report["artifacts"] != expected_artifacts:
        raise PilotError("canonical A3 artifact/checkpoint evidence mismatch")
    if any(item.get("checkpoint") != expected_artifacts["checkpoints"][index] for index, item in enumerate(report["steps"])):
        raise PilotError("canonical A3 update/checkpoint binding mismatch")
    expected_resume = expected_artifacts["checkpoints"][-1] if phase == "phase-a" else canonical_tensor_digest(phase_spec["resume_adapter_file"])
    if report["resume_source"] != expected_resume:
        raise PilotError("canonical A3 resume checkpoint binding mismatch")
    if report["contract_digest"] != contract_digest(
            phase, report["effective"], report["identity_manifest"], phase_spec=phase_spec,
            historical_evidence=report["attempt_1_historical_evidence"],
            historical_attempt2_evidence=report["attempt_2_historical_evidence"],
            authorization=report["authorization"],
            phase_a_admission_lineage=report.get("phase_a_admission_lineage")):
        raise PilotError("canonical A3 contract digest mismatch")
    if report["commands"] != report["effective"]["command"] or report["retry"] != RETRY_POLICY or report["fallback"] != FALLBACK_POLICY or report["non_claims"] != list(NON_CLAIMS):
        raise PilotError("canonical A3 command or policy binding mismatch")
    if report_path is not None and pathlib.Path(report_path).resolve() != pathlib.Path(paths[f"{phase}-report"]).resolve():
        raise PilotError("canonical A3 report path argument mismatch")


def validate_phase_b_dependency(report: dict[str, Any], marker: dict[str, Any], resume_source: pathlib.Path | str,
                                 phase_spec: dict[str, Any] | None = None,
                                 historical_evidence: list[dict[str, Any]] | None = None,
                                 historical_attempt2_evidence: dict[str, Any] | None = None,
                                 trusted_identity: dict[str, Any] | None = None,
                                 trusted_authorization: dict[str, str] | None = None,
                                 trusted_phase_a_authorization: dict[str, str] | None = None) -> None:
    if report.get("status") != "ok" or report.get("phase") != "phase-a" or marker.get("status") != "ok" or marker.get("phase") != "phase-a":
        raise PilotError("Phase A report and OK marker required")
    a2_paths: dict[str, str] | None = None
    if phase_spec is not None and phase_spec.get("attempt") == 3:
        a2_paths = phase_spec.get("namespace_paths")
        if not isinstance(a2_paths, dict) or not a2_paths:
            raise PilotError("canonical A3 namespace paths missing")
        if report.get("attempt") != 3 or report.get("namespace") != ATTEMPT3_NAMESPACE:
            raise PilotError("attempt-1 report cannot satisfy attempt-3 dependency")
        if marker.get("attempt") != 3 or marker.get("namespace") != ATTEMPT3_NAMESPACE:
            raise PilotError("attempt-1 marker cannot satisfy attempt-3 dependency")
    if report.get("provider_calls") != 2 or report.get("optimizer_updates") != 2:
        raise PilotError("Phase A report cardinality binding missing")
    if phase_spec is not None and phase_spec.get("attempt") == 3:
        if not isinstance(report.get("provider_evidence"), list):
            raise PilotError("Phase A canonical provider evidence missing")
    report_path = marker.get("report_path")
    expected_report_sha = marker.get("report_sha256")
    if a2_paths is not None:
        canonical_report_path = pathlib.Path(a2_paths["phase-a-report"])
        if pathlib.Path(resume_source) != pathlib.Path(a2_paths["phase-b-resume"]):
            raise PilotError("Phase A resume source path is not the canonical Phase B resume")
    else:
        canonical_report_path = pathlib.Path((phase_spec or PHASES["phase-a"])["report"])
        if not canonical_report_path.is_absolute():
            canonical_report_path = REPO_ROOT / canonical_report_path
    canonical_report_path = canonical_report_path.resolve()
    if not report_path or pathlib.Path(report_path).resolve() != canonical_report_path:
        raise PilotError("Phase A marker report path is not the canonical report")
    if not expected_report_sha:
        raise PilotError("Phase A report hash binding mismatch")
    observed_report_sha, hashed_report = _read_json_snapshot(canonical_report_path)
    if observed_report_sha != expected_report_sha:
        raise PilotError("Phase A report hash binding mismatch")
    if hashed_report != report:
        raise PilotError("Phase A report object does not equal hashed payload")
    if phase_spec is not None and phase_spec.get("attempt") == 3:
        _validate_attempt3_marker(marker, phase="phase-a", phase_spec=phase_spec,
                                  report=report, marker_path=pathlib.Path(a2_paths["phase-a-ok"]))
        validate_canonical_attempt3_report(report, phase_spec=phase_spec,
                                           report_path=canonical_report_path,
                                           historical_evidence=historical_evidence,
                                           historical_attempt2_evidence=historical_attempt2_evidence,
                                           trusted_identity=trusted_identity,
                                           trusted_authorization=trusted_authorization,
                                           trusted_phase_a_authorization=trusted_phase_a_authorization)
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
    observed_source = canonical_tensor_digest(source)
    if observed_source["file_sha256"] != source_info.get("file_sha256") or observed_source["canonical_tensor_digest_v1"] != source_info.get("canonical_tensor_digest_v1"):
        raise PilotError("Phase A resume source canonical digest mismatch")


def _effective(args: argparse.Namespace) -> dict[str, Any]:
    keys = set(COMMON_VALUES) | {"phase", "model", "data", "config", "adapter_path", "resume_adapter_file", "train", "test", "hf_dataset"}
    if getattr(args, "attempt", 1) != 1:
        keys |= {"attempt", "log_path"}
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
    if getattr(args, "attempt", 1) == 2:
        effective["command"] = canonical_attempt3_command(args.phase, phase)
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


def _validate_final_phase_lineage(report: dict[str, Any], phase: str) -> None:
    expected_keys = _A2_REPORT_KEYS | (frozenset({"phase_a_admission_lineage"}) if phase == "phase-b" else frozenset())
    _require_exact_keys(report, expected_keys, f"final {phase} report")
    if type(report["status"]) is not str or report["status"] != "ok" or type(report["attempt"]) is not int or report["attempt"] != 3 or type(report["namespace"]) is not str or report["namespace"] != ATTEMPT3_NAMESPACE:
        raise PilotError(f"final {phase} report is not canonical attempt-3 evidence")
    if type(report["phase"]) is not str or report["phase"] != phase:
        raise PilotError(f"final {phase} report phase binding mismatch")
    paths = report["namespace_paths"]
    if not isinstance(paths, dict) or type(report["report_path"]) is not str or report["report_path"] != paths.get(f"{phase}-report"):
        raise PilotError(f"final {phase} report path binding mismatch")
    if type(report["contract_digest"]) is not str or not _HEX64.fullmatch(report["contract_digest"]):
        raise PilotError(f"final {phase} contract binding missing")
    if not isinstance(report["authorization"], dict) or _authorization_json(report["authorization"]) != report["authorization"]:
        raise PilotError(f"final {phase} authorization schema mismatch")
    if not isinstance(report["attempt_1_historical_evidence"], list) or not isinstance(report["attempt_2_historical_evidence"], dict):
        raise PilotError(f"final {phase} historical evidence schema mismatch")
    if type(report["wall_seconds"]) not in (int, float) or not math.isfinite(float(report["wall_seconds"])) or report["wall_seconds"] < 0:
        raise PilotError(f"final {phase} wall time schema mismatch")
    if not isinstance(report["output_path"], str) or not isinstance(report["steps"], list):
        raise PilotError(f"final {phase} output/step schema mismatch")
    if phase == "phase-b" and not isinstance(report["phase_a_admission_lineage"], dict):
        raise PilotError("final Phase B admission lineage schema mismatch")


def _validate_final_phase_snapshot(report: dict[str, Any], phase: str, *,
                                   trusted_authorization: dict[str, str],
                                   trusted_identity: dict[str, Any],
                                   trusted_attempt_1_historical_evidence: list[dict[str, Any]],
                                   trusted_attempt_2_historical_evidence: dict[str, Any],
                                   trusted_phase_a_authorization: dict[str, str],
                                   trusted_phase_a_admission_lineage: dict[str, Any]
                                   ) -> tuple[dict[str, Any], str]:
    if not isinstance(report, dict) or type(report.get("report_path")) is not str:
        raise PilotError(f"final {phase} report path missing")
    report_path = pathlib.Path(report["report_path"])
    _, report_sha256, snapshot = _read_json_bytes_snapshot(report_path)
    if not _json_exact_equal(snapshot, report):
        raise PilotError(f"final {phase} supplied report does not equal immutable snapshot")
    paths = snapshot.get("namespace_paths")
    if not isinstance(paths, dict) or type(paths.get(f"{phase}-output")) is not str:
        raise PilotError(f"final {phase} namespace paths missing")
    root = pathlib.Path(paths[f"{phase}-report"]).resolve().parents[2]
    workspace = pathlib.Path(paths[f"{phase}-output"]).resolve().parent
    phase_spec = attempt3_phase_specs(repo_root=root, workspace=workspace)[phase]
    validate_canonical_attempt3_report(
        snapshot, phase_spec=phase_spec, report_path=report_path,
        historical_evidence=trusted_attempt_1_historical_evidence,
        historical_attempt2_evidence=trusted_attempt_2_historical_evidence,
        trusted_identity=trusted_identity,
        trusted_authorization=trusted_authorization,
        trusted_phase_a_authorization=trusted_phase_a_authorization,
        trusted_phase_a_admission_lineage=(trusted_phase_a_admission_lineage
                                           if phase == "phase-b" else None),
    )
    return snapshot, report_sha256


def validate_final_attempt3_report(final: dict[str, Any], phase_a: dict[str, Any], phase_b: dict[str, Any],
                                   *, trusted_phase_a_admission_lineage: dict[str, Any] | None = None,
                                   trusted_phase_a_authorization: dict[str, str] | None = None,
                                   trusted_phase_b_authorization: dict[str, str] | None = None,
                                   trusted_identity: dict[str, Any] | None = None,
                                   trusted_attempt_1_historical_evidence: list[dict[str, Any]] | None = None,
                                   trusted_attempt_2_historical_evidence: dict[str, Any] | None = None) -> None:
    expected_keys = frozenset({
        "status", "provider_calls", "optimizer_updates", "steps", "global_progression",
        "total_active_wall_seconds", "phase_a", "phase_b", "output_path", "contract_digest", "non_claims",
        "attempt", "namespace", "phase_a_report_path", "phase_b_report_path",
        "phase_a_contract_digest", "phase_b_contract_digest", "phase_authorizations",
        "attempt_1_historical_evidence", "attempt_2_historical_evidence",
        "phase_a_report_sha256", "phase_b_report_sha256",
    })
    if any(value is None for value in (
        trusted_phase_a_admission_lineage, trusted_phase_a_authorization,
        trusted_phase_b_authorization, trusted_identity,
        trusted_attempt_1_historical_evidence, trusted_attempt_2_historical_evidence,
    )):
        raise PilotError("final publication requires independently retained trusted A3/B3 authorization, identity, and history roots")
    _require_exact_keys(final, expected_keys, "final report")
    phase_a, phase_a_sha256 = _validate_final_phase_snapshot(
        phase_a, "phase-a", trusted_authorization=trusted_phase_a_authorization,
        trusted_identity=trusted_identity,
        trusted_attempt_1_historical_evidence=trusted_attempt_1_historical_evidence,
        trusted_attempt_2_historical_evidence=trusted_attempt_2_historical_evidence,
        trusted_phase_a_authorization=trusted_phase_a_authorization,
        trusted_phase_a_admission_lineage=trusted_phase_a_admission_lineage)
    phase_b, phase_b_sha256 = _validate_final_phase_snapshot(
        phase_b, "phase-b", trusted_authorization=trusted_phase_b_authorization,
        trusted_identity=trusted_identity,
        trusted_attempt_1_historical_evidence=trusted_attempt_1_historical_evidence,
        trusted_attempt_2_historical_evidence=trusted_attempt_2_historical_evidence,
        trusted_phase_a_authorization=trusted_phase_a_authorization,
        trusted_phase_a_admission_lineage=trusted_phase_a_admission_lineage)
    if final["phase_a_report_sha256"] != phase_a_sha256 or final["phase_b_report_sha256"] != phase_b_sha256:
        raise PilotError("final phase report hash binding mismatch")
    _validate_final_phase_lineage(phase_a, "phase-a")
    _validate_final_phase_lineage(phase_b, "phase-b")
    trusted_phase_a_authorization_json = _authorization_json(trusted_phase_a_authorization)
    trusted_phase_b_authorization_json = _authorization_json(trusted_phase_b_authorization)
    if not _json_exact_equal(phase_a["authorization"], trusted_phase_a_authorization_json):
        raise PilotError("final trusted Phase A3 authorization mismatch")
    if not _json_exact_equal(phase_b["authorization"], trusted_phase_b_authorization_json):
        raise PilotError("final trusted Phase B3 authorization mismatch")
    if phase_a["identity_manifest"] != trusted_identity or phase_b["identity_manifest"] != trusted_identity:
        raise PilotError("final trusted cross-phase identity mismatch")
    if phase_a["attempt_1_historical_evidence"] != trusted_attempt_1_historical_evidence or phase_b["attempt_1_historical_evidence"] != trusted_attempt_1_historical_evidence:
        raise PilotError("final trusted attempt-1 historical evidence mismatch")
    if phase_a["attempt_2_historical_evidence"] != trusted_attempt_2_historical_evidence or phase_b["attempt_2_historical_evidence"] != trusted_attempt_2_historical_evidence:
        raise PilotError("final trusted attempt-2 historical evidence mismatch")
    if not _json_exact_equal(phase_b["phase_a_admission_lineage"], trusted_phase_a_admission_lineage):
        raise PilotError("final Phase A admission lineage mismatch")
    if not _json_exact_equal(final["status"], "ok") or type(final["attempt"]) is not int or final["attempt"] != 3 or type(final["namespace"]) is not str or final["namespace"] != ATTEMPT3_NAMESPACE:
        raise PilotError("final report identity mismatch")
    expected_steps = phase_a["steps"] + phase_b["steps"]
    expected_total = float(phase_a["wall_seconds"]) + float(phase_b["wall_seconds"])
    expected_contract = hashlib.sha256(json.dumps(
        {"phase_a": phase_a["contract_digest"], "phase_b": phase_b["contract_digest"],
         "phase_authorizations": {"phase-a": trusted_phase_a_authorization_json, "phase-b": trusted_phase_b_authorization_json}},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected = {
        "status": "ok", "provider_calls": phase_a["provider_calls"] + phase_b["provider_calls"],
        "optimizer_updates": phase_a["optimizer_updates"] + phase_b["optimizer_updates"],
        "steps": expected_steps, "global_progression": [0, 1, 2, 3],
        "total_active_wall_seconds": expected_total, "phase_a": phase_a, "phase_b": phase_b,
        "output_path": phase_b["output_path"], "contract_digest": expected_contract,
        "non_claims": list(NON_CLAIMS), "attempt": 3, "namespace": ATTEMPT3_NAMESPACE,
        "phase_a_report_path": phase_a["report_path"], "phase_b_report_path": phase_b["report_path"],
        "phase_a_contract_digest": phase_a["contract_digest"], "phase_b_contract_digest": phase_b["contract_digest"],
        "phase_authorizations": {"phase-a": trusted_phase_a_authorization_json, "phase-b": trusted_phase_b_authorization_json},
        "attempt_1_historical_evidence": trusted_attempt_1_historical_evidence,
        "attempt_2_historical_evidence": trusted_attempt_2_historical_evidence,
        "phase_a_report_sha256": phase_a_sha256,
        "phase_b_report_sha256": phase_b_sha256,
    }
    for key, expected_value in expected.items():
        if key == "total_active_wall_seconds":
            if type(final[key]) not in (int, float) or not math.isfinite(float(final[key])) or final[key] != expected_value:
                raise PilotError(f"final {key} type/value mismatch")
        elif not _json_exact_equal(final[key], expected_value):
            raise PilotError(f"final {key} type/value mismatch")
    if phase_a["authorization"] == phase_b["authorization"]:
        raise PilotError("final phase authorizations must remain phase-specific")


def aggregate_reports(phase_a: dict[str, Any], phase_b: dict[str, Any], *, strict_lineage: bool = False) -> dict[str, Any]:
    if phase_a.get("status") != "ok" or phase_b.get("status") != "ok":
        raise PilotError("final report requires both phases to pass")
    strict_lineage = strict_lineage or phase_a.get("attempt") == 3 or phase_b.get("attempt") == 3
    if strict_lineage:
        _validate_final_phase_lineage(phase_a, "phase-a")
        _validate_final_phase_lineage(phase_b, "phase-b")
        if phase_a["authorization"] == phase_b["authorization"]:
            raise PilotError("final phase authorizations must remain phase-specific")
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
        {"phase_a": phase_a.get("contract_digest", ""), "phase_b": phase_b.get("contract_digest", ""),
         "phase_authorizations": {"phase-a": phase_a.get("authorization"), "phase-b": phase_b.get("authorization")}},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    final = {"status": "ok", "provider_calls": phase_a.get("provider_calls", 0) + phase_b.get("provider_calls", 0),
             "optimizer_updates": phase_a.get("optimizer_updates", 0) + phase_b.get("optimizer_updates", 0),
             "steps": steps, "global_progression": [0, 1, 2, 3], "total_active_wall_seconds": total,
             "phase_a": phase_a, "phase_b": phase_b, "output_path": phase_b.get("output_path", PHASES["phase-b"]["adapter_path"]),
             "contract_digest": final_contract, "non_claims": list(NON_CLAIMS)}
    if strict_lineage:
        final.update({"attempt": 3, "namespace": ATTEMPT3_NAMESPACE,
                      "phase_a_report_path": phase_a["report_path"], "phase_b_report_path": phase_b["report_path"],
                      "phase_a_contract_digest": phase_a["contract_digest"], "phase_b_contract_digest": phase_b["contract_digest"],
                      "phase_authorizations": {"phase-a": phase_a["authorization"], "phase-b": phase_b["authorization"]},
                      "attempt_1_historical_evidence": phase_a["attempt_1_historical_evidence"],
                      "attempt_2_historical_evidence": phase_a["attempt_2_historical_evidence"]})
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, choices=(1, 3), default=1)
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
    parser.add_argument("--launch-check-only", action="store_true")
    parser.add_argument("--authorization-json")
    parser.add_argument("--phase-a-authorization-json")
    parser.add_argument("--log-path")
    parser.add_argument("--log-fd", type=int)
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
    if not schema or len({item["path"] for item in schema}) != len(schema):
        raise PilotError("trainable schema must contain unique named leaves")
    if any(set(item) != {"path", "shape", "dtype"} or type(item["path"]) is not str or not item["path"]
           or not isinstance(item["shape"], list) or any(type(dim) is not int or dim < 0 for dim in item["shape"])
           or type(item["dtype"]) is not str or not item["dtype"] for item in schema):
        raise PilotError("trainable schema key/dtype/shape mismatch")
    return schema


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _array_meta(path: str, value: Any) -> _ArrayMeta:
    if type(path) is not str or not path:
        raise PilotError("MLX trainable leaf path must be a nonempty string")
    raw_shape = getattr(value, "shape", ())
    if not isinstance(raw_shape, (tuple, list)) or any(type(item) is not int or item < 0 for item in raw_shape):
        raise PilotError("MLX trainable leaf shape schema mismatch")
    shape = tuple(raw_shape)
    raw_dtype = getattr(value, "dtype", None)
    if raw_dtype is None:
        raise PilotError("MLX trainable leaf dtype schema missing")
    dtype = str(raw_dtype)
    if not dtype or dtype == "None":
        raise PilotError("MLX trainable leaf dtype schema mismatch")
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


def contract_digest(phase: str, effective: dict[str, Any], identity: dict[str, Any] | None = None,
                    phase_spec: dict[str, Any] | None = None,
                    historical_evidence: list[dict[str, Any]] | None = None,
                    historical_attempt2_evidence: dict[str, Any] | None = None,
                    authorization: dict[str, str] | None = None,
                    phase_a_admission_lineage: dict[str, Any] | None = None) -> str:
    identity_value = _immutable_identity(identity or {})
    spec = phase_spec or PHASES[phase]
    payload = {"phase": phase, "common": COMMON_VALUES, "phase_spec": spec,
               "effective": effective, "identity": identity_value}
    if spec.get("attempt") == 3:
        payload["attempt"] = 3
        payload["namespace"] = ATTEMPT3_NAMESPACE
        payload["attempt_1_historical_evidence_manifest"] = (
            list(historical_evidence) if historical_evidence is not None else list(ATTEMPT1_EVIDENCE_SHA256))
        payload["attempt_2_historical_evidence_manifest"] = (
            historical_attempt2_evidence if historical_attempt2_evidence is not None else {})
        payload["authorization"] = authorization or {}
        payload["active_log_binding"] = effective.get("log_path")
        if phase_a_admission_lineage is not None:
            payload["phase_a_admission_lineage"] = phase_a_admission_lineage
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
        phase_start = pathlib.Path(phase.get("phase_start_adapter_file") or
                                   (pathlib.Path(PHASES["phase-a"]["adapter_path"]) / "phase-a-start.safetensors"))
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


def _validate_artifacts(phase_name: str, output: pathlib.Path,
                        phase_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    phase = phase_spec or PHASES[phase_name]
    if phase.get("attempt") == 3:
        paths = phase.get("namespace_paths")
        if not isinstance(paths, dict) or not paths:
            raise PilotError("canonical A3 namespace paths missing")
        expected_output = pathlib.Path(paths[f"{phase_name}-output"])
        if output != expected_output:
            raise PilotError(f"canonical A3 artifact output binding mismatch: {output}")
        start = pathlib.Path(paths[f"{phase_name}-start-checkpoint"])
        checkpoints = [pathlib.Path(paths[f"{phase_name}-step{step}-checkpoint"])
                       for step in range(1, phase["iters"] + 1)]
        final = pathlib.Path(paths[f"{phase_name}-final-checkpoint"])
        config = pathlib.Path(paths[f"{phase_name}-config"])
    else:
        start = output / ("phase-a-start.safetensors" if phase_name == "phase-a" else "resume-start.safetensors")
        checkpoints = [output / f"{step:07d}_adapters.safetensors" for step in range(1, phase["iters"] + 1)]
        final = output / "adapters.safetensors"
        config = output / "adapter_config.json"
    for path in [start, *checkpoints, final, config]:
        if not path.is_file():
            raise PilotError(f"missing required pilot artifact: {path}")
    numbered = {path.name for path in output.glob("*_adapters.safetensors")}
    expected_numbered = {path.name for path in checkpoints}
    if numbered != expected_numbered:
        raise PilotError("checkpoint cardinality or unexpected numbered checkpoint")
    infos = {"start": _artifact_info(start), "checkpoints": [_artifact_info(path) for path in checkpoints], "final": _artifact_info(final),
             "config": {"path": str(config), "size": config.stat().st_size, "file_sha256": file_sha256(config)}}
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


def _phase_spec(phase: str, attempt: int = 1) -> dict[str, Any]:
    if attempt == 1:
        if phase not in PHASES:
            raise PilotError(f"unknown phase: {phase}")
        return PHASES[phase]
    if attempt == 3:
        return attempt3_phase_specs()[phase]
    raise PilotError("only attempt 1 and attempt 3 are supported")


def _marker_path(phase: str, status: str, phase_spec: dict[str, Any] | None = None) -> pathlib.Path:
    if phase_spec is not None and phase_spec.get("attempt") == 3:
        return pathlib.Path(phase_spec["namespace_paths"][f"{phase}-{status}"])
    return pathlib.Path(PILOT_WORKSPACE) / f".ds4-segmented-pilot-{phase}-{status}"


def _all_attempt_paths(phase: str, attempt: int = 1) -> list[pathlib.Path]:
    if attempt == 3:
        return _attempt3_absent_paths(attempt3_namespace())
    paths = [PILOT_FINAL_REPORT, pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-ok", pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-fail"]
    if phase == "phase-a":
        paths.extend(REPO_ROOT / spec["report"] for spec in PHASES.values())
        paths.extend(_marker_path(name, status) for name in PHASES for status in ("ok", "fail"))
    else:
        paths.extend([REPO_ROOT / PHASES[phase]["report"], _marker_path(phase, "ok"), _marker_path(phase, "fail")])
    return paths


def _check_attempt_gates(phase: str, attempt: int = 1, allow_active_log: bool = False,
                        authorization_json: str | dict[str, Any] | None = None,
                        trusted_identity: dict[str, Any] | None = None,
                        phase_a_authorization_json: str | dict[str, Any] | None = None) -> None:
    if attempt == 3:
        paths = attempt3_namespace()
        check_attempt3_launch(phase, paths[f"{phase}-log"], allow_active_log=allow_active_log,
                              authorization_json=authorization_json, trusted_identity=trusted_identity,
                              phase_a_authorization_json=phase_a_authorization_json)
        return
    for path in _all_attempt_paths(phase):
        ensure_absent(path)
    ensure_absent(PHASES[phase]["adapter_path"])


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError(f"identity command failed: {' '.join(args)}: {exc}") from exc


def _resource_skip_type(psutil: Any, error: BaseException) -> str | None:
    for name in ("ZombieProcess", "NoSuchProcess", "AccessDenied"):
        error_type = getattr(psutil, name, None)
        if error_type is not None and isinstance(error, error_type):
            return name
    return None


def _resource_skip_pid(error: BaseException, pid: int | None) -> int | None:
    if pid is not None:
        return pid
    candidate = getattr(error, "pid", None)
    return candidate if type(candidate) is int else None


def _resource_evidence(available: int, free: int, observed: list[list[int]],
                       skips: dict[str, Any]) -> dict[str, Any]:
    names = ("AccessDenied", "NoSuchProcess", "ZombieProcess")
    if set(skips) != set(names):
        raise ValueError("resource skip type schema mismatch")
    for name in names:
        if skips[name]["counts_by_type"] != len(skips[name]["pids_by_type"]) + skips[name]["unknown_pid_counts_by_type"]:
            raise ValueError("resource skip count mismatch")
    evidence = {"available_memory": available, "disk_free": free,
                "competing_processes": sorted(observed),
                "allowed_process_skips": {
                    "total": sum(skips[name]["counts_by_type"] for name in names),
                    "counts_by_type": {name: skips[name]["counts_by_type"] for name in names},
                    "pids_by_type": {name: sorted(skips[name]["pids_by_type"]) for name in names},
                    "unknown_pid_counts_by_type": {name: skips[name]["unknown_pid_counts_by_type"] for name in names},
                }}
    if evidence["allowed_process_skips"]["total"] != sum(evidence["allowed_process_skips"]["counts_by_type"].values()):
        raise ValueError("resource skip total mismatch")
    return evidence


def _resource_gate() -> dict[str, Any]:
    try:
        import psutil
    except Exception as exc:
        raise ResourceObserverError("import", exc) from exc
    try:
        available = int(psutil.virtual_memory().available)
    except Exception as exc:
        raise ResourceObserverError("virtual-memory", exc) from exc
    try:
        current_pid = os.getpid()
    except Exception as exc:
        raise ResourceObserverError("process-pid", exc) from exc
    names = ("AccessDenied", "NoSuchProcess", "ZombieProcess")
    skips = {name: {"counts_by_type": 0, "pids_by_type": [], "unknown_pid_counts_by_type": 0} for name in names}
    observed: list[list[int]] = []
    try:
        iterator = iter(psutil.process_iter(["pid"]))
    except Exception as exc:
        raise ResourceObserverError("process-enumeration", exc) from exc
    while True:
        try:
            process = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise ResourceObserverError("process-enumeration", exc) from exc
        pid: int | None = None
        try:
            pid = int(process.pid)
        except Exception as exc:
            skip_type = _resource_skip_type(psutil, exc)
            if skip_type is None:
                raise ResourceObserverError("process-pid", exc) from exc
            skip_pid = _resource_skip_pid(exc, None)
            skips[skip_type]["counts_by_type"] += 1
            if skip_pid is None:
                skips[skip_type]["unknown_pid_counts_by_type"] += 1
            else:
                skips[skip_type]["pids_by_type"].append(skip_pid)
            continue
        if pid == current_pid:
            continue
        try:
            rss = int(process.memory_info().rss)
        except Exception as exc:
            skip_type = _resource_skip_type(psutil, exc)
            if skip_type is None:
                raise ResourceObserverError("process-rss", exc, pid) from exc
            skip_pid = _resource_skip_pid(exc, pid)
            skips[skip_type]["counts_by_type"] += 1
            if skip_pid is None:
                skips[skip_type]["unknown_pid_counts_by_type"] += 1
            else:
                skips[skip_type]["pids_by_type"].append(skip_pid)
            continue
        observed.append([pid, rss])
    observed.sort(key=lambda item: (item[0], item[1]))
    competing = [item for item in observed if item[1] > 50 * 1024**3]
    if competing:
        raise PilotError(f"competing process exceeds RSS gate: {competing}")
    if available < PILOT_MEMORY_HEADROOM:
        raise PilotError("memory headroom below 32 GiB")
    try:
        free = shutil_disk_free(PILOT_WORKSPACE)
    except ResourceObserverError:
        raise
    except Exception as exc:
        raise ResourceObserverError("disk-free", exc) from exc
    if free < PILOT_DISK_MIN_FREE:
        raise PilotError("disk free below 1 GiB")
    try:
        return _resource_evidence(available, free, observed, skips)
    except Exception as exc:
        raise ResourceObserverError("resource-evidence", exc) from exc


def shutil_disk_free(path: str | pathlib.Path) -> int:
    import shutil
    try:
        return int(shutil.disk_usage(path).free)
    except Exception as exc:
        raise ResourceObserverError("disk-free", exc) from exc


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
        try:
            mlx_version = importlib.metadata.version("mlx")
        except importlib.metadata.PackageNotFoundError as exc:
            raise PilotError("MLX distribution metadata unavailable") from exc
        except Exception as exc:
            raise PilotError(f"MLX distribution metadata error: {type(exc).__name__}") from exc
        if type(mlx_version) is not str or not mlx_version:
            raise PilotError("MLX distribution metadata malformed: expected non-empty string")
        if mlx_version != "0.31.2":
            raise PilotError(f"MLX distribution metadata version mismatch: {mlx_version!r}")
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


def _phase_failure_report(phase_name: str, message: str | Exception, started: float, *, effective: dict[str, Any] | None = None,
                          identity: dict[str, Any] | None = None, contract: str | None = None,
                          output: pathlib.Path | None = None, lock_lifecycle: dict[str, Any] | None = None,
                          watchdog_cancelled: bool = False, phase_spec: dict[str, Any] | None = None,
                          historical_evidence: list[dict[str, Any]] | None = None,
                          historical_attempt2_evidence: dict[str, Any] | None = None,
                          authorization: dict[str, str] | None = None) -> dict[str, Any]:
    spec = phase_spec or PHASES[phase_name]
    report = {"status": "fail", "phase": phase_name, "attempt": spec.get("attempt", 1),
              "namespace": spec.get("namespace", "ds4-segmented-pilot"), "error": str(message), "wall_seconds": time.monotonic() - started,
              "effective": effective or {}, "commands": (effective or {}).get("command", []),
              "identity_manifest": identity or {}, "contract_digest": contract or "",
              "output_path": str(output or spec["adapter_path"]),
              "lock_lifecycle": lock_lifecycle or {}, "watchdog_cancelled": watchdog_cancelled,
              "exit_code": 1, "retry": RETRY_POLICY, "fallback": FALLBACK_POLICY, "non_claims": list(NON_CLAIMS)}
    if spec.get("attempt") == 3:
        report.update({"attempt_1_historical_evidence": historical_evidence or [],
                       "attempt_2_historical_evidence": historical_attempt2_evidence or {},
                       "authorization": authorization})
    if isinstance(message, ResourceObserverError):
        report["resource_observer_error"] = {"stage": message.stage, "exception_type": message.exception_type,
                                              "pid": message.pid}
    return report


def _ok_marker_paths(attempt: int = 1) -> list[pathlib.Path]:
    if attempt == 3:
        paths = attempt3_namespace()
        return [paths["phase-a-ok"], paths["phase-b-ok"], paths["final-ok"]]
    workspace = pathlib.Path(PILOT_WORKSPACE)
    return [_marker_path(phase, "ok") for phase in PHASES] + [workspace / ".ds4-segmented-pilot-ok"]


def _remove_ok_markers(attempt: int = 1) -> None:
    errors: list[Exception] = []
    for path in _ok_marker_paths(attempt):
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


def _marker_fields(phase: str, status: str, report_path: pathlib.Path, report: dict[str, Any],
                   phase_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = phase_spec or PHASES[phase]
    return {"phase": phase, "status": status, "attempt": spec.get("attempt", 1),
            "namespace": spec.get("namespace", "ds4-segmented-pilot"), "report_path": str(report_path),
            "report_sha256": file_sha256(report_path), "output_path": report.get("output_path", spec["adapter_path"]),
            "timestamp": time.time(), "exit_code": 0 if status == "ok" else 1,
            "contract_digest": report.get("contract_digest", "")}


def _validate_attempt3_marker(marker: dict[str, Any], *, phase: str,
                              phase_spec: dict[str, Any], report: dict[str, Any],
                              marker_path: pathlib.Path | str | None = None,
                              final: bool = False,
                              trusted_report_sha256: str | None = None) -> None:
    expected_keys = _ATTEMPT3_MARKER_PHASE_A_KEYS if phase == "phase-a" and not final else _ATTEMPT3_MARKER_BASE_KEYS
    _require_exact_keys(marker, expected_keys, "marker")
    paths = phase_spec.get("namespace_paths")
    if type(phase) is not str or phase not in ("phase-a", "phase-b") or phase_spec.get("attempt") != 3 or not isinstance(paths, dict):
        raise PilotError("canonical A3 marker phase spec missing")
    if type(marker["phase"]) is not str or marker["phase"] != phase or type(marker["status"]) is not str or marker["status"] != "ok":
        raise PilotError("canonical A3 marker identity mismatch")
    if type(marker["attempt"]) is not int or marker["attempt"] != 3:
        raise PilotError("canonical A3 marker attempt type/value mismatch")
    if type(marker["namespace"]) is not str or marker["namespace"] != ATTEMPT3_NAMESPACE:
        raise PilotError("canonical A3 marker namespace mismatch")
    report_key = "final-report" if final else f"{phase}-report"
    output_key = "phase-b-output" if final else f"{phase}-output"
    marker_key = "final-ok" if final else f"{phase}-ok"
    if type(marker["report_path"]) is not str or marker["report_path"] != paths[report_key] or type(marker["output_path"]) is not str or marker["output_path"] != paths[output_key]:
        raise PilotError("canonical A3 marker path binding mismatch")
    if type(marker["report_sha256"]) is not str or not _HEX64.fullmatch(marker["report_sha256"]):
        raise PilotError("canonical A3 marker report hash schema mismatch")
    observed_report_sha256 = (trusted_report_sha256 if trusted_report_sha256 is not None
                               else file_sha256(marker["report_path"]))
    if marker["report_sha256"] != observed_report_sha256:
        raise PilotError("canonical A3 marker report hash mismatch")
    if trusted_report_sha256 is None and marker["report_sha256"] != file_sha256(paths[report_key]):
        raise PilotError("canonical A3 marker report path mismatch")
    if type(marker["timestamp"]) not in (int, float) or not math.isfinite(float(marker["timestamp"])) or marker["timestamp"] <= 0:
        raise PilotError("canonical A3 marker timestamp schema mismatch")
    if type(marker["exit_code"]) is not int or marker["exit_code"] != 0:
        raise PilotError("canonical A3 marker exit code schema mismatch")
    if type(marker["contract_digest"]) is not str or not _HEX64.fullmatch(marker["contract_digest"]) or marker["contract_digest"] != report.get("contract_digest"):
        raise PilotError("canonical A3 marker lifecycle/contract mismatch")
    if marker_path is not None and pathlib.Path(marker_path).resolve() != pathlib.Path(paths[marker_key]).resolve():
        raise PilotError("canonical A3 marker path argument mismatch")
    if phase == "phase-a" and not final:
        source = report.get("resume_source")
        if not isinstance(source, dict) or type(marker["resume_source"]) is not str or marker["resume_source"] != source.get("path"):
            raise PilotError("canonical A3 marker resume path mismatch")
        if type(marker["resume_source_file_sha256"]) is not str or not _HEX64.fullmatch(marker["resume_source_file_sha256"]) or marker["resume_source_file_sha256"] != source.get("file_sha256"):
            raise PilotError("canonical A3 marker resume file hash mismatch")
        if type(marker["resume_source_canonical_tensor_digest_v1"]) is not str or not _HEX64.fullmatch(marker["resume_source_canonical_tensor_digest_v1"]) or marker["resume_source_canonical_tensor_digest_v1"] != source.get("canonical_tensor_digest_v1"):
            raise PilotError("canonical A3 marker resume digest mismatch")


def capture_phase_a_admission_lineage(report_path: pathlib.Path | str, marker_path: pathlib.Path | str,
                                     report: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    report_bytes, report_sha256, report_snapshot = _read_json_bytes_snapshot(report_path)
    marker_bytes, marker_sha256, marker_snapshot = _read_json_bytes_snapshot(marker_path)
    if report_snapshot != report or marker_snapshot != marker:
        raise PilotError("Phase A admission snapshot does not equal supplied evidence")
    return {
        "report": {"path": str(report_path), "sha256": report_sha256,
                   "bytes_b64": base64.b64encode(report_bytes).decode("ascii"),
                   "schema": sorted(report_snapshot)},
        "marker": {"path": str(marker_path), "sha256": marker_sha256,
                   "bytes_b64": base64.b64encode(marker_bytes).decode("ascii"),
                   "schema": sorted(marker_snapshot)},
        "authorization": json.loads(json.dumps(report_snapshot.get("authorization"))),
        "attempt_1_historical_evidence": json.loads(json.dumps(report_snapshot.get("attempt_1_historical_evidence"))),
        "attempt_2_historical_evidence": json.loads(json.dumps(report_snapshot.get("attempt_2_historical_evidence"))),
        "identity_manifest": json.loads(json.dumps(report_snapshot.get("identity_manifest"))),
        "contract_digest": report_snapshot.get("contract_digest"),
        "artifacts": json.loads(json.dumps(report_snapshot.get("artifacts"))),
        "resume_source": json.loads(json.dumps(report_snapshot.get("resume_source"))),
    }


def _write_failure_evidence(phase: str, report: dict[str, Any], phase_spec: dict[str, Any] | None = None) -> None:
    spec = phase_spec or PHASES[phase]
    attempt = int(spec.get("attempt", 1))
    # Roll back published success evidence before replacing any report it names.
    _remove_ok_markers(attempt)
    report_path = (pathlib.Path(spec["namespace_paths"][f"{phase}-report"])
                   if attempt == 3 else REPO_ROOT / spec["report"])
    final_path = (pathlib.Path(spec["namespace_paths"]["final-report"])
                  if attempt == 3 else PILOT_FINAL_REPORT)
    atomic_write_json(report_path, report)
    final = dict(report)
    final["phase_report"] = str(report_path)
    atomic_write_json(final_path, final)
    errors: list[Exception] = []
    final_marker = (pathlib.Path(spec["namespace_paths"]["final-fail"])
                    if attempt == 3 else pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-fail")
    for marker_path, marker in (
        (_marker_path(phase, "fail", spec), _marker_fields(phase, "fail", report_path, report, spec)),
        (final_marker, _marker_fields(phase, "fail", final_path, final, spec)),
    ):
        try:
            atomic_write_json(marker_path, marker)
        except Exception as exc:
            errors.append(exc)
    if errors:
        _remove_ok_markers(attempt)
        raise PilotError("failure evidence write failed: " + "; ".join(str(exc) for exc in errors)) from errors[0]


def _write_success_evidence(phase: str, report: dict[str, Any], phase_spec: dict[str, Any] | None = None,
                            phase_a_authorization: dict[str, str] | None = None,
                            phase_a_admission_lineage: dict[str, Any] | None = None,
                            phase_b_authorization: dict[str, str] | None = None,
                            trusted_identity: dict[str, Any] | None = None,
                            trusted_attempt_1_historical_evidence: list[dict[str, Any]] | None = None,
                            trusted_attempt_2_historical_evidence: dict[str, Any] | None = None) -> None:
    spec = phase_spec or PHASES[phase]
    attempt = int(spec.get("attempt", 1))
    try:
        report_path = (pathlib.Path(spec["namespace_paths"][f"{phase}-report"])
                       if attempt == 3 else REPO_ROOT / spec["report"])
        final_path = (pathlib.Path(spec["namespace_paths"]["final-report"])
                      if attempt == 3 else PILOT_FINAL_REPORT)
        atomic_write_json(report_path, report)
        marker = _marker_fields(phase, "ok", report_path, report, spec)
        if phase == "phase-a":
            source = report.get("resume_source")
            if source:
                marker["resume_source"] = source["path"]
                marker["resume_source_file_sha256"] = source["file_sha256"]
                marker["resume_source_canonical_tensor_digest_v1"] = source["canonical_tensor_digest_v1"]
        else:
            if attempt != 3:
                phase_a_report_path = REPO_ROOT / PHASES["phase-a"]["report"]
                phase_a = json.loads(phase_a_report_path.read_text(encoding="utf-8"))
                final = aggregate_reports(phase_a, report)
                atomic_write_json(final_path, final)
            else:
                phase_root = pathlib.Path(spec["namespace_paths"]["phase-a-report"]).resolve().parents[2]
                phase_workspace = pathlib.Path(spec["namespace_paths"]["phase-a-output"]).resolve().parent
                phase_a_spec = attempt3_phase_specs(repo_root=phase_root, workspace=phase_workspace)["phase-a"]
                phase_a_report_path = pathlib.Path(phase_a_spec["namespace_paths"]["phase-a-report"])
                phase_a_report_sha, phase_a = _read_json_snapshot(phase_a_report_path)
                phase_a_marker_path = pathlib.Path(spec["namespace_paths"]["phase-a-ok"])
                _, phase_a_marker = _read_json_snapshot(phase_a_marker_path)
                if phase_a_authorization is None or phase_a_admission_lineage is None:
                    raise PilotError("Phase B requires explicit Phase A authorization and admission lineage")
                if report.get("phase_a_admission_lineage") != phase_a_admission_lineage:
                    raise PilotError("Phase B admission lineage is not the retained pre-training lineage")
                validate_phase_b_dependency(
                    phase_a, phase_a_marker, pathlib.Path(spec["namespace_paths"]["phase-b-resume"]),
                    phase_spec=phase_a_spec,
                    historical_evidence=phase_a.get("attempt_1_historical_evidence"),
                    historical_attempt2_evidence=phase_a.get("attempt_2_historical_evidence"),
                    trusted_identity=phase_a.get("identity_manifest"),
                    trusted_phase_a_authorization=phase_a_authorization,
                )
                phase_b_report_sha, phase_b_snapshot = _read_json_snapshot(report_path)
                validate_canonical_attempt3_report(
                    phase_b_snapshot,
                    phase_spec=spec,
                    report_path=report_path,
                    historical_evidence=phase_a.get("attempt_1_historical_evidence"),
                    historical_attempt2_evidence=phase_a.get("attempt_2_historical_evidence"),
                    trusted_identity=phase_a.get("identity_manifest"),
                    trusted_phase_a_authorization=phase_a_authorization,
                )
                final = aggregate_reports(phase_a, phase_b_snapshot, strict_lineage=True)
                final.update({"phase_a_report_sha256": phase_a_report_sha, "phase_b_report_sha256": phase_b_report_sha})
                validate_final_attempt3_report(
                    final, phase_a, phase_b_snapshot,
                    trusted_phase_a_admission_lineage=phase_a_admission_lineage,
                    trusted_phase_a_authorization=phase_a_authorization,
                    trusted_phase_b_authorization=phase_b_authorization,
                    trusted_identity=trusted_identity,
                    trusted_attempt_1_historical_evidence=trusted_attempt_1_historical_evidence,
                    trusted_attempt_2_historical_evidence=trusted_attempt_2_historical_evidence,
                )
                atomic_write_json(final_path, final)
        if attempt == 3:
            _validate_attempt3_marker(marker, phase=phase, phase_spec=spec,
                                      report=report, marker_path=_marker_path(phase, "ok", spec))
        atomic_write_json(_marker_path(phase, "ok", spec), marker)
        if phase == "phase-b":
            final_marker = (pathlib.Path(spec["namespace_paths"]["final-ok"])
                            if attempt == 3 else pathlib.Path(PILOT_WORKSPACE) / ".ds4-segmented-pilot-ok")
            final_marker_payload = _marker_fields(phase, "ok", final_path, final, spec)
            if attempt == 3:
                _validate_attempt3_marker(final_marker_payload, phase="phase-b", phase_spec=spec,
                                          report=final, marker_path=final_marker, final=True)
            atomic_write_json(final_marker, final_marker_payload)
    except Exception:
        # Multi-file publication cannot be a filesystem transaction. Remove all
        # success markers before the caller writes bound failure evidence.
        _remove_ok_markers(attempt)
        raise


def _expected_validation_steps(phase: dict[str, Any]) -> list[int]:
    return [iteration - 1 for iteration in range(1, phase["iters"] + 1)
            if iteration == 1 or iteration % phase["steps_per_eval"] == 0 or iteration == phase["iters"]]


def _attest_log_fd(log_path: pathlib.Path | str, log_fd: int) -> None:
    try:
        path_stat = os.stat(log_path)
        fd_stat = os.fstat(log_fd)
    except OSError as exc:
        raise PilotError(f"attempt-3 log attestation failed: {log_path}: {exc}") from exc
    if not stat_is_regular(path_stat.st_mode) or not stat_is_regular(fd_stat.st_mode):
        raise PilotError(f"attempt-3 log attestation requires regular file: {log_path}")
    fields = ("st_dev", "st_ino", "st_mode", "st_size")
    if any(getattr(path_stat, field) != getattr(fd_stat, field) for field in fields):
        raise PilotError(f"attempt-3 log FD collision: {log_path}")


def stat_is_regular(mode: int) -> bool:
    import stat
    return stat.S_ISREG(mode)


def run_phase(args: argparse.Namespace, *, api: Any = None, preflight: Any = None) -> dict[str, Any]:
    phase_name = args.phase
    attempt = int(getattr(args, "attempt", 1) or 1)
    phase = _phase_spec(phase_name, attempt)
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
    historical_evidence: list[dict[str, Any]] = []
    historical_attempt2_evidence: dict[str, Any] = {}
    authorization: dict[str, str] | None = None
    phase_a_authorization: dict[str, str] | None = None
    phase_a_admission_lineage: dict[str, Any] | None = None
    phase_a_report: dict[str, Any] | None = None
    phase_a_marker: dict[str, Any] | None = None
    output = pathlib.Path(phase["adapter_path"])
    try:
        # Occupied evidence is a terminal one-attempt record. Check it before
        # any watchdog, config, or identity work can fail and overwrite it.
        try:
            _check_attempt_gates(phase_name, attempt, allow_active_log=attempt == 3,
                                 authorization_json=getattr(args, "authorization_json", None),
                                 phase_a_authorization_json=getattr(args, "phase_a_authorization_json", None))
            if attempt == 3:
                authorization = _authorization_json(args.authorization_json) if getattr(args, "authorization_json", None) else None
                phase_a_authorization = (_authorization_json(args.phase_a_authorization_json)
                                         if getattr(args, "phase_a_authorization_json", None) else None)
                historical_evidence = verify_attempt1_historical_evidence()
                historical_attempt2_evidence = verify_attempt2_historical_evidence()
                if getattr(args, "log_fd", None) is not None:
                    _attest_log_fd(args.log_path, int(args.log_fd))
        except Exception:
            attempt_gate_rejected = True
            raise
        watchdog = _install_timeout_watchdog(phase["timeout"])
        config = _read_config(args.config)
        effective = validate_pins(args, phase, config)
        contract = contract_digest(phase_name, effective, phase_spec=phase,
                                    historical_evidence=historical_evidence,
                                    historical_attempt2_evidence=historical_attempt2_evidence,
                                    authorization=authorization)
        _acquire_ft_lock(workspace)
        lock_acquired = True
        lock_lifecycle["acquired"] = True
        try:
            _check_attempt_gates(phase_name, attempt, allow_active_log=attempt == 3,
                                 authorization_json=getattr(args, "authorization_json", None),
                                 phase_a_authorization_json=getattr(args, "phase_a_authorization_json", None))
        except Exception:
            attempt_gate_rejected = True
            raise
        if attempt == 3 and getattr(args, "log_fd", None) is not None:
            _attest_log_fd(args.log_path, int(args.log_fd))
        if phase_name == "phase-b":
            phase_a_spec = attempt3_phase_specs()["phase-a"] if attempt == 3 else PHASES["phase-a"]
            phase_a_report_path = pathlib.Path(phase_a_spec["namespace_paths"]["phase-a-report"] if attempt == 3 else phase_a_spec["report"])
            if attempt == 1:
                phase_a_report_path = REPO_ROOT / phase_a_report_path
            if not phase_a_report_path.is_file():
                raise PilotError("Phase A report missing")
            _, phase_a_report = _read_json_snapshot(phase_a_report_path)
            marker_path = _marker_path("phase-a", "ok", phase_a_spec)
            if not marker_path.is_file():
                raise PilotError("Phase A OK marker missing")
            _, phase_a_marker = _read_json_snapshot(marker_path)
            if attempt == 3:
                validate_phase_b_dependency(phase_a_report, phase_a_marker, phase["resume_adapter_file"],
                                            phase_spec=phase_a_spec,
                                            historical_attempt2_evidence=historical_attempt2_evidence,
                                            trusted_phase_a_authorization=phase_a_authorization)
                phase_a_admission_lineage = capture_phase_a_admission_lineage(
                    phase_a_report_path, marker_path, phase_a_report, phase_a_marker)
            else:
                validate_phase_b_dependency(phase_a_report, phase_a_marker, phase["resume_adapter_file"])
        identity = (preflight or _runtime_preflight)(args, phase, config)
        if attempt == 3:
            _check_attempt_gates(phase_name, attempt, allow_active_log=True,
                                 authorization_json=getattr(args, "authorization_json", None),
                                 phase_a_authorization_json=getattr(args, "phase_a_authorization_json", None),
                                 trusted_identity=identity)
        if phase_name == "phase-b" and phase_a_report is not None:
            compare_immutable_identity(phase_a_report.get("identity_manifest", {}), identity)
            if attempt == 3:
                validate_phase_b_dependency(phase_a_report, phase_a_marker, phase["resume_adapter_file"],
                                            phase_spec=phase_a_spec, historical_evidence=historical_evidence,
                                            historical_attempt2_evidence=historical_attempt2_evidence,
                                            trusted_identity=identity,
                                            trusted_phase_a_authorization=phase_a_authorization)
        contract = contract_digest(phase_name, effective, identity, phase_spec=phase,
                                    historical_evidence=historical_evidence,
                                    historical_attempt2_evidence=historical_attempt2_evidence,
                                    authorization=authorization,
                                    phase_a_admission_lineage=phase_a_admission_lineage)
        output.mkdir(parents=True)
        runtime_api = api or _load_training_api()
        tree_flatten = _api_get(runtime_api, "tree_flatten")
        provider = _StepObservingProvider(lambda *_: (_ for _ in ()).throw(PilotError("provider not initialized")), phase_name,
                                          _api_get(runtime_api, "mx"), tree_flatten=tree_flatten)
        callback = _PilotTrainingCallback(phase_name)
        _execute_training(args, phase, runtime_api, output, provider, callback, config=config)
        artifacts = (_validate_artifacts(phase_name, output, phase_spec=phase)
                     if attempt == 3 else _validate_artifacts(phase_name, output))
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
        report = {"status": "ok", "phase": phase_name, "attempt": attempt,
                  "namespace": phase.get("namespace", "ds4-segmented-pilot"), "effective": effective, "contract_digest": contract,
                  "identity_manifest": identity, "provider_calls": len(provider.records), "optimizer_updates": len(callback.records),
                  "steps": callback.records, "provider_evidence": provider.records, "validation_evidence": callback.validation_records,
                  "artifacts": artifacts, "output_path": str(output), "commands": effective.get("command", []),
                  "wall_seconds": time.monotonic() - started, "retry": RETRY_POLICY, "fallback": FALLBACK_POLICY,
                  "non_claims": list(NON_CLAIMS), "global_mapping": [0] + global_steps(phase_name, callback_steps)}
        if attempt == 3:
            paths = phase["namespace_paths"]
            report.update({"attempt_1_historical_evidence": historical_evidence,
                           "attempt_2_historical_evidence": historical_attempt2_evidence,
                           "authorization": authorization,
                           **({"phase_a_admission_lineage": phase_a_admission_lineage}
                              if phase_name == "phase-b" else {}),
                           "report_path": str(phase["report"]), "log_path": paths[f"{phase_name}-log"],
                           "ok_marker_path": paths[f"{phase_name}-ok"], "fail_marker_path": paths[f"{phase_name}-fail"],
                           "final_report_path": paths["final-report"], "namespace_paths": dict(paths)})
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
        failed = _phase_failure_report(phase_name, error or "unknown failure", started, effective=effective, identity=identity,
                                       contract=contract, output=output, lock_lifecycle=lock_lifecycle,
                                       watchdog_cancelled=watchdog_cancelled, phase_spec=phase,
                                       historical_evidence=historical_evidence,
                                       historical_attempt2_evidence=historical_attempt2_evidence,
                                       authorization=authorization)
        if attempt_gate_rejected:
            # Existing one-attempt evidence is itself the terminal record. Never
            # delete or overwrite it with a second invocation's failure report.
            return failed
        try:
            if attempt == 3:
                _write_failure_evidence(phase_name, failed, phase_spec=phase)
            else:
                _write_failure_evidence(phase_name, failed)
        except Exception as evidence_error:
            failed["cleanup_warning"] = f"failure evidence write failed: {evidence_error}"
        return failed
    assert report is not None
    report["lock_lifecycle"] = lock_lifecycle
    report["watchdog_cancelled"] = watchdog_cancelled
    report["exit_code"] = 0
    try:
        if attempt == 3:
            validate_canonical_attempt3_report(report, phase_spec=phase,
                                               report_path=phase["report"],
                                               historical_evidence=historical_evidence,
                                               historical_attempt2_evidence=historical_attempt2_evidence,
                                               trusted_identity=identity,
                                               trusted_authorization=authorization,
                                               trusted_phase_a_authorization=phase_a_authorization,
                                               trusted_phase_a_admission_lineage=phase_a_admission_lineage)
        _write_success_evidence(phase_name, report, phase_spec=phase,
                                phase_a_authorization=phase_a_authorization,
                                phase_a_admission_lineage=phase_a_admission_lineage,
                                phase_b_authorization=authorization,
                                trusted_identity=identity,
                                trusted_attempt_1_historical_evidence=historical_evidence,
                                trusted_attempt_2_historical_evidence=historical_attempt2_evidence)
    except Exception as exc:
        failed = _phase_failure_report(phase_name, f"report/marker write failed: {exc}", started, effective=effective,
                                       identity=identity, contract=contract, output=output,
                                       lock_lifecycle=lock_lifecycle, watchdog_cancelled=watchdog_cancelled,
                                       phase_spec=phase,
                                       historical_evidence=historical_evidence,
                                       historical_attempt2_evidence=historical_attempt2_evidence,
                                       authorization=authorization)
        try:
            if attempt == 3:
                _write_failure_evidence(phase_name, failed, phase_spec=phase)
            else:
                _write_failure_evidence(phase_name, failed)
        except Exception as evidence_error:
            failed["cleanup_warning"] = f"failure evidence write failed: {evidence_error}"
        return failed
    return report


def _prepare_attempt3_launch_args(args: argparse.Namespace, phase_spec: dict[str, Any]) -> argparse.Namespace:
    identity = attempt3_launch_identity()
    pinned = dict(COMMON_VALUES)
    pinned.update({"attempt": 3, "model": identity["model"], "data": identity["data"], "config": identity["config"],
                   "adapter_path": phase_spec["adapter_path"], "resume_adapter_file": phase_spec["resume_adapter_file"],
                   "train": True, "test": False, "hf_dataset": False, "iters": phase_spec["iters"],
                   "steps_per_eval": phase_spec["steps_per_eval"], "log_path": phase_spec["namespace_paths"][f"{args.phase}-log"]})
    for key, expected in pinned.items():
        if getattr(args, key, None) is None:
            setattr(args, key, expected)
    args._command_argv = tuple(canonical_attempt3_command(args.phase, phase_spec))
    return args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.launch_check_only:
        if args.attempt != 3 or not args.log_path:
            print("launch-check-only requires --attempt 3 and --log-path", file=sys.stderr)
            return 1
        try:
            phase_spec = attempt3_phase_specs()[args.phase]
            args = _prepare_attempt3_launch_args(args, phase_spec)
            config = _read_config(args.config)
            validate_pins(args, phase_spec, config)
            trusted_identity = _runtime_preflight(args, phase_spec, config)
            print(json.dumps(check_attempt3_launch(args.phase, args.log_path,
                                                   trusted_identity=trusted_identity,
                                                   authorization_json=args.authorization_json), sort_keys=True))
        except Exception as exc:
            print(f"pilot launch check failed closed: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.attempt == 3 and not args.log_fd:
        print("attempt 3 requires --log-fd from the exclusive catalog wrapper", file=sys.stderr)
        return 1
    report = run_phase(args)
    if report.get("status") != "ok":
        print(f"pilot failed closed: {report.get('error', 'unknown failure')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
