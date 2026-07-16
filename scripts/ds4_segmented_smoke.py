"""DS4 segmented-training smoke activation entry point.

Thin wiring only: loads model + dataset via MLX-LM public APIs,
applies LoRA per the YAML config, constructs the DS4 segmented loss
provider, and calls ``train(..., loss_and_grad=provider)`` exactly
once per attempted microbatch. The trainer retains all optimizer,
accumulation, save, callback, and UI ownership. Default MLX-LM
training path is unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pathlib
import shutil
import signal
import sys
import threading
import time
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import yaml

from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, build_parser
from mlx_lm.tuner.callbacks import get_reporting_callbacks
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    print_trainable_parameters,
)
from mlx_lm.utils import save_config

from ds4_ft_mlx.segmented_loss_and_grad import make_ds4_segmented_loss_and_grad
from ds4_ft_mlx.lora_targets import build_lora_parameters

# ---------------------------------------------------------------------------
# Repository root for absolute artifact paths
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ARTIFACT_DIR = _REPO_ROOT / "agent-output" / "cmux-14-3"
_SMOKE_REPORT_PATH = _ARTIFACT_DIR / "smoke-report.json"

# ---------------------------------------------------------------------------
# Preflight constants
# ---------------------------------------------------------------------------
SMOKE_TIMEOUT_SECONDS = 1200
SMOKE_MEMORY_HEADROOM = 32 * 1024**3
SMOKE_DISK_MIN_FREE = 1 * 1024**3
SMOKE_LOCK_TIMEOUT_S = 60
SMOKE_LOCK_POLL_S = 2.0
SMOKE_ABORT_PREFLIGHT = 3
SMOKE_ABORT_TRAIN = 4
SMOKE_ABORT_TIMEOUT = 2

# ---------------------------------------------------------------------------
# Custom argparse type for segment-size validation
# ---------------------------------------------------------------------------
def _segment_size_type(value: str) -> int:
    try:
        v = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid segment-size value: {value!r} (must be an integer 1..4)"
        )
    if v < 1 or v > 4:
        raise argparse.ArgumentTypeError(
            f"invalid segment-size value: {v} (must be 1..4)"
        )
    return v


# ---------------------------------------------------------------------------
# Lock: single atomic .ds4-ft.lock with path+token ownership tracking
# Acquired path and PID token are stored so release only unlinks the
# exact lock this process acquired.
# ---------------------------------------------------------------------------
_LOCK_OWNED_PATH: pathlib.Path | None = None
_LOCK_OWNED_TOKEN: str | None = None

# Warnings accumulated during terminal cleanup; inspectable, never fatal.
_CLEANUP_WARNINGS: list[str] = []

# Last preflight failure reason — preserves detailed diagnostics that
# str(SystemExit(3)) truncates to just "3".
_LAST_FAIL_REASON: str | None = None

# Phase-1 pinned smoke identity — any deviation at direct entry point is rejected.
_PINNED_SMOKE_VALUES = {
    "max_seq_length": 4096,
    "iters": 1,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "mask_prompt": True,
    "grad_checkpoint": True,
    "segment_size": 1,
}

# Phase-1 pinned asset paths — enforced BEFORE preflight at direct entry point.
# Synthetic helpers that call _check_preflight directly are unaffected.
_PINNED_SMOKE_PATHS: dict[str, str | None] = {
    "model": "/Volumes/Data NVME/mlx-ft/ds4/model-4bit",
    "data": "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke",
    "adapter_path": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke",
    "config": "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json",
}


def _ft_lock_path(mlx_work: pathlib.Path) -> pathlib.Path:
    return mlx_work / ".ds4-ft.lock"


def _acquire_ft_lock(
    mlx_work: pathlib.Path,
    timeout_s: int = SMOKE_LOCK_TIMEOUT_S,
    poll_s: float = SMOKE_LOCK_POLL_S,
) -> None:
    global _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN
    lock = _ft_lock_path(mlx_work)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    token = str(os.getpid())
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(poll_s)
            continue
        except OSError as exc:
            last_error = f"lock open failed: {exc}"
            time.sleep(poll_s)
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={token}\n")
        except OSError as exc:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                lock.unlink()
            except OSError:
                pass
            last_error = f"lock write failed: {exc}"
            time.sleep(poll_s)
            continue
        _LOCK_OWNED_PATH = lock
        _LOCK_OWNED_TOKEN = token
        return
    raise SystemExit(
        last_error or f"lock acquisition failed after {timeout_s}s: {lock}"
    )


def _release_ft_lock() -> None:
    global _LOCK_OWNED_PATH, _LOCK_OWNED_TOKEN
    if _LOCK_OWNED_PATH is None:
        return
    if not _LOCK_OWNED_PATH.exists():
        _LOCK_OWNED_PATH = None
        _LOCK_OWNED_TOKEN = None
        return
    # Exact PID comparison — not substring. Content is "pid=TOKEN\n".
    try:
        content = _LOCK_OWNED_PATH.read_text()
    except OSError:
        return
    if _LOCK_OWNED_TOKEN is None:
        return
    stored = content.strip()
    if not stored.startswith("pid=") or stored.removeprefix("pid=") != _LOCK_OWNED_TOKEN:
        return
    try:
        _LOCK_OWNED_PATH.unlink()
    except FileNotFoundError:
        pass
    _LOCK_OWNED_PATH = None
    _LOCK_OWNED_TOKEN = None


# ---------------------------------------------------------------------------
# Marker paths
# ---------------------------------------------------------------------------
def _ok_marker_path(mlx_work: pathlib.Path) -> pathlib.Path:
    return mlx_work / ".ds4-segmented-smoke-ok"

def _fail_marker_path(mlx_work: pathlib.Path) -> pathlib.Path:
    return mlx_work / ".ds4-segmented-smoke-fail"

def _write_fail_marker(mlx_work: pathlib.Path, reason: str) -> None:
    global _LAST_FAIL_REASON
    _LAST_FAIL_REASON = reason
    marker = _fail_marker_path(mlx_work)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"status": "fail", "reason": reason, "timestamp": time.time()})
        + "\n", encoding="utf-8",
    )

def _write_ok_marker(mlx_work: pathlib.Path) -> None:
    marker = _ok_marker_path(mlx_work)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"status": "ok", "timestamp": time.time(), "exit_code": 0})
        + "\n", encoding="utf-8",
    )

def _clear_markers(mlx_work: pathlib.Path) -> None:
    for p in (_ok_marker_path(mlx_work), _fail_marker_path(mlx_work)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

# ---------------------------------------------------------------------------
# Smoke report
# ---------------------------------------------------------------------------
def _write_smoke_report(report: dict[str, Any]) -> None:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _SMOKE_REPORT_PATH.write_text(json.dumps(report) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_smoke_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.add_argument(
        "--segment-size",
        type=_segment_size_type,
        default=1,
        help="DS4 segmented provider segment size (1..4). Bounded smoke pins 1.",
    )
    return parser

def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    d = vars(args)
    config_path = d.get("config")
    if config_path is not None:
        cp = pathlib.Path(str(config_path))
        if not cp.is_file():
            raise SystemExit(f"config file not found: {config_path}")
        try:
            raw = cp.read_text(encoding="utf-8")
            if cp.suffix.lower() in (".yaml", ".yml"):
                cfg = yaml.safe_load(raw)
            else:
                cfg = json.loads(raw)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise SystemExit(f"config file read/parse failed: {config_path}: {exc}")
        if not isinstance(cfg, dict):
            raise SystemExit(
                f"config file must contain a mapping (dict), got {type(cfg).__name__}: {config_path}"
            )
        for k, v in cfg.items():
            if d.get(k) is None:
                d[k] = v
    for k, v in CONFIG_DEFAULTS.items():
        if d.get(k) is None:
            d[k] = v
    return args


# ---------------------------------------------------------------------------
# Recursive gradient tree flattener
# ---------------------------------------------------------------------------
def _flatten_gradient_tree(
    val: Any, prefix: str = "",
) -> list[tuple[str, Any, tuple, str]]:
    """Recursively flatten gradient tree.

    Returns list of (path, tensor, shape, dtype_str) for every leaf.
    Handles dict, list, tuple containers. Non-container leaves that
    have shape/dtype attributes are treated as tensors.
    """
    leaves: list[tuple[str, Any, tuple, str]] = []
    if isinstance(val, dict):
        for k, v in val.items():
            child = f"{prefix}.{k}" if prefix else k
            _collect_leaves(v, child, leaves)
    elif isinstance(val, (list, tuple)):
        for i, v in enumerate(val):
            child = f"{prefix}[{i}]"
            _collect_leaves(v, child, leaves)
    else:
        _collect_leaves(val, prefix, leaves)
    return leaves


def _collect_leaves(
    val: Any, path: str, out: list[tuple[str, Any, tuple, str]]
) -> None:
    """Collect gradient leaf or recurse into container.

    Non-container, non-tensor values are malformed and raise SystemExit.
    """
    if isinstance(val, dict):
        for k, v in val.items():
            child = f"{path}.{k}"
            _collect_leaves(v, child, out)
    elif isinstance(val, (list, tuple)):
        for i, v in enumerate(val):
            child = f"{path}[{i}]"
            _collect_leaves(v, child, out)
    elif hasattr(val, "shape") and hasattr(val, "dtype"):
        out.append((path, val, val.shape, str(val.dtype)))
    else:
        raise SystemExit(f"malformed gradient leaf at {path}: {type(val).__name__}")


# ---------------------------------------------------------------------------
# Schema comparison helper
# ---------------------------------------------------------------------------
def _compare_tree_schema(
    actual: Any, expected: Any, path: str = "root",
) -> list[str]:
    """Compare two gradient trees for exact container types, key order,
    paths, leaf shapes, and leaf dtypes at every node.

    Returns list of mismatch descriptions (empty = match).
    """
    mismatches: list[str] = []
    if isinstance(expected, dict):
        if type(actual) is not type(expected):
            mismatches.append(f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}")
            return mismatches
        if list(actual.keys()) != list(expected.keys()):
            mismatches.append(
                f"{path}: key order mismatch\n"
                f"  expected keys: {list(expected.keys())}\n"
                f"  actual keys:   {list(actual.keys())}"
            )
            return mismatches
        for k in expected:
            if k not in actual:
                mismatches.append(f"{path}.{k}: missing in actual")
                continue
            child = f"{path}.{k}"
            mismatches.extend(_compare_tree_schema(actual[k], expected[k], child))
    elif isinstance(expected, (list, tuple)):
        exp_type = type(expected)
        if type(actual) is not exp_type:
            mismatches.append(f"{path}: expected {exp_type.__name__}, got {type(actual).__name__}")
            return mismatches
        if len(actual) != len(expected):
            mismatches.append(f"{path}: length mismatch expected={len(expected)} actual={len(actual)}")
            return mismatches
        for i, (ae, ee) in enumerate(zip(actual, expected)):
            child = f"{path}[{i}]"
            mismatches.extend(_compare_tree_schema(ae, ee, child))
    elif hasattr(expected, "shape") and hasattr(expected, "dtype"):
        if not hasattr(actual, "shape") or not hasattr(actual, "dtype"):
            mismatches.append(f"{path}: expected tensor, got {type(actual).__name__}")
            return mismatches
        if tuple(actual.shape) != tuple(expected.shape):
            mismatches.append(
                f"{path}: shape mismatch expected={tuple(expected.shape)} actual={tuple(actual.shape)}"
            )
        if str(actual.dtype) != str(expected.dtype):
            mismatches.append(
                f"{path}: dtype mismatch expected={expected.dtype} actual={actual.dtype}"
            )
    return mismatches


def _check_schema_match(gradients: Any, trainable_params: Any) -> None:
    """Raise SystemExit if gradient tree does not match trainable_parameters
    in container types, key order, paths, leaf shapes, and leaf dtypes."""
    mismatches = _compare_tree_schema(gradients, trainable_params)
    if mismatches:
        raise SystemExit(
            "gradient schema mismatch:\n" + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Provider observation wrapper — captures exact scalar evidence
# ---------------------------------------------------------------------------
class _ObservingProvider:
    """Wraps a real provider to count calls and capture exact evidence.

    Captures materialized scalar loss, scalar token count, flattened
    gradient leaf schema (every leaf path, shape, dtype), and finiteness
    of every gradient leaf. Captures the batch to verify token count
    against the default_loss mask formula. Valid nested dict/list/tuple
    gradient trees pass; malformed or nonfinite leaves fail.

    The original provider delegate remains observable and is called
    exactly once.
    """

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.call_count: int = 0
        self.captured_loss: float | None = None
        self.captured_token_count: int | None = None
        self.captured_gradient_paths: list[str] | None = None
        self.captured_gradient_shapes: list[tuple] | None = None
        self.captured_gradient_dtypes: list[str] | None = None
        self.captured_gradient_leaf_count: int | None = None
        self.captured_all_finite: bool | None = None
        # Captured last batch for token count verification
        self._last_batch: tuple | None = None

    def __call__(self, model: Any, *batch: Any) -> tuple:
        self.call_count += 1
        self._last_batch = batch
        # Capture expected gradient schema from model.trainable_parameters()
        self._expected_tp_tree = model.trainable_parameters() if hasattr(model, "trainable_parameters") else None
        result = self._delegate(model, *batch)
        if result is None:
            return result
        (loss, token_count), gradients = result

        # Validate loss: must be mx.float32 scalar, finite
        if not hasattr(loss, "dtype"):
            raise SystemExit(f"loss has no dtype attribute")
        if loss.shape != ():
            raise SystemExit(f"loss shape is {loss.shape}, expected scalar ()")
        if loss.dtype != mx.float32:
            raise SystemExit(f"loss dtype is {loss.dtype}, expected mx.float32")
        if not bool(mx.isfinite(loss).item()):
            raise SystemExit(f"loss is not finite: {loss.item()}")
        self.captured_loss = float(loss.item())

        # Validate token count: must be mx.int32 scalar, positive
        if not hasattr(token_count, "dtype"):
            raise SystemExit(f"token_count has no dtype attribute")
        if token_count.shape != ():
            raise SystemExit(f"token_count shape is {token_count.shape}, expected scalar ()")
        if token_count.dtype != mx.int32:
            raise SystemExit(f"token_count dtype is {token_count.dtype}, expected mx.int32")
        if not bool(mx.all(token_count > 0).item()):
            raise SystemExit(f"token_count is not positive: {token_count.item()}")
        self.captured_token_count = int(token_count.item())

        # Recursively flatten gradient tree; malformed leaves raise SystemExit
        leaf_paths: list[str] = []
        leaf_shapes: list[tuple] = []
        leaf_dtypes: list[str] = []
        all_finite = True
        if gradients is not None:
            leaves = _flatten_gradient_tree(gradients)
            for lp, lt, ls, ld in leaves:
                leaf_paths.append(lp)
                leaf_shapes.append(ls)
                leaf_dtypes.append(ld)
                try:
                    if not bool(mx.all(mx.isfinite(lt)).item()):
                        all_finite = False
                except Exception:
                    all_finite = False
            if not leaves:
                all_finite = False
        else:
            all_finite = False
        self.captured_gradient_paths = leaf_paths if leaf_paths else None
        self.captured_gradient_shapes = leaf_shapes if leaf_shapes else None
        self.captured_gradient_dtypes = leaf_dtypes if leaf_dtypes else None
        self.captured_gradient_leaf_count = len(leaf_paths) if leaf_paths else 0
        self.captured_all_finite = all_finite

        # Compare returned gradients against expected trainable_parameters schema
        if gradients is not None and self._expected_tp_tree is not None:
            _check_schema_match(gradients, self._expected_tp_tree)

        return result

    def compute_expected_token_count(self) -> int:
        """Compute expected token count using the pinned default_loss mask:

            targets = tokens[:, 1:]
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = (steps >= start) & (steps <= end)
            expected = sum(mask)

        Raises SystemExit if batch is not available or formula fails."""
        if self._last_batch is None:
            raise SystemExit("no captured batch to compute expected token count")
        try:
            batch = self._last_batch
            tokens = batch[0]
            lengths = batch[1]
            if isinstance(lengths, (list, tuple)) and len(lengths) >= 2:
                lengths_arr = lengths[1]
            else:
                lengths_arr = lengths
            targets = tokens[:, 1:]
            steps = mx.arange(1, targets.shape[1] + 1)
            start = lengths_arr[:, 0:1]
            end = lengths_arr[:, 1:]
            mask = (steps >= start) & (steps <= end)
            return int(mx.sum(mask).item())
        except Exception as exc:
            raise SystemExit(f"token count formula failed: {exc}")


# ---------------------------------------------------------------------------
# Tokenizer helper — try to load tokenizer and count real tokens
# ---------------------------------------------------------------------------
def _try_token_count(model_path: pathlib.Path, text: str) -> int | None:
    """Try to load the tokenizer from model_path and token-count text.

    Returns the real token count, or None if the tokenizer cannot be loaded.
    """
    try:
        from mlx_lm.tokenizer_utils import load_tokenizer
        tok = load_tokenizer(str(model_path))
        ids = tok.encode(text)
        return len(ids)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def _check_process_rss_gb(threshold_gb: float = 50.0) -> int:
    try:
        import psutil
    except ImportError:
        raise SystemExit("psutil not available")
    count = 0
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            if info is None:
                continue
            mem = info.get("memory_info")
            if mem is None:
                continue
            rss = mem.rss
            if rss is None:
                continue
            if rss > threshold_gb * 1024**3:
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            raise SystemExit(f"process scan failure ({type(exc).__name__}): {exc}")
        except (OSError, TypeError, AttributeError) as exc:
            raise SystemExit(f"process info read failure ({type(exc).__name__}): {exc}")
    return count


def _check_preflight(
    args: argparse.Namespace, mlx_work: pathlib.Path,
    exit_code_if_fail: int = SMOKE_ABORT_PREFLIGHT,
) -> None:
    """Fail-closed preflight: lock, model identity, dataset schema, LoRA identity,
    MLX version, fork identity, adapter absence, RSS, memory, disk."""
    _acquire_ft_lock(mlx_work)
    model_path = pathlib.Path(args.model)
    if not model_path.is_dir():
        _write_fail_marker(mlx_work, f"model path not found: {model_path}")
        raise SystemExit(exit_code_if_fail)
    if not list(model_path.iterdir()):
        _write_fail_marker(mlx_work, f"model directory is empty: {model_path}")
        raise SystemExit(exit_code_if_fail)
    if not (model_path / "config.json").is_file():
        _write_fail_marker(mlx_work, f"model config.json not found: {model_path}")
        raise SystemExit(exit_code_if_fail)
    tokenizer_found = any(
        (model_path / f).is_file() for f in ("tokenizer.json", "tokenizer.model")
    )
    if not tokenizer_found:
        _write_fail_marker(mlx_work, f"no tokenizer file found in {model_path}")
        raise SystemExit(exit_code_if_fail)

    # Model weight files: require nonempty regular model-*.safetensors or weights/ with files
    weight_files = sorted(model_path.glob("model-*.safetensors"))
    weights_dir = model_path / "weights"
    have_weight_files = bool(weight_files and all(f.is_file() and f.stat().st_size > 0 for f in weight_files))
    have_weights_dir = weights_dir.is_dir() and any(f.is_file() and f.stat().st_size > 0 for f in weights_dir.iterdir())
    if not have_weight_files and not have_weights_dir:
        _write_fail_marker(mlx_work, f"no nonempty model weight files in {model_path}")
        raise SystemExit(exit_code_if_fail)

    # Dataset: require all three splits, parse EVERY nonblank row, check schema
    data_path = pathlib.Path(str(args.data))
    ds4_allowed_keys = {"prompt", "completion"}
    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        sf = data_path / split
        if not sf.is_file():
            _write_fail_marker(mlx_work, f"dataset split not found: {sf}")
            raise SystemExit(exit_code_if_fail)
        row_count = 0
        try:
            with sf.open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    row_count += 1
                    try:
                        row = json.loads(line_stripped)
                    except json.JSONDecodeError:
                        _write_fail_marker(mlx_work, f"invalid JSON in {split} line {line_no}")
                        raise SystemExit(exit_code_if_fail)
                    if not isinstance(row, dict):
                        _write_fail_marker(mlx_work, f"{split} line {line_no}: not a JSON object")
                        raise SystemExit(exit_code_if_fail)
                    keys = set(row.keys())
                    if keys != ds4_allowed_keys:
                        _write_fail_marker(mlx_work,
                            f"{split} line {line_no}: keys {keys} != {{prompt, completion}}")
                        raise SystemExit(exit_code_if_fail)
                    prompt = row.get("prompt", "")
                    completion = row.get("completion", "")
                    if not isinstance(prompt, str) or not prompt.strip():
                        _write_fail_marker(mlx_work, f"{split} line {line_no}: prompt must be nonempty string")
                        raise SystemExit(exit_code_if_fail)
                    if not isinstance(completion, str) or not completion.strip():
                        _write_fail_marker(mlx_work, f"{split} line {line_no}: completion must be nonempty string")
                        raise SystemExit(exit_code_if_fail)
                    # Token-length bound: use real tokenizer if available;
                    # otherwise apply conservative whitespace + density fallback.
                    max_len = 4096
                    combined = prompt + completion
                    real_tokens = _try_token_count(model_path, combined)
                    if real_tokens is not None:
                        if real_tokens > max_len:
                            _write_fail_marker(mlx_work,
                                f"{split} line {line_no}: {real_tokens} real tokens > {max_len}")
                            raise SystemExit(exit_code_if_fail)
                    else:
                        # Conservative fallback: whitespace bound + unbroken-string guard
                        approx_tokens = len(prompt.split()) + len(completion.split())
                        if approx_tokens > max_len:
                            _write_fail_marker(mlx_work,
                                f"{split} line {line_no}: approx {approx_tokens} tokens > {max_len}")
                            raise SystemExit(exit_code_if_fail)
                        # Guard against unbroken strings where whitespace splitting
                        # is unreliable (e.g. a 10000-char single word scores as 1 token):
                        # reject rows with chars-per-whitespace-token > 50 AND total chars > max_len.
                        total_chars = len(combined)
                        density = total_chars / max(approx_tokens, 1)
                        if density > 50 and total_chars > max_len:
                            _write_fail_marker(mlx_work,
                                f"{split} line {line_no}: unreliable whitespace bound "
                                f"(density={density:.0f} chars/token, {total_chars} chars > {max_len})")
                            raise SystemExit(exit_code_if_fail)
        except OSError as exc:
            _write_fail_marker(mlx_work, f"dataset read failed: {split}: {exc}")
            raise SystemExit(exit_code_if_fail)
        if row_count == 0:
            _write_fail_marker(mlx_work, f"dataset split is empty: {split}")
            raise SystemExit(exit_code_if_fail)

    # LoRA config: accept both top-level and nested lora_parameters format
    config_path = getattr(args, "config", None)
    if config_path is not None:
        cp = pathlib.Path(str(config_path))
        if not cp.is_file():
            _write_fail_marker(mlx_work, f"lora config not found: {config_path}")
            raise SystemExit(exit_code_if_fail)
        try:
            lora_raw = cp.read_text(encoding="utf-8")
            if cp.suffix.lower() in (".yaml", ".yml"):
                lora_cfg = yaml.safe_load(lora_raw)
            else:
                lora_cfg = json.loads(lora_raw)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            _write_fail_marker(mlx_work, f"lora config read/parse failed: {cp}: {exc}")
            raise SystemExit(exit_code_if_fail)
        if not isinstance(lora_cfg, dict):
            _write_fail_marker(mlx_work, f"lora config must be a dict: {cp}")
            raise SystemExit(exit_code_if_fail)
        # Unpack nested format: {"lora_parameters":{"rank":...,"scale":...,"dropout":...,"keys":[...]}}
        lp = lora_cfg.get("lora_parameters", lora_cfg)
        if not isinstance(lp, dict):
            _write_fail_marker(mlx_work, f"lora config body must be a dict: {cp}")
            raise SystemExit(exit_code_if_fail)
        # Canonical LoRA parameters via build_lora_parameters() — NOT hardcoded
        canonical = build_lora_parameters()
        canonical_keys = set(canonical["keys"])

        # Validate rank: must be finite numeric (not bool, not NaN/Inf) in (0..256]
        rank = lp.get("rank")
        if rank is None:
            _write_fail_marker(mlx_work, f"lora config missing required field 'rank': {cp}")
            raise SystemExit(exit_code_if_fail)
        if (isinstance(rank, bool) or not isinstance(rank, (int, float))
                or not math.isfinite(rank) or rank <= 0 or rank > 256):
            _write_fail_marker(mlx_work, f"lora config 'rank' out of range (1..256): {rank}")
            raise SystemExit(exit_code_if_fail)

        # Validate scale: must be finite numeric (not bool, not NaN/Inf) in (0..1000]
        scale = lp.get("scale")
        if scale is None:
            _write_fail_marker(mlx_work, f"lora config missing required field 'scale': {cp}")
            raise SystemExit(exit_code_if_fail)
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale <= 0 or scale > 1000.0):
            _write_fail_marker(mlx_work, f"lora config 'scale' out of range (0..1000): {scale}")
            raise SystemExit(exit_code_if_fail)

        # Require exact canonical key set — reject subsets, supersets, duplicates, wrong keys
        keys = lp.get("keys", None)
        if not keys:
            _write_fail_marker(mlx_work, f"lora config missing required 'keys' field: {cp}")
            raise SystemExit(exit_code_if_fail)
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            _write_fail_marker(mlx_work, f"lora config 'keys' must be a list of strings: {cp}")
            raise SystemExit(exit_code_if_fail)
        # Reject duplicates in the config
        if len(keys) != len(set(keys)):
            _write_fail_marker(mlx_work, f"lora config 'keys' contains duplicates: {keys}")
            raise SystemExit(exit_code_if_fail)
        # Reject any key not in canonical set
        for k in keys:
            if k not in canonical_keys:
                _write_fail_marker(mlx_work,
                    f"lora config key '{k}' not in canonical DS4 set {sorted(canonical_keys)}: {cp}")
                raise SystemExit(exit_code_if_fail)
        # Reject subset (config has fewer keys than canonical)
        config_key_set = set(keys)
        if config_key_set != canonical_keys:
            _write_fail_marker(mlx_work,
                f"lora config keys {sorted(config_key_set)} != canonical {sorted(canonical_keys)}: {cp}")
            raise SystemExit(exit_code_if_fail)

        # Validate dropout: must be finite numeric (not bool, not NaN/Inf) in [0, 1)
        dropout = lp.get("dropout", -1.0)
        if (isinstance(dropout, bool)
                or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout)
                or dropout < 0 or dropout >= 1.0):
            _write_fail_marker(mlx_work, f"lora config 'dropout' out of range [0..1): {dropout}")
            raise SystemExit(exit_code_if_fail)

    # MLX version check: require mlx.core.__version__ == 0.31.2, fail closed
    try:
        import mlx
        v = None
        if hasattr(mlx, "core") and hasattr(mlx.core, "__version__"):
            v = mlx.core.__version__
        if v is None:
            _write_fail_marker(mlx_work, "MLX core version not available")
            raise SystemExit(exit_code_if_fail)
        if v != "0.31.2":
            _write_fail_marker(mlx_work, f"MLX version is {v}, expected 0.31.2")
            raise SystemExit(exit_code_if_fail)
    except ImportError:
        _write_fail_marker(mlx_work, "mlx not importable")
        raise SystemExit(exit_code_if_fail)

    # Fork identity: verify mlx_lm.tuner.trainer resolves from vendor/mlx-lm
    try:
        import mlx_lm
        mlx_lm_dir = pathlib.Path(mlx_lm.__file__).resolve().parent
        expected_vendor = _REPO_ROOT / "vendor" / "mlx-lm" / "mlx_lm"
        if str(mlx_lm_dir).rstrip("/") != str(expected_vendor).rstrip("/"):
            _write_fail_marker(mlx_work,
                f"mlx_lm resolved from {mlx_lm_dir}, expected {expected_vendor}")
            raise SystemExit(exit_code_if_fail)
    except (ImportError, AttributeError, OSError) as exc:
        _write_fail_marker(mlx_work, f"mlx_lm fork identity check failed: {exc}")
        raise SystemExit(exit_code_if_fail)

    adapter_path = pathlib.Path(str(args.adapter_path))
    if adapter_path.exists():
        _write_fail_marker(mlx_work, f"adapter path exists: {adapter_path}")
        raise SystemExit(exit_code_if_fail)
    rss_count = _check_process_rss_gb(50.0)
    if rss_count > 0:
        _write_fail_marker(mlx_work, f"found {rss_count} processes >50 GiB RSS")
        raise SystemExit(exit_code_if_fail)
    try:
        model_dir_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    except OSError as exc:
        _write_fail_marker(mlx_work, f"model dir scan failed: {exc}")
        raise SystemExit(exit_code_if_fail)
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        _write_fail_marker(mlx_work, "psutil not available")
        raise SystemExit(exit_code_if_fail)
    except OSError as exc:
        _write_fail_marker(mlx_work, f"memory query failed: {exc}")
        raise SystemExit(exit_code_if_fail)
    required = model_dir_bytes + SMOKE_MEMORY_HEADROOM
    if available < required:
        _write_fail_marker(mlx_work,
            f"insufficient memory: {available//1024**3} GiB < {required//1024**3} GiB")
        raise SystemExit(exit_code_if_fail)
    adapter_parent = adapter_path.parent
    try:
        free = shutil.disk_usage(adapter_parent).free
    except OSError as exc:
        _write_fail_marker(mlx_work, f"disk query failed: {exc}")
        raise SystemExit(exit_code_if_fail)
    if free < SMOKE_DISK_MIN_FREE:
        _write_fail_marker(mlx_work,
            f"insufficient disk: {free//1024**3} GiB < {SMOKE_DISK_MIN_FREE//1024**3} GiB")
        raise SystemExit(exit_code_if_fail)


# ---------------------------------------------------------------------------
# Timeout watchdog (signal + backup thread)
# ---------------------------------------------------------------------------
_TIMEOUT_FLAG: bool = False
_TIMEOUT_REASON: str | None = None


def _timeout_handler(signum: int, frame: Any) -> None:
    global _TIMEOUT_FLAG, _TIMEOUT_REASON
    if not _TIMEOUT_FLAG:
        _TIMEOUT_FLAG = True
        _TIMEOUT_REASON = f"smoke timed out after {SMOKE_TIMEOUT_SECONDS}s"
    raise SystemExit(_TIMEOUT_REASON)


def _backup_watchdog() -> None:
    """Backup watchdog thread: if the primary signal.alarm does not fire
    (e.g. on a thread where signals are not delivered), this thread sends
    SIGALRM to the process to trigger the signal handler in the main thread."""
    deadline = time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5
    while time.monotonic() < deadline:
        if _TIMEOUT_FLAG:
            return
        time.sleep(1)
    # Send SIGALRM to this process — triggers signal handler in main thread
    os.kill(os.getpid(), signal.SIGALRM)


def _install_timeout_watchdog(mlx_work: pathlib.Path) -> None:
    """Install both signal.alarm and a backup threading watchdog."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SMOKE_TIMEOUT_SECONDS)
    t = threading.Thread(target=_backup_watchdog, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> _ObservingProvider:
    import numpy as np
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    training_callback = get_reporting_callbacks(
        args.report_to, project_name=args.project_name,
        log_dir=str(args.adapter_path), config=vars(args),
    )
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested {args.num_layers} layers but model has {len(model.layers)}"
        )
    if args.fine_tune_type == "full":
        for l in model.layers[-max(args.num_layers, 0):]:
            l.unfreeze()
        args.lora_parameters = None
    elif args.fine_tune_type in ("lora", "dora"):
        linear_to_lora_layers(
            model, args.num_layers, args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"unknown fine-tune-type {args.fine_tune_type}")
    if args.resume_adapter_file is not None:
        model.load_weights(args.resume_adapter_file, strict=False)
    print_trainable_parameters(model)
    adapter_path = pathlib.Path(str(args.adapter_path))
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")
    training_args = TrainingArgs(
        batch_size=args.batch_size, iters=args.iters,
        val_batches=args.val_batches, steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval, steps_per_save=args.save_every,
        adapter_file=str(adapter_file), max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})
    if optimizer_name == "adam":
        opt_class = optim.Adam
    elif optimizer_name == "adamw":
        opt_class = optim.AdamW
    elif optimizer_name == "muon":
        opt_class = optim.Muon
    elif optimizer_name == "sgd":
        opt_class = optim.SGD
    elif optimizer_name == "adafactor":
        opt_class = optim.Adafactor
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    opt = opt_class(learning_rate=lr, **optimizer_config)
    raw_provider = make_ds4_segmented_loss_and_grad(segment_size=args.segment_size)
    provider = _ObservingProvider(raw_provider)
    train(
        model=model, args=training_args, optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        training_callback=training_callback,
        loss_and_grad=provider,
    )
    return provider


def main(argv: list[str] | None = None) -> int:
    """Orchestrate: watchdog → parse → config → markers → preflight → run → validate → report.

    Watchdog is installed BEFORE parser construction; parser and all later
    phases are inside the try/finally for terminal cleanup.
    """
    start = time.monotonic()

    # --- Pre-scan argv for adapter-path to resolve workspace before parse ---
    pre_scan_work = pathlib.Path.cwd()
    if argv:
        for i, a in enumerate(argv):
            if a == "--adapter-path" and i + 1 < len(argv):
                pre_scan_work = pathlib.Path(argv[i + 1]).resolve().parent
                break
            if a.startswith("--adapter-path="):
                pre_scan_work = pathlib.Path(a.split("=", 1)[1]).resolve().parent
                break

    # Clear stale markers at pre-scanned workspace
    _clear_markers(pre_scan_work)

    # --- Watchdog installed before parser (covers full process startup) ---
    global _TIMEOUT_FLAG, _TIMEOUT_REASON, _LAST_FAIL_REASON
    _TIMEOUT_FLAG = False
    _TIMEOUT_REASON = None
    _LAST_FAIL_REASON = None
    _install_timeout_watchdog(pathlib.Path.cwd())

    mlx_work: pathlib.Path | None = pre_scan_work

    try:
        # --- Parser inside terminal lifecycle ---
        try:
            stderr_capture = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = stderr_capture
            try:
                parser = build_smoke_parser()
                args = parser.parse_args(argv)
            finally:
                sys.stderr = old_stderr
        except SystemExit as pexc:
            # argparse raises SystemExit(0) for --help, SystemExit(2) for errors.
            # Distinguish from timeout code 2.
            # Capture real argparse stderr text — str(SystemExit(2)) is just "2".
            parser_stderr = stderr_capture.getvalue().strip()
            pcode = pexc.code if isinstance(pexc.code, int) else 0
            preason = parser_stderr or str(pexc) or "argparse error"
            pclass = "argparse-help" if pcode == 0 else "argparse"
            try:
                _write_fail_marker(mlx_work or pathlib.Path.cwd(), preason)
            except Exception:
                pass
            try:
                _write_smoke_report({
                    "status": "fail", "failure_code": pclass,
                    "failure_message": preason,
                    "wall_clock_seconds": time.monotonic() - start,
                    "cleanup_warnings": list(_CLEANUP_WARNINGS),
                })
            except Exception:
                pass
            return pcode or SMOKE_ABORT_PREFLIGHT

        # --- Resolve workspace from parsed CLI args ---
        adapter_path = pathlib.Path(str(args.adapter_path)).resolve()
        mlx_work = adapter_path.parent.resolve()
        _clear_markers(mlx_work)

        # --- Config merge ---
        try:
            args = _merge_config(args)
        except SystemExit:
            _write_fail_marker(mlx_work, "config merge failed")
            _write_smoke_report({
                "status": "fail", "failure_code": "config",
                "failure_message": "config merge failed",
                "wall_clock_seconds": time.monotonic() - start,
            })
            return SMOKE_ABORT_PREFLIGHT

        # --- Pinned asset paths: reject any deviation from Phase-1 contract ---
        for attr, expected in _PINNED_SMOKE_PATHS.items():
            actual_val = getattr(args, attr, None)
            actual_str = str(actual_val) if actual_val is not None else ""
            expected_str = str(expected) if expected is not None else ""
            if actual_str != expected_str:
                _write_fail_marker(mlx_work,
                    f"pinned smoke path {attr}: expected {expected!r}, got {actual_val!r}")
                _write_smoke_report({
                    "status": "fail", "failure_code": "pinned-paths",
                    "failure_message":
                        f"pinned smoke path {attr}: expected {expected!r}, got {actual_val!r}",
                    "wall_clock_seconds": time.monotonic() - start,
                    "cleanup_warnings": list(_CLEANUP_WARNINGS),
                })
                return SMOKE_ABORT_PREFLIGHT

        # --- Pinned smoke identity: reject any deviation from Phase-1 contract ---
        for attr, expected in _PINNED_SMOKE_VALUES.items():
            actual = getattr(args, attr, None)
            if actual != expected:
                _write_fail_marker(mlx_work,
                    f"pinned smoke arg {attr}: expected {expected!r}, got {actual!r}")
                _write_smoke_report({
                    "status": "fail", "failure_code": "pinned-args",
                    "failure_message":
                        f"pinned smoke arg {attr}: expected {expected!r}, got {actual!r}",
                    "wall_clock_seconds": time.monotonic() - start,
                    "cleanup_warnings": list(_CLEANUP_WARNINGS),
                })
                return SMOKE_ABORT_PREFLIGHT

        # --- Preflight ---
        _check_preflight(args, mlx_work)

        # --- Training ---
        provider = _run(args)

        elapsed = time.monotonic() - start

        # --- Post-run validation (unconditional — no provider=None guard) ---
        if not (adapter_path / "adapters.safetensors").is_file():
            raise SystemExit("adapters.safetensors not found after train")
        if not (adapter_path / "adapter_config.json").is_file():
            raise SystemExit("adapter_config.json not found after train")

        if provider.call_count != 1:
            raise SystemExit(
                f"provider called {provider.call_count} times (expected 1)"
            )
        if provider.captured_loss is None:
            raise SystemExit("loss not captured")
        if provider.captured_token_count is None or provider.captured_token_count <= 0:
            raise SystemExit(
                f"invalid token count: {provider.captured_token_count}"
            )
        if provider.captured_all_finite is not True:
            raise SystemExit("gradients contain nonfinite values")
        if provider.captured_gradient_paths is None:
            raise SystemExit("gradient paths not captured")
        if provider.captured_gradient_shapes is None:
            raise SystemExit("gradient shapes not captured")
        if provider.captured_gradient_leaf_count is None or provider.captured_gradient_leaf_count == 0:
            raise SystemExit("zero gradient leaves captured")
        # Token count equality vs default_loss mask formula (required, never None)
        expected_tc = provider.compute_expected_token_count()
        if expected_tc != provider.captured_token_count:
            raise SystemExit(
                f"token count mismatch: captured={provider.captured_token_count} "
                f"expected={expected_tc}"
            )

        # --- Success report + marker (report must be durable before OK) ---
        report_data = {
            "status": "ok",
            "loss": provider.captured_loss,
            "token_count": provider.captured_token_count,
            "gradient_paths": provider.captured_gradient_paths,
            "gradient_shapes": provider.captured_gradient_shapes,
            "gradient_dtypes": provider.captured_gradient_dtypes,
            "gradient_leaf_count": provider.captured_gradient_leaf_count,
            "finite": provider.captured_all_finite,
            "provider_call_count": provider.call_count,
            "wall_clock_seconds": elapsed,
        }
        try:
            _write_smoke_report(report_data)
        except Exception:
            try:
                _write_fail_marker(mlx_work, "report write failed")
            except Exception:
                pass
            return SMOKE_ABORT_TRAIN
        # OK marker only after report succeeds; write failure returns TRAIN abort
        try:
            _write_ok_marker(mlx_work)
        except Exception:
            try:
                _write_fail_marker(mlx_work, "ok marker write failed")
            except Exception:
                pass
            return SMOKE_ABORT_TRAIN
        return 0

    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else SMOKE_ABORT_TRAIN
        # Preserve preflight exit code 3 from _check_preflight
        if code == SMOKE_ABORT_PREFLIGHT:
            pass  # preserve code 3
        elif code == SMOKE_ABORT_TIMEOUT:
            pass  # preserve code 2
        elif _TIMEOUT_FLAG:
            code = SMOKE_ABORT_TIMEOUT
        else:
            code = SMOKE_ABORT_TRAIN
        # Preserve detailed diagnostic: _LAST_FAIL_REASON is set by
        # _write_fail_marker BEFORE each raise in _check_preflight.
        # str(SystemExit(3)) == "3" — useless. Only use it as fallback.
        exc_str = str(exc)
        if exc_str != str(code) and exc_str not in ("", "None"):
            detailed = exc_str
        else:
            detailed = None
        reason = _TIMEOUT_REASON or detailed or _LAST_FAIL_REASON or exc_str or "unknown"
        fcode = "timeout" if _TIMEOUT_FLAG else ("preflight" if code == SMOKE_ABORT_PREFLIGHT else "training-error")
        try:
            _write_fail_marker(mlx_work or pathlib.Path.cwd(), reason)
        except Exception:
            pass
        try:
            _write_smoke_report({
                "status": "fail", "failure_code": fcode,
                "failure_message": reason,
                "wall_clock_seconds": time.monotonic() - start,
                "cleanup_warnings": list(_CLEANUP_WARNINGS),
            })
        except Exception:
            pass
        return code

    except MemoryError as exc:
        try:
            _write_fail_marker(mlx_work or pathlib.Path.cwd(), f"out of memory: {exc}")
        except Exception:
            pass
        try:
            _write_smoke_report({
                "status": "fail", "failure_code": "oom",
                "failure_message": str(exc),
                "wall_clock_seconds": time.monotonic() - start,
                "cleanup_warnings": list(_CLEANUP_WARNINGS),
            })
        except Exception:
            pass
        return SMOKE_ABORT_TRAIN

    except Exception as exc:
        try:
            _write_fail_marker(mlx_work or pathlib.Path.cwd(), f"unhandled: {exc}")
        except Exception:
            pass
        try:
            _write_smoke_report({
                "status": "fail", "failure_code": "exception",
                "failure_message": str(exc),
                "wall_clock_seconds": time.monotonic() - start,
                "cleanup_warnings": list(_CLEANUP_WARNINGS),
            })
        except Exception:
            pass
        return SMOKE_ABORT_TRAIN

    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
        try:
            _release_ft_lock()
        except Exception as e:
            _CLEANUP_WARNINGS.append(f"cleanup release: {e}")
        # Persist cleanup warnings durably: stderr + warning marker on disk.
        if _CLEANUP_WARNINGS:
            for w in _CLEANUP_WARNINGS:
                print(f"cleanup-warning: {w}", file=sys.stderr)
            try:
                marker = _ARTIFACT_DIR / "cleanup-warnings.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    json.dumps({
                        "warnings": _CLEANUP_WARNINGS,
                        "timestamp": time.time(),
                    }) + "\n", encoding="utf-8")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
