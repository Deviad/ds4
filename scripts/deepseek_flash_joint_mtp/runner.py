"""The single subprocess boundary for the pipeline."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .canonical import atomic_write_json, digest_payload


class RunnerError(RuntimeError):
    """Raised when a launch cannot be safely prepared or recorded."""


@dataclass(frozen=True)
class LaunchSpec:
    stage: str
    attempt_id: str
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    tool_identity: Mapping[str, object]
    input_identities: Mapping[str, object]
    temporary_outputs: tuple[Path, ...]
    final_outputs: tuple[Path, ...]

    def payload(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "attempt_id": self.attempt_id,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "environment": dict(self.environment),
            "tool_identity": dict(self.tool_identity),
            "input_identities": dict(self.input_identities),
            "temporary_outputs": [str(path) for path in self.temporary_outputs],
            "final_outputs": [str(path) for path in self.final_outputs],
        }


def _validate_launch(spec: LaunchSpec) -> None:
    if not spec.argv or any(not isinstance(item, str) or "\x00" in item for item in spec.argv):
        raise RunnerError("argv must be a non-empty NUL-free string array")
    if not Path(spec.argv[0]).is_absolute():
        raise RunnerError("argv[0] must be absolute")
    if not spec.cwd.is_absolute():
        raise RunnerError("cwd must be absolute")
    for key, value in spec.environment.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise RunnerError(f"invalid environment key: {key}")
        if not isinstance(value, str) or "\x00" in value:
            raise RunnerError(f"invalid environment value: {key}")
    for path in (*spec.temporary_outputs, *spec.final_outputs):
        if not path.is_absolute():
            raise RunnerError("output paths must be absolute")


@dataclass(frozen=True)
class ProcessResult:
    started_at: str
    ended_at: str
    pid: int | None
    status: dict[str, object]
    log_identity: dict[str, object] | None
    log_error: str | None
    forwarded_signals: tuple[int, ...]

    @property
    def return_code(self) -> int:
        if self.status["kind"] == "exit":
            return int(self.status["code"])
        return 128 + int(self.status["signal"])

    def as_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "pid": self.pid,
            "status": self.status,
            "log_identity": self.log_identity,
            "log_error": self.log_error,
            "forwarded_signals": list(self.forwarded_signals),
        }


def run_process(
    spec: LaunchSpec,
    attempt_dir: Path,
    *,
    manifest_event: Callable[[str, dict[str, object]], None],
    log_path_override: Path | None = None,
) -> ProcessResult:
    """Persist launch intent, run one child, and preserve its raw status."""

    _validate_launch(spec)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    for output in spec.temporary_outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    launch_payload = spec.payload()
    launch_payload["launch_digest"] = digest_payload(launch_payload)
    launch_path = attempt_dir / "launch.json"
    atomic_write_json(launch_path, launch_payload)
    manifest_event("launch-prepared", {"launch_digest": launch_payload["launch_digest"]})
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    child: subprocess.Popen[bytes] | None = None
    forwarded: list[int] = []
    interrupted = {"signal": None}
    grace_deadline: float | None = None
    old_handlers: dict[int, object] = {}

    def forward(signum: int, _frame: object) -> None:
        nonlocal grace_deadline
        forwarded.append(signum)
        if interrupted["signal"] is None:
            interrupted["signal"] = signum
            grace_deadline = time.monotonic() + 0.5
            target_signal = signum
        else:
            grace_deadline = time.monotonic()
            target_signal = signal.SIGKILL
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, target_signal)
            except ProcessLookupError:
                pass

    def kill_and_reap() -> None:
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.signal(signum, forward)
        try:
            child = subprocess.Popen(
                list(spec.argv),
                shell=False,
                cwd=str(spec.cwd),
                env=dict(spec.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            atomic_write_json(attempt_dir / "failure.json", {"failure_class": "launch", "diagnostic": str(exc)})
            raise RunnerError(str(exc)) from exc
        process_started = {"pid": child.pid, "started_at": started_at, "launch_digest": launch_payload["launch_digest"]}
        atomic_write_json(attempt_dir / "process-started.json", process_started)
        try:
            manifest_event("process-started", process_started)
        except Exception as exc:
            kill_and_reap()
            raise RunnerError(f"process-started manifest persistence failed: {exc}") from exc
        log_path = log_path_override or (attempt_dir / "stdout.log")
        log_error: str | None = None
        try:
            with log_path.open("xb") as log:
                output = b""
                while True:
                    try:
                        output, _ = child.communicate(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        if grace_deadline is not None and time.monotonic() >= grace_deadline:
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            grace_deadline = None
                log.write(output or b"")
                log.flush()
                os.fsync(log.fileno())
            try:
                manifest_event("log-published", {"log_path": str(log_path)})
            except Exception as exc:
                raise RunnerError(f"log-published manifest persistence failed: {exc}") from exc
        except OSError as exc:
            log_error = str(exc)
            try:
                child.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                kill_and_reap()
        raw_return = child.returncode
        ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if raw_return is not None and raw_return < 0:
            status = {"kind": "signal", "signal": -raw_return}
        else:
            status = {"kind": "exit", "code": int(raw_return or 0)}
        log_identity: dict[str, object] | None = None
        if log_error is None:
            from .canonical import identity_dict
            log_identity = identity_dict(log_path)
        result = ProcessResult(started_at, ended_at, child.pid, status, log_identity, log_error, tuple(forwarded))
        atomic_write_json(attempt_dir / "result.json", result.as_dict())
        manifest_event("process-finished", {"status": status, "result_digest": digest_payload(result.as_dict())})
        return result
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
