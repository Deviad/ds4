"""Canonical JSON, digests, atomic writes, and small-file identities."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


class CanonicalError(ValueError):
    """Raised when data cannot be represented canonically or safely."""


def _reject_constant(value: str) -> None:
    raise CanonicalError(f"non-finite JSON number: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_bytes(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CanonicalError) as exc:
        raise CanonicalError(str(exc)) from exc


def load_json(path: Path) -> Any:
    return load_json_bytes(path.read_bytes())


def _number(value: Decimal | int | float) -> str:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise CanonicalError("non-finite float")
        value = Decimal(str(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise CanonicalError("invalid decimal") from exc
    if not decimal.is_finite():
        raise CanonicalError("non-finite decimal")
    if decimal == 0:
        return "0"
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def canonical_dumps(value: Any) -> str:
    """Serialize JSON-compatible data without whitespace or unstable numbers."""

    def render(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, (Decimal, int, float)) and not isinstance(item, bool):
            return _number(item)
        if isinstance(item, str):
            if unicodedata.normalize("NFC", item) != item:
                raise CanonicalError("non-NFC string")
            return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise CanonicalError("JSON object keys must be strings")
            keys = sorted(item)
            return "{" + ",".join(
                render(key) + ":" + render(item[key]) for key in keys
            ) + "}"
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(render(part) for part in item) + "]"
        raise CanonicalError(f"unsupported JSON value: {type(item).__name__}")

    return render(value)


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def digest_payload(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    atomic_write_bytes(path, canonical_bytes(value) + b"\n", mode=mode)


def _safe_stat(path: Path) -> os.stat_result:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise CanonicalError(f"symlink is not an accepted identity: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise CanonicalError(f"not a regular file: {path}")
    return before


@dataclass(frozen=True)
class FileIdentity:
    kind: str
    canonical_path: str
    size: int
    sha256: str
    mode: int
    mtime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "canonical_path": self.canonical_path,
            "size": self.size,
            "sha256": self.sha256,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
        }


def file_identity(path: Path) -> FileIdentity:
    path = Path(path)
    before = _safe_stat(path)
    canonical = path.resolve(strict=True)
    file_digest = sha256_file(path)
    after = _safe_stat(path)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CanonicalError(f"file changed while hashing: {path}")
    return FileIdentity(
        kind="file",
        canonical_path=str(canonical),
        size=after.st_size,
        sha256=file_digest,
        mode=stat.S_IMODE(after.st_mode),
        mtime_ns=after.st_mtime_ns,
    )


def identity_dict(path: Path) -> dict[str, Any]:
    return file_identity(path).as_dict()
