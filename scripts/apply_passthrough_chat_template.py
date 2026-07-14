#!/usr/bin/env python3
"""Patch DS4 MLX tokenizer_config.json with a passthrough chat template.

The Epic 2 dataset already stores DeepSeek V4 chat-framed prompts and raw
assistant completions.  mlx_lm's prompt/completion loader still calls
``apply_chat_template`` over message dictionaries, so the tokenizer needs an
identity template that concatenates message contents without adding any tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

PASSTHROUGH_CHAT_TEMPLATE = "{{ messages | map(attribute='content') | join('') }}"
DEFAULT_MLX_WORK = "/Volumes/Data NVME/mlx-ft/ds4"


class PatchError(RuntimeError):
    """Raised when the tokenizer config cannot be safely patched."""


def default_tokenizer_config() -> pathlib.Path:
    mlx_work = os.environ.get("MLX_WORK", DEFAULT_MLX_WORK)
    return pathlib.Path(mlx_work) / "model-4bit" / "tokenizer_config.json"


def load_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PatchError(f"{path}: missing tokenizer_config.json") from exc
    except json.JSONDecodeError as exc:
        raise PatchError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PatchError(f"{path}: expected top-level JSON object")
    return data


def patch_config(path: pathlib.Path, *, force: bool = False, dry_run: bool = False) -> str:
    data = load_json_object(path)
    existing = data.get("chat_template")
    if existing == PASSTHROUGH_CHAT_TEMPLATE:
        return "unchanged"
    if existing is not None and not force:
        raise PatchError(
            f"{path}: chat_template already exists and differs; rerun with --force to replace"
        )
    data["chat_template"] = PASSTHROUGH_CHAT_TEMPLATE
    if dry_run:
        return "would-update"

    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)
    return "updated"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install passthrough DS4 chat_template into tokenizer_config.json."
    )
    parser.add_argument(
        "--tokenizer-config",
        type=pathlib.Path,
        default=default_tokenizer_config(),
        help="Path to tokenizer_config.json (default: $MLX_WORK/model-4bit/tokenizer_config.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-matching chat_template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report whether a change would be made without writing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    path = args.tokenizer_config.expanduser()
    try:
        status = patch_config(path, force=args.force, dry_run=args.dry_run)
    except PatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{status}: {path}")
    if status in {"updated", "would-update"}:
        print(f"chat_template={PASSTHROUGH_CHAT_TEMPLATE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
