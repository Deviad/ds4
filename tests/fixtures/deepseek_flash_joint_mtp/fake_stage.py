#!/usr/bin/env python3
"""Tiny child process used only by the Slice 1 fixture adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-digest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--checkpoint")
    parser.add_argument("--force-exit", type=int)
    parser.add_argument("--argv0", required=True)
    args = parser.parse_args()
    if os.environ.get("FAKE_IGNORE_TERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif os.environ.get("FAKE_TERM_EXIT_ZERO"):
        signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    if os.environ.get("FAKE_STAGE_SLEEP"):
        time.sleep(float(os.environ["FAKE_STAGE_SLEEP"]))
    dependencies = {}
    for item in args.dependency:
        name, separator, digest = item.partition("=")
        if not separator:
            return 64
        dependencies[name] = digest
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    resumed = False
    if checkpoint is not None:
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if (saved.get("stage") != args.stage or saved.get("run_id") != args.run_id or
                    saved.get("config_digest") != args.config_digest or
                    saved.get("dependencies") != dependencies):
                return 65
            resumed = True
        else:
            checkpoint.write_text(json.dumps({
                "stage": args.stage,
                "run_id": args.run_id,
                "config_digest": args.config_digest,
                "dependencies": dependencies,
            }, sort_keys=True))
            return 75
    if args.force_exit is not None:
        return args.force_exit
    observed_environment = dict(os.environ)
    payload = {
        "stage": args.stage,
        "run_id": args.run_id,
        "config_digest": args.config_digest,
        "dependencies": dependencies,
        "argv": [args.argv0] + sys.argv,
        "observed_environment": observed_environment,
        "resumed_from_checkpoint": resumed,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
