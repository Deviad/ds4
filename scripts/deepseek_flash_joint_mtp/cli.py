"""Public command-line surface for Slice 1."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import ConfigError
from .orchestrator import OrchestrationError, dry_run, plan, run, unavailable_action, verify
from .state import StateError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepseek_flash_joint_mtp_pipeline.py")
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--json", action="store_true")

    dry_parser = commands.add_parser("dry-run")
    dry_parser.add_argument("--config", required=True)
    dry_parser.add_argument("--fixture-root")
    dry_parser.add_argument("--json", action="store_true")

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--from-stage")
    run_parser.add_argument("--through-stage")

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument("--run-id", required=True)

    for name in ("promote", "cleanup"):
        action = commands.add_parser(name)
        action.add_argument("--config", required=True)
        action.add_argument("--run-id", required=True)
        action.add_argument("--confirm", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.config)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) if args.json else _human_plan(result))
            return 0
        if args.command == "dry-run":
            result = dry_run(args.config, args.fixture_root)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) if args.json else _human_plan(result))
            return 0
        if args.command == "run":
            code, run_id = run(args.config, run_id=args.run_id, resume=args.resume, from_stage=args.from_stage, through_stage=args.through_stage)
            print(json.dumps({"status": "ok" if code == 0 else "failed", "run_id": run_id, "exit_code": code}, sort_keys=True))
            return code
        if args.command == "verify":
            print(json.dumps(verify(args.config, args.run_id), sort_keys=True))
            return 0
        unavailable_action(args.config, args.run_id, args.command, args.confirm)
        return 0
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except StateError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except OrchestrationError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 5


def _human_plan(value: dict[str, object]) -> str:
    lines = [f"adapter: {value['adapter']}", f"registry: {value['registry_digest']}"]
    for node in value["nodes"]:
        lines.append(f"{node['name']}: {node['kind']} ({node.get('reason', node['adapter_status'])})")
    return "\n".join(lines)
