#!/usr/bin/env bash
set -o pipefail
PROJECT='/Users/spotted/projects/ds4-finetuning'
MLX_WORK='/Volumes/Data NVME/mlx-ft/ds4'
LOG="$PROJECT/agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log"
RCFILE="$PROJECT/agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.rc"
{
  echo "=== coder 13.3b-5 smoke-train start: $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "project=$PROJECT"
  echo "mlx_work=$MLX_WORK"
  echo "command=python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes"
  cd "$PROJECT" || exit 111
  unset SSLKEYLOGFILE
  . "$PROJECT/python-envs/mlx/.venv/bin/activate"
  rm -rf "$MLX_WORK/adapters-smoke"
  python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes
  rc=$?
  echo "=== coder 13.3b-5 smoke-train end: $(date '+%Y-%m-%d %H:%M:%S') exit=$rc ==="
  echo "$rc" > "$RCFILE"
  exit "$rc"
} >> "$LOG" 2>&1
