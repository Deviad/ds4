#!/usr/bin/env bash
set -uo pipefail
cd /Users/spotted/projects/ds4-finetuning || exit 99
LOG="agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.log"
RC="agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.rc"
rm -f "$RC"
echo "=== coder 13.3b-5b smoke-train start: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "project=$PWD"
echo "mlx_work=/Volumes/Data NVME/mlx-ft/ds4"
echo "command=python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes"
unset SSLKEYLOGFILE
. python-envs/mlx/.venv/bin/activate
python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes
rc=$?
echo "=== coder 13.3b-5b smoke-train end: $(date '+%Y-%m-%d %H:%M:%S') exit=$rc ==="
echo "$rc" > "$RC"
exit "$rc"
