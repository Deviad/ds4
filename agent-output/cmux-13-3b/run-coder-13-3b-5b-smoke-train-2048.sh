#!/usr/bin/env bash
set -uo pipefail
cd /Users/spotted/projects/ds4-finetuning || exit 99
RC="agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.rc"
echo "=== coder 13.3b-5b smoke-train-2048 fallback start: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "reason=4096 OOM: [METAL] Command buffer execution failed: Insufficient Memory"
echo "project=$PWD"
echo "mlx_work=/Volumes/Data NVME/mlx-ft/ds4"
echo "command=python3 scripts/finetune_ds4.py run-command smoke-train-2048 --execute --yes"
unset SSLKEYLOGFILE
. python-envs/mlx/.venv/bin/activate
rm -rf '/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-2048'
python3 scripts/finetune_ds4.py run-command smoke-train-2048 --execute --yes
rc=$?
echo "=== coder 13.3b-5b smoke-train-2048 fallback end: $(date '+%Y-%m-%d %H:%M:%S') exit=$rc ==="
echo "$rc" > "$RC"
exit "$rc"
