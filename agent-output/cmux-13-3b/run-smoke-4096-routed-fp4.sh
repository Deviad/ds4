#!/usr/bin/env bash
set -uo pipefail

PROJECT='/Users/spotted/projects/ds4-finetuning'
MLX_WORK='/Volumes/Data NVME/mlx-ft/ds4'
DATA='/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096'
MODEL="$MLX_WORK/model-4bit"
OUT="$MLX_WORK/adapters-smoke-memory400"
STAMP="$(date '+%Y%m%d-%H%M%S')"
LOG="$PROJECT/agent-output/cmux-13-3b/smoke-4096-routed-fp4-$STAMP.log"
PIDFILE="$PROJECT/agent-output/cmux-13-3b/smoke-4096-routed-fp4.pid"
RCFILE="$PROJECT/agent-output/cmux-13-3b/smoke-4096-routed-fp4.rc"
DONE="$PROJECT/agent-output/cmux-13-3b/smoke-4096-routed-fp4.done"
LATEST="$PROJECT/agent-output/cmux-13-3b/smoke-4096-routed-fp4.latest"

rm -f "$RCFILE" "$DONE"
printf '%s\n' "$LOG" > "$LATEST"
printf '%s\n' "$$" > "$PIDFILE"

exec > >(tee -a "$LOG") 2>&1

echo "=== DS4 4096 routed-FP4 smoke start: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "commit=$(git -C "$PROJECT" rev-parse HEAD)"
echo "pid=$$"
echo "project=$PROJECT"
echo "model=$MODEL"
echo "data=$DATA"
echo "output=$OUT"
echo "policy=mx.disable_compile graph_limit=400000000000 stop_peak=340000000000"

if ps -axo pid,command | awk -v self="$$" '$1 != self && /finetune_ds4|mlx_lm[.]lora|run_smoke_4096_routed_fp4[.]py/ {found=1} END {exit !found}'; then
  echo 'STOP: another heavy training process is active'
  echo 97 > "$RCFILE"
  exit 97
fi

for path in "$PROJECT/python-envs/mlx/.venv/bin/activate" "$MODEL" "$DATA" "$MLX_WORK/lora-config.json"; do
  if [[ ! -e "$path" ]]; then
    echo "STOP: missing prerequisite $path"
    echo 98 > "$RCFILE"
    exit 98
  fi
done

if [[ -e "$OUT" ]]; then
  ARCHIVE="$OUT.pre-5dee4ce-$STAMP"
  echo "archiving_existing_output=$ARCHIVE"
  mv "$OUT" "$ARCHIVE"
fi

unset SSLKEYLOGFILE
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/python-envs/mlx/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PROJECT/python-envs/mlx/.venv/bin/activate"
cd "$PROJECT"

COMMAND=(
  python "$PROJECT/agent-output/cmux-13-3b/run_smoke_4096_routed_fp4.py"
  --config "$MLX_WORK/lora-config.json"
  --model "$MODEL"
  --train
  --data "$DATA"
  --adapter-path "$OUT"
  --fine-tune-type lora
  --iters 20
  --batch-size 1
  --learning-rate 1e-5
  --max-seq-length 4096
  --mask-prompt
  --grad-checkpoint
  --val-batches 1
  --steps-per-report 1
)
printf 'command='; printf '%q ' "${COMMAND[@]}"; printf '\n'

"${COMMAND[@]}"
rc=$?
printf '%s\n' "$rc" > "$RCFILE"
echo "=== DS4 4096 routed-FP4 smoke end: $(date '+%Y-%m-%d %H:%M:%S') exit=$rc ==="
if [[ "$rc" -eq 0 ]]; then
  touch "$DONE"
  echo DS4_SMOKE_PANEL_DONE
else
  echo DS4_SMOKE_PANEL_FAILED
fi
exit "$rc"
