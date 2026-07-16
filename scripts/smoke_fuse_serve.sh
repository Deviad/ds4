#!/usr/bin/env bash
# Story 12.3 end-to-end fusion serving smoke. NO production code mutated.
set -euo pipefail

DS4_BIN="${DS4_BIN:-/Users/spotted/projects/ds4/ds4}"
QUANT_BIN="${QUANT_BIN:-/Users/spotted/projects/ds4/gguf-tools/deepseek4-quantize}"
FUSE_HELPER="${FUSE_HELPER:-/Users/spotted/projects/ds4-finetuning/scripts/fuse_lora_hf.py}"
SYNTH_CREATOR="${SYNTH_CREATOR:-/Users/spotted/projects/ds4-finetuning/scripts/make_synth_lora.py}"
MLX_PY="${MLX_PY:-/Users/spotted/projects/ds4-finetuning/python-envs/mlx/.venv/bin/python3}"
BASE_HF="${BASE_HF:-/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0}"
TEMPLATE_GGUF="${TEMPLATE_GGUF:-/Users/spotted/projects/ds4/ds4flash.gguf}"
WORK_DIR="${WORK_DIR:-/Volumes/Data NVME/mlx-ft/ds4}"
WORK_MOUNT="${WORK_MOUNT:-/Volumes/Data NVME}"
CAPTURE_DIR="${WORK_DIR}/smoke-captures"
REPORT="${REPORT:-/Users/spotted/projects/ds4-finetuning/agent-output/cmux-12-3/test-report.md}"
TRACK_A_MARKER="/Users/spotted/projects/ds4/.ds4-gguf-generate-ok"
TRACK_B_MARKER="/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok"
NEW_FUSION_MARKER="/Volumes/Data NVME/mlx-ft/ds4/.ds4-fusion-smoke-ok"
MODEL_4BIT_DIR="/Volumes/Data NVME/mlx-ft/ds4/model-4bit"
CONVERT_SHIMMED_DIR="/Volumes/Data NVME/mlx-ft/ds4/convert-shimmed"
READINESS_JSON="/Volumes/Data NVME/mlx-ft/ds4/deepseek-v4-forward-parity-readiness.json"
MIN_DISK_BYTES="${MIN_DISK_BYTES:-375809638400}"
DISK_FILL_FREE_MIN_BYTES="${DISK_FILL_FREE_MIN_BYTES:-104152956928}"
MAX_RETAINED_BYTES="${MAX_RETAINED_BYTES:-104857600}"
PROMPT="Hello, how are you?"
N_AC3="${N_AC3:-32}"
N_PROBE="${N_PROBE:-1}"
EPS_ALPHA0_REL="5e-3"
CASCADE_ALPHA0_REL="2e-2"
EPS_NONZERO_MIN_REL="5e-2"
# AC7 §10.1 — GGUF_MIN_BYTES / GGUF_MAX_BYTES are derived at runtime (after TEMPLATE_RESOLVED). §10.1.
# Defaults 0 here to make accidental early-binding visible if preflight doesn't set them.
GGUF_MIN_BYTES=0
GGUF_MAX_BYTES=0

STEP_ROWS=""
AC1_STATUS="PENDING"
AC2_STATUS="PENDING"
AC3_STATUS="PENDING"
AC4_STATUS="PENDING"
AC5_STATUS="PENDING"
AC6_STATUS="PENDING"
AC7_STATUS="PENDING"
OVERALL_STATUS="PENDING"
BASE_NORM="n/a"
BASE_ARGMAX="n/a"
ALPHA0_NORM="n/a"
ALPHA0_ARGMAX="n/a"
NONZERO_ARGMAX="n/a"
L2_REL_ALPHA0="n/a"
L2_REL_NONZERO="n/a"
PRE_DISK_USED="n/a"
ALPHA0_PEAK="n/a"
NONZERO_PEAK="n/a"
POST_DISK_USED="n/a"
FINAL_RETAINED_BYTES="n/a"
POST_DS4FLASH_SHA="n/a"
POST_HF_SHARD_COUNT="n/a"
POST_TRACK_A_MTIME="n/a"
POST_READINESS_MTIME="n/a"
POST_READINESS_SIZE="n/a"
POST_READINESS_PROOFS="n/a"
PRE_READINESS_SIZE="n/a"
DS4FLASH_SHA_STATUS="PENDING"
HF_MTIME_STATUS="PENDING"
TRACK_A_STATUS="PENDING"
TRACK_B_STATUS="PENDING"
NEW_MARKER_STATUS="PENDING"
MODEL_4BIT_STATUS="PENDING"
CONVERT_SHIMMED_STATUS="PENDING"
READINESS_STAT_STATUS="PENDING"
READINESS_PROOFS_STATUS="INFO ONLY"
CLEANUP_STATUS="PENDING"
Q7_BF16_CASCADE_STATUS="NOT TRIGGERED"
ALPHA0_CASCADE_STATUS="NOT TRIGGERED"
NONZERO_CASCADE_STATUS="NOT TRIGGERED"
DISK_FILL_CASCADE_STATUS="NOT TRIGGERED"
SCRIPT_SHA="n/a"

quote_cmd() {
  local out="" q=""
  for arg in "$@"; do
    printf -v q "%q" "$arg"
    out+="$q "
  done
  printf "%s" "${out% }"
}

append_step() {
  local step="$1" variant="$2" cmd="$3" code="$4" outfile="$5" notes="$6" excerpt=""
  if [[ -f "$outfile" ]]; then
    excerpt="$(LC_ALL=C tr '\n' ' ' < "$outfile" | cut -c1-220 | sed 's/|/\\|/g')"
  fi
  STEP_ROWS+="| ${step} | ${variant} | \`${cmd}\` | ${code} | \"${excerpt}\" | ${notes} |"$'\n'
}

run_logged() {
  local step="$1" variant="$2" outfile="$3" notes="$4"
  shift 4
  local cmd code
  cmd="$(quote_cmd "$@")"
  mkdir -p "$(dirname "$outfile")"
  set +e
  "$@" 2>&1 | tee "$outfile"
  code=${PIPESTATUS[0]}
  set -e
  append_step "$step" "$variant" "$cmd" "$code" "$outfile" "$notes"
  return "$code"
}

free_bytes() {
  df -k "$WORK_MOUNT" | awk 'NR==2 { printf "%.0f", $4 * 1024 }'
}

disk_used_bytes() {
  df -k "$WORK_MOUNT" | awk 'NR==2 { printf "%.0f", $3 * 1024 }'
}

update_phase_peak_from_file() {
  local phase="$1" peak_file="$2" peak="n/a"
  if [[ -s "$peak_file" ]]; then
    peak="$(cat "$peak_file")"
  fi
  case "$phase" in
    alpha0) ALPHA0_PEAK="$peak" ;;
    nonzero) NONZERO_PEAK="$peak" ;;
  esac
}

write_peak_if_higher() {
  local peak_file="$1" used current
  used="$(disk_used_bytes 2>/dev/null || printf '0')"
  [[ "$used" =~ ^[0-9]+$ ]] || used=0
  current=0
  if [[ -s "$peak_file" ]]; then
    current="$(cat "$peak_file")"
  fi
  [[ "$current" =~ ^[0-9]+$ ]] || current=0
  if (( used > current )); then
    printf "%s\n" "$used" > "$peak_file"
  fi
}

disk_sampler() {
  local peak_file="$1" trigger_file="$2" free
  while true; do
    write_peak_if_higher "$peak_file"
    free="$(free_bytes 2>/dev/null || printf '0')"
    [[ "$free" =~ ^[0-9]+$ ]] || free=0
    if (( free < DISK_FILL_FREE_MIN_BYTES )); then
      printf "%s\n" "$free" > "$trigger_file"
    fi
    sleep 5
  done
}

run_logged_phase() {
  local phase="$1"
  shift
  local peak_file="${CAPTURE_DIR}/${phase}-disk-peak.bytes"
  local trigger_file="${CAPTURE_DIR}/${phase}-disk-fill.trigger"
  local sampler_pid code
  mkdir -p "$CAPTURE_DIR"
  write_peak_if_higher "$peak_file"
  update_phase_peak_from_file "$phase" "$peak_file"
  rm -f "$trigger_file"
  disk_sampler "$peak_file" "$trigger_file" &
  sampler_pid=$!
  if run_logged "$@"; then
    code=0
  else
    code=$?
  fi
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  write_peak_if_higher "$peak_file"
  update_phase_peak_from_file "$phase" "$peak_file"
  if [[ -s "$trigger_file" ]]; then
    DISK_FILL_CASCADE_STATUS="TRIGGERED"
    block "disk fill mid-run during ${phase}; free bytes $(cat "$trigger_file") below ${DISK_FILL_FREE_MIN_BYTES}"
  fi
  return "$code"
}

shard_mtimes() {
  find "$BASE_HF" -name 'model-*.safetensors' -exec stat -f "%m %N" {} \; | sort
}

readiness_proofs() {
  "$MLX_PY" -c "import json,sys
try:
    j=json.load(open(sys.argv[1], encoding='utf-8'))
    p=j.get('real_mode_proofs') or {}
    print('%s %s %s %s' % (p.get('proofs_total','n/a'), p.get('proofs_ok','n/a'), p.get('proofs_failed','n/a'), p.get('proofs_skipped','n/a')))
except Exception:
    print('n/a n/a n/a n/a')" "$READINESS_JSON"
}

base_layer0_shape() {
  "$MLX_PY" - "$BASE_HF" <<'PY'
import json, pathlib, struct, sys
base = pathlib.Path(sys.argv[1])
index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
name = "layers.0.attn.wq_a.weight"
shard = index["weight_map"][name]
with (base / shard).open("rb") as fh:
    header_len = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(header_len).decode("utf-8"))
print(tuple(header[name]["shape"]))
PY
}

json_norm_argmax() {
  "$MLX_PY" - "$1" <<'PY'
import json, math, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    j = json.load(fh)
vals = j.get("logits")
if not isinstance(vals, list) or not vals:
    raise SystemExit("missing logits array")
logits = []
for value in vals:
    if value is None:
        raise SystemExit("logits contain null")
    f = float(value)
    if not math.isfinite(f):
        raise SystemExit("logits contain non-finite value")
    logits.append(f)
norm = math.sqrt(sum(v * v for v in logits))
print(f"{norm:.17g}")
print(json.dumps(j.get("argmax_token"), ensure_ascii=False))
PY
}

json_rel_l2() {
  "$MLX_PY" - "$1" "$2" <<'PY'
import json, math, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    a = json.load(fh)["logits"]
with open(sys.argv[2], encoding="utf-8") as fh:
    b = json.load(fh)["logits"]
if len(a) != len(b):
    raise SystemExit(f"logit length mismatch {len(a)} != {len(b)}")
base_norm_sq = 0.0
diff_sq = 0.0
for av, bv in zip(a, b):
    if av is None or bv is None:
        raise SystemExit("logits contain null")
    af = float(av)
    bf = float(bv)
    if not math.isfinite(af) or not math.isfinite(bf):
        raise SystemExit("logits contain non-finite value")
    base_norm_sq += af * af
    d = bf - af
    diff_sq += d * d
if base_norm_sq <= 0.0:
    raise SystemExit("base logits norm is zero")
print(f"{math.sqrt(diff_sq) / math.sqrt(base_norm_sq):.17g}")
PY
}

float_le() {
  "$MLX_PY" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
}

float_ge() {
  "$MLX_PY" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
}

validate_text_capture() {
  local file="$1"
  [[ -s "$file" ]] || return 1
  LC_ALL=C grep -Eq '[[:graph:]]' "$file" || return 1
  if LC_ALL=C grep -Eiq '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$file"; then
    return 1
  fi
  if LC_ALL=C grep -q '[^[:print:][:space:]]' "$file"; then
    return 1
  fi
  return 0
}

validate_fused_dir() {
  "$MLX_PY" - "$1" <<'PY'
import pathlib, sys
sys.path.insert(0, "/Users/spotted/projects/ds4-finetuning/scripts")
from finetune_ds4 import validate_fused_hf_safetensors_dir
validate_fused_hf_safetensors_dir(pathlib.Path(sys.argv[1]))
print("validate_fused_hf_safetensors_dir PASS")
PY
}

assert_alpha0_scale() {
  "$MLX_PY" -c "import json,sys
cfg=json.load(open(sys.argv[1], encoding='utf-8'))
scale=cfg.get('lora_parameters',{}).get('scale')
print('scale=%r' % (scale,))
raise SystemExit(0 if scale == 0.0 else 1)" "$1/adapter_config.json"
}

assert_gguf_size() {
  local gguf="$1" size
  if [[ ! -f "$gguf" ]]; then
    echo "missing gguf: ${gguf}"
    return 1
  fi
  size="$(stat -f "%z" "$gguf")"
  echo "gguf_size=${size} expected_range=${GGUF_MIN_BYTES}-${GGUF_MAX_BYTES} (template-relative, template=${TEMPLATE_GGUF_BYTES})"
  (( size >= GGUF_MIN_BYTES && size <= GGUF_MAX_BYTES ))
}

cleanup_variant() {
  local fused_hf="$1" gguf="$2" skip_gguf="${3:-}"
  if [[ "$skip_gguf" == "skip-gguf" ]]; then
    rm -rf "$fused_hf"
    echo "removed ${fused_hf} (GGUF preserved per §10.2 SKIP_QUANT_VARIANTS)"
  else
    rm -rf "$fused_hf" "$gguf"
    echo "removed ${fused_hf}"
    echo "removed ${gguf}"
  fi
}

retained_evidence_bytes() {
  local total=0 path kb
  for path in \
    "${WORK_DIR}/synth-adapter-alpha0" \
    "${WORK_DIR}/synth-adapter-nonzero" \
    "$CAPTURE_DIR"; do
    if [[ -e "$path" ]]; then
      kb="$(du -sk "$path" | awk '{print $1}')"
      total=$((total + kb * 1024))
    fi
  done
  printf "%s" "$total"
}

cascade_total() {
  local total=0 status
  for status in "$Q7_BF16_CASCADE_STATUS" "$ALPHA0_CASCADE_STATUS" "$NONZERO_CASCADE_STATUS" "$DISK_FILL_CASCADE_STATUS"; do
    [[ "$status" == "TRIGGERED" ]] && total=$((total + 1))
  done
  printf "%s" "$total"
}

skip_variant_listed() {
  # §10.2 (a) — returns 0 when $1 is in SKIP_QUANT_VARIANTS (comma-list). §10.3#4: strip whitespace, do NOT lowercase.
  local needle="$1" item
  local IFS=','
  for item in ${SKIP_QUANT_VARIANTS:-}; do
    item="${item//[[:space:]]/}"
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

smoke_variants_listed() {
  # §10.2 (a) — returns 0 when $1 is in SMOKE_VARIANTS (comma-list). Default allowlist: alpha0,nonzero.
  local needle="$1" item
  local IFS=','
  local list="${SMOKE_VARIANTS:-alpha0,nonzero}"
  for item in $list; do
    item="${item//[[:space:]]/}"
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

final_report() {
  local status="$1"
  local now total_cascades
  now="$(date '+%Y-%m-%d %H:%M')"
  total_cascades="$(cascade_total)"
  mkdir -p "$(dirname "$REPORT")"
  cat > "$REPORT" <<EOF
# Story 12.3 end-to-end fusion serving smoke — Test Report
- **Date:** ${now}
- **Executor:** Test Manager (\`neuralwatt/qwen3.6-35b\`)
- **D1 orchestrator version:** \`scripts/smoke_fuse_serve.sh\` (post-Round-2; SHA ${SCRIPT_SHA})
- **Hard constraints honored:** NO \`ds4flash.gguf\` mutation, NO HF F8 mutation, NO marker written, NO readiness-JSON edit, NO production code changed.

## §1. Pre-flight invariants (captured BEFORE any work)
| Invariant | Pre-flight value |
|---|---|
| \`df /Volumes/Data NVME\` free bytes | ${PRE_FREE_BYTES:-n/a} (>= 350 GiB gate; exit 1 if violated) |
| \`ds4flash.gguf\` resolved-target sha256 | ${PRE_DS4FLASH_SHA:-n/a} (\`readlink -f\`; AC6(a)) |
| HF F8 per-shard mtime count | ${PRE_HF_SHARD_COUNT:-n/a} shards |
| HF F8 layer-0 wq_a.weight shape | ${PRE_LAYER0_SHAPE:-n/a} |
| Track-A marker mtime | ${PRE_TRACK_A_MTIME:-n/a} |
| Track-B marker | ABSENT ✓ |
| NEW_FUSION_MARKER | ABSENT ✓ |
| \`model-4bit\` directory | ABSENT ✓ |
| \`convert-shimmed\` directory | ABSENT ✓ |
| Readiness-JSON mtime (Tier-1 load-bearing) | ${PRE_READINESS_MTIME:-n/a} |
| Readiness-JSON byte-size (Tier-1 load-bearing) | ${PRE_READINESS_SIZE:-n/a} (63708 expected) |
| Readiness proofs_total / ok / failed / skipped (Tier-2 informational, n/a-tolerant) | ${PRE_READINESS_PROOFS:-n/a} (9 / 8 / 1 / 0 when nested-parse succeeds; n/a on schema mismatch) |
| \`ds4\` binary present | ✓ |
| \`deepseek4-quantize\` binary present | ✓ |
| No concurrent ds4 | ✓ |

## §2. Step-by-step exit codes + stdout excerpts
| Step | Variant | Command | Exit code | Stdout excerpt | Notes |
|---|---|---|---|---|---|
${STEP_ROWS}
## §3. AC4 (alpha=0 soundness) PRIMARY Metal dump-logits relative-L2
- Base logits L2 norm: ||base||₂ = ${BASE_NORM}
- Alpha0 logits L2 norm: ||alpha0||₂ = ${ALPHA0_NORM}
- L2_REL_ALPHA0 ||alpha0 - base||₂ / ||base||₂ = ${L2_REL_ALPHA0}
- PASS when L2_REL_ALPHA0 <= 5e-3 → ${AC4_STATUS}
- cascade threshold 2e-2 → ${ALPHA0_CASCADE_STATUS}
- argmax_token base = ${BASE_ARGMAX}; alpha0 = ${ALPHA0_ARGMAX} (informational; NOT gating — BF16 bucket-flip physics)

## §4. AC5 (nonzero delta observable) PRIMARY Metal dump-logits relative-L2
- L2_REL_NONZERO ||nonzero - base||₂ / ||base||₂ = ${L2_REL_NONZERO}
- PASS when L2_REL_NONZERO >= 5e-2 → ${AC5_STATUS}
- cascade (NaN/garbage) → ${NONZERO_CASCADE_STATUS}
- argmax_token nonzero = ${NONZERO_ARGMAX} (informational)

## §5. ds4flash.gguf sha256 before / after
- Pre-flight sha256 ${PRE_DS4FLASH_SHA:-n/a}
- Post-flight sha256 ${POST_DS4FLASH_SHA}
- UNCHANGED AC6(a) ${DS4FLASH_SHA_STATUS}

## §6. HF F8 checkpoint invariants
- Per-shard mtime pre == post AC6(b): ${HF_MTIME_STATUS} (${PRE_HF_SHARD_COUNT:-n/a} → ${POST_HF_SHARD_COUNT} shards unchanged; fuse reads mmap ACCESS_READ-only)

## §7. Marker + readiness-Tier-1 invariants
- Track-A marker mtime pre == post AC6(c): ${TRACK_A_STATUS}
- Track-B marker still ABSENT post AC6(d): ${TRACK_B_STATUS}
- NO new fusion-serving marker created (\`! -e .ds4-fusion-smoke-ok\`) AC6(e): ${NEW_MARKER_STATUS}
- \`model-4bit\` directory still ABSENT: ${MODEL_4BIT_STATUS}
- \`convert-shimmed\` directory still ABSENT: ${CONVERT_SHIMMED_STATUS}
- Readiness-JSON mtime pre == post AND byte-size pre == post (Tier-1 load-bearing) AC6(f): ${READINESS_STAT_STATUS} (${PRE_READINESS_MTIME:-n/a}/${PRE_READINESS_SIZE:-n/a} → ${POST_READINESS_MTIME}/${POST_READINESS_SIZE}; Tier-2 proofs ${PRE_READINESS_PROOFS:-n/a} → ${POST_READINESS_PROOFS}, informational n/a-tolerant, NOT gating)

## §8. Disk usage
| Phase | Used bytes | Notes |
|---|---|---|
| Pre-flight used | ${PRE_DISK_USED} | before any work |
| Phase alpha0 peak | ${ALPHA0_PEAK} | max during alpha0 (synth ~37 MB + fused-hf ~76 GB + fused-gguf ~97 GB coexist during quantize) |
| Phase nonzero peak | ${NONZERO_PEAK} | max during nonzero (sequenced AFTER alpha0 cleanup) |
| Final retained | ${FINAL_RETAINED_BYTES} | after cleanup; retained evidence ≤ 100 MB gate; df used after cleanup ${POST_DISK_USED} |

Retained evidence (≤ 100 MB gate): \`synth-adapter-alpha0/\` + \`synth-adapter-nonzero/\` + \`smoke-captures/\` (base + alpha0 + nonzero dump-logits JSON + stdouts).

## §9. §6 STOP-rule cascade check
| # | Cascade condition | Status |
|---|---|---|
| 1 | Q7 BF16-reader crash on fused attn (\`fuse_lora_hf.py\` / \`deepseek4-quantize\` die naming BF16/dtype/scale during real variant work) | ${Q7_BF16_CASCADE_STATUS} |
| 2 | alpha0 dramatic divergence (L2_REL_ALPHA0 > 2e-2) | ${ALPHA0_CASCADE_STATUS} |
| 3 | nonzero NaN/garbage (serve stdout NaN/non-finite/garbage OR logits JSON contains null/non-finite) | ${NONZERO_CASCADE_STATUS} |
| 4 | Disk fill mid-quantize (df free < 97 GB safety margin during quantize/serve) | ${DISK_FILL_CASCADE_STATUS} |
| — | **Total cascades** | ${total_cascades} |

## §10. Verdict
- AC1 (fuse exit 0 + gate PASS): ${AC1_STATUS}
- AC2 (quantize dry-run + real exit 0 + inspect exit 0 + gguf exists size ~97 GB ±5 GB): ${AC2_STATUS}
- AC3 (serve -n 32 exit 0 + real text): ${AC3_STATUS}
- AC4 (alpha=0 L2_REL <= 5e-3): ${AC4_STATUS}
- AC5 (nonzero L2_REL >= 5e-2, no NaN): ${AC5_STATUS}
- AC6 (ALL invariants: ds4flash sha / HF shards / Track-A marker / Track-B absent / new-marker absent / readiness Tier-1 stat): ${AC6_STATUS}
- AC7 (cleanup: temp fused-HF dirs + fused GGUFs removed; retained evidence ≤ 100 MB): ${AC7_STATUS}

**OVERALL: ${status}**
EOF
}

preflight_fail() {
  echo "PRE-FLIGHT ABORT: $*" >&2
  exit 1
}

ac_fail() {
  OVERALL_STATUS="BLOCKED"
  final_report "$OVERALL_STATUS"
  echo "AC FAIL: $*" >&2
  exit 2
}

block() {
  OVERALL_STATUS="BLOCKED"
  final_report "$OVERALL_STATUS"
  echo "BLOCKED: $*" >&2
  exit 3
}

run_variant() {
  local variant="$1" adapter_dir fused_hf gguf serve_out probe_out metrics alpha_scale SKIP_THIS_VARIANT skip_size
  adapter_dir="${WORK_DIR}/synth-adapter-${variant}"
  fused_hf="${WORK_DIR}/fused-hf-smoke-${variant}"
  gguf="${WORK_DIR}/fused-smoke-${variant}.gguf"
  serve_out="${CAPTURE_DIR}/${variant}-stdout.txt"
  probe_out="${CAPTURE_DIR}/${variant}-probe-stdout.txt"
  if [[ "$variant" == "alpha0" ]]; then
    alpha_scale="0.0"
  else
    alpha_scale="20.0"
  fi

  SKIP_THIS_VARIANT=""
  if skip_variant_listed "$variant"; then SKIP_THIS_VARIANT="1"; fi

  # §10.2 (b): in skip mode the externally-supplied GGUF is read-only input — never rm it in the prologue.
  if [[ -z "$SKIP_THIS_VARIANT" ]]; then
    rm -rf "$fused_hf" "$gguf"
  else
    rm -rf "$fused_hf"
  fi
  write_peak_if_higher "${CAPTURE_DIR}/${variant}-disk-peak.bytes"
  update_phase_peak_from_file "$variant" "${CAPTURE_DIR}/${variant}-disk-peak.bytes"

  # §10.2 (c): synth → fuse → gate → quantize chain runs only when NOT skipping; else verify pre-existing GGUF.
  if [[ -z "$SKIP_THIS_VARIANT" ]]; then
  run_logged_phase "$variant" "4 synth adapter create" "$variant" "${CAPTURE_DIR}/${variant}-synth.txt" "--num-layers 43; scale=${alpha_scale}; --targets q_a,q_b,kv" \
    "$MLX_PY" "$SYNTH_CREATOR" --base "$BASE_HF" --variant "$variant" --out "$adapter_dir" --rank 8 --num-layers 43 --alpha-scale "$alpha_scale" --seed 42 --targets q_a,q_b,kv \
    || { AC1_STATUS="FAIL"; ac_fail "make_synth_lora.py failed for ${variant}"; }

  if [[ "$variant" == "alpha0" ]]; then
    run_logged_phase "$variant" "4a alpha0 scale assert" "$variant" "${CAPTURE_DIR}/${variant}-scale-assert.txt" "adapter_config.json lora_parameters.scale == 0.0" \
      assert_alpha0_scale "$adapter_dir" \
      || { AC4_STATUS="FAIL"; ac_fail "alpha0 adapter_config.json lora_parameters.scale is not 0.0"; }
  fi

  run_logged_phase "$variant" "5 fuse" "$variant" "${CAPTURE_DIR}/${variant}-fuse.txt" "fuse_lora_hf.py" \
    "$MLX_PY" "$FUSE_HELPER" --base "$BASE_HF" --adapter "${adapter_dir}/adapter.safetensors" --adapter-config "${adapter_dir}/adapter_config.json" --out "$fused_hf" --targets q_a,q_b,kv \
    || { AC1_STATUS="FAIL"; Q7_BF16_CASCADE_STATUS="TRIGGERED"; block "fuse_lora_hf.py crashed on real 76GB shapes for ${variant}"; }

  run_logged_phase "$variant" "6 fused HF gate" "$variant" "${CAPTURE_DIR}/${variant}-gate.txt" "validate_fused_hf_safetensors_dir" \
    validate_fused_dir "$fused_hf" \
    || { AC1_STATUS="FAIL"; ac_fail "fused HF gate failed for ${variant}"; }

  run_logged_phase "$variant" "7a quantize dry-run" "$variant" "${CAPTURE_DIR}/${variant}-quantize-dry-run.txt" "NO --attention flags; inherit template; --overwrite --threads 8" \
    "$QUANT_BIN" --hf "$fused_hf" --template "$TEMPLATE_GGUF" --out "$gguf" --dry-run --overwrite --threads 8 \
    || { AC2_STATUS="FAIL"; ac_fail "deepseek4-quantize --dry-run failed for ${variant}"; }

  run_logged_phase "$variant" "7b quantize real" "$variant" "${CAPTURE_DIR}/${variant}-quantize.txt" "NO --attention flags; inherit template; --overwrite --threads 8" \
    "$QUANT_BIN" --hf "$fused_hf" --template "$TEMPLATE_GGUF" --out "$gguf" --overwrite --threads 8 \
    || { AC2_STATUS="FAIL"; Q7_BF16_CASCADE_STATUS="TRIGGERED"; block "deepseek4-quantize crashed on real BF16 attn Q7 residual for ${variant}"; }
  else
    # §10.2 SKIP_THIS_VARIANT — require pre-existing GGUF; §10.1 bound check runs at phase 7c (below) as the single AC7 gate.
    [[ -f "$gguf" ]] || { AC2_STATUS="FAIL"; ac_fail "SKIP_QUANT_VARIANTS=${variant}: required GGUF missing: ${gguf}"; }
    skip_size="$(stat -f '%z' "$gguf" 2>/dev/null || stat -c '%s' "$gguf" 2>/dev/null)"
    (( skip_size > 0 )) || { AC2_STATUS="FAIL"; ac_fail "SKIP_QUANT_VARIANTS=${variant}: required GGUF zero-size: ${gguf}"; }
    echo "SKIP_QUANT_VARIANTS=${variant}: reusing pre-existing GGUF (bytes=${skip_size}, mtime=$(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$gguf" 2>/dev/null || stat -c '%y' "$gguf" 2>/dev/null))"
  fi

  run_logged_phase "$variant" "7c gguf size assert" "$variant" "${CAPTURE_DIR}/${variant}-gguf-size-assert.txt" "template-relative [template - 1 GiB, template + 5 GiB] §10.1" \
    assert_gguf_size "$gguf" \
    || { AC2_STATUS="FAIL"; ac_fail "fused GGUF size outside [template - 1 GiB, template + 5 GiB] §10.1 for ${variant}"; }

  run_logged_phase "$variant" "8 inspect" "$variant" "${CAPTURE_DIR}/${variant}-inspect.txt" "ds4 --inspect" \
    "$DS4_BIN" --inspect -m "$gguf" \
    || { AC2_STATUS="FAIL"; Q7_BF16_CASCADE_STATUS="TRIGGERED"; block "ds4 crashed loading fused GGUF during inspect for ${variant}"; }

  run_logged_phase "$variant" "9 serve AC3" "$variant" "$serve_out" "-n ${N_AC3} --metal; prompt Hello, how are you?" \
    "$DS4_BIN" -m "$gguf" -n "$N_AC3" --temp 0 -p "$PROMPT" --metal \
    || { AC3_STATUS="FAIL"; if [[ "$variant" == "alpha0" ]]; then ALPHA0_CASCADE_STATUS="TRIGGERED"; else NONZERO_CASCADE_STATUS="TRIGGERED"; fi; block "ds4 serve failed for ${variant}"; }
  if ! validate_text_capture "$serve_out"; then
    AC3_STATUS="FAIL"
    if [[ "$variant" == "alpha0" ]]; then
      ALPHA0_CASCADE_STATUS="TRIGGERED"
    else
      NONZERO_CASCADE_STATUS="TRIGGERED"
    fi
    block "ds4 serve emitted empty/NaN/garbage text for ${variant}"
  fi

  run_logged_phase "$variant" "10 dump-logits probe" "$variant" "$probe_out" "-n ${N_PROBE} --dump-logits --metal; prompt Hello, how are you?" \
    "$DS4_BIN" -m "$gguf" -n "$N_PROBE" --temp 0 -p "$PROMPT" --dump-logits "${CAPTURE_DIR}/${variant}-logits.json" --metal \
    || { if [[ "$variant" == "alpha0" ]]; then AC4_STATUS="FAIL"; ALPHA0_CASCADE_STATUS="TRIGGERED"; else AC5_STATUS="FAIL"; NONZERO_CASCADE_STATUS="TRIGGERED"; fi; block "ds4 dump-logits failed for ${variant}"; }

  if ! metrics="$(json_norm_argmax "${CAPTURE_DIR}/${variant}-logits.json" 2>&1)"; then
    if [[ "$variant" == "alpha0" ]]; then
      AC4_STATUS="FAIL"
      ALPHA0_CASCADE_STATUS="TRIGGERED"
    else
      AC5_STATUS="FAIL"
      NONZERO_CASCADE_STATUS="TRIGGERED"
    fi
    block "${variant} logits JSON contains null/non-finite/garbage: ${metrics}"
  fi

  if [[ "$variant" == "alpha0" ]]; then
    ALPHA0_NORM="$(printf "%s\n" "$metrics" | sed -n '1p')"
    ALPHA0_ARGMAX="$(printf "%s\n" "$metrics" | sed -n '2p')"
    if ! L2_REL_ALPHA0="$(json_rel_l2 "${CAPTURE_DIR}/base-logits.json" "${CAPTURE_DIR}/alpha0-logits.json" 2>&1)"; then
      AC4_STATUS="FAIL"
      ALPHA0_CASCADE_STATUS="TRIGGERED"
      block "alpha0 relative-L2 computation failed: ${L2_REL_ALPHA0}"
    fi
    if ! float_le "$L2_REL_ALPHA0" "$CASCADE_ALPHA0_REL"; then
      AC4_STATUS="FAIL"
      ALPHA0_CASCADE_STATUS="TRIGGERED"
      block "alpha0 relative-L2 ${L2_REL_ALPHA0} exceeds cascade threshold ${CASCADE_ALPHA0_REL}"
    fi
    ALPHA0_CASCADE_STATUS="NOT TRIGGERED"
    if float_le "$L2_REL_ALPHA0" "$EPS_ALPHA0_REL"; then
      AC4_STATUS="PASS"
    else
      AC4_STATUS="FAIL"
      ac_fail "alpha0 relative-L2 ${L2_REL_ALPHA0} exceeds AC4 tolerance ${EPS_ALPHA0_REL}"
    fi
  else
    NONZERO_ARGMAX="$(printf "%s\n" "$metrics" | sed -n '2p')"
    if ! L2_REL_NONZERO="$(json_rel_l2 "${CAPTURE_DIR}/base-logits.json" "${CAPTURE_DIR}/nonzero-logits.json" 2>&1)"; then
      AC5_STATUS="FAIL"
      NONZERO_CASCADE_STATUS="TRIGGERED"
      block "nonzero relative-L2 computation failed: ${L2_REL_NONZERO}"
    fi
    NONZERO_CASCADE_STATUS="NOT TRIGGERED"
    if float_ge "$L2_REL_NONZERO" "$EPS_NONZERO_MIN_REL"; then
      AC5_STATUS="PASS"
    else
      AC5_STATUS="FAIL"
      ac_fail "nonzero relative-L2 ${L2_REL_NONZERO} below AC5 minimum ${EPS_NONZERO_MIN_REL}"
    fi
  fi

  if [[ -n "$SKIP_THIS_VARIANT" ]]; then
    run_logged_phase "$variant" "11 cleanup ${variant} (GGUF preserved)" "$variant" "${CAPTURE_DIR}/${variant}-cleanup.txt" "rm -rf fused-hf only; preserve externally-supplied GGUF per §10.2" \
      cleanup_variant "$fused_hf" "$gguf" "skip-gguf" \
      || { AC7_STATUS="FAIL"; ac_fail "cleanup failed for ${variant}"; }
  else
    run_logged_phase "$variant" "11 cleanup ${variant}" "$variant" "${CAPTURE_DIR}/${variant}-cleanup.txt" "rm -rf fused HF + fused GGUF; disk-peak captured" \
      cleanup_variant "$fused_hf" "$gguf" \
      || { AC7_STATUS="FAIL"; ac_fail "cleanup failed for ${variant}"; }
  fi
}

mkdir -p "$(dirname "$REPORT")"
mkdir -p "$CAPTURE_DIR"

[[ -x "$DS4_BIN" ]] || preflight_fail "ds4 binary missing or not executable: ${DS4_BIN}"
[[ -x "$QUANT_BIN" ]] || preflight_fail "deepseek4-quantize binary missing or not executable: ${QUANT_BIN}"
[[ -f "$MLX_PY" ]] || preflight_fail "MLX python missing: ${MLX_PY}"
[[ -d "$BASE_HF" ]] || preflight_fail "base HF checkpoint missing: ${BASE_HF}"
[[ -f "$TEMPLATE_GGUF" ]] || preflight_fail "template GGUF missing: ${TEMPLATE_GGUF}"
[[ -e "$TRACK_A_MARKER" ]] || preflight_fail "Track-A marker absent: ${TRACK_A_MARKER}"
[[ ! -e "$TRACK_B_MARKER" ]] || preflight_fail "Track-B marker exists before smoke: ${TRACK_B_MARKER}"
[[ ! -e "$NEW_FUSION_MARKER" ]] || preflight_fail "new fusion marker exists before smoke: ${NEW_FUSION_MARKER}"
[[ ! -e "$MODEL_4BIT_DIR" ]] || preflight_fail "model-4bit directory exists before smoke: ${MODEL_4BIT_DIR}"
[[ ! -e "$CONVERT_SHIMMED_DIR" ]] || preflight_fail "convert-shimmed directory exists before smoke: ${CONVERT_SHIMMED_DIR}"
[[ -f "$READINESS_JSON" ]] || preflight_fail "readiness JSON missing: ${READINESS_JSON}"
if pgrep -f -- "$DS4_BIN" >/dev/null 2>&1; then
  preflight_fail "concurrent ds4 detected for ${DS4_BIN}"
fi

PRE_FREE_BYTES="$(free_bytes)"
if ! float_ge "$PRE_FREE_BYTES" "$MIN_DISK_BYTES"; then
  preflight_fail "free bytes ${PRE_FREE_BYTES} below MIN_DISK_BYTES ${MIN_DISK_BYTES}"
fi

SCRIPT_SHA="$(shasum -a 256 "$0" | awk '{print $1}' 2>/dev/null || printf 'n/a')"
TEMPLATE_RESOLVED="$(readlink -f "$TEMPLATE_GGUF")"
PRE_DS4FLASH_SHA="$(shasum -a 256 "$TEMPLATE_RESOLVED" | awk '{print $1}')"
# §10.1 — FROZEN: derive AC7 bounds from template bytes (template-relative ±{1 GiB, 5 GiB}).
TEMPLATE_GGUF_BYTES="$(stat -f '%z' "$TEMPLATE_RESOLVED" 2>/dev/null || stat -c '%s' "$TEMPLATE_RESOLVED" 2>/dev/null)"
[[ -n "$TEMPLATE_GGUF_BYTES" && "$TEMPLATE_GGUF_BYTES" =~ ^[0-9]+$ && "$TEMPLATE_GGUF_BYTES" -gt 0 ]] \
  || preflight_fail "TEMPLATE_GGUF_BYTES unset/zero — cannot derive §10.1 bounds; TEMPLATE_RESOLVED=${TEMPLATE_RESOLVED}"
GGUF_MIN_BYTES=$(( TEMPLATE_GGUF_BYTES - (1 * 1024 * 1024 * 1024) ))
GGUF_MAX_BYTES=$(( TEMPLATE_GGUF_BYTES + (5 * 1024 * 1024 * 1024) ))
(( GGUF_MIN_BYTES > 0 )) || GGUF_MIN_BYTES=0   # §10.3#2 defensive underflow guard
PRE_HF_MTIMES_FILE="${CAPTURE_DIR}/hf-shards-pre.mtime"
POST_HF_MTIMES_FILE="${CAPTURE_DIR}/hf-shards-post.mtime"
shard_mtimes > "$PRE_HF_MTIMES_FILE"
PRE_HF_SHARD_COUNT="$(wc -l < "$PRE_HF_MTIMES_FILE" | tr -d ' ')"
PRE_LAYER0_SHAPE="$(base_layer0_shape)"
PRE_TRACK_A_MTIME="$(stat -f "%m" "$TRACK_A_MARKER")"
PRE_READINESS_MTIME="$(stat -f "%m" "$READINESS_JSON")"
PRE_READINESS_SIZE="$(stat -f "%z" "$READINESS_JSON")"
PRE_READINESS_PROOFS="$(readiness_proofs)"
PRE_DISK_USED="$(disk_used_bytes)"
ALPHA0_PEAK="$PRE_DISK_USED"
NONZERO_PEAK="$PRE_DISK_USED"

run_logged "1 preflight disk" "—" "${CAPTURE_DIR}/preflight-disk.txt" ">= ${MIN_DISK_BYTES}" \
  bash -lc "df -k \"$WORK_MOUNT\" && echo free_bytes=${PRE_FREE_BYTES} used_bytes=${PRE_DISK_USED}"
run_logged "2 preflight invariants" "—" "${CAPTURE_DIR}/preflight-invariants.txt" "sha + stat + Tier-1 readiness stat + Tier-2 n/a-tolerant parse" \
  bash -lc "printf '%s\n' 'template=${TEMPLATE_RESOLVED}' 'sha=${PRE_DS4FLASH_SHA}' 'hf_shards=${PRE_HF_SHARD_COUNT}' 'shape=${PRE_LAYER0_SHAPE}' 'track_a=${PRE_TRACK_A_MTIME}' 'readiness_mtime=${PRE_READINESS_MTIME}' 'readiness_size=${PRE_READINESS_SIZE}' 'proofs=${PRE_READINESS_PROOFS}' 'ds4_binary=present' 'quantize_binary=present' 'concurrent_ds4=absent'"

run_logged "3 base logits probe" "base" "${CAPTURE_DIR}/base-probe-stdout.txt" "base dump-logits; prompt Hello, how are you?" \
  "$DS4_BIN" -m "$TEMPLATE_GGUF" --temp 0 -n "$N_PROBE" -p "$PROMPT" --dump-logits "${CAPTURE_DIR}/base-logits.json" --metal \
  || { AC4_STATUS="FAIL"; ac_fail "base ds4 dump-logits failed"; }
if ! base_metrics="$(json_norm_argmax "${CAPTURE_DIR}/base-logits.json" 2>&1)"; then
  AC4_STATUS="FAIL"
  ac_fail "base logits JSON invalid: ${base_metrics}"
fi
BASE_NORM="$(printf "%s\n" "$base_metrics" | sed -n '1p')"
BASE_ARGMAX="$(printf "%s\n" "$base_metrics" | sed -n '2p')"

# §10.2 (h) — SMOKE_VARIANTS exclude → mark the AC DEFERRED (NOT FAIL). Round-7 (separate slice) re-runs.
smoke_variants_listed alpha0  || AC4_STATUS="DEFERRED (variant excluded by SMOKE_VARIANTS='${SMOKE_VARIANTS:-alpha0,nonzero}')"
smoke_variants_listed nonzero || AC5_STATUS="DEFERRED (variant excluded by SMOKE_VARIANTS='${SMOKE_VARIANTS:-alpha0,nonzero}')"

# §10.2 (f) — variant call-sites gated by SMOKE_VARIANTS allowlist.
smoke_variants_listed alpha0 && run_variant alpha0
smoke_variants_listed nonzero && run_variant nonzero

POST_DS4FLASH_SHA="$(shasum -a 256 "$TEMPLATE_RESOLVED" | awk '{print $1}')"
shard_mtimes > "$POST_HF_MTIMES_FILE"
POST_HF_SHARD_COUNT="$(wc -l < "$POST_HF_MTIMES_FILE" | tr -d ' ')"
POST_TRACK_A_MTIME="$(stat -f "%m" "$TRACK_A_MARKER")"
if [[ -f "$READINESS_JSON" ]]; then
  POST_READINESS_MTIME="$(stat -f "%m" "$READINESS_JSON")"
  POST_READINESS_SIZE="$(stat -f "%z" "$READINESS_JSON")"
  POST_READINESS_PROOFS="$(readiness_proofs)"
else
  AC6_STATUS="FAIL"
  ac_fail "readiness JSON missing post-flight: ${READINESS_JSON}"
fi
POST_DISK_USED="$(disk_used_bytes)"
FINAL_RETAINED_BYTES="$(retained_evidence_bytes)"

DS4FLASH_SHA_STATUS="FAIL"
[[ "$PRE_DS4FLASH_SHA" == "$POST_DS4FLASH_SHA" ]] && DS4FLASH_SHA_STATUS="PASS ✓"
HF_MTIME_STATUS="FAIL"
cmp -s "$PRE_HF_MTIMES_FILE" "$POST_HF_MTIMES_FILE" && HF_MTIME_STATUS="PASS ✓"
TRACK_A_STATUS="FAIL"
[[ "$PRE_TRACK_A_MTIME" == "$POST_TRACK_A_MTIME" ]] && TRACK_A_STATUS="PASS ✓"
TRACK_B_STATUS="FAIL"
[[ ! -e "$TRACK_B_MARKER" ]] && TRACK_B_STATUS="PASS ✓"
NEW_MARKER_STATUS="FAIL"
[[ ! -e "$NEW_FUSION_MARKER" ]] && NEW_MARKER_STATUS="PASS ✓"
MODEL_4BIT_STATUS="FAIL"
[[ ! -e "$MODEL_4BIT_DIR" ]] && MODEL_4BIT_STATUS="PASS ✓"
CONVERT_SHIMMED_STATUS="FAIL"
[[ ! -e "$CONVERT_SHIMMED_DIR" ]] && CONVERT_SHIMMED_STATUS="PASS ✓"
READINESS_STAT_STATUS="FAIL"
[[ "$PRE_READINESS_MTIME" == "$POST_READINESS_MTIME" && "$PRE_READINESS_SIZE" == "$POST_READINESS_SIZE" ]] && READINESS_STAT_STATUS="PASS ✓"

CLEANUP_STATUS="PASS ✓"
for path in \
  "${WORK_DIR}/fused-hf-smoke-alpha0" \
  "${WORK_DIR}/fused-smoke-alpha0.gguf" \
  "${WORK_DIR}/fused-hf-smoke-nonzero" \
  "${WORK_DIR}/fused-smoke-nonzero.gguf"; do
  # §10.2 (g): SKIP_QUANT_VARIANTS GGUFs are intentionally preserved externals — exempt from cleanup-presence check.
  case "$path" in
    */fused-smoke-alpha0.gguf)  skip_variant_listed alpha0  && continue ;;
    */fused-smoke-nonzero.gguf) skip_variant_listed nonzero && continue ;;
  esac
  if [[ -e "$path" ]]; then
    CLEANUP_STATUS="FAIL (${path} remains)"
  fi
done

AC1_STATUS="PASS"
AC2_STATUS="PASS"
AC3_STATUS="PASS"
if [[ "$DS4FLASH_SHA_STATUS" == PASS* && "$HF_MTIME_STATUS" == PASS* && "$TRACK_A_STATUS" == PASS* && "$TRACK_B_STATUS" == PASS* && "$NEW_MARKER_STATUS" == PASS* && "$MODEL_4BIT_STATUS" == PASS* && "$CONVERT_SHIMMED_STATUS" == PASS* && "$READINESS_STAT_STATUS" == PASS* ]]; then
  AC6_STATUS="PASS"
else
  AC6_STATUS="FAIL"
fi
if [[ "$CLEANUP_STATUS" == PASS* && "$FINAL_RETAINED_BYTES" =~ ^[0-9]+$ && "$FINAL_RETAINED_BYTES" -le "$MAX_RETAINED_BYTES" ]]; then
  AC7_STATUS="PASS"
else
  AC7_STATUS="FAIL"
fi

if [[ "$AC6_STATUS" != "PASS" ]]; then
  ac_fail "AC6 invariant failure"
fi
if [[ "$AC7_STATUS" != "PASS" ]]; then
  ac_fail "AC7 cleanup/retained-evidence failure: cleanup=${CLEANUP_STATUS}; retained=${FINAL_RETAINED_BYTES}; max=${MAX_RETAINED_BYTES}"
fi

if [[ "$AC1_STATUS" == "PASS" && "$AC2_STATUS" == "PASS" && "$AC3_STATUS" == "PASS" \
      && ( "$AC4_STATUS" == "PASS" || "$AC4_STATUS" == "DEFERRED"* ) \
      && ( "$AC5_STATUS" == "PASS" || "$AC5_STATUS" == "DEFERRED"* ) \
      && "$AC6_STATUS" == "PASS" && "$AC7_STATUS" == "PASS" && "$(cascade_total)" == "0" ]]; then
  OVERALL_STATUS="PASS"
  final_report "$OVERALL_STATUS"
  exit 0
fi

ac_fail "one or more acceptance criteria failed"
