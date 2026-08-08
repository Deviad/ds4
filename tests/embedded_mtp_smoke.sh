#!/bin/sh
set -eu

model=${DS4_EMBEDDED_MTP_MODEL:-/Volumes/Data NVME/mlx-ft/ds4/joint-main-mtp-Q4K-final.gguf}
legacy_model=${DS4_LEGACY_GUARD_MAIN_MODEL:-ds4flash.gguf}
legacy_sidecar=$(mktemp "${TMPDIR:-/tmp}/ds4-legacy-mtp.XXXXXX")
trap 'rm -f "$legacy_sidecar"' EXIT

if [ ! -f "$model" ]; then
    echo "embedded MTP model not found: $model" >&2
    exit 2
fi
if [ ! -f "$legacy_model" ]; then
    echo "legacy guard base model not found: $legacy_model" >&2
    exit 2
fi

python3 - "$legacy_sidecar" <<'PY'
import struct
import sys

path = sys.argv[1]
name = b"mtp.0.hc_head_base.weight"
header = struct.pack("<IIQQ", 0x46554747, 3, 1, 0)
tensor = (struct.pack("<Q", len(name)) + name +
          struct.pack("<IQIQ", 1, 1, 0, 0))
data_pos = (len(header) + len(tensor) + 31) & ~31
with open(path, "wb") as f:
    f.write(header)
    f.write(tensor)
    f.write(b"\\0" * (data_pos - f.tell()))
    f.write(struct.pack("<f", 0.0))
PY

inspect=$(env -u DS4_MTP_PROBE -u DS4_MTP_FULL_LOGITS -u DS4_MTP_MIN_MARGIN \
    ./ds4 --inspect --metal --model "$model" 2>&1)
printf '%s\n' "$inspect" | grep -F \
    'embedded_mtp stages=3 bound_stages=3 source=main_model block_size=5 draft=2' >/dev/null

# -n 2: the speculative driver skips draft preparation when max_tokens == 1,
# so a second token is needed to make the embedded stages run.
generation=$(env -u DS4_MTP_PROBE -u DS4_MTP_FULL_LOGITS -u DS4_MTP_MIN_MARGIN \
    ./ds4 --metal --model "$model" -p x -n 2 --ctx 64 2>&1)
printf '%s\n' "$generation" | grep -F 'embedded_mtp stages_executed=0:1,1:1,2:1' >/dev/null
if printf '%s\n' "$generation" | grep -Eiq '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)'; then
    echo 'embedded MTP generation produced a non-finite diagnostic' >&2
    exit 1
fi

set +e
legacy_output=$(./ds4 --metal --model "$legacy_model" --mtp "$legacy_sidecar" \
    -p x -n 1 --ctx 64 2>&1)
legacy_rc=$?
set -e
if [ "$legacy_rc" -eq 0 ]; then
    echo 'legacy guard regression unexpectedly succeeded with an incomplete sidecar' >&2
    exit 1
fi
# An incomplete legacy sidecar must be rejected at support-model detection,
# before any binding is attempted.
printf '%s\n' "$legacy_output" | grep -F \
    'unsupported --mtp support model' >/dev/null
if printf '%s\n' "$legacy_output" | grep -Fq 'embedded three-stage GGUF'; then
    echo 'legacy guard regression was rejected as embedded MTP' >&2
    exit 1
fi

printf '%s\n' 'embedded MTP smoke: default draft=2; stages 0,1,2 executed; finite output; legacy guard passed'
