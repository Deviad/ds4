#!/usr/bin/env bash
# RED→GREEN test for Story 12.3 Round-6 §10 FROZEN spec on scripts/smoke_fuse_serve.sh.
# Scope: template-relative AC7 (§10.1) + SKIP_QUANT_VARIANTS/SMOKE_VARIANTS env contract (§10.2)
#        + forward-cascade guards (§10.3). NO heavy smoke executed here — static + helper-unit only.
# Frozen source-of-truth: agent-output/cmux-12-3/architecture.md §10.1–§10.5.
set -uo pipefail

SCRIPT="${SCRIPT:-/Users/spotted/projects/ds4-finetuning/scripts/smoke_fuse_serve.sh}"
PASS=0
FAIL=0
note() { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS + 1)); printf 'PASS: %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf 'FAIL: %s\n' "$1"; }

# Extract a single shell function body by name from the smoke script (def line through matching close brace).
extract_fn() {
  local fn="$1"
  sed -n "/^${fn}() {/,/^}/p" "$SCRIPT"
}

# ---------------------------------------------------------------------------
# Test 1 — syntax: `bash -n` exits 0 (mandatory, NO execution).
# ---------------------------------------------------------------------------
if bash -n "$SCRIPT" 2>/tmp/smoke_ac7_syntax.err; then
  ok "T1 bash -n syntax check exit 0"
else
  bad "T1 bash -n syntax check (see /tmp/smoke_ac7_syntax.err)"
fi

# ---------------------------------------------------------------------------
# Test 2 — §10.1 template-relative AC7: hardcoded 92/102 GiB literals DROPPED,
#          bounds derived from TEMPLATE_GGUF_BYTES with hard preflight guard +
#          underflow defense + template-relative echo + §10.1 ac_fail wording.
# ---------------------------------------------------------------------------
t2_ok=1
if grep -Eq '92 \* 1024|102 \* 1024' "$SCRIPT"; then
  bad "T2a hardcoded 92/102 GiB literals still present (§10.1 dropped them)"; t2_ok=0
fi
grep -q 'TEMPLATE_GGUF_BYTES=' "$SCRIPT"                              || { bad "T2b TEMPLATE_GGUF_BYTES derivation missing"; t2_ok=0; }
grep -q 'TEMPLATE_GGUF_BYTES - (1 \* 1024 \* 1024 \* 1024)' "$SCRIPT" || { bad "T2c GGUF_MIN_BYTES = template - 1 GiB missing"; t2_ok=0; }
grep -q 'TEMPLATE_GGUF_BYTES + (5 \* 1024 \* 1024 \* 1024)' "$SCRIPT" || { bad "T2d GGUF_MAX_BYTES = template + 5 GiB missing"; t2_ok=0; }
grep -q 'TEMPLATE_GGUF_BYTES unset/zero' "$SCRIPT"                    || { bad "T2e §10.3#1 preflight guard message missing"; t2_ok=0; }
grep -q 'GGUF_MIN_BYTES > 0' "$SCRIPT"                                || { bad "T2f §10.3#2 underflow defense missing"; t2_ok=0; }
grep -q 'template-relative, template=' "$SCRIPT"                      || { bad "T2g template-relative echo message missing"; t2_ok=0; }
grep -q 'outside \[template - 1 GiB, template + 5 GiB\] §10.1' "$SCRIPT" || { bad "T2h §10.1 ac_fail wording missing"; t2_ok=0; }
# Early-binding visibility: top-of-script initializers must be 0 (delayed derivation).
grep -Eq '^GGUF_MIN_BYTES=0$' "$SCRIPT"                               || { bad "T2i GGUF_MIN_BYTES early-binding sentinel 0 missing"; t2_ok=0; }
grep -Eq '^GGUF_MAX_BYTES=0$' "$SCRIPT"                               || { bad "T2j GGUF_MAX_BYTES early-binding sentinel 0 missing"; t2_ok=0; }
[[ "$t2_ok" == 1 ]] && ok "T2 §10.1 template-relative AC7 bounds + guards + wording"

# ---------------------------------------------------------------------------
# Test 3 — §10.2(a) skip_variant_listed CSV membership incl §10.3#4 whitespace strip.
# ---------------------------------------------------------------------------
t3_ok=1
fn3="$(extract_fn skip_variant_listed)"
if [[ -z "$fn3" ]]; then
  bad "T3 skip_variant_listed not defined"; t3_ok=0
else
  ( eval "$fn3"
    SKIP_QUANT_VARIANTS="alpha0,nonzero" skip_variant_listed alpha0 ) && r1=0 || r1=1
  ( eval "$fn3"
    SKIP_QUANT_VARIANTS="nonzero"        skip_variant_listed alpha0 ) && r2=0 || r2=1
  ( eval "$fn3"
    SKIP_QUANT_VARIANTS=""               skip_variant_listed alpha0 ) && r3=0 || r3=1
  # §10.3#4: whitespace after comma must still match (item="${item//[[:space:]]/}").
  ( eval "$fn3"
    SKIP_QUANT_VARIANTS="alpha0, nonzero" skip_variant_listed nonzero ) && r4=0 || r4=1
  [[ "$r1" == 0 ]] || { bad "T3a present-variant should match"; t3_ok=0; }
  [[ "$r2" == 1 ]] || { bad "T3b absent-variant must NOT match"; t3_ok=0; }
  [[ "$r3" == 1 ]] || { bad "T3c unset list must NOT match"; t3_ok=0; }
  [[ "$r4" == 0 ]] || { bad "T3d §10.3#4 whitespace-after-comma must match"; t3_ok=0; }
  [[ "$t3_ok" == 1 ]] && ok "T3 §10.2(a) skip_variant_listed CSV + whitespace strip"
fi

# ---------------------------------------------------------------------------
# Test 4 — §10.2(a) smoke_variants_listed CSV membership + default allowlist.
# ---------------------------------------------------------------------------
t4_ok=1
fn4="$(extract_fn smoke_variants_listed)"
if [[ -z "$fn4" ]]; then
  bad "T4 smoke_variants_listed not defined"; t4_ok=0
else
  ( eval "$fn4"; unset SMOKE_VARIANTS; smoke_variants_listed alpha0 )  && r1=0 || r1=1
  ( eval "$fn4"; unset SMOKE_VARIANTS; smoke_variants_listed nonzero ) && r2=0 || r2=1
  ( eval "$fn4"; SMOKE_VARIANTS="alpha0" smoke_variants_listed nonzero ) && r3=0 || r3=1
  ( eval "$fn4"; SMOKE_VARIANTS="alpha0" smoke_variants_listed alpha0 )  && r4=0 || r4=1
  [[ "$r1" == 0 ]] || { bad "T4a default allowlist must include alpha0"; t4_ok=0; }
  [[ "$r2" == 0 ]] || { bad "T4b default allowlist must include nonzero"; t4_ok=0; }
  [[ "$r3" == 1 ]] || { bad "T4c SMOKE_VARIANTS=alpha0 must EXCLUDE nonzero"; t4_ok=0; }
  [[ "$r4" == 0 ]] || { bad "T4d SMOKE_VARIANTS=alpha0 must include alpha0"; t4_ok=0; }
  [[ "$t4_ok" == 1 ]] && ok "T4 §10.2(a) smoke_variants_listed CSV + default allowlist"
fi

# ---------------------------------------------------------------------------
# Test 5 — §10.2(d) cleanup_variant 3-arg skip-gguf preserves GGUF;
#          §10.2(h) final overall gate accepts AC4/AC5 DEFERRED.
# ---------------------------------------------------------------------------
t5_ok=1
fn5="$(extract_fn cleanup_variant)"
if [[ -z "$fn5" ]]; then
  bad "T5 cleanup_variant not defined"; t5_ok=0
else
  tmpd="$(mktemp -d)"
  hf="${tmpd}/fused-hf"; gg="${tmpd}/fused.gguf"
  # skip-gguf mode: GGUF preserved, fused-hf removed.
  mkdir -p "$hf"; : > "$gg"
  ( eval "$fn5"; cleanup_variant "$hf" "$gg" "skip-gguf" ) >/dev/null 2>&1
  { [[ ! -e "$hf" ]] && [[ -e "$gg" ]]; } || { bad "T5a skip-gguf must preserve GGUF, remove fused-hf"; t5_ok=0; }
  # default mode: both removed (backward-compatible).
  mkdir -p "$hf"; : > "$gg"
  ( eval "$fn5"; cleanup_variant "$hf" "$gg" ) >/dev/null 2>&1
  { [[ ! -e "$hf" ]] && [[ ! -e "$gg" ]]; } || { bad "T5b default mode must remove both fused-hf + GGUF"; t5_ok=0; }
  rm -rf "$tmpd"
fi
# §10.2(h): final gate must accept DEFERRED for AC4/AC5 (pattern-match prefix).
grep -q '"$AC4_STATUS" == "DEFERRED"\*' "$SCRIPT" || { bad "T5c final gate must accept AC4 DEFERRED"; t5_ok=0; }
grep -q '"$AC5_STATUS" == "DEFERRED"\*' "$SCRIPT" || { bad "T5d final gate must accept AC5 DEFERRED"; t5_ok=0; }
grep -q 'DEFERRED (variant excluded by SMOKE_VARIANTS=' "$SCRIPT" || { bad "T5e DEFERRED init message missing"; t5_ok=0; }
# §10.2(f): variant call-sites guarded by smoke_variants_listed.
grep -q 'smoke_variants_listed alpha0 && run_variant alpha0' "$SCRIPT" || { bad "T5f alpha0 call-site gate missing"; t5_ok=0; }
grep -q 'smoke_variants_listed nonzero && run_variant nonzero' "$SCRIPT" || { bad "T5g nonzero call-site gate missing"; t5_ok=0; }
[[ "$t5_ok" == 1 ]] && ok "T5 §10.2(d)(f)(h) cleanup skip-gguf + DEFERRED gate + variant call-site guards"

# ---------------------------------------------------------------------------
note ""
note "RESULT: ${PASS} passed, ${FAIL} failed"
[[ "$FAIL" == 0 ]]
