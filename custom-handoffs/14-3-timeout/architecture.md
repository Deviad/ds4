# Slice 14.3b — Architecture: Smoke Timeout Repin 600s → 1200s

> Repin the bounded real-smoke hard timeout constant from 600 seconds to
> 1200 seconds. No production code edits in this document; this is the
> Architect handoff. Coder implements Phase 1 (constant + tests + docs).
> Phase 2 (real smoke with 1200s) is separately authorized, future.

## Triggering event

Story 14.3a (filtered-dataset repin) completed successfully: the smoke
reached trainer startup (trainable parameters reported, 0/1 iterations
visible on the progress bar) but timed out at 600 seconds with zero
completed iterations. Evidence:

- `agent-output/cmux-14-3/smoke-report.json`:
  `{"status":"fail","failure_code":"timeout","failure_message":"smoke timed out after 600s","wall_clock_seconds":603.217}`
- `agent-output/cmux-14-3/smoke-log.txt`: shows 0% progress, 0/1 iters, no
  loss value emitted.
- Adapter output directory was empty; instance lock was released cleanly.

The operator authorized a minimal timeout-only repin from 600s to 1200s
(20 minutes). This gives the single bounded smoke attempt enough
wall-clock time to complete one forward/backward pass against the real
4096-token model and filtered dataset.

## Scope

This slice changes exactly one integer literal in one production source
file. The timeout message, backup watchdog deadline, and `signal.alarm`
duration are all runtime-derived from this constant — no separate string
edits are needed. All other constants, parameters, paths, catalog
entries, safety gates, abort conditions, and protected files are
preserved exactly.

## Change (exhaustive — anything not listed is forbidden)

### C1. Hard timeout constant

**File:** `scripts/ds4_segmented_smoke.py`, line 55
**Change:** `SMOKE_TIMEOUT_SECONDS = 600` → `SMOKE_TIMEOUT_SECONDS = 1200`

This is the only production source edit. No other line in this file
changes.

### Runtime-derived sites (automatically follow — no separate edits)

The constant `SMOKE_TIMEOUT_SECONDS` is read at four sites, all of which
update automatically when the constant changes:

1. **`_timeout_handler`** (line 819):
   `_TIMEOUT_REASON = f"smoke timed out after {SMOKE_TIMEOUT_SECONDS}s"`
   → produces `"smoke timed out after 1200s"` at runtime.

2. **`_backup_watchdog`** (line 827):
   `deadline = time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5`
   → watchdog deadline becomes 1205s at runtime.

3. **`_install_timeout_watchdog`** (line 839):
   `signal.alarm(SMOKE_TIMEOUT_SECONDS)`
   → `signal.alarm(1200)` at runtime.

4. **Report writer** (line 1125-1126):
   `reason = _TIMEOUT_REASON or ...` and
   `fcode = "timeout" if _TIMEOUT_FLAG else ...`
   → reads the global set by the handler; produces `"smoke timed out
   after 1200s"` in the report's `failure_message` field.

**Architect confirms:** no separate string edit, no separate timeout
value, and no entanglement with any other logic. The change is fully
isolated to the one constant.

### C2. Synthetic tests

**File:** `tests/test_ds4_segmented_smoke.py`
**Change:** Add a new test class `TestTimeoutRepin` after `TestTimeoutPreflight`.

#### `TestTimeoutRepin` (new class)

**`test_timeout_constant_is_1200`**: Import
`scripts.ds4_segmented_smoke as mod` WITHOUT monkeypatching, assert
`mod.SMOKE_TIMEOUT_SECONDS == 1200`. Pure constant assertion — no
filesystem access, no monkeypatch. Fails if the constant is 600 or any
other value. Mutation-sensitive: changing the constant to 600 or 900
makes the test RED.

**`test_timeout_message_contains_1200s`**: Monkeypatch `signal.signal`
and `signal.alarm` to no-ops (same pattern as existing
`TestTimeoutPreflight` tests). Reset `mod._TIMEOUT_FLAG = False` and
`mod._TIMEOUT_REASON = None` (module globals). Call
`mod._timeout_handler(signal.SIGALRM, None)` directly and catch
`SystemExit`. Assert that `mod._TIMEOUT_REASON` equals
`"smoke timed out after 1200s"` (exact string). This verifies the
runtime-derived message reflects the new constant, not a stale 600s
string. No `signal.alarm` fires (no real timeout), no filesystem access.

**`test_lock_timeout_unchanged_at_60`**: Import
`scripts.ds4_segmented_smoke as mod` WITHOUT monkeypatching, assert
`mod.SMOKE_LOCK_TIMEOUT_S == 60`. Regression guard — the lock timeout
must not change from the Story 14.3 pin.

**`test_abort_timeout_unchanged_at_2`**: Import
`scripts.ds4_segmented_smoke as mod` WITHOUT monkeypatching, assert
`mod.SMOKE_ABORT_TIMEOUT == 2`. Regression guard — the abort exit code
must not change.

**`test_backup_watchdog_deadline_uses_1200`**: Import
`scripts.ds4_segmented_smoke as mod`, monkeypatch `time.monotonic` to
return a known value (e.g. 1000.0), monkeypatch `time.sleep` to no-op,
monkeypatch `os.kill` to no-op, and monkeypatch `mod._TIMEOUT_FLAG` to
`True` immediately so the watchdog returns early. Call `_backup_watchdog()`
and verify it does not raise. This confirms the function imports and
runs with `SMOKE_TIMEOUT_SECONDS` at 1200 without error. (Alternative:
just assert `mod.SMOKE_TIMEOUT_SECONDS + 5 == 1205` as a simpler
arithmetic check — both approaches are valid; the Coder may choose
either.)

#### Existing tests that remain GREEN (no changes needed)

- `TestTimeoutPreflight` (3 tests): Tests watchdog installation ordering,
  config failure markers, and thread creation. These monkeypatch
  `_install_timeout_watchdog` to a no-op and do not assert on the timeout
  value — they remain GREEN.
- `TestPinnedSmokeArgs` (4 tests): Tests `--iters`, `--batch-size`,
  `--segment-size`, `--grad-checkpoint` rejection — no timeout
  assertions, remain GREEN.
- `TestPinnedSmokePaths` (4 tests): Tests pinned paths — no timeout
  assertions, remain GREEN.
- `TestFilteredDatasetRepin` (2 tests from 14.3a): Tests dataset path
  constant — no timeout assertions, remain GREEN.
- `TestCatalog` (3+2 tests): Tests catalog entry structure — no timeout
  assertions, remain GREEN.
- `TestProtected` (3 tests): Protected file hashes — no timeout
  assertions, remain GREEN.
- All other test classes: No assertion on `SMOKE_TIMEOUT_SECONDS` or
  600s. The only match for "600" in the test file is a hex substring in
  a full-train catalog hash (`"26416001..."`) — not a timeout assertion.

### C3. Documentation updates

#### C3a. `docs/architecture.md`

**Change:** Amend the existing segmented-smoke paragraph (line ~74) with
one sentence recording the timeout repin. The hard timeout is a durable
boundary of the bounded real-smoke contract — the architecture doc should
reflect the current pin.

Sentence to append to the existing paragraph:

> The hard timeout was repinned from 600s to 1200s after the 14.3a
> filtered-dataset smoke reached trainer startup but timed out with zero
> completed iterations. The timeout constant `SMOKE_TIMEOUT_SECONDS` is
> the single source of truth: the signal alarm, backup watchdog thread,
  and report message all read it at runtime.

#### C3b. `docs/backlog.md`

**Status:** Already updated by BA (Story 14.3b exists at line 4766 with
the correctly formed user story and acceptance criteria). Architect
verifies presence. Status should be updated to reflect architecture is
complete (Coder ready).

### C4. `scripts/finetune_ds4.py` — NO change

The BA requirements ask the Architect to determine whether
`scripts/finetune_ds4.py` contains a timeout constant referencing the
smoke timeout that must be kept in sync.

**Adjudication: NO.** `scripts/finetune_ds4.py` defines
`DS4_GGUF_BASE_SMOKE_TIMEOUT = 900` (line 129) — this is a completely
separate timeout for the Track-A `ds4-smoke` base generation gate
(`subprocess.run(..., timeout=timeout_s)` at line 1663). It is not
connected to `SMOKE_TIMEOUT_SECONDS` in `ds4_segmented_smoke.py` in any
way. The `ds4-segmented-smoke` catalog entry does not embed a timeout
value — the timeout is enforced internally by the smoke script. No
`finetune_ds4.py` change is needed.

### C5. `docs/technical-spec.md` — NO change

Grep for `600`, `1200`, `timeout`, `SMOKE_TIMEOUT` in
`docs/technical-spec.md` returned no matches. The technical spec does
not reference the segmented smoke timeout. No change needed.

### C6. `custom-handoffs/14-3/requirements.md` — NO change

The original Story 14.3 requirements R14.3-4 pins `SMOKE_TIMEOUT_SECONDS
= 600` in the bounded smoke contract. This repin amends that pin from
600 to 1200. However, the original `custom-handoffs/14-3/requirements.md`
is a predecessor artifact — it documents the state at the time of the
original story. The amendment is recorded in:

1. `custom-handoffs/14-3-timeout/requirements.md` (BA, this slice)
2. `docs/backlog.md` Story 14.3b
3. `docs/architecture.md` (C3a)

If the BA requirements explicitly ask to amend the original R14.3-3 or
R14.3-4 tables in `custom-handoffs/14-3/requirements.md`, the Coder should
do so. If not, the predecessor artifact stays as-is and the amendment is
carried by the new slice's requirements and canonical docs.

**Architect determination:** The BA requirements for this slice do not
list `custom-handoffs/14-3/requirements.md` as an authorized docs change.
The timeout repin is recorded in `custom-handoffs/14-3-timeout/requirements.md`
and `docs/backlog.md`. No edit to the predecessor requirements file.

## What does NOT change (binding)

| # | Invariant | Verification |
|---|-----------|-------------|
| N1 | `SMOKE_LOCK_TIMEOUT_S = 60` unchanged | `test_lock_timeout_unchanged_at_60` |
| N2 | `SMOKE_ABORT_TIMEOUT = 2` unchanged | `test_abort_timeout_unchanged_at_2` |
| N3 | `SMOKE_ABORT_PREFLIGHT = 3` unchanged | Not directly tested; covered by existing preflight tests |
| N4 | `SMOKE_ABORT_TRAIN = 4` unchanged | Not directly tested; covered by existing training tests |
| N5 | `SMOKE_MEMORY_HEADROOM = 32 * 1024**3` unchanged | Not affected — no edit near this constant |
| N6 | `SMOKE_DISK_MIN_FREE = 1 * 1024**3` unchanged | Not affected |
| N7 | `SMOKE_LOCK_POLL_S = 2.0` unchanged | Not affected |
| N8 | All pinned smoke paths (model, data, adapter, config) unchanged | `TestPinnedSmokePaths` + `TestFilteredDatasetRepin` — no changes |
| N9 | All pinned smoke args (iters, batch-size, lr, max-seq-length, mask-prompt, grad-checkpoint, segment-size) unchanged | `TestPinnedSmokeArgs` — no changes |
| N10 | Default `smoke-train`, `full-train`, `continue-train` catalog entries unchanged | `TestCatalog` + `test_finetune_ds4.py` — no changes |
| N11 | `scripts/finetune_ds4.py` unchanged | No timeout constant references the segmented smoke |
| N12 | `segmented_loss_and_grad.py` (provider source, SHA-256 `205721...`) unchanged | `TestProtected.test_provider_source` |
| N13 | `test_ds4_segmented_loss_and_grad.py` (SHA-256 `618a0f...`) unchanged | `TestProtected.test_provider_test` |
| N14 | `test_mlx_lm_source.py` (SHA-256 `dec2c2...`) unchanged | `TestProtected.test_source_sentinel` — sentinel covers `python-envs/mlx/src/` only |
| N15 | No `vendor/mlx-lm/` inner file changes | Vendor HEAD `15b522f...` stays pinned |
| N16 | All abort conditions preserved | No abort logic touched |
| N17 | No retry/fallback/provider-call-count policy change | No retry logic touched |
| N18 | Watchdog installation mechanism (signal.SIGALRM + backup thread) unchanged | `TestTimeoutPreflight` — no changes |
| N19 | Lock release, fail marker, smoke report schema unchanged | No cleanup/marker logic touched |
| N20 | No `git commit` or `git push` | R-F-10 binding |

## Source-sentinel adjudication

Same as 14.3a: the source sentinel manifest in `test_mlx_lm_source.py`
covers only files under `python-envs/mlx/src/`. `scripts/ds4_segmented_smoke.py`
is not under that tree. Editing it (C1) does not change the sentinel
manifest. `test_mlx_lm_source.py` stays unchanged — blob SHA-256
`dec2c2b5...` remains intact.

**Adjudication: no source-sentinel repin needed.**

## File scope

| Path | Action | Justification |
|------|--------|---------------|
| `scripts/ds4_segmented_smoke.py` | EDIT (1 literal: line 55 `600` → `1200`) | Repin hard timeout constant |
| `tests/test_ds4_segmented_smoke.py` | EDIT (add `TestTimeoutRepin` class with 4-5 methods) | TDD: new constant, message, regression guards |
| `docs/architecture.md` | EDIT (1 sentence appended to segmented-smoke paragraph) | Record timeout repin as durable boundary |
| `docs/backlog.md` | VERIFY presence (BA already updated) | Status update only |

### Forbidden file changes

- `scripts/finetune_ds4.py` — no segmented-smoke timeout reference (C4 adjudication)
- `vendor/mlx-lm/` (any inner file; inner HEAD `15b522f...` frozen)
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` (provider source pin `205721...`)
- `tests/test_ds4_segmented_loss_and_grad.py` (blob `618a0f...`)
- `tests/test_mlx_lm_source.py` (blob `dec2c2...`)
- `tests/test_finetune_ds4.py` — no timeout assertions; no changes needed
- `docs/technical-spec.md` — no segmented-smoke timeout reference
- Any file under `agent-output/cmux-13-3b/`
- Any C/Objective-C/Metal/CUDA/ROCm/distributed/SSD/disk-cache file
- Any environment definition, package manifest, or config template
- `custom-handoffs/14-3/requirements.md` — predecessor artifact; amendment carried by new slice

## TDD sequence

### Red stage (test file only, no production edits)

1. Add `TestTimeoutRepin.test_timeout_constant_is_1200` — RED because
   constant is still 600.
2. Add `TestTimeoutRepin.test_timeout_message_contains_1200s` — RED
   because the handler produces `"smoke timed out after 600s"`.
3. Add `TestTimeoutRepin.test_lock_timeout_unchanged_at_60` — may
   initially PASS (constant is already 60). This is a regression guard.
4. Add `TestTimeoutRepin.test_abort_timeout_unchanged_at_2` — may
   initially PASS (constant is already 2). Regression guard.
5. (Optional) Add `TestTimeoutRepin.test_backup_watchdog_deadline_uses_1200`
   — RED because arithmetic `600 + 5 = 605 ≠ 1205`.

### Green stage (production edits)

1. C1: Change `SMOKE_TIMEOUT_SECONDS = 600` to `1200` in
   `scripts/ds4_segmented_smoke.py` line 55.
2. C3a: Amend `docs/architecture.md` with the timeout repin sentence.
3. Verify `docs/backlog.md` Story 14.3b presence and update status.

### Full test suite

Run the focused suite:
```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
  python-envs/mlx/.venv/bin/python -m pytest -q \
  tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py \
  tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py
```

All existing tests (236 from 14.3a) must remain GREEN. The new
`TestTimeoutRepin` tests must go GREEN after C1.

## Protected hashes (regression matrix)

| Protected artifact | Check | This slice touches? |
|---|---|---|
| Provider source `segmented_loss_and_grad.py` | blob SHA-256 `205721...` | No |
| Provider test `test_ds4_segmented_loss_and_grad.py` | blob SHA-256 `618a0f...` | No |
| Source sentinel `test_mlx_lm_source.py` | blob SHA-256 `dec2c2...` | No (see adjudication) |
| ADR 0028 | blob SHA-256 `0aa743...` | No |
| Path A 365-file digest | `7241924d...` | No |
| `vendor/mlx-lm` inner HEAD | `15b522f...` | No |
| `trainer.py` inner | `42e5ee2d...` | No |
| `test_tuner_trainer.py` inner | `275d6f3a...` | No |

## STOP-ESCALATE criteria

STOP and write `architect-stop.md` with error JSON if:

1. The timeout change cannot be isolated to `SMOKE_TIMEOUT_SECONDS` and
   its runtime-derived message. (Adjudicated: it CAN — the constant is
   the sole source of truth for alarm, watchdog, and message.)
2. The 1200s value requires changing lock timeout, abort timeout,
   preflight thresholds, or any parameter other than the hard timeout.
   (Adjudicated: it does NOT — all other constants are independent.)
3. `scripts/finetune_ds4.py` contains a catalog constant referencing
   the segmented smoke timeout. (Adjudicated: it does NOT —
   `DS4_GGUF_BASE_SMOKE_TIMEOUT = 900` is a separate Track-A timeout.)
4. The source-sentinel manifest requires repinning. (Adjudicated: NO —
   sentinel covers `python-envs/mlx/src/` only.)
5. Any real asset read/execution is required for architecture. (No —
   all analysis is from source code and existing evidence.)

None of these STOP conditions triggered. Design is unambiguous.

## Risks and caveats

1. **1200s may still be insufficient.** The 14.3a smoke reached trainer
   startup but showed 0/1 iterations at 600s. The model is large
   (4-bit quantized DeepSeek V4 Flash on M3 Ultra). A single
   forward/backward pass of one 4096-token microbatch through 43 layers
   with gradient checkpointing may take more than 1200s. If Phase 2 times
   out again at 1200s, the failure is terminal STOP/ESCALATE — no further
   timeout increases are authorized by this slice.

2. **The backup watchdog fires at 1205s (1200 + 5).** This is 5 seconds
   after the signal alarm. If the main thread is blocked in a C extension
   (e.g., MLX Metal kernel) and cannot receive SIGALRM, the backup thread
   sends SIGALRM via `os.kill(os.getpid(), signal.SIGALRM)` at 1205s.
   If MLX still doesn't return to Python, the process may hang beyond
   1205s. This is a pre-existing limitation of the watchdog design — not
   introduced or changed by this slice.

3. **`test_timeout_message_contains_1200s` calls `_timeout_handler`
   directly.** The handler raises `SystemExit`. The test must catch
   `SystemExit` and then assert on the module-global `_TIMEOUT_REASON`.
   The handler also sets `_TIMEOUT_FLAG = True`, which should be reset
   to `False` after the test to avoid cross-test contamination. Use
   `monkeypatch` or explicit reset.

## Coder Phase 1 can proceed now

The design is complete and unambiguous. Coder should:

1. Write the `TestTimeoutRepin` tests (red).
2. Change `SMOKE_TIMEOUT_SECONDS = 1200` (green) — one literal on line 55.
3. Update `docs/architecture.md` (C3a).
4. Verify `docs/backlog.md` Story 14.3b presence.
5. Run the full focused test suite under `python-envs/mlx/.venv`.
6. `git add` all modified test files (tracking hygiene HARD RULE).
7. No `git commit`, no `git push`.