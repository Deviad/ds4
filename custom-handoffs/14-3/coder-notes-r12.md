# Coder r12 notes

Revision: 14-3-coder-r12
Attempt: 1 (r11 blocker closure)

Functional hash (staged four-file): `e4cc2c96dcf2a790ca0fff4068b41107e2b3d33795c25701c001ea9114b13359`.
Command: `git diff --cached --binary -- scripts/ds4_segmented_smoke.py tests/test_ds4_segmented_smoke.py scripts/finetune_ds4.py docs/architecture.md | shasum -a 256`

## Closed r11 blockers (all five)

### 1. Registry staging (✓)
Staged `custom-handoffs/14-3/.pipeline-private/implementation-state.v1.tsv` so index and worktree both register `14-3-coder-r11` at hash `8911e91dea22a347e416c48057039d266c79f458332d6f96e191cfa01ce7d1c0`. Index and worktree are now byte-identical.

### 2. Exact pinned asset paths (✓)
Added `_PINNED_SMOKE_PATHS` dict in `scripts/ds4_segmented_smoke.py` with the four exact Phase-1 paths:
- model: `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`
- data: `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096`
- adapter_path: `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke`
- config: `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`

Enforced in `main()` BEFORE `_check_preflight` and after config merge. Any path deviation produces `failure_code: pinned-paths` and exit. Synthetic helpers that call `_check_preflight()` directly are unaffected; they still exercise preflight without real assets.

Tracked tests in `TestPinnedSmokePaths` (4 tests): arbitrary model/data/adapter rejection + valid paths accepted.

### 3. LoRA NaN/Inf/bool rejection (✓)
Added `math.isfinite()` check plus explicit `isinstance(x, bool)` rejection to rank, scale, and dropout validation in `_check_preflight`. NaN, Inf, True, and False are all rejected with `SystemExit`.

Tracked tests in `TestLoraNaN` (8 tests): NaN rank/scale/dropout, Inf rank/scale, bool rank/scale/dropout.

### 4. Argparse reason capture (✓)
Parser error handling now captures argparse stderr via `io.StringIO()` during `parse_args()`. Uses the captured stderr text as the primary failure message instead of `str(SystemExit(2))` which was just `"2"`. Preserves `failure_code: argparse` classification and `return code 2`.

Updated `TestParserClassification.test_argparse_error_classified_as_argparse_not_timeout` to assert the captured message is longer than 5 chars and contains "segment" — proving meaningful text is preserved.

### 5. Cleanup warning durable sidecar verification (✓)
Rewrote `TestCleanupWarningPersistence.test_cleanup_warnings_in_report_on_failure` to:
- Verify in-memory `_CLEANUP_WARNINGS` list (unchanged)
- Verify durable `agent-output/cmux-14-3/cleanup-warnings.json` sidecar file exists and contains the warning
- Clean up sidecar file after test (before/after)

No longer asserts that cleanup warnings appear in the smoke report (which is written before the finally block). The contract is now: warnings are durable via stderr + sidecar file, not via the pre-cleanup report.

## Per-blocker verification

| Blocker | Fix | Test evidence |
|---|---|---|
| 1. Registry staging | `git add` | `git diff --cached` shows r11, `git status` clean |
| 2. Pinned paths | `_PINNED_SMOKE_PATHS` + `main()` check | `TestPinnedSmokePaths` (4 tests) |
| 3. LoRA NaN/bool | `math.isfinite` + `isinstance(x, bool)` | `TestLoraNaN` (8 tests) |
| 4. Argparse reason | `io.StringIO` stderr capture | `TestParserClassification` updated |
| 5. Cleanup warning | Sidecar file verification | `TestCleanupWarningPersistence` updated |

## Suite results

- Focused smoke suite: 119 passed
- Finetune suite: 51 passed
- Source sentinel: 14 passed
- Protected hashes: all match
- `git diff --check`: clean
- py_compile: both files clean

## No real assets

No `/Volumes/Data NVME` paths read, stat'd, or opened. Phase 2, smoke, training, inference, backend, install, network — none accessed. No commit or push.

## Staging

All touched files staged:
- `scripts/ds4_segmented_smoke.py` (r12 edits)
- `tests/test_ds4_segmented_smoke.py` (r12 tests)
- `custom-handoffs/14-3/.pipeline-private/implementation-state.v1.tsv` (registry)

Protected files byte-identical:
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`
- `tests/test_ds4_segmented_loss_and_grad.py`
- `tests/test_mlx_lm_source.py`
- `vendor/mlx-lm/` inner files (none touched)
