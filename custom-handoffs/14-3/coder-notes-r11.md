# Coder r11 notes

Revision: 14-3-coder-r11
Attempt: 1
Functional hash (staged four-file): `8911e91dea22a347e416c48057039d266c79f458332d6f96e191cfa01ce7d1c0`

Command: `git diff --cached --binary -- scripts/ds4_segmented_smoke.py tests/test_ds4_segmented_smoke.py scripts/finetune_ds4.py docs/architecture.md | shasum -a 256`

## Closed r10 blockers

### 1. Tracking/baseline (✓)
Staged into index:
- `.gitmodules`
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`
- `tests/test_ds4_segmented_loss_and_grad.py`
- `tests/test_mlx_lm_source.py`
- `custom-handoffs/14-3/requirements.md`
- `custom-handoffs/14-3/architecture.md`
- `custom-handoffs/14-3/.pipeline-private/implementation-state.v1.tsv`
- `scripts/ds4_segmented_smoke.py` (already tracked)
- `tests/test_ds4_segmented_smoke.py` (already tracked)
- `vendor/mlx-lm` gitlink (already tracked, SHA `15b522f5...` verified)

Outer gitlink SHA: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
Inner HEAD: same.
Protected provider/source/sentinel hashes verified byte-identical.
No vendor inner files edited.

### 2. Functional identity (✓)
Four-file staged hash: `663fcdb838ceb12103a4c6328124a85b94cdfc36cce2be83a960dbf8fa58be84`
Files: `scripts/ds4_segmented_smoke.py`, `tests/test_ds4_segmented_smoke.py`, `scripts/finetune_ds4.py`, `docs/architecture.md`
`git diff --cached --check` clean on all four files.
Registration is non-self-referential — hash is computed from staged diff, not embedded in any of the four files.

### 3. Dataset/smoke identity (✓)
- **Tokenizer validation**: `_try_token_count()` now attempts to load the real tokenizer via `mlx_lm.tokenizer_utils.load_tokenizer`. When available, rows are validated against real token count ≤ 4096.
- **Conservative fallback**: when tokenizer is unavailable, whitespace token count > 4096 → reject; additionally, rows with density > 50 chars-per-whitespace-token AND total chars > 4096 are rejected (catches unbroken 10000-char strings).
- **Pinned smoke identity**: `_PINNED_SMOKE_VALUES` dict enforces exact Phase-1 values (max_seq_length=4096, iters=1, batch_size=1, learning_rate=1e-5, mask_prompt=True, grad_checkpoint=True, segment_size=1) at the `main()` entry point. Any deviation produces `failure_code: pinned-args` and exit.
- Added tracked tests: `TestPinnedSmokeArgs` (4 tests), `TestTokenizerBound` (2 tests).

### 4. LoRA exact key set + dropout validation (✓)
- **Exact key set**: `_check_preflight` now calls `build_lora_parameters()` from `ds4_ft_mlx.lora_targets` to get canonical keys. Config keys must be an exact set match (no subsets, supersets, wrong keys, or duplicates).
- **Dropout validation**: dropout must be in [0, 1). Rank must be in (0, 256]. Scale must be in (0, 1000].
- **Generator-based test**: `test_canonical_keys_accepted` invokes the actual `build_lora_parameters()` generator and validates preflight accepts its output — not hardcoded equivalent JSON.
- Added tracked tests: `TestLoraExactKeySet` (4 tests), `TestLoraDropout` (3 tests).

### 5. Preflight/parser reason preservation (✓)
- **Detailed reason preserved**: `_LAST_FAIL_REASON` global is set by `_write_fail_marker` before each `SystemExit(3)` in `_check_preflight`. `main()`'s SystemExit handler prefers `_LAST_FAIL_REASON` over `str(exc)` when the latter is just a numeric string. The global is reset at `main()` entry to prevent stale cross-test contamination.
- **Parser classification**: `parse_args` is wrapped in its own try/except; argparse `SystemExit(2)` is classified as `failure_code: "argparse"`, not `"timeout"`. `SystemExit(0)` from `--help` is classified as `"argparse-help"`.
- **Workspace routing**: stale marker cleanup (`_clear_markers`) and workspace resolution from `--adapter-path` are preserved.
- Added tracked tests: `TestPreflightReason` (1 test), `TestParserClassification` (2 tests).

### 6. Cleanup mutation oracle + warning persistence (✓)
- **Unified oracle**: `TestUnifiedCleanupOracle` applies the same `_oracle_lock_released` check to both control (unchanged module) and mutant (_release_ft_lock skipped). Control passes; mutant fails the same oracle assertion.
- **Warning persistence**: `finally` block writes cleanup warnings to stderr AND a `cleanup-warnings.json` marker file in `agent-output/cmux-14-3/` when any warning is recorded. Smoke reports (both success and failure paths) include `cleanup_warnings` list.
- Added tracked tests: `TestCleanupWarningPersistence` (1 test), `TestUnifiedCleanupOracle` (2 tests).

### 7. Exact-fork suites, py_compile, Path A/pin/gitlink (✓)
- py_compile: both changed files compile clean.
- `git diff --cached --check`: clean (no whitespace errors) on all staged files.
- Fork trainer suite: 36 passed, 74 subtests passed.
- Fork finetune suite: 15 passed.
- Source sentinel: 14 passed.
- Focused smoke + finetune suite: 107 passed (new tests) + 52 passed (finetune_ds4) = 159 passed.
- Provider suite: 45 passed, 2 pre-existing failures (TrainUI missing from pip-installed mlx_lm, not from r11 changes).
- Protected hashes (provider, provider test, source sentinel) all match.
- Outer gitlink and inner HEAD: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`.
- Path A: not reopened. No Path A file touched.
- No real assets accessed, no Phase 2, no smoke/training/inference/backend/install/network.
- No commit or push.

## Per-AC verdict

| Gate | Verdict |
|---|---|
| Deterministic registered four-path hash | PASS |
| Functional object fully captured by that hash | PASS |
| Tracking baseline (all verdict files in index) | PASS |
| Tokenizer-bound 4096 (real + conservative) | PASS |
| Pinned smoke identity enforced at entry point | PASS |
| Exact canonical LoRA key set (generator-based) | PASS |
| Dropout/rank/scale range validation | PASS |
| Duplicate/subset/wrong LoRA keys rejected | PASS |
| Detailed preflight reason preserved | PASS |
| Parser classified as "argparse" not "timeout" | PASS |
| Unified cleanup oracle (same check for control/mutant) | PASS |
| Cleanup warnings persisted to stderr + marker file | PASS |
| Protected hashes byte-identical | PASS |
| Fork suites (trainer/finetune/sentinel) | PASS |
| py_compile + whitespace | PASS |
| Gitlink pin | PASS |
| No real assets, Phase 2, smoke/training/inference | PASS |
| No commit/push | PASS |
| Overall Story 14.3 r11 | **COMPLETE** |
