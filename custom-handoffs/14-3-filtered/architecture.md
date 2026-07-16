# Slice 14.3a — Architecture: Filtered Smoke Dataset Repin

> Repin the bounded real-smoke contract's pinned dataset path from
> `mlx-4096` to the operator-created filtered copy `mlx-4096-smoke`.
> No production code edits in this slice; this document is the Architect
> handoff. Coder implements Phase 1 (path constant + catalog + tests).
> Phase 2 (real smoke) is separately authorized, future.

## Scope

This slice repins exactly one pinned constant and its corresponding
catalog entry. The original dataset at `mlx-4096` stays immutable and
unreferenced as the smoke entry point. All other pinned parameters,
safety gates, abort conditions, protected files, and default command
catalog entries are preserved exactly.

## Triggering event

The original pinned smoke dataset
`/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096`
failed preflight because `train.jsonl` line 56 is approximately 8,520
tokens, exceeding the 4096 max-seq-length bound. Evidence:
`agent-output/cmux-14-3/smoke-report.json`:
`{"status":"fail","failure_code":"preflight","failure_message":"train.jsonl line 56: approx 8520 tokens > 4096"}`.

The operator created a separate filtered copy at
`/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`
with the same `train.jsonl`, `valid.jsonl`, `test.jsonl` filenames. 66
rows were excluded using the smoke preflight's own whitespace-token
conservative bound (4096). All remaining rows pass the <=4096 fallback
bound. The original dataset is byte-identical and untouched.

## Changes (exhaustive — anything not listed is forbidden)

### C1. Pinned dataset path constant

**File:** `scripts/ds4_segmented_smoke.py`
**Change:** In `_PINNED_SMOKE_PATHS`, the `"data"` value changes from:

```
/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096
```

to:

```
/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke
```

This is a single string literal change in the existing constant dict.
The pinned-path gate in `main()` compares the CLI `--data` argument
against this constant before reaching preflight. Any path that does not
match produces `failure_code: "pinned-paths"` and immediate exit. This
mechanism is unchanged — only the value changes.

### C2. Command catalog entry

**File:** `scripts/finetune_ds4.py`, line ~1356, `command_catalog()` function
**Change:** In the `"ds4-segmented-smoke"` entry, the `--data` argument
changes from `{q(split_dir)}` to `{q(dataset_root / "mlx-4096-smoke")}`.

Current (line 1356):
```python
"ds4-segmented-smoke": [f"... --data {q(split_dir)} --adapter-path ..."],
```

After:
```python
"ds4-segmented-smoke": [f"... --data {q(dataset_root / "mlx-4096-smoke")} --adapter-path ..."],
```

Rationale: `split_dir = dataset_root / args.split_dir` where
`DEFAULT_SPLIT_DIR = "mlx-4096"`. The catalog currently generates
`--data <dataset_root>/mlx-4096`, matching the old pinned constant. After
the repin, the catalog must generate `--data <dataset_root>/mlx-4096-smoke`
to match the new pinned constant; otherwise the pinned-path gate rejects
the catalog-generated command.

`dataset_root` is in scope (computed at line 1302:
`dataset_root = path_arg(args.dataset_root)`). This is a literal path in
a command string, not a new env var, config key, or selector (F10
compliant). It mirrors how other catalog entries already use literal
paths like `mlx_work / 'model-4bit'` and `mlx_work / 'adapters-segmented-smoke'`.

`DEFAULT_SPLIT_DIR` stays `"mlx-4096"` — it feeds `split_dir` for all
default catalog entries, which are unchanged.

### C3. Synthetic tests

**File:** `tests/test_ds4_segmented_smoke.py`
**Change:** Add a new test class `TestFilteredDatasetRepin` and two test
methods in the existing `TestCatalog` class.

#### C3a. `TestFilteredDatasetRepin` (new class, after `TestPinnedSmokePaths`)

**`test_pinned_data_constant_is_filtered_path`**: Import
`scripts.ds4_segmented_smoke as mod` WITHOUT monkeypatching, assert
`mod._PINNED_SMOKE_PATHS["data"]` equals the exact filtered path string
`/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`.
This is a pure string comparison — no filesystem access, no monkeypatch.
The test fails if the constant still points to the original `mlx-4096`
or to any other path.

**`test_original_dataset_path_rejected`**: Monkeypatch `_PINNED_SMOKE_PATHS`
to pin `data` to the filtered path (matching the production value), pass
`--data` with the ORIGINAL `mlx-4096` path (without `-smoke`), and verify
`failure_code: "pinned-paths"`. Uses the same monkeypatch pattern as
existing `TestPinnedSmokePaths.test_arbitrary_data_rejected` (timeout,
signal, report, marker stubs). The original path string differs from the
filtered pin — the gate rejects on string inequality. No filesystem
access needed.

#### C3b. `TestCatalog` additions (existing class, after `test_catalog_redirects`)

**`test_segmented_smoke_catalog_uses_filtered_data`**: Build catalog via
`finetune_ds4.command_catalog(a)` with `split_dir="mlx-4096"` (matching
the default), assert the `ds4-segmented-smoke` entry string contains
`"mlx-4096-smoke"` (proving the catalog generates the filtered path).

**`test_default_catalog_entries_do_not_contain_filtered_path`**: Build
catalog via `finetune_ds4.command_catalog(a)` with
`split_dir="mlx-4096"`, assert that `smoke-train`, `full-train`, and
`continue-train` entry strings do NOT contain `"mlx-4096-smoke"`,
proving the default entries still use the original `split_dir`. (Note:
`mlx-4096` is a substring of `mlx-4096-smoke`, but `mlx-4096-smoke` as a
full substring does NOT appear in default entries that generate
`--data <root>/mlx-4096` — the trailing `-smoke` is absent.)

### C4. Documentation updates

#### C4a. `docs/architecture.md`

**Change:** Amend the existing segmented-smoke paragraph (line ~74)
with one sentence recording the repinned dataset path and its provenance.
The pinned dataset path is a durable contractual boundary of the
bounded real-smoke contract — the architecture doc should reflect the
current pin. Example sentence to append:

> The pinned dataset path was repinned from `mlx-4096` to
> `mlx-4096-smoke` (filtered copy, 66 rows excluded by the same
> whitespace-token conservative bound preflight uses) after the original
> failed preflight on a row exceeding 4096 tokens. The original dataset
> remains immutable and is rejected by the pinned-path gate.

#### C4b. `docs/backlog.md`

**Status:** Already updated by BA (Story 14.3a exists, status "READY FOR
ARCHITECTURE"). Architect verifies presence. Status should be updated to
reflect architecture is complete (Coder ready).

#### C4c. `custom-handoffs/14-3/requirements.md` R14.3-3

**Change:** Amend the asset identity table dataset row from `mlx-4096`
to `mlx-4096-smoke` with a filter provenance note. This is a contract
amendment recording the operator-authorized repin, not a relaxation of
any test, bound, or safety gate. The BA requirements explicitly authorize
this amendment.

## What does NOT change (binding)

| # | Invariant | Verification |
|---|-----------|-------------|
| N1 | Original dataset at `mlx-4096` not modified, read, or referenced as smoke entry point after repin | `test_original_dataset_path_rejected` + grep shows no `mlx-4096"` without `-smoke` in `_PINNED_SMOKE_PATHS` |
| N2 | All other pinned smoke paths (model, adapter, config) unchanged | `test_pinned_data_constant_is_filtered_path` can also assert other paths unchanged (optional) |
| N3 | All other pinned parameters (iters, batch-size, learning-rate, max-seq-length, mask-prompt, grad-checkpoint, segment-size, timeout) unchanged | Existing `TestPinnedSmokeArgs` tests (4 tests) — no changes needed, they use `_pin_test_paths` and exercise arg rejection, not data path |
| N4 | Default `smoke-train`, `full-train`, `continue-train` catalog entries unchanged in source, behavior, identity | `test_default_catalog_entries_do_not_contain_filtered_path` + existing `test_finetune_ds4.py` tests |
| N5 | `segmented_loss_and_grad.py` (provider source, SHA-256 `205721...`) unchanged | `TestProtected.test_provider_source` — no changes |
| N6 | `test_ds4_segmented_loss_and_grad.py` (SHA-256 `618a0f...`) unchanged | `TestProtected.test_provider_test` — no changes |
| N7 | `test_mlx_lm_source.py` (SHA-256 `dec2c2...`) unchanged | `TestProtected.test_source_sentinel` — no changes needed |
| N8 | No `vendor/mlx-lm/` inner file changes | Vendor HEAD `15b522f...` stay pinned |
| N9 | No new env var, config-file key, mode flag, or alternate dataset path selector | C2 uses a literal path — F10 compliant |
| N10 | All abort conditions preserved | No abort logic touched |
| N11 | No retry/fallback/provider-call-count policy change | No retry logic touched |

## Source-sentinel adjudication

The source sentinel manifest in `test_mlx_lm_source.py` covers only files
under `python-envs/mlx/src/` (19 pre-provider files counted by
`semantic_source_manifest(root, source)` walking that directory tree).
The sentinel constants are:
- `SEMANTIC_MLX_SRC_PRE_PROVIDER_COUNT = 19`
- `SEMANTIC_MLX_SRC_PRE_PROVIDER_SHA256 = "8881561e..."`
- `SEMANTIC_MLX_SRC_PROVIDER_PATH = "python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py"`

`scripts/finetune_ds4.py` is NOT under `python-envs/mlx/src/` and is NOT
counted by the sentinel manifest. Editing `scripts/finetune_ds4.py` (C2)
does NOT change the sentinel manifest. Therefore `test_mlx_lm_source.py`
does NOT need to be changed, and its blob SHA-256 `dec2c2b5...` remains
intact.

**Adjudication: no source-sentinel repin needed for this slice.**

## Filter-report provenance verification

### No filter-report.json exists

No `filter-report.json` file exists at `agent-output/cmux-14-3/` or
any other path examined. The filter provenance is carried in:

1. `agent-output/cmux-14-3/smoke-report.json` — evidences the original
   preflight failure (row 56, ~8520 tokens > 4096).
2. `custom-handoffs/14-3-filtered/requirements.md` (BA) — documents the
   operator's filtered copy creation: 66 rows excluded, same
   whitespace-token conservative bound (4096), all remaining rows pass
   <=4096, original dataset untouched.
3. `docs/backlog.md` Story 14.3a — records the repinned dataset path
   and filter provenance for the backlog.

### Verification path

The filter used the same whitespace-token conservative bound that the
smoke preflight uses when no tokenizer is available. By construction,
every row that passes the filter also passes the preflight. The
verification is not a separate document check — it IS the Phase 2 smoke
preflight:

1. **This slice (implementation)**: Repin the path constant and catalog
   entry. No real asset reads (R-F-9). The filter-report provenance is
   carried by the BA requirements and is trusted as operator-authorized.
   No filter-report.json needs to be read or created.

2. **Phase 2 (future, separately authorized)**: When the bounded smoke
   runs against the filtered copy, preflight scans every row in
   `train.jsonl`, `valid.jsonl`, `test.jsonl`. If the filter is correct,
   all rows pass <=4096. If the filter missed any row, the smoke aborts
   with `failure_code: "preflight"` — the same abort that caught the
   original failure. The pinned-path gate ensures only the filtered copy
   is used; the original `mlx-4096` path is rejected at entry before
   preflight even runs.

3. **No separate provenance artifact needed during implementation**:
   The filtered copy IS the provenance. Its existence and row counts are
   the operator's evidence. Creating or reading a filter-report.json
   during this implementation slice would violate R-F-9 (no real asset
   reads). The Phase 2 preflight is the natural verification gate.

## File scope

| Path | Action | Justification |
|------|--------|---------------|
| `scripts/ds4_segmented_smoke.py` | EDIT (1 literal in `_PINNED_SMOKE_PATHS`) | Repin data constant |
| `scripts/finetune_ds4.py` | EDIT (1 `--data` arg in `ds4-segmented-smoke` catalog entry) | Catalog matches pinned constant |
| `tests/test_ds4_segmented_smoke.py` | EDIT (add `TestFilteredDatasetRepin` class + 2 `TestCatalog` methods) | TDD: new path enforced, original rejected, default unchanged |
| `docs/architecture.md` | EDIT (1 sentence in segmented-smoke paragraph, line ~74) | Record repinned dataset path as durable boundary |
| `docs/backlog.md` | VERIFY presence (BA already updated) | Story 14.3a status update only |
| `custom-handoffs/14-3/requirements.md` | EDIT (R14.3-3 asset identity table dataset row) | Contract amendment, not relaxation |

### Forbidden file changes

- `vendor/mlx-lm/` (any inner file; inner HEAD `15b522f...` frozen)
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` (provider source pin `205721...`)
- `tests/test_ds4_segmented_loss_and_grad.py` (blob `618a0f...`)
- `tests/test_mlx_lm_source.py` (blob `dec2c2...`) — see source-sentinel adjudication
- `tests/test_finetune_ds4.py` — no changes needed; `DEFAULT_SPLIT_DIR` unchanged, `ds4-segmented-smoke` not tested there
- Any file under `agent-output/cmux-13-3b/`

## TDD sequence

### Red stage (test file only, no production edits)

1. Add `TestFilteredDatasetRepin.test_pinned_data_constant_is_filtered_path` — RED because constant still `mlx-4096`.
2. Add `TestFilteredDatasetRepin.test_original_dataset_path_rejected` — RED because pinned constant still `mlx-4096`, so original path matches the pin (no rejection).
3. Add `TestCatalog.test_segmented_smoke_catalog_uses_filtered_data` — RED because catalog still uses `split_dir` (resolves to `mlx-4096`, not `mlx-4096-smoke`).
4. Add `TestCatalog.test_default_catalog_entries_do_not_contain_filtered_path` — may initially PASS (default entries never had `mlx-4096-smoke`); this is a regression guard for the green stage.

### Green stage (production edits)

1. C1: Change `_PINNED_SMOKE_PATHS["data"]` to `mlx-4096-smoke` in `ds4_segmented_smoke.py`.
2. C2: Change `--data {q(split_dir)}` to `--data {q(dataset_root / "mlx-4096-smoke")}` in `finetune_ds4.py` catalog entry.
3. C4a: Amend `docs/architecture.md` with repinned path sentence.
4. C4c: Amend `custom-handoffs/14-3/requirements.md` R14.3-3 asset table.
5. Verify docs/backlog.md Story 14.3a presence and update status.

### Full test suite

All existing tests in `test_ds4_segmented_smoke.py` (T1 through T17,
TestPinnedSmokeArgs, TestLoraExactKeySet, TestPinnedSmokePaths, etc.)
must remain GREEN. The new tests must go GREEN.

`test_finetune_ds4.py` must remain GREEN — it does not test the
`ds4-segmented-smoke` catalog entry's `--data` value and does not break
from the catalog change. The `test_catalog_redirects` test in
`test_ds4_segmented_smoke.py::TestCatalog` checks only `"smoke-log.txt"`
and `"2>&1"` in the catalog entry — these remain present after C2.

`test_mlx_lm_source.py` must remain GREEN — no file under
`python-envs/mlx/src/` is touched.

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

1. The filtered dataset path, row counts, or exclusion method in the BA
   requirements contradict the operator's stated creation process or the
   `smoke-report.json` evidence.
2. The repin requires relaxing any pinned parameter, bound, or safety
   gate.
3. The repin requires touching any protected file (provider source,
   provider test, source sentinel, ADR 0028, Path A evidence, vendor
   inner files).
4. The source-sentinel manifest in `test_mlx_lm_source.py` requires
   repinning due to `scripts/finetune_ds4.py` edits. (Adjudicated: NO —
   sentinel covers `python-envs/mlx/src/` only, not `scripts/`.)
5. The catalog entry change (C2) would require introducing a new env
   var, config-file key, mode flag, or alternate dataset path selector.
   (Adjudicated: NO — C2 uses a literal path, same as existing
   `mlx_work / 'model-4bit'` pattern.)

None of these STOP conditions triggered. Design is unambiguous.

## Risks and caveats

1. **Phase 2 preflight is the only real verification of the filtered
   copy.** If the operator's filter missed a row, Phase 2 will abort with
   `failure_code: "preflight"` — same behavior as the original failure.
   This is safe by design: preflight is fail-closed.

2. **The `test_catalog_redirects` test uses `split_dir="x"`.** After C2,
   the `ds4-segmented-smoke` entry ignores `split_dir` for `--data`
   (uses `dataset_root / "mlx-4096-smoke"` instead). The test checks only
   `"smoke-log.txt"` and `"2>&1"`, so it passes. If a future test checks
   that `split_dir` appears in the segmented-smoke entry, it would fail —
   but that's expected behavior, not a regression.

3. **`DEFAULT_SPLIT_DIR` stays `"mlx-4096"`.** This is correct: the
   default catalog entries (`smoke-train`, etc.) still point to the
   original (unfiltered) dataset. Only `ds4-segmented-smoke` deviates to
   the filtered copy. If a future full-train or smoke-train run is
   attempted against the filtered copy, it would need separate
   authorization.

4. **`test_finetune_ds4.py` uses `split_dir="mlx-4096"` in its stub.**
   This matches `DEFAULT_SPLIT_DIR` and is not affected by C2. The
   `ds4-segmented-smoke` entry was not tested in `test_finetune_ds4.py`
   before this slice; no new tests are needed there.

## Coder Phase 1 can proceed now

The design is complete and unambiguous. Coder should:

1. Write the 4 new tests (red).
2. Implement C1 + C2 (green).
3. Update docs (C4a, C4c).
4. Run the full test suite under `python-envs/mlx/.venv`.
5. `git add` all modified/new test files (tracking hygiene HARD RULE).
6. No `git commit`, no `git push`.