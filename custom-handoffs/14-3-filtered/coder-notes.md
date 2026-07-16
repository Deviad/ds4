# Coder notes — Slice 14.3a filtered smoke dataset repin

## Implementation summary

Repinned the bounded real-smoke contract's pinned dataset path from `mlx-4096` to the operator-created filtered copy `mlx-4096-smoke`. Two production edits, one test file expansion, three doc amendments.

## Red stage

1. Added `TestFilteredDatasetRepin` class (3 tests):
   - `test_pinned_data_constant_is_filtered_path` — RED (constant still `mlx-4096`)
   - `test_original_dataset_path_rejected` — RED (original path matches old pin)
   - `test_filtered_path_accepted_by_pinned_gate` — pending green
2. Added `TestCatalog.test_catalog_segmented_uses_filtered_path` — RED (catalog used `split_dir`)
3. Added `TestCatalog.test_default_catalog_entries_unchanged` — passed (regression guard)

## Green stage

1. C1: `scripts/ds4_segmented_smoke.py` line 111 — changed `_PINNED_SMOKE_PATHS["data"]` from `mlx-4096` to `mlx-4096-smoke`
2. C2: `scripts/finetune_ds4.py` line 1356 — changed `--data {q(split_dir)}` to `--data {q(dataset_root / "mlx-4096-smoke")}` in `ds4-segmented-smoke` catalog entry
3. C4a: `docs/architecture.md` line 74 — appended filtered dataset repin sentence
4. C4c: `custom-handoffs/14-3/requirements.md` — amended R14.3-3 asset identity table and R14.3-4 command shape to `mlx-4096-smoke`

## Test results

Exact-fork full focused suite: **236 passed, 0 red** across:
- `tests/test_ds4_segmented_smoke.py` — 124 tests (+4 new)
- `tests/test_finetune_ds4.py` — 54 tests
- `tests/test_mlx_lm_source.py` — 14 tests
- `tests/test_ds4_segmented_loss_and_grad.py` — 44 tests

New tests:
- `TestFilteredDatasetRepin::test_pinned_data_constant_is_filtered_path` — GREEN
- `TestFilteredDatasetRepin::test_original_dataset_path_rejected` — GREEN
- `TestFilteredDatasetRepin::test_filtered_path_accepted_by_pinned_gate` — GREEN
- `TestCatalog::test_catalog_segmented_uses_filtered_path` — GREEN
- `TestCatalog::test_default_catalog_entries_unchanged` — GREEN

## Verification checks

- `_PINNED_SMOKE_PATHS["data"]` → `mlx-4096-smoke` ✓
- Catalog entry `--data` uses `dataset_root / "mlx-4096-smoke"` ✓
- Default catalog entries unchanged ✓
- Protected hashes:
  - Provider source `205721...` ✓
  - Provider test `618a0f...` ✓
  - Source sentinel `dec2c2...` ✓

## File scope

| Path | Action |
|------|--------|
| `scripts/ds4_segmented_smoke.py` | 1 literal edit |
| `scripts/finetune_ds4.py` | 1 catalog entry edit |
| `tests/test_ds4_segmented_smoke.py` | +5 tests (~95 lines) |
| `docs/architecture.md` | +1 sentence |
| `docs/backlog.md` | status update |
| `custom-handoffs/14-3/requirements.md` | 2 amendments |

## Unchanged invariants

- All pinned smoke parameters except data path
- Default catalog entries
- Provider source, provider test, source sentinel
- No vendor/mlx-lm/ edits
- No Path A evidence touched
- No real asset reads
- No commit/push
