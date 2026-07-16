# Story 14.4 — Coder r4 (fresh-clone test fix + hash cascade)

## r4 defect

`test_active_memory_matrix` called `.write_text()` on a path inside
`agent-output/cmux-14-2/` without ensuring the parent directory existed.
That directory is gitignored and untracked — existed on local checkout from
prior runs, absent in fresh clone → `FileNotFoundError`, 245/246.

## Fix (exact per architecture-r2)

### C1 — `tests/test_ds4_segmented_loss_and_grad.py` line 1021-1023
- Before: `(PROJECT_ROOT / "agent-output/cmux-14-2/memory-r4.log").write_text(...)`
- After: captures path in `_log_path`, calls `_log_path.parent.mkdir(parents=True, exist_ok=True)`, then `_log_path.write_text(...)`
- Idempotent: no-op if directory already exists

### C2 — hash cascade in `tests/test_ds4_segmented_smoke.py`
- Provider test blob: `618a0f22` → `8bf2a19f`
- Updated `TestProtected.test_provider_test` expected hash

## Verification

| Check | Result |
|-------|--------|
| RED: active_memory_matrix with dir absent | `FileNotFoundError` → RED confirmed |
| GREEN: after C1 from clean state | **1 passed** |
| Full suite (local) | **246 passed, 3 skipped, 2 subtests** |
| Synthetic staged clone (fresh, no pre-existing cmux-14-2) | **246 passed, 3 skipped, 2 subtests** |
| Submodule fetch from remote | `80fab4e` checked out |
| Path A | 365 files intact |
| Protected hashes (3/3) | OK (provider source `205721`, provider test `8bf2a1`, sentinel `24325e`) |
| Whitespace | exit 0 |
| 0 untracked, 0 unstaged | ✅ |
| 0 staged Mach-O binaries | ✅ |
| Staged | 146 paths (113 add / 12 mod / 21 del) |
| No outer commit, no push | ✅ |

## Staged handoffs
All current-slice chain-of-custody files force-added: architecture-r2,
task-coder-r2/r3/r4, task-reviewer-r2, task-tester-r2, test-report-r2,
review.md, review-final.md (placeholder), test-report-final.md (placeholder).

## Manifests
Rebuilt from final index: sorted, complete, self-referencing.
