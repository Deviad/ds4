# Story 14.3b — Coder r3 (final reviewer closure)

Revision: `14-3b-coder-r3`
Cached-diff functional hash: `sha256:29c0bdd1121ec7623093b74d84a680e8c367edbdb93c17a4fa7cf86a1a9a725d`

## r3 changes (vs r2)

### BLOCKER 1 — hash reconciled
- r2 notes carried stale header hash (`017193ce...`) that diverged from actual
  staged bytes. r3 recomputes `git diff --cached --binary` from current staged
  state: `29c0bdd1121ec7623093b74d84a680e8c367edbdb93c17a4fa7cf86a1a9a725d`.
  No self-reference in notes — only the hash recorded in this document.

### BLOCKER 2 — watchdog test: exact expression oracle
- Replaced weak `"SMOKE_TIMEOUT_SECONDS" in source` check with exact regex
  oracle requiring `time.monotonic()\s*\+\s*SMOKE_TIMEOUT_SECONDS\s*\+\s*5\b`
  in `_backup_watchdog` source.
- Verified: mutation `+5 → +6` correctly rejected (`source_oracle_match=False`).
- Arithmetic check (`1200 + 5 == 1205`) retained for constant-drift guard.

### BLOCKER 3 — backlog status corrected
- `docs/backlog.md:4766`: status changed from false `Reviewer PASS` to
  `[ IMPLEMENTED — code, tests, docs staged; 241/241 GREEN; Reviewer/Test
  Manager pending ]`.
- Acceptance criteria and user story unchanged.

## Production source

- `scripts/ds4_segmented_smoke.py`: `SMOKE_TIMEOUT_SECONDS = 1200` — one literal
  on line 55. No other production edit.

## Test changes (only TestTimeoutRepin)

- `test_backup_watchdog_deadline_uses_1200`: strengthened to exact expression
  oracle (`time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5` regex).
- All other 4 tests in `TestTimeoutRepin` unchanged from r2.

## Verification

| Check | Result |
|-------|--------|
| Full focused suite | **241 passed, 0 RED** |
| Timeout tests (TestTimeoutRepin) | **5/5 GREEN** |
| Protected hashes (TestProtected) | **3/3 GREEN** |
| Mutation sensitivity (+5→+6) | **Correctly rejected** |
| py_compile | OK (2/2) |
| `git diff --cached --check` | exit 0 |
| All 4 files tracked + staged | ✅ |
| No commit, no push | ✅ |
| No real smoke/training/inference/assets | ✅ |

## Reviewer closure items (all three)

1. ✅ Cached-diff hash registered as `14-3b-coder-r3`
2. ✅ Watchdog test uses exact expression oracle; +6 mutation fails
3. ✅ Backlog status reflects actual state (Reviewer/Test Manager pending)
