# Story 14.3b — Coder notes (1200s timeout repin)

Revision: `14-3b-coder-r1`
Four-file functional hash: `sha256:53e598f6a872b82e7bab3405106f9f88303a2997d5e4637ab1aecb36dfe19cc7`

## What changed

### Production source (1 literal)

- `scripts/ds4_segmented_smoke.py` line 55: `SMOKE_TIMEOUT_SECONDS = 600` → `1200`
  Runtime-derived sites (alarm, watchdog deadline, timeout message) update
  automatically. No separate string/edit needed — Architect adjudicated.

### Tests (5 new, all GREEN)

- `tests/test_ds4_segmented_smoke.py`: added `TestTimeoutRepin` class after
  `TestTimeoutPreflight`, 5 methods:
  - `test_timeout_constant_is_1200` — direct constant assertion
  - `test_timeout_message_contains_1200s` — monkeypatched handler, exact string match
  - `test_lock_timeout_unchanged_at_60` — regression guard
  - `test_abort_timeout_unchanged_at_2` — regression guard
  - `test_backup_watchdog_deadline_uses_1200` — arithmetic check (1200 + 5 = 1205)

### Documentation

- `docs/architecture.md`: appended repin sentence to segmented-smoke paragraph
- `docs/backlog.md`: Story 14.3b already present (BA); verified at line 4766

## TDD sequence

1. **RED**: 3 failures (constant 600≠1200, message 600s≠1200s, watchdog 605≠1205),
   2 passes (lock 60, abort 2) — confirmed.
2. **GREEN**: one literal change → all 5 GREEN; 241/241 full suite GREEN.
3. No real smoke, no training, no inference, no `/Volumes/Data NVME/` access.

## Verification

- Full focused suite: 241 passed, 0 RED (test_ds4_segmented_smoke, test_finetune_ds4,
  test_mlx_lm_source, test_ds4_segmented_loss_and_grad)
- Protected hashes: 3/3 GREEN (TestProtected)
- py_compile: OK
- All 4 files tracked and staged (`git ls-files` confirmed, `git add` done)
- No commit, no push

## Preserved (exhaustive)

All invariants N1-N20 intact:
- `SMOKE_LOCK_TIMEOUT_S = 60`, `SMOKE_ABORT_TIMEOUT = 2` unchanged
- All pinned paths, args, catalog entries unchanged
- `scripts/finetune_ds4.py`, `segmented_loss_and_grad.py`, vendor, sentinel untouched
- No retry/fallback/provider-call-count abort-condition changes
- Watchdog mechanism (signal.SIGALRM + backup thread) unchanged
