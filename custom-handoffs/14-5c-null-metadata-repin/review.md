# Story 14.5c independent review — r12b final

## Verdict

**PASS**

No blocking findings.
R12 F1/F2 remain substantively closed.
R12 F3/F4 now closed on current staged tree.

## Exact wording closure

Canonical staged text matches production and targeted oracle exactly:

`attempt-2 runtime target bindings are not immutable`

Counts:

- `docs/technical-spec.md`: exact text 1; old text 0.
- `scripts/ds4_segmented_pilot.py:775`: exact text 1; old text 0.
- `tests/test_ds4_segmented_pilot.py:4755`: exact text 1; old text 0.

Staged `docs/technical-spec.md` blob: `0912f923c1844b3e98ca278e42df2532507d69c3`.

## R12 substantive closures

Current staged test directly retains both required oracles:

- `tests/test_ds4_segmented_pilot.py:4709-4757` performs one coordinated path/size/SHA-256/payload substitution, proves weakened lower verifier accepts substituted evidence with valid report identity, then proves public guard rejects same rebound tuple with exact immutable-binding error and not unrelated missing-report error.
- `tests/test_ds4_segmented_pilot.py:4760-4874` injects every two Phase-A and four Phase-B/final publication write seams; verifies returned phase failure, exact persisted phase/final reports, both fail markers with report path/hash/contract/namespace/phase/attempt/exit/status bindings, and absence of every Phase-A/Phase-B/final OK marker.

Current staged blobs:

- `scripts/ds4_segmented_pilot.py`: `3ea6b2140d0155c9ecfd2f17fa139d4e7a899fe0`
- `scripts/finetune_ds4.py`: `a8812909ca77191c21956343fe33642bdbce516e`
- `tests/test_ds4_segmented_pilot.py`: `f596a21ca9e5359a3dcf5b1684ab4dd0ab969c84`

R12 handoff lacked individual prior blob identities sufficient for independent byte comparison. Reviewer therefore did not rely solely on prior `871` evidence; exact current-tree canonical suite was rerun.

## Independent verification

- Exact documented vendor-first six-file suite: `871 passed, 3 skipped, 1 warning, 2 subtests passed in 110.10s`.
- All six verdict-contributing test files tracked.
- Python compile check: PASS for pilot, catalog, pilot tests, and catalog tests.
- `git diff --cached --check`: PASS.
- `git diff --check`: PASS.
- Pre-review-artifact tree: 20 staged files, 0 unstaged files, 0 untracked files.
- Deterministic staged path/blob manifest SHA-256: `b7ca280d3453455b941bd7188db5baeabb9d6d93dae3d6bca95aee33fc742348`.
- Staged `.cmux-status`, nested `.cmux-status`, phase/final OK, and phase/final fail marker paths: 0.
- Protected C/Objective-C/Metal/CUDA set: 80 tracked files, 0 staged drift.
- Vendor `mlx-lm` HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`; clean.

A noncanonical diagnostic run without required vendor-first `PYTHONPATH` resolved site-installed `mlx_lm` and failed two trainer tests because that package lacks `TrainUI`. It is not verdict evidence. Exact command documented in `docs/technical-spec.md:752`, with vendor source first, passed canonical `871` suite above.

## Test Manager same-tree gate

- `custom-handoffs/14-5c-null-metadata-repin/test-report-r12b.md`: current **GREEN**.
- `.cmux-status/test-manager.done`: present.
- Git index mtime: `2026-07-18T13:21:58.525Z`.
- Test Manager report mtime: `2026-07-18T13:24:10.962Z`.
- Test Manager marker mtime: `2026-07-18T13:24:12.575Z`.
- Report and marker postdate unchanged index; final pre-review-artifact scan confirms no unstaged or untracked drift.

## Boundary

No real A3/B3 access, model/data load, provider call, training, inference, cleanup, commit, push, delegation, or cmux operation performed.
