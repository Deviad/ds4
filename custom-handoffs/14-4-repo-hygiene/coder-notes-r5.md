# Story 14.4 — Coder r5 (two-commit evidence-ordering closure)

## r5 change

Removed `review-final.md` and `test-report-final.md` placeholders from the
staged Commit 1 index. The completed Test Manager report (`GREEN`) remains
physically present on disk under `custom-handoffs/14-4-repo-hygiene/`
(ignored by gitignore). It will be force-added in Commit 2 together with the
Reviewer final report and backlog closeout after both gates PASS.

## Two-commit design

| Commit | Contents | When |
|--------|----------|------|
| **Commit 1** | Reviewed implementation: all hygiene changes, tracked files, manifests, pre-gate handoffs, inner vendor pin advancement | After Reviewer PASS on Commit 1 index |
| **Commit 2** | Final gate evidence: Reviewer report, Test Manager report, backlog closeout → COMPLETE status | After both gates PASS; parent commits |

Commit 1's manifest does NOT include final gate outputs. No gate status
claimed in Commit 1.

## Current Commit 1 staged state

| Metric | Value |
|--------|-------|
| Additions | 114 |
| Modifications | 12 |
| Deletions | 21 |
| Total staged | 145 |
| Untracked | 0 |
| Unstaged | 0 |
| Final gate reports in index | 0 (on disk only, ignored) |
| `git diff --cached --check` | exit 0 |
| Staged binaries | 0 |
| Full suite | **246 passed, 3 skipped, 2 subtests** |
| Synthetic fresh clone | **246 passed** (submodule from remote) |
| Path A | 365 intact |
| Protected hashes | 3/3 OK |
| Remote reachability | `80fab4e` on `origin/story-14-loss-and-grad-seam` |

## r4→r5 delta

- Removed: `review-final.md` (placeholder), `test-report-final.md` (placeholder) from index
- Staged count: 147 → 145 → 147 (placeholders removed, r5 files added)
- Additions: 114 → 112 → 114
- Added: two-commit design to backlog Story 14.4
