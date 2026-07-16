# Story 14.4 — BA final closeout (2026-07-16)

## Summary

Story 14.4 is COMPLETE. Reviewer PASS (final) and Test Manager GREEN (final)
verified repository reproducibility, portable wrappers, vendor pin resolution,
and fresh-clone test baseline. Commit 1 `94a93d5` is local; no outer push.

## Reviewer PASS (final)

- Report: `custom-handoffs/14-4-repo-hygiene/review-final.md`
- Reviewed staged tree: `cb7560973525a9445b972592d8e7b79a66ef86c2`
- Staged counts: 116 A, 12 M, 21 D, total 149
- Portable-wrapper gate: 14/14 `.pi/agents/bin/*.sh` PASS
- Synthetic fresh-clone gate: `246 passed, 3 skipped, 1 warning, 2 subtests passed in 23.66s`
- Vendor gate: gitlink `80fab4e4...`, parent `15b522f...`, remote reachable on `story-14-loss-and-grad-seam`
- Commit/deletion manifests: exact, no duplicates, sorted
- `git diff --cached --check`: exit `0`; zero unstaged/untracked/binary/symlink
- Protected hashes: provider `20572191...`, provider test `8bf2a19f...`, source-sentinel `24325ef3...`, Path A `7241924d...`, ADR 0028 `0aa743b...`
- Story 14.3 smoke report/log/provenance: index blobs equal `HEAD`
- Findings: none

## Test Manager GREEN (final)

- Report: `custom-handoffs/14-4-repo-hygiene/test-report-final.md`
- Synthetic commit: `f62acbca...`
- Portable wrappers: sourced/exec/missing-target/override all OK
- Fresh clone + recursive submodule fetch: OK
- Exact five-file suite: `246 passed, 3 skipped, 1 warning, 2 subtests passed`
- Path A: 365 / `7241924d...` intact
- Vendor gitlink reachable
- Smoke/protected hashes unchanged

## Commit 1

- SHA: `94a93d5121707384d79bb93cc536d2769540a12e`
- Message: "Make DS4 fine-tuning checkout reproducible"

## Vendor resolution

- Inner: `80fab4e419a57f9465bb9e2f4e90010d645e124c` (parent `15b522f...`)
- Remote: `refs/heads/story-14-loss-and-grad-seam` on `git@github.com:Deviad/mlx-lm.git`
- Outer gitlink updated to `80fab4e`

## Stale cleanup

- 111 historical root scratch Markdown files deleted
- 21 tracked runtime-state paths removed from index
- 174 Path A evidence files preserved as frozen evidence
- `python-envs/legacy-trans/` deleted
- Ambiguous `context.md` + `adapter-converter-implementation-gpt55.md` preserved, ignored
- Generated binaries preserved locally, untracked, ignored

## Non-claims

Repository reproducibility and fresh-clone test baseline only. No convergence,
OOM repair, throughput, or full-training readiness claim. Story 14.3 smoke
evidence intact.

## No outer push

Commit 1 is local. Supervisor/operator owns the push decision.