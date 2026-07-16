# Story 14.4 — Coder r5 evidence-ordering closure

Functional candidate already passes fresh synthetic clone 246/246. Resolve final-report self-reference with explicit two-commit design:

Commit 1 candidate (reviewed implementation/hygiene):
- Remove `review-final.md` and `test-report-final.md` placeholder additions from index. Keep completed Test Manager report physically present under ignored custom-handoffs for post-gate evidence commit.
- Do not include final gate outputs in Commit 1 manifest; they do not exist until after gate.
- Update coder-notes-r4 stale counts or add coder-notes-r5 superseding exact counts.
- Rebuild complete sorted Commit 1 manifest LAST from exact index; include coder-notes-r5 and all pre-gate handoffs, exclude final gate reports.
- Add a concise two-commit evidence-ordering note to Story 14.4 backlog: Commit 1 reviewed implementation; Commit 2 final reports + COMPLETE status after PASS/GREEN. Do not claim gates passed yet.
- Force-add task-coder-r5/coder-notes-r5. Zero unstaged/nonignored untracked; ignored final test report allowed and documented.
- Recheck cached diff, manifest equality, binary scan, Path A, protected hashes, synthetic fresh-clone exact suite. No code changes, no commit/push.

Final Reviewer/Tester will write ignored report files after reviewing Commit 1. Parent then commits Commit 1, stages reports/backlog closeout, and commits evidence as Commit 2. Marker when complete.