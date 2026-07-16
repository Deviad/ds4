# Story 14.4 — repository reproducibility and stale-file cleanup

User authorized: inspect what belongs to current implementation, commit legitimate slipped files, delete remaining stale/generated files. No push.

Read AGENTS.md, docs/architecture.md, docs/technical-spec.md, docs/backlog.md, commit af08c25, current git status, current source imports, untracked files, and inner vendor/mlx-lm status. Produce requirements and update docs/backlog.md with a trackable user story and acceptance criteria.

Required classification:
1. Current implementation files required for fresh-clone reproducibility, including untracked modules imported by committed Epic 14.3 source/tests, legitimate tests/scripts/docs/project-local .pi scaffolding, and vendor inner changes.
2. Stale/generated/ephemeral files safe to delete: cmux status markers, dispatch epochs, pipeline-private state, PID/RC files, caches, transient logs, obsolete handoffs/reviews and scratch root markdown.
3. Unrelated or ambiguous user work that must not be deleted without evidence.
4. Resolve vendor pin tension: outer gitlink is 15b522f..., but inner staged trainer.py/test changes may be required by current implementation. Require fresh-clone reproducibility and explicit reviewed pin update if committed.

Acceptance must require: tracked baseline tests; no untracked implementation dependencies; no generated binaries/cache/PID/RC/dispatch files committed; Reviewer PASS; Tester GREEN; no push; deletion manifest and commit manifest; current smoke evidence preserved; Path A permanent STOP preserved. Write requirements.md and marker.