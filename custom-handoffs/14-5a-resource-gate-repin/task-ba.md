# Story 14.5a — resource observer repair + attempt-2 repin

User authorized reviewed repair after Phase A attempt 1 failed closed in 0.045s before model/dataset/training. Read AGENTS, Story 14.5 requirements/architecture/code/tests, failure reports/log/markers, and backlog.

Evidence:
- error exactly `(pid=0)` = `psutil.AccessDenied(pid=0)` while `_resource_gate()` read process memory;
- 0 updates/provider calls, no adapter files;
- lock acquired/released once; fail reports and Phase-A/final fail markers durable;
- no retry performed.

Update docs/backlog.md with Story 14.5a user story and acceptance criteria. Requirements must:
1. Preserve attempt-1 reports/log/markers byte-for-byte; never delete/rename/overwrite them.
2. Repair process scan per-process: skip only `psutil.AccessDenied`, `NoSuchProcess`, `ZombieProcess`; retain observable skipped PID/type counts; all other errors fail; >50 GiB competing-process gate and current-process exclusion remain.
3. Mutation-sensitive tests for PID 0 access denied, disappearing/zombie process, genuine >50 GiB competitor, unknown exception, current process, memory/disk gates.
4. Repin a new immutable attempt namespace—not reuse canonical attempt-1 paths. Specify exact attempt-2 Phase A/B output/report/log/marker/final paths and update Phase B resume path. Existing attempt-1 artifacts must not block attempt 2, but any attempt-2 artifact must.
5. Preserve same model/dataset/config/training pins and budgets: A2 2 updates/2700s, B2 1 update/1500s, total 4200s; one attempt each, no automatic retry/fallback.
6. Reviewer PASS + Tester GREEN + fresh explicit operator authorization before Phase A2; separate authorization before Phase B2.
7. Preserve Path A, Story14.3 smoke, vendor/provider/CUDA/distributed/Metal invariants and non-claims.

Write requirements.md and marker. No real asset access, training, inference, cleanup, commit, push.