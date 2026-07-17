# Story 14.5a — Coder synthetic repair

Read requirements.md/architecture.md completely. Direct TDD red→green; no delegate/roles/real model-dataset-adapter access/training/inference/cleanup/commit/push.

Implement only reviewed scope:
1. Refactor `_resource_gate()` into per-process scan: skip/count exact psutil AccessDenied/NoSuchProcess/ZombieProcess with PID/type evidence; continue scanning; unknown/enumeration errors fail; current PID excluded; >50GiB, 32GiB headroom, 1GiB disk unchanged.
2. Add immutable attempt-2 namespace and exact A2/B2/final paths/phase specs/logs/reports/markers/output/resume. Attempt-1 constants/evidence remain untouched and can neither block nor satisfy attempt2.
3. Bind exact attempt-1 five-file size/SHA manifest from architecture; attempt2 preflight verifies but never modifies.
4. Implement explicit attempt-2 catalog commands, non-default. Preserve existing Story14.5 attempt1 and all default command bytes.
5. Collision before log/output mutation. Implement exclusive log creation/noclobber FD attestation from architecture; pre/post-lock collision recheck. Any attempt2 artifact blocks without writes/cleanup; B2 accepts only exact A2 dependencies.
6. Preserve Story14.5 training/identity/lock/timeout/terminal contracts and budgets A2 2/2700, B2 1/1500, total4200.
7. Add mutation-sensitive tests for resource exceptions/gates/schemas and every attempt2 path/attempt1 substitution/collision/log race/historical hash. Import/tests no `/Volumes` access.
8. Update docs architecture/technical/backlog actual synthetic status. Track all verdict files/handoffs; no `.cmux-status` staged.
9. Focused, exact six-file, real trainer, pycompile/diff/PathA/smoke/provider/vendor/attempt1 hash gates. Write coder-notes.md and marker.