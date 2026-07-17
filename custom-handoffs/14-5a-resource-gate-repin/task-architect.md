# Story 14.5a — Architect resource repair + attempt-2 namespace

Read requirements.md, Story14.5 architecture/code/tests, attempt-1 failure reports/log/markers, and canonical docs.

Design minimal durable repair:
- safe per-process psutil scan that skips/counts only AccessDenied/NoSuchProcess/ZombieProcess, continues scanning, preserves >50GiB/current-PID/headroom/disk gates, fails unknown errors;
- exact report schema for skipped PID/type evidence;
- immutable attempt-2 Phase A2/B2/final namespace from requirements, while attempt-1 bytes remain untouched and never accepted;
- CLI/catalog design (explicit phase-a2/phase-b2 or explicit attempt selector), exact command paths, reports/markers/logs/output/resume bindings, collision gates before log/output mutation;
- unchanged training/runtime budgets and all Story14.5 terminal/identity/lock contracts;
- mutation-sensitive tests including realistic fake psutil processes and path substitutions;
- separate Phase A2/Phase B2 authorization and visible run protocol;
- durable docs and evidence-hash capture proving attempt-1 immutability.

No real assets/training/inference, attempt-1 cleanup, commit, push. Write architecture.md and marker.