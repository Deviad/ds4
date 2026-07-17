# Story 14.5a — Architect r8 consumer applicability adjudication

Read r7 blocked review, task-coder-r8, coder-notes-r8, requirements/architecture, namespace/catalog/runtime.

Adjudicate three REDs: A2 step1 checkpoint, A2 step2 checkpoint, B2 step1 checkpoint mutations do not change launch/training argv. Determine whether each is legitimately N/A for executable catalog because runtime produces names from save cadence/namespace and validates them post-training, or whether catalog must bind them explicitly.

Define exact applicable-consumer matrix per namespace key. Do not invent unused shell assignments or meaningless CLI flags. If N/A, specify mutation oracle in true runtime consumers (phase specs/artifact validation/report/contract) and remove invalid catalog expectation. If required, specify minimal production binding. Preserve all gates. Write architecture-r8.md and marker. No real execution/commit/push.