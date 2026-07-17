# Story 14.5 — Coder r2 reviewer blocker closure

Read latest `custom-handoffs/standby/review.md` fully. Fix every P0/P1 via TDD; no real assets/commit/push, no delegation/subagents.

Binding corrections:
1. Wire exact vendor 80fab4e APIs: load_dataset(args, tokenizer), supported TrainingArgs fields, freeze+LoRA order, construct fresh optimizer after Phase B adapter load/proof, exact train(model, optimizer, train_dataset, val_dataset, args=...), callback one-dict API incl validation, output adapter_file paths/checkpoint cadence/config.
2. Invoke real Phase A/B continuity path: start/step/final checkpoint existence, SHA/canonical digest/schema/cardinality/progression; Phase B read+bind actual Phase A marker/report/checkpoint, save resume-start before optimizer, prove source==resume-start!=final. Remove fabricated dependency.
3. Implement final report/marker and exact lock-release→report→marker ordering for success/failure; one-attempt rejects any output/report/phase/final marker; correct wall timing; cancellable watchdog.
4. Runtime preflight gates required by architecture: interpreter/versions/resolved sources/gitlink/config/topology/provenance/dataset/model identity; competing process, memory/disk thresholds; timeout before preflight; recheck under lock. No real access in synthetic tests.
5. Correct observer finiteness/schema/shapes/dtypes/leaf count/loss/token/mask/elapsed evidence and pair provider/callback/checkpoint N.
6. Enforce every CLI/config pin including train, LoRA rank/scale/dropout/key set and exact rendered catalog commands.
7. Harden canonical digest: dtype, byte size, contiguous offsets, no gaps/trailing bytes.
8. Replace shallow tests with architecture T1-T18 terminal-path/mutation-sensitive tests executing run_phase/_execute_training/main with fakes and injected failures. Tests must kill mutations named by Reviewer.
9. Track architecture.md, requirements, task/coder notes/tests. Remove every `.cmux-status` path from index. Reconcile docs to delivered behavior only.
10. Run focused tests, exact Epic 14 suite, py_compile, protected hashes/Path A exact 365 digest/vendor/smoke identity, diff check, staged-file tracking. Write coder-notes-r2.md and marker.

Do not weaken requirements to fit implementation.