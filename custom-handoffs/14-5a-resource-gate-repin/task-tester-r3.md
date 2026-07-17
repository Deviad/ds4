# Story 14.5a Test Manager r3 — canonical A2 admission closure

Read in full before acting:
- `/Users/spotted/projects/ds4-finetuning/AGENTS.md`
- `docs/technical-spec.md`
- `custom-handoffs/14-5a-resource-gate-repin/requirements.md`
- `custom-handoffs/14-5a-resource-gate-repin/architecture.md`
- `custom-handoffs/14-5a-resource-gate-repin/task-coder-r3.md`
- `custom-handoffs/14-5a-resource-gate-repin/coder-notes-r3.md`

BEGIN NOW. Independent verification only. Do not edit production code or tests. Write only:
`custom-handoffs/14-5a-resource-gate-repin/test-report-r3.md`
and `.cmux-status/test-manager.done`; finish with terminal JSON `{"status":"ok","role":"Test Manager"}` on success or an error JSON on STOP.

Run and record independently:
- focused pilot resource/attempt-2/canonical/mutation tests;
- full `tests/test_ds4_segmented_pilot.py`;
- exact six-file synthetic suite:
  `tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py`;
- `git diff --check`;
- `git ls-files` for every test contributing to the verdict;
- direct protected-source hash checks (Story 14.3 smoke/provider, vendor, C/Metal/CLI/server/Metal MoE) and no-real-execution evidence.

Assess exact canonical A2 admission acceptance: complete synthetic report, every A2/B2 checkpoint/config path, report/marker/log bindings, coordinated mutation rejection, resource observer failure/skip accounting, and attempt-1 immutability. GREEN only with reproducible tracked evidence. No real model, dataset, adapter, training, inference, cleanup, commit, or push.
