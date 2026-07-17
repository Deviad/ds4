# Story 14.5a — Coder r6 central-source closure

## Result

Implemented direct TDD closure for all r5 blocking findings.

- `_a2_gradient_schema()` rejects duplicate gradient paths before keyed shape/dtype map comparison.
- Added coordinated duplicate schema/list/map mutation coverage.
- Added central `attempt2_launch_identity()` in `scripts/ds4_segmented_pilot.py` for exact interpreter, model, dataset, config, and pilot-script paths.
- `canonical_attempt2_command()`, launch preparation, and canonical effective-pin validation consume that central identity.
- `finetune_ds4.py` imports the central identity and renders the same interpreter/model/data/config values for launch-check and training command positions. Output, log, resume, checkpoint, config, report, and marker paths remain namespace-derived bindings; no independent A2 path reconstruction remains in the generated wrapper.
- Added mutation coverage proving central launch identity changes generated command bindings consistently.
- Added A2/B2 namespace mutation coverage through dependency admission, artifact validation, phase specs, and catalog rendering, including B2 config, resume source, start checkpoint, step-1 checkpoint, and final checkpoint.

## TDD evidence

- Red first: new duplicate-path admission test and central launch-identity test failed against r5.
- Green after implementation: targeted closure tests passed.

## Verification

- Canonical six-file suite: `427 passed, 3 skipped, 1 warning, 2 subtests passed in 21.08s`.
- Targeted closure tests: `21 passed`.
- `py_compile`: PASS.
- Generated Phase B catalog wrapper `bash -n`: PASS.
- `git diff --check`: PASS.
- Protected direct HEAD byte comparison: `PROTECTED_FILES=31 CHANGED=0`.
- All six verdict-contributing test files tracked by `git ls-files`.

## Boundary

No real model, dataset, training, inference, cleanup, commit, push, or marker staging performed. Canonical docs and real Phase A2/B2 authorization remain pending independent Reviewer PASS, Test Manager GREEN, and fresh operator authorization.
