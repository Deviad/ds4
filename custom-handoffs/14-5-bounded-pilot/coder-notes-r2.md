# Story 14.5 coder notes — r2 blocker closure

- No real model, dataset, adapter, training, inference, CUDA, distributed, or smoke execution.
- No commit or push.

## Closed blockers

- Wired the pinned vendor trainer seam: `load_dataset(args, tokenizer)`, freeze before LoRA, exact supported `TrainingArgs`, fresh Adam after Phase B load/proof, positional `train(model, optimizer, train_dataset, val_dataset, args=...)`, provider factory, and one-dict train/validation callbacks.
- Added real Phase A/B artifact progression and resume-start canonical equality/change proofs. Phase B requires the durable Phase A report, marker, report hash, contract digest, checkpoint file hash, and canonical digest bindings.
- Enforced release-lock → durable report → marker ordering, final report aggregation, failure evidence, one-attempt gates, cancellable watchdog, and timeout wall-clock accounting.
- Added fail-closed preflight identity/resource gates for interpreter, source hashes, vendor gitlink/HEAD, resolved MLX source, config/topology, model/dataset/provenance manifests, competing RSS, memory headroom, and disk free space.
- Replaced truthiness gradient checks with finite/schema/path/shape/dtype/leaf-count evidence and token-mask equality; paired provider N, callback N, and checkpoint N.
- Required all effective training/path pins; required exact LoRA rank/scale/dropout/key set; corrected catalog commands to the pinned absolute paths and valid quoted `bash -lc` syntax.
- Hardened canonical safetensors digest against unsupported dtypes, byte-size mismatch, gaps, overlaps, and trailing data.
- Expanded synthetic tests to execute `_execute_training` and `run_phase` with fakes, cover digest/layout/schema/callback/dependency/progression/resource-contract surfaces, and preserve the frozen smoke hash.
- Removed `.cmux-status` paths from the index; architecture/requirements/test evidence remain tracked/staged for the supervisor.
- Reconciled backlog status to synthetic implementation complete with independent review/test gates still required.

## Verification

- `python3 -m py_compile scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py tests/test_ds4_segmented_pilot.py`
- Focused Story 14.5 test: `21 passed, 1 warning`.
- Exact Epic 14 synthetic suite: `267 passed, 3 skipped, 1 warning, 2 subtests passed`.
- Path A manifest: `365` files; digest `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`.
- Catalog dry-run rendered exact Phase A/B paths, valid double-quoted paths inside `bash -lc`, `set -o pipefail`, unbuffered Python, and `tee`; no pilot command executed.
- `python3 -m py_compile scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py tests/test_ds4_segmented_pilot.py` passed.
- `git diff --check` and `git diff --cached --check` clean.
- Frozen Story 14.3 smoke hash `ec17950d...`, provider hash `20572191...`, and vendor `80fab4e...` remain exact; no vendor/Path A/runtime files changed.

Independent Reviewer PASS and Test Manager GREEN remain outstanding; real execution remains blocked.
