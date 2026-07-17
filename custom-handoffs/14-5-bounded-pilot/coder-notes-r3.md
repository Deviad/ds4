# Story 14.5 coder notes — r3 executable-path closure

## Scope

- Read `custom-handoffs/standby/review.md` (r2 BLOCKED) and `task-coder-r3.md`.
- Direct TDD implementation only. No delegation, commit, push, real model/dataset/adapter/training/inference/CUDA/distributed/smoke execution.

## Closed review findings

- `_StepObservingProvider` now accepts only real `((loss, token_count), gradients)` output.
- Provider validates scalar MLX `float32` loss, scalar MLX `int32` token count, exact nonempty model trainable schema, MLX finiteness, and exact mask-token count bounded by tensor length.
- Pinned `mlx.utils.tree_flatten` is used before every pilot safetensors save and for schema observation. Nested MLX save probe passes.
- Parser exposes `test` and `hf_dataset` namespace fields required by pinned `load_dataset(args, tokenizer)`.
- Each phase applies `numpy.random.seed(0)` then `mx.random.seed(0)` before model loading/training; call order is tested.
- Immutable identity is separated from dynamic resource observations. Phase B compares immutable identity before model load/training. Contract digest binds immutable assets, topology/runtime identity, effective pins, and command.
- Runtime gates now require MLX `0.31.2`, resolved vendor `mlx_lm`, exact outer/inner vendor identity and clean inner checkout, provenance-bound train/valid/test hashes, checkpoint schema/cardinality invariance, and exact validation callback order/count.
- Reports include commands, lock lifecycle, watchdog cancellation, and non-claims. Phase/final markers include output, timestamp, exit code, and contract digest.
- Backlog/architecture status now states r3 implementation remains under independent review; real execution remains blocked.

## Verification

- Focused pilot suite: `28 passed, 1 warning`.
- Exact Epic 14 synthetic/protected suite: `274 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `py_compile`: PASS.
- `git diff --check` and `git diff --cached --check`: PASS.
- Frozen Story 14.3 smoke SHA: `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8` PASS.
- Frozen segmented provider SHA: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` PASS.
- Vendor inner HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`; vendor clean PASS.
- Cited test file tracked PASS; no `.cmux-status` marker staged.

Real pilot execution remains unauthorized and blocked pending independent Reviewer PASS, Test Manager GREEN, and separate operator authorization.
