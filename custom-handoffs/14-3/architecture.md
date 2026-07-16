# Slice 14-3 — Architecture: DS4 Segmented Provider Activation + Bounded Real-Smoke Contract

## Verdict

**GO.** A standalone thin-wiring script plus one explicitly-requested `MLX_STEPS` entry is feasible without editing production code, the staged trainer fork, the provider source, the default command catalog, or any C/Metal/CUDA path. All open BA questions resolve to GO-compatible answers. Real smoke remains a Phase-2 gate controlled by operator authorization and is described here as a fail-closed pinned-command contract. No real asset is read, listed, stat'd, or opened during this Architect stage.

## Pre-flight links verified (read-only)

Read and adjudicated:

- `AGENTS.md` (project instructions, caveman defaults, role-pipeline gates).
- `docs/architecture.md` (scaffolding, fine-tuning toolchain subsection, data-marker policy, canonical-docs policy).
- `docs/technical-spec.md` (build/run/test/Python environments, §7 fine-tuning pipeline, §10.11 opaque packed-FP4 primitive gate, §10.12 Story 13.3b-5g first-backward OOM re-entry).
- `docs/backlog.md` Story 14.3 provisional section (user story and acceptance criteria present).
- `custom-handoffs/14-3/requirements.md` (BA's full requirements with R14.3-* sections).
- `agent-output/cmux-14-2a/{architecture,implementation,review,test-report}.md` (Story 14.2a closure: staged `trainer.py` worktree/index SHA-256 `42e5ee2d…`, `test_tuner_trainer.py` SHA-256 `275d6f3a…`, masked default-path hash `b53bdc54…`, dual single-compile `update_step` topology, direct-host `loss_and_grad(model, *batch)` call).
- `custom-handoffs/14-2-r11/review.md` (Reviewer r11 PASS: functional hash `75786d9e…` over the three authorized paths `segmented_loss_and_grad.py` blob `20572191…`, `test_ds4_segmented_loss_and_grad.py` blob `618a0f22…`, `test_mlx_lm_source.py` blob `dec2c2b5…`).
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` (the provider factory, pinned SHA-256 above).
- `vendor/mlx-lm/mlx_lm/tuner/trainer.py` (the staged `train()` signature, `loss_and_grad=` branch, provider output validator, finite gate).
- `vendor/mlx-lm/mlx_lm/tuner/lora.py` (the `build_parser`, `run()`, `train_model()` skeleton, default config merging, training args assembly).
- `vendor/mlx-lm/mlx_lm/tuner/datasets.py` (the `load_dataset`, `CompletionsDataset`, mask-prompt offset semantics).
- `scripts/finetune_ds4.py` (`MLX_STEPS` tuple, `command_catalog` dict, `LockFile` class, argparse plumbing).

## Verified runtime facts

1. **`train()` public signature accepts `loss_and_grad`** as the last positional-or-keyword parameter at the staged fork:

   ```python
   def train(
       model,
       optimizer,
       train_dataset,
       val_dataset=None,
       args: TrainingArgs = TrainingArgs(),
       loss: callable = default_loss,
       iterate_batches: callable = iterate_batches,
       training_callback: TrainingCallback = None,
       loss_and_grad=None,
   ):
   ```

   The custom branch (under `else:` for `if loss_and_grad is None`) calls `loss_and_grad(model, *batch)` exactly once per attempted microbatch inside the existing random-snapshot rollback `try`. Returns `((loss, token_count), gradients)`.

2. **Provider contract matches the trainer's custom branch**:

   - `segmented_loss_and_grad.py` returns `provider(model, batch, lengths) -> ((loss, token_count), gradients)` where loss is `mx.float32` scalar, token_count is `mx.int32` scalar, gradients tree has same shape/key order as `model.trainable_parameters()`.
   - Provider requires `getattr(model, "model_type", None) == "deepseek_v4_nn"` and the DS4 topology (`pipeline_layers`, `hc_head`, `norm`, `lm_head`, `embed_tokens`, `hc_mult`, `hidden_size`). When the model is loaded by `mlx_lm.load` against `model-4bit` (which is produced only via the DS4 fork's `deepseek_v4.py` model class), the attribute is set.
   - Provider rejects `segment_size` outside `{1, 2, 3, 4}` via `_validate_segment_size`.
   - Provider already catches exception and restores `model.state` and `random.state` through its `try/finally` — exceptions propagate to the trainer's random rollback boundary which restores the entry random snapshot.

3. **Existing trainer validators** (Story 14.2a, unchanged in this slice) already enforce on every custom-path call:

   - `_validate_provider_output` rejects non-tuple result, non-tuple metadata, non-array loss/token_count, wrong gradient container types, wrong key order, wrong shape, wrong dtype — exact `TypeError`/`ValueError` classes.
   - `_provider_finite_gate` runs `mx.all(mx.isfinite(...))` on loss and every gradient leaf and materializes everything before phase 2.
   - `_provider_gradient_schema` is captured once per microbatch from `model.trainable_parameters()` at the same identity the provider is asked to mutate.

4. **`lora.py`'s `train_model()` does NOT accept `loss_and_grad`** — it calls `train(model=, args=, optimizer=, train_dataset=, val_dataset=, training_callback=)` and ends. Therefore the script MUST replicate the `train_model()` body (freeze, linear_to_lora_layers, args+optimizer, save_config, `train()` call) and add `loss_and_grad=provider`. This is thin wiring, not a training framework fork.

5. **Adapter save in the trainer's `train()`**:

   - `adapter_config.json` is written by `lora.train_model` via `save_config(vars(args), adapter_path / "adapter_config.json")` BEFORE calling `train()`. If the smoke script bypasses `train_model`, it must perform this save itself (one-line import from `mlx_lm.utils.save_config`).
   - `trains` periodic save fires only when `it % args.steps_per_save == 0 and rank == 0`. For `iters=1` with default `steps_per_save=100`, periodic save does NOT occur.
   - The final unconditional save at `train()` end (`if rank == 0: mx.save_safetensors(args.adapter_file, …)`) DOES fire for `iters=1`. So `adapters.safetensors` will always be present after one iteration.
   - The pinned smoke therefore naturally satisfies BA R14.3-4 "adapter output must contain `adapters.safetensors` and `adapter_config.json`" because the script saves the config and `train()` saves the safetensors.

6. **`CompletionsDataset` mask-prompt semantics** (datasets.py L88–129):

   - `mask_prompt=True` causes the dataset to return `(tokens, offset)` where `offset = len(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, …))` — the offset is the prompt prefix length.
   - `iterate_batches` packs the offset into the `lengths` array second tuple element (`(offsets, lengths)` → column 0 is offset, column 1 is total length). Both `default_loss` and the DS4 provider build their `mask = mx.logical_and(steps >= lengths[:,0:1], steps <= lengths[:,1:])` from this same layout. Therefore `mask-prompt` is dataset-level and the provider consumes it transparently.

7. **`LockFile` class** lives at `scripts/finetune_ds4.py` L488–508 (O_CREAT|O_EXCL, fail-fast). The smoke script must reuse this style and add a bounded wait (BA R14.3-4 mandates a 60-second wait). Implemented inline as a tight retry with a 2-second poll interval.

8. **No real asset was accessed during this stage**: I did not read, list, stat, open, or import any path under `/Volumes/Data NVME/…`, model weights, checkpoint, tokenizer, dataset, adapter, or `mlx_lm.generate`. All probes were static source inspection of tracked files. MLX has not been imported and no Metal device has been initialized.

## Design decisions and rationale

### D1. Standalone script vs. inline heredoc

**Chosen: Standalone `scripts/ds4_segmented_smoke.py`.** Options adjudicated:

- **A) Standalone script** — small tracked file, testable in isolation with monkey-patches and synthetic args; mirrors the project convention (`scripts/` helpers, `torch_one_step_lora.py` style). Chosen.
- **B) Inline heredoc in `command_catalog["ds4-segmented-smoke"]`** — ugly multi-line Python string embedded in `finetune_ds4.py`, hard to test, hard to lint, no AST parse. Rejected.
- **C) Edit `lora.train_model` to accept and forward `loss_and_grad`** — production fork edit; violates vendor-edit freeze, trainers-recode rule, and "no permanent semantic variant" rule. Rejected.

### D2. Reuse `lora.build_parser()` vs. hand-rolled minimal argparse

**Chosen: Reuse `lora.build_parser()` + add `--segment-size`.** Rationale:

- `lora.build_parser()` already exposes every smoke argument the BA pinned (--model, --data, --adapter-path, --config, --iters, --batch-size, --learning-rate, --max-seq-length, --mask-prompt, --grad-checkpoint, plus the optimizer, val-batches, save-every, num-layers, fine-tune-type, lora_parameters used by `train_model`).
- Reusing it keeps the script thin and paving the same YAML+CLI convention that the canonical smoke command uses.
- Bypassing `lora.main()` and `lora.run()` means the script NEVER invokes `train_model` (which would lose `loss_and_grad`); it instead mirrors `train_model`'s body with our override.
- `--segment-size` is the only argument added; it lives in this script and never leaks up to the fork.

### D3. Adapter-config save responsibility

**Chosen: Script saves `adapter_config.json` directly.** Since the script bypasses `train_model`, it must perform the one-line `save_config(vars(args), adapter_path / "adapter_config.json")` itself. This satisfies BA R14.3-4 adapter-dir artifact list (`adapters.safetensors` from `train()` final save + `adapter_config.json` from the script).

### D4. ADR / canonical docs update

**Chosen: One paragraph in `docs/architecture.md` fine-tuning toolchain subsection; NO new ADR.** Rationale mirrors Story 14.2a's adjudication:

1. The change introduces no new public protocol — `train()` signature and provider protocol are unchanged.
2. No new backend/source regime; default `mlx_lm.lora` CLI is byte-unchanged in the catalog.
3. The activation is a thin wiring script + one MLX step entry, not a durable cross-subsystem decision.
4. ADR 0028 (opaque packed-FP4 Metal primitive) and ADR 0029 (MLX-LM source selection) remain authoritative; activation doesn't extend or invert either.

Docs change: add one paragraph in `docs/architecture.md` § Fine-tuning toolchain that pins the new step, the one-shot bounded contract, and the explicit non-default nature. Reviewer must enforce no stale `docs/architecture.md` drift.

### D5. BA open questions

- **Q1 — standalone vs inline**: A (standalone). See D1.
- **Q2 — adapter save**: Allow `train()` to save naturally at `iters=1` (final save fires; periodic save does NOT fire because `1 % 100 ≠ 0`). Script saves `adapter_config.json` itself. Adapter dir becomes evidence artifact, never reused.
- **Q3 — `--segment-size`**: Explicit CLI arg, default 1, validated via `make_ds4_segmented_loss_and_grad` factory. Bounded smoke pins `--segment-size 1`. One bounded contract explicit prevents ambiguity later operator wants re-pin different size in separate authorization.
- **Q4 — ADR**: No ADR; one paragraph in `docs/architecture.md`. See D4.

## File scope (Phase 1)

| Path | Action | Justification |
|---|---|---|
| `scripts/ds4_segmented_smoke.py` | NEW, tracked | The activation entry point. |
| `scripts/finetune_ds4.py` | EDIT, add two insertions | (a) Add `"ds4-segmented-smoke"` tuple entry to `MLX_STEPS`; (b) add `command_catalog["ds4-segmented-smoke"]` entry. Existing entries byte-unchanged. |
| `tests/test_ds4_segmented_smoke.py` | NEW, tracked | TDD Red/Green synthetic suite: provider wiring, default unchanged, mutations, preflight, abort conditions. |
| `docs/architecture.md` | EDIT, one paragraph | Pin activation boundary under Fine-tuning toolchain subsection. |
| `docs/backlog.md` | Already updated by BA; Architect verifies presence | No further edits. |

Forbidden Phase-1 file changes:
- `vendor/mlx-lm/` (any inner file; inner HEAD `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` pin and outer gitlink frozen).
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` (provider source pin `20572191…`).
- `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn.py` and other MLX model sources.
- `tests/test_ds4_segmented_loss_and_grad.py` (pin `618a0f22…`).
- `tests/test_mlx_lm_source.py` (pin `dec2c2b5…`).
- Any C/ObjC/Metal/CUDA/ROCm/distributed/SSD/disk-cache file.
- YAML/JSON config templates or `python-envs/*/pyproject.toml`.
- ADR directory (no new ADR this slice).

## Phase 1 — `scripts/ds4_segmented_smoke.py` contract

### Module layout

```python
"""DS4 segmented-training smoke activation entry point.

Thin wiring only: loads model + dataset via MLX-LM public APIs,
applies LoRA per the YAML config, constructs the DS4 segmented loss
provider, and calls ``train(..., loss_and_grad=provider)`` exactly
once. The trainer retains all optimizer, accumulation, save,
callback, and UI ownership. Default MLX-LM training path is
unchanged.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import signal
import sys
import threading
import time
import types

import mlx.optimizers as optim

from mlx_lm import load
from mlx_lm.tuner.callbacks import get_reporting_callbacks
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import save_config
from ..lora import CONFIG_DEFAULTS, build_parser  # type: ignore  (fork import)
from ds4_ft_mlx.segmented_loss_and_grad import make_ds4_segmented_loss_and_grad
```

**Note on the `..lora` import path**: We import `lora` from `mlx_lm.tuner.lora` (not `mlx_lm.lora`) to reuse `build_parser` and `CONFIG_DEFAULTS`. The pinned fork module at `vendor/mlx-lm/mlx_lm/tuner/lora.py` is structured under `mlx_lm.tuner`. (During Coder Phase 1 the exact relative import must be verified by the existing precursor pattern.)

### Functions (exact contract)

```python
def build_smoke_parser() -> argparse.ArgumentParser:
    parser = build_parser()              # fork's existing parser
    parser.add_argument(
        "--segment-size",
        type=int,
        default=1,
        help="DS4 segmented provider segment size (1..4). Bounded smoke pins 1."
    )
    return parser

def merge_config_and_defaults(args: argparse.Namespace) -> argparse.Namespace:
    # Reuse lora.main()'s config+defaults merge; do not call main() itself.
    ...
    return args

def acquire_smoke_lock(lock_path: pathlib.Path, timeout_s: int = 60, poll_s: float = 2.0) -> bool:
    # O_CREAT|O_EXCL retry loop with owner PID; Academy rule: fail-closed.
    ...

def check_preflight(args, model_size_bytes: int | None = None) -> None:
    # Lock + memory + disk checks; any failure raises SystemExit(3) with a clear marker.
    ...

def _run(args: argparse.Namespace) -> None:
    # Mirrors train_model body, then adds provider + loss_and_grad to train().
    ...

def main(argv: list[str] | None = None) -> int:
    # Parse, merge, preflight, set 600s timeout watchdog, call _run, write marker files, return exit code.
    ...

if __name__ == "__main__":
    raise SystemExit(main())
```

### `_run(args)` exact event order

```text
acquire_smoke_lock (O_EXCL; wait ≤ 60s; abort on failure)
→ model, tokenizer = load(args.model, trust_remote_code=True)
→ train_set, valid_set, test_set = load_dataset(args, tokenizer)
→ adapter_path = Path(args.adapter_path)
→ adapter_path.mkdir(parents=True, exist_ok=True)
→ save_config(vars(args), adapter_path / "adapter_config.json")
→ model.freeze()
→ linear_to_lora_layers(model, args.num_layers, args.lora_parameters, use_dora=False)
→ if args.resume_adapter_file: model.load_weights(args.resume_adapter_file, strict=False)
→ training_args = TrainingArgs(
      batch_size=args.batch_size,
      iters=args.iters,
      val_batches=args.val_batches,
      steps_per_report=args.steps_per_report,
      steps_per_eval=args.steps_per_eval,
      steps_per_save=args.save_every,
      adapter_file=adapter_path / "adapters.safetensors",
      max_seq_length=args.max_seq_length,
      grad_checkpoint=args.grad_checkpoint,
      grad_accumulation_steps=args.grad_accumulation_steps,
  )
→ optimizer = optim.Adam(learning_rate=args.learning_rate, **optimizer_config_for("adam"))
→ provider = make_ds4_segmented_loss_and_grad(segment_size=args.segment_size)
→ train(
      model=model,
      optimizer=optimizer,
      train_dataset=CacheDataset(train_set),
      val_dataset=CacheDataset(valid_set),
      args=training_args,
      loss_and_grad=provider,
  )
→ write OK marker
```

**Provider call count**: exactly once per attempted microbatch ensured by the trainer (Story 14.2a). For `iters=1, batch_size=1`, the trainer attempts exactly one microbatch, resulting in exactly one provider body entry.

### Argument pin (bounded smoke)

| Arg | Value | Notes |
|---|---|---|
| `--model` | `$MLX_WORK/model-4bit` | Loaded via `mlx_lm.load`; `model_type` resolved as `deepseek_v4_nn` by the DS4 NN model module. |
| `--data` | `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096` | Contains `train.jsonl`, `valid.jsonl`, `test.jsonl` with `prompt` + `completion` fields. |
| `--adapter-path` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke` | Must NOT exist before smoke; must contain `adapters.safetensors` + `adapter_config.json` after smoke. |
| `--config` | `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` | DS4-safe LoRA target allowlist; `num_layers`, `fine_tune_type`, `optimizer`, `optimizer_config`, `lora_parameters`, `seed`, `lr_schedule` fields consumed. |
| `--iters` | 1 | Exactly one attempted microbatch. |
| `--batch-size` | 1 | One real row per batch. |
| `--learning-rate` | 1e-5 | Conservative LR. |
| `--max-seq-length` | 4096 | Token-level boundary for the single bounded smoke. |
| `--mask-prompt` | True | Dataset offsets prompt tokens; provider + default loss both consume via `lengths[:,0:1]`. |
| `--grad-checkpoint` | True | Trainer applies `grad_checkpoint(model.layers[0])`; class-level patch covers all 43 decoder layers. |
| `--segment-size` | 1 | Provider `segment_size=1`; segment-bounded command-buffer lifetime. |

### Pinned Phase 2 smoke command (catalog entry)

```python
"ds4-segmented-smoke": [
    f"cd {q(mlx_work)} && unset SSLKEYLOGFILE && . {q(mlx_work / '.venv/bin/activate')} && "
    f"PYTHONPATH={q(project_root / 'python-envs' / 'mlx' / 'src')}:{q(project_root / 'vendor' / 'mlx-lm')} "
    f"python {q(project_root / 'scripts' / 'ds4_segmented_smoke.py')} "
    f"--model {q(mlx_work / 'model-4bit')} "
    f"--data {q(split_dir)} "
    f"--adapter-path {q(mlx_work / 'adapters-segmented-smoke')} "
    f"--config {q(lora_config)} "
    f"--iters 1 --batch-size 1 --learning-rate 1e-5 "
    f"--max-seq-length 4096 --mask-prompt --grad-checkpoint "
    f"--segment-size 1"
],
```

`split_dir` and `lora_config` come from existing `command_catalog` local variables (same variables the default `smoke-train` uses). This guarantees the dataset path mirrors the BA canonical `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096` through the same `--data {q(split_dir)}` pattern as `smoke-train`.

### finetune_ds4.py MLX_STEPS insertion

Add `"ds4-segmented-smoke"` as a new element in the `MLX_STEPS` tuple. Position: immediately after `"ds4-smoke"` (last element) so default steps sequence remains untouched. Existing elements are byte-identical.

## Phase 1 — TDD Red/Green test design

Tests live in `tests/test_ds4_segmented_smoke.py`. They use synthetic tiny fixtures only — no real model, no `/Volumes` path, no MLX dependency at import time, no Metal. They monkey-patch MLX-LM and the provider factory with stubs/recorders.

### T1 — `MLX_STEPS` and command catalog invariants

```python
def test_mlx_steps_contains_ds4_segmented_smoke_and_existing_steps():
    # import scripts.finetune_ds4 (or read source AST) and assert tuple.
```

```python
def test_existing_smoke_train_command_byte_unchanged():
    # The smoke-train command string in command_catalog is EXACTLY the same as
    # the repo baseline (computed by git show HEAD of scripts/finetune_ds4.py
    # or by a static snapshot hash). Pin via known substring patterns.
```

**Mutation kills**: Inserting `"ds4-segmented-smoke"` in the wrong position (e.g., before `smoke-train`) → fail. Deleting `smoke-train` from `MLX_STEPS` → fail. Changing the `smoke-train` string by one char → fail.

### T2 — Provider factory + train call exactly once

```python
def test_run_constructs_provider_once_and_calls_train_with_loss_and_grad(monkeypatch):
    # Monkeypatch:
    #   mlx_lm.load -> (StubModel, None)
    #   mlx_lm.tuner.datasets.load_dataset -> ([tiny], [], [])
    #   mlx_lm.tuner.utils.linear_to_lora_layers -> noop
    #   mlx_lm.tuner.utils.print_trainable_parameters -> noop
    #   mlx_lm.tuner.trainer.train -> recorder that asserts `loss_and_grad is not None`
    #   ds4_ft_mlx.segmented_loss_and_grad.make_ds4_segmented_loss_and_grad -> recorder
    #   mlx_lm.utils.save_config -> noop
    # Build args with segment-size=1, iters=1, batch-size=1.
    # Run _run(args).
    # Assert:
    #   provider_factory called exactly once with segment_size=1
    #   train called exactly once with loss_and_grad=<return of factory>
    #   train was not called without loss_and_grad kwarg
```

**Marker**: `'provider must be constructed once and passed to train()'`.

### T3 — Default fallback killed

```python
def test_run_must_not_omit_loss_and_grad(monkeypatch, tmp_path):
    # Copy scripts/ds4_segmented_smoke.py to tmp_path with one mutation:
    #   remove `loss_and_grad=provider` from the train() call.
    # Import the mutated module, run with same stubs.
    # Assert train was called with loss_and_grad=None.
    # Assert provider_factory was still called (this is the wiring that should have used it).
    # The test FAILS because the rider contract requires loss_and_grad=provider.
    # After T3 is Green (with correct source), rerunning this mutation must FAIL.
```

**Marker**: `'loss_and_grad must be passed to train()'`.

### T4 — Mutations in temp-overlap

Use the same `cp -R` + `str.replace` pattern as Story 14.2a; each recipe uses a unique `MUTATION` env var. Mutations:

- `no-loss-and-grad` — remove `loss_and_grad=provider` from `train()` call. T2 fails with the provider-not-passed marker.
- `no-provider-construction` — remove `provider = make_ds4_segmented_loss_and_grad(...)`. T2 fails because `loss_and_grad` is undefined.
- `wrong-segment-size` — replace `segment_size=args.segment_size` with `segment_size=2`. T2 fails because factory was called with `segment_size=2`; marker `'segment size must equal --segment-size'`.
- `cached-provider` — replace the factory call with a memoized singleton. T2 fails because the recorded `segment_size` is not always 1.
- `omitted-save-config` — remove `save_config` call. Separate test `test_adapter_config_saved_before_train` fails with `'adapter_config.json must be written'`.
- `omitted-lock` — comment out `acquire_smoke_lock` call. Preflight test fails with `'smoke lock must be acquired'`.
- `lock-timeout-bypass` — change `timeout_s=60` to `timeout_s=0`. Preflight test fails with `'lock timeout must be 60s, returning False immediately'`.

### T5 — Preflight abort ordering

```python
def test_preflight_runs_before_train(monkeypatch):
    # Stubs that record call order: lock_attempt, mem_check, disk_check, train.
    # Force mem_check to return False.
    # Run main().
    # Assert: lock_attempt called, mem_check called, train NOT called,
    # exit code is nonzero, fail marker written.
```

### T6 — Timeout watchdog

```python
def test_timeout_watchdog_aborts_after_600s(monkeypatch):
    # Stub train() to sleep(N) seconds and return after sleep.
    # Stub signal.alarm and threading.Timer to fire the handler immediately.
    # Assert main() raises TimeoutError or exits with exit code 2.
```

### T7 — OOM abort

```python
def test_out_of_memory_aborts_smoke(monkeypatch):
    # Stub train() to raise MemoryError or RuntimeError("Metal command-buffer out-of-memory").
    # Assert main() returns exit code 3, fail marker written, OK marker not written,
    # adapter directory left as-is (no cleanup).
```

### T8 — No real-asset access during Phase 1

```python
def test_smoke_script_imports_without_path_access(monkeypatch):
    # Monkeypatch pathlib.Path.stat and os.listdir to raise if any path
    # starting with /Volumes/Data NVME is queried.
    # Import the module. Assert no exception from the import itself.
    # Call main() with stubs that don't touch real paths.
```

### T9 — `--segment-size` validation

```python
def test_segment_size_0_rejected_by_argparse():
    # parse_args(['--segment-size', '0']) raises SystemExit.
```

```python
def test_segment_size_choices():
    # Accept 1,2,3,4; reject 0, 5, -1, 1.5.
```

### T10 — Default command catalog byte-unchanged

Run the full existing regression suite (`tests/test_finetune_ds4.py`, `tests/test_mlx_lm_source.py`) plus the new test file. Existing tests that probe `command_catalog` must remain green unchanged.

## Phase 2 — Pinned real-smoke command (operator-authorized)

**Phase 2 cannot begin until ALL of the following are met**:
1. Coder implementation complete and `git ls-files` lists every contributing file.
2. Reviewer returns PASS on exact staged paths.
3. Tester returns GREEN on the tracked test suite under `python-envs/mlx/.venv`.
4. Supervisor presents the exact pinned command below and BA's limits list.
5. Operator gives explicit authorization per slice.

### Pinned command (Phase 2)

```bash
cd "/Volumes/Data NVME/mlx-ft/ds4" \
  && unset SSLKEYLOGFILE \
  && . "/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/activate" \
  && PYTHONPATH="/Users/spotted/projects/ds4-finetuning/python-envs/mlx/src:/Users/spotted/projects/ds4-finetuning/vendor/mlx-lm" \
  python /Users/spotted/projects/ds4-finetuning/scripts/ds4_segmented_smoke.py \
    --model "/Volumes/Data NVME/mlx-ft/ds4/model-4bit" \
    --data "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096" \
    --adapter-path "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke" \
    --config "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json" \
    --iters 1 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 \
    --mask-prompt --grad-checkpoint --segment-size 1 \
  > /Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-3/smoke-log.txt 2>&1
```

### Preflight algorithm (run inside `scripts/ds4_segmented_smoke.py`)

```text
preflight(args, exit_code_if_fail=3):
  1. try_smoke_lock:
     lock_path = Path(args.mlx_work) / ".ds4-segmented-smoke.lock"
     attempt O_CREAT|O_EXCL with retry:
       - poll interval 2 seconds
       - total timeout 60 seconds
       - on failure: write fail marker, raise SystemExit(3)
       - record owner PID in lock file
  2. memory check:
     if not args.skip_preflight:
       model_dir_size = sum(f.stat().st_size for f in Path(args.model).rglob('*') if f.is_file())
       available = psutil.virtual_memory().available
       required = model_dir_size + SMOKE_MEMORY_HEADROOM
       if available < required:
         write fail marker, raise SystemExit(3) with diagnostic
  3. disk check:
     adapter_parent = Path(args.adapter_path).parent
     free = shutil.disk_usage(adapter_parent).free
     if free < SMOKE_DISK_MIN_FREE:
       write fail marker, raise SystemExit(3)
  4. adapter absence check (BA R14.3-3):
     if Path(args.adapter_path).exists():
       write fail marker, raise SystemExit(3) with 'adapter path must not exist before smoke'
  5. fail marker path pin: Path(args.mlx_work) / ".ds4-segmented-smoke-fail"
     ok marker path pin:   Path(args.mlx_work) / ".ds4-segmented-smoke-ok"
```

### Architect-pinned preflight constants

| Constant | Value | Rationale |
|---|---|---|
| `SMOKE_TIMEOUT_SECONDS` | 600 | Matches BA R14.3-4. Both `signal.alarm(600)` and a watchdog thread doubling the signal fallback are used. |
| `SMOKE_MEMORY_HEADROOM` | 32 GiB (`32 * 1024**3`) | Model dir size is the dominant base; the 32 GiB headroom covers LoRA graph, optimizer state, OS, MLX allocator cache, and one-segment activation transients. If peak observation exceeds this, the script aborts via Metal OOM pathways inside `train()` (the trainer does not wrap OOM explicitly; non-MemoryError exceptions propagate up to `main()`). |
| `SMOKE_DISK_MIN_FREE` | 1 GiB | Covers adapter checkpoint (LoRA rank on 43 layers is bounded by rank × num_layers × hidden_size × dtype, well under 1 GB) and a safety margin. BA R14.3-4 pins 1 GiB. |
| `SMOKE_LOCK_TIMEOUT_S` | 60 | BA R14.3-4 pins. |
| `SMOKE_LOCK_POLL_S` | 2 | Tight retry interval; 30 attempts before abort. |

### Post-train validation (Phase 2)

Inside `main()` after `train()` returns successfully:

```text
1. Assert Path(args.adapter_path) / "adapters.safetensors" exists.
2. Assert Path(args.adapter_path) / "adapter_config.json" exists.
3. Build smoke-report.json to agent-output/cmux-14-3/:
   {
     "loss": <float42 from train() deferred; schema verification only>,
     "token_count": <likewise>,
     "gradient_keys": <count from provider>,
     "finite": true,
     "provider_call_count": 1,
     "wall_clock_seconds": <elapsed>
   }
4. Write ok marker: Path(args.mlx_work) / ".ds4-segmented-smoke-ok".
5. Exit 0.
```

The trainer's `_provider_finite_gate` already materializes loss and gradient leaves BEFORE `update_step` — meaning the validation and finite check happen inside `train()` itself. The script does not need to re-validate the loss/gradients at the host boundary. The adapter directory existence check after `train()` is the only additional post-train assertion.

### Abort conditions (any → immediate termination, no retry)

1. OOM/MemoryError raised by MLX or OS.
2. Nonfinite loss or any gradient leaf (raised by `_provider_finite_gate`).
3. Gradient schema mismatch (raised by `_validate_provider_output`).
4. Token count ≤ 0 or mismatch with `default_loss` mask sum (caught by validation).
5. Wall-clock > 600s (signal/watchdog).
6. Instance lock acquisition failure (after 60s wait).
7. Model loading failure (corrupt weights, missing config, tokenizer error).
8. Dataset loading failure (missing file, malformed row, empty dataset).
9. Provider construction failure (segment_size validation from `_validate_segment_size`).
10. Adapter output directory already exists at smoke start (preflight gate).

**On any abort**:
- Write fail marker at `Path(args.mlx_work) / ".ds4-segmented-smoke-fail"`.
- Preserve all artifacts (log, partial adapter dir) as evidence.
- Do NOT delete adapter directory — it is evidence per BA R14.3-6.
- Do NOT retry, fall back to default loss, fall back to monolithic, or change segment size.
- Exit with code 3 (preflight abort) or code 4 (in-train abort) or code 2 (timeout).

### Forbidden real-smoke success claims

The smoke does NOT prove or claim:
- Real peak-memory reduction vs monolithic training.
- OOM repair or command-buffer lifetime fix.
- Convergence, loss quality, generalization.
- Throughput or speed.
- Full-training readiness.
- Correctness of provider math on real data (synthetic equivalence is Story 14.2 closure; real-data correctness is a separate story).
- Any extrapolation beyond one microbatch.

## Phase 2 — non-goals

- The smoke does NOT modify the fork, provider, default command catalog, or any production file.
- The smoke does NOT promote the adapter checkpoint for further training; it is evidence only.
- The smoke does NOT change Python environments, pin, source, or `pyproject.toml`.
- No commit or push.

## Resource safety / exclusive jobs

- Single instance lock at `Path(args.mlx_work) / ".ds4-segmented-smoke.lock"` — O_EXCL with 60s wait.
- No concurrency: the script is a single-process one-iteration trainer invocation.
- `mx.set_memory_limit` is NOT modified by the smoke; the existing MLX-LM wired-limit set during `train()` startup (L403) remains authoritative.
- `mx.disable_compile()` is NOT invoked by the smoke; per ADR 0028, primitive re-pin §10.11, the segmented provider internally manages its own `mx.vjp` segments and `mx.stop_gradient` boundaries.
- The host MLX venv at `/Volumes/Data NVME/mlx-ft/ds4/.venv` is the canonical interpreter; the config pin from `technical-spec.md` §6.1 stands.

## Protected paths (regression matrix)

| Protected artifact | Check | Consent needed? |
|---|---|---|
| `vendor/mlx-lm/` inner HEAD | `git -C vendor/mlx-lm rev-parse HEAD` == `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` | no |
| `vendor/mlx-lm` outer gitlink | `git rev-parse :vendor/mlx-lm` == same | no |
| `segmented_loss_and_grad.py` blob SHA-256 | `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` | no |
| `test_ds4_segmented_loss_and_grad.py` blob SHA-256 | `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6` | no |
| `test_mlx_lm_source.py` blob SHA-256 | `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65` | no |
| `smoke-train` command catalog string | unchanged | no |
| `smoke-train-2048` command catalog string | unchanged | no |
| `full-train` command catalog string | unchanged | no |
| `continue-train` command catalog string | unchanged | no |
| `eval`, `fuse`, `fuse-hf`, `fused-generate`, `ds4-smoke` catalog strings | unchanged | no |
| Masked default-path hash of `trainer.py` custom branch | `b53bdc549ff24b71c0b33224dbaaa30b67e42b7222e75e88e4ab87b6ad887126` | no |
| `trainer.py` worktree/index SHA-256 | `42e5ee2d13aad0ae31d6ebf63300ed260f186291ef80bf416d3395e40503468f` | no |
| `test_tuner_trainer.py` worktree/index SHA-256 | `275d6f3ad7dd32458baed4a2df48e2b46e5edaee86956f155467392ec11dbcf2` | no |
| Path A 365-file digest | `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af` | no |
| ADR 0028 SHA-256 | `0aa743b731e6393ca84e91bcd3d32bec655177779ee15e21a47ade19e322300b` | no |
| Epic 13 365-file section digest | the same Path A digest above | no |
| `adapters.safetensors` in `vendor/mlx-lm/` | remains absent before AND after Phase 1 work | no |
| Epic 14 / Story 14.2 functional hash (three authorized paths) | `75786d9e99184fe30cf752d2e2eb180612252ba0566ad8b5624fdd564fa3d74d` | no |

Heavy live jobs from other slices remain gated by `AGENTS.md` safety section and technical-spec §10.6.

## Build & test (Phase 1)

- All Phase-1 tests run under `python-envs/mlx/.venv` against the same venv MLX as the existing fine-tuning test suite.
- Exact tracked-test commands (run inside `python-envs/mlx/.venv`):

```bash
set -euo pipefail
cd /Users/spotted/projects/ds4-finetuning
MLX_PY='/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python'
test -x "$MLX_PY"

# Phase 1 focused suite
"$MLX_PY" -m pytest -q tests/test_ds4_segmented_smoke.py

# Default command catalog regression
"$MLX_PY" -m pytest -q tests/test_finetune_ds4.py

# Fork source-and-trainer regression (still uses fork venv; tracked paths)
cd vendor/mlx-lm
PYTHONPATH="$PWD" "$MLX_PY" -m pytest -q tests/test_tuner_trainer.py
PYTHONPATH="$PWD" "$MLX_PY" -m pytest -q tests/test_finetune.py
cd ..

# Source sentinel
python3 tests/test_mlx_lm_source.py

# Story 14.2 protected provider source/test
python3 tests/test_ds4_segmented_loss_and_grad.py
```

The Coder records the observed final focused-test count; every contributing file must be tracked before Reviewer/Test Manager accept it. No fabricated count is pinned here.

## TDD sequence

### Red stage — test file only

T2 and T3 must initially fail with the specific markers above before any production edit lands:

```bash
set -euo pipefail
cd /Users/spotted/projects/ds4-finetuning
MLX_PY='/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python'

# Assumes scripts/ds4_segmented_smoke.py and tests/test_ds4_segmented_smoke.py exist.
# T2 red: provider wiring not yet implemented.
"$MLX_PY" -m pytest -q tests/test_ds4_segmented_smoke.py::test_run_constructs_provider_once_and_calls_train_with_loss_and_grad \
  > agent-output/cmux-14-3/red-provider-wiring.log 2>&1 || true

# T3 red: loss_and_grad missing OR provider wiring missing.
"$MLX_PY" -m pytest -q tests/test_ds4_segmented_smoke.py::test_run_must_not_omit_loss_and_grad \
  > agent-output/cmux-14-3/red-loss-and-grad-omitted.log 2>&1 || true

grep -F 'provider must be constructed once and passed to train()' agent-output/cmux-14-3/red-provider-wiring.log
grep -F 'loss_and_grad must be passed to train()' agent-output/cmux-14-3/red-loss-and-grad-omitted.log

! grep -E 'ImportError|ModuleNotFoundError|ERROR collecting|not found' \
  agent-output/cmux-14-3/red-provider-wiring.log \
  agent-output/cmux-14-3/red-loss-and-grad-omitted.log
```

A collection/import failure is not a valid red.

### Green stage

1. Implement `scripts/ds4_segmented_smoke.py` with the minimal body matching the contract above.
2. Add `"ds4-segmented-smoke"` to `MLX_STEPS` and the catalog entry to `command_catalog`.
3. Add the one-paragraph note to `docs/architecture.md` fine-tuning toolchain subsection.
4. Run T2/T3 green, then the full focused suite, then the regression floor.
5. Run all 7 mutation recipes (T4); each fails on the targeted test.

## Coder Phase 1 can proceed now

YES — Coder may start Phase 1 implementation under this contract. Real smoke (Phase 2) remains blocked until Phase 1 is at the same revision with Reviewer PASS + Tester GREEN, AND a fresh explicit operator authorization of the exact pinned command above is recorded.

## STOP-ESCALATE

This slice does NOT reach a STOP-ESCALATE.

- No production code edit is required.
- Public `train()` API suffices to activate the provider without fork edits.
- Adapter save is a one-line `save_config` + train's final save.
- Bounded smoke is fail-closed via existing validators + a watchdog/lock + memory/disk/adapter-absence preflights.
- No real asset is required to design the contract.

## Risks and caveats

1. **MLX-version import path**: `from mlx_lm.tuner.lora import build_parser, CONFIG_DEFAULTS` — the exact relative import must be verified by the Coder at the time of Phase 1 implementation. If MLX-LM restructuring prevents reuse of `build_parser`, the Coder falls back to a hand-rolled minimal argparse that pins the same args; this is specified in the implementation plan, not as a STOP trigger.

2. **`lora_parameters` config field**: The `args.lora_parameters` attribute must exist before `linear_to_lora_layers` call — `lora.main()` reads it from CONFIG_DEFAULTS dict through the `vars(args)` namespace. The Coder must ensure `args.lora_parameters` is populated via the same YAML+defaults merge path as `lora.main()`.

3. **`save_every` defaults**: CONFIG_DEFAULTS has `save_every=100`, and `TrainingArgs.steps_per_save = args.save_every`. For `iters=1`, periodic save does NOT fire (`1 % 100 != 0`). The final unconditional save at `train()` end always fires. Coder should explicitly assert during Phase 1 synthetic tests that the script does NOT trigger periodic save and DOES emit final save (via train stub recording).

4. **Distributed.eager compile policy**: The existing `train()` at L403 sets the wired limit through `mx.metal.is_available()`. We do NOT modify this; the existing policy is authoritative. The smoke does NOT add a `mx.disable_compile()` wrap. Per ADR 0028 §10.11, the segmented provider internally manages `mx.vjp` and `mx.stop_gradient` boundaries; no eager override.

5. **`_validate_segment_size` boundary**: The provider factory validates `segment_size` in `{1, 2, 3, 4}`. Phase 2 pins `segment_size=1`. Phase 1 tests must NOT attempt to validate the factory with invalid values directly on a real call path; tests use the factory call recorder (mock).

6. **Path A absolute prohibition remains**: Per `docs/technical-spec.md` §10.12 and ADR 0028 reduction re-pin, no Path A re-entry or unbounded smoke is authorized. This slice does not alter that.

## Closure

GO is conditional on:
1. Coder Phase 1 implementation under this exact contract.
2. Independent Reviewer PASS on the exact staged paths.
3. Independent Test Manager GREEN on the tracked test suite under `python-envs/mlx/.venv`.
4. Supervisor presenting this exact pinned Phase 2 command above and the preflight constants above.
5. Fresh explicit operator authorization per slice.

After Phase 1 double-green, Phase 2 may proceed ONLY as an operator-authorized single bounded run of the pinned command.