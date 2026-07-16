# Epic 14 / Story 14.3 — DS4 segmented provider activation and bounded real-smoke contract

## Authorization and predecessor closure

Story 14.3 is separately authorized after Story 14.2 and Story 14.2a both closed
double-green. This story opens activation wiring and one bounded real-smoke
contract only. It does not authorize full training, convergence evaluation,
throughput benchmarking, OOM repair claims, Path A reopening, or any
unbounded/persistent real execution.

### Predecessor identities (immutable)

**Story 14.1** — complete double-green:
- Reviewer PASS: `agent-output/cmux-14-1/review-r6.md`
- Test Manager GREEN: `agent-output/cmux-14-1/test-report-r6c.md`
- Inner `HEAD` / outer gitlink: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
- Staged trainer SHA-256 (pre-14.2a): `7eda0fd4436a4e191b813ae6f6c4a4346e8cbe0c769d2fd75111182632382221`
- Staged trainer-test SHA-256 (pre-14.2a): `11035cfb0c8ab224772ca018e92fa75f195bea785fa58a07fa9c280b6f3dc5e9`
- Semantic MLX source baseline: 19 files, SHA-256 `8881561e55b573b4ed47676b0baf03da577f202697ab1b9ebe50efff92b3128bc`

**Story 14.2a** — complete double-green:
- Reviewer PASS: `agent-output/cmux-14-2a/review.md`
- Test Manager GREEN: `agent-output/cmux-14-2a/test-report.md`
- Staged trainer SHA-256: `42e5ee2d13aad0ae31d6ebf63300ed260f186291ef80bf416d3395e40503468f`
- Staged trainer-test SHA-256: `275d6f3ad7dd32458baed4a2df48e2b46e5edaee86956f155467392ec11dbcf2`
- Masked default-path trainer SHA-256: `b53bdc549ff24b71c0b33224dbaaa30b67e42b7222e75e88e4ab87b6ad887126`
- Inner artifact SHA-256: `e5cda46105d86bd81612daa66916a1c936bc15cdbfd4528f427176e459ccfff9`
- Inner `HEAD` / outer gitlink: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` (unchanged)

**Story 14.2** — complete double-green:
- Registered revision: `14-2-r10-coder-r10`
- Registered functional artifact hash: `75786d9e99184fe30cf752d2e2eb180612252ba0566ad8b5624fdd564fa3d74d`
- Reviewer PASS: `custom-handoffs/14-2-r11/review.md` (r11)
- Test Manager GREEN: `agent-output/cmux-14-2/test-report-r10.md` (r10)
- Staged provider SHA-256 (blob): `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`
- Staged provider-test SHA-256 (blob): `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6`
- Staged source-sentinel SHA-256 (blob): `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65`
- Mutation matrix: `17/17 RED`
- Focused suite: `47 passed`
- Trainer suite: `36 passed, 74 subtests passed`
- Finetune suite: `15 passed`
- Source sentinel: `14 tests, OK` (post-14.2 repin from 13)
- Active-memory matrix: `8/8 PASS`
- py_compile: `5/5 PASS`

Path A remains permanently stopped. Story 13.3b-5i closure is not reopened or
reinterpreted.

### Current activation gap

`scripts/finetune_ds4.py` has zero references to `loss_and_grad`,
`segmented_loss_and_grad`, `make_ds4_segmented_loss_and_grad`, `--segmented`,
`--provider`, or any custom-provider selection mechanism. The existing
`smoke-train` step uses `mlx_lm.lora` CLI which exposes no custom
`loss_and_grad` passthrough. The segmented provider module
`python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` exists and is
tracked but is completely unwired into any training command.

## Trackable user story

```
As a DS4 fine-tuning operator (WHO), I want the DS4 segmented loss-and-gradient
provider activated through a dedicated training entry point that explicitly
passes it to the Story 14.1 trainer seam while the default MLX-LM training
commands remain unchanged (WHAT), so that one bounded real-model smoke can prove
the provider runs a forward/backward pass against real 4096-token data without
OOM and produces finite loss/gradients through the immutable trainer-owned
pipeline (WHY).
```

## R14.3-1 — Minimal activation surface

The activation wires the segmented provider into a DS4-specific training entry
point. The default MLX-LM training path (`mlx_lm.lora` CLI, `smoke-train` step,
`full-train` step, `continue-train` step) must remain unchanged in behavior,
source, output, and identity.

Activation comprises exactly:

1. **A new MLX step name** (e.g., `ds4-segmented-smoke`) added to `MLX_STEPS`
   and `COMMAND_STEPS` in `scripts/finetune_ds4.py`. The step is not a default
   backend step; it must be explicitly requested.

2. **A Python entry point** that:
   - Imports `make_ds4_segmented_loss_and_grad` from
     `ds4_ft_mlx.segmented_loss_and_grad` (project-owned, `python-envs/mlx/src`)
   - Imports `train` from `mlx_lm.tuner.trainer` (vendor fork, `vendor/mlx-lm`)
   - Loads the real model from `$MLX_WORK/model-4bit` using `mlx_lm.load`
   - Constructs the provider:
     `provider = make_ds4_segmented_loss_and_grad(segment_size=1)`
   - Calls `train(..., loss_and_grad=provider)` with minimal smoke arguments
     (one iteration, batch-size 1, learning-rate 1e-5, max-seq-length 4096,
     mask-prompt, grad-checkpoint)
   - Activates the MLX venv, unsets `SSLKEYLOGFILE`, and sets `PYTHONPATH` to
     include both `python-envs/mlx/src` and `vendor/mlx-lm`

3. **No change to `mlx_lm.lora` CLI** — the CLI has no `--loss-and-grad` flag and
   none is added. No mode flag, capability bit, registry, provider-class
   dispatch, environment variable, config-file key, monkey patch, or alternate
   trainer loop is introduced into the fork, the CLI, or the generic trainer.

4. **No generic provider selection** — the segmented provider is constructed and
   passed explicitly by the entry point. There is no generic "use custom
   provider" toggle; the default path does not import or reference the
   provider module.

5. **No permanent semantic variant** — the activation step is a different
   command that uses a different code path (Python `train()` call, not
   `mlx_lm.lora` CLI). It does not add a diagnostic switch, fallback mode, or
   conditional branch to the default `smoke-train` or `full-train` steps.

6. **No Path A reopening** — no `composed_layer_sequential`, local-loss
   diagnostic, Path A report, or Story 13.3b file or evidence is touched.

## R14.3-2 — Allowed production/docs/test file scope for activation wiring

### Authorized production changes

1. `scripts/finetune_ds4.py` — add step name to `MLX_STEPS` and
   `COMMAND_STEPS`; add command catalog entry for the new step.

2. A new Python entry-point script (exact path to be pinned by Architect; the
   recommended location is `scripts/ds4_segmented_smoke.py` or an inline
   `python -c` block in the command catalog). This script imports the provider
   and calls `train()`. It is a thin wiring script, not a training framework or
   reusable library.

### Forbidden production changes

- `vendor/mlx-lm/` inner files (any) — no fork file edits, no new inner
  tracked path, no gitlink pin change. Inner `HEAD` and outer gitlink remain
  `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`.
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` — remains
  byte-identical at blob SHA-256
  `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`.
- `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn.py` and all other MLX model
  source files — no edit.
- `tests/test_ds4_segmented_loss_and_grad.py` — remains byte-identical at blob
  SHA-256 `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6`.
- `tests/test_mlx_lm_source.py` — remains byte-identical at blob SHA-256
  `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65`.
- No C/Objective-C/Metal/CUDA/ROCm/distributed/SSD/disk-cache file.
- No YAML/JSON config template, environment definition (`python-envs/*/`),
  package manifest, or architecture/ADR file unless the Architect determines a
  durable boundary change requires it.

### Authorized docs/test scope

- `docs/backlog.md` — this BA update (Story 14.2/14.2a status + Story 14.3
  user story and acceptance criteria).
- New test file(s) for activation wiring (exact scope pinned by Architect;
  likely `tests/test_finetune_ds4.py` additions or a new focused test for the
  entry-point step existence/catalog entry).
- `agent-output/cmux-14-3/` handoff artifacts and `.cmux-status/` markers.

### Optional ADR

If the Architect determines that adding a second MLX training entry point
(Python `train()` call) beside the existing `mlx_lm.lora` CLI path constitutes
a durable architectural boundary change, an ADR must be created or updated in
the same slice. If no durable boundary changes, no ADR is required.

## R14.3-3 — Real-asset manifest (by documented path; not read during BA)

The following asset paths are pinned for the eventual smoke. BA does not read,
map, stat, hash, or list any real asset. Coder/Tester must verify existence and
identity before the smoke runs, but only after implementation + Reviewer PASS +
Tester GREEN + explicit operator authorization.

| Asset | Path | Required checks before smoke |
|---|---|---|
| MLX workspace | `/Volumes/Data NVME/mlx-ft/ds4` | Directory exists; `.venv/bin/python` is the MLX interpreter; `config.json` present |
| MLX venv | `/Volumes/Data NVME/mlx-ft/ds4/.venv` | Activatable; `import mlx, mlx_lm` succeeds; MLX `0.31.2`; MLX-LM fork installed editable from `vendor/mlx-lm` |
| Converted model | `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` | Directory exists; contains MLX weights (`model-*.safetensors` or `weights/`), `config.json`, tokenizer files; `mlx_lm.load` succeeds |
| Dataset | `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke` | **Repinned 2026-07-16 (Story 14.3a):** filtered copy; 66 rows excluded from original `mlx-4096` using the smoke preflight whitespace-token conservative bound at 4096; all remaining rows pass ≤4096. Original `mlx-4096` untouched, immutable. Contains `train.jsonl`, `valid.jsonl`, `test.jsonl`; each row has exactly `prompt` and `completion` fields. |
| LoRA config | `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` | Exists; contains DS4-safe LoRA target allowlist from `mlx-lora-targets-check`; rank/alpha/target modules match `convert_lora_to_ds4.py` expectations |
| Adapter output | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke` | Does NOT exist before smoke; created by the smoke run; if present, must be removed before smoke or smoke aborts |
| Smoke log | `agent-output/cmux-14-3/smoke-log.txt` | Created by smoke run; captures stdout/stderr; archived as handoff evidence |

### Asset identity contract

- If any required asset is missing, misconfigured, or fails its identity check,
  the smoke command must NOT run. The result is STOP/NEEDS-INFO, not a guessed
  authorization or a fallback to a different model/data source.
- The dataset path `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`
  is the only authorized dataset for the real smoke (operator-authorized filtered
  copy; provenance in `agent-output/cmux-14-3/filtered-dataset-provenance.json`;
  input/kept/excluded: train 15170/15108/62, valid 819/816/3, test 824/823/1;
  total excluded 66). The original `mlx-4096` dataset is immutable and rejected
  by the pinned-path gate. No synthetic data, alternate split, or
  different token-length bucket is permitted for the real smoke.
- The model path `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` is the only
  authorized model. No raw FP8 checkpoint, HF original, or alternate conversion
  is permitted for the real smoke.
- Python path must include `python-envs/mlx/src` (for `ds4_ft_mlx`) and
  `vendor/mlx-lm` (for the fork `mlx_lm`) so the provider and trainer resolve to
  the exact tracked code.
- The MLX interpreter is `/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python`.

## R14.3-4 — Bounded 4096-token smoke contract

### Purpose

Prove the DS4 segmented loss-and-gradient provider runs exactly one real
forward/backward pass against the real 4096-token DeepSeek V4 Flash model
(`model-4bit`) and real 4096-token dataset through the immutable Story 14.1
trainer seam, producing finite loss and gradients without OOM or crash. This
smoke does NOT prove convergence, throughput, real peak-memory reduction, OOM
repair, command-buffer repair, full-training readiness, or adapter quality.

### Command shape (exact pinned form)

```bash
cd "/Volumes/Data NVME/mlx-ft/ds4" \
  && unset SSLKEYLOGFILE \
  && . "/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/activate" \
  && PYTHONPATH="/Users/spotted/projects/ds4-finetuning/python-envs/mlx/src:/Users/spotted/projects/ds4-finetuning/vendor/mlx-lm" \
  python <entry-point> \
    --model "/Volumes/Data NVME/mlx-ft/ds4/model-4bit" \
    --data "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke" \
    --adapter-path "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke" \
    --config "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json" \
    --iters 1 \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --max-seq-length 4096 \
    --mask-prompt \
    --grad-checkpoint \
    --segment-size 1
```

Where `<entry-point>` is the Python script created by Coder (R14.3-2). The
exact script path and argument parsing are pinned by Architect. The
`--segment-size 1` flag passes the value to `make_ds4_segmented_loss_and_grad(segment_size=1)`.

### Bounded scope

- Exactly one iteration (`--iters 1`).
- Batch size 1, one real row from the dataset.
- Max sequence length 4096 (token-level boundary).
- The first attempted training microbatch / first backward boundary is the
  load-bearing event. If the provider produces finite loss and gradients for
  this one microbatch without OOM, the smoke succeeds.
- No subsequent iteration, no eval pass, no distributed run, no generation, no
  fuse, no quantize, no splice, no DS4 runtime command.
- The smoke does not retry, fall back to default `smoke-train`, fall back to
  monolithic loss, use a different segment size, or use a different model/data
  path.

### Validation policy

After the smoke run:
1. **Loss**: the provider's returned `loss` scalar must be a finite
   `mx.float32` value. NaN or Inf is an abort.
2. **Token count**: the provider's returned `token_count` must be a positive
   `mx.int32` scalar matching `default_loss` mask sum for the row.
3. **Gradient schema**: the returned `gradients` tree must have exactly the
   same container types, key order, paths, leaf count, leaf shapes, and leaf
   dtypes as `model.trainable_parameters()`. Any mismatch is an abort.
4. **Finite gradients**: every gradient leaf must be finite (no NaN, no Inf).
   Any nonfinite leaf is an abort.
5. **Adapter save**: if `train()` attempts to save an adapter after the one
   iteration, the adapter directory `adapters-segmented-smoke` must contain
   `adapters.safetensors` and `adapter_config.json`. Save failure is an abort.
6. **Provider call count**: the provider body must execute exactly once for the
   one microbatch. Zero calls or >1 calls is an abort.

### Timeout

- Hard timeout: 600 seconds (10 minutes) from process start to exit.
- If the process exceeds 600 seconds, it is killed and the smoke fails.
- No partial result is accepted from a timed-out process.

### Memory/process preflight

- Before launching the smoke, the entry point or preflight must check:
  1. No other model process consuming >50 GB RSS is running (check
     `ps aux`/`vmmap` or equivalent).
  2. Available system memory is sufficient for the model + one segment graph
     (the model-4bit directory size is a lower bound; Architect pins the exact
     threshold).
  3. The instance lock (existing lock mechanism in `finetune_ds4.py`) is
     acquired. If lock acquisition fails, the smoke does not run.
  4. Disk space at the adapter output path is sufficient for a minimal adapter
     (at least 1 GB free).
- If any preflight check fails, the smoke does not run. Result is
  STOP/NEEDS-INFO.

### Instance lock

- The existing instance lock mechanism in `scripts/finetune_ds4.py` (lock file
  or equivalent) must be used. The smoke must not bypass the lock.
- If the lock is held by another process, the smoke waits up to 60 seconds,
  then aborts if the lock is still held.

### Log/artifact destinations

| Artifact | Destination |
|---|---|
| Smoke stdout/stderr | `agent-output/cmux-14-3/smoke-log.txt` |
| Training loss/gradient report | `agent-output/cmux-14-3/smoke-report.json` (structured: loss, token_count, gradient key count, finite check, provider call count, wall-clock time) |
| Adapter checkpoint | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke/` |
| Smoke marker | `.ds4-segmented-smoke-ok` or `.ds4-segmented-smoke-fail` at workspace root |

### Abort conditions (any → immediate termination, no retry)

1. OOM or memory-error exception from MLX or the OS.
2. Nonfinite loss or any nonfinite gradient leaf.
3. Gradient schema mismatch (path, shape, dtype, key order, container type).
4. Token count ≤ 0 or mismatch with `default_loss` mask sum.
5. Timeout exceeding 600 seconds.
6. Instance lock acquisition failure.
7. Model loading failure (corrupt weights, missing config, tokenizer error).
8. Dataset loading failure (missing file, malformed row, empty dataset).
9. Provider construction failure (`segment_size` rejection, model-type
   rejection, topology validation error).
10. Crash (segfault, signal, unhandled Python exception).
11. Any command that reads/writes an unexpected path (outside the pinned asset
    manifest + artifact destinations).
12. Any assertion failure, warning, or error from MLX/MLX-LM internals that
    indicates undefined behavior, shape corruption, or graph-computation failure.

### No retry/fallback

- The smoke runs exactly once. Failure is terminal.
- No retry with different `--segment-size`, different model, different dataset,
  reduced `--max-seq-length`, disabled `--grad-checkpoint`, or default
  `smoke-train` fallback.
- No fallback to `mlx_lm.lora` CLI without the provider.
- No post-hoc "partial success" interpretation — the smoke is pass/fail.

## R14.3-5 — Provider-selection and trainer-ownership evidence

Before the smoke runs, the implementation must demonstrate (via tracked tests)
the following evidence:

### Provider-selection evidence

1. The segmented provider is constructed explicitly by the activation entry
   point. The default `smoke-train` step does not import, reference, or
   construct the provider.
2. The default MLX-LM training path is byte-identical in behavior: the existing
   `smoke-train` command catalog entry is unchanged; no new branch,
   conditional, import, or side effect is added to the default path.
3. A mutation test proves that removing the provider from `train()` call falls
   back to the default loss path (no provider == default behavior, same as
   Story 14.1).
4. A mutation test proves that the activation step calls
   `make_ds4_segmented_loss_and_grad(segment_size=1)` exactly once and passes
   the result to `train(..., loss_and_grad=provider)`.
5. The provider construction is the ONLY altered behavior between the default
   `smoke-train` and the activation `ds4-segmented-smoke` step.

### Trainer-ownership evidence

1. Activation does not change any `vendor/mlx-lm` inner file. Inner `HEAD` and
   outer gitlink remain pinned at `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`.
2. Trainer-owned semantics remain exactly as Story 14.1/14.2a: accumulation,
   update, distributed averaging, validation, callbacks, checkpoint save,
   cache clearing, UI, report, and failure isolation are owned by the trainer,
   not the provider or the activation entry point.
3. The activation entry point does not import or call any trainer internal
   beyond `train()` and `default_loss` (or equivalent public imports).
4. The activation entry point does not set any environment variable that would
   alter trainer behavior beyond `PYTHONPATH` (for `ds4_ft_mlx` and fork
   `mlx_lm` resolution) and `SSLKEYLOGFILE` unsetting.
5. The provider's `loss_and_grad` result passes through the trainer's existing
   validation, finite gate, and phase-2 accumulation/update exactly as in
   Story 14.2 synthetic tests.

## R14.3-6 — Success/failure claims and rollback

### Allowed success claims (measured facts only)

After a successful smoke:
1. The provider produced finite float32 loss for one real 4096-token microbatch.
2. The provider produced finite gradients matching `model.trainable_parameters()`
   schema for that microbatch.
3. The provider body executed exactly once for that microbatch.
4. The wall-clock time from process start to exit is recorded.
5. The trainer accepted the provider's result through its phase-2 accumulation
   and update without error.
6. The adapter checkpoint was saved (if `train()` performed a save).
7. The model loaded successfully from `$MLX_WORK/model-4bit`.

### Forbidden success claims

The smoke does NOT prove or claim:
- Real peak-memory reduction vs monolithic training.
- OOM repair or command-buffer lifetime fix.
- Convergence, loss quality, or generalization.
- Throughput or speed.
- Full-training readiness.
- Correctness of the provider's segmented math on real data (synthetic
  equivalence was proven in Story 14.2; real-data correctness is a later
  separate concern).
- Any result extrapolated beyond one microbatch.

### Failure handling

1. On any abort condition (R14.3-4), the process terminates and the smoke is
   declared FAILED.
2. All captured evidence (smoke log, partial output, error traceback, crash
   dump if available) must be preserved in `agent-output/cmux-14-3/`.
3. The adapter directory `adapters-segmented-smoke` must NOT be promoted or used
   for further training. It is evidence only.
4. Failure does NOT authorize:
   - Redesign of the provider, trainer, or activation wiring.
   - A retry with different parameters.
   - A fallback to default training.
   - Additional smoke runs.
   - Full training.
   - Path A reopening or diagnostics.
5. Failure is a terminal STOP/ESCALATE to the supervisor and operator.
6. Recovery requires explicit operator authorization for a new, separately
   reviewed smoke contract.

### Rollback

- The smoke command is read-only with respect to the codebase: no source file
  is modified, no git operation is performed, no commit/push is attempted.
- The model at `$MLX_WORK/model-4bit` is not mutated. Training with
  `batch_size=1, iters=1` produces an adapter checkpoint but does not alter the
  base model.
- If the adapter directory was created, it may be deleted after evidence
  capture. It must NOT be left as a valid training checkpoint that could be
  confused with a real trained adapter.
- If any process left orphan resources (GPU memory, temp files), the
  preflight/entry point must clean them up. If cleanup fails, the failure is
  recorded as a warning but does not change the pass/fail verdict.

## R14.3-7 — Execution authorization gate

The implementation proceeds in two phases:

### Phase 1: Synthetic activation wiring (implementation)

- Coder implements the activation entry point and step wiring using synthetic
  fixtures only (tiny config, synthetic arrays, no real model, no real
  dataset).
- A tracked test proves the activation step exists in the command catalog,
  constructs the real provider class, and passes it to `train()`.
- A tracked test proves the default `smoke-train` and `full-train` steps are
  unchanged (same command string, same behavior).
- A tracked mutation test proves removing the provider from the activation step
  reverts to default loss behavior.
- Reviewer PASS and Tester GREEN must close Phase 1 before any real asset is
  read.
- No real model, checkpoint, tokenizer, dataset, adapter, GGUF, or
  `/Volumes/Data NVME/...` path is read, listed, mapped, stat'd, or opened
  during Phase 1.

### Phase 2: Bounded real smoke (separately authorized)

- Phase 2 does NOT begin until:
  1. Coder implementation is complete and tracked.
  2. Reviewer returns PASS.
  3. Tester returns GREEN.
  4. Supervisor presents the exact pinned command from R14.3-4 and the
     pinned limits (timeout, preflight, abort, artifacts).
  5. Operator gives explicit authorization to execute the smoke command.
- Phase 2 runs the single bounded smoke command from R14.3-4 exactly as pinned.
- Phase 2 is not a separate implementation slice; it is an execution gate owned
  by the supervisor/operator, not by Coder/Architect.
- If any pinned parameter needs to change (model path, dataset path, timeout,
  segment size, segment size domain), the change must be recorded as a backlog
  amendment before execution. No mid-execution parameter change is permitted.

## R14.3-8 — Tracking, staging, pin, protected evidence, no-commit/push gates

### Tracking and staging

- Every verdict-contributing source, test, entry point, handoff, and evidence
  file must be tracked and staged before Reviewer/Test Manager verdict.
- `git ls-files -- <file>` must list every BA/coder/evidence file cited in
  verdicts.
- Any untracked file the slice touched MUST be `git add`ed before declaring done.

### Inner submodule pin

- Inner `HEAD` and outer gitlink remain `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`.
- Inner cached paths remain exactly Story 14.1/14.2a's two files
  (`mlx_lm/tuner/trainer.py` and `tests/test_tuner_trainer.py`).
- Inner unstaged/untracked paths: none.
- Inner worktree/index hashes match.
- `vendor/mlx-lm/adapters.safetensors` remains absent before and after smoke.

### Protected evidence (regression matrix)

| Protected artifact | Check | Consent needed? |
|---|---|---|
| Story 14.1 trainer/test hashes | `git ls-files` + blob SHA-256 comparison | no |
| Story 14.2 provider/test/source hashes | `git ls-files` + blob SHA-256 comparison | no |
| Story 14.2a trainer/test default-path hash | masked SHA-256 comparison | no |
| Path A 365-file digest | SHA-256 `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af` | no |
| `segmented_loss_and_grad.py` provider source | blob SHA-256 `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` | no |
| `test_ds4_segmented_loss_and_grad.py` | blob SHA-256 `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6` | no |
| `test_mlx_lm_source.py` | blob SHA-256 `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65` | no |
| Default `smoke-train` command catalog entry | unchanged string in `finetune_ds4.py` | no |
| Default `full-train` command catalog entry | unchanged string in `finetune_ds4.py` | no |
| Epic 13 / Story 13.3b-5i evidence | 365 files at pinned digest | no |
| ADR 0028 | SHA-256 `0aa743b731e6393ca84e91bcd3d32bec655177779ee15e21a47ade19e322300b` | no |

### No-commit/push

- BA, Coder, Architect, Reviewer, and Tester MUST NOT run `git commit` or
  `git push` at any stage.
- All changes remain staged in the local worktree/index.
- The supervisor/operator owns the commit/push decision.

### Resource safety

- No more than one model process using >100 GB RAM at a time. The smoke
  preflight must verify no competing process before launch.
- Training and quantization are mutually exclusive on this host.
- The instance lock is intentional — the smoke MUST acquire it and MUST NOT
  bypass it.
- If the preflight detects resource contention from unrelated work, the smoke
  is PENDING-RESOURCE, not a code regression. The supervisor pauses and waits
  for the user to authorize killing the competing process.

## R14.3-9 — STOP/ESCALATE criteria

Write `custom-handoffs/14-3/ba-stop.md` and emit error JSON if any of:
- The activation choice cannot be specified without redesigning the trainer,
  model, or provider contracts (e.g., if `mlx_lm.lora` CLI must be patched to
  accept a custom provider, which violates the no-fork-edit rule).
- The requirements would require a default-path change, generic mode/registry,
  alternate trainer loop, provider-owned optimizer semantics, Path A
  reopening, or unbounded/ambiguous smoke.
- Any real asset read/execution is required to finish the requirements
  themselves (BA stage).
- Predecessor identity, protected evidence, or canonical backlog cannot be
  reconciled.
- Scope materially exceeds one activation-wiring slice plus one separately
  authorized bounded smoke.

## Acceptance criteria

1. `custom-handoffs/14-3/requirements.md` exists, contains testable activation
   (R14.3-1), asset (R14.3-3), smoke (R14.3-4), abort (R14.3-4 abort
   conditions), artifact (R14.3-4 log/artifact destinations), claim (R14.3-6),
   rollback (R14.3-6 rollback), and authorization (R14.3-7) requirements with no
   unresolved placeholder. **Proof: file exists and every R14.3-* section has
   concrete criteria.**

2. `docs/backlog.md` contains exact final predecessor identities (Story 14.2
   r10 functional hash `75786d9e...3d74d`, Story 14.2a trainer/test hashes)
   and one Story 14.3 user story in the exact form: `As a [type of user] (WHO),
   I want [some goal] (WHAT), so that [some reason] (WHY).` **Proof: grep for
   `75786d9e` and `As a.*WHO.*I want.*WHAT.*so that.*WHY` in `docs/backlog.md`.**

3. Default release/fork behavior remains unchanged unless DS4 activation is
   explicitly selected: no provider-specific trainer ownership or fallback in
   default path. **Proof: R14.3-1 section 4 and R14.3-5 section 2.**

4. Exactly one bounded 4096 smoke is specified but not executed; no real asset
   is read during BA work. **Proof: R14.3-4 defines one smoke; R14.3-7 Phase 2
   requires separate operator authorization; BA verified no command reads
   `/Volumes/Data NVME/...`.**

5. Path A permanent STOP and the single already-exhausted Path A smoke remain
   unchanged. **Proof: R14.3-8 protected evidence table and R14.3-9 STOP
   criteria.**

6. Any missing asset identity, command parameter, safety limit, or ownership
   proof produces STOP/NEEDS-INFO rather than guessed authorization. **Proof:
   R14.3-3 asset identity contract and R14.3-4 preflight/abort sections.**

7. All BA verdict files and backlog edits are tracked/staged; no commit or
   push. **Proof: `git ls-files -- custom-handoffs/14-3/requirements.md
   docs/backlog.md` returns both; `git diff --stat` shows no unstaged changes
   to these files; `git log --oneline -1` shows no new commit from BA.**

## Open questions

Q1: Should the activation entry point be a standalone Python script (e.g.,
`scripts/ds4_segmented_smoke.py`) or an inline `python -c` block in the
command catalog?
**Recommended:** Standalone script. An inline block is fragile for
multi-line imports, argument parsing, and error handling. A tracked,
testable script is cleaner and allows Coder to write synthetic fixtures
against it. Architect should adjudicate.

Q2: Should the smoke produce an adapter checkpoint (i.e., should `train()`
with `iters=1` save)? Or should the entry point suppress the save and only
prove forward/backward?
**Recommended:** Allow `train()` to save if it does so naturally; the
adapter is evidence, not a promoted checkpoint. But the adapter directory
must NOT be reused for further training. Architect should confirm whether
the existing `train()` save behavior triggers at `iters=1` or requires
`--save-every` / similar.

Q3: Does the activation entry point need a separate `--segment-size`
argument, or is `segment_size=1` hardcoded for the smoke?
**Recommended:** Accept `--segment-size` as an explicit argument defaulting
to 1, so the pinned smoke command includes it. This makes the one bounded
contract explicit and prevents ambiguity if a later operator wants to
re-pin a different size in a separate authorization. But the smoke itself
runs with `--segment-size 1` only.

Q4: Does adding a second MLX training entry point (Python `train()` call
beside `mlx_lm.lora` CLI) require an ADR?
**Recommended:** Only if the Architect determines it changes a durable
boundary. If the entry point is a thin one-off script that calls `train()`
and the default CLI path is unchanged, it may be a local tool, not an
architectural decision. Architect should adjudicate.

None of these questions block the definition of done. All have recommended
answers that do not require a prerequisite to exist or a real asset to be read.