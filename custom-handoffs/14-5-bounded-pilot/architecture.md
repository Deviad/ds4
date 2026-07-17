# Story 14.5 — Architecture: Bounded Two-Phase Checkpoint/Resume Pilot

## Verdict

**GO for synthetic implementation only. Real execution remains blocked pending Coder completion, independent Reviewer PASS, Test Manager GREEN, and separate operator authorization for each phase.**

Current public APIs at vendor pin `80fab4e419a57f9465bb9e2f4e90010d645e124c` can implement the pilot without changing vendor trainer semantics:

- `train(..., loss_and_grad=provider)` calls the provider once per local iteration.
- With `grad_accumulation_steps=1`, every local iteration takes the optimizer-update branch.
- `steps_per_save=1` writes `adapters.safetensors` and `{iteration:07d}_adapters.safetensors` after each update, then rewrites final `adapters.safetensors` after the loop.
- `model.load_weights(resume_adapter_file, strict=False)` loads adapter weights after creating the same LoRA topology and before constructing the fresh optimizer/training loop.
- `TrainingCallback.on_train_loss_report()` receives the completed local iteration after the update has been materialized.

The APIs prove **adapter-weight continuity only**. They do not restore optimizer state, MLX/NumPy RNG state, data cursor, scheduler state, or a trainer-owned global iteration. Story 14.5 must state those non-claims explicitly.

No real assets, training, inference, smoke, CUDA, or distributed execution occurred during architecture work.

---

## 1. Existing behavior and capability gap

### Frozen Story 14.3 smoke

`scripts/ds4_segmented_smoke.py` is intentionally unsuitable for this pilot:

- `_PINNED_SMOKE_VALUES["iters"] == 1`.
- `_PINNED_SMOKE_PATHS["adapter_path"]` is the Story 14.3 smoke output.
- Preflight rejects any existing adapter directory.
- `_ObservingProvider` retains only the latest call evidence.
- Success requires exactly one provider call.
- Reports and markers are fixed under Story 14.3 names.
- Watchdog is fixed at 1200 seconds.

The smoke script must remain byte-identical at current SHA-256:

`ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8`

The existing `ds4-segmented-smoke` catalog command, marker names, report path, and Story 14.3 artifacts remain unchanged.

### Vendor trainer facts at `80fab4e...`

`vendor/mlx-lm/mlx_lm/tuner/trainer.py`:

1. Iterates local steps using `range(1, args.iters + 1)`.
2. Calls `loss_and_grad(model, *batch)` once per local iteration.
3. With `grad_accumulation_steps=1`, `it % grad_accum_steps == 0` is true every iteration and `optimizer.update(model, grad)` runs.
4. Calls `mx.eval(state, losses, n_tokens, grad_accum)` before the training callback.
5. With `steps_per_report=1`, calls `on_train_loss_report` once per completed iteration with `iteration=it`.
6. With `steps_per_save=1`, saves after every completed iteration:
   - current `adapters.safetensors`;
   - `{it:07d}_adapters.safetensors`.
7. Saves final `adapters.safetensors` once more after the loop.
8. Validation occurs before the update when `it == 1`, `it % steps_per_eval == 0`, or `it == args.iters`.

Therefore:

- Phase A (`iters=2`, `steps_per_eval=2`) performs validation before updates 1 and 2, provider calls 1 and 2, updates 1 and 2, and writes `0000001_adapters.safetensors`, `0000002_adapters.safetensors`, and final `adapters.safetensors`.
- Phase B (`iters=1`, `steps_per_eval=1`) performs one validation, one provider call, one fresh-optimizer update, and writes `0000001_adapters.safetensors` plus final `adapters.safetensors`.

The pilot-level global step mapping is explicit metadata:

- Phase A local 1 → global 1;
- Phase A local 2 → global 2;
- Phase B resume offset 2, local 1 → global 3.

This is not a trainer-owned global iteration.

---

## 2. Minimal implementation surface

| Path | Action | Purpose |
|---|---|---|
| `scripts/ds4_segmented_pilot.py` | **NEW, tracked** | Dedicated fail-closed Phase A/Phase B entry point |
| `tests/test_ds4_segmented_pilot.py` | **NEW, tracked** | Synthetic mutation-sensitive contract suite |
| `scripts/finetune_ds4.py` | **EDIT, minimal** | Add two explicit non-default catalog steps |
| `docs/architecture.md` | **EDIT, one bounded-pilot paragraph** | Durable activation/checkpoint boundary |
| `docs/technical-spec.md` | **EDIT, one Story 14.5 section** | Exact commands, artifacts, verification, non-claims |
| `docs/backlog.md` | BA already updated | Verify Story 14.0 correction and Story 14.5 user story/status |

No ADR is required. This is a bounded operational proof using the already accepted trainer/provider architecture, not a new durable architecture decision.

Forbidden changes:

- `scripts/ds4_segmented_smoke.py`;
- `tests/test_ds4_segmented_smoke.py` unless only a protected-source hash guard is required (not expected);
- `vendor/mlx-lm/`;
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`;
- Path A files/evidence;
- C/Objective-C/Metal/CUDA/ROCm/distributed/SSD/GGUF/runtime files;
- default `smoke-train`, `full-train`, `continue-train`, or `ds4-segmented-smoke` command strings.

---

## 3. Pilot entry-point structure

### 3.1 Explicit phase contract

The script exposes one required phase selector and the normal MLX-LM arguments used by the catalog:

```text
--phase {phase-a,phase-b}
--model
--data
--adapter-path
--config
--resume-adapter-file (Phase B only)
--train
--fine-tune-type
--num-layers
--iters
--batch-size
--learning-rate
--max-seq-length
--mask-prompt
--grad-checkpoint
--grad-accumulation-steps
--seed
--optimizer
--val-batches
--steps-per-report
--steps-per-eval
--save-every
--segment-size
```

After config parsing/merge, the entry point compares every effective path and parameter against immutable phase specifications. Any mismatch is `pinned-paths`, `pinned-args`, or `config-conflict` and aborts before model loading.

The LoRA config is parsed separately before merge. If it contains a contract-critical field with a value different from the pinned value, reject it even when the explicit CLI value would otherwise override it. Do not silently ignore conflicting config values.

### 3.2 Immutable constants

```python
PILOT_VENDOR_SHA = "80fab4e419a57f9465bb9e2f4e90010d645e124c"
PILOT_PROVIDER_SHA256 = "20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518"
PILOT_WORKSPACE = "/Volumes/Data NVME/mlx-ft/ds4"
PILOT_INTERPRETER = "/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python"
PILOT_MODEL = "/Volumes/Data NVME/mlx-ft/ds4/model-4bit"
PILOT_DATA = "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke"
PILOT_CONFIG = "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json"
PILOT_PROVENANCE = "agent-output/cmux-14-3/filtered-dataset-provenance.json"
PILOT_LOCK_TIMEOUT = 60
PILOT_LOCK_POLL = 2
PILOT_MEMORY_HEADROOM = 32 * 1024**3
PILOT_DISK_MIN_FREE = 1 * 1024**3
PILOT_TOTAL_BUDGET = 4200
```

Shared effective training identity:

```python
COMMON_VALUES = {
    "max_seq_length": 4096,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "mask_prompt": True,
    "grad_checkpoint": True,
    "segment_size": 1,
    "grad_accumulation_steps": 1,
    "seed": 0,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "num_layers": 16,
    "val_batches": 25,
    "steps_per_report": 1,
    "save_every": 1,
    "report_to": None,
    "project_name": None,
    "trust_remote_code": False,
    "lr_schedule": None,
    "clear_cache_threshold": 0,
}
```

Phase specifications:

```python
PHASES = {
    "phase-a": {
        "iters": 2,
        "steps_per_eval": 2,
        "timeout": 2700,
        "global_offset": 0,
        "adapter_path": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a",
        "resume_adapter_file": None,
    },
    "phase-b": {
        "iters": 1,
        "steps_per_eval": 1,
        "timeout": 1500,
        "global_offset": 2,
        "adapter_path": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b",
        "resume_adapter_file": "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors",
    },
}
```

LoRA topology must also match the accepted Story 14.3 config: rank `8`, scale `20.0`, dropout `0.0`, and the exact canonical DS4 key set returned by `build_lora_parameters()`. The script records the effective topology digest in both reports.

### 3.3 No generic mode or fallback

`--phase` selects exactly one of two immutable pilot contracts. It is not a generic training mode, registry, alternate backend selector, or parameter override mechanism. Unknown phases and additional paths fail closed.

---

## 4. Exact catalog commands

Add two names to `MLX_STEPS`:

```text
ds4-segmented-pilot-phase-a
ds4-segmented-pilot-phase-b
```

Both must be excluded from `DEFAULT_BACKEND_STEPS["local-mlx"]`. They are explicit operator-only steps.

Use the pinned interpreter directly; do not rely on whichever `python` happens to be active. Run through `bash -lc` with `set -o pipefail`, unbuffered output, and `tee` so execution is visible and logged.

### Phase A

```bash
bash -lc 'set -o pipefail
cd "/Users/spotted/projects/ds4-finetuning"
unset SSLKEYLOGFILE
PYTHONUNBUFFERED=1 \
PYTHONPATH="/Users/spotted/projects/ds4-finetuning/python-envs/mlx/src:/Users/spotted/projects/ds4-finetuning/vendor/mlx-lm" \
"/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python" \
"/Users/spotted/projects/ds4-finetuning/scripts/ds4_segmented_pilot.py" \
  --phase phase-a \
  --model "/Volumes/Data NVME/mlx-ft/ds4/model-4bit" \
  --data "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke" \
  --adapter-path "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a" \
  --config "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json" \
  --train --fine-tune-type lora --num-layers 16 \
  --iters 2 --batch-size 1 --learning-rate 1e-5 \
  --max-seq-length 4096 --mask-prompt --grad-checkpoint \
  --grad-accumulation-steps 1 --seed 0 --optimizer adam \
  --val-batches 25 --steps-per-report 1 --steps-per-eval 2 \
  --save-every 1 --segment-size 1 \
  2>&1 | tee "/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5/phase-a-log.txt"'
```

### Phase B

Identical common parameters, with:

```text
--phase phase-b
--adapter-path "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b"
--resume-adapter-file "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors"
--iters 1
--steps-per-eval 1
```

Log to `agent-output/cmux-14-5/phase-b-log.txt`.

The catalog uses `project_root`, `dataset_root`, and `mlx_work` to quote paths, but generated values must equal the exact paths above. Synthetic tests compare complete normalized commands and assert all pre-existing default/smoke command strings remain byte-identical.

---

## 5. Phase lifecycle

Each phase is a separate process. No process remains active while the supervisor verifies Phase A and requests Phase B authorization.

### 5.1 Shared startup order

1. Record process start monotonic/UTC timestamps.
2. Parse CLI and select immutable `PhaseSpec`.
3. Install phase-specific `signal.alarm(timeout)` and backup watchdog before config/model work.
4. Perform a read-only precheck of marker/output paths.
5. Acquire `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock` with 60-second maximum and 2-second polling.
6. Recheck marker/output paths while holding the lock.
7. Strictly validate effective CLI/config pins.
8. Validate source/vendor/interpreter identities.
9. Validate model/dataset/config/provenance identities and safety resources.
10. Load model and dataset, freeze model, create exact LoRA topology.
11. Create the phase output directory only after all fail-closed preflight gates pass.
12. Save config/start snapshot or load-and-prove resume snapshot.
13. Construct fresh Adam optimizer and per-step observers.
14. Call vendor `train()` once.
15. Validate artifacts/evidence/cardinality/digests.
16. Cancel watchdog and release only the owned lock.
17. Atomically write report(s), then marker(s).

No pilot marker is cleared automatically. Existing output or phase/final marker means the attempt has already started or completed and is a terminal STOP. This enforces one attempt per phase.

### 5.2 Phase A

Preconditions:

- Phase A output directory absent.
- All Phase A, Phase B, and final pilot OK/fail markers absent.
- No previous final pilot report indicating an attempt.

After LoRA topology construction and before optimizer creation/update:

```python
mx.eval(model.trainable_parameters())
mx.save_safetensors("phase-a-start.safetensors", sorted_trainable_weights)
```

Then run `train()` with two iterations.

Required output:

```text
phase-a-start.safetensors
0000001_adapters.safetensors
0000002_adapters.safetensors
adapters.safetensors
adapter_config.json
```

Postconditions:

- provider records: exactly 2;
- completed train callbacks: exactly 2, iterations `[1, 2]`;
- inferred optimizer updates under pinned trainer/grad-accumulation contract: exactly 2;
- checkpoints 1 and 2 exist and validate;
- `start != step1`, `step1 != step2` canonical digests;
- step2 canonical digest equals final canonical digest;
- two finite/schema/token evidence records;
- report durable before Phase A OK marker.

### 5.3 Phase B

Phase B preconditions:

- Phase A output exists.
- Phase A OK marker exists; Phase A fail marker absent.
- Phase A report exists, has `status="ok"`, exact cardinality 2, and valid report hash binding from marker.
- Resume source is the exact `0000002_adapters.safetensors` path.
- Resume source file SHA-256 and canonical tensor digest match Phase A report.
- Current common identities equal Phase A common identities.
- Phase B output directory absent.
- Phase B/final markers absent.

After constructing the identical LoRA topology:

1. Record the resume-source tensor manifest.
2. Load exact source using `model.load_weights(path, strict=False)`.
3. Materialize all trainable weights.
4. Save `resume-start.safetensors` before optimizer construction/update.
5. Compare exact sorted tensor names, dtype, shape, and canonical digest to the Phase A source.
6. Require resume-start canonical digest differs from Phase A `phase-a-start` digest.

Only after these checks does Phase B construct a fresh Adam optimizer and call `train()` once.

Required output:

```text
resume-start.safetensors
0000001_adapters.safetensors
adapters.safetensors
adapter_config.json
```

Postconditions:

- resume-start canonical digest equals Phase A step2 source digest;
- provider records exactly 1;
- completed train callbacks exactly 1 at local iteration 1;
- inferred optimizer update exactly 1;
- global mapping 2 → 3 recorded;
- Phase B final canonical digest differs from resume-start;
- local checkpoint canonical digest equals final adapter canonical digest;
- all evidence finite/schema/token valid;
- phase report and final pilot report durable before Phase B/final OK markers.

---

## 6. Proving updates and per-step evidence

### 6.1 Step observer

Implement a pilot-specific `_StepObservingProvider`. Do not reuse the smoke `_ObservingProvider`, which stores only the latest call.

For every provider invocation, append an immutable record containing:

- provider call ordinal;
- phase;
- local and mapped global step;
- float32 scalar finite loss;
- int32 scalar positive token count;
- expected mask token count and equality;
- sorted gradient paths;
- gradient shapes and dtypes;
- gradient leaf count;
- all-finite result;
- provider-call elapsed time.

Use the smoke-tested schema/token formulas or import only pure helpers (`_flatten_gradient_tree`, `_check_schema_match`) without modifying the smoke file. Pilot tests pin the smoke source hash so any accidental smoke edit fails.

### 6.2 Completed-update observer

Implement `_PilotTrainingCallback(TrainingCallback)` with `steps_per_report=1`.

For every `on_train_loss_report` call, append:

- callback iteration;
- finite reported train loss;
- learning rate;
- reported tokens/second and iterations/second;
- derived `train_step_wall_seconds = 1 / iterations_per_second`;
- update ordinal equal to callback iteration.

Why this proves optimizer-update count:

- vendor gitlink/HEAD is pinned to `80fab4e...`;
- `grad_accumulation_steps=1` is pinned;
- in that exact trainer, `optimizer.update` occurs before `mx.eval` and before the callback on every local iteration;
- the callback iteration list must be exact and checkpoint files must exist after those completed iterations.

The report labels this as `optimizer_update_ordinal` derived from the verified trainer control flow. It does not claim optimizer-state persistence.

After `train()` returns, pair provider record N, callback N, and checkpoint N. A missing/duplicate/out-of-order item is terminal failure.

---

## 7. Safetensors identity and canonical tensor digest

No existing repository helper provides the required canonical digest. Implement a small read-only helper in `ds4_segmented_pilot.py`.

### 7.1 File SHA-256

Compute normal file SHA-256 in bounded chunks (for example 8 MiB). Never use `read_bytes()` for potentially large artifacts.

### 7.2 Canonical tensor digest v1

Parse the safetensors format directly:

1. Read the 8-byte little-endian header length.
2. Parse the JSON header with duplicate-key rejection.
3. Exclude `__metadata__`.
4. Validate every tensor has `dtype`, `shape`, and two integer `data_offsets` within the data section.
5. Reject duplicate names, invalid/overlapping offsets, negative dimensions, non-regular files, or trailing/out-of-bounds data.
6. Sort tensor names lexicographically.
7. Hash an unambiguous framed stream:
   - version tag `DS4_CANONICAL_TENSOR_V1`;
   - UTF-8 name length + name;
   - dtype length + dtype;
   - rank + each dimension as unsigned 64-bit integers;
   - tensor byte length;
   - exact raw tensor data bytes in chunks.

This digest is independent of JSON key order, header whitespace, and incidental metadata, while preserving tensor names, dtype, shape, and bytes.

Every safetensors artifact report includes:

```json
{
  "path": "...",
  "file_sha256": "...",
  "canonical_tensor_digest_v1": "...",
  "tensor_count": 96,
  "tensors": [{"name":"...","dtype":"...","shape":[...],"nbytes":...}]
}
```

Synthetic tests build tiny valid safetensors fixtures proving:

- reordered header keys/metadata can change file SHA while canonical digest remains equal;
- one tensor byte change changes canonical digest;
- name/dtype/shape change changes canonical digest;
- invalid offsets/duplicates fail closed.

---

## 8. Identity manifest and drift gates

### 8.1 Phase A identity

Record a deterministic aggregate manifest over:

- source-tree commit and relevant clean/dirty state;
- pilot source SHA-256;
- provider source SHA-256 `205721...`;
- smoke source SHA-256 `ec179...`;
- vendor outer gitlink and inner HEAD, both `80fab4e...`;
- interpreter exact path and Python version;
- MLX version `0.31.2`;
- `mlx_lm` resolved module path under `vendor/mlx-lm`;
- LoRA config file SHA-256 and normalized effective topology;
- dataset provenance file SHA-256;
- `train.jsonl`, `valid.jsonl`, `test.jsonl` SHA-256 values, matched to tracked provenance;
- model metadata/tokenizer/index files;
- every model payload file relative path, size, and chunked SHA-256.

The current repository does not expose a separate accepted exact model-payload manifest. Therefore the pilot must chunk-hash model payloads during real preflight, or STOP/NEEDS-INFO if a reviewed exact manifest is supplied later. File names/sizes/mtimes alone are not exact identity.

No real hash scan occurs during synthetic implementation/review.

### 8.2 Phase B identity

Recompute the same manifest and require exact equality to Phase A for all common assets/source identities. Phase-specific output/resume fields are compared against their exact PhaseSpec.

Any mismatch blocks model loading/training.

### 8.3 Contract digest

Compute a JSON-canonical SHA-256 over normalized common identity, effective training parameters, topology, and phase-specific fields. Phase A marker stores the Phase A report SHA and contract digest. Phase B verifies marker → report → checkpoint bindings before loading.

---

## 9. Reports and markers

### 9.1 Paths

Reports:

```text
agent-output/cmux-14-5/phase-a-report.json
agent-output/cmux-14-5/phase-b-report.json
agent-output/cmux-14-5/pilot-report.json
```

Markers:

```text
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-ok
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-ok
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-fail
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-ok
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail
```

### 9.2 Durable write protocol

Use same-directory temporary files, flush + `fsync`, then `os.replace` for reports and markers. Marker JSON binds:

- phase/status;
- report path and file SHA-256;
- contract digest;
- output path;
- resume checkpoint file/canonical hashes when applicable;
- timestamp and exit code.

Write order:

- Phase A success: release lock → Phase A report → Phase A OK marker.
- Phase A failure: release lock → Phase A fail report → final pilot fail report → Phase A fail marker → final fail marker.
- Phase B success: release lock → Phase B report → final pilot report → Phase B OK marker → final OK marker.
- Phase B failure: release lock → Phase B fail report → final pilot fail report → Phase B fail marker → final fail marker.

An OK marker is never written before all required reports are durable. A report or marker write failure is terminal; make best-effort fail evidence and print the failure visibly.

### 9.3 Final report

The final report contains exact command strings, common/phase identities, Phase A/B evidence, call/update totals `2 + 1 = 3`, global mapping, artifact hashes, continuity proofs, active wall-clock total, lock lifecycle, cleanup warnings, and non-claims.

`total_active_wall_seconds = phase_a.wall + phase_b.wall` must be `<= 4200`. Human approval/wait time between processes is excluded.

---

## 10. Lock, timeout, cleanup, and abort semantics

### 10.1 Lock

Reuse the Story 14.3 lock algorithm and path:

```text
/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock
```

- O_EXCL acquisition;
- 60-second maximum;
- 2-second polling;
- exact `pid=TOKEN` ownership;
- release only owned lock.

The pilot may import the tested lock helpers from `ds4_segmented_smoke` or copy their exact small implementation. Importing is preferred to avoid divergence; smoke bytes remain unchanged.

### 10.2 Timeouts

Phase A installs a 2700-second alarm; Phase B installs a 1500-second alarm. Each also starts a daemon backup watchdog that sends SIGALRM to the current process at timeout + 5 seconds if the primary alarm does not run.

No combined process-level 4200-second timer is needed because phases are separate. The final report enforces the summed active budget.

### 10.3 Cleanup ownership

On every terminal path:

1. cancel signal alarm;
2. release only the owned lock;
3. record release result/cleanup warnings;
4. preserve output/checkpoints/logs;
5. write fail/success reports and markers in the ordering above.

Never delete or overwrite phase output. Never remove Story 14.3 outputs. Partial pilot artifacts remain evidence.

Immediate terminal abort includes all R14.5-7 conditions. No retry, fallback, smaller sequence, changed segment size, alternate dataset/model/backend, or smoke rerun.

---

## 11. Resume semantics and explicit non-claims

Phase B starts a new process and creates:

- new model object;
- same LoRA topology;
- new Adam optimizer with empty optimizer state;
- NumPy seed reset to 0;
- MLX seed reset to 0;
- new dataset iterator beginning from its newly seeded ordering;
- local trainer iteration beginning at 1.

The checkpoint load transfers trainable adapter tensor values only. It does not transfer:

- Adam moments/step counter;
- RNG state;
- dataset cursor/order position;
- LR scheduler state;
- validation state;
- trainer global iteration.

The only allowed continuity claim is:

> Before Phase B update 1, the materialized trainable adapter tensor set exactly equals the Phase A step-2 checkpoint by canonical tensor digest and tensor schema; after Phase B update 1, those adapter weights changed.

---

## 12. Synthetic test plan

Create `tests/test_ds4_segmented_pilot.py`. Every verdict-contributing file must be tracked before Reviewer/Test Manager verdict.

### T1 — Exact phase specs

Assert all paths, iters, call counts, global offsets, eval/save/report cadence, and timeouts exactly match R14.5. Mutating any value fails.

### T2 — Parser/config fail closed

Conflicting CLI/config values for sequence length, batch, LR, seed, optimizer, LoRA topology, cadence, paths, resume source, or phase fail before load/train.

### T3 — Catalog exactness and non-default status

Assert both pilot steps are in `MLX_STEPS`, absent from default local-MLX steps, use absolute pinned interpreter, `pipefail`, `tee`, exact args/logs, and do not alter any existing default/smoke catalog string.

### T4 — Output/marker one-attempt gate

Pre-existing output or any same-phase/final marker blocks launch. No marker is cleared.

### T5 — Phase dependency

Phase B fails without valid Phase A report, Phase A OK marker, report-hash binding, exact resume path, and exact recorded checkpoint hashes.

### T6 — Resume proof

Tiny synthetic safetensors fixtures prove matching resume-start passes; fresh/unloaded/partially loaded weights fail; resume-start equal to Phase A start fails.

### T7 — Canonical digest

Cover header reorder/metadata independence and name/dtype/shape/data sensitivity plus malformed files.

### T8 — Per-step evidence cardinality

Fake trainer/provider events prove Phase A requires exactly 2 provider + 2 callback + 2 checkpoint records; Phase B requires 1 each. Missing, duplicate, reordered, nonfinite, schema drift, or token mismatch fails.

### T9 — Update/global mapping

Assert local `[1,2]` maps global `[1,2]`, Phase B local `[1]` maps global `[3]`, and total calls/updates equal 3.

### T10 — Checkpoint progression

Phase A requires start != step1 != step2 and step2 == final. Phase B requires source == resume-start != final and local checkpoint == final.

### T11 — Identity drift

Model/dataset/config/provider/pilot/vendor/interpreter manifest mismatch blocks before training. Synthetic files only.

### T12 — Watchdogs

Assert `signal.alarm(2700)` and `signal.alarm(1500)`, backup deadline expression, timeout failure classification, no retry, and lock release.

### T13 — Terminal cleanup

Inject exceptions at parser, preflight, start-save, load, provider, callback, checkpoint validation, report, marker, timeout, and success. Assert owned lock released exactly once and partial evidence preserved.

### T14 — Final marker ordering

Final OK marker impossible until both phase reports and final report are durably written. Any write failure yields fail status.

### T15 — Frozen smoke/default paths

Assert:

- `scripts/ds4_segmented_smoke.py` SHA-256 remains `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8`;
- provider source remains `205721...`;
- Story 14.3 report/log/provenance unchanged;
- default command strings unchanged;
- pilot import/tests perform no `/Volumes/Data NVME/...` access.

### T16 — Path A/vendor isolation

Assert Path A count/digest remains `365` / `7241924d...`, vendor gitlink/HEAD remains `80fab4e...`, and no vendor inner diff exists.

### T17 — Full report aggregation

Synthetic Phase A/B reports compose exact final totals, hashes, wall-clock budget, progression, non-claims, and marker bindings.

### T18 — Mutation oracles

At minimum mutate: phase iters, timeouts, save cadence, resume path/hash check, canonical digest bytes/schema, provider cardinality, callback cardinality, final marker order, no-retry branch, output-exists gate, lock release, smoke hash, and default-command strings. Each targeted test must turn RED.

---

## 13. Build/test gate

Focused synthetic gate:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
PYTHONDONTWRITEBYTECODE=1 \
python-envs/mlx/.venv/bin/python -m pytest -q \
  tests/test_ds4_segmented_pilot.py \
  tests/test_ds4_segmented_smoke.py \
  tests/test_finetune_ds4.py \
  tests/test_mlx_lm_source.py \
  tests/test_ds4_segmented_loss_and_grad.py \
  tests/test_ds4_gguf_base_smoke.py
```

Also run:

- `git diff --check`;
- `git ls-files` for every cited test;
- smoke/provider/protected hashes;
- Path A count/digest;
- vendor outer gitlink/inner HEAD/status;
- synthetic no-real-path-access oracle;
- fresh synthetic clone if Story 14.4 commit is not yet the current outer HEAD.

No real pilot command is run during implementation/review/testing.

---

## 14. Durable documentation updates

Coder updates:

### `docs/architecture.md`

Add one paragraph under DS4 segmented activation:

- pilot is a separate non-default entry point;
- vendor trainer remains owner of updates/checkpoint cadence;
- Phase B proves adapter-weight equality only;
- no optimizer/RNG/data-cursor/global-step continuity claim;
- smoke contract remains frozen.

### `docs/technical-spec.md`

Add a Story 14.5 section containing:

- exact Phase A/B catalog commands;
- asset/parameter pins;
- output/checkpoint/report/marker paths;
- canonical digest definition;
- timeout/lock/abort semantics;
- synthetic test gate;
- future visible execution/authorization sequence;
- non-claims.

No new ADR.

---

## 15. Future visible execution protocol

Real execution is separately authorized and must use the `panel-runner`/cmux visible pane protocol.

1. Present exact Phase A command, pins, output path, and 2700-second timeout.
2. Obtain explicit operator authorization.
3. Run Phase A in one visible pane with `tee` and pipefail.
4. Poll until terminal; process must exit fully.
5. Read/verify Phase A report, marker, identities, checkpoint hashes, cardinality, and lock release.
6. Present exact Phase B command, exact resume SHA/canonical digest, and 1500-second timeout.
7. Obtain separate explicit authorization.
8. Reuse the same visible pane after prior process exit.
9. Poll until terminal and verify Phase B/final reports and markers on screen.

Security/auth/resource/backoff issues are reported immediately. No opaque delegate or detached PID/log execution.

---

## 16. STOP/NEEDS-INFO conditions

Synthetic implementation stops if:

- vendor public APIs cannot produce required checkpoint names/cadence;
- strict-false load followed by resume-start save cannot prove exact trainable tensor equality;
- optimizer-update evidence cannot be tied to pinned trainer control flow and callbacks;
- config topology cannot be pinned without changing Story 14.3/vendor behavior;
- smoke/default command strings drift;
- any test requires real asset access;
- any verdict-contributing file is untracked;
- Path A/provider/vendor/protected history changes.

Future real execution additionally stops if:

- exact model/dataset/config/provider/pilot/vendor identity cannot be established;
- chunked model hashing cannot complete inside the phase hard budget;
- output or marker already exists;
- Phase A report/marker/checkpoint binding fails;
- budgets require increase;
- any contract value must change.

No STOP condition is triggered for synthetic implementation. Current APIs are sufficient for the dedicated pilot.

---

## 17. Coder implementation order

1. Add `tests/test_ds4_segmented_pilot.py` with RED contract tests.
2. Add minimal `scripts/ds4_segmented_pilot.py` until tests turn GREEN.
3. Add two non-default catalog entries without changing existing strings.
4. Add mutation oracles.
5. Update canonical architecture/technical spec.
6. Run focused/full synthetic gates and tracking checks.
7. Stage every verdict-contributing test and handoff artifact.
8. Obtain independent Reviewer PASS and Test Manager GREEN.
9. Do not run real assets/training/inference.
10. Do not commit or push during this slice.