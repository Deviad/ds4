# Epic 14 / Story 14.5 — Bounded multi-step training + resume pilot

## Authorization and boundary

Operator authorized requirements/design work after Story 14.4 closure.

This BA slice:

- corrects stale Story 14.0 status using already-completed downstream evidence;
- defines smallest meaningful bounded real pilot proving more than one optimizer
  update plus adapter-weight resume continuity;
- authorizes synthetic implementation/review/test only after Architect GO;
- does **not** authorize real model, dataset, adapter, training, inference, smoke,
  CUDA, or distributed access/execution.

No real asset access, execution, deletion, commit, or push during BA.

## Story 14.0 status correction

Story 14.0 status `READY FOR ARCHITECTURE` is stale. Mark it COMPLETE without
rewriting history:

- preserve its original bootstrap-only scope and initial pin
  `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`;
- record that Story 14.4 later advanced the vendor pin to
  `80fab4e419a57f9465bb9e2f4e90010d645e124c`;
- cite downstream completion: Stories 14.1–14.4, Story 14.4 Reviewer PASS,
  Test Manager GREEN, remote submodule reachability, and synthetic fresh-clone
  exact suite `246 passed, 3 skipped`;
- preserve Story 14.0's historical no-fork-edit/no-real-execution boundary.

## Trackable user story

```text
As a DS4 fine-tuning operator (WHO), I want a two-phase bounded segmented
training pilot with two initial optimizer updates followed by one verified
adapter-weight resume update (WHAT), so that checkpoint creation and continuation
from prior trained adapter weights are proven before any larger training run
(WHY).
```

## R14.5-1 — Current capability gap and mandatory implementation phase

The existing `scripts/ds4_segmented_smoke.py` cannot execute this pilot:

1. `_PINNED_SMOKE_VALUES` requires `iters=1`.
2. `_PINNED_SMOKE_PATHS` requires the Story 14.3 smoke adapter path.
3. Preflight rejects any existing adapter output path.
4. Post-run validation requires exactly one provider call.
5. Report schema captures one final observation, not per-step observations.
6. Story 14.3 smoke artifacts and markers must remain immutable.
7. Existing `--resume-adapter-file` loads adapter weights only; it does not
   restore optimizer state, RNG state, dataset cursor, or a global iteration.

Therefore real execution is BLOCKED pending a separately reviewed minimal
implementation phase.

### Minimal implementation surface

Architect must design a dedicated explicit pilot entry point, recommended:

- `scripts/ds4_segmented_pilot.py`;
- explicit catalog step(s) in `scripts/finetune_ds4.py`;
- tracked synthetic tests, recommended `tests/test_ds4_segmented_pilot.py`.

The implementation may reuse Story 14.3 preflight, lock, observer, report, and
watchdog concepts, but must not relax or repurpose the frozen smoke contract.

Forbidden:

- changing Story 14.3 smoke parameters, paths, report, or markers;
- changing default `smoke-train`, `full-train`, or `continue-train` behavior;
- editing `vendor/mlx-lm` trainer/provider semantics;
- adding a generic registry/mode/fallback to MLX-LM;
- changing CUDA, distributed, Metal production inference, CPU, SSD streaming,
  GGUF, or Path A code/evidence.

Implementation authorization remains synthetic only. Reviewer PASS and Test
Manager GREEN are mandatory before real execution authorization.

## R14.5-2 — Exact immutable asset identity

Both phases pin exactly:

| Asset | Exact path / identity |
|---|---|
| MLX workspace | `/Volumes/Data NVME/mlx-ft/ds4` |
| MLX interpreter | `/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python` |
| Model | `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` |
| Dataset | `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke` |
| LoRA config | `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` |
| Provider | `ds4_ft_mlx.segmented_loss_and_grad.make_ds4_segmented_loss_and_grad(segment_size=1)` |
| Vendor trainer | gitlink `80fab4e419a57f9465bb9e2f4e90010d645e124c` |
| Dataset provenance | `agent-output/cmux-14-3/filtered-dataset-provenance.json` |

Before Phase A, preflight records SHA-256/identity for model metadata, dataset
splits, config, provider source, pilot source, and vendor gitlink. Before Phase
B, the same identities must match exactly. Payload-heavy model files need not be
rehash-scanned if Story 14.3's accepted identity procedure provides an exact
manifest; no identity may be guessed.

Any identity mismatch → terminal STOP/NEEDS-INFO. No alternate path.

## R14.5-3 — Shared training identity

Both phases use exactly:

| Parameter | Value |
|---|---:|
| `max_seq_length` | `4096` |
| `batch_size` | `1` |
| `learning_rate` | `1e-5` |
| `mask_prompt` | `true` |
| `grad_checkpoint` | `true` |
| `segment_size` | `1` |
| `grad_accumulation_steps` | `1` |
| `seed` | `0` |
| `fine_tune_type` | `lora` |
| optimizer | same config-selected optimizer as Story 14.3; expected `adam` |
| `val_batches` | `25` |
| `steps_per_report` | `1` |
| `steps_per_save` / `save_every` | `1` |
| retry | none |
| fallback | none |
| distributed | disabled / world size `1` |

Config-path values that conflict with the pinned table must be rejected, not
silently merged.

## R14.5-4 — Phase A: two-update pilot

### Exact identity

- Phase: `phase-a`
- Iterations / optimizer updates: exactly `2`
- Provider calls: exactly `2`
- Local steps: `1`, `2`
- Global steps: `1`, `2`
- Save cadence: every step (`save_every=1`)
- Eval cadence: `steps_per_eval=2`; current trainer's mandatory first/final
  validation behavior remains active
- Output directory:
  `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a`
- Log:
  `agent-output/cmux-14-5/phase-a-log.txt`
- Report:
  `agent-output/cmux-14-5/phase-a-report.json`
- OK/fail markers:
  `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-{ok,fail}`

### Output-directory gate

Phase A output directory must not exist before launch. Existing directory →
STOP, no deletion or overwrite.

### Required artifacts

Phase A must produce:

- `phase-a-start.safetensors` — trainable adapter weights before update 1;
- `0000001_adapters.safetensors` — checkpoint after update 1;
- `0000002_adapters.safetensors` — checkpoint after update 2;
- `adapters.safetensors` — final adapter after update 2;
- `adapter_config.json`;
- SHA-256 and canonical tensor digest for every safetensors artifact.

Canonical tensor digest means deterministic digest over sorted tensor name,
dtype, shape, and raw tensor bytes, independent of incidental file metadata.

### Phase A progression proof

- start digest differs from step-1 digest;
- step-1 digest differs from step-2 digest;
- step-2 canonical digest equals final `adapters.safetensors` canonical digest;
- exactly two optimizer updates occurred;
- exactly two per-step finite evidence records exist.

Any equality where change is required, missing artifact, or count mismatch →
terminal failure.

## R14.5-5 — Phase B: one verified resume update

Phase B may start only after Phase A PASS report and marker are durable.

### Exact identity

- Phase: `phase-b`
- Iterations / optimizer updates: exactly `1`
- Provider calls: exactly `1`
- Local step: `1`
- Global step start: `2`
- Global step end: `3`
- Resume source:
  `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors`
- Output directory:
  `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b`
- Save cadence: every step (`save_every=1`)
- Eval cadence: `steps_per_eval=1`
- Log:
  `agent-output/cmux-14-5/phase-b-log.txt`
- Report:
  `agent-output/cmux-14-5/phase-b-report.json`
- OK/fail markers:
  `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-{ok,fail}`

### Output-directory and source gates

- Phase B output directory must not exist before launch.
- Phase A resume source must exist, be regular, and match the hash recorded in
  Phase A report.
- Phase A report/marker/config/model/dataset/provider/vendor identities must
  still match.

### Resume continuity proof

Before Phase B update 1:

1. Construct the same LoRA topology.
2. Load the exact Phase A `0000002_adapters.safetensors` with `strict=False`
   only as required by existing LoRA loading semantics.
3. Save loaded trainable weights as `resume-start.safetensors` before any
   optimizer update.
4. Prove `resume-start.safetensors` canonical tensor digest equals Phase A
   step-2 resume-source canonical digest.
5. Prove resume-start digest differs from Phase A `phase-a-start.safetensors`
   digest.

After Phase B update 1:

- produce `0000001_adapters.safetensors`, `adapters.safetensors`, and
  `adapter_config.json`;
- prove Phase B final canonical digest differs from resume-start digest;
- prove local step `1` maps to global step `3`;
- record one finite per-step evidence record and exactly one provider call.

This proves **adapter-weight continuity only**. It explicitly does not prove
optimizer-state, RNG-state, data-cursor, scheduler-state, or exact interruption
resume. Full-state resume requires a separate future story.

## R14.5-6 — Time budgets

Measured Story 14.3 baseline:

- one complete process/iteration: `1025.602482 s`;
- initial validation: `816.30 s`;
- derived non-validation remainder: `209.302482 s`.

### Phase A budget derivation

Two updates with mandatory first/final validation:

```text
2 × 816.30 + 2 × 209.302482 = 2051.204964 s estimated
2051.204964 × 1.25 = 2564.006205 s
rounded hard budget = 2700 s (45 minutes)
```

### Phase B budget derivation

One resumed update with validation:

```text
1025.602482 × 1.25 = 1282.003103 s
rounded hard budget = 1500 s (25 minutes)
```

### Total budget

- Phase A hard timeout: `2700 s`
- Phase B hard timeout: `1500 s`
- Total active wall-clock budget: `4200 s` (70 minutes)

Total budget counts process startup, preflight, loading, validation, training,
checkpoint hashing, reporting, and cleanup. Supervisor waiting/approval time
between phases is excluded, but no process may remain active between phases.

Timeout in either phase → kill owned process, write fail report/marker, release
lock, preserve evidence, stop. No partial success.

## R14.5-7 — Lock, preflight, abort, and no-retry behavior

Reuse Story 14.3 safety semantics:

- single `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock`;
- lock wait maximum `60 s`, polling `2 s`;
- exact PID-token ownership; release only owned lock;
- fail if competing process exceeds `50 GiB` RSS;
- model-size + `32 GiB` available-memory headroom;
- at least `1 GiB` free at each output parent;
- signal alarm + independent backup watchdog;
- terminal cleanup on success, exception, signal, timeout, or report failure.

Immediate abort:

- OOM/memory error;
- timeout;
- lock/preflight/identity failure;
- nonfinite loss or gradient;
- token count nonpositive or mask mismatch;
- gradient schema/path/shape/dtype drift;
- provider call or optimizer-update count drift;
- missing/hash-mismatched checkpoint;
- resume-start digest mismatch;
- output path already exists;
- model/dataset/config/provider/vendor drift;
- crash, signal, or report/marker write failure;
- unexpected path read/write.

One attempt per phase. Phase A failure blocks Phase B. Phase B failure makes
pilot FAILED. No retry, reduced length, changed segment size, changed dataset,
default trainer fallback, monolithic loss fallback, smoke rerun, or alternate
backend.

## R14.5-8 — Measurable evidence and final report

Each per-step record must include:

- phase, local step, global step;
- finite float32 loss;
- positive int32 token count and expected mask token count;
- gradient leaf count, sorted paths, shapes, dtypes;
- all-finite result;
- provider call ordinal;
- optimizer-update ordinal;
- checkpoint path, file SHA-256, canonical tensor digest;
- step wall time.

Final report:

`agent-output/cmux-14-5/pilot-report.json`

Required summary:

- overall status;
- exact commands and pinned parameters;
- source tree/commit and vendor gitlink identities;
- Phase A/B start/end times and wall clocks;
- total active wall clock ≤ `4200 s`;
- provider calls `2 + 1 = 3`;
- optimizer updates `2 + 1 = 3`;
- global progression `0 → 1 → 2`, resume at `2`, then `2 → 3`;
- all losses/gradients finite;
- all checkpoint/config/report/marker paths and hashes;
- Phase A start/step1/step2 change proofs;
- Phase B resume-start equality to Phase A step2;
- Phase B final change proof;
- lock acquisition/release and cleanup warnings;
- explicit non-claims.

Final marker after both phases and report durability:

`/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-ok`

A fail marker is written instead on any failure. Final report must be preserved
regardless of verdict.

## R14.5-9 — Synthetic implementation gates

Before real authorization, tracked mutation-sensitive tests must prove without
real assets:

1. exact Phase A/B paths and parameter pins;
2. Phase A iters/calls/updates = `2`; Phase B = `1`; total = `3`;
3. save cadence `1`, report cadence `1`, eval cadence `2` then `1`;
4. timeouts `2700`, `1500`, total `4200`;
5. Phase B blocked without Phase A PASS marker/report/source hash;
6. output directories rejected when pre-existing;
7. resume-start canonical digest equals prior checkpoint digest;
8. restarted/unloaded/fresh weights are detected and rejected;
9. per-step finite/schema/token evidence list has exact cardinality;
10. one-attempt/no-retry/no-fallback behavior;
11. lock release on every terminal path;
12. smoke script/catalog/markers/report remain byte-identical;
13. default training commands remain unchanged;
14. no Path A, CUDA, distributed, production inference, or real-asset access;
15. final report/marker only after both phases succeed.

Reviewer must return PASS and Test Manager GREEN on tracked tests and exact
candidate identity before real execution.

## R14.5-10 — Visible execution protocol

Real execution, if separately authorized, must use visible cmux/panel execution,
not opaque delegation or detached PID/log jobs.

Sequence:

1. Supervisor presents exact Phase A command, pins, and `2700 s` budget.
2. Operator explicitly authorizes Phase A.
3. Run Phase A in visible pane; poll with supervision protocol until terminal.
4. Supervisor reads Phase A report/marker and verifies all gates.
5. Supervisor presents exact Phase B command, resume hash, and `1500 s` budget.
6. Operator explicitly authorizes Phase B.
7. Run Phase B in same visible pane after prior process fully exits.
8. Verify final report/marker on screen; report any auth/security/resource issue
   immediately.

No real execution occurs during BA/Architect/Coder/Reviewer/Test Manager
implementation work.

## R14.5-11 — Claims and protected history

Allowed claims after success only:

- two Phase A optimizer updates and one Phase B optimizer update completed;
- all three provider calls returned finite evidence;
- checkpoints were created at pinned cadence;
- Phase B loaded adapter weights exactly matching Phase A step-2 checkpoint;
- Phase B changed those resumed weights after one update;
- measured wall times and hashes.

Forbidden claims:

- convergence, quality, generalization, or useful training;
- throughput/performance improvement;
- OOM or command-buffer repair;
- full-training readiness;
- optimizer/RNG/dataset-cursor/scheduler continuity;
- exact interruption recovery;
- behavior beyond three bounded updates;
- CUDA/distributed readiness.

Protected:

- Path A permanent STOP and 365-file digest;
- Stories 14.0–14.4 history/evidence;
- Story 14.3 smoke report/log/provenance, marker, and adapter evidence;
- no smoke rerun;
- no mutation of model, dataset, or prior adapter outputs.

## R14.5-12 — STOP/ESCALATE

STOP/NEEDS-INFO if:

- current public APIs cannot support a dedicated pilot without changing vendor
  trainer semantics;
- exact adapter-weight continuity cannot be captured before Phase B update;
- current config values conflict with pinned contract;
- Phase A/B outputs cannot be isolated from Story 14.3 artifacts;
- required synthetic test cannot be made mutation-sensitive;
- budget requires increase before execution;
- any real asset access is needed to finish requirements/architecture/code
  review;
- protected history/evidence would change;
- untracked verdict-contributing file exists.

Any contract change requires BA + Architect repin before execution.

## Acceptance criteria

1. Story 14.0 canonical status is COMPLETE while preserving original bootstrap
   history and noting Story 14.4's later pin advance.
2. Story 14.5 appears in `docs/backlog.md` using exact WHO/WHAT/WHY format.
3. Phase A exactly 2 updates/calls; Phase B exactly 1 resumed update/call;
   total exactly 3.
4. Model, filtered dataset, config, provider, 4096 length, batch 1, LR 1e-5,
   mask-prompt, grad-checkpoint, and segment size 1 are exactly pinned.
5. Save cadence 1; resume source and separate output paths exact.
6. Hard budgets are Phase A `2700 s`, Phase B `1500 s`, total `4200 s`, derived
   from Story 14.3 measured `1025.602482 s` and `816.30 s` validation.
7. Per-step finite/schema/token/call/update/checkpoint/hash evidence and resume
   equality/change proofs are required.
8. Mandatory synthetic implementation phase closes Reviewer PASS + Test Manager
   GREEN before any real authorization.
9. One attempt per phase; no retry/fallback/smoke rerun/alternate backend.
10. Visible cmux/panel execution and final report required for future real run.
11. Non-claims and Path A/Story 14.3 protections explicit.
12. No real training/inference/asset access, commit, or push during this slice.
