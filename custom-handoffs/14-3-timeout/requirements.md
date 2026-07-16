# Epic 14 / Story 14.3b — Timeout repin from 600s to 1200s

## Authorization

Operator explicitly authorized a minimal timeout-only repin after the
filtered-dataset smoke (Story 14.3a) reached trainer startup but timed out at
600 seconds with 0/1 iterations. Evidence:

- `agent-output/cmux-14-3/smoke-log.txt`: shows 0% progress, 0/1 iterations, no
  loss value emitted.
- `agent-output/cmux-14-3/smoke-report.json`: `"failure_code": "timeout"`,
  `"failure_message": "smoke timed out after 600s"`,
  `"wall_clock_seconds": 603.217`.
- Adapter output directory (`adapters-segmented-smoke`) was empty; instance
  lock was released.

This repin changes only the hard timeout constant from 600 seconds to 1200
seconds (20 minutes). It does NOT claim convergence, OOM repair, correctness
beyond one bounded attempt, throughput, or full-training readiness. It does
NOT authorize a retry, a redesign, a parameter change other than the timeout,
or any new execution. It defines requirements for the architectural and
implementation repin (changing the constant and corresponding docs/tests) only.

## Predecessor closure (immutable)

This story inherits all predecessor identities, protected evidence, and pins
from Story 14.3 / Story 14.3a. The complete predecessor closure table is in
`custom-handoffs/14-3/requirements.md` (R14.3-8). Key protected hashes:

- `segmented_loss_and_grad.py`: blob SHA-256
  `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`
- `test_ds4_segmented_loss_and_grad.py`: blob SHA-256
  `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6`
- `test_mlx_lm_source.py`: blob SHA-256
  `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65`
- Inner `HEAD` / outer gitlink: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
- Path A 365-file evidence digest: `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`

All remain protected and byte-identical. No Path A reopening.

## Trackable user story

```
As a DS4 fine-tuning operator (WHO), I want the bounded real-smoke hard
timeout repinned from 600 seconds to 1200 seconds (WHAT), so that the
single authorized smoke attempt has enough wall-clock time to complete one
forward/backward pass against the real 4096-token model and dataset without
premature timeout (WHY).
```

## R14.3b-1 — Minimal timeout-only change

### Change (exhaustive)

1. The smoke hard timeout constant `SMOKE_TIMEOUT_SECONDS` in
   `scripts/ds4_segmented_smoke.py` (currently `600`) changes to `1200`.

That is the only production source change. No other constant, function, control
flow, import, argument, catalog entry, default path, or file is modified.

### Preserved (exhaustive — everything else stays identical)

- Filtered dataset path: `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`
- Model path: `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`
- LoRA config: `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`
- Adapter output: `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke`
- Max sequence length: 4096
- Iterations: 1 (`--iters 1`)
- Batch size: 1 (`--batch-size 1`)
- Learning rate: 1e-5 (`--learning-rate 1e-5`)
- Prompt masking: enabled (`--mask-prompt`)
- Gradient checkpointing: enabled (`--grad-checkpoint`)
- Segment size: 1 (`--segment-size 1`)
- Instance lock mechanism: unchanged (lock acquisition, 60s wait, abort on failure)
- Preflight checks: unchanged (memory, lock, disk, no competing process)
- Report schema: unchanged (loss, token_count, gradient key count, finite
  check, provider call count, wall-clock time)
- Abort conditions: unchanged (OOM, nonfinite loss/gradient, schema mismatch,
  token count ≤0, timeout [now 1200s], lock failure, model load failure,
  dataset load failure, provider construction failure, crash, unexpected path
  access, internal assertion/shape/graph error)
- No retry, no fallback, no alternative model/data/segment-size/params
- Lock timeout (`SMOKE_LOCK_TIMEOUT_S = 60`): unchanged
- Abort timeout (`SMOKE_ABORT_TIMEOUT = 2`): unchanged
- Watchdog installation mechanism (`signal.SIGALRM`, `signal.alarm`): unchanged
- Watchdog handler message format: updated to reflect 1200s in the timeout
  message string (`_TIMEOUT_REASON`), because the handler constructs the
  message from `SMOKE_TIMEOUT_SECONDS` at runtime — no separate string edit
  is needed if the handler reads the constant (Architect confirms)
- Smoke log/report/artifact destinations: unchanged
- Default `smoke-train`, `full-train`, `continue-train` command catalog
  entries: unchanged in source, behavior, identity
- `vendor/mlx-lm/` inner files: unchanged, inner gitlink unchanged
- `segmented_loss_and_grad.py`, `test_ds4_segmented_loss_and_grad.py`,
  `test_mlx_lm_source.py`: unchanged

## R14.3b-2 — Allowed production/docs/test file scope

### Authorized production change

1. `scripts/ds4_segmented_smoke.py` — change `SMOKE_TIMEOUT_SECONDS = 600` to
   `SMOKE_TIMEOUT_SECONDS = 1200`. If the timeout message string is hardcoded
   separately from the constant, that string also changes to reflect 1200s.
   No other edit in this file.

### Authorized docs/test scope

1. `docs/backlog.md` — Story 14.3b user story + acceptance criteria (this BA
   update).
2. `tests/test_ds4_segmented_smoke.py` or equivalent tracked test file —
   synthetic tests proving the constant is 1200 and the timeout message/report
   reflects 1200s. Exact test scope pinned by Architect.
3. `docs/architecture.md` or an ADR — only if the Architect determines the
   timeout change constitutes a durable boundary change. If not, no ADR is
   required.
4. `agent-output/cmux-14-3-timeout/` handoff artifacts and `.cmux-status/`
   markers.

### Forbidden changes

- Any file in `vendor/mlx-lm/`
- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`
- `tests/test_ds4_segmented_loss_and_grad.py`
- `tests/test_mlx_lm_source.py`
- Any C/Objective-C/Metal/CUDA/ROCm/distributed/SSD/disk-cache file
- `scripts/finetune_ds4.py` (unless Architect identifies a durable catalog
  constant referencing the smoke timeout that must be kept in sync — if so,
  only the constant value changes, nothing else)
- Any environment definition, package manifest, or config template
- Path A evidence, Epic 13 evidence, ADR 0028

## R14.3b-3 — Synthetic test contract

Tracked synthetic tests must prove:

1. `SMOKE_TIMEOUT_SECONDS` equals `1200` (direct constant assertion or import
   check). Mutation-sensitive: changing the constant to any other value
   (including 600) makes the test RED.

2. The timeout handler message or report `failure_message` field contains
   `1200s` (not `600s`) when the timeout fires. This can be tested by
   mocking the signal or calling the handler function directly.

3. `SMOKE_LOCK_TIMEOUT_S` remains `60` (unchanged from Story 14.3). This is a
   regression guard ensuring the lock timeout was not accidentally changed.

4. `SMOKE_ABORT_TIMEOUT` remains `2` (unchanged). Same regression guard.

5. All other pinned smoke parameters in the smoke command shape
   (`--iters 1`, `--batch-size 1`, `--learning-rate 1e-5`,
   `--max-seq-length 4096`, `--mask-prompt`, `--grad-checkpoint`,
   `--segment-size 1`) are present and unchanged in the command/catalog
   string. This is a regression guard ensuring the timeout repin did not
   accidentally alter any other parameter.

No real model, dataset, or `/Volumes/Data NVME/...` path is read during these
tests. Tests are pure constant/string/catalog assertions.

## R14.3b-4 — Non-claims

This repin does NOT claim or authorize:

- Convergence, loss quality, or generalization.
- OOM repair or command-buffer lifetime fix.
- Correctness of the provider's segmented math on real data.
- Throughput or speed.
- Full-training readiness.
- Any result beyond one bounded 4096-token microbatch.
- A retry of the smoke. The 1200s timeout is a single attempt; if it times out
  again, failure is terminal STOP/ESCALATE.
- A parameter change other than the timeout.

## R14.3b-5 — STOP/ESCALATE criteria

Emit error JSON and write `custom-handoffs/14-3-timeout/ba-stop.md` if any of:

- The timeout change cannot be isolated to `SMOKE_TIMEOUT_SECONDS` and its
  runtime-derived message (i.e., the constant is entangled with other logic
  that must change).
- The 1200s value would require changing lock timeout, abort timeout, preflight
  thresholds, or any parameter other than the hard timeout.
- Any real asset read/execution is required to finish the requirements
  themselves (BA stage).
- Predecessor identity, protected evidence, or canonical backlog cannot be
  reconciled.
- Scope materially exceeds one constant change + corresponding docs/tests.

## Acceptance criteria

1. `custom-handoffs/14-3-timeout/requirements.md` exists with testable
   timeout-change (R14.3b-1), scope (R14.3b-2), synthetic test (R14.3b-3),
   non-claims (R14.3b-4), and STOP (R14.3b-5) requirements with no unresolved
   placeholder. **Proof: file exists and every R14.3b-* section has concrete
   criteria.**

2. `docs/backlog.md` contains one Story 14.3b user story in the exact form:
   `As a [type of user] (WHO), I want [some goal] (WHAT), so that [some reason]
   (WHY).` **Proof: grep for `14.3b` and `As a.*WHO.*I want.*WHAT.*so
   that.*WHY` near Story 14.3b in `docs/backlog.md`.**

3. The only production change specified is `SMOKE_TIMEOUT_SECONDS` from 600 to
   1200 in `scripts/ds4_segmented_smoke.py`; all other pinned parameters,
   paths, locks, abort conditions, and catalog entries are explicitly
   preserved. **Proof: R14.3b-1 preserve list is exhaustive; R14.3b-2
   forbidden list covers all other files.**

4. No real smoke, training, inference, or `/Volumes/Data NVME/...` access is
   authorized or required during BA, architecture, or implementation. **Proof:
   R14.3b-3 tests are pure constant/string/catalog assertions; R14.3b-2
   scope does not include execution.**

5. The repin explicitly states it does not claim convergence, OOM repair, or
   correctness beyond one bounded attempt. **Proof: R14.3b-4 non-claims
   section.**

6. All BA verdict files and backlog edits are tracked/staged; no commit or
   push. **Proof: `git ls-files -- custom-handoffs/14-3-timeout/requirements.md
   docs/backlog.md` returns both; `git diff --stat` shows no unstaged changes
   to these files; `git log --oneline -1` shows no new commit from BA.**