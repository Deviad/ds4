# ADR 0008: Two-track parity gates (DS4-GGUF base smoke-generate vs MLX shimmed-checkpoint forward parity)

Date: 2026-06-19
Status: Accepted

> **Note on numbering:** this is ADR **0008**, not 0007. ADR 0007
> (`0007-expert-block-geometry-shape-authoritative.md`, landed by Story 11.15c)
> is unchanged. `docs/architecture.md` and `docs/backlog.md` cite this ADR as 0008.

## Context

The plan had a **single correctness gate** — `.deepseek-v4-forward-parity-ok` —
that conflated two genuinely independent questions:

1. **Can the real DS4/Metal runtime smoke-generate from the immutable base
   `ds4flash.gguf`?** — a *deployment/runtime* question. The base GGUF and the
   `ds4` binary both exist today, and this track is independent of any
   fine-tuning stack (no MLX, no conversion, no `model-4bit`, no adapter).
2. **Can MLX load/forward the shimmed FP8 checkpoint with full parity, enabling
   `convert-shimmed` → `model-4bit` → MLX adapter training?** — a
   *training-prep* question. It remains blocked by routed expert-decode dispatch
   (11.15c follow-up), `hc_mult>1` multi-layer forward (11.11), and real-scale
   integration.

Because the MLX `smoke-generate` story (which needs `model-4bit`, which does not
exist) was the only "smoke-generate" referenced, the DS4 runtime's own
base-generation capability had **no independent gate** and was implicitly held
hostage to the entire MLX stack. That conflates blocker #4 (the classic
"missing base smoke gate") with the MLX forward-parity blockers.

This violates two existing decisions:
- **ADR 0001** — the Metal/DS4 runtime is the production path; its correctness
  should be proven on its own track, not gated behind the training-prep stack.
- **ADR 0002** — each gate must be parity-first and **independently**
  fail-closed, bound to proven evidence (not aspirations). A single conflation
  marker prevents either question from being answered on its own merits.

The base GGUF and `ds4` binary both exist, so Track A **could be proven today**;
the conflation was the only thing holding it back.

## Decision

Split the single conflated gate into **two independent tracks with distinct
markers**, neither creating or validating the other:

- **Track A — DS4-GGUF base smoke-generate.** The real `ds4`/Metal runtime
  smoke-generates from the immutable base `ds4flash.gguf` (no adapter, no
  conversion, no `model-4bit`). **Marker: `.ds4-gguf-generate-ok`** (new). This
  is the gate for Story 11.19 (DS4-GGUF base smoke impl).
- **Track B — MLX shimmed-checkpoint forward parity.** Existing track; gates
  `convert-shimmed` → `model-4bit` → MLX adapter `smoke-train`/`smoke-generate`.
  **Marker: `.deepseek-v4-forward-parity-ok`** (unchanged semantics, now scoped
  to the MLX track only).

**Track-independence invariant (core):** neither track creates or validates the
other's marker.
- Track A success **never** creates `model-4bit` or
  `.deepseek-v4-forward-parity-ok`.
- Track B success **never** creates `.ds4-gguf-generate-ok`.
- `convert-shimmed` remains gated on **Track B only**.

Both markers remain parity-first and bound to proven evidence per ADR 0002:
- Track A: prompt/output + `ds4` binary/version + determinism (same prompt ⇒
  byte-stable tokens across runs).
- Track B: parity-report hash (the proven forward-parity run). A marker without
  its bound evidence is rejected before any dependent step runs.

Track A is **distinct from** Story 5.5 (final fused/mixed GGUF after
fine-tuning + splice) and Story 7.1 (LoRA adapter loading beside the base):
Track A is **base-only runtime generation** — no adapter, no fusion, no training
dependency.

## Consequences

- **What gets easier:** DS4 base-generation smoke (Story 11.19) can be proven
  **now**, unblocked from the MLX stack. It becomes a foundation for Story 7.x
  runtime-LoRA deployment smoke and gives a real deployment-generation signal
  today (base + binary already exist).
- **What stays hard / out of scope here:** the MLX fine-tune path
  (`convert-shimmed` → `model-4bit` → adapter training) is **still gated** on
  Track B; Track A does not relax any Track-B blocker. MLX `smoke-generate`
  remains blocked until routed expert-decode dispatch, `hc_mult>1` multi-layer
  forward, and real-scale integration land.
- **Track-B blocker reword (completed by Story 11.20):** the
  `forward_parity_blockers()` 4th tuple item was reworded from the conflated
  `"full model load/forward over the shimmed checkpoint and generation smoke"`
  to the MLX/shimmed-Track-B scoped `"full MLX shimmed-checkpoint load/forward
  and MLX generation smoke (Track B only; DS4-GGUF base generation is its own
  Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)"`, dropping
  DS4-GGUF base generation as a forward-parity blocker per this ADR. The other 3
  items, the tuple length (4), and the order are byte-unchanged; the 4 asserting
  test sites were updated consistently (one renamed to
  `test_forward_parity_blockers_track_b_scoped` with a negative regression guard
  against the old conflation substring).
- **Marker honesty (ADR 0002):** `.ds4-gguf-generate-ok` is written **only** by
  Story 11.19's proven base-generation gate, never by the spec and never
  speculatively. It stays absent until that gate actually runs and binds its
  evidence. `.deepseek-v4-forward-parity-ok` and `model-4bit` likewise stay
  absent until their own Track-B gates are proven.

### Marker path pin (Story 11.19)

The Track-A marker `.ds4-gguf-generate-ok` is pinned to
**`/Users/spotted/projects/ds4/.ds4-gguf-generate-ok`** (`DS4_ROOT`, the home of
the `ds4` binary + base `ds4flash.gguf`) — deliberately separate from the
MLX-track marker dir `/Volumes/Data NVME/mlx-ft/ds4/`. It is written **only** by
Story 11.19's gate helper `ds4_gguf_base_smoke_check(args)` in
`scripts/finetune_ds4.py`, which runs one bounded, deterministic, production-Metal
invocation (`ds4 -m ds4flash.gguf -p <prompt> -n 12 --temp 0 --metal`) under an
opt-in env gate (`DS4_GGUF_BASE_SMOKE=1`), clears any stale marker first, asserts
`returncode==0` + real generated text + base-GGUF immutability, and writes the
marker bound to a sidecar evidence record (prompt, `-n`, binary sha256/version,
GGUF realpath/size/header-sha256, stdout excerpt, ISO timestamp). The gate skips
cleanly (skip ≠ proof, no marker) when the env gate is unset or either artifact
is missing; it never imports torch/MLX and never touches the Track-B chain.
