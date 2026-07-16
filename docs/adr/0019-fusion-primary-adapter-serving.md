# ADR 0019: Fusion is the primary (Q)LoRA adapter-serving bridge; runtime `--lora` is an unimplemented gap

Date: 2026-06-21
Status: Accepted

## Context

The DS4 fine-tuning project's end goal is that a (Q)LoRA-trained adapter runs on
the antirez DS4 C-engine (DwarfStar) on Apple Silicon. Two facts drove a
strategic pivot on 2026-06-21 (Story 12.1 reconciliation, parent-pre-verified
F1/F2/F6):

1. **The serving foundation is already proven (F1).** The Track-A marker
   `/Users/spotted/projects/ds4/.ds4-gguf-generate-ok` is present (2026-06-19):
   the C-engine/Metal runtime smoke-generates the immutable base
   `ds4flash.gguf` with no dependency on the fine-tuning stack (ADR 0008 Track A).
   This is the serving foundation Epic 12 stands on.

2. **Runtime `--lora` does not exist (F2).** `ds4_cli.c` rejects unknown flags;
   there is zero LoRA-apply code in the C engine (`ds4.c`/`ds4_metal.m`/
   `metal/*.metal`/`ds4_server.c`). `scripts/convert_lora_to_ds4.py` is a
   safetensors **tensor-key rename only** tool — it copies tensor bytes
   byte-for-byte, stores `alpha` as metadata (`ds4_metadata["ds4_lora_alpha"] =
   str(float(lora_alpha))`, L333), and never multiplies. `python-envs/mlx/src/
   ds4_ft_mlx/lora_targets.py` defines an allowlist (`q_a`/`q_b`/`kv`) with no
   runtime consumer. So the prior backlog framing ("runtime LoRA primary,
   fusion deferred") is false and is inverted: **fusion is the primary and only
   currently-viable serving bridge; runtime `--lora` is the deferred
   unimplemented gap.**

The MLX B-track (training: `fp8-shim → convert-shimmed → model-4bit → mlx_lm.lora
train`) remains blocked (F5, blockers B0–B3; `full_forward_parity=false`), so
there is no real trained adapter yet. However, the fusion bridge's
**contract** — how a trained adapter reaches the C-engine once it exists — must
be specified now so no slice, reviewer, or operator proceeds on the false
"runtime `--lora`" assumption. The `fuse` step declared in
`scripts/finetune_ds4.py` L953 has three resolution gaps (G1 format / G2 gate /
G3 base) that must be closed by design.

No existing ADR covers this. ADR 0001 codifies the Metal production path but
not the adapter bridge. ADR 0008 codifies Track-A/Track-B independence and the
base-serving foundation but not the adapter-serving bridge or the runtime-`--lora`
gap. ADR 0002 binds parity-first/fail-closed/marker-honesty. This ADR fills the
gap.

## Decision

1. **The C-engine Metal runtime is the canonical serving runtime (re-affirms ADR
   0001).** A trained adapter reaches it via **offline FUSION**, not via a live
   `--lora` flag:
   ```
   adapter → fuse (numpy-delta on the original HF F8 checkpoint, TODAY;
                    OR mlx_lm fuse --dequantize, future)
          → HF safetensors dir → deepseek4-quantize --hf → fused GGUF (new)
          → ds4 -m <fused.gguf> --metal
   ```
   The fused GGUF is loaded by the **same** Track-A serving path proven by F1;
   fusion is a **bridge A←B** (consumes Track-B's adapter output, feeds
   Track-A's serving), **not a third track**. It inherits **neither** track's
   marker (ADR 0008 track-independence).

2. **The today-viable bridge is the numpy-delta bypass, not `mlx_lm fuse`.**
   `mlx_lm fuse --dequantize` is **not** today-viable: it requires a loadable
   MLX DS4 model (G3/B-track), a key-remap (`self.*`→`layers.N.*`), and
   `deepseek4-quantize` expert-BF16 support (its `dequant_fp4_weight` hard-dies
   on non-`I8`+`F8_E8M0` experts). `mlx_lm fuse --export-gguf` is rejected
   outright for DS4 (it emits generic llama-arch GGUF missing the `deepseek4.*`
   metadata `ds4.c` requires). The numpy-delta helper reads the **original HF F8
   checkpoint** safetensors directly (no `mlx_lm` load → no G3 blocker →
   B-track-independent for a synthetic-delta smoke), applies the same merge math
   `mlx_lm`'s `LoRALinear.fuse()` uses (`delta = ((scale * lora_b.T) @
   lora_a.T)`, `scale = alpha/rank` pre-baked), emits the fused attention weights
   as BF16 (dropping their companion `F8_E8M0` scale tensors), and byte-copies
   everything else (experts stay `I8`+`F8_E8M0`; reader-accepted). This feeds
   the existing, unchanged `deepseek4-quantize --hf` reader and `ds4 -m` path.

3. **Fusion targets are the 3 trained attention modules only — `q_a`/`q_b`/`kv`
   — NOT `lm_head`.** `lora_targets.py` `SUPPORTED_ALIASES = ("q_a","q_b","kv")`
   forbids `lm_head`/`output`/embeddings/MLP/experts. (The `output`/`lm_head`
   entry in `convert_lora_to_ds4.py` `SUPPORTED_TARGETS` serves the DS4 adapter
   rename format for the non-existent runtime `--lora` path; it is never
   trained and never fused.) The fuse base is the **original HF F8 checkpoint**
   (experts `I8`+`F8_E8M0`, attention `F8_E4M3`+`F8_E8M0` — both reader-accepted),
   **not** `hf-f8shim` (which shims experts to BF16 and would die in the
   quantizer).

4. **The `fuse` model-4bit gate (`scripts/finetune_ds4.py` L4382–4383) is NOT
   lifted.** The `fuse` (mlx_lm) step is properly Track-B-gated; lifting it would
   conflate Track A/Track B (an unproven-MLX-forward base reaching the served
   GGUF via `mlx_lm.fuse`, reusing neither marker as evidence). The numpy-fuse
   bridge is a **separate `fuse-hf` step** that bypasses the gate entirely
   (never reads `model-4bit`) and carries its **own** evidence gate
   (`validate_fused_hf_safetensors_dir`), reusing no marker. The fused serving is
   proven by the end-to-end smoke (Track-A-style evidence: exit 0 + real text +
   determinism + base immutability), not by a borrowed marker.

5. **Runtime `--lora` is an unimplemented gap, explicitly deferred.** It would
   require new production code on the Metal runtime (ADR 0001 production path):
   a per-layer apply kernel (`W + scale·B^T@A^T` or KV-aware), a runtime
   consumer for the `convert_lora_to_ds4.py` adapter format, and
   `ds4_cli.c`/`ds4_server.c` `--lora` plumbing. It is a **FUTURE Epic 12
   sub-track**, deferred until a proven frequent-adapter-swap need is
   demonstrated. The re-quant cost of fusion (76–164 GB file per adapter
   change) is acceptable for the user's infrequent train→serve workflow and
   argues against premature runtime-`--lora` investment.

6. **never-mutate-`ds4flash.gguf` invariant holds.** Fusion writes a **new**
   explicitly-named GGUF; `ds4flash.gguf` is never mutated in place
   (`docs/backlog.md` L296, Story 7.4 AC — byte-intact per BA reconciliation).
   Pairs with ADR 0008's base-GGUF immutability assertion in
   `ds4_gguf_base_smoke_check`.

## Consequences

- **What gets easier:** the fusion serving bridge is contract-valid TODAY without
  any production-code change and without unblocking the MLX B-track. A
  synthetic-delta end-to-end smoke (alpha=0 parity anchor + small-nonzero run)
  can be run B-track-independently once the `scripts/fuse_lora_hf.py` helper is
  coded (FROZEN-design in `agent-output/cmux-12-1/architecture.md` §6.1) and the
  76–164 GB resource is available. The bridge reuses the proven Track-A serving
  path (F1) and the unchanged `deepseek4-quantize --hf` reader (BF16 attention
  via `tensor_to_f32`, `I8`+`F8_E8M0` experts via `dequant_fp4_weight`).

- **What stays hard / out of scope here:** the `mlx_lm fuse --dequantize` clean
  MLX-native path is deferred (needs B-track unblock + key-remap + quantizer
  expert-BF16 support). Runtime `--lora` is deferred (FUTURE Epic 12 sub-track;
  UN-FROZEN — a future slice must FROZEN-design it first per 11.46 cascade
  discipline). The double-quant dtype round-trip (BF16-fuse → Q2 re-quant) is
  acceptable for small-LoRA-delta **pending** the parity smoke passing tolerance
  (≥95% token identity OR logit cosine ≥ 0.999 for the alpha=0 anchor); on
  failure, fail-closed per ADR 0002 (block fusion serving).

- **Marker discipline (re-affirms ADR 0002 + ADR 0008):** fusion inherits no
  marker. `.ds4-gguf-generate-ok` stays scoped to base-only generation;
  `.deepseek-v4-forward-parity-ok` + `model-4bit` stay scoped to Track B. Any
  future fusion-serving marker (e.g. `.ds4-fusion-smoke-ok`) is an **UN-FROZEN
  cascade** — a future slice must FROZEN-design it (path, writer, evidence
  binding) before writing it; STOP + escalate per 11.46 if attempted
  half-spec'd. The fused serving before such a marker exists is proven only by
  the smoke's Track-A-style evidence sidecar, never by a borrowed marker.

- **Scope guard:** `scripts/finetune_ds4.py`'s `fuse` step (L953) and its
  `model-4bit` gate (L4382–4383) are NOT lifted/edited by this ADR — they stay
  Track-B-gated. The numpy-fuse bridge is a NEW separate step + helper + gate
  (`fuse-hf` + `scripts/fuse_lora_hf.py` + `validate_fused_hf_safetensors_dir`),
  FROZEN-design'd in `agent-output/cmux-12-1/architecture.md` §6, for a future
  Coder slice. `deepseek4-quantize.c` and `ds4.c` are NOT edited (the numpy
  bridge reuses the existing `--hf` reader + serving path unchanged). CPU/SSD/
  CUDA/distributed paths untouched.

## Cross-refs

- **ADR 0001** (whole-model Metal graph = production path) — `ds4 -m
  <fused.gguf> --metal` is that path; runtime `--lora` gap sits on this
  production runtime.
- **ADR 0002** (parity-first, fail-closed, marker evidence) — fusion smoke
  marker must be marker-honest; runtime `--lora` fail-closed until proven; the
  Q8 dtype round-trip parity gate binds.
- **ADR 0008** (two-track parity gates, Track A = DS4-GGUF base smoke,
  Track B = MLX forward parity) — fusion bridge consumes Track-B adapter output,
  feeds Track-A serving, inherits **neither** marker; track-independence
  invariant binds; base-GGUF immutability assertion pairs with the L296
  never-mutate invariant.
- **ADR 0005** (agent-output not canonical) — `agent-output/cmux-12-1/` is
  handoff evidence; `docs/backlog.md` is canonical.
- **ADR 0006** (canonical backlog) — the BA reconciliation ledger flips (4 false
  `[x]` → `[ ]`; Story 5.3/7.4 inverted runtime-LoRA-primary → fusion-primary;
  L296 never-mutate + L969 11.25 provenance preserved byte-intact).

## Implementation references (not part of this ADR's decision; FROZEN-design for a future slice)

> **ADR Status (`Accepted`) + core Decision (fusion primary; runtime `--lora`
> unimplemented gap) are UNCHANGED by the micro-revision below.** The 2026-06-21
> Reviewer (xhigh) BLOCK on Story 12.1 surfaced 5 byte-defects in the FROZEN
> §6 byte-spec (defects in the FUTURE-code DESIGN spec, not in shipped code).
> `agent-output/cmux-12-1/architecture.md` §6.4 (MICRO-REVISION) supersedes the
> §6.1/§6.2 byte-level contract; this section is amended to point there. No
> production code, marker, readiness-JSON, or proof-counter was touched.

- `agent-output/cmux-12-1/architecture.md` **§6.4 MICRO-REVISION (2026-06-21,
  Reviewer BLOCK repair)** — authoritative corrected byte-spec; supersedes
  §6.1/§6.2 byte-level contract (helper file + step/gate wiring in §6.1/§6.2
  still stand as the FROZEN future-code STRUCTURE; only the byte-spec details
  are superseded). §6.1/§6.2 cited ONLY for the structure now.
  Corrected byte-spec (see §6.4 for full evidence):
  - **Per-format delta + shapes (B1+B3):** MLX internal adapter — lowercase
    `lora_a` `(input_dims, r)` + `lora_b` `(r, output_dims)`, config field
    `lora_parameters.{rank,scale,dropout,keys}`, scale pre-baked, formula
    `delta = (scale * (lora_b.T @ lora_a.T)).astype(weight_dtype)`. HF/PEFT
    synthetic adapter — `lora_A.weight` `(r, in)` + `lora_B.weight` `(out, r)`,
    config `lora_alpha` + `target_modules`, formula
    `delta = (lora_alpha / rank) * (lora_B.weight @ lora_A.weight)`.
  - **Adapter-config parsing (B2):** detect format from `adapter_config.json` —
    `lora_parameters` object ⇒ MLX; `lora_alpha` + `target_modules` ⇒ PEFT.
  - **HF scale tensor name (B4):** `.scale`, NOT `.weight_scale`; derived via
    `s/.weight$/.scale/` (enumerated keys: `layers.N.attn.wq_a.scale`,
    `layers.N.attn.wq_b.scale`, `layers.N.attn.wkv.scale`).
  - **Per-target A/B pairing (B5):** each of `q_a`/`q_b`/`kv` pairs its OWN
    `lora_A`+`lora_B` (MLX `lora_a`+`lora_b`); do NOT cross-pair a target's A
    with another target's B.
  - **hf-f8shim rejection precise reason:** the blocker is the expert SCALE
    dtype `BF16` (not `F8_E8M0`); the expert WEIGHT stays `I8`. `die` fires at
    `deepseek4-quantize.c` L716, not on BF16 expert weights.
- `mlx_lm/tuner/lora.py` — `LoRALinear.fuse` delta line (~L52):
  `delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)`;
  `LoRALinear.__init__` shapes (~L88-93): `lora_a` `(input_dims, r)`, `lora_b`
  `(r, output_dims)`; defaults `r=8`, `scale=20.0`, `dropout=0.0` (scale
  pre-baked — authoritative MLX merge-math precedent).
- `mlx_lm/tuner/utils.py` L60-62 + L80-82: `r=config["rank"]`,
  `scale=config["scale"]`, `dropout=config["dropout"]`; `~L134` `lora_parameters`
  config path (MLX adapter config is `lora_parameters.{rank,scale,dropout,keys}`,
  NOT HF `lora_alpha`/`target_modules`).
- `scripts/finetune_ds4.py` L3961-3969: `build_lora_parameters(rank=8,
  scale=20.0, dropout=0.0)` → `"lora_parameters": lora_parameters` (MLX adapter
  config writer).
- `deepseek4-quantize.c` L660–676 (`tensor_to_f32` BF16 branch), L683 (F8 attn
  dequant requires `F8_E4M3` weight + `F8_E8M0` scale), **L716**
  (`dequant_fp4_weight` `die("bad FP4 weight/scale dtype")` — experts require
  `I8` weight + `F8_E8M0` scale; this is the line `hf-f8shim` fails on because
  its expert scale is BF16), L922–953 (`layer_map` HF input keys — preserve
  verbatim), L1183–1186 (scale-name derivation `.weight`→`.scale`, regex
  `s/.weight$/.scale/`), L1615–1640 (`write_full_gguf` — unchanged downstream).
- `scripts/convert_lora_to_ds4.py` L33–41 (suffix map; key-rename only),
  L158–161, L332–333 (`ds4_lora_alpha` metadata-only — confirms zero merge
  math in the converter).
- `python-envs/mlx/src/ds4_ft_mlx/lora_targets.py` (`SUPPORTED_ALIASES =
  ("q_a","q_b","kv")`; `lm_head`/`output`/embeddings/MLP forbidden).
