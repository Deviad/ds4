# Story/Epic 13.3b — Sub-slice breakdown

Architect r0. Per-slice scope / blast-radius / AC / STOP-rules / dev-days.
STOP-rule applied: any slice >7 dev-days is split. Total envelope drives Epic re-scope
(architecture.md §10). All slices SERIAL (file-mutating coder, reviewed before next).

AC convention (every slice): (a) parity vs torch `modeling_deepseek_v4.py` reference for
the ported math; (b) tiny CSA fixtures (11.14/11.15) STAY GREEN; (c) backward AC (13.3a-3)
STAYS GREEN; (d) `git diff --check` clean + FROZEN helper bodies byte-intact (AST/grep proof).

---

## 13.3b-1 — HCA real path + RoPE oracle  (FIRST; Epic go/no-go)

- **Scope:** NEW `_hca_compressor_mlx` (torch HCACompressor:362, stateless branch);
  NEW `_attention_real_mlx` HCA branch (multi-head MLA + compressed-KV append +
  block_bias, reusing cr=0 `_attention_mlx:632` body); RoPE oracle pinning compress-rope
  contract (theta=160000 + yarn; full vs trailing slice — architecture.md §3.5).
- **Blast-radius:** `deepseek_v4.py` ADD helpers (no body edits); NEW tests. No nn-file
  edit yet (helper-level + functional tests).
- **AC:** (a) `_hca_compressor_mlx` output matches torch `DeepseekV4HCACompressor.forward`
  (past_key_values=None) within tol on a real-dim fixture (head_dim=512, rate=128);
  (a') HCA `_attention_real_mlx` matches torch `DeepseekV4Attention.forward` for an HCA
  layer; (a'') rope oracle: MLX compress-rope cos/sin == torch for one compress layer;
  + (b)(c)(d).
- **STOP-rules:** compress rope needs a math/kernel primitive FROZEN cannot supply
  additively → STOP (flips port→kernel work). Any tiny test RED → STOP.
- **Dev-days:** 5 (oracle 1.5 + HCA compressor 1.5 + HCA attn 2). ≤7 OK.

## 13.3b-2 — CSA + Indexer real path  (SPLIT-armed)

- **Scope:** NEW `_indexer_scorer_mlx` (torch IndexerScorer:446); NEW `_indexer_mlx`
  (torch Indexer:462 — own compressor via FROZEN `_csa_windowed_compressor_mlx` reuse,
  q rope, top-k, `-1` sentinel + clamp, future-mask); CSA branch of `_attention_real_mlx`
  (FROZEN windowed compressor real keys → compressed_kv; indexer top-k → `[B,1,S,T]`
  block_bias; KV append; multi-head attn). torch CSACompressor:589 + Attention:805.
- **Blast-radius:** `deepseek_v4.py` ADD helpers; NEW tests.
- **AC:** (a) `_indexer_mlx` top-k indices == torch `DeepseekV4Indexer` indices (incl `-1`
  sentinels) on real-dim fixture (index_n_heads=64, index_head_dim=128, index_topk=512,
  rate=4); (a') CSA `_attention_real_mlx` matches torch CSA Attention.forward; + (b)(c)(d).
- **STOP-rule SPLIT (PRESCRIBED, est >7d):** split into
  - **13.3b-2a Indexer** (`_indexer_scorer_mlx` + `_indexer_mlx`, top-k parity): 4 d.
  - **13.3b-2b CSA attention** (compressor real-keys + block_bias build + multi-head): 4 d.
- **Dev-days:** 8 → SPLIT 4 + 4. (MLX masked-topk-with-sentinel emulation is the cost driver.)

## 13.3b-3 — GroupedLinear parity + dispatch + AttentionNN guard lift + integration

- **Scope:** confirm cr=0 o_a application == torch `DeepseekV4GroupedLinear.forward:328`
  bmm at o_groups=8 (REUSE if identical; else NEW `_grouped_linear_mlx`). ADD additive
  dispatch branch in `_attention_mlx:640` (architecture.md §4.2). LIFT `AttentionNN:153`
  guard; add compressor/indexer submodules + real `__call__` branch. End-to-end nn forward
  on a real-dim multi-layer (sliding+CSA+HCA) tiny-but-real-shaped config.
- **Blast-radius:** `deepseek_v4.py` ADD dispatch branch (additive); `deepseek_v4_nn.py`
  `AttentionNN` edits (SANCTIONED); NEW tests.
- **AC:** (a) GroupedLinear parity vs torch; (a') nn forward over mixed-layer-type config
  produces finite `mx.array` `[B,S,vocab]` and matches functional helpers; (b)(c)(d);
  (e) dispatch: tiny input still routes `_csa_attention_mlx` (assert branch unchanged).
- **STOP-rule:** dispatch edit makes any tiny input change branch → STOP.
- **Dev-days:** 4. ≤7 OK.

## 13.3b-4 — real ckpt remap script + load

- **Scope:** NEW `scripts/remap_ds4_nn_weights.py`: rename table (architecture.md §5.1),
  CSA/indexer ape TRANSPOSE `(rate,2*out)→(2*out,rate)`, per-expert→stacked `(n_routed,…)`,
  drop redundant BF16 `.scale` sidecars (keep FP4 expert scales), `model.` prefix,
  w1=gate/w3=up/w2=down shared-expert rename, mtp drop. Then `model.load_weights` on nn
  param tree (after `sanitize`).
- **Blast-radius:** NEW convert script; NO FROZEN/nn-port code edits (convert side only).
- **AC:** remapped key set == nn `model.parameters()` key set exactly (no missing/extra);
  load succeeds strict; one real CSA + one real HCA + one sliding layer load + forward
  finite; (b)(c) regression; (d). NO full 162GB run required for grading (sub-checkpoint /
  single-shard or shape-only assertion acceptable, mirror 13.3a-3 Q9 unit-style).
- **STOP-rule:** ckpt carries a tensor with no nn param target (unknown arch piece) → STOP.
- **Dev-days:** 3. ≤7 OK.

## 13.3b-5 — smoke-train 43-layer real

- **Scope:** `mlx_lm.lora --train --iters 20` against remapped real ckpt; confirm loss
  finite + decreasing-ish, adapters save/load. Memory: FP4 experts stay quantized
  (ADR 0024), LoRA attention-only.
- **Blast-radius:** NONE (run-only; uses prior slices). NEW test/run script.
- **AC:** train loop runs ≥20 iters without NaN/OOM; loss finite; adapter round-trips;
  (b)(c) regression suite still GREEN post-run.
- **STOP-rule:** OOM on Mac Studio M3 Ultra at real dims → STOP, escalate (memory re-scope).
  Avoid large CPU runs (AGENTS.md Safety).
- **Dev-days:** 2–3. ≤7 OK.

---

## Totals

| slice | dev-days |
|---|---|
| 13.3b-1 HCA + rope oracle | 5 |
| 13.3b-2a Indexer | 4 |
| 13.3b-2b CSA attention | 4 |
| 13.3b-3 GroupedLinear + dispatch + guard lift + integration | 4 |
| 13.3b-4 remap + load | 3 |
| 13.3b-5 smoke-train | 2–3 |
| **total** | **22–23 dev-days ≈ 4.4–4.6 weeks** |

> Exceeds 3-week single-story ceiling → 13.3b re-scoped to **Epic** with five gated
> stories (architecture.md §10). 13.3b-1 is go/no-go: its rope oracle confirms FROZEN
> reuse is viable before committing the rest of the Epic.
