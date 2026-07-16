# Requirements — Story 13.3b-2a (Indexer real path: top-k + -1 sentinel + future-mask)

BA. THIN. Architect r0 owns design (`architecture.md` §3.2 `_indexer_scorer_mlx`,
§3.3 `_indexer_mlx`, §2.1 reuse map, §8 frozen list + ADR 0026). This doc =
LOCKED Q-list + acceptance tests + blast-radius + STOP-rules. NO design.
Caveman-ultra. Code/paths/line-numbers/numbers byte-exact.

FROZEN file: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`.
torch ref: `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`.
real cfg: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`.
Predecessor: 13.3b-1 complete (HEAD `1f5d8e4`). FIRST half SPLIT 13.3b-2
(Architect STOP-rule 8d → 2a indexer 4d / 2b CSA attention 4d). 13.3b-2b
(CSA attention wiring, block_bias build, multi-head) is a LATER slice — OUT here.

---

## §0 GO VERDICT (BA recon, resolved) — **GO**

AC2 load-bearing concern: "does MLX top-k + -1 sentinel + future-mask match
torch `Indexer.forward:577-586` additively, no FROZEN-primitive gap?" BA verified
all emulation primitives exist in `mx.core`: `mx.argsort` / `mx.argpartition`
(desc via negate), `mx.take_along_axis` (gather), `mx.where` (sentinel),
`mx.full`, fp32 cast. torch `topk` tie order is implementation-defined; MLX
sort/partition will NOT reproduce torch's exact tie order — but parity is graded
**set-based** (SET of non-sentinel picks + their scores, NOT exact tie ordering),
exactly the FROZEN `_csa_indexer_mlx:489` Python-sort precedent. **No
FROZEN-primitive gap → AC2 STOP does NOT fire → 13.3b-2a proceeds. No `ba-stop.md`.**

Rope landmine (resolved GO): torch indexer rope (q + compressed_kv) =
**compress-yarn over TRAILING qk_rope_head_dim=64** of `index_head_dim=128`
(`DeepseekV4Indexer.rope_layer_type="compress"`, modeling:491; rotary dim =
`config.head_dim=512 * partial_rotary_factor=0.125 = 64`, modeling:139-141;
`apply_rotary_pos_emb:357` rotates only last `rope_dim=64`). This is the SAME
compress-yarn-tail contract 13.3b-1 proved (`_compress_rope_yarn_tail_tables_mlx:545`).
It is **NOT** the FROZEN `_csa_windowed_compressor_mlx:347` baked-in full-channel
plain-theta rope (`_rope_full_tables_mlx` + `_apply_rope_full_mlx`), nor the
FROZEN tiny `_csa_indexer_mlx:436` full-channel rope. ⇒ NEW `_indexer_mlx`
must apply compress-yarn-tail rope via REUSED 13.3b-1 helper; achievable
additively (proven 13.3b-1). No new kernel.

---

## §A — LOCKED Q-list (Coder MUST follow; Architect owns the *design*, BA pins the *contract*)

### Q1 — indexer compressor: overlap semantics + dims + real ckpt keys (§3.3)

**LOCKED.** Indexer compressor = Ca/Cb **OVERLAP** windowed pooling at
`out_dim=index_head_dim=128`, `compress_rate=rate=4` (CSA layer; cfg
`compress_ratios` = `4` on CSA layers, layer 2..41 alternating). Port torch
`DeepseekV4Indexer.forward:519-555` (stateless `cache_layer=None` branch):

- `kv = x @ kv_proj.T`, `gate = x @ gate_proj.T`, both `[B, S, 2*index_head_dim=256]`
  (modeling:519-520).
- `usable = (S // rate) * rate`; `chunk = [:, :usable]`; `first_window_position=0`
  (modeling:523-524).
- `n_windows = usable // rate`; reshape `[B, n_win, rate, 256]`; `chunk_gate +=
  position_bias` (ape `(rate, 2*index_head_dim)` token-major, NO transpose at
  helper input — convert-side transpose is 13.3b-4 CSA/indexer-only) (modeling:531-532).
- Ca/Cb overlap build (modeling:535-541): `new_kv[B, n_win, 2*rate, 128]`;
  `new_kv[:,:,rate:] = chunk_kv[..., head_dim:]` (Cb = trailing 128 of 256);
  `new_kv[:, 1:, :rate] = chunk_kv[:, :-1, :, :head_dim]` (Ca = leading 128 of
  *previous* window); window 0 Ca = zeros / gate=-inf. Identical Ca/Cb layout to
  FROZEN `_csa_windowed_compressor_mlx:387-407` at `out_dim=index_head_dim=128`.
- `compressed = kv_norm( (new_kv * softmax(new_gate, axis=2, fp32)).sum(axis=2) )`
  `[B, n_win, 128]` (modeling:548-549).
- **RoPE (LANDMINE, see Q2)**: compress-yarn-tail at positions `arange(n_win)*rate`,
  NOT FROZEN windowed full-plain rope.
- `compressed_kv` `[B, n_win, 128]` feeds `_indexer_scorer_mlx` (Q3).

BA pin (Coder assert via safetensors header, ONE real CSA layer 2, NO tensor load,
shape/name only — verified by BA 2026-06-26):

| key | shape | dtype |
|---|---|---|
| `layers.2.attn.indexer.compressor.wkv.weight` | `(256, 4096)` = `(2*index_head_dim, hidden)` | BF16 |
| `layers.2.attn.indexer.compressor.wgate.weight` | `(256, 4096)` | BF16 |
| `layers.2.attn.indexer.compressor.ape` | `(4, 256)` = `(rate, 2*index_head_dim)` | F32 |
| `layers.2.attn.indexer.compressor.norm.weight` | `(128,)` = `(index_head_dim)` | BF16 |
| `layers.2.attn.indexer.weights_proj.weight` | `(64, 4096)` = `(index_n_heads, hidden)` | BF16 |
| `layers.2.attn.indexer.wq_b.weight` | `(8192, 1024)` = `(index_n_heads*index_head_dim, q_lora_rank)` | BF16 |

> **Ca/Cb overlap WARNING for Coder:** FROZEN `_csa_windowed_compressor_mlx:347`
> gives the *correct overlap pooling math* BUT bakes its own rope
> (`_rope_full_tables_mlx` full-channel + plain `compress_rope_theta`, no yarn),
> which MISMATCHES torch real indexer compress-yarn-trailing-64 rope. So you
> CANNOT call FROZEN `_csa_windowed_compressor_mlx` end-to-end and get parity.
> Resolution (Architect §3.3, additive, mirrors how 13.3b-1 `_hca_compressor_mlx`
> handled the non-overlap case): in the NEW `_indexer_mlx`, build the overlap
> pool inline (or factor an additive helper) and apply rope via the REUSED
> `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx`. **Editing
> the FROZEN `_csa_windowed_compressor_mlx` body to swap its rope is FORBIDDEN
> (S-frozen-body STOP, see Q7).**

### Q2 — indexer q + compressed-kv RoPE: full-vs-trailing? yarn? positions? (§3.3)

**LOCKED — REUSE 13.3b-1 helper.** Both q-rope and compressed_kv-rope use the
**compress-yarn over TRAILING `qk_rope_head_dim=64`** of `index_head_dim=128`,
positions differ:

1. **Channel scope = TRAILING 64** (NOT full 128). torch `apply_rotary_pos_emb:357`
   rotates last `rope_dim` channels; `rope_dim = config.head_dim=512 *
   partial_rotary_factor=0.125 = 64` (modeling:139-141). Leading 64 of
   index_head_dim untouched. ⇒ REUSE FROZEN `_apply_rope_tail_mlx` /
   `_broadcast_rope_tail_table_mlx` with `qk_rope_head_dim=64`.
2. **Yarn = YES**, identical contract to 13.3b-1 (theta=`compress_rope_theta=160000`,
   factor=16, beta_fast=32, beta_slow=1, original_max=65536, attention_factor=1.0).
   ⇒ REUSE 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` with
   `head_dim=index_head_dim=128`, `qk_rope_head_dim=64`. **Do NOT** reuse
   `_rope_full_tables_mlx` (that is the tiny/self-consistent FROZEN path, wrong
   for real parity).
3. **q computation** (modeling:564-565): `q = (q_residual @ wq_b.T).view(B, S,
   index_n_heads=64, index_head_dim=128)`; apply compress-yarn-tail rope at
   `positions = position_ids` (= `arange(seq_len)` stateless). `wq_b` shape
   `(8192, 1024)` so `q_residual[B,S,1024] @ wq_b.T[1024,8192]` → `[B,S,8192]` →
   reshape `[B,S,64,128]`.
4. **compressed_kv rope** (modeling:551-555): `positions = arange(n_win)*rate +
   first_window_position(0)`; apply compress-yarn-tail rope on `[B, n_win, 128]`.
5. **fp32 strategy**: rope tables fp32 (helper already), apply in input dtype;
   downstream scorer casts fp32 anyway (Q3).

### Q3 — `_indexer_scorer_mlx` math + weights_proj shape (§3.2)

**LOCKED.** Port torch `DeepseekV4IndexerScorer.forward:455-459`, fp32 accum:

- `scores = relu( einsum('bShd,bTd->bShT', q.float(), compressed_kv.float()) ) *
  index_head_dim**-0.5` → `[B, S, index_n_heads=64, T]` (modeling:456-457;
  `softmax_scale = index_head_dim**-0.5`, modeling:451). NOTE q enters scorer as
  `[B, S, H, index_head_dim]` (post-rope, post-`.transpose(1,2)` modeling:565);
  `compressed_kv` `[B, T, index_head_dim]`, broadcast over H via `.unsqueeze(1)`
  (modeling:456).
- `weights = (hidden @ weights_proj.T).float() * index_n_heads**-0.5` →
  `[B, S, H]` (modeling:458; `weights_scaling = index_n_heads**-0.5`,
  modeling:452). `weights_proj.weight` shape `(index_n_heads=64, hidden=4096)`
  (BA-verified ckpt) ⇒ `hidden[B,S,4096] @ weights_proj.T[4096,64]` → `[B,S,64]`.
- `return (scores * weights[..., None]).sum(axis=2)` → `[B, S, T]` (modeling:459).
- **fp32 accum** throughout (torch `.float()` on q, compressed_kv, weights). MLX:
  cast inputs `mx.float32`, return fp32 `[B,S,T]` (downstream topk wants fp32).

### Q4 — top-k + -1 sentinel + future-mask: exact recipe (§3.3, HARD part)

**LOCKED.** Port torch `DeepseekV4Indexer.forward:567-586`. MLX has NO native
masked-topk-with-sentinel → emulate (all primitives exist, §0):

1. `index_scores = _indexer_scorer_mlx(...)` `[B, S, T]` (Q3).
2. `compressed_len = T`; `top_k = min(index_topk, compressed_len)` (cfg
   `index_topk=512`) (modeling:568-569).
3. **If `compressed_len == 0`**: return `index_scores.topk(...)` is unreachable;
   mirror torch — no `compressed_len>0` branch entered, indexer returns empty /
   the `[B, n_win=0]` path. (Parity fixtures use `S > rate` so `T>0`.)
4. **future-mask** (modeling:577-581): `causal_threshold = (position_ids + 1) //
   rate` `[B, S]`; `entry_indices = arange(T)`; `future_mask = entry_indices[1,1,T]
   >= causal_threshold[B,S,1]` `[B, S, T]`; `index_scores = where(future_mask,
   -inf, index_scores)`. ("future" = compressed window `i` past the query's
   completed-block horizon; `>=` not `>`.)
5. **top-k indices** (modeling:582): `top_k_indices = topk(index_scores_masked,
   top_k, axis=-1).indices` `[B, S, k]`, DESCENDING by score. MLX emulation:
   `mx.argsort(-scores, axis=-1)[..., :top_k]` (or `mx.argpartition` + sort the
   partition), DESC. Ties: MLX order ≠ torch order — **graded set-based** (Q6/AC2).
6. **-1 sentinel** (modeling:583-584): `invalid = top_k_indices >=
   causal_threshold[B,S,1]`; `return where(invalid, -1, top_k_indices)`. Sentinel
   is applied to the **INDEX VALUE** (output is the `[B,S,k]` index tensor with
   `-1` where the pick is still a future/invalid entry — happens for early queries
   with fewer than `top_k` completed blocks). NOT a separate mask. Output dtype
   integer indices.
7. Output `[B, S, top_k]` integer indices incl `-1` sentinels. (Downstream
   13.3b-2b CSA attention clamps/drops `-1`; OUT of scope here.)

> **Tie-break FLAG (Coder + Reviewer + Tester):** at equal scores torch `topk`
> and MLX `argsort`/`argpartition` may order picks differently. AC2 grades the
> SET of valid (non-`-1`) picked indices AND their scores, NOT the exact
> per-position ordering at ties. Mirrors FROZEN `_csa_indexer_mlx:489-505`
> deterministic-but-not-torch-order precedent. Exact-order matching of ties is
> DEFERRED (and is a non-goal — torch order is itself implementation-defined).

### Q5 — standalone helpers, NO wiring (scope fence)

**LOCKED.** 13.3b-2a adds ONLY two NEW additive helpers to FROZEN
`deepseek_v4.py`, in the 13.3b real-port module section beside
`_hca_compressor_mlx:591` / `_attention_real_mlx:658`:

- `_indexer_scorer_mlx(q, compressed_kv, hidden, *, index_n_heads, index_head_dim,
  weights_proj)` → scores `[B,S,T]` (§3.2 signature; Architect-final).
- `_indexer_mlx(args, hidden, q_residual, *, position_ids, index_topk, weights)` →
  top_k_indices `[B,S,k]` incl `-1` (§3.3 signature; Architect-final).

**NO** dispatch branch edit. **NO** wiring into `_attention_real_mlx`. **NO** call
site. The HCA path (`_attention_real_mlx:658`) stays UNCHANGED — it still raises
`NotImplementedError("index_topk is CSA-only...")` for `index_topk is not None`
(deepseek_v4.py:683). Helper-level + functional parity tests ONLY. CSA attention
wiring (block_bias build, kv-append, multi-head consumer of indexer output) is
13.3b-2b. nn-port guard lift is 13.3b-3. convert is 13.3b-4.

### Q6 — fixtures: synthesize real dims + anti-circularity (mirror 13.3b-1 §Q6)

**LOCKED.** Synthesize random weights at real indexer dims: `index_n_heads=64`,
`index_head_dim=128`, `index_topk=512`, `rate=4`, `hidden=4096`,
`q_lora_rank=1024`, `compress_rope_theta=160000`, yarn `{factor:16, beta_fast:32,
beta_slow:1, original_max:65536}`, `rms_norm_eps=1e-6`, `qk_rope_head_dim=64`.
Use `S > rate` (e.g. `S=257` like 13.3b-1) so `T = S//rate > 0` and early-query
sentinel cases (`causal_threshold < top_k`) exercise the `-1` path.

- **Anti-circularity (ADR 0007 §4, proven legit 13.3b-1):** torch reference
  computes expected from the **SAME synthesized weights/config** (load into
  `DeepseekV4IndexerScorer` / `DeepseekV4Indexer`, `past_key_values=None`). MLX
  helper reads the SAME arrays. torch is the ONLY oracle — NO second MLX
  primitive used as oracle.
- **ONE real cross-check:** assert synthesized key-names/shapes match ONE real
  CSA ckpt layer (layer 2) safetensors header (Q1 table) — shape/name only, NO
  tensor load, NO 162GB read. Mirror `tests/test_hca_compressor_mlx_parity.py`
  `_real_header_shape` skip-if-absent pattern.

### Q7 — STOP conditions (Architect subslice-breakdown + ADR 0026)

**LOCKED.** Emit `coder-stop.md` + `{"status":"error",...}` if:

- **(S-rope-indexer)**: indexer rope needs a contract FROZEN cannot supply
  additively (e.g. full-channel yarn where 13.3b-1 helper is trailing-only, AND
  the variant can't be derived additively) → STOP. **BA recon: does NOT fire** —
  indexer rope = trailing-64 + yarn = exactly 13.3b-1 `_compress_rope_yarn_tail_tables_mlx`
  (Q2). Reuse, no new kernel.
- **(S-frozen-body)**: any FROZEN symbol BODY would need EDIT (not additive new
  symbol). Forbidden bodies incl: `_csa_windowed_compressor_mlx:347` (reused
  pooling — do NOT edit to swap rope, Q1 WARNING), `_csa_indexer_mlx:436`,
  `_csa_attention_mlx:515`, `_csa_compressor_mlx:423`, `_attention_mlx`,
  `_attention_real_mlx:658`, `_hca_compressor_mlx:591`,
  `_compress_rope_yarn_tail_tables_mlx:545`, `_apply_rope_*`,
  `_hyperconnection_mlx`, `_hyperhead_mlx`, parity `Model`, `sanitize_weights`,
  the 9 ADR-0017 forbidden symbols → STOP. ADDITIVE ONLY.
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) gain ANY introduced
  RED → STOP.
- **(S-parity-topk)**: AC2 set-based parity unachievable even with set semantics
  (e.g. MLX argsort/argpartition cannot reproduce torch's *valid-pick SET*, not
  just tie-order) → STOP-ESCALATE. **BA recon: does NOT fire** — set membership
  is determined by scores+threshold, framework-agnostic; only tie-ORDER differs,
  which AC2 already excludes.
- **(S-backward)**: 13.3a-3 backward AC gains introduced RED → STOP.
- **byte-intactness**: FROZEN bodies byte-intact (source-hash + AST, not zero-line
  diff — AGENTS.md untracked-file caveat). `git diff --check` clean.

---

## §B — Acceptance criteria (TDD red-first)

- **AC1** `tests/test_indexer_scorer_mlx_parity.py` (NEW): `_indexer_scorer_mlx`
  output `[B,S,T]` matches torch `DeepseekV4IndexerScorer.forward:455` within tol
  at real dims (index_n_heads=64, index_head_dim=128, hidden=4096). fp32 accum +
  `weights_proj` shape `(64,4096)` asserted. Q3.
- **AC2** `tests/test_indexer_mlx_parity.py` (NEW): `_indexer_mlx` top_k_indices
  `[B,S,k]` (incl `-1` sentinels) matches torch `DeepseekV4Indexer.forward:511`
  **SET-based** — for each `[b,t]`, the set of non-`-1` picks AND their scores
  match torch; count of `-1` sentinels matches; NOT exact tie ordering. Future-mask
  + `-1` sentinel + top_k=min(index_topk,T) recipe (Q4) confirmed. Q2 rope-reuse
  + Q1 overlap + Q6 anti-circular header cross-check included.
- **tolerances** (mirror 13.3b-1 BA-pattern: cross-framework BF16, cos/sin 1e-3 /
  rotated 1e-2): scorer scores fp32 **atol=1e-3** (intra path may tighten to
  1e-5 if fp32 end-to-end); future-mask / `-1` sentinel **EXACT** (integer index +
  mask, no tolerance); valid-pick SET **exact membership**, pick-scores **atol=1e-3**.
- **AC convention (a)(b)(c)(d)** per subslice-breakdown: (a) parity vs torch
  ported math; (b) tiny CSA fixtures (11.14/11.15) GREEN; (c) backward AC
  (13.3a-3) GREEN; (d) `git diff --check` clean + FROZEN bodies byte-intact
  (AST/grep proof).
- **regression**: 13.3b-1 (AC0 rope-oracle + AC1 HCA compressor + AC2 HCA attn +
  AC3 dispatch) GREEN; 13.3a nn (12, incl backward AC) GREEN; 13.2 FP4 (9) GREEN;
  tiny CSA (11.14/11.15) GREEN. Full suite: **561+2new** PASS / **1** RED
  (pre-existing Test #8) / **13** SKIP / **0 introduced RED**.

---

## §C — Blast-radius (Architect §8 + subslice-breakdown — confirmed set)

| # | File | Change | Sanction |
|---|---|---|---|
| 1 | `deepseek_v4.py` | ADD `_indexer_scorer_mlx` + `_indexer_mlx` (additive, 13.3b module section). NO body edits to ANY FROZEN helper. NO dispatch branch (13.3b-2b). | ADR 0026 additive |
| 2 | `tests/test_indexer_scorer_mlx_parity.py` | NEW (AC1) | — |
| 3 | `tests/test_indexer_mlx_parity.py` | NEW (AC2) | — |

**OUT of scope (explicit fences):** CSA attention wiring / block_bias build /
kv-append / multi-head indexer consumer (13.3b-2b); `_attention_real_mlx`
dispatch + AttentionNN guard lift + GroupedLinear (13.3b-3); nn-port edits
(13.3b-3); convert/remap (13.3b-4); smoke-train (13.3b-5).

**sha-pin cascade (SLICE-INVARIANT):** 13.3b-2a edits `deepseek_v4.py` (additive
2 helpers) ⇒ the file's content changes ⇒ advance EXACTLY the right number of
sha16 pin sites that reference `deepseek_v4.py` content (per AGENTS.md;
Reviewer verifies the count). Do NOT under- or over-advance.

---

## §D — STOP-ESCALATE verdict

surface:81 only. AC2 top-k recon = **SOLVABLE** (set-based AC + all MLX
primitives present, §0/Q7). No `ba-stop.md`. Story 13.3b-2a proceeds to Architect
review (design already drafted §3.2/§3.3) → Coder.
