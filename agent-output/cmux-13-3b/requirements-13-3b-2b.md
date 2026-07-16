# Requirements — Story 13.3b-2b (CSA attention wiring: compressor real keys + block_bias + KV-append + multi-head)

BA. THIN. Architect r0 owns design (`architecture.md` §3.4 unified `_attention_real_mlx`
CSA+HCA, §3.5 RoPE parity, §2.1 reuse map, §8 frozen list + ADR 0026). This doc =
LOCKED Q-list + acceptance tests + blast-radius + STOP-rules + rope-landmine +
**ape-transpose re-adjudication**. NO design. Caveman-ultra. Code/paths/line-numbers/numbers byte-exact.

FROZEN file: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`.
torch ref: `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`.
real cfg: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`.
Predecessor: 13.3b-2a complete (HEAD `a2c20b0`). SECOND half SPLIT 13.3b-2. FIRST half
13.3b-2a (`_indexer_mlx:620` + `_indexer_scorer_mlx:591`) GREEN, CONSUMED here for block_bias.

---

## §0 GO VERDICT (BA recon, resolved) — **GO**

Load-bearing AC2 concern: "can CSA attention branch (CSA compressor real-keys cr=4 +
block_bias from `_indexer_mlx` + KV-append + multi-head) reach parity with torch
`DeepseekV4Attention.forward` CSA layer, additively, no FROZEN-primitive gap?" BA verified
against torch source. Findings:

1. **CSA compressor = SINGLE-head, out_dim=head_dim=512** (NOT multi-head). torch
   `CSACompressor.forward` (modeling:623-702) produces `compressed [B,n_win,head_dim=512]`
   → `compressed_kv = compressed.unsqueeze(1)` = `[B,1,n_win,512]` (modeling:686). The
   "multi-head" is the ATTENTION (q has `num_attention_heads=64` heads broadcasting
   against the single `num_key_value_heads=1` kv head), NOT the compressor. Identical
   single-kv-head pattern to HCA `_hca_compressor_mlx:806` (`expand_dims(compressed, 1)`).
2. **Ca/Cb overlap at rate=4, 2*head_dim=1024** — SAME pooling structure as shipped
   `_indexer_mlx:679-695` but out_dim=512 (not 128). All MLX primitives proven 13.3b-2a.
3. **Rope landmine = option α (reimplement inline + compress-yarn-tail)** — CONFIRMED
   (see §0.1). Identical resolution to 13.3b-2a. No new kernel.
4. **block_bias build = scatter from `_indexer_mlx` top-k** (modeling:693-702). MLX has
   no native `scatter_` but `mx.put_along_axis` / one-hot-comparison emulate it; all
   primitives present. AC graded set-based-on-indexer + exact mask values.
5. **KV-append + multi-head + grouped-o = REUSE HCA `_attention_real_mlx:870-910`** byte-pattern.
   CSA uses SAME `o_groups=8` grouped output (torch `Attention` is layer-type-agnostic
   for o-proj, modeling:870-872).

⇒ No FROZEN-primitive gap → AC2 STOP does NOT fire → 13.3b-2b proceeds. **No `ba-stop.md`.**

### §0.1 ROPE LANDMINE re-adjudication (carryover 13.3b-2a) — **Option α LOCKED**

FROZEN `_csa_windowed_compressor_mlx:347` bakes full-channel plain-θ rope
(`_rope_full_tables_mlx:300` + `_apply_rope_full_mlx:311`, deepseek_v4.py:394-395,415).
torch real CSA compressor rope = **compress-yarn over TRAILING `qk_rope_head_dim=64`**
of head_dim=512 (`CSACompressor.rope_layer_type="compress"`, modeling:610; rotary
dim = `head_dim=512 * partial_rotary_factor=0.125 = 64`, modeling:139-141;
`apply_rotary_pos_emb:356-357` rotates only last `rope_dim=64`), theta=`compress_rope_theta=160000`
+ yarn factor=16. ⇒ FROZEN windowed compressor rope **MISMATCHES** torch.

- **Option α (LOCKED)**: reimplement the Ca/Cb overlap pooling INLINE in the new CSA
  attention helper (additive, NOT a FROZEN body edit), apply compress-yarn-tail via
  REUSED 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266`,
  positions `arange(n_win)*rate`. SAME pattern as shipped `_indexer_mlx:677-707`,
  consistent with Architect §3.5 parity-for-all-compress-layers.
- **Option β (REJECT)**: reuse FROZEN `_csa_windowed_compressor_mlx` end-to-end →
  wrong rope → parity broken.

> **§2.1↔§3.5 tension (FLAGGED, resolved):** Architect §2.1 reuse-map said "CSA
> compressor = FROZEN `_csa_windowed_compressor_mlx:347` (out_dim parametric)" — but
> that was BEFORE BA 13.3b-2a discovered the baked-in-rope mismatch. §3.5 (RoPE parity
> for ALL compress layers, theta=160000+yarn) wins. Option α is consistent with §3.5 and
> with the shipped `_indexer_mlx` precedent. **Editing FROZEN
> `_csa_windowed_compressor_mlx` body to swap its rope is FORBIDDEN (S-frozen-body, Q7).**

### §0.2 ape-transpose RE-ADJUDICATION (brief Q2 premise INVERTED) — **NO transpose at helper input**

> **BRIEF CORRECTION (cite torch evidence).** Task brief §3 + Q2 claimed: "torch
> `CSACompressor` uses `ape[:out_dim,:].T` — DIFFERENT from HCA no-transpose". **This is
> WRONG at the torch math layer.** torch `CSACompressor.forward:648`:
> `chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias`
> adds `position_bias` shape `(compress_rate=4, 2*head_dim=1024)` (modeling:618)
> **DIRECTLY**, token-major, **NO `.T`, NO `[:out_dim,:]` slice** — IDENTICAL to HCA
> `HCACompressor.forward:648`→wait HCA modeling:415 `+ self.position_bias` shape
> `(rate, head_dim)` also direct.
>
> The `ape[:out_dim,:].T` the brief refers to lives ONLY inside the FROZEN MLX helper
> `_csa_windowed_compressor_mlx:387-388` — and ONLY because that helper STORES ape
> convert-transposed `(2*out_dim, rate)` (its assert :368) and transposes it back to
> token-major internally. **Option α does NOT call that FROZEN helper** → it consumes ape
> token-major `(rate, 2*out_dim)` DIRECTLY, EXACTLY like the shipped `_indexer_mlx:680`
> (`+ weights["indexer_compressor_ape"].reshape((1,1,rate,2*out_dim))`, NO transpose).
>
> **LOCK: the CSA compressor `ape`/`position_bias` is consumed token-major
> `(rate=4, 2*head_dim=1024)` with NO transpose** at helper input (synthesize the test
> fixture directly so; the convert-side transpose architecture.md §5.2 is a 13.3b-4
> concern ONLY for feeding the FROZEN helper, which option α bypasses). HCA (13.3b-1)
> and CSA (here) and indexer (13.3b-2a) ALL consume ape token-major no-transpose under
> option α. The brief's "HCA=NO / CSA=YES transpose mirror" is VOID — both NO.

---

## §A — LOCKED Q-list (Coder MUST follow; Architect owns *design*, BA pins *contract*)

### Q1 — CSA compressor: single-head shape + Ca/Cb overlap cr=4 + real ckpt keys + rope (§3.4, landmine)

**LOCKED — option α (reimplement inline).** Port torch `CSACompressor.forward`
(modeling:635-686), `past_key_values=None`/`cache_layer=None` branch:

- `out_dim = head_dim = 512`; `rate = compress_rate_csa = 4` (cfg `compress_ratios`
  = `4` on the 21 CSA layers: 2,4,…,40,42). SINGLE kv head (§0.1).
- `kv = hidden @ wkv.T`, `gate = hidden @ wgate.T`, both `[B,S,2*head_dim=1024]`
  (modeling:635-636). Uses **`hidden_states`** for kv/gate (NOT q_residual; q_residual
  feeds the indexer only, modeling:693).
- `usable = (S//rate)*rate`; `first_window_position=0`; `n_windows = usable//rate`
  (modeling:639-645).
- reshape `[B, n_win, rate, 1024]`; `chunk_gate += position_bias` (ape `(rate=4,
  2*head_dim=1024)` token-major, **NO transpose** §0.2) (modeling:648).
- **Ca/Cb overlap build** (modeling:650-662): `new_kv[B, n_win, 2*rate, head_dim]`;
  `new_kv[:,:,rate:] = chunk_kv[..., head_dim:]` (Cb = trailing 512 of 1024, current
  window); `new_kv[:, 1:, :rate] = chunk_kv[:, :-1, :, :head_dim]` (Ca = leading 512
  of *previous* window); window 0 Ca = zero-kv / `-inf`-gate (softmax weight 0). IDENTICAL
  Ca/Cb layout to shipped `_indexer_mlx:681-692` at out_dim=512 (vs 128).
- `compressed = kv_norm( (new_kv * softmax(new_gate, axis=2, dtype=fp32).cast).sum(axis=2) )`
  `[B, n_win, 512]` (modeling:673-675). `kv_norm` = RMSNorm over head_dim=512.
- **RoPE (landmine §0.1)**: compress-yarn-tail TRAILING-64 of head_dim=512, positions
  `arange(n_win)*rate` (modeling:676-680), via REUSED `_compress_rope_yarn_tail_tables_mlx:545`
  + `_apply_rope_tail_mlx:266` (`head_dim=512`, `qk_rope_head_dim=64`). Mirror shipped
  `_hca_compressor_mlx:795-805` rope block.
- `compressed_kv = compressed[:, None]` `[B,1,n_win,512]` (modeling:686).

**BA-verified real ckpt keys** (safetensors header, ONE real CSA layer 2, NO tensor load —
Coder re-assert shape/name only):

| key | shape | dtype |
|---|---|---|
| `layers.2.attn.compressor.wkv.weight` | `(1024, 4096)` = `(2*head_dim, hidden)` | BF16 |
| `layers.2.attn.compressor.wgate.weight` | `(1024, 4096)` | BF16 |
| `layers.2.attn.compressor.ape` | `(4, 1024)` = `(rate, 2*head_dim)` | F32 |
| `layers.2.attn.compressor.norm.weight` | `(512,)` = `(head_dim)` | BF16 |

(indexer ckpt keys `layers.2.attn.indexer.compressor.*` etc. ALREADY consumed/asserted
13.3b-2a `requirements-13-3b-2a.md` Q1 — reuse `_indexer_mlx` as-is, do NOT re-port.)

> **Ca/Cb WARNING for Coder:** do NOT call FROZEN `_csa_windowed_compressor_mlx:347`
> end-to-end (it bakes wrong rope §0.1). Reimplement the overlap pool inline mirroring
> the shipped `_indexer_mlx:677-707` block at out_dim=512, then apply compress-yarn-tail.
> Editing the FROZEN helper body to swap rope is FORBIDDEN (S-frozen-body, Q7).

### Q2 — CSA ape transpose — **RE-ADJUDICATED: NO transpose** (see §0.2)

**LOCKED.** The brief's "torch CSA `ape[:out_dim,:].T`, DIFFERENT from HCA" premise is
INVERTED (§0.2). torch `CSACompressor.forward:648` adds `position_bias (rate=4,
2*head_dim=1024)` DIRECTLY (modeling:618 def). Under option α (inline, FROZEN helper
bypassed) the CSA compressor `ape` is consumed **token-major `(rate=4, 2*head_dim=1024)`,
NO `.T`, NO `[:out_dim,:]` slice** — IDENTICAL to shipped `_indexer_mlx:680`. Synthesize
the test fixture `ape` directly as `(4, 1024)`. The HCA-vs-CSA transpose-mirror the brief
posited does NOT exist; both are no-transpose under option α.

### Q3 — block_bias build from `_indexer_mlx` top-k (§3.4 — the CSA-specific piece)

**LOCKED.** Consume shipped `_indexer_mlx(args, hidden, q_residual, weights, *,
position_ids, index_topk)` → `top_k_indices [B,S,k]` (incl `-1` sentinels, int32; HEAD
`a2c20b0`, deepseek_v4.py:620-749). Port torch `CSACompressor.forward:693-702`:

- `compressed_len = n_win` (the CSA compressor's window count = S//rate, SAME rate=4 as
  indexer ⇒ aligns with `compressed_kv` axis-2 length).
- `valid = top_k_indices >= 0` `[B,S,k]` (modeling:695).
- `safe_indices = where(valid, top_k_indices, compressed_len)` — clamp `-1` sentinel to
  the extra OOB column `compressed_len` (modeling:699).
- `block_bias = full((B, 1, S, compressed_len + 1), -inf)` (modeling:700).
- `scatter_(-1, safe_indices[:,None], 0.0)` — set every picked (valid) window → `0.0`,
  the rest stay `-inf` (modeling:701). The extra `+1` column absorbs all `-1` clamps,
  then is DROPPED.
- `block_bias = block_bias[..., :compressed_len]` → `[B,1,S,compressed_len]` (modeling:702).
- **dtype**: `block_bias` built at attention-score dtype (compute path), fill values
  `0.0` (visible) / `-inf` (masked). EXACT mask values (no tolerance).
- **MLX scatter**: no native `scatter_`; emulate via `mx.put_along_axis(block_bias,
  safe_indices_expanded, 0.0, axis=-1)` OR one-hot/`mx.max` over picks. Coder owns op
  choice; AC = mask-value parity vs torch given the SAME `top_k_indices`.
- **mask combine** (CSA attention path, mirror HCA `_attention_real_mlx:872-876`):
  `mask = concat([sliding_causal_mask, block_bias], axis=-1)` `[B,1,S,S+n_win]`. block_bias
  is CONCATENATED to the mask (NOT added to raw scores separately) — torch
  `Attention.forward:840-842` cats block_bias onto attention_mask. The sliding causal
  part = FROZEN `_causal_sliding_mask_mlx:914` (REUSE, as HCA does).

### Q4 — KV append (§3.4 step 3 — reuse HCA `_attention_real_mlx:870`)

**LOCKED.** `kv_full = concat([kv_by_head, compressed_kv], axis=2)` (modeling:832,
`torch.cat([kv, compressed_kv], dim=2)`). axis=2 (window/seq axis). Ordering: `kv` first
(real tokens), `compressed_kv` appended. `kv_by_head = expand_dims(kv, 1)` `[B,1,S,512]`;
`compressed_kv [B,1,n_win,512]` → `kv_full [B,1,S+n_win,512]`. SINGLE kv head broadcasts
over 64 q-heads at score time. IDENTICAL to HCA `_attention_real_mlx:869-870`. REUSE.

### Q5 — multi-head attention + grouped-o (§3.4 step 5-6 — REUSE HCA tail)

**LOCKED — REUSE HCA `_attention_real_mlx:880-910` byte-pattern.** torch
`Attention.forward` o-projection is layer-type-AGNOSTIC (modeling:846-872; same
`eager_attention_forward` + `apply_rotary_pos_emb(-sin)` rope-undo + `o_a_proj`(grouped)
+ `o_b_proj` for sliding/CSA/HCA). ⇒ CSA path:

- `scale = head_dim**-0.5`; `q_by_head = q.transpose(0,2,1,3)` `[B,64,S,512]`.
- `scores = (q_by_head @ kv_full.transpose(0,1,3,2)) * scale + mask` (mask from Q3).
- per-head sink: `sink_logits = sinks.reshape((1,64,1,1))`; concat to scores; softmax;
  drop sink col; `attended = probs @ kv_full` (modeling:717-742 eager + 854-858, sinks
  `[64]`).
- rope-undo: `_apply_rope_tail_mlx(attended, cos, -sin, qk_rope_head_dim=64)` (compress-yarn-tail
  cos/sin from step-1 q/kv rope, NEGATED sin) (modeling:868).
- **grouped-o (o_groups=8)**: SAME block-diagonal `o_a_proj` (shape `(o_groups*o_lora_rank,
  heads_per_group*head_dim)` = `(8*1024, 8*512)`) + `o_b_proj` `(hidden=4096,
  o_groups*o_lora_rank=8192)` as HCA `_attention_real_mlx:892-910`. CSA does NOT differ.
  REUSE. (GroupedLinear-vs-cr0 parity confirm = 13.3b-3, OUT here — CSA reuses the
  proven HCA grouped-o code path.)
- q/kv rope = compress-yarn-tail (CSA layer rope_layer_type="compress", modeling:777),
  SAME as HCA `_attention_real_mlx:855-865`. REUSE.

### Q6 — CSA attention helper shape + standalone + fixtures + anti-circularity (§3.4, pattern 13.3b-1/2a)

**LOCKED — NEW `_csa_attention_real_mlx` helper (additive), NOT a branch edit to
`_attention_real_mlx`.** Rationale: 13.3b-1 `_attention_real_mlx:828-834` hard-guards
`compression_ratio != 128 → NotImplementedError` + `index_topk is not None →
NotImplementedError`. Adding a CSA branch INSIDE `_attention_real_mlx` REQUIRES relaxing
those guards = EDITING the 13.3b-1 HCA-branch body bytes (S-frozen-body risk, brief's
"HCA branch body MUST stay byte-identical"). A NEW sibling helper `_csa_attention_real_mlx`
is PURELY additive, keeps HCA bytes intact, and 13.3b-3 (which OWNS dispatch) wires
cr=4→`_csa_attention_real_mlx`, cr=128→`_attention_real_mlx`.

> **§3.4 "unified `_attention_real_mlx`" tension (FLAGGED, resolved):** Architect §3.4
> sketched a unified helper, written BEFORE the 13.3b-1/2a split materialized
> `_attention_real_mlx` as an HCA-only symbol with hard guards. Additive-only +
> byte-intact-HCA-branch (AGENTS.md, brief S-frozen-body) ⇒ NEW sibling helper now.
> 13.3b-3 MAY refactor to a unified dispatch at wiring time (its call); 13.3b-2b ships
> the CSA branch as an additive HELPER. Consistent with subslice-breakdown.md 13.3b-2b
> ("CSA branch of `_attention_real_mlx`" = the CSA real path, produced additively).

Signature LOCK (Architect-final at review): `_csa_attention_real_mlx(args, x, weights,
*, index_topk) -> mx.array` — mirror `_attention_real_mlx:819-825` signature; internally
compute q_a/q_residual/q/kv (REUSE :845-865 math), call inline CSA compressor (Q1) +
`_indexer_mlx` (Q3) + build block_bias (Q3) + KV-append (Q4) + multi-head/grouped-o (Q5).

**Fixtures + anti-circularity** (mirror 13.3b-1 `test_attention_real_mlx_hca_parity.py` +
13.3b-2a `test_indexer_mlx_parity.py`):
- Synthesize random weights at real CSA dims: `head_dim=512`, `num_attention_heads=64`,
  `num_key_value_heads=1`, `o_groups=8`, `o_lora_rank=1024`, `q_lora_rank=1024`,
  `qk_rope_head_dim=64`, `compress_ratio=4`, `index_topk=512`, `index_n_heads=64`,
  `index_head_dim=128`, `hidden=4096`, `compress_rope_theta=160000`, yarn `{factor:16,
  beta_fast:32, beta_slow:1, original_max:65536}`, `rms_norm_eps=1e-6`,
  `sliding_window=128`. Use `S > rate` (e.g. `S=257` like 13.3b-1/2a) so `n_win = S//4 > 0`
  and early-query sentinel/causal-threshold paths fire.
- **Anti-circularity (ADR 0007 §4, proven 13.3b-1/2a):** torch reference computes expected
  from the SAME synthesized weights/config (load into `DeepseekV4CSACompressor` /
  `DeepseekV4Attention` CSA layer, `past_key_values=None`). MLX helper reads the SAME
  arrays. torch is the ONLY oracle — NO second MLX primitive as oracle.
- **ONE real cross-check:** assert synthesized CSA-compressor key-names/shapes match real
  CSA ckpt layer 2 safetensors header (Q1 table) — shape/name only, NO tensor load, NO
  162GB read. Mirror `_real_header_shape` skip-if-absent pattern.

### Q7 — STOP conditions (architecture.md §9 + ADR 0026 + 13.3b-2a pattern)

**LOCKED.** Emit `coder-stop.md` + `{"status":"error",...}` if:

- **(S-rope-csa)**: CSA compressor rope needs a contract FROZEN cannot supply additively.
  BA recon: does NOT fire — option α = compress-yarn-tail = REUSED 13.3b-1 helper, proven
  13.3b-2a. STOP only if code disproves.
- **(S-frozen-body)**: any FROZEN/upstream symbol BODY would need EDIT (not additive-new) →
  STOP. Forbidden bodies incl: `_csa_windowed_compressor_mlx:347` (even reused — byte-identical;
  do NOT edit to swap rope, Q1), `_attention_mlx` cr=0 body, **`_attention_real_mlx:819` HCA
  branch body (MUST stay byte-identical; CSA ships as NEW sibling helper, Q6)**,
  `_hca_compressor_mlx:752`, **`_indexer_mlx:620` + `_indexer_scorer_mlx:591` (13.3b-2a —
  byte-identical, CONSUMED as-is)**, `_compress_rope_yarn_tail_tables_mlx:545`,
  `_apply_rope_tail_mlx:266`, `_apply_rope_full_mlx:311`/`_rope_full_tables_mlx:300`,
  `_csa_attention_mlx:515`, `_csa_compressor_mlx:419`, `_csa_indexer_mlx:435`,
  `_csa_config_error:317`, `_require_csa_config:335`, `_hyperconnection_mlx`,
  `_hyperhead_mlx`, `_causal_sliding_mask_mlx:914`, parity `Model:2059`, `sanitize_weights`,
  9 ADR-0017 forbidden symbols. ADDITIVE ONLY.
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) gain ANY introduced RED → STOP.
- **(S-parity-csa)**: CSA attention output cannot match torch within tol after best-effort →
  STOP with failing assertion + torch-vs-MLX delta.
- **(S-backward)**: 13.3a-3 backward AC gains introduced RED → STOP.
- **byte-intactness**: FROZEN bodies byte-intact (source-hash + AST, not zero-line diff —
  AGENTS.md untracked caveat). `git diff --check` clean.

---

## §B — Acceptance criteria → trackable tests (TDD red-first)

- **AC1** — `tests/test_csa_compressor_real_mlx_parity.py` (NEW): the inline CSA compressor
  (Ca/Cb overlap cr=4 out_dim=512 + correct compress-yarn-tail rope) output
  `compressed_kv [B,1,n_win,512]` matches torch `DeepseekV4CSACompressor.forward`
  (`past_key_values=None`) within tol at real dims (head_dim=512, rate=4, hidden=4096).
  Q1 overlap + Q2 NO-transpose + §0.1 rope. (torch CSACompressor also runs its indexer at
  modeling:693; the AC1 fixture either compares ONLY the compressed_kv slice OR compares
  the full `(compressed_kv, block_bias)` tuple — Coder/Architect pin; compressed_kv parity
  is the load-bearing AC1 assert.)
- **AC2** — `tests/test_csa_attention_real_mlx_parity.py` (NEW): `_csa_attention_real_mlx`
  (compressor + block_bias from `_indexer_mlx` + KV-append + multi-head + grouped-o) output
  `[B,S,hidden=4096]` matches torch `DeepseekV4Attention.forward` for a CSA layer within tol
  at real dims. Q3 block_bias + Q4 KV-append + Q5 multi-head/grouped-o + Q6 anti-circular
  header cross-check. block_bias mask values graded EXACT given the SAME `_indexer_mlx`
  top_k_indices; attention output graded within tol (cross-framework BF16).
- **tolerances** (mirror 13.3b-1/2a): cross-framework BF16 — compressed_kv / attention
  output **atol=1e-2** (rotated path); intra-fp32 cos/sin / mask values **EXACT or 1e-5**;
  block_bias `0`/`-inf` mask values **EXACT** (no tolerance).
- **AC convention (a)(b)(c)(d)** per subslice-breakdown: (a) parity vs torch ported math;
  (b) tiny CSA fixtures (11.14/11.15) GREEN; (c) backward AC (13.3a-3) GREEN; (d)
  `git diff --check` clean + FROZEN bodies byte-intact (AST/grep proof).
- **regression**: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2 incl `_indexer_mlx` reuse) + 13.3a
  nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite:
  **563+2new** PASS / **1** RED (pre-existing Test #8) / **13** SKIP / **0 introduced RED**.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST. `git diff --check`
  clean. **sha-pin cascade SLICE-INVARIANT**: deepseek_v4.py edited additively (NEW
  `_csa_attention_real_mlx` + inline CSA compressor) → advance EXACTLY the right count of
  sha16 pin sites referencing deepseek_v4.py content. **Reviewer verifies the count using
  the rigorous whole-tree scan, NOT Test Manager 13.3b-1/2a's grep-subset.**

---

## §C — Blast-radius (Architect §8 + subslice-breakdown — confirmed set)

| # | File | Change | Sanction |
|---|---|---|---|
| 1 | `deepseek_v4.py` | ADDITIVE ONLY: NEW `_csa_attention_real_mlx` helper (inline CSA compressor option α + block_bias build from `_indexer_mlx` + KV-append + multi-head/grouped-o), in 13.3b module section beside `_attention_real_mlx:819`. NO body edits to ANY FROZEN/13.3b-1/13.3b-2a helper. NO dispatch branch in `_attention_mlx` (13.3b-3). | ADR 0026 additive |
| 2 | `tests/test_csa_compressor_real_mlx_parity.py` | NEW (AC1) | — |
| 3 | `tests/test_csa_attention_real_mlx_parity.py` | NEW (AC2) | — |

**OUT of scope (explicit fences):** dispatch branch in `_attention_mlx` cr=4→CSA /
cr=128→HCA (13.3b-3); GroupedLinear-vs-cr0 parity confirm + `_grouped_linear_mlx`
(13.3b-3 — CSA REUSES proven HCA grouped-o); `AttentionNN` guard lift + nn-port edits
(13.3b-3); unify `_csa_attention_real_mlx`+`_attention_real_mlx` (13.3b-3 dispatch call);
convert/remap (13.3b-4); smoke-train (13.3b-5).

**FROZEN — MUST NOT EDIT BODY** (Q7 list): `_csa_windowed_compressor_mlx:347`,
`_attention_real_mlx:819` (HCA branch byte-identical), `_hca_compressor_mlx:752`,
`_indexer_mlx:620`, `_indexer_scorer_mlx:591`, `_compress_rope_yarn_tail_tables_mlx:545`,
`_apply_rope_tail_mlx:266`, `_apply_rope_full_mlx:311`, `_rope_full_tables_mlx:300`,
`_attention_mlx` cr=0 body, `_csa_attention_mlx:515`, `_csa_compressor_mlx:419`,
`_csa_indexer_mlx:435`, `_csa_config_error:317`, `_require_csa_config:335`,
`_causal_sliding_mask_mlx:914`, `_hyperconnection_mlx`, `_hyperhead_mlx`, parity
`Model:2059`, `sanitize_weights`, 9 ADR-0017 forbidden symbols.

---

## §D — STOP-ESCALATE verdict

surface:81 only. AC2 CSA-attention recon = **SOLVABLE** (option α compressor proven
13.3b-2a; block_bias scatter primitives present; KV-append/multi-head/grouped-o REUSE
HCA; ape NO-transpose §0.2; rope REUSED). No FROZEN-primitive gap. No `ba-stop.md`. Story
13.3b-2b proceeds to Architect review (design §3.4) → Coder.

## Deliverables (this BA pass)
1. THIS doc.
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. `docs/backlog.md` 13.3b-2b row.
4. In-pane JSON `{"status":"ok","role":"BA"}` surface:81 only.
