# DeepSeek V4 Flash architecture mapping dossier

Status: documentation-only mapping pass for Story 11.1. This dossier does not start conversion or training and does not read safetensors shard payloads. It compares source code and the shimmed checkpoint index only.

## Sources inspected

- DeepSeek V4 Flash config: `/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0/config.json`
- Hugging Face Transformers DeepSeek V4 implementation: `/Volumes/Data NVME/mlx-ft/ds4/.venv/lib/python3.14/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` and `configuration_deepseek_v4.py`
- DeepSeek V4 reference inference code in the HF snapshot: `inference/model.py`
- MLX-LM references: `mlx_lm.models.deepseek_v3`, `mlx_lm.models.deepseek_v32`, and `mlx_lm.models.mla`
- DS4 runtime/layout code: `ds4.c`, `ds4.h`, `ds4_lora.c`, `ds4_lora.h`
- Project MLX scaffold/mapping code: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` and `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_mapping.py`
- Shimmed checkpoint index only: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/model.safetensors.index.json`

Requested `context.md` and `plan.md` were not present in the repository root when this dossier was written.

## DeepSeek V4 Flash shape and schedule summary

From the local Flash `config.json` and Transformers configuration:

| Field | Flash value | Notes |
|---|---:|---|
| `model_type` | `deepseek_v4` | Requires MLX-LM module registration under this exact name. |
| `hidden_size` | 4096 | DS4 Flash shape uses this as `DS4_N_EMBD`. |
| `num_hidden_layers` | 43 | Base block count; does not include MTP. |
| `num_nextn_predict_layers` | 1 | Upstream checkpoint contains MTP tensors; support is not proven. |
| `num_attention_heads` | 64 | Shared KV / MQA attention. |
| `num_key_value_heads` | 1 | V4 uses a single shared KV head, unlike ordinary GQA. |
| `head_dim` | 512 | V4 full Q/KV head dim. |
| `qk_rope_head_dim` | 64 | Trailing partial RoPE slice; `partial_rotary_factor = 64 / 512`. |
| `q_lora_rank` | 1024 | `wq_a -> q_norm -> wq_b`. |
| `o_groups` | 8 | Grouped output projection. |
| `o_lora_rank` | 1024 | Intermediate per-group output dim. |
| `sliding_window` | 128 | Sliding-window branch present in every attention block. |
| `compress_ratios` / `layer_types` | 44 ratios / 43 decoder layer types | Layers 0/1 are `sliding_attention`; layer 42 is `compressed_sparse_attention`. The trailing `compress_ratios[43]=0` is not a normal decoder layer in the 43-layer Transformers schedule. |
| `index_n_heads` / `index_head_dim` / `index_topk` | 64 / 128 / 512 | Lightning Indexer applies to CSA (`compress_ratio == 4`) layers. |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 6 | Routed MoE in every block. |
| `n_shared_experts` | 1 | Shared expert always present. |
| `num_hash_layers` | 3 | First 3 layers use hash routing through `tid2eid`. |
| `expert_dtype` (routed) | `I8` + `F8_E8M0` scale | **Verified (no FP4):** routed experts are signed-I8 weights with 1-D block scales, `block_size=16` on axis 1 (shape-authoritative; e.g. weight `[2048,2048]`, scale `[2048,128]`). See ADR 0007. |
| `expert_dtype` (shared) | `F8_E4M3` + `F8_E8M0` scale | **Verified:** shared experts are F8_E4M3 weights with genuine 2-D `128×128` sub-block scales (e.g. weight `[2048,4096]`, scale `[16,32]`). Matches declared `weight_block_size`. See ADR 0007. |
| `quantization_config` | FP8 e4m3 + UE8M0 scales, `weight_block_size=[128,128]` | Governs the `fmt=e4m3` fp8 path (shared experts + non-expert linears); **advisory, not authoritative** for the I8 routed micro-block (block_size 16). Block geometry is derived from observed weight/scale shapes (ADR 0007). Shim rewrites dtype for safetensors loading only. |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 20 | Hyper-Connections are architectural, not optional residual sugar. |

## Reference architecture comparison

### Transformers DeepSeek V4

Transformers contains real V4 classes: `DeepseekV4RotaryEmbedding`, `DeepseekV4HCACache`, `DeepseekV4CSACache`, `DeepseekV4GroupedLinear`, `DeepseekV4HCACompressor`, `DeepseekV4Indexer`, `DeepseekV4CSACompressor`, `DeepseekV4Attention`, `DeepseekV4HyperConnection`, `DeepseekV4HyperHead`, `DeepseekV4Experts`, `DeepseekV4TopKRouter`, `DeepseekV4HashRouter`, and `DeepseekV4SparseMoeBlock`.

Key semantics to preserve:

- RoPE is interleaved and partial: the trailing `qk_rope_head_dim` slice is rotated; cos/sin are repeated interleaved, not duplicated by halves.
- V4 has three attention layer types: `sliding_attention`, `compressed_sparse_attention` (CSA, rate 4), and `heavily_compressed_attention` (HCA, rate 128).
- CSA uses overlapping two-series compression (`2 * head_dim`) and a Lightning Indexer.
- HCA uses non-overlapping compression (`head_dim`) and no indexer.
- Compressed branches use `compress_rope_theta=160000` and YaRN with attention factor forced to 1.0.
- Core attention is shared-KV MQA: `q_a_proj`, `q_a_norm`, `q_b_proj`, `kv_proj`, `kv_norm`, then grouped output `o_a_proj` and `o_b_proj`.
- Hyper-Connections wrap both attention and FFN, with Sinkhorn-derived pre/post/combination weights; the head also has a hyper-head.
- MoE has hash routing in the first `num_hash_layers`, top-k routing elsewhere, routed `I8 + F8_E8M0` experts (ADR 0007), and one shared expert.
- `num_nextn_predict_layers` exists in config; MTP tensors are present but are not proven safe for this MLX route.

### DeepSeek V4 reference inference code from the HF snapshot

The snapshot `inference/model.py` mirrors the same architecture with names that match the checkpoint index more directly:

- Attention: `wq_a`, `q_norm`, `wq_b`, `wkv`, `kv_norm`, `wo_a`, `wo_b`, `attn_sink`.
- Compressor: `ape`, `wkv`, `wgate`, `norm`.
- Indexer: `wq_b`, `weights_proj`, and an internal compressor.
- Hyper-Connections: top-level `hc_head_*` and per-layer `hc_attn_*` / `hc_ffn_*`.
- MoE: `gate.weight`, hash `gate.tid2eid`, optional `gate.bias`, routed `experts.{i}.w1/w2/w3`, and `shared_experts.w1/w2/w3`.
- MTP: `MTPBlock` has `e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`, and its own head HC tensors.

### MLX-LM `deepseek_v3.py`, `deepseek_v32.py`, and `mla.py`

MLX-LM gives useful pieces but is not a V4 implementation:

- `deepseek_v3.py` implements MLA with low-rank query projection and latent KV (`kv_a_proj_with_mqa`, `embed_q`, `unembed_out`) plus standard residual layers.
- `deepseek_v32.py` adds an `Indexer`, but it indexes MLX-LM latent KV and does not implement V4 CSA/HCA compressors, overlapping compression, HCA, Hyper-Connections, grouped output projection, hash routing, or MTP.
- `mla.py` provides `MultiLinear` / `QuantizedMultiLinear`, useful for per-head linear algebra in V3/V32, but V4 Flash uses direct shared `wkv`, grouped output `wo_a`/`wo_b`, and compressor/indexer projections that do not map to a shallow `deepseek_v3`/`deepseek_v32` rename.
- MLX-LM `deepseek_v32.sanitize()` removes MTP layers for V32-style checkpoints. That is not sufficient proof for V4 Flash; MTP remains fail-closed here.

### DS4 runtime/layout code

DS4 is the deployment/runtime reference for this project. Relevant DS4 facts:

- DS4 Flash shape defaults match the local Flash config: 43 layers, 4096 hidden, 64 heads, 512 head dim, 1024 Q LoRA rank, 1024 output LoRA rank, 256 routed experts, 6 experts/token, 2048 expert intermediate, 3 hash layers, 128 sliding window, 64x128 indexer, `index_top_k=512`, and `n_hc=4`.
- DS4 GGUF names differ from the HF safetensors index names. Examples: `blk.%u.attn_q_a.weight`, `blk.%u.attn_q_b.weight`, `blk.%u.attn_kv.weight`, `blk.%u.attn_output_a.weight`, `blk.%u.attn_output_b.weight`, `blk.%u.attn_compressor_*`, `blk.%u.indexer.*`, `blk.%u.hc_attn_*`, `blk.%u.hc_ffn_*`, `blk.%u.ffn_gate_tid2eid.weight`.
- DS4 validates tensor layout and format per family: FP32 norms/sinks/HC scales/bases; F16/Q8-ish attention pieces depending on quantized GGUF variant; Q2/Q4/IQ2/Q8 routed expert formats in deployment GGUF; I32 hash routing table for hash layers.
- DS4 implements V4 indexer QAT/Hadamard behavior for 128-wide indexer rows. This is a parity requirement for MLX, not optional.
- DS4 runtime LoRA currently applies CPU routes for `output`, `attn_q_a`, `attn_q_b`, and `attn_kv`. Some additional canonical names may be recognized by conversion/parsing code, but recognition is not runtime application support. MLX must not attach all-linear LoRA safely by default.

## Shimmed checkpoint index inventory

The shimmed index contains 69,187 tensor names. This inventory used only `model.safetensors.index.json`; no tensor shard payloads were opened.

The project tensor-name scanner classifies the index as follows:

| Family | Count | Example(s) | Status |
|---|---:|---|---|
| `embedding` | 1 | `embed.weight` | mapped |
| `output` | 1 | `head.weight` | mapped |
| `norm` | 173 | `layers.0.attn_norm.weight`, `layers.0.ffn_norm.weight`, `norm.weight` | mapped; forward parity required |
| `attention.q_a` / `_scale` | 43 / 43 | `layers.0.attn.wq_a.weight`, `.scale` | mapped; dequant/scale parity required |
| `attention.q_b` / `_scale` | 43 / 43 | `layers.0.attn.wq_b.weight`, `.scale` | mapped; dequant/scale parity required |
| `attention.kv` / `_scale` | 43 / 43 | `layers.0.attn.wkv.weight`, `.scale` | mapped; dequant/scale parity required |
| `attention.output_a` / `_scale` | 43 / 43 | `layers.0.attn.wo_a.weight`, `.scale` | mapped; grouped output parity required |
| `attention.output_b` / `_scale` | 43 / 43 | `layers.0.attn.wo_b.weight`, `.scale` | mapped; grouped output parity required |
| `attention.sink` | 43 | `layers.0.attn.attn_sink` | mapped; sparse attention parity required |
| `compressor` | 248 | `layers.10.attn.compressor.ape`, `wkv`, `wgate`, `norm` | mapped by name; CSA/HCA semantics blocked until parity tests |
| `indexer` | 63 | `layers.10.attn.indexer.wq_b.weight`, `weights_proj`, indexer compressor tensors | mapped by name; Lightning Indexer parity blocked |
| `hyperconnection` | 261 | `hc_head_base`, `layers.0.hc_attn_fn`, `layers.0.hc_ffn_scale` | mapped by name; HC Sinkhorn/pre/post parity blocked |
| `moe.router` | 86 | `layers.0.ffn.gate.weight`, `layers.0.ffn.gate.tid2eid`, `layers.10.ffn.gate.bias` | mapped by name; hash/top-k parity required |
| `moe.expert.w1/w2/w3` | 11,008 each | `layers.0.ffn.experts.0.w1.weight` | mapped by name; **I8** weights; decode arithmetic + shape-authoritative geometry proven (11.15b/c); routed dispatch fail-closed pending independent full-path reference (ADR 0007) |
| `moe.expert.w1/w2/w3_scale` | 11,008 each | `layers.0.ffn.experts.0.w1.scale` | mapped by name; **F8_E8M0** 1-D block scale (block_size 16, axis 1); proven (11.15b/c) |
| `moe.shared.w1/w2/w3` | 43 each | `layers.0.ffn.shared_experts.w1.weight` | mapped by name; shared expert parity required |
| `moe.shared.w1/w2/w3_scale` | 43 each | `layers.0.ffn.shared_experts.w1.scale` | mapped by name; dequant/scale parity required |
| `mtp.intentional-review-required` | 1,575 | `mtp.0.attn.wq_a.weight`, `mtp.0.e_proj.weight`, `mtp.0.ffn.experts.0.w1.scale` | review-required; not supported/proven |
| unknown | 0 | none | none in current index scanner output |

## Concrete HF/MLX/DS4 name mapping sketch

This is a semantic mapping target, not an approval to convert.

| HF safetensors family | Transformers V4 module | HF reference inference name | DS4 GGUF/runtime family | MLX-LM existing analogue | MLX V4 status |
|---|---|---|---|---|---|
| `embed.weight` | `embed_tokens` | `embed` | token embedding | `nn.Embedding` | mapped |
| `head.weight` | `lm_head` / `DeepseekV4HyperHead` input | `head.weight` | output/head | `nn.Linear` head | mapped, HC-head parity needed |
| `norm.weight` | final RMSNorm | `norm` | final norm | RMSNorm | mapped |
| `layers.N.attn.wq_a.*` | `self_attn.q_a_proj` | `wq_a` | `attn_q_a` | V3/V32 `q_a_proj` | mapped; LoRA-compatible target candidate |
| `layers.N.attn.q_norm.weight` | `self_attn.q_a_norm` | `q_norm` | `attn_q_a_norm` equivalent | V3/V32 `q_a_layernorm` | mapped by norm family |
| `layers.N.attn.wq_b.*` | `self_attn.q_b_proj` | `wq_b` | `attn_q_b` | V3/V32 `q_b_proj` | mapped; LoRA-compatible target candidate |
| `layers.N.attn.wkv.*` | `self_attn.kv_proj` | `wkv` | `attn_kv` | no direct V3 latent-KV match | mapped; LoRA-compatible target candidate |
| `layers.N.attn.kv_norm.weight` | `self_attn.kv_norm` | `kv_norm` | `attn_kv_a_norm` equivalent | V3/V32 `kv_a_layernorm` shape differs | mapped by norm family |
| `layers.N.attn.attn_sink` | attention sink parameter | `attn_sink` | `attn_sinks` | none | mapped; sparse attention parity needed |
| `layers.N.attn.wo_a.*` | `self_attn.o_a_proj` / grouped linear | `wo_a` | `attn_output_a` | none | mapped; grouped output blocked |
| `layers.N.attn.wo_b.*` | `self_attn.o_b_proj` | `wo_b` | `attn_output_b` | V3/V32 `o_proj` only superficially related | mapped; grouped output blocked |
| `layers.N.attn.compressor.*` | `HCA/CSACompressor` | `compressor.ape/wkv/wgate/norm` | `attn_compressor_*` | none | mapped by name; blocked on compressor parity |
| `layers.N.attn.indexer.*` | `DeepseekV4Indexer` / scorer | `indexer.wq_b`, `weights_proj`, `indexer.compressor.*` | `indexer.*` and `indexer_compressor_*` | V32 has simpler `Indexer` | mapped by name; blocked on Lightning Indexer parity |
| `layers.N.hc_attn_*` | `DeepseekV4HyperConnection` for attention | `hc_attn_*` | `hc_attn_*` | none | mapped by name; blocked on HC parity |
| `layers.N.hc_ffn_*` | `DeepseekV4HyperConnection` for FFN | `hc_ffn_*` | `hc_ffn_*` | none | mapped by name; blocked on HC parity |
| `hc_head_*` | `DeepseekV4HyperHead` | `hc_head_*` | top-level HC head | none | mapped by name; blocked on head HC parity |
| `layers.N.ffn.gate.weight` | `TopKRouter` / `HashRouter` | `gate.weight` | `ffn_gate_inp` equivalent | V3/V32 `MoEGate.weight` | mapped; router parity required |
| `layers.N.ffn.gate.tid2eid` | `HashRouter` | `gate.tid2eid` | `ffn_gate_tid2eid.weight` | none | mapped; hash routing parity required |
| `layers.N.ffn.gate.bias` | `TopKRouter` bias | `gate.bias` | router bias equivalent | V3/V32 correction bias differs | mapped; top-k parity required |
| `layers.N.ffn.experts.E.w1/w2/w3.*` | routed experts | `experts.E.w1/w2/w3` | routed expert GGUF tensors | V3/V32 `SwitchGLU` after stacking | name-mapped; **I8 + F8_E8M0** (bs16, axis1); decode proven, routed dispatch fail-closed pending independent reference (ADR 0007) |
| `layers.N.ffn.shared_experts.w1/w2/w3.*` | shared expert MLP | `shared_experts` | shared FFN tensors | V3/V32 shared expert MLP | mapped; dequant parity required |
| `mtp.0.*` | MTPBlock / next-token prediction layer | `mtp.0.*` | DS4 has MTP GGUF loader paths, but MLX training support is not proven | V32 sanitizer drops MTP-like extra layers | review-required; fail-closed |

## Mapped tensor families

These are name-mapped and have a reference module family, but most still require parity tests before conversion:

- Embedding and output/head.
- Final and per-layer RMSNorms.
- Core attention `wq_a`, `q_norm`, `wq_b`, `wkv`, `kv_norm`, `attn_sink`, `wo_a`, `wo_b`.
- CSA/HCA compressor tensors by name.
- CSA Lightning Indexer tensors by name.
- Hyper-Connection and hyper-head tensors by name.
- MoE router, hash table, routed experts, and shared expert tensors by name.

## Blocked tensor families / semantics

These must remain blocked until executable parity tests exist:

- FP8 block scales and UE8M0 semantics for non-expert linear families. The BF16 shim makes safetensors loadable; it is not proof that per-block scale application matches Transformers/DS4.
- Routed expert dispatch through `dequantize_expert_packed("i8")`. The routed packing is verified `I8 + F8_E8M0` (block_size 16, axis 1, shape-authoritative — **not** FP4); the per-block decode arithmetic and geometry are proven (11.15b/c), but public dispatch stays fail-closed until an independent full-path reference confirms consumer-convention/orientation (spec §10.9, ADR 0007).
- CSA/HCA compressor behavior: overlapping CSA two-series windows, HCA non-overlap, APE/position bias, chunk buffering, compressed RoPE positions, and causal masking.
- Lightning Indexer behavior: indexer compressor, query projection from `q_residual`, ReLU weighted scorer, top-k invalid sentinel handling, Hadamard/QAT behavior noted in DS4.
- Hyper-Connections: Sinkhorn pre/post/combination math for block residual streams and hyper-head.
- Grouped output projection `wo_a -> wo_b`: must preserve group reshape and per-group projection, not collapse to V3/V32 `o_proj`.
- Hash routing and top-k routing: first 3 layers use `tid2eid`; later layers use score+bias top-k with `sqrtsoftplus` and route scale.
- MLX LoRA target allowlist: only DS4-supported target families should be enabled; all-linear attachment remains unsafe.

## Review-required tensor families

- `mtp.0.*` (1,575 tensor names) is review-required. Do not mark MTP supported unless there is proof from load/forward parity and a clear conversion/runtime policy.
- Current fail-closed behavior is correct: `deepseek-v4-mapping-check` reports `mtp.intentional-review-required` and does not produce a valid `.deepseek-v4-mapping-ok` marker for the real shimmed index.
- Possible future policies, each requiring proof: implement MTP end-to-end, deliberately strip it with evidence that MLX-LM conversion/training/runtime do not need it, or keep it blocked.

## Unknown tensor families

- Current scanner output over the shimmed index reports zero unknown/unmapped non-MTP names.
- This does not unblock conversion because review-required MTP and blocked semantics remain.

## Fail-closed acceptance gates implied by this dossier

Before `convert-shimmed` may run, require all of the following:

1. Import gate: `mlx_lm.models.deepseek_v4` resolves to project-controlled code.
2. Tiny-config gate: config parsing works and construction remains fail-closed until parity is implemented.
3. Mapping gate: no unknown or review-required families, and marker is bound to the current shimmed index hash.
4. Forward parity gate: tiny load/forward parity exists against Transformers or reference inference for attention, compressor/indexer, HC, MoE/router, and output head.
5. Dequant parity gate: FP8/UE8M0 and the verified I8+F8_E8M0 (routed) / F8_E4M3+F8_E8M0 (shared) expert fixtures match trusted DS4/Transformers/reference behavior or fail closed (ADR 0007).
6. LoRA allowlist gate: MLX LoRA attaches only to target families known to convert to DS4 canonical safetensors.

## Bottom line

A shallow MLX-LM `deepseek_v3.py` or `deepseek_v32.py` rename is explicitly unsafe. V4 Flash differs in direct shared-KV attention, CSA/HCA compressors, grouped output projection, Hyper-Connections, hash/top-k MoE routing, I8+F8_E8M0 routed experts (ADR 0007), and MTP checkpoint content. The current correct state is: name mapping has useful coverage, but conversion/training must remain blocked until MTP policy, dequantization, and load/forward parity are proven.
