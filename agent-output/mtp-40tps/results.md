# Path-to-40-t/s diagnostics — fusion sweep + GPU trace

Model: joint-main-mtp-L3742-mixed.gguf, M3 Ultra (Mac Studio), Metal, ctx 512,
greedy, speculation disabled for clean per-token paths. Harness `fusion-sweep.sh`.

## Baseline decomposition (from engine's DS4_METAL_GRAPH_TOKEN_PROFILE)

Wall 30.7 ms/token (32.5 t/s):

| phase | ms/token |
|---|---|
| GPU execute (commit -> complete) | 25.3 |
| CPU command encoding | 2.0 |
| logits readback | ~0 |
| sampler + MTP probe + gaps | ~3.4 |

## Fusion A/B sweep — fusions are worth <3%

Each variant toggles one existing fusion OFF via its DISABLE env; all else
identical (n=40 tokens each):

| variant | t/s | GPU exec ms | delta vs baseline |
|---|---|---|---|
| baseline | 32.15 | 25.51 | — |
| no_routed_pair_swiglu | 31.96 | 25.62 | +0.11 ms |
| no_shared_swiglu | 31.78 | 25.77 | +0.26 ms |
| no_iq2_expert_views | 31.99 | 25.51 | +0.00 ms |
| no_moe_id_pair_swiglu | 31.94 | 25.60 | +0.09 ms |
| no_q4_grouped_experts | 31.85 | 25.58 | +0.07 ms |
| moe_write_clamped | 31.85 | 25.65 | +0.14 ms |

All six existing fusions together save ~0.7 ms/token (<3%). They are working;
they are also nearly exhausted as a lever. Two candidate flags were dropped
before running: `DS4_BATCHED_FFN` is a CPU-reference-path flag, and the Q4
expert table is gated off for in-RAM Flash models — both would have been no-ops.

## GPU trace — the GPU is saturated, not gapped

`xctrace record --template "Metal System Trace"` over a 20-token generation
(trace at /tmp/ds4gpu.trace, parsed intervals in /tmp/ds4-gpu-intervals.json):

- 1183 GPU intervals for the ds4 process; merged busy-union = 886.4 ms inside a
  903.9 ms active span = **98.1% GPU busy, 1.9% idle**.
- The single largest idle gap is 17.5 ms at process startup (warmup); during
  decode the gaps between command buffers are sub-microsecond.
- The serialized stage profiler inflates native token cost ~4.5x (125 ms vs
  25.3 ms) and its per-stage numbers do not decompose cleanly against the
  native total — treated as unreliable for attribution and discarded.

## Where this leaves the 40 t/s target

The 25 ms/token budget is inside GPU kernel execution, streaming 6.26 GiB of
active weights at ~253 GB/s = **32% of the M3 Ultra's ~800 GB/s peak**. Not
bandwidth-saturated, not scheduling-gapped, not fusion-limited: the kernels are
latency/occupancy-bound on small per-expert matvecs (18 dispatches/layer of
2-6 MB each across 43 layers).

Reaching 40 t/s (25 ms/token) requires GPU execute <= ~20 ms = streaming at
~320 GB/s, a +26% kernel-efficiency gain.

## What cannot attribute the 25 ms (tried)

- Metal System Trace / Game Performance templates expose command-buffer
  intervals only; per-kernel (shader timeline) attribution was Disabled in the
  template configuration and xctrace CLI offers no flag to enable it.
- The engine's DS4_METAL_DECODE_STAGE_PROFILE serializes each stage into its
  own command buffer; isolated small kernels run ~5x slower than when scheduled
  back-to-back in the native single-buffer path, so its numbers overstate and
  distort attribution.

## Per-stage GPU attribution — measured via engine patch

Counter sampling proved unusable on this GPU family: the device advertises
stage-boundary counter support but trips a hard assertion
(`sampleCountersInBuffer not supported on this device`) when a compute encoder
actually calls it, and reports dispatch-boundary as unsupported. Fallback
implemented instead (ds4_metal.m + ds4.c, env-gated, fails soft):
`DS4_METAL_GPU_STAGE_TIMING=1` combined with `DS4_METAL_DECODE_STAGE_PROFILE=1`
labels each serialized stage segment and reports the command buffer's own
`GPUStartTime`/`GPUEndTime` — the true GPU duration of that segment, free of
CPU encode/sync latency. Patch verified: default path byte-identical (cmp vs
same-env reference), zero diagnostic output when env unset,
`make embedded-mtp-test` PASS.

Measurement (8.66 fully-profiled tokens, ctx 512, greedy, spec disabled):
serialized GPU total 26.9 ms/token — within 6% of the native 25.3 ms, so the
attribution is representative, and the earlier 4.5x serialization inflation was
CPU wall-clock, not GPU work.

| stage | ms/token | share |
|---|---|---|
| routed_moe | 5.69 | 21.2% |
| attn_output | 5.14 | 19.1% |
| q_path | 3.85 | 14.3% |
| attention | 3.09 | 11.5% |
| compressor_indexer | 1.80 | 6.7% |
| shared_gate_up | 1.47 | 5.5% |
| attn_hc_pre | 1.41 | 5.3% |
| ffn_hc_pre | 1.29 | 4.8% |
| router | 1.20 | 4.4% |
| kv_path | 1.03 | 3.8% |
| shared_down | 0.93 | 3.5% |

Attention half 16.3 ms/token (61%), FFN/MoE half 10.6 ms/token (39%).

### What this changes

The working assumption going in was that the IQ2/Q2-K expert GEMVs dominated.
They do not. routed_moe is the single largest stage (21%) but its implied
streaming rate is already at/above the memory roofline (expert bytes per token
divided by 5.69 ms exceeds 800 GB/s, i.e. it is bandwidth-saturated or benefits
from expert reuse) — headroom there is small. The attention projection path —
attn_output + q_path + attention = 12.1 ms, 45% of decode GPU time — consists of
small LoRA matvecs running at an estimated ~450 GB/s (q_path byte estimate
1.6 GiB/token ÷ 3.85 ms), well below peak. That latency-bound family is where
kernel work pays.

### 18k-context attribution (the real workload)

Same measurement with context18k.prompt.txt, ctx 32768, 9.52 fully-profiled
tokens. Serialized GPU total 37.25 ms/token vs native 41.3 ms (gap 4.0 ms,
consistent with the 3.9 ms gap at ctx 512 — non-GPU overhead is constant,
the serialized GPU numbers track native cost at both contexts).

| stage | 18k ms/tok | 512 ms/tok | delta | 18k share |
|---|---|---|---|---|
| attention | 9.69 | 3.09 | +6.60 | 26.0% |
| routed_moe | 5.83 | 5.69 | +0.14 | 15.7% |
| attn_output | 5.27 | 5.14 | +0.13 | 14.1% |
| compressor_indexer | 4.94 | 1.80 | +3.14 | 13.3% |
| q_path | 3.93 | 3.85 | +0.08 | 10.5% |
| shared_gate_up | 1.52 | 1.47 | +0.05 | 4.1% |
| attn_hc_pre | 1.45 | 1.41 | +0.04 | 3.9% |
| ffn_hc_pre | 1.34 | 1.29 | +0.06 | 3.6% |
| router | 1.23 | 1.20 | +0.03 | 3.3% |
| kv_path | 1.09 | 1.03 | +0.06 | 2.9% |
| shared_down | 0.96 | 0.93 | +0.03 | 2.6% |

The context-length penalty is concentrated almost perfectly: total growth
512->18k is +10.35 ms/token, of which +10.05 ms (97%) is the attention family
and +0.14 ms (1%) routed_moe. Within that, the `attention` kernel itself grew
+6.60 ms (3.09 -> 9.69) and `compressor_indexer` +3.14 ms (1.80 -> 4.94).
Everything else is context-flat, confirming the weight-streaming half is
unaffected by sequence length.

At 18k the attention half is 26.37 ms = 71% of decode GPU time; ffn/moe 29%.

### Final verdict on the 40 t/s target

At the real 18k context the target needs 37.25 -> <=25 ms/token, a 33% cut.
The two growth stages name themselves: the attention kernel (9.69 ms, 26%) and
compressor_indexer (4.94 ms, 13%) — together 40% of the token, and the entire
reason long context is slower than short. Both scale with sequence state rather
than weight bytes, so they are algorithm/kernel problems (indexer sparsity,
compressed-KV attention scheduling), not quantization or bandwidth problems.
The routed-MoE expert path, at 15.7% and context-flat near the memory roofline,
is confirmed a dead lever at every context length measured.

## Inside compressor_indexer — score vs top-k vs prep (measured)

Follow-on slice. `DS4_METAL_INDEXER_STAGE_PROFILE` already split this sub-path
into score / topk / attention segments, but reported wall clock only. Wiring the
existing `ds4_gpu_stage_mark` into `metal_graph_indexer_stage_profile_boundary`
(ds4.c:17596, six lines, same pattern as the two other boundary helpers) makes
those segments report `GPUStartTime`/`GPUEndTime` like the stage table above.
Plan in `indexer-split-plan.md`; raw output in `indexer-split-18k.stderr`.

First structural finding: **the indexer runs on 21 of 43 layers, not all of
them.** Flash's layout is `il<2 -> ratio 0`, then alternating 4 / 128
(ds4.c:632-638), and the indexer sub-path is gated on `ratio == 4`
(ds4.c:15438). So per-invocation cost is ~2x what a 43-layer average implies.

Second: **at ctx 512 the indexer never runs at all.** It additionally requires
`n_comp > DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (default 1024,
ds4.c:13503) with `n_comp ~ pos/4`, i.e. `pos > ~4096`. A ctx-512 run with the
profiler on emits zero indexer lines (`indexer-split-512.stderr`, 0 matches).
The 1.80 ms measured at ctx 512 is therefore compressor-only, and the +3.14 ms
growth to 18k is the indexer switching on, not the compressor scaling.

Decode GPU time, 12 tokens at ctx 32768, summed over the 21 ratio-4 layers:

| sub-stage | ms/token | per layer | share of the 4.94 ms stage |
|---|---|---|---|
| indexer top-k | 1.14 | 54.5 us | 23% |
| indexer score | 0.93 | 44.4 us | 19% |
| indexer prep (q-proj, RoPE, QAT, weight-proj) | ~1.06 (residual) | ~50 us | ~22% |
| compressor update (both caches) | ~1.80 (= ctx-512 value) | ~86 us | ~36% |

score + topk = 2.08 ms/token measured directly; the 2.86 ms residual against the
published 4.94 ms splits into the ctx-512 compressor baseline (1.80 ms) and
~1.06 ms of indexer prep, which is inferred by subtraction rather than measured.

### What this changes

**Top-k costs more than the score it consumes.** The working assumption was that
scoring 4502 rows against 64 heads dominated; instead the argsort + merge-pass
selection is 23% more expensive than the scoring pass (54.5 us vs 44.4 us per
layer). Selecting 512 of ~4500 with a full bitonic argsort plus ~5 merge passes
is the wrong algorithm for that ratio; a radix-select or heap-select is O(n)
against the current O(n log n). That is the single most attractive kernel target
found so far, and it is self-contained in `ds4_gpu_indexer_topk_tensor`
(ds4_metal.m:12561).

**The score kernel is nowhere near bandwidth-bound.** It reads 4502 rows x 128
float32 = 2.31 MB in 44.4 us = **52 GB/s, about 6.5% of peak**. It is not
streaming-limited; it is limited by the dispatch shape. `grid (n_comp,1,1)`,
`tg (32,4,1)` = 128 threads, with 3 threadgroup barriers per head-group and 16
head-groups, i.e. 48 barriers per row x 4502 rows per layer
(metal/dsv4_misc.metal:169-195, ds4_metal.m:12319-12322). Only `tid == 0`
accumulates. Widening rows-per-threadgroup and replacing the psum-plus-barrier
reduction with a simdgroup reduction is a well-defined, low-risk rewrite.

**The indexer cache is float32 while the attention cache is FP8/F16.** 512
bytes/row of f32 feeding a kernel that only produces a dot-product score
(metal/dsv4_misc.metal:157). Halving it to f16 halves the bytes, but given 52
GB/s that saves little until the occupancy problem is fixed first.

Headroom if both are addressed: score + topk is 2.08 of 37.25 ms/token (5.6%).
Even eliminating them entirely does not reach 40 t/s alone; the 33% cut still
needs the 9.69 ms attention kernel. But the top-k rewrite is cheap, isolated,
and does not touch the model's candidate set.

## Threshold sweep — the default is wrong below ~16k

Harness `threshold-sweep.sh`, raw runs in `thsweep/`.

First finding is a design one: **at the 18k prompt the knob is saturated and
cannot discriminate.** The gate is `n_comp > threshold` with `n_comp ~ pos/4`,
so 18k gives `n_comp ~ 4502`, above the largest legal threshold (4096). Every
setting picks sparse. Verified rather than assumed: thresholds 64 and 4096 at
18k produced byte-identical output at 24.13 vs 24.04 t/s. The knob only has
authority for `n_comp` in 64..4096, i.e. roughly 256..16384 tokens.

So the sweep was reframed as a dense-vs-sparse crossover measurement: threshold
64 forces sparse as early as legal, 4096 holds dense as long as legal, and the
pair brackets the true crossover at each length.

| tokens | n_comp | sparse (th=64) | dense (th=4096) | delta |
|---|---|---|---|---|
| ~2215 | 553 | 31.32 | 31.45 | +0.13 |
| ~4431 | 1107 | 25.10 | 30.94 | **+5.84** |
| 6789 | 1697 | 24.93 | 30.44 | **+5.51** |
| 9409 | 2352 | 24.50 | 29.56 / 29.38 | **+5.0** |
| 13067 | 3266 | 24.36 | 29.00 / 28.58 | **+4.5** |
| 16551 | 4137 | 24.33 | 24.35 | +0.02 (both sparse) |

The 9409 row first measured +0.30, inconsistent with both neighbours; two
repeats gave +5.00 and +4.92, so the original was a bad run and the repeats are
reported. 13067 was repeated for the same reason and held at +4.6.

### The default costs ~5 t/s across most of the useful range

Dense indexed attention beats the sparse path by **4.5-5.8 t/s (18-23%)** from
about 4400 to 16500 tokens. The default threshold of 1024 switches to sparse at
`n_comp > 1024`, i.e. **~4096 tokens** — precisely where the measurements show
dense is still clearly winning. The in-code comment (ds4.c:13497) reasons that
"around the 2K frontier the sparse path's score/top-k setup dominates the
smaller attention scan", and the direction is right; the frontier is just placed
about 4x too low. The crossover is somewhere past 16k, not at 4k.

This is consistent with the sub-stage measurements above: the indexer chain
costs a fixed ~3.14 ms/token once engaged, and that only pays for itself when
the attention scan it shortens is large enough. Below ~16k it is not.

Raising the default from 1024 to 4096 would recover ~5 t/s for every context
between 4k and 16k at zero implementation cost. It changes nothing at 18k+,
where `n_comp` exceeds every legal threshold anyway.

### Correctness caveat, and why this is not a free win

Sparse and dense are **not** output-equivalent, as expected — sparse attends to
512 selected rows plus the SWA window, dense attends to all of them. At 12000
bytes (~4400 tokens) the two paths diverge in wording ("the user says" vs "the
user asks"). Each path is individually deterministic across repeats, so this is
a real path difference, not run-to-run noise. At 18000/24000/36000 bytes the
outputs happened to be identical, but that is luck of the sampling, not a
guarantee.

So this is a speed/quality knob, not a pure speedup. The measurements say the
default currently pays ~5 t/s for sparsity in a range where sparsity buys little
scan reduction. Whether the quality delta justifies that is a model-behaviour
question this sweep does not answer, and changing the default should be gated on
an eval comparing the two paths at 4k-16k rather than on throughput alone.

## Quality eval — dense vs sparse at 4k-16k

The throughput sweep left an open question: dense is ~5 t/s faster in this band,
but the paths are not output-equivalent, so is the speed bought with accuracy?
Harness `sparse-quality-eval.py` + `run-multihop.py`, raw runs in `qualeval/`
and `multihop/`.

Design: needle-in-haystack with exact-match ground truth. A haystack of
distractor records carries needles at controlled depths; the answer is a
5-digit code, so it cannot be guessed and there is no model-judge in the loop.
This targets exactly the failure sparse attention would cause — the 512-row
selection missing the rows holding the answer. `--nothink`, greedy, one process
per run so no KV state leaks between configurations.

**Result: 60 runs, zero accuracy difference.**

| eval | sparse | dense | mean t/s (sparse -> dense) |
|---|---|---|---|
| single needle, 8 depths x 3 lengths | 24/24 | 24/24 | 23.58 -> 28.61 |
| multi-hop, k=4 and k=8 codes per reply | 36/36 codes | 36/36 codes | 24.88 -> 29.81 |
| combined | 60/60 | 60/60 | 23.84 -> 28.85 (**+21.0%**) |

Not one disagreement across 30 paired trials, at any depth (0.13 to 1.0) or any
length. The speed is not being bought with retrieval accuracy on this task.

### What this does not establish

Both evals saturated at 100%, which is a ceiling effect, not a demonstration of
equivalence. Specifically:

- With 0 failures in 30 paired trials, the rule of three puts the 95% upper
  bound on the difference rate at **10%**. This rules out a large quality gap,
  not a small one.
- Retrieval is the task sparse attention is *designed* to survive — the indexer
  selects rows by relevance, and a uniquely-keyed needle is maximally easy to
  select. It is the right first probe because it is the direct failure mode, but
  it is close to a best case for sparsity.
- Not tested: reasoning that must integrate many weakly-keyed rows, summarization
  where no single row is decisive, code comprehension over a long file, or
  anything where the relevant context is diffuse rather than pointed. A
  512-row budget could bind there while leaving needle retrieval intact.
- The earlier wording divergence at ~4400 tokens ("says" vs "asks") shows the
  paths do produce different text; this eval shows that difference did not cost
  correctness on these questions.

The indexer was verified to actually engage on the sparse runs (189 and 315
indexer stage invocations at 16000 and 38000 bytes), so the tie is a real tie
and not the sparse path silently failing to activate.

### Recommendation

Raising `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` from 1024 to 4096 is
supported by the evidence gathered: +21% throughput between ~4.4k and ~16.5k
tokens, with no measured accuracy cost on long-context retrieval. It is a
one-line default change (ds4.c:13503) affecting nothing at 18k+, where the knob
is saturated anyway.

The residual risk is the diffuse-context regime this eval did not cover. A
reasonable gate before changing the default would be one run of the existing
`ds4-eval` harness under both settings — though note its embedded questions are
short, so the indexer never engages on them and it would measure nothing here.
A long-context reasoning eval would be needed instead, which does not currently
exist in this repo.

## Persistent KV reuse — measured on the real 18k prompt

Plan in `kv-reuse-plan.md`, harness `kv-reuse-measure.py`, raw runs in
`kvreuse/`. Server on port 8077, ctx 32768, `--kv-disk-dir /tmp/ds4-kvtest`,
`--kv-disk-space-mb 8192`, requests over HTTP so prompt bytes are exact.

Correction to an earlier claim in this session: persistent KV reuse is **not**
an unbuilt feature. `ds4_session_sync()` (ds4.h:249) already does in-process
prefix reuse, and `ds4_kvstore.c` is a complete content-addressed disk
checkpoint store with eviction, prefix matching, and eight server flags. This
slice measured what exists; it built nothing.

### Latency result

| condition | wall | note |
|---|---|---|
| cold (empty cache) | 39.98 s | full prefill, 18004 tok @ ~474 t/s |
| warm, same process | 6.89 s | disk hit + 1620-token replay |
| warm, repeated | 6.95 s | reproducible |
| divergent prefix (1 byte changed near top) | 40.86 s | full re-prefill, as designed |

**Warm is 5.78x faster end to end, saving 33.1 s (83%).** Engine-side, the
prefill work drops from 37.95 s to 4.30 s (0.41 s disk load + 3.89 s replay),
**8.8x cheaper**. Checkpoint is 237.99 MiB for 16384 tokens, written in
46-127 ms and loaded in 397-413 ms.

The mechanism is visible in the server log and behaves exactly as designed:

```
kv cache stored tokens=16384 trimmed=1620 reason=cold key=token-text size=237.99 MiB save=110.4 ms
kv cache hit text tokens=16384 text=47494 quant=2 key=token-text load=397.4 ms
chat ctx=16384..18004:1620 prefill chunk 1620/1620 (100.0%) chunk=416.11 t/s 3.893s
```

The divergent-prefix case confirms the failure mode is a clean full re-prefill,
not corruption: `live kv cache miss live=18052 prompt=18006 common=6
reason=token-mismatch`.

### The 2048-alignment gate costs ~3.9 s of the warm path

`--kv-cache-boundary-align-tokens` (default 2048) truncates the 18004-token
save to 16384, leaving 1620 tokens replayed on every warm hit — 3.89 s of the
6.9 s warm total, i.e. **56% of the remaining warm latency**. Raising the
checkpoint to the full prefix would cut warm latency to roughly 3 s. Whether
the alignment exists for a correctness reason was not investigated; it is a
candidate for a follow-up slice, not a change to make blindly.

### Correctness: gate not passed, and the reason is not KV reuse

The plan's primary gate was byte-identical warm-vs-cold output. **Warm replies
differed from cold.** Investigating rather than reporting the speed number
alone:

- warm1 and warm2 also differ **from each other**.
- Three identical requests to a server started with **no `--kv-disk-dir` at
  all** produced three different replies (`det1/det2/det3` in `kvreuse/`).
- Two CLI runs of the same prompt at `--temp 0 --nothink` were byte-identical.

So the engine core is deterministic and the **server request path is not**,
independently of KV reuse. KV reuse is therefore not shown to corrupt output —
but it is also **not shown to preserve it**, because the baseline is too noisy
to detect a reuse-induced difference. The gate is unresolved, not passed.

Two differences between the CLI reference and the server path are candidates
and were not isolated: the server runs in THINKING mode (log shows `gen=48
THINKING`) where the CLI test used `--nothink`, and MTP speculation is active
on the server while the CLI comparison used the default. Either could
introduce order-dependent behaviour. Identifying which is a prerequisite to
closing the correctness question, and is the next slice.

### Cache sizing observation

Six checkpoints totalling 1.5 GiB accumulated across four requests — each
request stored both a `reason=cold` entry (237.99 MiB) and a `reason=evict`
entry (259.89 MiB) at the post-generation frontier. At the **default** 4096 MB
budget this working pattern would begin evicting after roughly eight requests,
and an eviction costs a full 38 s re-prefill on next use. Worth a sizing review
for long agent sessions; not changed here.

## Correctness gate CLOSED, and the cause was the harness

The earlier KV-reuse slice failed its byte-identity gate. Root cause found and
proven: `./ds4-server --help thinking` states **"In thinking mode, client
sampling knobs are ignored like the official API."** The measurement sent
`temperature: 0` in thinking mode, so it was silently discarded and every reply
sampled at the server default. The non-determinism was the harness, not KV
reuse and not the server.

Re-run in non-thinking mode (`model: deepseek-chat` + `thinking:
{type: disabled}`), where temperature is honoured:

| condition | wall | cached_tokens | vs cold |
|---|---|---|---|
| cold | 40.76 s | 0 | — |
| warm1 | 7.56 s | 16384 | **byte-identical** |
| warm2 | 7.02 s | 16384 | **byte-identical** |

**Gate passed.** Reuse is confirmed active (`cached_tokens: 16384`) and output
is byte-identical to a cold full prefill.

Diagnosis proven rather than assumed: two thinking-mode requests at
`temperature: 0` hitting the **same** warm cache (`cached=16384` both) still
produced different replies (311 vs 321 bytes). Same cache state, different
output → the variance is sampling, not KV reuse.

## Default change 1: indexer sparse threshold 1024 -> 4096 (LANDED)

`ds4.c:13505`. Evidence is the sweep and quality eval above: +21% throughput
from ~4.4k to ~16.5k tokens, 60/60 on both paths in the retrieval eval.

Verified after the change:

| case | before | after | output |
|---|---|---|---|
| 36000-byte prompt (~13k tok) | 24.32 t/s (env=1024) | **28.87 t/s** | — |
| ctx 512 short prompt | 31.94 t/s | 31.94 t/s | **byte-identical** to pre-change |
| 18k prompt (knob saturated) | 24.04-24.13 t/s | 24.09 t/s | **byte-identical** to prior run |

`make` clean, `make embedded-mtp-test` PASS. The old behaviour remains
reachable via `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=1024`, confirmed to
still produce 24.32 t/s.

## Default change 2: KV boundary alignment — NOT changed, and why

The warm path spends 3.89 s replaying the 1620 tokens that the 2048 alignment
declines to checkpoint. Disabling alignment is a large win:

| setting | warm wall | cached | replayed | speedup vs cold |
|---|---|---|---|---|
| align 2048 (default) | 7.02-7.56 s | 16384 | 1620 | 5.59x |
| align 0 | 3.64-4.15 s | 17972 | 32 (trim only) | **10.46x** |

That is **47% faster warm**, and it held on a longer generation: 256 tokens
cold 47.5 s vs warm 10.8 s, byte-identical.

**I am not changing this default.** The alignment is load-bearing by design,
not an arbitrary rounding — ds4_kvstore.c:35-39 states it explicitly:

```
/* Tokenizers may merge text across the prompt boundary. Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * cases where canonical prompt tokenization spells the same bytes differently.
 * The 2048 alignment also matches the backend prefill chunk schedule, which
 * keeps compressor row finalization identical to a cold full prompt. */
```

The alignment matches the prefill chunk schedule so **compressor row
finalization stays identical to a cold prompt** — exactly the machinery this
whole investigation has been measuring. My align=0 tests came back
byte-identical on this prompt, but that is one prompt whose length happens to
land safely; the comment describes a class of failure (tokenizer merges at the
boundary, compressor rows finalized mid-chunk) that a single passing case does
not rule out. Changing it needs a test matrix across prompt lengths that land
at different offsets within a 4096-token prefill chunk, which is a slice of its
own.

Recommended interim: users who want the faster warm path can set
`--kv-cache-boundary-align-tokens 0` per-deployment and measure their own
workload. The 3.4 s is available today without a default change.

## Alignment test matrix — align=0 survived every boundary tested

Plan in `align-matrix-plan.md`, harness `run-align-matrix.py` +
`make-align-prompts.py`, raw runs in `alignmatrix/`.

The question: ds4_kvstore.c:35-39 says the 2048 alignment "matches the backend
prefill chunk schedule, which keeps compressor row finalization identical to a
cold full prompt". If that is load-bearing, checkpointing at a length that is
**not** chunk-compatible should be able to diverge from a cold prefill.

Method: build prompts whose token counts land on chosen residues mod the
4096-token prefill chunk, using `--dump-tokens` (no GPU run, ~40 ms) to
binary-search byte prefixes. Per case: fresh cache dir, fresh server, cold
request, warm request, byte-compare. 128-token generations so divergence has
room to appear. Non-thinking + `temperature: 0`, since thinking mode ignores
sampling knobs.

### Control arm first

`align=2048` passed 12/12 with reuse engaged in every case, mean warm 8.36 s vs
cold 29.93 s. The harness detects reuse and would have caught a mismatch.

### Experimental arm

`align=0` stored checkpoints at **six distinct residues mod 4096** — 484, 996,
2020, 3044, 3556, 4068 — all mid-chunk, exactly the condition the comment warns
about. **12/12 byte-identical**, mean warm 5.55 s.

| arm | cases | reuse engaged | byte-identical | mean warm | residues mod 4096 |
|---|---|---|---|---|---|
| align 2048 (control) | 12 | 12/12 | **12/12** | 8.36 s | {0, 2048} |
| align 0 | 12 | 12/12 | **12/12** | 5.55 s | {484, 996, 2020, 3044, 3556, 4068} |

**34% faster warm** across twelve prompt lengths, with no correctness
difference at any chunk offset.

### A gap the first pass missed

Every stored length in that matrix was divisible by 4, so it never tested a
boundary **mid compressor group** — the compressor emits a row every `ratio`
tokens, and ratio-4 layers are the ones carrying the indexer. `store_len`
subtracts trim=32, and 32 is itself divisible by 4, so the residue is inherited
from the prompt length; all twelve prompts happened to be 0 mod 4.

Closed with a second matrix at a fixed ~10240-token anchor, walking byte length
until one prompt landed on each of `stored % 4 ∈ {0,1,2,3}`:

| case | stored | stored%4 | identical |
|---|---|---|---|
| mod4-0 | 10212 | 0 | yes |
| mod4-1 | 10213 | 1 | yes |
| mod4-2 | 10214 | 2 | yes |
| mod4-3 | 10215 | 3 | yes |

4/4 byte-identical with reuse engaged. Checkpointing part-way through a
compressor group did not change output.

### Verdict, with its limit stated

16 align=0 cases, 0 mismatches, spanning six chunk residues and all four
compressor-group residues, at 8k-16k tokens. The rule of three puts the 95%
upper bound on the failure rate at **18.8%** — this is meaningful evidence, not
proof, and the bound is wide because 16 cases is not many.

What the matrix does **not** cover: contexts past 16k (where `n_comp` exceeds
the sparse threshold and the indexer path changes), the continued-frontier save
path (`--kv-cache-continued-interval-tokens`, its own alignment interaction),
the SWA 128-row window boundary specifically, and multi-turn sessions where
several checkpoints chain.

**Recommendation: still do not change the default in this slice.** The evidence
now supports align=0 being safe in the range tested, and the win is real (34%
warm, 2.81 s). But the comment describes a design invariant, and the honest
position is that 16 passing cases in one band is grounds for a maintainer to
revisit the invariant deliberately — not grounds for an agent to silently drop
it. The per-deployment flag `--kv-cache-boundary-align-tokens 0` remains
available to anyone who wants the win today.

## Alignment matrix extended — past 16k and through the continued-frontier path

Plan in `align-matrix-ext-plan.md`, harnesses `run-align-matrix.py` and
`run-continued-arm.py`, raw runs in `alignext/`. This closes the two gaps the
first matrix declared.

### Arm A: past 16k, where the indexer is live

Above ~16.4k tokens `n_comp` exceeds the sparse threshold, so the indexer
engages and the attention implementation changes. Every earlier align case ran
below that boundary.

| length | align2048 stored | align0 stored | align0 residue mod 4096 | identical |
|---|---|---|---|---|
| 18000 tok | 16384 | 17972 | 1588 | both yes |
| 21000 tok | 20480 | 20972 | 492 | both yes |
| 24000 tok | 20480 | 23972 | 3492 | both yes |

3/3 in each arm, reuse engaged everywhere. Mean warm 9.73 s (align2048) vs
7.27 s (align0), **25% faster**, cold ~50 s in both.

The same mod-4 gap that the first matrix had applied here too — all three
align0 cases landed on `stored % 4 == 0`, so mid-compressor-group above 16k was
still untested. Closed with a 21k anchor covering all four residues:

| case | stored | stored%4 | identical |
|---|---|---|---|
| a21k-mod4-0 | 20972 | 0 | yes |
| a21k-mod4-1 | 20973 | 1 | yes |
| a21k-mod4-2 | 20974 | 2 | yes |
| a21k-mod4-3 | 20975 | 3 | yes |

4/4 byte-identical with the indexer active.

### Arm B: the continued-frontier path

`ds4_kvstore_continued_store_target()` fires only when
`live_tokens % step == 0` exactly, where step is the continued interval rounded
**up** to a multiple of the alignment. The two arms therefore checkpoint at
different frontiers: align2048 -> 10240, align0 -> 10000. Each arm got a prompt
~120 tokens below its own frontier and generated 256 tokens to cross it.

| arm | expected frontier | continued save observed | stage-2 identical | warm vs cold |
|---|---|---|---|---|
| align2048 | 10240 | **10240** | yes | 8.0 s vs 24.8 s |
| align0 | 10000 | **10000** | yes | 3.8 s vs 24.3 s |

Both arms emitted `reason=continued` at exactly the predicted frontier, so the
path was genuinely exercised rather than skipped. Stage 2 then re-requested
with prompt+continuation so the continued checkpoint was the matching prefix,
and compared against a cold run of that same extended prompt: byte-identical in
both arms. align0 is **53% faster** on the continued reload.

### Cumulative evidence for align=0

| source | align0 cases | failures |
|---|---|---|
| first matrix (8-16k, six chunk residues) | 12 | 0 |
| first matrix (mod-4 residues) | 4 | 0 |
| arm A (18-24k, indexer live) | 3 | 0 |
| arm A mod-4 at 21k | 4 | 0 |
| arm B continued-frontier stage 2 | 1 | 0 |
| **total** | **24** | **0** |

Rule of three: 95% upper bound on the failure rate now **12.5%**, down from
18.8%. Coverage spans 8k-24k tokens, six distinct residues mod the 4096-token
prefill chunk, all four compressor-group residues at two magnitudes, both sides
of the indexer activation boundary, and the continued-frontier save and reload.

Speed, consistently: 34% faster warm at 8-16k, 25% at 18-24k, 53% on the
continued reload.

### Remaining untested

- Multi-turn sessions chaining three or more checkpoints (arm B covers one
  continued save and one reload).
- The SWA 128-row window boundary specifically, as opposed to the ratio-4
  compressor boundary.
- Contexts past 24k, up to the 32768 allocation.
- Eviction interacting with alignment under a full disk budget.

### Verdict

The two gaps that blocked the recommendation are closed, and the invariant in
ds4_kvstore.c:35-39 did not produce a single observable difference in 24
align=0 cases spanning the conditions it names. The evidence now supports
changing the default; what remains is a judgement about how much residual risk
a 12.5% statistical bound represents for a correctness margin someone wrote
deliberately. That call is still the maintainer's, and the change is one line:
`KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS` at ds4_kvstore.c:41.

## Final four gaps closed

Harnesses `run-multiturn-chain.py`, `run-eviction.py`, and the shared
`run-align-matrix.py`; raw runs in `alignfinal/`.

### SWA 128-row boundary

`DS4_N_SWA = 128` and `raw_window = 128` (ds4.c:11209). Four prompts at a 21k
anchor placed the stored length on SWA residues 0, 1, 64, and 127:

| case | stored | stored%128 | identical |
|---|---|---|---|
| swa0 | 20992 | 0 | yes |
| swa1 | 20993 | 1 | yes |
| swa64 | 21056 | 64 | yes |
| swa127 | 20991 | 127 | yes |

4/4 byte-identical, reuse engaged, indexer active at this length.

### Contexts past 24k

27k passes byte-identical (stored 26972, warm 7.9 s vs cold 63.4 s).

**30k initially showed no reuse at all** — `reuse_engaged: false`, warm 70.0 s
vs cold 70.4 s. Cause found in the log rather than guessed: no `reason=cold`
save is emitted because 30004 exceeds `cold_max_tokens` (default 30000,
ds4_kvstore.c:1322). Confirmed by raising the gate to 32000, after which the
same prompt cached 29972 tokens and ran warm in 5.5 s vs 68.2 s cold,
byte-identical. So this is a documented gate doing its job, not an align
failure — but it means **the default configuration silently stops checkpointing
above ~30k tokens**, which is worth knowing independently of alignment.

### Multi-turn chain, 3+ checkpoints

First attempt produced only 2 checkpoints because 96-token turns never crossed
a 10000-token continued frontier. Re-run with
`--kv-cache-continued-interval-tokens 512` and 400-token turns over a 27k base:

| turn | prompt tokens | cached | wall |
|---|---|---|---|
| 1 | 27015 | 0 | 80.5 s |
| 2 | 27428 | 27415 | 16.8 s |
| 3 | 27844 | 27828 | 17.8 s |
| 4 | 28258 | 28244 | 16.9 s |

**10 checkpoints written across the chain** (continued saves at 4096, 8192,
12288, 16384, 20480, 24576, then cold at 26983, then continued at 27136 and
28160), each turn loading the previous turn's state. align0 final reply
byte-identical to a cold replay of the same conversation.

### Eviction under a full budget

512 MiB budget against ~240-290 MiB checkpoints, three distinct prompts, then
re-request the first:

| arm | checkpoints written | survived in cache | first prompt on re-request | output |
|---|---|---|---|---|
| align0 | 3 | 1 (337.6 MiB) | evicted -> cold fallback | identical to reference |
| align2048 | 3 | 1 (318.7 MiB) | evicted -> cold fallback | identical to reference |

Eviction genuinely fired in both arms, and the evicted entry fell back to a
clean cold prefill rather than loading stale state. Both arms also matched a
same-config reference run.

### A harness error I made, and the correction

The align2048 chain first reported `final_identical: false`, which looked like
a control failure. It was not. Investigation:

- Two fresh processes with no kv-disk: **identical** (1891 bytes).
- kv align=2048, cold then warm on one process: **identical** (1803).
- kv align=0, two fresh processes: **identical** (1825).

Each configuration is internally deterministic and reproducible; different
configurations produce different text for the same input. The chain's warm
turn-4 reply (1894) was produced by a server that had served three prior turns
in-process, and replaying that same turn sequence reproduced **all four turns
byte-for-byte**, including the 1894 final. So the variable was conversation
history, and comparing a mid-conversation reply against a single-shot cold
replay was an invalid comparison on my part, not engine drift.

This does not weaken the align result: within every configuration tested, warm
output equals cold output, which is the property under test.

### Cumulative evidence

| source | align0 cases | failures |
|---|---|---|
| first matrix (8-16k, chunk + mod-4 residues) | 16 | 0 |
| extension (18-24k, indexer live, mod-4, continued) | 8 | 0 |
| SWA residues 0/1/64/127 | 4 | 0 |
| deep context 27k and 30k | 2 | 0 |
| multi-turn chain turns (10 checkpoints) | 4 | 0 |
| eviction, survived and evicted paths | 2 | 0 |
| **total** | **36** | **0** |

Rule of three: 95% upper bound now **8.3%**, down from 12.5%. Coverage spans
8k-30k tokens, six chunk residues, all four compressor-group residues, four SWA
residues, both sides of the indexer boundary, chained continued checkpoints, and
eviction in both survived and evicted states.

### Independent finding worth acting on

`cold_max_tokens = 30000` silently disables cold checkpointing above ~30k
tokens while `--ctx` allows 32768. A user running near the context limit gets
no KV reuse and no diagnostic saying why. Not an alignment issue; a separate
default worth reviewing.

## cold_max_tokens: a silent gate that degrades reuse above 30k

Follow-up to the 30k observation in the alignment work. Investigated, measured,
and deliberately **not** changed — the fix is a default-policy decision.

### Two gates share one option

`cold_max_tokens` (default 30000, ds4_kvstore.c:34) controls two different
things:

1. **Save gate**, ds4_server.c:10196-10201. A cold checkpoint is considered
   only when `prompt_for_sync->len <= cold_max_tokens`. Above it,
   `cold_store_len` stays 0 and no cold save happens. There is no `else`, so
   **nothing is logged.** Confirmed on the d30k run: the entire server log
   contains no line mentioning a skip, and grepping for skip/blocked/exceeds
   returns zero matches. The only kv lines are a `live kv cache miss` and an
   `evict` store.
2. **Load gate**, ds4_kvstore.c:1322. A checkpoint *larger* than `cold_max` is
   loaded once and then `unlink`ed — treated as single-use. This one does log
   (`consumed file=`), and reads as deliberate.

Gate 2 looks intentional. Gate 1 is the one with no diagnostic.

### The impact is a degradation, not a cliff

My first reading was that reuse stops entirely above 30k. Tested at ctx 65536
with the default `cold_max`, and that was **too strong**: the continued-save
path still fires and supplies partial reuse.

| configuration | cached / prompt | cold | warm |
|---|---|---|---|
| 30004 tok, ctx 65536, cold_max default 30000 | 20480 / 30004 (68%) | 67.1 s | 26.0 s |
| 30004 tok, ctx 32768, cold_max raised to 32000 | 29972 / 30004 (99.9%) | 68.2 s | 5.5 s |

The 20480 came from `reason=continued` at the aligned 10240-step frontier, not
from a cold save. So above `cold_max` the server falls back to the last
continued frontier and replays everything after it: 9524 tokens instead of 32.

**Cost of the gate: 20.5 s per warm request, 4.7x slower warm than necessary.**
The size depends on how far the prompt sits past the last continued frontier,
so the worst case is a prompt just above `cold_max` and just past a frontier;
at the default 10240 step that is up to ~10k replayed tokens, roughly 24 s.

### Why the default looks misplaced

`cold_max_tokens` is absolute and independent of `--ctx`, so the affected band
grows with context:

| --ctx | tokens above cold_max | share of context |
|---|---|---|
| 32768 | 2768 | 8% |
| 65536 | 35536 | 54% |
| 100000 | 70000 | 70% |
| 393216 | 363216 | 92% |

The help text's own example is
`./ds4-server --ctx 100000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192`,
which under the default gets degraded cold reuse across 70% of its context
without saying so.

The gate does bound checkpoint size (~408 MiB at 30k, ~1.4 GiB at 100k tokens,
extrapolated from the measured 238 MiB / 17.5k tokens). But
`--kv-disk-space-mb` already bounds disk use directly, and the eviction tests
above show reclamation works correctly in both align arms — so size control
already has a dedicated, working mechanism.

### Recommended, not applied

Three options, in increasing order of change:

1. **Log the skip.** One `kv_logf` in the `else` of the save gate, saying the
   cold checkpoint was skipped because the prompt exceeds `cold_max_tokens`.
   Purely additive, no behaviour change, removes the silence.
2. **Scale the default with `--ctx`** rather than a fixed 30000, so the gate
   keeps its intent (bound one checkpoint's size) without swallowing most of a
   large context.
3. **Retire the save-side gate** and rely on `--kv-disk-space-mb` plus
   eviction, which are already the size-control mechanism.

Option 1 is safe enough to land on its own evidence. Options 2 and 3 change
reuse behaviour for every large-context deployment and want a maintainer
decision, so nothing here is applied.

## What removing cold_max_tokens would actually do

Tested rather than reasoned about, because the question is whether
`--kv-disk-space-mb` alone is a sufficient replacement.

### The budget path already handles oversized checkpoints, and logs it

Ran a 21k prompt (needs ~292 MiB) against a **200 MiB** budget with `cold_max`
raised out of the way, so the budget was the only thing that could refuse:

```
kv cache skipped tokens=20480 reason=cold because estimated file size
291.77 MiB (294.69 MiB with safety) exceeds ...
```

Both requests ran at 47 s with `cached=0`, output byte-identical, and **zero
.kv files left behind**. So the budget path degrades cleanly to no-cache
operation, with a log line naming the actual numbers — precisely what the
`cold_max` save gate fails to do.

The budget already provides all three pieces: a size check with 1% safety
headroom (`ds4_kvstore_file_size_fits`), a refusal that logs, and eviction to
reclaim space (verified earlier in both align arms).

### Checkpoint size is linear in tokens

Measured across 12 distinct checkpoint sizes from this session's runs:

| tokens | size | KiB/token |
|---|---|---|
| 4096 | 76.7 MiB | 19.16 |
| 10212 | 156.9 MiB | 15.74 |
| 16004 | 233.0 MiB | 14.91 |
| 21068 | 299.5 MiB | 14.56 |
| 28658 | 399.1 MiB | 14.26 |

Mean 15.15 KiB/token, converging near 14.3 at larger sizes. Linear, so
checkpoint size is predictable from token count.

### So removal would not create unbounded files

Under the default 4096 MiB budget a single checkpoint stops fitting at roughly
**274k tokens**, versus `cold_max`'s fixed 30000. Removing the save gate moves
the cutoff from an absolute constant to one derived from the disk budget the
user actually chose — it does not remove a cutoff.

| --ctx | checkpoint at full ctx | under default 4096 MiB budget |
|---|---|---|
| 32768 | 485 MiB | fits |
| 65536 | 970 MiB | fits |
| 100000 | 1479 MiB | fits |
| 393216 | 5818 MiB | refused by budget |

### But there is a real argument for keeping a per-entry cap

Eviction frees space by deleting **other** entries, so one large checkpoint that
fits the budget can evict everything else and monopolise the cache:

| single checkpoint | size | share of default budget |
|---|---|---|
| 30000 tok | 444 MiB | 11% |
| 65000 tok | 962 MiB | 23% |
| 100000 tok | 1479 MiB | 36% |
| 150000 tok | 2219 MiB | 54% |

With `--ctx 100000` and the default budget, one checkpoint would take 36% of the
cache. Several concurrent long conversations would thrash — each new one evicts
the previous, and every request pays full prefill anyway, which is worse than
the current degradation.

So `cold_max=30000` is a coherent policy: cap one entry at ~11% of the default
budget, leaving room for several concurrent conversations.

### Conclusion

Removing the gate entirely is **not** the right fix. Its two real defects are
narrower:

1. **It is silent.** The save gate has no `else` and logs nothing, while the
   budget path in the same file logs a precise reason. This is a
   straightforward inconsistency to fix.
2. **It is absolute.** A fixed 30000 scales with neither `--ctx` nor
   `--kv-disk-space-mb`, so its intent (bound one entry's share of the cache) is
   expressed as a constant that stops matching once either is raised.

Expressing the same intent as a fraction of the budget would preserve the
anti-monopolisation property while removing the dead band:

| cap | 4096 MiB budget | 8192 MiB | 16384 MiB |
|---|---|---|---|
| 25% of budget | 69k tokens | 138k | 277k |
| 50% of budget | 138k tokens | 277k | 554k |

Recommended: add the missing log line (safe, additive, no behaviour change),
and consider the fraction-of-budget form as a separate change. Neither applied
— the second alters reuse behaviour for every deployment.

## Three changes landed, and one measurement that invalidates earlier baselines

### --mtp-draft 1 was silently ignored (fixed, ea2c41ef3)

Embedded three-stage models raise the draft depth when the caller leaves it
unset, but the test was `e->mtp_draft_tokens <= 1` while all three front ends
initialised the option to 1. An explicit `--mtp-draft 1` was therefore
indistinguishable from unset and silently overridden. Fixed by using 0 as the
unset sentinel and keying the embedded default off `opt` rather than the
normalised engine value.

Verified: no flag -> `draft=2`; `--mtp-draft 1` -> `draft=1` (was 2);
`--mtp-draft 5` -> `draft=5`.

### The consequence: every SPEC_DISABLE baseline in this document was inflated

`DS4_MTP_SPEC_DISABLE=1` only skips the speculative *call*
(ds4_cli.c:484,1155). Drafting itself is gated on `mtp_draft_tokens > 1`
(ds4.c:27660-27661), so with the env var set the engine still leaves `draft=2`,
drafts every token, and discards the result. `--mtp-draft 1` skips the drafter
entirely.

Measured on the same prompt, byte-identical output:

| context | `--mtp-draft 1` (true nospec) | `DS4_MTP_SPEC_DISABLE=1` |
|---|---|---|
| 18k | 28.31 t/s | 23.96 t/s |

That is 18% of wasted drafting overhead carried by every `SPEC_DISABLE`
measurement in this document, including the 41.3 ms/token decode baseline.

### Speculation is a net loss on this model

With the flag now working, draft 2 can be compared against true non-speculative
decoding. Three runs each at 13k, byte-identical output:

| configuration | mean | stdev |
|---|---|---|
| `--mtp-draft 1` | 34.31 t/s | 0.40 |
| default draft 2 | 28.79 t/s | 0.14 |

**Speculation costs 16.1% at 13k and 13.6% at 18k.** The distributions do not
overlap. The earlier "1.545x speculative win" was draft-5 against draft-2, a bad
configuration against a good one; against no drafting at all, draft 2 loses.

This is a finding, not a change: `DS4_MTP_DEFAULT_DRAFT` is left at 2. Whether
the embedded default should be 1 is a maintainer decision, and the acceptance
rate may differ on other prompts or models.

### KV checkpoint alignment default removed (32c4ce935)

`KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS` 2048 -> 0, on the 36-case evidence
above. Verified on a live server at the new default: checkpoint grows from
16384 to 17972 tokens, warm path 7.3 s -> **4.1 s**, output byte-identical to
cold. The tokenizer-boundary trim is unchanged and alignment stays available
via `--kv-cache-boundary-align-tokens`.

## Attention kernel — the growth mechanism, and why the stage is already flat

Map in `attention-kernel-map.md`. A survey delegate produced the dispatch
geometry and loop structure; its section 4 (what grows with context) was wrong
and is corrected in that file.

The delegate attributed growth to the raw-window span. That cannot be right:
the raw loop is bounded by `args.n_raw`, the fixed 128-row SWA window.

The actual mechanism is the visible clamp on selected rows
(metal/dsv4_misc.metal:753-754):

```
    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
```

with an early `stop = true` once a top-k index exceeds `visible`. The kernel
attends `min(top_k, pos/ratio)` rows, not a flat 512:

| context | rows attended (raw + selected) | 16-row batches | barriers |
|---|---|---|---|
| ctx 512 | 128 + 128 = 256 | 16 | 32 |
| ctx 18k | 128 + 512 = 640 | 40 | 80 |

Ratio 2.50x against a measured 3.14x (9.69/3.09), so row count explains most of
the growth; the remainder is plausibly gather locality, since at ctx 512 the
visible rows are the earliest and near-contiguous while at 18k the 512 indices
are scattered across 4500 rows.

### What this changes about the target

**The stage saturates at 512 selected rows from ~4k tokens onward.** It is flat,
not growing, across the entire 4k-30k range. The earlier framing — attention
grows with context and that is why 18k is slow — is wrong beyond 4k. There is no
context-scaling left to fix there.

The remaining levers are per-row cost and the 80 barriers per token per layer:
256-thread threadgroups with 2 barriers per 16-row batch, a gather over
scattered indices, and a dtype-switchable compressed cache
(`comp_kv_f16`). Splitting the raw-window pass from the selected-row pass, or
widening the batch beyond 16 rows to halve the barrier count, are the concrete
candidates. None attempted here.

### Free lever (swept above)

`DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` (64..4096, default 1024) moves the
dense/sparse crossover with no code change. The in-code comment (ds4.c:13497)
says the score/top-k setup dominates the smaller attention scan below the ~2K
frontier — now quantified: setup is 2.08 ms/token. A sweep at 18k would show
whether 2048 or 4096 nets out better. It must not be confused with
`DS4_N_INDEXER_TOP_K`; lowering that changes the model's candidate set.

### Method notes for this slice

- Patch is diagnostic-only and fails soft. Default-path output verified
  byte-identical (`cmp` of two 24-token greedy runs, both `31.4-31.9 t/s`), zero
  diagnostic output when the env vars are unset, `make` clean,
  `make embedded-mtp-test` PASS.
- The stderr stream carries two series: 105 samples labelled `score`/`topk`/
  `attention` are the **prefill** batch path (5 chunks x 21 layers, positions
  0/4096/8192/12288/16384), and 252 samples labelled `decode_*` are decode
  (12 tokens x 21 layers). Only the `decode_*` series is used above.
- Prefill's own indexer cost is large and unanalyzed here: 14.77 ms score +
  2.42 ms topk + 43.12 ms attention per chunk per layer-set. The 18k prefill
  measured 473.7 t/s in this run.

### Correction to the stage map

An earlier survey of this code inverted the stage-label mapping, reading each
label as naming the region that *follows* it. The boundary helper reports
`now - *stage_t0` and then resets (ds4.c:17646-17657), so a label names the
region that *ends* at it. `compressor_indexer` is ds4.c:15346-15646 (compressor
updates + the full indexer chain); `attention` is 15646-15695 (the indexed
attention dispatch). The published tables were always correct; only the
intermediate analysis was wrong. Details in `indexer-map.md`.

### Remaining caveats

- Serialized measurement excludes native inter-stage overlap; the constant
  ~4 ms native-vs-serialized gap at both contexts validates the comparison.
- q_path byte estimate derived from shape constants (q_a 4096x1024, q_b
  1024x32768, q8_0); bandwidth percentages carry that uncertainty.
- 18k run used the same 18k code-review prompt as all prior benchmarks.

## Baseline reconciliation — the 15.56 vs 24.04 discrepancy resolved

Flagged three times this session as unreconciled: `results.md` recorded 24.2 t/s
nospec at 18k while the mixed-quantization work recorded 15.56 plain and 24.04
draft-2, suggesting the baseline might already include speculation and that the
headroom arithmetic could be wrong. Resolved by reading the old run logs and
re-measuring.

**Both eras used ctx=32768.** The suspicion that the old runs were at 65536 was
wrong; `context buffers ... (ctx=32768 ...)` appears in both
`context18k-plain.stderr` and `context18k-draft2.stderr`. Context setting was
never the confound.

**The old "plain" run was not plain.** `context18k-plain.stderr` reports
`draft=5`, not draft 1. Its 14.31 t/s is the cost of an over-deep draft, where
verification loses more than acceptance wins. So the widely-quoted 1.545x was
draft-5 versus draft-2 — a bad configuration against a good one — not plain
decoding versus speculative decoding.

Re-measured today on the current binary, same prompt, ctx 32768:

| configuration | generation |
|---|---|
| `DS4_MTP_SPEC_DISABLE=1` (true nospec) | 24.19 t/s |
| default draft=2 | 24.32 t/s |
| old draft=5 run | 14.31 t/s |

**Speculation is worth ~0.5% at 18k (24.19 -> 24.32), not 1.545x.** The
speculative win reported earlier was real only relative to a misconfigured
baseline.

### Consequence for this session's arithmetic

The 41.3 ms/token figure underpinning every decode-lever calculation equals
24.21 t/s, which matches today's measured nospec 24.19 t/s within noise. The
baseline **is** speculation-free as assumed. Every headroom number in this
document stands unchanged.

### Unrelated finding: --mtp-draft 1 does not take effect

`./ds4 --mtp-draft 1 ...` still logs `draft=2` and runs at 24.28 t/s. The
embedded-MTP default assignment at ds4.c:26180 only replaces
`mtp_draft_tokens` when it is `<= 1`, so an explicit `--mtp-draft 1` is
indistinguishable from unset and gets overwritten by `DS4_MTP_DEFAULT_DRAFT`.
The help text says "Default: 1 (2 on embedded three-stage MTP models)", so
requesting 1 explicitly is silently ignored on exactly the models where a user
might want it. Not fixed here — it is a CLI contract bug needing its own slice,
and `DS4_MTP_SPEC_DISABLE=1` is the working way to get non-speculative decode.

## Caveats

- Fusion-sweep numbers are ctx 512. At 18k ctx the engine measures 24.19 t/s
  nospec (verified above); the fusion decomposition was not repeated there and
  attention/KV costs differ.
- The 6.26 GiB/token figure is derived from shape constants and quant types
  (routed 6-of-256 experts + shared + attention + drafter), not a byte-exact
  trace of buffer reads.
- Run-to-run GPU-exec noise is ~0.1-0.2 ms; fusion deltas at or below that are
  treated as zero.
