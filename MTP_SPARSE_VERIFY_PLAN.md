# MTP / DSpark Sparse Verify Completion Plan

## Goal

Validate and ship the sparse speculative-verification path that is already in
the DS4 engine, including MTP-aware fine-tuning output. The remaining work is
regression coverage, real-model correctness validation, performance gating,
and fine-tuning-pipeline integration—not a new target-verifier implementation.

## Verified current implementation

The current tree already contains the core path:

- `gguf-tools/deepseek4-quantize.c` builds standalone DSpark support GGUFs from
  `mtp.*` tensors via `--dspark-support`.
- `ds4.c` detects and loads `DS4_SUPPORT_DSPARK`, runs legacy MTP and DSpark
  draft generation, and routes both through the shared target verifier.
- `metal_graph_verify_suffix_tops()` runs the target batch through all model
  layers using `metal_graph_encode_layer_batch()`.
- Ratio-4 CSA layers with more compressed rows than
  `DS4_N_INDEXER_TOP_K` use batched indexer scoring, top-k selection, and
  `ds4_gpu_attention_indexed_mixed_batch_heads_tensor()` rather than full
  compressed-cache attention.
- The Metal indexed-attention wrapper sorts every multi-token top-k row into
  ascending cache order with `kernel_dsv4_sort_i32_rows_asc` before attention.
  Single-token fast decode may retain score order outside quality mode.
- Flash attention is architecture-specific: the first two layers use raw SWA,
  ratio-4 layers use CSA top-k after the threshold, and ratio-128 layers attend
  their much smaller compressed cache. Sparse verify should preserve that same
  per-layer architecture rather than forcing CSA top-k onto every layer.
- Existing live-test entry points include `--mtp-verify-depth` and
  `--dspark-verify-depth`.

## Exact production call chain

```text
CLI/server generation loop
  -> ds4_session_eval_speculative_argmax()
     -> ds4_session_eval_probe_tp()                 # target token + draft prep
     -> ds4_session_eval_dspark_speculative_argmax()# DSpark support
        or legacy MTP branch in the same function
     -> metal_graph_verify_suffix_tops()
     -> metal_graph_verify_suffix_tops_impl()
     -> metal_graph_encode_layer_batch()            # every target layer
     -> metal_graph_encode_layer_attention_batch()
     -> ratio-4 and n_comp > DS4_N_INDEXER_TOP_K:
          ds4_gpu_indexer_scores_decode_batch_tensor()
          ds4_gpu_indexer_topk_tensor()
          ds4_gpu_attention_indexed_mixed_batch_heads_tensor()
            -> kernel_dsv4_sort_i32_rows_asc        # multi-token verify
            -> kernel_dsv4_indexed_mixed_attention_heads8
```

A partial acceptance rolls the speculative frontier back and replays only the
accepted prefix through normal one-token decode. A full acceptance keeps the
batch verifier state and reads the final verifier-logit row.

---

## Remaining user stories

### Story 1 — Lock the sparse route and cache-order invariant

**Status:** Complete in the working tree. Reviewer `PASS` and Tester `GREEN`
are recorded for staged revision `23934014-AF84-4ABE-AB9E-EAD96426949B`.
The focused model-free Metal target runs six route, fallback, transition, sort,
kernel, and binding tests. The repository-wide `make` and `make test` remain
blocked by the independently reproduced clean-HEAD thirteen-symbol Metal link
baseline; no unrelated linker repair is included in this story.

As a DS4 maintainer, I want focused regression coverage for the existing
verifier route, so future attention optimizations cannot silently restore dense
verification or remove cache-order sorting.

**Acceptance criteria:**

- A targeted test exercises a verify batch with more than one token on a
  ratio-4 layer where `n_comp > DS4_N_INDEXER_TOP_K`.
- The test proves the path reaches batched indexer scoring, top-k selection,
  and indexed mixed attention; it fails if the verifier uses full
  compressed-cache attention instead.
- The test proves multi-token selected rows are sorted ascending before the
  indexed-attention kernel consumes them.
- The single-token score-order optimization and the short-context
  `n_comp <= DS4_N_INDEXER_TOP_K` path are covered as intentional, separate
  behavior.
- Both DSpark and legacy MTP are shown to enter the shared verifier; no duplicate
  verifier implementation is added.

### Story 2 — Prove target correctness with real model artifacts

As a DS4 operator, I want the existing speculative path validated against plain
target decode, so accepted drafts never change the generated token stream or
corrupt cache state.

**Acceptance criteria:**

- Produce or identify a compatible DSpark/MTP support GGUF and verify that the
  current loader recognizes it alongside the target GGUF.
- Run the existing `--mtp-verify-depth` and `--dspark-verify-depth` live tests
  with their required model artifacts; record pass, fail, or skip explicitly.
- At temperature zero, compare speculative and plain decode over prompts that
  cover short and long context, structured output, code, prose, and near-tie
  logits.
- Cover full acceptance, partial acceptance, first-draft rejection, EOS, and
  rollback/replay. The committed token sequence and subsequent logits must
  match the selected release semantics.
- Confirm main-model KV, compressor, indexer, and speculative-cache frontiers
  remain valid after both full and partial acceptance.

### Story 3 — Benchmark and make the release decision

As a DS4 operator, I want reproducible measurements of the existing path, so it
is enabled only when sparse verification produces a real end-to-end gain.

**Acceptance criteria:**

- Benchmark plain decode, legacy MTP, and DSpark on the same target/support
  model pair at short, medium, and long context.
- Record end-to-end tokens per second, draft acceptance, verify time, replay
  time, memory overhead, and the frequency of sparse versus short-context
  fallback attention.
- Confirm that disabling speculative decoding leaves normal Metal performance
  unchanged.
- Release gate: no correctness regression and no non-speculative performance
  regression. The performance target remains at least a 1.2x end-to-end gain
  before promoting the feature beyond opt-in.
- Commit a benchmark report with exact hardware, model files, quantization,
  command lines, environment switches, and raw result locations.
- Update user-facing documentation with the supported mode, opt-in command,
  limitations, and the measured release decision.

### Story 4 — Produce MTP-capable fine-tuned model artifacts

As a DS4 fine-tuning operator, I want the fine-tuning pipeline to preserve and
package the MTP/DSpark support tensors, so fine-tuned target models can use the
same sparse speculative-verification path.

**Acceptance criteria:**

- The fine-tuning pipeline exposes an explicit MTP/DSpark option and carries it
  through hybrid-model assembly, conversion, and quantization.
- MTP/DSpark support tensors come from the compatible original checkpoint and
  are not silently dropped or rewritten by a LoRA merge that targets only the
  main model.
- The pipeline produces a target GGUF plus the support GGUF or equivalent
  artifact layout expected by the current DS4 loader.
- The generated pair loads successfully and passes the real-model correctness
  gates from Story 2, including speculative-versus-plain temperature-zero
  comparison.
- A no-MTP fine-tuning run remains unchanged and does not acquire support-model
  dependencies or additional runtime overhead.
- Fine-tuning documentation describes the option, source-checkpoint
  compatibility requirements, generated files, and the DS4 invocation needed
  to enable the result.

## Execution order

```text
Story 1 (regression lock) -> Story 2 (real-model correctness)
                          -> Story 3 (benchmark and release gate)
                          -> Story 4 (fine-tuning integration)
```

Story 1 should be small and deterministic. Story 2 establishes the correctness
gate reused by Stories 3 and 4. Stories 3 and 4 may proceed in parallel after
Story 2, but both must pass before this MTP plan is complete.

The fine-tuning implementation may remain in the separate fine-tuning worktree;
that repository boundary does not remove it from this MTP plan.

## References

- DS4 target verifier and speculative orchestration: `ds4.c`
- Metal top-k sorting and indexed attention wrapper: `ds4_metal.m`
- Metal sort and indexed-attention kernels: `metal/dsv4_misc.metal`
- Support GGUF conversion: `gguf-tools/deepseek4-quantize.c`
- Live verification tests: `tests/ds4_test.c`
- Fine-tuning integration worktree: `ds4-finetuning`
- Fine-tuning pipeline entry point: `scripts/finetune_e2e.py`
- Fine-tuning documentation: `docs/finetuning-guide.md`
- Prior sparse-verifier evidence: `Deviad/llama.cpp` commit `5217c92fa`
- Dense DSpark comparison: `antirez/ds4#502`
