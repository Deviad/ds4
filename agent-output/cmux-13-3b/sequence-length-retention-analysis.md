# DeepSeek V4 fine-tuning sequence-length retention audit

Date: 2026-07-14

## Purpose

Determine whether `--max-seq-length 4096` is sufficient for the reasoning-heavy Computer Science fine-tuning dataset, where completions include explicit thinking traces.

This file is evidence for a later BA re-pin of Story 13.6. It does not authorize changing the current 4096-token smoke, which remains the first backward/memory gate.

## Inputs and method

- Dataset: `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096`
- Model/tokenizer: `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer.json`
- Splits: train 15,170; validation 819; test 824
- Schema: `prompt`, `completion`
- Tokenizer loaded directly with `tokenizers.Tokenizer.from_file(...)` because Transformers does not recognize the remapped `deepseek_v4_nn` model type.
- Token counts use `encode(..., add_special_tokens=False)` on prompt and completion separately. The passthrough chat template adds no additional framing because the dataset prompt is already framed.
- Completion retention at cap `L` is `max(0, min(completion_tokens, L - prompt_tokens))`.

## Dataset distribution

### Train

- Prompt tokens: p50 63; p75 1,709; p90 2,211; p95 2,333; p99 2,589; max 4,036.
- Completion tokens: p50 465; p75 969; p90 1,798; p95 2,445; p99 5,421; max 21,780.
- Combined tokens: p50 724; p75 2,547; p90 3,125; p95 3,675; p99 6,413; max 22,573.
- Explicit `<think>` marker: 15,170/15,170 (100%).
- Code fence: 1,838/15,170 (12.1%).

### Validation

- Completion tokens: p50 460; p95 2,542; p99 5,519; max 20,913.
- Combined tokens: p50 809; p95 3,812; p99 6,246; max 23,104.
- Explicit `<think>` marker: 819/819 (100%).

### Test

- Completion tokens: p50 448; p95 2,571; p99 5,012; max 15,632.
- Combined tokens: p50 656; p95 3,755; p99 6,259; max 15,691.
- Explicit `<think>` marker: 824/824 (100%).

## Completion-target retention

| Cap | Train fully retained | Train target tokens retained | Validation fully retained | Validation target tokens retained | Test fully retained | Test target tokens retained |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 68.44% | 59.68% | 65.20% | 56.85% | 68.93% | 59.59% |
| 4,096 | 96.82% | 89.78% | 96.21% | 89.33% | 96.24% | 91.17% |
| 8,192 | 99.35% | 96.08% | 99.51% | 95.96% | 99.64% | 97.84% |
| 16,384 | 99.97% | 99.89% | 99.88% | 99.05% | 100.00% | 100.00% |
| 32,768 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

At 4,096:

- Train: 483 examples (3.18%) have truncated targets; no prompt consumes the entire cap.
- Validation: 31 examples (3.79%) have truncated targets; no zero-target record.
- Test: 31 examples (3.76%) have truncated targets; no zero-target record.
- The small long-example tail contains disproportionate supervision: about 10.22% of train completion tokens are discarded.

At 2,048:

- 2,803 train records (18.48%), 174 validation records (21.25%), and 159 test records (19.30%) have zero target tokens after prompt masking.
- This explains the observed 2048 validation `NaN`; 2048 is not a valid completion-only fallback.

## Model context

Local config:

- `max_position_embeddings = 1,048,576`
- YaRN `original_max_position_embeddings = 65,536`
- YaRN factor 16

The model context does not force the 4,096 cap. The cap is a training resource/throughput choice.

## Decision recommendation

1. Keep 4,096 for the current 20-iteration smoke. It preserves non-empty completion targets and is the minimum proof for backward, memory, optimizer, and adapter saving.
2. Do not run the full 5,000-iteration Story 13.6 solely at 4,096 without a BA/Architect re-pin. It would discard about 10.22% of train target tokens, concentrated in long explicit-thinking examples.
3. After the 4K smoke is GREEN, run an 8K smoke. 8K retains 96.08% of train target tokens and fully retains 99.35% of train examples.
4. Run a bounded 16K smoke before committing to the final schedule. 16K retains 99.89% of train target tokens and fully retains all but four train examples.
5. Preferred final policy: length-bucketed or phased training—4K ordinary examples, 8K longer reasoning/code examples, and 16K long-tail examples. If the trainer cannot bucket lengths safely, use 8K as the minimum fixed cap and document the remaining 3.92% target-token truncation.
6. Inference generation limits remain separate from training caps. The pretrained model can generate beyond 4K, but supervised long-thinking behavior is weakened when fine-tuning targets are truncated.

## Canonical follow-up

After the active Story 13.3b-5f primitive review/test gate completes, BA should update:

- `docs/backlog.md` Story 13.6 acceptance criteria;
- `docs/technical-spec.md` final-training sequence policy;
- any full-training command that still hardcodes `--max-seq-length 4096`.

No current smoke command changes are authorized by this evidence file alone.
