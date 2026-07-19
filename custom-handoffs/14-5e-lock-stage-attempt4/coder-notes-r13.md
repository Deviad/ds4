# Story 14.5e — Coder r13 exact publication bytes + early rollback

## Result

GREEN focused and canonical synthetic verification. No real A4/B4 model, dataset, provider, training, inference, cleanup, commit, or push ran.

## RED-first corrections

- Added tracked publication seam tests for byte-only serializer rewrites and semantic mutations at the Phase B success marker, final report, and final success marker.
- Added tracked early disagreement tests for Phase A authorization, Phase A admission-lineage, and Phase B authorization roots; every attempt-4 success marker and final report must be absent after rejection.
- The byte-only matrix was RED before the production correction: current-file hash comparison accepted semantically identical rewritten bytes.

## Production correction

- Added `_serialize_json_bytes()` as the single canonical JSON serializer and `_write_exact_json()` as the publication writer/readback boundary. Expected bytes and SHA-256 are retained before each write; persisted bytes, digest, parsed value, and strict marker validation are checked after write.
- A4 marker report hashes now use the retained report digest instead of rehashing the current report path. Final marker validation uses the retained final-report digest.
- Attempt-4 publication rollback is initialized before trusted/untrusted authorization and admission-root comparisons. Phase A authorization, Phase A admission-lineage, and Phase B authorization disagreement remove all attempt-4 success markers before raising.

## Validation

- Focused `tests/test_ds4_segmented_pilot.py`: **735 passed, 1 warning**.
- Canonical documented six-file command: **981 passed, 3 skipped, 1 warning, 2 subtests passed**.
- `py_compile` for `scripts/ds4_segmented_pilot.py`, `scripts/finetune_ds4.py`, `tests/test_ds4_segmented_pilot.py`, and `tests/test_finetune_ds4.py`: PASS.
- `git diff --check` and `git diff --cached --check`: PASS.
- Every verdict-contributing canonical test file is tracked by `git ls-files`.
- Protected C/Objective-C/Metal/CUDA/vendor paths: no diff.

Independent Reviewer PASS and Test Manager GREEN remain separate gates. No verdict is fabricated.
