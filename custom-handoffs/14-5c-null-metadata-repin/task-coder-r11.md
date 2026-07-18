# Story 14.5c — Coder r11 final evidence closure

Read latest standby review. Test-first; production only if test exposes defect.

1. True loaded-MLX negative matrix: invoke normal mx.load/production loaded-schema path for missing/extra/duplicate tensor names, dtype, shape, layout, payload mismatch; assert fail closed at load or schema/value boundary. Keep positive exact keys/dtype/shape/values.
2. Every A3/B3/final marker field: independent missing, wrong JSON type (including every numeric/digest/resume field), valid-type wrong value, extra key.
3. Drive each 2 A3 + 4 B3/final success write seam through `run_phase()` caller failure path; assert exact phase/final failure reports and fail marker report path/hash/contract/attempt3; no OK survives.
4. Attempt2 coordinated mutation instrumentation must prove path/hash/target guards reached; add weakened-verifier mutant that causes oracle failure/acceptance and production rejection.
5. Correct docs/coder notes. Stage exact no drift/markers. Vendor-first canonical six-file. No real artifacts/training/cleanup/commit/push/delegate/cmux. Write coder-notes-r11.md.