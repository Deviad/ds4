# Story 14.5c — Coder

Read requirements.md and architecture.md fully. Implement strict TDD RED→GREEN, synthetic only.

Scope: scripts/ds4_segmented_pilot.py; scripts/finetune_ds4.py; tracked tests/test_ds4_segmented_pilot.py + any explicitly architecture-listed tracked tests; docs/architecture.md; docs/technical-spec.md; coder-notes.md.

Implement absent/null/object metadata acceptance at existing canonical parser seam; reject all other types; preserve duplicate/tensor/digest semantics and real mx.save_safetensors integration. Repin live runtime/catalog from attempt2 to explicit fixed attempt3 only; parser rejects attempt2 live. Central A3/B3 paths/spec/commands; attempt2 command retirement; immutable read-only attempt1 + consumed attempt2 historical verification/binding; exact authorization/revision/command/source/protected identity and pre-log no-write ordering; canonical A3/B3 report/admission/publication/resume matrices. No arbitrary attempt registry/dynamic suffix/attempt4.

Protect all unrelated runtime/vendor/PathA/CUDA/Metal/SSD files. No real model/dataset/attempt artifacts access, training, cleanup, commit, push, delegate, or cmux. Stage exact slice, write coder-notes.md + marker.