# Story 14.3a Test Manager final

Validate filtered smoke dataset repin. Read requirements.md, architecture.md, coder-notes.md and r12 gates.

Run exact-fork `tests/test_ds4_segmented_smoke.py`, `tests/test_finetune_ds4.py`, `tests/test_mlx_lm_source.py`, `tests/test_ds4_segmented_loss_and_grad.py`, py_compile, plus new filtered-path/catalog tests. Verify current four-file functional hash, Path A tracked-path count 365/digest `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`, protected hashes, `.gitmodules`/vendor pin, ADRs, and filtered dataset report/counts without rerunning real smoke. No real assets/Phase 2/commit/push. Write only `custom-handoffs/14-3-filtered/test-report.md` and marker. GREEN only reproducible evidence.