# Coder r10 notes

Revision: 14-3-coder-r10
Attempt: 1
Functional hash (staged four-file): `d9b4f9e60ae4158423f41dd09de62611355427525c022973faefef077ae2e584`

Command: `git diff --cached --binary -- scripts/ds4_segmented_smoke.py tests/test_ds4_segmented_smoke.py scripts/finetune_ds4.py docs/architecture.md | shasum -a 256`

## Staged files
- `scripts/ds4_segmented_smoke.py` (newly tracked)
- `tests/test_ds4_segmented_smoke.py` (newly tracked)

`scripts/finetune_ds4.py` and `docs/architecture.md` are already tracked, unchanged in this slice — no staged diff, included in hash for four-file consistency.

## Test results

### test_ds4_segmented_smoke.py
88 passed, 0 failed

### test_mlx_lm_source.py
13 passed, 2 failed (pre-existing, not r10 scope)
- `test_current_repository_submodule_metadata_and_gitlink` — submodule gitlink not staged
- `test_path_a_evidence_exact_hash_and_tracked_paths` — agent-output files not tracked

### test_finetune_ds4.py
52 passed, 0 failed

### Protected path checks
- Path A protected source hash: byte-identical (test_provider_source PASS)
- Path A protected test hash: byte-identical (test_provider_test PASS)
- Source sentinel hash: byte-identical (test_source_sentinel PASS)

## Per-AC verdict

### 1. LoRA canonical allowlist (Reviewer finding 1)
- `test_generated_nested_lora_accepted` now monkeypatches `mlx_lm.__file__` to vendor path so fork identity check passes; canonical DS4 keys (`self_attn.q_a_proj`, `self_attn.q_b_proj`, `self_attn.kv_proj`) are accepted. PASS.
- `test_missing_keys_rejected` — rejects config with no `keys` field. PASS.
- `test_wrong_allowlist_rejected` — rejects config with wrong key `q_proj`. PASS.
- Smoke script `_check_preflight` uses canonical DS4 allowlist set. 

### 2. Dataset 4096 bound and pinned smoke identity (Reviewer finding 2)
- Conservative whitespace token bound hardcoded to 4096 in smoke script (was `getattr(args, "max_seq_length", 4096)`, now always 4096).
- `test_token_bound_5000_rejected` — 5000 whitespace tokens → rejected. PASS.
- `test_max_seq_length_bound_pinned` — 4100 tokens with `max_seq_length=8192` → rejected because bound is pinned at 4096. PASS.

### 3. Preflight code 3 / detailed reason (Reviewer finding 3)
No changes needed — already implemented in r9.
- `test_preflight_exit_code_preserved` PASS
- `test_preflight_reason_report` PASS
- `test_parser_failure_workspace_marker` PASS
- `test_parse_failure_inside_terminal_lifecycle` PASS

### 4. Exact recursive schema (Reviewer finding 4)
No changes needed — already implemented in r9.
- `test_list_subclass_rejected` PASS
- `test_tuple_subclass_rejected` PASS

### 5. MLX version/fork fail closed (Reviewer finding 5)
No changes needed — already implemented in r9.
- `test_mlx_version_none_rejected` PASS

### 6. Behavioral cleanup mutation with warning (Reviewer finding 6)
- `test_cleanup_release_control` — sets `_LOCK_OWNED_PATH` before `main()`, verifies lock is released (None after cleanup). PASS.
- `test_cleanup_release_mutant_still_holds` — sets `_LOCK_OWNED_PATH` before `main()`, verifies lock survives in mutant (not None after cleanup). PASS.
- `test_cleanup_release_exception_warning` — monkeypatches `_release_ft_lock` to raise OSError; verifies `_CLEANUP_WARNINGS` records the exception. PASS.
- Smoke script: added module-level `_CLEANUP_WARNINGS: list[str]` list; `finally` block records release exceptions instead of silent `pass`.

### 7. mlx-lm==0.31.3 pin (Reviewer finding 7)
`python-envs/mlx/pyproject.toml` already contains exact `mlx-lm==0.31.3` (not `>=`). No changes needed.

### 8. Functional identity (Task item 8)
- Staged diff computed: `d9b4f9e60ae4158423f41dd09de62611355427525c022973faefef077ae2e584`
- Four files: `scripts/ds4_segmented_smoke.py`, `tests/test_ds4_segmented_smoke.py`, `scripts/finetune_ds4.py`, `docs/architecture.md`
- No full-diff self-reference.
- `git diff --cached --check`: clean (no whitespace errors).

## Tracking hygiene
Both `scripts/ds4_segmented_smoke.py` and `tests/test_ds4_segmented_smoke.py` were previously untracked. Now staged with `git add`. `git ls-files` confirms both are tracked in index.

## Known pre-existing failures (outside r10 scope)
- `test_mlx_lm_source.py::test_current_repository_submodule_metadata_and_gitlink` — submodule gitlink not in index
- `test_mlx_lm_source.py::test_path_a_evidence_exact_hash_and_tracked_paths` — agent-output Path A files untracked

These do not block r10. Protected source/test/sentinel hashes match.
