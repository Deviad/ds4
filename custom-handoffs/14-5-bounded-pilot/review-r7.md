# Story 14.5 independent review — r7 final

## Verdict: PASS

R6 blocking findings closed. Phase B now binds marker SHA-256, strict JSON parsing, and supplied report equality to one byte snapshot. Terminal-test tmp containment covers every specified direct terminal family through an enforced AST inventory. Retained-partial lock cleanup and evidence ordering remain green through the real `run_phase()` terminal path. No new blocking defect found.

## R7 closure verification

### Phase B immutable report snapshot — CLOSED

`scripts/ds4_segmented_pilot.py:114-123,274-290`

- `_read_json_snapshot()` performs one `Path.read_bytes()` call.
- SHA-256 and strict JSON parsing consume that same in-memory payload.
- `_reject_duplicate_keys` rejects duplicate keys at every object depth.
- Validator compares marker hash to snapshot hash, then compares snapshot object to the report supplied by `run_phase()`.
- No report-path hash/reopen sequence remains in the validator.

Independent tmp-only race probes:

```text
replacement_after_snapshot: accepted bound old snapshot; canonical_reads=1; path replaced=true
supplied_unbound_replacement: rejected; canonical_reads=1; Phase A report object does not equal hashed payload
duplicate_keys: rejected; Phase A hashed report payload is not readable JSON
```

This proves a replacement after the single read cannot split marker hashing from parsing. The bound snapshot remains internally consistent; an unbound supplied object and duplicate-key payload fail closed.

`tests/test_ds4_segmented_pilot.py:343-409` also preserves a mutation-sensitive regression check against the legacy `file_sha256()`/reopen boundary.

### Comprehensive terminal tmp containment — CLOSED

`tests/test_ds4_segmented_pilot.py:28-52,719-742`

Independent AST inventory over the current test source found:

```text
covered terminal tests: 22
unguarded: []
families: _acquire_ft_lock, _install_timeout_watchdog, _release_ft_lock,
          _write_failure_evidence, _write_success_evidence, run_phase
```

Every test directly invoking a specified terminal family installs `install_tmp_fs_guard()`. The meta-test prevents later unguarded direct terminal tests from entering the suite.

Representative direct terminal/binding run:

```text
7 passed, 1 warning in 0.10s
```

Covered report binding, retained-partial quarantine, partial-progress cleanup ordering, watchdog-cancellation failure, terminal mutation inventory, resolved-tmp operations, and Phase A success publication.

### Retained-partial lock and evidence behavior — RETAINED GREEN

Focused and representative runs preserve the r6-verified behavior:

- acquisition failure with retained partial ownership enters real `run_phase()` cleanup;
- exactly one recorded release attempt occurs;
- canonical `.ds4-ft.lock` pathname is removed by quarantine;
- lifecycle remains truthful: `acquired=false`, `released=false`, `release_attempts=1`, `quarantined=true`;
- failure progression remains training failure → watchdog cancellation → release → failure report → fail marker;
- failure markers bind surviving reports; success markers do not survive terminal publication failure.

## Independent test evidence

Focused pilot suite:

```text
71 passed, 1 warning in 0.31s
```

Direct two-node real-trainer targets:

```text
2 passed, 1 warning in 0.06s
```

Exact documented six-file suite:

```text
317 passed, 3 skipped, 1 warning, 2 subtests passed in 21.61s
```

Three skips remain the authorized protected real-GGUF gates. No real model, dataset, adapter, training, inference, CUDA, distributed, network, or protected GGUF execution occurred.

Evidence normalization: coder notes report `2 passed, 45 deselected` from a selection run. Final review uses the direct two-node command above; no deselection occurs. This does not affect the green target results.

## Reproducibility and scope gates

- `python3 -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py`: PASS, with bytecode redirected outside the checkout.
- `git diff --check`: PASS.
- `git diff --cached --check`: PASS.
- All six verdict-contributing test files: TRACKED via `git ls-files`.
- Provider SHA-256: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`.
- Smoke SHA-256: `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8`.
- Vendor inner HEAD / outer gitlink: `80fab4e419a57f9465bb9e2f4e90010d645e124c`; vendor clean.
- Protected provider/smoke/vendor paths show no scope drift.
- No `.cmux-status` marker was staged before this review artifact.
- Canonical docs remain conservative: synthetic implementation/review only; future Phase A and Phase B execution requires separate operator authorization in a visible panel; continuity and readiness non-claims remain explicit.

## Gate result

Reviewer gate: PASS.

Real execution remains unauthorized. Test Manager verdict and explicit operator authorization remain separate downstream gates.
