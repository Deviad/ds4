# Story 14.5b — canonical MLX version source and A2 reauthorization architecture

## 1. Verdict

**GO for minimal TDD repair and synthetic verification only.**

No real model or dataset access, provider execution, training, inference, namespace mutation, cleanup, deletion, overwrite, Phase A2/B2 invocation, commit, or push is authorized.

The fixed, still-empty attempt-2 namespace is retained. No attempt-3 repin is required. `scripts/finetune_ds4.py` already supplies the required generated-wrapper test boundary and remains byte-for-byte protected.

## 2. Evidence and failure disposition

The only authorized A2 invocation at `c910d1ba33912236ee87f3f9bdfb5b31edece6e7` exited `1` before opening its log:

```text
pilot launch check failed closed: MLX version mismatch: None
```

Recorded facts remain immutable:

- `mlx.__version__ is None`;
- `importlib.metadata.version("mlx") == "0.31.2"`;
- training/provider calls/optimizer updates were `0/0/0`;
- every fixed A2 output, checkpoint, config, log, report, phase marker, final report, and final marker remained absent;
- no cleanup, overwrite, fallback, or retry occurred;
- the authorization was consumed.

Canonical record:

```text
agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md
SHA-256 cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41
```

The repair does not rewrite that event or imply that no wrapper process was previously launched under the attempt-2 ordinal.

## 3. Minimal production design

### 3.1 Import and lookup style

Add one standard-library import at module scope:

```python
import importlib.metadata
```

Keep `import mlx` and `import mlx_lm` inside `_runtime_preflight()`. Importing `mlx` remains mandatory runtime-availability evidence even though its version attribute is ignored.

After successful module imports, obtain the installed version only through:

```python
mlx_version = importlib.metadata.version("mlx")
```

Do not introduce a second version helper, alternate identity field, environment seam, configuration seam, shell command, package-map lookup, `pip` call, MLX-LM version lookup, or fallback.

### 3.2 Exception and value contract

Use explicit fail-closed categories in this order:

1. `import mlx` or `import mlx_lm` raises `ImportError`:
   preserve the existing runtime-source-unavailable `PilotError` category.
2. `importlib.metadata.version("mlx")` raises `importlib.metadata.PackageNotFoundError`:
   raise `PilotError` with a distribution-metadata-unavailable diagnostic.
3. The metadata call raises any other `Exception`:
   raise `PilotError` with a distribution-metadata-error diagnostic and the exception type; chain the original exception.
4. Metadata returns an empty value or a value whose exact type is not `str`:
   raise `PilotError` with a distribution-metadata-malformed diagnostic.
5. Metadata returns a string other than exact `"0.31.2"`:
   raise `PilotError` with a distribution-metadata-version-mismatch diagnostic containing the observed representation.
6. Metadata returns exact `"0.31.2"`:
   continue preflight and place that same value in `identity_manifest.immutable.mlx_version`.

Recommended stable diagnostics:

```text
MLX distribution metadata unavailable
MLX distribution metadata error: <ExceptionType>
MLX distribution metadata malformed: expected non-empty string
MLX distribution metadata version mismatch: <repr(value)>
```

Use no `str()`, `.strip()`, parsing, normalization, compatible range, suffix removal, or newer-version acceptance. Whitespace-only strings, `0.31.2+local`, `0.31.2.dev0`, and all nearby/newer versions reach the mismatch branch unchanged. Do not catch `PilotError` in the generic metadata-exception branch.

`mlx.__version__` must not be read, copied, logged as authoritative, or used in any decision. Its absence, `None`, correct value, incorrect value, spoofed value, or property behavior cannot grant or deny admission.

### 3.3 Existing identity and lifecycle

The existing immutable key remains exactly `mlx_version`; no report schema change occurs. `_validate_canonical_identity()` continues requiring `immutable["mlx_version"] == "0.31.2"`. Contract digests naturally bind the accepted metadata value and repaired pilot source hash.

Ordering outside this version-source substitution remains unchanged:

```text
attempt-2 collision checks
→ launch-check preflight
→ log opening through noclobber FD
→ runtime preflight
→ output directory creation
→ training API load/provider/update
→ evidence publication
```

The launch-check path must still finish `_runtime_preflight()` and `check_attempt2_launch()` before the generated wrapper opens the phase log. Negative metadata outcomes therefore cannot reach `_load_training_api()`, provider construction, output creation, checkpoint/config writes, report/marker publication, or final aggregation.

## 4. Test seams

All tests remain in tracked `tests/test_ds4_segmented_pilot.py` and exercise the loaded production module. No verdict may rely on source-text matching or a helper that reimplements the expected decision.

### 4.1 Valid-preflight fixture

Create one temp-only fixture that makes every unrelated `_runtime_preflight()` prerequisite valid while preserving the production version branch:

- set `REPO_ROOT`, `PILOT_WORKSPACE`, `PILOT_MODEL`, `PILOT_DATA`, `PILOT_CONFIG`, and `PILOT_INTERPRETER` to temporary values;
- create only synthetic temporary model/data/config/provenance/vendor-module files needed for successful prerequisite checks;
- stub `_git_output`, `_resource_gate()`, `_validate_lora_config()`, manifest/provenance reads, and frozen provider/smoke hashes with valid deterministic values;
- install synthetic importable `mlx` and `mlx_lm` modules in `sys.modules`;
- place synthetic `mlx_lm.__file__` below the temporary pinned vendor root;
- monkeypatch only `pilot.importlib.metadata.version` for the version mutation;
- fail the fixture if `_load_training_api()`, provider execution, optimizer update, or evidence publication occurs.

The fixture must call production `_runtime_preflight()` or production `main(... --launch-check-only ...)`; it must not replace `_runtime_preflight()`.

### 4.2 Module-attribute seam

Construct the fake `mlx` module independently from metadata behavior:

- absent attribute: do not set `__version__`;
- `None`: set `__version__ = None`;
- exact spoof: set `__version__ = "0.31.2"`;
- wrong spoof: set `__version__ = "999.0-spoof"`.

Only one source changes per matrix row. A separate case with exact metadata and an `ImportError` for `mlx` proves mandatory module availability remains fail-closed.

### 4.3 Direct launch-check seam

Call production `main()` with exact attempt-2 `--launch-check-only` arguments and a temporary central namespace. For every negative metadata case assert:

- return code `1`;
- stderr contains the expected metadata category;
- `_load_training_api()` call count is zero;
- provider-call and optimizer-update counters are zero;
- every A2/final destination was absent before the call and remains absent afterward.

The direct test must use a fresh fixture per case. It may not create and then delete a destination.

### 4.4 Generated-wrapper seam

Use unchanged `scripts.finetune_ds4.py::_pilot_attempt2_command()` to render the actual catalog wrapper against a temporary pilot runner. The runner imports the production `scripts/ds4_segmented_pilot.py`, installs the same valid prerequisite fixture, records invocation kind, then delegates to production `main()`.

For each negative metadata case:

- run the generated `bash -lc` wrapper in a subprocess;
- assert exact nonzero status propagation;
- trace contains only `launch-check`, never `train`;
- phase log remains absent, proving the wrapper did not execute its noclobber log-open step;
- phase output/start/step-1/step-2/final checkpoint/config remain absent;
- phase report/OK/fail and final report/OK/fail remain absent;
- training API/provider/update counters remain zero.

This uses the generated wrapper as shipped without editing `scripts/finetune_ds4.py`. No new production seam or wrapper repin is justified.

## 5. Mutation matrix

### 5.1 Positive production-preflight rows

| Distribution metadata | `mlx.__version__` | Expected |
|---|---|---|
| `"0.31.2"` | absent | pass; identity `mlx_version == "0.31.2"` |
| `"0.31.2"` | `None` | pass; identity `mlx_version == "0.31.2"` |
| `"0.31.2"` | `"999.0-spoof"` | pass; spoof ignored |

### 5.2 Negative production-preflight and launch rows

| Distribution metadata behavior | `mlx.__version__` | Expected diagnostic |
|---|---|---|
| `"0.31.1"` | `"0.31.2"` | metadata version mismatch |
| `"0.31.2+local"` | `"0.31.2"` | metadata version mismatch |
| `"0.31.3"` | absent | metadata version mismatch |
| `PackageNotFoundError("mlx")` | any | metadata unavailable |
| `RuntimeError("metadata broken")` | any | metadata error: `RuntimeError` |
| `""` | any | metadata malformed |
| `None` | any | metadata malformed |
| integer `312` | any | metadata malformed |

Run every negative row through direct production launch-check. Run at least mismatch, suffix, missing, generic exception, empty, `None`, and integer rows through the generated-wrapper boundary. Parameterization is preferred, but each row must independently reach the metadata branch with all earlier guards valid.

### 5.3 Availability row

Exact metadata `"0.31.2"` plus failed `import mlx` must fail as runtime source identity unavailable. It must not become a metadata-only success path.

## 6. No-write destination set

For Phase A2, assert all exact temporary counterparts remain absent:

```text
phase-a-output
phase-a-start-checkpoint
phase-a-step1-checkpoint
phase-a-step2-checkpoint
phase-a-final-checkpoint
phase-a-config
phase-a-log
phase-a-report
phase-a-ok
phase-a-fail
final-report
final-ok
final-fail
```

No test may pre-create then remove these paths. Existing collision tests continue proving occupied paths reject before mutation.

## 7. Namespace and authorization adjudication

Retain `ATTEMPT2_NAMESPACE = "ds4-segmented-pilot-attempt-2"` and every current A2/B2/final path.

Rationale:

- the failed invocation stopped before opening the log;
- all A2/final destinations remained absent;
- attempt-1 evidence is separate and immutable;
- unchanged collision-before-write gates prevent overwrite;
- an attempt-3 repin would broaden catalog, path, report, digest, test, and documentation changes without adding protection.

Audit lineage for any future A2 invocation must record:

1. failed revision `c910d1ba33912236ee87f3f9bdfb5b31edece6e7` and exact pre-log failure;
2. zero calls/updates and absent destinations;
3. consumed prior authorization;
4. exact repaired revision;
5. Reviewer PASS and Test Manager GREEN on that same revision;
6. immediate attempt-1 manifest verification and A2/final absence check;
7. fresh explicit operator authorization bound to one exact command;
8. resulting exit and evidence.

Fresh authorization permits one visible invocation only. Any exit consumes it. It is not a continuation or retry under the old authorization and creates no retry entitlement.

B2 remains blocked until exact A2 report/OK marker, checkpoint/config SHA-256 and canonical tensor digests, identity, cardinality, lock release, absent failure evidence, and fixed namespace are independently verified. B2 then requires separate explicit authorization.

## 8. Exact implementation scope

Permitted mutable files:

```text
scripts/ds4_segmented_pilot.py
tests/test_ds4_segmented_pilot.py
docs/technical-spec.md
custom-handoffs/14-5b-mlx-version-source/architecture.md
custom-handoffs/14-5b-mlx-version-source/coder-notes.md
custom-handoffs/14-5b-mlx-version-source/review.md
custom-handoffs/14-5b-mlx-version-source/test-report.md
.cmux-status/*.done
```

`docs/backlog.md` and `requirements.md` are BA-owned canonical inputs and must not be rewritten by Coder. `docs/architecture.md` needs no change because subsystem boundaries do not change; the durable operational contract is recorded in `docs/technical-spec.md`.

No ADR: this corrects an implementation source-of-truth bug inside the existing fail-closed pilot boundary; it does not introduce a new subsystem or public API.

## 9. Protected bytes and baselines

These exact current baselines must remain unchanged:

| Protected object | Baseline |
|---|---|
| `scripts/finetune_ds4.py` | SHA-256 `de303d1166f2ecd2428dc15f63c8be8ed22e9f30274008afd35be458ef4b188b` |
| `scripts/ds4_segmented_smoke.py` | SHA-256 `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8` |
| `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` | SHA-256 `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` |
| failure lineage record | SHA-256 `cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41` |
| `vendor/mlx-lm` gitlink and HEAD | `80fab4e419a57f9465bb9e2f4e90010d645e124c` |
| all 22 root `ds4*.{c,h,m,cu}` runtime files | sorted `<sha256><two spaces><path>\n` manifest SHA-256 `07e2e75adeccb4f9b86f12924f41eb1d17d0ffcbda4b1ad0d76e8866b5bd085b` |
| all 19 `metal/**` files | same manifest format SHA-256 `c67e0ed61758ccd4bd1faca421b697a749fb3a10328f9b2fb4d27b994f0ba13f` |

The five historical attempt-1 evidence entries remain exactly the size/SHA-256 manifest already pinned in `ATTEMPT1_EVIDENCE_SHA256`. Synthetic implementation/review must not access external model, dataset, config, checkpoint, output, marker, or attempt-1 files to recompute them. Existing synthetic manifest tests and source constants are the permitted proof boundary.

No provider/vendor package, `python-envs` dependency definition, model, dataset, config, Story 14.3 smoke, Path A, CUDA, ROCm, distributed, Metal, CPU, SSD-streaming, GGUF, CLI/server, or inference source change is permitted.

## 10. TDD and verification sequence

1. Verify `tests/test_ds4_segmented_pilot.py` is tracked.
2. Add production-boundary positive and negative metadata tests.
3. Add direct launch-check no-call/no-write matrix.
4. Add generated-wrapper no-call/no-write matrix using unchanged `scripts/finetune_ds4.py`.
5. Run targeted tests and capture RED against `mlx.__version__` behavior.
6. Implement only the module-scope import and `_runtime_preflight()` substitution/diagnostics.
7. Run targeted GREEN.
8. Run canonical six-file regression.
9. Run `py_compile`, tracking, whitespace, scope, and protected-hash checks.
10. Reviewer and Test Manager independently verify the same exact revision.

Targeted command:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py \
  -k 'mlx_distribution_metadata or mlx_version_source or metadata_launch_check'
```

Canonical regression:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
python-envs/mlx/.venv/bin/python -m pytest -q \
  tests/test_ds4_segmented_pilot.py \
  tests/test_ds4_segmented_smoke.py \
  tests/test_finetune_ds4.py \
  tests/test_mlx_lm_source.py \
  tests/test_ds4_segmented_loss_and_grad.py \
  tests/test_ds4_gguf_base_smoke.py
```

Structural checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m py_compile \
  scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py
git diff --check
git ls-files --error-unmatch -- tests/test_ds4_segmented_pilot.py
```

Reviewer and Test Manager must verify every verdict-contributing test file with `git ls-files` before citing pass counts. Any untracked contributing test blocks the verdict.

## 11. STOP/ESCALATE

Stop and return to BA + Architect for any proposal or discovered need to:

- read, trust, record, or fall back to `mlx.__version__`;
- accept anything other than exact metadata string `0.31.2`;
- normalize, coerce, parse, suppress, or recover from metadata errors;
- make `mlx` import optional;
- edit `scripts/finetune_ds4.py` or repin its wrapper;
- change fixed namespace, paths, identity schema, digest, budgets, cardinality, lock, logging, retry, fallback, or training semantics;
- clean, delete, rename, overwrite, or allocate an alternate namespace;
- mutate attempt-1 or failed-A2 historical evidence;
- access real assets or run A2/B2;
- change provider/vendor, Story 14.3, Path A, CUDA, ROCm, distributed, Metal, CPU, SSD, GGUF, CLI/server, or inference paths;
- proceed with an untracked verdict-contributing test;
- treat requirements, architecture, or prior authorization as permission for a real invocation.

## 12. Coder handoff

Implement TDD RED → minimal GREEN exactly within Section 8. Keep `scripts/finetune_ds4.py` byte-identical. Update no additional canonical document unless implementation discovers a durable contract mismatch; STOP before broadening scope. Produce coder notes only. Do not commit, push, access real assets, mutate namespaces, or run A2/B2.
