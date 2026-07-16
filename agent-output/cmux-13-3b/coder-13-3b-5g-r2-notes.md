# Story 13.3b-5g r2 Coder notes

Diagnostic-only remediation complete. No production, vendor, primitive, Metal, inference, SSD, CUDA, distributed, model-loader, real model, shard, dataset, external config, smoke, full training, commit, or real-asset access.

Changed diagnostic assets only:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`

Implemented r2 contract:

- Exact 64-row plan/order retained and identity-hashed.
- Probe B/D uses synthetic packed payload, applicable base quantize, freeze, one rank-8/scale-20/dropout-0 LoRA conversion on q_a/q_b/kv for last `min(D,16)` layers, exact `6L` trainable keys, trainable-key hash evidence.
- Differentiated losses bind forward once.
- Probe C measures DOT `CustomKernel` nodes from exported differentiated graph and asserts exact `6*D*N`; route-score gradient exported with activation gradient to expose all six primitive nodes per non-empty expert.
- `active_after_graph` measured after differentiated graph construction; sequential Probe C records max per-stage graph-active and peak values, sums nodes.
- Probe C metadata separates E=2 low-E control from normalized E=8/32/128/256, K=6, 384-assignment expert axis.
- Probe D fixture is load-bearing: exact schema, 43 layers, config derivation, fixture hash, canonical source hashes, malformed-copy failures.
- Probe D mask no longer mutates module registration; full path equals ordinary forward, branch replacements preserve shape/dtype and input-gradient evidence.
- Probe A now requires non-empty exact gradient-key equality, shape equality, key hashes, separate loss/gradient finiteness, and all-key parity.
- Top-level `cell_evidence` has one entry per row with fresh child PID, nonce, serial monotonic interval, trainable-key evidence, input-gradient evidence, and measured node count.
- Child process outcomes normalized for success, non-zero, signal, timeout, and malformed output.

TDD evidence:

- RED before report regeneration: `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_multilayer_peak.py` failed on missing `cell_evidence`, missing `probe_c_comparison_contract`, and missing Probe A key evidence in old report.
- Focused GREEN: `8 passed, 1 warning in 2.06s`.
- Tracked baseline GREEN: `629 passed, 18 skipped, 2 warnings, 96 subtests passed in 139.56s`.
- `make`: exit 0, `Nothing to be done for 'all'.`
- `git diff --check`: exit 0.
- `git diff --cached --check`: exit 0.
- Tracked files verified by `git ls-files`: helper, focused test, fixture, regenerated report.

Report summary:

- rows: 64
- evidence entries: 64
- all row outcomes: `exit_code=0`, `signal=null`, `error=null`
- unique child nonces: 64
- Probe C custom nodes: exact `6 * depth * nonempty_experts` for all Probe C rows
- memory limit: `8_000_000_000`
- timeout: `180`
- execution mode: `serial_fresh_subprocess_per_cell`
- max concurrency: `1`
