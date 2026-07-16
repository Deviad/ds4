# Story 13.3b-5g — Coder diagnostic-only multi-layer peak proof

## Summary

Implemented the diagnostic-only synthetic peak slice. No real model, shard, dataset, external config, second smoke, production source, primitive source, site-packages, or commit.

Added tracked diagnostic artifacts:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`

## TDD RED → GREEN

RED command:

```bash
python-envs/mlx/.venv/bin/python -m pytest tests/test_deepseek_v4_nn_multilayer_peak.py -q
```

Initial result: `4 failed`, because the helper, fixture, and report did not exist.

GREEN commands:

```bash
python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_multilayer_peak_probe.py run --output agent-output/cmux-13-3b/multilayer-peak-report.json
python-envs/mlx/.venv/bin/python -m pytest tests/test_deepseek_v4_nn_multilayer_peak.py -q
python-envs/mlx/.venv/bin/python -m pytest $(git ls-files 'tests/test_*.py') -q
make
```

Results:

- 5g focused tests: `4 passed`.
- Tracked Python baseline: `480 passed, 15 skipped, 86 subtests passed`.
- `make`: `Nothing to be done for all`.

## Report completeness

`agent-output/cmux-13-3b/multilayer-peak-report.json` contains 64 serial fresh-subprocess rows and no hidden/omitted cells:

- Probe B depth matrix: 9 rows — checkpoint-on `D={1,2,4,8,16,43}`, checkpoint-off `D={1,4,8}`.
- Probe C routed node-pressure matrix: 40 rows — one-graph and matched sequential controls for `D={1,8,16,43}` × `E={2,8,32,128,256}`.
- Probe D component ablations: 15 rows — `full`, `no_routed`, `no_shared`, `no_attention`, `routed_only` at compression ratios `0`, `4`, and `128`.
- Missing pinned cells: none.
- Captured subprocess failures: none.

Probe A:

- layer count: `43`.
- decoder layer count: `43`.
- checkpointed forward calls: `43`.
- checkpoint wrapper: `checkpointed_fn`.
- checkpoint-on/off finite loss and gradients: true.
- gradient max absolute diff: `0.0`.
- gradient max relative diff: `0.0`.

Peak summary:

- Probe B max backward peak: `492,694,696` bytes.
- Probe C one-graph max backward peak: `6,615,992` bytes.
- Probe C sequential max backward peak: `1,544,424` bytes.
- Probe D max backward peak: `7,443,004,916` bytes, below the `8,000,000,000` diagnostic limit.

## Scope, tracking, hashes

Verified tracked with `git ls-files`:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`

Checks:

- `git diff --check`: clean.
- `git diff --cached --check`: clean.

Protected hashes recorded in `coder-13-3b-5g-scope-checks.log`:

- `ds4.c`: `a9cb4d37d1b5ce34580aec077857bd47f30217ba4e18a36697d86de8e877f957`
- `ds4_metal.m`: `6624500152a779c1201e55c2c56d74221ee4f4b8982ff47d181e90c8b265a1c6`
- `ds4_cuda.cu`: `5a325e571fe05e929de249492d2bf564c37b759ca87d7a1230d97b735a56d727`
- `ds4_distributed.c`: `ad333f0bb30be211d745e9b0c3a391f0f8e8b368f763d8afe2e591de308afb8c`
- `ds4_ssd.c`: `2b2ba6e2278c24196bde916f7149f9c2a9cbc87460d5ce6c421651899f4a4c78`
- `deepseek_v4.py`: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
- `routed_fp4_metal.py`: `3f5a676756cee50e55db946f0b129cb8b772c64757b14c1a4508e0b515380f5a`
- `ds4_routed_fp4_train.metal`: `639c10cef35c061dbc02b6f3b5af60989fc1c1f887191f1765bea0600a180961`

## Boundary

Diagnostic report classifies nothing and authorizes nothing. Architect re-entry remains required after independent Reviewer PASS and Test Manager GREEN. No commit performed.
