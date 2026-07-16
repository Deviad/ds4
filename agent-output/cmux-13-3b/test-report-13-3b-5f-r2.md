GREEN

Independent retest: GREEN.
No product or test edits made.

Ran
- `tests/test_deepseek_v4_nn_routed_fp4_metal.py`: 18 passed
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py`: 21 passed
- tracked-only full suite: 612 passed, 18 skipped, 96 subtests passed
- `make`: Nothing to be done for `all`
- `git diff --check` and `git diff --cached --check`: clean
- `git ls-files` verified tracked verdict files:
  - `tests/helpers/routed_fp4_memory_probe.py`
  - `tests/test_deepseek_v4_nn_routed_fp4_metal.py`
  - `tests/test_deepseek_v4_nn_sparse_routed_backward.py`

File-backed probe evidence
- fixed-assignment `E=2/4/8`:
  - E=2: peak delta `70,648,416`, boundary samples present, all finite
  - E=4: peak delta `99,357,588`, boundary samples present, all finite
  - E=8: peak delta `109,954,088`, boundary samples present, all finite
  - spread: `39,305,672` bytes (< 64 MiB)
- real-dimension no-shard:
  - `H=4096`, `I=2048`, `T=512`, `K=2`, `E=2`
  - operation peak delta: `194,566,304` bytes (< 2 GiB)
  - all finite: true
- formula:
  - `assignment_y = 384 MiB`
  - `assignment_i = 192 MiB`
  - `a_partial = 24 MiB`
  - `envelope = 1504 MiB` (< 2 GiB)

Scope / hash checks
- `docs/adr/0024-fp4-e2m1-on-the-fly-moe-dequant-contract.md`: `33dc15ce2d7ed64cb77509cb95ab48300988f83bec5513b6656a644bcef6c751`
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
- `ds4.c`: `a9cb4d37d1b5ce34580aec077857bd47f30217ba4e18a36697d86de8e877f957`
- `ds4_metal.m`: `6624500152a779c1201e55c2c56d74221ee4f4b8982ff47d181e90c8b265a1c6`
- `ds4_cuda.cu`: `5a325e571fe05e929de249492d2bf564c37b759ca87d7a1230d97b735a56d727`
- `ds4_distributed.c`: `ad333f0bb30be211d745e9b0c3a391f0f8e8b368f763d8afe2e591de308afb8c`
- `ds4_ssd.c`: `2b2ba6e2278c24196bde916f7149f9c2a9cbc87460d5ce6c421651899f4a4c78`

Path inspection
- no production small-dimension serial-overwrite branch found in `routed_fp4_metal.py`
- structural test confirms `8x8x32` tiled kernels
- real-dimension probe exercised the same tiled primitive path
- no lingering probe process or shard access observed
