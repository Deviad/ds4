GREEN

Independent verdict: GREEN.
File-backed probe used after stdin/spawn failure; no product or test files edited.

Evidence
- `tests/test_deepseek_v4_nn_routed_fp4_metal.py`: 4 passed.
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py`: 21 passed.
- Full tracked suite: 598 passed, 18 skipped, 96 subtests passed.
- `make`: Nothing to be done for `all`.
- `git diff --check`: clean.
- `git ls-files` confirmed tracked verdict tests:
  - `tests/test_deepseek_v4_nn_routed_fp4_metal.py`
  - `tests/test_deepseek_v4_nn_sparse_routed_backward.py`

Independent file-backed probe
- Fixed-assignment E=2/4/8 peak deltas:
  - E=2: 148,402,356 bytes
  - E=4: 151,990,564 bytes
  - E=8: 159,088,988 bytes
  - spread: 10,686,632 bytes (< 64 MiB)
- Real-dimension no-shard R=1/8/32/96:
  - R=1: fwd 0.018396s, bwd 0.003250s, fwd peak 24,580, bwd peak 41,992
  - R=8: fwd 0.031234s, bwd 0.004481s, fwd peak 196,612, bwd peak 335,908
  - R=32: fwd 0.080738s, bwd 0.011929s, fwd peak 786,436, bwd peak 1,343,620
  - R=96: fwd 0.211529s, bwd 0.029176s, fwd peak 2,359,300, bwd peak 4,030,852
  - all finite: true
- Formula check:
  - route metadata: 98,304 bytes
  - one-expert dequant reference: 100,663,296 bytes
  - documented active delta: 402,653,184 bytes
  - worst-case active: 1,140,850,688 bytes
  - envelope: 1,504 MiB
  - budget: 1,342,177,280 bytes

Scope / hash checks
- `ds4.c`: `a9cb4d37d1b5ce34580aec077857bd47f30217ba4e18a36697d86de8e877f957`
- `ds4_metal.m`: `6624500152a779c1201e55c2c56d74221ee4f4b8982ff47d181e90c8b265a1c6`
- `ds4_cuda.cu`: `5a325e571fe05e929de249492d2bf564c37b759ca87d7a1230d97b735a56d727`
- `ds4_distributed.c`: `ad333f0bb30be211d745e9b0c3a391f0f8e8b368f763d8afe2e591de308afb8c`
- `ds4_ssd.c`: `2b2ba6e2278c24196bde916f7149f9c2a9cbc87460d5ce6c421651899f4a4c78`
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
- `docs/adr/0024-fp4-e2m1-on-the-fly-moe-dequant-contract.md`: `33dc15ce2d7ed64cb77509cb95ab48300988f83bec5513b6656a644bcef6c751`

No lingering probe process, model load, or shard access observed after validation.
