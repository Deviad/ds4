# Reviewer BLOCKED — Story 13.3b-4

Verdict: BLOCKED.

Blocking axis: Axis 5 / AC2 legitimacy.

Reason:

`tests/test_remap_ds4_nn_weights_load.py` does not perform the required real sub-checkpoint load+forward finite check. It builds `_tiny_mixed_config()` and `_tiny_ckpt_style_weights()` with synthetic `_zeros` / `_ones`, then strict-loads that fabricated tensor dict.

Evidence:

```text
tests/test_remap_ds4_nn_weights_load.py:25  def _tiny_mixed_config() -> dict[str, object]:
tests/test_remap_ds4_nn_weights_load.py:80  def _tiny_ckpt_style_weights(cfg: dict[str, object]) -> dict[str, mx.array]:
tests/test_remap_ds4_nn_weights_load.py:100-170 raw[...] = _zeros/_ones(...)
tests/test_remap_ds4_nn_weights_load.py:182 remapped, report = remap.remap_weight_dict(_tiny_ckpt_style_weights(cfg), config=cfg)
tests/test_remap_ds4_nn_weights_load.py:192 model.load_weights(list(remapped.items()), strict=True)
```

No `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` payload load, no `safetensors.safe_open` / `load_file`, no actual selected checkpoint tensor payloads in AC2. AC1 uses the real checkpoint only for header/index keyset.

Required fix:

Add reviewer-visible AC2 coverage that loads actual safetensors payload tensors from the shimmed checkpoint for at least one sliding + one CSA + one HCA sub-checkpoint path, remaps with `remap_weight_dict`, strict-loads nn module, and runs finite forward; or get BA/Architect to explicitly revise AC2 if synthetic tiny topology is intended to be sufficient.

Other axes:

- Scope: GREEN; exactly 6 staged files, no protected staged edits.
- FROZEN/nn-port/shim byte+AST identity: GREEN.
- Remap mechanics on real headers: GREEN; `1460` expected keys == `1460` remapped, `0` missing, `0` extra.
- Ape verbatim: GREEN; no `.T`/transpose/swapaxes/permute/moveaxis; ape shapes `(4,1024)`, `(4,256)`, `(128,512)` identity.
- Tracking/commit gating: GREEN; new tests staged `A`, HEAD still `58194a9`.
- sha-pin cascade: GREEN; no protected pin touch.
- Regression: tests pass locally (`3 passed`; full `572 passed, 13 skipped`), but BLOCKED remains due AC2 coverage gap.

Full review: `agent-output/cmux-13-3b/review-13-3b-4.md`.
