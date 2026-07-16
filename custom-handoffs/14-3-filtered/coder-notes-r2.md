# Coder notes r2 — Story 14.3a reviewer blocker closure

## Blocker 1 — contradictory authorization sentence fixed

`custom-handoffs/14-3/requirements.md:185-186` now reads:

> The dataset path `...mlx-4096-smoke` is the only authorized dataset for
> the real smoke (operator-authorized filtered copy; provenance in
> `agent-output/cmux-14-3/filtered-dataset-provenance.json`; input/kept/
> excluded: train 15170/15108/62, valid 819/816/3, test 824/823/1; total
> excluded 66). The original `mlx-4096` dataset is immutable and rejected
> by the pinned-path gate.

No other asset-contract sentence contradicts the repin. Original
dataset is explicitly identified as immutable/rejected.

## Blocker 2 — provenance JSON referenced consistently

`agent-output/cmux-14-3/filtered-dataset-provenance.json` is already
tracked (`git ls-files` confirms). Referenced in:

| Doc | What was added |
|-----|---------------|
| `custom-handoffs/14-3/requirements.md` R14.3-3 asset contract | provenance JSON path + exact split counts |
| `docs/architecture.md` § Fine-tuning toolchain | provenance JSON path + split counts + per-split SHA-256 + excluded-row manifest |
| `docs/backlog.md` Story 14.3a filter provenance | provenance JSON path + exact split counts + source/filtered SHA-256 per split + excluded-row manifest |

Exact counts preserved: train 15170/15108/62, valid 819/816/3, test
824/823/1, total excluded 66. No real dataset was altered.

## Blocker 3 — default-catalog test strengthened to exact byte identity

`tests/test_ds4_segmented_smoke.py::TestCatalog::test_default_catalog_entries_unchanged`
now:

1. Computes normalized SHA-256 digests (temp-dir prefix replaced with
   `${TMP}`) for `smoke-train`, `full-train`, `continue-train`.
2. Compares against approved immutable baseline digests.
3. Any change to a default entry — including flag reordering,
   whitespace, or accidental contamination — produces a different
   digest and the test fails.
4. Defense-in-depth: also asserts no `mlx-4096-smoke` substring leaked
   into any default entry.

Baseline digests (stable across CI and local runs):

| Entry | Normalized SHA-256 |
|-------|--------------------|
| smoke-train | `7b04545d1e54ad69f971fb6e4bac9240f551f88df217ddb9bd456017b5eec0fa` |
| full-train | `26416001f3d87e0cddabdfb3189a7e2702cebc9244533a70a3a8809c6dfdee0a` |
| continue-train | `534bbf0c6345e342d21fe8bc9d2e8a4cefd70f531e3cd995588eff926150a2b2` |

Segmented entry explicitly uses filtered path; all defaults unchanged.

## Verification results

Exact-fork focused suite: **236 passed, 0 red, 1 benign
PytestAssertRewriteWarning** in 10.00s.

```
236 passed, 1 warning in 10.00s
```

### Checks passing

- Protected blob hashes: provider source `205721...` ✓, provider test
  `618a0f...` ✓, source sentinel `dec2c2...` ✓
- py_compile: 5/5 PASS
- Path A: 365 rows, digest `7241924d...` OK
- Vendor inner HEAD: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` ✓
- Vendor outer gitlink: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` ✓
- Provenance JSON: tracked ✓
- All modified files staged ✓

## File scope

| Path | Action |
|------|--------|
| `custom-handoffs/14-3/requirements.md` | 1 contract sentence replaced (blocker 1) |
| `docs/architecture.md` | 1 sentence extended with provenance ref (blocker 2) |
| `docs/backlog.md` | 6 filter-provenance bullet points updated (blocker 2) |
| `tests/test_ds4_segmented_smoke.py` | `test_default_catalog_entries_unchanged` strengthened to hash-based identity oracle (blocker 3) |

## Unchanged invariants

- All pinned smoke parameters (iters, batch-size, learning-rate,
  max-seq-length, mask-prompt, grad-checkpoint, segment-size, timeout)
- Default catalog entries (byte-identical; digests verified)
- Provider source, provider test, source sentinel (blob hashes verified)
- No vendor/mlx-lm/ edits
- No Path A evidence touched
- No real asset reads
- No commit/push
