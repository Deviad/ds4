# Story 13.3b-5f — opaque packed-FP4 Metal/MLX training primitive

**Decision: GO to one TDD RED → GREEN implementation slice within this design.** Installed MLX 0.31.2 can launch the required training-only custom Metal kernels and can expose their exact first-order VJP through an outer `mx.grad` / `mx.value_and_grad` while treating captured packed weights and scales as constants. The selected design has no full dequantized expert matrix, no `E*H*I` operation term, no `O(H²I)` recomputation, and a conservative real-config routed-operation envelope of `1504 MiB` including headroom.

No full model, model shard, dataset, or real smoke was loaded or run. No production or FROZEN code was changed.

## 1. Inputs and evidence read

Read in full or section-indexed in full where source was larger:

- `agent-output/cmux-13-3b/task-architect-13-3b-5f-metal-primitive.md`
- `agent-output/cmux-13-3b/requirements-13-3b-5f-metal-primitive.md`
- `agent-output/cmux-13-3b/architecture-13-3b-5e-peak-adjudication.md`
- current routed-expert and sparse custom-VJP implementation in `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
- exact FROZEN `_dequantize_fp4_block_scale_mlx` and `_moe_mlx` contracts in `deepseek_v4.py`
- ADR 0024, ADR 0025, and ADR 0026
- `docs/architecture.md`, `python-envs/mlx/pyproject.toml`, root `Makefile`, and the explicit production Metal source loader list in `ds4_metal.m`
- installed MLX 0.31.2 API docstrings for `mx.fast.metal_kernel`, `mx.custom_function`, and `mx.vjp`
- MLX v0.31.2 `custom_metal_kernels.rst` plus the v0.31.2 quantized-Metal source organization
- project Metal threadgroup/SIMD conventions, especially `metal/dense.metal`, `metal/moe.metal`, and production source loading boundaries

Story 13.3b-5e remains the RED/root-cause authority. Its fixed-assignment locked peaks are `584,225,254`, `779,041,700`, and `1,179,289,384` bytes for E=2/4/8. Active bytes grew after every expert with cache zero; final flat active memory did not validate peak. The stopped Python path creates four `forward_one` graphs per non-empty expert and retains their FP32 dequant arrays and nested VJP tape until the outer gradient evaluates.

## 2. Light feasibility probes

All probes used `python-envs/mlx/.venv/bin/python`, MLX `0.31.2`, Apple Metal, tiny synthetic arrays, and `mx.disable_compile()`. They created no project files and loaded no model/shard.

### 2.1 API and transform boundary

Installed signatures and contracts:

```text
mx.fast.metal_kernel(
    name, input_names, output_names, source,
    header="", ensure_row_contiguous=True, atomic_outputs=False
)

kernel(
    inputs, template, grid, threadgroup,
    output_shapes, output_dtypes,
    init_value=..., verbose=...
)

@mx.custom_function
@function.vjp
```

MLX documents captured arrays as constants with no gradients. A custom-function probe captured `c`, launched one Metal forward kernel and one Metal VJP kernel, then ran outer `mx.value_and_grad(..., argnums=(0,1))`:

```text
value 40.0
dx [2.0, 3.0, 4.0, 5.0]
dc [0.0, 0.0, 0.0, 0.0]
dtypes float32 float32 float32
finite True
```

This proves the required first-order transform shape: activation cotangent returned; closure/frozen input cotangent absent. A DOT export of an unevaluated custom Metal result contains one `CustomKernel` node, not the kernel's internal arithmetic. The outer graph can retain opaque kernel nodes without seeing tile-local dequant operations.

### 2.2 Dtypes, outputs, threadgroups, bounds, and atomics

A padded-grid kernel used static `threadgroup float tile[32]`, `threadgroup_barrier`, BF16 input, a shape guard, and FP32 output:

```text
tg_bounds_bf16 [1.5, 2.0, 3.0, 4.0, 5.0] float32
```

A two-output kernel launched successfully. A separate `atomic_outputs=True`, `init_value=0` FP32 probe accumulated 64 threads:

```text
atomic_float 64.0 float32
```

The selected design does not require atomics, but MLX 0.31.2 can provide them if later diagnostics need them.

### 2.3 Exact packed tile

A real custom Metal `8x8x32`-shape probe used:

- 256 threads / eight SIMDgroups;
- a static `float weight_tile[8][32]`;
- LSB-first nibble decode;
- the exact ADR 0024 E2M1 LUT;
- BF16 direct scales;
- `simd_sum` FP32 reduction.

Against an independent NumPy literal-LUT reference:

```text
shape_dtype (2, 3) float32
max_abs 0.0
```

This validates the core dequant/reuse tile and the required Metal built-ins under the installed release.

### 2.4 Current dtype and clamp derivative

Current `x @ fp32_dequant_weight.T` returns FP32 for BF16, FP16, and FP32 `x`. The primitive therefore fixes `y`, `dx`, and `a` to FP32.

MLX 0.31.2 clip VJP evidence at `limit=2`:

```text
upper clip: derivative at +2 == 0
both-sided clip: derivative at -2 == 0 and at +2 == 0
strict interior: derivative == 1
```

The Metal derivative masks must be `u1 < limit` and `-limit < u3 < limit`; equality receives zero. `metal::precise::exp` sigmoid differed from `mx.sigmoid` by at most one FP32 ULP (`1.1920928955078125e-07`) over 10,001 points in `[-20,20]`.

## 3. Exact expert contract

Let:

```text
R = number of unique selected token rows for one expert
H = hidden_size
I = moe_intermediate_size
Hp = ceil(H/32)*32
Ip = ceil(I/32)*32
L = swiglu_limit
```

Current packed arrays are:

```text
w1_weight uint8     [E, I, Hp/2]
w1_scale  bfloat16  [E, I, Hp/32]
w3_weight uint8     [E, I, Hp/2]
w3_scale  bfloat16  [E, I, Hp/32]
w2_weight uint8     [E, H, Ip/2]
w2_scale  bfloat16  [E, H, Ip/32]
```

For projection output row `n` and logical input coordinate `k`:

```text
byte = packed[n, k//2]
code = (byte & 0x0f) if k even else ((byte >> 4) & 0x0f)
weight[n,k] = E2M1_LUT[code] * float32(scale[n,k//32])
```

The exact LUT is:

```text
(+0.0,+0.5,+1.0,+1.5,+2.0,+3.0,+4.0,+6.0,
 -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0)
```

No E8M0 conversion, exponent adjustment, native `gather_qmm(mode="mxfp4")`, or alternate nibble order is allowed.

For logical gathered rows `x_e[r]=x_flat[rows_e[r]]`:

```text
u1     = x_e @ W1.T
gate   = min(u1, L)
u3     = x_e @ W3.T
up     = clamp(u3, -L, L)
sig    = sigmoid(gate)
silu   = gate * sig
hidden = silu * up
y_e    = hidden @ W2.T
```

Padding is only on projection input axes. For `h>=H` or `i>=I`, activation is exactly zero; output stays `[R,I]` for the pair projection and `[R,H]` for down projection. Accumulation and output are FP32.

## 4. Exact first-order derivative

The routed block has duplicate-collapsed selected-score denominator:

```text
D_t  = sum over unique selected experts e of score[t,e]
D'_t = D_t + 1e-20
f_t,e = rsf * score[t,e] / D'_t
routed_t = sum_e f_t,e * y_t,e
```

Given routed-output cotangent `g_t`, the primitive backward for one expert receives unscaled `g_e`, normalized factor `f_e`, the same logical `x_e`, and frozen packed weights/scales.

Use:

```text
gu      = g_e @ W2                    # [R,I]
a_e     = rowwise_sum(hidden * gu)    # == rowwise_dot(g_e,y_e)
dhidden = f_e[:,None] * gu

dup = dhidden * silu * 1(-L < u3 < L)
dgate = (
    dhidden * up
    * (sig + gate*sig*(1-sig))
    * 1(u1 < L)
)

dx_e = dgate @ W1 + dup @ W3
```

The backward emits only:

```text
dx_e float32 [R,H]
a_e  float32 [R]
```

It emits no packed-weight or scale cotangent. Higher-order gradients are unsupported and must fail closed or remain undocumented; tests make only a first-order claim.

The Python custom VJP uses its forward output `routed` and computes once:

```text
common_t = dot(g_t, routed_t)
dscore_t,e = rsf*a_t,e/D'_t - common_t/D'_t
```

Proof:

```text
common_t = rsf * sum_j(score_t,j*a_t,j) / D'_t

dscore_t,e
= rsf * (a_t,e*D'_t - sum_j(score_t,j*a_t,j)) / (D'_t)^2
= rsf*a_t,e/D'_t - common_t/D'_t
```

This is the exact prior Q derivative without a Q forward pass. Unselected scores receive zero. Duplicate routes contribute one denominator term and one score cotangent because assignment planning collapses duplicates before either operation.

## 5. Selected kernel decomposition

No alternatives remain open for Coder. Implement these six kernels.

### 5.1 Common tile

Every matrix kernel uses:

```text
BM=8 token rows
BN=8 output coordinates
BK=32 reduction coordinates
threadgroup=(256,1,1)
eight 32-lane SIMDgroups
one SIMDgroup per token row
FP32 lane accumulators
```

Grid:

```text
grid=(256, ceil(R/8), ceil(N/8))
threadgroup_position_in_grid.y = row tile
threadgroup_position_in_grid.z = output tile
```

For each K block:

1. SIMDgroup 0 loads/dequantizes the `8x32` packed-weight tile into static threadgroup FP32 storage.
2. Barrier.
3. Eight row SIMDgroups load one activation per lane and update eight lane accumulators.
4. Barrier before the weight tile is overwritten.
5. After all K blocks, each accumulator uses `simd_sum`; lane 0 writes guarded outputs.

Pair kernels stage both `w1` and `w3` tiles (`2*8*32*4 = 2048` bytes). Single-weight kernels stage 1024 bytes. `BK=32` means every staged tile uses exactly one scale block. There is no full dequant output or device scratch proportional to `H*I`.

All kernels use `ensure_row_contiguous=False` and generated `*_shape` / `*_strides` metadata. This prevents implicit MLX copies of expert slices. Outputs are row contiguous by `metal_kernel` contract.

### 5.2 Kernel signatures

#### K1 — `ds4_fp4_pair_swiglu_forward`

```text
inputs:
  x_flat float32 [T,H]
  rows int32 [R]
  w1,w3 uint8 [I,Hp/2]
  s1,s3 bfloat16 [I,Hp/32]
  limit float32 [1]
output:
  hidden float32 [R,I]
grid: (256,ceil(R/8),ceil(I/8))
```

Gather `x_flat[rows[r],k]` inside the kernel. Use zero for `k>=H`. Accumulate `u1/u3`, apply exact clips and precise sigmoid once per output.

#### K2 — `ds4_fp4_down_forward`

```text
inputs: hidden [R,I], w2 [H,Ip/2], s2 [H,Ip/32]
output: y [R,H]
grid: (256,ceil(R/8),ceil(H/8))
```

Use zero for `i>=I` when reducing over `Ip`.

#### K3 — `ds4_fp4_down_input_vjp`

```text
inputs: g_flat [T,H], rows [R], w2 [H,Ip/2], s2 [H,Ip/32]
output: gu [R,I]
grid: (256,ceil(R/8),ceil(I/8))
```

Compute the logical transpose product `gu[r,i]=sum_h g_flat[rows[r],h]*W2[h,i]` without transposing or dequantizing `w2` globally. Scale lookup is `s2[h,i//32]`.

#### K4 — `ds4_fp4_pair_swiglu_vjp_terms`

```text
inputs:
  x_flat [T,H], rows [R], gu [R,I], f [R]
  w1,w3 [I,Hp/2], s1,s3 [I,Hp/32], limit [1]
outputs:
  dgate [R,I]
  dup [R,I]
  a_partial [R,ceil(I/8)]
grid: (256,ceil(R/8),ceil(I/8))
```

Recompute `u1/u3` once per `(r,i)`. Lane 0 writes strict-mask derivatives. Each output-tile group sums its eight `hidden[r,i]*gu[r,i]` terms into one uniquely owned `a_partial[r,tile_i]`; no atomic is needed.

#### K5 — `ds4_fp4_pair_input_vjp`

```text
inputs:
  dgate,dup [R,I]
  w1,w3 [I,Hp/2], s1,s3 [I,Hp/32]
output:
  dx [R,H]
grid: (256,ceil(R/8),ceil(H/8))
```

Compute logical transpose products over I:

```text
dx[r,h] = sum_i(dgate[r,i]*W1[i,h] + dup[r,i]*W3[i,h])
```

No projection is recomputed inside the H loop.

#### K6 — `ds4_fp4_reduce_a`

```text
input: a_partial [R,ceil(I/8)]
output: a [R]
grid: (256,R,1)
threadgroup: (256,1,1)
```

Each lane accumulates columns stride 256, then two-level SIMDgroup/threadgroup FP32 reduction. One group owns each row, so no atomic is needed.

### 5.3 Bounds and failure checks

Wrapper rejects before launch unless:

- `x_flat` and `g_flat` rank 2 and FP32 after explicit promotion;
- `rows` rank 1 int32, in `[0,T)`, unique within this expert, and token-order preserving;
- packed weights are uint8 rank 2 and scales are BF16 rank 2;
- `2*w1.shape[1]==2*w3.shape[1]==Hp`, `w1.shape[0]==w3.shape[0]==I`;
- `2*w2.shape[1]==Ip`, `w2.shape[0]==H`;
- every scale shape matches output rows by logical-input blocks;
- `Hp` and `Ip` are multiples of 32 with padding `<32`;
- `limit` is finite and positive;
- `R==0` skips all launches;
- grid dimensions fit unsigned Metal indexing; the fixed 256-thread group is launch-proven on the target M3 Ultra.

Any mismatch raises a deterministic `ValueError`; unsupported platform/version raises `RuntimeError`. No fallback executes `forward_one` or nested `mx.vjp`.

## 6. Python wrapper and outer custom-function boundary

### 6.1 Source packaging

Canonical Metal source:

```text
python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal
```

It contains one marked common-header section and six marked kernel-body sections. `routed_fp4_metal.py` loads it with `importlib.resources.files("ds4_ft_mlx")`, parses exact markers, and lazily constructs six singleton `mx.fast.metal_kernel` callables. Add package data in `python-envs/mlx/pyproject.toml`:

```toml
[tool.setuptools.package-data]
ds4_ft_mlx = ["metal/*.metal"]
```

Do not place the file in root `metal/`: root `Makefile` uses `$(wildcard metal/*.metal)` as a production dependency, while `ds4_metal.m` has an explicit production source list. Package-local placement makes build/runtime isolation structural.

### 6.2 Public narrow API

```python
packed_fp4_forward_one(
    x_flat, rows,
    w1, s1, w3, s3, w2, s2,
    *, hidden_size, intermediate_size, limit,
) -> y_e

packed_fp4_input_vjp_one(
    x_flat, g_flat, rows, factor,
    w1, s1, w3, s3, w2, s2,
    *, hidden_size, intermediate_size, limit,
) -> (dx_e, a_e)

routed_fp4(
    x_flat, scores_flat, rows_by_expert, experts,
    *, routed_scaling_factor,
) -> routed_flat
```

Only `routed_fp4` is called by `SparseMoeBlockNN`. The two one-expert functions are testable implementation units.

### 6.3 Assignment plan

From `mx.stop_gradient(indices_flat)`, host code builds:

```text
rows_by_expert[e]       # unique rows, original token order
assignment_rows [A]     # expert-id segment order
assignment_eids [A]     # same order
segment_offsets [E+1]
A = sum_e R_e <= T*K
```

Expert segments are ordered by ascending expert id, matching the current outer expert loop. Empty segments are skipped. Activations, scores, factors, outputs, and cotangents stay on MLX/Metal; only detached int32 route metadata crosses the host boundary.

### 6.4 Complete routed custom function

Selected pseudocode:

```python
x32 = x_flat.astype(mx.float32)

@mx.custom_function
def _routed(x32, scores):
    selected = scores[assignment_rows, assignment_eids]
    denom = mx.zeros((T,), scores.dtype).at[assignment_rows].add(selected)
    dprime_a = denom[assignment_rows] + 1e-20
    factor = rsf * selected / dprime_a

    y_parts = []
    for eid, rows, segment in nonempty_segments:
        y_parts.append(packed_fp4_forward_one(
            x32, rows, expert[eid].packed_and_scales, ...
        ))
    y_assignment = mx.concatenate(y_parts, axis=0)
    contribution = y_assignment * factor[:, None]
    return mx.zeros((T,H), mx.float32).at[assignment_rows].add(contribution)

@_routed.vjp
def _routed_vjp(primals, g, routed):
    x32, scores = primals
    selected = scores[assignment_rows, assignment_eids]
    denom = mx.zeros((T,), scores.dtype).at[assignment_rows].add(selected)
    dprime_a = denom[assignment_rows] + 1e-20
    factor = rsf * selected / dprime_a

    dx_parts = []
    a_parts = []
    for eid, rows, segment in nonempty_segments:
        dx_e, a_e = packed_fp4_input_vjp_one(
            x32, g, rows, factor[segment],
            expert[eid].packed_and_scales, ...
        )
        dx_parts.append(dx_e)
        a_parts.append(a_e)

    dx_assignment = mx.concatenate(dx_parts, axis=0)
    a_assignment = mx.concatenate(a_parts, axis=0)
    dx = mx.zeros((T,H), mx.float32).at[assignment_rows].add(dx_assignment)

    common = mx.sum(g * routed, axis=-1)
    dscore_assignment = (
        rsf * a_assignment / dprime_a
        - common[assignment_rows] / dprime_a
    )
    dscores = mx.zeros_like(scores).at[
        assignment_rows, assignment_eids
    ].add(dscore_assignment)
    return dx, dscores
```

The custom function has positional primals only `(x32, scores)`. Packed arrays, scales, rows, offsets, and `rsf` are closure constants. MLX therefore requests no VJP for them. The explicit cast before the custom function preserves the current FP32 expert math; if the upstream activation is BF16/FP16, the cast's ordinary MLX VJP returns the cotangent in the upstream dtype.

One denominator scatter, one routed scatter, one dx scatter, and one score scatter replace repeated per-expert full accumulators. The graph may contain one opaque kernel chain per non-empty expert, but its arrays sum over assignments; no chain emits `[H,I]` or `[I,H]` dequant data.

A private test-only telemetry callback may evaluate and sample after expert outputs for acceptance evidence. It must not alter math, select a semantic variant, or be enabled by the trainer. The load-bearing GREEN remains the complete operation peak from a fresh process, not flat final active bytes.

## 7. Memory bound

Let `A=sum_e R_e<=T*K` after duplicate collapse. The kernels gather `x` and `g` by row internally, so no global `x_assignment[A,H]` or `g_assignment[A,H]` array exists.

### 7.1 Per-assignment and per-expert sizes at H=4096, I=2048

```text
one FP32 H row = 4096*4 = 16 KiB
one FP32 I row = 2048*4 =  8 KiB
one a_partial row = ceil(2048/8)*4 = 1 KiB

forward retained assignment rows:
  hidden + y + contribution = 8 + 16 + 16 = 40 KiB per assignment
  one expert segment = 40 KiB * R_e

backward retained assignment rows:
  gu + dgate + dup + a_partial + dx
  = 8 + 8 + 8 + 1 + 16 = 41 KiB per assignment
  one expert segment = 41 KiB * R_e
```

Packed payload per real expert is `12 MiB` weights plus `1.5 MiB` BF16 scales; all 256 experts are `3.375 GiB` frozen baseline residency. Payload construction/materialization stays outside measured operation delta. No FP32 dequant counterpart is allocated.

### 7.2 Real configured assignment bound

For `E=256,K=6,T=4096,H=4096,I=2048`:

```text
A <= T*K = 24576
A*H*4 = 384 MiB
A*I*4 = 192 MiB
A*ceil(I/8)*4 = 24 MiB
T*H*4 = 64 MiB
T*E*4 = 4 MiB
```

Forward conservative simultaneous arrays:

| Buffer | MiB |
|---|---:|
| all hidden segments | 192 |
| concatenated y assignments | 384 |
| weighted contribution assignments | 384 |
| input x | 64 |
| routed output | 64 |
| scores | 4 |
| **Total** | **1092** |

Backward conservative simultaneous arrays:

| Buffer | MiB |
|---|---:|
| gu assignments | 192 |
| dgate assignments | 192 |
| dup assignments | 192 |
| a partials | 24 |
| dx assignments | 384 |
| x, routed, g, dx | 256 |
| scores, dscores | 8 |
| **Total** | **1248** |

`factor`, `D'`, assignment metadata, `a`, and selected score cotangents total below 1 MiB. Static threadgroup scratch is a few KiB per resident group, not per logical grid group.

Reserve 256 MiB for graph metadata, any measured layout materialization not represented above, and allocator variance:

```text
1248 MiB + 256 MiB = 1504 MiB operation envelope
2 GiB - 1504 MiB = 544 MiB headroom
```

This is below the `<=2 GiB` formula gate and leaves 26.6% binary headroom. A real-dimension no-shard measured operation must still be `<2 GiB`; formula does not waive measurement.

### 7.3 Fixed-assignment E independence

Small GREEN probe: `T=1024,K=2,A=2048,H=1024,I=512`, E=2/4/8, all experts non-empty.

```text
A*H*4 = 8 MiB
A*I*4 = 4 MiB
forward assignment arrays = 4 + 8 + 8 = 20 MiB
backward assignment arrays = 4 + 4 + 4 + 0.5 + 8 = 20.5 MiB
```

Only scores/dscores (`T*E*4`) and small opaque-node/metadata counts vary with E; there is no dequantized `E*H*I` term. The required measured spread remains `max(peak)-min(peak)<=64 MiB`.

## 8. Complexity and smoke credibility

Per expert:

```text
forward: pair projection 2*R*H*I + down R*I*H = 3*R*H*I MAC terms
backward: down transpose R*H*I
          pair recompute 2*R*H*I
          pair input VJP 2*R*I*H
        = 5*R*H*I MAC terms
```

Total exact forward+backward is `8*R*H*I`, linear in assignment count and standard projection dimensions. K4 recomputes `u1/u3` once; K5 consumes saved derivative terms. It does not recompute `u1/u3` for every H coordinate, so the rejected `O(R*H²*I)` shape is absent.

`BM=8` reuses each dequantized weight tile across eight token rows. Average real balanced routing has `R_e≈A/E=96`, so the selected tile has useful row reuse. Six launches per non-empty expert are accepted for correctness and boundedness. Before the later smoke, a no-shard real-dimension benchmark must report per-kernel and end-to-end p50/p95 for representative `R={1,8,32,96}` and extrapolate 43 layers × 20 iterations. A result showing launch-dominated or plainly impractical runtime returns to Architect; it does not authorize semantic approximation.

## 9. Platform and backend isolation

`routed_fp4_metal.py` must fail closed unless:

```text
sys.platform == "darwin"
platform.machine() == "arm64"
importlib.metadata.version("mlx") == "0.31.2"
mx.metal.is_available() is True
default device is GPU
```

Kernel construction is lazy. Importing other fine-tuning helpers must not compile Metal. Unsupported hosts receive an actionable error naming Story 13.3b-5f/ADR 0028; they do not call the old Python composition.

Isolation facts:

- package-local `.metal` source is absent from root `Makefile`'s `metal/*.metal` wildcard;
- `ds4_metal.m` keeps its explicit source list unchanged;
- `deepseek_v4.py` stays at approved SHA-256 `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`;
- CPU reference, default Metal inference, SSD streaming, CUDA, ROCm, distributed inference, model loading, datasets, and site-packages remain untouched;
- shared expert and all routing selection logic remain in `deepseek_v4_nn.py` unchanged except the routed primitive dispatch.

Recorded pre-slice production hashes for later direct comparison include:

```text
ds4.c           a9cb4d37d1b5ce34580aec077857bd47f30217ba4e18a36697d86de8e877f957
ds4_metal.m     6624500152a779c1201e55c2c56d74221ee4f4b8982ff47d181e90c8b265a1c6
ds4_cuda.cu     5a325e571fe05e929de249492d2bf564c37b759ca87d7a1230d97b735a56d727
ds4_distributed.c ad333f0bb30be211d745e9b0c3a391f0f8e8b368f763d8afe2e591de308afb8c
ds4_ssd.c       2b2ba6e2278c24196bde916f7149f9c2a9cbc87460d5ce6c421651899f4a4c78
```

Coder regenerates every hash rather than copying this snapshot. Reviewer/Test Manager compare a generated manifest of all protected production files and existing root `metal/*.metal`, not a zero-line git diff.

## 10. Exact edit map

### Coder production/training edits

1. `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`
   - common LUT/decode/shape helpers;
   - six marked kernel bodies only.
2. `python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py`
   - platform/version guards;
   - source loader/parser and lazy kernel cache;
   - shape/dtype validation;
   - one-expert forward/backward wrappers;
   - complete routed custom function and exact score identity.
3. `python-envs/mlx/pyproject.toml`
   - package the training `.metal` file.
4. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
   - preserve route selection and `_host_unique_rows_per_expert` semantics;
   - replace only the stopped nested `forward_one` custom-VJP body with `routed_fp4`;
   - retain one shared-expert call.
5. `docs/technical-spec.md`
   - record focused tests, platform gate, no-shard memory commands, double-green gate, and still-forbidden smoke.
6. `docs/backlog.md`
   - BA verifies Story 13.3b-5f acceptance state; Coder does not rewrite requirements.

### Tests

1. New tracked `tests/test_deepseek_v4_nn_routed_fp4_metal.py`.
2. Strengthen tracked `tests/test_deepseek_v4_nn_sparse_routed_backward.py` without relaxing existing expectations.
3. Optional tracked fresh-process probe helper under `tests/` only if subprocess isolation cannot remain inside the test module.

### Files forbidden

- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
- root `metal/*.metal`
- `ds4_metal.m`, `ds4.c`, C/CUDA/ROCm/distributed/SSD sources
- model shards, datasets, site-packages
- ADR 0024

## 11. TDD RED → GREEN plan

All verdict tests must be tracked before Coder declares done.

### RED 1 — stopped lifetime behavior

Fresh processes, compile disabled, `T=1024,K=2,H=1024,I=512`, E=2/4/8, identical A=2048, all experts non-empty. Run current Python custom-VJP path before replacement. Record active/cache/peak at expert boundaries and final evaluation. Assert material E-scaled peak, not exact allocator bytes.

### RED 2 — primitive absent/fail closed

Before implementation, tests require import/API, package Metal source, six kernel names, platform guard, and no-fallback behavior. They must fail for the correct missing-boundary reason.

### GREEN 1 — packed tile and forward parity

Independent reference uses a literal E2M1 table and independent nibble unpack. Fixtures must include:

- every nibble code, both byte positions;
- distinct bytes/scales for w1/w3/w2;
- at least two 32-logical-input scale blocks;
- nonuniform selected rows and token order;
- aligned real-like and padded tiny H/I;
- FP32 output shape/dtype/finiteness;
- `max_abs<=1e-6`.

No expected value may call production dequant or primitive helpers.

### GREEN 2 — one-expert VJP

Compare `dx` and `a` with ordinary-MLX/dense independent reference at `atol=rtol=1e-5`. Include:

- nonuniform `g` and `f`;
- positive/negative activations;
- raw `u1` below, exactly at, and above `+L`;
- raw `u3` below, exactly at, between, exactly at, and above `±L`;
- non-zero finite expected leaves.

Explicitly assert zero derivative at equality boundaries.

### GREEN 3 — routed score identity

Compare simplified identity with the prior Q formula on learned/hash-style plans, duplicate routes, empty experts, and nonuniform scores/g. Assert exact selected support and `1e-5` parity.

### GREEN 4 — whole sparse and semantic invariants

Retain/extend existing tests for:

- learned/hash routes;
- stable lower-index ties;
- correction bias selection only;
- duplicate collapse;
- empty skip;
- exact normalization;
- token order;
- one shared expert call;
- w1/w3/w2 order;
- focused LoRA gradient tree, all finite, at least one non-zero expected update;
- no route/packed-weight/scale gradient.

### GREEN 5 — opacity and closure

- DOT graph sees `CustomKernel` nodes, not dequant arithmetic.
- API returns only y or `(dx,a)`; no dense matrix output.
- source rejects `_dequantize_fp4_block_scale_mlx`, `mx.repeat`, native mxfp4/E8M0 substitution, and fallback `mx.vjp(forward_one)`.
- outer `value_and_grad` returns activation/score cotangents and zero/no closure cotangents.

### GREEN 6 — memory

Fresh-process small E=2/4/8 operation telemetry:

```text
spread = max(peak_delta)-min(peak_delta) <= 64 MiB
```

Record baseline, active, cache, peak after each expert boundary and final evaluation. Final active alone is not evidence.

Real-dimension no-shard probe:

- `H=4096,I=2048`, minimal representative `R` sufficient to launch every tile/boundary path;
- one reused synthetic expert payload, no 256-expert allocation requirement;
- outer transform returns finite y/dx/a;
- measured operation peak `<2 GiB`;
- packed setup outside measured baseline;
- include the `A=24576` formula and `1504 MiB` envelope.

### GREEN 7 — regressions, tracking, and scope

- focused FP4/MoE/remap/LoRA suites;
- full tracked non-live suite;
- `make` build check because architecture isolation claims no production effect;
- `git diff --check`;
- `git ls-files -- <every verdict test>` non-empty;
- direct SHA manifest for FROZEN and all production backends/root Metal sources;
- approved predecessor chain-of-custody checks from BA requirements.

Reviewer PASS and Test Manager GREEN remain mandatory before any real smoke.

## 12. Risks and controls

1. **Floating reduction order:** SIMD reductions may differ from ordinary MLX matmul. Control: FP32 accumulation, precise sigmoid, strict tolerance tests. STOP on parity failure; do not relax tolerances without Architect re-pin.
2. **Implicit copies:** default `ensure_row_contiguous=True` can hide copies. Control: fixed `False`, stride-aware indexing, operation telemetry. STOP on an unexplained E-scaled copy.
3. **Scatter accumulation order:** duplicate token rows meet in one MLX scatter. Control: expert-id segment ordering and dense/Q references. If tolerance fails, replace only the assignment combine with one deterministic token-row Metal reduction; do not restore repeated full accumulators.
4. **Kernel launch overhead:** six launches per non-empty expert. Control: real-dimension no-shard timing for representative R before smoke. No semantic fusion until parity/memory GREEN.
5. **MLX release drift:** private custom-kernel behavior can change. Control: exact `mlx==0.31.2` guard for this story; a later version requires probes and ADR amendment.
6. **Packaging drift:** missing `.metal` package data fails only after install. Control: installed/editable import test reads source through `importlib.resources` from a fresh process.
7. **Higher-order transforms:** not supported. Control: document first-order only and avoid any higher-order claim.

## 13. STOP conditions

STOP and return to Architect if any of these occurs:

- Metal kernels cannot reproduce forward `<=1e-6` or VJP/score `1e-5` tolerances;
- exact MLX clamp-boundary masks cannot be matched;
- any full FP32 w1/w3/w2 matrix appears as an MLX output or retained buffer;
- fixed-assignment peak has a material `E*H*I` slope or E spread exceeds 64 MiB;
- measured real-dimension no-shard operation reaches `>=2 GiB` or formula exceeds 2 GiB;
- hidden contiguity/layout copies invalidate the 1504 MiB envelope;
- implementation requires root production Metal, FROZEN, inference, SSD, CUDA, ROCm, distributed, shard, dataset, or site-package edits;
- host materialization expands beyond detached integer route metadata;
- frozen packed weights/scales receive cotangents;
- the only workable derivative is `O(R*H²*I)` or requires Q recomputation;
- unsupported platforms silently fall back;
- any verdict test is untracked or predecessor chain of custody is unresolved;
- Reviewer or Test Manager does not independently pass/green.

## 14. Durable documentation completed now

- Created `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md`.
- Amended ADR 0025 to revoke the disproved `mx.eval` lifetime claim and point to ADR 0028.
- Updated `docs/architecture.md` with the package-local training component and production-isolation map.
- ADR 0024 unchanged.

No real 4096 smoke is authorized by this architecture. Only the existing double-green gate can authorize exactly one later smoke.
