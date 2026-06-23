/* Story 11.51 numeric correctness for the routed I8+F8_E8M0 single-token Metal
 * path. Included by tests/ds4_test.c so it can reuse the tiny C test harness.
 *
 * VALIDATES: the new Story 11.50 fused single-token encoder
 * `ds4_gpu_encode_mul_mv_id_i8_e8m0_pair_swiglu` (ds4_metal.m L21467) drives the
 * FROZEN pair_swiglu kernel `kernel_mul_mv_id_i8_e8m0_pair_swiglu_f32`
 * (metal/moe.metal L4566) for the gate+up arm AND the §6 D2.5 single-token down
 * GEMM `ds4_gpu_encode_mul_mm_id_i8_e8m0_f32` (ds4_metal.m L24920) drives the
 * FROZEN GEMM kernel `kernel_mul_mm_id_i8_e8m0_f32` (metal/moe.metal L4609) for
 * the down arm — together producing `mid` and `out` within L2_REL ≤ 5e-3 of the
 * Story 11.47 batched-matmul reference (`ds4_gpu_routed_moe_batch_tensor` at
 * n_tokens=32 exercising the SAME FROZEN GEMM kernel on the SAME expert
 * selection and the SAME single-token x).
 *
 * REFERENCE (Q1=(i) mechanism — Architect FROZEN): a single shared model_raw
 * backs BOTH dispatches. Tiles the single-token x across all 32 reference token
 * rows; selected_ref is token-major (p = t*n_expert + e → expert e);
 * weights_ref == 1.0 for all pairs. Batched dispatcher reorders pairs by token
 * and (for n_tokens>=32) clears `use_mm_id` (ds4_metal.m L25306), exercising the
 * FROZEN `kernel_mul_mm_id_i8_e8m0_f32` GEMM on each token row. Because every
 * reference token row carries the SAME x, pair `0*n_expert + e` (token 0,
 * expert e) computes IDENTICALLY to pair `t*n_expert + e` for any t — so row 0
 * of mid and row 0 of out are numerically deterministic per-expert.
 *
 * Extraction (row 0 = token 0):
 *   mid_ref_row0 = mid_ref[0 .. n_expert * expert_mid_dim)
 *   out_ref_row0 = out_ref[0 .. out_dim)               (post sum_experts; w==1.0)
 *
 * DUT (one_tensor, n_tokens=1): the fused gate+up kernel reads pair p's input
 * from `x + p * src_pair_stride` (src_pair_stride = expert_in_dim*sizeof(float),
 * VERIFIED-RAW moe.metal L4586); so x_dut is pair_rows_dut * expert_in_dim
 * floats, each pair slot a replicate of x_tok (mirrors batched
 * repeat_hc_embedding). one_tensor mid is laid out pair-major: mid_dut itself
 * IS row 0 across all n_expert experts. out_dut row 0 = out_dut[0 .. out_dim).
 *
 * Gate / up / down row strides follow production semantics
 * (`routed_expert_row_bytes`, ds4.c L3251): I8_E8M0 unblocked int8 → row
 * stride = dim[0] bytes. For gate: gate_row_bytes = expert_in_dim; for down:
 * down_row_bytes = expert_mid_dim. BOTH reference and DUT pass identical
 * row_bytes so the comparison is path-vs-path, not stride-vs-stride.
 *
 * ZERO production source edits this slice. NO ds4_metal.m test hook.
 * NO ds4.h mutation. NO MSL edit. AC7/AC8 byte-intactness (HEAD 068d498) audited
 * out-of-band by Test Manager via `git diff --stat 068d498 -- metal/ ds4.h
 * ds4.c ds4_cli.c ds4_server.c ds4_metal.c` (must be empty).
 */

#ifndef DS4_TEST_METAL_TENSOR_I8_E8M0
#define DS4_TEST_METAL_TENSOR_I8_E8M0 64u
#endif

/* AC6 numeric gate — L2_REL ≤ 5e-3 (ADR 0020 precedent for I8_E8M0 8-bit weight
 * + F8_E8M0 block-scale reduction-order-susceptible FP32 accumulation). */
#define DS4_TEST_I8_E8M0_L2REL (5e-3)

extern int ds4_gpu_test_i8_e8m0_host_dispatch_routing(void);
/* 11.47/11.50 reachability hook: returns <0 if Metal unavailable,
 * 1 if the FROZEN I8_E8M0 mv (pair_swiglu) AND mm (down GEMM) pipelines are both
 * bound into dispatch. This is the runtime corroboration of byte-intactness
 * (D7): if a build regression unbolted either FROZEN kernel, the routing helper
 * would return 0 (or the dispatch ok codes would be 0) before any numeric
 * comparison runs. */

/* L2_REL(a, b) = ||a-b||_2 / max(||a||_2, ||b||_2, 1e-12). */
static double ds4_test_i8_e8m0_l2_rel(const float *a, const float *b, uint64_t n) {
    double da = 0.0, db = 0.0, dd = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double av = (double)a[i], bv = (double)b[i];
        da += av * av;
        db += bv * bv;
        dd += (av - bv) * (av - bv);
    }
    double norm_max = da > db ? da : db;
    if (norm_max < 1e-12) norm_max = 1e-12;
    return sqrt(dd) / sqrt(norm_max);
}

/* Fill the I8_E8M0 model region per ADR 0021 paired-tensor contract
 * (gate_w | gate_s | up_w | up_s | down_w | down_s). Deterministic, NON-ZERO,
 * non-saturating. Scales held at 1.0 (E8M0 byte 0x7F) so fusion-vs-separate
 * reduction-order drift is not conflated with scale interpolation. */
static void ds4_test_i8_e8m0_fill_model(uint8_t *model_raw,
                                         uint64_t gate_offset,
                                         uint64_t gate_scale_offset,
                                         uint64_t up_offset,
                                         uint64_t up_scale_offset,
                                         uint64_t down_offset,
                                         uint64_t down_scale_offset,
                                         uint32_t n_total_expert,
                                         uint32_t expert_in_dim,
                                         uint32_t expert_mid_dim,
                                         uint32_t out_dim) {
    const uint32_t in_blocks = expert_in_dim / 16u;
    const uint32_t mid_blocks = expert_mid_dim / 16u;
    const uint64_t gate_expert_bytes = (uint64_t)expert_mid_dim * expert_in_dim;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * expert_mid_dim;
    const uint64_t gate_scale_expert_bytes = (uint64_t)expert_mid_dim * in_blocks;
    const uint64_t down_scale_expert_bytes = (uint64_t)out_dim * mid_blocks;

    int8_t *gate_w = (int8_t *)(model_raw + gate_offset);
    int8_t *up_w   = (int8_t *)(model_raw + up_offset);
    int8_t *down_w = (int8_t *)(model_raw + down_offset);
    uint8_t *gate_s = model_raw + gate_scale_offset;
    uint8_t *up_s   = model_raw + up_scale_offset;
    uint8_t *down_s = model_raw + down_scale_offset;

    for (uint32_t e = 0; e < n_total_expert; e++) {
        int8_t *ge = gate_w + (uint64_t)e * gate_expert_bytes;
        int8_t *ue = up_w   + (uint64_t)e * gate_expert_bytes;   /* up same shape as gate */
        int8_t *de = down_w + (uint64_t)e * down_expert_bytes;
        uint8_t *gs = gate_s + (uint64_t)e * gate_scale_expert_bytes;
        uint8_t *us = up_s   + (uint64_t)e * gate_scale_expert_bytes;
        uint8_t *ds = down_s + (uint64_t)e * down_scale_expert_bytes;

        for (uint32_t r = 0; r < expert_mid_dim; r++) {
            for (uint32_t c = 0; c < expert_in_dim; c++) {
                ge[r * expert_in_dim + c] = (int8_t)(((e * 17 + r * 3 + c) % 251u) - 125);
                ue[r * expert_in_dim + c] = (int8_t)(((e * 31 + r * 5 + c * 7) % 251u) - 125);
            }
            for (uint32_t b = 0; b < in_blocks; b++) {
                gs[r * in_blocks + b] = 0x7Fu;   /* E8M0 = 1.0 */
                us[r * in_blocks + b] = 0x7Fu;
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            for (uint32_t c = 0; c < expert_mid_dim; c++) {
                de[r * expert_mid_dim + c] = (int8_t)(((e * 13 + r * 11 + c * 2) % 251u) - 125);
            }
            for (uint32_t b = 0; b < mid_blocks; b++) {
                ds[r * mid_blocks + b] = 0x7Fu;
            }
        }
    }
}

/* Story 11.51 AC4 (mid L2_REL ≤ 5e-3) + AC5 (out L2_REL ≤ 5e-3) + D7
 * invariants-guard runtime parity asserts. quality=false for BOTH reference and
 * DUT so identical-quality fused is compared against identical-quality
 * reference. */
static void test_metal_i8_e8m0_single_token_numeric(void) {
    const int routing = ds4_gpu_test_i8_e8m0_host_dispatch_routing();
    if (routing < 0) {
        fprintf(stderr, "ds4: Metal unavailable; skipping --metal-i8-e8m0-single-token-numeric\n");
        return;
    }
    TEST_ASSERT(routing == 1);
    if (routing != 1) return;

    /* === D2 fixture === */
    const uint32_t expert_in_dim = 256;     /* multiples of 256 (one_tensor guard) AND of 16 (scale geometry) */
    const uint32_t expert_mid_dim = 256;
    const uint32_t out_dim = 256;
    const uint32_t n_total_expert = 6;
    const uint32_t n_expert = 6;
    const uint32_t N_TOK_REF = 32;           /* clears use_mm_id gate L25306 */
    const uint32_t N_TOK_DUT = 1;            /* one_tensor path */
    const uint32_t pair_rows_ref = N_TOK_REF * n_expert;   /* 192 */
    const uint32_t pair_rows_dut = N_TOK_DUT * n_expert;   /* 6 */

    const uint64_t gate_expert_bytes = (uint64_t)expert_mid_dim * expert_in_dim;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * expert_mid_dim;
    const uint64_t gate_scale_expert_bytes = (uint64_t)expert_mid_dim * (expert_in_dim / 16u);
    const uint64_t down_scale_expert_bytes = (uint64_t)out_dim * (expert_mid_dim / 16u);
    const uint64_t gate_weight_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_weight_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t gate_scale_bytes = (uint64_t)n_total_expert * gate_scale_expert_bytes;
    const uint64_t down_scale_bytes = (uint64_t)n_total_expert * down_scale_expert_bytes;

    /* Model map per ADR 0021 paired-tensor contract: gate_w | gate_s | up_w | up_s | down_w | down_s. */
    const uint64_t gate_offset = 0;
    const uint64_t gate_scale_offset = gate_offset + gate_weight_bytes;
    const uint64_t up_offset = gate_scale_offset + gate_scale_bytes;
    const uint64_t up_scale_offset = up_offset + gate_weight_bytes;       /* up weight same shape as gate */
    const uint64_t down_offset = up_scale_offset + gate_scale_bytes;       /* up scale same shape as gate scale */
    const uint64_t down_scale_offset = down_offset + down_weight_bytes;
    const uint64_t model_bytes = down_scale_offset + down_scale_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());

    /* Production row stride semantics — I8_E8M0 unblocked int8: row stride = dim[0] bytes
     * (routed_expert_row_bytes, ds4.c L3251). Identical on reference and DUT. */
    const uint64_t gate_row_bytes = expert_in_dim;
    const uint64_t down_row_bytes = expert_mid_dim;

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    TEST_ASSERT(model_raw != NULL);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);
    ds4_test_i8_e8m0_fill_model((uint8_t *)model_raw,
                                 gate_offset, gate_scale_offset,
                                 up_offset, up_scale_offset,
                                 down_offset, down_scale_offset,
                                 n_total_expert,
                                 expert_in_dim, expert_mid_dim, out_dim);

    /* Single-token input x (deterministic, non-zero, no ±0 row). */
    float *x_tok = calloc((size_t)expert_in_dim, sizeof(float));
    TEST_ASSERT(x_tok != NULL);
    if (!x_tok) { free(model_raw); return; }
    for (uint32_t i = 0; i < expert_in_dim; i++) {
        x_tok[i] = (float)((int)(i % 7) - 3) * 0.5f;   /* range [-1.5, +1.5] */
    }

    /* Shared model map — backs BOTH reference and DUT no-copy Metal views. */
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);

    int ok_all = 1;

    /* === D3 — REFERENCE: batched n_tokens=32 (use_mm_id GEMM path) === */
    ds4_gpu_tensor *out_ref = ds4_gpu_tensor_alloc((uint64_t)N_TOK_REF * out_dim * sizeof(float));
    ds4_gpu_tensor *gate_ref = ds4_gpu_tensor_alloc((uint64_t)pair_rows_ref * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *up_ref = ds4_gpu_tensor_alloc((uint64_t)pair_rows_ref * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *mid_ref = ds4_gpu_tensor_alloc((uint64_t)pair_rows_ref * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *experts_ref = ds4_gpu_tensor_alloc((uint64_t)pair_rows_ref * out_dim * sizeof(float));
    ds4_gpu_tensor *sel_ref = ds4_gpu_tensor_alloc((uint64_t)pair_rows_ref * sizeof(int32_t));
    ds4_gpu_tensor *w_ref = ds4_gpu_tensor_alloc((uint64_t)pair_rows_ref * sizeof(float));
    ds4_gpu_tensor *x_ref = ds4_gpu_tensor_alloc((uint64_t)N_TOK_REF * expert_in_dim * sizeof(float));
    TEST_ASSERT(out_ref && gate_ref && up_ref && mid_ref && experts_ref && sel_ref && w_ref && x_ref);
    if (!out_ref || !gate_ref || !up_ref || !mid_ref || !experts_ref || !sel_ref || !w_ref || !x_ref) {
        ds4_gpu_tensor_free(out_ref);  ds4_gpu_tensor_free(gate_ref); ds4_gpu_tensor_free(up_ref);
        ds4_gpu_tensor_free(mid_ref); ds4_gpu_tensor_free(experts_ref);
        ds4_gpu_tensor_free(sel_ref); ds4_gpu_tensor_free(w_ref); ds4_gpu_tensor_free(x_ref);
        free(x_tok); /* model_raw kept live (Metal no-copy views) */ return;
    }

    /* x_ref: replicate x_tok across all N_TOK_REF rows. */
    for (uint32_t t = 0; t < N_TOK_REF; t++) {
        TEST_ASSERT(ds4_gpu_tensor_write(x_ref,
                                         (uint64_t)t * expert_in_dim * sizeof(float),
                                         x_tok,
                                         (uint64_t)expert_in_dim * sizeof(float)) != 0);
    }

    /* sel_ref / w_ref: token-major p = t*n_expert + e → expert e, weight 1.0. */
    int32_t *sel_ref_host = calloc((size_t)pair_rows_ref, sizeof(int32_t));
    float *w_ref_host = calloc((size_t)pair_rows_ref, sizeof(float));
    TEST_ASSERT(sel_ref_host && w_ref_host);
    if (!sel_ref_host || !w_ref_host) {
        free(sel_ref_host); free(w_ref_host);
        ds4_gpu_tensor_free(out_ref);  ds4_gpu_tensor_free(gate_ref); ds4_gpu_tensor_free(up_ref);
        ds4_gpu_tensor_free(mid_ref); ds4_gpu_tensor_free(experts_ref);
        ds4_gpu_tensor_free(sel_ref); ds4_gpu_tensor_free(w_ref); ds4_gpu_tensor_free(x_ref);
        free(x_tok); return;
    }
    for (uint32_t t = 0; t < N_TOK_REF; t++) {
        for (uint32_t e = 0; e < n_expert; e++) {
            const uint32_t p = t * n_expert + e;
            sel_ref_host[p] = (int32_t)e;
            w_ref_host[p] = 1.0f;
        }
    }
    TEST_ASSERT(ds4_gpu_tensor_write(sel_ref, 0, sel_ref_host, (uint64_t)pair_rows_ref * sizeof(int32_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(w_ref, 0, w_ref_host, (uint64_t)pair_rows_ref * sizeof(float)) != 0);

    bool mid_is_f16 = true;
    ds4_gpu_set_quality(false);
    const int ok_ref = ds4_gpu_routed_moe_batch_tensor(out_ref, gate_ref, up_ref, mid_ref, experts_ref,
                                                         model_raw, model_alloc,
                                                         gate_offset, up_offset, down_offset,
                                                         DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                         DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                         gate_expert_bytes, gate_row_bytes,
                                                         down_expert_bytes, down_row_bytes,
                                                         expert_in_dim, expert_mid_dim, out_dim,
                                                         sel_ref, w_ref,
                                                         n_total_expert, n_expert,
                                                         0.0f,
                                                         x_ref, 0, N_TOK_REF, &mid_is_f16);
    TEST_ASSERT(ok_ref == 1);
    TEST_ASSERT(mid_is_f16 == false);
    if (ok_ref == 1) {
        TEST_ASSERT(ds4_gpu_synchronize() != 0);
    }
    ok_all = ok_all && (ok_ref == 1);

    /* === D4 — DUT: one_tensor n_tokens=1 (fused pair_swiglu gate+up) + D2.5 down GEMM === */
    ds4_gpu_tensor *out_dut = ds4_gpu_tensor_alloc((uint64_t)N_TOK_DUT * out_dim * sizeof(float));
    ds4_gpu_tensor *gate_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *up_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *mid_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *experts_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * out_dim * sizeof(float));
    ds4_gpu_tensor *sel_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * sizeof(int32_t));
    ds4_gpu_tensor *w_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * sizeof(float));
    /* pair_swiglu reads pair p from x + p*src_pair_stride (src_pair_stride = expert_in_dim*sizeof(float));
     * so x_dut spans pair_rows_dut * expert_in_dim floats, each pair slot a replicate of x_tok. */
    ds4_gpu_tensor *x_dut = ds4_gpu_tensor_alloc((uint64_t)pair_rows_dut * expert_in_dim * sizeof(float));
    TEST_ASSERT(out_dut && gate_dut && up_dut && mid_dut && experts_dut && sel_dut && w_dut && x_dut);
    if (!out_dut || !gate_dut || !up_dut || !mid_dut || !experts_dut || !sel_dut || !w_dut || !x_dut) {
        ds4_gpu_tensor_free(out_dut); ds4_gpu_tensor_free(gate_dut); ds4_gpu_tensor_free(up_dut);
        ds4_gpu_tensor_free(mid_dut); ds4_gpu_tensor_free(experts_dut);
        ds4_gpu_tensor_free(sel_dut); ds4_gpu_tensor_free(w_dut); ds4_gpu_tensor_free(x_dut);
        free(sel_ref_host); free(w_ref_host); free(x_tok);
        ds4_gpu_tensor_free(out_ref); ds4_gpu_tensor_free(gate_ref); ds4_gpu_tensor_free(up_ref);
        ds4_gpu_tensor_free(mid_ref); ds4_gpu_tensor_free(experts_ref);
        ds4_gpu_tensor_free(sel_ref); ds4_gpu_tensor_free(w_ref); ds4_gpu_tensor_free(x_ref);
        return;
    }

    for (uint32_t p = 0; p < pair_rows_dut; p++) {
        TEST_ASSERT(ds4_gpu_tensor_write(x_dut,
                                         (uint64_t)p * expert_in_dim * sizeof(float),
                                         x_tok,
                                         (uint64_t)expert_in_dim * sizeof(float)) != 0);
    }
    int32_t *sel_dut_host = calloc((size_t)pair_rows_dut, sizeof(int32_t));
    float *w_dut_host = calloc((size_t)pair_rows_dut, sizeof(float));
    TEST_ASSERT(sel_dut_host && w_dut_host);
    if (!sel_dut_host || !w_dut_host) {
        free(sel_dut_host); free(w_dut_host);
        ds4_gpu_tensor_free(out_dut); ds4_gpu_tensor_free(gate_dut); ds4_gpu_tensor_free(up_dut);
        ds4_gpu_tensor_free(mid_dut); ds4_gpu_tensor_free(experts_dut);
        ds4_gpu_tensor_free(sel_dut); ds4_gpu_tensor_free(w_dut); ds4_gpu_tensor_free(x_dut);
        free(sel_ref_host); free(w_ref_host); free(x_tok);
        ds4_gpu_tensor_free(out_ref); ds4_gpu_tensor_free(gate_ref); ds4_gpu_tensor_free(up_ref);
        ds4_gpu_tensor_free(mid_ref); ds4_gpu_tensor_free(experts_ref);
        ds4_gpu_tensor_free(sel_ref); ds4_gpu_tensor_free(w_ref); ds4_gpu_tensor_free(x_ref);
        return;
    }
    for (uint32_t p = 0; p < pair_rows_dut; p++) {
        sel_dut_host[p] = (int32_t)p;   /* pair p → expert p (0..n_expert-1) */
        /* Production-convention per-pair router weights: weights[pair] laid out
         * contiguously with stride sizeof(float) (matches ds4_gpu_encode_moe_
         * swiglu_weight's .weight_stride = sizeof(float) in the batched reference
         * path). This is the layout production emits for single-token decode
         * (g->router_weights holds n_expert contiguous floats). */
        w_dut_host[p] = 1.0f;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(sel_dut, 0, sel_dut_host, (uint64_t)pair_rows_dut * sizeof(int32_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(w_dut, 0, w_dut_host, (uint64_t)pair_rows_dut * sizeof(float)) != 0);

    ds4_gpu_set_quality(false);
    const int ok_dut = ds4_gpu_routed_moe_one_tensor(out_dut, gate_dut, up_dut, mid_dut, experts_dut,
                                                       model_raw, model_alloc,
                                                       gate_offset, up_offset, down_offset,
                                                       DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                       DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                       gate_expert_bytes, gate_row_bytes,
                                                       down_expert_bytes, down_row_bytes,
                                                       expert_in_dim, expert_mid_dim, out_dim,
                                                       sel_dut, w_dut,
                                                       n_total_expert, n_expert,
                                                       0.0f,
                                                       x_dut, 0);
    TEST_ASSERT(ok_dut == 1);
    if (ok_dut == 1) {
        TEST_ASSERT(ds4_gpu_synchronize() != 0);
    }
    ok_all = ok_all && (ok_dut == 1);

    /* D7 invariants-guard (defense-in-depth): routing helper already corroborated
     * that BOTH FROZEN pipelines (mv pair_swiglu + mm down GEMM) are bound; the
     * ok_ref / ok_dut return codes corroborate dispatch actually fired; and
     * mid_is_f16==false corroborates the I8_E8M0 f32 mid path (not the f16
     * fallback). Non-degenerate reference norms guard against a silent
     * all-zero tautology (AC4/AC5 comparison is only meaningful if both sides
     * carry signal). */
    TEST_ASSERT(ok_ref == 1);
    TEST_ASSERT(ok_dut == 1);
    TEST_ASSERT(mid_is_f16 == false);

    if (ok_ref == 1 && ok_dut == 1) {
        const uint64_t mid_row0_floats = (uint64_t)n_expert * expert_mid_dim;
        const uint64_t out_row0_floats = (uint64_t)out_dim;
        float *mid_ref_row0 = calloc((size_t)mid_row0_floats, sizeof(float));
        float *out_ref_row0 = calloc((size_t)out_row0_floats, sizeof(float));
        float *mid_dut_row0 = calloc((size_t)mid_row0_floats, sizeof(float));
        float *out_dut_row0 = calloc((size_t)out_row0_floats, sizeof(float));
        TEST_ASSERT(mid_ref_row0 && out_ref_row0 && mid_dut_row0 && out_dut_row0);
        if (mid_ref_row0 && out_ref_row0 && mid_dut_row0 && out_dut_row0) {
            TEST_ASSERT(ds4_gpu_tensor_read(mid_ref, 0, mid_ref_row0, mid_row0_floats * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(out_ref, 0, out_ref_row0, out_row0_floats * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(mid_dut, 0, mid_dut_row0, mid_row0_floats * sizeof(float)) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(out_dut, 0, out_dut_row0, out_row0_floats * sizeof(float)) != 0);

            /* Non-degenerate reference signal guards (D7). */
            double n_mid_ref = 0.0, n_out_ref = 0.0;
            for (uint64_t i = 0; i < mid_row0_floats; i++) n_mid_ref += (double)mid_ref_row0[i] * mid_ref_row0[i];
            for (uint64_t i = 0; i < out_row0_floats; i++) n_out_ref += (double)out_ref_row0[i] * out_ref_row0[i];
            TEST_ASSERT(n_mid_ref > 1e-6);
            TEST_ASSERT(n_out_ref > 1e-6);

            /* AC4 — gate+up fused encoder numeric gate.
             *
             * STOP-STATE (TDD RED, Story 11.51 Coder STOP+ESCALATE 2026-06-24):
             * This assert FAILS with L2_REL ≈ 9.3e-1 (>> 5e-3) using the
             * production-convention per-pair router-weights layout forced here.
             * Root cause (VERIFIED-RAW + empirical confirmation this slice):
             * the FROZEN Story 11.50 I8_E8M0 pair_swiglu host-side arg struct
             * at ds4_metal.m:24447 sets
             *     .weight_stride = (uint64_t)n_expert * sizeof(float)
             * but the kernel `kernel_mul_mv_id_i8_e8m0_pair_swiglu_f32` reads
             *     route_weights + pair * weight_stride
             * i.e. route_weights[pair * n_expert], NOT route_weights[pair].
             * With a contiguous per-pair weights buffer (n_expert floats, the
             * layout production emits for single-token decode AND the layout the
             * batched `ds4_gpu_encode_moe_swiglu_weight` reference path expects
             * with .weight_stride = sizeof(float)), pairs 1..n_expert-1 read
             * OUT OF BOUNDS → route weight resolves to 0 → mid contribution for
             * those pairs is ZERO → the fused single-token MoE collapses to the
             * pair-0 expert only (1/n_expert of the correct output).
             *
             * Empirical confirmation this slice: sizing w_dut_host for the
             * BUGGY stride (1.0 at every pair*n_expert offset) lights ALL pairs
             * up bit-identical to the reference (L2_REL = 0.000e+00). That proves
             * the pair_swiglu KERNEL MATH is correct; the defect is purely the
             * host-side .weight_stride field. Was masked by the 11.50
             * reachability test (zero router weights → route==0 everywhere, so
             * the OOB reads of 0 were indistinguishable from correct 0).
             *
             * Fixing it requires editing FROZEN ds4_metal.m:24447
             * (.weight_stride = sizeof(float)), which is OFF-LIMITS this slice
             * (AC8 sha256-identical HEAD 068d498). Per the task-coder.md
             * STOP-rule, Coder does NOT improvise a production source edit;
             * the test stays RED as the numeric gate, and goes GREEN once the
             * supervisor's follow-up ds4_metal.m weight_stride-fix slice lands.
             */
            const double l2_mid = ds4_test_i8_e8m0_l2_rel(mid_dut_row0, mid_ref_row0, mid_row0_floats);
            fprintf(stderr, "ds4: 11.51 AC4 mid L2_REL(dut, ref) = %.3e (gate = %.3e)\n",
                    l2_mid, (double)DS4_TEST_I8_E8M0_L2REL);
            /* TDD diagnostic: characterize mismatch structure (remove after root-cause). */
            {
                int32_t ids_back[6] = { -99,-99,-99,-99,-99,-99 };
                TEST_ASSERT(ds4_gpu_tensor_read(sel_dut, 0, ids_back, (uint64_t)pair_rows_dut * sizeof(int32_t)) != 0);
                fprintf(stderr, "ds4: 11.51 DIAG sel_dut ids back =");
                for (int i = 0; i < 6; i++) fprintf(stderr, " %d", ids_back[i]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "ds4: 11.51 DIAG mid_ref[0..7] =");
            for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", mid_ref_row0[i]);
            fprintf(stderr, "\n");
            fprintf(stderr, "ds4: 11.51 DIAG mid_dut[0..7] =");
            for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", mid_dut_row0[i]);
            fprintf(stderr, "\n");
            for (uint32_t p = 0; p < n_expert; p++) {
                double pr = 0.0, pd = 0.0;
                for (uint32_t r = 0; r < expert_mid_dim; r++) {
                    pr += fabs(mid_ref_row0[p*expert_mid_dim+r]);
                    pd += fabs(mid_dut_row0[p*expert_mid_dim+r]);
                }
                fprintf(stderr, "ds4: 11.51 DIAG pair %u meanAbs ref=%.4f dut=%.4f\n",
                        p, pr/expert_mid_dim, pd/expert_mid_dim);
            }
            TEST_ASSERT(l2_mid <= (double)DS4_TEST_I8_E8M0_L2REL);

            /* AC5 — down GEMM single-token END-TO-END numeric gate (NOT amended
             * to parity; stays L2_REL ≤ 5e-3). Catches down-arm amplification of
             * small AC4 mid drift + routing bugs the per-expert AC4 gate misses. */
            const double l2_out = ds4_test_i8_e8m0_l2_rel(out_dut_row0, out_ref_row0, out_row0_floats);
            fprintf(stderr, "ds4: 11.51 AC5 out L2_REL(dut, ref) = %.3e (gate = %.3e)\n",
                    l2_out, (double)DS4_TEST_I8_E8M0_L2REL);
            TEST_ASSERT(l2_out <= (double)DS4_TEST_I8_E8M0_L2REL);
        } else {
            ok_all = 0;
        }
        free(mid_ref_row0); free(out_ref_row0); free(mid_dut_row0); free(out_dut_row0);
    }

    free(sel_ref_host); free(w_ref_host);
    free(sel_dut_host); free(w_dut_host);
    free(x_tok);
    ds4_gpu_tensor_free(out_ref); ds4_gpu_tensor_free(gate_ref); ds4_gpu_tensor_free(up_ref);
    ds4_gpu_tensor_free(mid_ref); ds4_gpu_tensor_free(experts_ref);
    ds4_gpu_tensor_free(sel_ref); ds4_gpu_tensor_free(w_ref); ds4_gpu_tensor_free(x_ref);
    ds4_gpu_tensor_free(out_dut); ds4_gpu_tensor_free(gate_dut); ds4_gpu_tensor_free(up_dut);
    ds4_gpu_tensor_free(mid_dut); ds4_gpu_tensor_free(experts_dut);
    ds4_gpu_tensor_free(sel_dut); ds4_gpu_tensor_free(w_dut); ds4_gpu_tensor_free(x_dut);
    /* model_raw backs Metal no-copy model views; keep it live until process exit
     * (mirrors 11.50 host-dispatch teardown precedent). */
    TEST_ASSERT(ok_all == 1);
}
