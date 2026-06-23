/* Story 11.47 host-dispatch smoke for the routed I8+F8_E8M0 Metal path.
 * Included by tests/ds4_test.c so it can reuse the tiny C test harness.
 */

#ifndef DS4_TEST_METAL_TENSOR_I8_E8M0
#define DS4_TEST_METAL_TENSOR_I8_E8M0 64u
#endif

extern int ds4_gpu_test_i8_e8m0_host_dispatch_routing(void);

/* Story 11.50 D2.d/D2.e/D2.h + D3.2/D3.3 — single-token (n_tokens=1) I8+E8M0
 * routed-expert dispatch-reachability via `ds4_gpu_routed_moe_one_tensor`.
 * REACHABILITY ONLY: zeroed weights ⇒ gate/up/down outputs are deterministic
 * zeros; we assert only that the dispatch returns ok==1 and the device syncs.
 * Numeric correctness (L2_REL vs 11.47 batched reference) is owned by 11.51.
 *
 * AC13 (silent-numerics guard) is enforced STATICALLY: the GREEN path the
 * helper exercises rides the §6 D2.5 down GEMM via
 * `ds4_gpu_encode_mul_mm_id_i8_e8m0_f32` (ds4_metal.m §6 branch, before the
 * generic mv fallback at the `} else if (ok) {` down arm). The prior Architect
 * STOP pathology was the down arm falling through to that generic mv|pair_swiglu
 * path; the §6 branch + AC4 relax + AC6 fuse-override close it. The Test Manager
 * xhigh audit (grep/code-read) confirms the §6 branch precedes the generic
 * fallback and references `ds4_gpu_encode_mul_mm_id_i8_e8m0_f32`; this runtime
 * helper corroborates it by reaching dispatch without the fail-closed gate. */
static void test_metal_i8_e8m0_dispatch_single_token(bool quality) {
    const int routing = ds4_gpu_test_i8_e8m0_host_dispatch_routing();
    if (routing < 0) {
        fprintf(stderr, "ds4: Metal unavailable; skipping --metal-i8-e8m0-dispatch single-token (%s)\n",
                quality ? "quality" : "fast");
        return;
    }
    TEST_ASSERT(routing == 1);
    if (routing != 1) return;

    const uint32_t expert_in_dim = 256;   /* multiples of 256 (one_tensor guard) AND of 16 (scale geometry) */
    const uint32_t expert_mid_dim = 256;
    const uint32_t out_dim = 256;
    const uint32_t n_total_expert = 6;
    const uint32_t n_expert = 6;
    const uint32_t n_tokens = 1;           /* one_tensor path */
    const uint32_t pair_rows = n_tokens * n_expert;   /* = n_expert */

    const uint64_t gate_expert_bytes = (uint64_t)expert_mid_dim * expert_in_dim;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * expert_mid_dim;
    const uint64_t gate_scale_expert_bytes = (uint64_t)expert_mid_dim * (expert_in_dim / 16u);
    const uint64_t down_scale_expert_bytes = (uint64_t)out_dim * (expert_mid_dim / 16u);
    const uint64_t gate_weight_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_weight_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t gate_scale_bytes = (uint64_t)n_total_expert * gate_scale_expert_bytes;
    const uint64_t down_scale_bytes = (uint64_t)n_total_expert * down_scale_expert_bytes;

    /* Model tensor layout per ADR 0007: gate_w | gate_s | up_w | up_s | down_w | down_s. */
    const uint64_t gate_offset = 0;
    const uint64_t gate_scale_offset = gate_offset + gate_weight_bytes;
    const uint64_t up_offset = gate_scale_offset + gate_scale_bytes;
    const uint64_t up_scale_offset = up_offset + gate_weight_bytes;
    const uint64_t down_offset = up_scale_offset + gate_scale_bytes;
    const uint64_t down_scale_offset = down_offset + down_weight_bytes;
    const uint64_t model_bytes = down_scale_offset + down_scale_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    TEST_ASSERT(model_raw != NULL);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);
    (void)gate_scale_offset;
    (void)up_scale_offset;
    (void)down_scale_offset;

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)pair_rows * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)pair_rows * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)pair_rows * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc((uint64_t)pair_rows * out_dim * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)pair_rows * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)pair_rows * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * expert_in_dim * sizeof(float));
    TEST_ASSERT(out && gate && up && mid && experts && selected && weights && x);
    if (!out || !gate || !up || !mid || !experts || !selected || !weights || !x) {
        ds4_gpu_tensor_free(out);  ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(x); free(model_raw); return;
    }

    int32_t *selected_host = calloc((size_t)pair_rows, sizeof(int32_t));
    float *weights_host = calloc((size_t)pair_rows, sizeof(float));
    TEST_ASSERT(selected_host && weights_host);
    if (!selected_host || !weights_host) {
        free(selected_host); free(weights_host);
        ds4_gpu_tensor_free(out);  ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(x); free(model_raw); return;
    }
    for (uint32_t p = 0; p < pair_rows; p++) {
        selected_host[p] = (int32_t)p;   /* each pair → expert p (0..n_expert-1) */
        weights_host[p] = 1.0f;
    }
    TEST_ASSERT(ds4_gpu_tensor_write(selected, 0, selected_host, (uint64_t)pair_rows * sizeof(int32_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(weights, 0, weights_host, (uint64_t)pair_rows * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_fill_f32(x, 0.0f, (uint64_t)n_tokens * expert_in_dim) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);

    ds4_gpu_set_quality(quality);
    /* AC8 (quality=false) + AC9 (quality=true): REACHABILITY only — ok==1 + sync. */
    const int ok = ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, experts,
                                                  model_raw, model_alloc,
                                                  gate_offset, up_offset, down_offset,
                                                  DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                  DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                  gate_expert_bytes,
                                                  gate_expert_bytes,    /* gate_row_bytes */
                                                  down_expert_bytes,
                                                  down_expert_bytes,    /* down_row_bytes */
                                                  expert_in_dim, expert_mid_dim, out_dim,
                                                  selected, weights,
                                                  n_total_expert, n_expert,
                                                  0.0f,    /* clamp off; reachability only */
                                                  x, 0);
    TEST_ASSERT(ok == 1);
    if (ok == 1) {
        TEST_ASSERT(ds4_gpu_synchronize() != 0);
    }

    free(selected_host);
    free(weights_host);
    ds4_gpu_tensor_free(out);  ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(experts);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    /* model_raw backs Metal no-copy model views; keep it live until process exit. */
}

static void test_metal_i8_e8m0_dispatch(void) {
    const int routing = ds4_gpu_test_i8_e8m0_host_dispatch_routing();
    if (routing < 0) {
        fprintf(stderr, "ds4: Metal unavailable; skipping --metal-i8-e8m0-dispatch\n");
        return;
    }
    TEST_ASSERT(routing == 1);
    if (routing != 1) return;

    const uint32_t expert_in_dim = 256;
    const uint32_t expert_mid_dim = 256;
    const uint32_t out_dim = 256;
    const uint32_t n_total_expert = 6;
    const uint32_t n_expert = 6;
    const uint32_t n_tokens = 32;
    const uint32_t pair_rows = n_tokens * n_expert;

    const uint64_t gate_expert_bytes = (uint64_t)expert_mid_dim * expert_in_dim;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * expert_mid_dim;
    const uint64_t gate_scale_expert_bytes = (uint64_t)expert_mid_dim * (expert_in_dim / 16u);
    const uint64_t down_scale_expert_bytes = (uint64_t)out_dim * (expert_mid_dim / 16u);
    const uint64_t gate_weight_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_weight_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t gate_scale_bytes = (uint64_t)n_total_expert * gate_scale_expert_bytes;
    const uint64_t down_scale_bytes = (uint64_t)n_total_expert * down_scale_expert_bytes;

    const uint64_t gate_offset = 0;
    const uint64_t gate_scale_offset = gate_offset + gate_weight_bytes;
    const uint64_t up_offset = gate_scale_offset + gate_scale_bytes;
    const uint64_t up_scale_offset = up_offset + gate_weight_bytes;
    const uint64_t down_offset = up_scale_offset + gate_scale_bytes;
    const uint64_t down_scale_offset = down_offset + down_weight_bytes;
    const uint64_t model_bytes = down_scale_offset + down_scale_bytes;
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    TEST_ASSERT(model_raw != NULL);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);
    (void)gate_scale_offset;
    (void)up_scale_offset;
    (void)down_scale_offset;

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)pair_rows * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)pair_rows * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)pair_rows * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc((uint64_t)pair_rows * out_dim * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)pair_rows * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)pair_rows * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * expert_in_dim * sizeof(float));
    TEST_ASSERT(out && gate && up && mid && experts && selected && weights && x);
    if (!out || !gate || !up || !mid || !experts || !selected || !weights || !x) {
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(gate);
        ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(x);
        free(model_raw);
        return;
    }

    int32_t *selected_host = calloc((size_t)pair_rows, sizeof(int32_t));
    float *weights_host = calloc((size_t)pair_rows, sizeof(float));
    TEST_ASSERT(selected_host && weights_host);
    if (!selected_host || !weights_host) {
        free(selected_host);
        free(weights_host);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(gate);
        ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weights);
        ds4_gpu_tensor_free(x);
        free(model_raw);
        return;
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t e = 0; e < n_expert; e++) {
            const uint32_t p = t * n_expert + e;
            selected_host[p] = (int32_t)e;
            weights_host[p] = 1.0f;
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(selected, 0, selected_host, (uint64_t)pair_rows * sizeof(int32_t)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(weights, 0, weights_host, (uint64_t)pair_rows * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_tensor_fill_f32(x, 0.0f, (uint64_t)n_tokens * expert_in_dim) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);

    bool mid_is_f16 = true;
    ds4_gpu_set_quality(false);
    const int ok = ds4_gpu_routed_moe_batch_tensor(out,
                                                   gate,
                                                   up,
                                                   mid,
                                                   experts,
                                                   model_raw,
                                                   model_alloc,
                                                   gate_offset,
                                                   up_offset,
                                                   down_offset,
                                                   DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                   DS4_TEST_METAL_TENSOR_I8_E8M0,
                                                   gate_expert_bytes,
                                                   expert_in_dim,
                                                   down_expert_bytes,
                                                   expert_mid_dim,
                                                   expert_in_dim,
                                                   expert_mid_dim,
                                                   out_dim,
                                                   selected,
                                                   weights,
                                                   n_total_expert,
                                                   n_expert,
                                                   0.0f,
                                                   x,
                                                   0,
                                                   n_tokens,
                                                   &mid_is_f16);
    TEST_ASSERT(ok == 1);
    TEST_ASSERT(mid_is_f16 == false);
    TEST_ASSERT(ds4_gpu_synchronize() != 0);

    free(selected_host);
    free(weights_host);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(experts);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(x);
    /* model_raw backs Metal no-copy model views; keep it live until process exit. */

    /* Story 11.50 D3.2/D3.3 — single-token one_tensor dispatch-reachability
     * (AC8 fast mode + AC9 quality mode); AC13 silent-numerics guard documented in
     * `test_metal_i8_e8m0_dispatch_single_token`. */
    test_metal_i8_e8m0_dispatch_single_token(false);   /* AC8 */
    test_metal_i8_e8m0_dispatch_single_token(true);    /* AC9 */
}
