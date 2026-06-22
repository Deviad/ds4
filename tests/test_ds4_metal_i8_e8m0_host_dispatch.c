/* Story 11.47 host-dispatch smoke for the routed I8+F8_E8M0 Metal path.
 * Included by tests/ds4_test.c so it can reuse the tiny C test harness.
 */

#ifndef DS4_TEST_METAL_TENSOR_I8_E8M0
#define DS4_TEST_METAL_TENSOR_I8_E8M0 64u
#endif

extern int ds4_gpu_test_i8_e8m0_host_dispatch_routing(void);

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
}
