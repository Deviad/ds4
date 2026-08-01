#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_mtp_sparse_route.h"
#include "ds4.h"
#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __func__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static bool test_mtp_sparse_route_verify_batch(void) {
    const uint32_t top_k = 8u;
    const ds4_mtp_verify_batch_plan plan =
        ds4_mtp_verify_batch_plan_for(4u, top_k + 1u, top_k, 17u, 2u, 32u);

    REQUIRE(plan.use_batch_path);
    REQUIRE(plan.route == DS4_MTP_BATCH_ATTN_INDEXED_CSA);
    REQUIRE(plan.batch_indexer_scores);
    REQUIRE(plan.topk_selection);
    REQUIRE(plan.indexed_mixed_attention);
    REQUIRE(!plan.static_mixed_attention);
    return true;
}

static bool test_mtp_sparse_route_threshold_transition(void) {
    const uint32_t top_k = 8u;
    const uint32_t ratio = 4u;
    const uint32_t pos0 = 17u;
    const uint32_t n_tokens = 2u;
    const uint32_t raw_cap = 32u;
    const uint32_t pre_update_n_comp = top_k;
    const uint32_t post_update_n_comp = pre_update_n_comp + 1u;
    const ds4_mtp_verify_batch_plan pre_update_plan =
        ds4_mtp_verify_batch_plan_for(ratio,
                                       pre_update_n_comp,
                                       top_k,
                                       pos0,
                                       n_tokens,
                                       raw_cap);
    const ds4_mtp_verify_batch_plan post_update_plan =
        ds4_mtp_verify_batch_plan_for(ratio,
                                       post_update_n_comp,
                                       top_k,
                                       pos0,
                                       n_tokens,
                                       raw_cap);

    REQUIRE(pre_update_plan.use_batch_path);
    REQUIRE(pre_update_plan.route == DS4_MTP_BATCH_ATTN_STATIC_MIXED);
    REQUIRE(!pre_update_plan.batch_indexer_scores);
    REQUIRE(!pre_update_plan.topk_selection);
    REQUIRE(!pre_update_plan.indexed_mixed_attention);
    REQUIRE(pre_update_plan.static_mixed_attention);
    REQUIRE(post_update_plan.use_batch_path);
    REQUIRE(post_update_plan.route == DS4_MTP_BATCH_ATTN_INDEXED_CSA);
    REQUIRE(post_update_plan.batch_indexer_scores);
    REQUIRE(post_update_plan.topk_selection);
    REQUIRE(post_update_plan.indexed_mixed_attention);
    REQUIRE(!post_update_plan.static_mixed_attention);
    return true;
}

static bool test_mtp_sparse_route_fallbacks(void) {
    const uint32_t top_k = 8u;
    const ds4_mtp_verify_batch_plan boundary =
        ds4_mtp_verify_batch_plan_for(4u, top_k, top_k, 17u, 2u, 32u);
    const ds4_mtp_verify_batch_plan below =
        ds4_mtp_verify_batch_plan_for(4u, top_k - 1u, top_k, 17u, 2u, 32u);
    const ds4_mtp_verify_batch_plan ratio128 =
        ds4_mtp_verify_batch_plan_for(128u, top_k + 1u, top_k, 17u, 2u, 32u);
    const ds4_mtp_verify_batch_plan raw =
        ds4_mtp_verify_batch_plan_for(0u, top_k + 1u, top_k, 17u, 2u, 32u);

    REQUIRE(boundary.use_batch_path);
    REQUIRE(boundary.route == DS4_MTP_BATCH_ATTN_STATIC_MIXED);
    REQUIRE(boundary.static_mixed_attention);
    REQUIRE(!boundary.indexed_mixed_attention);
    REQUIRE(!boundary.batch_indexer_scores);
    REQUIRE(!boundary.topk_selection);
    REQUIRE(below.route == DS4_MTP_BATCH_ATTN_STATIC_MIXED);
    REQUIRE(!below.indexed_mixed_attention);
    REQUIRE(ratio128.route == DS4_MTP_BATCH_ATTN_STATIC_MIXED);
    REQUIRE(!ratio128.indexed_mixed_attention);
    REQUIRE(raw.route == DS4_MTP_BATCH_ATTN_FULL_RAW);
    REQUIRE(!raw.static_mixed_attention);
    REQUIRE(!raw.indexed_mixed_attention);
    return true;
}

static bool test_mtp_sparse_route_sort_policy(void) {
    const ds4_mtp_indexed_attention_command_plan multi_token =
        ds4_mtp_indexed_attention_command_plan_for(2u, false);
    const ds4_mtp_indexed_attention_command_plan fast_decode =
        ds4_mtp_indexed_attention_command_plan_for(1u, false);
    const ds4_mtp_indexed_attention_command_plan quality_decode =
        ds4_mtp_indexed_attention_command_plan_for(1u, true);

    REQUIRE(multi_token.n_events == 2u);
    REQUIRE(multi_token.events[0] == DS4_MTP_INDEXED_ATTN_SORT_TOPK);
    REQUIRE(fast_decode.n_events == 1u);
    REQUIRE(fast_decode.events[0] == DS4_MTP_INDEXED_ATTN_ATTEND_ORIGINAL_TOPK);
    REQUIRE(quality_decode.n_events == 2u);
    REQUIRE(quality_decode.events[0] == DS4_MTP_INDEXED_ATTN_SORT_TOPK);
    return true;
}

static bool test_mtp_sparse_route_sort_kernel_rows(void) {
    const uint32_t row_width = 8u;
    const uint32_t n_rows = 2u;
    const int32_t source_values[] = {
        7, 1, 6, 0, 5, 2, 4, 3,
        15, 9, 14, 8, 13, 10, 12, 11,
    };
    const int32_t expected_values[] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    };
    const uint64_t bytes = (uint64_t)row_width * n_rows * sizeof(int32_t);
    ds4_gpu_tensor *source = NULL;
    ds4_gpu_tensor *sorted = NULL;
    int32_t actual_values[sizeof(expected_values) / sizeof(expected_values[0])] = { 0 };
    bool passed = false;

    source = ds4_gpu_tensor_alloc(bytes);
    sorted = ds4_gpu_tensor_alloc(bytes);
    if (!source || !sorted) {
        fprintf(stderr, "%s:%d: tensor allocation failed\n", __func__, __LINE__);
        goto cleanup;
    }
    if (ds4_gpu_tensor_write(source, 0, source_values, bytes) == 0) {
        fprintf(stderr, "%s:%d: tensor write failed\n", __func__, __LINE__);
        goto cleanup;
    }
    if (ds4_gpu_sort_i32_rows_asc_tensor(sorted, source, row_width, n_rows) == 0) {
        fprintf(stderr, "%s:%d: production row-sort kernel failed\n", __func__, __LINE__);
        goto cleanup;
    }
    if (ds4_gpu_tensor_read(sorted, 0, actual_values, bytes) == 0) {
        fprintf(stderr, "%s:%d: tensor read failed\n", __func__, __LINE__);
        goto cleanup;
    }
    for (size_t i = 0; i < sizeof(expected_values) / sizeof(expected_values[0]); i++) {
        if (actual_values[i] != expected_values[i]) {
            fprintf(stderr, "%s:%d: row %zu value %d != %d\n",
                    __func__, __LINE__, i, actual_values[i], expected_values[i]);
            goto cleanup;
        }
    }
    passed = true;

cleanup:
    ds4_gpu_tensor_free(sorted);
    ds4_gpu_tensor_free(source);
    return passed;
}

static bool test_mtp_sparse_route_sorted_binding(void) {
    const ds4_mtp_indexed_attention_command_plan multi_token =
        ds4_mtp_indexed_attention_command_plan_for(2u, false);
    const ds4_mtp_indexed_attention_command_plan fast_decode =
        ds4_mtp_indexed_attention_command_plan_for(1u, false);
    const ds4_mtp_indexed_attention_command_plan quality_decode =
        ds4_mtp_indexed_attention_command_plan_for(1u, true);

    REQUIRE(multi_token.n_events == 2u);
    REQUIRE(multi_token.events[0] == DS4_MTP_INDEXED_ATTN_SORT_TOPK);
    REQUIRE(multi_token.events[1] == DS4_MTP_INDEXED_ATTN_ATTEND_SORTED_TOPK);
    REQUIRE(fast_decode.n_events == 1u);
    REQUIRE(fast_decode.events[0] == DS4_MTP_INDEXED_ATTN_ATTEND_ORIGINAL_TOPK);
    REQUIRE(quality_decode.n_events == 2u);
    REQUIRE(quality_decode.events[0] == DS4_MTP_INDEXED_ATTN_SORT_TOPK);
    REQUIRE(quality_decode.events[1] == DS4_MTP_INDEXED_ATTN_ATTEND_SORTED_TOPK);
    return true;
}

typedef bool (*mtp_sparse_route_test_fn)(void);

typedef struct {
    const char *name;
    mtp_sparse_route_test_fn fn;
} mtp_sparse_route_test_case;

int main(void) {
    const mtp_sparse_route_test_case tests[] = {
        { "test_mtp_sparse_route_verify_batch", test_mtp_sparse_route_verify_batch },
        { "test_mtp_sparse_route_threshold_transition", test_mtp_sparse_route_threshold_transition },
        { "test_mtp_sparse_route_fallbacks", test_mtp_sparse_route_fallbacks },
        { "test_mtp_sparse_route_sort_policy", test_mtp_sparse_route_sort_policy },
        { "test_mtp_sparse_route_sort_kernel_rows", test_mtp_sparse_route_sort_kernel_rows },
        { "test_mtp_sparse_route_sorted_binding", test_mtp_sparse_route_sorted_binding },
    };
    const size_t n_tests = sizeof(tests) / sizeof(tests[0]);

    for (size_t i = 0; i < n_tests; i++) {
        printf("RUN %s\n", tests[i].name);
        if (!tests[i].fn()) {
            fprintf(stderr, "FAIL %s\n", tests[i].name);
            ds4_gpu_cleanup();
            return 1;
        }
        printf("PASS %s\n", tests[i].name);
    }
    ds4_gpu_cleanup();
    printf("PASS all %zu MTP sparse route tests\n", n_tests);
    return 0;
}
