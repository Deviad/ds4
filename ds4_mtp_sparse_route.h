#ifndef DS4_MTP_SPARSE_ROUTE_H
#define DS4_MTP_SPARSE_ROUTE_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    DS4_MTP_BATCH_ATTN_FULL_RAW = 0,
    DS4_MTP_BATCH_ATTN_STATIC_MIXED = 1,
    DS4_MTP_BATCH_ATTN_INDEXED_CSA = 2,
} ds4_mtp_batch_attention_route;

typedef struct {
    ds4_mtp_batch_attention_route route;
    bool use_batch_path;
    bool batch_indexer_scores;
    bool topk_selection;
    bool indexed_mixed_attention;
    bool static_mixed_attention;
} ds4_mtp_verify_batch_plan;

static inline ds4_mtp_batch_attention_route ds4_mtp_batch_attention_route_for(
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t top_k) {
    if (ratio == 4u && n_comp > top_k) {
        return DS4_MTP_BATCH_ATTN_INDEXED_CSA;
    }
    if (ratio != 0u) return DS4_MTP_BATCH_ATTN_STATIC_MIXED;
    return DS4_MTP_BATCH_ATTN_FULL_RAW;
}

static inline ds4_mtp_verify_batch_plan ds4_mtp_verify_batch_plan_for(
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t raw_cap) {
    const ds4_mtp_batch_attention_route route =
        ds4_mtp_batch_attention_route_for(ratio, n_comp, top_k);
    const bool use_batch_path = ratio != 0u && pos0 != 0u &&
                                n_tokens != 0u && n_tokens <= raw_cap;
    const bool indexed = use_batch_path && route == DS4_MTP_BATCH_ATTN_INDEXED_CSA;
    const bool static_mixed = use_batch_path && route == DS4_MTP_BATCH_ATTN_STATIC_MIXED;
    return (ds4_mtp_verify_batch_plan) {
        .route = route,
        .use_batch_path = use_batch_path,
        .batch_indexer_scores = indexed,
        .topk_selection = indexed,
        .indexed_mixed_attention = indexed,
        .static_mixed_attention = static_mixed,
    };
}

typedef enum {
    DS4_MTP_INDEXED_ATTN_SORT_TOPK = 0,
    DS4_MTP_INDEXED_ATTN_ATTEND_ORIGINAL_TOPK = 1,
    DS4_MTP_INDEXED_ATTN_ATTEND_SORTED_TOPK = 2,
} ds4_mtp_indexed_attention_event;

typedef struct {
    uint32_t n_events;
    ds4_mtp_indexed_attention_event events[2];
} ds4_mtp_indexed_attention_command_plan;

static inline ds4_mtp_indexed_attention_command_plan
    ds4_mtp_indexed_attention_command_plan_for(
        uint32_t n_tokens,
        bool quality_mode) {
    if (n_tokens > 1u || quality_mode) {
        return (ds4_mtp_indexed_attention_command_plan) {
            .n_events = 2u,
            .events = {
                DS4_MTP_INDEXED_ATTN_SORT_TOPK,
                DS4_MTP_INDEXED_ATTN_ATTEND_SORTED_TOPK,
            },
        };
    }
    return (ds4_mtp_indexed_attention_command_plan) {
        .n_events = 1u,
        .events = { DS4_MTP_INDEXED_ATTN_ATTEND_ORIGINAL_TOPK },
    };
}

#endif
