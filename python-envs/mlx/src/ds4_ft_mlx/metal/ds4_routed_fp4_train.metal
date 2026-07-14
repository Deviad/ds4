// Story 13.3b-5f package-local training-only packed-FP4 primitive.
// Not part of root production Metal inference kernels.

// DS4_ROUTED_FP4_TRAIN_COMMON_BEGIN
inline float ds4_fp4_lut(uint code) {
    switch (code & 0x0f) {
        case 0u: return 0.0f;
        case 1u: return 0.5f;
        case 2u: return 1.0f;
        case 3u: return 1.5f;
        case 4u: return 2.0f;
        case 5u: return 3.0f;
        case 6u: return 4.0f;
        case 7u: return 6.0f;
        case 8u: return -0.0f;
        case 9u: return -0.5f;
        case 10u: return -1.0f;
        case 11u: return -1.5f;
        case 12u: return -2.0f;
        case 13u: return -3.0f;
        case 14u: return -4.0f;
        default: return -6.0f;
    }
}

inline float ds4_sigmoid(float x) {
    return 1.0f / (1.0f + metal::precise::exp(-x));
}

inline float ds4_fp4_weight(
    const device uint8_t* packed,
    const device bfloat16_t* scale,
    const constant int64_t* packed_strides,
    const constant int64_t* scale_strides,
    uint row,
    uint logical_col
) {
    const uint byte_col = logical_col >> 1;
    const uint scale_col = logical_col >> 5;
    const uint8_t byte = packed[row * packed_strides[0] + byte_col * packed_strides[1]];
    const uint code = (logical_col & 1u) == 0u ? (uint(byte) & 0x0fu) : ((uint(byte) >> 4) & 0x0fu);
    return ds4_fp4_lut(code) * float(scale[row * scale_strides[0] + scale_col * scale_strides[1]]);
}
// DS4_ROUTED_FP4_TRAIN_COMMON_END

// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN ds4_fp4_pair_swiglu_forward
const uint lane = thread_position_in_threadgroup.x & 31u;
const uint simd_row = thread_position_in_threadgroup.x >> 5;
const uint r = threadgroup_position_in_grid.y * 8u + simd_row;
const uint out_base = threadgroup_position_in_grid.z * 8u;
threadgroup float w1_tile[8][32];
threadgroup float w3_tile[8][32];
float u1[8];
float u3[8];
for (uint j = 0; j < 8u; ++j) { u1[j] = 0.0f; u3[j] = 0.0f; }
for (uint k0 = 0; k0 < uint(hidden_size); k0 += 32u) {
    if (simd_row == 0u) {
        const uint k = k0 + lane;
        for (uint j = 0; j < 8u; ++j) {
            const uint i = out_base + j;
            const bool in_bounds = i < uint(intermediate_size) && k < uint(hidden_size);
            w1_tile[j][lane] = in_bounds ? ds4_fp4_weight(w1, s1, w1_strides, s1_strides, i, k) : 0.0f;
            w3_tile[j][lane] = in_bounds ? ds4_fp4_weight(w3, s3, w3_strides, s3_strides, i, k) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint token = r < uint(rows_shape[0]) ? uint(rows[r * rows_strides[0]]) : 0u;
    const uint k = k0 + lane;
    const float xv = (r < uint(rows_shape[0]) && k < uint(hidden_size))
        ? float(x_flat[token * x_flat_strides[0] + k * x_flat_strides[1]])
        : 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        u1[j] = metal::fma(xv, w1_tile[j][lane], u1[j]);
        u3[j] = metal::fma(xv, w3_tile[j][lane], u3[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint j = 0u; j < 8u; ++j) {
    u1[j] = simd_sum(u1[j]);
    u3[j] = simd_sum(u3[j]);
}
if (r < uint(rows_shape[0]) && lane == 0u) {
    const float lim = float(limit[0]);
    for (uint j = 0u; j < 8u; ++j) {
        const uint i = out_base + j;
        if (i < uint(intermediate_size)) {
            const float gate = metal::min(u1[j], lim);
            const float up = metal::min(metal::max(u3[j], -lim), lim);
            hidden[r * uint(intermediate_size) + i] = gate * ds4_sigmoid(gate) * up;
        }
    }
}
// DS4_ROUTED_FP4_TRAIN_KERNEL_END ds4_fp4_pair_swiglu_forward

// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN ds4_fp4_down_forward
const uint lane = thread_position_in_threadgroup.x & 31u;
const uint simd_row = thread_position_in_threadgroup.x >> 5;
const uint r = threadgroup_position_in_grid.y * 8u + simd_row;
const uint out_base = threadgroup_position_in_grid.z * 8u;
threadgroup float w_tile[8][32];
float acc[8];
for (uint j = 0; j < 8u; ++j) { acc[j] = 0.0f; }
for (uint k0 = 0; k0 < uint(intermediate_size); k0 += 32u) {
    if (simd_row == 0u) {
        const uint i = k0 + lane;
        for (uint j = 0; j < 8u; ++j) {
            const uint h = out_base + j;
            w_tile[j][lane] = (h < uint(hidden_size) && i < uint(intermediate_size))
                ? ds4_fp4_weight(w2, s2, w2_strides, s2_strides, h, i)
                : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = k0 + lane;
    const float hv = (r < uint(hidden_shape[0]) && i < uint(intermediate_size))
        ? float(hidden[r * hidden_strides[0] + i * hidden_strides[1]])
        : 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = metal::fma(hv, w_tile[j][lane], acc[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint j = 0u; j < 8u; ++j) { acc[j] = simd_sum(acc[j]); }
if (r < uint(hidden_shape[0]) && lane == 0u) {
    for (uint j = 0u; j < 8u; ++j) {
        const uint h = out_base + j;
        if (h < uint(hidden_size)) { y[r * uint(hidden_size) + h] = acc[j]; }
    }
}
// DS4_ROUTED_FP4_TRAIN_KERNEL_END ds4_fp4_down_forward

// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN ds4_fp4_down_input_vjp
const uint lane = thread_position_in_threadgroup.x & 31u;
const uint simd_row = thread_position_in_threadgroup.x >> 5;
const uint r = threadgroup_position_in_grid.y * 8u + simd_row;
const uint out_base = threadgroup_position_in_grid.z * 8u;
threadgroup float w_tile[8][32];
float acc[8];
for (uint j = 0; j < 8u; ++j) { acc[j] = 0.0f; }
for (uint k0 = 0; k0 < uint(hidden_size); k0 += 32u) {
    if (simd_row == 0u) {
        const uint h = k0 + lane;
        for (uint j = 0; j < 8u; ++j) {
            const uint i = out_base + j;
            w_tile[j][lane] = (i < uint(intermediate_size) && h < uint(hidden_size))
                ? ds4_fp4_weight(w2, s2, w2_strides, s2_strides, h, i)
                : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint token = r < uint(rows_shape[0]) ? uint(rows[r * rows_strides[0]]) : 0u;
    const uint h = k0 + lane;
    const float gv = (r < uint(rows_shape[0]) && h < uint(hidden_size))
        ? float(g_flat[token * g_flat_strides[0] + h * g_flat_strides[1]])
        : 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = metal::fma(gv, w_tile[j][lane], acc[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint j = 0u; j < 8u; ++j) { acc[j] = simd_sum(acc[j]); }
if (r < uint(rows_shape[0]) && lane == 0u) {
    for (uint j = 0u; j < 8u; ++j) {
        const uint i = out_base + j;
        if (i < uint(intermediate_size)) { gu[r * uint(intermediate_size) + i] = acc[j]; }
    }
}
// DS4_ROUTED_FP4_TRAIN_KERNEL_END ds4_fp4_down_input_vjp

// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN ds4_fp4_pair_swiglu_vjp_terms
const uint lane = thread_position_in_threadgroup.x & 31u;
const uint simd_row = thread_position_in_threadgroup.x >> 5;
const uint r = threadgroup_position_in_grid.y * 8u + simd_row;
const uint out_base = threadgroup_position_in_grid.z * 8u;
threadgroup float w1_tile[8][32];
threadgroup float w3_tile[8][32];
float u1[8];
float u3[8];
for (uint j = 0; j < 8u; ++j) { u1[j] = 0.0f; u3[j] = 0.0f; }
for (uint k0 = 0; k0 < uint(hidden_size); k0 += 32u) {
    if (simd_row == 0u) {
        const uint k = k0 + lane;
        for (uint j = 0; j < 8u; ++j) {
            const uint i = out_base + j;
            const bool in_bounds = i < uint(intermediate_size) && k < uint(hidden_size);
            w1_tile[j][lane] = in_bounds ? ds4_fp4_weight(w1, s1, w1_strides, s1_strides, i, k) : 0.0f;
            w3_tile[j][lane] = in_bounds ? ds4_fp4_weight(w3, s3, w3_strides, s3_strides, i, k) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint token = r < uint(rows_shape[0]) ? uint(rows[r * rows_strides[0]]) : 0u;
    const uint k = k0 + lane;
    const float xv = (r < uint(rows_shape[0]) && k < uint(hidden_size))
        ? float(x_flat[token * x_flat_strides[0] + k * x_flat_strides[1]])
        : 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        u1[j] = metal::fma(xv, w1_tile[j][lane], u1[j]);
        u3[j] = metal::fma(xv, w3_tile[j][lane], u3[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint j = 0u; j < 8u; ++j) {
    u1[j] = simd_sum(u1[j]);
    u3[j] = simd_sum(u3[j]);
}
float a_sum = 0.0f;
if (r < uint(rows_shape[0]) && lane == 0u) {
    for (uint j = 0u; j < 8u; ++j) {
        const uint i = out_base + j;
        if (i < uint(intermediate_size)) {
            const float lim = float(limit[0]);
            const float gate = metal::min(u1[j], lim);
            const float up = metal::min(metal::max(u3[j], -lim), lim);
            const float sig = ds4_sigmoid(gate);
            const float silu = gate * sig;
            const float hidden_v = silu * up;
            const float gu_v = float(gu[r * gu_strides[0] + i * gu_strides[1]]);
            a_sum += hidden_v * gu_v;
            const float dhidden = float(f[r * f_strides[0]]) * gu_v;
            const float mask_up = (u3[j] > -lim && u3[j] < lim) ? 1.0f : 0.0f;
            const float mask_gate = (u1[j] < lim) ? 1.0f : 0.0f;
            dup[r * uint(intermediate_size) + i] = dhidden * silu * mask_up;
            dgate[r * uint(intermediate_size) + i] = dhidden * up * (sig + gate * sig * (1.0f - sig)) * mask_gate;
        }
    }
    if (out_base < uint(intermediate_size)) {
        a_partial[r * uint(a_tiles) + threadgroup_position_in_grid.z] = a_sum;
    }
}
// DS4_ROUTED_FP4_TRAIN_KERNEL_END ds4_fp4_pair_swiglu_vjp_terms

// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN ds4_fp4_pair_input_vjp
const uint lane = thread_position_in_threadgroup.x & 31u;
const uint simd_row = thread_position_in_threadgroup.x >> 5;
const uint r = threadgroup_position_in_grid.y * 8u + simd_row;
const uint out_base = threadgroup_position_in_grid.z * 8u;
threadgroup float w1_tile[8][32];
threadgroup float w3_tile[8][32];
float acc[8];
for (uint j = 0; j < 8u; ++j) { acc[j] = 0.0f; }
for (uint k0 = 0; k0 < uint(intermediate_size); k0 += 32u) {
    if (simd_row == 0u) {
        const uint i = k0 + lane;
        for (uint j = 0; j < 8u; ++j) {
            const uint h = out_base + j;
            const bool in_bounds = i < uint(intermediate_size) && h < uint(hidden_size);
            w1_tile[j][lane] = in_bounds ? ds4_fp4_weight(w1, s1, w1_strides, s1_strides, i, h) : 0.0f;
            w3_tile[j][lane] = in_bounds ? ds4_fp4_weight(w3, s3, w3_strides, s3_strides, i, h) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = k0 + lane;
    const float dg = (r < uint(dgate_shape[0]) && i < uint(intermediate_size))
        ? float(dgate[r * dgate_strides[0] + i * dgate_strides[1]])
        : 0.0f;
    const float du = (r < uint(dup_shape[0]) && i < uint(intermediate_size))
        ? float(dup[r * dup_strides[0] + i * dup_strides[1]])
        : 0.0f;
    for (uint j = 0u; j < 8u; ++j) {
        acc[j] = metal::fma(dg, w1_tile[j][lane], acc[j]);
        acc[j] = metal::fma(du, w3_tile[j][lane], acc[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint j = 0u; j < 8u; ++j) { acc[j] = simd_sum(acc[j]); }
if (r < uint(dgate_shape[0]) && lane == 0u) {
    for (uint j = 0u; j < 8u; ++j) {
        const uint h = out_base + j;
        if (h < uint(hidden_size)) { dx[r * uint(hidden_size) + h] = acc[j]; }
    }
}
// DS4_ROUTED_FP4_TRAIN_KERNEL_END ds4_fp4_pair_input_vjp

// DS4_ROUTED_FP4_TRAIN_KERNEL_BEGIN ds4_fp4_reduce_a
const uint row = threadgroup_position_in_grid.x;
const uint lane = thread_position_in_threadgroup.x;
threadgroup float partial[8];
float acc = 0.0f;
if (row < uint(a_partial_shape[0])) {
    for (uint c = lane; c < uint(a_partial_shape[1]); c += 256u) {
        acc += float(a_partial[row * a_partial_strides[0] + c * a_partial_strides[1]]);
    }
}
float sg = simd_sum(acc);
if ((lane & 31u) == 0u) { partial[lane >> 5] = sg; }
threadgroup_barrier(mem_flags::mem_threadgroup);
float total_in = lane < 8u ? partial[lane] : 0.0f;
float total = simd_sum(total_in);
if (lane == 0u && row < uint(a_partial_shape[0])) { a[row] = total; }
// DS4_ROUTED_FP4_TRAIN_KERNEL_END ds4_fp4_reduce_a
