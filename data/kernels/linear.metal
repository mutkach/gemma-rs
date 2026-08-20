kernel void linear_bf16(
    device const bfloat16* input [[buffer(0)]],
    device const bfloat16* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint32_t &in_dim[[buffer(3)]],
    constant uint32_t &out_dim[[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {

    uint32_t idx = gid;
    if (idx >= out_dim) return;

    bfloat16 sum = 0.0f;
    for (uint32_t i = 0; i < in_dim; i++) {
        sum += input[i] * weights[idx * in_dim + i];
    }
    output[idx] = sum;
}

kernel void linear_tiled_bf16(
    device const bfloat16* input [[buffer(0)]],
    device const bfloat16* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint32_t &in_dim[[buffer(3)]],
    constant uint32_t &out_dim[[buffer(4)]],
    uint gid [[thread_position_in_grid]]

) {

    uint32_t idx = gid;
    uint32_t offset = gid % 32;
    if (idx >= out_dim) return;

    bfloat16 sum = 0.0f;
    for (uint32_t i = offset; i < in_dim; i += 32) {
        sum += input[i] * weights[idx * in_dim + i];
    }
    output[idx] = sum;
}
