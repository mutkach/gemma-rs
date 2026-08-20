#include <metal_stdlib>


kernel void simple_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(4)]],
    constant uint32_t &batch_size [[buffer(5)]],
    constant uint32_t &seq_len [[buffer(6)]],
    constant uint32_t &num_heads [[buffer(7)]],
    constant uint32_t &head_dim [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint32_t head_id = gid.y;
    uint32_t seq_idx = gid.x;





    if (head_id >= num_heads || seq_idx >= seq_len) return;

    // TODO: Implement attention computation

}

kernel void tiled_gqa_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* o_proj [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint32_t &batch_size [[buffer(5)]],
    constant uint32_t &seq_len [[buffer(6)]],
    constant uint32_t &num_heads [[buffer(7)]],
    constant uint32_t &num_kv_heads [[buffer(8)]],
    constant uint32_t &head_dim [[buffer(9)]],
    constant uint32_t &tile_size [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
) {

// TODO

}
