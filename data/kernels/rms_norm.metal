#include <metal_stdlib>

using namespace metal;

// one kernel per row

template <typename T>
kernel void rms_norm(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint32_t &rows [[buffer(2)]],
    constant uint32_t &cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    T sum = 0;
    if (gid >= rows) return;
    for (size_t i = 0; i < cols; i++) {
        uint32_t idx = gid * cols + i;
        sum += input[idx] * input[idx];
    }
    T mean_squared = sum / cols;
    for (size_t i = 0; i < cols; i++) {
        uint32_t idx = gid * cols + i;
        output[idx] = input[idx] / sqrt(mean_squared);
    }
}
