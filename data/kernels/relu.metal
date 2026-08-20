#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void relu(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint32_t &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = max(T(0.0f), input[gid]);
}
