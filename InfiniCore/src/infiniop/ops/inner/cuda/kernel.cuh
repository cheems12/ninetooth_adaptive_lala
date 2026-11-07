#ifndef __INNER_KERNEL_CUH__
#define __INNER_KERNEL_CUH__

#include <type_traits>

template <size_t BLOCK_SIZE, typename T>
INFINIOP_CUDA_KERNEL innerKernel(
    const T *input, const T *other, T *out,
    size_t total_elements, size_t oper_len, size_t *out_shape,
    ptrdiff_t *input_strides, ptrdiff_t *other_strides, ptrdiff_t *out_strides,
    size_t input_ndim, size_t other_ndim, size_t out_ndim) {

    size_t out_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_index >= total_elements) {
        return;
    }

    size_t out_offset = device::nvidia::indexToOffset(out_index, out_ndim, out_shape, out_strides);

    size_t out_dim_pos = out_ndim - 1;
    size_t index = out_index;

    ptrdiff_t input_offset = 0;
    ptrdiff_t other_offset = 0;

    for (int i = (int)other_ndim - 2; i >= 0; i--) {
        other_offset += (index % out_shape[out_dim_pos]) * other_strides[i];
        index /= out_shape[out_dim_pos--];
    }
    for (int i = (int)input_ndim - 2; i >= 0; i--) {
        input_offset += (index % out_shape[out_dim_pos]) * input_strides[i];
        index /= out_shape[out_dim_pos--];
    }

    using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    AccT tmp = AccT{0};
    for (size_t i = 0; i < oper_len; i++) {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            tmp += __bfloat162float(input[input_offset])
                 * __bfloat162float(other[other_offset]);
        } else if constexpr (std::is_same_v<T, half>) {
            tmp += __half2float(input[input_offset])
                 * __half2float(other[other_offset]);
        } else {
            tmp += static_cast<AccT>(input[input_offset])
                 * static_cast<AccT>(other[other_offset]);
        }
        input_offset += input_strides[input_ndim - 1];
        other_offset += other_strides[other_ndim - 1];
    }

    if constexpr (std::is_same_v<T, cuda_bfloat16>) {
        out[out_offset] = __float2bfloat16_rn(static_cast<float>(tmp));
    } else if constexpr (std::is_same_v<T, half>) {
        out[out_offset] = __float2half_rn(static_cast<float>(tmp));
    } else {
        out[out_offset] = static_cast<T>(tmp);
    }
}

#endif // __INNER_KERNEL_CUH__
