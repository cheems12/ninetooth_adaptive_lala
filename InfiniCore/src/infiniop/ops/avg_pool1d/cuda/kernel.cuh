#ifndef __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__
#define __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__

#include <type_traits>

template <typename T>
__device__ void avgPool1dKernel(
    T *y,
    const T *x,
    size_t batch,
    size_t channels,
    size_t in_width,
    size_t out_width,
    size_t kernel_size,
    size_t stride,
    size_t padding,

    ptrdiff_t y_stride_batch,
    ptrdiff_t y_stride_channel,
    ptrdiff_t y_stride_width,
    ptrdiff_t x_stride_batch,
    ptrdiff_t x_stride_channel,
    ptrdiff_t x_stride_width) {

    size_t total_elements = batch * channels * out_width;

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        size_t ow = idx % out_width;
        size_t temp = idx / out_width;
        size_t c = temp % channels;
        size_t b = temp / channels;

        size_t y_offset = b * y_stride_batch + c * y_stride_channel + ow * y_stride_width;

        long long start_w = static_cast<long long>(ow * stride) - padding;

        // Accumulate low precision inputs in FP32.  Besides improving the
        // reduction accuracy, this avoids relying on implicit integer
        // construction of DTK's gfx906 bfloat16 type.
        using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
        AccT sum = AccT{0};

        for (size_t k = 0; k < kernel_size; ++k) {
            long long iw = start_w + k;

            if (iw >= 0 && iw < static_cast<long long>(in_width)) {
                size_t x_offset = b * x_stride_batch + c * x_stride_channel + iw * x_stride_width;
                if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                    sum += __bfloat162float(x[x_offset]);
                } else if constexpr (std::is_same_v<T, half>) {
                    sum += __half2float(x[x_offset]);
                } else {
                    sum += static_cast<AccT>(x[x_offset]);
                }
            }
        }

        const AccT average = sum / static_cast<AccT>(kernel_size);
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            y[y_offset] = __float2bfloat16_rn(static_cast<float>(average));
        } else if constexpr (std::is_same_v<T, half>) {
            y[y_offset] = __float2half_rn(static_cast<float>(average));
        } else {
            y[y_offset] = static_cast<T>(average);
        }
    }
}

#endif
