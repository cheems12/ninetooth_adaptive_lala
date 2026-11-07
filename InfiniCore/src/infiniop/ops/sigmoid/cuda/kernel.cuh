#ifndef __SIDMOID_CUDA_H__
#define __SIDMOID_CUDA_H__

namespace op::sigmoid::cuda {
typedef struct SigmoidOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        // sigmoid(x) = 1 / (1 + exp(-x))
        if constexpr (std::is_same_v<T, half2>) {
#if defined(ENABLE_HYGON_API)
            const float2 xf = __half22float2(x);
            return __floats2half2_rn(
                1.0f / (1.0f + expf(-xf.x)),
                1.0f / (1.0f + expf(-xf.y)));
#else
            half2 denominator = __hadd2(make_half2(1, 1), h2exp(__hneg2(x)));
            return h2rcp(denominator);
#endif
        } else if constexpr (std::is_same_v<T, half>) {
#if defined(ENABLE_HYGON_API)
            const float xf = __half2float(x);
            return __float2half_rn(1.0f / (1.0f + expf(-xf)));
#else
            half denominator = __hadd(__float2half(1.0f), hexp(__hneg(x)));
            return hrcp(denominator);
#endif
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
#if defined(ENABLE_HYGON_API)
            const float xf = __bfloat162float(x);
            return __float2bfloat16(1.0f / (1.0f + expf(-xf)));
#else
            cuda_bfloat16 denominator = __float2bfloat16(__fadd_rn(1.0f, __expf(__bfloat162float(-x))));
            return __float2bfloat16(1.0f) / denominator;
#endif
        } else if constexpr (std::is_same_v<T, float>) {
            if (x >= 0.0f) {
                float z = expf(-x);
                return 1.0f / (1.0f + z);
            } else {
                float z = expf(x);
                return z / (1.0f + z);
            }
        } else { // double
            return 1.0 / (1.0 + exp(-x));
        }
    }
} SigmoidOp;
} // namespace op::sigmoid::cuda

#endif // __SIDMOID_CUDA_H__
