#ifndef TIMESTEP_EMBEDDING_NINETOOTHED_H
#define TIMESTEP_EMBEDDING_NINETOOTHED_H

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/timestep_embedding.h"
#include "../../../ninetoothed/utils.h"

#include <cmath>
#include <vector>

namespace op::timestep_embedding::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle,
               size_t num_timesteps,
               size_t embedding_dim,
               infiniDtype_t input_dtype)
        : InfiniopDescriptor{handle->device, handle->device_id},
          num_timesteps_{num_timesteps},
          embedding_dim_{embedding_dim},
          input_dtype_{input_dtype} {}

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t timestep_desc) {
        CHECK_OR_RETURN(timestep_desc->shape().size() == 1,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->shape().size() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->shape()[0] == timestep_desc->shape()[0]
                            && output_desc->shape()[1] > 0
                            && output_desc->shape()[1] % 2 == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_F32,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(timestep_desc->dtype(),
                    INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(output_desc->isContiguous()
                            && timestep_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        *desc_ptr = new Descriptor(handle,
                                   timestep_desc->shape()[0],
                                   output_desc->shape()[1],
                                   timestep_desc->dtype());
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(void *output,
                             const void *timestep,
                             float max_period,
                             void *stream) const {
        if (max_period <= 0.0f) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (num_timesteps_ == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        const std::vector<Size> shape{num_timesteps_, embedding_dim_};
        const std::vector<Stride> output_strides{
            static_cast<Stride>(embedding_dim_), 1};
        const std::vector<Stride> timestep_strides{1, 0};

        auto output_nt{::ninetoothed::Tensor{
            output, shape, output_strides}};
        auto timestep_nt{::ninetoothed::Tensor{
            timestep, shape, timestep_strides}};
        auto log_max_period_nt{
            ::ninetoothed::Tensor<float>{std::log(max_period)}};

        if (launch_timestep_embedding(stream,
                                      output_nt,
                                      timestep_nt,
                                      log_max_period_nt,
                                      input_dtype_,
                                      256)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    size_t num_timesteps_;
    size_t embedding_dim_;
    infiniDtype_t input_dtype_;
};

} // namespace op::timestep_embedding::ninetoothed

#endif // TIMESTEP_EMBEDDING_NINETOOTHED_H
