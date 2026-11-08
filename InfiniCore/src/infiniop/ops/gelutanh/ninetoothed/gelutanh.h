#ifndef GELUTANH_NINETOOTHED_H
#define GELUTANH_NINETOOTHED_H

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/gelutanh.h"
#include "../../../ninetoothed/utils.h"

#include <vector>

namespace op::gelutanh::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle,
               infiniopTensorDescriptor_t output_desc,
               infiniopTensorDescriptor_t input_desc)
        : InfiniopDescriptor{handle->device, handle->device_id},
          output_shape_{output_desc->shape()},
          output_strides_{output_desc->strides()},
          input_shape_{input_desc->shape()},
          input_strides_{input_desc->strides()},
          dtype_{output_desc->dtype()} {}

    size_t workspaceSize() const { return 0; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        std::vector<infiniopTensorDescriptor_t> input_descs) {
        CHECK_OR_RETURN(handle != nullptr && desc_ptr != nullptr
                            && output_desc != nullptr && input_descs.size() == 1
                            && input_descs[0] != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        const auto input_desc = input_descs[0];
        CHECK_OR_RETURN(output_desc->shape() == input_desc->shape(),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->ndim() >= 1 && output_desc->ndim() <= 5,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dtype() == input_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(output_desc->dtype(),
                    INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32,
                    INFINI_DTYPE_F64);
        CHECK_OR_RETURN(!output_desc->hasBroadcastDim(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        *desc_ptr = new Descriptor(handle, output_desc, input_desc);
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(void *, size_t,
                             void *output,
                             std::vector<const void *> inputs,
                             void *stream) const {
        if (inputs.size() != 1) {
            return INFINI_STATUS_BAD_PARAM;
        }
        auto input_nt{::ninetoothed::Tensor{
            inputs[0], input_shape_, input_strides_}};
        auto output_nt{::ninetoothed::Tensor{
            output, output_shape_, output_strides_}};
        if (launch_gelutanh(stream, input_nt, output_nt,
                            output_shape_.size(), dtype_, 256)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;
    std::vector<Size> output_shape_;
    std::vector<Stride> output_strides_;
    std::vector<Size> input_shape_;
    std::vector<Stride> input_strides_;
    infiniDtype_t dtype_;
};

} // namespace op::gelutanh::ninetoothed

#endif // GELUTANH_NINETOOTHED_H
