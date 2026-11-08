#ifndef UPSAMPLE_BILINEAR_NINETOOTHED_H
#define UPSAMPLE_BILINEAR_NINETOOTHED_H

#include "../info.h"
#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/upsample_bilinear.h"
#include "../../../ninetoothed/utils.h"

#include <utility>
#include <vector>

namespace op::upsample_bilinear::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle, UpsampleBilinearInfo info)
        : InfiniopDescriptor{handle->device, handle->device_id},
          info_{std::move(info)} {}

    size_t workspaceSize() const { return 0; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int align_corners) {
        auto info = UpsampleBilinearInfo::create(
            output_desc, input_desc, align_corners);
        CHECK_RESULT(info);
        CHECK_DTYPE(input_desc->dtype(),
                    INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(!output_desc->hasBroadcastDim(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);
        *desc_ptr = new Descriptor(handle, info.take());
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *, size_t,
        void *output,
        const void *input,
        void *stream) const {
        const std::vector<Size> input_shape{
            info_.n(), info_.c(), info_.h_in(), info_.w_in()};
        const std::vector<Size> output_shape{
            info_.n(), info_.c(), info_.h_out(), info_.w_out()};
        std::vector<Stride> input_strides(4);
        std::vector<Stride> output_strides(4);
        for (size_t i = 0; i < 4; ++i) {
            input_strides[i] = info_.input_stride(i);
            output_strides[i] = info_.output_stride(i);
        }
        auto input_nt{::ninetoothed::Tensor{
            input, input_shape, input_strides}};
        auto output_nt{::ninetoothed::Tensor{
            output, output_shape, output_strides}};
        if (launch_upsample_bilinear(
                stream, input_nt, output_nt,
                info_.dtype(), info_.align_corners(), 256)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;
    UpsampleBilinearInfo info_;
};

} // namespace op::upsample_bilinear::ninetoothed

#endif // UPSAMPLE_BILINEAR_NINETOOTHED_H
