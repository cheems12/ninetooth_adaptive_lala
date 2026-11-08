#ifndef INDEX_COPY_NINETOOTHED_H
#define INDEX_COPY_NINETOOTHED_H

#include "../info.h"
#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/index_copy.h"
#include "../../../ninetoothed/utils.h"

#include <utility>
#include <vector>

namespace op::index_copy::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle, IndexCopyInfo info)
        : InfiniopDescriptor{handle->device, handle->device_id},
          info_{std::move(info)} {}

    size_t workspaceSize() const { return 0; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int64_t dim,
        infiniopTensorDescriptor_t index_desc,
        infiniopTensorDescriptor_t source_desc) {
        auto info = IndexCopyInfo::create(
            output_desc, input_desc, dim, index_desc, source_desc);
        CHECK_RESULT(info);
        CHECK_DTYPE(input_desc->dtype(),
                    INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(output_desc->isContiguous()
                            && source_desc->isContiguous()
                            && index_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);
        *desc_ptr = new Descriptor(handle, info.take());
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(void *, size_t,
                             void *output,
                             const void *,
                             const void *index,
                             const void *source,
                             void *stream) const {
        if (info_.outer_size() == 0 || info_.index_len() == 0
            || info_.inner_size() == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        const std::vector<Size> source_shape{
            info_.outer_size(), info_.index_len(), info_.inner_size()};
        const std::vector<Stride> source_strides{
            static_cast<Stride>(info_.index_len() * info_.inner_size()),
            static_cast<Stride>(info_.inner_size()), 1};
        const std::vector<Size> output_shape{
            info_.outer_size(), info_.dim_size(), info_.inner_size()};
        const std::vector<Stride> output_strides{
            static_cast<Stride>(info_.dim_size() * info_.inner_size()),
            static_cast<Stride>(info_.inner_size()), 1};
        const std::vector<Size> index_shape{info_.index_len()};
        const std::vector<Stride> index_strides{1};

        auto source_nt{::ninetoothed::Tensor{
            source, source_shape, source_strides}};
        auto index_nt{::ninetoothed::Tensor{
            index, index_shape, index_strides}};
        auto output_nt{::ninetoothed::Tensor{
            output, output_shape, output_strides}};

        if (launch_index_copy(stream, source_nt, index_nt, output_nt,
                              info_.dtype(), info_.idx_dtype(), 256)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;
    IndexCopyInfo info_;
};

} // namespace op::index_copy::ninetoothed

#endif // INDEX_COPY_NINETOOTHED_H
