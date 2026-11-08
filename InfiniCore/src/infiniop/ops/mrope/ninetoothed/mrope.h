#ifndef MROPE_NINETOOTHED_H
#define MROPE_NINETOOTHED_H

#include "../info.h"
#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/mrope.h"
#include "../../../ninetoothed/utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace op::mrope::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle, MRoPEInfo info)
        : InfiniopDescriptor{handle->device, handle->device_id},
          info_{std::move(info)} {}

    size_t workspaceSize() const { return 0; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t q_out_desc,
        infiniopTensorDescriptor_t k_out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t cos_desc,
        infiniopTensorDescriptor_t sin_desc,
        infiniopTensorDescriptor_t positions_desc,
        int head_size,
        int rotary_dim,
        int section_t,
        int section_h,
        int section_w,
        bool interleaved) {
        auto info = MRoPEInfo::create(
            q_out_desc, k_out_desc, q_desc, k_desc, cos_desc, sin_desc,
            positions_desc, head_size, rotary_dim,
            section_t, section_h, section_w, interleaved);
        CHECK_RESULT(info);
        CHECK_DTYPE(q_desc->dtype(),
                    INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        *desc_ptr = new Descriptor(handle, info.take());
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *, size_t,
        void *q_out, void *k_out,
        const void *q, const void *k,
        const void *cos, const void *sin,
        const void *positions,
        void *stream) const {
        if (info_.num_tokens == 0) {
            return INFINI_STATUS_SUCCESS;
        }

        const std::vector<Size> cos_shape{
            info_.max_position_embeddings, info_.half_rotary_dim};
        const std::vector<Stride> cos_strides{
            info_.cos_stride_position, 1};
        const std::vector<Stride> sin_strides{
            info_.sin_stride_position, 1};
        const std::vector<Size> positions_shape{3, info_.num_tokens};
        const std::vector<Stride> positions_strides{
            info_.positions_has_axes ? info_.positions_stride_axis : 0,
            info_.positions_stride_token};

        auto cos_nt{::ninetoothed::Tensor{cos, cos_shape, cos_strides}};
        auto sin_nt{::ninetoothed::Tensor{sin, cos_shape, sin_strides}};
        auto positions_nt{::ninetoothed::Tensor{
            positions, positions_shape, positions_strides}};
        auto section_t_nt{::ninetoothed::Tensor<int64_t>{
            static_cast<int64_t>(info_.section_t)}};
        auto section_h_nt{::ninetoothed::Tensor<int64_t>{
            static_cast<int64_t>(info_.section_h)}};
        auto section_w_nt{::ninetoothed::Tensor<int64_t>{
            static_cast<int64_t>(info_.section_w)}};

        const auto status_q = launch_one(
            stream, q_out, q, info_.num_q_heads,
            info_.q_out_stride_token, info_.q_stride_token,
            cos_nt, sin_nt, positions_nt,
            section_t_nt, section_h_nt, section_w_nt);
        if (status_q != INFINI_STATUS_SUCCESS) {
            return status_q;
        }
        return launch_one(
            stream, k_out, k, info_.num_kv_heads,
            info_.k_out_stride_token, info_.k_stride_token,
            cos_nt, sin_nt, positions_nt,
            section_t_nt, section_h_nt, section_w_nt);
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    infiniStatus_t launch_one(
        void *stream,
        void *output,
        const void *input,
        size_t num_heads,
        ptrdiff_t output_stride_token,
        ptrdiff_t input_stride_token,
        ::ninetoothed::Tensor<> cos,
        ::ninetoothed::Tensor<> sin,
        ::ninetoothed::Tensor<> positions,
        ::ninetoothed::Tensor<int64_t> section_t,
        ::ninetoothed::Tensor<int64_t> section_h,
        ::ninetoothed::Tensor<int64_t> section_w) const {
        const std::vector<Size> shape{
            info_.num_tokens, num_heads, info_.head_size};
        const std::vector<Stride> input_strides{
            input_stride_token, static_cast<Stride>(info_.head_size), 1};
        const std::vector<Stride> output_strides{
            output_stride_token, static_cast<Stride>(info_.head_size), 1};
        auto input_nt{::ninetoothed::Tensor{input, shape, input_strides}};
        auto output_nt{::ninetoothed::Tensor{output, shape, output_strides}};
#if defined(MROPE_NINETOOTHED_DEBUG)
        fprintf(stderr,
                "[mrope] in=%p out=%p shape=(%llu,%llu,%llu) "
                "in_str=(%lld,%lld,%lld) out_str=(%lld,%lld,%lld) "
                "rotary=%llu half=%llu sec=(%llu,%llu,%llu) ilv=%d "
                "cos_str=%lld pos_str=(%lld,%lld) has_axes=%d\n",
                input, output,
                (unsigned long long)shape[0], (unsigned long long)shape[1],
                (unsigned long long)shape[2],
                (long long)input_strides[0], (long long)input_strides[1],
                (long long)input_strides[2],
                (long long)output_strides[0], (long long)output_strides[1],
                (long long)output_strides[2],
                (unsigned long long)info_.rotary_dim,
                (unsigned long long)info_.half_rotary_dim,
                (unsigned long long)info_.section_t,
                (unsigned long long)info_.section_h,
                (unsigned long long)info_.section_w,
                (int)info_.interleaved,
                (long long)info_.cos_stride_position,
                (long long)info_.positions_stride_axis,
                (long long)info_.positions_stride_token,
                (int)info_.positions_has_axes);
#endif
        if (launch_mrope(stream,
                         input_nt, cos, sin, positions, output_nt,
                         section_t, section_h, section_w,
                         info_.data_type, info_.position_type,
                         info_.interleaved, 128)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        return INFINI_STATUS_SUCCESS;
    }

    MRoPEInfo info_;
};

} // namespace op::mrope::ninetoothed

#endif // MROPE_NINETOOTHED_H
