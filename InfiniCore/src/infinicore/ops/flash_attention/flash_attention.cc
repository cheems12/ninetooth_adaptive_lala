#include "infinicore/ops/flash_attention.hpp"

#include "../../utils.hpp"

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ops/arange.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

namespace infinicore::op {

#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
namespace {

bool can_use_hygon_sdpa(const Tensor &out,
                        const Tensor &q,
                        const Tensor &k,
                        const Tensor &v,
                        const Tensor &total_kv_len) {
    return q->device().getType() == Device::Type::HYGON
        && out->ndim() == 4
        && q->ndim() == 4
        && k->ndim() == 4
        && v->ndim() == 4
        && total_kv_len->ndim() == 1
        && q->shape()[0] == k->shape()[0]
        && q->shape()[0] == v->shape()[0]
        && q->shape()[0] == total_kv_len->shape()[0]
        && q->shape()[1] == k->shape()[1]
        && q->shape()[1] == v->shape()[1]
        && q->shape()[3] == k->shape()[3]
        && k->shape()[2] == v->shape()[2];
}

void run_hygon_sdpa(Tensor out,
                    const Tensor &q,
                    const Tensor &k,
                    const Tensor &v,
                    const Tensor &total_kv_len,
                    float scale,
                    bool is_causal) {
    adaptor::set_aten_stream_to_infinicore();

    const auto out_at = adaptor::to_aten_tensor(out);
    const auto q_at = adaptor::to_aten_tensor(q);
    const auto k_at = adaptor::to_aten_tensor(k);
    const auto v_at = adaptor::to_aten_tensor(v);
    const auto lengths_at = adaptor::to_aten_tensor(total_kv_len);

    const int64_t query_len = q_at.size(-2);
    const int64_t key_len = k_at.size(-2);
    const auto key_positions = at::arange(key_len, lengths_at.options());

    // A boolean SDPA mask uses true for elements that participate in
    // attention.  Keep the batch-specific KV lengths on device so this path
    // does not introduce a device-to-host synchronization on every call.
    auto attention_mask = key_positions.unsqueeze(0)
                              .lt(lengths_at.reshape({-1, 1}))
                              .unsqueeze(1)
                              .unsqueeze(1);

    if (is_causal) {
        const auto query_positions = at::arange(query_len, lengths_at.options());
        const auto causal_mask = query_positions.unsqueeze(1)
                                     .ge(key_positions.unsqueeze(0))
                                     .unsqueeze(0)
                                     .unsqueeze(0);
        attention_mask = attention_mask.logical_and(causal_mask);
    }

    auto result = at::scaled_dot_product_attention(
        q_at,
        k_at,
        v_at,
        attention_mask,
        0.0,
        false,
        std::optional<double>(static_cast<double>(scale)));
    out_at.copy_(result);
}

} // namespace
#endif

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashAttention);

FlashAttention::FlashAttention(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k, v, total_kv_len, scale, is_causal);
}

void FlashAttention::execute(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FlashAttention, out, q, k, v, total_kv_len, scale, is_causal);
}

Tensor flash_attention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal) {
    Shape shape = q->shape();
    int idx = shape.size() - 1;
    shape[idx] = v->shape()[idx];
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    flash_attention_(out, q, k, v, total_kv_len, scale, is_causal);
    return out;
}

void flash_attention_(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &total_kv_len, float scale, bool is_causal) {
#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)
    // The hand-written NineToothed kernel is the portable fallback, but on
    // gfx906 its small square tiles launch many kernels and are considerably
    // slower than the container's tuned ATen/HIP SDPA implementation.
    // Preserve graph recording semantics by using the eager fast path only
    // outside a recorded graph.
    if (!context::isGraphRecording() && can_use_hygon_sdpa(out, q, k, v, total_kv_len)) {
        run_hygon_sdpa(out, q, k, v, total_kv_len, scale, is_causal);
        return;
    }
#endif
    FlashAttention::execute(out, q, k, v, total_kv_len, scale, is_causal);
}
} // namespace infinicore::op
