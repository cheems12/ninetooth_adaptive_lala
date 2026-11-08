#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/timestep_embedding.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)
#include "nvidia/timestep_embedding_nvidia.cuh"
#endif
#ifdef ENABLE_NINETOOTHED
#include "ninetoothed/timestep_embedding.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateTimestepEmbeddingDescriptor(
    infiniopHandle_t handle,
    infiniopTimestepEmbeddingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t timestep_desc) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::timestep_embedding::NAMESPACE::Descriptor::create(           \
            handle,                                                             \
            reinterpret_cast<op::timestep_embedding::NAMESPACE::Descriptor **>( \
                desc_ptr),                                                      \
            output_desc,                                                        \
            timestep_desc)

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_HYGON_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_HYGON, ninetoothed);
#else
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_METAX_API)
        CREATE(INFINI_DEVICE_METAX, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_ILUVATAR_API)
        CREATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_QY_API)
        CREATE(INFINI_DEVICE_QY, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_ALI_API)
        CREATE(INFINI_DEVICE_ALI, ninetoothed);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopTimestepEmbedding(
    infiniopTimestepEmbeddingDescriptor_t desc,
    void *output,
    const void *timestep,
    float max_period,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                              \
    case CASE:                                                                  \
        return reinterpret_cast<                                                \
                   const op::timestep_embedding::NAMESPACE::Descriptor *>(desc) \
            ->calculate(output, timestep, max_period, stream)

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_HYGON_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_HYGON, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_METAX_API)
        CALCULATE(INFINI_DEVICE_METAX, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_ILUVATAR_API)
        CALCULATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_QY_API)
        CALCULATE(INFINI_DEVICE_QY, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_ALI_API)
        CALCULATE(INFINI_DEVICE_ALI, ninetoothed);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyTimestepEmbeddingDescriptor(
    infiniopTimestepEmbeddingDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                          \
    case CASE:                                                            \
        delete reinterpret_cast<                                          \
            const op::timestep_embedding::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        DESTROY(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_HYGON_API
#ifdef ENABLE_NINETOOTHED
        DESTROY(INFINI_DEVICE_HYGON, ninetoothed);
#else
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_METAX_API)
        DESTROY(INFINI_DEVICE_METAX, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_ILUVATAR_API)
        DESTROY(INFINI_DEVICE_ILUVATAR, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_QY_API)
        DESTROY(INFINI_DEVICE_QY, ninetoothed);
#endif
#if defined(ENABLE_NINETOOTHED) && defined(ENABLE_ALI_API)
        DESTROY(INFINI_DEVICE_ALI, ninetoothed);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
