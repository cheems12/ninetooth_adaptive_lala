#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gelutanh.h"

#ifdef ENABLE_CPU_API
#include "cpu/gelutanh_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#include "nvidia/gelutanh_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/gelutanh_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/gelutanh_moore.h"
#endif
#ifdef ENABLE_NINETOOTHED
#include "ninetoothed/gelutanh.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateGeluTanhDescriptor(
    infiniopHandle_t handle,
    infiniopGeluTanhDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::gelutanh::NAMESPACE::Descriptor::create(                     \
            handle,                                                             \
            reinterpret_cast<op::gelutanh::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                             \
            {x_desc})

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_QY_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_QY, ninetoothed);
#else
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#endif
#ifdef ENABLE_HYGON_API
        // The native Hygon elementwise implementation supports all official
        // GELU-Tanh dtypes and arbitrary-rank/strided tensors.  Keep using it
        // even when NineToothed is enabled for the other preliminary operators.
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_METAX, ninetoothed);
#else
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
#if defined(ENABLE_ALI_API) && defined(ENABLE_NINETOOTHED)
        CREATE(INFINI_DEVICE_ALI, ninetoothed);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetGeluTanhWorkspaceSize(infiniopGeluTanhDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                  \
        *size = reinterpret_cast<op::gelutanh::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_NVIDIA, ninetoothed)
#else
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_ILUVATAR, ninetoothed)
#else
        GET(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#endif
#ifdef ENABLE_QY_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_QY, ninetoothed)
#else
        GET(INFINI_DEVICE_QY, nvidia)
#endif
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_METAX, ninetoothed)
#else
        GET(INFINI_DEVICE_METAX, metax)
#endif
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore)
#endif
#if defined(ENABLE_ALI_API) && defined(ENABLE_NINETOOTHED)
        GET(INFINI_DEVICE_ALI, ninetoothed)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopGeluTanh(
    infiniopGeluTanhDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                     \
        return reinterpret_cast<const op::gelutanh::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x}, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_QY_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_QY, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_METAX, ninetoothed);
#else
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
#if defined(ENABLE_ALI_API) && defined(ENABLE_NINETOOTHED)
        CALCULATE(INFINI_DEVICE_ALI, ninetoothed);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyGeluTanhDescriptor(infiniopGeluTanhDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        delete reinterpret_cast<const op::gelutanh::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_NVIDIA, ninetoothed);
#else
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_ILUVATAR, ninetoothed);
#else
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_QY_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_QY, ninetoothed);
#else
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#endif
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_METAX, ninetoothed);
#else
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
#endif
#if defined(ENABLE_ALI_API) && defined(ENABLE_NINETOOTHED)
        DELETE(INFINI_DEVICE_ALI, ninetoothed);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
