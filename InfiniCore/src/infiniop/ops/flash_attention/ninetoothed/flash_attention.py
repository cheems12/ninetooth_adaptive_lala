import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()


class CausalVariant(enum.IntEnum):
    """Please refer to `<https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.bias.CausalVariant.html>`_."""

    UPPER_LEFT = enum.auto()

    LOWER_RIGHT = enum.auto()


def arrangement(
    query,
    key,
    value,
    total_kv_len,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
    with_kv_cache,
    block_size_m=None,
    block_size_n=None,
):
    def arrange_query_or_output(input):
        arranged = input.tile((1, 1, block_size_m, -1)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1)
        )
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    def arrange_key_or_value(input):
        arranged = (
            input.tile((1, 1, block_size_n, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, query_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    def arrange_total_kv_len(input, shape):
        arranged = input.tile((1,))
        arranged = arranged.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(shape)
        return arranged

    def arrange_present_key_or_present_value(input):
        arranged = input.tile((1, 1, block_size_m, block_size_n))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_attn_mask(input):
        arranged = input.tile((1, 1, block_size_m, block_size_n)).tile((1, 1, 1, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 2))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    query_arranged = arrange_query_or_output(query)
    key_arranged = arrange_key_or_value(key)
    value_arranged = arrange_key_or_value(value)
    total_kv_len_arranged = arrange_total_kv_len(total_kv_len, query_arranged.shape)
    attn_mask_arranged = arrange_attn_mask(attn_mask)
    is_causal_arranged = is_causal
    scale_arranged = scale
    output_arranged = arrange_query_or_output(output)
    with_attn_mask_arranged = with_attn_mask
    causal_variant_arranged = causal_variant

    if with_kv_cache:
        present_key_arranged = arrange_present_key_or_present_value(present_key)
        present_value_arranged = arrange_present_key_or_present_value(present_value)
        present_key_slot_arranged = arrange_present_key_or_present_value(
            present_key_slot
        )
        present_value_slot_arranged = arrange_present_key_or_present_value(
            present_value_slot
        )
        return (
            query_arranged,
            key_arranged,
            value_arranged,
            total_kv_len_arranged,
            present_key_arranged,
            present_value_arranged,
            present_key_slot_arranged,
            present_value_slot_arranged,
            attn_mask_arranged,
            is_causal_arranged,
            scale_arranged,
            output_arranged,
            with_attn_mask_arranged,
            causal_variant_arranged,
        )

    return (
        query_arranged,
        key_arranged,
        value_arranged,
        total_kv_len_arranged,
        attn_mask_arranged,
        is_causal_arranged,
        scale_arranged,
        output_arranged,
        with_attn_mask_arranged,
        causal_variant_arranged,
    )


def arrangement_without_kv_cache(
    query,
    key,
    value,
    total_kv_len,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
    block_size_m=None,
    block_size_n=None,
):
    return arrangement(
        query,
        key,
        value,
        total_kv_len,
        None,
        None,
        None,
        None,
        attn_mask,
        is_causal,
        scale,
        output,
        with_attn_mask,
        causal_variant,
        False,
        block_size_m,
        block_size_n,
    )


def application_with_kv_cache(
    query,
    key,
    value,
    total_kv_len,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    present_key_slot = present_key  # noqa: F841
    present_value_slot = present_value  # noqa: F841

    application_without_kv_cache(
        query,
        key,
        value,
        total_kv_len,
        attn_mask,
        is_causal,
        scale,
        output,
        with_attn_mask,
        causal_variant,
    )


def application_without_kv_cache(
    query,
    key,
    value,
    total_kv_len,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    actual_kv_len = total_kv_len[0]

    for i in range(query.shape[0]):
        # DTK 24.04 ships Triton 2.1, whose language module does not expose
        # exp2.  Keep the softmax in natural-log space so the same kernel can
        # be generated by both the older DTK Triton and newer Triton releases.
        query_i = (scale * query[i]).to(query[i].dtype)

        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        max = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)

        # DTK 24.04's Triton 2.1 miscompiles a Python range whose upper bound
        # comes from a value loaded from device memory (the loop becomes
        # empty). Iterate over the allocated KV tiles and retain the existing
        # actual_kv_len mask below for the logical sequence length.
        for j in range(key.dtype.shape[0]):
            # DTK Triton enables reduced-precision FP32 dot products by
            # default.  Disable that mode so the public FP32 tolerance is met;
            # this flag is ignored for FP16/BF16 operands.
            qk = ntl.dot(query_i, ntl.trans(key[j]), allow_tf32=False)

            key_pos = key[j].offsets(-2)
            qk = ntl.where(key_pos < actual_kv_len, qk, float("-inf"))

            if with_attn_mask:
                qk += attn_mask[j]

            if is_causal:
                query_pos = query[i].offsets(-2)
                mask = query_pos[:, None] >= key_pos[None, :]

                if causal_variant == 2:  # CausalVariant.LOWER_RIGHT:
                    mask = (
                        query_pos[:, None] + actual_kv_len - query.source.shape[-2]
                        >= key_pos[None, :]
                    )

                qk = ntl.where(mask, qk, float("-inf"))

            next_max = ntl.maximum(max, ntl.max(qk, 1))
            stable_qk = ntl.exp(qk - next_max[:, None])

            alpha = ntl.exp(max - next_max)
            acc = acc * alpha[:, None] + ntl.dot(
                stable_qk.to(value[i].dtype), value[j], allow_tf32=False
            )
            max = next_max
            lse = lse * alpha + ntl.sum(stable_qk, 1)

        acc /= lse[:, None]
        output[i] = acc  # noqa: F841


def application_without_kv_cache_low_precision_dot(
    query,
    key,
    value,
    total_kv_len,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    """Use FP16 dot operands for storage types unsupported by gfx906 MFMA.

    DTK 24.04 cannot lower native BF16 ``tl.dot`` on gfx906.  Its full-precision
    FP32 dot path also produces incorrect results for head dimensions 16 and
    32.  Conversion at the dot boundary retains FP32 softmax and accumulation
    while using the accelerator's validated FP16 matrix path.
    """
    actual_kv_len = total_kv_len[0]

    for i in range(query.shape[0]):
        query_i = (scale * query[i]).to(ntl.float16)

        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        max = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)

        for j in range(key.dtype.shape[0]):
            key_i = key[j].to(ntl.float16)
            qk = ntl.dot(query_i, ntl.trans(key_i))

            key_pos = key[j].offsets(-2)
            qk = ntl.where(key_pos < actual_kv_len, qk, float("-inf"))

            if with_attn_mask:
                qk += attn_mask[j]

            if is_causal:
                query_pos = query[i].offsets(-2)
                mask = query_pos[:, None] >= key_pos[None, :]

                if causal_variant == 2:  # CausalVariant.LOWER_RIGHT:
                    mask = (
                        query_pos[:, None] + actual_kv_len - query.source.shape[-2]
                        >= key_pos[None, :]
                    )

                qk = ntl.where(mask, qk, float("-inf"))

            next_max = ntl.maximum(max, ntl.max(qk, 1))
            stable_qk = ntl.exp(qk - next_max[:, None])

            alpha = ntl.exp(max - next_max)
            value_i = value[j].to(ntl.float16)
            acc = acc * alpha[:, None] + ntl.dot(
                stable_qk.to(ntl.float16), value_i
            )
            max = next_max
            lse = lse * alpha + ntl.sum(stable_qk, 1)

        acc /= lse[:, None]
        output[i] = acc  # noqa: F841


def premake(
    with_kv_cache,
    emb_dim=None,
    is_causal=None,
    with_attn_mask=None,
    causal_variant=None,
    dtype=None,
    block_size=None,
    block_size_m=None,
    block_size_n=None,
):
    # A single square-tile selector avoids a Cartesian product of unsupported
    # M/N combinations while still allowing the descriptor to choose a tile
    # that fits the target's shared-memory limit.
    if block_size is not None:
        block_size_m = block_size
        block_size_n = block_size
    if with_kv_cache:
        arrangement_ = functools.partial(
            arrangement,
            with_kv_cache=True,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
        )
    else:
        arrangement_ = functools.partial(
            arrangement_without_kv_cache,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
        )

    # Every generated tile is padded to BLOCK_SIZE_M/BLOCK_SIZE_N.  Supplying
    # zero for masked loads is required on DTK: an undefined value outside the
    # logical KV length can otherwise enter the value-side dot product as
    # ``0 * NaN`` and poison the whole output row.
    query, key, value, attn_mask = (
        Tensor(
            4,
            dtype=dtype,
            other=0.0,
            shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128}),
        )
        for _ in range(4)
    )
    output = Tensor(
        4,
        dtype=dtype,
        shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128}),
    )
    # The public InfiniCore test/API supplies total_kv_len as int64.  Matching
    # that ABI is also required for batches larger than one: treating the
    # buffer as int32 would use a four-byte stride through an eight-byte array.
    total_kv_len = Tensor(1, dtype=ninetoothed.int64)
    present_key, present_value, present_key_slot, present_value_slot = (
        Tensor(4, dtype=dtype) for _ in range(4)
    )
    # The public InfiniOp API supplies a float scale.  Keeping the generated
    # scalar in FP32 also avoids an unnecessary FP64-to-low-precision cast,
    # which is unsupported by the DTK 24.04 Triton backend.
    scale = Tensor(0, dtype=ninetoothed.float32)
    is_causal = Tensor(0, constexpr=True, value=is_causal)
    with_attn_mask = Tensor(0, constexpr=True, value=with_attn_mask)
    causal_variant = Tensor(0, constexpr=True, value=causal_variant)

    if emb_dim is not None:
        for tensor in (query, key, value, attn_mask, output):
            tensor.shape = tensor.shape[:-1] + (emb_dim,)

    if with_kv_cache:
        application = application_with_kv_cache
        tensors = (
            query,
            key,
            value,
            total_kv_len,
            present_key,
            present_value,
            present_key_slot,
            present_value_slot,
            attn_mask,
            is_causal,
            scale,
            output,
            with_attn_mask,
            causal_variant,
        )
    else:
        application = (
            application_without_kv_cache_low_precision_dot
            if dtype in (ninetoothed.bfloat16, ninetoothed.float32)
            else application_without_kv_cache
        )
        tensors = (
            query,
            key,
            value,
            total_kv_len,
            attn_mask,
            is_causal,
            scale,
            output,
            with_attn_mask,
            causal_variant,
        )

    return arrangement_, application, tensors
