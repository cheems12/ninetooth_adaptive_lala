import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(
    input,
    cos,
    sin,
    positions,
    output,
    section_t,
    section_h,
    section_w,
    block_size=None,
):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # Input and output are normalized by InfiniCore to
    # [tokens, heads, head_size].
    input = input.tile((1, 1, block_size))
    # These tensors are random-access lookup tables.  Unit tiles express the
    # actual per-program footprint while application code indexes `.source`.
    cos = cos.tile((1, 1))
    sin = sin.tile((1, 1))
    positions = positions.tile((1, 1))
    output = output.tile((1, 1, block_size))
    return input, cos, sin, positions, output, section_t, section_h, section_w


def application_non_interleaved(
    input, cos, sin, positions, output, section_t, section_h, section_w
):
    token = input.offsets(0)[:, None, None]
    head = input.offsets(1)[None, :, None]
    dim = input.offsets(2)[None, None, :]
    half = cos.source.shape[1]
    rotary_dim = half * 2
    pair_dim = dim % half

    t_end = ntl.cast(section_t, ntl.int64)
    h_end = t_end + ntl.cast(section_h, ntl.int64)
    axis = ntl.where(pair_dim < t_end, 0, ntl.where(pair_dim < h_end, 1, 2))
    position = ntl.cast(positions.source[axis, token], ntl.int64)
    valid_position = (position >= 0) & (position < cos.source.shape[0])
    position = ntl.where(valid_position, position, 0)

    table_cos = ntl.cast(cos.source[position, pair_dim], ntl.float32)
    table_sin = ntl.cast(sin.source[position, pair_dim], ntl.float32)
    first = ntl.cast(input.source[token, head, pair_dim], ntl.float32)
    second = ntl.cast(
        input.source[token, head, pair_dim + half], ntl.float32
    )
    original = input.source[token, head, dim]
    rotated = ntl.where(
        dim < half,
        first * table_cos - second * table_sin,
        second * table_cos + first * table_sin,
    )
    output = ntl.where(dim < rotary_dim, ntl.cast(rotated, input.dtype), original)


def application_interleaved(
    input, cos, sin, positions, output, section_t, section_h, section_w
):
    token = input.offsets(0)[:, None, None]
    head = input.offsets(1)[None, :, None]
    dim = input.offsets(2)[None, None, :]
    half = cos.source.shape[1]
    rotary_dim = half * 2
    pair_dim = dim % half

    section_h_i64 = ntl.cast(section_h, ntl.int64)
    section_w_i64 = ntl.cast(section_w, ntl.int64)
    mod = pair_dim % 3
    h_axis = (mod == 1) & (pair_dim < section_h_i64 * 3)
    w_axis = (mod == 2) & (pair_dim < section_w_i64 * 3)
    axis = ntl.where(h_axis, 1, ntl.where(w_axis, 2, 0))
    position = ntl.cast(positions.source[axis, token], ntl.int64)
    valid_position = (position >= 0) & (position < cos.source.shape[0])
    position = ntl.where(valid_position, position, 0)

    table_cos = ntl.cast(cos.source[position, pair_dim], ntl.float32)
    table_sin = ntl.cast(sin.source[position, pair_dim], ntl.float32)
    first = ntl.cast(input.source[token, head, pair_dim], ntl.float32)
    second = ntl.cast(
        input.source[token, head, pair_dim + half], ntl.float32
    )
    original = input.source[token, head, dim]
    rotated = ntl.where(
        dim < half,
        first * table_cos - second * table_sin,
        second * table_cos + first * table_sin,
    )
    output = ntl.where(dim < rotary_dim, ntl.cast(rotated, input.dtype), original)


def premake(dtype=None, position_dtype=None, interleaved=False, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    application = (
        application_interleaved if interleaved else application_non_interleaved
    )
    tensors = (
        Tensor(3, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=dtype),
        Tensor(2, dtype=position_dtype),
        Tensor(3, dtype=dtype),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.int64),
        Tensor(0, dtype=ninetoothed.int64),
    )
    return arrangement_, application, tensors
