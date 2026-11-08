import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(input, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    # Input pixels are gathered through `input.source`; a unit tile prevents
    # the AOT tuner from treating the complete input image as one local tile.
    return input.tile((1, 1, 1, 1)), output.tile((1, 1, 1, block_size))


def application_align_corners(input, output):
    n = output.offsets(0)[:, None, None, None]
    c = output.offsets(1)[None, :, None, None]
    out_h = output.offsets(2)[None, None, :, None]
    out_w = output.offsets(3)[None, None, None, :]

    in_h_size = input.source.shape[2]
    in_w_size = input.source.shape[3]
    out_h_size = output.source.shape[2]
    out_w_size = output.source.shape[3]

    scale_h = ntl.where(
        out_h_size > 1,
        ((in_h_size - 1) * 1.0) / (out_h_size - 1),
        0.0,
    )
    scale_w = ntl.where(
        out_w_size > 1,
        ((in_w_size - 1) * 1.0) / (out_w_size - 1),
        0.0,
    )
    real_h = ntl.cast(out_h, ntl.float32) * scale_h
    real_w = ntl.cast(out_w, ntl.float32) * scale_w

    h0_unclamped = ntl.cast(ntl.floor(real_h), ntl.int64)
    w0_unclamped = ntl.cast(ntl.floor(real_w), ntl.int64)
    h1_unclamped = h0_unclamped + 1
    w1_unclamped = w0_unclamped + 1

    h_lambda = real_h - ntl.cast(h0_unclamped, ntl.float32)
    w_lambda = real_w - ntl.cast(w0_unclamped, ntl.float32)

    h0 = ntl.where(
        h0_unclamped < 0,
        0,
        ntl.where(h0_unclamped >= in_h_size, in_h_size - 1, h0_unclamped),
    )
    h1 = ntl.where(
        h1_unclamped < 0,
        0,
        ntl.where(h1_unclamped >= in_h_size, in_h_size - 1, h1_unclamped),
    )
    w0 = ntl.where(
        w0_unclamped < 0,
        0,
        ntl.where(w0_unclamped >= in_w_size, in_w_size - 1, w0_unclamped),
    )
    w1 = ntl.where(
        w1_unclamped < 0,
        0,
        ntl.where(w1_unclamped >= in_w_size, in_w_size - 1, w1_unclamped),
    )

    v00 = ntl.cast(input.source[n, c, h0, w0], ntl.float32)
    v01 = ntl.cast(input.source[n, c, h0, w1], ntl.float32)
    v10 = ntl.cast(input.source[n, c, h1, w0], ntl.float32)
    v11 = ntl.cast(input.source[n, c, h1, w1], ntl.float32)

    top = v00 + (v01 - v00) * w_lambda
    bottom = v10 + (v11 - v10) * w_lambda
    output = ntl.cast(top + (bottom - top) * h_lambda, input.dtype)


def application_half_pixel(input, output):
    n = output.offsets(0)[:, None, None, None]
    c = output.offsets(1)[None, :, None, None]
    out_h = output.offsets(2)[None, None, :, None]
    out_w = output.offsets(3)[None, None, None, :]

    in_h_size = input.source.shape[2]
    in_w_size = input.source.shape[3]
    out_h_size = output.source.shape[2]
    out_w_size = output.source.shape[3]

    scale_h = (in_h_size * 1.0) / out_h_size
    scale_w = (in_w_size * 1.0) / out_w_size
    real_h = (ntl.cast(out_h, ntl.float32) + 0.5) * scale_h - 0.5
    real_w = (ntl.cast(out_w, ntl.float32) + 0.5) * scale_w - 0.5

    h0_unclamped = ntl.cast(ntl.floor(real_h), ntl.int64)
    w0_unclamped = ntl.cast(ntl.floor(real_w), ntl.int64)
    h1_unclamped = h0_unclamped + 1
    w1_unclamped = w0_unclamped + 1

    h_lambda = real_h - ntl.cast(h0_unclamped, ntl.float32)
    w_lambda = real_w - ntl.cast(w0_unclamped, ntl.float32)

    h0 = ntl.where(
        h0_unclamped < 0,
        0,
        ntl.where(h0_unclamped >= in_h_size, in_h_size - 1, h0_unclamped),
    )
    h1 = ntl.where(
        h1_unclamped < 0,
        0,
        ntl.where(h1_unclamped >= in_h_size, in_h_size - 1, h1_unclamped),
    )
    w0 = ntl.where(
        w0_unclamped < 0,
        0,
        ntl.where(w0_unclamped >= in_w_size, in_w_size - 1, w0_unclamped),
    )
    w1 = ntl.where(
        w1_unclamped < 0,
        0,
        ntl.where(w1_unclamped >= in_w_size, in_w_size - 1, w1_unclamped),
    )

    v00 = ntl.cast(input.source[n, c, h0, w0], ntl.float32)
    v01 = ntl.cast(input.source[n, c, h0, w1], ntl.float32)
    v10 = ntl.cast(input.source[n, c, h1, w0], ntl.float32)
    v11 = ntl.cast(input.source[n, c, h1, w1], ntl.float32)

    top = v00 + (v01 - v00) * w_lambda
    bottom = v10 + (v11 - v10) * w_lambda
    output = ntl.cast(top + (bottom - top) * h_lambda, input.dtype)


def premake(dtype=None, align_corners=False, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    application = (
        application_align_corners if align_corners else application_half_pixel
    )
    return arrangement_, application, (Tensor(4, dtype=dtype), Tensor(4, dtype=dtype))
