import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(output, timestep, log_max_period, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    tile_shape = (1, block_size)
    return (
        output.tile(tile_shape),
        timestep.tile(tile_shape),
        log_max_period,
    )


def application(output, timestep, log_max_period):
    half_dim = output.source.shape[-1] // 2
    output_dim = output.offsets(-1)
    frequency_index = output_dim % half_dim
    frequency = ntl.exp(
        -log_max_period
        * ntl.cast(frequency_index, ntl.float32)
        / ntl.cast(half_dim, ntl.float32)
    )
    angle = ntl.cast(timestep, ntl.float32) * frequency
    output = ntl.where(  # noqa: F841
        output_dim < half_dim,
        ntl.cos(angle),
        ntl.sin(angle),
    )


def premake(dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(2, dtype=ninetoothed.float32),
        Tensor(2, dtype=dtype),
        Tensor(0, dtype=ninetoothed.float32),
    )
    return arrangement_, application, tensors
