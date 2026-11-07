import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor


def arrangement(source, index, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    # InfiniCore normalizes the tensors to [outer, indexed, inner].  Let the
    # source tensor define the launch grid and keep the destination available
    # for an indexed store.
    source = source.tile((1, 1, block_size))
    # The index and destination are accessed through their source tensors.
    # Unit tiles describe their per-program footprint to the AOT tuner without
    # changing the indexed loads and stores below.
    index = index.tile((1,))
    output = output.tile((1, 1, block_size))
    return source, index, output


def application(source, index, output):
    outer = source.offsets(0)[:, None, None]
    index_offset = source.offsets(1)
    inner = source.offsets(2)[None, None, :]
    destination = ntl.cast(index.source[index_offset], ntl.int64)[None, :, None]
    output.source[outer, destination, inner] = source


def premake(dtype=None, index_dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (
        Tensor(3, dtype=dtype),
        Tensor(1, dtype=index_dtype),
        Tensor(3, dtype=dtype),
    )
    return arrangement_, application, tensors
