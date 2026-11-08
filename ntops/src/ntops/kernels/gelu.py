import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from ntops.kernels.element_wise import arrangement


def default_application(input, output):
    output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))  # noqa: F841


def tanh_application(input, output):
    input_loaded = input
    input_squared = input_loaded * input_loaded
    input_cubed = input_squared * input_loaded
    tanh_argument = 0.7978845608028654 * (
        input_loaded + 0.044715 * input_cubed
    )
    # DTK 24.04 provides a Triton 2.1 fork without tl.tanh.  Express tanh in
    # terms of exp so NineToothed can AOT compile the same semantics on Hygon.
    tanh_value = 2.0 / (1.0 + ntl.exp(-2.0 * tanh_argument)) - 1.0

    output = (  # noqa: F841
        0.5
        * input_loaded
        * (1 + tanh_value)
    )


def premake(ndim, approximate, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)

    if approximate == "tanh":
        application = tanh_application
    else:
        application = default_application

    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))

    return arrangement_, application, tensors
