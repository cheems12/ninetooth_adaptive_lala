import ninetoothed
from ntops.kernels.mrope import premake

import infiniop.ninetoothed.build


def build():
    infiniop.ninetoothed.build.build(
        premake,
        {
            "dtype": (
                ninetoothed.float16,
                ninetoothed.bfloat16,
                ninetoothed.float32,
            ),
            "position_dtype": (ninetoothed.int32, ninetoothed.int64),
            "interleaved": (False, True),
            "block_size": (128,),
        },
        caller="cuda",
        op_name="mrope",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
