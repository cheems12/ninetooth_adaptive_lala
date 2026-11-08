import ninetoothed
from ntops.kernels.index_copy import premake

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
            "index_dtype": (ninetoothed.int32, ninetoothed.int64),
            "block_size": (256,),
        },
        caller="cuda",
        op_name="index_copy",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
