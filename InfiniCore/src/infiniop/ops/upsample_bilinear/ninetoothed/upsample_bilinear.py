import ninetoothed
from ntops.kernels.upsample_bilinear import premake

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
            "align_corners": (False, True),
            "block_size": (256,),
        },
        caller="cuda",
        op_name="upsample_bilinear",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
