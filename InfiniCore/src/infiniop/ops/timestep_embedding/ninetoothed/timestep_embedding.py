import ninetoothed

import infiniop.ninetoothed.build
from ntops.kernels.timestep_embedding import premake


def build():
    infiniop.ninetoothed.build.build(
        premake,
        {
            "dtype": (
                ninetoothed.float16,
                ninetoothed.bfloat16,
                ninetoothed.float32,
            ),
            "block_size": (256,),
        },
        caller="cuda",
        op_name="timestep_embedding",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
