import ninetoothed
from ntops.kernels.gelu import premake as gelu_premake

import infiniop.ninetoothed.build


def premake(ndim, dtype=None, block_size=None):
    return gelu_premake(ndim, "tanh", dtype=dtype, block_size=block_size)


def build():
    infiniop.ninetoothed.build.build(
        premake,
        {
            "ndim": range(1, 6),
            # 官方测试 test/infiniop/gelutanh.py 含 F64 用例
            # （`_TENSOR_DTYPES` 里有 float64），Hygon 的 HIP 实现也支持 F64，
            # 因此九齿路径必须同样覆盖，否则 descriptor 创建直接失败。
            "dtype": (
                ninetoothed.float16,
                ninetoothed.bfloat16,
                ninetoothed.float32,
                ninetoothed.float64,
            ),
            "block_size": (256,),
        },
        caller="cuda",
        op_name="gelutanh",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
