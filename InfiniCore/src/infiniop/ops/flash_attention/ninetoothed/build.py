import os

import ninetoothed

import infiniop.ninetoothed.build

from . import flash_attention


def build():

    env_vars_to_check = ["MACA_HOME", "MACA_PATH", "MACA_ROOT"]
    if any(var in os.environ for var in env_vars_to_check):
        return

    with_kv_cache_values = (0,)
    emb_dim_values = (16, 32, 64, 128, 256)
    is_causal_values = (0, 1)
    with_attn_mask_values = (0,)
    # Keep constexpr values as plain integers.  IntEnum stringification is
    # preserved by NineToothed's AOT dispatcher generator and would otherwise
    # emit Python-only expressions such as ``CausalVariant.UPPER_LEFT`` into C.
    causal_variant_values = (1, 2)
    dtype_values = (ninetoothed.float16, ninetoothed.bfloat16, ninetoothed.float32)
    # Keep square tiles so the descriptor can select the largest specialization
    # that remains within gfx906's LDS limit for each dtype/head dimension.
    block_size_values = (16, 32, 64)

    constexpr_param_grid = {
        "with_kv_cache": with_kv_cache_values,
        "emb_dim": emb_dim_values,
        "is_causal": is_causal_values,
        "with_attn_mask": with_attn_mask_values,
        "causal_variant": causal_variant_values,
        "dtype": dtype_values,
        "block_size": block_size_values,
    }

    infiniop.ninetoothed.build.build(
        flash_attention.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="flash_attention",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
