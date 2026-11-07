# Isolated test workaround for platform cache filesystem; not production-safe.
import os, errno, shutil
_original_replace = os.replace
def _test_replace(src, dst):
    try:
        return _original_replace(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(src, dst)
        os.unlink(src)
os.replace = _test_replace
# Diagnostic LLVM 17 compatibility experiment: drop GEP no-wrap hint.
from triton.backends.amd.compiler import HIPBackend
_original_make_amdgcn = HIPBackend.make_amdgcn
def _compat_make_amdgcn(src, metadata, options):
    return _original_make_amdgcn(src.replace("getelementptr inbounds nuw ", "getelementptr inbounds "), metadata, options)
HIPBackend.make_amdgcn = staticmethod(_compat_make_amdgcn)
import os
from pathlib import Path
import ninetoothed
from ntops.kernels.scaled_dot_product_attention import premake
from infiniop.ninetoothed.build import build
out = Path(os.environ['IFAB_AFS_HOME']) / 'h3-build-support/mha-aot-f32math-unit8b'
build(premake, {'with_kv_cache':(0,), 'emb_dim':(64,128), 'is_causal':(0,1), 'with_attn_mask':(0,), 'causal_variant':(1,), 'dtype':(ninetoothed.float16,ninetoothed.bfloat16), 'block_size_m':(32,), 'block_size_n':(32,)}, caller='cuda', op_name='mha_f32math', output_dir=out)
print('AOT pilot built', flush=True)
