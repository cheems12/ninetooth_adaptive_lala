from pathlib import Path
import os,re
root=Path(os.environ['IFAB_AFS_HOME'])/'h3-build-support'
src=root/'mha-aot-f32math-unit8b'; dst=root/'mha-aot-f32math-unit8b-hip'; dst.mkdir(exist_ok=True)
rep={'CUmodule':'hipModule_t','CUfunction':'hipFunction_t','CUcontext':'hipCtx_t','CUstream':'hipStream_t','CUresult':'hipError_t','CUdeviceptr':'uintptr_t','CUDA_SUCCESS':'hipSuccess','CUDA_ERROR_INVALID_CONTEXT':'hipErrorInvalidContext','CUDA_ERROR_INVALID_HANDLE':'hipErrorInvalidResourceHandle','CUDA_ERROR_INVALID_VALUE':'hipErrorInvalidValue','CUDA_ERROR_OUT_OF_MEMORY':'hipErrorOutOfMemory','CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN':'hipDeviceAttributeMaxSharedMemoryPerBlock','CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES':'hipFuncAttributeMaxDynamicSharedMemorySize','CU_FUNC_CACHE_PREFER_SHARED':'hipFuncCachePreferShared','cuCtxGetCurrent':'hipCtxGetCurrent','cuFuncSetAttribute':'hipFuncSetAttribute','cuFuncSetCacheConfig':'hipFuncSetCacheConfig','cuGetErrorString':'hipDrvGetErrorString','cuLaunchKernel':'hipModuleLaunchKernel','cuModuleGetFunction':'hipModuleGetFunction','cuModuleLoadData':'hipModuleLoadData','cuModuleUnload':'hipModuleUnload'}
pattern=re.compile(r'\b('+'|'.join(map(re.escape,sorted(rep,key=len,reverse=True)))+r')\b')
count=0
for p in src.iterdir():
 if p.suffix not in ('.cpp','.h','.c'): continue
 t=p.read_text()
 if p.suffix=='.cpp':
  t=t.replace('#include <cuda.h>','#include <hip/hip_runtime_api.h>')
  t=pattern.sub(lambda m:rep[m.group()],t)
  t=t.replace(', 4 * 32, 1, 1,',', 4 * 64, 1, 1,')
  if p.name.endswith('.triton.launch.cpp'):
   t=t.replace('#include <unordered_map>','#include <map>')
   t=t.replace('std::unordered_map<hipCtx_t, State>','std::map<std::pair<int, hipCtx_t>, State>')
   a=t.index('    if (context == nullptr) {'); b=t.index('    try {',a)
   t=t[:a]+'''    int device = -1;
    result = hipGetDevice(&device);
    if (result != hipSuccess) { launch_mutex.unlock(); return result; }
    auto cache_key = std::make_pair(device, context);

'''+t[b:]
   t=t.replace('context_states.find(context)','context_states.find(cache_key)').replace('context_states.emplace(context, state)','context_states.emplace(cache_key, state)')
   count+=1
 (dst/p.name).write_text(t)
assert count==8, count
print('staged multi-device host adapters:',count,flush=True)
