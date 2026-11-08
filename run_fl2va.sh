#!/usr/bin/env bash
# Run the recorded Hygon experiment, or check its runtime without generating video.
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:---check}
case "$MODE" in --check|--run) ;; *) echo "Usage: bash run_fl2va.sh [--check|--run]" >&2; exit 2;; esac

# Optional: restore site-specific offline packages; it must be a trusted local file.
if [[ -n ${FL2VA_ENV:-} ]]; then
    [[ -f $FL2VA_ENV ]] || { echo "Missing FL2VA_ENV: $FL2VA_ENV" >&2; exit 2; }
    source "$FL2VA_ENV"
elif [[ -f /opt/dtk/env.sh ]]; then
    source /opt/dtk/env.sh
fi

ASSETS="$ROOT/artifacts/hygon-gfx936"
CORE_ARCHIVE="$ASSETS/infinicore-runtime-gfx936-20261108.tar.gz"
for path in "$CORE_ARCHIVE" "$ASSETS/_infinilm_producer_sync_trace.so" "$ASSETS/mha_ninetoothed_hygon.so" "$ASSETS/mha_full.so" "$ASSETS/mha72.so"; do
    [[ -r $path ]] || { echo "Missing runtime file: $path" >&2; exit 2; }
done
command -v python >/dev/null || { echo 'Python 3 is required' >&2; exit 2; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/fl2va-run.XXXXXXXX")
mkdir -p "$WORK/python/infinicore/lib" "$WORK/python/infinilm/lib" "$WORK/lib" "$WORK/deps"
cp -a "$ROOT/InfiniCore/python/infinicore/." "$WORK/python/infinicore/"
cp -a "$ROOT/InfiniLM/python/infinilm/." "$WORK/python/infinilm/"
python - "$CORE_ARCHIVE" "$WORK/python/infinicore/lib" <<'PY'
import sys, tarfile
from pathlib import Path
expected = {'_infinicore.cpython-311-x86_64-linux-gnu.so', 'libinfiniccl.so', 'libinfinicore_cpp_api.so', 'libinfiniop.so', 'libinfinirt.so'}
with tarfile.open(sys.argv[1], 'r:gz') as archive:
    for member in archive.getmembers():
        if member.name not in expected or not member.isfile():
            raise SystemExit(f'Unexpected archive member: {member.name}')
        with archive.extractfile(member) as src, (Path(sys.argv[2]) / member.name).open('wb') as dst:
            dst.write(src.read())
        expected.remove(member.name)
if expected:
    raise SystemExit(f'Missing runtime files: {sorted(expected)}')
PY
cp "$ASSETS/_infinilm_producer_sync_trace.so" "$WORK/python/infinilm/lib/_infinilm.so"
cp "$ASSETS/mha_full.so" "$ASSETS/mha72.so" "$WORK/lib/"
cp "$WORK/python/infinicore/lib/"*.so "$WORK/lib/"
if [[ -f "$ROOT/third_party/h3-python-extra-wheels.zip" ]]; then
    python - "$ROOT/third_party/h3-python-extra-wheels.zip" "$WORK/deps" <<'PY'
import io, pathlib, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as outer:
    for name in outer.namelist():
        if name.endswith('.whl') and not pathlib.PurePosixPath(name).name.lower().startswith('numpy-'):
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as wheel:
                wheel.extractall(sys.argv[2])
PY
fi
if [[ -d "$ROOT/third_party/wheels" ]]; then
    python - "$ROOT/third_party/wheels" "$WORK/deps" <<'PY'
import pathlib, sys, zipfile
for path in pathlib.Path(sys.argv[1]).glob('*.whl'):
    with zipfile.ZipFile(path) as wheel:
        wheel.extractall(sys.argv[2])
PY
fi
if [[ -f "$ROOT/third_party/h3-vision-runtime.tar" ]]; then
    mkdir -p "$WORK/vision"
    python - "$ROOT/third_party/h3-vision-runtime.tar" "$WORK/vision" <<'PY'
import pathlib, sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    for member in archive.getmembers():
        name = pathlib.PurePosixPath(member.name)
        if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
            raise SystemExit(f'Unexpected archive member: {member.name}')
    archive.extractall(sys.argv[2])
PY
    export PYTHONPATH="$WORK/vision${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ -f "$ROOT/third_party/diffusers-source.zip" ]]; then
    python - "$ROOT/third_party/diffusers-source.zip" "$WORK/diffusers" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
    export PYTHONPATH="$WORK/diffusers/src${PYTHONPATH:+:$PYTHONPATH}"
fi
# Additional offline packages or a site-specific diffusers checkout can be supplied here.
export PYTHONPATH="$WORK/python:$WORK/deps${FL2VA_EXTRA_PYTHONPATH:+:$FL2VA_EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
TORCH_LIB=$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$WORK/lib:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INFINI_ROOT="$WORK"
PLUGIN="$ASSETS/mha_ninetoothed_hygon.so"
export LD_PRELOAD="$PLUGIN${LD_PRELOAD:+:$LD_PRELOAD}"
echo "Runtime workspace: $WORK"
python - <<'PY'
import sys, torch, infinicore, infinilm
from infinilm.lib import _infinilm
print('Python:', sys.version.split()[0], 'PyTorch:', torch.__version__, 'HIP:', torch.version.hip)
print('Devices:', torch.cuda.device_count(), 'InfiniLM extension:', _infinilm.__file__)
if torch.cuda.device_count() < 8:
    raise SystemExit('The recorded configuration requires 8 available DCUs.')
import diffusers, transformers, safetensors, tokenizers, av, accelerate, einops, janus, xxhash, torchvision, triton
print('Python imports: OK')
PY
if [[ $MODE == --check ]]; then
    python "$ROOT/InfiniLM/examples/minimax_h3.py" fl2va --help >/dev/null
    echo 'Entry point check: OK. No video was generated.'
    exit 0
fi
: "${FIRST_FRAME:?Set FIRST_FRAME to the input image path}"
: "${PROMPT:?Set PROMPT to the text prompt}"
: "${OUTPUT:?Set OUTPUT to the desired output video path}"
MODEL_PATH=${MODEL_PATH:-/public/home/aclsylqidf/shared/MiniMax-H3/diffusers}
[[ -d $MODEL_PATH && -r $FIRST_FRAME ]] || { echo 'Model directory or input image not found/readable' >&2; exit 2; }
[[ ! -e $OUTPUT ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
mkdir -p "$(dirname -- "$OUTPUT")"
cd "$ROOT/InfiniLM"
python -u examples/minimax_h3.py fl2va --model-path="$MODEL_PATH" --text="$PROMPT" \
    --gpus="${GPUS:-0,1,2,3,4,5,6,7}" --steps="${STEPS:-50}" --seed="${SEED:-42}" \
    --num-frames="${NUM_FRAMES:-124}" --first-frame="$FIRST_FRAME" --output="$OUTPUT" \
    2>&1 | tee "${OUTPUT}.log"
