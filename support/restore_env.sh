#!/usr/bin/env bash
# Source this script inside a newly started InfiniFabric shell.
source "$IFAB_AFS_HOME/h3-setup.sh"
mkdir -p "$TMPDIR/h3vision" "$TMPDIR/h3wheels" "$TMPDIR/h3safe"
tar xf "$IFAB_AFS_HOME/h3-vision-runtime.tar" -C "$TMPDIR/h3vision"
unzip -qo "$IFAB_AFS_HOME/h3-python-extra-wheels.zip" -d "$TMPDIR/h3wheels"
python -c 'import os,glob,zipfile; root=os.environ["TMPDIR"]; [zipfile.ZipFile(p).extractall(root+"/h3safe") for p in glob.glob(root+"/h3wheels/*.whl") if not os.path.basename(p).lower().startswith("numpy")]'
export PYTHONPATH="$TMPDIR/h3safe:$TMPDIR/h3vision:$IFAB_AFS_HOME/h3-work/InfiniCore/python:$IFAB_AFS_HOME/h3-vendor:$IFAB_AFS_HOME/h3-python:$IFAB_AFS_HOME/h3-work/diffusers/src:$IFAB_AFS_HOME/h3-work/InfiniLM/python"
export LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/torch/lib:$IFAB_AFS_HOME/h3-work/install/lib:$LD_LIBRARY_PATH"
