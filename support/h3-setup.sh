#!/usr/bin/env bash
export PATH="$IFAB_AFS_HOME/tools:$PATH"
export INFINI_ROOT="$IFAB_AFS_HOME/h3-work/install"
export GIT_CONFIG_NOSYSTEM=1 HYGON_ARCH=gfx936
export PYTHONPATH="$IFAB_AFS_HOME/h3-vendor:$IFAB_AFS_HOME/h3-python:$IFAB_AFS_HOME/h3-extra:$IFAB_AFS_HOME/h3-vision-python:$IFAB_AFS_HOME/h3-work/diffusers/src:$IFAB_AFS_HOME/h3-work/InfiniLM/python:${PYTHONPATH:-}"
export FA_SO_PATH="$(find "$IFAB_AFS_HOME/h3-vendor" -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit)"
source /opt/dtk/env.sh
