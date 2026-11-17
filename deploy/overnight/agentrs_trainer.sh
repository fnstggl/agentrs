#!/bin/bash
# Supervisor wrapper for the AGENTRS overnight run (trainer). Installed by deploy/overnight/install.sh.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export HF_HOME=${HF_HOME:-/workspace/.hf_home} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=/workspace/agentrs/.venv/bin/python
export AGENTRS_DEVICE=cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PY" -u -m agentrs.econ.overnight --config configs/overnight/a6000.yaml 2>&1
