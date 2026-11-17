#!/bin/bash
# Supervisor wrapper for the AGENTRS overnight run (live). Installed by deploy/overnight/install.sh.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export HF_HOME=${HF_HOME:-/workspace/.hf_home} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=/workspace/agentrs/.venv/bin/python
export AGENTRS_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
exec nice -n 5 "$PY" -u -m agentrs.live run --config configs/overnight/live.yaml --max-hours 10 2>&1
