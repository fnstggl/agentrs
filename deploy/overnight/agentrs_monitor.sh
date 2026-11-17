#!/bin/bash
# Supervisor wrapper for the AGENTRS overnight run (monitor). Installed by deploy/overnight/install.sh.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export HF_HOME=${HF_HOME:-/workspace/.hf_home} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=/workspace/agentrs/.venv/bin/python
exec "$PY" -u -m agentrs.econ.monitor --run-dir runs/overnight --interval 60 2>&1
