#!/bin/bash
# Verified GitHub Releases checkpoints of runs/cam_pfix (initial, <= 3 h, new best, final); idles until
# the run has its first checkpoint.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/agentrs/src
exec nice -n 5 /workspace/agentrs/.venv/bin/python -u -m agentrs.gh_backup watch --run runs/cam_pfix --every-h 3 --interval 120 2>&1
