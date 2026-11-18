#!/bin/bash
# CAM-v2 experiment queue (branch feat/cam-v2): starts the controlled experiments only as GPU capacity
# frees up; never touches other runs. Stop starting new jobs: touch /workspace/agentrs/runs/cam2_v22/QUEUE_STOP
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs-cam2-exp-v22   # frozen experiment code (tag cam2-exp-v22)
export PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/agentrs-cam2-exp-v22/src
exec /workspace/agentrs/.venv/bin/python -u scripts/cam2_queue.py --queue configs/cam2/queue_v22.yaml 2>&1
