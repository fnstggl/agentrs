#!/bin/bash
# Superlearner experiment queue (frozen experiment code in /workspace/agentrs-sl-exp): smoke-tests each job's exact
# production command, then starts jobs as GPU / disk capacity allows. Stop new starts: touch /workspace/agentrs/runs/sl/QUEUE_STOP
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs-sl-exp
export PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/agentrs-sl-exp/src
exec /workspace/agentrs/.venv/bin/python -u scripts/sl_queue.py --queue configs/sl/queue_phase1.yaml 2>&1
