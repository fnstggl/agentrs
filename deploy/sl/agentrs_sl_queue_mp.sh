#!/bin/bash
# Superlearner Phase 2 (Melting Pot) queue (frozen experiment code in /workspace/agentrs-sl-exp): smoke-tests each job's exact
# production command, then starts jobs as GPU / disk capacity allows. Stop new starts: touch /workspace/agentrs/runs/sl_mp/QUEUE_STOP
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs-sl-exp-mp
export PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/agentrs-sl-exp-mp/src
exec /workspace/agentrs/.venv/bin/python -u scripts/sl_queue.py --queue configs/sl/queue_phase2_mp.yaml 2>&1
