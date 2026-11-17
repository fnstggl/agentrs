#!/bin/bash
# Hourly verified off-instance backup of runs/overnight (checkpoints, run state, experience, reports).
# Needs AGENTRS_BACKUP_TARGET in ${WORKSPACE}/.env, e.g. hf://<user>/agentrs-overnight (after
# `hf auth login`) or rclone:<remote>:<bucket>/agentrs (after `rclone config`). Exits 0 if unset.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
if [ -z "${AGENTRS_BACKUP_TARGET:-}" ]; then
  echo "AGENTRS_BACKUP_TARGET is not set: no external backup configured"; exit 0
fi
exec /workspace/agentrs/.venv/bin/python -u -m agentrs.backup watch --runs runs/overnight --interval ${AGENTRS_BACKUP_INTERVAL:-3600} 2>&1
