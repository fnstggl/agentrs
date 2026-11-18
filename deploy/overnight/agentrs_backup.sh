#!/bin/bash
# GitHub Releases checkpoint backups of runs/overnight (mandatory destination): an initial verified
# checkpoint, then at most every AGENTRS_GH_BACKUP_EVERY_H hours (default 3), on every new best validated
# policy (>= 45 min apart) and the final checkpoint at the end. Each upload is verified before publishing.
# Token: ~/.config/agentrs/github_token (chmod 600) or GITHUB_TOKEN; repo: AGENTRS_GH_CKPT_REPO
# (default fnstggl/agentrs). Legacy hf:// / rclone: targets: python -m agentrs.backup (AGENTRS_BACKUP_TARGET).
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/agentrs/src
exec nice -n 5 /workspace/agentrs/.venv/bin/python -u -m agentrs.gh_backup watch --run runs/overnight \
  --every-h ${AGENTRS_GH_BACKUP_EVERY_H:-3} --interval 120 2>&1
