#!/bin/bash
# Install the CAM product-fix run and its GitHub checkpoint watcher as supervisor programs.
set -euo pipefail
cd "$(dirname "$0")"
for p in cam_pfix cam_pfix_backup; do
  install -m 755 agentrs_$p.sh /opt/supervisor-scripts/agentrs_$p.sh
  install -m 644 agentrs_$p.conf /etc/supervisor/conf.d/agentrs_$p.conf
done
supervisorctl reread
supervisorctl update agentrs_cam_pfix agentrs_cam_pfix_backup
