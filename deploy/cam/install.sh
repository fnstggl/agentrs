#!/bin/bash
# Install the CAM learner + its GitHub checkpoint watcher as supervisor programs (autostart).
set -euo pipefail
cd "$(dirname "$0")"
for p in cam cam_backup; do
  install -m 755 agentrs_$p.sh /opt/supervisor-scripts/agentrs_$p.sh
  install -m 644 agentrs_$p.conf /etc/supervisor/conf.d/agentrs_$p.conf
done
supervisorctl reread && supervisorctl add agentrs_cam 2>/dev/null; supervisorctl add agentrs_cam_backup 2>/dev/null; supervisorctl update agentrs_cam agentrs_cam_backup
