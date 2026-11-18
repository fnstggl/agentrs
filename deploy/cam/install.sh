#!/bin/bash
# Install the CAM learner, the matched no-world-model ablation and the CAM GitHub checkpoint watcher
# as supervisor programs (autostart; a finished run exits immediately on restart).
set -euo pipefail
cd "$(dirname "$0")"
for p in cam cam_nowm cam_backup cam_transfer; do
  install -m 755 agentrs_$p.sh /opt/supervisor-scripts/agentrs_$p.sh
  install -m 644 agentrs_$p.conf /etc/supervisor/conf.d/agentrs_$p.conf
done
supervisorctl reread
supervisorctl update agentrs_cam agentrs_cam_nowm agentrs_cam_backup agentrs_cam_transfer
