#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
install -m 755 agentrs_cam2_queue_v21.sh /opt/supervisor-scripts/agentrs_cam2_queue_v21.sh
install -m 644 agentrs_cam2_queue_v21.conf /etc/supervisor/conf.d/agentrs_cam2_queue_v21.conf
supervisorctl reread
supervisorctl update agentrs_cam2_queue_v21
