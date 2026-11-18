#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
install -m 755 agentrs_cam2_queue_v22.sh /opt/supervisor-scripts/agentrs_cam2_queue_v22.sh
install -m 644 agentrs_cam2_queue_v22.conf /etc/supervisor/conf.d/agentrs_cam2_queue_v22.conf
supervisorctl reread
supervisorctl update agentrs_cam2_queue_v22
