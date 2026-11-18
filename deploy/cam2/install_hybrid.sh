#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
install -m 755 agentrs_cam_hybrid_queue.sh /opt/supervisor-scripts/agentrs_cam_hybrid_queue.sh
install -m 644 agentrs_cam_hybrid_queue.conf /etc/supervisor/conf.d/agentrs_cam_hybrid_queue.conf
supervisorctl reread
supervisorctl update agentrs_cam_hybrid_queue
