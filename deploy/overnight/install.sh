#!/bin/bash
# Install the overnight supervisor programs. autostart=true: after an instance stop/start they come back and
# the trainer resumes from its latest checkpoint (a finished run exits immediately).
set -euo pipefail
cd "$(dirname "$0")"
for p in trainer live monitor github backup; do
  install -m 755 agentrs_$p.sh /opt/supervisor-scripts/agentrs_$p.sh
  install -m 644 agentrs_$p.conf /etc/supervisor/conf.d/agentrs_$p.conf
done
supervisorctl reread && supervisorctl update
