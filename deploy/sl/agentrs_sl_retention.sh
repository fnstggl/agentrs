#!/bin/bash
# Superlearner storage retention: every 15 min removes only redundant checkpoint copies (see scripts/sl_retention.py).
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs-sl
exec /workspace/agentrs/.venv/bin/python -u scripts/sl_retention.py --apply --loop 900 2>&1
