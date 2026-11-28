#!/bin/bash
# Operator tool: start a live CAM Live task (separated services). RUN BY THE OPERATOR, not by an agent:
# step 3 asks the operator to type the objective id before the authorization record is signed — signing is the
# operator's explicit act and is never done on the operator's behalf.
#
#   bash cam_live/tools/launch_live.sh [--accept-ephemeral-storage] cam_live/tasks/discord_100_planning.yaml 8 \
#        --read 'https://*' --read 'http://*'
#
# arguments: [--accept-ephemeral-storage] TASK_FILE HOURS [authorize_live.py options: --read / --owned /
#            --preauthorize / --account / --note]
#   --accept-ephemeral-storage  the operator accepts that this machine has no persistent volume (the doctor's
#           persistent_storage check would otherwise block live mode); recorded in run.json / preflight.json and
#           shown in status. Run data is then lost with the machine: copy `cam-live plan-report --out F` off the box.
#   HOURS   length of the authorization window (the task's own limits still bound the run)
# environment: CAM_LIVE_RUN_DIR (default runs/cam_live/live/<objective>-<UTC time>), CAM_LIVE_OPERATOR_KEY
#   (default /root/.config/camlive/operator_ed25519), CAM_LIVE_CONF (default /root/.config/camlive)
#
# What it does:
#   1. stores the Discord bot credential in the run's vault and the verifier's own copy (cam_live.harness.credentials,
#      read from the operator's 0600 file; nothing is printed);
#   2. places the operator's public key where the authorization service looks (<run>/auth/keys/operator.pub);
#   3. shows what will be authorized and signs the live record after the operator types the objective id
#      (tools/authorize_live.py; the private key never leaves that tool);
#   4. `cam-live run`: the supervisor starts the services, registers the task and the record, runs the preflight
#      against the running services (read-only Discord identity / permission check) and starts the learner only if
#      every live item is confirmed. Otherwise nothing is started and the missing items are printed.
# Afterwards: `cam-live status --run-dir R`, `cam-live dashboard --run-dir R`, `cam-live stop --run-dir R`,
#   `python -m cam_live.harness.operator_requests --run-dir R list`.
set -euo pipefail
RUNOPT=()
if [ "${1:-}" = "--accept-ephemeral-storage" ]; then RUNOPT+=(--accept-ephemeral-storage); shift; fi
[ $# -ge 2 ] || { sed -n '2,18p' "$0"; exit 2; }
TASK=$(readlink -f "$1"); HOURS=$2; shift 2
[ -t 0 ] || { echo "launch_live.sh: run this in an interactive terminal (the signature needs the operator's typed confirmation)" >&2; exit 2; }
CONF=${CAM_LIVE_CONF:-/root/.config/camlive}
KEY=${CAM_LIVE_OPERATOR_KEY:-$CONF/operator_ed25519}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH=$REPO/cam_live/src
PY=$REPO/.venv/bin/python
OBJ=$($PY -c "import sys; from cam_live.runtime.tasks import load_task; print(load_task(sys.argv[1], for_run=True).objective_id)" "$TASK")
R=${CAM_LIVE_RUN_DIR:-$REPO/runs/cam_live/live/$OBJ-$(date -u +%Y%m%dT%H%M%SZ)}
[ ! -e "$R/state.json" ] || { echo "launch_live.sh: $R already holds a run" >&2; exit 3; }
mkdir -p "$R"
echo "run directory: $R"
$PY -m cam_live.harness.credentials set discord_bot_token --run-dir "$R" --from-file "$CONF/discord_bot_token"
mkdir -p "$R/auth/keys"
install -m 0644 "$CONF/operator_ed25519.pub" "$R/auth/keys/operator.pub"
$PY "$REPO/cam_live/tools/authorize_live.py" --task "$TASK" --private "$KEY" --hours "$HOURS" \
    --out "$R.signed.json" "$@"
$PY -m cam_live.runtime.cli run --task "$TASK" --authorization "$R.signed.json" --run-dir "$R" "${RUNOPT[@]}"
echo
echo "follow: tail -f $R/logs/supervisor.log ; $PY -m cam_live.runtime.cli status --run-dir $R"
