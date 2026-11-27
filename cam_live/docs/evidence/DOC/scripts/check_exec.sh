#!/bin/bash
# Executes the read-only / offline commands (scratch CAM_LIVE_HOME; nothing contacts an external service).
S=/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DOC/repro
export CAM_LIVE_HOME=$S/home
cd /workspace/camlive
P=.venv/bin/python
OUT=$S/out
LIST=$S/exec_results.tsv
: > $LIST
run() {
  local id="$1"; shift
  local log="$OUT/exec_$id.txt"
  local t0=$(date +%s)
  timeout 900 "$@" > "$log" 2>&1
  local code=$?
  local dt=$(( $(date +%s) - t0 ))
  local first=$(grep -m1 -v '^\s*$' "$log" | cut -c1-150)
  printf '%s\t%s\t%ss\t%s\t%s\n' "$code" "$id" "$dt" "$*" "$first" >> $LIST
}
run doctor .venv/bin/cam-live doctor
run doctor_strict .venv/bin/cam-live doctor --strict
run preflight_planning .venv/bin/cam-live preflight --task cam_live/tasks/discord_100_planning.yaml
run preflight_discord100 .venv/bin/cam-live preflight --task cam_live/tasks/discord_100.yaml
run preflight_lab .venv/bin/cam-live preflight --task cam_live/tasks/lab_staging.yaml --no-doctor
run init_no_agent .venv/bin/cam-live init --no-agent --home $S/home
run inspect_harness_lab .venv/bin/cam-live inspect-harness --lab
run browser_find $P -m cam_live.harness.browser --find cambrowser
run playwright_version $P -m playwright --version
run prune_dry_scratch $P cam_live/tools/prune_runs.py $S/home
run verify_completion_only .venv/bin/cam-live verify-completion --only IN-05,CM-13,DL-02 --quiet
run status_no_run .venv/bin/cam-live status --run-dir $S/home/runs/none
run collect_runtime $P -m pytest cam_live/tests/runtime --collect-only -q --rootdir=cam_live
run collect_storage $P -m pytest cam_live/tests/storage --collect-only -q --rootdir=cam_live
