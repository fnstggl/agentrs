#!/bin/bash
# DOC command check for docs/REPRODUCE.md, part 2: the read-only / offline commands, executed. Every command runs
# from /workspace/camlive with CAM_LIVE_HOME in the scratch directory. Nothing here contacts an external service,
# starts a live run, starts a service as root (the dashboard is a 127.0.0.1 process of this shell, stopped at the
# end), pushes, or reads a credential. CPU only: CUDA_VISIBLE_DEVICES is emptied for every command that builds a
# model, and CAM_LIVE_GPU_LOCK points to a file that does not exist there, so any attempt to take the GPU lock fails.
# The doctor and the preflights run with this machine's real environment (they only inspect).
#   bash cam_live/docs/evidence/DOC/scripts/check_exec.sh [SCRATCH]      (results: SCRATCH/exec_results.tsv)
S=${1:-/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DOC/r3_re}
export CAM_LIVE_HOME=$S/home HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd /workspace/camlive || exit 2
P=.venv/bin/python
CL=.venv/bin/cam-live
OUT=$S/out
LIST=$S/exec_results.tsv
mkdir -p "$OUT"
: > "$LIST"
CPU=(env CUDA_VISIBLE_DEVICES= CAM_LIVE_GPU_LOCK=$S/no_such_gpu.lock OMP_NUM_THREADS=8)
# run ID PATTERN CMD...: exit status, seconds, command, first output line, last line matching PATTERN
run() {
  local id="$1" pat="$2"; shift 2
  local log="$OUT/exec_$id.txt"
  local t0
  t0=$(date +%s)
  timeout 900 "$@" < /dev/null > "$log" 2>&1
  local code=$?
  local dt=$(( $(date +%s) - t0 ))
  local first last
  first=$(grep -m1 -v '^\s*$' "$log" | cut -c1-140)
  last=""
  [ -n "$pat" ] && last=$(grep -E "$pat" "$log" | tail -1 | cut -c1-180)
  printf '%s\t%s\t%ss\t%s\t%s\t%s\n' "$code" "$id" "$dt" "$*" "$first" "$last" >> "$LIST"
}
T=cam_live/tasks
# ---- health, harness, preflight (real environment of this machine)
run doctor 'checks ok' $CL doctor
run doctor_strict 'checks ok' $CL doctor --strict
run doctor_json '' $CL doctor --json
run doctor_lock_absent 'gpu_lock|checks ok' env CAM_LIVE_GPU_LOCK=$S/no_such_gpu.lock $CL doctor
for t in discord_100_planning discord_100 discord_launch_check lab_staging utility_staging staging/discord_100_planning_staging; do
  run "preflight_${t//\//_}" '^LIVE|^STAGING' $CL preflight --task $T/$t.yaml
done
run preflight_planning_accept '^LIVE|^STAGING|ACCEPTED' $CL preflight --task $T/discord_100_planning.yaml --accept-ephemeral-storage
run preflight_launch_check_accept '^LIVE|^STAGING|ACCEPTED' $CL preflight --task $T/discord_launch_check.yaml --accept-ephemeral-storage
run run_preflight_only_lab '^LIVE|^STAGING' $CL run --task $T/lab_staging.yaml --preflight-only --no-doctor
run init_no_agent 'dashboard_token_file' $CL init --no-agent --home $S/home
run inspect_harness_lab 'missing|discord|Discord' $CL inspect-harness --lab
run browser_find '' $P -m cam_live.harness.browser --find cambrowser
run status_no_run '' $CL status --run-dir $S/home/runs/none
run validate_live_refused '' $CL validate --mode live
run verify_completion_only 'requirements:' $CL verify-completion --only IN-05,CM-13,DL-02 --quiet
run launch_live_noargs '' bash cam_live/tools/launch_live.sh
# ---- a scratch diagnostic run (tiny CPU configuration, LabWorld, in-process) and the commands that read a run
R=$S/run_lab
rm -rf "$R" "$S/export_lab"
run run_lab_inproc 'run ended' "${CPU[@]}" $CL run --task $T/lab_staging.yaml --inproc --foreground --max-iterations 4 \
    --config $S/debug.yaml --run-dir "$R" --no-doctor
run status_run 'status:|checkpoint:' $CL status --run-dir "$R"
run journal_run '' $CL journal --run-dir "$R"
run plan_report_run '' $CL plan-report --run-dir "$R" --out $S/plan_report_lab.md
run export_checkpoint_run '' "${CPU[@]}" $CL export-checkpoint --run-dir "$R" --out $S/export_lab
run prune_dry_home 'DRY RUN' $P cam_live/tools/prune_runs.py $S/home
run prune_dry_run 'DRY RUN' $P cam_live/tools/prune_runs.py "$R"
run evaluate_transfer_cpu 'executed|S_q|wrote|result' "${CPU[@]}" $CL evaluate-transfer --checkpoint $S/export_lab \
    --baseline $S/export_lab --domains T1 --seeds 0 --lifetimes 1 --decisions 2 --jobs 2 --out $S/transfer_cpu.json
# ---- dashboard (127.0.0.1, view-only), three requests, then stopped
"$CL" dashboard --run-dir "$R" --port 18787 --view-only > "$OUT/exec_dashboard.txt" 2>&1 &
DPID=$!
for _ in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18787/ && break; sleep 1; done
TOK=$(cat $S/home/dashboard.token 2>/dev/null)
c1=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18787/)
c2=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:18787/?token=$TOK")
c3=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" http://127.0.0.1:18787/api/status)
kill $DPID; wait $DPID 2>/dev/null
printf '%s\t%s\t%s\t%s\t%s\t%s\n' 0 dashboard_view_only - "$CL dashboard --run-dir R --port 18787 --view-only" \
    "no token: $c1; ?token=: $c2; Bearer /api/status: $c3" "stopped" >> "$LIST"
# ---- staging validation, CPU debug configuration (the documented --debug path; GPU hidden, lock file absent)
rm -rf "$S/val_debug"                  # a new directory: an existing one's state would be reused
run validate_debug 'report:' "${CPU[@]}" $CL validate --mode staging --debug --decisions 12 --out $S/val_debug
# ---- tests: collection per directory and of the whole suite in one process; one small suite executed
for d in credit diagnostics eval harness harness/auth harness/compute harness/env harness/skills integration nn predict \
         reward runtime storage test_secret_scan.py test_verify_completion.py; do
  id=${d%.py}; run "collect_${id//\//_}" 'collected|error' "${CPU[@]}" $P -m pytest cam_live/tests/$d --collect-only -q -p no:cacheprovider --rootdir=cam_live
done
run collect_all_one_process 'collected|error' "${CPU[@]}" $P -m pytest cam_live/tests --collect-only -q -p no:cacheprovider --rootdir=cam_live
run cam-live_test_storage 'passed|failed|exit code' "${CPU[@]}" $CL test --suite storage --results-dir $S/junit
# ---- secret scan, pattern part only: tools/secret_scan.py:main over fdab7d3~3..fdab7d3 with real_secrets() replaced
# by an empty list (the real-credential part reads /root/.config/camlive and /root/.config/agentrs, which this agent
# does not read); git log / git cat-file only, nothing written
run secret_scan_patterns_only 'secret scan' $P -I -c "import sys; sys.path.insert(0, 'cam_live/tools'); import secret_scan as s; \
s.real_secrets = lambda: []; sys.exit(s.main(['fdab7d3~3', 'fdab7d3']))"
echo "exec checks: $(wc -l < "$LIST") commands; results in $LIST"
