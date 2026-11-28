#!/bin/bash
# DOC command check for docs/REPRODUCE.md, part 1: every documented entry point with --help (argparse parses --help
# before any work, so nothing else happens). Records exit status and the first non-empty output line per command.
# Offline: no command here contacts a network, starts a service or uses the GPU.
#   bash cam_live/docs/evidence/DOC/scripts/check_help.sh [SCRATCH]      (results: SCRATCH/help_results.tsv)
S=${1:-/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DOC/r3_re}
export CAM_LIVE_HOME=$S/home HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd /workspace/camlive || exit 2
P=.venv/bin/python
CL=.venv/bin/cam-live
OUT=$S/out
LIST=$S/help_results.tsv
mkdir -p "$OUT" "$CAM_LIVE_HOME"
: > "$LIST"
run() {
  local id="$1"; shift
  local log="$OUT/help_$id.txt"
  timeout 300 "$@" < /dev/null > "$log" 2>&1
  local code=$?
  local first
  first=$(grep -m1 -v '^\s*$' "$log" | cut -c1-150)
  printf '%s\t%s\t%s\t%s\n' "$code" "$id" "$*" "$first" >> "$LIST"
}
# the cam-live console script (pyproject.toml [project.scripts] cam-live = cam_live.runtime.cli:main)
run cam-live $CL --help
for s in doctor init test validate inspect-harness preflight run status pause resume stop approve reject \
         export-checkpoint journal plan-report dashboard evaluate-transfer verify-completion; do
  run "cam-live_$s" $CL $s --help
done
# python -m entry points (every module with a __main__ block that an operator or evaluator starts)
for m in cam_live.runtime.cli cam_live.runtime.loop_service cam_live.runtime.verifier_service \
         cam_live.harness.credentials cam_live.harness.auth.opctl cam_live.harness.auth.launcher \
         cam_live.harness.auth.service cam_live.harness.service cam_live.harness.sandbox_host \
         cam_live.harness.operator_requests cam_live.harness.staging cam_live.harness.browser \
         cam_live.harness.grounding.bench cam_live.eval.cli cam_live.eval.oe_runner cam_live.eval.oe_batch \
         cam_live.eval.proposal_probe cam_live.eval.transfer cam_live.eval.report cam_live.eval.v3_report \
         cam_live.eval.meta_checks cam_live.verify_completion cam_live.nn.bench cam_live.nn.fast_compare \
         cam_live.predict.diagnose cam_live.reward.diagnose cam_live.reward.evidence cam_live.credit.diagnose; do
  run "$m" $P -m $m --help
done
run opctl_init-operator-key $P -m cam_live.harness.auth.opctl init-operator-key --help
for s in run anchors batch transfer-batch analyse integrity meta scaling large; do
  run "eval.cli_$s" $P -m cam_live.eval.cli $s --help
done
for s in sequence transfer; do run "oe_runner_$s" $P -m cam_live.eval.oe_runner $s --help; done
# tools/
for t in authorize_live discord_live_check prune_runs planning_rehearsal; do
  run "tools_$t" $P cam_live/tools/$t.py --help
done
for s in prepare serve holds check; do
  run "planning_rehearsal_$s" $P cam_live/tools/planning_rehearsal.py $s --help
done
# secret_scan.py has no argparse: any argument count other than 2 prints its usage line and exits 2 (no git, no file read)
run tools_secret_scan $P cam_live/tools/secret_scan.py --help
# launch_live.sh without arguments prints its header and exits 2 before anything else (tools/launch_live.sh:32)
run tools_launch_live_noargs bash cam_live/tools/launch_live.sh
# evidence scripts named in REPRODUCE.md
run run_separated $P cam_live/docs/evidence/validation/separated/run_separated.py --help
run run_live_observe $P cam_live/docs/evidence/validation/discord_live/run_live_observe.py --help
run playwright_version $P -m playwright --version
echo "help checks: $(awk -F'\t' '$1==0' "$LIST" | wc -l) of $(wc -l < "$LIST") exit 0; results in $LIST"
