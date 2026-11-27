#!/bin/bash
# Runs each command with --help (no side effects past argparse), records exit status and the first output line.
S=/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/DOC/repro
export CAM_LIVE_HOME=$S/home
cd /workspace/camlive
P=.venv/bin/python
OUT=$S/out
LIST=$S/help_results.tsv
: > $LIST
run() {
  local id="$1"; shift
  local log="$OUT/$id.txt"
  timeout 300 "$@" > "$log" 2>&1
  local code=$?
  local first=$(grep -m1 -v '^\s*$' "$log" | cut -c1-150)
  printf '%s\t%s\t%s\t%s\n' "$code" "$id" "$*" "$first" >> $LIST
}
run cam-live .venv/bin/cam-live --help
for s in doctor init test validate inspect-harness preflight run status pause resume stop approve reject export-checkpoint journal dashboard evaluate-transfer verify-completion; do
  run "cam-live_$s" .venv/bin/cam-live $s --help
done
run cli_module $P -m cam_live.runtime.cli --help
for m in cam_live.runtime.loop_service cam_live.runtime.verifier_service cam_live.harness.credentials cam_live.harness.auth.opctl cam_live.harness.auth.launcher cam_live.harness.auth.service cam_live.harness.service cam_live.harness.operator_requests cam_live.harness.staging cam_live.harness.browser cam_live.harness.grounding.bench cam_live.eval.cli cam_live.eval.oe_runner cam_live.eval.proposal_probe cam_live.eval.transfer cam_live.eval.report cam_live.verify_completion cam_live.nn.bench cam_live.nn.fast_compare cam_live.predict.diagnose cam_live.reward.diagnose cam_live.reward.evidence cam_live.credit.diagnose; do
  run "$m" $P -m $m --help
done
for s in run anchors batch analyse integrity meta scaling large; do
  run "eval.cli_$s" $P -m cam_live.eval.cli $s --help
done
for s in sequence transfer; do run "oe_runner_$s" $P -m cam_live.eval.oe_runner $s --help; done
for t in authorize_live discord_live_check prune_runs; do run "tools_$t" $P cam_live/tools/$t.py --help; done
