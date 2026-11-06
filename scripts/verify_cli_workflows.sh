#!/usr/bin/env bash
# Executes every documented CLI workflow at smoke scale (tiny models, few steps) to verify the
# README commands exist and run. Usage: bash scripts/verify_cli_workflows.sh [workdir]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
W=${1:-runs/cli_check}
rm -rf "$W"; mkdir -p "$W"
M=smoke
A="python -m agentrs"
set -x
$A setup-check
$A run-episode --family form_validation --actor teacher -v --out "$W/episodes"
printf 'type(e1, "x")\nfinish()\n' | $A record-demo --family fill_field --consent --demonstrator-id tester --out "$W/demos"
$A gen-data --out "$W/data/train" --split train --families click_button,fill_field,menu --n-per-family 6
$A gen-data --out "$W/data/train" --split train --families click_button,fill_field,menu --n-per-family 3 --actor random
$A gen-data --out "$W/data/val" --split val --families click_button,fill_field,menu --n-per-family 2
$A data-summary --store "$W/data/train"
$A train-wm --data "$W/data/train" --val-data "$W/data/val" --model $M --steps 6 --eval-every 3 --out "$W/ckpt/wm"
$A eval-wm --world-model "$W/ckpt/wm" --train-data "$W/data/train" --val-data "$W/data/val" --n-generate 4 --n-drift 2 --out "$W/wm_eval.json"
$A train-bc --data "$W/data/train" --val-data "$W/data/val" --model $M --steps 6 --eval-every 3 --out "$W/ckpt/policy_bc"
$A train-bc --data "$W/data/train" --val-data "$W/data/val" --model $M --steps 9 --eval-every 3 --out "$W/ckpt/policy_bc" --resume "$W/ckpt/policy_bc"
$A train-critic --data "$W/data/train" --val-data "$W/data/val" --model $M --steps 4 --out "$W/ckpt/critic"
$A train-rl --policy "$W/ckpt/policy_bc" --critic "$W/ckpt/critic" --model $M --algo ppo --iterations 1 --episodes-per-iter 4 --families click_button,menu --out "$W/rl"
$A train-rl --policy "$W/ckpt/policy_bc" --critic "$W/ckpt/critic" --model $M --algo ppo --iterations 2 --episodes-per-iter 4 --families click_button,menu --out "$W/rl" --resume "$W/rl"
$A run-episode --family login --actor planner --policy "$W/rl/policy" --world-model "$W/ckpt/wm" --critic "$W/rl/critic" -v
$A evaluate --policy "$W/rl/policy" --sets val,test_family --n-per-family 1 --families click_button,menu
$A evaluate --actor planner --policy "$W/rl/policy" --world-model "$W/ckpt/wm" --critic "$W/rl/critic" --sets val --n-per-family 1 --families click_button
$A compare --a "$W/ckpt/policy_bc" --b "$W/rl/policy" --sets val --n-per-family 1 --families click_button,menu
$A continual --config configs/continual/smoke.yaml --set root="$W/continual" cycles=1
$A rollback --registry "$W/continual/registry.sqlite" --component policy
$A experiment configs/experiments/smoke.yaml --set out_dir="$W/exp" name=smoke_D rl_iterations=1 cache_dir="$W/cache"
$A baselines --config configs/experiments/smoke.yaml --variants A,C --seeds 0 --prefix smoke --set out_dir="$W/exp" rl_iterations=1 cache_dir="$W/cache"
$A report --runs "$W"/exp/smoke_* --out "$W/report.md"
$A visualize --run "$W/exp/smoke_D"
$A visualize --store "$W/rl/experience" --world-model "$W/ckpt/wm" | head -40
python scripts/planner_sweep.py "$W/exp/smoke_D" --sets val --grid horizon=1,2 uncertainty_penalty=0,1 --n-per-family 1
set +x
echo "ALL CLI WORKFLOWS RAN"
