#!/usr/bin/env bash
# Launch the v2 comparison for the given seeds (default: 0). Each run uses 1 CPU thread;
# variants of the same seed share cached BC / world-model / critic initial stages.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs/final/logs
for seed in ${SEEDS:-0}; do
  for v in A B C D Dimag; do
    nohup python scripts/run_experiment.py configs/experiments/v2_${v}.yaml --threads 1 \
      --set seed=${seed} name=v2_${v}_s${seed} > runs/final/logs/v2_${v}_s${seed}.log 2>&1 &
  done
done
wait
