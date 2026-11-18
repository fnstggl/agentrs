#!/bin/bash
# Learning-speed transfer test: does CAM experience on negotiation + product make a never-seen family
# (business development) faster to learn? Two arms with identical configs, seeds and budgets on bizdev
# only: (a) fresh CAM, (b) CAM initialised from the main run's best validated checkpoint (frozen
# snapshot taken once the main run has >= MIN_ITER iterations). Curves: success vs env interactions.
set -uo pipefail
MAIN=${MAIN:-/workspace/agentrs/runs/cam}
OUT=${OUT:-/workspace/agentrs/runs/cam_transfer}
MIN_ITER=${MIN_ITER:-150}
ITERS=${ITERS:-60}
PY=/workspace/agentrs/.venv/bin/python
export PYTHONPATH=${AGENTRS_CAM_SRC:-/workspace/agentrs-cam/src} OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
if [ ! -f "$OUT/init/weights.pt" ]; then
  echo "waiting for $MAIN to reach iteration $MIN_ITER with a best checkpoint"
  until [ -f "$MAIN/ckpt/best/DONE" ] && [ "$($PY -c "import json;print(json.load(open('$MAIN/state.json'))['iter'])" 2>/dev/null || echo 0)" -ge "$MIN_ITER" ]; do
    sleep 120
  done
  mkdir -p "$OUT/init" && cp "$MAIN/ckpt/best/weights.pt" "$MAIN/ckpt/best/meta.json" "$OUT/init/"
fi
COMMON="train_families={\"bizdev\":1.0} heldout_families=[] max_iters=$ITERS eval_every=5 eval_episodes=200 test_episodes=400 planning_k=0 hours=3 gpu_mem_fraction=0.14 mb_episodes=32"
for arm in scratch pretrained; do
  INIT=""
  [ "$arm" = pretrained ] && INIT="init_from=$OUT/init"
  $PY -u -m agentrs.cam.train --config /workspace/agentrs-cam/configs/cam/a6000.yaml \
      --set run_dir=$OUT/$arm $COMMON $INIT || exit 1
done
echo "transfer test complete"
