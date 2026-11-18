#!/bin/bash
# CAM product-fix run (branch feat/cam-product-fix, its own worktree so the running CAM programs' source
# is untouched). Waits until the no-world-model ablation has finished (its GPU share is reused) and
# enough GPU memory is free, then trains; on restart it resumes from runs/cam_pfix/ckpt/latest.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
SRC=${AGENTRS_CAM_PFIX_SRC:-/workspace/agentrs-cam-pfix}
RUN=/workspace/agentrs/runs/cam_pfix
WAIT_RUN=/workspace/agentrs/runs/cam_nowm
PY=/workspace/agentrs/.venv/bin/python
cd /workspace/agentrs
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC/src
if [ ! -f "$RUN/state.json" ]; then
  echo "waiting for $WAIT_RUN to finish (or stop running) and >= 6 GB free GPU memory"
  while true; do
    phase=$($PY -c "import json;print(json.load(open('$WAIT_RUN/state.json')).get('phase'))" 2>/dev/null || echo missing)
    st=$(supervisorctl status agentrs_cam_nowm 2>/dev/null | awk '{print $2}')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if { [ "$phase" = done ] || { [ "$st" != RUNNING ] && [ "$st" != STARTING ]; }; } && [ "${free:-0}" -ge 6000 ]; then
      echo "starting: ablation phase=$phase program=$st free_gpu_mib=$free"
      break
    fi
    sleep 60
  done
fi
exec $PY -u -m agentrs.cam.train --config $SRC/configs/cam/product_fix.yaml 2>&1
