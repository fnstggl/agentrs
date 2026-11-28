#!/bin/bash
# PS full-size evidence: every GPU section takes the shared GPU lock itself (each hold <= 840 s).
#   holds 1-2: the substantive protocol, persistent state on and off in the same hold (ps_oe_paired), objective 0, 1
#   hold 3:    compatibility of the live planning run's checkpoint with the PS code (copies only)
# usage: ps_gpu_jobs.sh PATCHED_SRC OUT_DIR SCRATCH
set -u
SRC=$1; OUT=$2; SCR=$3
export CAM_LIVE_GPU_LOCK=/workspace/agentrs/runs/cam_unified/gpu_heavy.lock
export CAM_LIVE_HOME=$SCR/home_gpu HF_HUB_OFFLINE=1 HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=$SRC
mkdir -p "$OUT" "$CAM_LIVE_HOME"
PY=/workspace/camlive/.venv/bin/python
S=/workspace/camlive/cam_live/docs/evidence/PS/scripts
LIVE=/workspace/camlive/runs/cam_live/live/discord-100-planning-20261128T083454Z
PLAN=${PLAN:-SW:rle,MECH:7}
DEC=${DEC:-12}
ROOT=/workspace/camlive/runs/cam_live/eval/ps/paired_gpu
for I in ${OBJECTIVES:-0 1}; do
  echo "[$(date -u +%H:%M:%S)] paired objective $I start"
  $PY $S/ps_oe_paired.py --plan $PLAN --objective $I --decisions $DEC --root $ROOT --out $OUT/paired.json \
      --hold-s 840 --threads 8 >> $OUT/paired.log 2>&1
  echo "[$(date -u +%H:%M:%S)] paired objective $I exit $?"
done
if [ "${SKIP_COMPAT:-0}" != 1 ]; then
  echo "[$(date -u +%H:%M:%S)] compat start"
  $PY $S/ps_compat_live.py --run-dir $LIVE --work $SCR/compat_live_work --out $OUT/compat_live.json > $OUT/compat_live.log 2>&1
  echo "[$(date -u +%H:%M:%S)] compat exit $?"
fi
