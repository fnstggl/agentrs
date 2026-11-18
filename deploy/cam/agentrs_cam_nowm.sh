#!/bin/bash
# CAM continual learner (new fully trainable core; no Qwen). Resumes from runs/cam/ckpt/latest.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/agentrs
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=${AGENTRS_CAM_SRC:-/workspace/agentrs/src}
exec /workspace/agentrs/.venv/bin/python -u -m agentrs.cam.train --config /workspace/agentrs-cam/configs/cam/ablate_nowm.yaml 2>&1
