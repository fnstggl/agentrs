#!/bin/bash
# SG real-model runs (Qwen3-4B + 50m core, open-ended generation on), one GPU-lock hold each (<= 840 s of work),
# from the patched copy (HEAD + SG files + the SG agent hook). Sequential: each waits for the shared lock.
SG=/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/SG
cd "$SG" || exit 1
export CAM_LIVE_HOME=$SG/home_real
export CAM_LIVE_GPU_LOCK=/workspace/agentrs/runs/cam_unified/gpu_heavy.lock
export PYTHONPATH=$SG/patched2/cam_live/src
export OMP_NUM_THREADS=4
mkdir -p "$CAM_LIVE_HOME" runs
PY=/workspace/camlive/.venv/bin/python
run() {  # name, args...
  local name=$1; shift
  if [ -f "runs/$name.json" ] && grep -q '"executed": true' "runs/$name.json"; then echo "skip $name"; return; fi
  echo "$(date -u +%H:%M:%S) start $name"
  nice -n 10 $PY sg_runner.py --real --seed 0 --threads 4 --hold-s 820 --out "runs/$name.json" --root "runs/$name" "$@" \
      > "runs/$name.log" 2>&1
  echo "$(date -u +%H:%M:%S) end $name exit $?"
  rm -rf "runs/$name/agent/learner/embed_cache" "runs/$name"/obj*/checkpoints
}
run real_skills     --plan CU2:store_entry,CU2:store_entry,CU:record_item --decisions 12
run real_primitive  --plan CU2:store_entry,CU2:store_entry,CU:record_item --decisions 12 --no-skills
run real_goalfree   --plan CUFREE:8,CU2:store_entry,CU:record_item --decisions 12
run real_nogoalfree --plan CU2:store_entry,CU:record_item --decisions 12
