#!/bin/bash
cd /workspace/camlive
export PYTHONPATH=<isolated copy of the tree>/cam_live/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python cam_live/docs/verification/repro/g3core/probe_rw_flipped_frozen.py --rw-dir cam_live/docs/evidence/RW --seeds 0 1 2 3 4 5 6 7 8 9 --out <out>/flipped_frozen.json > <out>/flipped_frozen.log 2>&1
.venv/bin/python cam_live/docs/evidence/RW/rw_meta_diag.py --arm drift --utility direct --seeds 0 1 2 3 4 5 6 7 8 9 --decisions 600 --out <out>/drift.json > <out>/drift.log 2>&1
.venv/bin/python cam_live/docs/evidence/RW/rw_meta_diag.py --arm pooled --utility direct --seeds 0 1 2 --decisions 600 --out <out>/learned_lr3e-3.json > <out>/learned_lr3e-3.log 2>&1
.venv/bin/python cam_live/docs/evidence/RW/rw_meta_diag.py --arm pooled --utility direct --lr-reward 1e-4 --seeds 0 1 2 3 4 5 --decisions 600 --out <out>/learned_lr1e-4.json > <out>/learned_lr1e-4.log 2>&1
.venv/bin/python cam_live/docs/evidence/RW/rw_meta_diag.py --arm frozen --utility direct --seeds 0 1 2 3 4 5 --decisions 600 --out <out>/frozen.json > <out>/frozen.log 2>&1
echo ALLDONE > <out>/done.txt
