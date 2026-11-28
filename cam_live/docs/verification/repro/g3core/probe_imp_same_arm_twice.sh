#!/bin/bash
# G3-core probe IMP-4 (D-IMP1 cause): run IMP's on arm twice and the off arm once with the SAME seed (debug scale,
# 1 lifetime, 3 decisions) and compare each decision's recorded inputs (one-step values, imagined continuations) and
# executed actions. Usage: probe_imp_same_arm_twice.sh <freeze> <workdir> <outdir>
F=$1; W=$2; O=$3
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH=$F/cam_live/src PYTHONHASHSEED=0
run() { nice -n 10 /workspace/camlive/.venv/bin/python -m cam_live.predict.imp_eval run --arm $1 --seed 1 --family T1 \
          --lifetimes 1 --decisions 3 --threads 2 --workdir $W/$2 --out $O/$2.json > $O/$2.log 2>&1; }
run on imp_same_arm_rep1; run on imp_same_arm_rep2; run off imp_same_seed_off
/workspace/camlive/.venv/bin/python - "$O" <<'PY'
import json, sys
o = sys.argv[1]
L = lambda n: json.load(open(f"{o}/{n}.json"))["lifetimes"][0]["decision_rows"]
a, c, b = L("imp_same_arm_rep1"), L("imp_same_arm_rep2"), L("imp_same_seed_off")
def d(x, y, k):
    v = [abs(p[k] - q[k]) for p, q in zip(x["cands"], y["cands"]) if p[k] is not None and q[k] is not None]
    return max(v) if v else float("nan")
for i in range(min(len(a), len(b), len(c))):
    print(f"decision {i}: on-rep1 {a[i]['operation']} {json.dumps(a[i]['arguments'])} | on-rep2 {c[i]['operation']} "
          f"{json.dumps(c[i]['arguments'])} | off {b[i]['operation']} {json.dumps(b[i]['arguments'])}")
    print(f"   on vs on: max|d one-step| {d(a[i], c[i], 2):.2e} max|d im_cont| {d(a[i], c[i], 4):.2e} | on vs off: "
          f"max|d one-step| {d(a[i], b[i], 2):.2e} max|d im_cont| {d(a[i], b[i], 4):.2e}")
PY
