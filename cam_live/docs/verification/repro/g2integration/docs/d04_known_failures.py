"""G2-integration / docs probe d04: docs/KNOWN_FAILURES.md at a tree - do the cited evidence paths exist, and are the
rows marked open still open in the code? Static reads only (no import of cam_live, nothing executed).

usage (from /workspace/camlive):
  .venv/bin/python -I cam_live/docs/verification/repro/g2integration/docs/d04_known_failures.py <SRC>
SRC = tree with cam_live/ (e.g. the frozen copy of a5ab772). Prints HOLDS / BROKEN per check.
"""
import glob
import os
import re
import sys

SRC = os.path.abspath(sys.argv[1])
CL = os.path.join(SRC, "cam_live")
PKG = os.path.join(CL, "src", "cam_live")
kf = open(os.path.join(CL, "docs", "KNOWN_FAILURES.md")).read()
n = {"HOLDS": 0, "BROKEN": 0}


def claim(cid, ok, what):
    k = "HOLDS" if ok else "BROKEN"
    n[k] += 1
    print(f"{k:6}  {cid}: {what}")


def read(rel):
    return open(os.path.join(PKG, rel)).read()


# ---- 1. evidence paths cited in table rows
rows = [ln for ln in kf.splitlines() if ln.startswith("| KF-")]
missing, checked = [], 0
for ln in rows:
    rid = ln.split("|")[1].strip()
    ev = ln.split("|")[3] if len(ln.split("|")) > 4 else ""
    for p in re.findall(r"`((?:docs|src|tests)/[^`\s]+)`", ev):
        p = p.split("::")[0].rstrip(",;")
        base = os.path.join(CL, p)
        if "*" in p:
            ok = bool(glob.glob(base))
        else:
            ok = os.path.exists(base) or (p.startswith("src/") and os.path.exists(os.path.join(CL, "src", p[4:])))
        checked += 1
        if not ok:
            missing.append(f"{rid} {p}")
claim("evidence paths of KF rows exist", not missing, f"{checked} checked, missing: {missing}")

# ---- 2. rows whose state is 'open' / 'workaround' at 204e9d4 and were not re-stated later in the register
later = kf.split("## 2026-11-27 23:45 UTC")[1] if "## 2026-11-27 23:45 UTC" in kf else ""
def restated(rid):
    return f"| {rid} |" in later

pf = read("runtime/preflight.py")
cfg = open(os.path.join(CL, "configs", "cam_live.yaml")).read()
inner = read("reward/inner.py")
asm = read("reward/assemble.py")
agent = read("agent.py")
val = read("validation.py")
pyproj = open(os.path.join(CL, "pyproject.toml")).read()
cli_doc = read("eval/cli.py").split('"""')[1]
oe_doc = read("eval/oe_runner.py").split('"""')[1]
checks = [
    ("KF-D01", "no override for persistent_storage", "ACCEPTABLE_RISKS" not in pf),
    ("KF-D07", "default config does not bound the GPU-lock hold", "gpu_hold_s" not in cfg),
    ("KF-E01", "no loss reads ExperienceRecord.credit (policy update)", "credit_fn" not in inner),
    ("KF-E02", "info_gain never produced (assembler reads only extras['info_gain'])", "info_gain_nats" not in asm),
    ("KF-E03", "epistemic_after / learning_progress never written", "_annotate_epistemic_after" not in agent),
    ("KF-E05", "R_psi lifetime context reset at every objective change", "self.meta.new_objective()" not in agent),
    ("KF-E10", "whole suite cannot be collected in one process", "import-mode=importlib" not in pyproj),
    ("KF-E11", "validate --debug keeps the given (real) configuration", "master item B" not in val),
    ("KF-E12", "eval docstrings name missing commands", ("baselines_cpu" in cli_doc or "continual" in cli_doc
                                                          or " fresh" in oe_doc)),
]
for rid, what, still in checks:
    claim(f"{rid} (register: open{', re-stated later' if restated(rid) else ''})", still,
          f"'{what}' is {'still true' if still else 'no longer true'} in the code")

# ---- 3. results recorded elsewhere at the tree that the register does not mention
ev = os.path.join(CL, "docs", "evidence", "validation")
newer = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ev, "staging_2026112[8-9]*.json")))
claim("KF-R02 cites the latest real-configuration staging validation", not newer or any(x in kf for x in newer),
      f"newer evidence files not in the register: {newer}")
for key, what in (("INT3B-4", "prompt window (12 of 12 decisions over the window)"),
                  ("INT3B-5", "one-process suite failure (tolerance test)"),
                  ("INT3A-5", "promotion without held-out evidence"), ("G2L-6", "estimator selection not connected"),
                  ("DX-1", "agent hook for estimator selection not applied"), ("R3-M1", "root follows learner symlinks"),
                  ("R3-m2", "OP-09 false refusals"), ("approvals_exercised", "staging check failing")):
    claim(f"register mentions {key}", key in kf, what)
print(f"summary: {n['HOLDS']} HOLD, {n['BROKEN']} BROKEN")
