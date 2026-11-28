"""G2-integration / docs probe d05: selected claims of MODEL_ARCHITECTURE.md (DL-02) and MATH_NOTES.md (DL-04) against
a tree (static reads), plus a comparison of re-measured numbers with the committed evidence.

usage (from /workspace/camlive):
  .venv/bin/python -I cam_live/docs/verification/repro/g2integration/docs/d05_model_math_claims.py <SRC> \
      [<param_counts.json from count_params.py run on SRC>] [<math_check output run on SRC>]
Re-measure first (CPU; loads Qwen3-4B in float32, ~16 GB RAM, ~3 min under load):
  cd <scratch> && PYTHONPATH=<SRC>/cam_live/src CUDA_VISIBLE_DEVICES= CAM_LIVE_HOME=<scratch>/home HF_HUB_OFFLINE=1 \
      /workspace/camlive/.venv/bin/python <SRC>/cam_live/docs/evidence/DOC/scripts/count_params.py <scratch>/pc.json
  cd <SRC>/cam_live && PYTHONPATH=<SRC>/cam_live/src CUDA_VISIBLE_DEVICES= \
      /workspace/camlive/.venv/bin/python docs/evidence/DOC/scripts/math_check.py > <scratch>/math.txt
"""
import json
import os
import re
import sys

SRC = os.path.abspath(sys.argv[1])
CL = os.path.join(SRC, "cam_live")
PKG = os.path.join(CL, "src", "cam_live")
n = {"HOLDS": 0, "BROKEN": 0}


def claim(cid, ok, what):
    k = "HOLDS" if ok else "BROKEN"
    n[k] += 1
    print(f"{k:6}  {cid}: {what}")


def read(rel):
    return open(os.path.join(PKG, rel)).read()


agent, inner, asm, learner = read("agent.py"), read("reward/inner.py"), read("reward/assemble.py"), read("credit/learner.py")
so = agent[agent.index("def set_objective"):]
so = so[:so.index("\n    def ", 10)]
claim("MA 8 step 1 / 9 / 10.2: set_objective calls MetaLearner.new_lifetime (R_psi context reset)",
      "self.meta.new_lifetime()" in so, "set_objective calls " + ("new_lifetime" if "new_lifetime()" in so else
                                                                 "new_objective (context kept)" if "new_objective()" in so else "?"))
claim("MA 10.1: every core_forward call passes state=None",
      not re.search(r"core_forward\([^)]*,\s*(?!None)[a-z_]+\s*\)", agent + inner + learner), "no carried state found")
claim("MA 7 / 10.7, MN 11.3: the integrated agent runs evidence-based estimator selection",
      any(s in agent for s in ("attach_experiments", "evaluate_experiments", "compare_with_experiment")),
      "agent.py calls attach_experiments / evaluate_experiments / compare_with_experiment"
      if any(s in agent for s in ("attach_experiments", "evaluate_experiments")) else
      "agent.py never calls attach_experiments / evaluate_experiments / compare_with_experiment (G2L-6, DX-1)")
claim("MN 11.3 / 14 / KF-E01: no training loss reads ExperienceRecord.credit", "credit_fn" not in inner and "credit_fn" not in agent,
      "reward/inner.py credit_fn / credited_goal_return and agent.py sets meta.inner.credit_fn" if "credit_fn" in agent else "-")
claim("MN 7.1 / 13 / 14: info_gain has no producer (assembler reads extras['info_gain'] only)", "info_gain_nats" not in asm,
      "assembler falls back to extras['info_gain_nats']" if "info_gain_nats" in asm else "-")
claim("MN 7.1 / 13: epistemic_after and learning_progress are never written",
      "_annotate_epistemic_after" not in agent and '"learning_progress"' not in agent, "agent.py annotates both (LRN-3)")
claim("MN 12: slow guard = one mean paired change over all providers", "_provider_verdict" not in learner,
      "per-provider verdicts and deferral without held-out evidence (G2L-4/5)" if "_provider_verdict" in learner else "-")
fast = read("nn/fast.py")
claim("MN 2.3: Poisson mode search takes the G-step only from the right", "okg = (~left)" in fast,
      "G-step on either side, convergence also requires the root condition (G2L-1)" if "okg = (~left)" not in fast else "-")
# re-measured numbers
if len(sys.argv) > 2:
    ev = json.load(open(os.path.join(CL, "docs", "evidence", "DOC", "param_counts.json")))
    new = json.load(open(sys.argv[2]))
    for lab in ("default", "debug"):
        if lab not in new or lab not in ev:
            continue
        a, b = ev[lab]["count_parameters"], new[lab]["count_parameters"]
        keys = ("by_group", "heads", "experiential_breakdown", "total_persistent_trainable", "adapters_lora", "adapters_bridge")
        diff = [k for k in keys if a.get(k) != b.get(k)]
        claim(f"MA 1 parameter counts ({lab}) equal the committed evidence", not diff,
              f"total {b.get('total_persistent_trainable')}; differing keys {diff}")
if len(sys.argv) > 3:
    ref = [l.strip() for l in open(os.path.join(CL, "docs", "evidence", "DOC", "math_check.txt")) if l[:1].isdigit()]
    got = [l.strip() for l in open(sys.argv[3]) if l[:1].isdigit()]
    claim("MN header: math_check.py output equals docs/evidence/DOC/math_check.txt", ref == got and len(ref) == 5,
          f"{len(got)} lines, equal={ref == got}")
print(f"summary: {n['HOLDS']} HOLD, {n['BROKEN']} BROKEN")
