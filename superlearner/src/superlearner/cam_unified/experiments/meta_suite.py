"""Meta-reward experiments D, E, F, I, J (CAM_UNIFIED_SPEC_META_REWARDS.md Part 15; plan 2026-11-26 §5 E2/E3), run
ONLY through the canonical loop ``runtime.run_lifetime`` with provenance rows (shared machinery with
experiments.dom_suite: builder, worlds, label store, integrity scan, GPU-busy and semantic-token accounting).

    python -m superlearner.cam_unified.experiments.meta_suite list
    python -m superlearner.cam_unified.experiments.meta_suite run --exp F --condition F-obs --smoke --run-dir DIR
    python -m superlearner.cam_unified.experiments.meta_suite run --exp D --condition D5 --seed 2 --run-dir DIR

Every condition declares what it needs (``requires``). Before anything runs the prerequisites are checked against
the code at this commit; an unmet one writes a provenance row with ``status: "BLOCKED"`` and the reason instead of
running a condition that would silently measure something else (e.g. R_psi is the seam stub that returns 0; the
canonical loop does not yet deliver the legacy reward channel a baseline needs). Non-compliant reference
conditions (native reward, sparse verified terminal reward as the ACTOR reward) are labelled ``compliant: False``.

Prerequisites:
  rpsi_real        selfeval.reward_model.RewardModel has parameters (not the seam stub)            (W8)
  meta_step        the learner hook exposes meta_step() and the preset has meta_every > 0           (W7/W8)
  credit           credit.longdelay RUDDER/COCOA wired into the learner (learner.credit is set)     (W8)
  curriculum       autonomy.curriculum wired into the loop (learner.curriculum is set)              (W8/W1)
  loop_reward_mode the canonical loop accepts reward_mode other than observation_only             (W1)
  goal_free        the loop supports CAM-generated goals (run_lifetime accepts goal_free=True)      (W1/W8)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from . import dom_suite as DS

PREREGS = {"D": "superlearner/docs/experiments/U-MR-1.md", "E": "superlearner/docs/experiments/U-MR-1.md",
           "F": "superlearner/docs/experiments/U-MR-1.md", "I": "superlearner/docs/experiments/U-MR-1.md",
           "J": "superlearner/docs/experiments/U-MR-1.md"}
EXPERIMENT_ID = "U-MR-1"


@dataclass(frozen=True)
class Cond:
    cid: str
    exp: str
    description: str
    overrides: tuple = ()                 # CamConfig overrides ((field, value), ...)
    reward_mode: str = "observation_only"
    requires: tuple = ()
    compliant: bool = True
    learn: bool = True
    worlds: str = "train"                 # train | val | integrity | probe


def _c(cid, exp, desc, **kw):
    kw["overrides"] = tuple(sorted((kw.pop("overrides", None) or {}).items()))
    return Cond(cid, exp, desc, **kw)


CONDITIONS = {c.cid: c for c in [
    # ---- Experiment D: learning efficiency (identical architecture, seven reward conditions)
    _c("D1", "D", "historical Crucible native (shaped) reward as the actor reward", reward_mode="legacy",
       requires=("loop_reward_mode",), compliant=False),
    _c("D2", "D", "sparse verified terminal utility as the actor reward (beta_ver U_ver)",
       reward_mode="sparse_verified", requires=("loop_reward_mode",), compliant=False),
    _c("D3", "D", "learned goal-progress evaluation only (Phi shaping; lambda_ig = lambda_c = 0, rho off, no meta)",
       overrides={"lambda_ig": 0.0, "lambda_c": 0.0, "meta_every": 0}, requires=("rpsi_real",)),
    _c("D4", "D", "hand-designed intrinsic motivation only (information gain - cost; Phi and rho off)",
       overrides={"meta_every": 0}, requires=("rpsi_real",)),
    _c("D5", "D", "meta-learned intrinsic reward (full R_psi, meta step every 16 updates)",
       requires=("rpsi_real", "meta_step")),
    _c("D6", "D", "meta-learned reward + retrospective credit (RUDDER / COCOA)",
       requires=("rpsi_real", "meta_step", "credit")),
    _c("D7", "D", "full architecture with the autonomous curriculum C_chi",
       requires=("rpsi_real", "meta_step", "credit", "curriculum")),
    # ---- Experiment E: novel objectives (held-out families; evaluation without learning)
    _c("E-cam", "E", "trained CAM evaluator + R_psi on held-out (val) families, weights-only, no learning",
       requires=("rpsi_real",), learn=False, worlds="val"),
    _c("E-frozen", "E", "frozen pretrained semantic model on the same held-out families (DOM condition A)",
       learn=False, worlds="val"),
    # ---- Experiment F: reward integrity (adversarial diagnostic worlds, envs/integrity.py)
    _c("F-obs", "F", "CAM in observation_only on the integrity worlds with the reward model of this commit "
                     "(engineering baseline; no integrity claim while R_psi is the seam stub)", worlds="integrity"),
    _c("F-rpsi", "F", "CAM with meta-learned R_psi on the integrity worlds: internal reward vs verified utility",
       requires=("rpsi_real", "meta_step"), worlds="integrity"),
    # ---- Experiment I: learning without an objective (exploration budget, then weights-only evaluation)
    _c("I-none", "I", "no exploration (weights-only evaluation of the untrained agent)", learn=False),
    _c("I-random", "I", "random exploration (uniform tool choice) then weights-only evaluation",
       requires=("goal_free",)),
    _c("I-fixed", "I", "fixed curriculum of self-goals", requires=("goal_free",)),
    _c("I-curiosity", "I", "ordinary curiosity (information gain only)", requires=("goal_free", "rpsi_real")),
    _c("I-meta", "I", "meta-learned autonomous curriculum", requires=("goal_free", "rpsi_real", "curriculum")),
    # ---- Experiment J: meta-learned reward verification (ProbeEnv: test first, optimum 10)
    _c("J-meta", "J", "R_psi meta-learned on ProbeEnv (informative test)", requires=("rpsi_real", "meta_step"),
       worlds="probe"),
    _c("J-phi", "J", "Phi only (no rho, no meta step) on ProbeEnv", overrides={"meta_every": 0},
       requires=("rpsi_real",), worlds="probe"),
    _c("J-lp", "J", "learning-progress-only intrinsic reward (the pre-2026-11-26 lp_coef path)",
       requires=("rpsi_real",), worlds="probe"),
    _c("J-uninformative", "J", "R_psi meta-learned on ProbeEnv with an uninformative test (must stay <= 0.1)",
       requires=("rpsi_real", "meta_step"), worlds="probe"),
]}
INTEGRITY_WORLDS = ("integrity:easy_useless", "integrity:noisy_tv", "integrity:duplicate_outcomes",
                    "integrity:short_term", "integrity:misleading_correlation", "integrity:fabricated_evidence",
                    "integrity:attractive_ineffective")


# ============================================================================================ prerequisites
def prerequisites(preset: str = "dom_dev") -> dict:
    """{name: (ok, reason)} for every prerequisite, measured on the code of this commit (no GPU needed)."""
    out = {}
    try:
        from ..config import make_preset
        from ..selfeval.reward_model import RewardModel
        cfg = make_preset(preset)
        m = RewardModel(cfg)
        n = sum(p.numel() for p in m.parameters()) if hasattr(m, "parameters") else 0
        out["rpsi_real"] = (n > 0, f"RewardModel has {n} parameters" + (" (seam stub: reward = 0)" if not n else ""))
    except Exception as e:
        out["rpsi_real"] = (False, f"RewardModel unavailable: {e!r}"[:200])
    fn, loop = DS.resolve_loop()
    real_loop = loop == "runtime.run_lifetime"
    sig = inspect.signature(fn) if real_loop else None
    out["loop_reward_mode"] = (real_loop, "runtime.run_lifetime present" if real_loop else
                               "runtime.run_lifetime absent: the run_steps delegation implements observation_only only")
    out["goal_free"] = (bool(sig and "goal_free" in sig.parameters),
                        "run_lifetime(goal_free=...) " + ("accepted" if sig and "goal_free" in sig.parameters else
                                                          "not available"))
    try:
        from ..learn import integrated as LI
        hook = getattr(LI, "IntegratedLearner", None)
    except Exception:
        hook = None
    out["meta_step"] = (hook is not None and callable(getattr(hook, "meta_step", None)),
                        "learn.integrated.IntegratedLearner.meta_step " + ("present" if hook is not None and
                                                                           callable(getattr(hook, "meta_step", None))
                                                                           else "absent"))
    out["credit"] = (hook is not None and "credit" in getattr(hook, "__init__").__code__.co_varnames
                     if hook is not None else False, "IntegratedLearner(credit=...) "
                     + ("accepted" if hook is not None and "credit" in hook.__init__.__code__.co_varnames else "absent"))
    out["curriculum"] = (hook is not None and "curriculum" in hook.__init__.__code__.co_varnames
                         if hook is not None else False, "IntegratedLearner(curriculum=...) "
                         + ("accepted" if hook is not None and "curriculum" in hook.__init__.__code__.co_varnames
                            else "absent"))
    return {k: (bool(v[0]), v[1]) for k, v in out.items()}


# ============================================================================================ one condition
@dataclass
class MetaSpec:
    exp: str
    condition: str
    run_dir: str
    preset: str = "dom_dev"
    seed: int = 0
    set: str = "dev"
    smoke: bool = False
    n_envs: int = 2
    decisions: int = 16
    update_every: int = 4
    device: str = "cuda"
    profile_gpu: bool = True
    families: tuple = ()
    agent_kw: dict = field(default_factory=dict)
    legacy_decision_path: bool = False     # declared: run_lifetime(require_cam_eval=False), recorded as a deviation


PROBE_OPTIMUM = 10.0     # ProbeEnv: test, then launch only when good (spec Exp J; plan §5 E3)


class ProbeLabels:
    """LabelSource over a capabilities ProbeEnv (operator side): one verified label per finished episode and row,
    utility = episode score / PROBE_OPTIMUM from the environment's own ScoreEvaluator (never shown to the agent)."""

    def __init__(self, env):
        from ..capabilities.envs import ScoreEvaluator
        self.env, self.ev, self.seen, self.episode = env, ScoreEvaluator(env), set(), 0

    def take_labels(self) -> list:
        from ..envs.labels import LabelRecord
        out = []
        done = self.env.done() if callable(getattr(self.env, "done", None)) else []
        for b, d in enumerate(done):
            key = (b, self.episode)
            if d and key not in self.seen:
                self.seen.add(key)
                score = float(self.ev.evaluate_goal(b).score)
                out.append(LabelRecord(env_id=f"probe#{b}", episode=self.episode, status="success" if score >= 5
                                       else "failure", utility=score / PROBE_OPTIMUM, value=score,
                                       t_resolved_days=None, evaluator="capabilities.ScoreEvaluator"))
        if done and all(done):
            self.episode += 1
        return out

    def utilities(self) -> list:
        return [{"env_id": f"probe#{b}", "family": "probe", "utility": float(self.ev.evaluate_goal(b).score) /
                 PROBE_OPTIMUM, "resolved": bool(d)} for b, d in enumerate(self.env.done())]


def _worlds(cond: Cond, spec: MetaSpec) -> tuple:
    if cond.worlds == "integrity":
        return tuple(spec.families or INTEGRITY_WORLDS), "integrity"
    split = "val" if (cond.worlds == "val" or spec.set == "confirm") else "train"
    return tuple(DS.families_for(split, spec.families or (DS.SMOKE_FAMILIES if split == "train" else None))), split


def run_meta_condition(spec: MetaSpec, semantic=None) -> dict:
    import torch
    from ..envs.labels import LabelStore
    from ..memory.checkpoint import model_arch_hash
    from ..observe.accounting import GpuBusy, TokenMeter
    cond = CONDITIONS[spec.condition]
    if cond.exp != spec.exp:
        raise ValueError(f"condition {cond.cid} belongs to experiment {cond.exp}, not {spec.exp}")
    if spec.set == "confirm":
        ok, why = DS.prereg_committed(PREREGS[spec.exp])
        if not ok:
            raise RuntimeError(f"confirm-set evaluation refused: {why}")
        if spec.smoke:
            raise ValueError("a smoke run is never a confirm evaluation")
    run_dir = os.path.abspath(os.path.join(spec.run_dir, cond.cid, f"seed{spec.seed}"))
    os.makedirs(run_dir, exist_ok=True)
    cv = DS.code_state()
    base = {"experiment": f"{EXPERIMENT_ID}/Exp{cond.exp}", "condition": cond.cid, "seed": spec.seed,
            "set": spec.set, "smoke": spec.smoke, "run_id": f"{EXPERIMENT_ID}-{cond.cid}-s{spec.seed}-"
                                                            f"{uuid.uuid4().hex[:8]}",
            "t_wall": time.time(), "code_commit": cv["commit"], "code_dirty": cv["dirty"], "preset": spec.preset,
            "compliant": cond.compliant, "description": cond.description}
    pre = prerequisites(spec.preset)
    unmet = {r: pre.get(r, (False, "unknown prerequisite"))[1] for r in cond.requires if not pre.get(r, (False,))[0]}
    if unmet:
        row = {**base, "status": "BLOCKED", "blocked_by": unmet, "reward_mode": cond.reward_mode,
               "metrics": {}, "deviations": []}
        DS.append_jsonl(os.path.join(os.path.abspath(spec.run_dir), "eval", "provenance.jsonl"), row)
        return row
    if cond.cid == "E-frozen":
        r = DS.RunSpec(condition="A", run_dir=os.path.join(spec.run_dir, cond.cid), preset=spec.preset,
                       seed=spec.seed, set="confirm" if spec.set == "confirm" else "dev", smoke=spec.smoke,
                       families=tuple(spec.families), n_envs=spec.n_envs, decisions=spec.decisions,
                       device=spec.device, profile_gpu=spec.profile_gpu, experiment=base["experiment"])
        return DS.run_condition(r, semantic=semantic)
    worlds, split = _worlds(cond, spec)
    st = DS.build_stack(spec.preset, os.path.join(run_dir, "target"), device=spec.device, seed=spec.seed,
                        overrides=dict(cond.overrides), semantic=semantic, agent_kw=spec.agent_kw)
    if cond.reward_mode == "observation_only" and st.cfg.reward_mode != "observation_only":
        raise RuntimeError(f"compliant conditions run observation_only; the preset has {st.cfg.reward_mode}")
    seed0 = (DS.CONFIRM_SEED0 if spec.set == "confirm" else DS.DEV_SEED0) + 1000 * spec.seed + 50_000
    if cond.worlds == "probe":
        from ..capabilities.envs import ProbeEnv
        adapter = ProbeEnv(spec.n_envs, seed=seed0, horizon=3, informative=cond.cid != "J-uninformative")
        labels = ProbeLabels(adapter)
        operator, env_src, fam_list = labels, "capabilities.envs.ProbeEnv", ["probe"] * spec.n_envs
    else:
        adapter, operator, labels, env_src, fam_list = DS.make_world(worlds, seed0, spec.n_envs,
                                                                     split if split != "integrity" else "train")
    store = LabelStore(os.path.join(run_dir, "labels"))
    run_lifetime, loop = DS.resolve_loop()
    deviations = []
    if loop != "runtime.run_lifetime":
        deviations.append(f"canonical loop: {loop}")
    if not st.builder.startswith("build.build_agent"):
        deviations.append(f"factory: {st.builder}")
    if "require_cam_eval" in DS.loop_kw(loop, spec.legacy_decision_path):
        deviations.append(DS.LEGACY_DECISION_DEVIATION)
    if not pre["rpsi_real"][0]:
        deviations.append(f"actor reward model: {pre['rpsi_real'][1]}")
    tm = TokenMeter()
    lm = getattr(st.semantic, "lm", None)
    if lm is not None:
        tm.attach(lm)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with GpuBusy(enabled=spec.profile_gpu) as gb:
        res = run_lifetime(st.agent, adapter, labels=labels, label_store=store, reward_mode=cond.reward_mode,
                           learner=st.learner if cond.learn else None, days=None,
                           max_decisions=DS.rounds_for(spec.decisions, spec.n_envs),
                           update_every=spec.update_every if cond.learn else 0, think_s=None, checkpoint=None,
                           on_step=None, **DS.loop_kw(loop, spec.legacy_decision_path))
    tm.detach()
    stats = DS._lifetime_stats(res)
    utils = operator.utilities() if isinstance(operator, ProbeLabels) else DS.operator_utilities(operator)
    us = [u["utility"] for u in utils if isinstance(u.get("utility"), (int, float))]
    metrics = {"verified_utility_end_mean": sum(us) / len(us) if us else None, "labels_joined": len(store),
               "worlds_resolved": sum(1 for u in utils if u.get("resolved"))}
    if cond.worlds == "integrity":
        traps = []
        for i in range(len(getattr(operator, "worlds", []))):
            try:
                v = operator.private_verify(i)["verify"]
                traps.append({k: v.get(k) for k in v if k not in ("outcome",)})
            except Exception as e:
                traps.append({"error": repr(e)[:120]})
        metrics["trap_records"] = len(traps)
        _write(os.path.join(run_dir, "integrity_private.json"), {"traps": traps, "note": "operator-side, private"})
    counts = st.model.count_parameters() if hasattr(st.model, "count_parameters") else {}
    row = {**base, "status": "RAN", "t_wall": time.time(), "loop": loop, "builder": st.builder,
           "env_source": env_src, "cfg_hash": DS.cfg_hash(st.cfg),
           "arch_hash": model_arch_hash(st.model, getattr(st.agent, "weights_exclude", ())), "ckpt": None,
           "ckpt_sha256": None, "reward_mode": st.cfg.reward_mode, "semantic_model": st.cfg.sem_model,
           "params": {"cam_total": counts.get("cam_total"), "cam_trainable": counts.get("cam_trainable")},
           "families": fam_list, "world_seed0": seed0, "n_envs": spec.n_envs, "learning": cond.learn,
           "decisions": stats.get("decisions"), "env_steps": stats.get("env_steps"), "updates": stats.get("updates"),
           "gpu_busy_s": gb.busy_s, "gpu_span_s": stats.get("gpu_span_s"), "semantic_tokens": tm.tokens,
           "wall_s": time.time() - t0, "peak_gpu_mb": stats.get("peak_gpu_mb"),
           "integrity": DS.integrity_scan(st.archive, st.cfg, store), "metrics": metrics,
           "condition_detail": {"overrides": dict(cond.overrides), "requires": list(cond.requires),
                                "prerequisites": {k: v[1] for k, v in pre.items()}},
           "deviations": deviations, "utilities": utils}
    DS._emit(DS.RunSpec(condition=cond.cid, run_dir=spec.run_dir, seed=spec.seed, set=spec.set,
                        experiment=base["experiment"]), row, run_dir, fam_list, getattr(st.agent, "version", None))
    return row


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="conditions and the prerequisites met at this commit")
    r = sub.add_parser("run")
    r.add_argument("--exp", required=True, choices=sorted({c.exp for c in CONDITIONS.values()}))
    r.add_argument("--condition", required=True, help="comma-separated condition ids, or 'all'")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--preset", default="dom_dev")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--set", default="dev", choices=("dev", "confirm"))
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--n-envs", type=int, default=2)
    r.add_argument("--decisions", type=int, default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--no-profile", action="store_true")
    r.add_argument("--dashboard", action="store_true")
    r.add_argument("--legacy-decision-path", action="store_true",
                   help="declare the legacy decision path: run_lifetime(require_cam_eval=False), recorded as a deviation")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        pre = prerequisites()
        for c in CONDITIONS.values():
            unmet = [x for x in c.requires if not pre[x][0]]
            print(f"{c.cid:16s} Exp{c.exp} {'RUNNABLE' if not unmet else 'BLOCKED by ' + ','.join(unmet):40s} "
                  f"{'' if c.compliant else '[non-compliant reference] '}{c.description}")
        for k, (ok, why) in pre.items():
            print(f"  prerequisite {k}: {'met' if ok else 'unmet'} ({why})")
        return 0
    ids = [c.cid for c in CONDITIONS.values() if c.exp == a.exp] if a.condition == "all" else a.condition.split(",")
    for cid in ids:
        spec = MetaSpec(exp=a.exp, condition=cid, run_dir=a.run_dir, preset=a.preset, seed=a.seed, set=a.set,
                        smoke=a.smoke, n_envs=a.n_envs, decisions=a.decisions or (6 * a.n_envs if a.smoke else 256),
                        device=a.device, profile_gpu=not a.no_profile, legacy_decision_path=a.legacy_decision_path)
        row = run_meta_condition(spec)
        print(json.dumps({k: row.get(k) for k in ("experiment", "condition", "status", "blocked_by", "loop",
                                                  "decisions", "gpu_busy_s", "semantic_tokens", "metrics")},
                         default=str))
    if a.dashboard:
        from ..observe.dashboard import build_dashboard
        print(build_dashboard(a.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
