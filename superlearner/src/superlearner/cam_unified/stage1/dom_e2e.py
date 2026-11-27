"""W1 real end-to-end check (plan 2026-11-26 §3 W1): `build.build_agent(preset)` on real Crucible rapid training
worlds plus real ow_tiny OpenWorld worlds, one batch through the canonical loop `runtime.run_lifetime`,
observation-only, labels to the learner-side LabelStore, learner updates through the LearnerHook.

    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock python -m superlearner.cam_unified.stage1.dom_e2e \\
        --preset dom --rapid 16 --ow 2 --rounds 16 --update-every 8 --run DIR --out FILE.json

Reported (`run_dom_e2e`): peak GPU memory (acting, each update); decide time per round at B (agent.act on real
observations and tools); each update's wall time, decisions consumed and losses; the core hash before / after; reward
delivery (every Transition.rewards is None, every record reward 0); version stamps; how many executed actions carry
a cam_eval stamp (the authorisation assertion is armed by the config; `require_cam_eval=False` is a declared legacy
decision path); labels appended; an integrity canary scan of everything the actor side wrote (archive records'
observations / actions / results and the decision traces) for the private world identities (rapid family names,
"family#seed", "ow_tiny"): content hits, and separately hits in the adapter-assigned id fields (env_id, goal_id).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch

GB = 2 ** 30


class _TimedLearner:
    """LearnerHook wrapper that times each step and records its peak GPU memory and the decisions it consumed."""

    def __init__(self, inner):
        self.inner = inner
        self.reward_model = getattr(inner, "reward_model", None)
        self.log: list = []

    def on_step(self, tr) -> None:
        self.inner.on_step(tr)

    def step(self) -> dict:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        try:
            out = self.inner.step()
        except torch.OutOfMemoryError as e:
            out = {"updated": False, "error": "OutOfMemoryError", "detail": str(e)[:300]}
        torch.cuda.synchronize()
        self.log.append({"s": time.perf_counter() - t, "peak_GB": torch.cuda.max_memory_allocated() / GB,
                         "updated": bool(out.get("updated")), "error": out.get("error"),
                         "decisions": len(out.get("consumed_decisions") or []), "reason": out.get("reason"),
                         "losses": out.get("losses"), "version": out.get("version"),
                         # IntegratedLearner (W7) UpdateInfo: applied / measured core-gradient shares, the DOM-08
                         # table and the per-objective group norms (present on probed updates)
                         "core_share": out.get("core_share"), "core_share_measured": out.get("core_share_measured"),
                         "dom08": out.get("dom08"), "share_guard": out.get("share_guard"),
                         "grad_norm_by_objective": (out.get("grad_table") or {}).get("norm"),
                         "grad_absent": (out.get("grad_table") or {}).get("absent"),
                         "reward": out.get("reward"), "meta": out.get("meta"), "labels": out.get("labels"),
                         "policy": {k: (out.get("policy") or {}).get(k) for k in (
                             "n_fresh_steps", "n_stale_steps", "kl", "clip_frac", "tool_entropy")},
                         "clipped": out.get("clipped"), "update": out.get("update"), "integrity": out.get("integrity")})
        return out

    def meta_step(self) -> dict:
        return self.inner.meta_step() if hasattr(self.inner, "meta_step") else {"updated": False}

    def state_dict(self):
        return self.inner.state_dict() if hasattr(self.inner, "state_dict") else None

    def load_state_dict(self, d) -> None:
        if hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(d)


ID_KEYS = ("env_id", "env_ids", "goal_id")       # adapter-assigned identifiers (reported separately)


def _strip_ids(x):
    if isinstance(x, dict):
        return {k: _strip_ids(v) for k, v in x.items() if k not in ID_KEYS}
    if isinstance(x, list):
        return [_strip_ids(v) for v in x]
    return x


def canary_scan(system, canaries: list) -> dict:
    """Occurrences of private strings in what the actor side wrote: archived records (observations, action, result,
    predictions, extra) and decision traces, with env-id fields removed (reported separately: `env_id_hits`)."""
    hits: dict = {}
    id_hits: dict = {}
    n_rec = 0
    system.archive.flush()
    for part in system.archive.scan(batch_size=512):
        for r in part:
            n_rec += 1
            d = r.to_dict()
            for k in ("env_id", "goal_id"):
                if any(c in str(d.get(k) or "") for c in canaries):
                    id_hits[k] = id_hits.get(k, 0) + 1
            blob = json.dumps(_strip_ids(d), default=str)
            for c in canaries:
                if c in blob:
                    hits.setdefault(f"archive:{c}", 0)
                    hits[f"archive:{c}"] += 1
    system.recorder.flush()
    n_tr = 0
    tdir = os.path.join(system.run_dir, "traces")
    for root, _, files in os.walk(tdir):
        for f in files:
            if not f.endswith((".jsonl", ".json")):
                continue
            for line in open(os.path.join(root, f)):
                line = line.strip()
                if not line:
                    continue
                n_tr += 1
                try:
                    blob = json.dumps(_strip_ids(json.loads(line)), default=str)
                except ValueError:
                    blob = line
                for c in canaries:
                    if c in blob:
                        hits.setdefault(f"traces:{c}", 0)
                        hits[f"traces:{c}"] += 1
    return {"canaries": len(canaries), "records_scanned": n_rec, "trace_lines_scanned": n_tr, "content_hits": hits,
            "content_clean": not hits, "id_field_hits": id_hits, "clean": not hits and not id_hits}


def intent_permutation_kl(model, inp, perm=None) -> dict:
    """KL(pi_tool(. | I_b) || pi_tool(. | I_perm(b))) per row of stored decision inputs `inp` (StepInputs of one
    round): the generator pointer's tool distribution with the row's own intent tokens vs. another row's (a cyclic
    derangement by default; observations, tools and R_t held fixed; a lower bound on the action KL, the tool being a
    marginal of the action). The identity "permutation" is the control (recompute noise floor). DOM intent mode
    only (cfg.n_intent > 0)."""
    with torch.no_grad():
        lp, allowed, lists, I = model.tool_distribution(inp)
        B = int(I.shape[0])
        if perm is None:
            perm = torch.roll(torch.arange(B, device=I.device), 1)
        lpp, _, _, _ = model.tool_distribution(inp, lambda X: X[perm])
        lpi, _, _, _ = model.tool_distribution(inp, lambda X: X)
    lp, lpp, lpi = lp.float(), lpp.float(), lpi.float()

    def kl(a, b):
        m = torch.isfinite(a) & torch.isfinite(b)
        return (a.exp() * (a - b)).masked_fill(~m, 0.0).sum(-1)
    k, k0 = kl(lp, lpp), kl(lp, lpi)
    dI = (I.float() - I.float()[perm]).flatten(1).norm(dim=-1)
    return {"B": B, "perm": [int(x) for x in perm.tolist()], "n_tools": [len(t) for t in lists],
            "kl_tool_mean": float(k.mean()), "kl_tool_min": float(k.min()), "kl_tool_max": float(k.max()),
            "kl_tool_per_row": [float(x) for x in k.tolist()],
            "kl_identity_control_max": float(k0.abs().max()),
            "intent_row_distance_mean": float(dI.mean()), "intent_norm_mean": float(I.float().flatten(1).norm(dim=-1).mean())}


def run_dom_e2e(preset: str = "dom_dev", *, n_rapid: int = 4, n_ow: int = 1, rounds: int = 16, update_every: int = 8,
                run_dir: Optional[str] = None, seed: int = 0, require_cam_eval: Optional[bool] = None,
                overrides: Optional[dict] = None, fresh_max_steps: Optional[int] = None,
                sem_model: Optional[str] = None, semantic=None, intent_kl: bool = True) -> dict:
    """See the module docstring. `require_cam_eval` None = armed by the config (raises ActionNotAuthorised on the
    first unstamped action); False = declared legacy decision path (counted, not refused)."""
    import tempfile
    from ..build import ConcatLabels, OperatorLabels, SystemConfig, build_agent, describe
    from ..core.bench import rapid_lever_specs
    from ..envs.experience import ExperienceOperator
    from ..decision.select import verify_stamp
    from ..runtime import ConcatEnv, GuardedEnv, cam_eval_stamp, run_lifetime
    from .harness_agent import core_hash
    run_dir = os.path.abspath(run_dir or tempfile.mkdtemp(prefix=f"dom_e2e_{preset}_"))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sc = SystemConfig(deterministic=False, grad_probe_every=1, archive_autoflush=False,
                      fresh_max_steps=fresh_max_steps or (update_every + 1), checkpoint_semantic_cache=False)
    over = {"grad_checkpoint": True, **dict(overrides or {})}
    sysm = build_agent(preset, env_kind="mixed", device="cuda", run_dir=run_dir, overrides=over, system=sc,
                       seed=seed, sem_model=sem_model, semantic=semantic)
    out: dict = {"preset": preset, "run_dir": run_dir, "build_s": time.perf_counter() - t0,
                 "build_GB": torch.cuda.memory_allocated() / GB, "overrides": over,
                 "fresh_max_steps": sc.fresh_max_steps}
    d = describe(sysm)
    out["parameters"] = {"cam_trainable": d["parameters"]["cam_trainable"], "cam_total": d["parameters"]["cam_total"],
                         "core": d["parameters"]["components"]["core"]["total"],
                         "semantic": d["semantic_model"], "trainer_groups": d["trainer"]["groups"]}
    specs = rapid_lever_specs(n_rapid, seed0=2000 + seed)
    op = ExperienceOperator(specs)
    parts, srcs = [op.adapter()], [OperatorLabels(op, evaluator="experience")]
    canaries = sorted({s.name.split(":", 1)[1] for s in specs} | {f"{s.name}#{s.seed}" for s in specs})
    ow_ad = ow_ev = None
    if n_ow:
        from crucible.research import llm_eval as L
        from ..envs.openworld import open_world
        from ..envs.openworld_eval import OpenWorldEvaluator
        goal = L.open_economy_goal(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=8,
                                   window_days=3, revenue_cents=10_000, requests=100, sustain_checks=2,
                                   endowment=3_000_000)
        ow_ad, ow_op = open_world([goal] * n_ow, seed=seed)
        ow_ev = OpenWorldEvaluator(ow_op)
        parts.append(ow_ad)
        if hasattr(ow_ev, "take_labels"):
            srcs.append(ow_ev)
        canaries.append("ow_tiny")
    env = GuardedEnv(ConcatEnv(parts) if len(parts) > 1 else parts[0])
    agent = sysm.agent
    act_s, peaks, seen = [], [], []
    act0 = agent.act

    def timed_act(obs, tools):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        a = act0(obs, tools)
        torch.cuda.synchronize()
        act_s.append(time.perf_counter() - t)
        peaks.append(torch.cuda.max_memory_allocated() / GB)
        return a
    dec_timings: list = []
    _dec0 = sysm.model.decide

    def _timed_decide(*a_, **k_):
        d_ = _dec0(*a_, **k_)
        dec_timings.append({k: float(v) for k, v in (getattr(d_, "timings", None) or {}).items()
                            if isinstance(v, (int, float))})
        return d_
    sysm.model.decide = _timed_decide
    agent.act = timed_act
    learner = _TimedLearner(sysm.learner)
    h0 = core_hash(sysm.model)
    kl_state: dict = {}

    def on_step(s_, tr):
        seen.append(tr)
        if intent_kl and "before" not in kl_state and int(getattr(sysm.cfg, "n_intent", 0) or 0) > 0 and agent.fresh:
            inp = agent.fresh[-1].inputs                 # the round just decided, before any learner update
            try:
                kl_state["before"] = {"agent_version": int(agent.version), "round": len(seen),
                                      **intent_permutation_kl(sysm.model, inp)}
                kl_state["inp"] = inp
            except Exception as e:                       # reported, never silently passed
                kl_state["before"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    t1 = time.perf_counter()
    try:
        life = run_lifetime(agent, env, labels=ConcatLabels(srcs), label_store=sysm.label_store, learner=learner,
                            max_decisions=rounds, update_every=update_every, require_cam_eval=require_cam_eval,
                            transcript=os.path.join(run_dir, "transcript.jsonl"), on_step=on_step)
    finally:
        agent.act = act0
        sysm.model.__dict__.pop("decide", None)
    out["lifetime_s"] = time.perf_counter() - t1
    h1 = core_hash(sysm.model)
    acts = [r.action for tr in seen for r in tr.records if r.action is not None]
    out.update({
        "B": env.B, "rounds": life.steps, "decisions": life.compute["decisions"], "reward_mode": life.reward_mode,
        "rewards_none_every_round": all(tr.rewards is None for tr in seen),
        "record_rewards_zero": all(r.reward == 0.0 for tr in seen for r in tr.records),
        "operator_returns": life.returns, "labels_taken": life.labels_taken, "labels_appended": life.labels_appended,
        "versions": life.versions, "authorisation": life.authorisation,
        "actions_stamped": sum(cam_eval_stamp(a) is not None for a in acts), "actions": len(acts),
        "actions_stamp_verified": sum(bool(verify_stamp(a)[0]) for a in acts),
        "decide_s": act_s, "decide_s_warm_median": sorted(act_s[1:])[len(act_s[1:]) // 2] if len(act_s) > 1 else None,
        "decide_peak_GB": max(peaks) if peaks else None, "decide_phase_s": dec_timings, "updates": learner.log,
        "n_updates": sum(u["updated"] for u in learner.log), "core_hash_before": h0, "core_hash_after": h1,
        "core_hash_changed": h0 != h1, "phase_wall_s": life.accounting["phase_wall_s"],
        "by_tool": life.accounting["by_tool"], "dropped": life.dropped,
        "agent_version": int(agent.version), "env_exceptions": list(env.errors)})
    if intent_kl:
        out["intent_permutation_kl"] = {"before_training": kl_state.get("before")}
        if "inp" in kl_state:                            # the same stored inputs after the updates
            try:
                out["intent_permutation_kl"]["after_training_same_inputs"] = {
                    "agent_version": int(agent.version), **intent_permutation_kl(sysm.model, kl_state["inp"])}
            except Exception as e:
                out["intent_permutation_kl"]["after_training_same_inputs"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    out["canary_scan"] = canary_scan(sysm, canaries)
    out["peak_GB"] = max([out["decide_peak_GB"] or 0.0] + [u["peak_GB"] for u in learner.log])
    if ow_ad is not None:
        ow_ad.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="dom_dev")
    ap.add_argument("--rapid", type=int, default=4)
    ap.add_argument("--ow", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--update-every", type=int, default=8)
    ap.add_argument("--fresh-max-steps", type=int, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-cam-eval", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args(argv)
    over = {}
    for kv in a.override:
        k, v = kv.split("=", 1)
        try:
            over[k] = json.loads(v)
        except ValueError:
            over[k] = v
    req = {"auto": None, "on": True, "off": False}[a.require_cam_eval]
    r = run_dom_e2e(a.preset, n_rapid=a.rapid, n_ow=a.ow, rounds=a.rounds, update_every=a.update_every,
                    run_dir=a.run, seed=a.seed, require_cam_eval=req, overrides=over,
                    fresh_max_steps=a.fresh_max_steps)
    txt = json.dumps(r, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
