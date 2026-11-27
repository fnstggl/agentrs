"""cam_live.reward.diagnose — debug-scale evidence for the reward workstream on the real model.

    PYTHONPATH=cam_live/src .venv/bin/python -m cam_live.reward.diagnose --out cam_live/docs/evidence/C
    ... --gpu-cost          additionally measures a meta-update at the 50m preset on the GPU (shared lock)

Everything runs on ``CamLiveConfig.debug()`` (tiny random backbone, debug core) in LabWorld, a diagnostic world with
known ground truth. It checks mathematics and plumbing; it is not evidence about any real task, and nothing here is
live experience (records carry ``provenance="diagnostic"``).

Sections written to ``diagnostics.json``:
    meta_gradient    finite-difference agreement of the meta-gradient on the real adaptable subset, K = 1, 2, 4;
                     exact zeros when the reward or theta_K is detached; the inner update changes the policy
    psi_isolation    the inner loss and the real policy update leave psi bit-identical
    experiment       many short LabWorld lifetimes: bootstrap-only vs learned reward with meta-updates vs
                     verified-outcome-only policy gradient (same worlds, same number of decisions)
    integrity        noisy gauge, fabricated / unsigned / duplicated / imagined outcomes, reward-magnitude scaling
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..config import CamLiveConfig
from ..contracts import (ActionSpec, ExperienceRecord, GoalStatus, OutcomeObservation, OutcomeQuery, OutcomeStatus,
                         SimClock, Trust, VerifiedOutcome)
from ..diagnostics.collect import LabCollector, Lifetime, uniform_policy, verified_return
from ..diagnostics.memstore import MemoryStore
from ..diagnostics.world import LabConfig, LabVerifier, LabWorld, lab_task
from .inner import GroupClip, InnerBatch, InnerConfig, POLICY_PREFIX, TRUNK_PREFIX
from .integrity import params_hash
from .meta import MetaGradient, QueryData, finite_difference, flat
from .metalearner import MetaConfig, MetaLearner

DAY = 86400.0
#: short worlds: 16 decisions, uniform behaviour reaches the target in ~30 % of worlds, "only run interventions"
#: in ~55 %, "prepare the last channel, then run interventions" in ~90 % (``scripted_baselines``)
LAB = dict(n_channels=4, horizon_s=1.5 * DAY, target=35, budget=100.0, mean_delay_s=(600.0, 5400.0),
           outcome_horizon_s=2 * 3600.0)
KINDS = ("bootstrap", "learned", "verified_pg")


@dataclass
class ExperimentConfig:
    n_lifetimes: int = 320
    n_decisions: int = 16
    update_every: int = 8            # decisions between policy updates (support block; the next block is its query)
    inner_lr: float = 0.1
    inner_steps: int = 2
    lr_reward: float = 3e-3
    min_independent: int = 4         # lifetimes (independent signed outcomes) per meta-update
    entropy_coef: float = 0.1
    mix_step: float = 0.25           # "mixed" learner: promotion step of the bootstrap mix
    mix_tolerance: float = 0.01      # ... non-inferiority margin of the promotion rule (verified-utility units)
    heldout: float = 0.25            # ... fraction of independent groups reserved for promotion evidence
    reward_hidden: int = 64
    reward_life: int = 16
    policy_hidden: int = 64
    max_context: int = 40
    lab: Dict[str, Any] = dataclasses.field(default_factory=lambda: dict(LAB))


# ================================================================================================= building blocks
class LabMeasurementAuth:
    """Authentication of LabWorld verifier measurements by asking the verifier again (LabWorld has no signature on
    ``OutcomeObservation``; change request C-1). Only observations whose query this object was told about, and
    whose value the verifier reproduces, pass."""

    def __init__(self) -> None:
        self._q: Dict[str, Tuple[OutcomeQuery, LabVerifier]] = {}

    def register(self, record: ExperienceRecord, verifier: LabVerifier) -> None:
        for q in list(record.pending_queries) + (list(record.result.expected_outcomes) if record.result else []):
            self._q[q.query_id] = (q, verifier)

    def __call__(self, o: OutcomeObservation) -> bool:
        hit = self._q.get(o.query_id)
        if hit is None or not o.verified:
            return False
        q, verifier = hit
        again = verifier.measure([q])[0]
        return again.status == o.status and again.value == o.value and again.verified


def make_stack(seed: int, ec: ExperimentConfig, *, mix: float = 1.0, mix_step: float = 0.0, heldout: float = 0.0,
               meta_mix: Optional[float] = 0.0, check_measurement: Optional[Callable[..., bool]] = None,
               **meta_over: Any) -> Tuple[Any, MemoryStore, SimClock, MetaLearner]:
    """Debug model + in-memory store + shared simulated clock + MetaLearner for LabWorld."""
    from ..nn.model import build_model
    cfg = CamLiveConfig.debug(seed=seed)
    model = build_model(cfg)
    store, clock = MemoryStore(), SimClock()
    key = LabVerifier(LabWorld(LabConfig(seed=0)))               # holds the verifier key; signs nothing here
    mc = MetaConfig.from_cfg(cfg, inner_lr=ec.inner_lr, inner_steps=ec.inner_steps, lr_reward=ec.lr_reward,
                             min_independent=ec.min_independent, heldout_fraction=heldout, mix=mix,
                             mixer_step=mix_step, mixer_min_independent=4, meta_mix=meta_mix,
                             reward_hidden=ec.reward_hidden, reward_life=ec.reward_life,
                             policy_hidden=ec.policy_hidden, provenance="diagnostic", **meta_over)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10_007 + seed)
        ml = MetaLearner(model, store, cfg, check_signature=key.check_signature,
                         check_measurement=check_measurement, clock=clock, meta_cfg=mc,
                         inner_cfg=InnerConfig(entropy_coef=ec.entropy_coef, half_life_s=DAY))
    return model, store, clock, ml


def model_policy(ml: MetaLearner, store: MemoryStore, generator: torch.Generator, theta: Optional[Dict[str, Tensor]] = None
                 ) -> Callable[[List[ActionSpec], Dict[str, Any]], Tuple[int, List[float]]]:
    """Behaviour policy for ``LabCollector``: sample from pi_theta(. | b, C) exactly and log every probability."""
    model = ml.model

    @torch.no_grad()
    def fn(cands: List[ActionSpec], info: Dict[str, Any]) -> Tuple[int, List[float]]:
        tok = model.tokenize([store.items(info["context_refs"])])
        b = model.core_forward(tok, None).b
        e_a, a_mask = model.encode_actions([cands])
        e_g = model.encode_goals([info["task"].objective])
        lat = InnerBatch(exp_ids=[], b=b, e_a=e_a, a_mask=a_mask, e_g=e_g,
                         chosen=torch.zeros(1, dtype=torch.long), behaviour_logp=torch.zeros(1, dtype=torch.float64),
                         ts=torch.zeros(1, dtype=torch.float64), z=model.decision_latent(b, e_a, e_g))
        p = ml.subset.forward(theta if theta is not None else ml.subset.live(), lat).probs[0, :len(cands)]
        p = p.to("cpu", torch.float64).clamp_min(0.0)
        p = p / p.sum()
        return int(torch.multinomial(p, 1, generator=generator)), [float(x) for x in p]
    return fn


def action_category(rec: ExperienceRecord, unlocked: bool, n_channels: int) -> str:
    a = rec.decision.chosen
    if a.operation == "lab.unlock":
        return "unlock_repeat" if unlocked else "unlock_first"
    if a.operation == "lab.act":
        if a.arguments.get("channel") == n_channels - 1:
            return "act_last_unlocked" if unlocked else "act_last_locked"
        return "act_other"
    return {"lab.probe": "probe", "lab.noise": "noise_gauge", "wait": "wait"}.get(a.operation, a.operation)


def categorise(records: Sequence[ExperienceRecord], n_channels: int) -> List[str]:
    out, unlocked = [], False
    for r in records:
        out.append(action_category(r, unlocked, n_channels))
        unlocked = unlocked or r.decision.chosen.operation == "lab.unlock"
    return out


CATEGORIES = ("unlock_first", "unlock_repeat", "act_last_unlocked", "act_last_locked", "act_other", "probe",
              "noise_gauge", "wait")


def reward_profile(ml: MetaLearner, seed: int, ec: ExperimentConfig, n_worlds: int = 24) -> Dict[str, Dict[str, float]]:
    """What R_psi pays for: mean learned reward by action category on held-out worlds under uniform behaviour,
    in units of the reward's standard deviation over those decisions, relative to the mean over all of them
    (only relative reward matters: the inner update standardises returns)."""
    store = MemoryStore()
    vals: Dict[str, List[float]] = {c: [] for c in CATEGORIES}
    boot: Dict[str, List[float]] = {c: [] for c in CATEGORIES}
    allv: List[float] = []
    allb: List[float] = []
    psi = {k: v.detach() for k, v in ml.reward.psi().items()}
    saved = ml.inner.life_states
    ml.inner.life_states = {}                                    # fresh lifetimes: zero lifetime context
    try:
        for i in range(n_worlds):
            world = LabWorld(LabConfig(seed=900_000 + seed * 1000 + i, **ec.lab))
            lt = LabCollector(store, max_context=ec.max_context).run(world, uniform_policy(seed * 7 + i),
                                                                     ec.n_decisions, seed=i)
            ib = ml.inner.build_batch(store, lt.records)
            with torch.no_grad():
                r = ml.inner.learned_reward(psi, ib).cpu().tolist()
            by_id = dict(zip(ib.exp_ids, r))
            by_boot = dict(zip(ib.exp_ids, ib.r_boot.cpu().tolist()))
            for rec, cat in zip(lt.records, categorise(lt.records, world.cfg.n_channels)):
                vals[cat].append(by_id[rec.exp_id])
                allv.append(by_id[rec.exp_id])
                boot[cat].append(by_boot[rec.exp_id])
                allb.append(by_boot[rec.exp_id])
    finally:
        ml.inner.life_states = saved
    mean = sum(allv) / len(allv)
    sd = math.sqrt(sum((x - mean) ** 2 for x in allv) / max(len(allv) - 1, 1)) or 1.0
    bmean = sum(allb) / len(allb)
    bsd = math.sqrt(sum((x - bmean) ** 2 for x in allb) / max(len(allb) - 1, 1)) or 1.0
    return {c: {"n": len(v), "raw_mean": (sum(v) / len(v)) if v else float("nan"),
                "relative": ((sum(v) / len(v) - mean) / sd) if v else float("nan"),
                "bootstrap_relative": ((sum(boot[c]) / len(boot[c]) - bmean) / bsd) if v else float("nan")}
            for c, v in vals.items()} | \
        {"_all": {"n": len(allv), "raw_mean": mean, "raw_std": sd, "relative": 0.0, "bootstrap_relative": 0.0}}


# ====================================================================================================== experiment
def run_learner(kind: str, seed: int, ec: ExperimentConfig, profile: bool = True) -> Dict[str, Any]:
    """One learner over ``ec.n_lifetimes`` LabWorld lifetimes (worlds depend on ``seed`` and the lifetime index only,
    so every learner of a seed meets the same worlds and takes the same number of decisions).

    bootstrap     policy updates on the labelled bootstrap reward only (mix fixed at 1, no meta-update)
    learned       policy updates on R_psi only (mix fixed at 0); R_psi meta-updated on signed lifetime outcomes
    mixed         the cold-start path: mix starts at 1, shadow meta-updates by importance weighting, promotion rule
    verified_pg   REINFORCE on the signed lifetime outcome at the end of each lifetime (no intrinsic reward)
    """
    torch.set_num_threads(1)
    t0 = time.time()
    mixes = {"bootstrap": (1.0, 0.0, 0.0), "learned": (0.0, 0.0, 0.0), "mixed": (1.0, ec.mix_step, ec.heldout),
             "verified_pg": (1.0, 0.0, 0.0)}
    mix, step, heldout = mixes[kind]
    model, store, clock, ml = make_stack(seed, ec, mix=mix, mix_step=step, heldout=heldout,
                                         mixer_tolerance=ec.mix_tolerance)
    gen = torch.Generator().manual_seed(77_000 + seed)
    act = model_policy(ml, store, gen)
    clip = GroupClip({POLICY_PREFIX: ml.mc.clip_policy, TRUNK_PREFIX: ml.mc.clip_trunk})
    achieved, utility, totals, mixes_seen, metas, cats = [], [], [], [], [], []
    u_mean, u_var = None, 0.05
    theta_start = ml.subset.snapshot()
    for L in range(ec.n_lifetimes):
        world = LabWorld(LabConfig(seed=100_000 * (seed + 1) + L, **ec.lab), clock=clock)
        ml.new_lifetime()
        buf: List[ExperienceRecord] = []

        def on_step(rec: ExperienceRecord) -> None:
            ml.internal_rewards([rec])
            buf.append(rec)
            if kind != "verified_pg" and len(buf) >= ec.update_every:
                ml.policy_update(list(buf))
                buf.clear()
        lt = LabCollector(store, max_context=ec.max_context, seq_start=L * 1000).run(
            world, act, ec.n_decisions, seed=seed * 100_003 + L, on_step=on_step,
            task=lab_task(world, objective_id=f"obj_lab_s{seed}_l{L}"))      # deterministic outcome-group keys
        if kind != "verified_pg" and len(buf) >= 2:
            ml.policy_update(list(buf))
        U = verified_return(lt)
        if kind == "verified_pg":
            if u_mean is None:
                u_mean = U
            adv = (U - u_mean) / math.sqrt(u_var + 1e-8)
            ib = ml.inner.build_batch(store, lt.records, policy_only=True)
            theta = ml.subset.snapshot()
            for _ in range(ec.inner_steps):
                th = {n: v.requires_grad_(True) for n, v in theta.items()}
                out = ml.subset.forward(th, ib)
                loss = -(out.log_prob(ib.chosen) * adv).mean() - ec.entropy_coef * out.entropy().mean()
                g = clip(list(th), torch.autograd.grad(loss, list(th.values()), allow_unused=True))
                theta = {n: (v.detach() if gi is None else (v - ec.inner_lr * gi).detach())
                         for (n, v), gi in zip(th.items(), g)}
            ml.subset.write(theta)
            u_var = 0.95 * u_var + 0.05 * (U - u_mean) ** 2
            u_mean = 0.95 * u_mean + 0.05 * U
        else:
            res = ml.admit_objective(lt.verified, utility=U)
            assert res.accepted, res.reason
            if kind in ("learned", "mixed"):
                m = ml.meta_update()
                if m.applied:
                    metas.append({"lifetime": L, "estimator": m.estimator, "grad_norm": m.meta_grad_norm,
                                  "n_independent": m.n_independent_outcomes, "mix": m.bootstrap_mix,
                                  "max_weight": m.max_importance_weight, "objective": m.outer_objective})
        achieved.append(float(lt.verified.status == GoalStatus.ACHIEVED))
        utility.append(U)
        totals.append(float(lt.verified.value))
        mixes_seen.append(ml.mixer.mix)
        cats.append(categorise(lt.records, world.cfg.n_channels))
    tail = cats[-max(1, ec.n_lifetimes // 4):]
    flat_tail = [c for row in tail for c in row]
    head = [c for row in cats[:max(1, ec.n_lifetimes // 4)] for c in row]
    freq = lambda rows: {c: rows.count(c) / max(len(rows), 1) for c in CATEGORIES}
    theta_end = ml.subset.snapshot()
    out: Dict[str, Any] = {
        "kind": kind, "seed": seed, "achieved": achieved, "utility": utility, "adopters": totals,
        "mix": mixes_seen, "meta_updates": metas, "n_meta_applied": len(metas),
        "n_meta_skipped": int(ml.stats["meta_skipped"]), "policy_updates": int(ml.stats["policy_updates"]),
        "action_frequency_first_quarter": freq(head), "action_frequency_last_quarter": freq(flat_tail),
        "theta_change_norm": math.sqrt(sum(float(((theta_end[n] - theta_start[n]) ** 2).sum()) for n in theta_end)),
        "ledger": {"entries": len(ml.ledger), "rejected": len(ml.ledger.rejected),
                   "chain_ok": ml.ledger.verify_chain()[0]},
        "seconds": time.time() - t0}
    if profile and kind in ("learned", "mixed"):
        out["reward_profile"] = reward_profile(ml, seed, ec)
    return out


def scripted_baselines(ec: ExperimentConfig, n_worlds: int = 300) -> Dict[str, Dict[str, float]]:
    """Reference points for the experiment's worlds (no model, no learning): fixed rules over operation types."""
    import random

    def rule(kind: str, seed: int) -> Callable[[List[ActionSpec], Dict[str, Any]], Tuple[int, List[float]]]:
        rng, state = random.Random(seed), {"unlocked": False}

        def fn(cands: List[ActionSpec], info: Dict[str, Any]) -> Tuple[int, List[float]]:
            ops = [c.operation for c in cands]
            p = [1.0 / len(cands)] * len(cands)
            if kind == "unlock_then_interventions" and not state["unlocked"] and "lab.unlock" in ops:
                state["unlocked"] = True
                return ops.index("lab.unlock"), p
            idx = list(range(len(cands))) if kind == "uniform" else \
                ([i for i, o in enumerate(ops) if o == "lab.act"] or list(range(len(cands))))
            return rng.choice(idx), p
        return fn
    out = {}
    for kind in ("uniform", "interventions_only", "unlock_then_interventions"):
        ach, tot = [], []
        for i in range(n_worlds):
            lt = LabCollector(MemoryStore(), max_context=ec.max_context).run(
                LabWorld(LabConfig(seed=i, **ec.lab)), rule(kind, i), ec.n_decisions, seed=i)
            ach.append(float(lt.verified.status == GoalStatus.ACHIEVED))
            tot.append(float(lt.verified.value))
        out[kind] = {"achieved": sum(ach) / n_worlds, "mean_adopters": sum(tot) / n_worlds, "n_worlds": n_worlds}
    return out


def _job(args: Tuple[str, int, Dict[str, Any]]) -> Dict[str, Any]:
    kind, seed, ecd = args
    return run_learner(kind, seed, ExperimentConfig(**ecd))


def _ci(values: Sequence[float]) -> Dict[str, float]:
    from ..predict.calibration import studentised_ci
    return {k: float(v) for k, v in studentised_ci(list(values), 0.95).items()}


def _verdict(ci: Dict[str, float]) -> str:
    if ci["lo"] > 0:
        return "helped"
    if ci["hi"] < 0:
        return "hurt"
    return "no detectable difference"


def summarise(runs: Sequence[Dict[str, Any]], ec: ExperimentConfig, block: int = 20) -> Dict[str, Any]:
    kinds = sorted({r["kind"] for r in runs}, key=lambda k: (KINDS + ("mixed",)).index(k))
    seeds = sorted({r["seed"] for r in runs})
    by = {(r["kind"], r["seed"]): r for r in runs}
    n = ec.n_lifetimes
    q = max(1, n // 4)
    curves: Dict[str, Any] = {}
    final: Dict[str, Dict[int, float]] = {}
    area: Dict[str, Dict[int, float]] = {}
    for k in kinds:
        blocks = []
        for s0 in range(0, n, block):
            per_seed = [sum(by[k, s]["achieved"][s0:s0 + block]) / len(by[k, s]["achieved"][s0:s0 + block])
                        for s in seeds]
            per_seed_u = [sum(by[k, s]["utility"][s0:s0 + block]) / len(by[k, s]["utility"][s0:s0 + block])
                          for s in seeds]
            blocks.append({"lifetimes": [s0, min(n, s0 + block)], "achieved": _ci(per_seed),
                           "verified_utility": _ci(per_seed_u)})
        curves[k] = blocks
        final[k] = {s: sum(by[k, s]["achieved"][-q:]) / q for s in seeds}
        area[k] = {s: sum(by[k, s]["achieved"]) / n for s in seeds}
    out: Dict[str, Any] = {"seeds": seeds, "kinds": kinds, "block": block, "curves": curves,
                           "final_quarter_achieved": {k: _ci(list(final[k].values())) for k in kinds},
                           "mean_over_all_lifetimes_achieved": {k: _ci(list(area[k].values())) for k in kinds}}
    comp: Dict[str, Any] = {}
    for a, b in (("learned", "bootstrap"), ("learned", "verified_pg"), ("bootstrap", "verified_pg"),
                 ("mixed", "bootstrap"), ("mixed", "learned")):
        if a in kinds and b in kinds:
            d_final = _ci([final[a][s] - final[b][s] for s in seeds])
            d_area = _ci([area[a][s] - area[b][s] for s in seeds])
            comp[f"{a}_minus_{b}"] = {"final_quarter": d_final, "final_quarter_verdict": _verdict(d_final),
                                      "all_lifetimes": d_area, "all_lifetimes_verdict": _verdict(d_area)}
    out["paired_differences"] = comp
    for k in kinds:
        out.setdefault("action_frequency_last_quarter", {})[k] = {
            c: sum(by[k, s]["action_frequency_last_quarter"][c] for s in seeds) / len(seeds) for c in CATEGORIES}
        out.setdefault("action_frequency_first_quarter", {})[k] = {
            c: sum(by[k, s]["action_frequency_first_quarter"][c] for s in seeds) / len(seeds) for c in CATEGORIES}
    for k in ("learned", "mixed"):
        if k in kinds:
            prof = [by[k, s]["reward_profile"] for s in seeds if "reward_profile" in by[k, s]]
            if prof:
                out.setdefault("reward_pays_for", {})[k] = {
                    c: _ci([p[c]["relative"] for p in prof if p[c]["n"] > 0]) for c in CATEGORIES}
                out["bootstrap_pays_for"] = {
                    c: _ci([p[c]["bootstrap_relative"] for p in prof if p[c]["n"] > 0]) for c in CATEGORIES}
            out.setdefault("meta", {})[k] = {
                "applied_per_seed": [by[k, s]["n_meta_applied"] for s in seeds],
                "skipped_per_seed": [by[k, s]["n_meta_skipped"] for s in seeds],
                "estimators": sorted({m["estimator"] for s in seeds for m in by[k, s]["meta_updates"]}),
                "final_mix_per_seed": [by[k, s]["mix"][-1] for s in seeds]}
    # where each learner ended up: concentration on something other than interventions, and the static gauge
    acts = ("act_last_unlocked", "act_last_locked", "act_other")
    gauge: Dict[str, Any] = {}
    for k in kinds:
        last = {s: by[k, s]["action_frequency_last_quarter"] for s in seeds}
        first = {s: by[k, s]["action_frequency_first_quarter"] for s in seeds}
        row: Dict[str, Any] = {
            "frequency_first_quarter": _ci([first[s]["noise_gauge"] for s in seeds]),
            "frequency_last_quarter": _ci([last[s]["noise_gauge"] for s in seeds]),
            "frequency_last_quarter_per_seed": {str(s): last[s]["noise_gauge"] for s in seeds},
            "seeds_with_gauge_above_25pct_at_end": [s for s in seeds if last[s]["noise_gauge"] > 0.25],
            "seeds_collapsed_on_a_non_intervention": [
                s for s in seeds if max(v for c, v in last[s].items() if c not in acts) > 0.4]}
        if k in ("learned", "mixed") and all("reward_profile" in by[k, s] for s in seeds):
            rel = {s: by[k, s]["reward_profile"]["noise_gauge"]["relative"] for s in seeds}
            row["learned_reward_relative_per_seed"] = {str(s): rel[s] for s in seeds}
            row["seeds_where_learned_reward_pays_gauge_above_average"] = [s for s in seeds if rel[s] > 0]
            row["seeds_where_gauge_is_paid_and_chosen_above_25pct"] = [
                s for s in seeds if rel[s] > 0 and last[s]["noise_gauge"] > 0.25]
        gauge[k] = row
    out["noisy_gauge"] = gauge
    statements = []
    names = {"learned": "learned reward from scratch (mix 0) with meta-updates",
             "mixed": "cold-start path (bootstrap mix with shadow meta-updates and the promotion rule)",
             "bootstrap": "bootstrap reward only", "verified_pg": "verified-outcome-only policy gradient"}
    for key_, c in comp.items():
        a, b_ = key_.split("_minus_")
        d = c["final_quarter"]
        statements.append(f"{names[a]} vs {names[b_]}: {c['final_quarter_verdict']} "
                          f"(difference in verified goal achievement over the last quarter of lifetimes "
                          f"{d['mean']:+.3f}, 95% interval [{d['lo']:+.3f}, {d['hi']:+.3f}], {int(d['n'])} seeds)")
    for k in ("learned", "mixed"):
        if k in gauge and "seeds_where_gauge_is_paid_and_chosen_above_25pct" in gauge[k]:
            bad = gauge[k]["seeds_where_gauge_is_paid_and_chosen_above_25pct"]
            statements.append(f"{names[k]}: the static gauge is paid above average by the learned reward and chosen "
                              f"in more than 25% of decisions at the end in {len(bad)} of {len(seeds)} seeds "
                              f"{bad}; seeds collapsed on a non-intervention: "
                              f"{gauge[k]['seeds_collapsed_on_a_non_intervention']}")
    out["statements"] = statements
    out["seconds_total"] = sum(r["seconds"] for r in runs)
    return out


def labworld_experiment(seeds: Sequence[int], ec: ExperimentConfig, kinds: Sequence[str] = KINDS,
                        workers: int = 0, block: int = 20) -> Dict[str, Any]:
    jobs = [(k, s, dataclasses.asdict(ec)) for s in seeds for k in kinds]
    if workers and workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            runs = list(ex.map(_job, jobs))
    else:
        runs = [_job(j) for j in jobs]
    out = summarise(runs, ec, block)
    out["config"] = dataclasses.asdict(ec)
    out["runs"] = [{k: v for k, v in r.items() if k not in ("adopters",)} for r in runs]
    return out


# ============================================================================================= single-lifetime rig
def lab_lifetime(ml: MetaLearner, store: MemoryStore, clock: SimClock, ec: ExperimentConfig, *, seed: int,
                 n_decisions: Optional[int] = None, update_every: Optional[int] = None,
                 auth: Optional[LabMeasurementAuth] = None, generator: Optional[torch.Generator] = None,
                 goal_free_rows: Sequence[int] = ()) -> Tuple[Lifetime, LabWorld, List[Any]]:
    """Act with the live policy in one LabWorld, rewarding every decision and updating the policy every
    ``update_every`` decisions (0: never). Returns (lifetime, world, learning-update records)."""
    world = LabWorld(LabConfig(seed=seed, **ec.lab), clock=clock)
    verifier = LabVerifier(world)
    gen = generator or torch.Generator().manual_seed(50_000 + seed)
    act = model_policy(ml, store, gen)
    every = ec.update_every if update_every is None else update_every
    buf: List[ExperienceRecord] = []
    updates: List[Any] = []

    def on_step(rec: ExperienceRecord) -> None:
        if len(buf) + len(updates) * max(every, 1) in goal_free_rows:
            rec.goal_text = ""                                   # a goal-free step (learned null goal)
            store.update(rec)
        if auth is not None:
            auth.register(rec, verifier)
        ml.internal_rewards([rec])
        buf.append(rec)
        if every and len(buf) >= every:
            updates.append(ml.policy_update(list(buf)))
            buf.clear()
    lt = LabCollector(store, max_context=ec.max_context, seq_start=int(ml.stats["policy_updates"]) * 1000 + 1).run(
        world, act, n_decisions or ec.n_decisions, seed=seed, on_step=on_step,
        task=lab_task(world, objective_id=f"obj_lab_{seed}"))
    return lt, world, updates


# ==================================================================================================== meta-gradient
def meta_gradient_check(Ks: Sequence[int] = (1, 2, 4), seed: int = 0, per_tensor: int = 4, eps: float = 1e-4,
                        inner_lr: float = 0.1) -> Dict[str, Any]:
    """Float64 check on the real model's adaptable subset (policy head + decision trunk of the debug model):
    autograd meta-gradient vs central finite differences of the outer surrogate, on stored LabWorld decisions.
    Two support rows are goal-free so that the null-goal parameter of R_psi is exercised. The utilities of the
    query rows are fixed constants here (this is a check of the derivative, not of learning)."""
    ec = ExperimentConfig()
    model, store, clock, ml = make_stack(seed, ec, mix=0.0)
    lt, _, _ = lab_lifetime(ml, store, clock, ec, seed=4242 + seed, update_every=0, goal_free_rows=(2, 5))
    sup, qry = lt.records[:8], lt.records[8:]
    D = torch.float64
    ib = ml.inner.build_batch(store, sup).to(D)
    ibq = ml.inner.build_batch(store, qry, policy_only=True).to(D)
    theta = ml.subset.snapshot(D)
    psi = {k: v.detach().clone().to(D).requires_grad_(True) for k, v in ml.reward.psi().items()}
    U = torch.linspace(-1.0, 1.0, ibq.n, dtype=D)
    clip = GroupClip({POLICY_PREFIX: ml.mc.clip_policy, TRUNK_PREFIX: ml.mc.clip_trunk})
    out: Dict[str, Any] = {"n_theta": ml.subset.n_params(), "n_psi": sum(p.numel() for p in psi.values()),
                           "n_support": ib.n, "n_query": ibq.n, "goal_free_support_rows": int((~ib.has_goal).sum()),
                           "eps": eps, "inner_lr": inner_lr, "dtype": "float64", "K": {}}
    names, sizes = list(psi), [p.numel() for p in psi.values()]
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)
    g_rng = torch.Generator().manual_seed(seed)
    coords = [offs[i] + int(j) for i, s in enumerate(sizes)
              for j in torch.randperm(s, generator=g_rng)[:per_tensor].tolist()]
    direction = torch.randn(offs[-1], dtype=D, generator=g_rng)
    direction = direction / direction.norm()
    for K in Ks:
        mg = MetaGradient(ml.inner.inner_loss, ml.inner.logp, inner_lr=inner_lr, K=K, grad_transform=clip)
        q = QueryData(batch=ibq, utility=U, baseline=0.0)

        def objective(p: Dict[str, Tensor]) -> float:
            tr = mg.inner_update(theta, p, [ib], create_graph=False)
            return float(mg.outer_surrogate(tr.detached(), q).surrogate)
        t0 = time.time()
        trace = mg.inner_update(theta, psi, [ib], create_graph=True)
        res = mg.meta_gradient(trace, q, "on_policy")
        seconds = time.time() - t0
        g = flat(res.grads)
        fd = finite_difference(objective, {k: v.detach() for k, v in psi.items()}, eps=eps, coords=coords)
        base = flat({k: v.detach() for k, v in psi.items()})

        def unflat(v: Tensor) -> Dict[str, Tensor]:
            return {n: v[offs[i]:offs[i + 1]].view(psi[n].shape) for i, n in enumerate(names)}
        dd_fd = (objective(unflat(base + eps * direction)) - objective(unflat(base - eps * direction))) / (2 * eps)
        dd = float(g @ direction)
        # the two ablations that must remove the pathway
        mg_det = MetaGradient(lambda th, p, b, theta_sg=None: ml.inner.functional(th, p, b, detach_reward=True),
                              ml.inner.logp, inner_lr=inner_lr, K=K, grad_transform=clip)
        res_det = mg_det.meta_gradient(mg_det.inner_update(theta, psi, [ib], create_graph=True), q, "on_policy")
        surr_stop = mg.outer_surrogate(trace.detached(), q).surrogate
        # what a plain ``v.detach()`` baseline inside the inner loss would give (the path theta_k -> V is lost)
        mg_naive = MetaGradient(lambda th, p, b: ml.inner.functional(th, p, b), ml.inner.logp, inner_lr=inner_lr,
                                K=K, grad_transform=clip)
        g_naive = flat(mg_naive.meta_gradient(mg_naive.inner_update(theta, psi, [ib], create_graph=True), q,
                                              "on_policy").grads)
        with torch.no_grad():
            p0, pK = ml.inner.probs(theta, ib), ml.inner.probs(trace.detached(), ib)
            pq0, pqK = ml.inner.probs(theta, ibq), ml.inner.probs(trace.detached(), ibq)
        per = {n: float(v.norm()) for n, v in res.grads.items()}
        out["K"][str(K)] = {
            "rel_error_sampled_coords": float((fd[coords] - g[coords]).norm() / g[coords].norm()),
            "max_abs_error_sampled_coords": float((fd[coords] - g[coords]).abs().max()),
            "n_coords": len(coords), "directional_autograd": dd, "directional_fd": dd_fd,
            "directional_rel_error": abs(dd - dd_fd) / max(abs(dd_fd), 1e-300),
            "grad_norm": res.grad_norm, "grad_norm_per_tensor": per,
            "every_tensor_nonzero": all(v > 0 for v in per.values()),
            "min_tensor_norm": min(per.values()),
            "grad_norm_reward_detached": res_det.grad_norm,
            "rel_error_if_value_baseline_plainly_detached": float((g_naive - g).norm() / g.norm()),
            "surrogate_has_graph_when_theta_K_detached": bool(surr_stop.requires_grad),
            "policy_prob_mean_abs_change_support": float((pK - p0).abs().sum(-1).mean()),
            "policy_prob_max_abs_change_support": float((pK - p0).abs().max()),
            "policy_prob_mean_abs_change_query": float((pqK - pq0).abs().sum(-1).mean()),
            "seconds_autograd": seconds}
    return out


# ==================================================================================================== psi isolation
def psi_isolation_check(seed: int = 0) -> Dict[str, Any]:
    """The policy-improvement loss (as a LossProvider on a graph-carrying batch, and as the real inner update)
    never moves psi."""
    from ..learning_api import encode_decisions
    from .integrity import depends_on
    ec = ExperimentConfig()
    model, store, clock, ml = make_stack(seed, ec, mix=0.5)
    lt, _, _ = lab_lifetime(ml, store, clock, ec, seed=777 + seed, update_every=0)
    h0 = params_hash(ml.reward)
    batch = encode_decisions(model, store, lt.records[:8], with_next=True, grad=True)
    loss, metrics = ml.inner.loss(model, batch, store)
    path = depends_on(loss, ml.reward)
    others = [p for g, ps in model.param_groups().items() if g not in ("reward", "backbone_frozen") for p in ps]
    opt = torch.optim.SGD(others, lr=0.05)
    opt.zero_grad()
    loss.backward()
    reached = sum(1 for p in ml.reward.parameters() if p.grad is not None and bool((p.grad != 0).any()))
    moved_before = params_hash(ml.subset.live())
    opt.step()
    h1 = params_hash(ml.reward)
    upd = ml.policy_update(lt.records[8:])
    h2 = params_hash(ml.reward)
    return {"loss": float(loss.detach()), "loss_has_grad_path_to_psi": bool(path),
            "psi_tensors_with_gradient_after_backward": reached,
            "psi_bit_identical_after_optimizer_step": h1 == h0,
            "psi_bit_identical_after_policy_update": h2 == h0,
            "theta_changed_by_optimizer_step": params_hash(ml.subset.live()) != moved_before,
            "policy_prob_mean_abs_change_policy_update": upd.prediction_change["policy_prob_mean_abs_change"],
            "policy_update_weight_delta": upd.weight_delta, "mix": metrics["mix"]}


# ======================================================================================================== integrity
def integrity_check(seed: int = 0) -> Dict[str, Any]:
    """Outcomes that must never reach the outer objective, and reward-magnitude scaling."""
    ec = ExperimentConfig(min_independent=2)
    auth = LabMeasurementAuth()
    model, store, clock, ml = make_stack(seed, ec, mix=0.0, check_measurement=auth)
    lts = []
    for i in range(3):                                           # three independent lifetimes
        ml.new_lifetime()
        lts.append(lab_lifetime(ml, store, clock, ec, seed=999 + seed + i, auth=auth)[:2])
    lt, world = lts[0]
    verifier = LabVerifier(world)
    good = lt.verified
    out: Dict[str, Any] = {}
    n0 = len(ml.ledger)
    forged = VerifiedOutcome.from_dict({**good.to_dict(), "value": good.value + 1000.0, "status": GoalStatus.ACHIEVED})
    unsigned = VerifiedOutcome.from_dict({**good.to_dict(), "signature": ""})
    made_up = VerifiedOutcome(objective_id=good.objective_id, status=GoalStatus.ACHIEVED, value=1e6, target=1.0,
                              evaluated_at=clock.now(), signature="0" * 64)
    attempts = {"tampered_value_with_old_signature": ml.admit_objective(forged, utility=1e6).reason,
                "unsigned": ml.admit_objective(unsigned, utility=1e6).reason,
                "fabricated_signature": ml.admit_objective(made_up, utility=1e6).reason,
                "self_authored_objective": ml.admit_objective(good, utility=1e6, objective_origin=Trust.SELF).reason}
    out["ledger_entries_after_invalid_objectives"] = len(ml.ledger) - n0
    first = ml.admit_objective(good, utility=verified_return(lt))
    dup = ml.admit_objective(good, utility=verified_return(lt))
    attempts["duplicate"] = dup.reason
    out["genuine_objective_accepted"] = bool(first.accepted) and all(
        ml.admit_objective(l.verified, utility=verified_return(l)).accepted for l, _ in lts[1:])
    rec = lt.records[0]
    fake = OutcomeObservation(query_id="q_invented", status=OutcomeStatus.OBSERVED, value=1e6, observed_at=clock.now(),
                              verified=True, source="labworld", dedup_key="invented")
    q_real = next((q for r in lt.records for q in (r.result.expected_outcomes if r.result else [])), None)
    attempts["measurement_with_unknown_query"] = ml.ledger.admit_measurement(fake, exp_ids=[rec.exp_id]).reason
    if q_real is not None:
        real = verifier.measure([q_real])[0]
        attempts["measurement_value_altered"] = ml.ledger.admit_measurement(
            dataclasses.replace(real, value=(real.value or 0.0) + 500.0, dedup_key="altered"),
            exp_ids=[rec.exp_id]).reason
        attempts["measurement_unverified_flag"] = ml.ledger.admit_measurement(
            dataclasses.replace(real, verified=False, dedup_key="unverified"), exp_ids=[rec.exp_id]).reason
        attempts["measurement_imagined"] = ml.ledger.admit_measurement(
            dataclasses.replace(real, dedup_key="imagined"), exp_ids=[rec.exp_id], imagined=True).reason
        attempts["measurement_self_sourced"] = ml.ledger.admit_measurement(
            dataclasses.replace(real, source="world_model", dedup_key="selfsrc"), exp_ids=[rec.exp_id]).reason
    out["rejections"] = attempts
    out["all_invalid_rejected"] = all(bool(r) for r in attempts.values())
    eps, used, held = ml._collect()
    util = [u for e in eps if e.ibq is not None for u, h in zip(e.utility, e.has_u) if h]
    out["max_abs_query_utility"] = max((abs(u) for u in util), default=0.0)
    out["fabricated_value_in_any_query_utility"] = any(abs(u) > 1e3 for u in util)
    out["independent_groups"] = len(used) + len(held)
    h_psi = params_hash(ml.reward)
    m = ml.meta_update()
    out["psi_changed_by_meta_update"] = params_hash(ml.reward) != h_psi
    out["meta_update"] = {"applied": m.applied, "skipped_reason": m.skipped_reason, "estimator": m.estimator,
                          "n_independent": m.n_independent_outcomes, "n_query": m.n_query}
    out["ledger_chain_ok"] = ml.ledger.verify_chain()[0]
    # reward-magnitude scaling: the same support batch, R_psi output multiplied by c
    ib = ml.inner.build_batch(store, lt.records[:8]).with_mix(0.0)
    theta0 = ml.subset.snapshot()
    psi = {k: v.detach().clone() for k, v in ml.reward.psi().items()}

    def step(p: Dict[str, Tensor]) -> Tensor:
        th = ml.engine.inner_update(theta0, p, [ib], ml.mc.inner_lr, 1, create_graph=False).detached()
        return flat({n: th[n] - theta0[n] for n in th})
    d1 = step(psi)
    bound = ml.reward.bound
    scaling = {}
    for c in (10.0, 1000.0):
        ml.reward.bound = bound * c                                   # r -> c * r exactly (tanh argument unchanged
        big = {k: (v * c if k.startswith("out.") else v) for k, v in psi.items()}   # when out is scaled with it)
        dc = step(big)
        scaling[f"output_x{c:g}"] = {"update_norm_ratio": float(dc.norm() / d1.norm()),
                                     "cosine": float(torch.nn.functional.cosine_similarity(dc, d1, dim=0)),
                                     "max_abs_diff": float((dc - d1).abs().max())}
    ml.reward.bound = bound
    for c in (10.0, 1000.0):                                          # inflate the pre-bound output only
        big = {k: (v * c if k.startswith("out.") else v) for k, v in psi.items()}
        dc = step(big)
        with torch.no_grad():
            r = ml.inner.learned_reward(big, ib)
        scaling[f"pre_bound_x{c:g}"] = {"update_norm_ratio": float(dc.norm() / d1.norm()),
                                        "cosine": float(torch.nn.functional.cosine_similarity(dc, d1, dim=0)),
                                        "max_abs_reward": float(r.abs().max()), "bound": bound}
    out["reward_scaling"] = scaling
    out["update_norm_bound_per_step"] = ml.mc.inner_lr * math.sqrt(ml.mc.clip_policy ** 2 + ml.mc.clip_trunk ** 2)
    out["update_norm_unscaled"] = float(d1.norm())
    return out


# ========================================================================================================== GPU cost
def meta_cost_50m(Ks: Sequence[int] = (2, 4), n_worlds: int = 4, timeout_s: float = 1800.0) -> Dict[str, Any]:
    """Time and GPU memory of a meta-update at the 50m preset on the real backbone configuration (Qwen3-4B frozen,
    cached latents), under the shared GPU lock. LabWorld decisions under uniform behaviour supply the rows."""
    from ..nn.gpu import free_gpu_memory, gpu_lock
    from ..nn.model import build_model
    cfg = CamLiveConfig()
    cfg.device = "cuda"
    ec = ExperimentConfig()
    out: Dict[str, Any] = {"preset": cfg.core.scale, "backbone": cfg.backbone.name, "K": {}}
    with gpu_lock(timeout_s=timeout_s, min_free_gb=16.0) as t_lock:
        try:
            t0 = time.time()
            model = build_model(cfg)
            out["load_seconds"] = time.time() - t0
            store, clock = MemoryStore(), SimClock()
            key = LabVerifier(LabWorld(LabConfig(seed=0)))
            mc = MetaConfig.from_cfg(cfg, provenance="diagnostic", heldout_fraction=0.0, mix=0.0, min_independent=2)
            ml = MetaLearner(model, store, cfg, check_signature=key.check_signature, clock=clock, meta_cfg=mc)
            counts = model.count_parameters()
            out.update({"n_theta": ml.subset.n_params(), "n_psi": sum(p.numel() for p in ml.reward.parameters()),
                        "experiential_params": counts["experiential"], "d_model": model.d_model,
                        "backbone_pretrained": counts["backbone_pretrained"]})
            torch.cuda.synchronize()
            base_mem = torch.cuda.memory_allocated()
            out["gpu_allocated_after_load_gb"] = base_mem / 2 ** 30
            lts = []
            t0 = time.time()
            for i in range(n_worlds):
                world = LabWorld(LabConfig(seed=31 + i, **ec.lab), clock=clock)
                ml.new_lifetime()
                lts.append(LabCollector(store, max_context=ec.max_context, seq_start=i * 1000 + 1).run(
                    world, uniform_policy(i), ec.n_decisions, seed=i))
            recs = [r for lt in lts for r in lt.records]
            out["n_decisions"] = len(recs)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ib = ml.inner.build_batch(store, recs[:16])
            ibq = ml.inner.build_batch(store, recs[16:48], policy_only=True)
            torch.cuda.synchronize()
            out["encode_seconds_first_48_rows"] = time.time() - t0
            out["encode_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            t0 = time.time()
            ml.inner.build_batch(store, recs[:16])
            torch.cuda.synchronize()
            out["encode_seconds_16_rows_cached_text"] = time.time() - t0
            theta0 = ml.subset.snapshot()
            psi = ml.reward.psi()
            U = torch.linspace(-1.0, 1.0, ibq.n, device=ibq.b.device)
            for K in Ks:
                row: Dict[str, Any] = {}
                for mode in ("unroll", "checkpoint"):
                    times, peak = [], 0
                    for rep in range(4):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        m0 = torch.cuda.memory_allocated()
                        t0 = time.time()
                        tr = ml.engine.inner_update(theta0, psi, [ib], ml.mc.inner_lr, K, create_graph=True, mode=mode)
                        res = ml.engine.meta_gradient(tr, QueryData(batch=ibq, utility=U, baseline=0.0), "on_policy")
                        torch.cuda.synchronize()
                        if rep:                                     # first repetition warms up the kernels
                            times.append(time.time() - t0)
                        peak = max(peak, torch.cuda.max_memory_allocated() - m0)
                        del tr
                    row[mode] = {"seconds_mean": sum(times) / len(times), "seconds_min": min(times),
                                 "extra_peak_gpu_mb": peak / 2 ** 20, "grad_norm": res.grad_norm,
                                 "supported": res.supported}
                out["K"][str(K)] = row
            # end to end through MetaLearner.meta_update (real policy updates, signed outcomes, re-encoded queries)
            e2e = {}
            for K in Ks:
                ml.mc.inner_steps = K
                ml2_recs = []
                for i in range(n_worlds):
                    world = LabWorld(LabConfig(seed=500 + 10 * K + i, **ec.lab), clock=clock)
                    ml.new_lifetime()
                    buf: List[ExperienceRecord] = []
                    t_pol: List[float] = []

                    def on_step(rec: ExperienceRecord) -> None:
                        ml.internal_rewards([rec])
                        buf.append(rec)
                        if len(buf) >= ec.update_every:
                            torch.cuda.synchronize()
                            t1 = time.time()
                            ml.policy_update(list(buf), steps=K)
                            torch.cuda.synchronize()
                            t_pol.append(time.time() - t1)
                            buf.clear()
                    lt = LabCollector(store, max_context=ec.max_context, seq_start=(100 + 10 * K + i) * 1000).run(
                        world, model_policy(ml, store, torch.Generator().manual_seed(i)), ec.n_decisions, seed=i,
                        on_step=on_step)
                    ml.admit_objective(lt.verified, utility=verified_return(lt))
                    ml2_recs.append(t_pol)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_allocated()
                t0 = time.time()
                rec_m = ml.meta_update()
                torch.cuda.synchronize()
                e2e[str(K)] = {"seconds": time.time() - t0,
                               "extra_peak_gpu_mb": (torch.cuda.max_memory_allocated() - m0) / 2 ** 20,
                               "applied": rec_m.applied, "skipped_reason": rec_m.skipped_reason,
                               "estimator": rec_m.estimator, "n_query": rec_m.n_query,
                               "n_independent": rec_m.n_independent_outcomes,
                               "meta_grad_norm": rec_m.meta_grad_norm,
                               "policy_update_seconds_mean": sum(sum(t) for t in ml2_recs)
                               / max(1, sum(len(t) for t in ml2_recs))}
            out["end_to_end_meta_update"] = e2e
            out["gpu_peak_total_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            out["lock_held_seconds"] = time.time() - t_lock
        finally:
            model = ml = store = None
            free_gpu_memory()
    return out


# ============================================================================================================= main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join("cam_live", "docs", "evidence", "C"))
    ap.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19")
    ap.add_argument("--lifetimes", type=int, default=320)
    ap.add_argument("--kinds", default="bootstrap,learned,mixed,verified_pg")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--skip-experiment", action="store_true")
    ap.add_argument("--gpu-cost", action="store_true", help="only measure the 50m meta-update cost on the GPU")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.gpu_cost:
        cost = meta_cost_50m()
        cost["command"] = "PYTHONPATH=cam_live/src .venv/bin/python -m cam_live.reward.diagnose --gpu-cost"
        with open(os.path.join(a.out, "meta_cost_50m.json"), "w") as f:
            json.dump(cost, f, indent=1, sort_keys=True)
        print(json.dumps({k: v for k, v in cost.items() if k in ("K", "end_to_end_meta_update", "gpu_peak_total_gb",
                                                                 "lock_held_seconds")}, indent=1))
        return 0
    torch.set_num_threads(4)
    path = os.path.join(a.out, "diagnostics.json")
    out: Dict[str, Any] = {}
    if os.path.exists(path) and a.skip_experiment:
        with open(path) as f:
            out = json.load(f)
        if "experiment" in out:
            out["experiment"].setdefault("command_experiment", out.get("command", ""))
    out["command"] = ("PYTHONPATH=cam_live/src .venv/bin/python -m cam_live.reward.diagnose "
                      f"--seeds {a.seeds} --lifetimes {a.lifetimes} --kinds {a.kinds}")
    out["scale"] = "CamLiveConfig.debug(): tiny random backbone, debug core, CPU; LabWorld (diagnostic provenance)"
    t0 = time.time()
    out["meta_gradient"] = meta_gradient_check()
    out["psi_isolation"] = psi_isolation_check()
    out["integrity"] = integrity_check()
    out["seconds_checks"] = time.time() - t0
    if not a.skip_experiment:
        ec = ExperimentConfig(n_lifetimes=a.lifetimes)
        out["experiment"] = labworld_experiment([int(s) for s in a.seeds.split(",")], ec, a.kinds.split(","),
                                                workers=a.workers)
        out["experiment"]["scripted_baselines"] = scripted_baselines(ec)
    elif "experiment" in out and out["experiment"].get("runs"):
        old = out["experiment"]                                  # re-summarise the stored runs, nothing is re-run
        ec = ExperimentConfig(**old["config"])
        runs = old["runs"]
        out["experiment"] = {**summarise(runs, ec, old.get("block", 20)), "config": old["config"], "runs": runs}
        out["experiment"]["scripted_baselines"] = scripted_baselines(ec)
        out["command"] = old.get("command_experiment", "") or out["command"]
        out["command_checks"] = ("PYTHONPATH=cam_live/src .venv/bin/python -m cam_live.reward.diagnose "
                                 "--skip-experiment   (checks re-run, stored experiment runs re-summarised)")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    brief = {"meta_gradient": {K: {k: v[k] for k in ("rel_error_sampled_coords", "directional_rel_error",
                                                    "every_tensor_nonzero", "grad_norm_reward_detached")}
                               for K, v in out["meta_gradient"]["K"].items()},
             "psi_isolation": out["psi_isolation"], "integrity_rejections": out["integrity"]["rejections"]}
    if "experiment" in out:
        e = out["experiment"]
        brief["experiment"] = {"final_quarter_achieved": e["final_quarter_achieved"],
                               "statements": e.get("statements")}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
