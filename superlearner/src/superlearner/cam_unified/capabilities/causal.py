"""C01 / C03 through the integrated world model: the mandatory causal-model tests of spec 6.6 on real recorded
sequences of the assembled agent (real tokenizer over Qwen3-0.6B, B's UnifiedCore, C's WorldModel / OutcomeHead).

Protocol (`run_causal`): for each test world the agent (untrained policy: CAM's own action generator sampling at
temperature 1, i.e. its initial exploratory behaviour) acts and its EventRecords go to E's archive; the episodes are
read back from the archive and tokenised with the agent's own `tokenise_sequences` (exactly the replay path), split
into training and held-out sequences by episode. The model's world objective (`CamUnifiedModel.world_loss` on the
acting-consistent `filter_sequence`, i.e. the agent's objective_world) is optimised by D's CombinedTrainer, followed
by the EMA target update. Measurements (all on held-out episodes or explicit interventions):

1. different actions -> different predictions from the same filtered latent state (prior transition per tool);
2. delayed vs immediate consequences (DelayEffectEnv: open-loop forecast of the reward of the 1st and 3rd decision
   after `press_now` / `press_later` / waiting);
3. interaction of two simultaneous interventions (InteractionEnv: interaction contrast of the predicted payout of
   the four lever settings vs truth 2.0, and against an additive ridge regression on the same sequences);
4. predictions follow an unannounced mechanism change (SwitchEnv: filtered one-step predictions after the switch
   versus unswitched episodes with identical prefixes);
5. multistep open-loop error vs horizon against no-change / trend-persistence / key-mean / linear-regression
   baselines on the same held-out (start, h) pairs (world/baselines.py `evaluate_multistep`);
6. changing a predicted intermediate event changes later predictions (DelayEffectEnv: the imagined future after
   `press_later` with the stochastic latent of the intermediate transition taken from a press_later vs a waiting
   posterior; and the imagined reward after replacing the first imagined action);
7. an imagined future does not mutate the actual environment, the archive, Z or the FastState (planner run between
   snapshots; bit-identical comparison).

DOM / observation-only variant (plan 2026-11-26 §3 W6 real end-to-end check; ``CausalSettings.preset="dom_dev"``):
the stack is `config.make_preset(preset)` (8 x 384 core with Qwen3-0.6B, block transition on R_t, no EMA target,
per-group free bits, consistency, gated belief fusion, reward_mode "observation_only"). The verified return is then
never a world-model target or an observed outcome: every test reads the *observable* payout instead, the change of
the points counter the worlds show the agent (outcome key ``num:points@me[points]``, predicted in symlog space and
reported in payout units by symexp), and every prediction starts from the agent filter's own prior input
(`world.train_seq.filter_sequence`: R_t and the belief tokens decide() uses). Training uses ``train_steps`` with a
linear learning-rate decay to ``lr_final_frac`` (``lr_decay``); t1 is averaged over every step of every held-out
batch; the consistency gap MSE(Z^_{t+1}, Z_post_{t+1}) / Var(Z_post) is measured on held-out batches (`consistency`).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..interfaces import Action, SequenceBatch
from ..outcome_spec import CORE_INDEX
from ..world._util import symexp
from ..world.baselines import (KeyMeanBaseline, LinearRegressionBaseline, NoChangeBaseline, TrendPersistenceBaseline,
                               WorldModelForecaster, evaluate_multistep)
from .envs import DelayEffectEnv, InteractionEnv, ProbeEnv, ScoreEvaluator, SwitchEnv, _points
from .imagination import config_override, imagination_settings
from .loop import Episode, mean, train_agent
from .stack import build_stack, capability_config

R_IDX = CORE_INDEX["reward"]


def _mem(tag: str) -> dict:
    """GPU memory at a phase of the run (GB; logged when CAM_W6_MEMLOG=1)."""
    if not torch.cuda.is_available():
        return {}
    m = {"phase": tag, "allocated_gb": torch.cuda.memory_allocated() / 2 ** 30,
         "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "t": time.time()}
    if os.environ.get("CAM_W6_MEMLOG") == "1":
        print("[W6 mem]", m, flush=True)
    return m


def obs_only(model) -> bool:
    """Observation-only measurements: the verified return is no world-model target; read the observable payout."""
    return str(getattr(model.cfg, "reward_mode", "legacy")) == "observation_only"


def payout_index(ob, key: str = "num:points@me[points]") -> torch.Tensor:
    """[B] query index of the observable payout outcome per row (-1 when the row has no such query)."""
    out = []
    for keys in (ob.keys or []):
        out.append(next((q for q, k in enumerate(keys) if k == key), -1))
    return torch.tensor(out, dtype=torch.long)


def payout_of(model, mean: torch.Tensor, ob, key: str = "num:points@me[points]") -> torch.Tensor:
    """Predicted payout [B] from per-query predictive means [B,Q]: the verified-reward query (legacy) or symexp of
    the observable points-counter change (observation-only; NaN where a row has no such query)."""
    mean = mean.float().cpu()
    if not obs_only(model):
        return mean[:, R_IDX]
    idx = payout_index(ob, key)
    sel = mean.gather(1, idx.clamp(min=0).view(-1, 1)).squeeze(1)
    return torch.where(idx >= 0, symexp(sel), torch.full_like(sel, float("nan")))


def _filt(model, seq, aux=None):
    """The agent's acting-consistent filter (no burn-in: every step is a measurement step)."""
    from ..world.train_seq import filter_window
    return filter_window(model, seq, aux, burn_in=0)


def _prior_input(filt, t: int):
    """(prior input [B,S,D], belief tokens) at step t: R_t and the pure belief tokens on the DOM path."""
    Rs = getattr(filt, "Rs", None)
    Z = Rs[t] if (Rs is not None and len(Rs) > t and Rs[t] is not None) else filt.Zs[t]
    bt = getattr(filt, "belief_tokens", None)
    return Z, (bt[t] if bt else filt.beliefs[t])


class AgentFilterForecaster:
    """Open-loop forecasts from the agent's own filter state (DOM path) under the shared forecasting API of
    world/baselines.py: start from R_start / B_start of `filter_sequence` (which have seen O_{<=start} only) and roll
    the recorded actions through the prior (reasoner on imagined states)."""

    def __init__(self, model, n_samples: int = 8, seed: int = 0):
        self.model, self.n_samples, self.seed = model, n_samples, seed

    def fit(self, batches):
        return self

    @torch.no_grad()
    def predict(self, batch, start, horizons):
        from ..world.train_seq import reason_fn_of
        m = self.model
        dev = batch.steps[0].obs.x.device
        g = torch.Generator(device=dev).manual_seed(self.seed)
        filt = _filt(m, SequenceBatch(batch.steps[:start + 1]))
        Z0, _ = _prior_input(filt, start)
        bt = getattr(filt, "belief_tokens", None) or filt.beliefs
        f = m.world_model.forecast(batch, start, tuple(horizons), beliefs=list(bt),
                                   n_samples=self.n_samples, generator=g, Z0=Z0, reason_fn=reason_fn_of(m))
        return {k: v["mean"].detach().float().cpu() for k, v in f.items()}


@dataclass
class CausalSettings:
    seed: int = 0
    B: int = 32
    record_episodes: int = 12           # episodes per world (x B rows)
    heldout_frac: float = 0.25
    train_steps: int = 300
    batch_seqs: int = 32
    lr: float = 2e-3
    grad_clip: float = 10.0
    horizons: tuple = (1, 2, 3, 4)
    worlds: tuple = ("delay", "interaction", "switch")
    cfg_overrides: dict = field(default_factory=dict)
    preset: str = ""                    # "" = capability_config (legacy); "dom_dev" etc. = config.make_preset(preset)
    lr_decay: bool = False              # linear decay of every learning rate to lr_final_frac over train_steps
    lr_final_frac: float = 0.05
    payout_key: str = "num:points@me[points]"   # observable payout outcome (observation-only measurements)
    wm_kwargs: dict = field(default_factory=dict)   # WorldModel constructor overrides (e.g. open_loop_samples)
    sem_cache_mb: int = 0                # > 0: bound the semantic encoder's hidden-state cache (GPU memory budget)
    core_kwargs: dict = field(default_factory=dict)  # UnifiedCore constructor overrides (e.g. emb_std)


def make_env(name: str, B: int, seed: int = 0, **kw):
    if name == "delay":
        return DelayEffectEnv(B, seed, **kw)
    if name == "interaction":
        return InteractionEnv(B, seed, **kw)
    if name == "switch":
        return SwitchEnv(B, seed, **kw)
    raise ValueError(name)


def episodes_from_archive(archive) -> list:
    """Decision records grouped by (env, actor, episode), each sorted by step."""
    eps: dict = {}
    for batch in archive.scan():
        for r in batch:
            if r.action is None:
                continue
            eps.setdefault((r.env_id, r.actor_id, r.episode_id), []).append(r)
    out = [sorted(v, key=lambda r: r.step) for v in eps.values()]
    out.sort(key=lambda recs: (recs[0].t_wall, recs[0].env_id))
    return out


def record(stack, env, n_episodes: int, seed0: int) -> list:
    """The agent acts (no updates) for n_episodes of every row; returns the archived episodes."""
    H = env.horizon
    train_agent(stack, env, ScoreEvaluator(env), n_episodes * H, update_every=0,
                seeds=[seed0 + b for b in range(env.B)])
    stack.archive.flush()
    return episodes_from_archive(stack.archive)


def groups_of(episodes: list, n: int, rng: random.Random) -> list:
    """Fixed groups of `n` archived episodes (each group is one replay batch)."""
    order = list(range(len(episodes)))
    rng.shuffle(order)
    out = []
    for i in range(0, len(order), n):
        part = [episodes[j] for j in order[i:i + n]]
        if len(part) >= 2:
            out.append(part)
    return out


def tokenise(agent, groups: list) -> list:
    """Tokenised (SequenceBatch, ReplayAux) per group with the agent's replay tokenisation (evaluation only: the
    tokens hold autograd graphs into the tokenizer, so training re-tokenises after every step like the agent)."""
    with torch.no_grad():
        return [agent.tokenise_sequences(g) for g in groups]


def train_world(stack, groups: list, steps: int, lr_decay: bool = False, lr_final_frac: float = 0.05,
                metrics: Optional[list] = None) -> list:
    """Optimise the model's world objective (the agent's objective_world: acting-consistent filter + WorldModel
    loss) on fixed episode groups, cycled, re-tokenised every step exactly as `CamUnifiedAgent.update` does; each
    step bumps the model version (weight-dependent caches) and updates the EMA target (a no-op without one).
    ``lr_decay``: every parameter group's learning rate decays linearly from its base value to ``lr_final_frac`` of
    it over the run. ``metrics`` (a list) collects per-step scalar loss terms. Returns the loss history."""
    model, trainer, agent = stack.model, stack.trainer, stack.agent
    opt = getattr(trainer, "opt", None)
    base = [g.get("base_lr", g["lr"]) for g in opt.param_groups] if (lr_decay and opt is not None) else None
    hist = []
    for i in range(steps):
        if base is not None:
            f = 1.0 - (1.0 - lr_final_frac) * (i / max(1, steps - 1))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
                if "base_lr" in g:
                    g["base_lr"] = b * f
        model.train()
        seq, aux = agent.tokenise_sequences(groups[i % len(groups)])
        filt = model.filter_sequence(seq, aux)
        out = model.world_loss(seq, filt)
        trainer.step({"world": out["world"]})
        model.version = model.version + 1
        hist.append(float(out["world"].detach()))
        if i < 2 or (i + 1) % 100 == 0:
            _mem(f"train_step_{i + 1}")
        if metrics is not None:
            metrics.append({k: float(v.detach()) for k, v in out.items()
                            if torch.is_tensor(v) and v.numel() == 1})
    if base is not None:
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b
            if "base_lr" in g:
                g["base_lr"] = b
    model.eval()
    return hist


def _with_episode(seq: SequenceBatch, steps: Optional[int] = None) -> SequenceBatch:
    return SequenceBatch(seq.steps[:steps] if steps else seq.steps)


@torch.no_grad()
def forecast(model, seq: SequenceBatch, start: int, horizons, n_samples: int = 8) -> dict:
    """Open-loop forecast from step ``start``. DOM path (single filter): from the agent filter's prior input and
    belief tokens at ``start`` (they have seen O_{<=start} only), reasoner on imagined states."""
    from ..world.train_seq import reason_fn_of, single_filter
    g = torch.Generator(device=model.device).manual_seed(0)
    if single_filter(model):
        filt = _filt(model, SequenceBatch(seq.steps[:start + 1]))
        Z0, _ = _prior_input(filt, start)
        bt = getattr(filt, "belief_tokens", None) or filt.beliefs
        return model.world_model.forecast(seq, start, tuple(horizons), beliefs=list(bt), n_samples=n_samples,
                                          generator=g, Z0=Z0, reason_fn=reason_fn_of(model))
    return model.world_model.forecast(seq, start, tuple(horizons), n_samples=n_samples, generator=g)


@torch.no_grad()
def scripted_sequences(stack, env, plans: list, seed0: int) -> tuple:
    """Run fixed tool plans (one per row; each entry a tool name or an Action) through the env + archive-free
    tokenisation, using the agent-visible records only. Returns (SequenceBatch, ReplayAux, hidden states)."""
    from ..interfaces import EventRecord, Provenance
    model = stack.model
    ob = env.reset([seed0 + b for b in range(len(plans))])
    B = env.B
    recs = [[] for _ in range(B)]
    scores = [0.0] * B
    obs = [list(o) for o in ob.observations]
    t = list(ob.t)
    ev = ScoreEvaluator(env)
    state = model.init_state(B)
    for k in range(len(plans[0])):
        acts = [p[k] if isinstance(p[k], Action) else Action(tool=p[k]) for p in plans]
        results = env.execute(acts)
        adv = env.advance_time([model.think_days] * B)
        new = [float(ev.evaluate_goal(b).score) for b in range(B)]
        nxt = [list(results[b].observations) + list(adv.observations[b]) for b in range(B)]
        oo = obs_only(model)
        specs = [state.trackers[b].realize(obs[b], acts[b], results[b], nxt[b],
                                           None if oo else new[b] - scores[b]) for b in range(B)]
        for b in range(B):
            recs[b].append(EventRecord(event_id=f"s{b}_{k}", env_id=env.env_ids[b], actor_id="script",
                                       episode_id=f"ep{b}", step=k, model_version=0, t_sim=t[b],
                                       observations=list(obs[b]), action=acts[b], result=results[b],
                                       reward=0.0 if oo else new[b] - scores[b], done=bool(adv.done[b]),
                                       provenance=Provenance.REAL,
                                       extra={"outcomes": [s.to_dict() for s in specs[b]],
                                              "dt": float(adv.t[b] - t[b]), "goal": {"goal_id": "g", "text":
                                                                                       env.goals()[b].text,
                                                                                       "metrics": {},
                                                                                       "deadline_days": None}}))
        scores, obs, t = new, nxt, list(adv.t)
    seq, aux = stack.agent.tokenise_sequences(recs)
    return seq, aux, [dict(h) for h in env.hidden]


# ------------------------------------------------------------------------------------------------ the seven tests

def measure_actions_differ(stack, heldout, all_steps: bool = False) -> dict:
    """(1) From the same filtered state, each contemplated tool's prior transition gives different payout
    predictions; reported as the mean absolute pairwise difference and the per-tool means. Legacy: step 2 of the
    first held-out batch; ``all_steps``: averaged over every step of every held-out batch (DOM acceptance)."""
    model = stack.model
    cand = stack.info.get("t1_actions") or {t.name: Action(tool=t.name) for t in stack.info["tools"]}
    names = list(cand)
    preds = {n: [] for n in names}
    batches = heldout if all_steps else heldout[:1]
    for seq, aux in batches:
        with torch.no_grad():
            filt = _filt(model, seq, aux) if obs_only(model) else model.filter_sequence(seq, aux)
        ts = range(len(seq.steps)) if all_steps else [2]
        for t in ts:
            st = seq.steps[t]
            Z, bel = _prior_input(filt, t)
            B = Z.shape[0]
            with torch.no_grad(), model._autocast():
                for n in names:
                    acts = [copy.deepcopy(cand[n]) for _ in range(B)]
                    A = model.tokenizer.action(acts, [stack.info["tools"]] * B)
                    Zh, _ = model.world_model.prior(Z, A, bel, st.dt, sample=False)
                    d = model.world_model.outcome_dist(Zh, st.outcomes)
                    preds[n].append(payout_of(model, d.mean(), st.outcomes))
    P = {n: torch.cat(v) for n, v in preds.items()}
    diffs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = ~(torch.isnan(P[a]) | torch.isnan(P[b]))
            if bool(m.any()):
                diffs.append(float((P[a][m] - P[b][m]).abs().mean()))
    return {"pred_reward": {n: float(torch.nanmean(v)) for n, v in P.items()},
            "mean_abs_pairwise_diff": mean(diffs), "n_rows": int(next(iter(P.values())).numel())}


def measure_delay(stack, seed0: int) -> dict:
    """(2) Open-loop forecasts after a fixed prefix of waits, then press_now / press_later / wait at step 2."""
    env = DelayEffectEnv(3, noise=0.0)
    plans = [["wait_turn"] * 2 + [first] + ["wait_turn"] * 5 for first in ("press_now", "press_later", "wait_turn")]
    seq, aux, _ = scripted_sequences(stack, env, plans, seed0)
    f = forecast(stack.model, seq, 2, (1, 2, 3))
    pred = torch.stack([payout_of(stack.model, f[h]["mean"], seq.steps[2 + h - 1].outcomes) for h in (1, 2, 3)], 1)
    truth = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 0]])
    return {"pred": pred.tolist(), "truth": truth.tolist(), "max_abs_err": float((pred - truth).abs().max()),
            "delayed_effect_pred": float(pred[1, 2]), "immediate_effect_pred": float(pred[0, 0])}


def measure_interaction(stack, train_batches, seed0: int) -> dict:
    """(3) Predicted payout of the four lever settings (one-step open-loop forecast from step 1)."""
    env = InteractionEnv(4, noise=0.0)
    settings = [(False, False), (True, False), (False, True), (True, True)]
    plans = [[Action("set_levers", {"x": False, "w": False}), Action("set_levers", {"x": x, "w": w}),
              Action("set_levers", {"x": False, "w": False}), Action("set_levers", {"x": False, "w": False})]
             for x, w in settings]
    seq, aux, _ = scripted_sequences(stack, env, plans, seed0)
    f = forecast(stack.model, seq, 1, (1,))
    pred = payout_of(stack.model, f[1]["mean"], seq.steps[1].outcomes)
    truth = torch.tensor([InteractionEnv.truth(x, w) for x, w in settings])
    contrast = float(pred[3] - pred[1] - pred[2] + pred[0])
    lin = LinearRegressionBaseline(horizons=(1,)).fit([b for b, _ in train_batches]).predict(seq, 1, (1,))[1]
    lin = payout_of(stack.model, lin, seq.steps[1].outcomes) if lin.dim() == 2 else lin.cpu()
    lin_contrast = float(lin[3] - lin[1] - lin[2] + lin[0])
    return {"pred": pred.tolist(), "truth": truth.tolist(), "max_abs_err": float((pred - truth).abs().max()),
            "interaction_contrast": contrast, "truth_contrast": 2.0, "linear_regression_contrast": lin_contrast}


def measure_switch(stack, seed0: int) -> dict:
    """(4) Filtered one-step predictions (Z_t from O_<=t, then the prior for the next pull) in episodes that switch
    at decision 3 versus episodes that never switch; identical before the switch."""
    model = stack.model
    out = {}
    for name, fixed in (("switch_at_3", 3), ("no_switch", 99)):
        env = SwitchEnv(16, noise=0.0, fixed_switch=fixed)
        seq, aux, _ = scripted_sequences(stack, env, [["pull_lever"] * 8 for _ in range(16)], seed0)
        with torch.no_grad():
            filt = _filt(model, seq, aux) if obs_only(model) else model.filter_sequence(seq, aux)
        preds = []
        with torch.no_grad(), model._autocast():
            for t in range(8):
                st = seq.steps[t]
                Z, bel = _prior_input(filt, t)
                Zh, _ = model.world_model.prior(Z, st.act, bel, st.dt, sample=False)
                if obs_only(model) and model.cfg.use_fast_adaptation:
                    # the predictive the agent acts with (particles + Kalman fusion of the filter's belief)
                    from ..core.belief_runtime import predictive_moments
                    fast = filt_fast_at(model, seq, t)
                    ob = model.channels_of(fast, st.outcomes)
                    m, _ = predictive_moments(model, fast, model.outcome_head.features(Zh, ob), ob)
                else:
                    m = model.world_model.outcome_dist(Zh, st.outcomes).mean()
                preds.append(float(torch.nanmean(payout_of(model, m, st.outcomes))))
        out[name] = preds
    out["truth_switch"] = [1.0] * 3 + [-1.0] * 5
    out["post_switch_pred_mean_t5_7"] = mean(out["switch_at_3"][5:])
    out["no_switch_pred_mean_t5_7"] = mean(out["no_switch"][5:])
    out["pre_switch_max_gap"] = max(abs(a - b) for a, b in zip(out["switch_at_3"][:3], out["no_switch"][:3]))
    out["switch_correct"] = bool(out["post_switch_pred_mean_t5_7"] < -0.7 and out["no_switch_pred_mean_t5_7"] > 0.7)
    return out


@torch.no_grad()
def filt_fast_at(model, seq: SequenceBatch, t: int):
    """The belief B_t of the replay filter (outcomes < t absorbed with the world-model likelihood and the Kalman
    layer, as the runtime does when acting)."""
    from ..core.belief_runtime import dom_beliefs, kalman_base
    from ..world.train_seq import prior_on_R
    B = seq.steps[0].obs.B
    dev = seq.steps[0].obs.x.device
    filt = _filt(model, SequenceBatch(seq.steps[:max(1, t)]))
    fast = model.beliefs.init_fast(B, device=dev)
    with model._autocast():
        for i in range(t):
            st = seq.steps[i]
            Z, bel = _prior_input(filt, i)
            Bt = model.beliefs.tokens(fast)
            ob = model.channels_of(fast, model.belief_view(st.outcomes))
            Zh, _ = model.world_model.prior(Z, st.act, Bt, st.dt, sample=False)
            feat = model.outcome_head.features(Zh, ob)
            fast, _ = model.beliefs.update(fast, filt.Zs[i], st.act, ob, model.outcome_head,
                                           feat=feat if dom_beliefs(model) else None)
            if model._kalman_on():
                fast = model.kalman_update(fast, feat, ob, kalman_base(model, feat, ob, fast))
    return fast


@torch.no_grad()
def measure_consistency(stack, heldout) -> dict:
    """Consistency gap on held-out batches: MSE(Z^_{t+1}, Z_post_{t+1}) / Var(Z_post) of the teacher-forced
    transition and of the mode-latent one-step prior (the state decide() scores), from the world loss diagnostics."""
    model = stack.model
    tf, m1 = [], []
    for seq, aux in heldout:
        filt = _filt(model, seq, aux)
        out = model.world_loss(seq, filt)
        if "cons_mse_var" in out:
            tf.append(float(out["cons_mse_var"]))
        if "cons_mse_var_mode1" in out:
            m1.append(float(out["cons_mse_var_mode1"]))
    return {"cons_mse_var_teacher_forced": mean(tf) if tf else None,
            "cons_mse_var_mode1": mean(m1) if m1 else None, "n_batches": len(heldout)}


def measure_multistep(stack, train_batches, heldout, horizons) -> dict:
    """(5) Open-loop error vs horizon of the world model and the baselines on identical held-out (start, h)."""
    from ..world.train_seq import single_filter
    wmf = AgentFilterForecaster(stack.model, n_samples=8) if single_filter(stack.model) else \
        WorldModelForecaster(stack.model.world_model, n_samples=8)
    fc = {"world_model": wmf,
          "no_change": NoChangeBaseline(), "trend": TrendPersistenceBaseline(), "key_mean": KeyMeanBaseline(),
          "linear_regression": LinearRegressionBaseline(horizons=tuple(horizons))}
    train = [b for b, _ in train_batches]
    for f in fc.values():
        f.fit(train)
    with torch.no_grad():
        res = evaluate_multistep(fc, [b for b, _ in heldout], tuple(horizons))
    return res


@torch.no_grad()
def measure_intermediate_event(stack, seed0: int, n_rows: int = 16) -> dict:
    """(6) Switch world: from the same real state (two real pulls, no switch), the imagined intermediate outcome of
    the next pull is set to +1 or to -1 and the FastState clone is updated with it exactly as the planner does
    (BeliefModule.update + Kalman update on the imagined OutcomeBatch); the predicted reward of the pull after it
    (prior transition from the imagined latent, scored with each revised belief) is compared. Changing the predicted
    intermediate event changes the later predicted consequence."""
    from ..outcome_spec import with_targets
    model = stack.model
    env = SwitchEnv(n_rows, noise=0.0, fixed_switch=99)
    ep = Episode(model, env, ScoreEvaluator(env), [seed0 + b for b in range(n_rows)], greedy=True)
    ep.step()
    ep.step()
    dec, tools = ep.decide()
    state = ep.state
    fast_before = state.fast.clone()
    specs, prosp = model.prospective_batch(state)
    out = {}
    with model._autocast():
        A = dec.act_tokens.to(model.device)
        feat = model.outcome_head.features(dec.Z_hat, prosp)
        from ..core.belief_runtime import kalman_base
        base = kalman_base(model, feat, prosp, state.fast)
        mean1, _, _ = model.predict_outcomes(dec.Z_hat, state.fast, prosp)
        for name, val in (("imagined_plus1", 1.0), ("imagined_minus1", -1.0)):
            ys = mean1.clone().float()
            obs_m = torch.zeros_like(prosp.q.mask)
            for b in range(n_rows):
                for q, k in enumerate(prosp.keys[b]):
                    if k == "reward" and not obs_only(model):
                        ys[b, q], obs_m[b, q] = val, True
                    elif k is not None and k.startswith("num:points"):
                        ys[b, q], obs_m[b, q] = math.copysign(math.log1p(abs(val)), val), True
            imagined = with_targets(prosp, ys, obs_m & prosp.q.mask)
            f1, _ = model.beliefs.update(state.fast.clone(), dec.Z, A, model.belief_view(imagined),
                                         model.outcome_head)
            f1 = model.kalman_update(f1, feat, model.belief_view(model.channels_of(state.fast, imagined)), base)
            Z2, _ = model.world_model.prior(dec.Z_hat, A, model.beliefs.tokens(f1), dec.dt, sample=False)
            m2, _, _ = model.predict_outcomes(Z2, f1, model.channels_of(f1, prosp))
            out[name] = float(torch.nanmean(payout_of(model, m2, prosp)))
    out["next_pred_real_belief"] = float(torch.nanmean(payout_of(model, mean1, prosp)))
    out["later_pred_difference"] = out["imagined_plus1"] - out["imagined_minus1"]
    out["real_fast_state_untouched"] = same_fast(fast_before, state.fast)
    return out


FAST_FIELDS = ("particles", "logw", "pscale", "bayes_mu", "bayes_P", "bayes_noise", "bayes_n", "surprise", "steps")


def same_tensor(a, b) -> bool:
    """Bit-identical (NaN == NaN: FastState.surprise marks empty entries with NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(a.nan_to_num(0.0), b.nan_to_num(0.0)))
    return bool(torch.equal(a, b))


def same_fast(f1, f2) -> bool:
    return all(same_tensor(getattr(f1, n), getattr(f2, n)) for n in FAST_FIELDS) and f1.channels == f2.channels


def _snapshot(env, stack, state) -> dict:
    return {"hidden": copy.deepcopy(env.hidden), "t": list(env.t), "seq": list(env.seq), "k": list(env.k),
            "returns": list(env.returns), "archive": stack.archive.count() if stack.archive is not None else None,
            "Z": state.Z.clone(), "fast": None if state.fast is None else state.fast.clone()}


def measure_imagination_isolated(stack, seed0: int) -> dict:
    """(7) A planner decision (imagined rollouts) between two snapshots of the environment, the archive count,
    Z and the FastState: everything but the decision's own state advance (Z_t, set by `decide` before planning) is
    bit-identical; the imagined trajectories carry provenance IMAGINED."""
    model = stack.model
    env = ProbeEnv(8, horizon=3)
    ep = Episode(model, env, ScoreEvaluator(env), [seed0 + b for b in range(8)], greedy=True)
    ep.step()                                     # one real step so the state is not the initial one
    before = _snapshot(env, stack, ep.state)
    fast_before = ep.state.fast.clone()
    with config_override(model, use_imagination=True), imagination_settings(model, 2, 4, 2):
        dec, tools = ep.decide()
    after = _snapshot(env, stack, ep.state)
    same_env = all(before[k] == after[k] for k in ("hidden", "t", "seq", "k", "returns", "archive"))
    fast_same = same_fast(fast_before, ep.state.fast)
    prov = dec.trajectory.provenance.value if dec.trajectory is not None else None
    n_imag = sum(len(i) for i in dec.imagined)
    return {"env_and_archive_unchanged": bool(same_env), "fast_state_unchanged": bool(fast_same),
            "imagined_items": n_imag, "trajectory_provenance": prov,
            "imagination_belief_updates": int(model.calls.get("imagination.beliefs.update", 0))}


def run_causal(settings: Optional[CausalSettings] = None, root: Optional[str] = None, device=None,
               log_every: int = 0) -> dict:
    st = settings or CausalSettings()
    t0 = time.time()
    out = {"settings": asdict(st), "worlds": {}}
    tmp = root or tempfile.mkdtemp(prefix="cam_y_causal_")
    rng = random.Random(st.seed)
    for w in st.worlds:
        if st.preset:
            from ..config import make_preset
            cfg = make_preset(st.preset, use_imagination=False, use_memory=False, lr=st.lr, grad_clip=st.grad_clip,
                              train_horizons=tuple(st.horizons), **st.cfg_overrides)
        else:
            cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, grad_clip=st.grad_clip,
                                    train_horizons=tuple(st.horizons), **st.cfg_overrides)
        ckw = {}
        if st.wm_kwargs:
            ckw["world_model"] = dict(st.wm_kwargs)
        if st.core_kwargs:
            ckw["core"] = dict(st.core_kwargs)
        stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"{w}_s{st.seed}"),
                            component_kwargs=ckw or None)
        env = make_env(w, st.B)
        stack.info["tools"] = env.available_tools()[0]
        if w == "interaction":
            # one tool with two boolean arguments: the contemplated interventions are the four lever settings
            stack.info["t1_actions"] = {f"x{int(x)}w{int(v)}": Action("set_levers", {"x": x, "w": v})
                                        for x in (False, True) for v in (False, True)}
        if st.sem_cache_mb and hasattr(stack.semantic, "cache_max_bytes"):
            stack.semantic.cache_max_bytes = int(st.sem_cache_mb) * 2 ** 20
        mem = [_mem("built")]
        eps = record(stack, env, st.record_episodes, st.seed * 10_007)
        # the agent's fresh on-policy buffer (decision inputs incl. FastStates) is not used by the world objective
        stack.agent.fresh = []
        stack.agent._last_fresh = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem.append(_mem("recorded"))
        rng.shuffle(eps)
        n_ho = max(2, int(len(eps) * st.heldout_frac))
        heldout_g = groups_of(eps[:n_ho], st.batch_seqs, rng)
        train_g = groups_of(eps[n_ho:], st.batch_seqs, rng)
        tm: list = []
        hist = train_world(stack, train_g, st.train_steps, st.lr_decay, st.lr_final_frac, metrics=tm)
        mem.append(_mem("trained"))
        heldout, train = tokenise(stack.agent, heldout_g), tokenise(stack.agent, train_g)
        res = {"n_train_episodes": len(eps) - n_ho, "n_heldout_episodes": n_ho, "loss_first": hist[0],
               "loss_last10": mean(hist[-10:]), "components": stack.real_component_report(),
               "config": {k: getattr(cfg, k) for k in ("preset", "reward_mode", "wm_transition_blocks",
                                                       "wm_ema_target", "kl_free_bits_group", "wm_consistency",
                                                       "n_layers", "d_model", "n_state", "sem_test_model")},
               "train_metrics_last10": {k: mean([m[k] for m in tm[-10:] if k in m]) for k in (tm[-1] if tm else {})},
               "world_model_has_target_core": getattr(stack.model.world_model, "target_core", None) is not None,
               "memory": mem}
        dev0 = 100_000 + 1000 * st.seed
        res["t1_actions_differ"] = measure_actions_differ(stack, heldout, all_steps=bool(st.preset))
        if st.preset:
            res["consistency"] = measure_consistency(stack, heldout)
        if w == "delay":
            res["t2_delay"] = measure_delay(stack, dev0)
            res["t5_multistep"] = measure_multistep(stack, train, heldout, st.horizons)
            res["t7_imagination_isolated"] = measure_imagination_isolated(stack, dev0 + 11)
        elif w == "interaction":
            res["t3_interaction"] = measure_interaction(stack, train, dev0)
        elif w == "switch":
            res["t4_switch"] = measure_switch(stack, dev0)
            res["t5_multistep"] = measure_multistep(stack, train, heldout, st.horizons)
            res["t6_intermediate_event"] = measure_intermediate_event(stack, dev0 + 7)
        out["worlds"][w] = res
    out["wall_s"] = time.time() - t0
    return out


# ================================================================================================ W6 LP probe
# Integrated learning progress at the first informative step (plan 2026-11-26 §3 W6: "> 0.1 nats"): the real dom_dev
# stack trained with the world objective *and* the DOM support/query objective (`adapt_loss_all_k` through
# CamUnifiedModel.adapt_loss) on recorded SwitchEnv episodes, then the hindsight LP of `observe_outcomes` (full fused
# predictive) LP_0 = log p(o_1 | B_1) - log p(o_1 | B_0) on fresh unswitched episodes (payout +1 every pull).

@torch.no_grad()
def measure_lp(stack, seed0: int, n_rows: int = 16, steps: int = 4) -> dict:
    """Hindsight LP per step of a greedy evaluation episode (`observe_outcomes` 'lp': nats per observed outcome,
    and the total over the step's observed outcomes); step index t reports LP_{t-1} (o_t under B_t vs B_{t-1})."""
    model = stack.model
    env = SwitchEnv(n_rows, noise=0.0, fixed_switch=99)
    ep = Episode(model, env, ScoreEvaluator(env), [seed0 + b for b in range(n_rows)], greedy=True)
    per, tot, ig = [], [], []
    for _ in range(steps):
        rec = ep.step()
        lp = rec.info.get("lp")
        n = model.belief_view(rec.info["outcomes"]).y_mask.float().sum(-1).cpu()
        per.append(None if lp is None else float(lp.float().mean()))
        tot.append(None if lp is None else float((lp.float().cpu() * n).mean()))
        bc = rec.info.get("belief_change")
        ig.append(None if bc is None else float(bc["info_gain"].mean()))
    return {"lp_per_outcome": per, "lp_total": tot, "info_gain": ig,
            "first_informative_lp_total": tot[1] if len(tot) > 1 else None,
            "first_informative_lp_per_outcome": per[1] if len(per) > 1 else None}


def train_world_adapt(stack, groups: list, st: CausalSettings) -> list:
    """World objective + the DOM support/query objective (`CamUnifiedModel.adapt_loss`, all k) on fixed episode
    groups, cycled, re-tokenised every step, with the configured linear LR decay. Returns per-step loss records."""
    model, trainer, agent = stack.model, stack.trainer, stack.agent
    opt = getattr(trainer, "opt", None)
    base = [g.get("base_lr", g["lr"]) for g in opt.param_groups] if (st.lr_decay and opt is not None) else None
    hist = []
    for i in range(st.train_steps):
        if base is not None:
            f = 1.0 - (1.0 - st.lr_final_frac) * (i / max(1, st.train_steps - 1))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
                if "base_lr" in g:
                    g["base_lr"] = b * f
        model.train()
        seq, aux = agent.tokenise_sequences(groups[i % len(groups)])
        filt = model.filter_sequence(seq, aux)
        w = model.world_loss(seq, filt)
        a, am = model.adapt_loss(seq, filt, None)
        trainer.step({"world": w["world"], "adapt": a})
        model.version = model.version + 1
        hist.append({"world": float(w["world"].detach()), "adapt": float(a.detach()),
                     "adapt_kalman": am.get("adapt_kalman"), "adapt_ks": am.get("adapt_ks")})
    if base is not None:
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b
            if "base_lr" in g:
                g["base_lr"] = b
    model.eval()
    return hist


def run_lp_probe(st: CausalSettings, root: Optional[str] = None, device=None) -> dict:
    """Record SwitchEnv episodes with the untrained dom_dev agent, train world + adapt objectives for
    ``st.train_steps`` steps (LR decay as configured), measure LP before and after."""
    from ..config import make_preset
    t0 = time.time()
    tmp = root or tempfile.mkdtemp(prefix="cam_w6_lp_")
    rng = random.Random(st.seed)
    cfg = make_preset(st.preset or "dom_dev", use_imagination=False, use_memory=False, lr=st.lr,
                      grad_clip=st.grad_clip, train_horizons=tuple(st.horizons), **st.cfg_overrides)
    ckw = {}
    if st.wm_kwargs:
        ckw["world_model"] = dict(st.wm_kwargs)
    if st.core_kwargs:
        ckw["core"] = dict(st.core_kwargs)
    stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"lp_s{st.seed}"),
                        component_kwargs=ckw or None)
    env = make_env("switch", st.B)
    stack.info["tools"] = env.available_tools()[0]
    eps = record(stack, env, st.record_episodes, st.seed * 10_007)
    groups = groups_of(eps, st.batch_seqs, rng)
    dev0 = 200_000 + 1000 * st.seed
    before = measure_lp(stack, dev0)
    model = stack.model
    hist = train_world_adapt(stack, groups, st)
    after = measure_lp(stack, dev0)
    return {"settings": asdict(st), "before": before, "after": after, "train_first": hist[:3],
            "train_last": hist[-3:], "adapt_last10": mean([h["adapt"] for h in hist[-10:]]),
            "world_last10": mean([h["world"] for h in hist[-10:]]), "components": stack.real_component_report(),
            "wall_s": time.time() - t0}


# ================================================================================================ W6 ProbeEnv margin
# "ProbeEnv h=1 evaluator margin between test and safe actions is >= 0 once trained" (plan 2026-11-26 §3 W6). W5's
# decision evaluator (r_psi + Phi + gamma V) is not part of this branch, so the margin is measured with the W6
# one-step evaluator on observable outcomes: Q(a) = E[payout(a) | b] + E_{y ~ p(y | b, a)} max_{a'} E[payout(a') | b'(a,
# y)] (`world.voi.decision_voi` with `outcome_q` on the points counter, symexp of the predicted symlog delta; cloned
# belief updates; observation-only).
# Truth (p_good 0.5, win 10, loss 10, safe 1, test cost 1): Q(test) = -1 + 5.5 = 4.5, Q(safe) = 1 + 1 = 2, margin 2.5.

class PointsProbeEnv(ProbeEnv):
    """ProbeEnv with the observable points counter of the causal worlds (observation-only payouts): every result
    and the initial observations carry the running total."""

    name = "probe_pts"

    def _new(self, rng, b):
        h = super()._new(rng, b)
        h["total"] = 0.0
        return h

    def _obs0(self, h, b):
        return list(super()._obs0(h, b)) + [_points(0.0)]

    def _act(self, h, a, rng, b):
        ok, obs, r, err = super()._act(h, a, rng, b)
        h["total"] += float(r)
        return ok, list(obs) + [_points(h["total"])], r, err


@torch.no_grad()
def measure_probe_margin(stack, seed0: int, n_rows: int = 16, n_y: int = 8,
                         payout_key: str = "num:points@me[points]") -> dict:
    """Q(run_test) - Q(play_safe) at the first decision of fresh PointsProbeEnv episodes (see the section note)."""
    from ..world.voi import VOIState, decision_voi, outcome_q
    model = stack.model
    model.eval()
    env = PointsProbeEnv(n_rows, horizon=3)
    ep = Episode(model, env, ScoreEvaluator(env), [seed0 + b for b in range(n_rows)], greedy=True)
    dec, _ = ep.decide()
    state = ep.state
    for b in range(n_rows):
        state.trackers[b].ingest(ep.obs_lists[b])
    _, prosp = model.prospective_batch(state)
    goal = model.tokenizer.goal(env.goals())
    info = [[Action("run_test"), Action("play_safe")] for _ in range(n_rows)]
    cont = [[Action("launch"), Action("play_safe")] for _ in range(n_rows)]
    q = outcome_q({payout_key: 1.0}, transform=symexp)        # payout units (the delta is predicted in symlog)
    with model._autocast():
        now = q(model, dec.R, state.fast, goal, info, prosp, dec.dt).float()             # [B, 2]
    v = decision_voi(model, VOIState(dec.R, state.fast, goal, prosp, dt=dec.dt), info, n_y, next_actions=cont,
                     q_fn=q, seed=seed0)
    after = v["voi"] + v["base"].unsqueeze(-1)                                          # E_y max_a' Q(b'(a, y), a')
    Q = now + after
    margin = Q[:, 0] - Q[:, 1]
    qa = v["q_after"][:, 0]                                                             # [B, n_y] after the test
    return {"margin_mean": float(margin.mean()), "margin_min": float(margin.min()),
            "frac_rows_margin_ge_0": float((margin >= 0).float().mean()),
            "Q_test": float(Q[:, 0].mean()), "Q_safe": float(Q[:, 1].mean()),
            "now_test": float(now[:, 0].mean()), "now_safe": float(now[:, 1].mean()),
            "after_test": float(after[:, 0].mean()), "after_safe": float(after[:, 1].mean()),
            "voi_test": float(v["voi"][:, 0].mean()), "voi_safe": float(v["voi"][:, 1].mean()),
            "q_base_launch": float(v["q_base"][:, 0].mean()), "q_base_safe": float(v["q_base"][:, 1].mean()),
            "q_after_test_sd_over_y": float(qa.std(-1).mean()), "n_rows": n_rows, "n_y": n_y,
            "truth": {"Q_test": 4.5, "Q_safe": 2.0, "margin": 2.5}}


def run_probe_margin(st: CausalSettings, root: Optional[str] = None, device=None) -> dict:
    """Record PointsProbeEnv episodes (horizon 3, exploring starts: every first tool occurs) with the untrained
    dom_dev agent, train world + adapt objectives, measure the h=1 margin before and after."""
    from ..config import make_preset
    t0 = time.time()
    tmp = root or tempfile.mkdtemp(prefix="cam_w6_probe_")
    rng = random.Random(st.seed)
    cfg = make_preset(st.preset or "dom_dev", use_imagination=False, use_memory=False, lr=st.lr,
                      grad_clip=st.grad_clip, train_horizons=tuple(st.horizons), **st.cfg_overrides)
    ckw = {}
    if st.wm_kwargs:
        ckw["world_model"] = dict(st.wm_kwargs)
    if st.core_kwargs:
        ckw["core"] = dict(st.core_kwargs)
    stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"probe_s{st.seed}"),
                        component_kwargs=ckw or None)
    if st.sem_cache_mb and hasattr(stack.semantic, "cache_max_bytes"):
        stack.semantic.cache_max_bytes = int(st.sem_cache_mb) * 2 ** 20
    env = PointsProbeEnv(st.B, horizon=3, exploring_starts=True)
    stack.info["tools"] = env._all_tools()
    eps = record(stack, env, st.record_episodes, st.seed * 10_007)
    stack.agent.fresh = []
    stack.agent._last_fresh = {}
    groups = groups_of(eps, st.batch_seqs, rng)
    dev0 = 300_000 + 1000 * st.seed
    before = measure_probe_margin(stack, dev0)
    hist = train_world_adapt(stack, groups, st)
    after = measure_probe_margin(stack, dev0)
    return {"settings": asdict(st), "before": before, "after": after, "train_first": hist[:3],
            "train_last": hist[-3:], "world_last10": mean([h["world"] for h in hist[-10:]]),
            "adapt_last10": mean([h["adapt"] for h in hist[-10:]]), "n_episodes": len(eps),
            "first_tools": sorted({e[0].action.tool for e in eps if e and e[0].action is not None}),
            "components": stack.real_component_report(), "wall_s": time.time() - t0}


# ================================================================================================ W6 DOM driver
# Real end-to-end check of plan 2026-11-26 §3 W6: run_causal on the real dom_dev stack, one world and seed per call
# (short GPU holds), and the pass/fail summary of the plan's criteria over the per-(world, seed) result files.
#
#     python -m superlearner.cam_unified.capabilities.causal run --world delay --seed 0 --out r.json
#     python -m superlearner.cam_unified.capabilities.causal summary r_*.json --out summary.json
#
# Criteria (held-out data or explicit interventions, payout in observable units): t1 mean |pairwise action
# difference| > 0.5 over all held-out steps; t2 delay max error < 0.3; |t3 interaction contrast - 2| < 0.4; t4 switch
# correct (post-switch < -0.7, unswitched > 0.7); consistency MSE/Var < 0.1 (teacher-forced and mode-latent prior).

DOM_SETTINGS = dict(preset="dom_dev", train_steps=800, lr=1e-3, lr_decay=True, lr_final_frac=0.05, grad_clip=10.0,
                    B=32, record_episodes=12, batch_seqs=4, horizons=(1, 2, 3, 4),
                    wm_kwargs={"open_loop_samples": 2}, cfg_overrides={"grad_checkpoint": True}, sem_cache_mb=256,
                    # the dom preset's state_emb_std=0.02 (W3's switch, not yet read by the seam core) through the
                    # real UnifiedCore's own constructor argument
                    core_kwargs={"emb_std": 0.02})


def criteria(worlds: dict) -> dict:
    """{criterion: {"value", "pass"}} for the worlds present in one result."""
    out = {}
    for w, r in worlds.items():
        t1 = r.get("t1_actions_differ", {}).get("mean_abs_pairwise_diff")
        if w != "switch":                  # the switch world has a single action: no pair to compare
            out[f"{w}/t1_mean_abs_pairwise_diff"] = {"value": t1, "pass": t1 is not None and t1 == t1 and t1 > 0.5}
        c = r.get("consistency") or {}
        for k in ("cons_mse_var_teacher_forced", "cons_mse_var_mode1"):
            v = c.get(k)
            out[f"{w}/{k}"] = {"value": v, "pass": v is not None and v < 0.1}
        if "t2_delay" in r:
            v = r["t2_delay"]["max_abs_err"]
            out[f"{w}/t2_max_abs_err"] = {"value": v, "pass": v == v and v < 0.3}
        if "t3_interaction" in r:
            v = r["t3_interaction"]["interaction_contrast"]
            out[f"{w}/t3_contrast"] = {"value": v, "pass": v == v and abs(v - 2.0) < 0.4}
        if "t4_switch" in r:
            s = r["t4_switch"]
            out[f"{w}/t4_switch"] = {"value": [s["post_switch_pred_mean_t5_7"], s["no_switch_pred_mean_t5_7"]],
                                     "pass": bool(s.get("switch_correct"))}
    return out


def dom_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summary", "lp", "probe"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--world", default="delay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="CausalSettings override key=json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        kw = dict(DOM_SETTINGS)
        for item in a.set:
            k, v = item.split("=", 1)
            try:
                kw[k] = json.loads(v)
            except json.JSONDecodeError:
                kw[k] = v
        if "horizons" in kw:
            kw["horizons"] = tuple(kw["horizons"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        res = run_probe_margin(CausalSettings(seed=a.seed, worlds=("probe",), **kw), device=a.device)
        res["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        res["command"] = " ".join([sys.executable, "-m", "superlearner.cam_unified.capabilities.causal"]
                                  + list(argv if argv is not None else sys.argv[1:]))
        m = res["after"]["margin_mean"]
        res["criterion"] = {"margin_mean": m, "pass": m == m and m >= 0}
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print("margin before", res["before"]["margin_mean"], "after", m, "->",
              "PASS" if res["criterion"]["pass"] else "FAIL")
        return 0
    if a.cmd == "lp":
        kw = dict(DOM_SETTINGS)
        for item in a.set:
            k, v = item.split("=", 1)
            try:
                kw[k] = json.loads(v)
            except json.JSONDecodeError:
                kw[k] = v
        if "horizons" in kw:
            kw["horizons"] = tuple(kw["horizons"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        res = run_lp_probe(CausalSettings(seed=a.seed, worlds=("switch",), **kw), device=a.device)
        res["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        res["command"] = " ".join([sys.executable, "-m", "superlearner.cam_unified.capabilities.causal"]
                                  + list(argv if argv is not None else sys.argv[1:]))
        lp = res["after"]["first_informative_lp_total"]
        res["criterion"] = {"first_informative_lp_total": lp, "pass": lp is not None and lp > 0.1}
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print("LP before", res["before"]["lp_total"], "after", res["after"]["lp_total"],
              "->", "PASS" if res["criterion"]["pass"] else "FAIL")
        return 0
    if a.cmd == "run":
        kw = dict(DOM_SETTINGS)
        for item in a.set:
            k, v = item.split("=", 1)
            try:
                kw[k] = json.loads(v)
            except json.JSONDecodeError:
                kw[k] = v
        if "horizons" in kw:
            kw["horizons"] = tuple(kw["horizons"])
        st = CausalSettings(seed=a.seed, worlds=(a.world,), **kw)
        t0 = time.time()
        root = tempfile.mkdtemp(prefix=f"w6_causal_{a.world}_s{a.seed}_")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        res = run_causal(st, root=root, device=a.device)
        res["criteria"] = criteria(res["worlds"])
        res["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        res["wall_s_total"] = time.time() - t0
        res["command"] = " ".join([sys.executable, "-m", "superlearner.cam_unified.capabilities.causal"]
                                  + list(argv if argv is not None else sys.argv[1:]))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        for k, v in res["criteria"].items():
            print(f"{k}: {v['value']} -> {'PASS' if v['pass'] else 'FAIL'}")
        return 0
    rows = {}
    for fn in a.files:
        with open(fn) as f:
            r = json.load(f)
        seed = r["settings"]["seed"]
        for k, v in criteria(r["worlds"]).items():
            rows.setdefault(k, {})[seed] = v
    summ = {k: {"per_seed": v, "all_pass": all(x["pass"] for x in v.values()), "n_seeds": len(v)}
            for k, v in sorted(rows.items())}
    with open(a.out, "w") as f:
        json.dump(summ, f, indent=1, default=str)
    for k, v in summ.items():
        print(k, "ALL PASS" if v["all_pass"] else "FAIL", {s: x["value"] for s, x in v["per_seed"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(dom_main())
