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
"""
from __future__ import annotations

import copy
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..interfaces import Action, SequenceBatch
from ..outcome_spec import CORE_INDEX
from ..world.baselines import (KeyMeanBaseline, LinearRegressionBaseline, NoChangeBaseline, TrendPersistenceBaseline,
                               WorldModelForecaster, evaluate_multistep)
from .envs import DelayEffectEnv, InteractionEnv, ProbeEnv, ScoreEvaluator, SwitchEnv
from .imagination import config_override, imagination_settings
from .loop import Episode, mean, train_agent
from .stack import build_stack, capability_config

R_IDX = CORE_INDEX["reward"]


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


def train_world(stack, groups: list, steps: int) -> list:
    """Optimise the model's world objective (the agent's objective_world: acting-consistent filter + WorldModel
    loss) on fixed episode groups, cycled, re-tokenised every step exactly as `CamUnifiedAgent.update` does; each
    step bumps the model version (weight-dependent caches) and updates the EMA target. Returns the loss history."""
    model, trainer, agent = stack.model, stack.trainer, stack.agent
    hist = []
    for i in range(steps):
        model.train()
        seq, aux = agent.tokenise_sequences(groups[i % len(groups)])
        filt = model.filter_sequence(seq, aux)
        out = model.world_loss(seq, filt)
        trainer.step({"world": out["world"]})
        model.version = model.version + 1
        hist.append(float(out["world"].detach()))
    model.eval()
    return hist


def _with_episode(seq: SequenceBatch, steps: Optional[int] = None) -> SequenceBatch:
    return SequenceBatch(seq.steps[:steps] if steps else seq.steps)


@torch.no_grad()
def forecast(model, seq: SequenceBatch, start: int, horizons, n_samples: int = 8) -> dict:
    g = torch.Generator(device=model.device).manual_seed(0)
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
        specs = [state.trackers[b].realize(obs[b], acts[b], results[b], nxt[b], new[b] - scores[b]) for b in range(B)]
        for b in range(B):
            recs[b].append(EventRecord(event_id=f"s{b}_{k}", env_id=env.env_ids[b], actor_id="script",
                                       episode_id=f"ep{b}", step=k, model_version=0, t_sim=t[b],
                                       observations=list(obs[b]), action=acts[b], result=results[b],
                                       reward=new[b] - scores[b], done=bool(adv.done[b]), provenance=Provenance.REAL,
                                       extra={"outcomes": [s.to_dict() for s in specs[b]],
                                              "dt": float(adv.t[b] - t[b]), "goal": {"goal_id": "g", "text":
                                                                                       env.goals()[b].text,
                                                                                       "metrics": {},
                                                                                       "deadline_days": None}}))
        scores, obs, t = new, nxt, list(adv.t)
    seq, aux = stack.agent.tokenise_sequences(recs)
    return seq, aux, [dict(h) for h in env.hidden]


# ------------------------------------------------------------------------------------------------ the seven tests

def measure_actions_differ(stack, heldout) -> dict:
    """(1) From the same filtered state Z_t, each contemplated tool's prior transition gives different reward
    predictions; reported as the mean absolute pairwise difference and the per-tool means."""
    model = stack.model
    seq, aux = heldout[0]
    filt = model.filter_sequence(seq, aux)
    Z = filt.Zs[2]
    st = seq.steps[2]
    B = Z.shape[0]
    names = [t.name for t in stack.info["tools"]]
    preds = {}
    with torch.no_grad(), model._autocast():
        for n in names:
            acts = [Action(tool=n) for _ in range(B)]
            A = model.tokenizer.action(acts, [stack.info["tools"]] * B)
            Zh, _ = model.world_model.prior(Z, A, filt.beliefs[2], st.dt, sample=False)
            d = model.world_model.outcome_dist(Zh, st.outcomes)
            preds[n] = d.mean()[:, R_IDX].float().cpu()
    diffs = [float((preds[a] - preds[b]).abs().mean()) for i, a in enumerate(names) for b in names[i + 1:]]
    return {"pred_reward": {n: float(v.mean()) for n, v in preds.items()}, "mean_abs_pairwise_diff": mean(diffs)}


def measure_delay(stack, seed0: int) -> dict:
    """(2) Open-loop forecasts after a fixed prefix of waits, then press_now / press_later / wait at step 2."""
    env = DelayEffectEnv(3, noise=0.0)
    plans = [["wait_turn"] * 2 + [first] + ["wait_turn"] * 5 for first in ("press_now", "press_later", "wait_turn")]
    seq, aux, _ = scripted_sequences(stack, env, plans, seed0)
    f = forecast(stack.model, seq, 2, (1, 2, 3))
    pred = torch.stack([f[h]["mean"][:, R_IDX] for h in (1, 2, 3)], 1).cpu()
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
    pred = f[1]["mean"][:, R_IDX].cpu()
    truth = torch.tensor([InteractionEnv.truth(x, w) for x, w in settings])
    contrast = float(pred[3] - pred[1] - pred[2] + pred[0])
    lin = LinearRegressionBaseline(horizons=(1,)).fit([b for b, _ in train_batches]).predict(seq, 1, (1,))[1]
    lin = lin[:, R_IDX].cpu() if lin.dim() == 2 else lin.cpu()
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
        filt = model.filter_sequence(seq, aux)
        preds = []
        with torch.no_grad(), model._autocast():
            for t in range(8):
                st = seq.steps[t]
                Zh, _ = model.world_model.prior(filt.Zs[t], st.act, filt.beliefs[t], st.dt, sample=False)
                preds.append(float(model.world_model.outcome_dist(Zh, st.outcomes).mean()[:, R_IDX].float().mean()))
        out[name] = preds
    out["truth_switch"] = [1.0] * 3 + [-1.0] * 5
    out["post_switch_pred_mean_t5_7"] = mean(out["switch_at_3"][5:])
    out["no_switch_pred_mean_t5_7"] = mean(out["no_switch"][5:])
    out["pre_switch_max_gap"] = max(abs(a - b) for a, b in zip(out["switch_at_3"][:3], out["no_switch"][:3]))
    return out


def measure_multistep(stack, train_batches, heldout, horizons) -> dict:
    """(5) Open-loop error vs horizon of the world model and the baselines on identical held-out (start, h)."""
    fc = {"world_model": WorldModelForecaster(stack.model.world_model, n_samples=8),
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
        base, _, _ = model.base_moments(dec.Z_hat, state.fast, prosp)
        mean1, _, _ = model.predict_outcomes(dec.Z_hat, state.fast, prosp)
        for name, val in (("imagined_plus1", 1.0), ("imagined_minus1", -1.0)):
            ys = mean1.clone().float()
            obs_m = torch.zeros_like(prosp.q.mask)
            for b in range(n_rows):
                for q, k in enumerate(prosp.keys[b]):
                    if k == "reward":
                        ys[b, q], obs_m[b, q] = val, True
                    elif k is not None and k.startswith("num:points"):
                        ys[b, q], obs_m[b, q] = math.copysign(math.log1p(abs(val)), val), True
            imagined = with_targets(prosp, ys, obs_m & prosp.q.mask)
            f1, _ = model.beliefs.update(state.fast.clone(), dec.Z, A, model.belief_view(imagined),
                                         model.outcome_head)
            f1 = model.kalman_update(f1, feat, model.belief_view(model.channels_of(state.fast, imagined)), base)
            Z2, _ = model.world_model.prior(dec.Z_hat, A, model.beliefs.tokens(f1), dec.dt, sample=False)
            m2, _, _ = model.predict_outcomes(Z2, f1, model.channels_of(f1, prosp))
            out[name] = float(m2[:, R_IDX].float().mean())
    out["next_pred_real_belief"] = float(mean1[:, R_IDX].float().mean())
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
        cfg = capability_config(use_imagination=False, use_memory=False, lr=st.lr, grad_clip=st.grad_clip,
                                train_horizons=tuple(st.horizons), **st.cfg_overrides)
        stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"{w}_s{st.seed}"))
        env = make_env(w, st.B)
        stack.info["tools"] = env.available_tools()[0]
        eps = record(stack, env, st.record_episodes, st.seed * 10_007)
        rng.shuffle(eps)
        n_ho = max(2, int(len(eps) * st.heldout_frac))
        heldout_g = groups_of(eps[:n_ho], st.batch_seqs, rng)
        train_g = groups_of(eps[n_ho:], st.batch_seqs, rng)
        hist = train_world(stack, train_g, st.train_steps)
        heldout, train = tokenise(stack.agent, heldout_g), tokenise(stack.agent, train_g)
        res = {"n_train_episodes": len(eps) - n_ho, "n_heldout_episodes": n_ho, "loss_first": hist[0],
               "loss_last10": mean(hist[-10:]), "components": stack.real_component_report()}
        dev0 = 100_000 + 1000 * st.seed
        res["t1_actions_differ"] = measure_actions_differ(stack, heldout)
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
