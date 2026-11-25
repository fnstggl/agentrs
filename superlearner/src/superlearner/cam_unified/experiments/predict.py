"""Support/query prediction harness (Experiment 1 conditions A, C, D; Experiment 2 intervention-effect accuracy).

Identical support for every learner. For each evaluation task one world runs a fixed *design policy* (seeded per
task: every decision sets one uniformly chosen lever to a uniformly chosen value of its public range), so each support
step is one decision followed by exactly one facility tick. The agent-view record of every step (observations before
the decision, the action, its result, the observations after it and the verified reward the agent receives) and the
numeric view of the same tick (Crucible's architecture-neutral vector: settings in force, gauge readings, reward) are
stored; every learner conditions on the first k steps of the *same* stream.

Queries. At every k in K_SET the world is forked (evaluator side, `mechrun.fork_rewards`) for `n_queries` candidate
interventions - holding the current settings plus random single-lever moves - and the true reward of the next tick
under each is recorded. Queries are never shown to any learner and the support never contains them (strict
separation): a learner predicts each query from (support_<k, the candidate action) only.

Scores per (task, k), all from each learner's own predictive distribution of the next-tick reward:
* ``logscore`` - mean log predictive density of the realised query rewards (proper scoring rule; nats);
* ``sqerr``    - mean squared error of the predictive mean (reward units);
* ``cover90``  - fraction of query rewards inside the central 90% predictive interval (calibration);
* ``regret``   - decision quality from the same identical support: the true reward lost by choosing the candidate
  with the highest predicted mean (ties broken uniformly) instead of the best candidate, normalised by the task's private
  (reference - default) net per tick (1 = the whole improvement the reference policy achieves over the defaults).

Condition C adds the predecessor notes of `mechrun.PredecessorNotes` to the reset observations (only learners that
read text can use them; their numeric content is kept for a numeric reference). Condition D runs the same design
through `mechrun.DelayedPartial`: half the gauges hidden, money (and the agent's reward) `delay` ticks late; numeric
learners get the same information (hidden gauge columns masked, rewards of the last `delay` support steps missing).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..interfaces import Action, ActionSource, EntityRegistry, EventRecord, GoalSpec, Provenance, new_id
from . import mechrun as M

K_SET = (0, 1, 2, 4, 8, 16, 32, 64)


@dataclass
class SupportEpisode:
    task: object
    condition: str
    lever_names: list                  # shown names, engine order
    lo: np.ndarray                     # [L] public ranges
    hi: np.ndarray
    integer: list
    x0: np.ndarray                     # [L] normalised resting settings
    X: np.ndarray                      # [T, L] normalised settings in force during tick t (after decision t)
    G: np.ndarray                      # [T, Gn] symlog gauge readings of tick t (hidden columns zero)
    gmask: np.ndarray                  # [Gn] 1 = observed gauge
    Y: np.ndarray                      # [T] true reward of tick t
    delay: int                         # rewards of support step t are known at k > t + delay
    steps: list                        # [T] agent-view dicts {obs, action, result, next_obs, reward}
    obs_final: list                    # observations delivered after the last support step (before decision T)
    queries: dict                      # k -> {"actions": [Action], "X": [Q, L], "Y": [Q]}
    scale_gap: float                   # (reference - default) net per tick / money scale (regret normaliser)
    notes: list = field(default_factory=list)        # condition C: [(lever j, value raw, base net, after net)]
    tools: list = field(default_factory=list)        # the env's tool list (ToolSchema)
    goal: Optional[GoalSpec] = None

    def revealed(self, k: int) -> np.ndarray:
        """[k] bool: rewards of the first k support steps a learner knows at decision k."""
        t = np.arange(k)
        return t < k - self.delay if self.delay > 0 else np.ones(k, dtype=bool)


def _norm(v, lo, hi) -> float:
    return 0.0 if hi <= lo else 2.0 * (float(v) - lo) / (hi - lo) - 1.0


def _set_action(name: str, value, integer: bool, tools: Sequence) -> Action:
    """A design / query action (ActionSource.BASELINE). Its ``logp_parts["decode"]`` holds only the schema key of the
    tool (``{"key", "name"}``) so that A's tokenizer resolves the exact schema (lever names are short neutral names
    that can repeat across worlds with different ranges); there is no generator decode record - these actions are
    never scored by a policy objective."""
    v = int(round(value)) if integer else float(round(float(value), 4))
    tool = next(t for t in tools if t.name == f"set_{name}")
    return Action(tool.name, {"value": v}, source=ActionSource.BASELINE,
                  logp_parts={"decode": {"key": tool.key, "name": tool.name}})


def collect_support(tasks: Sequence, *, k_max: int = 64, n_queries: int = 12, design_seed: int = 0,
                    condition: str = "A", delay: int = 3, hide_frac: float = 0.5, notes_m: int = 0,
                    surface=None, ks: Sequence[int] = K_SET) -> list:
    """Run the design policy on all `tasks` (one batched world) and return one SupportEpisode per task."""
    cond = condition.upper()
    world = M.make_world(tasks, surface=surface, seed=design_seed)
    dp = M.DelayedPartial(world, delay=delay, hide_frac=hide_frac, seed=design_seed) if cond == "D" else None
    ob = dp.reset() if dp is not None else world.adapter.reset()
    B = len(tasks)
    op = world.operator
    notes_obs = [[] for _ in range(B)]
    notes_raw = [[] for _ in range(B)]
    if notes_m:
        pn = M.PredecessorNotes()
        for b in range(B):
            r = random.Random(f"notes|{tasks[b].id}|{design_seed}")
            obs = pn.notes(world, b, notes_m, r)
            notes_obs[b] = obs
            r = random.Random(f"notes|{tasks[b].id}|{design_seed}")      # replay the same draws for the numbers
            for _ in range(notes_m):
                e = op.envs[b]
                j = r.randrange(len(e.lever_spec))
                lv = e.lever_spec[j]
                v = r.uniform(lv.lo, lv.hi)
                v = float(int(round(v))) if lv.integer else round(v, 3)
                base, after = pn._experiment(e.task, op.spec, j, v)
                notes_raw[b].append((j, v, base, after))
                r.randrange(3)                                         # the wording draw
    obs_now = [list(ob.observations[b]) + notes_obs[b] for b in range(B)]
    rngs = [random.Random(f"design|{t.id}|{design_seed}") for t in tasks]
    qrngs = [random.Random(f"query|{t.id}|{design_seed}") for t in tasks]
    envs = op.envs
    L = [len(e.lever_spec) for e in envs]
    meta = [_env_meta(world, b, dp) for b in range(B)]
    steps = [[] for _ in range(B)]
    Xs = [[] for _ in range(B)]
    Gs = [[] for _ in range(B)]
    queries = [dict() for _ in range(B)]
    think = M.THINK_S / M.DAY
    tools0 = world.adapter.available_tools()
    for t in range(k_max + 1):
        if t in ks:
            for b in range(B):
                queries[b][t] = make_queries(world, b, qrngs[b], n_queries, tools0[b], meta[b])
        if t == k_max:
            break
        acts = []
        for b, e in enumerate(envs):
            j = rngs[b].randrange(L[b])
            lv = e.lever_spec[j]
            v = rngs[b].uniform(lv.lo, lv.hi)
            acts.append(_set_action(e.lever_shown[j], v, lv.integer, tools0[b]))
        res = world.adapter.execute(acts)
        dur = [(r.duration if r is not None else 0.0) + think for r in res]
        adv = dp.advance(dur) if dp is not None else world.adapter.advance_time(dur)
        rew = dp.take_rewards(adv.done) if dp is not None else world.evaluator.take_rewards()
        for b, e in enumerate(envs):
            lo, hi, _, gm = meta[b]
            nxt = list(res[b].observations) + list(adv.observations[b])
            steps[b].append({"obs": obs_now[b], "action": acts[b], "result": res[b], "next_obs": nxt,
                             "reward": float(rew[b]), "t": float(ob.t[b]) if t == 0 else float(prev_t[b])})
            Xs[b].append([_norm(e.raw[j], lo[j], hi[j]) for j in range(L[b])])
            last = e.env.sims[0].last[0]
            g = np.array([float(last["readings"][n]) for n in e.gauge_engine], dtype=np.float64)
            Gs[b].append(np.sign(g) * np.log1p(np.abs(g)) * gm)
            obs_now[b] = nxt
        prev_t = list(adv.t)
        if any(adv.done):
            raise RuntimeError("a support episode ended before k_max decisions; use tasks with horizon > k_max")
    out = []
    for b, task in enumerate(tasks):
        e = envs[b]
        lo, hi, x0, gm = meta[b]
        gap = (task.reference_net - task.default_net) / max(1e-9, float(op.envs[b].goal.meta["reward_scale_cents"])) \
            if task.validated else float("nan")
        out.append(SupportEpisode(task, cond, list(e.lever_shown), lo, hi, [lv.integer for lv in e.lever_spec], x0,
                                  np.array(Xs[b]), np.array(Gs[b]), gm, np.array(world.rewards[b][:k_max]),
                                  delay if cond == "D" else 0, steps[b], obs_now[b], queries[b], float(gap),
                                  notes_raw[b], list(tools0[b]), world.adapter.goals()[b]))
    return out


def make_queries(world, b: int, rng: random.Random, n_queries: int, tools: Sequence, meta: tuple) -> dict:
    """Evaluator side: hold + (n_queries - 1) random single-lever moves from env b's current settings, with the true
    next-tick reward of each from a fork of the world."""
    e = world.operator.envs[b]
    lo, hi = meta[0], meta[1]
    L = len(e.lever_spec)
    cur = list(e.raw)
    acts, raws = [_set_action(e.lever_shown[0], cur[0], e.lever_spec[0].integer, tools)], [list(cur)]
    acts[0].logp_parts["hold"] = True
    for _ in range(n_queries - 1):
        j = rng.randrange(L)
        lv = e.lever_spec[j]
        v = rng.uniform(lv.lo, lv.hi)
        v = float(int(round(v))) if lv.integer else v
        raw = list(cur)
        raw[j] = float(lv.clip(v))
        acts.append(_set_action(e.lever_shown[j], raw[j], lv.integer, tools))
        raws.append(raw)
    ys = M.fork_rewards(world, b, raws)
    return {"actions": acts, "raw": raws, "Y": np.array(ys),
            "X": np.array([[_norm(r[j], lo[j], hi[j]) for j in range(L)] for r in raws])}


def _env_meta(world, b: int, dp=None) -> tuple:
    e = world.operator.envs[b]
    L = len(e.lever_spec)
    lo = np.array([lv.lo for lv in e.lever_spec], dtype=np.float64)
    hi = np.array([lv.hi for lv in e.lever_spec], dtype=np.float64)
    x0 = np.array([_norm(e.raw[j], lo[j], hi[j]) for j in range(L)])
    gm = np.ones(len(e.gauge_engine))
    if dp is not None:
        for k_, name in enumerate(e.gauge_shown):
            if name in dp.hidden[b]:
                gm[k_] = 0.0
    return lo, hi, x0, gm


class TrajectoryRecorder:
    """`mechrun.run_batch` round callback that records an acting learner's own trajectory as SupportEpisodes
    (condition B: agent-selected experiments) with forked queries at every round in `ks` (evaluator side). Steps are
    decisions; X / G / Y hold the settings in force after the decision, the readings of the latest tick and the reward
    the learner received for the decision."""

    def __init__(self, world, ks: Sequence[int] = K_SET, n_queries: int = 12, seed: int = 0, delayed=None,
                 condition: str = "B"):
        self.w, self.ks, self.nq, self.seed, self.dp, self.cond = world, set(ks), n_queries, seed, delayed, condition
        self.steps, self.X, self.G, self.Y, self.queries, self.meta = [], [], [], [], [], []
        self.prev = None
        self.tools = None
        self.final = []

    def __call__(self, r, acts, res, rew, nob) -> None:
        w = self.w
        B = w.B
        if r == 0:
            self.tools = w.adapter.available_tools()
            self.meta = [_env_meta(w, b, self.dp) for b in range(B)]
            self.steps, self.X, self.G, self.Y = ([[] for _ in range(B)] for _ in range(4))
            self.queries = [dict() for _ in range(B)]
            self.qrng = [random.Random(f"query|{w.tasks[b].id}|{self.seed}") for b in range(B)]
        else:
            for b in range(B):
                if acts[b] is None:
                    continue
                e = w.operator.envs[b]
                lo, hi, _, gm = self.meta[b]
                self.steps[b].append({"obs": list(self.prev.observations[b]), "action": acts[b], "result": res[b],
                                      "next_obs": list(nob.observations[b]), "reward": float(rew[b]),
                                      "t": float(self.prev.t[b])})
                self.X[b].append([_norm(e.raw[j], lo[j], hi[j]) for j in range(len(e.lever_spec))])
                last = e.env.sims[0].last[0] if e.env.sims[0].last else None
                if last is not None and last.get("readings"):
                    g = np.array([float(last["readings"][n]) for n in e.gauge_engine], dtype=np.float64)
                else:
                    g = np.zeros(len(e.gauge_engine))
                self.G[b].append(np.sign(g) * np.log1p(np.abs(g)) * gm)
                self.Y[b].append(float(rew[b]))
        if r in self.ks:
            for b in range(B):
                if not nob.done[b]:
                    self.queries[b][r] = make_queries(w, b, self.qrng[b], self.nq, self.tools[b], self.meta[b])
        self.prev = nob

    def episodes(self) -> list:
        w = self.w
        out = []
        for b, task in enumerate(w.tasks):
            lo, hi, x0, gm = self.meta[b]
            gap = (task.reference_net - task.default_net) / max(1e-9, float(w.operator.envs[b].goal.meta[
                "reward_scale_cents"])) if task.validated else float("nan")
            out.append(SupportEpisode(task, self.cond, list(w.operator.envs[b].lever_shown), lo, hi,
                                      [lv.integer for lv in w.operator.envs[b].lever_spec], x0,
                                      np.array(self.X[b]).reshape(len(self.X[b]), len(lo)),
                                      np.array(self.G[b]).reshape(len(self.G[b]), len(gm)), gm, np.array(self.Y[b]),
                                      self.dp.delay if self.dp is not None else 0, self.steps[b],
                                      list(self.prev.observations[b]), self.queries[b], float(gap), [],
                                      list(self.tools[b]), w.adapter.goals()[b]))
        return out


# ============================================================================ scoring

def gaussian_logpdf(y, mean, var):
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    return -0.5 * (np.log(2 * math.pi * var) + (np.asarray(y) - mean) ** 2 / var)


def student_logpdf(y, loc, scale2, dof):
    scale2 = np.maximum(np.asarray(scale2, dtype=np.float64), 1e-12)
    dof = np.broadcast_to(np.asarray(dof, dtype=np.float64), scale2.shape)
    lg = np.vectorize(math.lgamma)
    z2 = (np.asarray(y) - loc) ** 2 / scale2
    return (lg((dof + 1) / 2) - lg(dof / 2) - 0.5 * np.log(dof * math.pi * scale2)
            - (dof + 1) / 2 * np.log1p(z2 / dof))


def _t_central_mass(z, dof, n: int = 801) -> np.ndarray:
    """P(|T| <= z) for a standard Student-t with `dof` (trapezoid integration of the density; |error| < 1e-4 for the
    dof >= 2 that occur here)."""
    z = np.abs(np.asarray(z, dtype=np.float64))
    dof = np.broadcast_to(np.asarray(dof, dtype=np.float64), z.shape)
    out = np.empty_like(z)
    for i, (zi, di) in enumerate(zip(z.ravel(), dof.ravel())):
        g = np.linspace(0.0, min(zi, 200.0), n)
        c = math.exp(math.lgamma((di + 1) / 2) - math.lgamma(di / 2)) / math.sqrt(di * math.pi)
        pdf = c * (1 + g * g / di) ** (-(di + 1) / 2)
        out.ravel()[i] = min(1.0, 2.0 * np.trapezoid(pdf, g))
    return out


def inside90(y, mean, var, dof=None) -> np.ndarray:
    """Whether each y lies in the central 90% interval of its predictive."""
    sd = np.sqrt(np.maximum(np.asarray(var, dtype=np.float64), 1e-24))
    z = (np.asarray(y) - mean) / sd
    if dof is None:
        return np.abs(z) <= 1.6448536269514722
    return _t_central_mass(z, dof) <= 0.9


def score_queries(ep: SupportEpisode, k: int, mean, var, dof=None) -> dict:
    """Scores of one learner's predictive (mean, var[, dof]) [Q] for the queries at k. With `dof` the predictive is
    Student-t with squared scale `var`; else Gaussian."""
    q = ep.queries[k]
    y = q["Y"]
    mean, var = np.asarray(mean, dtype=np.float64), np.asarray(var, dtype=np.float64)
    lp = student_logpdf(y, mean, var, dof) if dof is not None else gaussian_logpdf(y, mean, var)
    # the candidate the predictive ranks best; ties (within 1e-3 of the mean predictive sd) are broken uniformly, so
    # a flat predictive (no information yet) scores the expected regret of a random choice among the tied candidates
    tol = 1e-6 + 1e-3 * float(np.mean(np.sqrt(np.maximum(var, 0))))
    tied = mean >= np.max(mean) - tol
    reg = float(np.max(y) - np.mean(y[tied]))
    return {"logscore": float(np.mean(lp)), "sqerr": float(np.mean((mean - y) ** 2)),
            "cover90": float(np.mean(inside90(y, mean, var, dof))),
            "regret": reg / ep.scale_gap if ep.scale_gap and math.isfinite(ep.scale_gap) and ep.scale_gap > 0
            else float("nan"), "regret_raw": reg}


# ============================================================================ CAM Unified as a predictor

def cam_records(model, ep: SupportEpisode, upto: Optional[int] = None) -> list:
    """The agent-view EventRecords of the support stream, with realised outcome specs from the model's own
    OutcomeTracker (exactly what CamUnifiedAgent.observe would archive), plus a final context record holding the
    observations delivered after the last support step (action None, no outcomes)."""
    tr = model.new_tracker()
    g = ep.goal
    gd = {"goal_id": g.goal_id, "text": g.text, "metrics": dict(g.metrics), "deadline_days": g.deadline_days} \
        if g is not None else None
    n = len(ep.steps) if upto is None else int(upto)
    env_id, ep_id = f"support-{ep.task.id}", new_id("sup_")
    recs = []
    for t, s in enumerate(ep.steps[:n]):
        specs = tr.realize(s["obs"], s["action"], s["result"], s["next_obs"], s["reward"])
        recs.append(EventRecord(new_id("e_"), env_id, "design", ep_id, t, 0, s["t"], list(s["obs"]), s["action"],
                                s["result"], reward=s["reward"], provenance=Provenance.REAL,
                                extra={"outcomes": [x.to_dict() for x in specs], "dt": M.TICK_S / M.DAY, "goal": gd,
                                       "retrieved": [], "option": None}))
    last_obs = ep.steps[n]["obs"] if n < len(ep.steps) else ep.obs_final
    recs.append(EventRecord(new_id("e_"), env_id, "design", ep_id, n, 0, 0.0, list(last_obs), None, None,
                            extra={"outcomes": [], "dt": 0.0, "goal": gd, "retrieved": [], "option": None}))
    return recs, tr


@torch.no_grad()
def cam_predict(stack, episodes: Sequence[SupportEpisode], ks: Sequence[int] = K_SET, chunk: int = 16) -> dict:
    """{(episode index, k): (mean [Q], var [Q])} of CAM Unified's predictive of the next-tick reward of each query.

    The support enters through the model's acting-consistent filter (`CamUnifiedModel.filter_sequence`: Z_t =
    F(Z_{t-1}, [O_t; A_{t-1}; TIME; B_t; ...])) and the support belief B_k is formed exactly as in the adaptation
    objective (`CamUnifiedModel.adapt_loss`): particle update and Kalman/NIG update for every support step t < k. Each
    query is then predicted from (Z_k, B_k, the query action) through the world model's prior and the outcome head
    mixed over particles and combined with the Kalman layer (`predict_outcomes`) - the same computation `decide`
    uses for its predictions. Query outcomes never enter."""
    model, agent = stack.model, stack.agent
    model.eval()
    out = {}
    kmax = max(ks)
    for c0 in range(0, len(episodes), chunk):
        eps = list(episodes[c0:c0 + chunk])
        B = len(eps)
        for ep in eps:
            model.tokenizer.register_tools(ep.tools)
        recs, trackers = zip(*[cam_records(model, ep, upto=kmax) for ep in eps])
        seq, aux = agent.tokenise_sequences(list(recs))
        filt = model.filter_sequence(seq, aux)
        Zs = filt.Zs
        steps = seq.steps
        dev = Zs[0].device
        fast = model.beliefs.init_fast(B, device=dev)
        fast.channels = [{} for _ in range(B)]
        kal = model._kalman_on()
        snaps = {}
        with model._autocast():
            for t in range(kmax + 1):
                if t in ks:
                    snaps[t] = fast.clone()
                if t == kmax:
                    break
                st = steps[t]
                ob = model.belief_view(st.outcomes)
                if not bool(ob.y_mask.any()):
                    continue
                Bt = model.beliefs.tokens(fast)
                Z_hat, _ = model.world_model.prior(Zs[t], st.act, Bt, st.dt, sample=False)
                ob = model.channels_of(fast, ob)
                base_mean, _, feat = model.base_moments(Z_hat, fast, ob)
                fast_n, _ = model.beliefs.update(fast, Zs[t], st.act, ob, model.outcome_head)
                if kal:
                    fast_n = model.kalman_update(fast_n, feat, ob, base_mean)
                fast = fast_n
            rspec = [s for s in trackers[0].prospective() if s.key == "reward"][0]
            from ..outcome_spec import build_outcome_batch, lookup_channels
            for k in ks:
                fk = snaps[k]
                acts, rows, tl = [], [], []
                for i, ep in enumerate(eps):
                    for a in ep.queries[k]["actions"]:
                        acts.append(a)
                        rows.append(i)
                        tl.append(ep.tools)
                idx = torch.tensor(rows, dtype=torch.long, device=dev)
                Z = Zs[k][idx]
                f = fk.select(idx)
                A = model.tokenizer.action(acts, tool_lists=tl)
                dt = model.expected_dt(Z, A, acts, tl)
                Bt = model.beliefs.tokens(f)
                Z_hat, _ = model.world_model.prior(Z, A, Bt, dt, sample=False)
                specs = [[rspec] for _ in rows]
                chans = [lookup_channels(f, r, specs[r]) for r in range(len(rows))]
                ob = build_outcome_batch(specs, model.query_encoder, [EntityRegistry(4, seed=r) for r in range(len(rows))],
                                         chans, dev)
                mean, var, _ = model.predict_outcomes(Z_hat, f, ob)
                mean, var = mean[:, 0].float().cpu().numpy(), var[:, 0].float().cpu().numpy()
                j = 0
                for i, ep in enumerate(eps):
                    Q = len(ep.queries[k]["actions"])
                    out[(c0 + i, k)] = (mean[j:j + Q], var[j:j + Q])
                    j += Q
    return out
