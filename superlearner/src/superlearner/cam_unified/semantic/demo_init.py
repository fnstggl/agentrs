"""Declared demo-init condition (architecture decision 2; spec Parts 2.3, 9.2, 12; `cfg.declared_conditions` must
contain "demo-init:example_inductor"): behaviour cloning of the integrated CamUnifiedModel on transcripts of Crucible's
**non-privileged** ExampleInductor.

Pipeline (every stage uses the real components; nothing here is a test double):

1. `record_demo_dataset(tasks, out_dir)` runs the demonstrator through workstream G's recorder (`envs.demos.
   record_demos`, which refuses privileged controls and held-out families/seeds) inside the real OpenWorld, wrapped in
   `ToolCapture`, which records the tool list the demonstrator was offered at each of its decisions. Each recorded
   decision becomes a `DemoDecision` (the observations the integrated agent would receive at that point - the
   previous action's tool result merged with the new world events exactly as `runtime.run_steps` merges them - the
   tool list the agent would get from the adapter, the public goal, the demonstrator's action and its real outcome).
2. `teacher_forced_rollout(model, episodes)` replays the transcripts through the model's own acting computation
   (`teacher_forced_decide`, the same context assembly as `CamUnifiedModel.decide`: tokenised observations + entity
   tokens, previous ACTION token, TIME, belief and option tokens, retrieved memory, goal and budget; reasoning; then
   the world-model prior, outcome predictions and `observe_outcomes` belief / Kalman updates with the real results) with
   the demonstrated action in place of a sampled one, and keeps the per-decision `StepInputs`.
3. `demo_log_prob(model, inputs, actions, evidence)` recomputes the decision with gradients (`encode_step` in acting
   mode, as `policy_evaluate` does) and returns log pi(a_demo | context) split into its tool / evidence / typed-head /
   string-token parts; `BehaviourCloning` minimises -log pi over minibatches, so the decoder path (tool pointer,
   evidence pointer, typed heads, bridge, LoRA, length head) and the latent core (tokenizer projections, entity
   encoder, backbone, reasoner, belief/option/budget tokens) are trained jointly. The latent option is not a demo
   label: its log-probability is excluded from the objective.
4. `evaluate_demo_match` (greedy decoding on held-out-seed transcripts: tool accuracy, exact-argument rate, demo
   log-probability) and `act_on_tasks` (the BC'd model acting through `CamUnifiedAgent` + `runtime.run_steps` on real
   OpenWorld seeds it was not trained on: schema-valid and executed-ok rates) measure the result.

Evidence labels. A demonstration has no evidence pointer decisions; `evidence_labels` derives them from observable
data only: the most recent observations whose rendered text contains an argument value of the demonstrated action
(ids, handles, numbers with >= 3 characters). They supervise CAM's evidence pointer (spec 2.3 "retrieving evidence
needed for a decision") and show the string decoder where exact identifiers come from (numbers equal to an evidence
number are encoded as copies).

Never used: privileged transcripts (refused upstream and re-checked here), evaluator-only state, goal details or
validation/test families/seeds (`worldgen.SplitGuard` inside the recorder; `check_condition` refuses a config without
the declared condition).
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

from ..interfaces import (Action, ActionSource, ExecutionResult, GoalSpec, Observation, ObsKind, ToolSchema)
from .tokenizer import parse_obs_ref, render_observation

CONDITION = "demo-init:example_inductor"
DEMONSTRATOR = "example_inductor"


class UndeclaredCondition(RuntimeError):
    """Behaviour cloning was requested without the declared demo-init condition in cfg.declared_conditions."""


def check_condition(cfg) -> None:
    if CONDITION not in tuple(getattr(cfg, "declared_conditions", ()) or ()):
        raise UndeclaredCondition(f"behaviour cloning needs {CONDITION!r} in cfg.declared_conditions "
                                  f"(got {tuple(getattr(cfg, 'declared_conditions', ()) or ())})")


# ============================================================================ recording
class ToolCapture:
    """Wraps a non-privileged Crucible `Policy` and records, per call of `act`, the simulated time and the tool specs
    the policy was offered (the demonstrator's view of the action space). Everything else is delegated unchanged."""

    def __init__(self, inner):
        from ..envs.agent_protocol import is_privileged
        if is_privileged(inner):
            raise RuntimeError(f"{type(inner).__name__} is a privileged control; its transcripts are never training "
                               f"data")
        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.calls: list = []                    # [(time_s, specs)]

    def __getattr__(self, k):                    # reset / set_goal / observe_result / usage / ...
        return getattr(self.__dict__["inner"], k)

    def act(self, d: dict, specs: list):
        self.calls.append((int(d.get("time", 0)), [dict(s) for s in specs]))
        return self.inner.act(d, specs)


def _tools_digest(tools: Sequence[ToolSchema]) -> str:
    return hashlib.sha256(json.dumps([t.to_dict() for t in tools], sort_keys=True, default=str).encode()
                          ).hexdigest()[:20]


@dataclass
class DemoDecision:
    """One demonstrated decision with exactly the inputs the integrated agent would have at that point."""
    step: int
    t: float                                   # simulated days
    observations: list                         # [Observation] (previous tool result + new world events)
    tools_id: str                              # key into DemoEpisode.tool_table
    action: Action                             # the demonstrator's action (source SCRIPTED, no decode record)
    result: Optional[ExecutionResult]
    reward: float = 0.0                        # verified return increment the harness delivered
    next_t: Optional[float] = None


@dataclass
class DemoEpisode:
    key: str                                   # "<family>:<seed>" (bookkeeping only: never a model input)
    family: str
    seed: int
    goal: GoalSpec                             # agent-visible goal (agent.public_goal of the adapter's GoalSpec)
    decisions: list
    tool_table: dict = field(default_factory=dict)      # tools_id -> [ToolSchema]
    success: Optional[bool] = None
    verified_return: float = 0.0

    def tools(self, d: DemoDecision) -> list:
        return self.tool_table[d.tools_id]

    def to_json(self) -> str:
        return json.dumps({"key": self.key, "family": self.family, "seed": self.seed, "goal": self.goal.to_dict(),
                           "success": self.success, "verified_return": self.verified_return,
                           "tool_table": {k: [t.to_dict() for t in v] for k, v in self.tool_table.items()},
                           "decisions": [{"step": d.step, "t": d.t, "observations": [o.to_dict() for o in
                                                                                     d.observations],
                                          "tools_id": d.tools_id, "action": d.action.to_dict(),
                                          "result": None if d.result is None else d.result.to_dict(),
                                          "reward": d.reward, "next_t": d.next_t} for d in self.decisions]},
                          default=str)

    @staticmethod
    def from_json(s: str) -> "DemoEpisode":
        j = json.loads(s)
        g = j["goal"]
        goal = GoalSpec(**{k: g[k] for k in ("goal_id", "text", "metrics", "deadline_days", "family", "meta",
                                              "decision_budget", "funds_budget") if k in g})
        decs = [DemoDecision(d["step"], d["t"], [Observation.from_dict(o) for o in d["observations"]], d["tools_id"],
                             Action.from_dict(d["action"]),
                             None if d["result"] is None else ExecutionResult.from_dict(d["result"]), d["reward"],
                             d.get("next_t")) for d in j["decisions"]]
        tt = {k: [ToolSchema.from_dict(t) for t in v] for k, v in j["tool_table"].items()}
        return DemoEpisode(j["key"], j["family"], int(j["seed"]), goal, decs, tt, j.get("success"),
                           float(j.get("verified_return", 0.0)))


def save_episodes(episodes: Sequence[DemoEpisode], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for e in episodes:
            fh.write(e.to_json() + "\n")
    os.replace(tmp, path)


def load_episodes(path: str) -> list:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                e = DemoEpisode.from_json(line)
                for d in e.decisions:
                    if d.action.source is not ActionSource.SCRIPTED:
                        raise RuntimeError(f"{e.key} step {d.step}: not a scripted demonstration action")
                out.append(e)
    return out


def record_demo_dataset(tasks: Sequence[Any], out_dir: Optional[str] = None, *, days: Optional[float] = None,
                        think_s: float = 600.0, seed: int = 0, policy_factory: Optional[Callable[[], Any]] = None
                        ) -> list:
    """Record the non-privileged demonstrator on `tasks` (GoalContract-like OpenWorld tasks of training families /
    seeds) and convert every decision into a DemoDecision. Returns [DemoEpisode] (one per task; also written to
    out_dir/episodes.jsonl together with the raw transcript out_dir/transcript.jsonl when out_dir is given)."""
    from ..agent import public_goal as agent_public_goal
    from ..envs import ow_convert as C
    from ..envs.demos import record_demos
    from ..envs.openworld import open_world, public_goal, wait_tool
    from ..runtime import FROM_START, _max_seq, _merge

    if policy_factory is None:
        from crucible.usefulness.agents import ExampleInductor
        policy_factory = ExampleInductor
    made: list = []

    def factory():
        w = ToolCapture(policy_factory())
        made.append(w)
        return w

    per_env: list = []
    t0 = time.time()
    out = record_demos(list(tasks), os.path.join(out_dir, "transcript.jsonl") if out_dir else None,
                       policy_factory=factory, think_s=think_s, sink=per_env.append, days=days, seed=seed)
    if out["demonstrator"] != DEMONSTRATOR:
        raise RuntimeError(f"unexpected demonstrator {out['demonstrator']!r}")
    wrappers = made[-len(tasks):]                 # made[0] is record_demos' privilege probe
    max_wait_s = inspect.signature(open_world).parameters["max_wait_s"].default
    episodes = []
    by_env: dict = {}
    for recs in per_env:
        by_env.setdefault(recs[0].env_id, []).extend(recs)
    env_ids = list(by_env)
    if len(env_ids) != len(tasks):
        raise RuntimeError(f"{len(env_ids)} recorded environments for {len(tasks)} tasks")
    for b, task in enumerate(tasks):
        recs = [r for r in by_env[env_ids[b]] if r.action is not None]
        calls = wrappers[b].calls
        if len(calls) != len(recs):
            raise RuntimeError(f"task {b}: {len(calls)} demonstrator calls for {len(recs)} acted records")
        horizon = int(getattr(task, "horizon_s", 30 * 86400))
        goal = agent_public_goal(public_goal(task, horizon))
        table: dict = {}
        decs: list = []
        seen = FROM_START
        prev_res: list = []
        for k, (r, (t_s, specs)) in enumerate(zip(recs, calls)):
            if abs(t_s - r.t_sim * 86400.0) > 1.0:
                raise RuntimeError(f"task {b} step {k}: demonstrator call at {t_s}s, record at {r.t_sim * 86400}s")
            tools = [C.tool_schema(s) for s in specs] + [wait_tool(max_wait_s)]
            tid = _tools_digest(tools)
            table.setdefault(tid, tools)
            obs, seen = _merge(seen, prev_res, r.observations)
            a = Action(tool=r.action.tool, args=copy.deepcopy(r.action.args), purpose=r.action.purpose,
                       code=r.action.code, source=ActionSource.SCRIPTED)
            decs.append(DemoDecision(k, float(r.t_sim), obs, tid, a, r.result, float(r.reward),
                                     float(r.t_sim + float(r.extra.get("dt_days", 0.0)))))
            prev_res = list(r.result.observations) if r.result is not None else []
        w = task.world
        episodes.append(DemoEpisode(f"{w['family']}:{int(w['seed'])}", w["family"], int(w["seed"]), goal, decs,
                                    table, bool(out["successes"][b]), float(out["returns"][b])))
    if out_dir:
        save_episodes(episodes, os.path.join(out_dir, "episodes.jsonl"))
        with open(os.path.join(out_dir, "record_summary.json"), "w") as fh:
            json.dump({"tasks": [e.key for e in episodes], "successes": out["successes"], "returns": out["returns"],
                       "success_day": out["success_day"], "decisions": [len(e.decisions) for e in episodes],
                       "demonstrator": out["demonstrator"], "wall_s": time.time() - t0}, fh, indent=1)
    return episodes


# ============================================================================ evidence labels
_VAL_MIN_CHARS = 3


def _arg_values(v, out: list) -> None:
    if isinstance(v, dict):
        for x in v.values():
            _arg_values(x, out)
    elif isinstance(v, list):
        for x in v:
            _arg_values(x, out)
    elif isinstance(v, bool) or v is None:
        return
    elif isinstance(v, (int, float)):
        s = repr(v) if isinstance(v, float) else str(v)
        if len(s.lstrip("-")) >= _VAL_MIN_CHARS:
            out.append(s)
    elif isinstance(v, str) and _VAL_MIN_CHARS <= len(v) <= 200 and "\n" not in v:
        out.append(v)


def evidence_labels(observations: Sequence[Observation], action: Action, max_evidence: int,
                    candidates: Optional[Sequence[int]] = None, chars: int = 1500) -> list:
    """Observation indices (newest first, <= max_evidence) whose rendered text contains an argument value of
    `action` as a whole token (ids, handles, names, numbers with >= 3 characters). `candidates` restricts the result
    to observations that have tokens (the evidence pointer's candidates)."""
    vals = []
    _arg_values(action.args or {}, vals)
    if not vals:
        return []
    cand = set(range(len(observations))) if candidates is None else set(candidates)
    pats = [re.compile(r"(?<![\w.-])" + re.escape(v) + r"(?![\w-])") for v in dict.fromkeys(vals)]
    found, covered = [], set()
    for i in sorted(cand, key=lambda i: (observations[i].t, i), reverse=True):
        text = render_observation(observations[i], chars)
        if observations[i].value is not None:
            text += " " + repr(observations[i].value)
        hit = {j for j, p in enumerate(pats) if j not in covered and p.search(text)}
        if hit:
            found.append(i)
            covered |= hit
        if len(found) >= max_evidence or len(covered) == len(pats):
            break
    return sorted(found)


# ============================================================================ teacher forcing
@dataclass
class ForcedStep:
    """One demonstrated decision prepared for the policy objective (StepInputs row of the acting computation)."""
    inputs: Any                                 # model.StepInputs (B = 1)
    action: Action
    evidence: Optional[list]                    # evidence labels (observation indices) or None
    episode: str
    step: int
    is_wait: bool
    unrepresentable: Optional[str] = None       # why the action cannot be written by the generator (excluded)


def _budget_rows(goals, t0, now, funds, funds0):
    rows = []
    for g, a, n, f, f0 in zip(goals, t0, now, funds, funds0):
        dl = None if g is None or g.deadline_days is None else float(g.deadline_days)
        tot = sum(f.values()) if f else None
        rows.append({"days_left": None if dl is None else dl - (n - a), "days_total": dl, "funds_left": tot,
                     "funds_total": f0, "decisions_left": None, "decisions_total": None})
    return rows


@torch.no_grad()
def teacher_forced_decide(model, state, obs_lists: list, now: list, tools: list, goals: list, actions: list,
                          budget=None, greedy: bool = False, generator: Optional[torch.Generator] = None):
    """`CamUnifiedModel.decide` with the actions given (teacher forcing): the identical context assembly, latent update,
    reasoning, option step, ACTION tokens of the given actions, expected duration, world-model prior, outcome
    predictions and value, in acting (eval) mode. Mutates `state` exactly like decide (Z, option, last actions /
    time, entity registries). Returns the `model.Decision` (logp 0: the behaviour policy is the demonstrator)."""
    from ..model import (Decision, StepInputs, acting_mode, belief_summary, check_shape, check_tokens, detach_tokens,
                         gen_on)
    cfg = model.cfg
    B, D = state.B, cfg.d_model
    dev = state.Z.device
    with acting_mode(model):
        regs_snapshot = copy.deepcopy(state.registries)
        now_t = torch.tensor([float(x) for x in now], device=dev)
        dt_prev = torch.where(state.last_now >= 0, (now_t - state.last_now).clamp(min=0), torch.full_like(now_t, -1))
        with model._autocast():
            G, Bt = model._goal_and_belief(goals, state.fast, B)
            M, ids, scores = (None, [[] for _ in range(B)], None)
            if cfg.use_memory:
                M, ids, scores = model.retrieve(state.Z, G, Bt, None, list(now))
            inp = StepInputs(state.Z.detach(), None if state.fast is None else state.fast.clone(), regs_snapshot,
                             obs_lists, list(now), list(state.last_actions), list(goals), list(tools),
                             detach_tokens(M), [list(i) for i in ids], dt_prev, budget, state.option.clone(),
                             state.option_steps.clone(), state.option.clone(),
                             torch.zeros(B, dtype=torch.bool, device=dev), state.option_steps.clone())
            inp.registries = state.registries
            enc = model.encode_step(inp, pre=(G, Bt))
            inp.registries = regs_snapshot
            if model.option_policy is not None:
                ost = model.option_policy.step(enc.R, enc.goal, state.option, state.option_steps, greedy,
                                               gen_on(generator, dev))
                inp.option, inp.option_new, inp.option_steps = ost.option.long(), ost.new.bool(), ost.steps.long()
                state.option, state.option_steps = inp.option.clone(), inp.option_steps.clone()
            model.tokenizer.register_tools([t for lst in tools for t in lst])
            A_tok = check_tokens(model.tokenizer.action(list(actions), tool_lists=list(tools)), B, D,
                                 "tokenizer.action(demo)")
            dt = model.expected_dt(enc.Z, A_tok, actions, tools)
            Z_hat, _ = model.world_model.prior(enc.Z, A_tok, enc.belief, dt, sample=False)
            check_shape(Z_hat, (B, cfg.n_state, D), "world_model.prior Z_hat")
            specs, prosp = model.prospective_batch(state)
            mean, var, _ = model.predict_outcomes(Z_hat, state.fast, prosp)
            value = model._value(model.value_head(enc.R, enc.goal))
        state.Z = enc.Z.detach().float()
        state.last_actions = list(actions)
        state.last_now = now_t
        predicted = [{s.key: float(mean[b, q]) for q, s in enumerate(specs[b])} for b in range(B)]
        uncertainty = [{s.key: float(var[b, q]) for q, s in enumerate(specs[b])} for b in range(B)]
        iters = enc.rinfo.get("iters")
        reasoning = [{"iters": float(iters[b]) if iters is not None else 0.0, "halt_p": []} for b in range(B)]
        retrieved = [[{"id": str(i), "score": float(scores[b][j]) if scores is not None else 0.0}
                      for j, i in enumerate(ids[b])] for b in range(B)]
        behavior = [{"policy": "demonstration", "logp_kind": "none", "selection": None, "n_candidates": 1}
                    for _ in range(B)]
        return Decision(list(actions), inp, enc.Z.detach().float(), enc.R.detach().float(), Z_hat.detach().float(),
                        detach_tokens(A_tok), detach_tokens(enc.ctx), specs, mean, var, predicted, uncertainty,
                        value, torch.zeros(B, device=dev), dt, reasoning, retrieved, [[] for _ in range(B)],
                        [[] for _ in range(B)],
                        belief_summary(None if state.fast is None else state.fast.logw) or [{} for _ in range(B)],
                        behavior, [{"ctx_tokens": dict(d)} for d in enc.dropped])


def _with_builtins(model, tools: list) -> list:
    """Make sure the generator's built-in tools are registered with the tokenizer (their ACTION tokens)."""
    gen = model.action_generator
    if getattr(gen, "builtins", None):
        model.tokenizer.register_tools(gen.builtins)
    return tools


@torch.no_grad()
def teacher_forced_rollout(model, episodes: Sequence[DemoEpisode], *, batch: int = 8, seed: int = 0,
                           max_evidence: Optional[int] = None, keep: Optional[Callable] = None) -> list:
    """Replay demonstrated episodes through the model's acting computation (B episodes in lockstep). Returns
    [ForcedStep] in episode/step order (`keep(episode, decision) -> bool` filters what is kept; all decisions are
    replayed so the latent state, beliefs and registries evolve exactly as when acting)."""
    from ..agent import default_funds
    from ..model import BudgetState
    out: list = []
    max_ev = int(max_evidence if max_evidence is not None else model.cfg.max_evidence)
    for c0 in range(0, len(episodes), batch):
        eps = list(episodes[c0: c0 + batch])
        state = model.init_state(len(eps), seed=seed + c0)
        t0 = [None] * len(eps)
        funds = [dict() for _ in eps]
        funds0 = [None] * len(eps)
        T = max(len(e.decisions) for e in eps)
        for k in range(T):
            rows = [b for b, e in enumerate(eps) if k < len(e.decisions)]
            sub = state.select(rows)
            decs = [eps[b].decisions[k] for b in rows]
            obs_lists = [list(d.observations) for d in decs]
            now = [float(d.t) for d in decs]
            tools = _with_builtins(model, [eps[b].tools(d) for b, d in zip(rows, decs)])
            goals = [eps[b].goal for b in rows]
            for j, b in enumerate(rows):
                if t0[b] is None:
                    t0[b] = now[j]
                f = default_funds(obs_lists[j], model.money_unit)
                if f:
                    funds[b].update(f)
                    if funds0[b] is None:
                        funds0[b] = sum(funds[b].values())
            budget = BudgetState.from_rows(_budget_rows(goals, [t0[b] for b in rows], now, [funds[b] for b in rows],
                                                        [funds0[b] for b in rows]), device=model.device)
            acts = [d.action for d in decs]
            dec = teacher_forced_decide(model, sub, obs_lists, now, tools, goals, acts, budget)
            for j, b in enumerate(rows):
                d = decs[j]
                if keep is None or keep(eps[b], d):
                    out.append(ForcedStep(dec.inputs.select([j]), d.action, None, eps[b].key, d.step,
                                          d.action.tool == "wait"))
            results = [d.result if d.result is not None else ExecutionResult(ok=False, error="missing") for d in decs]
            nxt = [list(eps[b].decisions[k + 1].observations) if k + 1 < len(eps[b].decisions) else []
                   for b in rows]
            model.observe_outcomes(sub, dec, results, nxt, [float(d.reward) for d in decs])
            state.scatter(rows, sub)
    gen = model.action_generator
    for st in out:
        st.unrepresentable = demo_unrepresentable(gen, st.inputs.tools[0], st.action)
    _attach_evidence(model, out, max_ev)
    return out


def demo_unrepresentable(gen, tools: Sequence[ToolSchema], action: Action) -> Optional[str]:
    """None when the generator can write `action` exactly (tool available and generable, arguments inside the
    compiled grammar's capability reductions, every string within its token cap); otherwise the reason. Such
    demonstrations are excluded from behaviour cloning and counted (never silently repaired)."""
    from .grammar import GrammarError, drive_walks, encode_value
    tool = next((t for t in list(tools) + list(getattr(gen, "builtins", [])) if t.name == action.tool), None)
    if tool is None:
        return "tool not in the decision's tool list"
    g = gen.grammar(tool)
    if not g.generable:
        return f"tool not generable: {g.error}"
    choices, holes, flags = [], [], {}
    try:
        encode_value(g.root, action.args or {}, gen.opts, choices, holes, flags, None)
        it = iter(choices)
        tpl = drive_walks([g.root], gen.opts, lambda rows, decs: [next(it) for _ in decs])[0]
        prog = gen.filler.program(tpl)
        if len(prog.holes) != len(holes):
            return "string holes do not match the arguments"
        for k, (h, (kind, val)) in enumerate(zip(prog.holes, holes)):
            gen.filler.tokenize_hole(kind, val, h, prog.cap[k])
    except (GrammarError, StopIteration) as e:
        return f"{type(e).__name__}: {str(e)[:160]}"
    return None


def _attach_evidence(model, steps: list, max_ev: int) -> None:
    """Evidence labels restricted to the observations that receive tokens under the tokenizer's budget (the
    evidence pointer's candidates), for tools whose arguments use evidence."""
    tok = model.tokenizer
    gen = model.action_generator
    for st in steps:
        inp = st.inputs
        tool = next((t for t in list(inp.tools[0]) + list(getattr(gen, "builtins", [])) if t.name == st.action.tool),
                    None)
        if tool is None or not gen.grammar(tool).uses_evidence:
            st.evidence = None
            continue
        obs = inp.obs_lists[0]
        order = sorted(range(len(obs)), key=lambda i: (obs[i].t, i), reverse=True)
        budget, kept = tok.max_obs_tokens, []
        for i in order:                         # the tokenizer's recency-suffix budget (no tokens computed here)
            n = tok._n_tokens(obs[i])
            if n > budget:
                break
            kept.append(i)
            budget -= n
        st.evidence = evidence_labels(obs, st.action, max_ev, candidates=kept) if kept else None


# ============================================================================ the objective
def demo_log_prob(model, inputs, actions: list, evidence: Optional[list] = None, exact_rows: bool = False) -> tuple:
    """log pi(a_demo | context) of stored decisions, recomputed with gradients through the acting computation (as
    `CamUnifiedModel.policy_evaluate`, without the latent option term). Returns ({tool, evidence, heads, tokens}: [B]
    each, entropy parts)."""
    from ..model import StepInputs, acting_mode
    regs = copy.deepcopy(inputs.registries)
    live = StepInputs(**{**inputs.__dict__, "registries": regs})
    cfg = model.cfg
    with model._autocast():
        mem = None
        if cfg.use_memory and any(inputs.memory_ids):
            mem = model.memory_tokens(inputs.memory_ids, inputs.now)
        with acting_mode(model):
            enc = model.encode_step(live, memory=mem)
            opt_tok = model._option_token(inputs.option, inputs.option_steps) if model.option_policy is not None \
                else None
            goal_in = model._goal_inputs(enc, opt_tok, inputs.budget)
            T = model.tokenizer.tools(inputs.tools)
            return model.action_generator.log_prob_parts(enc.R, goal_in, enc.belief, T, enc.obs, inputs.tools,
                                                         inputs.obs_lists, actions, evidence=evidence,
                                                         exact_rows=exact_rows)


def _stack_inputs(steps: list):
    from ..model import StepInputs
    return StepInputs.cat([s.inputs for s in steps])


def _evidence_arg(steps: list) -> Optional[list]:
    if all(s.evidence is None for s in steps):
        return None
    return [s.evidence for s in steps]


@dataclass
class BCConfig:
    epochs: int = 3
    batch: int = 8
    lr: float = 3e-4                       # core / heads (cfg.lr)
    lr_sem: float = 1e-4                   # LoRA + bridge + length head (cfg.lr_sem)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    wait_share: float = 0.5                # at most this share of the training decisions are idle waits (subsampled)
    seed: int = 0


def semantic_param_ids(model) -> set:
    gen = model.action_generator
    ids = {id(p) for p in gen.lora}
    ids |= {id(p) for p in gen.bridge.parameters()}
    if getattr(gen, "length_head", None) is not None:
        ids |= {id(p) for p in gen.length_head.parameters()}
    return ids


def make_optimizer(model, bc: BCConfig) -> torch.optim.Optimizer:
    sem = semantic_param_ids(model)
    ps = [p for p in model.parameters() if p.requires_grad]
    lora = {id(p) for p in model.action_generator.lora}
    for p in model.action_generator.lora:
        if all(p is not q for q in ps):
            ps.append(p)
    core = [p for p in ps if id(p) not in sem]
    semp = [p for p in ps if id(p) in sem]
    assert lora <= {id(p) for p in semp}
    return torch.optim.AdamW([{"params": core, "lr": bc.lr, "name": "core"},
                              {"params": semp, "lr": bc.lr_sem, "name": "semantic"}], weight_decay=bc.weight_decay)


def balance_waits(steps: list, wait_share: float, rng: random.Random) -> list:
    acts = [s for s in steps if not s.is_wait]
    waits = [s for s in steps if s.is_wait]
    n_w = len(waits) if wait_share >= 1 else min(len(waits), int(round(len(acts) * wait_share / (1 - wait_share))))
    return acts + rng.sample(waits, n_w)


class BehaviourCloning:
    """-log pi(a_demo | context) over the integrated model (declared demo-init condition only)."""

    def __init__(self, model, bc: Optional[BCConfig] = None):
        check_condition(model.cfg)
        self.model = model
        self.bc = bc or BCConfig()
        self.opt = make_optimizer(model, self.bc)
        self.rng = random.Random(self.bc.seed)
        self.updates = 0
        self.log: list = []

    def params(self) -> list:
        return [p for g in self.opt.param_groups for p in g["params"]]

    def step(self, steps: list) -> dict:
        m = self.model
        m.train()
        inp = _stack_inputs(steps)
        parts, _ = demo_log_prob(m, inp, [s.action for s in steps], _evidence_arg(steps))
        lp = sum(parts.values())
        loss = -lp.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.params(), self.bc.grad_clip)
        core_g = sum(float(p.grad.norm()) ** 2 for p in m.core.parameters() if p.grad is not None) ** 0.5
        lora_g = sum(float(p.grad.norm()) ** 2 for p in m.action_generator.lora if p.grad is not None) ** 0.5
        self.opt.step()
        self.updates += 1
        sem = getattr(m.action_generator, "enc", None)
        if sem is not None and hasattr(sem, "mark_updated"):
            sem.mark_updated()                   # LoRA changed: decoder version (and merged acting weights)
        m.bump_weights_epoch()
        rec = {"update": self.updates, "loss": float(loss.detach()), "grad_norm": float(gn), "core_grad": core_g,
               "lora_grad": lora_g, "n": len(steps), **{f"lp_{k}": float(v.detach().mean()) for k, v in
                                                        parts.items()}}
        self.log.append(rec)
        return rec

    def epoch(self, steps: list, on_batch: Optional[Callable] = None, deadline: Optional[float] = None) -> dict:
        data = balance_waits(representable(steps), self.bc.wait_share, self.rng)
        self.rng.shuffle(data)
        # batches of similar cost: waits never touch the language model
        data.sort(key=lambda s: s.is_wait)
        batches = [data[i: i + self.bc.batch] for i in range(0, len(data), self.bc.batch)]
        self.rng.shuffle(batches)
        losses = []
        for i, b in enumerate(batches):
            if deadline is not None and time.time() > deadline:
                break
            r = self.step(b)
            losses.append(r["loss"])
            if on_batch is not None:
                on_batch(i, r)
        return {"batches": len(losses), "of": len(batches), "loss_mean": sum(losses) / max(1, len(losses)),
                "decisions": sum(len(b) for b in batches[: len(losses)])}


# ============================================================================ evaluation
def representable(steps: list) -> list:
    return [s for s in steps if s.unrepresentable is None]


@torch.no_grad()
def demo_logp_table(model, steps: list, batch: int = 8) -> dict:
    """Mean demo log-probability (generator parts) over the representable steps, split into waits and other
    actions."""
    model.eval()
    steps = representable(steps)
    vals = {"wait": [], "act": []}
    for i in range(0, len(steps), batch):
        ch = steps[i: i + batch]
        parts, _ = demo_log_prob(model, _stack_inputs(ch), [s.action for s in ch], _evidence_arg(ch))
        lp = sum(parts.values())
        for s, v in zip(ch, lp.tolist()):
            vals["wait" if s.is_wait else "act"].append(v)
    mean = lambda v: (sum(v) / len(v)) if v else None
    return {"logp_wait": mean(vals["wait"]), "logp_act": mean(vals["act"]),
            "logp_all": mean(vals["wait"] + vals["act"]), "n_wait": len(vals["wait"]), "n_act": len(vals["act"])}


@torch.no_grad()
def greedy_actions(model, steps: list, batch: int = 8) -> list:
    """The model's greedy decoder action at each stored decision (inputs from a teacher-forced rollout)."""
    from ..model import StepInputs, acting_mode
    model.eval()
    out = []
    for i in range(0, len(steps), batch):
        ch = steps[i: i + batch]
        inp = _stack_inputs(ch)
        live = StepInputs(**{**inp.__dict__, "registries": copy.deepcopy(inp.registries)})
        with model._autocast(), acting_mode(model):
            enc = model.encode_step(live)
            opt_tok = model._option_token(inp.option, inp.option_steps) if model.option_policy is not None else None
            goal_in = model._goal_inputs(enc, opt_tok, inp.budget)
            T = model.tokenizer.tools(inp.tools)
            out += model.action_generator.sample(enc.R, goal_in, enc.belief, T, enc.obs, inp.tools, inp.obs_lists,
                                                 greedy=True, rescore=False)
    return out


def _same_args(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@torch.no_grad()
def evaluate_demo_match(model, steps: list, batch: int = 8) -> dict:
    """Greedy decoding at the demonstrated decisions (inputs from the teacher-forced rollout): tool accuracy and
    exact-argument rate, overall and on non-wait decisions."""
    from crucible.core.schema import validate
    res = {"n": 0, "tool_ok": 0, "args_ok": 0, "n_act": 0, "tool_ok_act": 0, "args_ok_act": 0, "schema_valid": 0,
           "excluded_unrepresentable": len(steps) - len(representable(steps))}
    steps = representable(steps)
    tools_seen: dict = {}
    acts = greedy_actions(model, steps, batch)
    for s, a in zip(steps, acts):
        tl = s.inputs.tools[0]
        res["n"] += 1
        ok_t = a.tool == s.action.tool
        ok_a = ok_t and _same_args(a.args, s.action.args)
        schema = next((t.parameters for t in list(tl) + list(model.action_generator.builtins) if t.name == a.tool),
                      None)
        res["schema_valid"] += int(schema is not None and not validate(schema, a.args))
        res["tool_ok"] += ok_t
        res["args_ok"] += ok_a
        if not s.is_wait:
            res["n_act"] += 1
            res["tool_ok_act"] += ok_t
            res["args_ok_act"] += ok_a
        tools_seen[a.tool] = tools_seen.get(a.tool, 0) + 1
    n, na = max(1, res["n"]), max(1, res["n_act"])
    res.update(tool_acc=res["tool_ok"] / n, exact_args_rate=res["args_ok"] / n, tool_acc_act=res["tool_ok_act"] / na,
               exact_args_rate_act=res["args_ok_act"] / na, schema_valid_rate=res["schema_valid"] / n,
               greedy_tools=dict(sorted(tools_seen.items(), key=lambda kv: -kv[1])[:20]))
    return res


def act_on_tasks(model, tasks: Sequence[Any], n_decisions: int, *, seed: int = 0, greedy: bool = False) -> dict:
    """The model acting through the integrated agent on real OpenWorld tasks (runtime.run_steps over
    OpenWorldAdapter, verified rewards from OpenWorldEvaluator): schema-valid and executed-ok rates."""
    from crucible.core.schema import validate
    from ..agent import CamUnifiedAgent
    from ..envs.openworld import open_world
    from ..envs.openworld_eval import OpenWorldEvaluator
    from ..runtime import run_steps
    agent = CamUnifiedAgent(model, seed=seed, greedy=greedy)
    ad, op = open_world(list(tasks), seed=seed)
    ev = OpenWorldEvaluator(op)
    checks = {"n": 0, "schema_valid": 0, "ok": 0, "non_wait": 0, "non_wait_ok": 0, "tools": {}, "errors": {}}
    try:
        tool_lists: list = []
        orig_act = agent.act

        def act(ob, tools):                   # observe what the agent saw and did (no change to the decision)
            acts = orig_act(ob, tools)
            tool_lists.append((tools, acts))
            return acts
        agent.act = act
        rep = run_steps(agent, ad, n_decisions, evaluator=ev)
    finally:
        ad.close()
    gen = model.action_generator
    for (tools, acts), row in zip(tool_lists, rep.steps):
        for b, a in enumerate(acts):
            if a is None:
                continue
            checks["n"] += 1
            schema = next((t.parameters for t in list(tools[b]) + list(gen.builtins) if t.name == a.tool), None)
            checks["schema_valid"] += int(schema is not None and not validate(schema, a.args))
            ok = bool(row["ok"][b])
            checks["ok"] += ok
            checks["tools"][a.tool] = checks["tools"].get(a.tool, 0) + 1
            if a.tool != "wait":
                checks["non_wait"] += 1
                checks["non_wait_ok"] += ok
    n = max(1, checks["n"])
    checks.update(schema_valid_rate=checks["schema_valid"] / n, executed_ok_rate=checks["ok"] / n,
                  non_wait_ok_rate=checks["non_wait_ok"] / max(1, checks["non_wait"]),
                  returns=rep.returns, compute=rep.compute.to_dict() if hasattr(rep.compute, "to_dict") else None,
                  tools=dict(sorted(checks["tools"].items(), key=lambda kv: -kv[1])[:25]))
    return checks


# ============================================================================ the declared run (resumable CLI)
TRAIN_TASKS = (("ow_tiny", 100_000, 16), ("ow_market", 100_000, 4))     # (family, first seed, n): dev offsets
HELD_TASKS = (("ow_tiny", 100_500, 4),)                                  # held-out transcripts (same split, unseen)
ACT_TASKS = (("ow_tiny", 100_600, 2),)                                   # acting evaluation worlds (unseen)


def make_tasks(spec) -> list:
    from crucible.research import llm_eval as L
    out = []
    for fam, s0, n in spec:
        for i in range(n):
            out.append(L.open_economy_goal(world_family=fam, seed=s0 + i,
                                           overrides={"n_consumers": 800} if fam == "ow_tiny" else {}, days=8))
    return out


def run_config(scale: str = "dev", **over):
    """The demo-init run's config: the declared condition; imagination off while replaying and acting (the decoder
    policy is what BC trains; planner decisions would score many decoder samples per decision)."""
    from ..config import make_config
    return make_config(scale, use_imagination=False, declared_conditions=(CONDITION,), **over)


def _build(cfg, seed: int, test_model: bool = False):
    from ..agent import CamUnifiedAgent
    from ..model import build_model
    torch.manual_seed(seed)
    kw = {"semantic": {"test": True}} if test_model else None
    model = build_model(cfg, component_kwargs=kw).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, CamUnifiedAgent(model, seed=seed)


def _json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _wait_subset(steps: list, n: int, seed: int) -> list:
    acts = [s for s in steps if not s.is_wait]
    waits = [s for s in steps if s.is_wait]
    return acts + random.Random(seed).sample(waits, min(n, len(waits)))


def cmd_record(out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    res = {}
    for name, spec in (("train", TRAIN_TASKS), ("held", HELD_TASKS)):
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        eps = record_demo_dataset(make_tasks(spec), d)
        res[name] = {"episodes": [e.key for e in eps], "decisions": sum(len(e.decisions) for e in eps),
                     "non_wait": sum(1 for e in eps for x in e.decisions if x.action.tool != "wait"),
                     "successes": [e.success for e in eps]}
    _json(os.path.join(out, "record.json"), res)
    return res


def _dataset_stats(steps: list) -> dict:
    return {"decisions": len(steps), "waits": sum(s.is_wait for s in steps),
            "unrepresentable": sum(s.unrepresentable is not None for s in steps),
            "unrepresentable_tools": sorted({s.action.tool for s in steps if s.unrepresentable}),
            "with_evidence_labels": sum(1 for s in steps if s.evidence)}


def cmd_train(out: str, minutes: float, scale: str = "dev", seed: int = 0, test_model: bool = False,
              epochs: int = 4) -> dict:
    """Resumable BC: evaluates the initial model once, then runs epochs (rollout refresh + BC pass) until `epochs`
    are done or `minutes` elapse; weights (E's save_weights) + optimiser state are saved after every epoch / chunk."""
    from ..memory.checkpoint import CheckpointOps
    deadline = time.time() + 60.0 * minutes
    cfg = run_config(scale)
    model, agent = _build(cfg, seed, test_model)
    ops = CheckpointOps(agent)
    train = load_episodes(os.path.join(out, "train", "episodes.jsonl"))
    held = load_episodes(os.path.join(out, "held", "episodes.jsonl"))
    ck = os.path.join(out, "ckpt")
    os.makedirs(ck, exist_ok=True)
    prog_path = os.path.join(out, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"epochs": [], "chunks": 0}
    bc = BehaviourCloning(model, BCConfig(seed=seed))
    if prog["epochs"] or prog.get("partial"):
        ops.load_weights_only(os.path.join(ck, "weights.safetensors"))
        bc.opt.load_state_dict(torch.load(os.path.join(ck, "optim.pt"), map_location=model.device))
        bc.updates = int(prog.get("updates", 0))
    else:
        init_dir = os.path.join(out, "init")
        os.makedirs(init_dir, exist_ok=True)
        ops.save_weights(os.path.join(init_dir, "weights.safetensors"))          # the no-demo arm
        t0 = time.time()
        hs = teacher_forced_rollout(model, held, batch=10, seed=seed)
        prog["held_dataset"] = _dataset_stats(hs)
        prog["before"] = {"held_logp": demo_logp_table(model, hs),
                          "held_match": evaluate_demo_match(model, _wait_subset(hs, 60, seed))}
        prog["before"]["eval_s"] = time.time() - t0
        _json(prog_path, prog)
    prog["chunks"] = int(prog.get("chunks", 0)) + 1
    while len(prog["epochs"]) < epochs and time.time() < deadline:
        last = prog["epochs"][-1] if prog["epochs"] else None
        if last is not None and time.time() + 1.05 * (last["rollout_s"] + last["bc_s"]) > deadline:
            break                                     # the next epoch would not fit in this chunk: leave it for the next
        t0 = time.time()
        steps = teacher_forced_rollout(model, train, batch=10, seed=seed + len(prog["epochs"]))
        prog["train_dataset"] = _dataset_stats(steps)
        t1 = time.time()
        r = bc.epoch(steps, deadline=deadline)
        r.update(rollout_s=t1 - t0, bc_s=time.time() - t1, updates=bc.updates,
                 last=bc.log[-1] if bc.log else None)
        ops.save_weights(os.path.join(ck, "weights.safetensors"))
        torch.save(bc.opt.state_dict(), os.path.join(ck, "optim.pt"))
        if r["batches"] < r["of"]:
            prog["partial"] = r                       # deadline inside the epoch: the epoch is redone next chunk
            prog["updates"] = bc.updates
            break
        prog.pop("partial", None)
        prog["epochs"].append(r)
        prog["updates"] = bc.updates
        _json(prog_path, prog)
    prog["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
    _json(prog_path, prog)
    return prog


def cmd_evaluate(out: str, minutes: float, scale: str = "dev", seed: int = 0, test_model: bool = False,
                 n_act: int = 30, arms: Sequence[str] = ("bc", "init")) -> dict:
    """After BC: held-out demo log-probability / greedy match, then both arms (BC'd weights and the initial no-demo
    weights) acting through the integrated agent on unseen OpenWorld seeds."""
    from ..memory.checkpoint import CheckpointOps
    cfg = run_config(scale)
    model, agent = _build(cfg, seed, test_model)
    ops = CheckpointOps(agent)
    path = os.path.join(out, "evaluation.json")
    ev = json.load(open(path)) if os.path.exists(path) else {}
    held = load_episodes(os.path.join(out, "held", "episodes.jsonl"))
    if "after" not in ev:
        ops.load_weights_only(os.path.join(out, "ckpt", "weights.safetensors"))
        hs = teacher_forced_rollout(model, held, seed=seed)
        ev["after"] = {"held_logp": demo_logp_table(model, hs),
                       "held_match": evaluate_demo_match(model, _wait_subset(hs, 60, seed))}
        _json(path, ev)
    tasks = make_tasks(ACT_TASKS)
    for arm in arms:
        key = f"acting_{arm}"
        if key in ev:
            continue
        ops.load_weights_only(os.path.join(out, "ckpt" if arm == "bc" else "init", "weights.safetensors"))
        t0 = time.time()
        ev[key] = act_on_tasks(model, tasks, n_act, seed=seed)
        ev[key]["wall_s"] = time.time() - t0
        _json(path, ev)
    return ev


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="CAM Unified declared demo-init condition (behaviour cloning on "
                                             "non-privileged ExampleInductor transcripts)")
    ap.add_argument("cmd", choices=("record", "train", "evaluate"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=22.0)
    ap.add_argument("--scale", default="dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-act", type=int, default=30)
    ap.add_argument("--test-model", action="store_true", help="Qwen3-0.6B (cfg.sem_test_model) instead of 1.7B")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        r = cmd_record(a.out)
    elif a.cmd == "train":
        r = cmd_train(a.out, a.minutes, a.scale, a.seed, a.test_model, a.epochs)
    else:
        r = cmd_evaluate(a.out, a.minutes, a.scale, a.seed, a.test_model, a.n_act)
    print(json.dumps(r, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
