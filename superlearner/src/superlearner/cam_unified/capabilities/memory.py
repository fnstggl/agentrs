"""E03 through the complete architecture: a delayed cue needed many decisions later (spec 10.6).

Protocol (`run_memory`): `DelayedCueEnv` - decision 0 shows the cue colour, the next `delay` decisions bring
interfering colour mentions (the agent should wait), the last decision must press the cue's button. Each
observation is in the core's working context only at the decision it is delivered, so at the answer the cue is
available only through the persistent state Z (kept across decisions by the core) or through lifetime memory:
EventArchive (E) -> Consolidator segments -> Retriever segment encoder + MemoryIndex keys -> query
q = Q(Z, G, B) -> top-k MEMORY tokens in the core's context. Conditions (same budget):

* "ablated": cfg.use_memory = False (Z only);
* "memory": cfg.use_memory = True with the agent's default retrieval (no filters: the whole index);
* "memory_episode": cfg.use_memory = True with retrieval scoped to the agent's current episode (MemoryIndex filter
  {"episode_id": ...}; a declared retrieval-scoping condition that isolates whether the retrieved memory content
  carries the cue from whether the learned query can find it among other episodes' segments).

Training: the agent's own objectives through `runtime.run_steps` (consolidation happens in `agent.update()`);
replay sequences span whole episodes so the world model can learn to keep the cue in Z. Evaluation (greedy, fresh
development seeds): answer accuracy, and for the memory conditions the retrieval hit rate - the fraction of answer
decisions whose retrieved segments contain the cue event of the same episode - against the number of segments in
the index.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..interfaces import ObservationBatch
from ..runtime import SEED_STRIDE
from .envs import DelayedCueEnv, ScoreEvaluator
from .loop import mean, train_agent
from .stack import build_stack, capability_config


class ListRecorder:
    """In-memory TraceRecorderP (DecisionTrace objects; the update links are kept as given)."""

    def __init__(self):
        self.traces: list = []
        self.links: list = []

    def record(self, traces: list) -> None:
        self.traces.extend(traces)

    def link_updates(self, decision_ids: list, update_id: str) -> None:
        self.links.append((list(decision_ids), update_id))


class EpisodeScopedRetrieval:
    """Scope the model's retrieval to each row's current episode (MemoryIndex filter on episode_id) for the decisions
    `agent.act` makes; installed as instance attributes (no class is modified) and removed by `remove()`."""

    def __init__(self, agent):
        self.agent = agent
        self.model = agent.model
        self._filters = None
        orig_act, orig_decide = agent.act, self.model.decide

        def act(obs_batch, tools_batch):
            rows = [b for b in range(len(obs_batch.env_ids)) if not obs_batch.done[b]]
            if agent.state is None or list(obs_batch.env_ids) != agent.env_ids:
                agent.reset(obs_batch.env_ids)
            self._filters = [{"episode_id": agent.episode_ids[b]} for b in rows]
            return orig_act(obs_batch, tools_batch)

        def decide(*a, **k):
            if self._filters is not None and k.get("retrieval_filters") is None:
                k["retrieval_filters"] = self._filters
            return orig_decide(*a, **k)

        self._old = (agent.__dict__.get("act"), self.model.__dict__.get("decide"))
        agent.act = act
        self.model.__dict__["decide"] = decide

    def remove(self) -> None:
        for obj, name, old in ((self.agent, "act", self._old[0]), (self.model, "decide", self._old[1])):
            if old is None:
                obj.__dict__.pop(name, None)
            else:
                obj.__dict__[name] = old


@dataclass
class MemorySettings:
    seed: int = 0
    condition: str = "memory_episode"     # ablated | memory | memory_episode
    B: int = 16
    delay: int = 10
    n_colours: int = 4
    distractors: int = 2
    train_rounds: int = 360
    update_every: int = 1
    lr: float = 1e-3
    eval_rows: int = 32
    eval_episodes: int = 2
    eval_seed0: int = 100_000
    cfg_overrides: dict = field(default_factory=dict)


def evaluate_cue(stack, st: MemorySettings) -> dict:
    """Greedy agent episodes on development seeds with archiving and consolidation (no learner update)."""
    agent = stack.agent
    env = DelayedCueEnv(st.eval_rows, delay=st.delay, n_colours=st.n_colours, distractors=st.distractors)
    ev = ScoreEvaluator(env)
    old_g, old_rec = agent.greedy, agent.recorder
    rec = ListRecorder()
    agent.greedy, agent.recorder = True, rec
    accs, hits, n_segments, retrieved_any, in_ctx = [], [], [], [], []
    try:
        for e in range(st.eval_episodes):
            ob = env.reset([st.eval_seed0 + SEED_STRIDE * e + b for b in range(st.eval_rows)])
            agent.reset(ob.env_ids)
            agent.set_goals(env.goals())
            cue_eid: dict = {}
            scores = [0.0] * env.B
            last_traces = None
            while not all(ob.done):
                n0 = len(rec.traces)
                acts = agent.act(ob, env.available_tools())
                res = env.execute(acts)
                adv = env.advance_time([(r.duration if r is not None else 0.0) + stack.model.think_days
                                        for r in res])
                new = [float(ev.evaluate_goal(b).score) for b in range(env.B)]
                rew = [n - s for n, s in zip(new, scores)]
                scores = new
                nxt = [list(res[b].observations if res[b] is not None else []) + list(adv.observations[b])
                       for b in range(env.B)]
                nob = ObservationBatch(list(ob.env_ids), nxt, list(adv.t), list(adv.done))
                recs = agent.observe(res, rew, nob)
                for r in recs:
                    if r.action is not None and r.step == 0:
                        cue_eid[r.env_id] = r.event_id
                if stack.consolidator is not None:
                    stack.consolidator.step()
                last_traces = rec.traces[n0:]
                ob = nob
            accs.append(mean(scores[b] > 0.5 for b in range(env.B)))
            if stack.index is not None and last_traces:
                ver = stack.model.retriever.key_version
                n_segments.append(stack.index.count(ver) if ver in stack.index.versions() else 0)
                for tr in last_traces:
                    ids = [x["id"] for x in tr.retrieved]
                    retrieved_any.append(float(bool(ids)))
                    src = set()
                    for sid in ids:
                        src.update(stack.model.retriever.resolve_ref("seg:" + sid))
                    hits.append(float(cue_eid.get(tr.env_id) in src))
                    ctx_src = set()
                    for ref in tr.context_refs:
                        if isinstance(ref, str) and ref.startswith("seg:"):
                            ctx_src.update(stack.model.retriever.resolve_ref(ref))
                    in_ctx.append(float(cue_eid.get(tr.env_id) in ctx_src))
    finally:
        agent.greedy, agent.recorder = old_g, old_rec
        agent.flush("end_of_run")
    return {"accuracy": mean(accs), "chance": 1.0 / st.n_colours, "retrieval_hit_rate": mean(hits) if hits else None,
            "retrieved_any_rate": mean(retrieved_any) if retrieved_any else None,
            "cue_memory_in_context_rate": mean(in_ctx) if in_ctx else None,
            "index_segments": n_segments[-1] if n_segments else None,
            "top_k": stack.cfg.mem_top_k}


def run_memory(settings: Optional[MemorySettings] = None, root: Optional[str] = None, device=None,
               log_every: int = 0) -> dict:
    st = settings or MemorySettings()
    t0 = time.time()
    use_mem = st.condition != "ablated"
    H = st.delay + 2
    cfg = capability_config(use_imagination=False, use_memory=use_mem, lr=st.lr, mem_segment=8, **st.cfg_overrides)
    tmp = root or tempfile.mkdtemp(prefix="cam_y_memory_")
    stack = build_stack(cfg, device=device, seed=st.seed, memory_root=os.path.join(tmp, f"{st.condition}_s{st.seed}"),
                        agent_kwargs=dict(seq_len=H, n_seqs=4, adapt_seq_len=H, adapt_n_seqs=2))
    scope = EpisodeScopedRetrieval(stack.agent) if st.condition == "memory_episode" else None
    env = DelayedCueEnv(st.B, delay=st.delay, n_colours=st.n_colours, distractors=st.distractors)
    ev = ScoreEvaluator(env)
    before = evaluate_cue(stack, st)
    tr = train_agent(stack, env, ev, st.train_rounds, update_every=st.update_every,
                     seeds=[st.seed * 10_007 + b for b in range(st.B)], log_every=log_every)
    after = evaluate_cue(stack, st)
    if scope is not None:
        scope.remove()
    rpr = tr["reward_per_round"]
    per_ep = [sum(rpr[i:i + H]) for i in range(0, len(rpr) - H + 1, H)]
    return {"settings": asdict(st), "components": stack.real_component_report(),
            "train": {"n_updates": tr["n_updates"], "final_losses": tr["final_losses"], "compute": tr["compute"],
                      "return_per_episode_first3": per_ep[:3], "return_per_episode_last3": per_ep[-3:]},
            "untrained": before, "trained": after,
            "retriever_has_loss": hasattr(stack.model.retriever, "loss") if stack.model.retriever is not None else None,
            "retriever_has_memory_tokens": hasattr(stack.model.retriever, "memory_tokens")
            if stack.model.retriever is not None else None,
            "wall_s": time.time() - t0}
