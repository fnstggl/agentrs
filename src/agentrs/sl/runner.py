"""Experience collection: many worlds stepped in lockstep with batched inference on the shared core.

A *stream* is one learner-controlled agent in one world instance. A *trial* is K consecutive episodes of the
same task (same hidden rules / counterparties / map); the agent's belief and episodic memory persist across the
episodes of a trial and are cleared between trials (adaptation mode "full"). Ablation modes: "episode" (cleared
every episode: no cross-episode memory), "none" (belief cleared every step and no memory: no adaptation beyond
the current observation), "gru" (recurrent belief only, no external memory).

Each call to ``collect(T)`` advances every stream by T decisions and returns a segment per stream with the
recurrent state at its start, so the learner can recompute it exactly under updated weights.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from agentrs.sl.featurize import LABEL_IDS, MAX_CAND, collate, featurize, label_ids
from agentrs.sl.model import SLCore, clear, symlog
from agentrs.sl.world import GoalStatus, RewardContract, Task, World

MODES = ("full", "episode", "none", "gru")


@dataclass
class WorldSlot:
    world: World
    task: Task
    streams: list                       # agent ids controlled by the learner
    episode: int = 0
    ep_steps: int = 0
    trial_id: int = 0
    before: dict = field(default_factory=dict)   # agent -> GoalStatus at the start of the current step
    ep_return: dict = field(default_factory=dict)
    goal_feat: dict = field(default_factory=dict)


class Runner:
    def __init__(self, model: SLCore, families: dict, make_world: Callable, sample_task: Callable, n_worlds: int,
                 trial_episodes: int = 3, split: str = "train", seed: int = 0, mode: str = "full",
                 contract: Optional[RewardContract] = None, device="cpu", max_episode_steps: int = 2000,
                 fixed_tasks: Optional[list] = None, greedy: bool = False, dt_ref: Optional[dict] = None):
        assert mode in MODES
        self.m, self.dev, self.mode, self.greedy = model, device, mode, greedy
        self.families = families                      # family -> sampling weight
        self.make_world, self.sample_task = make_world, sample_task
        self.K, self.split, self.max_ep = trial_episodes, split, max_episode_steps
        self.rc = contract or RewardContract()
        self.rng = random.Random(seed)
        self.fixed = list(fixed_tasks) if fixed_tasks else None   # evaluation: exact task list, one trial each
        self.label_cache: dict = {}
        self._contracts: dict = {}
        self.slots: list = []
        self.trials_done: list = []                   # finished trial records (per stream)
        self.dt_ref = dt_ref if dt_ref is not None else {}
        for _ in range(n_worlds):
            s = self._new_slot()
            if s is None:
                break
            self.slots.append(s)
        self.n = sum(len(s.streams) for s in self.slots)
        self.state = model.init_state(self.n, device)
        self.pending_reset = torch.ones(self.n, dtype=torch.bool)   # new trial at next step
        self.pending_newep = torch.ones(self.n, dtype=torch.bool)
        self.steps_total = 0
        self.stats = {"env_s": 0.0, "infer_s": 0.0}

    # ------------------------------------------------------------------ tasks
    def _next_task(self) -> Optional[Task]:
        if self.fixed is not None:
            return self.fixed.pop(0) if self.fixed else None
        fams = list(self.families)
        fam = self.rng.choices(fams, weights=[self.families[f] for f in fams])[0]
        return self.sample_task(fam, self.rng, self.split)

    def _new_slot(self) -> Optional[WorldSlot]:
        t = self._next_task()
        if t is None:
            return None
        w = self.make_world(t.family)
        w.initialize_world(t)
        return self._start(WorldSlot(world=w, task=t, streams=self._controlled(w)))

    @staticmethod
    def _controlled(w: World) -> list:
        return list(getattr(w, "controlled", None) or w.agents())

    def _start(self, s: WorldSlot) -> WorldSlot:
        s.ep_steps = 0
        s.before = {a: s.world.evaluate_goal(a) for a in s.streams}
        s.ep_return = {a: 0.0 for a in s.streams}
        s.goal_feat = {a: featurize(s.task.family, s.world.goal(a)) for a in s.streams}
        return s

    def _contract(self, w) -> RewardContract:
        ov = getattr(w, "reward_overrides", None)
        if not ov:
            return self.rc
        key = w.family
        c = self._contracts.get(key)
        if c is None:
            from dataclasses import asdict
            from agentrs.sl.world import RewardSpec
            c = self._contracts[key] = RewardContract(RewardSpec(**{**asdict(self.rc.s), **ov}))
        return c

    def _labels(self, fam: str, lab: str) -> np.ndarray:
        k = (fam, lab)
        v = self.label_cache.get(k)
        if v is None:
            v = self.label_cache[k] = label_ids(fam, lab)
        return v

    # ------------------------------------------------------------------ one decision for every stream
    @torch.no_grad()
    def _act(self, rec: bool):
        m, dev = self.m, self.dev
        index = [(si, a) for si, s in enumerate(self.slots) for a in s.streams]
        t0 = time.time()
        feats = [featurize(self.slots[si].task.family, self.slots[si].world.observe(a)) for si, a in index]
        gfeats = [self.slots[si].goal_feat[a] for si, a in index]
        spaces = [self.slots[si].world.available_actions(a) for si, a in index]
        self.stats["env_s"] += time.time() - t0
        t1 = time.time()
        newtrial = self.pending_reset.clone()
        newep = self.pending_newep.clone()
        st = self.state
        apply_resets(m, st, self.mode, newtrial.to(dev), newep.to(dev))
        ob = m.encode(collate(feats, dev))
        ge = m.encode_goal(collate(gfeats, dev))
        flags = torch.stack([newep.float(), symlog(torch.tensor([float(self.slots[si].ep_steps) for si, _ in index]))], -1).to(dev)
        use_mem = torch.full((self.n,), uses_memory(m, self.mode), dtype=torch.bool, device=dev)
        pre = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()} if rec else None
        st = m.step_belief(st, ob, ge, flags, use_mem)
        ctx = m.context(st, ge)
        hi, term_logit = m.skill_dists(ctx, st["z"])
        force = newep.to(dev)
        if m.c.skills > 1:
            term = torch.bernoulli(torch.sigmoid(term_logit)).bool() | force
            znew = torch.distributions.Categorical(logits=hi).sample() if not self.greedy else hi.argmax(-1)
            z = torch.where(term, znew, st["z"])
            lp_sk = torch.where(force, torch.zeros_like(term_logit),
                                torch.where(term, F_logsig(term_logit), F_logsig(-term_logit)))
            lp_sk = lp_sk + torch.where(term, torch.log_softmax(hi, -1).gather(1, z[:, None]).squeeze(1), torch.zeros_like(lp_sk))
        else:
            term = force
            z = torch.zeros_like(st["z"])
            lp_sk = torch.zeros(self.n, device=dev)
        st["z"] = z
        q = m.policy_query(ctx, z)
        v = m.value_of(ctx, z)
        # slot-by-slot decoding (batched across streams at each slot position)
        prefixes = [[] for _ in range(self.n)]
        cands, chosen = [[] for _ in range(self.n)], [[] for _ in range(self.n)]
        lp_a = torch.zeros(self.n, device=dev)
        a_emb = torch.zeros(self.n, m.c.da, device=dev)
        active = list(range(self.n))
        qcur = q
        for j in range(max(sp.max_len if sp.kind == "sequential" else 1 for sp in spaces)):
            rows, opts = [], []
            for i in active:
                sp = spaces[i]
                if sp.kind == "discrete":
                    if j > 0:
                        continue
                    labs = sp.labels
                else:
                    labs = sp.allowed(prefixes[i])
                    if labs == ["END"] or not labs:
                        continue
                rows.append(i)
                opts.append(labs[:MAX_CAND])
            if not rows:
                break
            fam_of = [self.slots[index[i][0]].task.family for i in rows]
            ids = np.zeros((len(rows), MAX_CAND, LABEL_IDS), np.int64)
            msk = np.zeros((len(rows), MAX_CAND), bool)
            for r, (f, labs) in enumerate(zip(fam_of, opts)):
                for c, lab in enumerate(labs):
                    ids[r, c] = self._labels(f, lab)
                msk[r, :len(labs)] = True
            ids_t = torch.as_tensor(ids, device=dev)
            emb = m.labels(ids_t)                                         # (R, C, da)
            rt = torch.as_tensor(rows, device=dev)
            logits = torch.einsum("rcd,rd->rc", emb, qcur[rt]) / (m.c.da ** 0.5)
            logits = logits.masked_fill(~torch.as_tensor(msk, device=dev), -1e9)
            pick = logits.argmax(-1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()
            lp = torch.log_softmax(logits, -1).gather(1, pick[:, None]).squeeze(1)
            lp_a.index_add_(0, rt, lp)
            pe = emb[torch.arange(len(rows), device=dev), pick]
            a_emb.index_add_(0, rt, pe)
            qn = qcur.clone()
            qn[rt] = m.slot(pe, qcur[rt])
            qcur = qn
            pk = pick.tolist()
            for r, i in enumerate(rows):
                lab = opts[r][pk[r]]
                prefixes[i].append(lab)
                cands[i].append(ids[r, :len(opts[r])])
                chosen[i].append(pk[r])
            active = [i for i in rows if spaces[i].kind == "sequential"]
        self.stats["infer_s"] += time.time() - t1
        # execute
        t2 = time.time()
        for i, (si, a) in enumerate(index):
            sp = spaces[i]
            act = prefixes[i] if sp.kind == "sequential" else prefixes[i][0]
            self.slots[si].world.execute(a, act)
        for s in self.slots:
            s.world.advance_time()
        rewards, discs, dones, infos = [], [], [], []
        for i, (si, a) in enumerate(index):
            s = self.slots[si]
            cons = s.world.observe_consequences(a)
            after = s.world.evaluate_goal(a)
            timeout = s.ep_steps + 1 >= self.max_ep
            if timeout and not cons.done:
                cons.done = True
            r = self._contract(s.world)(s.before[a], after, cons)
            s.before[a] = after
            s.ep_return[a] += r
            fam = s.task.family
            ref = self.dt_ref.setdefault(fam, max(cons.dt, 1e-6) if cons.dt > 0 else 1.0)
            if cons.dt > 0:
                self.dt_ref[fam] = 0.999 * ref + 0.001 * cons.dt
            dtn = min(20.0, max(0.25, cons.dt / max(ref, 1e-6))) if cons.dt > 0 else 1.0
            rewards.append(r)
            discs.append(self._contract(s.world).s.gamma ** dtn)
            dones.append(cons.done)
            infos.append((fam, after, cons))
        self.stats["env_s"] += time.time() - t2
        r_t = torch.tensor(rewards, dtype=torch.float32, device=dev)
        r_feat = torch.stack([symlog(r_t), torch.tensor(dones, dtype=torch.float32, device=dev), torch.ones_like(r_t)], -1)
        st = m.write_memory(st, a_emb, r_feat, use_mem)
        self.state = st
        # episode / trial bookkeeping
        self.pending_reset[:] = False
        self.pending_newep[:] = False
        ep_end_rows = []
        for si, s in enumerate(self.slots):
            s.ep_steps += 1
        done_slots = {}
        for i, (si, a) in enumerate(index):
            if dones[i]:
                done_slots.setdefault(si, []).append(i)
        for si, rows_ in done_slots.items():
            s = self.slots[si]
            if len(rows_) < len(s.streams):
                continue  # multi-agent world: wait until every controlled agent is done
            for i in rows_:
                fam, after, _ = infos[i]
                self.trials_done.append({"family": fam, "task": s.task, "trial": s.trial_id, "episode": s.episode,
                                         "success": bool(after.success), "score": after.score, "progress": after.progress,
                                         "violated": after.violated, "return": s.ep_return[index[i][1]],
                                         "steps": s.ep_steps, "mode": self.mode, "trial_end": s.episode + 1 >= self.K})
            ep_end_rows += rows_
            s.episode += 1
            if s.episode >= self.K:  # trial over: new task
                t = self._next_task()
                if t is None:
                    s.world = None
                    continue
                s.world.close()
                w = self.make_world(t.family)
                w.initialize_world(t)
                self.slots[si] = WorldSlot(world=w, task=t, streams=self._controlled(w), trial_id=s.trial_id + 1)
                self._start(self.slots[si])
                for i in rows_:
                    self.pending_reset[i] = True
                    self.pending_newep[i] = True
            else:
                s.world.reset_world()
                self._start(s)
                for i in rows_:
                    self.pending_newep[i] = True
        self.steps_total += self.n
        out = None
        if rec:
            out = {"feat": feats, "gfeat": gfeats, "flags": flags.cpu(), "newtrial": newtrial, "newep": newep,
                   "term": term.cpu(), "z": z.cpu(), "cands": cands, "chosen": chosen, "lp": (lp_a + lp_sk).cpu(),
                   "v": v.cpu(), "r": r_t.cpu(), "disc": torch.tensor(discs), "done": torch.tensor(dones),
                   "trial_end": torch.tensor([bool(self.pending_reset[i]) for i in range(self.n)]),
                   "fam": [self.slots[si].task.family if self.slots[si].world is not None else "" for si, _ in index]}
        return out, pre

    def collect(self, T: int) -> dict:
        """Advance every stream by T decisions; returns the segment with its starting recurrent state."""
        steps, start = [], None
        for t in range(T):
            rec, pre = self._act(True)
            if t == 0:
                start = pre
            steps.append(rec)
        boot = self.bootstrap_values()
        return {"steps": steps, "start": start, "boot": boot, "mode": self.mode}

    @torch.no_grad()
    def bootstrap_values(self) -> torch.Tensor:
        """Value of the next (not yet acted) state of every stream under the current weights, without committing."""
        m, dev = self.m, self.dev
        index = [(si, a) for si, s in enumerate(self.slots) for a in s.streams]
        feats = [featurize(self.slots[si].task.family, self.slots[si].world.observe(a)) for si, a in index]
        gfeats = [self.slots[si].goal_feat[a] for si, a in index]
        st = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in self.state.items()}
        reset = self.pending_reset.to(dev)
        apply_resets(m, st, self.mode, reset, self.pending_newep.to(dev))
        ob = m.encode(collate(feats, dev))
        ge = m.encode_goal(collate(gfeats, dev))
        flags = torch.stack([self.pending_newep.float(), symlog(torch.tensor([float(self.slots[si].ep_steps) for si, _ in index]))], -1).to(dev)
        use_mem = torch.full((self.n,), uses_memory(m, self.mode), dtype=torch.bool, device=dev)
        st = m.step_belief(st, ob, ge, flags, use_mem)
        v = m.value_of(m.context(st, ge), st["z"])
        return torch.where(reset, torch.zeros_like(v), v).cpu()  # a trial boundary cuts the bootstrap

    def run_trials(self) -> list:
        """Evaluation: run until every fixed task's trial has finished; returns the per-episode records."""
        assert self.fixed is not None
        while any(s.world is not None for s in self.slots):
            live = [s for s in self.slots if s.world is not None]
            if len(live) < len(self.slots):
                # drop finished worlds (keep recurrent state rows aligned by rebuilding)
                self._compact()
            self._act(False)
        return self.trials_done

    def _compact(self):
        keep_rows, new_slots, r = [], [], 0
        for s in self.slots:
            k = len(s.streams)
            if s.world is not None:
                keep_rows += list(range(r, r + k))
                new_slots.append(s)
            r += k
        idx = torch.tensor(keep_rows, dtype=torch.long)
        self.slots = new_slots
        self.n = len(keep_rows)
        self.state = {k: (v[idx.to(v.device)] if torch.is_tensor(v) else v) for k, v in self.state.items()}
        self.pending_reset = self.pending_reset[idx]
        self.pending_newep = self.pending_newep[idx]


def uses_memory(m, mode: str) -> bool:
    return mode in ("full", "episode") and m.c.memory


def apply_resets(m, st: dict, mode: str, newtrial: torch.Tensor, newep: torch.Tensor) -> None:
    """Identical at collection and in the learner's replay. Trial start: clear belief and memory. New episode in
    the same trial: "full"/"gru" keep belief (and memory) - cross-episode adaptation; "episode" clears both.
    "none": cleared before every decision."""
    m.reset_rows(st, torch.nonzero(newtrial).flatten(), keep_memory=False)
    if mode == "episode":
        m.reset_rows(st, torch.nonzero(newep).flatten(), keep_memory=False)
    elif mode in ("full", "gru"):
        clear(st, newep & ~newtrial, ("prev_a", "prev_r"))
    if mode == "none":
        m.reset_rows(st, torch.arange(st["h"].shape[0], device=st["h"].device), keep_memory=False)


def F_logsig(x):
    return torch.nn.functional.logsigmoid(x)
