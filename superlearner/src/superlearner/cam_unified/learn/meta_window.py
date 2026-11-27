"""The psi meta step inside the canonical loop (plan 2026-11-26 §2 "L_meta (outer)" / "Inner"; W8 change request to
W7: `meta_step()` calls MetaRewardTrainer start -> act_with / inner_step x K -> query -> outer_step every meta_every
updates).

W8's `selfeval.metagrad.MetaRewardTrainer` needs ON-POLICY windows acted with the fast weights phi_k and a query
window acted with phi_K whose episodes carry verified utilities. In the canonical loop the learner cannot act, so
`MetaWindowDriver` spreads one meta step over the rounds that follow `learner.meta_step()`:

    begin (meta_step)  fast = trainer.start(); act with phi_0 (trainer.act_with)
    inner k = 0..K-1   collect `window_rounds` rounds of the agent's fresh steps acted with phi_k,
                       fast = trainer.inner_step(fast, fast_policy.window_batch(window)) -> act with phi_{k+1}
    query              collect `window_rounds` rounds acted with phi_K, then keep acting with phi_K until the query
                       episodes' verified labels are in the LabelStore (at most `max_wait_rounds`)
    outer              trainer.outer_step(fast, fast_policy.query_batch(window, utilities)) (psi changes, version
                       bumps), restore the slow weights, then `trainer.commit(fast)` (consolidate phi_K, plan §2) and
                       bump the policy version, so every decision acted inside the window is lag >= 1 for the next
                       slow update (V-trace with its recorded behaviour log-probabilities, never a PPO ratio)

While a window is open the learner's slow step is suspended (`IntegratedLearner.step` returns updated=False): a slow
update would write into parameters that hold phi_k and be undone by act_with's restore. A window that cannot get its
query utilities in time is aborted (slow weights restored, psi unchanged, logged). Verified utilities are read only
here, on the learner side, and reach psi only through the outer loss (never an actor input or loss).

`fp` is W8's `selfeval.fast_policy` module (window / window_batch / query_batch), injectable for tests.
"""
from __future__ import annotations

import math
import time
from typing import Optional


class MetaWindowDriver:
    def __init__(self, trainer, *, fp=None, window_rounds: int = 8, max_wait_rounds: int = 64, kappa: float = 0.0,
                 commit: bool = True, min_query_rows: int = 1):
        if fp is None:
            from ..selfeval import fast_policy as fp                 # W8
        self.trainer = trainer
        self.fp = fp
        self.window_rounds = int(window_rounds)
        self.max_wait_rounds = int(max_wait_rounds)
        self.kappa = float(kappa)
        self.commit_fast = bool(commit)
        self.min_query_rows = int(min_query_rows)
        self.K = int(getattr(getattr(trainer, "mrl", None), "K", getattr(trainer, "K", 2)))
        self.phase: Optional[str] = None          # None | "inner" | "query" | "wait"
        self.k = 0
        self.fast = None
        self._cm = None
        self._steps: list = []
        self._seen: set = set()
        self._rounds = 0
        self._wait = 0
        self._query_steps: list = []
        self.log: list = []
        self.pending: list = []                   # finished / aborted meta steps not yet reported by an update

    @property
    def active(self) -> bool:
        return self.phase is not None

    # ------------------------------------------------------------------ phi substitution
    def _enter(self, fast) -> None:
        self._exit()
        self._cm = self.trainer.act_with(fast)
        self._cm.__enter__()

    def _exit(self) -> None:
        if self._cm is not None:
            cm, self._cm = self._cm, None
            cm.__exit__(None, None, None)

    # ------------------------------------------------------------------ protocol
    def meta_step(self, learner) -> dict:
        """Called by IntegratedLearner.meta_step(): opens a meta window (no-op while one is open)."""
        if self.active:
            return {"started": False, "reason": f"meta window open (phase {self.phase}, k={self.k})"}
        self.fast = self.trainer.start()
        self._enter(self.fast)
        self.phase, self.k = "inner", 0
        self._steps, self._rounds, self._wait = [], 0, 0
        self._seen = {id(s) for s in getattr(learner.agent, "fresh", [])}
        self.t0 = time.time()
        return {"started": True, "K": self.K, "window_rounds": self.window_rounds,
                "psi_version": int(getattr(self.trainer, "version", 0))}

    def _new_steps(self, learner) -> list:
        out = []
        for s in getattr(learner.agent, "fresh", []):
            if id(s) not in self._seen:
                self._seen.add(id(s))
                out.append(s)
        return out

    def on_round(self, learner) -> Optional[dict]:
        """Called by IntegratedLearner.on_step after each round; returns the outer-step entry when one finished."""
        if not self.active:
            return None
        new = self._new_steps(learner)
        if self.phase in ("inner", "query"):
            self._steps.extend(new)
            self._rounds += 1
            if self._rounds < self.window_rounds:
                return None
            steps, self._steps, self._rounds = self._steps, [], 0
            if not steps:
                return self.abort(learner, "no decisions in the window")
            if self.phase == "inner":
                w = self.fp.window(learner.model, steps)
                self.fast, st = self.trainer.inner_step(self.fast, self.fp.window_batch(w, self.kappa),
                                                        on_policy=True)
                self.k += 1
                self._enter(self.fast)                  # act with phi_{k}
                if self.k >= self.K:
                    self.phase = "query"
                return None
            self._query_steps = steps
            self.phase = "wait"
        if self.phase == "wait":
            U = self._utilities(learner, self._query_steps)
            n = sum(u is not None for u in U.values())
            if n >= max(1, min(self.min_query_rows, len(U))) and n == len(U):
                return self._outer(learner, U)
            self._wait += 1
            if self._wait > self.max_wait_rounds:
                if n >= self.min_query_rows:
                    return self._outer(learner, U)
                return self.abort(learner, f"no verified query utility within {self.max_wait_rounds} rounds")
        return None

    def _utilities(self, learner, steps) -> dict:
        """env_id -> verified utility of the query episode of that env (None while pending), from the LabelStore."""
        eps = {}
        for s in steps:
            envs = s.env_ids or [str(r) for r in s.rows]
            for j, e in enumerate(envs):
                eps.setdefault(str(e), learner._register_episode(e, s.episode_ids[j]))
        labs = {}
        if learner.label_store is not None:
            for lab in learner.label_store.iterate():
                if getattr(lab, "kind", "terminal") == "terminal" and lab.utility is not None:
                    labs[(str(lab.env_id), int(lab.episode))] = float(lab.utility)
        return {e: labs.get((e, ep)) for e, ep in eps.items()}

    def _outer(self, learner, U: dict) -> dict:
        w = self.fp.window(learner.model, self._query_steps)
        q = self.fp.query_batch(w, {e: u for e, u in U.items() if u is not None})
        base = None
        if learner.label_store is not None:      # score baseline: mean verified utility of earlier labels
            us = [float(l.utility) for l in learner.label_store.iterate() if l.utility is not None]
            base = sum(us) / len(us) if us else None
        st = self.trainer.outer_step(self.fast, q, score_baseline=base)
        self._exit()                               # slow weights back, then consolidate phi_K into them
        if self.commit_fast:
            self.trainer.commit(self.fast)
        learner._bump_policy_version()
        entry = {"meta": True, "outer": {k: v for k, v in (st or {}).items() if _jsonable(v)},
                 "query_rows": int(sum(u is not None for u in U.values())), "inner_steps": self.k,
                 "committed": self.commit_fast, "wall_s": time.time() - getattr(self, "t0", time.time())}
        self._reset()
        self.log.append(entry)
        self.pending.append(entry)
        return entry

    def abort(self, learner, reason: str) -> dict:
        self._exit()
        if self.k > 0 or self._query_steps:
            learner._bump_policy_version()         # decisions acted with phi_k are off-policy for the slow weights
        entry = {"meta": False, "aborted": reason, "inner_steps": self.k}
        self._reset()
        self.log.append(entry)
        self.pending.append(entry)
        return entry

    def _reset(self) -> None:
        self.phase, self.k, self.fast = None, 0, None
        self._steps, self._query_steps, self._rounds, self._wait = [], [], 0, 0

    def state_dict(self) -> dict:
        sd = self.trainer.state_dict() if hasattr(self.trainer, "state_dict") else None
        return {"trainer": sd, "log": list(self.log[-50:])}

    def load_state_dict(self, d: dict) -> None:
        if d.get("trainer") is not None and hasattr(self.trainer, "load_state_dict"):
            self.trainer.load_state_dict(d["trainer"])
        self.log = list(d.get("log") or [])


def _jsonable(v) -> bool:
    if isinstance(v, (int, str, bool)) or v is None:
        return True
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, (list, tuple)):
        return all(_jsonable(x) for x in v)
    if isinstance(v, dict):
        return all(_jsonable(x) for x in v.values())
    return False
