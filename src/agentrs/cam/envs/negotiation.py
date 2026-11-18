"""Multi-issue procurement negotiation with a counterparty that has hidden preferences, a hidden
reservation utility, hidden patience, a hidden concern that the right argument addresses, and its own
concession policy. The agent (buyer) must reach an agreement whose utility *for the agent* is at
least its target; the counterparty accepts only offers that are good enough for *its* configured
objective at that moment, counter-proposes otherwise, and walks away when its patience runs out.

Train/val counterparty policies: boulware, conceder, linear, tit-for-tat; preferences ~ Dirichlet(1).
Held-out (test): deadline-driven and argument-anchored counterparties, peaked preferences
(Dirichlet(0.3)) and higher reservation utilities.
"""
from __future__ import annotations

import itertools

import numpy as np

from agentrs.cam.envs.base import CamEnv, Contract, Grammar, Status, StepResult, nb

ISSUES = ["price", "qty", "delivery", "warranty"]
LEVELS = [f"N{i}" for i in range(11)]
REASONS = ["volume", "longterm", "market", "quality", "urgent"]
ALL = np.array(list(itertools.product(range(11), repeat=4)), dtype=np.int16)  # 14641 offers
TRAIN_TYPES = ["boulware", "conceder", "linear", "titfortat"]
TEST_TYPES = ["deadline", "anchored"]


class NegotiationEnv(CamEnv):
    family = "negotiation"
    keywords = ["GOAL", "NEGOTIATE", "BUY", "W", "DIR", "UP", "DOWN", "TARGET", "ROUND", "LAST", "CP", "NONE",
                "COUNTER", "ACCEPTED", "REJECTED", "LEFT", "ANSWER", "MOOD", "OK", "TENSE", "ANGRY", "KNOW", "HIGH",
                "MED", "LOW", "BEST", "ASK", "PROPOSE", "ACCEPT", "WALK", "ARG", "NOARG", *ISSUES, *REASONS]
    events = ["counter", "accept", "cp_walk", "agreement", "answer"]
    n_features = 14
    max_steps = 14
    _shared_attrs = ("u_a_all", "u_c_all")

    def reset(self, seed: int, split: str = "train") -> None:
        self._seed(seed, split)
        r, g = self.rng, self.np
        test = split == "test"
        self.cp_type = r.choice(TEST_TYPES if test else TRAIN_TYPES)
        self.w_a = g.dirichlet(np.ones(4))
        self.w_c = g.dirichlet(np.full(4, 0.3 if test else 1.0))
        self.d_a = np.array([-1, 1, -1, 1])  # buyer: low price, high qty, short delivery, long warranty
        # counterparty is opposed on price; on other issues opposed or aligned (integrative issues)
        self.d_c = np.array([1] + [(-d if r.random() < 0.6 else d) for d in self.d_a[1:]])
        self.r_c = r.uniform(0.5, 0.7) if test else r.uniform(0.3, 0.6)
        self.patience = r.randint(2, 4)
        self.concern = r.choice(REASONS)
        self.cp_deadline = r.randint(8, 13)
        self.u_a_all = self._u(ALL, self.w_a, self.d_a)
        self.u_c_all = self._u(ALL, self.w_c, self.d_c)
        feas = self.u_c_all >= self.r_c
        self.best_possible = float(self.u_a_all[feas].max()) if feas.any() else 0.0
        self.target = r.uniform(0.72, 0.9) * self.best_possible
        self.round = 0
        self.last_cp = None            # counteroffer (levels)
        self.last_resp = "NONE"
        self.known: dict[int, str] = {}
        self.answer = None
        self.agreement = None
        self.arg_bonus = 0.0
        self.last_agent_uc = None
        self.tft_drop = 0.0
        self.best_cp_ua = 0.0
        self.patience0 = self.patience
        self.end_reason = None

    def __getstate__(self):  # the utility tables are a pure function of the hidden parameters: don't serialise them
        d = self.__dict__.copy()
        d.pop("u_a_all", None)
        d.pop("u_c_all", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if "w_a" in d:
            self.u_a_all = self._u(ALL, self.w_a, self.d_a)
            self.u_c_all = self._u(ALL, self.w_c, self.d_c)

    @staticmethod
    def _u(x, w, d):
        v = np.where(d > 0, x / 10.0, (10 - x) / 10.0)
        return (v * w).sum(-1)

    def _threshold(self) -> float:
        tau = min(1.0, self.round / self.cp_deadline)
        r = self.r_c
        if self.cp_type == "boulware":
            base = r + (1 - r) * (1 - tau ** 3)
        elif self.cp_type == "conceder":
            base = r + (1 - r) * (1 - tau ** 0.33)
        elif self.cp_type == "linear":
            base = r + (1 - r) * (1 - tau)
        elif self.cp_type == "titfortat":
            base = max(r, 0.95 - self.tft_drop)
        elif self.cp_type == "deadline":
            base = 0.92 if tau < 0.7 else r
        else:  # anchored: barely moves unless addressed with the right argument
            base = r + (1 - r) * (1 - 0.4 * tau)
        return max(r, base - self.arg_bonus)

    def goal_tokens(self):
        t = ["GOAL", "NEGOTIATE", "BUY"]
        for i, n in enumerate(ISSUES):
            t += ["W", n, nb(self.w_a[i]), "DIR", "UP" if self.d_a[i] > 0 else "DOWN"]
        return t + ["TARGET", nb(self.target)]

    def obs_tokens(self):
        t = ["ROUND", nb(self.round, 0, 14), "CP", self.last_resp]
        if self.last_cp is not None:
            t += ["LAST"] + [LEVELS[int(v)] for v in self.last_cp] + ["BEST", nb(self.best_cp_ua)]
        mood = "OK" if self.patience >= 3 else ("TENSE" if self.patience == 2 else "ANGRY")
        t += ["MOOD", mood]
        if self.answer is not None:
            t += ["ANSWER"] + list(self.answer)
        for i, imp in sorted(self.known.items()):
            t += ["KNOW", ISSUES[i], imp]
        return t

    def features(self):
        f = np.zeros(self.n_features, dtype=np.float32)
        f[0] = self.round / 14
        if self.last_cp is not None:
            f[1:5] = np.asarray(self.last_cp) / 10
            f[5] = self.u_a_all[self._idx(self.last_cp)]
        f[6] = self.best_cp_ua
        f[7] = {"OK": 1.0, "TENSE": 0.5, "ANGRY": 0.0}["OK" if self.patience >= 3 else ("TENSE" if self.patience == 2 else "ANGRY")]
        for i, imp in self.known.items():
            f[8 + i] = {"HIGH": 1.0, "MED": 0.5, "LOW": 0.1}[imp]
        f[12] = float(self.last_resp == "COUNTER")
        f[13] = float(self.agreement is not None)
        return f

    @staticmethod
    def _idx(x) -> int:
        a, b, c, d = (int(v) for v in x)
        return ((a * 11 + b) * 11 + c) * 11 + d

    def grammar(self):
        return GRAMMAR

    def progress(self):
        if self.agreement is not None:
            return min(1.0, float(self.u_a_all[self._idx(self.agreement)]) / max(1e-6, self.target))
        return min(1.0, self.best_cp_ua / max(1e-6, self.target)) * 0.9  # lockable offer on the table

    def _finish(self, ok: bool) -> None:
        self.done = True
        u = float(self.u_a_all[self._idx(self.agreement)]) if self.agreement is not None else 0.0
        self.success = bool(ok and self.agreement is not None and u >= self.target - 1e-9)

    def step(self, action):
        ev = {k: False for k in self.events}
        typ = action[0]
        self.round += 1
        self.answer = None
        el = 1.0
        if typ == "ASK":
            i = ISSUES.index(action[1])
            rank = int((self.w_c > self.w_c[i]).sum())
            imp = "HIGH" if rank == 0 else ("LOW" if rank == 3 else "MED")
            if self.rng.random() < 0.15:
                imp = self.rng.choice(["HIGH", "MED", "LOW"])
            self.known[i] = imp
            self.answer = ("ASK", ISSUES[i], imp)
            ev["answer"] = True
            self.last_resp = "NONE" if self.last_cp is None else "COUNTER"
            el = self.rng.uniform(0.5, 3)
        elif typ == "PROPOSE":
            offer = np.array([LEVELS.index(a) for a in action[1:5]])
            arg = action[5]
            if arg != "NOARG" and arg == self.concern and self.arg_bonus == 0.0:
                self.arg_bonus = 0.12 if self.cp_type == "anchored" else 0.05
            uc = float(self.u_c_all[self._idx(offer)])
            if self.last_agent_uc is not None and uc > self.last_agent_uc:
                self.tft_drop += 0.8 * (uc - self.last_agent_uc)
            self.last_agent_uc = uc
            el = self.rng.uniform(2, 24)
            if uc >= self._threshold():
                self.agreement = offer
                self.last_resp = "ACCEPTED"
                ev["accept"] = ev["agreement"] = True
                self._finish(True)
            else:
                if uc < self.r_c - 0.12:
                    self.patience -= 1
                if self.patience <= 0 or self.round >= self.cp_deadline + 3:
                    self.last_resp = "LEFT"
                    ev["cp_walk"] = True
                    self.end_reason = "constraint:no_walkout"
                    self._finish(False)
                else:
                    thr = self._threshold()
                    band = (self.u_c_all >= thr) & (self.u_c_all <= thr + 0.06)
                    if not band.any():
                        band = self.u_c_all >= thr
                    cand = ALL[band]
                    j = int(np.abs(cand - offer).sum(-1).argmin())
                    self.last_cp = cand[j]
                    self.last_resp = "COUNTER"
                    ev["counter"] = True
                    self.best_cp_ua = max(self.best_cp_ua, float(self.u_a_all[self._idx(self.last_cp)]))
        elif typ == "ACCEPT":
            if self.last_cp is not None:
                self.agreement = self.last_cp
                ev["agreement"] = True
                self.last_resp = "ACCEPTED"
                self._finish(True)
            else:  # nothing on the table to accept: invalid, but the round is used up
                self.last_resp = "REJECTED"
                if self.round >= self.max_steps:
                    self._finish(False)
                return self._out(ev, el, valid=False)
        elif typ == "WALK":
            self.last_resp = "LEFT"
            self.end_reason = "abandoned"
            self._finish(False)
        if not self.done and self.round >= self.max_steps:
            self._finish(False)
        return self._out(ev, el)

    def _out(self, ev, el, valid=True):
        return StepResult(elapsed_h=el, events=ev, done=self.done, success=self.success if self.done else None,
                          valid=valid)

    # ------------------------------------------------------------------ task contract
    MILESTONES = ("learned_preference", "counteroffer", "offer_50pct", "offer_80pct", "agreement")

    def contract(self):
        return Contract(clock="round", horizon=float(self.max_steps), resources=("goodwill",),
                        constraints=("no_walkout",), milestones=self.MILESTONES,
                        failure_causes=("deadline", "constraint:no_walkout", "abandoned", "agreement_below_target"),
                        risk_budget=0.25)

    def _ua(self):
        return float(self.u_a_all[self._idx(self.agreement)]) if self.agreement is not None else 0.0

    def task_status(self):
        mood = 1.0 if self.patience >= 3 else (0.5 if self.patience == 2 else 0.0)  # observable as MOOD
        ms = {"learned_preference": bool(self.known), "counteroffer": self.last_cp is not None,
              "offer_50pct": self.best_cp_ua >= 0.5 * self.target, "offer_80pct": self.best_cp_ua >= 0.8 * self.target,
              "agreement": self.agreement is not None}
        return Status(progress=self.progress(), clock=float(self.round), time_left=max(0.0, 1 - self.round / self.max_steps),
                      resources={"goodwill": mood}, milestones=ms,
                      violated="no_walkout" if self.end_reason == "constraint:no_walkout" else None,
                      costs={"rounds": float(self.round)})

    def failure_cause(self):
        if not self.done or self.success:
            return None
        if self.end_reason:
            return self.end_reason
        return "agreement_below_target" if self.agreement is not None else "deadline"

    def hindsight_goals(self):
        """An agreement below the target verifiably achieved the same negotiation goal with the target
        lowered to the agreement's utility (floored to the goal's numeric bin)."""
        if not self.done or self.success or self.agreement is None:
            return []
        u = self._ua()
        k = int(np.floor(u * 20 + 1e-9))
        if k <= 0:
            return []
        g = self.goal_tokens()
        return [(g[:-1] + [f"N{k}"], None)]

    def hidden(self):
        return {"cp_type": self.cp_type, "w_c": self.w_c.round(3).tolist(), "r_c": round(self.r_c, 3),
                "concern": self.concern, "target": round(self.target, 3), "best_possible": round(self.best_possible, 3)}


GRAMMAR = Grammar({"ASK": [ISSUES], "PROPOSE": [LEVELS, LEVELS, LEVELS, LEVELS, ["NOARG", *REASONS]],
                        "ACCEPT": [], "WALK": []}, name="negotiation")
