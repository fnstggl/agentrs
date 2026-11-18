"""Business development / coordination: secure commitments from enough partners for a joint project.

Five potential partners with hidden incentives (revenue share, exclusivity, speed, social proof),
hidden minimum share, and hidden dependencies (one partner only commits after its 'anchor' partner
has committed; a rival refuses if its rival committed). Partners respond after a delay with interest,
a typed objection, a decline or a commitment. Commitments consume a limited pool of revenue share.
Success (verified by the environment): at least ``k`` commitments with total promised share within
the cap before the deadline. Held-out (test): more dependencies, more rivalry, tighter cap.
"""
from __future__ import annotations

import numpy as np

from agentrs.cam.envs.base import CamEnv, Contract, Grammar, Status, StepResult, nb

NP = 5
PARTNERS = [f"P{i}" for i in range(NP)]
WANTS = ["share", "exclusive", "speed", "proof"]
SHARES = [f"N{i}" for i in range(11)]  # x 5 %


class BizDevEnv(CamEnv):
    family = "bizdev"
    keywords = ["GOAL", "SECURE", "COMMITMENTS", "CAP", "STEP", "POOL", "PARTNER", "STATUS", "UNKNOWN", "INTERESTED",
                "OBJECTION", "DECLINED", "COMMITTED", "PENDING", "WANTS", "RESEARCH", "PITCH", "FOLLOWUP", "WAIT",
                "EXCL", "NOEXCL", "FAST", "SLOW", *PARTNERS, *WANTS]
    events = ["reply", "objection", "commit", "decline"]
    n_features = 18
    max_steps = 20

    def reset(self, seed: int, split: str = "train") -> None:
        self._seed(seed, split)
        r = self.rng
        test = split == "test"
        self.want = [r.choice(WANTS) for _ in range(NP)]
        self.min_share = [r.randint(2, 6) for _ in range(NP)]  # x 5 %
        self.anchor = [None] * NP
        self.rival = [None] * NP
        for i in range(NP):
            if r.random() < (0.5 if test else 0.25):
                self.anchor[i] = r.choice([j for j in range(NP) if j != i])
            if r.random() < (0.35 if test else 0.15):
                self.rival[i] = r.choice([j for j in range(NP) if j != i])
        self.k = 3
        self.cap = 12 if test else 15  # x 5 % of revenue share available
        self.status = ["UNKNOWN"] * NP
        self.promised = [0] * NP
        self.known: dict[int, str] = {}
        self.pending: list[tuple[int, int, str]] = []  # (due step, partner, reply)
        self.step_no = 0
        self.last_obj: dict[int, str] = {}
        self.commit_steps: list[int] = []
        self.replies = 0

    def goal_tokens(self):
        return ["GOAL", "SECURE", nb(self.k, 0, 5), "COMMITMENTS", "CAP", nb(self.cap, 0, 20)]

    def obs_tokens(self):
        pool = self.cap - sum(self.promised)
        t = ["STEP", nb(self.step_no, 0, 20), "POOL", nb(max(0, pool), 0, 20)]
        for i in range(NP):
            t += ["PARTNER", PARTNERS[i], "STATUS", self.status[i]]
            if i in self.last_obj:
                t += ["OBJECTION", self.last_obj[i]]
            if i in self.known:
                t += ["WANTS", self.known[i]]
        return t

    def features(self):
        f = np.zeros(self.n_features, dtype=np.float32)
        f[0] = self.step_no / 20
        f[1] = (self.cap - sum(self.promised)) / 20
        codes = {"UNKNOWN": 0.0, "PENDING": 0.25, "INTERESTED": 0.5, "OBJECTION": 0.4, "DECLINED": 0.1, "COMMITTED": 1.0}
        for i in range(NP):
            f[2 + i] = codes[self.status[i]]
            f[7 + i] = float(i in self.known)
        f[12] = sum(s == "COMMITTED" for s in self.status) / self.k
        f[13] = len(self.pending) / NP
        return f

    def grammar(self):
        return GRAMMAR

    def progress(self):
        return min(1.0, sum(s == "COMMITTED" for s in self.status) / self.k)

    def _decide(self, i: int, share: int, excl: bool, fast: bool) -> str:
        committed = [s == "COMMITTED" for s in self.status]
        if self.rival[i] is not None and committed[self.rival[i]]:
            return "DECLINED"
        ok = share >= self.min_share[i]
        w = self.want[i]
        if w == "share":
            ok = share >= self.min_share[i] + 2
        elif w == "exclusive":
            ok = ok and excl
        elif w == "speed":
            ok = ok and fast
        elif w == "proof":
            ok = ok and sum(committed) >= 1
        if ok and self.anchor[i] is not None and not committed[self.anchor[i]]:
            return "OBJECTION:proof"
        if ok:
            return "COMMITTED"
        return f"OBJECTION:{w}"

    def step(self, action):
        ev = {k: False for k in self.events}
        typ = action[0]
        self.step_no += 1
        valid = True
        if typ == "RESEARCH":
            i = PARTNERS.index(action[1])
            self.known[i] = self.want[i] if self.rng.random() > 0.15 else self.rng.choice(WANTS)
        elif typ == "PITCH":
            i = PARTNERS.index(action[1])
            if self.status[i] in ("COMMITTED", "DECLINED", "PENDING"):
                valid = False
            else:
                share, excl, fast = SHARES.index(action[2]), action[3] == "EXCL", action[4] == "FAST"
                excl_taken = any(self.status[j] == "COMMITTED" and self.promised[j] < 0 for j in range(NP))
                self.status[i] = "PENDING"
                self.pending.append((self.step_no + self.rng.randint(1, 3), i, f"{share}|{int(excl)}|{int(fast)}|{int(excl_taken)}"))
        elif typ == "FOLLOWUP":
            i = PARTNERS.index(action[1])
            self.pending = [(min(d, self.step_no + 1) if p == i else d, p, x) for d, p, x in self.pending]
        # deliver due replies
        due = [x for x in self.pending if x[0] <= self.step_no]
        self.pending = [x for x in self.pending if x[0] > self.step_no]
        for _, i, x in due:
            share, excl, fast, _ = (int(v) for v in x.split("|"))
            dec = self._decide(i, share, bool(excl), bool(fast))
            ev["reply"] = True
            self.replies += 1
            if dec == "COMMITTED":
                if sum(self.promised) + share > self.cap:
                    dec = "OBJECTION:share"
                else:
                    self.promised[i] = share
            if dec == "COMMITTED":
                self.status[i] = "COMMITTED"
                self.commit_steps.append(self.step_no)
                self.last_obj.pop(i, None)
                ev["commit"] = True
            elif dec == "DECLINED":
                self.status[i] = "DECLINED"
                ev["decline"] = True
            else:
                self.status[i] = "OBJECTION"
                self.last_obj[i] = dec.split(":")[1]
                ev["objection"] = True
        n = sum(s == "COMMITTED" for s in self.status)
        if n >= self.k or self.step_no >= self.max_steps:
            self.done = True
            self.success = bool(n >= self.k and sum(self.promised) <= self.cap)
        return StepResult(elapsed_h=self.rng.uniform(4, 72), events=ev, done=self.done,
                          success=self.success if self.done else None, valid=valid)

    # ------------------------------------------------------------------ task contract
    MILESTONES = ("researched", "reply", "commit_1", "commit_2")

    def contract(self):
        return Contract(clock="step", horizon=float(self.max_steps), resources=("share_pool",), constraints=(),
                        milestones=self.MILESTONES, failure_causes=("deadline",))

    def task_status(self):
        n = sum(s == "COMMITTED" for s in self.status)
        ms = {"researched": bool(self.known), "reply": self.replies > 0, "commit_1": n >= 1, "commit_2": n >= 2}
        return Status(progress=self.progress(), clock=float(self.step_no), time_left=max(0.0, 1 - self.step_no / self.max_steps),
                      resources={"share_pool": max(0, self.cap - sum(self.promised)) / self.cap}, milestones=ms,
                      costs={"share_promised": float(sum(self.promised)), "steps": float(self.step_no)})

    def hindsight_goals(self):
        """Fewer commitments than required (within the cap) verifiably achieved 'secure n commitments'
        at the step of the n-th commitment."""
        n = len(self.commit_steps)
        if not self.done or self.success or n < 1 or sum(self.promised) > self.cap:
            return []
        return [(["GOAL", "SECURE", nb(n, 0, 5), "COMMITMENTS", "CAP", nb(self.cap, 0, 20)], self.commit_steps[n - 1])]

    def hidden(self):
        return {"want": self.want, "min_share": self.min_share, "anchor": self.anchor, "rival": self.rival}


GRAMMAR = Grammar({"RESEARCH": [PARTNERS], "PITCH": [PARTNERS, SHARES, ["EXCL", "NOEXCL"], ["FAST", "SLOW"]],
                        "FOLLOWUP": [PARTNERS], "WAIT": []}, name="bizdev")
