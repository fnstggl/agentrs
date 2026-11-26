"""Kernel module `pop` (host `pop`): the autonomous population (contract section 2, MASTER_SPEC Part 3).

State: the column table `pop` (`schema.columns`), one row per individual, all classes together. Nothing is written
outside an applier: every cohort change is one `columns.batch` (a content-addressed blob of row indices and values
plus one hash-chained `cols.batch` event).

Activation is event-driven. Each individual has `next_active_at` (drawn from its own rhythm, inside its waking day)
and may be woken earlier by `wake(...)` (a message, a notification). The periodic process `pop.tick` (default every
15 simulated minutes) takes only the individuals that are due - a vectorised comparison, not a per-individual loop -
and for each decides whether to start a session (`models.Decision("session_start")`) and which registered activity
to spend it on (learned habit `habit[slot]` + goal affinity + Gumbel noise). Activities are registered by modules
with `register_activity(world, name, handler, owner=...)`; a handler receives the cohort (`handler(ctx, idx, now)`)
and may return the utility each individual experienced (float array), which drives satisfaction and habit learning
(an activity that pays off above the individual's running satisfaction becomes more habitual, one that disappoints
less so). The daily process `pop.daily` resets attention of those who spent any, applies employment changes and
slow knowledge decay.

Money: `wallet_cents` is income received outside the ledger and not yet deposited; income accrues lazily
(`income_cents` per day since `income_day`). Before an individual pays anything, `money.deposit` materialises it as a
principal with a bank account (`ids.ensure_principal(..., account=True)`) and moves the accrued amount from the
exogenous economy (`acct_external`) into that account; every payment is then a real `bank` move.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..substrate import attention, bus, columns, ids
from . import models, profiles, schema, semantics
from .schema import NEVER, TABLE

DAY = 86400
MAX_INIT_CHUNK = 250_000
SEGMENTS = ("casual", "enthusiast", "shopper", "social")
DEFAULT_MIX = {"casual": 0.45, "enthusiast": 0.2, "shopper": 0.15, "social": 0.2}
N_EMPLOYERS = 200


# ============================================================================================ public API (contract 2)
def _registry(world) -> dict:
    return world.runtime.setdefault("pop.activities", {"order": [], "spec": {}})


def register_activity(world, name: str, handler, *, owner: str, prior: float = 0.0, goals: tuple = ()) -> None:
    """Register an activity: `handler(ctx, idx, now)` is called with the cohort that chose `name` this tick (in
    `Module.install`). The handler may return a float array of experienced utilities (one per row of `idx`).
    `prior` is the mean initial habit, `goals` the goal codes (`schema.GOALS`) the activity serves."""
    import re
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,40}", name):
        raise ProtocolError(f"bad activity name {name!r}")
    reg = _registry(world)
    if name in reg["spec"]:
        if reg["spec"][name]["owner"] != owner:
            raise ProtocolError(f"activity {name} already registered by {reg['spec'][name]['owner']}")
        reg["spec"][name]["handler"] = handler
        return
    reg["order"].append(name)
    reg["spec"][name] = {"handler": handler, "owner": owner, "slot": len(reg["order"]) - 1, "prior": float(prior),
                         "goals": tuple(int(g) for g in goals)}


def activities(world) -> list[str]:
    return list(_registry(world)["order"])


def activity_slot(world, name: str) -> int:
    return _registry(world)["spec"][name]["slot"]


def wake(ctx: Ctx, idx: np.ndarray, at: int, activity: str) -> None:
    """Schedule a triggered activation of `idx` at time `at` for `activity` (a message, a notification). An earlier
    pending wake-up is kept. One recorded batch."""
    reg = _registry(ctx.world)
    if activity not in reg["spec"]:
        raise InvalidAction(f"unknown activity {activity!r}")
    ix = np.unique(np.asarray(idx, np.int64))
    if ix.size == 0:
        return
    cur = columns.col(ctx.world, TABLE, "wake_at")[ix]
    at = max(int(at), ctx.now)
    sel = ix[cur > at]
    if sel.size == 0:
        return
    columns.batch(ctx, TABLE, "pop.wake", sel, {"wake_at": np.full(sel.size, at, np.int64),
                                                "wake_act": np.full(sel.size, reg["spec"][activity]["slot"], np.int16)},
                  extra={"activity": activity, "at": at})


def class_members(world, cls: int) -> np.ndarray:
    """Row indices of the individuals of participant class `cls` (1 behavioural, 2 adaptive, 3 strategic,
    4 semantic)."""
    return np.flatnonzero(columns.col(world, TABLE, "class") == int(cls))


def n_individuals(world) -> int:
    return columns.rows(world, TABLE)


FEATURE_DOC = {
    "interests": "float32[n, semantics.DIM] unit interest vector in the shared space",
    "interest_coords": "float32[n, D] the same interests as coordinates of the space's topic subspace",
    "knowledge": "float32[n, D] accumulated topic knowledge (0..1 per dimension)",
    "prefs": "float32[n, P] preferences, names in schema.PREF_NAMES",
    "skills": "float32[n, S] skills, names in schema.SKILL_NAMES",
    "beliefs": "float32[n, B] beliefs, names in schema.BELIEF_NAMES",
    "goals": "uint8[n, G] goal codes (schema.GOALS)",
    "habit": "float32[n, H] activity habits (slots: population.activities(world))",
    "satisfaction": "float32[n] overall satisfaction",
    "attention_left_min": "float32[n] attention minutes left today",
    "attention_budget_min": "float32[n] daily attention budget",
    "attention_ratio": "float32[n] left / budget",
    "wallet_cents": "int64[n] spendable money: undeposited + accrued income + bank balance when materialised",
    "experience": "float32[n] cumulative sessions",
    "n_sessions": "int32[n] sessions so far",
    "last_activity": "int16[n] slot of the last activity (-1 none)",
    "employed": "bool[n] has an employer",
    "class": "uint8[n] participant class",
    "policy_group": "int16[n] policy group (classes 2-4)",
    "hour": "float32[n] local hour of day now",
    "contacts": "int32[n, C] social contacts (row indices, -1 none)",
    "recent": "int32[n, R] recently visited resources (res rows, -1 none)",
}


def features(world, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Documented individual features for behaviour models (`FEATURE_DOC`). Read-only copies."""
    ix = np.asarray(idx, np.int64)
    c = lambda name: columns.col(world, TABLE, name)[ix]  # noqa: E731
    bud = c("attention_budget_min").astype(np.float32)
    left = c("attention_left_min").astype(np.float32)
    out = {
        "interests": semantics.lift(c("interests")), "interest_coords": c("interests").astype(np.float32),
        "knowledge": c("knowledge").astype(np.float32),
        "prefs": c("prefs").astype(np.float32), "skills": c("skills").astype(np.float32),
        "beliefs": c("beliefs").astype(np.float32), "goals": c("goals").copy(), "habit": c("habit").copy(),
        "satisfaction": c("satisfaction").copy(), "attention_left_min": left, "attention_budget_min": bud,
        "attention_ratio": left / np.maximum(bud, 1e-3), "wallet_cents": spendable(world, ix),
        "experience": c("experience").copy(), "n_sessions": c("n_sessions").copy(),
        "last_activity": c("last_activity").copy(), "employed": c("employer") >= 0, "class": c("class").copy(),
        "policy_group": c("policy_group").copy(),
        "hour": np.full(ix.size, (world.now % DAY) / 3600.0, np.float32),
        "contacts": c("contacts").copy(), "recent": c("recent").copy(),
    }
    return out


def accrued(world, idx: np.ndarray) -> np.ndarray:
    """Income earned since the last deposit (lazy daily accrual), int64[n]."""
    ix = np.asarray(idx, np.int64)
    day = world.now // DAY
    inc = columns.col(world, TABLE, "income_cents")[ix].astype(np.int64)
    since = columns.col(world, TABLE, "income_day")[ix].astype(np.int64)
    return inc * np.maximum(0, day - since)


def spendable(world, idx: np.ndarray) -> np.ndarray:
    """Money an individual can spend: undeposited wallet + accrued income + bank balance (if it has an account)."""
    ix = np.asarray(idx, np.int64)
    out = columns.col(world, TABLE, "wallet_cents")[ix].astype(np.int64) + accrued(world, ix)
    bank = world.modules.get("bank")
    if bank is not None:
        st = world.state
        for j, i in enumerate(ix):
            p = f"p{int(i)}"
            if p in st.tables["principals"]:
                a = bank.primary_account(st, p)
                if a is not None:
                    out[j] += bank.balance(st, a)
    return out


# ============================================================================================ events
bus.event_type("pop.genesis", S.obj({"profile": {"type": "string"}, "counts": {"type": "object", "additionalProperties": True},
                                     "n": {"type": "integer"}, "dims": {"type": "object", "additionalProperties": True},
                                     "models": {"type": "object", "additionalProperties": True}, "activities": {"type": "array"}},
                                    ["profile", "counts", "n", "dims"]), owner="pop",
               doc="the population of a world: profile, class counts, dimensions, decision models in use")
bus.event_type("pop.policy_update", S.obj({
    "group": {"type": "string"}, "step": {"type": "integer", "minimum": 0}, "loss": {"type": "number"},
    "weights_hash": {"type": "string", "pattern": "[0-9a-f]{64}"}, "n_examples": {"type": "integer", "minimum": 0},
    "prev_hash": {"type": "string"}, "params": {"type": "integer"}, "examples": {"type": "array"},
    "checkpoint": {"type": ["string", "null"]}, "loss_before": {"type": "number"}, "arch": {"type": "string"},
    "optimizer_steps": {"type": "integer"}, "data": {"type": "object", "additionalProperties": True}},
    ["group", "step", "loss", "weights_hash", "n_examples"]), owner="pop",
    doc="a neural policy group's parameters changed by learning from its participants' experience")


bus.event_type("pop.decision_model", S.obj({"key": {"type": "string"}, "model_id": {"type": "string"},
                                            "model_version": {"type": "string"}, "fidelity_id": {"type": "string"},
                                            "kind": {"type": "string"}},
                                           ["key", "model_id", "model_version", "fidelity_id"]), owner="pop",
               doc="a decision model was used for the first time in this world (decision provenance registry)")


# ============================================================================================ the module
@register_module
class Population(Module):
    """Persistent individuals, profiles, activation scheduling, attention and time, habits, satisfaction and the
    class 1 decision models. Config: {profile, n?, interest_dim?, classes?, income_per_day_cents?: [lo, hi],
    seed_mix?: {segment: weight}, tick_s?: 900, adaptive_window_s?: 7200, dims?: {...}, calibration?: bool}."""
    name = "pop"
    host = "pop"
    version = "1"
    depends = ("artifacts", "cols")

    def install(self, world):
        r = profiles.resolve(self.config)
        self.prof = r
        self.d = schema.dims(r["dims"])
        self.counts = r["counts"]
        self.tick_s = int(self.config.get("tick_s", 900))
        if self.tick_s < 60 or DAY % self.tick_s:
            raise ProtocolError("tick_s must divide a day and be >= 60 s")
        self.window = int(self.config.get("adaptive_window_s", 7200))
        cols = schema.columns(self.d)
        cols["income_day"] = ("int32", (), 0)
        columns.declare(world, TABLE, r["n"], cols)
        st = world.state
        st.declare_table("pop_policies")
        st.declare_table("pop_policy_log", index=("group",))
        st.declare_table("pop_decision_models")
        world.runtime["pop.models"] = models.build(self.config)
        _registry(world)
        bus.enforce(world, owner="pop")
        if r["init"] == "derived":
            # the initial state is a pure function of the world spec (like declared fill values): computed here, so
            # creation, replay and regeneration start from the same arrays without a multi-GB genesis blob
            for lo in range(0, r["n"], MAX_INIT_CHUNK):
                hi = min(r["n"], lo + MAX_INIT_CHUNK)
                for k, v in self._initial(world, lo, hi, now=0, derived=True).items():
                    st.arrays[f"{TABLE}.{k}"][lo:hi] = v

    # ------------------------------------------------------------------ genesis and initial state
    def genesis(self, ctx):
        w = ctx.world
        n = self.prof["n"]
        if self.prof["init"] == "batch":
            for lo in range(0, n, MAX_INIT_CHUNK):
                hi = min(n, lo + MAX_INIT_CHUNK)
                vals = self._initial(w, lo, hi, now=ctx.now)
                columns.batch(ctx, TABLE, "pop.init", np.arange(lo, hi), vals, mode="set",
                              extra={"rows": [lo, hi]})
        if n:
            attention.reset(ctx)
        first = (ctx.now // self.tick_s + 1) * self.tick_s
        ctx.schedule(first, "pop.tick", {}, prio=40, period=self.tick_s, key="pop.tick")
        ctx.schedule((ctx.now // DAY + 1) * DAY, "pop.daily", {}, prio=30, period=DAY, key="pop.daily")
        bus.emit(ctx, "pop.genesis", {"profile": self.prof["profile"], "counts": dict(self.counts), "n": int(n),
                                      "dims": dict(self.d), "models": models.status(w),
                                      "activities": activities(w)}, vis="internal")

    def _initial(self, world, lo: int, hi: int, now: int, derived: bool = False) -> dict[str, np.ndarray]:
        """Heterogeneous initial state of rows [lo, hi): a deterministic function of the world seed."""
        d = self.d
        m = hi - lo
        rng = world.rng_for("pop-init", lo, hi)
        cnt = self.counts
        rows = np.arange(lo, hi)
        b1, b2, b3 = cnt["lightweight"], cnt["lightweight"] + cnt["adaptive"], \
            cnt["lightweight"] + cnt["adaptive"] + cnt["strategic"]
        cls = np.where(rows < b1, 1, np.where(rows < b2, 2, np.where(rows < b3, 3, 4))).astype(np.uint8)
        mix = dict(self.config.get("seed_mix") or DEFAULT_MIX)
        wts = np.array([float(mix.get(s, 0.0)) for s in SEGMENTS])
        if wts.sum() <= 0:
            raise ProtocolError("seed_mix needs a positive weight")
        seg = rng.choice(len(SEGMENTS), size=m, p=wts / wts.sum()).astype(np.uint8)
        # interests: unit coordinates in the shared space's topic subspace, drawn from the embeddings of a real topic
        # corpus (MIND-small news subcategories; population.semantics, D-16)
        interests = semantics.interest_prior_coords(rng, m, d["D"])
        # preferences (schema.PREF_NAMES), segment-shifted
        P = d["P"]
        prefs = rng.beta(2.0, 2.0, size=(m, P))
        shift = {0: {}, 1: {1: 0.25, 3: 0.2, 6: -0.05}, 2: {0: -0.1, 7: 0.25, 5: 0.1}, 3: {6: 0.3, 4: 0.2}}
        for s, sh in shift.items():
            mask = seg == s
            for j, v in sh.items():
                if j < P:
                    prefs[mask, j] = np.clip(prefs[mask, j] + v, 0.02, 0.98)
        knowledge = np.clip(np.abs(interests) * rng.uniform(0.0, 0.5, (m, 1)), 0, 1)
        G = d["G"]
        primary = np.select([seg == 1, seg == 2, seg == 3], [1, 2, 3], 4)
        goals = np.zeros((m, G), np.uint8)
        goals[:, 0] = primary
        if G > 1:
            goals[:, 1] = rng.integers(1, 5, m)
        skills = np.clip(rng.beta(2.5, 2.5, (m, d["S"])), 0.02, 0.98)
        beliefs = np.clip(0.5 + rng.normal(0, 0.12, (m, d["B"])), 0.05, 0.95)
        lo_inc, hi_inc = (self.config.get("income_per_day_cents") or [3_000, 30_000])
        employed = rng.random(m) < 0.68
        income = np.exp(rng.uniform(math.log(lo_inc), math.log(hi_inc), m))
        income = np.where(employed, income, income * 0.35).astype(np.int32)
        employer = np.where(employed, rng.integers(0, N_EMPLOYERS, m), -1).astype(np.int32)
        commit_start = np.where(employed, np.clip(rng.normal(9 * 60, 50, m), 6 * 60, 13 * 60), 0).astype(np.int16)
        commit_len = np.where(employed, np.clip(rng.normal(8 * 60, 40, m), 4 * 60, 10 * 60), 0).astype(np.int16)
        day_start = np.clip(rng.normal(7 * 60, 55, m), 4 * 60, 11 * 60).astype(np.int16)
        day_len = np.clip(rng.normal(16 * 60, 50, m), 13 * 60, 18 * 60).astype(np.int16)
        heavy = (seg == 1) | (seg == 3)
        mean_gap = np.exp(rng.normal(np.where(heavy, math.log(7 * 3600), math.log(13 * 3600)), 0.45)).astype(np.float32)
        session_min = np.exp(rng.normal(math.log(18.0), 0.45, m)).astype(np.float32)
        budget = np.exp(rng.normal(np.where(heavy, math.log(120.0), math.log(75.0)), 0.35)).astype(np.float32)
        # first activation: uniform over the first rhythm period, moved into the individual's waking day
        nxt = schedule_times(now, rng.uniform(0.05, 1.0, m) * mean_gap, day_start, day_len, rng)
        # habits: every slot gets individual noise; at genesis (batch mode) the activities registered by then add
        # their prior and goal affinity (in derived mode they are not known yet when `pop` installs)
        reg = _registry(world)
        habit = rng.normal(0.3, 0.5, (m, d["H"])).astype(np.float32)
        if not derived:
            for name in reg["order"]:
                sp = reg["spec"][name]
                if sp["slot"] < d["H"]:
                    bonus = np.zeros(m)
                    for g in sp["goals"]:
                        bonus += 0.4 * (goals == g).any(axis=1)
                    habit[:, sp["slot"]] += (sp["prior"] - 0.3 + bonus).astype(np.float32)
        # social contacts: small world (ring lattice over all rows + 10% random rewiring)
        n = self.prof["n"]
        C = d["C"]
        offs = np.array([o for k in range(1, C // 2 + 1) for o in (k, -k)] + ([C // 2 + 1] if C % 2 else []))[:C]
        contacts = (rows[:, None] + offs[None, :]) % max(n, 1)
        rew = rng.random((m, C)) < 0.10
        contacts = np.where(rew, rng.integers(0, max(n, 1), (m, C)), contacts).astype(np.int32)
        contacts[contacts == rows[:, None]] = -1
        if n <= 1:
            contacts[:] = -1
        trust = np.clip(0.5 + rng.normal(0, 0.12, (m, C)), 0.05, 0.95)
        wallet = (income.astype(np.int64) * rng.integers(2, 12, m)).astype(np.int64)
        policy_group = np.where(cls == 2, 0, np.where(cls == 3, 100 + (rows - b2) % 2, np.where(cls == 4, 200, -1)))
        out = {
            "class": cls, "policy_group": policy_group.astype(np.int16), "segment": seg,
            "interests": interests.astype(np.float16), "prefs": prefs.astype(np.float16),
            "knowledge": knowledge.astype(np.float16), "goals": goals, "skills": skills.astype(np.float16),
            "beliefs": beliefs.astype(np.float16), "experience": rng.exponential(20.0, m).astype(np.float32),
            "wallet_cents": wallet, "income_cents": income, "income_day": np.full(m, now // DAY, np.int32),
            "wtp_scale": np.exp(rng.normal(0, 0.4, m)).astype(np.float16), "employer": employer,
            "commit_start_min": commit_start, "commit_len_min": commit_len, "day_start_min": day_start,
            "day_len_min": day_len, "mean_gap_s": mean_gap, "session_min": session_min,
            "next_active_at": np.where(cls <= 2, nxt, NEVER).astype(np.int64), "habit": habit,
            "contacts": contacts, "contact_trust": trust.astype(np.float16),
            "attention_budget_min": budget, "attention_left_min": budget,
        }
        return out

    # ------------------------------------------------------------------ the tick: activations and sessions
    @process("tick")
    def pr_tick(self, ctx):
        w = ctx.world
        now = ctx.now
        if not columns.rows(w, TABLE):
            return
        nxt = columns.col(w, TABLE, "next_active_at")
        wk = columns.col(w, TABLE, "wake_at")
        cls = columns.col(w, TABLE, "class")
        due = np.flatnonzero(((nxt <= now) | (wk <= now)) & (cls <= 2))
        if due.size and self.window > self.tick_s and now % self.window:
            # event aggregation: adaptive participants that are due wait for the next decision round, so their
            # shared policy decides for them in one batch (MASTER_SPEC 13.2/13.3)
            due = due[cls[due] != 2]
        if due.size == 0:
            return
        self.run_cohort(ctx, due, now)

    def run_cohort(self, ctx: Ctx, due: np.ndarray, now: int) -> dict:
        """Activate `due`: session decision, activity choice, dispatch, then the recorded session batch."""
        w = ctx.world
        reg = _registry(w)
        ds = w.runtime["pop.models"]
        c = lambda name: columns.col(w, TABLE, name)[due]  # noqa: E731
        bud = c("attention_budget_min").astype(np.float64)
        left = c("attention_left_min").astype(np.float64)
        habit = c("habit").astype(np.float64)
        prefs = c("prefs").astype(np.float64)
        minute = (now % DAY) // 60
        cs, cl = c("commit_start_min").astype(np.int64), c("commit_len_min").astype(np.int64)
        in_commit = (cl > 0) & (((minute - cs) % 1440) < cl)
        woken = c("wake_at") <= now
        wact = c("wake_act").astype(np.int64)
        hour = minute / 60.0
        slots = [reg["spec"][a]["slot"] for a in reg["order"]]
        F = {"attention_ratio": left / np.maximum(bud, 1e-3), "satisfaction": c("satisfaction").astype(np.float64),
             "habit_max": habit[:, slots].max(axis=1) if slots else np.zeros(due.size),
             "experience_log": np.log1p(c("experience").astype(np.float64)), "in_commitment": in_commit.astype(float),
             "woken": woken.astype(float), "hour_sin": np.full(due.size, math.sin(2 * math.pi * hour / 24)),
             "hour_cos": np.full(due.size, math.cos(2 * math.pi * hour / 24))}
        rng = ctx.rng("pop.tick")
        start = ds["session_start"].sample(F, rng) & (left > 0.5) & bool(slots)
        chosen = np.full(due.size, -1, np.int64)
        st_ix = np.flatnonzero(start)
        if st_ix.size:
            goals = c("goals")[st_ix]
            logits = np.zeros((st_ix.size, len(slots)))
            for j, a in enumerate(reg["order"]):
                sp = reg["spec"][a]
                logits[:, j] = habit[st_ix, sp["slot"]]
                for g in sp["goals"]:
                    logits[:, j] += 0.5 * (goals == g).any(axis=1)
            logits += rng.gumbel(size=logits.shape) * (0.6 + 0.6 * prefs[st_ix, 1:2])
            pick = np.asarray(slots)[np.argmax(logits, axis=1)]
            forced = woken[st_ix] & np.isin(wact[st_ix], slots)
            pick = np.where(forced, wact[st_ix], pick)
            chosen[st_ix] = pick
        util = np.zeros(due.size, np.float64)
        by_act = {}
        decided: list[str] = []
        w.runtime["pop.tick.decisions"] = decided       # record_decisions lists its batches here (provenance)
        try:
            for a in reg["order"]:
                sp = reg["spec"][a]
                sel = np.flatnonzero(chosen == sp["slot"])
                if sel.size == 0:
                    continue
                by_act[a] = int(sel.size)
                out = sp["handler"](ctx, due[sel], now)
                if out is not None:
                    u = np.asarray(out, np.float64).reshape(-1)
                    if u.shape[0] != sel.size or not np.all(np.isfinite(u)):
                        raise ProtocolError(f"activity {a} returned {u.shape[0]} utilities for {sel.size} "
                                            f"individuals")
                    util[sel] = np.clip(u, -2.0, 2.0)
        finally:
            w.runtime.pop("pop.tick.decisions", None)
        # ---- recorded session outcome for every activated individual (re-read: handlers changed state)
        sat = c("satisfaction").astype(np.float64)
        habit = c("habit").astype(np.float32)
        last_act = c("last_activity").astype(np.int64)
        n_sess = c("n_sessions").astype(np.int64)
        n_sw = c("n_switches").astype(np.int64)
        exp_ = c("experience").astype(np.float64)
        last_u = c("last_utility").astype(np.float64)
        last_at = c("last_session_at").astype(np.int64)
        had = chosen >= 0
        rows = np.flatnonzero(had)
        lr = 0.15 + 0.25 * prefs[:, 1]           # novelty seekers update habits faster
        if rows.size:
            sl = chosen[rows]
            h = habit[rows, sl].astype(np.float64)
            h_new = np.clip(h + lr[rows] * (util[rows] - sat[rows]), -4.0, 4.0)
            habit[rows] *= np.float32(0.995)       # unused habits fade slowly
            habit[rows, sl] = h_new.astype(np.float32)
            loyal = prefs[rows, 5]
            beta = 0.25 - 0.12 * loyal
            sat[rows] = (1 - beta) * sat[rows] + beta * util[rows]
            n_sw[rows] += (last_act[rows] >= 0) & (last_act[rows] != sl)
            last_act[rows] = sl
            n_sess[rows] += 1
            exp_[rows] += 1.0
            last_u[rows] = util[rows]
            last_at[rows] = now
        gaps = c("mean_gap_s").astype(np.float64) * rng.exponential(1.0, due.size) * \
            np.where(had, 1.0, 0.5) / (0.6 + 0.8 * np.clip(sat, 0, 1))
        nxt = schedule_times(now, np.clip(gaps, 600, 3 * DAY), c("day_start_min"), c("day_len_min"), rng)
        left_plat = int(np.sum(had & (habit[np.arange(due.size), np.maximum(chosen, 0)] < -1.0)))
        vals = {"next_active_at": nxt, "wake_at": np.where(woken, NEVER, c("wake_at")).astype(np.int64),
                "wake_act": np.where(woken, -1, wact).astype(np.int16), "satisfaction": sat.astype(np.float32),
                "habit": habit, "n_sessions": n_sess.astype(np.int32), "n_switches": n_sw.astype(np.int32),
                "last_activity": last_act.astype(np.int16), "experience": exp_.astype(np.float32),
                "last_utility": last_u.astype(np.float32), "last_session_at": last_at}
        columns.batch(ctx, TABLE, "pop.session", due, vals, mode="set",
                      extra={"t": int(now), "activations": int(due.size), "sessions": int(rows.size),
                             "woken": int(woken.sum()), "by_activity": by_act, "left_platform": left_plat,
                             "switches": int(np.sum(n_sw - c("n_switches"))),
                             "mean_utility": float(util[rows].mean()) if rows.size else 0.0,
                             "decisions": decided})
        return {"activations": int(due.size), "sessions": int(rows.size), "by_activity": by_act}

    # ------------------------------------------------------------------ daily
    @process("daily")
    def pr_daily(self, ctx):
        w = ctx.world
        n = columns.rows(w, TABLE)
        if not n:
            return
        spent = columns.col(w, TABLE, "attention_spent_min")
        who = np.flatnonzero(spent > 0)
        if who.size:
            attention.reset(ctx, who)
        rng = ctx.rng("pop.daily")
        # employment: rare job loss and job finding (income and the daily commitment window follow)
        emp = columns.col(w, TABLE, "employer")
        u = rng.random(n)
        lose = (emp >= 0) & (u < 0.002)
        find = (emp < 0) & (u < 0.02)
        ch = np.flatnonzero(lose | find)
        if ch.size:
            inc = columns.col(w, TABLE, "income_cents")[ch].astype(np.int64)
            gets = find[ch]
            pend = accrued(w, ch)
            new_emp = np.where(gets, rng.integers(0, N_EMPLOYERS, ch.size), -1).astype(np.int32)
            new_inc = np.where(gets, inc / 0.35, inc * 0.35).astype(np.int32)
            columns.batch(ctx, TABLE, "pop.employment", ch, {
                "employer": new_emp, "income_cents": new_inc,
                # income earned so far is kept at the old rate before the rate changes
                "wallet_cents": columns.col(w, TABLE, "wallet_cents")[ch] + pend,
                "income_day": np.full(ch.size, ctx.now // DAY, np.int32),
                "commit_start_min": np.where(gets, np.clip(rng.normal(9 * 60, 50, ch.size), 360, 780), 0).astype(np.int16),
                "commit_len_min": np.where(gets, np.clip(rng.normal(480, 40, ch.size), 240, 600), 0).astype(np.int16)},
                extra={"lost": int(lose.sum()), "found": int(find.sum())})
        # knowledge fades a little for those who were active (others' knowledge is read with its age)
        last = columns.col(w, TABLE, "last_session_at")
        act = np.flatnonzero(last >= ctx.now - DAY)
        if act.size:
            k = columns.col(w, TABLE, "knowledge")[act].astype(np.float32) * np.float32(0.985)
            goals = columns.col(w, TABLE, "goals")[act].copy()
            # goals drift: a secondary goal is re-drawn now and then
            if goals.shape[1] > 1:
                redraw = rng.random(act.size) < 0.04
                goals[redraw, 1] = rng.integers(1, 5, int(redraw.sum()))
            columns.batch(ctx, TABLE, "pop.daily", act, {"knowledge": k.astype(np.float16), "goals": goals},
                          extra={"day": int(ctx.now // DAY)})

    # ------------------------------------------------------------------ learning records (classes 2 and 3)
    @applier("pop.policy_update")
    def ap_policy_update(self, ev):
        p = ev.payload
        st = self.world.state
        rec = {"group": p["group"], "step": p["step"], "loss": p["loss"], "weights_hash": p["weights_hash"],
               "n_examples": p["n_examples"], "t": ev.t, "checkpoint": p.get("checkpoint"),
               "prev_hash": p.get("prev_hash"), "params": p.get("params"), "arch": p.get("arch"),
               "examples": list(p.get("examples") or []), "loss_before": p.get("loss_before"),
               "optimizer_steps": p.get("optimizer_steps"), "data": dict(p.get("data") or {})}
        st.put("pop_policy_log", f"{p['group']}|{p['step']:08d}", rec)
        old = st.get("pop_policies", p["group"])
        ck = p.get("checkpoint") or (old or {}).get("last_checkpoint")
        cks = list((old or {}).get("checkpoints") or [])
        if p.get("checkpoint"):
            cks.append([p["step"], p["checkpoint"], p["weights_hash"]])
        st.put("pop_policies", p["group"], {"group": p["group"], "step": p["step"], "weights_hash": p["weights_hash"],
                                            "t": ev.t, "last_checkpoint": ck, "checkpoints": cks,
                                            "updates": (old or {}).get("updates", 0) + 1})

    @applier("pop.decision_model")
    def ap_decision_model(self, ev):
        p = ev.payload
        self.world.state.insert("pop_decision_models", p["key"], {
            "model_id": p["model_id"], "model_version": p["model_version"], "fidelity_id": p["fidelity_id"],
            "kind": p.get("kind", ""), "first_t": ev.t})

    @applier("pop.genesis")
    def ap_genesis(self, ev):
        self.world.state.meta_set("pop.profile", ev.payload["profile"])

    # ------------------------------------------------------------------ the individual's own view
    @endpoint("me", S.obj(), read_only=True, cost_s=0)
    def ep_me(self, ctx):
        """Your own state as a member of the population (only individuals of the population have one)."""
        i = ids.index(ctx.actor)
        if i is None or i >= columns.rows(ctx.world, TABLE):
            raise NotFound("you are not a member of the population")
        f = features(ctx.world, np.array([i]))
        return {"individual": ctx.actor, "satisfaction": round(float(f["satisfaction"][0]), 4),
                "attention_left_min": round(float(f["attention_left_min"][0]), 2),
                "wallet_cents": int(f["wallet_cents"][0]), "goals": [schema.GOALS.get(int(g), "?") for g in f["goals"][0]],
                "employed": bool(f["employed"][0]), "sessions": int(f["n_sessions"][0]),
                "class": schema.CLASSES.get(int(f["class"][0]), "?")}

    def check_invariants(self, world):
        errs = []
        try:
            n = columns.rows(world, TABLE)
        except NotFound:
            return ["pop table missing"]
        if n != self.prof["n"]:
            errs.append(f"pop has {n} rows, profile says {self.prof['n']}")
        if n:
            wal = columns.col(world, TABLE, "wallet_cents")
            if int(wal.min()) < 0:
                errs.append("negative undeposited wallet")
            cls = columns.col(world, TABLE, "class")
            if not np.isin(cls, [1, 2, 3, 4]).all():
                errs.append("unknown participant class")
            kn = columns.col(world, TABLE, "known")
            res_rows = columns.rows(world, schema.RES_TABLE) if schema.RES_TABLE in columns.tables(world) else 0
            if kn.size and int(kn.max()) >= res_rows:
                errs.append("known resource index beyond the resource table")
        return errs


def schedule_times(now: int, gaps: np.ndarray, day_start: np.ndarray, day_len: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """now + gap, moved forward to the start of the individual's waking day (+ up to 45 min) when it falls outside
    it. Vectorised; int64 seconds."""
    t = now + np.asarray(gaps, np.float64).astype(np.int64)
    ds = np.asarray(day_start, np.int64) * 60
    dl = np.asarray(day_len, np.int64) * 60
    tod = t % DAY
    since = (tod - ds) % DAY
    asleep = since >= dl
    push = (DAY - since) + rng.integers(0, 45 * 60, t.shape[0])
    return np.where(asleep, t + push, t).astype(np.int64)


def money_deposit(ctx: Ctx, i: int) -> str | None:
    """Materialise individual `i` with a bank account and deposit its undeposited wallet plus accrued income from the
    exogenous economy (`acct_external`, ledger kind "income"). Returns the account id (None without a bank)."""
    w = ctx.world
    bank = w.modules.get("bank")
    if bank is None:
        return None
    p = ids.ensure_principal(ctx, int(i), account=True)
    acct = bank.primary_account(ctx.state, p)
    ix = np.array([int(i)])
    amount = int(columns.col(w, TABLE, "wallet_cents")[i]) + int(accrued(w, ix)[0])
    if amount > 0:
        bank.move(ctx, "acct_external", acct, amount, "income", "income deposited")
    columns.batch(ctx, TABLE, "pop.deposit", ix, {"wallet_cents": np.zeros(1, np.int64),
                                                  "income_day": np.array([ctx.now // DAY], np.int32)},
                  extra={"amount": amount})
    return acct


def require_member(ctx: Ctx) -> int:
    i = ids.index(ctx.actor)
    if i is None:
        raise Forbidden("only members of the population")
    return i
