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

import contextlib
import math

import numpy as np

from ..core import schema as S
from ..core.errors import Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..substrate import attention, bus, columns, ids
from . import activation, models, profiles, schema, semantics
from .schema import NEVER, TABLE

DAY = 86400
MAX_INIT_CHUNK = 250_000
SEGMENTS = ("casual", "enthusiast", "shopper", "social")
DEFAULT_MIX = {"casual": 0.45, "enthusiast": 0.2, "shopper": 0.15, "social": 0.2}
N_EMPLOYERS = 200
# Triggered wake-ups are part of the population's public API (contract section 1: `wake`): any subsystem schedules
# them through `wake`, which keeps the earliest pending wake-up and checks the activity; no module writes the two
# columns directly except `pop` itself (which clears them when the wake-up fires).
WAKE_INTERFACE = "pop_wake"
columns.register_interface(WAKE_INTERFACE, ("wake_at", "wake_act"))


@contextlib.contextmanager
def running_as(ctx: Ctx, module: str | None):
    """Run a block as kernel module `module` for column write ownership (`substrate.columns`): an activity handler
    is its owner module's code even though the population's tick calls it, so it writes the owner's columns (and,
    within its package, its siblings'), never the population's own; the population's API (`experience`,
    `money_deposit`) writes the population's columns as `pop` on behalf of whichever module calls it."""
    prev = getattr(ctx, "module", None)
    if module is not None and module not in ctx.world.modules:
        module = prev
    ctx.module = module
    try:
        yield ctx
    finally:
        ctx.module = prev


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


def register_class_runtime(world, cls: int, handler, *, owner: str) -> None:
    """Bind a participant class (3 strategic, 4 semantic) to a runtime of its own. Binding contract (the class-4
    semantic participant runtime of workstream D, `crucible.semantic.generalists`, and class 3, `pop_strategic`):

    * Rows. The class's rows of the population table ARE the participants: identity (`ids.pid(i)`, principal
      `p<i>`, materialised with `ids.ensure_principal`), interests (`interests` / `interest_coords`), preferences,
      skills, beliefs, goals, money (`spendable`, `pay`, `money_deposit`), contacts, habits and history (`features`)
      live there like every other individual's; a runtime keeps only its own mind state (its own state table) and
      never a second copy of a per-person fact.
    * Activation. Call in `Module.install` (before genesis) so the class's rows get a first activation time. The
      population's tick hands the due rows (`next_active_at` or a pending `wake` reached) to
      `handler(ctx, idx, now) -> next_active_at` (int64 per row; None = now + 6 h), running as `owner`.
    * Session and provenance. The call runs inside a population session: every `record_decisions` batch the handler
      records (on the population table or a relation table) is listed by the session batch `pop.session` that the
      tick then records for `idx` (kind `pop.session`, provenance model `class<c>.<owner>`, fidelity
      `pop.decision.class_runtime`, `extra.class`), so `check_provenance` counts those rows as organic. A runtime's
      tool calls (`ctx.call(..., as_principal=p<i>)`) are the participant's own actions.
    * Attention. Time a participant spends is drawn from its daily budget with `substrate.attention.spend`.
    * Learning from outcomes. `experience(ctx, idx, activity, utility, item_vecs=...)` updates interests,
      knowledge, satisfaction and habits from realised outcomes, as for every individual."""
    rt = world.runtime.setdefault("pop.class_runtimes", {})
    if int(cls) in rt and rt[int(cls)]["owner"] != owner:
        raise ProtocolError(f"class {cls} is bound to {rt[int(cls)]['owner']}")
    rt[int(cls)] = {"handler": handler, "owner": owner}


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
                  extra={"activity": activity, "at": at}, _iface=WAKE_INTERFACE)


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


def session_budget(world, idx: np.ndarray) -> np.ndarray:
    """Minutes the current session of individuals `idx` may still take (the population's session-length decision,
    `pop.session_length`, capped by the attention left today). An activity handler spends at most this much of the
    session (float32[n]); outside a session it is the attention left."""
    ix = np.asarray(idx, np.int64)
    left = columns.col(world, TABLE, "attention_left_min")[ix].astype(np.float32)
    if world.runtime.get("pop.tick.decisions") is None:
        return left
    return np.minimum(columns.col(world, TABLE, "session_budget_min")[ix].astype(np.float32), left)


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
    def check_habit_slots(self, world) -> None:
        """Every registered activity needs a habit slot (column `habit`, H = dims.H): a world that registers more
        activities than the profile has slots fails at creation instead of failing every tick."""
        k = len(_registry(world)["order"])
        if k > self.d["H"]:
            raise ProtocolError(f"pop: {k} activities are registered ({', '.join(_registry(world)['order'])}) but the "
                                f"population has {self.d['H']} habit slots; configure the pop module with "
                                f"dims: {{H: >= {k}}}")

    def genesis(self, ctx):
        w = ctx.world
        n = self.prof["n"]
        self.check_habit_slots(w)
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
        # first activation: the start of the individual's waking day (now, if it is awake), where it takes its
        # first daily decision (population.activation)
        awake_left = activation.waking_left_s(now, day_start, day_len)
        nxt = np.where(awake_left > 0, now + rng.integers(60, 3600, m),
                       activation.next_wake_s(now, day_start, day_len, rng)).astype(np.int64)
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
            "next_active_at": np.where((cls <= 2) | np.isin(cls, list(world.runtime.get("pop.class_runtimes", {}))),
                                       nxt, NEVER).astype(np.int64), "habit": habit,
            "contacts": contacts, "contact_trust": trust.astype(np.float16),
            "attention_budget_min": budget, "attention_left_min": budget,
        }
        # a 7-day history before the first simulated day (otherwise the fitted return model sees dormant users)
        out.update(activation.genesis_history(rng, seg, session_min, int(now // DAY)))
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
        runtimes = w.runtime.get("pop.class_runtimes") or {}
        for c_, rt in sorted(runtimes.items()):
            if c_ <= 2:
                continue
            bound = np.flatnonzero(((nxt <= now) | (wk <= now)) & (cls == c_))
            if bound.size:
                decided: list[str] = []
                w.runtime["pop.tick.decisions"] = decided
                try:
                    with running_as(ctx, rt["owner"]):
                        nx = rt["handler"](ctx, bound, now)
                finally:
                    w.runtime.pop("pop.tick.decisions", None)
                nx = np.full(bound.size, now + 6 * 3600, np.int64) if nx is None else np.asarray(nx, np.int64)
                from .api import record_decisions
                with running_as(ctx, "pop"):
                    record_decisions(ctx, "pop.session", bound, {
                        "next_active_at": np.maximum(nx, now + 60), "wake_at": np.full(bound.size, NEVER, np.int64),
                        "last_session_at": np.full(bound.size, now, np.int64),
                        "n_sessions": columns.col(w, TABLE, "n_sessions")[bound] + 1},
                        model_id=f"class{c_}.{rt['owner']}", model_version="1",
                        fidelity_id="pop.decision.class_runtime",
                        extra={"t": int(now), "activations": int(bound.size), "sessions": int(bound.size), "woken": 0,
                               "by_activity": {f"class{c_}.{rt['owner']}": int(bound.size)}, "left_platform": 0,
                               "switches": 0, "mean_utility": 0.0, "decisions": decided, "class": int(c_)})
        due = np.flatnonzero(((nxt <= now) | (wk <= now)) & (cls <= 2))
        if due.size and self.window > self.tick_s and now % self.window:
            # event aggregation: adaptive participants that are due wait for the next decision round, so their
            # shared policy decides for them in one batch (MASTER_SPEC 13.2/13.3)
            due = due[cls[due] != 2]
        if due.size == 0:
            return
        self.run_cohort(ctx, due, now)

    def run_cohort(self, ctx: Ctx, due: np.ndarray, now: int) -> dict:
        """Activate `due`: the day's decisions at the first activation of a day (fitted daily_return@1 and
        extra_sessions@1, population.activation), who starts a session now, its length, the activity, dispatch to
        the activity's handler, then the recorded learning and scheduling batches. Every population-table batch of
        the cohort carries decision provenance and is listed by the cohort's `pop.session` batch."""
        from .api import record_decisions
        w = ctx.world
        reg = _registry(w)
        self.check_habit_slots(w)
        c = lambda name: columns.col(w, TABLE, name)[due]  # noqa: E731
        today = int(now // DAY)
        rng = ctx.rng("pop.tick")
        bud = c("attention_budget_min").astype(np.float64)
        left = c("attention_left_min").astype(np.float64)
        prefs = c("prefs").astype(np.float64)
        woken = c("wake_at") <= now
        wact = c("wake_act").astype(np.int64)
        slots = [reg["spec"][a]["slot"] for a in reg["order"]]
        decided: list[str] = []
        w.runtime["pop.tick.decisions"] = decided       # record_decisions lists its batches here (provenance)
        try:
            # ---- the day's decisions (first activation of the simulated day)
            day_left = c("day_left").astype(np.int64)
            new = c("act_day").astype(np.int64) != today
            nd = np.flatnonzero(new)
            first_at = np.full(due.size, -1, np.int64)
            if nd.size:
                active, planned, pv1, pv2 = activation.decide_day(ctx, due[nd], today, ctx.rng("pop.day"))
                tdn = c("td_n")[nd].astype(np.int64)
                vals = {"active": active, "act_day": np.full(nd.size, today, np.int32),
                        "sessions": planned.astype(np.int32), "day_left": planned.astype(np.int16),
                        # the last day with sessions is complete: it joins the observed history
                        "extra_obs": c("extra_obs")[nd] + np.maximum(tdn - 1, 0),
                        "days_obs": c("days_obs")[nd] + (tdn > 0), "sl_n": c("sl_n")[nd] + tdn,
                        "sl_log_sum": c("sl_log_sum")[nd] + c("td_log_sum")[nd],
                        "sl_items": c("sl_items")[nd] + c("td_items")[nd],
                        "td_n": np.zeros(nd.size, np.int16), "td_log_sum": np.zeros(nd.size, np.float32),
                        "td_items": np.zeros(nd.size, np.float32)}
                with running_as(ctx, "pop"):
                    record_decisions(ctx, "pop.day_active", due[nd], vals, **activation.provenance_args(pv1),
                                     extra={"t": int(now), "day": today, "active": int(active.sum()),
                                            **activation.served_extra(pv1)})
                    a = np.flatnonzero(active)
                    if a.size and pv2 is not None:
                        record_decisions(ctx, "pop.day_sessions", due[nd[a]], {"sessions": planned[a].astype(np.int32)},
                                         **activation.provenance_args(pv2),
                                         extra={"t": int(now), "day": today, "sessions": int(planned[a].sum()),
                                                **activation.served_extra(pv2)})
                day_left[nd] = planned
                # when the first planned session happens: the minimum of the planned uniform draws over the rest of
                # the waking day
                wl = activation.waking_left_s(now, c("day_start_min")[nd], c("day_len_min")[nd])
                gap = activation.next_session_gap_s(wl, planned, rng)
                first_at[nd] = np.where(np.isfinite(gap), now + np.minimum(gap, 2 * DAY), -1).astype(np.int64)
            # ---- who starts a session now
            planned_now = (day_left > 0) & ~new                         # a planned session's time has come
            first_now = new & (day_left > 0) & (first_at >= 0) & (first_at < now + self.tick_s)
            start = (woken | planned_now | first_now) & (left > 0.5) & bool(slots)
            st_ix = np.flatnonzero(start)
            sess_budget = np.zeros(due.size)
            if st_ix.size:
                sess_budget[st_ix] = activation.session_budget(ctx.rng("pop.duration"),
                                                               c("session_min")[st_ix], left[st_ix])
                with running_as(ctx, "pop"):
                    record_decisions(ctx, "pop.duration", due[st_ix],
                                     {"session_budget_min": sess_budget[st_ix].astype(np.float32)},
                                     model_id=activation.LENGTH_MODEL[0], model_version=activation.LENGTH_MODEL[1],
                                     fidelity_id="pop.decision.session_length", extra={"t": int(now)})
            # ---- which activity
            chosen = np.full(due.size, -1, np.int64)
            habit = c("habit").astype(np.float64)
            if st_ix.size:
                goals = c("goals")[st_ix]
                logits = np.zeros((st_ix.size, len(slots)))
                derived = self.prof["init"] == "derived"
                for j, a_ in enumerate(reg["order"]):
                    sp = reg["spec"][a_]
                    logits[:, j] = habit[st_ix, sp["slot"]]
                    if derived:            # derived initial state: the activities' priors were not known at install
                        logits[:, j] += sp["prior"] - 0.3
                    for g in sp["goals"]:
                        logits[:, j] += 0.5 * (goals == g).any(axis=1)
                logits += rng.gumbel(size=logits.shape) * (0.6 + 0.6 * prefs[st_ix, 1:2])
                pick = np.asarray(slots)[np.argmax(logits, axis=1)]
                ad = w.modules.get("pop_adaptive")
                if ad is not None and hasattr(ad, "choose_activity"):
                    c2 = np.flatnonzero(c("class")[st_ix] == 2)
                    if c2.size:                 # class 2: the shared adaptive policy chooses the activity
                        pick[c2] = ad.choose_activity(ctx, due[st_ix[c2]], np.asarray(slots), logits[c2], rng)
                forced = woken[st_ix] & np.isin(wact[st_ix], slots)
                pick = np.where(forced, wact[st_ix], pick)
                chosen[st_ix] = pick
            spent0 = c("attention_spent_min").astype(np.float64)
            util = np.zeros(due.size, np.float64)
            by_act = {}
            items_rep = w.runtime.setdefault("pop.tick.items", {})
            items_rep.clear()
            for a_ in reg["order"]:
                sp = reg["spec"][a_]
                sel = np.flatnonzero(chosen == sp["slot"])
                if sel.size == 0:
                    continue
                by_act[a_] = int(sel.size)
                with running_as(ctx, sp["owner"]):
                    out = sp["handler"](ctx, due[sel], now)
                if out is not None:
                    u = np.asarray(out, np.float64).reshape(-1)
                    if u.shape[0] != sel.size or not np.all(np.isfinite(u)):
                        raise ProtocolError(f"activity {a_} returned {u.shape[0]} utilities for {sel.size} "
                                            f"individuals")
                    util[sel] = np.clip(u, -2.0, 2.0)
            # ---- recorded learning outcome for every individual that took a session (re-read: handlers wrote)
            had = chosen >= 0
            rows = np.flatnonzero(had)
            spent1 = c("attention_spent_min").astype(np.float64)
            minutes = np.maximum(spent1 - spent0, 0.0)          # the attention ledger resets only in pop.daily
            ad = w.modules.get("pop_adaptive")
            if ad is not None and hasattr(ad, "reward_activity") and rows.size:
                ad.reward_activity(ctx, due[rows], util[rows], chosen[rows])
            if rows.size:
                sat = c("satisfaction").astype(np.float64)
                habit = c("habit").astype(np.float32)
                last_act = c("last_activity").astype(np.int64)
                n_sess = c("n_sessions").astype(np.int64)
                n_sw0 = c("n_switches").astype(np.int64)
                n_sw = n_sw0.copy()
                exp_ = c("experience").astype(np.float64)
                lr = 0.15 + 0.25 * prefs[:, 1]           # novelty seekers update habits faster
                sl = chosen[rows]
                h = habit[rows, sl].astype(np.float64)
                habit[rows] *= np.float32(0.995)       # unused habits fade slowly
                habit[rows, sl] = np.clip(h + lr[rows] * (util[rows] - sat[rows]), -4.0, 4.0).astype(np.float32)
                beta = 0.25 - 0.12 * prefs[rows, 5]
                sat[rows] = (1 - beta) * sat[rows] + beta * util[rows]
                n_sw[rows] += (last_act[rows] >= 0) & (last_act[rows] != sl)
                last_act[rows] = sl
                n_sess[rows] += 1
                exp_[rows] += 1.0
                rep = np.array([items_rep.get(int(i), -1) for i in due[rows]], np.int64)
                items = np.where(rep >= 0, rep, activation.items_from_minutes(minutes[rows]))
                items = np.maximum(items, 1)
                hd = c("hist_day").copy()
                hs = c("hist_sess").copy()
                hi = c("hist_items").copy()
                activation.ring_update(hd, hs, hi, rows, today, items)
                dl_after = np.where(planned_now[rows] | first_now[rows], np.maximum(day_left[rows] - 1, 0),
                                    day_left[rows])
                day_left[rows] = dl_after
                vals = {"satisfaction": sat[rows].astype(np.float32), "habit": habit[rows],
                        "n_sessions": n_sess[rows].astype(np.int32), "n_switches": n_sw[rows].astype(np.int32),
                        "last_activity": last_act[rows].astype(np.int16), "experience": exp_[rows].astype(np.float32),
                        "last_utility": util[rows].astype(np.float32),
                        "last_session_at": np.full(rows.size, now, np.int64),
                        "hist_day": hd[rows], "hist_sess": hs[rows], "hist_items": hi[rows],
                        "last_active_day": np.full(rows.size, today, np.int32),
                        "td_n": (c("td_n")[rows].astype(np.int64) + 1).astype(np.int16),
                        "td_log_sum": (c("td_log_sum")[rows] + activation.log1p_minutes(minutes[rows])
                                       ).astype(np.float32),
                        "td_items": (c("td_items")[rows] + items).astype(np.float32),
                        "session_end_at": (now + 60 * minutes[rows]).astype(np.int64)}
                with running_as(ctx, "pop"):
                    record_decisions(ctx, "pop.learn", due[rows], vals, model_id=activation.LEARN_MODEL[0],
                                     model_version=activation.LEARN_MODEL[1],
                                     fidelity_id="pop.decision.habit_learning",
                                     extra={"t": int(now), "sessions": int(rows.size),
                                            "minutes": round(float(minutes[rows].sum()), 3)})
                switches = int(np.sum(n_sw - n_sw0))
                left_plat = int(np.sum(habit[rows, sl] < -1.0))
            else:
                switches = left_plat = 0
        finally:
            w.runtime.pop("pop.tick.decisions", None)
            w.runtime.pop("pop.tick.items", None)
        # ---- scheduling of the cohort's next activation (every activated individual)
        end_at = c("session_end_at").astype(np.int64)
        wl = activation.waking_left_s(now, c("day_start_min"), c("day_len_min"))
        after = np.where(had, np.maximum(end_at, now) - now, 0).astype(np.float64)
        gap = activation.next_session_gap_s(np.maximum(wl - after, 0.0), day_left, rng)
        nxt_sess = np.where(np.isfinite(gap), now + after + np.maximum(gap, activation.SESSION_GAP_S), -1)
        nxt_wake = activation.next_wake_s(now, c("day_start_min"), c("day_len_min"), rng)
        nxt = np.where((day_left > 0) & (nxt_sess >= 0) & (nxt_sess < now + wl), nxt_sess, nxt_wake)
        nxt = np.where(new & ~had & (first_at >= 0), first_at, nxt)        # first planned session still ahead
        nxt = np.maximum(nxt.astype(np.int64), now + 60)
        wake_at = np.where(woken, NEVER, c("wake_at")).astype(np.int64)
        wake_act = np.where(woken, -1, wact).astype(np.int16)
        day_left = np.where(~had & (left <= 0.5), 0, day_left)         # no attention left today: the plan ends
        extra = {"t": int(now), "activations": int(due.size), "sessions": int(rows.size), "woken": int(woken.sum()),
                 "day_decisions": int(nd.size), "by_activity": by_act, "left_platform": left_plat,
                 "switches": switches, "mean_utility": float(util[rows].mean()) if rows.size else 0.0,
                 "decisions": decided}
        from .api import record_decisions as _rd
        with running_as(ctx, "pop"):
            _rd(ctx, "pop.session", due, {"next_active_at": nxt, "wake_at": wake_at, "wake_act": wake_act,
                                          "day_left": day_left.astype(np.int16)},
                model_id=activation.SCHEDULE_MODEL[0], model_version=activation.SCHEDULE_MODEL[1],
                fidelity_id="pop.decision.schedule", extra=extra)
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
        from .api import record_decisions
        rng = ctx.rng("pop.daily")
        # employment: rare job loss and job finding (income and the daily commitment window follow); finding a job
        # is likelier with more of the skills an individual has built up through its own activity
        emp = columns.col(w, TABLE, "employer")
        sk = columns.col(w, TABLE, "skills").astype(np.float64).mean(axis=1)
        u = rng.random(n)
        lose = (emp >= 0) & (u < 0.002)
        find = (emp < 0) & (u < 0.02 * (0.4 + 1.2 * sk))
        ch = np.flatnonzero(lose | find)
        if ch.size:
            inc = columns.col(w, TABLE, "income_cents")[ch].astype(np.int64)
            gets = find[ch]
            pend = accrued(w, ch)
            new_emp = np.where(gets, rng.integers(0, N_EMPLOYERS, ch.size), -1).astype(np.int32)
            new_inc = np.where(gets, inc / 0.35, inc * 0.35).astype(np.int32)
            record_decisions(ctx, "pop.employment", ch, {
                "employer": new_emp, "income_cents": new_inc,
                # income earned so far is kept at the old rate before the rate changes
                "wallet_cents": columns.col(w, TABLE, "wallet_cents")[ch] + pend,
                "income_day": np.full(ch.size, ctx.now // DAY, np.int32),
                "commit_start_min": np.where(gets, np.clip(rng.normal(9 * 60, 50, ch.size), 360, 780), 0).astype(np.int16),
                "commit_len_min": np.where(gets, np.clip(rng.normal(480, 40, ch.size), 240, 600), 0).astype(np.int16)},
                model_id="pop.employment", model_version="1", fidelity_id="pop.decision.employment",
                extra={"lost": int(lose.sum()), "found": int(find.sum())})
        # knowledge fades a little for those who were active (others' knowledge is read with its age)
        last = columns.col(w, TABLE, "last_session_at")
        act = np.flatnonzero(last >= ctx.now - DAY)
        if act.size:
            k = columns.col(w, TABLE, "knowledge")[act].astype(np.float32) * np.float32(0.985)
            goals = columns.col(w, TABLE, "goals")[act].copy()
            # goals follow experience: now and then the secondary goal becomes the goal served by the activity the
            # individual has come to like most (its strongest learned habit)
            reg = _registry(w)
            if goals.shape[1] > 1 and reg["order"]:
                redraw = np.flatnonzero(rng.random(act.size) < 0.04)
                if redraw.size:
                    hab = columns.col(w, TABLE, "habit")[act[redraw]].astype(np.float64)
                    names = reg["order"]
                    sl = np.array([reg["spec"][a]["slot"] for a in names])
                    best = np.argmax(hab[:, sl], axis=1)
                    served = np.array([(reg["spec"][a]["goals"] or (4,))[0] for a in names])
                    goals[redraw, 1] = served[best].astype(np.uint8)
            record_decisions(ctx, "pop.daily", act, {"knowledge": k.astype(np.float16), "goals": goals},
                             model_id="pop.drift", model_version="1", fidelity_id="pop.decision.daily_drift",
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

    # ------------------------------------------------------------------ blob retention (crucible.substrate.retention)
    def blob_roots(self, world):
        """Blobs the neural classes rebuild their weights from (adaptive.py/strategic.py `_rebuild`): the latest
        checkpoint of every policy group and the example batches of the updates logged after it."""
        st = world.state
        out = []
        for g, meta in sorted(st.tables.get("pop_policies", {}).items()):
            cks = meta.get("checkpoints") or []
            if not cks:
                continue
            step0 = cks[-1][0]
            out.append(cks[-1][1])
            for key in st.find_ids("pop_policy_log", "group", g):
                lg = st.tables["pop_policy_log"][key]
                if lg["step"] > step0:
                    out.extend(r for r in lg.get("examples") or [] if isinstance(r, str) and r.startswith("sha256:"))
        return out

    def superseded_blobs(self, world):
        """Checkpoints replaced by a later checkpoint of the same group: no rebuild from this state on reads them
        (their weights hashes stay in the hash-chained log and in pop_policies)."""
        return [ck[1] for _g, meta in sorted(world.state.tables.get("pop_policies", {}).items())
                for ck in (meta.get("checkpoints") or [])[:-1]]

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
    with running_as(ctx, "pop"):
        columns.batch(ctx, TABLE, "pop.deposit", ix, {"wallet_cents": np.zeros(1, np.int64),
                                                      "income_day": np.array([ctx.now // DAY], np.int32)},
                      extra={"amount": amount})
    return acct


def require_member(ctx: Ctx) -> int:
    i = ids.index(ctx.actor)
    if i is None:
        raise Forbidden("only members of the population")
    return i
