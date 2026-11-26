"""Kernel module `pop_semantic`: the runtime of class 4 of the population (compact semantic generalists).

Binding. `install` binds participant class 4 of the population to this runtime
(`crucible.population.register_class_runtime`): the class-4 rows of the shared population table are the semantic
participants, with their identity, interests, skills, money, contacts and history living in that table like every
other individual's. The population's tick hands every due class-4 row to `_activate` inside its recorded session
machinery; the handler returns each participant's next activation time (event-driven: participants are idle between
activations, MASTER_SPEC 13.2).

Persistence (SP-3.3c). Each participant's own mind state - persona, goals with their status, memory, action
history, the identifiers it has seen, perception cursors, counters - is a row of table `semp_state`, written only by
the appliers of `semp.agent` (genesis) and `semp.update` (one per participant per activation, carrying only what the
activation added). It is part of every snapshot, fork and replay of the world; a resumed world continues each
participant exactly where it stopped. Nothing lives on a Python object between activations.

Config: ``{steps: 3, shortlist: 10, gap_s: [14400, 28800], patience: 4, max_failures: 4, brain: "lm" | "ablated",
routes: {capability: model id}, memory: 40, history: 80}``. The brain is the shared models of
`crucible.semantic.routing` (one weight copy per model per process); ``brain="ablated"`` replaces every model
decision by a uniform random one (SP-3.2d ablation).

Participants act only through `ctx.call(tool, args, as_principal=<their own principal>)`: the same tools, schemas,
authorisation and evaluator-tool filtering as any participant's `World.execute`. They hold no tool of their own.
"""
from __future__ import annotations

import numpy as np

from ...core import schema as S
from ...core.errors import ProtocolError
from ...core.kernel import Module, applier, register_module
from ...substrate import bus, columns, ids
from . import mind as M

NAME = "pop_semantic"
TABLE = "semp_state"
CLASS = 4
DAY = 86400

_ANY = {"type": "object", "additionalProperties": True}
bus.event_type("semp.agent", S.obj({"i": {"type": "integer"}, "principal": S.s_str(maxLength=40),
                                    "account": {"type": ["string", "null"]}, "persona": _ANY},
                                   ["i", "principal", "persona"]),
               owner=NAME, doc="a class-4 semantic participant joins (genesis): its principal, account and persona")
bus.event_type("semp.update", S.obj({"i": {"type": "integer"}, "memory_add": {"type": "array"},
                                     "history_add": {"type": "array"}, "known_add": _ANY, "goals_set": {"type": "array"},
                                     "active": {"type": ["integer", "null"]}, "seen": _ANY, "counts": _ANY,
                                     "next_at": {"type": "integer"}, "steps": {"type": "array"}},
                                    ["i", "memory_add", "history_add", "goals_set", "counts", "next_at"]),
               owner=NAME, doc="what one activation of a semantic participant added to its own state")

SKILL_WORDS = {"writing": "writing", "technical": "technical work and programming", "social": "dealing with people",
               "planning": "planning", "evaluating": "judging quality", "searching": "finding information",
               "reading": "reading", "shopping": "finding good deals"}
TRAITS = {"sociability": ("sociable", "reserved"), "novelty_seeking": ("curious about new things", "set in your ways"),
          "price_sensitivity": ("careful with money", "generous with money"),
          "impulsiveness": ("impulsive", "deliberate")}


def persona_of(world, i: int, contacts: list[str]) -> dict:
    """A participant's self-description from its own population state (interests, skills, traits, work, income)."""
    from ...population import interest_coords, features
    from ...population import schema as PS
    from ...population import semantics as sem
    ix = np.array([int(i)])
    f = features(world, ix)
    c = interest_coords(world, ix)[0].astype(np.float32)
    names = sem.topic_names()
    tc = sem.topic_coords(c.shape[0])
    sims = tc @ c
    top = [int(j) for j in np.argsort(-sims, kind="stable")[:3]]
    topics = [f"{names[j].replace('_', ' ').replace('-', ' ')} ({', '.join(sem.topic_words(j)[:3])})" for j in top]
    sk = f["skills"][0]
    good = [SKILL_WORDS[n] for n, v in zip(PS.SKILL_NAMES, sk) if v >= 0.6 and n in SKILL_WORDS][:3]
    pr = f["prefs"][0]
    traits = []
    for n, (hi, lo) in TRAITS.items():
        v = float(pr[PS.PREF_NAMES.index(n)])
        if v >= 0.65:
            traits.append(hi)
        elif v <= 0.25:
            traits.append(lo)
    inc = int(columns.col(world, "pop", "income_cents")[i])
    about = (f"You are interested in {', '.join(topics[:-1])} and {topics[-1]}. " +
             (f"You are good at {', '.join(good)}. " if good else "") +
             (f"You are {', '.join(traits)}. " if traits else "") +
             ("You have a job. " if bool(f["employed"][0]) else "You have no job at the moment. "))
    return {"about": about.strip(), "topics": [names[j] for j in top], "skills": good, "traits": traits,
            "contacts": list(contacts), "income_per_day_cents": inc}


@register_module
class SemanticGeneralists(Module):
    """Class-4 semantic participants of the population (see the module docstring)."""
    name = NAME
    host = "semp"
    version = "1"
    depends = ("pop", "bank", "identity")

    def install(self, world):
        from ... import population
        world.state.declare_table(TABLE)
        self.cfg = {"steps": 3, "shortlist": 10, "gap_s": [4 * 3600, 8 * 3600], "patience": 4, "max_failures": 4,
                    "brain": "lm", "routes": {}, "memory": 40, "history": 80, **self.config}
        population.register_class_runtime(world, CLASS, self._activate, owner=NAME)
        bus.enforce(world, owner=NAME)

    # ------------------------------------------------------------------ runtime objects (never state)
    def brain(self, world):
        b = world.runtime.get("semp.brain")
        if b is None:
            from . import brain
            kw = {"routes": dict(self.cfg.get("routes") or {})} if self.cfg["brain"] == "lm" else {}
            b = world.runtime["semp.brain"] = brain.make(self.cfg["brain"], **kw)
        return b

    def mind_config(self) -> M.Config:
        c = self.cfg
        return M.Config(steps=int(c["steps"]), shortlist=int(c["shortlist"]), patience=int(c["patience"]),
                        max_failures=int(c["max_failures"]), memory=int(c["memory"]), history=int(c["history"]))

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        w = ctx.world
        members = np.flatnonzero(columns.col(w, "pop", "class") == CLASS)
        principals = [ids.pid(int(i)) for i in members]
        bank = w.modules["bank"]
        for i, p in zip(members.tolist(), principals):
            ids.ensure_principal(ctx, int(i), account=True)
            acct = bank.primary_account(ctx.state, p)
            others = [q for q in principals if q != p]
            bus.emit(ctx, "semp.agent", {"i": int(i), "principal": p, "account": acct,
                                         "persona": persona_of(w, int(i), others)}, vis="internal")

    @applier("semp.agent")
    def ap_agent(self, ev):
        p = ev.payload
        self.world.state.put(TABLE, str(int(p["i"])), {
            "i": int(p["i"]), "principal": p["principal"], "account": p.get("account"), "persona": p["persona"],
            "goals": [], "active": None, "memory": [], "history": [], "known": {}, "seen": {}, "counts": {},
            "next_at": None, "joined": ev.t})

    @applier("semp.update")
    def ap_update(self, ev):
        p = ev.payload
        st = self.world.state
        key = str(int(p["i"]))
        row = st.get(TABLE, key)
        if row is None:
            raise ProtocolError(f"semantic participant {key} unknown")
        cfg = getattr(self, "cfg", {"memory": 40, "history": 80, "goals": 24})
        goals = {g["id"]: g for g in row["goals"]}
        for g in p["goals_set"]:
            goals[g["id"]] = g
        known = dict(row["known"])
        known.update(p.get("known_add") or {})
        if len(known) > 120:
            known = dict(list(known.items())[-120:])
        st.patch(TABLE, key, memory=(row["memory"] + list(p["memory_add"]))[-int(cfg.get("memory", 40)):],
                 history=(row["history"] + list(p["history_add"]))[-int(cfg.get("history", 80)):],
                 goals=[goals[k] for k in sorted(goals)][-24:], active=p.get("active"), known=known,
                 seen=p.get("seen") or row["seen"], counts=p["counts"], next_at=int(p["next_at"]),
                 last_steps=list(p.get("steps") or []))

    # ------------------------------------------------------------------ activation (called by the population tick)
    def _activate(self, ctx, idx, now):
        w = ctx.world
        rows = [w.state.get(TABLE, str(int(i))) for i in idx]
        agents = [M.Agent.from_row(r) for r in rows if r is not None]
        for a in agents:
            self._deposit(ctx, a.i)
        if agents:
            n0, seq0, mark = len(ctx.events), w.seq, ctx.state.savepoint()
            try:
                M.activate(ctx, agents, self.brain(w), self.mind_config())
            except Exception as e:  # noqa: BLE001 - a fault of the runtime itself (e.g. the model ran out of memory)
                # undo this activation exactly (as a refused call is undone) and record the fault, so the
                # participants keep their schedule instead of the population's tick failing and retrying them at
                # every tick (a standard civilisation froze every class-4 participant after day 1 this way)
                ctx.state.rollback_to(mark)
                del ctx.events[n0:]
                w.seq = seq0
                fault = f"{type(e).__name__}: {str(e)[:200]}"
                agents = [M.Agent.from_row(r) for r in rows if r is not None]
                for a in agents:
                    a.count("activation_faults")
                    a.step_log.append({"fault": fault})
        nxt = self._next_times(ctx, np.asarray(idx, np.int64), now)
        by = {a.i: a for a in agents}
        for k, i in enumerate(np.asarray(idx, np.int64).tolist()):
            a = by.get(int(i))
            if a is not None:
                bus.emit(ctx, "semp.update", M.delta(a, int(nxt[k])), vis="internal")
        return nxt

    def _deposit(self, ctx, i: int) -> None:
        """Income the individual earned outside the world arrives in its bank account (the population's own
        mechanics: undeposited wallet plus accrued income)."""
        from ... import population
        dep = getattr(population, "deposit", None)
        if dep is None:                    # not exported by the population API in this checkout
            from ...population.core import money_deposit as dep
        dep(ctx, int(i))

    def _next_times(self, ctx, idx: np.ndarray, now: int) -> np.ndarray:
        """Next activation of each participant: a gap drawn from `gap_s`, moved into its waking hours."""
        w = ctx.world
        lo, hi = (int(x) for x in self.cfg["gap_s"])
        rng = ctx.rng("semp.next", int(now))
        gaps = rng.integers(lo, max(lo + 1, hi), idx.size)
        ds = columns.col(w, "pop", "day_start_min")[idx].astype(np.int64) * 60
        dl = columns.col(w, "pop", "day_len_min")[idx].astype(np.int64) * 60
        t = now + gaps
        tod = t % DAY
        since = (tod - ds) % DAY
        asleep = since >= dl
        return np.where(asleep, t + (DAY - since) + rng.integers(0, 1800, idx.size), t).astype(np.int64)

    def check_invariants(self, world):
        errs = []
        if "pop" not in world.modules:
            return errs
        cls = columns.col(world, "pop", "class")
        for key, row in world.state.scan(TABLE):
            i = int(row["i"])
            if i >= cls.shape[0] or int(cls[i]) != CLASS:
                errs.append(f"semantic participant {key} is not a class-4 individual")
            if row["principal"] != ids.pid(i):
                errs.append(f"semantic participant {key} has the wrong principal")
            if row.get("active") is not None and not any(g["id"] == row["active"] for g in row["goals"]):
                errs.append(f"semantic participant {key}: active goal missing")
        return errs


def rows(world) -> list[dict]:
    """Every semantic participant's state row (evaluator side)."""
    return [r for _, r in sorted(world.state.scan(TABLE), key=lambda kv: int(kv[0]))] if TABLE in \
        world.state.tables else []
