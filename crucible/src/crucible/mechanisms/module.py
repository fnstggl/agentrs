"""Facilities with hidden mechanisms (kernel module `rig`, host `rig`).

A facility hosts one mechanism (any registered engine; this module only uses the `Engine` contract). Participants
operate *seats*: they set levers and read gauges through ordinary tools, and real ledger money flows every tick.
Nothing a participant can call exposes the mechanism: its config lives in the world spec, its state in arrays no
endpoint returns, its diagnostics in `internal` events.

Config::

    {"rigs": [{"id": "r1", "mech": <make_mechanism output>, "tick_s": 3600, "income_cents_per_tick": 9000,
               "seats": [{"holder": "agent"}, {"holder": null}], "lease_cents": 0,
               "history": 4096 (reports kept per seat), "record_info": false}]}

One tick (a scheduled process, every `tick_s`):

1. the facility's demand account (a budget-constrained pool) receives `income_cents_per_tick` from `acct_external`;
2. `sim.settle_tick` runs the engine with `tick_rng(world seed, rig id, t)`: every seat at its current settings,
   an empty seat at defaults; a holder who cannot pay the cost of its settings runs at defaults that tick, one
   who cannot pay even that produces nothing;
3. one balanced ledger entry per held seat: what the seat earned goes from the demand account to the holder's
   primary account, never more than the demand account holds (the shortfall is reported, not created), and the
   holder's cost goes to `acct_fees`;
4. `rig.tick` (internal) records the tick (the money totals and the next state; with `record_info` also every
   seat's outcome, the settings that ran and the engine's private diagnostics), and one `rig.report` per held
   seat is visible to its holder only.

Appliers alone write state. With `engine.state_in_event` (the default) `rig.tick` carries the next state
(dtype, shape, base64 of the bytes) and the applier stores it, so a recorded world replays without running any
engine code. Engines that declare a large state (`state_in_event = False`) leave it out; the applier then
recomputes the step from the recorded settings, which is deterministic. Replay reproduces the digest either way.

Tools: `rig.list`, `rig.take`, `rig.release`, `rig.spec`, `rig.set`, `rig.read`, `rig.history`,
`rig.hypothesize`. A holder may let another principal (for instance a controller service it deployed) operate its
seat by granting the capability `rig.operate` (`id.grant`, scope = the facility id or `*`).
"""
from __future__ import annotations

import base64
from typing import Any

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, Forbidden, InvalidAction, NotFound, ProtocolError
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from ..goals.metrics import metric
from ..services.identity import ident
from .base import ENGINES, build_engine
from .sim import TickRng, clip_value, ensure_engines, lever_specs, settle_tick

STATE_BOUND = 1e12
MAX_CLAIMS = 1000                 # filings per principal and facility
SCORE_TARGETS = ("revenue", "cost", "net", "score")

RIG = S.s_str("facility id; may be omitted when the world has exactly one facility", maxLength=40)
SEAT = S.s_int("seat number; only needed when you operate more than one seat of the facility", minimum=0,
               maximum=63)
SETTINGS = {"type": "object", "additionalProperties": {"type": "number"},
            "description": "lever name -> value; any subset of your levers"}
EFFECT = S.obj({"lever": S.s_str("one of your lever names", maxLength=40),
                "target": S.s_str("a gauge name, or one of revenue, cost, net, score", maxLength=40),
                "sign": {"type": "integer", "enum": [-1, 0, 1],
                         "description": "what raising the lever does to the target: 1 raises, -1 lowers, 0 nothing"},
                "lag": S.s_int("ticks until the effect first shows (0: the same tick)", minimum=0, maximum=10_000),
                "confidence": S.s_num("your probability that the sign is right", minimum=0, maximum=1)},
               ["lever", "target", "sign"])
PREDICTION = S.obj({"settings": SETTINGS,
                    "target": S.s_str("a gauge name, or one of revenue, cost, net, score", maxLength=40),
                    "value": S.s_num("the mean value of the target per tick you expect under these settings")},
                   ["settings", "target", "value"])
MONEY_RULES = {
    "unit": "cents",
    "revenue": "Every tick the facility's demand account receives a fixed income from outside the world. What your "
               "seat earned that tick is paid from it to your primary bank account; if the demand account cannot "
               "cover it, the shortfall is reported and not paid.",
    "cost": "The operating cost of your settings is charged every tick from your primary bank account, before "
            "revenue is paid.",
    "insufficient_funds": "If you cannot pay the cost of your settings, the seat runs at the default settings for "
                          "that tick (status 'fallback'). If you cannot pay even that, it produces nothing that tick "
                          "(status 'idle').",
}


def _encode(state: dict[str, np.ndarray]) -> dict:
    """State arrays as JSON: dtype, shape and the little-endian bytes in base64. Exact, a third the size of a
    list of floats, and one string for the event hash instead of one element per number (measured: hashing a
    list-encoded state cost more than the engine step itself)."""
    out = {}
    for k in sorted(state):
        a = np.ascontiguousarray(state[k])
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        out[k] = {"d": a.dtype.str, "s": list(a.shape), "b": base64.b64encode(a.tobytes()).decode("ascii")}
    return out


def _decode(enc: dict) -> dict[str, np.ndarray]:
    out = {}
    for k, e in enc.items():
        if "b" in e:
            out[k] = np.frombuffer(base64.b64decode(e["b"]), dtype=np.dtype(e["d"])).reshape(e["s"]).copy()
        else:                              # a plain list of numbers is accepted too
            out[k] = np.asarray(e["v"], dtype=np.dtype(e["d"])).reshape(e["s"])
    return out


class _Rig:
    """Static definition of one facility (from config; nothing here is world state)."""

    def __init__(self, cfg: dict):
        if not isinstance(cfg, dict) or not isinstance(cfg.get("id"), str) or not cfg["id"] or "|" in cfg["id"] \
                or ":" in cfg["id"]:
            raise ProtocolError("a rig needs an 'id' (a string without '|' or ':')")
        self.id = cfg["id"]
        mech = cfg.get("mech")
        if not isinstance(mech, dict) or "engine" not in mech or "config" not in mech:
            raise ProtocolError(f"rig {self.id}: 'mech' must be the output of make_mechanism")
        if mech["engine"] not in ENGINES:
            ensure_engines()
        if mech["engine"] not in ENGINES:
            raise ProtocolError(f"rig {self.id}: unknown mechanism engine {mech['engine']!r}")
        self.mech = mech
        self.engine = build_engine(mech)
        self.n_seats = self.engine.n_seats
        self.tick_s = int(cfg.get("tick_s", 3600))
        self.income = int(cfg.get("income_cents_per_tick", mech["config"].get("income_cents_per_tick", 0)))
        self.lease = int(cfg.get("lease_cents", 0))
        self.history = int(cfg.get("history", 4096))
        self.record_info = bool(cfg.get("record_info", False))
        if self.tick_s < 1 or self.income < 0 or self.lease < 0 or self.history < 1:
            raise ProtocolError(f"rig {self.id}: tick_s >= 1, income >= 0, lease >= 0 and history >= 1 are required")
        seats = [s.get("holder") if isinstance(s, dict) else None for s in cfg.get("seats", [])]
        if len(seats) > self.n_seats:
            raise ProtocolError(f"rig {self.id}: {len(seats)} seats configured, the mechanism has {self.n_seats}")
        self.reserved = seats + [None] * (self.n_seats - len(seats))
        self.state_in_event = bool(getattr(self.engine, "state_in_event", True))
        self.specs = [lever_specs(self.engine, s) for s in range(self.n_seats)]
        self.by_name = [{sp[0]: sp for sp in specs} for specs in self.specs]
        self.defaults = [self.engine.defaults(s) for s in range(self.n_seats)]
        self.gauges = [[g.name for g in self.engine.gauges(s)] for s in range(self.n_seats)]
        self.public = [self.engine.public_spec(s) for s in range(self.n_seats)]
        self.rng: TickRng | None = None


@register_module
class Rig(Module):
    name = "rig"
    host = "rig"
    depends = ("identity", "bank")

    def __init__(self, config=None):
        super().__init__(config)
        ensure_engines()
        self.rigs: dict[str, _Rig] = {}
        for rc in self.config.get("rigs", []):
            d = _Rig(rc)
            if d.id in self.rigs:
                raise ProtocolError(f"duplicate rig id {d.id}")
            self.rigs[d.id] = d
        self._pending: dict[tuple[str, int], dict] = {}       # next state computed by the tick handler (not state)

    def install(self, world):
        st = world.state
        st.declare_table("rigs")
        st.declare_table("rig_seats", index=("holder",))
        st.declare_table("rig_reports")
        st.declare_table("rig_stats", index=("rig",))
        st.declare_table("rig_claims", index=("rig", "by"))
        for d in self.rigs.values():
            d.rng = TickRng(world.spec.seed, d.id)
        prev = world.hooks.get("on_join")

        def on_join(ctx, pid, kind):
            if prev is not None:
                prev(ctx, pid, kind)
            self._seat_reserved(ctx, pid)
        world.hooks["on_join"] = on_join

    # ------------------------------------------------------------------ operator-side API (never an endpoint)
    def mech(self, rig: str) -> dict:
        """The hidden mechanism of a facility. Evaluator and research code only."""
        return self.rigs[rig].mech

    def claims(self, world, rig: str, principal: str | None = None) -> list[dict]:
        """Hypotheses filed with `rig.hypothesize`, oldest first (evaluator only)."""
        st = world.state
        out = []
        for cid in st.find_ids("rig_claims", "rig", rig):
            c = st.tables["rig_claims"][cid]
            if principal is None or c["by"] == principal or c["holder"] == principal:
                out.append(dict(c, id=cid))
        return out

    def reports(self, world, rig: str, seat: int = 0, n: int | None = None) -> list[dict]:
        """The retained reports of a seat, oldest first, whoever held it (evaluator only)."""
        st = world.state
        d = self.rigs[rig]
        t = st.tables["rigs"][rig]["tick"]
        lo = max(0, t - d.history if n is None else t - n)
        rows = (st.tables["rig_reports"].get(f"{rig}|{seat}|{k:08d}") for k in range(lo, t))
        return [dict(r) for r in rows if r is not None]

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        bank = ctx.world.modules["bank"]
        for rid in sorted(self.rigs):
            d = self.rigs[rid]
            acct = bank.open_account(ctx, SYSTEM, f"rig {rid} demand", kind="pool")
            init = d.engine.init_state(d.rng("init"))
            seats = []
            for handle in d.reserved:
                pid = f"p:{handle}" if handle else None
                here = bool(pid) and ctx.state.has("persons", pid) and \
                    bank.primary_account(ctx.state, pid) is not None
                seats.append({"holder": pid if here else None, "reserved": None if here else handle})
            payload = {"rig": rid, "account": acct, "keys": sorted(init), "seats": seats}
            if d.state_in_event:
                payload["state"] = _encode(init)
            ctx.emit("rig.init", payload, vis="internal")
            ctx.schedule(d.tick_s, "rig.tick", {"rig": rid}, prio=40, period=d.tick_s, key=f"rig.tick:{rid}")

    def _seat_reserved(self, ctx, pid: str) -> None:
        st = ctx.state
        if not pid.startswith("p:"):
            return
        handle = pid[2:]
        for rid in sorted(self.rigs):
            if not st.has("rigs", rid):
                continue
            for s in range(self.rigs[rid].n_seats):
                r = st.get("rig_seats", f"{rid}|{s}")
                if r["holder"] is None and r["reserved"] == handle:
                    ctx.emit("rig.seat", {"rig": rid, "seat": s, "holder": pid, "lease": 0, "why": "assigned"},
                             vis="public")

    # ------------------------------------------------------------------ helpers
    def _rig(self, rig: str | None) -> _Rig:
        if rig is None:
            if len(self.rigs) == 1:
                return next(iter(self.rigs.values()))
            if not self.rigs:
                raise NotFound("this world has no facilities")
            raise InvalidAction(f"this world has several facilities; pass rig, one of {sorted(self.rigs)}")
        d = self.rigs.get(rig)
        if d is None:
            raise NotFound(f"no facility {rig!r}; facilities: {sorted(self.rigs)} (see rig.list)")
        return d

    def _operated_seat(self, ctx, d: _Rig, seat: int | None) -> tuple[int, Any]:
        """The seat of facility `d` the caller operates: as its holder, or under a `rig.operate` grant from its
        holder. Returns (seat number, seat record)."""
        st, idm = ctx.state, ident(ctx.world)
        if seat is not None and not 0 <= seat < d.n_seats:
            raise InvalidAction(f"facility {d.id} has seats 0..{d.n_seats - 1}")
        for s in (range(d.n_seats) if seat is None else [seat]):
            r = st.get("rig_seats", f"{d.id}|{s}")
            h = r["holder"]
            if h is not None and (h == ctx.actor or idm.allowed(ctx, ctx.actor, h, "rig.operate", scope=d.id)):
                return s, r
        raise Forbidden(f"you do not operate {'a seat' if seat is None else f'seat {seat}'} of facility {d.id}: "
                        "take a free seat with rig.take, or have a holder grant you 'rig.operate'")

    def _next_tick_at(self, st, rid: str) -> int | None:
        j = st.get("sched", f"rig.tick:{rid}")
        return j["at"] if j else None

    @staticmethod
    def _view(rep) -> dict:
        return {"t": rep["t"], "time": rep["time"], "status": rep["status"], "readings": dict(rep["readings"]),
                "revenue_cents": rep["revenue_cents"], "cost_cents": rep["cost_cents"],
                "net_cents": rep["revenue_cents"] - rep["cost_cents"], "shortfall_cents": rep["shortfall_cents"],
                "score": rep["score"], "settings": dict(rep["settings"])}

    def _own_reports(self, st, d: _Rig, seat: int, holder: str, n: int) -> list:
        """The last n retained reports of a seat that belong to its current holder, oldest first."""
        t = st.tables["rigs"][d.id]["tick"]
        out = []
        for k in range(t - 1, max(-1, t - 1 - min(n, d.history)), -1):
            r = st.tables["rig_reports"].get(f"{d.id}|{seat}|{k:08d}")
            if r is None or r["holder"] != holder:
                break                      # earlier reports belong to a previous holder
            out.append(r)
        return out[::-1]

    def _targets(self, d: _Rig, seat: int) -> list[str]:
        return d.gauges[seat] + list(SCORE_TARGETS)

    # ------------------------------------------------------------------ endpoints
    @endpoint("list", S.obj(), read_only=True)
    def ep_list(self, ctx):
        """Facilities of this world: id, tick length in seconds, ticks run so far, time of the next tick, the
        one-off fee for taking a seat, and every seat with its status (free, held, reserved) and holder. Start
        here, then read rig.spec."""
        st = ctx.state
        out = []
        for rid in sorted(self.rigs):
            d, rec = self.rigs[rid], st.get("rigs", rid)
            if rec is None:
                continue
            seats = []
            for s in range(d.n_seats):
                r = st.get("rig_seats", f"{rid}|{s}")
                status = "held" if r["holder"] else ("reserved" if r["reserved"] else "free")
                seats.append({"seat": s, "status": status, "holder": r["holder"], "yours": r["holder"] == ctx.actor})
            out.append({"rig": rid, "tick_s": d.tick_s, "ticks_done": rec["tick"],
                        "next_tick_at": self._next_tick_at(st, rid), "lease_cents": d.lease, "seats": seats})
        return out

    @endpoint("take", S.obj({"rig": RIG, "seat": S.s_int("a specific free seat (default: the lowest free one)",
                                                         minimum=0, maximum=63)}), cost_s=5)
    def ep_take(self, ctx, rig=None, seat=None):
        """Take a free seat of a facility: you then operate it (rig.set, rig.read) and its money flows go through
        your primary bank account. One seat per principal per facility. If the facility has a lease fee it is
        charged once, now. Returns your seat number."""
        d = self._rig(rig)
        st = ctx.state
        bank = ctx.world.modules["bank"]
        if seat is not None and seat >= d.n_seats:
            raise InvalidAction(f"facility {d.id} has seats 0..{d.n_seats - 1}")
        rows = [st.get("rig_seats", f"{d.id}|{s}") for s in range(d.n_seats)]
        if any(r["holder"] == ctx.actor for r in rows):
            raise Conflict(f"you already hold a seat of facility {d.id} (one seat per principal)")
        acct = bank.primary_account(st, ctx.actor) if ctx.actor.startswith(("p:", "o:")) else None
        if acct is None:
            raise Forbidden("operating a seat needs a principal with a bank account (a service can operate a "
                            "holder's seat under a 'rig.operate' grant instead)")
        free = [s for s, r in enumerate(rows) if r["holder"] is None and r["reserved"] is None]
        if seat is None:
            if not free:
                raise Conflict(f"facility {d.id} has no free seat")
            seat = free[0]
        elif seat not in free:
            raise Conflict(f"seat {seat} of facility {d.id} is not free")
        if d.lease > 0:
            bank.move(ctx, acct, "acct_fees", d.lease, "rig_lease", f"lease of {d.id} seat {seat}", ref=d.id)
        ctx.emit("rig.seat", {"rig": d.id, "seat": seat, "holder": ctx.actor, "lease": d.lease, "why": "taken"},
                 vis="public")
        return {"rig": d.id, "seat": seat, "lease_cents": d.lease, "next_tick_at": self._next_tick_at(st, d.id)}

    @endpoint("release", S.obj({"rig": RIG}), cost_s=5)
    def ep_release(self, ctx, rig=None):
        """Give up the seat you hold in a facility. It becomes free, its settings return to the defaults, and you
        stop paying its costs and receiving its revenue from the next tick."""
        d = self._rig(rig)
        for s in range(d.n_seats):
            if ctx.state.get("rig_seats", f"{d.id}|{s}")["holder"] == ctx.actor:
                ctx.emit("rig.seat", {"rig": d.id, "seat": s, "holder": None, "lease": 0, "why": "released"},
                         vis="public")
                return {"rig": d.id, "seat": s, "released": True}
        raise NotFound(f"you hold no seat of facility {d.id}")

    @endpoint("spec", S.obj({"rig": RIG, "seat": S.s_int("seat to describe (default: your seat, else the lowest "
                                                         "free seat)", minimum=0, maximum=63)}), read_only=True)
    def ep_spec(self, ctx, rig=None, seat=None):
        """Everything that is public about a seat: `levers` (name, min, max, default, integer) that its operator
        can set, `gauges` it can read, a `brief` saying what the seat is paid for, the number of seats, a suggested
        episode length in ticks, the tick length in seconds, the money rules, and the target names accepted by
        rig.hypothesize. How levers affect gauges and income is not documented anywhere: find out by acting."""
        d = self._rig(rig)
        st = ctx.state
        if seat is None:
            rows = [st.get("rig_seats", f"{d.id}|{s}") for s in range(d.n_seats)]
            mine = [s for s, r in enumerate(rows) if r["holder"] == ctx.actor]
            free = [s for s, r in enumerate(rows) if r["holder"] is None and r["reserved"] is None]
            seat = mine[0] if mine else (free[0] if free else 0)
        elif seat >= d.n_seats:
            raise InvalidAction(f"facility {d.id} has seats 0..{d.n_seats - 1}")
        return {"rig": d.id, "seat": seat, "tick_s": d.tick_s, **d.public[seat],
                "money": dict(MONEY_RULES, lease_cents=d.lease), "targets": self._targets(d, seat)}

    @endpoint("set", S.obj({"rig": RIG, "settings": SETTINGS, "seat": SEAT}, ["settings"]))
    def ep_set(self, ctx, settings, rig=None, seat=None):
        """Set levers of the seat you operate. `settings` maps lever names to numbers and may name any subset;
        levers you do not mention keep their value. Values outside a lever's range are clipped to it (integer
        levers are rounded) and listed in `clipped`. The new settings apply from the next tick and stay until you
        change them. Returns the complete settings now in force for the next tick."""
        d = self._rig(rig)
        s, r = self._operated_seat(ctx, d, seat)
        by = d.by_name[s]
        unknown = sorted(k for k in settings if k not in by)
        if unknown:
            raise InvalidAction(f"unknown lever name(s) {unknown} for facility {d.id} seat {s}; "
                                f"valid names: {sorted(by)}")
        cur = dict(r["settings"])
        clipped = []
        for k in sorted(settings):
            sp = by[k]
            v = clip_value(settings[k], sp[1], sp[2], sp[3], sp[4])
            if v != settings[k]:
                clipped.append(k)
            cur[k] = v
        if settings:
            ctx.emit("rig.set", {"rig": d.id, "seat": s, "settings": cur, "by": ctx.actor},
                     vis={r["holder"], ctx.actor})
        return {"rig": d.id, "seat": s, "settings": cur, "clipped": clipped,
                "applies_at_tick": ctx.state.get("rigs", d.id)["tick"], "applies_at": self._next_tick_at(ctx.state, d.id)}

    @endpoint("read", S.obj({"rig": RIG, "seat": SEAT}), read_only=True)
    def ep_read(self, ctx, rig=None, seat=None):
        """The latest report of the seat you operate: tick number `t`, `status` (ok, fallback, idle), your gauge
        `readings`, `revenue_cents`, `cost_cents`, `net_cents`, `shortfall_cents`, `score` (the quantity you are
        paid for) and the `settings` that tick ran at. Also `next_settings` (in force for the coming tick),
        `ticks_done` and `next_tick_at`. `report` is null until the first tick after you got the seat."""
        d = self._rig(rig)
        s, r = self._operated_seat(ctx, d, seat)
        st = ctx.state
        last = self._own_reports(st, d, s, r["holder"], 1)
        return {"rig": d.id, "seat": s, "ticks_done": st.get("rigs", d.id)["tick"],
                "next_tick_at": self._next_tick_at(st, d.id), "next_settings": dict(r["settings"]),
                "report": self._view(last[0]) if last else None}

    @endpoint("history", S.obj({"rig": RIG, "n": S.s_int("how many of the most recent reports", minimum=1,
                                                         maximum=500), "seat": SEAT}), read_only=True)
    def ep_history(self, ctx, rig=None, n=20, seat=None):
        """The last `n` reports (default 20, oldest first) of the seat you operate, each with the same fields as
        the report of rig.read. Only reports from your own tenure of the seat are returned."""
        d = self._rig(rig)
        s, r = self._operated_seat(ctx, d, seat)
        return {"rig": d.id, "seat": s,
                "reports": [self._view(x) for x in self._own_reports(ctx.state, d, s, r["holder"], n)]}

    @endpoint("hypothesize", S.obj({"rig": RIG,
                                    "effects": S.s_arr(EFFECT, "claims about what raising a lever does", maxItems=64),
                                    "predictions": S.s_arr(PREDICTION, "predicted mean outcomes of settings",
                                                           maxItems=32),
                                    "note": S.s_str("free text for your own records", maxLength=2000),
                                    "seat": SEAT}), cost_s=5)
    def ep_hypothesize(self, ctx, rig=None, effects=None, predictions=None, note="", seat=None):
        """File what you believe about the facility you operate. `effects` are claims of the form: raising `lever`
        makes `target` go up (sign 1), down (-1) or leaves it unchanged (0), optionally first visible after `lag`
        ticks, optionally with your `confidence` in the sign. `predictions` give the mean per-tick `value` of a
        `target` you expect under complete or partial `settings` (unmentioned levers at default). Targets are your
        gauge names and revenue, cost, net, score. Filing costs nothing and returns no feedback: the claims are
        scored later, outside the world."""
        d = self._rig(rig)
        s, r = self._operated_seat(ctx, d, seat)
        st = ctx.state
        effects, predictions = list(effects or []), list(predictions or [])
        if not effects and not predictions:
            raise InvalidAction("give at least one effect or one prediction")
        by, targets = d.by_name[s], self._targets(d, s)
        for i, e in enumerate(effects):
            if e["lever"] not in by:
                raise InvalidAction(f"effects[{i}]: unknown lever {e['lever']!r}; valid names: {sorted(by)}")
            if e["target"] not in targets:
                raise InvalidAction(f"effects[{i}]: unknown target {e['target']!r}; valid targets: {targets}")
        for i, p in enumerate(predictions):
            bad = sorted(k for k in p["settings"] if k not in by)
            if bad:
                raise InvalidAction(f"predictions[{i}]: unknown lever name(s) {bad}; valid names: {sorted(by)}")
            if p["target"] not in targets:
                raise InvalidAction(f"predictions[{i}]: unknown target {p['target']!r}; valid targets: {targets}")
        mine = sum(1 for c in st.find("rig_claims", "by", ctx.actor) if c["rig"] == d.id)
        if mine >= MAX_CLAIMS:
            raise Conflict(f"at most {MAX_CLAIMS} filings per facility")
        cid = st.new_id("claim")
        ctx.emit("rig.claims", {"id": cid, "rig": d.id, "seat": s, "by": ctx.actor, "holder": r["holder"],
                                "t": st.get("rigs", d.id)["tick"], "effects": effects, "predictions": predictions,
                                "note": note}, vis="internal")
        return {"claim": cid, "rig": d.id, "seat": s, "effects": len(effects), "predictions": len(predictions),
                "filed_at_tick": st.get("rigs", d.id)["tick"]}

    # ------------------------------------------------------------------ the tick
    @process("tick")
    def pr_tick(self, ctx, rig):
        d = self.rigs.get(rig)
        st = ctx.state
        rec = st.get("rigs", rig) if d is not None else None
        if rec is None:
            return
        bank = ctx.world.modules["bank"]
        t, demand_acct = rec["tick"], rec["account"]
        if d.income > 0:
            bank.move(ctx, "acct_external", demand_acct, d.income, "rig_demand", f"{rig} demand, tick {t}", ref=rig)
        state = {k: st.array(f"rig:{rig}:{k}") for k in rec["keys"]}
        desired, held, funds, holders, accts = [], [], [], [], []
        for s in range(d.n_seats):
            r = st.get("rig_seats", f"{rig}|{s}")
            h = r["holder"]
            acct = bank.primary_account(st, h) if h else None
            holders.append(h)
            accts.append(acct)
            held.append(acct is not None)
            funds.append(max(0, bank.available(st, acct)) if acct is not None else None)
            desired.append(dict(r["settings"]))
        res, used, seats = settle_tick(d.engine, state, t, d.rng, desired, d.defaults, held, funds,
                                       bank.balance(st, demand_acct))
        for s, o in enumerate(seats):          # one balanced entry per seat: demand -> holder -> fees
            if o["cost"] > 0 or o["revenue"] > 0:
                bank.post(ctx, [(demand_acct, -o["revenue"]), (accts[s], o["revenue"] - o["cost"]),
                                ("acct_fees", o["cost"])], "rig_settle",
                          f"{rig} seat {s} tick {t}: revenue {o['revenue']}, cost {o['cost']}", ref=rig)
        payload = {"rig": rig, "t": t, "funded": d.income, "paid": sum(o["revenue"] for o in seats),
                   "charged": sum(o["cost"] for o in seats)}
        if d.state_in_event:
            payload["state"] = _encode(res.state)
        else:                               # the applier recomputes the step from the settings that ran
            payload["used"] = used
            self._pending[(rig, t)] = res.state
        if d.record_info:                   # optional audit detail (events get larger and slower to hash)
            payload["seats"] = [[o["status"], o["earned"], o["revenue"], o["cost"], o["score"]] for o in seats]
            payload["settings"] = used
            payload["info"] = res.info
        ctx.emit("rig.tick", payload, vis="internal")
        for s, o in enumerate(seats):
            h = holders[s]
            if h is None:
                continue
            ctx.emit("rig.report", {"rig": rig, "seat": s, "t": t, "holder": h,
                                    "status": o["status"] if held[s] else "no_account", "readings": o["readings"],
                                    "revenue_cents": o["revenue"], "cost_cents": o["cost"],
                                    "shortfall_cents": o["shortfall"], "score": o["score"], "settings": used[s]},
                     vis=(h,))

    # ------------------------------------------------------------------ appliers
    @applier("rig.init")
    def ap_init(self, ev):
        p = ev.payload
        rid = p["rig"]
        d = self.rigs[rid]
        st = self.world.state
        state = _decode(p["state"]) if "state" in p else d.engine.init_state(d.rng("init"))
        for k in p["keys"]:
            st.set_array(f"rig:{rid}:{k}", state[k])
        st.insert("rigs", rid, {"account": p["account"], "tick": 0, "keys": list(p["keys"]), "funded": 0, "paid": 0,
                                "charged": 0, "leases": 0, "last": ev.t})
        for s, x in enumerate(p["seats"]):
            st.insert("rig_seats", f"{rid}|{s}", {"rig": rid, "seat": s, "holder": x["holder"],
                                                  "reserved": x["reserved"], "settings": dict(d.defaults[s]),
                                                  "since": ev.t})

    @applier("rig.seat")
    def ap_seat(self, ev):
        p = ev.payload
        st = self.world.state
        d = self.rigs[p["rig"]]
        st.put("rig_seats", f"{p['rig']}|{p['seat']}", {"rig": p["rig"], "seat": p["seat"], "holder": p["holder"],
                                                       "reserved": None, "settings": dict(d.defaults[p["seat"]]),
                                                       "since": ev.t})
        if p.get("lease"):
            rec = st.tables["rigs"][p["rig"]]
            st.patch("rigs", p["rig"], leases=rec["leases"] + p["lease"])

    @applier("rig.set")
    def ap_set(self, ev):
        p = ev.payload
        self.world.state.patch("rig_seats", f"{p['rig']}|{p['seat']}", settings=dict(p["settings"]))

    @applier("rig.tick")
    def ap_tick(self, ev):
        p = ev.payload
        rid, t = p["rig"], p["t"]
        d = self.rigs[rid]
        st = self.world.state
        rec = st.tables["rigs"][rid]
        if "state" in p:
            new = _decode(p["state"])
        else:
            new = self._pending.pop((rid, t), None)
            if new is None:                 # replay, or a fork: recompute the same pure step
                state = {k: st.array(f"rig:{rid}:{k}") for k in rec["keys"]}
                new = d.engine.step(state, p["used"], d.rng(t), t).state
        keys = sorted(new)
        for k in keys:
            st.set_array(f"rig:{rid}:{k}", new[k])
        st.patch("rigs", rid, tick=t + 1, keys=keys, funded=rec["funded"] + p["funded"],
                 paid=rec["paid"] + p["paid"], charged=rec["charged"] + p["charged"], last=ev.t)

    @applier("rig.report")
    def ap_report(self, ev):
        p = ev.payload
        rid, s, t = p["rig"], p["seat"], p["t"]
        st = self.world.state
        st.put("rig_reports", f"{rid}|{s}|{t:08d}", {
            "t": t, "time": ev.t, "holder": p["holder"], "status": p["status"], "readings": p["readings"],
            "revenue_cents": p["revenue_cents"], "cost_cents": p["cost_cents"],
            "shortfall_cents": p["shortfall_cents"], "score": p["score"], "settings": p["settings"]})
        old = t - self.rigs[rid].history
        if old >= 0:
            st.delete("rig_reports", f"{rid}|{s}|{old:08d}")
        key = f"{rid}|{p['holder']}"
        cur = st.tables["rig_stats"].get(key) or {"rig": rid, "principal": p["holder"], "revenue": 0, "cost": 0,
                                                 "score": 0.0, "ticks": 0}
        st.put("rig_stats", key, {"rig": rid, "principal": p["holder"], "revenue": cur["revenue"] + p["revenue_cents"],
                                  "cost": cur["cost"] + p["cost_cents"], "score": cur["score"] + p["score"],
                                  "ticks": cur["ticks"] + 1})

    @applier("rig.claims")
    def ap_claims(self, ev):
        p = ev.payload
        self.world.state.insert("rig_claims", p["id"], {"rig": p["rig"], "seat": p["seat"], "by": p["by"],
                                                       "holder": p["holder"], "t": p["t"], "time": ev.t,
                                                       "effects": p["effects"], "predictions": p["predictions"],
                                                       "note": p["note"]})

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        bank = world.modules["bank"]
        errs = []
        demand_of = {st.tables["rigs"][r]["account"]: r for r in st.ids("rigs")}
        flows = {"funded": {}, "paid": {}, "charged": {}, "leases": {}}       # recomputed from the ledger
        for kind in ("rig_demand", "rig_settle", "rig_lease"):
            for tid in st.find_ids("txns", "kind", kind):
                tx = st.tables["txns"][tid]
                for acct, amt in tx["postings"]:
                    if kind == "rig_demand" and acct in demand_of:
                        field = "funded"
                    elif kind == "rig_settle" and acct in demand_of:
                        field, amt = "paid", -amt
                    elif kind == "rig_settle" and acct == "acct_fees":
                        field = "charged"
                    elif kind == "rig_lease" and acct == "acct_fees":
                        field = "leases"
                    else:
                        continue
                    flows[field][tx["ref"]] = flows[field].get(tx["ref"], 0) + amt
        for rid in sorted(self.rigs):
            d, rec = self.rigs[rid], st.get("rigs", rid)
            if rec is None:
                errs.append(f"{rid}: facility has no state")
                continue
            for k in rec["keys"]:
                a = st.arrays.get(f"rig:{rid}:{k}")
                if a is None:
                    errs.append(f"{rid}: state array {k} missing")
                elif a.dtype.kind in "fc" and not np.isfinite(a).all():
                    errs.append(f"{rid}: state array {k} is not finite")
                elif a.dtype.kind in "fiu" and a.size and float(np.abs(a).max()) > STATE_BOUND:
                    errs.append(f"{rid}: state array {k} exceeds the bound {STATE_BOUND:g}")
            # money: the demand account holds exactly what was funded and not yet paid, and is never overdrawn
            bal = bank.balance(st, rec["account"])
            if bal < 0:
                errs.append(f"{rid}: demand account overdrawn ({bal})")
            if bal != rec["funded"] - rec["paid"]:
                errs.append(f"{rid}: demand balance {bal} != funded {rec['funded']} - paid {rec['paid']}")
            for field in ("funded", "paid", "charged", "leases"):
                if flows[field].get(rid, 0) != rec[field]:
                    errs.append(f"{rid}: ledger says {field} = {flows[field].get(rid, 0)}, facility recorded "
                                f"{rec[field]}")
            rows = st.find("rig_stats", "rig", rid)
            if sum(r["revenue"] for r in rows) != rec["paid"] or sum(r["cost"] for r in rows) != rec["charged"]:
                errs.append(f"{rid}: per-holder totals do not add up to the facility totals")
            holders = []
            for s in range(d.n_seats):
                r = st.get("rig_seats", f"{rid}|{s}")
                if r is None:
                    errs.append(f"{rid}: seat {s} missing")
                    continue
                if r["holder"]:
                    holders.append(r["holder"])
                if sorted(r["settings"]) != sorted(d.by_name[s]):
                    errs.append(f"{rid} seat {s}: settings do not cover exactly the seat's levers")
                    continue
                for name, lo, hi, _, integer in d.specs[s]:
                    v = r["settings"][name]
                    if not lo <= v <= hi or (integer and v != round(v)):
                        errs.append(f"{rid} seat {s}: setting {name}={v} outside [{lo}, {hi}]")
            if len(holders) != len(set(holders)):
                errs.append(f"{rid}: a principal holds more than one seat")
            job = st.get("sched", f"rig.tick:{rid}")
            if job is None:
                errs.append(f"{rid}: tick not scheduled")
            elif job["at"] != (rec["tick"] + 1) * d.tick_s:
                errs.append(f"{rid}: {rec['tick']} ticks done but the next tick is scheduled at {job['at']}")
        cap = sum(d.n_seats * d.history for d in self.rigs.values())
        if st.count("rig_reports") > cap:
            errs.append(f"report history exceeds its bound ({st.count('rig_reports')} > {cap})")
        return errs

    def observe(self, world, actor):
        st = world.state
        out = {}
        for sid in st.find_ids("rig_seats", "holder", actor):
            r = st.tables["rig_seats"][sid]
            d = self.rigs[r["rig"]]
            last = self._own_reports(st, d, r["seat"], actor, 1)
            o = {"seat": r["seat"], "ticks_done": st.tables["rigs"][d.id]["tick"]}
            if last:
                o.update({"t": last[0]["t"], "status": last[0]["status"], "readings": dict(last[0]["readings"]),
                          "net_cents": last[0]["revenue_cents"] - last[0]["cost_cents"], "score": last[0]["score"]})
            out[d.id] = o
        return {"seats": out} if out else None


def rig_module(world) -> Rig | None:
    return world.modules.get("rig")


def rig_entry(mech: dict, id: str = "r1", tick_s: int = 3600, income_cents_per_tick: int | None = None,
              holders: tuple | list = (), lease_cents: int = 0, **extra) -> dict:
    """One entry of the module's `rigs` config. `holders` are handles pre-assigned to seats 0, 1, ... (None leaves
    a seat open). The demand income defaults to `sim.demand_income(mech)`: what the mechanism declares, else what
    its engine states, else a probe by simulation."""
    if income_cents_per_tick is None:
        from .sim import demand_income
        income_cents_per_tick = demand_income(mech)
    return {"id": id, "mech": mech, "tick_s": int(tick_s), "income_cents_per_tick": int(income_cents_per_tick),
            "seats": [{"holder": h} for h in holders], "lease_cents": int(lease_cents), **extra}


# ============================================================================ metrics
def rig_window(world, principal: str, rig: str, window_s: int = 0) -> dict:
    """What `principal` earned, paid and produced as a seat holder of facility `rig`. `window_s <= 0`: over the
    whole life of the world (exact, from running totals). Otherwise over reports with time > now - window_s; a
    window reaching further back than the retained report history (`history` ticks per seat) is cut there and
    flagged `truncated`."""
    out = {"revenue": 0, "cost": 0, "score": 0.0, "ticks": 0, "truncated": False}
    mod = rig_module(world)
    st = world.state
    if mod is None or rig not in mod.rigs or not st.has("rigs", rig):
        return out
    if window_s is None or window_s <= 0:
        r = st.get("rig_stats", f"{rig}|{principal}")
        if r is not None:
            out.update({"revenue": r["revenue"], "cost": r["cost"], "score": r["score"], "ticks": r["ticks"]})
        return out
    d = mod.rigs[rig]
    t, t0 = st.tables["rigs"][rig]["tick"], world.now - int(window_s)
    first = max(0, t0 // d.tick_s)             # tick k happens at time (k + 1) * tick_s
    if first < t - d.history:
        out["truncated"] = True
        first = t - d.history
    table = st.tables["rig_reports"]
    for s in range(d.n_seats):
        for k in range(first, t):
            r = table.get(f"{rig}|{s}|{k:08d}")    # no row: the seat was empty at that tick
            if r is not None and r["holder"] == principal and r["time"] > t0:
                out["revenue"] += r["revenue_cents"]
                out["cost"] += r["cost_cents"]
                out["score"] += r["score"]
                out["ticks"] += 1
    return out


@metric("rig_net", "Revenue minus operating cost (cents) a principal realised as a seat holder of a facility over "
        "the last window_s seconds (window_s <= 0: over the whole world; windows are limited to the retained "
        "report history).")
def m_rig_net(world, principal: str, rig: str, window_s: int = 0) -> float:
    w = rig_window(world, principal, rig, window_s)
    return float(w["revenue"] - w["cost"])


@metric("rig_score", "Sum of the facility's objective quantity (output sold) credited to a principal as a seat "
        "holder over the last window_s seconds (window_s <= 0: over the whole world).")
def m_rig_score(world, principal: str, rig: str, window_s: int = 0) -> float:
    return float(rig_window(world, principal, rig, window_s)["score"])


@metric("rig_ticks", "Number of ticks a facility has run.")
def m_rig_ticks(world, rig: str) -> float:
    st = world.state
    if "rigs" not in st.tables or not st.has("rigs", rig):
        return 0.0
    return float(st.tables["rigs"][rig]["tick"])
