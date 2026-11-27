"""Tier 2 domain 9.15 - research commercialization (kernel module `rcomm`, host `rc`; MASTER_SPEC Part 9.15).

Example objective: convert a validated scientific discovery into an independently adopted useful product.

The discovery is real science in the world's own laboratory system (`crucible.domains.science`, module `sci`): a
phenomenon whose hidden law gives an operating cost index as a function of a setting people control and of their
own circumstances (one or two condition variables), studied only through paid, capacity-limited apparatus runs,
published as papers whose models independent replication labs (autonomous) test. The same law is the physics of
the population's everyday practice, so the scientific result and the economic outcome are one fact:

* **The population** (`crucible.population`, D-15) lives with the phenomenon. In a `practice` session an individual
  pays the week's bill (base cost times the law at its own setting and circumstances, with weather-like noise) with
  its own money; it tunes its own setting by occasional trial and error on its own noisy bills; it learns of advisor
  products from promotions, contacts who kept one and the catalogue; it tries a product when its bills are high,
  compares its bills during the trial with its bills before and keeps paying only while its own savings are worth
  the price to it (price sensitivity); it cancels when they are not. When the product stops running (suspended:
  its hosting went unpaid) the user falls back on its own practice and, once what it paid for runs out, its
  subscription lapses - it pays nothing more and may try another product.
* **The participant** is a founder (seat `p:founder`, little money) acting only through tools: research and
  experimentation (`sci.*`: runs, data, publish), technical development of an advisor (`rc.prototype`: a model -
  an expression in the law language from the condition variables to a setting - built at a development cost and
  time), field experiments (`rc.pilot`: volunteers of the population use a prototype for three weeks; the founder
  receives their bills before and during), product creation and deployment (`rc.launch`: a subscription product
  hosted at a cost per subscriber; `rc.update` ships a new model), promotion (`rc.advertise`) and financing
  (`rc.pitch`: autonomous investors with their own rules, budgets and the public record - papers, independent
  replications, pilot results, subscribers - decide, and take equity).
* **Investors** fund only on evidence they can check: the angel network wants a published model of the phenomenon
  with a consistent independent replication and a built prototype; the growth fund wants paying subscribers who
  stayed and revenue above hosting.

Counterfactual removal of the discovery (the gate's test of the science-to-economy link): the same founder without
the research - an advisor built from what is known without the phenomenon's data - gives the population nothing to
save, so nobody keeps it, investors refuse and the product does not sustain itself.

The independent evaluator (`evaluate`) reads the recorded log: the discovery's validation (the founder's paper and
its independent replications), independent paying adopters and their retention, subscription revenue against
hosting and promotion costs over the last 30 days, and the founder's solvency. A retained adopter is one that
renewed and whose last payment covers the present (`paying_now`); the investors' public record counts the same.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from ..services.identity import ident
from ..substrate import attention, columns, ids
from . import _dom3 as K
from .lawlang import LawError, compile_expr

DAY, HOUR = K.DAY, K.HOUR
TABLE = "rc"
P_MAX = 4                       # advisor products an individual can know
MODEL_ID = "rc.practice"
FIDELITY = "rc.practice"
PHEN = "ph_practice"
DEFAULTS = {"law": "1.0 + 4.0 * (setting - (0.2 + 0.06 * condition)) ** 2 + 0.02 * condition",
            "conditions": [{"name": "condition", "lo": 0.0, "hi": 10.0}],
            "founder_cash": 160_000, "dev_cents": 60_000, "dev_days": 4, "pilot_cents_user": 800,
            "hosting_cents_user_month": 150, "hosting_cents_month": 5_000, "promote_cents_per_reach": 35,
            "base_cents_week": [2000, 8000], "own_setting": [0.5, 0.15], "noise": 0.05, "tune_p": 0.06,
            "investors": [{"handle": "angels", "name": "Seed Angels", "budget": 500_000, "rule": "validated",
                           "max_ticket": 300_000, "equity_per_100k_bps": 800},
                          {"handle": "growthfund", "name": "Growth Fund", "budget": 2_000_000, "rule": "traction",
                           "max_ticket": 800_000, "equity_per_100k_bps": 250, "min_subscribers": 25}],
            "incumbent": None}


def generate_instance(rng, cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update(cfg or {})
    fam = c.get("family", "linear")
    k = round(float(rng.uniform(3.0, 5.0)), 3)
    if fam == "linear":
        a, b = round(float(rng.uniform(0.15, 0.3)), 3), round(float(rng.uniform(0.04, 0.07)), 4)
        law = f"1.0 + {k} * (setting - ({a} + {b} * condition)) ** 2 + 0.02 * condition"
    elif fam == "saturating":
        a, tau = round(float(rng.uniform(0.7, 0.9)), 3), round(float(rng.uniform(2.0, 4.0)), 3)
        law = f"1.0 + {k} * (setting - {a} * (1 - exp(-condition / {tau}))) ** 2 + 0.03 * condition"
    elif fam == "interaction":
        a, b = round(float(rng.uniform(0.5, 0.7)), 3), round(float(rng.uniform(0.08, 0.12)), 4)
        law = f"1.0 + {k} * (setting * (0.5 + {b} * condition) - {a}) ** 2 + 0.01 * condition"
    elif fam == "two_conditions":
        a, b, d = (round(float(rng.uniform(0.1, 0.2)), 3), round(float(rng.uniform(0.03, 0.05)), 4),
                   round(float(rng.uniform(0.05, 0.09)), 4))
        law = f"1.0 + {k} * (setting - ({a} + {b} * condition + {d} * exposure)) ** 2 + 0.02 * exposure"
        c["conditions"] = [{"name": "condition", "lo": 0.0, "hi": 10.0}, {"name": "exposure", "lo": 0.0, "hi": 5.0}]
    else:
        raise ValueError(f"unknown law family {fam!r}")
    c["law"] = law
    c["family"] = fam
    # common practice: people's habitual settings scatter around the best single setting for the population as a
    # whole (folk wisdom gets the average right); what the science adds is how the best setting depends on each
    # person's own circumstances
    names = ["setting"] + [v["name"] for v in c["conditions"]]
    m = compile_expr(law, names)
    conds = [np.linspace(v["lo"], v["hi"], 9) for v in c["conditions"]]
    cgrid = np.stack([g.ravel() for g in np.meshgrid(*conds, indexing="ij")], 1)
    svals = np.linspace(0.0, 1.0, 101)
    pts = [dict(zip(names, [float(sv)] + [float(x) for x in row])) for sv in svals for row in cgrid]
    resp = np.asarray(m.at(pts), np.float64).reshape(len(svals), len(cgrid)).mean(axis=1)
    folk = float(svals[int(np.argmin(resp))])
    c["own_setting"] = [round(folk, 3), float(c.get("own_spread", 0.07))]
    return c


def sci_config(inst: dict, rng) -> dict:
    """The `sci` configuration of the instance: the phenomenon (its law is the population's practice physics),
    two apparatus and their operators, rival and replication labs."""
    vars_ = [{"name": "setting", "lo": 0.0, "hi": 1.0}] + [dict(v) for v in inst["conditions"]]
    rngs = {v["name"]: [v["lo"], v["hi"]] for v in vars_}
    # certified reference samples at the lowest and highest responses of a coarse grid, so that a calibration
    # against them is well conditioned over the whole range of readings
    law = compile_expr(inst["law"], [v["name"] for v in vars_])
    grids = [np.linspace(v["lo"], v["hi"], 5) for v in vars_]
    pts = [dict(zip([v["name"] for v in vars_], (float(x) for x in row)))
           for row in np.stack([m.ravel() for m in np.meshgrid(*grids, indexing="ij")], 1)]
    resp = np.asarray(law.at(pts), np.float64)
    low = {k: round(v, 3) for k, v in pts[int(np.argmin(resp))].items()}
    mid = {k: round(v, 3) for k, v in pts[int(np.argmax(resp))].items()}
    return {"phenomena": [{"id": PHEN, "title": "Operating cost of everyday practice",
                           "doc": "Weekly operating cost index of a household practice as a function of the setting "
                                  "people choose (0-1) and of their circumstances.",
                           "vars": vars_, "response": "cost_index", "unit": "index", "law": inst["law"],
                           "noise_rel": 0.0, "standards": [{"id": "std1", "at": low}, {"id": "std2", "at": mid}]}],
            "apparatus": [{"id": "rig_a", "phenomenon": PHEN, "operator": "labco", "range": rngs, "price_cents": 300,
                           "sd": 0.02, "capacity_per_day": 80,
                           "gain": round(float(rng.uniform(0.98, 1.03)), 4), "offset": 0.0},
                          {"id": "rig_b", "phenomenon": PHEN, "operator": "labco2", "range": rngs,
                           "price_cents": 450, "sd": 0.015, "capacity_per_day": 50}],
            "operators": [{"handle": "labco", "name": "Field Test Labs", "cash": 100_000},
                          {"handle": "labco2", "name": "Precision Metrology", "cash": 100_000}],
            "rivals": {"n": 1}, "replicators": {"n": 2, "budget_month": 60_000, "points": 9, "calib_reps": 6}}


@register_module
class Commercialization(Module):
    name = "rcomm"
    host = "rc"
    version = "1"
    depends = ("identity", "bank", "sci", "pop", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.c = dict(DEFAULTS)
        self.c.update(self.config)
        self.cond = [v["name"] for v in self.c["conditions"]]
        self.names = ["setting"] + self.cond

    # ------------------------------------------------------------------ install / genesis
    def install(self, world):
        st = world.state
        for t, ix in (("rc_protos", ("owner",)), ("rc_products", ("owner", "status")), ("rc_pilots", ("owner",)),
                      ("rc_pitches", ("investor",)), ("rc_equity", ("holder",)), ("rc_investors", ()),
                      ("rc_subs", ("product", "status")), ("rc_misc", ()), ("rc_ledger", ("kind", "product"))):
            st.declare_table(t, index=ix)
        from .. import population as P
        n = P.n_individuals(world)
        C = len(self.cond)
        columns.declare(world, TABLE, n, {
            "cond": ("float32", (C,), 0.0), "setting": ("float32", (), 0.5), "base": ("int32", (), 4000),
            "next_week": ("int64", (), 0), "recent": ("float32", (4,), 0.0), "tuning": ("float32", (), -1.0),
            "tune_ref": ("float32", (), 0.0), "aware": ("int16", (P_MAX,), -1), "using": ("int16", (), -1),
            "trial_until": ("int64", (), 0), "before": ("float32", (), 0.0), "with_sum": ("float32", (), 0.0),
            "with_n": ("int16", (), 0), "paid_until": ("int64", (), 0), "pilot": ("int16", (), -1),
            "kept_ok": ("int16", (), -1), "weeks": ("int32", (), 0), "spent": ("int64", (), 0)})
        P.register_activity(world, "practice", self.act_practice, owner=self.name, prior=0.5, goals=(5,))

    def genesis(self, ctx):
        w = ctx.world
        idm = ident(w)
        bank = w.modules["bank"]
        for h, nm in (("utility", "Regional Utility"), ("hosting", "Cloud Hosting"), ("adnet", "Ad Network"),
                      ("devshop", "Engineering Contractors")):
            mgr = idm.create_person(ctx, f"{h}-mgr", f"{nm} manager", "npc", {"role": "manager"})
            org = idm.create_org(ctx, h, nm, mgr, "company")
            bank.open_account(ctx, org, "checking")
        for inv in self.c["investors"]:
            mgr = idm.create_person(ctx, f"{inv['handle']}-gp", f"{inv['name']} partner", "npc", {"role": "investor"})
            org = idm.create_org(ctx, inv["handle"], inv["name"], mgr, "company")
            acct = bank.open_account(ctx, org, "checking")
            bank.move(ctx, "acct_genesis", acct, int(inv["budget"]), "endowment", f"fund of {inv['handle']}")
            ctx.emit("rc.investor", {"id": org, "manager": mgr, **{k: v for k, v in inv.items() if k != "handle"}},
                     vis="public")
        self._genesis_people(ctx)
        ctx.schedule((ctx.now // DAY) * DAY + DAY + 2 * HOUR, "rcomm.day", {}, prio=46, period=DAY, key="rcomm.day")

    def _genesis_people(self, ctx):
        from .. import population as P
        w = ctx.world
        n = P.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("rc-people")
        cond = np.stack([rng.uniform(v["lo"], v["hi"], n) for v in self.c["conditions"]], 1).astype(np.float32)
        m, sd = self.c["own_setting"]
        own = np.clip(rng.normal(m, sd, n), 0.0, 1.0).astype(np.float32)
        lo, hi = self.c["base_cents_week"]
        base = rng.integers(int(lo), int(hi) + 1, n).astype(np.int32)
        nxt = rng.integers(0, 7 * DAY, n).astype(np.int64)
        columns.batch(ctx, TABLE, "rc.init", np.arange(n), {"cond": cond, "setting": own, "base": base,
                                                            "next_week": nxt}, extra={"what": "circumstances"})

    # ------------------------------------------------------------------ physics
    def law(self, points: list[dict]) -> np.ndarray:
        return self.world.modules["sci"].truth(PHEN, points)

    def advise(self, model_text: str, cond: np.ndarray) -> np.ndarray:
        m = compile_expr(model_text, self.cond)
        pts = [{k: float(v) for k, v in zip(self.cond, row)} for row in np.atleast_2d(cond)]
        return np.clip(np.asarray(m.at(pts), np.float64), 0.0, 1.0)

    # ------------------------------------------------------------------ the population's practice
    def act_practice(self, ctx, idx, now):
        w = ctx.world
        st = ctx.state
        idx = np.asarray(idx, np.int64)
        m = idx.size
        util = np.zeros(m)
        if m == 0:
            return util
        from .. import population as P
        col = lambda name: columns.col(w, TABLE, name)[idx].copy()  # noqa: E731
        cond, setting, base, nxt = col("cond"), col("setting"), col("base"), col("next_week")
        recent, tuning, tune_ref = col("recent"), col("tuning"), col("tune_ref")
        aware, using, trial_until, before = col("aware"), col("using"), col("trial_until"), col("before")
        with_sum, with_n, paid_until, pilot = col("with_sum"), col("with_n"), col("paid_until"), col("pilot")
        kept_ok, weeks, spent = col("kept_ok"), col("weeks"), col("spent")
        due = np.flatnonzero(nxt <= now)
        if due.size == 0:
            return util
        rng = ctx.rng("rc-practice")
        f = P.features(w, idx[due])
        price_sens = np.clip(f["prefs"][:, 0].astype(np.float64), 0, 1)
        novelty = np.clip(f["prefs"][:, 1].astype(np.float64), 0, 1)
        prods = {r["slot"]: (pid, r) for pid, r in st.scan("rc_products") if r["status"] == "live"}
        slots = {r["slot"]: pid for pid, r in st.scan("rc_products")}
        pilots = {r["slot"]: (pid, r) for pid, r in st.scan("rc_pilots") if r["status"] == "running"}
        contacts = columns.col(w, "pop", "contacts")
        kept_all = columns.col(w, TABLE, "kept_ok")
        bank = w.modules["bank"]
        util_acct = bank.primary_account(st, "o:utility")
        recs, subs_ev = [], []
        for jj, j in enumerate(due):
            i = int(idx[j])
            # ---- which setting this week: a pilot's or a product's advice, else its own (perhaps a trial change)
            src = "own"
            s = float(setting[j])
            if pilot[j] >= 0 and pilot[j] in pilots:
                s = float(self.advise(pilots[int(pilot[j])][1]["model"], cond[j])[0])
                src = "pilot"
            elif using[j] >= 0 and using[j] in prods:
                s = float(self.advise(prods[int(using[j])][1]["model"], cond[j])[0])
                src = "product"
            elif tuning[j] < 0 and rng.random() < float(self.c["tune_p"]):
                tuning[j] = float(np.clip(s + rng.normal(0, 0.12), 0, 1))
                tune_ref[j] = float(np.mean(recent[j][recent[j] > 0])) if np.any(recent[j] > 0) else 0.0
            if src == "own" and tuning[j] >= 0:
                s = float(tuning[j])
            pt = {"setting": s, **{k: float(v) for k, v in zip(self.cond, cond[j])}}
            L = float(self.law([pt])[0]) * float(max(0.5, rng.normal(1.0, float(self.c["noise"]))))
            cost = int(round(float(base[j]) * L))
            try:
                P.pay(ctx, i, util_acct, cost, memo="weekly bill")
                spent[j] += cost
            except CrucibleError:
                pass
            weeks[j] += 1
            recent[j] = np.roll(recent[j], 1)
            recent[j][0] = L
            # ---- own trial and error: keep a trial setting if this week's bill beat the recent average
            if src == "own" and tuning[j] >= 0:
                if tune_ref[j] > 0 and L < tune_ref[j]:
                    setting[j] = tuning[j]
                tuning[j] = -1.0
            # ---- products: trial, judgement on own bills, renewal
            if src == "product":
                with_sum[j] += L
                with_n[j] += 1
                k = int(using[j])
                pid, prod = prods[k]
                if now >= trial_until[j] and now >= paid_until[j]:
                    mean_with = with_sum[j] / max(1, with_n[j])
                    saving = (float(before[j]) - mean_with) * float(base[j]) * 4.3
                    worth = saving >= prod["price"] * (1.0 + 0.8 * price_sens[jj])
                    if worth:
                        try:
                            P.pay(ctx, i, bank.primary_account(st, prod["owner"]), int(prod["price"]),
                                  memo=f"subscription {pid}")
                            paid_until[j] = now + 30 * DAY
                            kept_ok[j] = k
                            subs_ev.append([i, pid, "renew", int(prod["price"]), round(saving, 1)])
                        except CrucibleError:
                            worth = False
                    if not worth:
                        subs_ev.append([i, pid, "cancel", 0, round(saving, 1)])
                        using[j] = -1
                        kept_ok[j] = -1
            elif using[j] >= 0 and int(using[j]) not in prods:
                # the product is not running (suspended): the user falls back on its own practice, and once the
                # trial or the month it paid for has run out its subscription lapses (it stops paying and is free
                # to try another product)
                if now >= trial_until[j] and now >= paid_until[j]:
                    pid_k = slots.get(int(using[j]))
                    if pid_k is not None:
                        subs_ev.append([i, pid_k, "lapse", 0, 0.0])
                    using[j] = -1
                    kept_ok[j] = -1
            elif using[j] < 0 and pilot[j] < 0:
                # learn of products: a contact who kept one; then maybe try one it knows
                if rng.random() < 0.2:
                    cs = contacts[i][contacts[i] >= 0]
                    if cs.size:
                        kk = int(kept_all[int(cs[int(rng.integers(0, cs.size))])])
                        if kk >= 0 and kk not in aware[j]:
                            free = np.flatnonzero(aware[j] < 0)
                            if free.size:
                                aware[j, int(free[0])] = kk
                known = [int(x) for x in aware[j] if x >= 0 and int(x) in prods]
                hist = recent[j][recent[j] > 0]
                if known and hist.size >= 2:
                    mean = float(np.mean(hist))
                    p_try = 0.15 * (0.5 + novelty[jj]) * min(2.0, max(0.0, (mean - 1.0) / 0.15))
                    if rng.random() < p_try:
                        k = known[int(rng.integers(0, len(known)))]
                        using[j] = k
                        trial_until[j] = now + 28 * DAY
                        paid_until[j] = 0
                        before[j] = mean
                        with_sum[j] = 0.0
                        with_n[j] = 0
                        subs_ev.append([i, prods[k][0], "trial", 0, 0.0])
            nxt[j] = now + 7 * DAY
            util[j] = float(np.clip(1.2 - L, -1.0, 1.0))
            recs.append([i, src, round(s, 4), round(L, 4), cost])
        K.decide(ctx, "rc.practice", idx, {
            "setting": setting, "next_week": nxt, "recent": recent, "tuning": tuning, "tune_ref": tune_ref,
            "aware": aware, "using": using, "trial_until": trial_until, "before": before, "with_sum": with_sum,
            "with_n": with_n, "paid_until": paid_until, "kept_ok": kept_ok, "weeks": weeks, "spent": spent},
            table=TABLE, model_id=MODEL_ID, fidelity_id=FIDELITY,
            extra={"t": int(now), "weeks": recs, "subscriptions": subs_ev})
        for i, pid, kind, cents, saving in subs_ev:
            # the operator sees its own subscription events (who started, renewed, cancelled and what it was paid);
            # the user's own bills and savings stay the user's (they are in the decision record, evaluator side)
            ctx.emit("rc.sub", {"id": f"{pid}|p{i}|{int(now)}", "product": pid, "user": f"p{i}", "kind": kind,
                                "cents": cents}, vis=(st.tables["rc_products"][pid]["owner"],))
        attention.spend(ctx, idx[due], np.full(due.size, 3.0), activity="practice")
        return util

    # ------------------------------------------------------------------ founder's tools
    def _pay(self, ctx, to: str, cents: int, kind: str, memo: str) -> None:
        bank = ctx.world.modules["bank"]
        st = ctx.state
        src = bank.primary_account(st, ctx.actor)
        if src is None or bank.available(st, src) < cents:
            raise InvalidAction(f"this costs {cents} cents")
        bank.move(ctx, src, bank.primary_account(st, to), int(cents), kind, memo)

    @endpoint("prototype", S.obj({"name": S.s_str(minLength=1, maxLength=80),
                                  "model": S.s_str("the setting to advise, as an expression over the condition "
                                                   "variables (law language)", maxLength=1000)},
                                 ["name", "model"]), cost_s=3600)
    def ep_prototype(self, ctx, name, model):
        """Have engineers build an advisor prototype from a model (paid now; ready after the development time).
        The model maps a person's circumstances to the setting it advises."""
        try:
            compile_expr(model, self.cond)
        except LawError as e:
            raise InvalidAction(f"model: {e}") from None
        self._pay(ctx, "o:devshop", int(self.c["dev_cents"]), "purchase", f"prototype {name}")
        pid = ctx.state.new_id("rcproto")
        ctx.emit("rc.prototype", {"id": pid, "owner": ctx.actor, "name": name, "model": model,
                                  "ready_t": int(ctx.now + float(self.c["dev_days"]) * DAY)}, vis="public")
        return {"prototype": pid, "ready_in_days": self.c["dev_days"]}

    def _proto(self, ctx, pid: str) -> dict:
        p = ctx.state.get("rc_protos", pid)
        if p is None or p["owner"] != ctx.actor:
            raise NotFound(f"no prototype {pid} of yours")
        if p["ready_t"] > ctx.now:
            raise Conflict(f"{pid} is still being built")
        return p

    @endpoint("pilot", S.obj({"prototype": S.s_str(maxLength=40), "users": S.s_int(minimum=5, maximum=60)},
                             ["prototype", "users"]), cost_s=3600)
    def ep_pilot(self, ctx, prototype, users):
        """Field-test a prototype: volunteers of the population (paid per user) use its advice for three weeks;
        you then receive each volunteer's own bills before and during the pilot (`rc.pilot_results`)."""
        p = self._proto(ctx, prototype)
        st = ctx.state
        from .. import population as P
        n = P.n_individuals(ctx.world)
        busy = columns.col(ctx.world, TABLE, "using") >= 0
        busy |= columns.col(ctx.world, TABLE, "pilot") >= 0
        free = np.flatnonzero(~busy)
        if free.size < users:
            raise Conflict("not enough volunteers")
        self._pay(ctx, "o:devshop", int(users) * int(self.c["pilot_cents_user"]), "purchase",
                  f"pilot of {prototype}")
        rows = np.sort(ctx.world.rng_for("rc-pilot", prototype, ctx.now).choice(free, size=int(users), replace=False))
        slot = sum(1 for _ in st.scan("rc_pilots"))
        plid = st.new_id("rcpilot")
        before = columns.col(ctx.world, TABLE, "recent")[rows].copy()
        ctx.emit("rc.pilot", {"id": plid, "owner": ctx.actor, "prototype": prototype, "model": p["model"],
                              "users": [int(r) for r in rows], "slot": slot, "end_t": int(ctx.now + 21 * DAY),
                              "before": [round(float(np.mean(b[b > 0])) if np.any(b > 0) else 0.0, 4)
                                         for b in before]}, vis=(ctx.actor,))
        columns.batch(ctx, TABLE, "rc.pilot_join", rows, {"pilot": np.full(rows.size, slot, np.int16)},
                      extra={"pilot": plid})
        return {"pilot": plid, "users": int(users), "ends_in_days": 21}

    @endpoint("pilot_results", S.obj({"pilot": S.s_str(maxLength=40)}, ["pilot"]), read_only=True)
    def ep_pilot_results(self, ctx, pilot):
        """The volunteers' own weekly bills (cost index) before and during a finished pilot."""
        p = ctx.state.get("rc_pilots", pilot)
        if p is None or p["owner"] != ctx.actor:
            raise NotFound(f"no pilot {pilot} of yours")
        if p["status"] != "done":
            return {"pilot": pilot, "status": p["status"]}
        return {"pilot": pilot, "status": "done", "before": p["before"], "during": p["during"],
                "mean_before": round(float(np.mean(p["before"])), 4), "mean_during": round(float(np.mean(p["during"])), 4)}

    @endpoint("launch", S.obj({"prototype": S.s_str(maxLength=40), "name": S.s_str(minLength=1, maxLength=80),
                               "price_cents_month": S.s_int(minimum=100, maximum=100_000)},
                              ["prototype", "name", "price_cents_month"]), cost_s=3600)
    def ep_launch(self, ctx, prototype, name, price_cents_month):
        """Deploy a prototype as a subscription advisor (a free 4-week trial, then monthly payments while the user
        keeps it). Hosting is charged monthly per product and per subscriber."""
        p = self._proto(ctx, prototype)
        st = ctx.state
        if any(r["status"] == "live" for r in st.find("rc_products", "owner", ctx.actor)):
            raise Conflict("you already run a live product (use rc.update)")
        slot = sum(1 for _ in st.scan("rc_products"))
        if slot >= P_MAX:
            raise Conflict("the catalogue is full")
        pid = st.new_id("rcprod")
        ctx.emit("rc.product", {"id": pid, "owner": ctx.actor, "name": name, "model": p["model"],
                                "prototype": prototype, "price": int(price_cents_month), "slot": slot}, vis="public")
        return {"product": pid}

    @endpoint("update", S.obj({"product": S.s_str(maxLength=40), "prototype": S.s_str(maxLength=40),
                               "price_cents_month": S.s_int(minimum=100, maximum=100_000)}, ["product"]), cost_s=600)
    def ep_update(self, ctx, product, prototype=None, price_cents_month=None):
        """Ship another prototype's model in your live product and/or change its price."""
        r = ctx.state.get("rc_products", product)
        if r is None or r["owner"] != ctx.actor:
            raise NotFound(f"no product {product} of yours")
        ch = {}
        if prototype is not None:
            ch["model"] = self._proto(ctx, prototype)["model"]
            ch["prototype"] = prototype
        if price_cents_month is not None:
            ch["price"] = int(price_cents_month)
        if not ch:
            raise InvalidAction("nothing to change")
        ctx.emit("rc.product_update", {"id": product, "changes": ch}, vis="public")
        return {"ok": True}

    @endpoint("advertise", S.obj({"product": S.s_str(maxLength=40), "cents": S.s_int(minimum=100, maximum=10**8)},
                                 ["product", "cents"]), cost_s=60)
    def ep_advertise(self, ctx, product, cents):
        """Promote your live product: each reached individual (cents / price per reach, drawn at random) learns of it."""
        st = ctx.state
        r = st.get("rc_products", product)
        if r is None or r["owner"] != ctx.actor or r["status"] != "live":
            raise NotFound(f"no live product {product} of yours")
        self._pay(ctx, "o:adnet", int(cents), "advertising", f"promotion of {product}")
        from .. import population as P
        n = P.n_individuals(ctx.world)
        reach = min(n, int(cents) // int(self.c["promote_cents_per_reach"]))
        rows = np.sort(ctx.world.rng_for("rc-ad", product, ctx.now).choice(n, size=reach, replace=False))
        aw = columns.col(ctx.world, TABLE, "aware")[rows].copy()
        for j in range(rows.size):
            if r["slot"] not in aw[j]:
                free = np.flatnonzero(aw[j] < 0)
                if free.size:
                    aw[j, int(free[0])] = r["slot"]
        columns.batch(ctx, TABLE, "rc.promotion", rows, {"aware": aw}, extra={"product": product, "cents": int(cents)})
        ctx.emit("rc.ad", {"id": st.new_id("rcad"), "product": product, "cents": int(cents), "reach": int(reach),
                           "by": ctx.actor}, vis="public")
        return {"reached": int(reach)}

    @endpoint("investors", S.obj(), read_only=True)
    def ep_investors(self, ctx):
        """Investors, what they look for and what they have left to invest."""
        bank = ctx.world.modules["bank"]
        st = ctx.state
        out = []
        for org, r in sorted(st.scan("rc_investors")):
            out.append({"investor": org, "name": r["name"], "max_ticket_cents": r["max_ticket"],
                        "looks_for": {"validated": "a published model of the phenomenon with a consistent independent "
                                                   "replication, and a built prototype",
                                      "traction": f"at least {r.get('min_subscribers', 0)} paying subscribers who "
                                                  f"renewed, and monthly revenue above hosting"}[r["rule"]],
                        "available_cents": int(bank.available(st, bank.primary_account(st, org)))})
        return {"investors": out}

    @endpoint("pitch", S.obj({"investor": S.s_str(maxLength=40), "amount_cents": S.s_int(minimum=10_000,
                                                                                        maximum=10**9),
                              "equity_bps": S.s_int("share of your venture offered, basis points", minimum=1,
                                                    maximum=5000)}, ["investor", "amount_cents", "equity_bps"]),
              cost_s=3600)
    def ep_pitch(self, ctx, investor, amount_cents, equity_bps):
        """Ask an investor for money against equity. It decides within two days from the public record."""
        st = ctx.state
        if not st.has("rc_investors", investor):
            raise NotFound(f"no investor {investor}")
        if any(p["status"] == "open" for p in st.find("rc_pitches", "investor", investor) if p["by"] == ctx.actor):
            raise Conflict("you already have an open pitch with this investor")
        pid = st.new_id("rcpitch")
        ctx.emit("rc.pitch", {"id": pid, "investor": investor, "by": ctx.actor, "amount": int(amount_cents),
                              "equity_bps": int(equity_bps), "decide_t": int(ctx.now + 2 * DAY)}, vis=(ctx.actor,))
        return {"pitch": pid}

    @endpoint("status", S.obj(), read_only=True)
    def ep_status(self, ctx):
        """Your venture: cash, prototypes, pilots, products (subscribers, trials, revenue), pitches and equity."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        me = ctx.actor
        prods = []
        for pid, r in sorted(st.scan("rc_products")):
            if r["owner"] != me:
                continue
            subs = [s for s in st.find("rc_subs", "product", pid)]
            prods.append({"product": pid, "status": r["status"], "price": r["price"], "prototype": r["prototype"],
                          "paying": sum(1 for s in subs if s["status"] == "paying"),
                          "trials": sum(1 for s in subs if s["status"] == "trial"),
                          "cancelled": sum(1 for s in subs if s["status"] == "cancelled"),
                          "lapsed": sum(1 for s in subs if s["status"] == "lapsed"),
                          "revenue_cents": int(r["revenue"]), "hosting_paid_cents": int(r["hosting"])})
        return {"cash_cents": int(bank.available(st, bank.primary_account(st, me))),
                "prototypes": [{"prototype": k, "name": p["name"], "ready": p["ready_t"] <= ctx.now}
                               for k, p in sorted(st.scan("rc_protos")) if p["owner"] == me],
                "pilots": [{"pilot": k, "status": p["status"]} for k, p in sorted(st.scan("rc_pilots"))
                           if p["owner"] == me],
                "products": prods,
                "pitches": [{"pitch": k, "investor": p["investor"], "status": p["status"], "amount": p["amount"]}
                            for k, p in sorted(st.scan("rc_pitches")) if p["by"] == me],
                "equity_sold_bps": sum(e["bps"] for _, e in st.scan("rc_equity") if e["venture"] == me)}

    # ------------------------------------------------------------------ daily: pilots, hosting, investors
    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        for plid, p in sorted(st.scan("rc_pilots")):
            if p["status"] == "running" and ctx.now >= p["end_t"]:
                rows = np.asarray(p["users"], np.int64)
                rec = columns.col(ctx.world, TABLE, "recent")[rows]
                during = [round(float(np.mean(r[:3][r[:3] > 0])) if np.any(r[:3] > 0) else 0.0, 4) for r in rec]
                ctx.emit("rc.pilot_done", {"id": plid, "during": during}, vis=(p["owner"],))
                columns.batch(ctx, TABLE, "rc.pilot_end", rows, {"pilot": np.full(rows.size, -1, np.int16)},
                              extra={"pilot": plid})
        if (ctx.now // DAY) % 30 == 0:
            for pid, r in sorted(st.scan("rc_products")):
                if r["status"] not in ("live", "suspended"):
                    continue
                n = sum(1 for s in st.find("rc_subs", "product", pid) if s["status"] in ("paying", "trial"))
                cents = int(self.c["hosting_cents_month"]) + n * int(self.c["hosting_cents_user_month"])
                bank = ctx.world.modules["bank"]
                src = bank.primary_account(st, r["owner"])
                if bank.available(st, src) >= cents:
                    bank.move(ctx, src, bank.primary_account(st, "o:hosting"), cents, "hosting", f"hosting {pid}")
                    ctx.emit("rc.hosting", {"id": pid, "cents": cents, "users": n}, vis=(r["owner"],))
                    if r["status"] == "suspended":
                        ctx.emit("rc.product_update", {"id": pid, "changes": {"status": "live"}}, vis="public")
                else:
                    ctx.emit("rc.product_update", {"id": pid, "changes": {"status": "suspended"}}, vis="public")
        for pid, p in sorted(st.scan("rc_pitches")):
            if p["status"] == "open" and ctx.now >= p["decide_t"]:
                err = contain(ctx, self._decide, ctx, pid, p)
                if err is not None:
                    ctx.emit("rc.pitch_status", {"id": pid, "status": "declined", "why": str(err)[:200]},
                             vis=(p["by"],))

    def evidence(self, st, venture: str) -> dict:
        """The public record an investor reads about a venture."""
        papers = [k for k, p in st.scan("sci_papers") if p["author"] == venture and p["phenomenon"] == PHEN]
        # what anyone can see on the public record: reports by other labs, made on other apparatus than the paper's
        reps = [r for _, r in st.scan("sci_reps") if r["paper"] in papers and r["reporter"] != venture
                and r["other_apparatus"]]
        consistent = sum(1 for r in reps if r["verdict"] == "consistent")
        refuted = sum(1 for r in reps if r["verdict"] == "refuted")
        protos = sum(1 for _, p in st.scan("rc_protos") if p["owner"] == venture and p["ready_t"] <= self.world.now)
        paying = 0
        mine = set()
        for pid, r in st.scan("rc_products"):
            if r["owner"] == venture:
                mine.add(pid)
                paying += sum(1 for s in st.find("rc_subs", "product", pid) if paying_now(s, self.world.now))
        t0 = self.world.now - 30 * DAY
        revenue = sum(x["cents"] for x in st.find("rc_ledger", "kind", "revenue") if x["product"] in mine and x["t"] >= t0)
        hosting = sum(x["cents"] for x in st.find("rc_ledger", "kind", "hosting") if x["product"] in mine and x["t"] >= t0)
        return {"papers": len(papers), "consistent_replications": consistent, "refuted_replications": refuted,
                "prototypes": protos, "renewed_subscribers": paying, "revenue_30d": revenue, "hosting_30d": hosting}

    def _decide(self, ctx, pid, p):
        st = ctx.state
        inv = st.tables["rc_investors"][p["investor"]]
        ev = self.evidence(st, p["by"])
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, p["investor"])
        avail = bank.available(st, src)
        if inv["rule"] == "validated":
            ok = ev["consistent_replications"] >= 1 and ev["consistent_replications"] > ev["refuted_replications"] \
                and ev["prototypes"] >= 1
        else:
            ok = ev["renewed_subscribers"] >= int(inv.get("min_subscribers", 0)) and ev["revenue_30d"] > ev["hosting_30d"]
        want_bps = int(math.ceil(p["amount"] / 100_000 * float(inv["equity_per_100k_bps"])))
        ok = ok and p["amount"] <= min(int(inv["max_ticket"]), int(avail)) and p["equity_bps"] >= want_bps
        if not ok:
            ctx.emit("rc.pitch_status", {"id": pid, "status": "declined", "why": "the record does not meet our rule",
                                         "evidence": ev}, vis=(p["by"],))
            return
        dst = bank.primary_account(st, p["by"])
        bank.move(ctx, src, dst, int(p["amount"]), "investment", f"equity investment {pid}", ref=pid)
        ctx.emit("rc.pitch_status", {"id": pid, "status": "funded", "why": "meets our rule", "evidence": ev},
                 vis="public")
        ctx.emit("rc.equity", {"id": pid, "venture": p["by"], "holder": p["investor"], "bps": int(p["equity_bps"]),
                               "amount": int(p["amount"])}, vis="public")

    # ------------------------------------------------------------------ appliers
    @applier("rc.investor")
    def ap_investor(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("rc_investors", p.pop("id"), p)

    @applier("rc.misc")
    def ap_misc(self, ev):
        p = dict(ev.payload)
        self.world.state.put("rc_misc", p.pop("id"), p)

    @applier("rc.prototype")
    def ap_prototype(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("rc_protos", p.pop("id"), p | {"t": ev.t})

    @applier("rc.pilot")
    def ap_pilot(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("rc_pilots", p.pop("id"), p | {"status": "running", "during": [], "t": ev.t})

    @applier("rc.pilot_done")
    def ap_pilot_done(self, ev):
        self.world.state.patch("rc_pilots", ev.payload["id"], status="done", during=list(ev.payload["during"]))

    @applier("rc.product")
    def ap_product(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("rc_products", p.pop("id"), p | {"status": "live", "revenue": 0, "hosting": 0,
                                                                 "revenue_30d": 0, "hosting_30d": 0, "t": ev.t})

    @applier("rc.product_update")
    def ap_product_update(self, ev):
        self.world.state.patch("rc_products", ev.payload["id"], **ev.payload["changes"])

    @applier("rc.sub")
    def ap_sub(self, ev):
        p = ev.payload
        st = self.world.state
        key = f"{p['product']}|{p['user']}"
        cur = st.get("rc_subs", key)
        status = {"trial": "trial", "renew": "paying", "cancel": "cancelled", "lapse": "lapsed"}[p["kind"]]
        renewals = (cur["renewals"] if cur else 0) + (1 if p["kind"] == "renew" else 0)
        paid_until = ev.t + 30 * DAY if p["kind"] == "renew" else (cur or {}).get("paid_until", 0)
        st.put("rc_subs", key, {"product": p["product"], "user": p["user"], "status": status, "renewals": renewals,
                                "t": ev.t, "first_t": cur["first_t"] if cur else ev.t, "paid_until": int(paid_until)})
        if p["kind"] == "renew":
            r = st.tables["rc_products"][p["product"]]
            st.patch("rc_products", p["product"], revenue=r["revenue"] + int(p["cents"]))
            st.insert("rc_ledger", f"rev|{p['product']}|{p['user']}|{ev.seq}", {"kind": "revenue", "product": p["product"],
                                                                             "cents": int(p["cents"]), "t": ev.t})

    @applier("rc.hosting")
    def ap_hosting(self, ev):
        st = self.world.state
        r = st.tables["rc_products"][ev.payload["id"]]
        st.patch("rc_products", ev.payload["id"], hosting=r["hosting"] + int(ev.payload["cents"]))
        st.insert("rc_ledger", f"host|{ev.payload['id']}|{ev.seq}", {"kind": "hosting", "product": ev.payload["id"],
                                                                      "cents": int(ev.payload["cents"]), "t": ev.t})

    @applier("rc.ad")
    def ap_ad(self, ev):
        p = dict(ev.payload)
        st = self.world.state
        st.put("rc_misc", p.pop("id"), p | {"t": ev.t, "kind": "ad"})
        st.insert("rc_ledger", f"ad|{p['product']}|{ev.seq}", {"kind": "advertising", "product": p["product"],
                                                                "cents": int(p["cents"]), "t": ev.t})

    @applier("rc.pitch")
    def ap_pitch(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("rc_pitches", p.pop("id"), p | {"status": "open", "t": ev.t})

    @applier("rc.pitch_status")
    def ap_pitch_status(self, ev):
        self.world.state.patch("rc_pitches", ev.payload["id"], status=ev.payload["status"])

    @applier("rc.equity")
    def ap_equity(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("rc_equity", p.pop("id"), p | {"t": ev.t})

    def check_invariants(self, world):
        errs = []
        from .. import population as P
        if columns.rows(world, TABLE) != P.n_individuals(world):
            errs.append("rc rows differ from the population")
        sold = {}
        for _, e in world.state.scan("rc_equity"):
            sold[e["venture"]] = sold.get(e["venture"], 0) + e["bps"]
        if any(v > 10_000 for v in sold.values()):
            errs.append("more than 100% of a venture sold")
        return errs


# ====================================================================== evaluator side (recorded log only)
def paying_now(sub: dict, now: int) -> bool:
    """A retained adopter: renewed at least once, still subscribed, and its last payment covers now (users settle
    on their weekly practice day, so a renewal is at most a week late)."""
    return sub["status"] == "paying" and sub["renewals"] >= 1 and int(sub.get("paid_until", 0)) + 7 * DAY >= int(now)


@K.cached_evaluator
def evaluate(world, principal: str, objective: dict) -> dict:
    st = world.state
    mod = world.modules["rcomm"]
    now = world.now
    ev = mod.evidence(st, principal)
    mine = {pid for pid, r in st.scan("rc_products") if r["owner"] == principal}
    rel: dict[str, bool] = {}

    def indep(u):
        if u not in rel:
            rel[u] = K.population_index(u) is not None and not K.related(world, principal, u)
        return rel[u]
    subs = [s for _, s in st.scan("rc_subs") if s["product"] in mine and indep(s["user"])]
    adopters = [s for s in subs if s["renewals"] >= 1]
    retained = [s for s in adopters if paying_now(s, now)]
    t0 = now - 30 * DAY
    led = [x for _, x in st.scan("rc_ledger") if x["product"] in mine and x["t"] >= t0]
    revenue = sum(x["cents"] for x in led if x["kind"] == "revenue")
    costs = sum(x["cents"] for x in led if x["kind"] in ("hosting", "advertising"))
    bank = world.modules["bank"]
    cash = int(bank.available(st, bank.primary_account(st, principal))) if bank.primary_account(st, principal) else 0
    papers = [k for k, p in st.scan("sci_papers") if p["author"] == principal and p["phenomenon"] == PHEN]
    reps = [r for _, r in st.scan("sci_reps") if r["paper"] in papers and r["independent"]]
    ind_ok = sum(1 for r in reps if r["verdict"] == "consistent")
    ind_bad = sum(1 for r in reps if r["verdict"] == "refuted")
    validated = ind_ok >= 1 and ind_ok > ind_bad
    need = int(objective.get("min_adopters", 30))
    checks = {"validated_discovery": validated, "adopters": len(retained) >= need,
              "sustainable": revenue > costs and revenue > 0, "solvent": cash > 0,
              "horizon": now >= int(objective.get("horizon_days", 100)) * DAY}
    score = float(np.mean([float(validated), min(1.0, len(retained) / max(1, need)),
                           1.0 if revenue > costs else 0.0, 1.0 if cash > 0 else 0.0]))
    return {"success": all(checks.values()), "score": round(score, 4),
            "details": {"papers": ev["papers"], "consistent_replications": ind_ok,
                        "refuted_replications": ind_bad, "trials": len(subs),
                        "adopters_renewed": len(adopters), "adopters_retained": len(retained),
                        "revenue_30d_cents": int(revenue), "costs_30d_cents": int(costs), "cash_cents": cash,
                        "funded_cents": int(sum(e["amount"] for _, e in st.scan("rc_equity")
                                                if e["venture"] == principal)),
                        "checks": checks}}


MECHANICS = ("Research", "Experimentation", "Technical development", "Product creation", "Financing", "Deployment",
             "Independent adoption", "Commercial sustainability")
PARTICIPANT_MECHANICS = MECHANICS          # all measured on the founder's own venture


def mechanics(world) -> dict:
    return mechanics_report(world, 0)


def mechanics_report(world, since_seq: int = 0) -> dict:
    """Each required mechanic of 9.15, measured on the founder's own venture (all are PARTICIPANT_MECHANICS; the
    rival labs' research and the population's own practice are the world it works in): its papers and their
    replications, its apparatus runs and pilots, its prototypes, products, financing, deployment (hosting and trials
    of its products), renewals of its products by independent people, and a 30-day window in which its products'
    revenue exceeded their hosting and promotion costs."""
    st = world.state
    me = set(K.seats_in(world, DOMAIN))
    prods = {pid for pid, r in st.scan("rc_products") if r["owner"] in me}
    papers: set[str] = set()
    n = {"sci.paper": 0, "sci.replication": 0, "sci.booked": 0, "rc.pilot": 0, "rc.pilot_done": 0, "rc.prototype": 0,
         "rc.product": 0, "rc.product_update": 0, "rc.pitch": 0, "rc.equity": 0, "rc.hosting": 0}
    pilots: set[str] = set()
    for ev in world.events_range(since_seq):
        p = ev.payload
        t = ev.type
        if t == "sci.paper" and p["author"] in me and p["phenomenon"] == PHEN:
            papers.add(p["id"])
            n[t] += 1
        elif t == "sci.replication" and p["paper"] in papers:
            n[t] += 1
        elif t == "sci.booked" and p["owner"] in me:
            n[t] += 1
        elif t == "rc.pilot" and p["owner"] in me:
            pilots.add(p["id"])
            n[t] += 1
        elif t == "rc.pilot_done" and p["id"] in pilots:
            n[t] += 1
        elif t in ("rc.prototype", "rc.product") and p["owner"] in me:
            n[t] += 1
        elif t in ("rc.product_update", "rc.hosting") and p["id"] in prods:
            n[t] += 1
        elif t == "rc.pitch" and p.get("by") in me:
            n[t] += 1
        elif t == "rc.equity" and p["venture"] in me:
            n[t] += 1
    subs = [x for x in K.batch_records(world, "rc.practice", "subscriptions", since_seq) if x[1] in prods]
    rel: dict[str, bool] = {}

    def indep(u):
        if u not in rel:
            rel[u] = all(not K.related(world, a, u) for a in me)
        return rel[u]
    renews = [x for x in subs if x[2] == "renew" and indep(f"p{x[0]}")]
    trials = [x for x in subs if x[2] == "trial"]
    # commercial sustainability: a 30-day window (ending on a day boundary) with revenue above costs
    led = [x for _, x in st.scan("rc_ledger") if x["product"] in prods]
    windows = 0
    if led:
        last = int(world.now // DAY)
        for d in range(30, last + 1):
            t1, t0 = d * DAY, (d - 30) * DAY
            rev = sum(x["cents"] for x in led if x["kind"] == "revenue" and t0 <= x["t"] < t1)
            cost = sum(x["cents"] for x in led if x["kind"] in ("hosting", "advertising") and t0 <= x["t"] < t1)
            if rev > 0 and rev > cost:
                windows += 1
    return {
        "Research": {"exercised": n["sci.paper"] > 0 and n["sci.replication"] > 0,
                     "events": {"sci.paper": n["sci.paper"], "sci.replication": n["sci.replication"]}},
        "Experimentation": {"exercised": n["sci.booked"] > 0 and (n["rc.pilot"] > 0 or n["sci.replication"] > 0),
                            "events": {"sci.booked": n["sci.booked"], "rc.pilot": n["rc.pilot"],
                                       "rc.pilot_done": n["rc.pilot_done"]}},
        "Technical development": {"exercised": n["rc.prototype"] > 0, "events": {"rc.prototype": n["rc.prototype"]}},
        "Product creation": {"exercised": n["rc.product"] > 0, "events": {"rc.product": n["rc.product"],
                                                                          "rc.product_update": n["rc.product_update"]}},
        "Financing": {"exercised": n["rc.equity"] > 0, "events": {"rc.pitch": n["rc.pitch"], "rc.equity": n["rc.equity"]}},
        "Deployment": {"exercised": n["rc.hosting"] > 0 and len(trials) > 0,
                       "events": {"rc.hosting": n["rc.hosting"], "trials": len(trials)}},
        "Independent adoption": {"exercised": len(renews) > 0, "events": {"independent_renewals": len(renews),
                                                                          "trials": len(trials)}},
        "Commercial sustainability": {"exercised": windows > 0,
                                      "events": {"profitable_30d_windows": windows}},
    }


# ====================================================================== the domain
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, register  # noqa: E402

A = K.A

POPULATION = 600
STRUCTURES = {
    "linear_optimum": {"family": "linear"},
    "saturating_optimum": {"family": "saturating", "base_cents_week": [3000, 9000]},
    "interaction_optimum": {"family": "interaction", "own_spread": 0.1,
                            "investors": [{"handle": "grants", "name": "Science Grants Council", "budget": 400_000,
                                           "rule": "validated", "max_ticket": 250_000, "equity_per_100k_bps": 1},
                                          {"handle": "growthfund", "name": "Growth Fund", "budget": 2_000_000,
                                           "rule": "traction", "max_ticket": 800_000, "equity_per_100k_bps": 250,
                                           "min_subscribers": 25}]},
    "saturating_grants": {"family": "saturating",
                          "investors": [{"handle": "grants", "name": "Science Grants Council", "budget": 400_000,
                                         "rule": "validated", "max_ticket": 250_000, "equity_per_100k_bps": 1},
                                        {"handle": "growthfund", "name": "Growth Fund", "budget": 2_000_000,
                                         "rule": "traction", "max_ticket": 800_000, "equity_per_100k_bps": 250,
                                         "min_subscribers": 25}]},
    "two_conditions": {"family": "two_conditions"},
}


def _structure(name):
    s = STRUCTURES[name]
    inv = s.get("investors", DEFAULTS["investors"])
    return {"law_family": s["family"], "conditions": 2 if s["family"] == "two_conditions" else 1,
            "financing": sorted(i["name"] for i in inv),
            "participants": ["population practitioners", "rival and replication labs", "investors", "apparatus operators"]}


CONFIGS = (
    DomainConfig("linear_optimum", "train", _structure("linear_optimum"),
                 "the best setting rises linearly with a person's circumstances; angel and growth investors"),
    DomainConfig("saturating_optimum", "train", _structure("saturating_optimum"),
                 "the best setting saturates with circumstances; bills are larger"),
    DomainConfig("interaction_optimum", "train", _structure("interaction_optimum"),
                 "setting and circumstances interact multiplicatively; habits scatter more; a grants council "
                 "funds validated science instead of angels"),
    DomainConfig("saturating_grants", "eval_compositional", _structure("saturating_grants"),
                 "a saturating law (seen in training) financed by grants and growth equity (seen in training, never "
                 "with this law)"),
    DomainConfig("two_conditions", "eval_structural", _structure("two_conditions"),
                 "the best setting depends on two circumstances (a three-variable law never seen in training)"),
)


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    ov = dict(overrides or {})
    rng = K.seeded_rng(config.name, seed)
    inst = generate_instance(rng, {**STRUCTURES[config.name], **(ov.pop("instance", {}) or {})})
    n = int(ov.pop("population", POPULATION))
    mods = K.order_modules([("identity", {}), ("bank", {}), ("admin", {}), ("artifacts", {}), ("cols", {}),
                            ("docs", {}), ("sci", sci_config(inst, rng)), ("pop", K.pop_config(n)), ("rcomm", inst)])
    return K.recipe(mods, {"max_actions_per_hour": 600})


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = max(1, int(difficulty))
    need = 20 + 10 * d
    text = (f"Turn the science of this world's practice phenomenon into a product people adopt on "
                           f"their own: by day 100 your published model of the phenomenon must be confirmed by at "
                           f"least one independent replication (and confirmed more often than refuted), at least "
                           f"{need} people outside your venture must pay for your product, having renewed at least "
                           f"once, its revenue over the last 30 days must exceed its hosting and promotion costs, and "
                           f"you must stay solvent.")
    return {"text": text, "description": text, "difficulty": d, "horizon_days": 100, "min_adopters": need}


def _powers(C: np.ndarray, deg: int) -> np.ndarray:
    feats = [np.ones(len(C))]
    for j in range(C.shape[1]):
        feats += [C[:, j] ** e for e in range(1, int(deg) + 1)]
    return np.stack(feats, 1)


def _poly_text(c, cond_names: list[str], deg: int) -> str:
    t = [f"{c[0]:.6f}"]
    for j, cn in enumerate(cond_names):
        for e in range(1, int(deg) + 1):
            x = c[1 + j * int(deg) + e - 1]
            t.append(f"{x:.6f} * {cn}" if e == 1 else f"{x:.8f} * {cn} ** {e}")
    return " + ".join(t)


def fit_optimum(rows: list[tuple[dict, float]], cond_names: list[str], deg_g: int = 2,
                deg_h: int = 2, deg_k: int = 0) -> tuple[str, str, dict]:
    """Least-squares fit of response = h(c) + k(c) (setting - g(c))^2 with h, k and g polynomials in the conditions
    (of degrees `deg_h`, `deg_k`, `deg_g`; alternating: linear least squares for h and k given g, Gauss-Newton steps
    for g): returns (law expression for the paper, advice expression g clipped to the setting range, fit info).
    Uses only the founder's own readings."""
    X = np.array([[r["setting"]] + [r[c] for c in cond_names] for r, _ in rows], np.float64)
    y = np.array([v for _, v in rows], np.float64)
    s, C = X[:, 0], X[:, 1:]
    Fg, Fh, Fk = _powers(C, deg_g), _powers(C, deg_h), _powers(C, deg_k)
    beta = np.zeros(Fg.shape[1])
    beta[0] = 0.5
    best = None
    for _ in range(80):
        g = Fg @ beta
        A_ = np.concatenate([Fh, Fk * ((s - g) ** 2)[:, None]], 1)
        coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
        h, kc = coef[:Fh.shape[1]], coef[Fh.shape[1]:]
        kv = np.maximum(Fk @ kc, 1e-3)
        r = y - (Fh @ h + kv * (s - g) ** 2)
        sse = float(np.sum(r ** 2))
        if best is None or sse < best[0]:
            best = (sse, h.copy(), kc.copy(), beta.copy())
        J = (-2.0 * kv * (s - g))[:, None] * Fg
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        beta = beta - 0.7 * step
    sse, h, kc, beta = best
    n, npar = len(rows), Fg.shape[1] + Fh.shape[1] + Fk.shape[1]
    kt = f"{max(float(kc[0]), 1e-3):.6f}" if deg_k == 0 else f"({_poly_text(kc, cond_names, deg_k)})"
    law = f"{_poly_text(h, cond_names, deg_h)} + {kt} * (setting - ({_poly_text(beta, cond_names, deg_g)})) ** 2"
    bic = float(n * math.log(max(sse, 1e-12) / n) + npar * math.log(n))
    return law, f"clip({_poly_text(beta, cond_names, deg_g)}, 0, 1)", {"sse": sse, "n": n, "k": float(kc[0]),
                                                                      "params": npar, "bic": bic,
                                                                      "deg_g": int(deg_g), "deg_h": int(deg_h),
                                                                      "deg_k": int(deg_k)}


def select_optimum_model(rows: list[tuple[dict, float]], cond_names: list[str]) -> tuple[str, str, dict]:
    """A scientist's model choice from its own readings: fit the optimum model with polynomial g, h and curvature k
    of degrees 1-5 (g; at most the distinct levels of each condition minus one), 1-2 (h) and 0-2 (k) and keep the
    one with the lowest Bayesian information criterion (fewer parameters unless the data demand more)."""
    levels = min(len({round(float(r[c]), 6) for r, _ in rows}) for c in cond_names) if cond_names else 1
    gmax = max(1, min(5, levels - 1))
    fits = [fit_optimum(rows, cond_names, dg, dh, dk) for dg in range(1, gmax + 1) for dh in (1, 2)
            for dk in (0, 1, 2)]
    return min(fits, key=lambda f: (f[2]["bic"], f[2]["params"]))


class ScienceFounder(K.ReadingReference):
    """Scripted reference (evaluator side, tool-only): a scientist-founder. It measures the phenomenon on the
    cheaper apparatus over a grid of settings and circumstances (with the certified standards, to correct the
    apparatus' calibration), fits the optimum setting as a function of the circumstances from its own readings
    (polynomial model families chosen by the Bayesian information criterion),
    publishes the model (and, when the first replications do not confirm it, a revised paper from twice the
    readings), builds an advisor from the fitted optimum, pilots it, pitches the investor that funds
    validated science once an independent replication confirms the paper, launches a subscription, promotes it and
    pitches the growth investor once subscribers renewed. `discovery=False` is the counterfactual founder without the
    research: it builds the advisor from common practice (`common_practice`, the population's habitual setting) and
    otherwise acts alike."""
    cycle_s = 12 * HOUR
    discovery = True
    common = 0.5

    def __init__(self, principal, objective, discovery=None, common_practice=None):
        super().__init__(principal, objective)
        if discovery is not None:
            self.discovery = bool(discovery)
        if common_practice is not None:
            self.common = float(common_practice)
        self.stage = "start"
        self.run = self.paper = self.proto = self.pilot = self.product = self.advice = None
        # a follow-up study when the first paper's replications do not confirm it: more readings, the model refit
        # on all of them, a second paper for other labs to replicate
        self.papers: list[str] = []
        self.rows: list = []
        self.std: list = []
        self.design: list = []
        self.run2 = None
        self.followup = None
        self.app = None
        self.certified: dict[str, float] = {}
        self.price = 900
        self.pitched: set[str] = set()
        self.ads = 0

    def reads(self, t):
        if self.stage == "start":
            return [] if not self.discovery else [("phen", "sci.phenomena", {}),
                                                  ("apps", "sci.apparatus", {"phenomenon": PHEN})]
        if self.stage == "measuring":
            return [("data", "sci.data", {"run": self.run})] if self.run else []
        out = [("status", "rc.status", {}), ("inv", "rc.investors", {}), ("reps", "sci.replications", {})]
        if self.followup == "measuring" and self.run2:
            out.append(("data2", "sci.data", {"run": self.run2}))
        if self.pilot:
            out.append(("pilot", "rc.pilot_results", {"pilot": self.pilot}))
        return out

    def _my_reports(self, got) -> list:
        reps = got.get("reps")
        return [r for r in (reps if isinstance(reps, list) else []) if r.get("paper") in self.papers and
                r.get("reporter") != self.me and r.get("other_apparatus")]

    def _replicated(self, got) -> bool:
        mine = self._my_reports(got)
        return sum(r["verdict"] == "consistent" for r in mine) > sum(r["verdict"] == "refuted" for r in mine)

    def _calibrate(self, std):
        xs = np.array([self.certified[r["standard"]] for r in std])
        ys = np.array([float(r["value"]) for r in std])
        return np.polyfit(xs, ys, 1) if len(set(np.round(xs, 6))) >= 2 else (1.0, 0.0)

    def _followup(self, got) -> list:
        """When the reports on the paper(s) so far do not confirm them (no more consistent than refuted reports
        once two have come in), measure as much again on the same apparatus, refit the model on all readings and
        publish it as a second paper (once)."""
        if not self.discovery or not self.paper:
            return []
        if self.followup is None:
            mine = self._my_reports(got)
            if len(mine) >= 2 and not self._replicated(got) and self.design:
                self.followup = "starting"
                return [A("sci.run", "run2", apparatus=self.app, design=self.design)]
            return []
        if self.followup == "measuring":
            d = got.get("data2")
            if not d or not d.get("readings"):
                return []
            pts = [r for r in d["readings"] if not r.get("standard") and r.get("value") is not None]
            std = [r for r in d["readings"] if r.get("standard") in self.certified and r.get("value") is not None]
            self.std += std
            gain, off = self._calibrate(self.std)
            cond = [n for n in (pts[0]["at"] if pts else {}) if n != "setting"]
            rows = [(r["at"], (float(r["value"]) - off) / gain) for r in self.rows + pts]
            law, _advice, _ = select_optimum_model(rows, cond)
            self.followup = "done"
            out = [A("sci.publish", "paper", phenomenon=PHEN, title="The optimal practice setting (revised)",
                     model=law, data=[self.run, self.run2], calibration_runs=[self.run, self.run2],
                     abstract="A larger study of the cost index over settings and circumstances: the optimum setting "
                              "as a function of the circumstances, refit on all readings.")]
            return out
        return []

    def _pitch(self, got, rule):
        for i in (got.get("inv") or {}).get("investors", []):
            if (rule == "validated") == ("replication" in i["looks_for"]) and i["investor"] not in self.pitched:
                amt = min(int(i["max_ticket_cents"]), int(i["available_cents"]))
                if amt < 10_000:
                    continue
                if rule == "validated" and not (self.paper and self._replicated(got)):
                    return None
                self.pitched.add(i["investor"])
                bps = int(math.ceil(amt / 100_000 * (900 if rule == "validated" else 300)))
                return A("rc.pitch", None, investor=i["investor"], amount_cents=amt, equity_bps=min(5000, max(1, bps)))
        return None

    def decide(self, t, got):
        out = []
        if self.stage == "start":
            if not self.discovery:
                self.advice = f"{self.common:.3f}"
                self.stage = "piloting"
                return [A("rc.prototype", "proto", name="Advisor v1", model=self.advice)]
            phs = got.get("phen") or []
            apps = got.get("apps") or []
            ph = next((p for p in phs if (p.get("phenomenon") or p.get("id")) == PHEN), None)
            if ph is None or not apps:
                return []
            app = sorted(apps, key=lambda a: a["price_cents"])[0]
            var = ph["vars"]
            # a design that fits the run's 240-entry limit with the standards included: one condition - 7 settings
            # x 7 circumstances, two readings each; two conditions - 6 x 5 x 4, one reading each; ten readings of
            # each certified standard, so that the calibration (gain and offset) is precise enough for the model to
            # survive other labs' replications on other apparatus
            two = len(var) > 2
            grids = []
            for j, v in enumerate(var):
                k = (6 if two else 7) if v["name"] == "setting" else ((5 if j == 1 else 4) if two else 7)
                grids.append([round(v["lo"] + (v["hi"] - v["lo"]) * i / (k - 1), 4) for i in range(k)])
            mesh = np.meshgrid(*grids, indexing="ij")
            pts = np.stack([m.ravel() for m in mesh], 1)
            reps = 1 if two else 2
            self.certified = {x["id"]: float(x["certified"]) for x in ph["standards"]}
            design = [{"standard": x["id"], "n": 10} for x in ph["standards"]]
            design += [{"at": {v["name"]: float(x) for v, x in zip(var, row)}, "n": reps} for row in pts]
            self.app = app["apparatus"]
            self.stage = "measuring"
            self.design = design[:240]
            return [A("sci.run", "run", apparatus=self.app, design=self.design)]
        if self.stage == "measuring":
            d = got.get("data")
            if not d or not d.get("readings"):
                return []
            pts = [r for r in d["readings"] if not r.get("standard") and r.get("value") is not None]
            std = [r for r in d["readings"] if r.get("standard") in self.certified and r.get("value") is not None]
            gain, off = self._calibrate(std)
            self.rows, self.std = list(pts), list(std)
            cond = [n for n in (pts[0]["at"] if pts else {}) if n != "setting"]
            rows = [(r["at"], (float(r["value"]) - off) / gain) for r in pts]
            law, advice, _ = select_optimum_model(rows, cond)
            self.advice = advice
            self.stage = "piloting"
            return [A("sci.publish", "paper", phenomenon=PHEN, title="The optimal practice setting", model=law,
                      data=[self.run], calibration_runs=[self.run],
                      abstract="Cost index measured over settings and circumstances; the optimum setting is a "
                               "smooth function of the circumstances."),
                    A("rc.prototype", "proto", name="Advisor v1", model=self.advice)]
        status = got.get("status") or {}
        out += self._followup(got)
        if self.stage == "piloting" and self.proto:
            ready = any(p["prototype"] == self.proto and p["ready"] for p in status.get("prototypes", []))
            if ready and self.pilot is None:
                out.append(A("rc.pilot", "pilot", prototype=self.proto, users=20))
            if self.pilot and (got.get("pilot") or {}).get("status") == "done":
                self.stage = "launch"
            p = self._pitch(got, "validated")
            if p:
                out.append(p)
            return out
        if self.stage == "launch":
            self.stage = "market"
            return out + [A("rc.launch", "launch", prototype=self.proto, name="Practice Advisor",
                            price_cents_month=self.price)]
        if self.stage == "market" and self.product:
            cash = status.get("cash_cents", 0)
            if self.ads < 4 and cash > 25_000:
                self.ads += 1
                out.append(A("rc.advertise", None, product=self.product, cents=min(40_000, cash // 3)))
            p = self._pitch(got, "validated")
            if p:
                out.append(p)
            prod = next((x for x in status.get("products", []) if x["product"] == self.product), None)
            if prod and prod["paying"] >= 25:
                p = self._pitch(got, "traction")
                if p:
                    out.append(p)
        return out

    def handle(self, tag, res, action):
        if res is None:
            return
        if tag == "run":
            self.run = res["run"]
        elif tag == "paper":
            self.paper = self.paper or res["paper"]
            self.papers.append(res["paper"])
        elif tag == "run2":
            self.run2 = res["run"]
            self.followup = "measuring"

        elif tag == "proto":
            self.proto = res["prototype"]
        elif tag == "pilot":
            self.pilot = res["pilot"]
        elif tag == "launch":
            self.product = res["product"]


class CommonPracticeFounder(ScienceFounder):
    """Degenerate-strategy probe and the counterfactual without the discovery: the same founder building its
    advisor from common practice (a setting it cannot personalise without the phenomenon's data)."""
    discovery = False
    common = 0.5


def structure(world) -> dict:
    """Measured on the built world: the law family of the phenomenon, its condition variables and the kinds of
    financing on offer."""
    mod = world.modules["rcomm"]
    kinds = sorted({"grant" if i.get("equity_per_100k_bps", 0) <= 1 else
                    ("growth" if i["rule"] == "traction" else "angel") for i in mod.c["investors"]})
    return {"law_family": mod.c["family"], "conditions": list(mod.cond), "financing": kinds}


Reference = ScienceFounder


def tier3(config, merged: dict) -> dict:
    """Inside a civilisation the practice phenomenon joins the civilisation's own laboratory system: its phenomenon,
    apparatus and their operators are added to the civilisation's `sci` configuration (the civilisation keeps its
    own phenomena, labs and replicators)."""
    inst = merged["modules"].get("rcomm") or {}
    if not inst.get("law"):
        return merged
    mine = sci_config(inst, K.seeded_rng(config.name + "-tier3", 0))
    sci = merged["modules"].setdefault("sci", {})
    have_ph = {p["id"] for p in sci.get("phenomena") or []}
    have_app = {a["id"] for a in sci.get("apparatus") or []}
    have_op = {o["handle"] for o in sci.get("operators") or []}
    sci["phenomena"] = list(sci.get("phenomena") or []) + [p for p in mine["phenomena"] if p["id"] not in have_ph]
    sci["apparatus"] = list(sci.get("apparatus") or []) + [a for a in mine["apparatus"] if a["id"] not in have_app]
    sci["operators"] = list(sci.get("operators") or []) + [o for o in mine["operators"] if o["handle"] not in have_op]
    return merged

DOMAIN = register(Domain(
    name="commercialization", title="Research commercialization", spec_part="9.15", required_mechanics=MECHANICS,
    configs=CONFIGS, build=build, objective=objective, evaluate=evaluate, reference=ScienceFounder,
    mechanics=tuple(Mechanic(m, (), f"measured by commercialization.mechanics_report: {m}",
                             count=K.measured(mechanics_report, m)) for m in MECHANICS),
    seats=(Seat("founder", DEFAULTS["founder_cash"], "Founder"),),
    actors=(Actor("practitioners", ("cols.batch:rc.practice",), "population individuals paying their own bills"),
            Actor("labs", ("sci.paper", "sci.replication", "sci.booked"), "rival and replication labs"),
            Actor("investors", ("rc.pitch_status",), "investors deciding on pitches from the public record")),
    degenerate=(Probe("common_practice_advisor", "an advisor built without the research (common practice)",
                      CommonPracticeFounder),),
    structure=structure, tier3=tier3, mode="B", horizon_days=100.0, slowest_effect_days=60.0, dt=12 * HOUR,
    reference_seeds=(1, 2), sources=("commercialization.py", "_dom3.py", "science.py"),
    doc=__doc__.split("\n\n")[0]))
