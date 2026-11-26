"""Work activities: the population's demand for capabilities and products (module `work`, host `work`).

D-15 / D-18. Demand in a v5 world comes from the one population (`crucible.population`), not from a separate buyer
population. Individuals with an occupation perform *work*: a stream of tasks of one activity family, each task a
pipeline of one or two steps, each step an instance of a public problem format (`economy.needs`, including
procedurally generated categories registered there). The relation graph this realises is generic
(`innovation.relations`): an activity *requires* a capability per step, a provider of that capability *lowers the
time cost* of the step and the employer's acceptance standard *constrains* its outcome; the individual's working
time *bounds* the activity; using a provider *loads* the provider's compute.

Per task and step the individual either
* does it itself (DIY): it costs its own minutes (drawn per individual and step) and is always accepted, or
* delegates it to a provider it relies on - a service product it is entitled to (`prod.use`) or a declared
  capability it adopted (`caps`, a direct priced call) - on its OWN task input: it spends a few minutes preparing
  and checking, pays what the provider charges, and observes the outcome: the response status (failures,
  overloads, outages) and whether its work is *accepted downstream*. Acceptance is world physics: the employer
  accepts the output when the format's verifier scores it at or above the individual's acceptance standard. The
  individual never reads the score or the hidden truth, only accepted / rejected; a rejected or failed delegated
  step is redone by hand (DIY minutes on top).

Usefulness is therefore anchored to the do-it-yourself outside option: a provider is worth
`tasks x [P(accept) x (DIY - use) - (1 - P(accept)) x use] x value of time - price` per month to an individual,
with P(accept) its own belief - a Beta posterior over its own accepted / rejected outcomes, started from public
ratings (only verified-independent purchasers count, see `commerce.products`), word of mouth (contacts tell what
they rely on and how it works for them) and a skeptical prior. Willingness to pay and retention follow from the
individual's own state (value of time, workload, standard, price sensitivity and loyalty preferences of the
population) and its experienced outcomes - not from constants.

Activities registered with the population (`popbridge.register_activity`):
* `work` - the working session. Professionals are woken for it at the start of their working day (`wake`); a
  session processes the day's tasks (1-3, real executions), records outcomes (`work.outcomes`) and then the
  individual's decisions through `record_decisions`: cancel a subscription that does not pay off (trials end
  without charge; paid periods at period end), request a refund through a support ticket after a bad paid
  period, open a support ticket after failures and weigh how the merchant handled it, rate what it paid for,
  recommend a provider that works for it on the public forum, and schedule a `shop` session when a step is
  unserved or underserved.
* `shop` - discovery and choice: the individual searches the world's search engine with a description of its
  task (organic results, sponsored results charged by `ads`, forum posts that name hosts), asks its contacts,
  keeps candidates whose declared interface accepts its own input (a product serving the public format, or a
  capability declaration whose input schema its input satisfies), estimates each one's value from its beliefs
  and the public prices, and - if one beats doing it itself - subscribes (a trial when offered, from its own
  account through a bank mandate), buys, or tries a declared capability on its own input and adopts it if the
  outcome was accepted. Responses to sponsored results use the fitted commercial-response model
  (`calibration` `commercial_response@1`, response types re-based to a search click rate).

Daily process `work.day`: deposits the accrued income of individuals that pay standing charges, wakes
professionals for their next working day, and lets an occupation's need end now and then (exogenous churn).

Nothing here reads a provider's code, a generator's hidden truth for a decision, or evaluator verdicts. Decisions
carry the model id `work.value` (fidelity register row FR-WORK-1, an uncalibrated documented assumption).
"""
from __future__ import annotations

import math

import numpy as np

from ..core import schema as S
from ..core.errors import CrucibleError
from ..core.kernel import Module, applier, endpoint, process, register_module
from ..substrate import columns, ids
from . import popbridge as pb

DAY = 86400
TABLE = "work"
MODEL = {"model_id": "work.value", "model_version": "1", "fidelity_id": "FR-WORK-1"}
MAX_TASKS = 3
FAMILY_DEFAULTS = {"share": 1.0, "tasks_day": [1.0, 2.5], "diy_min": [12.0, 45.0], "size": [0.2, 0.6],
                   "standard": [0.88, 0.98], "budget_min": [60.0, 180.0], "then": None, "then_diy_min": [8.0, 30.0],
                   "title": None}
DEFAULTS = {"families": [], "professional_share": 0.6, "use_min": 2.0, "shop_p": 0.6, "shop_every_days": 3,
            "need_end_month_p": 0.03, "prior_accept": 0.5, "prior_weight": 2.0, "rating_weight": 0.6,
            "wom": True, "forum_board": "tools", "post_p": 0.08, "search_limit": 8, "ads_click_rate": 0.03,
            "ticket_patience_days": 3, "refund_threshold": 0.35, "first_day": 1, "rate_after": 5,
            "publish_formats": False, "ask_p": 0.15, "ask_board": "help"}


def _draw(rng, lohi, n):
    lo, hi = float(lohi[0]), float(lohi[1])
    return rng.uniform(lo, hi, n)


def _needs():
    from .needs import NEEDS
    return NEEDS


@register_module
class Work(Module):
    name = "work"
    host = "work"
    depends = ("identity", "bank", "pop", "cols", "products", "subs", "search", "support")

    # ------------------------------------------------------------------ setup
    def cfg(self, k):
        return self.config.get(k, DEFAULTS[k])

    def families(self) -> list[dict]:
        out = []
        for f in self.cfg("families"):
            g = {**FAMILY_DEFAULTS, **f}
            if g["need"] not in _needs():
                raise ValueError(f"work family needs a registered problem format, not {g['need']!r}")
            if g["then"] is not None and g["then"] not in _needs():
                raise ValueError(f"unknown second step {g['then']!r}")
            out.append(g)
        return out

    def install(self, world):
        self.fams = self.families()
        n = pb.n_individuals(world)
        columns.declare(world, TABLE, n, {
            "fam": ("int16", (), -1), "tasks": ("float32", (), 0.0), "size": ("float32", (), 0.0),
            "std": ("float32", (), 0.0), "vot": ("float32", (), 0.0), "diy": ("float32", (2,), 0.0),
            "budget": ("float32", (), 0.0), "ad_type": ("int16", (), 0), "active": ("bool", (), False),
            "minutes": ("float64", (), 0.0), "diy_minutes": ("float64", (), 0.0), "done": ("float64", (), 0.0),
            "lost": ("float64", (), 0.0), "delegated": ("float64", (), 0.0), "spent": ("float64", (), 0.0),
            "last_shop": ("int64", (), -1), "dormant_until": ("int64", (), -1), "shop_at": ("int64", (), -1),
            "work_day": ("int32", (), -1),
            # the last decision of each kind (decision provenance lives in the batch extras)
            "choice": ("int32", (), -1), "ev": ("float32", (), 0.0)})
        st = world.state
        st.declare_table("work_exp", index=("i", "prov", "step_need"))
        st.declare_table("work_provs")
        pb.register_activity(world, "work", self.act_work, owner="work", prior=0.2, goals=(5,))
        pb.register_activity(world, "shop", self.act_shop, owner="work", prior=-0.2, goals=(2,))
        self._types = None

    def genesis(self, ctx):
        w = ctx.world
        n = pb.n_individuals(w)
        if not n or not self.fams:
            return
        rng = w.rng_for("work-genesis")
        f = pb.features(w, np.arange(n))
        shares = np.array([float(x["share"]) for x in self.fams])
        employed = f["employed"].astype(bool)
        prof = (rng.random(n) < float(self.cfg("professional_share"))) & (employed | (rng.random(n) < 0.25))
        fam = np.where(prof, rng.choice(len(self.fams), size=n, p=shares / shares.sum()), -1).astype(np.int16)
        skills = f["skills"].astype(np.float64)
        tech = skills[:, 4] if skills.shape[1] > 4 else np.full(n, 0.5)
        plan = skills[:, 7] if skills.shape[1] > 7 else np.full(n, 0.5)
        vals = {k: np.zeros(n, np.float32) for k in ("tasks", "size", "std", "budget")}
        diy = np.zeros((n, 2), np.float32)
        for j, g in enumerate(self.fams):
            m = fam == j
            k = int(m.sum())
            if not k:
                continue
            vals["tasks"][m] = _draw(rng, g["tasks_day"], k)
            vals["size"][m] = _draw(rng, g["size"], k)
            vals["std"][m] = _draw(rng, g["standard"], k)
            vals["budget"][m] = _draw(rng, g["budget_min"], k)
            # hand work is slower for the less skilled
            skill = 0.5 * (tech[m] + plan[m])
            diy[m, 0] = _draw(rng, g["diy_min"], k) * (1.35 - 0.7 * skill)
            if g["then"] is not None:
                diy[m, 1] = _draw(rng, g["then_diy_min"], k) * (1.35 - 0.7 * skill)
        # value of a working minute: from the individual's means (spendable money) and employment
        wallet = np.maximum(f["wallet_cents"].astype(np.float64), 1000.0)
        vot = np.exp(rng.normal(np.log(wallet / 60000.0 * 12.0), 0.35)) * np.where(employed, 1.0, 0.6)
        vot = np.clip(vot, 2.0, 120.0)
        types = self._ad_types(rng, n)
        cols = {"fam": fam, "tasks": vals["tasks"], "size": vals["size"], "std": vals["std"],
                "vot": vot.astype(np.float32), "diy": diy, "budget": vals["budget"], "ad_type": types,
                "active": fam >= 0}
        columns.batch(ctx, TABLE, "work.genesis", np.arange(n), cols, mode="set",
                      extra={"families": [g["need"] for g in self.fams], "professionals": int((fam >= 0).sum())})
        if self.cfg("publish_formats"):
            self._publish_formats(ctx)
        first = (ctx.now // DAY + int(self.cfg("first_day"))) * DAY
        ctx.schedule(first, "work.day", {}, prio=35, period=DAY, key="work.day")
        ctx.schedule(first + 3600, "work.hourly", {}, prio=36, period=3600, key="work.hourly")
        self._wake_workday(ctx, first)

    def _publish_formats(self, ctx) -> None:
        """World generation: the public documentation of the work formats (a standards body's documents, indexed
        by search). Procedurally generated categories are not published - nobody has documented them yet."""
        docs = ctx.world.modules.get("docs")
        if docs is None:
            return
        idm = ctx.world.modules["identity"]
        org = idm.create_org(ctx, "formats", "Formats Registry", None, "group")
        for need in sorted({g["need"] for g in self.fams} | {g["then"] for g in self.fams if g["then"]}):
            if need.startswith("gen."):
                continue
            nd = _needs()[need]
            docs.create_doc(ctx, org, f"Format {need}: {nd.title}",
                            nd.spec_text() + f"\nKeywords: {' '.join(nd.keywords)}\n", "public")

    def _ad_types(self, rng, n) -> np.ndarray:
        m = self._ad_model()
        if m is None:
            return np.zeros(n, np.int16)
        return m.sample_types(rng, n).astype(np.int16)

    def _ad_model(self):
        if getattr(self, "_admodel", None) is None:
            try:
                from ..calibration import registry
                m = registry.load("commercial_response@1")
                self._admodel = m
                self._adtab = m.transport(control_visit_rate=float(self.cfg("ads_click_rate")))
            except Exception:  # noqa: BLE001 - a world without the fitted artefact responds without ad uplift
                self._admodel = False
                self._adtab = None
        return self._admodel or None

    # ------------------------------------------------------------------ helpers
    def _col(self, name):
        return columns.col(self.world, TABLE, name)

    def _steps(self, i: int) -> list[tuple[int, str]]:
        g = self.fams[int(self._col("fam")[i])]
        out = [(0, g["need"])]
        if g["then"] is not None:
            out.append((1, g["then"]))
        return out

    def task_text(self, need: str) -> str:
        """How an individual describes the task when it looks for help (public format title and vocabulary)."""
        nd = _needs()[need]
        g = next((x for x in self.fams if x["need"] == need and x.get("title")), None)
        title = g["title"] if g else nd.title
        return f"{title} {' '.join(nd.keywords[:8])}"

    def exp(self, st, i: int, prov: str) -> dict | None:
        return st.get("work_exp", f"{i}|{prov}")

    def relied(self, st, i: int, need: str) -> list[tuple[str, dict]]:
        """Providers the individual currently relies on for `need` (its own adoption records)."""
        out = []
        for key in st.find_ids("work_exp", "i", int(i)):
            r = st.tables["work_exp"][key]
            if r["active"] and r["step_need"] == need:
                out.append((r["prov"], r))
        return out

    def belief(self, st, i: int, prov: str) -> tuple[float, float]:
        """(P(accept), evidence weight) from the individual's own outcomes with a provider."""
        r = self.exp(st, i, prov)
        a0 = float(self.cfg("prior_accept")) * float(self.cfg("prior_weight"))
        b0 = (1 - float(self.cfg("prior_accept"))) * float(self.cfg("prior_weight"))
        if r is None:
            return a0 / (a0 + b0), 0.0
        a, b = a0 + r["ok"], b0 + r["rej"] + r["fail"]
        return a / (a + b), float(r["uses"])

    # ------------------------------------------------------------------ provider access (real tools)
    def _provider(self, st, prov: str) -> dict | None:
        """Public description of a provider: a launched service product or an active capability declaration."""
        if prov.startswith("prd_") and "products" in st.tables:
            p = st.get("products", prov)
            if p is None or p["status"] != "launched" or (p["delivery"] or {}).get("type") != "service":
                return None
            return {"kind": "product", "id": prov, "owner": p["owner"], "need": p.get("need"),
                    "host": p["delivery"]["host"], "rating": p.get("vrating_sum", 0), "ratings": p.get("vrating_n", 0)}
        if prov.startswith("cdecl_") and "cap_decls" in st.tables:
            d = st.get("cap_decls", prov)
            if d is None or d["status"] != "active":
                return None
            return {"kind": "decl", "id": prov, "owner": d["provider"], "host": d["host"], "inputs": d["inputs"],
                    "endpoint": d["endpoint"], "name": d["name"], "rating": 0, "ratings": 0}
        return None

    def compatible(self, st, prov: dict, need: str, sample: dict) -> bool:
        """Interface match: the product serves the public format, or a declared input schema accepts the
        individual's own input."""
        if prov["kind"] == "product":
            if prov.get("need") == need:
                return True
            for did in st.find_ids("cap_decls", "host", prov["host"]) if "cap_decls" in st.tables else ():
                d = st.tables["cap_decls"][did]
                if d["status"] == "active" and not S.validate(d["inputs"], sample, "input"):
                    return True
            return False
        return not S.validate(prov["inputs"], sample, "input")

    def _call(self, ctx, p: str, prov: dict, inp: dict) -> dict:
        """One real execution of a provider on the individual's own input, as the individual."""
        try:
            if prov["kind"] == "product":
                res = ctx.call("prod.use", {"product": prov["id"], "input": inp}, as_principal=p)
            else:
                res = ctx.call(f"{prov['host']}.{prov['endpoint']}", inp, as_principal=p)
        except CrucibleError as e:
            return {"status": 403 if e.code == "forbidden" else 400, "error": e.message}
        return res if isinstance(res, dict) else {"status": 500}

    @staticmethod
    def _accepted(need: str, inp, hidden, res: dict, std: float) -> bool:
        """World physics: the employer accepts the delegated output when the format's verifier scores it at or
        above the individual's standard. Only the yes/no reaches the individual."""
        if res.get("status") != 200:
            return False
        try:
            return float(_needs()[need].score(inp, hidden, res.get("body"))) >= float(std)
        except Exception:  # noqa: BLE001 - a malformed output is rejected
            return False

    # ------------------------------------------------------------------ the working session
    def act_work(self, ctx, idx: np.ndarray, now: int):
        w, st = ctx.world, ctx.state
        fam = self._col("fam")[idx]
        util = np.zeros(idx.size)
        today = now // DAY
        rows = np.flatnonzero((fam >= 0) & self._col("active")[idx] & (self._col("work_day")[idx] < today))
        if rows.size == 0:
            return util
        columns.batch(ctx, TABLE, "work.day_started", idx[rows], {"work_day": np.full(rows.size, today, np.int32)},
                      mode="set")
        rng = ctx.rng("work.session")
        use_min = float(self.cfg("use_min"))
        tasks = self._col("tasks")[idx]
        size, std, vot = self._col("size")[idx], self._col("std")[idx], self._col("vot")[idx]
        diy, budget = self._col("diy")[idx], self._col("budget")[idx]
        upd = {k: np.zeros(idx.size) for k in ("minutes", "diy_minutes", "done", "lost", "delegated", "spent")}
        out_rows: list[list] = []
        follow: list[tuple[int, str, str]] = []           # (row, provider, need) to reconsider after the session
        shop: list[int] = []
        for r in rows:
            i = int(idx[r])
            p = ids.pid(i)
            n_t = int(min(MAX_TASKS, rng.poisson(float(tasks[r]))))
            if n_t == 0:
                continue
            steps = self._steps(i)
            spent_min = baseline = 0.0
            cash = 0
            for _ in range(n_t):
                if spent_min > float(budget[r]):
                    upd["lost"][r] += 1
                    continue
                for s, need in steps:
                    d = float(diy[r, s])
                    baseline += d
                    rel = self.relied(st, i, need)
                    if not rel:
                        spent_min += d
                        upd["diy_minutes"][r] += d
                        continue
                    prov_id, _ = rel[0]
                    prov = self._provider(st, prov_id)
                    if prov is not None and prov["kind"] == "product" and \
                            w.modules["products"].entitlement(st, prov_id, {p}) is None:
                        # the subscription ended (period end, non-payment, merchant): no longer relied on
                        ctx.emit("work.rely", {"key": f"{i}|{prov_id}", "i": i, "prov": prov_id, "active": False,
                                               "why": "no entitlement"}, vis="internal")
                        spent_min += d
                        upd["diy_minutes"][r] += d
                        continue
                    trng = w.rng_for("work-task", i, now, s, len(out_rows))
                    inp, hidden = _needs()[need].generate(trng, float(size[r]))
                    if prov is None:
                        res = {"status": 410}
                    else:
                        before = self._balance(st, p)
                        res = self._call(ctx, p, prov, inp)
                        cash += max(0, before - self._balance(st, p))
                    ok = self._accepted(need, inp, hidden, res, float(std[r]))
                    m = use_min + (0.0 if ok else d)
                    spent_min += m
                    if not ok:
                        upd["diy_minutes"][r] += d
                    upd["delegated"][r] += 1
                    out_rows.append([i, prov_id, need, 1, int(ok), int(res.get("status") == 200 and not ok),
                                     int(res.get("status") != 200), round(m, 3)])
                    follow.append((r, prov_id, need))
                upd["done"][r] += 1
            upd["minutes"][r] += spent_min
            upd["spent"][r] += cash
            value = (baseline - spent_min) * float(vot[r]) - cash
            util[r] = float(np.clip(value / max(1.0, baseline * float(vot[r])), -1.0, 1.0)) + 0.1
            if any(not self.relied(st, i, need) for _, need in steps):
                shop.append(r)
        if out_rows:
            ctx.emit("work.outcomes", {"rows": out_rows, "t": int(now)}, vis="internal")
        sel = np.flatnonzero(upd["minutes"] + upd["lost"] > 0)
        if sel.size:
            columns.batch(ctx, TABLE, "work.session", idx[sel], {k: upd[k][sel] for k in upd}, mode="add",
                          extra={"t": int(now), "sessions": int(sel.size)})
        self._after_session(ctx, idx, follow, now)
        self._maybe_shop(ctx, idx, np.array(sorted(set(shop)), np.int64), now)
        return util

    @staticmethod
    def _balance(st, p: str) -> int:
        tot = 0
        for a in st.find_ids("accounts", "owner", p):
            tot += int(st.tables["accounts"][a]["balance"])
        return tot

    # ------------------------------------------------------------------ decisions after experience
    def _after_session(self, ctx, idx, follow, now):
        st = ctx.state
        seen = set()
        cancel_i, cancel_v, refund_i, ticket_i, rate_i, post_i = [], [], [], [], [], []
        use_min = float(self.cfg("use_min"))
        for r, prov_id, need in follow:
            i = int(idx[r])
            if (i, prov_id) in seen:
                continue
            seen.add((i, prov_id))
            e = self.exp(st, i, prov_id)
            if e is None or not e["active"]:
                continue
            p = ids.pid(i)
            pacc, nev = self.belief(st, i, prov_id)
            s = 0 if self._steps(i)[0][1] == need else 1
            d = float(self._col("diy")[i, s])
            vot = float(self._col("vot")[i])
            tasks_m = float(self._col("tasks")[i]) * 22
            price_m = self._monthly_cost(st, p, prov_id, tasks_m)
            ev = tasks_m * (pacc * (d - use_min) - (1 - pacc) * use_min) * vot - price_m
            prov = self._provider(st, prov_id)
            # failures: one support ticket per incident window, weighed later
            if e["fail_recent"] >= 2 and not e.get("ticket") and prov is not None:
                tid = self._ticket(ctx, p, prov, "Requests are failing",
                                   f"{e['fail_recent']} of my recent requests failed.", 2)
                if tid:
                    ticket_i.append(i)
                    ctx.emit("work.exp_note", {"key": f"{i}|{prov_id}", "ticket": tid, "ticket_t": int(now)},
                             vis="internal")
            patience = float(self.cfg("ticket_patience_days")) * DAY
            neglected = bool(e.get("ticket")) and self._ticket_neglected(st, e, now, patience)
            if neglected:
                ev -= abs(ev) * 0.5 + 0.5 * price_m
            if nev >= 3 and ev < 0:
                cancel_i.append(i)
                cancel_v.append(ev)
                self._drop(ctx, p, i, prov_id, prov, refund=pacc < float(self.cfg("refund_threshold")))
                if pacc < float(self.cfg("refund_threshold")):
                    refund_i.append(i)
            elif nev >= int(self.cfg("rate_after")) and not e.get("rated") and prov is not None \
                    and prov["kind"] == "product":
                stars = int(np.clip(round(1 + 4 * pacc - (1.0 if neglected else 0.0)), 1, 5))
                try:
                    ctx.call("prod.rate", {"product": prov_id, "stars": stars,
                                           "text": "works for my tasks" if stars >= 4 else
                                           ("mixed results" if stars == 3 else "my work keeps getting rejected")},
                             as_principal=p)
                    rate_i.append(i)
                except CrucibleError:
                    pass
                ctx.emit("work.exp_note", {"key": f"{i}|{prov_id}", "rated": True}, vis="internal")
                if stars >= 4 and ctx.rng("work.post", i).random() < float(self.cfg("post_p")):
                    if self._post(ctx, p, need, prov):
                        post_i.append(i)
        if cancel_i:
            pb.record_decisions(ctx, "work.cancel", np.array(cancel_i), {"ev": np.array(cancel_v, np.float32)},
                                table=TABLE, **MODEL)
        for kind, lst in (("work.ticket", ticket_i), ("work.rate", rate_i), ("work.refund_request", refund_i),
                          ("work.recommend", post_i)):
            if lst:
                pb.record_decisions(ctx, kind, np.array(sorted(set(lst))),
                                    {"choice": np.ones(len(set(lst)), np.int32)}, table=TABLE, **MODEL)

    def _ticket_neglected(self, st, e, now, patience) -> bool:
        t = st.get("sup_tickets", e["ticket"]) if "sup_tickets" in st.tables else None
        if t is None:
            return False
        if t["status"] == "open" and now - int(e["ticket_t"]) > patience:
            return True
        # "resolved" without an effect: failures continued after the ticket was resolved
        return t["status"] == "resolved" and e["fail_recent"] >= 2 and int(e["last_fail_t"]) > int(t.get("updated")
                                                                                                    or e["ticket_t"])

    def _ticket(self, ctx, p, prov, subject, body, severity) -> str | None:
        try:
            if prov["kind"] == "product":
                return ctx.call("prod.support", {"product": prov["id"], "subject": subject, "body": body,
                                                 "severity": severity}, as_principal=p)["ticket"]
            return ctx.call("support.open", {"host": prov["host"], "subject": subject, "body": body,
                                             "severity": severity}, as_principal=p)["ticket"]
        except CrucibleError:
            return None

    def _post(self, ctx, p, need, prov) -> bool:
        try:
            ctx.call("forum.post", {"board": self.cfg("forum_board"),
                                    "title": f"{prov['host']} works well for {_needs()[need].title.lower()}",
                                    "body": f"I use {prov['host']}.crucible for {self.task_text(need)}. "
                                            "My results get accepted."}, as_principal=p)
            return True
        except CrucibleError:
            return False

    def _monthly_cost(self, st, p: str, prov_id: str, tasks_m: float) -> float:
        """What relying on a provider costs the individual per month: its subscription price, or the per-call price
        times its monthly tasks."""
        if prov_id.startswith("prd_"):
            for sid in st.find_ids("sub_subs", "customer", p):
                s = st.tables["sub_subs"][sid]
                if s["product"] == prov_id and s["status"] in ("trialing", "active", "past_due"):
                    return float(s["price"]) / (1 if s["interval"] == "month" else 12)
            return 0.0
        prov = self._provider(st, prov_id)
        if prov is None or "cloud" not in self.world.modules:
            return 0.0
        dep = self.world.modules["cloud"].live_deployment(st, prov["host"])
        price = int((dep or {}).get("manifest", {}).get("endpoints", {}).get(prov["endpoint"], {}).get("price_cents", 0))
        return float(price) * tasks_m

    def _drop(self, ctx, p, i, prov_id, prov, refund: bool) -> None:
        """Stop relying on a provider: cancel the subscription (a trial ends now, a paid period at its end) or drop
        the adopted capability; ask for a refund of a bad paid period."""
        st = ctx.state
        if prov_id.startswith("prd_"):
            for sid in st.find_ids("sub_subs", "customer", p):
                s = st.tables["sub_subs"][sid]
                if s["product"] != prov_id or s["status"] not in ("trialing", "active", "past_due"):
                    continue
                try:
                    if s["status"] == "trialing" or s["status"] == "past_due":
                        ctx.call("subs.cancel", {"subscription": sid, "at_period_end": False}, as_principal=p)
                    elif not s["cancel_at_period_end"]:
                        ctx.call("subs.cancel", {"subscription": sid, "at_period_end": True}, as_principal=p)
                except CrucibleError:
                    pass
                if refund and s["status"] == "active" and s["last_invoice"] and prov is not None:
                    inv = st.get("sub_invoices", s["last_invoice"])
                    if inv is not None and inv["amount"] - inv["refunded"] > 0 and ctx.now - inv["t"] < 14 * DAY:
                        self._ticket(ctx, p, prov, f"Refund request {s['last_invoice']}",
                                     f"Please refund invoice {s['last_invoice']}: my work kept getting rejected.", 3)
        elif prov is not None:
            try:
                ctx.call("caps.drop_service", {"declaration": prov_id, "reason": "results not accepted"},
                         as_principal=p)
            except CrucibleError:
                pass
        ctx.emit("work.rely", {"key": f"{i}|{prov_id}", "i": int(i), "prov": prov_id, "active": False,
                               "why": "not worth it"}, vis="internal")

    # ------------------------------------------------------------------ discovery and choice
    def _maybe_shop(self, ctx, idx, rows, now):
        if rows.size == 0:
            return
        last = self._col("last_shop")[idx[rows]]
        due = rows[(last < 0) | (now - last > int(self.cfg("shop_every_days")) * DAY)]
        if due.size == 0:
            return
        rng = ctx.rng("work.shopwake")
        go = due[rng.random(due.size) < float(self.cfg("shop_p"))]
        if go.size:
            # the population clears pending wake-ups of a session it just ran, so the shop session is scheduled
            # through this module's hourly process
            at = now + rng.integers(1800, 6 * 3600, go.size).astype(np.int64)
            columns.batch(ctx, TABLE, "work.shop_planned", idx[go], {"shop_at": at}, mode="set")

    @process("hourly")
    def pr_hourly(self, ctx):
        at = self._col("shop_at")
        due = np.flatnonzero((at >= 0) & (at <= ctx.now + 3600))
        if due.size == 0:
            return
        for t in sorted(set(at[due].tolist())):
            sel = due[at[due] == t]
            pb.wake(ctx, sel, int(max(t, ctx.now)), "shop")
        columns.batch(ctx, TABLE, "work.shop_woken", due, {"shop_at": np.full(due.size, -1, np.int64)}, mode="set")

    def act_shop(self, ctx, idx: np.ndarray, now: int):
        w, st = ctx.world, ctx.state
        util = np.zeros(idx.size)
        fam = self._col("fam")[idx]
        rows = np.flatnonzero((fam >= 0) & self._col("active")[idx])
        if rows.size == 0:
            return util
        f = pb.features(w, idx[rows])
        prefs = f["prefs"].astype(np.float64)
        contacts = f["contacts"]
        chosen, evs, shopped, asked = [], [], [], []
        for k, r in enumerate(rows):
            i = int(idx[r])
            p = ids.pid(i)
            shopped.append(i)
            for s, need in self._steps(i):
                if self.relied(st, i, need):
                    continue
                sample, _ = _needs()[need].generate(w.rng_for("work-sample", i, need), float(self._col("size")[i]))
                cands = self._candidates(ctx, p, i, need, contacts[k], sample)
                best, best_ev = None, 0.0
                d = float(self._col("diy")[i, s])
                vot = float(self._col("vot")[i])
                tasks_m = float(self._col("tasks")[i]) * 22
                ps = float(prefs[k, 0]) if prefs.shape[1] > 0 else 0.5
                for prov_id, (prov, prior_acc, how) in sorted(cands.items()):
                    pacc, nev = self.belief(st, i, prov_id)
                    if nev == 0:
                        pacc = prior_acc
                    e = self.exp(st, i, prov_id)
                    if e is not None and not e["active"] and e["uses"] >= 3 and pacc < 0.6:
                        continue                                   # tried, did not work, dropped
                    price_m, trial = self._offer_price(st, prov, tasks_m)
                    if price_m is None:
                        continue
                    gain = tasks_m * (pacc * (d - float(self.cfg("use_min"))) -
                                      (1 - pacc) * float(self.cfg("use_min"))) * vot
                    ev = gain - price_m * (0.6 + 0.8 * ps)
                    if trial:
                        ev += 0.15 * abs(ev) + 0.1 * price_m      # trying costs nothing up front
                    if ev > best_ev:
                        best, best_ev = (prov_id, prov), ev
                if best is not None and self._adopt(ctx, p, i, need, best[0], best[1], sample):
                    chosen.append(i)
                    evs.append(best_ev)
                    util[r] += 0.3
                elif not cands and ctx.rng("work.ask", i).random() < float(self.cfg("ask_p")):
                    if self._ask(ctx, p, i, need, sample, d * vot * tasks_m):
                        asked.append(i)
        if shopped:
            columns.batch(ctx, TABLE, "work.shopped", np.array(shopped),
                          {"last_shop": np.full(len(shopped), now, np.int64)}, mode="set")
        if chosen:
            pb.record_decisions(ctx, "work.adopt", np.array(chosen), {"ev": np.array(evs, np.float32)},
                                table=TABLE, **MODEL)
        if asked:
            pb.record_decisions(ctx, "work.ask", np.array(sorted(set(asked))),
                                {"choice": np.ones(len(set(asked)), np.int32)}, table=TABLE, **MODEL)
        return util

    def _ask(self, ctx, p, i, need, sample, monthly_value) -> bool:
        """Post a public request for help with an unserved task: what it is and an example of the individual's own
        input (truncated) - the demand signal an entrepreneur can read."""
        import json
        ex = json.dumps(sample, sort_keys=True)[:600]
        try:
            ctx.call("forum.post", {"board": self.cfg("ask_board"),
                                    "title": f"Looking for a service: {self.task_text(need)[:120]}",
                                    "body": f"I do this by hand and it takes me about "
                                            f"{float(self._col('diy')[i, 0]):.0f} minutes a task. Example input: {ex}"},
                     as_principal=p)
            return True
        except CrucibleError:
            return False

    def _candidates(self, ctx, p, i, need, contacts, sample) -> dict:
        """{provider: (description, prior P(accept), channel)} from search (organic, sponsored), forum posts naming
        hosts and contacts' word of mouth; only interface-compatible providers are kept."""
        st, w = ctx.state, ctx.world
        rng = w.rng_for("work-consider", i, ctx.now)
        out: dict = {}
        try:
            res = ctx.call("search.query", {"q": self.task_text(need)[:200], "limit": int(self.cfg("search_limit"))},
                           as_principal=p)["results"]
        except CrucibleError:
            res = []
        tab = getattr(self, "_adtab", None)
        t = int(self._col("ad_type")[i])
        rank = 0
        for rres in res:
            if rres.get("sponsored"):
                pv = float(tab["p1_visit"][t]) if tab is not None else float(self.cfg("ads_click_rate"))
            else:
                pv = 0.9 / (1.0 + 0.35 * rank)
                rank += 1
            if rng.random() >= pv:
                continue
            for prov_id in self._resolve_result(st, rres):
                prov = self._provider(st, prov_id)
                if prov is not None and prov_id not in out and self.compatible(st, prov, need, sample):
                    out[prov_id] = (prov, self._public_prior(prov), "sponsored" if rres.get("sponsored") else "search")
        if self.cfg("wom"):
            for j in [int(c) for c in contacts if int(c) >= 0][:6]:
                for key in st.find_ids("work_exp", "i", j):
                    r = st.tables["work_exp"][key]
                    if not r["active"] or r["step_need"] != need or r["uses"] < 2:
                        continue
                    prov = self._provider(st, r["prov"])
                    if prov is None or r["prov"] in out or not self.compatible(st, prov, need, sample):
                        continue
                    told = (r["ok"] + 1.0) / (r["uses"] + 2.0)        # what the contact says about it
                    if rng.random() < 0.7:
                        out[r["prov"]] = (prov, 0.5 * self._public_prior(prov) + 0.5 * told, "contact")
        return out

    def _resolve_result(self, st, r) -> list[str]:
        ref = r.get("ref")
        if isinstance(ref, str) and ref.startswith(("prd_", "cdecl_")):
            return [ref]
        out = []
        for h in r.get("hosts") or []:
            if "products" in st.tables:
                out += [pid for pid, p in st.scan("products")
                        if p["status"] == "launched" and (p["delivery"] or {}).get("host") == h]
            if "cap_decls" in st.tables:
                out += [d for d in st.find_ids("cap_decls", "host", h) if st.tables["cap_decls"][d]["status"] == "active"]
        return out

    def _public_prior(self, prov) -> float:
        """Prior P(accept) from public verified ratings (stars 1..5), shrunk to the skeptical prior."""
        base = float(self.cfg("prior_accept"))
        n = int(prov.get("ratings") or 0)
        if n <= 0:
            return base
        mean = float(prov["rating"]) / n
        wgt = float(self.cfg("rating_weight")) * n / (n + 4.0)
        return (1 - wgt) * base + wgt * (mean - 1.0) / 4.0

    def _offer_price(self, st, prov, tasks_m) -> tuple[float | None, bool]:
        """Monthly cost of the cheapest way to use a provider, and whether a free trial is offered."""
        if prov["kind"] == "product":
            subs = self.world.modules["subs"]
            plans = subs.public_plans(st, prov["id"])
            monthly = [pl for pl in plans if pl["interval"] == "month"]
            if not monthly:
                return None, False
            pl = min(monthly, key=lambda x: (x["price_cents"], x["plan"]))
            return float(pl["price_cents"]), int(pl["trial_days"]) > 0
        dep = self.world.modules["cloud"].live_deployment(st, prov["host"])
        if dep is None:
            return None, False
        price = int(dep["manifest"]["endpoints"].get(prov["endpoint"], {}).get("price_cents", 0))
        return float(price) * tasks_m, False

    def _adopt(self, ctx, p, i, need, prov_id, prov, sample) -> bool:
        st = ctx.state
        pb.fund(ctx, i)
        try:
            if prov["kind"] == "product":
                subs = ctx.world.modules["subs"]
                if subs.live_subscription(st, prov_id, p) is None:
                    plans = [pl for pl in subs.public_plans(st, prov_id) if pl["interval"] == "month"]
                    if not plans:
                        return False
                    pl = min(plans, key=lambda x: (x["price_cents"], x["plan"]))
                    ctx.call("subs.subscribe", {"plan": pl["plan"]}, as_principal=p)
            else:
                res = ctx.call("caps.try", {"service_host": prov["host"], "capability": prov_id, "input": sample},
                               as_principal=p)
                # the individual judges the trial by its own downstream outcome on its own input
                _, hidden = _needs()[need].generate(ctx.world.rng_for("work-sample", i, need),
                                                    float(self._col("size")[i]))
                ok = res.get("ok") and self._accepted(need, sample, hidden, {"status": 200, "body": res.get("output")},
                                                      float(self._col("std")[i]))
                ctx.emit("work.outcomes", {"rows": [[int(i), prov_id, need, 1, int(bool(ok)),
                                                     int(bool(res.get("ok")) and not ok), int(not res.get("ok")),
                                                     float(self.cfg("use_min"))]], "t": int(ctx.now)},
                         vis="internal")
                if not ok:
                    ctx.emit("work.rely", {"key": f"{i}|{prov_id}", "i": int(i), "prov": prov_id, "active": False,
                                           "why": "trial not accepted"}, vis="internal")
                    return False
                ctx.call("caps.adopt_service", {"declaration": prov_id, "note": "works on my own tasks"},
                         as_principal=p)
        except CrucibleError:
            return False
        ctx.emit("work.rely", {"key": f"{i}|{prov_id}", "i": int(i), "prov": prov_id, "active": True,
                               "need": need, "why": "adopted"}, vis="internal")
        return True

    # ------------------------------------------------------------------ daily
    def _wake_workday(self, ctx, day_start: int) -> None:
        fam = self._col("fam")
        act = self._col("active")
        who = np.flatnonzero((fam >= 0) & act)
        if who.size == 0:
            return
        rng = ctx.rng("work.wake")
        # working days: about 22 of 30
        who = who[rng.random(who.size) < 22 / 30]
        if who.size == 0:
            return
        at = day_start + int(9 * 3600) + rng.integers(-3600, 3600, who.size)
        for t in sorted(set((at // 900 * 900).tolist())):
            sel = who[(at // 900 * 900) == t]
            pb.wake(ctx, sel, int(t), "work")

    @process("day")
    def pr_day(self, ctx):
        st = ctx.state
        w = ctx.world
        # standing charges are paid from the bank: deposit the income of those who have them
        payers = sorted({ids.index(s["customer"]) for _, s in st.scan("sub_subs")
                         if s["status"] in ("trialing", "active", "past_due") and ids.index(s["customer"]) is not None})
        for i in payers:
            pb.fund(ctx, i)
        # an occupation's need ends now and then (exogenous churn the seller cannot prevent)
        fam, act = self._col("fam"), self._col("active")
        prof = np.flatnonzero((fam >= 0) & act)
        rng = ctx.rng("work.end")
        p_end = 1 - (1 - float(self.cfg("need_end_month_p"))) ** (1 / 30)
        ends = prof[rng.random(prof.size) < p_end]
        back = np.flatnonzero((fam >= 0) & ~act & (self._col("dormant_until") >= 0) &
                              (self._col("dormant_until") <= ctx.now))
        if ends.size:
            columns.batch(ctx, TABLE, "work.need_ended", ends, {
                "active": np.zeros(ends.size, bool),
                "dormant_until": ctx.now + rng.integers(20, 90, ends.size).astype(np.int64) * DAY}, mode="set")
            for i in ends.tolist():
                for prov_id, _ in [(r["prov"], r) for k, r in
                                   [(k, st.tables["work_exp"][k]) for k in st.find_ids("work_exp", "i", int(i))]
                                   if r["active"]]:
                    self._drop(ctx, ids.pid(i), i, prov_id, self._provider(st, prov_id), refund=False)
            pb.record_decisions(ctx, "work.need_end", ends, {"choice": np.zeros(ends.size, np.int32)}, table=TABLE,
                                **MODEL)
        if back.size:
            columns.batch(ctx, TABLE, "work.need_back", back, {"active": np.ones(back.size, bool),
                                                               "dormant_until": np.full(back.size, -1, np.int64)},
                          mode="set")
        self._wake_workday(ctx, ctx.now)

    # ------------------------------------------------------------------ appliers
    @applier("work.outcomes")
    def ap_outcomes(self, ev):
        st = self.world.state
        for i, prov, need, uses, ok, rej, fail, minutes in ev.payload["rows"]:
            key = f"{i}|{prov}"
            r = st.get("work_exp", key)
            if r is None:
                r = {"i": int(i), "prov": prov, "step_need": need, "uses": 0, "ok": 0, "rej": 0, "fail": 0,
                     "minutes": 0.0, "active": False, "first": ev.t, "fail_recent": 0, "last_fail_t": -1,
                     "ticket": None, "ticket_t": -1, "rated": False}
                st.insert("work_exp", key, r)
                r = st.tables["work_exp"][key]
            st.patch("work_exp", key, uses=r["uses"] + uses, ok=r["ok"] + ok, rej=r["rej"] + rej,
                     fail=r["fail"] + fail, minutes=round(r["minutes"] + minutes, 3), last=ev.t,
                     fail_recent=(r["fail_recent"] + 1) if fail else max(0, r["fail_recent"] - 1) if ok else
                     r["fail_recent"], last_fail_t=ev.t if fail else r["last_fail_t"])

    @applier("work.rely")
    def ap_rely(self, ev):
        st = self.world.state
        p = ev.payload
        r = st.get("work_exp", p["key"])
        if r is None:
            st.insert("work_exp", p["key"], {"i": p["i"], "prov": p["prov"], "step_need": p.get("need"),
                                             "uses": 0, "ok": 0, "rej": 0, "fail": 0, "minutes": 0.0,
                                             "active": bool(p["active"]), "first": ev.t, "fail_recent": 0,
                                             "last_fail_t": -1, "ticket": None, "ticket_t": -1, "rated": False,
                                             "since": ev.t})
        else:
            ch = {"active": bool(p["active"])}
            if p.get("need"):
                ch["step_need"] = p["need"]
            if p["active"]:
                ch["since"] = ev.t
            st.patch("work_exp", p["key"], **ch)

    @applier("work.exp_note")
    def ap_note(self, ev):
        p = dict(ev.payload)
        key = p.pop("key")
        self.world.state.patch("work_exp", key, **p)

    # ------------------------------------------------------------------ the individual's own view
    @endpoint("mine", S.obj(), read_only=True, cost_s=0)
    def ep_mine(self, ctx):
        """Your own work activity (members of the population only): tasks, what you rely on, your outcomes."""
        i = ids.index(ctx.actor)
        if i is None or i >= columns.rows(ctx.world, TABLE) or int(self._col("fam")[i]) < 0:
            return {"working": False}
        st = ctx.state
        return {"working": bool(self._col("active")[i]), "steps": [n for _, n in self._steps(i)],
                "relies_on": [{"provider": r["prov"], "uses": r["uses"], "accepted": r["ok"]}
                              for k in st.find_ids("work_exp", "i", int(i)) for r in [st.tables["work_exp"][k]]
                              if r["active"]]}

    def check_invariants(self, world):
        errs = []
        st = world.state
        for key, r in st.scan("work_exp"):
            if r["ok"] + r["rej"] + r["fail"] != r["uses"]:
                errs.append(f"work_exp {key}: outcomes do not add up to uses")
        if TABLE in columns.tables(world):
            if float(self._col("minutes").min(initial=0.0)) < 0:
                errs.append("negative working minutes")
        return errs


# ============================================================================ evaluator-side measurement
def resource_use(world) -> dict:
    """World resource consumption attributable to work activities (evaluator side): working minutes, minutes done
    by hand, tasks completed and lost for lack of time, delegated steps."""
    out = {}
    if TABLE not in columns.tables(world):
        return out
    for k in ("minutes", "diy_minutes", "done", "lost", "delegated", "spent"):
        out[k] = float(columns.col(world, TABLE, k).sum())
    return out
