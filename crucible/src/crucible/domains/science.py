"""Experimental science (kernel module `sci`, host `sci`): phenomena with hidden laws, measuring apparatus,
paid and queued experiments, datasets, publications with machine-checkable models, and independent replication.

World physics (config, never returned by any tool):

* a **phenomenon** relates a few named condition variables to one response through a hidden law written in
  `crucible.domains.lawlang` (for example a saturating response times a linear one). Public: its variables and their
  physical ranges, the response name, and certified reference standards (samples whose true response at stated
  conditions is certified - a laboratory calibration standard);
* an **apparatus** measures one phenomenon inside its own settable range (a sub-box of the physical ranges) with a
  stated precision (public), at a price per measurement (paid to its operator through `bank`), with a finite
  capacity (measurements per day: runs queue behind each other, so other researchers' use delays yours). Hidden:
  its calibration - a reading is `gain * truth + offset + noise`; gain may drift with use.

Measurement: `noise ~ N(0, sd) + N(0, rel * |truth|)`, drawn with the world's seeded randomness when the run
completes (`sci.complete`, a scheduled process); the readings are stored as a JSON dataset artifact owned by the
experimenter (`crucible.substrate.artifacts`) and returned only to the experimenter (or to everyone once shared or
cited in a publication). An apparatus states its precision the way instrument datasheets do: an absolute part
(`stated_sd`) plus a part relative to the reading (`stated_rel`). By default the datasheet is honest (the stated
parts equal the true `sd` and `rel`); a configuration may give an apparatus an optimistic datasheet (`sd_mult`, or a
`stated_rel` below its `rel`), which only replicate measurements reveal.

Science as an institution: `sci.publish` records a paper - a public `docs` document plus a model (an expression in
the law language) and the runs it rests on (which become public); the module computes the model's fit on that data.
`sci.replicate` files a replication report: the module itself tests the paper's model against the reporter's own
completed runs (corrected by the reporter's own measurements of the certified standards on the same apparatus, when
given) with a generalised chi-square test: each reading's variance is the apparatus' stated precision (absolute and
relative parts, at the model's prediction) and the uncertainty of the reporter's own calibration (the weighted
least-squares covariance of gain and offset from the standards it measured) is propagated as a correlated term, so a
true law is refuted at about the nominal rate (`p_threshold`) and a wrong one is refuted as data accumulate. Reports
state whether the reporter is related to the author and whether it used other apparatus than the paper's data
(independence is recorded, never assumed). The same statistic is shown with a paper (`sci.publish`) on the author's
own data: a statistics service over the participant's own measurements, never a reward or an evaluator verdict.

Novel-condition evaluation (`challenge` config, off by default): a standards bureau with a traceable instrument
that can be set anywhere inside the phenomenon's physical ranges - beyond what any bookable apparatus reaches -
announces blind prediction challenges every `every_days`: it measures the response at freshly drawn conditions
outside every bookable range (outside the published data when the apparatus cover the whole physical box), tests
every paper published on the phenomenon against these readings with the same statistic, and publishes the readings
(a public dataset) and the per-paper results (`sci.challenge`). How well a claimed law extrapolates is thus an
observable fact of the world, measured with noise like everything else.

Autonomous participants (module config, all acting only through the tools above, as themselves):

* **rival labs** design their own studies (random designs on the cheapest apparatus), fit the polynomial models
  their methods allow (degree chosen by AIC on their own data), publish, and start new studies; they compete for
  apparatus capacity;
* **replication labs** (funded monthly from outside the world, ledger kind `grant`) pick published papers they have
  not replicated, prefer apparatus the paper did not use, measure the certified standards on it first, measure a
  spread design over its range, and file the replication report computed by the module.

Tools: sci.phenomena, sci.apparatus, sci.run, sci.runs, sci.data, sci.share, sci.publish, sci.papers, sci.paper,
sci.replicate, sci.replications, sci.challenges.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from ..services.identity import ident
from .lawlang import LawError, compile_expr

DAY = 86400
PRICE_KIND = "experiment"
POINT = {"type": "object", "properties": {
    "at": {"type": "object", "additionalProperties": {"type": "number"},
           "description": "condition: variable name -> value (every variable of the phenomenon)"},
    "standard": S.s_str("instead of 'at': measure this certified reference standard", maxLength=40),
    "n": S.s_int("replicate measurements at this point (default 1)", minimum=1, maximum=50)},
    "additionalProperties": False}
RUN_IDS = S.s_arr(S.s_str(maxLength=40), "run ids", maxItems=40)
DEFAULTS = {"max_design": 240, "max_pending": 6, "p_threshold": 0.01}
RIVAL = {"n": 0, "prefix": "rival", "budget": [40_000, 80_000], "points": [6, 12], "every_days": 2, "studies": 3,
         "cooldown_days": [3, 8], "start_day": 0}
REPLICATOR = {"n": 0, "prefix": "repl", "budget_month": 60_000, "points": 8, "reps": 1, "every_days": 1,
              "max_parallel": 2, "start_day": 0, "calib_reps": 2}
CHALLENGE = {"every_days": 14, "start_day": 14, "points": 8, "sd": 0.15, "rel": 0.005, "handle": "bureau",
             "name": "Standards Bureau", "max_papers": 60, "pass_p": 0.01}


def _chi2_sf(x: float, k: int) -> float:
    from scipy.special import gammaincc
    if k <= 0:
        return 1.0
    return float(gammaincc(k / 2.0, max(0.0, x) / 2.0))


@register_module
class Science(Module):
    name = "sci"
    host = "sci"
    depends = ("identity", "bank", "artifacts", "docs")

    # ------------------------------------------------------------------ setup
    def install(self, world):
        st = world.state
        st.declare_table("sci_phen")
        st.declare_table("sci_app", index=("phenomenon",))
        st.declare_table("sci_runs", index=("owner", "apparatus", "status"))
        st.declare_table("sci_papers", index=("author", "phenomenon"))
        st.declare_table("sci_reps", index=("paper", "reporter"))
        st.declare_table("sci_npc", index=("role",))
        self.cfg = {**DEFAULTS, **{k: v for k, v in self.config.items() if k in DEFAULTS}}
        self.laws = {}
        self.vars = {}
        for ph in self.config.get("phenomena") or []:
            names = [v["name"] for v in ph["vars"]]
            self.vars[ph["id"]] = names
            self.laws[ph["id"]] = compile_expr(ph["law"], names)
        self.apps = {a["id"]: a for a in self.config.get("apparatus") or []}
        for a in self.apps.values():
            if a["phenomenon"] not in self.laws:
                raise ValueError(f"sci: apparatus {a['id']} measures unknown phenomenon {a['phenomenon']}")
        st.declare_table("sci_chal", index=("phenomenon",))
        st.declare_table("sci_bureau")
        ch = self.config.get("challenge")
        self.challenge = {**CHALLENGE, **ch} if ch else None

    def truth(self, phenomenon: str, points: list[dict]) -> np.ndarray:
        """Noise-free response (world physics; used by world mechanics and by evaluators, never by a tool)."""
        return self.laws[phenomenon].at(points)

    def _phen_cfg(self, phenomenon: str) -> dict:
        return next(ph for ph in self.config.get("phenomena") or [] if ph["id"] == phenomenon)

    def true_rel(self, apparatus: str) -> float:
        """The apparatus' true relative noise (hidden; default: the phenomenon's `noise_rel`)."""
        a = self.apps[apparatus]
        return float(a["rel"]) if a.get("rel") is not None else float(self._phen_cfg(a["phenomenon"]).get("noise_rel", 0.0))

    def genesis(self, ctx):
        w = ctx.world
        idm, bank = w.modules["identity"], w.modules["bank"]
        ops = {}
        for o in self.config.get("operators") or []:
            owner = idm.create_person(ctx, f"{o['handle']}-head", f"{o['name']} director", "npc", {"role": "owner"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, o["handle"], o["name"], owner, "company", {"sector": "instruments"})
            acct = bank.open_account(ctx, oid, "operating")
            if o.get("cash"):
                bank.move(ctx, "acct_genesis", acct, int(o["cash"]), "endowment", f"capital of {oid}")
            ops[o["handle"]] = (oid, acct)
        for ph in self.config.get("phenomena") or []:
            stds = []
            for s in ph.get("standards") or []:
                v = float(self.truth(ph["id"], [s["at"]])[0])
                stds.append({"id": s["id"], "at": {k: float(x) for k, x in s["at"].items()},
                             "certified": round(v, 4)})
            ctx.emit("sci.phenomenon", {"id": ph["id"], "title": ph.get("title", ph["id"]), "doc": ph.get("doc", ""),
                                        "vars": [{"name": v["name"], "lo": float(v["lo"]), "hi": float(v["hi"]),
                                                  "unit": v.get("unit", "")} for v in ph["vars"]],
                                        "response": ph.get("response", "response"),
                                        "unit": ph.get("unit", ""), "standards": stds}, vis="public")
        for a in self.apps.values():
            oid, acct = ops[a["operator"]]
            ctx.emit("sci.apparatus", {"id": a["id"], "phenomenon": a["phenomenon"], "title": a.get("title", a["id"]),
                                       "operator": oid, "account": acct,
                                       "range": {k: [float(v[0]), float(v[1])] for k, v in a["range"].items()},
                                       "price_cents": int(a["price_cents"]), "stated_sd": float(a["sd"]),
                                       "stated_rel": float(a["stated_rel"]) if a.get("stated_rel") is not None
                                       else self.true_rel(a["id"]),
                                       "capacity_per_day": int(a["capacity_per_day"]),
                                       "setup_s": int(a.get("setup_s", 1800))}, vis="public")
        self._genesis_npcs(ctx)
        if self.challenge:
            ch = self.challenge
            head = idm.create_person(ctx, f"{ch['handle']}-head", f"{ch['name']} director", "npc", {"role": "owner"})
            bank.open_account(ctx, head, "checking")
            bid = idm.create_org(ctx, ch["handle"], ch["name"], head, "group", {"sector": "standards"})
            bank.open_account(ctx, bid, "operating")
            ctx.emit("sci.bureau", {"org": bid, "every_days": int(ch["every_days"]), "points": int(ch["points"]),
                                    "stated_sd": float(ch["sd"]), "stated_rel": float(ch["rel"])}, vis="public")
            ctx.schedule(ctx.now + int(ch["start_day"]) * DAY + 12 * 3600, "sci.challenge", {}, prio=60,
                         period=int(ch["every_days"]) * DAY, key="sci.challenge")

    def _genesis_npcs(self, ctx):
        w = ctx.world
        idm, bank = w.modules["identity"], w.modules["bank"]
        rng = w.rng_for("sci-npcs")
        rows = []
        rv = {**RIVAL, **(self.config.get("rivals") or {})}
        phs = sorted(self.laws)
        endow = []
        for i in range(int(rv["n"])):
            pid = idm.create_person(ctx, f"{rv['prefix']}{i:02d}", f"Rival lab {i}", "npc", {"role": "researcher"})
            acct = bank.open_account(ctx, pid, "checking")
            endow.append((acct, int(rng.integers(rv["budget"][0], rv["budget"][1] + 1))))
            rows.append({"id": pid, "role": "rival", "account": acct, "phenomenon": rv.get("phenomenon") or phs[i % len(phs)],
                         "stage": "design", "runs": [], "studies": 0, "next": int(rv["start_day"]) * DAY + i * 3600,
                         "slot": i})
        rp = {**REPLICATOR, **(self.config.get("replicators") or {})}
        for i in range(int(rp["n"])):
            pid = idm.create_person(ctx, f"{rp['prefix']}{i:02d}", f"Replication lab {i}", "npc",
                                    {"role": "replicator"})
            acct = bank.open_account(ctx, pid, "checking")
            endow.append((acct, int(rp["budget_month"])))
            rows.append({"id": pid, "role": "replicator", "account": acct, "calib": {}, "pending": {}, "done": [],
                         "next": int(rp["start_day"]) * DAY + i * 3600, "slot": i})
        if endow:
            bank.post(ctx, [("acct_genesis", -sum(c for _, c in endow))] + endow, "endowment", "research budgets")
        if rows:
            ctx.emit("sci.npcs", {"rows": rows}, vis="internal")
            ctx.schedule(ctx.now + 3 * 3600, "sci.npc_day", {}, prio=65, period=DAY, key="sci.npc_day")
            if int(rp["n"]):
                ctx.schedule(ctx.now + 30 * DAY, "sci.grants", {}, prio=5, period=30 * DAY, key="sci.grants")

    # ------------------------------------------------------------------ helpers
    def _app(self, st, aid: str) -> dict:
        a = st.get("sci_app", aid)
        if a is None:
            raise NotFound(f"no apparatus {aid}")
        return a

    def _phen(self, st, pid: str) -> dict:
        p = st.get("sci_phen", pid)
        if p is None:
            raise NotFound(f"no phenomenon {pid}")
        return p

    def _run(self, ctx, rid: str, *, readable: bool = True) -> dict:
        r = ctx.state.get("sci_runs", rid)
        if r is None:
            raise NotFound(f"no run {rid}")
        if readable and not (r["shared"] or r["owner"] in ctx.world.principals_of(ctx.actor)):
            raise Forbidden("this run is neither yours nor shared")
        return r

    def _points(self, st, app: dict, design: list) -> list[dict]:
        """Expand a design into measurement points [{at, standard}], validating ranges."""
        ph = self._phen(st, app["phenomenon"])
        names = [v["name"] for v in ph["vars"]]
        stds = {s["id"]: s for s in ph["standards"]}
        out = []
        for i, p in enumerate(design):
            n = int(p.get("n", 1))
            if "standard" in p:
                if "at" in p:
                    raise InvalidAction(f"design[{i}]: give either 'at' or 'standard'")
                s = stds.get(p["standard"])
                if s is None:
                    raise InvalidAction(f"design[{i}]: no standard {p['standard']!r} for {ph['id']}")
                out += [{"at": dict(s["at"]), "standard": s["id"]}] * n
                continue
            at = p.get("at")
            if not isinstance(at, dict) or set(at) != set(names):
                raise InvalidAction(f"design[{i}]: 'at' must set exactly the variables {names}")
            for k in names:
                lo, hi = app["range"][k]
                v = float(at[k])
                if not (math.isfinite(v) and lo <= v <= hi):
                    raise InvalidAction(f"design[{i}]: {k}={at[k]} outside this apparatus' range [{lo}, {hi}]")
            out += [{"at": {k: float(at[k]) for k in names}}] * n
        return out

    def readings(self, st, rid: str) -> list[dict]:
        r = st.tables["sci_runs"][rid]
        return [{"at": p["at"], "standard": p.get("standard"), "value": v} for p, v in zip(r["points"], r["values"] or [])]

    @staticmethod
    def precision(app: dict, values) -> np.ndarray:
        """Stated variance of a reading at true response `values`: the apparatus' datasheet precision (absolute part
        plus a part relative to the response)."""
        v = np.abs(np.asarray(values, dtype=float))
        return float(app["stated_sd"]) ** 2 + (float(app.get("stated_rel", 0.0)) * v) ** 2

    def correction(self, st, owner: str, apparatus: str, calibration_runs: list[str]) -> tuple:
        """(gain, offset, n standards, cov) estimated by weighted least squares from the certified standards measured
        in `calibration_runs` (completed runs of `owner` on `apparatus`), with the 2x2 covariance of (offset, gain)
        under the apparatus' stated precision. With one distinct standard only the offset is corrected (gain 1);
        without standards: (1, 0, 0, zeros)."""
        app = st.tables["sci_app"][apparatus]
        stds = {s["id"]: s["certified"] for s in st.tables["sci_phen"][app["phenomenon"]]["standards"]}
        xs, ys = [], []
        for rid in calibration_runs:
            r = st.get("sci_runs", rid)
            if r is None or r["owner"] != owner or r["apparatus"] != apparatus or r["status"] != "done":
                continue
            for p, v in zip(r["points"], r["values"]):
                if p.get("standard") in stds:
                    xs.append(stds[p["standard"]])
                    ys.append(v)
        zero = np.zeros((2, 2))
        if not xs:
            return 1.0, 0.0, 0, zero
        xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
        wts = 1.0 / np.maximum(self.precision(app, xs), 1e-12)

        def offset_only():
            o = float(np.sum(wts * (ys - xs)) / np.sum(wts))
            return 1.0, o, len(xs), np.array([[1.0 / float(np.sum(wts)), 0.0], [0.0, 0.0]])
        if len(set(np.round(xs, 6))) < 2:
            return offset_only()
        A = np.stack([np.ones(len(xs)), xs], 1)
        info = A.T @ (A * wts[:, None])
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return offset_only()
        o, g = cov @ (A.T @ (wts * ys))
        if not (0.2 < g < 5.0):
            return offset_only()
        return float(g), float(o), len(xs), cov

    def fit_stats(self, st, model, owner: str, runs: list[str], calibration_runs: list[str]) -> dict:
        """Generalised chi-square test of `model` against the readings of `runs` (all by `owner`), corrected per
        apparatus by the owner's standards in `calibration_runs`. Per apparatus, the covariance of the corrected
        readings is the stated precision at the model's prediction (diagonal) plus the propagated uncertainty of the
        owner's gain/offset estimate (one correlated term shared by every reading of that apparatus)."""
        z2, n = 0.0, 0
        res, ymeans = [], []
        by_app: dict[str, list] = {}
        for rid in runs:
            r = st.tables["sci_runs"][rid]
            for p, v in zip(r["points"], r["values"]):
                if not p.get("standard"):
                    by_app.setdefault(r["apparatus"], []).append((p["at"], float(v)))
        for aid in sorted(by_app):
            app = st.tables["sci_app"][aid]
            pts = [a for a, _ in by_app[aid]]
            raw = np.array([v for _, v in by_app[aid]], dtype=float)
            g, o, _, cov = self.correction(st, owner, aid, calibration_runs)
            y = (raw - o) / g
            m = model.at(pts)
            if not np.all(np.isfinite(m)):
                return {"n": len(pts), "chi2": None, "p": 0.0, "rmse": None, "rel_err": None, "valid": False}
            d = y - m
            var = self.precision(app, m) / (g * g)
            J = np.stack([-np.ones(len(y)) / g, -y / g], 1)
            sigma = np.diag(np.maximum(var, 1e-12)) + J @ cov @ J.T
            try:
                z2 += float(d @ np.linalg.solve(sigma, d))
            except np.linalg.LinAlgError:
                z2 += float(np.sum(d * d / np.maximum(var, 1e-12)))
            n += len(pts)
            res.append(d)
            ymeans.append(np.abs(y))
        if n == 0:
            return {"n": 0, "chi2": None, "p": None, "rmse": None, "rel_err": None, "valid": False}
        d = np.concatenate(res)
        scale = float(np.mean(np.concatenate(ymeans))) or 1.0
        return {"n": n, "chi2": round(z2, 4), "p": round(_chi2_sf(z2, n), 6), "rmse": round(float(np.sqrt(np.mean(d ** 2))), 6),
                "rel_err": round(float(np.mean(np.abs(d))) / scale, 6), "valid": True}

    # ------------------------------------------------------------------ public tools
    @endpoint("phenomena", S.obj(), read_only=True)
    def ep_phenomena(self, ctx):
        """Phenomena under study: their condition variables and physical ranges, the response measured, and the
        certified reference standards (true response at stated conditions) available for calibration."""
        return [dict(p, id=pid) for pid, p in ctx.state.scan("sci_phen")]

    @endpoint("apparatus", S.obj({"phenomenon": S.s_str(maxLength=40)}), read_only=True)
    def ep_apparatus(self, ctx, phenomenon=None):
        """Measuring apparatus: phenomenon, settable range, price per measurement, stated precision (standard
        deviation of a reading: `stated_sd` plus `stated_rel` times the response), capacity, current queue
        (measurements waiting) and operator."""
        out = []
        for aid, a in ctx.state.scan("sci_app"):
            if phenomenon is not None and a["phenomenon"] != phenomenon:
                continue
            q = max(0, int(a["busy_until"]) - ctx.now)
            out.append({"apparatus": aid, "phenomenon": a["phenomenon"], "title": a["title"], "operator": a["operator"],
                        "range": a["range"], "price_cents": a["price_cents"], "stated_sd": a["stated_sd"],
                        "stated_rel": a.get("stated_rel", 0.0),
                        "capacity_per_day": a["capacity_per_day"], "queue_wait_s": q, "measured": a["measured"]})
        return out

    @endpoint("run", S.obj({"apparatus": S.s_str(maxLength=40), "design": S.s_arr(POINT, "measurement points",
                                                                                  minItems=1, maxItems=240),
                            "account": S.s_str("account to pay from (default: your primary account)", maxLength=40),
                            "note": S.s_str(maxLength=300)}, ["apparatus", "design"]), cost_s=300)
    def ep_run(self, ctx, apparatus, design, account=None, note=""):
        """Book measurements on an apparatus. Each point is measured `n` times (or a certified standard is measured
        instead). The price per measurement is paid now; the run joins the apparatus' queue and its readings arrive
        when it completes (an event, then sci.data)."""
        st = ctx.state
        app = self._app(st, apparatus)
        pts = self._points(st, app, design)
        if len(pts) > int(self.cfg["max_design"]):
            raise InvalidAction(f"at most {self.cfg['max_design']} measurements per run")
        pending = [r for r in st.find("sci_runs", "owner", ctx.actor) if r["status"] == "queued"]
        if len(pending) >= int(self.cfg["max_pending"]):
            raise Conflict(f"you already have {len(pending)} runs waiting")
        bank = ctx.world.modules["bank"]
        if account is None:
            account = bank.primary_account(st, ctx.actor)
            if account is None:
                raise Forbidden("you need a bank account")
        cost = int(app["price_cents"]) * len(pts)
        ident(ctx.world).authorize(ctx, bank.account_owner(st, account), "bank.spend", amount=cost, scope=account)
        rid = st.new_id("srun")
        bank.move(ctx, account, app["account"], cost, PRICE_KIND, f"{len(pts)} measurements on {apparatus}", ref=rid)
        start = max(ctx.now, int(app["busy_until"]))
        dur = int(app["setup_s"]) + int(math.ceil(len(pts) * DAY / max(1, int(app["capacity_per_day"]))))
        ready = start + dur
        ctx.emit("sci.booked", {"id": rid, "owner": ctx.actor, "apparatus": apparatus, "phenomenon": app["phenomenon"],
                                "points": pts, "n": len(pts), "cost": cost, "ready_at": ready, "note": note[:300]},
                 vis=(ctx.actor,))
        ctx.emit("sci.queue", {"apparatus": apparatus, "busy_until": ready, "add": len(pts)}, vis="public")
        ctx.schedule(ready, "sci.complete", {"run": rid}, prio=40)
        return {"run": rid, "measurements": len(pts), "cost_cents": cost, "ready_at": ready}

    @endpoint("runs", S.obj({"status": S.s_enum(["queued", "done"])}), read_only=True)
    def ep_runs(self, ctx, status=None):
        """Your runs (newest last): apparatus, size, cost, status and when it is or was ready."""
        out = []
        st = ctx.state
        for rid in st.find_ids("sci_runs", "owner", ctx.actor):     # (rows carry no id: the id is the key)
            r = st.tables["sci_runs"][rid]
            if status is None or r["status"] == status:
                out.append({"run": rid, "apparatus": r["apparatus"], "phenomenon": r["phenomenon"],
                            "measurements": r["n"], "cost_cents": r["cost"], "status": r["status"],
                            "ready_at": r["ready_at"], "dataset": r["dataset"], "shared": r["shared"]})
        return out

    @endpoint("data", S.obj({"run": S.s_str(maxLength=40)}, ["run"]), read_only=True)
    def ep_data(self, ctx, run):
        """Readings of one completed run that is yours, shared, or cited by a publication: one row per measurement
        (the condition, or the standard measured, and the reading)."""
        r = self._run(ctx, run)
        if r["status"] != "done":
            raise Conflict(f"run {run} is still queued (ready at {r['ready_at']})")
        return {"run": run, "apparatus": r["apparatus"], "phenomenon": r["phenomenon"], "owner": r["owner"],
                "dataset": r["dataset"], "readings": self.readings(ctx.state, run)}

    @endpoint("share", S.obj({"run": S.s_str(maxLength=40)}, ["run"]), cost_s=10)
    def ep_share(self, ctx, run):
        """Make one of your completed runs (and its dataset artifact) public."""
        r = self._run(ctx, run)
        if r["owner"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("not your run")
        if r["status"] != "done":
            raise Conflict("run not completed")
        self._share(ctx, run)
        return {"run": run, "shared": True}

    def _share(self, ctx, rid: str) -> None:
        r = ctx.state.tables["sci_runs"][rid]
        if r["shared"]:
            return
        from ..substrate import artifacts
        ctx.emit("sci.shared", {"run": rid}, vis="public")
        if r["dataset"]:
            ctx.run_as(r["owner"], lambda c: artifacts.publish(c, r["dataset"]))

    @endpoint("publish", S.obj({"phenomenon": S.s_str(maxLength=40), "title": S.s_str(minLength=1, maxLength=200),
                                "abstract": S.s_str(maxLength=4000),
                                "model": S.s_str("the law you claim, as an expression over the phenomenon's variables",
                                                 maxLength=2000),
                                "data": RUN_IDS, "calibration_runs": RUN_IDS}, ["phenomenon", "title", "model", "data"]),
              cost_s=1800)
    def ep_publish(self, ctx, phenomenon, title, model, data, abstract="", calibration_runs=()):
        """Publish a paper: a public document, the model you claim (expression language: + - * / **, comparisons,
        'a if c else b', exp log sqrt abs min max tanh sigmoid step clip pow, numeric constants and the
        phenomenon's variable names) and the runs it rests on (your own completed runs; they become public). The
        fit of the model to that data is computed and shown with the paper; independent labs may replicate it."""
        st = ctx.state
        self._phen(st, phenomenon)
        try:
            m = compile_expr(model, self.vars[phenomenon])
        except LawError as e:
            raise InvalidAction(f"model: {e}") from None
        runs = list(dict.fromkeys(data))
        cal = list(dict.fromkeys(calibration_runs or []))
        for rid in runs + cal:
            r = self._run(ctx, rid)
            if r["owner"] not in ctx.world.principals_of(ctx.actor):
                raise Forbidden(f"run {rid} is not yours")
            if r["status"] != "done" or r["phenomenon"] != phenomenon:
                raise InvalidAction(f"run {rid} is not a completed run on {phenomenon}")
        stats = self.fit_stats(st, m, st.tables["sci_runs"][runs[0]]["owner"], runs, cal)
        body = (f"{abstract}\n\nModel ({phenomenon}): {m.text}\nData: {', '.join(runs)}\n"
                f"Calibration: {', '.join(cal) or 'none'}\nFit: {json.dumps(stats)}")
        doc = ctx.call("docs.create", {"title": title, "body": body[:20000], "visibility": "public"})["doc"]
        pid = st.new_id("paper")
        apps = sorted({st.tables["sci_runs"][r]["apparatus"] for r in runs})
        for rid in runs + cal:
            self._share(ctx, rid)
        ctx.emit("sci.paper", {"id": pid, "author": ctx.actor, "phenomenon": phenomenon, "title": title,
                               "abstract": abstract[:4000], "model": m.text, "data": runs, "calibration_runs": cal,
                               "apparatus": apps, "doc": doc, "fit": stats}, vis="public")
        return {"paper": pid, "doc": doc, "fit": stats}

    @endpoint("papers", S.obj({"phenomenon": S.s_str(maxLength=40), "author": S.s_str(maxLength=60),
                               "limit": S.s_int(minimum=1, maximum=200)}), read_only=True)
    def ep_papers(self, ctx, phenomenon=None, author=None, limit=50):
        """Published papers (oldest first) with their model, fit and replication tally."""
        out = []
        for pid, p in ctx.state.scan("sci_papers"):
            if (phenomenon and p["phenomenon"] != phenomenon) or (author and p["author"] != author):
                continue
            out.append(self._public_paper(ctx.state, pid, p))
        return out[:limit]

    def _public_paper(self, st, pid, p):
        reps = [st.tables["sci_reps"][r] for r in st.find_ids("sci_reps", "paper", pid)]
        return {"paper": pid, "author": p["author"], "phenomenon": p["phenomenon"], "title": p["title"],
                "model": p["model"], "data": list(p["data"]), "apparatus": list(p["apparatus"]), "doc": p["doc"],
                "fit": p["fit"], "t": p["t"],
                "replications": {"consistent": sum(1 for r in reps if r["verdict"] == "consistent"),
                                 "refuted": sum(1 for r in reps if r["verdict"] == "refuted"),
                                 "independent_consistent": sum(1 for r in reps if r["verdict"] == "consistent"
                                                               and r["independent"])}}

    @endpoint("paper", S.obj({"paper": S.s_str(maxLength=40)}, ["paper"]), read_only=True)
    def ep_paper(self, ctx, paper):
        """One paper with its replication reports."""
        p = ctx.state.get("sci_papers", paper)
        if p is None:
            raise NotFound(f"no paper {paper}")
        out = self._public_paper(ctx.state, paper, p)
        out["abstract"] = p["abstract"]
        out["reports"] = [self._public_rep(dict(ctx.state.tables["sci_reps"][x], id=x))
                          for x in ctx.state.find_ids("sci_reps", "paper", paper)]
        return out

    def _public_rep(self, r):
        return {k: r[k] for k in ("id", "paper", "reporter", "apparatus", "n", "chi2", "p", "rel_err", "verdict",
                                  "related_to_author", "other_apparatus", "independent", "t")}

    @endpoint("replicate", S.obj({"paper": S.s_str(maxLength=40), "runs": RUN_IDS, "calibration_runs": RUN_IDS},
                                 ["paper", "runs"]), cost_s=600)
    def ep_replicate(self, ctx, paper, runs, calibration_runs=()):
        """File a replication report on a paper from your own completed runs on its phenomenon. The model is tested
        against your readings (corrected with your measurements of certified standards on the same apparatus, if you
        give calibration runs) by a generalised chi-square test using the apparatus' stated precision and the
        uncertainty of your calibration; the report is public."""
        st = ctx.state
        p = st.get("sci_papers", paper)
        if p is None:
            raise NotFound(f"no paper {paper}")
        runs = list(dict.fromkeys(runs))
        cal = list(dict.fromkeys(calibration_runs or []))
        for rid in runs + cal:
            r = self._run(ctx, rid)
            if r["owner"] != ctx.actor:
                raise Forbidden(f"run {rid} is not yours")
            if r["status"] != "done" or r["phenomenon"] != p["phenomenon"]:
                raise InvalidAction(f"run {rid} is not a completed run on {p['phenomenon']}")
            if rid in p["data"]:
                raise InvalidAction(f"run {rid} is part of the paper's own data")
        if any(r["paper"] == paper and r["reporter"] == ctx.actor for r in st.find("sci_reps", "reporter", ctx.actor)):
            raise Conflict("you already reported on this paper")
        m = compile_expr(p["model"], self.vars[p["phenomenon"]])
        stats = self.fit_stats(st, m, ctx.actor, runs, cal)
        if not stats["n"]:
            raise InvalidAction("no measurements at conditions (only standards) in these runs")
        from ..goals.metrics import is_related
        related = bool(is_related(ctx.world, p["author"], ctx.actor))
        apps = sorted({st.tables["sci_runs"][r]["apparatus"] for r in runs})
        other = not (set(apps) & set(p["apparatus"]))
        verdict = "consistent" if stats["valid"] and stats["p"] >= float(self.cfg["p_threshold"]) else "refuted"
        rid = st.new_id("srep")
        ctx.emit("sci.replication", {"id": rid, "paper": paper, "reporter": ctx.actor, "runs": runs,
                                     "calibration_runs": cal, "apparatus": apps, "n": stats["n"], "chi2": stats["chi2"],
                                     "p": stats["p"], "rel_err": stats["rel_err"], "verdict": verdict,
                                     "related_to_author": related, "other_apparatus": other,
                                     "independent": (not related) and other}, vis="public")
        for x in runs + cal:
            self._share(ctx, x)
        return {"report": rid, "verdict": verdict, "p": stats["p"], "n": stats["n"], "independent": (not related) and other}

    @endpoint("replications", S.obj({"reporter": S.s_str(maxLength=60)}), read_only=True)
    def ep_replications(self, ctx, reporter=None):
        """Replication reports (all, or by one reporter)."""
        st = ctx.state
        ids = st.find_ids("sci_reps", "reporter", reporter) if reporter else [k for k, _ in st.scan("sci_reps")]
        return [self._public_rep(dict(st.tables["sci_reps"][k], id=k)) for k in ids]

    # ------------------------------------------------------------------ processes
    @process("complete")
    def pr_complete(self, ctx, run):
        """A queued run completes: the apparatus measures every point (world physics plus noise) and the readings
        are stored as a dataset artifact owned by the experimenter."""
        st = ctx.state
        r = st.get("sci_runs", run)
        if r is None or r["status"] != "queued":
            return
        acfg = self.apps[r["apparatus"]]
        app = st.tables["sci_app"][r["apparatus"]]
        truth = self.truth(r["phenomenon"], [p["at"] for p in r["points"]])
        rng = ctx.rng("sci-measure", run)
        k = np.arange(len(truth), dtype=float) + float(app["measured"])
        gain = float(acfg.get("gain", 1.0)) + float(acfg.get("drift_per_measure", 0.0)) * k
        sd = float(acfg["sd"]) * float(acfg.get("sd_mult", 1.0))
        rel = self.true_rel(r["apparatus"])
        noise = rng.normal(0.0, sd, len(truth)) + rel * np.abs(truth) * rng.normal(0.0, 1.0, len(truth))
        vals = gain * truth + float(acfg.get("offset", 0.0)) + noise
        values = [round(float(v), 4) for v in vals]
        from ..substrate import artifacts
        rows = [{"at": p["at"], "standard": p.get("standard"), "value": v} for p, v in zip(r["points"], values)]
        data = json.dumps({"run": run, "apparatus": r["apparatus"], "phenomenon": r["phenomenon"], "readings": rows},
                          sort_keys=True).encode()
        ref = artifacts.put(ctx, data, "application/json", owner=r["owner"],
                            meta={"run": run, "apparatus": r["apparatus"]}, kind="dataset")
        ctx.emit("sci.completed", {"id": run, "values": values, "dataset": ref}, vis=(r["owner"],))

    @process("challenge")
    def pr_challenge(self, ctx):
        """The standards bureau's blind prediction challenge (see the module doc): fresh conditions outside every
        bookable range, measured on its traceable instrument, every paper on the phenomenon tested against them."""
        ch = self.challenge
        if not ch:
            return
        st = ctx.state
        k = len(st.tables["sci_chal"])
        bureau = next((r for _, r in st.scan("sci_bureau")), None) if "sci_bureau" in st.tables else None
        if bureau is None:
            return
        from ..substrate import artifacts
        for pid in sorted(self.laws):
            papers = [(x, p) for x, p in st.scan("sci_papers") if p["phenomenon"] == pid][: int(ch["max_papers"])]
            if not papers:
                continue
            ph = st.tables["sci_phen"][pid]
            names = [v["name"] for v in ph["vars"]]
            rng = ctx.rng("sci-challenge", pid, k)
            pts = self.novel_points(st, pid, papers, rng, int(ch["points"]))
            truth = self.truth(pid, pts)
            sd, rel = float(ch["sd"]), float(ch["rel"])
            vals = truth + rng.normal(0.0, sd, len(pts)) + rel * np.abs(truth) * rng.normal(0.0, 1.0, len(pts))
            vals = np.round(vals, 4)
            results = []
            for x, p in papers:
                m = compile_expr(p["model"], names).at(pts)
                if not np.all(np.isfinite(m)):
                    results.append({"paper": x, "valid": False, "chi2": None, "p": 0.0, "rmse": None,
                                    "rel_err": None, "verdict": "failed"})
                    continue
                d = vals - m
                var = np.maximum(sd ** 2 + (rel * np.abs(m)) ** 2, 1e-12)
                chi2 = float(np.sum(d * d / var))
                pv = _chi2_sf(chi2, len(pts))
                results.append({"paper": x, "valid": True, "chi2": round(chi2, 4), "p": round(pv, 6),
                                "rmse": round(float(np.sqrt(np.mean(d * d))), 6),
                                "rel_err": round(float(np.mean(np.abs(d))) / (float(np.mean(np.abs(vals))) or 1.0), 6),
                                "verdict": "passed" if pv >= float(ch["pass_p"]) else "failed"})
            rows = [{"at": a, "value": float(v)} for a, v in zip(pts, vals)]
            cid = f"chal_{k:04d}_{pid}"
            data = json.dumps({"challenge": cid, "phenomenon": pid, "readings": rows}, sort_keys=True).encode()
            ref = ctx.run_as(bureau["org"], lambda c: artifacts.put(c, data, "application/json", owner=bureau["org"],
                                                                     meta={"challenge": cid}, kind="dataset"))
            ctx.run_as(bureau["org"], lambda c: artifacts.publish(c, ref))
            # the bureau publishes its challenge as itself (an institution's decision, not a system record)
            payload = {"id": cid, "phenomenon": pid, "readings": rows, "dataset": ref, "stated_sd": sd,
                       "stated_rel": rel, "results": results}
            ctx.run_as(bureau["org"], lambda c: c.emit("sci.challenge", payload, vis="public"))

    def novel_points(self, st, phenomenon: str, papers, rng, n: int) -> list[dict]:
        """Conditions no bookable apparatus can set (some variable outside the union of the apparatus' ranges, inside
        the physical range); when the apparatus cover the whole physical box: outside the bounding box of the
        papers' data; otherwise anywhere in the physical box."""
        ph = st.tables["sci_phen"][phenomenon]
        names = [v["name"] for v in ph["vars"]]
        lo = np.array([float(v["lo"]) for v in ph["vars"]])
        hi = np.array([float(v["hi"]) for v in ph["vars"]])
        apps = [a for _, a in st.scan("sci_app") if a["phenomenon"] == phenomenon]
        alo = np.array([min(a["range"][k][0] for a in apps) for k in names]) if apps else hi
        ahi = np.array([max(a["range"][k][1] for a in apps) for k in names]) if apps else lo
        dlo, dhi = hi.copy(), lo.copy()
        for _, p in papers:
            for rid in p["data"]:
                r = st.get("sci_runs", rid)
                for q in (r["points"] if r else []):
                    if not q.get("standard"):
                        x = np.array([float(q["at"][k]) for k in names])
                        dlo, dhi = np.minimum(dlo, x), np.maximum(dhi, x)
        out = []
        for box in ((alo, ahi), (dlo, dhi), None):
            for _ in range(400 * n):
                if len(out) >= n:
                    break
                x = rng.uniform(lo, hi)
                if box is None or np.any((x < box[0] - 1e-9) | (x > box[1] + 1e-9)):
                    out.append({k: round(float(v), 4) for k, v in zip(names, x)})
            if len(out) >= n:
                break
        return out[:n]

    @endpoint("challenges", S.obj({"phenomenon": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=100)}),
              read_only=True)
    def ep_challenges(self, ctx, phenomenon=None, limit=20):
        """The standards bureau's blind prediction challenges (newest last): the conditions it measured (outside
        every bookable range), its readings and stated precision, and how each published paper's model fared."""
        out = []
        for cid, c in ctx.state.scan("sci_chal"):
            if phenomenon is None or c["phenomenon"] == phenomenon:
                out.append(dict(c, challenge=cid))
        return out[-int(limit):]

    @process("grants")
    def pr_grants(self, ctx):
        """Monthly outside funding of the replication labs (exogenous research funding)."""
        rp = {**REPLICATOR, **(self.config.get("replicators") or {})}
        posts = [(r["account"], int(rp["budget_month"])) for _, r in ctx.state.scan("sci_npc") if r["role"] == "replicator"]
        if posts:
            ctx.world.modules["bank"].post(ctx, [("acct_external", -sum(c for _, c in posts))] + posts, "grant",
                                           "replication funding")

    @process("npc_day")
    def pr_npc_day(self, ctx):
        for pid in ctx.state.ids("sci_npc"):
            r = ctx.state.tables["sci_npc"][pid]
            if ctx.now < int(r["next"]):
                continue
            fn = self._rival if r["role"] == "rival" else self._replicator
            err = contain(ctx, fn, ctx, pid)
            if err is not None:
                ctx.emit("sci.npc_failed", {"id": pid, "error": f"{type(err).__name__}: {err}"[:200]}, vis="internal")

    def _call(self, ctx, who, tool, args):
        try:
            return ctx.call(tool, args, as_principal=who), None
        except CrucibleError as e:
            return None, e

    def _set(self, ctx, pid, **ch):
        ctx.emit("sci.npc", {"id": pid, "changes": ch}, vis="internal")

    # ------------------------------------------------------------------ rival labs
    def _rival(self, ctx, pid):
        st = ctx.state
        r = st.tables["sci_npc"][pid]
        cfg = {**RIVAL, **(self.config.get("rivals") or {})}
        rng = ctx.rng("sci-rival", pid)
        nxt = ctx.now + int(cfg["every_days"]) * DAY
        if r["stage"] == "design":
            apps, _ = self._call(ctx, pid, "sci.apparatus", {"phenomenon": r["phenomenon"]})
            if not apps:
                return self._set(ctx, pid, next=nxt)
            a = sorted(apps, key=lambda x: (x["price_cents"], x["apparatus"]))[0]
            k = int(rng.integers(cfg["points"][0], cfg["points"][1] + 1))
            design = [{"at": {v: round(float(rng.uniform(*a["range"][v])), 3) for v in sorted(a["range"])}, "n": 1}
                      for _ in range(k)]
            res, err = self._call(ctx, pid, "sci.run", {"apparatus": a["apparatus"], "design": design})
            if err is not None:
                return self._set(ctx, pid, next=nxt + 5 * DAY)
            return self._set(ctx, pid, stage="wait", runs=[res["run"]], next=int(res["ready_at"]) + 600)
        if r["stage"] == "wait":
            runs = [st.tables["sci_runs"][x] for x in r["runs"]]
            if any(x["status"] != "done" for x in runs):
                return self._set(ctx, pid, next=nxt)
            model = self._poly_fit(st, r["phenomenon"], r["runs"])
            if model is None:
                return self._set(ctx, pid, stage="design", next=nxt)
            res, err = self._call(ctx, pid, "sci.publish", {"phenomenon": r["phenomenon"],
                                                            "title": f"An empirical law for {r['phenomenon']}",
                                                            "abstract": "Least-squares polynomial fit.",
                                                            "model": model, "data": list(r["runs"])})
            studies = int(r["studies"]) + 1
            cool = int(rng.integers(cfg["cooldown_days"][0], cfg["cooldown_days"][1] + 1))
            return self._set(ctx, pid, stage="design" if studies < int(cfg["studies"]) else "done", runs=[],
                             studies=studies, next=ctx.now + cool * DAY)

    def _poly_fit(self, st, phenomenon, runs) -> str | None:
        """The rival labs' method: a polynomial of degree 1 or 2 in every variable (AIC chooses)."""
        names = self.vars[phenomenon]
        X, y = [], []
        for rid in runs:
            for row in self.readings(st, rid):
                if row["standard"] is None:
                    X.append([row["at"][k] for k in names])
                    y.append(row["value"])
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        best = None
        for deg in (1, 2):
            cols = [np.ones(len(y))] + [X[:, i] for i in range(len(names))]
            terms = ["1"] + list(names)
            if deg == 2:
                cols += [X[:, i] ** 2 for i in range(len(names))]
                terms += [f"{k} ** 2" for k in names]
            A = np.stack(cols, 1)
            if len(y) <= A.shape[1]:
                continue
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            rss = float(np.sum((A @ coef - y) ** 2))
            aic = len(y) * math.log(max(rss, 1e-12) / len(y)) + 2 * A.shape[1]
            if best is None or aic < best[0]:
                best = (aic, coef, terms)
        if best is None:
            return None
        _, coef, terms = best
        parts = [f"{round(float(c), 6)!r}" if t == "1" else f"{round(float(c), 6)!r} * {t}" for c, t in zip(coef, terms)]
        return " + ".join(parts).replace("+ -", "- ")

    # ------------------------------------------------------------------ replication labs
    def _replicator(self, ctx, pid):
        st = ctx.state
        r = st.tables["sci_npc"][pid]
        cfg = {**REPLICATOR, **(self.config.get("replicators") or {})}
        nxt = ctx.now + int(cfg["every_days"]) * DAY
        pending = dict(r["pending"])
        calib = dict(r["calib"])
        done = list(r["done"])
        # 1. finished work: calibrations become usable, completed measurements become reports
        for paper, job in sorted(pending.items()):
            if job["stage"] == "calibrating":
                c = st.get("sci_runs", job["cal"])
                if c is not None and c["status"] == "done":
                    calib[job["app"]] = job["cal"]
                    pending[paper] = dict(job, stage="ready")
                continue
            if job["stage"] == "measuring":
                runs = [st.tables["sci_runs"][x] for x in [job["run"]] + ([job["cal"]] if job["cal"] else [])]
                if all(x["status"] == "done" for x in runs):
                    self._call(ctx, pid, "sci.replicate", {"paper": paper, "runs": [job["run"]],
                                                           "calibration_runs": [job["cal"]] if job["cal"] else []})
                    pending.pop(paper)
                    done.append(paper)
        # 2. measurement runs for papers whose apparatus is now calibrated
        for paper, job in sorted(pending.items()):
            if job["stage"] == "ready":
                rid = self._replication_run(ctx, pid, job["app"], cfg)
                if rid is not None:
                    pending[paper] = dict(job, run=rid, cal=calib.get(job["app"]), stage="measuring")
        # 3. new papers
        if len(pending) < int(cfg["max_parallel"]):
            papers, _ = self._call(ctx, pid, "sci.papers", {"limit": 200})
            for p in papers or []:
                if len(pending) >= int(cfg["max_parallel"]):
                    break
                if p["paper"] in done or p["paper"] in pending or p["author"] == pid:
                    continue
                apps, _ = self._call(ctx, pid, "sci.apparatus", {"phenomenon": p["phenomenon"]})
                if not apps:
                    continue
                apps.sort(key=lambda a: (a["apparatus"] in p["apparatus"], a["queue_wait_s"] > 3 * DAY,
                                         a["price_cents"], a["apparatus"]))
                a = apps[0]["apparatus"]
                phen = st.tables["sci_phen"][p["phenomenon"]]
                if a in calib or not phen["standards"]:
                    rid = self._replication_run(ctx, pid, a, cfg)
                    if rid is not None:
                        pending[p["paper"]] = {"app": a, "run": rid, "cal": calib.get(a), "stage": "measuring"}
                else:
                    design = [{"standard": s["id"], "n": int(cfg["calib_reps"])} for s in phen["standards"]]
                    res, err = self._call(ctx, pid, "sci.run", {"apparatus": a, "design": design})
                    if err is None:
                        pending[p["paper"]] = {"app": a, "run": None, "cal": res["run"], "stage": "calibrating"}
        self._set(ctx, pid, pending=pending, calib=calib, done=done, next=nxt)

    def _replication_run(self, ctx, pid, apparatus, cfg) -> str | None:
        a = ctx.state.tables["sci_app"][apparatus]
        names = sorted(a["range"])
        k = int(cfg["points"])
        per = max(2, int(round(k ** (1.0 / max(1, len(names))))))
        grids = [np.linspace(a["range"][v][0], a["range"][v][1], per) for v in names]
        mesh = np.meshgrid(*grids, indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], 1)
        design = [{"at": {v: round(float(x), 4) for v, x in zip(names, row)}, "n": int(cfg["reps"])} for row in pts]
        res, err = self._call(ctx, pid, "sci.run", {"apparatus": apparatus, "design": design})
        return None if err is not None else res["run"]

    # ------------------------------------------------------------------ appliers
    @applier("sci.phenomenon")
    def ap_phen(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sci_phen", p.pop("id"), p)

    @applier("sci.apparatus")
    def ap_app(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sci_app", p.pop("id"), dict(p, busy_until=0, measured=0, runs=0))

    @applier("sci.booked")
    def ap_booked(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sci_runs", p.pop("id"), dict(p, status="queued", values=None, dataset=None,
                                                             shared=False, t=ev.t))

    @applier("sci.queue")
    def ap_queue(self, ev):
        st = self.world.state
        a = st.tables["sci_app"][ev.payload["apparatus"]]
        st.patch("sci_app", ev.payload["apparatus"], busy_until=int(ev.payload["busy_until"]))

    @applier("sci.completed")
    def ap_completed(self, ev):
        st = self.world.state
        r = st.tables["sci_runs"][ev.payload["id"]]
        st.patch("sci_runs", ev.payload["id"], status="done", values=list(ev.payload["values"]),
                 dataset=ev.payload["dataset"], done_t=ev.t)
        a = st.tables["sci_app"][r["apparatus"]]
        st.patch("sci_app", r["apparatus"], measured=int(a["measured"]) + int(r["n"]), runs=int(a["runs"]) + 1)

    @applier("sci.shared")
    def ap_shared(self, ev):
        self.world.state.patch("sci_runs", ev.payload["run"], shared=True)

    @applier("sci.paper")
    def ap_paper(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sci_papers", p.pop("id"), dict(p, t=ev.t))

    @applier("sci.replication")
    def ap_rep(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sci_reps", p.pop("id"), dict(p, t=ev.t))

    @applier("sci.npcs")
    def ap_npcs(self, ev):
        for r in ev.payload["rows"]:
            self.world.state.insert("sci_npc", r["id"], {k: v for k, v in r.items() if k != "id"})

    @applier("sci.npc")
    def ap_npc(self, ev):
        self.world.state.patch("sci_npc", ev.payload["id"], **ev.payload["changes"])

    @applier("sci.npc_failed")
    def ap_npc_failed(self, ev):
        pass

    @applier("sci.bureau")
    def ap_bureau(self, ev):
        self.world.state.insert("sci_bureau", ev.payload["org"], dict(ev.payload))

    @applier("sci.challenge")
    def ap_challenge(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("sci_chal", p.pop("id"), dict(p, t=ev.t))

    def check_invariants(self, world):
        errs = []
        st = world.state
        for rid, r in st.scan("sci_runs"):
            if r["status"] == "done" and (r["values"] is None or len(r["values"]) != r["n"]):
                errs.append(f"run {rid}: {r['n']} measurements, {len(r['values'] or [])} readings")
        return errs

    def observe(self, world, actor):
        st = world.state
        mine = st.find("sci_runs", "owner", actor)
        if not mine and not st.find_ids("sci_papers", "author", actor):
            return None
        return {"runs_queued": sum(1 for r in mine if r["status"] == "queued"),
                "runs_done": sum(1 for r in mine if r["status"] == "done"),
                "papers": len(st.find_ids("sci_papers", "author", actor))}
