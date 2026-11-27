"""Tier 2 domain 9.6 Technology invention (workstream I-2): invent a technology that measurably improves another
industry.

The world (a restriction of the civilisation generator, D-17): the industry of a civilisation profile - firms run by
organisational minds that make materials, components and products on machines whose hidden operating optima they
do not know (each runs its process designs at the operating point it started with), regional grids, freight, the one
population as households buying the goods (`civgoods`) - plus a technology market (`techmkt`, below).

A technology is EXECUTABLE: a program in a repository implementing the public interface `process_control/1` -
`recommend(request) -> {"speed": s, "temp": t}`, where the request is the adopting plant's OWN production history on
one machine model (operating point, units started, units made, energy of each run) and the machine's limits. The
inventor declares it (`tech.declare`: repository, entry point, compatible machine models, licence fee per run).

Plants are the industry's firms. A plant whose measured yield on a machine model is below what it aspires to
publishes a need (`tech.needs`: the operation and machine model, never its numbers). It considers declared
technologies compatible with its models; it tries one by running the program in the sandbox on its own history
every day of a trial (the program may explore), operating its production design at the recommended point; at the
end of the trial it compares its own measured cost per good unit (input units and energy per unit made) during the
trial's last days with its runs before, keeps the technology (paying the licence per run from then on, re-running
the program weekly) or reverts to its old point and does not try it again. Technical constraints are enforced by the
world: the sandbox's time and memory limits, the interface's types, the machine's operating range; a program that
fails a constraint fails the trial. Nothing about who adopts is scripted: it follows the plants' own measurements.

Objective (public): plants of an industry adopt the technology and that industry measurably improves. Evaluator
(evaluator side): adopters are independent plants that kept the technology on their own measurements; the gain is
the expected output per unit of input and energy of every unit they produced since adopting, at the operating points
they ran, against the same units at the point they ran before (the counterfactual without the technology), from the
world's hidden physics - noise-free, so a technology that changes nothing scores zero however lucky a trial was.

Configurations (structure measured on the built world: which operation the market serves, supply-chain shape,
freight topology shape, energy regime):

    train   forming_standard   technologies for forming machines (components) in the standard chain on a ring
    train   assembly_diamond   technologies for assembly machines (products) in a diamond chain on a star
    train   extraction_linear  technologies for extraction (mines) in a linear chain on a line, scarce energy
    eval_compositional  forming_diamond   forming in the diamond chain on a star (seen parts, new combination)
    eval_structural     any_wide          any operation in a wide chain on a complete graph with scarce energy and
                                          clustered households (never seen in training)
"""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np

from ..core import schema as S
from ..core.canonical import to_jsonable
from ..core.errors import CrucibleError, Forbidden, InvalidAction, NotFound, ProtocolError, contain
from ..core.kernel import Module, applier, endpoint, process, register_module
from ..services.identity import ident
from . import base
from ..goals.metrics import metric as goal_metric
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, ScriptedReference, Seat, act_

NAME = "invention"
SEAT = "inventor"
HORIZON_DAYS = 40.0
N_POP = 400
DAY = 86400
TARGET = {1: {"adopters": 1, "gain": 0.03}, 2: {"adopters": 2, "gain": 0.06}, 3: {"adopters": 3, "gain": 0.10}}
INTERFACE = "process_control/1"
HARNESS_FILE = "_crucible_tech_harness.py"
HARNESS_TARGET = "_crucible_tech_harness:run"
HARNESS = '''"""Installed by Crucible next to your files: calls your technology once per request."""
import importlib
import json
import os
import traceback


def run(job, caller=None):
    mod_name, _, fn_name = job["target"].partition(":")
    if mod_name.endswith(".py"):
        mod_name = mod_name[:-3]
    mod_name = mod_name.replace("/", ".")
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name or "recommend")
    except BaseException as e:  # noqa: BLE001
        return {"load_error": (type(e).__name__ + ": " + str(e).replace(os.getcwd() + os.sep, ""))[:300]}
    try:
        out = fn(job["request"])
        json.dumps(out)
        return {"ok": True, "out": out}
    except BaseException as e:  # noqa: BLE001
        tb = traceback.extract_tb(e.__traceback__)
        where = " at %s:%d" % (os.path.basename(tb[-1].filename), tb[-1].lineno) if tb else ""
        return {"ok": False, "error": (type(e).__name__ + where)[:300]}
'''
DEFAULTS = {"ops": ["form"], "aspiration": [0.82, 0.93], "try_p": 0.5, "trial_days": 10, "eval_days": 4,
            "min_trial_units": 150, "max_trial_days": 16, "keep_gain": 0.02, "review_every_days": 7, "timeout_s": 5.0,
            "history": 40, "vendors": True}

# Technologies already on the market at genesis (real programs, run in the sandbox like any other): a legacy
# vendor's controller for machine models outside this market (plants record it as incompatible with theirs) and a
# free open-source controller written against the previous interface version (it answers `temperature` where
# process_control/1 asks `temp`, so the world's interface check rejects it in every trial) supporting half of the
# market's machine models.
LEGACY_SOURCE = '''"""Legacy process controller (process_control/1): nudge speed up while yield holds."""


def recommend(request):
    h = request.get("history") or []
    if not h:
        return {"speed": 0.5, "temp": 0.5}
    last = h[-1]
    y = last["made"] / max(1, last["units"])
    step = 0.02 if y >= 0.85 else -0.02
    return {"speed": min(1.0, max(0.0, last["speed"] + step)), "temp": last["temp"]}
'''
OPENCTL_SOURCE = '''"""openctl 0.9: a free process controller (written for process_control/0)."""


def recommend(request):
    h = request.get("history") or []
    best = max(h, key=lambda r: r["made"] / max(1, r["units"])) if h else {"speed": 0.5, "temp": 0.5}
    return {"speed": round(best["speed"], 4), "temperature": round(best["temp"], 4)}
'''


# ============================================================================================ the technology market
@register_module
class TechMarket(Module):
    """Kernel module `techmkt` (host `tech`): executable process technologies and their adoption by the industry's
    plants (see the module docstring above). Fidelity record FR-TECH-ADOPT (plausible, unvalidated: trial-based
    adoption on the plant's own measured outcomes, an aspiration level, one trial per technology)."""
    name = "techmkt"
    host = "tech"
    version = "1"
    depends = ("identity", "bank", "code", "industry", "society")

    def __init__(self, config=None):
        super().__init__(config)
        cfg = dict(DEFAULTS)
        for k, v in (self.config or {}).items():
            if k not in DEFAULTS:
                raise ProtocolError(f"techmkt: unknown config key {k!r}")
            cfg[k] = v
        self.cfg = cfg

    def install(self, world):
        st = world.state
        st.declare_table("tech_inventions", index=("owner", "status"))
        st.declare_table("tech_trials", index=("plant", "invention", "status"))
        st.declare_table("tech_needs", index=("plant",))
        st.declare_table("tech_market")

    def genesis(self, ctx):
        # the market's scope as built: the machine operations whose plants it serves
        ctx.emit("tech.market", {"ops": sorted(self.cfg["ops"]), "interface": INTERFACE}, vis="public")
        if self.cfg["vendors"]:
            self._vendors(ctx)
        ctx.schedule(ctx.now + 2 * DAY + 11 * 3600, "techmkt.review", {}, prio=70, period=DAY, key="techmkt.review")

    def _vendors(self, ctx) -> None:
        """The technologies on the market at genesis (see LEGACY_SOURCE / OPENCTL_SOURCE)."""
        from .common import make_org
        models = ctx.world.modules["industry"].phys["models"]
        ops = self.cfg["ops"]
        market = sorted(m for m, d in models.items() if d["op"] in ops or "any" in ops)
        other = sorted(m for m in models if m not in market)
        code = ctx.world.modules["code"]
        offers = []
        if other:
            offers.append(("legacyctl", "Legacy Controls", "Legacy process controller", LEGACY_SOURCE, other, 4_000))
        if market:
            half = market[::2]
            offers.append(("openctl", "openctl project", "openctl 0.9 (free)", OPENCTL_SOURCE, half, 0))
        for handle, name, title, src, mods, fee in offers:
            org, lead = make_org(ctx, handle, name, "company")
            rid = code.create_repo(ctx, org, "controls", "public", title, "open")
            ref = code.commit_files(ctx, rid, {"control.py": src}, title, lead)
            ctx.emit("tech.declared", {"id": ctx.state.new_id("tech"), "owner": org, "repo": rid, "ref": ref,
                                       "target": "control.py:recommend", "models": list(mods),
                                       "licence_cents_per_run": int(fee), "title": title, "doc": title,
                                       "interface": INTERFACE, "status": "active"}, vis="public")

    # ------------------------------------------------------------------ plants and their own records
    def plants(self, st) -> list[str]:
        """The industry's firms run by organisational minds that own machines of the market's operations."""
        ind = self.world.modules["industry"]
        out = set()
        for _, m in st.scan("minds"):
            org = m.get("org")
            if not org or m.get("kind") != "firm":
                continue
            for mid in st.find_ids("machines", "owner", org):
                mdl = ind.phys["models"][st.tables["machines"][mid]["model"]]
                if mdl["op"] in self.cfg["ops"] or "any" in self.cfg["ops"]:
                    out.add(org)
        return sorted(out)

    def lines(self, st, org) -> list[dict]:
        """(design, machine, model) of the plant's own production on machines of the market's operations."""
        ind = self.world.modules["industry"]
        seen, out = set(), []
        for _, r in sorted(st.scan("runs")):
            if r["owner"] != org:
                continue
            m = st.tables["machines"].get(r["machine"])
            if m is None:
                continue
            mdl = ind.phys["models"][m["model"]]
            if not (mdl["op"] in self.cfg["ops"] or "any" in self.cfg["ops"]):
                continue
            key = (r["design"], m["model"])
            if key not in seen:
                seen.add(key)
                out.append({"design": r["design"], "machine": r["machine"], "model": m["model"], "op": mdl["op"]})
        return out

    def history(self, st, org, model) -> list[dict]:
        """The plant's own finished runs on machines of `model`, newest last (what it would hand a technology)."""
        out = []
        for _, r in sorted(st.scan("runs")):
            if r["owner"] != org or r["status"] != "done":
                continue
            m = st.tables["machines"].get(r["machine"])
            if m is None or m["model"] != model:
                continue
            out.append({"speed": round(float(r["speed"]), 4), "temp": round(float(r["temp"]), 4),
                        "units": int(r["units"]), "made": int(r.get("made") or 0), "kwh": round(float(r["kwh"]), 3),
                        "t": int(r.get("t0") or 0), "broke": bool(r.get("broke"))})
        return out[-int(self.cfg["history"]):]

    @staticmethod
    def cost_per_good(rows: list[dict]) -> float | None:
        """Input units plus energy (kWh, at a nominal 0.1 input unit per kWh) per unit made over runs (runs a
        breakdown interrupted are a maintenance matter, not the process's: left out)."""
        rows = [r for r in rows if not r.get("broke")]
        units = sum(r["units"] for r in rows)
        made = sum(r["made"] for r in rows)
        kwh = sum(r["kwh"] for r in rows)
        if made <= 0:
            return None
        return (units + 0.1 * kwh) / made

    # ------------------------------------------------------------------ running a technology (sandbox)
    def recommend(self, ctx, inv: dict, plant: str, request: dict) -> dict:
        """Run the technology on a plant's request in the sandbox (recorded, so replay does not rerun it) and check
        the interface's constraints. Returns {ok, speed, temp} or {ok: False, error}."""
        code = ctx.world.modules["code"]
        files = dict(code.files_with_deps(ctx, inv["repo"], inv["ref"], inv["owner"]))
        files[HARNESS_FILE] = HARNESS
        pool = code.sandbox_pool(ctx.world)
        key = f"{ctx.world.world_id}|tech|{inv['repo']}|{inv['ref']}"
        job = {"kind": "call", "target": HARNESS_TARGET, "caller": inv["owner"],
               "args": {"target": inv["target"], "request": request}}
        timeout = float(self.cfg["timeout_s"])

        def go():
            w = pool.get(key, lambda: files)
            r = w.run(job, rpc=None, timeout=timeout)
            res = r.get("result") if isinstance(r.get("result"), dict) else {}
            return {"ok": bool(r.get("ok")), "result": res, "error": str(r.get("error") or "")[:200]}
        res = ctx.external("tech.recommend", go)
        out = res.get("result") or {}
        if not res.get("ok") or out.get("load_error") or not out.get("ok"):
            return {"ok": False, "error": (out.get("load_error") or out.get("error") or res.get("error")
                                           or "the program did not run")[:200]}
        rec = out.get("out")
        if not isinstance(rec, dict) or not all(isinstance(rec.get(k), (int, float)) and not isinstance(rec.get(k), bool)
                                                for k in ("speed", "temp")):
            return {"ok": False, "error": "the output is not {speed: number, temp: number} (interface "
                                          f"{INTERFACE})"}
        s, t = float(rec["speed"]), float(rec["temp"])
        if not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0) or not (math.isfinite(s) and math.isfinite(t)):
            return {"ok": False, "error": "the operating point is outside the machine's range [0, 1]"}
        return {"ok": True, "speed": round(s, 4), "temp": round(t, 4)}

    # ------------------------------------------------------------------ the plants' daily review
    @process("review")
    def pr_review(self, ctx):
        st = ctx.state
        for org in self.plants(st):
            err = contain(ctx, self._review, ctx, org)
            if err is not None:
                ctx.emit("tech.failed", {"plant": org, "error": f"{type(err).__name__}: {err}"[:200]}, vis="internal")

    def _owner_of(self, st, org) -> str | None:
        for _, m in st.scan("minds"):
            if m.get("org") == org:
                return m.get("owner")
        return None

    def _review(self, ctx, org):
        st = ctx.state
        cfg = self.cfg
        rng = ctx.rng("techmkt.review", org)
        who = self._owner_of(st, org)
        if who is None:
            return
        asp = float(ctx.world.rng_for("techmkt-aspiration", org).uniform(*cfg["aspiration"]))
        for line in self.lines(st, org):
            hist = self.history(st, org, line["model"])
            if len(hist) < 2:
                continue
            active = [st.tables["tech_trials"][t] | {"id": t} for t in st.find_ids("tech_trials", "plant", org)
                      if st.tables["tech_trials"][t]["design"] == line["design"] and
                      st.tables["tech_trials"][t]["status"] in ("trial", "adopted")]
            if active:
                self._continue(ctx, org, who, line, active[0], hist)
                continue
            made = sum(r["made"] for r in hist[-6:])
            units = sum(r["units"] for r in hist[-6:])
            yld = made / units if units else 1.0
            need_id = f"{org}|{line['design']}"
            if yld < asp and not st.has("tech_needs", need_id):
                ctx.emit("tech.need", {"id": need_id, "plant": org, "op": line["op"], "model": line["model"]},
                         vis="public")
            if yld >= asp:
                continue
            tried = {st.tables["tech_trials"][t]["invention"] for t in st.find_ids("tech_trials", "plant", org)}
            cands = []
            for iid in st.find_ids("tech_inventions", "status", "active"):
                inv = st.tables["tech_inventions"][iid]
                if iid in tried or inv["owner"] in (org, who):
                    continue
                if line["model"] not in inv["models"]:
                    ctx.emit("tech.incompatible", {"plant": org, "invention": iid, "model": line["model"]},
                             vis="internal")
                    continue
                cands.append((int(inv["licence_cents_per_run"]), iid))
            if not cands or rng.random() > float(cfg["try_p"]):
                continue
            _, iid = sorted(cands)[0]                 # the cheapest compatible technology first
            self._start_trial(ctx, org, who, line, iid, hist)

    def _start_trial(self, ctx, org, who, line, iid, hist):
        st = ctx.state
        inv = st.tables["tech_inventions"][iid]
        d = st.tables["designs"][line["design"]]
        tid = st.new_id("ttrial")
        base_rows = [r for r in hist if abs(r["speed"] - d["speed"]) < 1e-6 and abs(r["temp"] - d["temp"]) < 1e-6
                     and r["t"] >= ctx.now - 14 * DAY]
        ctx.emit("tech.trial", {"id": tid, "plant": org, "invention": iid, "design": line["design"],
                                "model": line["model"], "from": [d["speed"], d["temp"]], "started": ctx.now,
                                "baseline": self.cost_per_good(base_rows or hist), "status": "trial",
                                "inventor": inv["owner"]}, vis=(org, inv["owner"]))
        self._apply(ctx, org, who, line, st.tables["tech_trials"][tid] | {"id": tid}, hist)

    def _apply(self, ctx, org, who, line, trial, hist) -> bool:
        st = ctx.state
        inv = st.tables["tech_inventions"][trial["invention"]]
        req = {"interface": INTERFACE, "model": line["model"], "operation": line["op"], "history": hist,
               "limits": {"speed": [0.0, 1.0], "temp": [0.0, 1.0]}}
        rec = self.recommend(ctx, inv, org, req)
        if not rec["ok"]:
            ctx.emit("tech.rejected", {"trial": trial["id"], "plant": org, "invention": trial["invention"],
                                       "why": rec["error"]}, vis=(org, inv["owner"]))
            self._end(ctx, org, who, line, trial, "failed", rec["error"])
            return False
        ctx.emit("tech.recommended", {"trial": trial["id"], "plant": org, "invention": trial["invention"],
                                      "speed": rec["speed"], "temp": rec["temp"]}, vis=(org,))
        try:
            ctx.call("make.tune", {"design": line["design"], "speed": rec["speed"], "temp": rec["temp"]},
                     as_principal=who)
        except CrucibleError as e:
            self._end(ctx, org, who, line, trial, "failed", e.message[:200])
            return False
        return True

    def _continue(self, ctx, org, who, line, trial, hist):
        st = ctx.state
        cfg = self.cfg
        days = (ctx.now - trial["started"]) / DAY
        inv = st.tables["tech_inventions"].get(trial["invention"])
        if inv is None or inv["status"] != "active":
            return self._end(ctx, org, who, line, trial, "reverted", "the technology was withdrawn")
        if trial["status"] == "trial":
            recent = [r for r in hist if r["t"] >= ctx.now - float(cfg["eval_days"]) * DAY and r["t"] >= trial["started"]]
            enough = sum(r["units"] for r in recent) >= int(cfg["min_trial_units"])
            if days < float(cfg["trial_days"]) or (not enough and days < float(cfg["max_trial_days"])):
                self._apply(ctx, org, who, line, trial, hist)
                return
            now_c = self.cost_per_good(recent)
            base_c = trial["baseline"]
            ok = now_c is not None and base_c is not None and now_c <= base_c * (1.0 - float(cfg["keep_gain"]))
            ctx.emit("tech.trial_done", {"trial": trial["id"], "plant": org, "invention": trial["invention"],
                                         "baseline": base_c, "trial": now_c, "kept": bool(ok)},
                     vis=(org, inv["owner"]))
            if ok:
                ctx.emit("tech.adopted", {"trial": trial["id"], "plant": org, "invention": trial["invention"],
                                          "at": ctx.now}, vis="public")
            else:
                self._end(ctx, org, who, line, trial, "reverted", "no improvement in our own measurements")
            return
        # adopted: pay the licence for the runs since the last payment, re-run the technology now and then
        runs = [r for r in hist if r["t"] > int(trial.get("paid_until") or trial["started"])]
        fee = int(inv["licence_cents_per_run"]) * len(runs)
        if fee > 0:
            bank = ctx.world.modules["bank"]
            src = bank.primary_account(st, org)
            dst = bank.primary_account(st, inv["owner"])
            if src and dst:
                pay = min(fee, max(0, bank.available(st, src)))
                if pay > 0:
                    bank.move(ctx, src, dst, pay, "licence", f"technology {trial['invention']}: {len(runs)} runs")
                ctx.emit("tech.licence_paid", {"trial": trial["id"], "plant": org, "invention": trial["invention"],
                                               "runs": len(runs), "cents": int(pay),
                                               "until": max(r["t"] for r in runs)}, vis=(org, inv["owner"]))
        if days >= float(cfg["trial_days"]) + float(cfg["review_every_days"]) * (1 + int(trial.get("reviews", 0))):
            self._apply(ctx, org, who, line, trial, hist)
            ctx.emit("tech.reviewed", {"trial": trial["id"]}, vis=(org,))

    def _end(self, ctx, org, who, line, trial, status, why):
        try:
            ctx.call("make.tune", {"design": line["design"], "speed": trial["from"][0], "temp": trial["from"][1]},
                     as_principal=who)
        except CrucibleError:
            pass
        ctx.emit("tech.trial_status", {"trial": trial["id"], "status": status, "why": why[:200]},
                 vis=(org, trial["inventor"]))

    # ------------------------------------------------------------------ tools
    @endpoint("declare", S.obj({"repo": S.s_str(maxLength=81), "target": S.s_str("module.py:function", maxLength=120),
                                "models": S.s_arr(S.s_str(maxLength=20), "machine models it supports", maxItems=12),
                                "licence_cents_per_run": S.s_int(minimum=0, maximum=10_000_000),
                                "title": S.s_str(maxLength=120), "doc": S.s_str(maxLength=2000),
                                "as_org": S.s_str(maxLength=40)}, ["repo", "target", "models"]), cost_s=60)
    def ep_declare(self, ctx, repo, target, models, licence_cents_per_run=0, title="", doc="", as_org=None):
        """Declare an executable technology of interface process_control/1 (`recommend(request) -> {speed, temp}`;
        request: {interface, model, operation, history: [{speed, temp, units, made, kwh, t, broke}], limits}). Plants of
        the industry try it on their own production when it supports their machine model."""
        st = ctx.state
        owner = ctx.actor
        if as_org:
            owner = ident(ctx.world).resolve_handle(st, as_org)
            ident(ctx.world).authorize(ctx, owner, "org.admin")
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        ref = code.repo(st, repo)["head"]
        if ref is None:
            raise InvalidAction(f"{repo} is empty")
        mod = target.partition(":")[0]
        if (mod if mod.endswith(".py") else mod.replace(".", "/") + ".py") not in code.tree(st, repo, ref):
            raise NotFound(f"{mod} not in {repo}")
        known = set(ctx.world.modules["industry"].phys["models"])
        bad = sorted(set(models) - known)
        if bad:
            raise InvalidAction(f"unknown machine models {bad}")
        iid = st.new_id("tech")
        ctx.emit("tech.declared", {"id": iid, "owner": owner, "repo": repo, "ref": ref, "target": target,
                                   "models": sorted(set(models)), "licence_cents_per_run": int(licence_cents_per_run),
                                   "title": title[:120], "doc": doc[:2000], "interface": INTERFACE, "status": "active"},
                 vis="public")
        return {"invention": iid, "ref": ref}

    @endpoint("inventions", S.obj(), read_only=True)
    def ep_inventions(self, ctx):
        """Declared technologies (public): owner, title, models, licence, adopters."""
        st = ctx.state
        out = []
        for iid, inv in sorted(st.scan("tech_inventions")):
            adopters = {st.tables["tech_trials"][t]["plant"] for t in st.find_ids("tech_trials", "invention", iid)
                        if st.tables["tech_trials"][t]["status"] == "adopted"}
            out.append({"invention": iid, **{k: inv[k] for k in ("owner", "title", "models", "licence_cents_per_run",
                                                                 "status", "interface")}, "adopters": len(adopters)})
        return out

    @endpoint("needs", S.obj(), read_only=True)
    def ep_needs(self, ctx):
        """What the industry's plants want improved (public): operation and machine model, how many plants."""
        cnt: dict = {}
        for _, n in ctx.state.scan("tech_needs"):
            k = (n["op"], n["model"])
            cnt[k] = cnt.get(k, 0) + 1
        return [{"operation": k[0], "model": k[1], "plants": v} for k, v in sorted(cnt.items())]

    @endpoint("mine", S.obj({"as_org": S.s_str(maxLength=40)}), read_only=True)
    def ep_mine(self, ctx, as_org=None):
        """Your technologies' trials and adoptions (the plants' own verdicts), licence fees received."""
        st = ctx.state
        owner = ident(ctx.world).resolve_handle(st, as_org) if as_org else ctx.actor
        if as_org:
            ident(ctx.world).authorize(ctx, owner, "org.admin")
        out = []
        for iid in st.find_ids("tech_inventions", "owner", owner):
            trials = [st.tables["tech_trials"][t] for t in st.find_ids("tech_trials", "invention", iid)]
            out.append({"invention": iid, "trials": len(trials),
                        "adopted": sum(1 for t in trials if t["status"] == "adopted"),
                        "failed": [t.get("why") for t in trials if t["status"] == "failed"][:5],
                        "reverted": sum(1 for t in trials if t["status"] == "reverted"),
                        "fees_cents": sum(int(t.get("fees", 0)) for t in trials)})
        return out

    @endpoint("withdraw", S.obj({"invention": S.s_str(maxLength=40)}, ["invention"]))
    def ep_withdraw(self, ctx, invention):
        st = ctx.state
        inv = st.get("tech_inventions", invention)
        if inv is None:
            raise NotFound("no invention")
        if inv["owner"] not in ctx.world.principals_of(ctx.actor):
            raise Forbidden("not yours")
        ctx.emit("tech.withdrawn", {"id": invention}, vis="public")
        return {"status": "withdrawn"}

    # ------------------------------------------------------------------ appliers
    @applier("tech.market")
    def ap_market(self, ev):
        self.world.state.put("tech_market", "scope", dict(ev.payload))

    @applier("tech.declared")
    def ap_declared(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("tech_inventions", p.pop("id"), p | {"t": ev.t})

    @applier("tech.withdrawn")
    def ap_withdrawn(self, ev):
        self.world.state.patch("tech_inventions", ev.payload["id"], status="withdrawn")

    @applier("tech.need")
    def ap_need(self, ev):
        p = dict(ev.payload)
        self.world.state.put("tech_needs", p.pop("id"), p | {"t": ev.t})

    @applier("tech.trial")
    def ap_trial(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("tech_trials", p.pop("id"), p | {"t": ev.t, "fees": 0, "reviews": 0})

    @applier("tech.trial_status")
    def ap_trial_status(self, ev):
        self.world.state.patch("tech_trials", ev.payload["trial"], status=ev.payload["status"], why=ev.payload["why"])

    @applier("tech.adopted")
    def ap_adopted(self, ev):
        self.world.state.patch("tech_trials", ev.payload["trial"], status="adopted", adopted_at=ev.payload["at"],
                               paid_until=ev.payload["at"])

    @applier("tech.licence_paid")
    def ap_paid(self, ev):
        st = self.world.state
        t = st.tables["tech_trials"][ev.payload["trial"]]
        st.patch("tech_trials", ev.payload["trial"], fees=int(t.get("fees", 0)) + int(ev.payload["cents"]),
                 paid_until=int(ev.payload["until"]))

    @applier("tech.reviewed")
    def ap_reviewed(self, ev):
        st = self.world.state
        t = st.tables["tech_trials"][ev.payload["trial"]]
        st.patch("tech_trials", ev.payload["trial"], reviews=int(t.get("reviews", 0)) + 1)

    @applier("tech.recommended")
    def ap_recommended(self, ev):
        return None

    @applier("tech.rejected")
    def ap_rejected(self, ev):
        return None

    @applier("tech.trial_done")
    def ap_trial_done(self, ev):
        return None

    @applier("tech.incompatible")
    def ap_incompatible(self, ev):
        return None

    @applier("tech.failed")
    def ap_failed(self, ev):
        return None


# ============================================================================================ world
SHAPES = {
    "forming_standard": {"ops": ["form"], "chain": "standard", "regions": 4, "links": "ring", "energy": "normal",
                         "placement": "mixed"},
    "assembly_diamond": {"ops": ["assemble"], "chain": "diamond", "regions": 4, "links": "star", "energy": "normal",
                         "placement": "mixed"},
    "extraction_linear": {"ops": ["extract"], "chain": "linear", "regions": 3, "links": "line", "energy": "scarce",
                          "placement": "mixed"},
    "forming_diamond": {"ops": ["form"], "chain": "diamond", "regions": 4, "links": "star", "energy": "normal",
                        "placement": "mixed"},
    "any_wide": {"ops": ["any"], "chain": "wide", "regions": 4, "links": "complete", "energy": "scarce",
                 "placement": "blocks"},
}
SPLIT = {"forming_standard": "train", "assembly_diamond": "train", "extraction_linear": "train",
         "forming_diamond": "eval_compositional", "any_wide": "eval_structural"}
DOCS = {
    "forming_standard": "a market for technologies of forming machines (component makers) in the standard chain on "
                        "a four-region ring",
    "assembly_diamond": "technologies for assembly machines (product makers) in a diamond chain on a star network",
    "extraction_linear": "technologies for extraction machines (mines) along a linear chain on a line, scarce energy",
    "forming_diamond": "forming technologies in the diamond chain on a star: parts seen in training, new combination",
    "any_wide": "technologies for any operation in a wide chain on a complete graph, scarce energy, clustered "
                "households: structure no training world has",
}


def _profile(config):
    from ..civ import profiles as P
    bal = P.PROFILES["balanced"]
    sh = SHAPES[config.name]
    return replace(bal, name=f"invention_{config.name}",
                   industry={**bal.industry, "chain": sh["chain"], "regions": sh["regions"], "links": sh["links"]},
                   energy={**bal.energy, "regime": sh["energy"]}, social={**bal.social, "placement": sh["placement"]})


def build(config, seed, overrides):
    pop = {"n": int(overrides.get("n", N_POP)), "dims": {"H": 16}}
    tm = {"ops": list(SHAPES[config.name]["ops"])}
    tm.update(overrides.get("techmkt") or {})
    return base.civ_recipe(seed, profile=_profile(config), subsystems=["industry", "supply_chains"],
                           modules=["civgoods", "civgen", "civlinks", "market", "code", "techmkt", "seat",
                                    "elsewhere"],
                           population_modules=("pop",), module_config={"pop": pop, "techmkt": tm},
                           params=overrides.get("params"))


def tier3(config, merged):
    if "techmkt" not in merged["module_list"]:
        merged["module_list"].append("techmkt")
    merged["modules"]["techmkt"] = {"ops": list(SHAPES[config.name]["ops"])}
    return merged


def structure(world):
    from .base import _freight_shape, _supply_shape
    st = world.state
    grids = sorted({"scarce" if (r.get("grid") or {}).get("capacity", 0) < 6000 else "ample"
                    for _, r in st.scan("regions")})
    scope = st.get("tech_market", "scope") or {}
    # (the population's placement is measured by the framework's core fingerprint as social locality)
    return {"market_ops": sorted(scope.get("ops") or []), "supply_chain": _supply_shape(world),
            "freight": _freight_shape(world), "grids": grids}


# ============================================================================================ objective and evaluator
def _draw(rng, difficulty, world):
    """The goal generator's draw: the number of independent adopting plants and the efficiency gain of the level,
    for the operations the world's technology market serves (built state)."""
    t = TARGET.get(int(difficulty), TARGET[3])
    scope = world.state.get("tech_market", "scope") or {}
    return {"adopters": int(t["adopters"]), "gain": float(t["gain"]), "ops": sorted(scope.get("ops") or []),
            "interface": str(scope.get("interface") or INTERFACE), "window_days": HORIZON_DAYS}


def _text(p):
    return (f"Invent an executable technology (interface {p['interface']}: tech.declare) that measurably improves "
            f"this world's industry ({', '.join(p['ops'])} machines): within {p['window_days']:.0f} days at least "
            f"{p['adopters']} independent plants must adopt it on their own measurements, and their output per unit "
            f"of input and energy from then on must beat what the same plants achieve without it by at least "
            f"{100 * p['gain']:.0f}%.")


@goal_metric("invention_independent_adopters", "EVALUATOR-ONLY. Independent plants that adopted a technology of the "
             "principal on their own measurements.")
def _m_adopters(world, principal):
    return float(evaluate(world, principal, {})["details"]["adopters"])


@goal_metric("invention_expected_gain", "EVALUATOR-ONLY. Expected output per unit of input and energy of the "
             "adopters' runs since adoption vs. the same units at their previous operating point (hidden physics).")
def _m_gain(world, principal):
    return float(evaluate(world, principal, {})["details"].get("gain", 0.0))


GOAL = GoalFamily(
    "adopted_process_technology", draw=_draw, text=_text,
    success=lambda p: {"all": [{"metric": "invention_independent_adopters", "args": {"principal": "$self"},
                                "op": ">=", "value": int(p["adopters"])},
                               {"metric": "invention_expected_gain", "args": {"principal": "$self"}, "op": ">=",
                                "value": float(p["gain"])}]},
    progress=lambda p: {"metric": "invention_expected_gain", "args": {"principal": "$self"},
                        "target": float(p["gain"])},
    window_days=lambda p: float(p["window_days"]),
    doc="independent plants adopt the technology on their own measurements and their expected efficiency rises")


def objective(world, principal, difficulty):
    """The objective generated for a participant (what `Domain.make_objective` returns)."""
    return DOMAIN.make_objective(world, principal, difficulty)


def expected_efficiency(ind, model: str, speed: float, temp: float) -> float:
    """Evaluator side: the expected units made per unit of input and energy (0.1 input unit per kWh) at an
    operating point, from the machine model's hidden physics (`economy.industry`: yield = y_max (0.6 + 0.4 q),
    energy per unit = e (0.6 + 0.8 temp + 0.4 speed^2)); wear scales every point alike and cancels in ratios."""
    md = ind.phys["models"][model]
    y = md["y_max"] * (0.6 + 0.4 * ind.quality(md, speed, temp))
    return y / (1.0 + 0.1 * ind.energy_per_unit(md, speed, temp))


def evaluate(world, principal, obj):
    """Adopters: independent plants that kept the technology on their own measurements. Gain: over every unit the
    adopters produced on the adopted line since adopting, the expected output per unit of input and energy at the
    operating points they actually ran, against the same units at the operating point they ran before (the
    counterfactual without the technology), from the world's hidden physics - noise-free, so a technology that
    changes nothing scores zero however lucky a trial was."""
    from ..goals.metrics import is_related
    st = world.state
    ind = world.modules["industry"]
    mine = set(world.principals_of(principal)) | {principal}
    trials = [t for _, t in sorted(st.scan("tech_trials")) if t["status"] == "adopted" and t["inventor"] in mine and
              not any(is_related(world, t["plant"], p) for p in mine)]
    adopters = sorted({t["plant"] for t in trials})
    p = (obj or {}).get("params") or {}
    need_n = int(p.get("adopters", 1))
    if not adopters:
        return {"success": False, "score": 0.0, "details": {"adopters": 0}}
    num = den = 0.0
    units = 0
    for t in trials:
        for _, r in st.scan("runs"):
            if r["design"] != t["design"] or r["owner"] != t["plant"] or int(r.get("t0") or 0) < int(t["adopted_at"]):
                continue
            m = st.tables["machines"].get(r["machine"])
            if m is None:
                continue
            u = int(r["units"])
            num += u * expected_efficiency(ind, m["model"], float(r["speed"]), float(r["temp"]))
            den += u * expected_efficiency(ind, m["model"], float(t["from"][0]), float(t["from"][1]))
            units += u
    gain = num / den - 1.0 if den > 0 else 0.0
    target = float(p.get("gain", TARGET[1]["gain"]))
    fees = sum(int(t.get("fees", 0)) for _, t in st.scan("tech_trials") if t["inventor"] in mine)
    ok = len(adopters) >= need_n and gain >= target
    score = 0.5 * min(1.0, len(adopters) / need_n) + 0.5 * max(0.0, min(1.0, gain / target))
    return {"success": bool(ok), "score": round(float(score), 4),
            "details": {"adopters": len(adopters), "plants": adopters, "units_since_adoption": units,
                        "gain": round(float(gain), 4), "licence_fees_cents": fees}}


# ============================================================================================ policies
PATTERN_SEARCH = '''"""Adaptive process control (interface process_control/1): a star experiment and a quadratic fit on the plant's
own runs.

It scores each operating point the plant has run by units made per unit of input and energy. Around the plant's own
starting point it first tries the four points one step (0.15) away along each axis; with those five points it fits a
parabola along each axis and recommends the vertex (at most 0.3 from the start, inside the machine's range); later it
recommends whichever measured point scored best. A plant whose runs are slow (under one a day) has no time for a
five-point star within a trial: it tries one step up along each axis and then the step that looks better along each
axis (down where up was worse), then whichever measured point scored best."""


def _score(rows):
    rows = [r for r in rows if not r.get("broke")] or rows
    units = sum(r["units"] for r in rows)
    made = sum(r["made"] for r in rows)
    kwh = sum(r["kwh"] for r in rows)
    return made / (units + 0.1 * kwh) if units else 0.0


def _vertex(x0, f_minus, f0, f_plus, h):
    den = f_minus - 2.0 * f0 + f_plus
    if den >= -1e-9:                       # not concave: go to the better side
        return x0 + (h if f_plus > f_minus else -h)
    return x0 + h * (f_minus - f_plus) / (2.0 * den)


def _clip(x):
    return round(min(0.95, max(0.05, x)), 3)


def _slow(start, pts, h):
    up = [(_clip(start[0] + h), start[1]), (start[0], _clip(start[1] + h))]
    for c in up:
        if c not in pts:
            return {"speed": c[0], "temp": c[1]}
    sc = {k: _score(v) for k, v in pts.items()}
    f0 = sc[start]
    ds = h if sc[up[0]] > f0 else -h
    dt = h if sc[up[1]] > f0 else -h
    v = (_clip(start[0] + ds), _clip(start[1] + dt))
    if v not in pts:
        return {"speed": v[0], "temp": v[1]}
    best = max(sc, key=lambda k: (sc[k], k))
    return {"speed": best[0], "temp": best[1]}


def recommend(request):
    hist = request.get("history") or []
    if not hist:
        return {"speed": 0.5, "temp": 0.5}
    pts = {}
    for r in hist:
        pts.setdefault((round(r["speed"], 3), round(r["temp"], 3)), []).append(r)
    start = (round(hist[0]["speed"], 3), round(hist[0]["temp"], 3))
    h = 0.15
    span = (hist[-1]["t"] - hist[0]["t"]) / 86400.0 if len(hist) > 1 else 0.0
    if len(hist) > 2 and span > 0 and (len(hist) - 1) / span < 1.0:
        return _slow(start, pts, h)
    star = [(round(min(0.95, max(0.05, start[0] + ds)), 3), round(min(0.95, max(0.05, start[1] + dt)), 3))
            for ds, dt in ((h, 0), (-h, 0), (0, h), (0, -h))]
    for c in star:
        if c not in pts:
            return {"speed": c[0], "temp": c[1]}
    sc = {k: _score(v) for k, v in pts.items()}
    f0 = sc[start]
    s = _vertex(start[0], sc[star[1]], f0, sc[star[0]], h)
    t = _vertex(start[1], sc[star[3]], f0, sc[star[2]], h)
    s = min(start[0] + 0.3, max(start[0] - 0.3, s))
    t = min(start[1] + 0.3, max(start[1] - 0.3, t))
    v = (round(min(0.95, max(0.05, s)), 3), round(min(0.95, max(0.05, t)), 3))
    if v not in pts:
        return {"speed": v[0], "temp": v[1]}
    best = max(sc, key=lambda k: (sc[k], k))
    return {"speed": best[0], "temp": best[1]}
'''


class InventorReference(ScriptedReference):
    """Tool-only reference (labelled scripted in evidence): read the industry's needs and the machine catalogue,
    write an adaptive process controller (a pattern search on the adopting plant's own runs), declare it for every
    machine model of the market's operations at a modest licence, then watch the plants' verdicts. It reads only
    tool results."""
    burst = 6
    replan_s = 2 * DAY
    source = PATTERN_SEARCH
    licence = 300

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.stage = "start"
        self.repo = None
        self.models: list = []
        self.declared = None

    def plan(self, obs):
        if self.stage == "start":
            self.stage = "wait"
            return [act_("make.catalog", "catalog"), act_("tech.needs", "needs"),
                    act_("code.create_repo", "repo", name="process-control", visibility="private",
                         description="adaptive process control")]
        if self.stage == "commit" and self.repo:
            self.stage = "wait"
            return [act_("code.commit", "commit", repo=self.repo, files={"control.py": self.source},
                         message="pattern-search process controller")]
        if self.stage == "declare":
            self.stage = "watch"
            return [act_("tech.declare", "declared", repo=self.repo, target="control.py:recommend", models=self.models,
                         licence_cents_per_run=self.licence, title="Adaptive operating-point control",
                         doc="Finds better operating points from your own runs.")]
        if self.stage == "watch":
            return [act_("tech.mine", None)]
        return []

    def on_result(self, tag, result, action):
        if result is None:
            return
        if tag == "catalog":
            ops = self.objective.get("text", "")
            want = [op for op in ("form", "assemble", "extract", "refine", "blend") if op in ops] or \
                ["form", "assemble", "extract", "refine", "blend"]
            if "any" in ops:
                want = ["form", "assemble", "extract", "refine", "blend"]
            self.models = sorted(m["model"] for m in result["machine_models"] if m["op"] in want)
        elif tag == "repo":
            self.repo = result.get("repo") or result.get("id")
            self.stage = "commit"
        elif tag == "commit":
            self.stage = "declare"
        elif tag == "declared":
            self.declared = result["invention"]


class BrokenProgram(InventorReference):
    """Degenerate: a declared technology that violates the interface (returns an operating point out of range)."""
    source = 'def recommend(request):\n    return {"speed": 3.0, "temp": -1.0}\n'


class Freeze(InventorReference):
    """Degenerate: a technology that just repeats the plant's current operating point (no invention)."""
    source = ('def recommend(request):\n    h = request.get("history") or [{"speed": 0.5, "temp": 0.5}]\n'
              '    return {"speed": h[-1]["speed"], "temp": h[-1]["temp"]}\n')


# ============================================================================================ mechanics and actors
def _downstream(world, since):
    """Production runs of adopting plants after their adoption (their energy and inputs per unit made changed): the
    seqs of those runs' events."""
    st = world.state
    adopted = {t["plant"]: t.get("adopted_at", 0) for _, t in st.scan("tech_trials") if t["status"] == "adopted"}
    return base.events_where(world, since, ("make.run",), lambda e: e.payload.get("owner") in adopted and
                             e.t >= adopted[e.payload["owner"]])


MECHANICS = (
    Mechanic("Executable inventions", ("tech.declared", "tech.recommended"), "programs declared and run in the sandbox"),
    Mechanic("Technical constraints", ("tech.rejected",), "a program's output refused by the world's interface, range "
             "or sandbox checks (the market's openctl answers the previous interface version)"),
    Mechanic("Independent evaluation", ("tech.trial_done",), "plants' own measured comparisons at the end of a trial"),
    Mechanic("Adoption", ("tech.adopted", "tech.trial"), "trials and adoptions by plants"),
    Mechanic("Compatibility", ("tech.incompatible",), "a technology passed over because it does not support the "
             "plant's machine model"),
    Mechanic("Economic consequences", ("tech.licence_paid",), "licence fees adopters pay per production run"),
    Mechanic("Downstream resource changes", (), "adopters' production runs at the new operating points",
             count=_downstream),
)

ACTORS = (
    Actor("households", ("cols.batch:goods.session",), "population individuals buying goods"),
    Actor("plants", ("make.run", "tech.need"), "the industry's firms producing and stating needs"),
    Actor("haulier", ("freight.shipment",), "the freight operator moving the industry's goods"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], {k: (str(v) if isinstance(v, list) else v) for k, v in SHAPES[n].items()},
                              DOCS[n]) for n in SHAPES)


DOMAIN = Domain(
    name=NAME, title="Technology invention", spec_part="9.6", required_mechanics=(), configs=_configs(), build=build,
    goal=GOAL, evaluate=evaluate, reference=InventorReference, mechanics=MECHANICS,
    seats=(Seat(SEAT, 300_000, "Inventor", "an engineer who invents process technologies"),), actors=ACTORS,
    degenerate=(Probe("broken_program", "a technology that violates its interface", BrokenProgram),
                Probe("freeze", "a technology that changes nothing", Freeze)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=14.0, dt=12 * 3600,
    reference_seeds=(0, 1, 2), reference_configs=("forming_standard",), sources=(), doc=__doc__)

base.register(DOMAIN)
