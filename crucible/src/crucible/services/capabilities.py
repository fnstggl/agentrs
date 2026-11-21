"""Capabilities: a descriptive registry of agent-invented functionality, with installation, delegation, trials,
evidence and an observed technology graph (module `caps`, host `caps`).

A *capability* is any live, deployed service (agent-written code) together with a descriptor saying what it
claims to do and how an organisation can put it to work:

    {"host": "autoscale", "title": "...", "description": "...", "tags": [...free words...],
     "params":   {"instance": {"kind": "instance"}, "host": {"kind": "host"}},      # resources the installer binds
     "requires": [{"cap": "cloud.manage", "scope": "$instance"}, {"cap": "analytics.read", "scope": "$host"}],
     "triggers": [{"endpoint": "tick", "every_s": 3600, "args": {"instance": "$instance", "host": "$host"}}],
     "monthly_cents": 0, "trial_days": 10}

Nothing in the world enumerates what capabilities may exist; the registry only *indexes* what has been published.
Installing one (`caps.install`) is composed entirely from general primitives: capability grants from the
installer to the service principal (`id.grant`), triggers that call the service as the installer (`wf`), and
payments (endpoint prices per call, optional subscription). The service then acts on the installer's resources
within exactly the delegated scope; the kernel enforces it.

Evidence (verification provenance) is recorded in separate, typed records:

* `outcome` (Mode B, world-state outcome): for an installation run as a **switchback trial** (days alternate
  between capability on and off, by schedule fixed at install time), the registry computes - from the installer's
  authoritative operating journal (`ops`), not from anything the provider or installer says - the difference in
  daily operating result between on and off days, its standard error and the effect relative to revenue.
* `test` (Mode A, executable correctness): `caps.test` runs a tester-supplied suite (generator + checker programs
  in the tester's repository, executed in the sandbox) against the live endpoint and records scores, failures and
  CPU per call, with the suite's commit and the deployment's commit.
* `adoption` (Mode C, revealed preference): the registry counts installers and paying callers that are
  *independent* of the provider (`goals.metrics.related_parties`); related parties never count.
* `review` (Mode D, subjective): free-text and 1-5 ratings by principals who installed or paid; kept apart.

Categories (`caps.categories`) are computed descriptively from published descriptors that have independent
adopters (token-overlap clusters) - an index of what exists, never a list of what is allowed. The technology graph
(`caps.graph`) is observed: service-to-service calls (composition), installations, and `improves` lineage.
"""
from __future__ import annotations

import math
import re

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

DAY = 86400
RESOURCE_KINDS = ("instance", "host", "account", "listing", "repo", "facility", "machine", "value")
CAP_RE = r"[a-z]+(\.[a-z_*]+)?"
STOP = set("a an the and or of to for in on with by your you our is are it this that be as at from can will "
           "service api call calls per each use using any all its into than more less".split())


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9]+", text.lower()) if t not in STOP and len(t) > 2}


def _bind(x, bindings: dict):
    if isinstance(x, str) and x.startswith("$"):
        k = x[1:]
        if k not in bindings:
            raise InvalidAction(f"unbound parameter {x}")
        return bindings[k]
    if isinstance(x, dict):
        return {k: _bind(v, bindings) for k, v in x.items()}
    if isinstance(x, list):
        return [_bind(v, bindings) for v in x]
    return x


@register_module
class Capabilities(Module):
    name = "caps"
    depends = ("identity", "bank", "cloud", "wf")

    def install(self, world):
        st = world.state
        st.declare_table("capabilities", index=("provider", "host", "status"))
        st.declare_table("cap_installs", index=("capability", "installer", "status"))
        st.declare_table("cap_evidence", index=("capability", "by", "kind"))
        st.declare_table("cap_deps", index=("src", "dst"))
        st.declare_table("cap_payers", index=("host", "payer"))
        st.declare_table("cap_markets")
        world.hooks["caps"] = self
        prev = world.hooks.get("on_service_call")

        def on_call(ctx, caller, host, ep, status, price):
            if prev:
                prev(ctx, caller, host, ep, status, price)
            self.on_service_call(ctx, caller, host, ep, status, price)
        world.hooks["on_service_call"] = on_call
        world.hooks.setdefault("resource_owner", {})
        ro = world.hooks["resource_owner"]
        ro.setdefault("instance", lambda st, rid: (st.get("instances", rid) or {}).get("owner"))
        ro.setdefault("host", lambda st, rid: (st.get("hosts", rid) or {}).get("owner"))
        ro.setdefault("account", lambda st, rid: (st.get("accounts", rid) or {}).get("owner"))
        ro.setdefault("listing", lambda st, rid: (st.get("listings", rid) or {}).get("seller"))
        ro.setdefault("repo", lambda st, rid: (st.get("repos", rid) or {}).get("owner"))

    def genesis(self, ctx):
        ctx.schedule(ctx.now + DAY - 300, "caps.daily", {}, prio=79, period=DAY, key="caps.daily")

    # ------------------------------------------------------------------ helpers
    def provider_of_host(self, st, host):
        h = st.get("hosts", host)
        return h["owner"] if h else None

    def related(self, world, a: str, b: str) -> bool:
        from ..goals.metrics import related_parties
        return b in related_parties(world, a)

    def is_delegated(self, st, org: str, resource: str, day: int | None = None) -> bool:
        """True if an installed capability currently operates `resource` for `org` (its trigger is active today).
        Organisations' own default routines stand down on delegated resources."""
        for iid in st.find_ids("cap_installs", "installer", org):
            i = st.tables["cap_installs"][iid]
            if i["status"] in ("active", "trial") and i["on_today"] and resource in i["bindings"].values():
                return True
        return False

    # ------------------------------------------------------------------ publish / discover
    @endpoint("publish", S.obj({
        "host": S.s_str("your live service host", maxLength=31), "title": S.s_str(maxLength=120),
        "description": S.s_str("what it does, what problem it addresses, how to use it", maxLength=4000),
        "tags": S.s_arr(S.s_str(maxLength=30), maxItems=12),
        "params": S.s_any_obj("installer-bound parameters: {name: {kind: instance|host|account|listing|repo|"
                              "facility|machine|value, doc}}"),
        "requires": S.s_arr(S.s_any_obj(), "capabilities to delegate: [{cap, scope: '$param' | '*'}]", maxItems=8),
        "triggers": S.s_arr(S.s_any_obj(), "[{endpoint, every_s | on_event, args}] run as the installer",
                            maxItems=4),
        "monthly_cents": S.s_int(minimum=0, maximum=100_000_000), "trial_days": S.s_int(minimum=0, maximum=60),
        "improves": S.s_str("capability this one supersedes or improves on", maxLength=40),
        "open_source": S.s_bool("make the service repository readable by anyone")}, ["host", "title", "description"]),
        cost_s=60)
    def ep_publish(self, ctx, host, title, description, tags=None, params=None, requires=None, triggers=None,
                   monthly_cents=0, trial_days=0, improves=None, open_source=False):
        """Publish a capability descriptor for a live service you operate (a new version if already published)."""
        st = ctx.state
        cloud = ctx.world.modules["cloud"]
        dep = cloud.live_deployment(st, host)
        if dep is None:
            raise NotFound(f"{host} is not a live service")
        owner = dep["owner"]
        ident(ctx.world).authorize(ctx, owner, "cloud.deploy")
        params = params or {}
        for k, p in params.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", k) or not isinstance(p, dict) or p.get("kind") not in RESOURCE_KINDS:
                raise InvalidAction(f"param {k!r}: kind must be one of {RESOURCE_KINDS}")
        reqs = []
        for r in requires or []:
            cap, scope = r.get("cap"), r.get("scope", "*")
            if not isinstance(cap, str) or not re.fullmatch(CAP_RE, cap) or cap.startswith("admin"):
                raise InvalidAction(f"bad required capability {cap!r}")
            if scope != "*" and not (isinstance(scope, str) and scope.startswith("$") and scope[1:] in params
                                     and params[scope[1:]]["kind"] != "value"):
                raise InvalidAction(f"scope of {cap} must be '*' or a resource parameter")
            reqs.append({"cap": cap, "scope": scope, **({"limit": int(r["limit"])} if r.get("limit") else {})})
        trigs = []
        eps = dep["manifest"]["endpoints"]
        for t in triggers or []:
            if t.get("endpoint") not in eps:
                raise InvalidAction(f"trigger endpoint {t.get('endpoint')!r} not served by {host}")
            if t.get("every_s") is None and not t.get("on_event"):
                raise InvalidAction("trigger needs every_s or on_event")
            if t.get("every_s") is not None and int(t["every_s"]) < 600:
                raise InvalidAction("triggers run at most every 600 s")
            trigs.append({"endpoint": t["endpoint"], "every_s": t.get("every_s"), "on_event": t.get("on_event"),
                          "args": t.get("args", {})})
        if improves is not None and not st.has("capabilities", improves):
            raise NotFound("improves: no capability " + improves)
        prev = [c for c in st.find("capabilities", "host", host) if c["status"] == "active"]
        cid = st.new_id("cap")
        ctx.emit("caps.published", {
            "id": cid, "provider": owner, "host": host, "title": title, "description": description,
            "tags": sorted(set(t.lower() for t in tags or [])), "params": params, "requires": reqs,
            "triggers": trigs, "monthly_cents": int(monthly_cents), "trial_days": int(trial_days),
            "improves": improves, "supersedes": [c["id"] for c in prev], "commit": dep["commit"],
            "endpoints": {n: {"doc": e["doc"], "params": e["params"], "price_cents": e["price_cents"]}
                          for n, e in sorted(eps.items())}}, vis="public")
        if open_source:
            repo = st.get("repos", dep["repo"])
            if repo is not None and repo.get("visibility") != "public":
                try:
                    ctx.call("code.set_visibility", {"repo": dep["repo"], "visibility": "public", "license": "open"})
                except CrucibleError:
                    pass
        from .search import index
        index(ctx, cid, "capability", title, f"{description}\nTags: {' '.join(tags or [])}\nEndpoints: "
              f"{' '.join(sorted(eps))}\nRequires: {' '.join(r['cap'] for r in reqs)}", owner, ref=cid, hosts=[host])
        ctx.world.learn(ctx, ctx.actor, "host", host)
        return {"capability": cid, "supersedes": [c["id"] for c in prev]}

    def summary(self, world, cid: str) -> dict:
        st = world.state
        c = st.tables["capabilities"][cid]
        ins = [st.tables["cap_installs"][i] for i in st.find_ids("cap_installs", "capability", cid)]
        indep = sorted({i["installer"] for i in ins if i["status"] in ("active",) and i["independent"]})
        ev = [st.tables["cap_evidence"][e] for e in st.find_ids("cap_evidence", "capability", cid)]
        pub_out = [e for e in ev if e["kind"] == "outcome" and e["public"]]
        tests = [e for e in ev if e["kind"] == "test" and e["public"]]
        payers = {p["payer"] for p in st.find("cap_payers", "host", c["host"]) if p["independent"]}
        return {"capability": cid, "host": c["host"], "provider": c["provider"], "title": c["title"],
                "status": c["status"], "monthly_cents": c["monthly_cents"], "trial_days": c["trial_days"],
                "t": c["t"], "installs": len(ins), "active_installs": sum(1 for i in ins if i["status"] == "active"),
                "independent_adopters": len(indep), "independent_payers": len(payers),
                "public_outcomes": [{"evidence": e["id"], "by": e["by"], "independent": e["independent"],
                                     **{k: e["data"][k] for k in ("effect_per_day", "se", "rel_effect", "days_on",
                                                                  "days_off") if k in e["data"]}} for e in pub_out][-10:],
                "public_tests": [{"evidence": e["id"], "by": e["by"], **e["data"]} for e in tests][-5:],
                "rating": (sum(e["data"]["stars"] for e in ev if e["kind"] == "review") /
                           max(1, sum(1 for e in ev if e["kind"] == "review"))) if any(e["kind"] == "review" for e in ev) else None}

    @endpoint("view", S.obj({"capability": S.s_str(maxLength=40)}, ["capability"]), read_only=True)
    def ep_view(self, ctx, capability):
        """Full descriptor, adoption statistics and public evidence for a capability."""
        c = ctx.state.get("capabilities", capability)
        if c is None:
            raise NotFound("no capability " + capability)
        ctx.world.learn(ctx, ctx.actor, "host", c["host"])
        return {**{k: c[k] for k in ("description", "tags", "params", "requires", "triggers", "endpoints",
                                     "improves", "supersedes", "commit")}, **self.summary(ctx.world, capability)}

    @endpoint("search", S.obj({"q": S.s_str(maxLength=300), "limit": S.s_int(minimum=1, maximum=50)}, ["q"]),
              read_only=True)
    def ep_search(self, ctx, q, limit=10):
        """Full-text search over published capabilities (ranked by relevance; statistics attached)."""
        search = ctx.world.modules.get("search")
        st = ctx.state
        out = []
        if search is not None:
            sp, org, _ = search.search(ctx.world, q, "capability", limit * 2)
            seen = set()
            for r in sp + org:
                cid = r.get("ref")
                c = st.get("capabilities", cid) if cid else None
                if c is None or c["status"] != "active" or cid in seen:
                    continue
                seen.add(cid)
                out.append({**self.summary(ctx.world, cid), "description": c["description"][:300], "tags": c["tags"]})
                if len(out) >= limit:
                    break
        return out

    @endpoint("list", S.obj({"provider": S.s_str(maxLength=40), "limit": S.s_int(minimum=1, maximum=200)}),
              read_only=True)
    def ep_list(self, ctx, provider=None, limit=50):
        """Active capabilities (newest first), optionally by provider."""
        st = ctx.state
        ids = st.find_ids("capabilities", "status", "active")
        if provider:
            pv = ident(ctx.world).resolve_handle(st, provider)
            ids = [i for i in ids if st.tables["capabilities"][i]["provider"] == pv]
        return [self.summary(ctx.world, i) for i in list(reversed(ids))[:limit]]

    # ------------------------------------------------------------------ install / delegate
    @endpoint("install", S.obj({
        "capability": S.s_str(maxLength=40), "org": S.s_str("installing organisation", maxLength=40),
        "bindings": {"type": "object", "additionalProperties": {"type": ["string", "number", "integer", "boolean"]},
                     "description": "param -> your resource id (or a value)"},
        "trial": S.s_bool("run as a switchback trial (alternate on/off days) with a world-verified outcome"),
        "trial_days": S.s_int(minimum=4, maximum=60),
        "publish_outcome": S.s_bool("make the verified trial outcome public")}, ["capability", "org"]), cost_s=60)
    def ep_install(self, ctx, capability, org, bindings=None, trial=False, trial_days=None, publish_outcome=False):
        """Put a capability to work for an organisation you administer: creates the delegated grants and triggers
        the descriptor asks for, bound to your resources, and (unless in trial) the subscription."""
        st = ctx.state
        c = st.get("capabilities", capability)
        if c is None or c["status"] != "active":
            raise NotFound("no active capability " + capability)
        org = ident(ctx.world).resolve_handle(st, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        dup = [i for i in st.find("cap_installs", "installer", org)
               if i["capability"] == capability and i["status"] in ("active", "trial", "paused")]
        if dup:
            raise Conflict("already installed")
        bindings = dict(bindings or {})
        ro = ctx.world.hooks.get("resource_owner", {})
        for k, p in c["params"].items():
            if k not in bindings:
                raise InvalidAction(f"bind parameter {k} ({p['kind']})")
            if p["kind"] != "value":
                fn = ro.get(p["kind"])
                if fn is None or fn(st, str(bindings[k])) != org:
                    raise Forbidden(f"{k}: {bindings[k]} is not a {p['kind']} owned by {org}")
        svc = f"svc:{c['host']}"
        grants = []
        for r in c["requires"]:
            scope = "*" if r["scope"] == "*" else str(_bind(r["scope"], bindings))
            g = ctx.call("id.grant", {"grantee": svc, "cap": r["cap"], "owner": org, "scope": scope,
                                      **({"limit": r["limit"]} if r.get("limit") else {})})
            grants.append(g["grant"])
        wf = ctx.world.modules["wf"]
        trigs = []
        for t in c["triggers"]:
            trigs.append(wf.create_trigger(ctx, org, ctx.actor, f"{c['host']}.{t['endpoint']}", _bind(t["args"], bindings),
                                           t["every_s"], t["on_event"], 0, None, f"cap {capability}"))
        schedule = None
        if trial:
            n = int(trial_days or max(6, c["trial_days"] or 8))
            n += n % 2
            rng = ctx.rng("trial", org, capability)
            schedule = []
            for _ in range(n // 2):        # randomised pairs: one on day and one off day per pair
                schedule.extend([1, 0] if rng.random() < 0.5 else [0, 1])
        iid = st.new_id("inst_cap")
        indep = not self.related(ctx.world, org, c["provider"])
        ctx.emit("caps.installed", {"id": iid, "capability": capability, "installer": org, "by": ctx.actor,
                                    "bindings": bindings, "grants": grants, "triggers": trigs, "independent": indep,
                                    "schedule": schedule, "start_day": ctx.now // DAY + 1,
                                    "publish_outcome": bool(publish_outcome)}, vis=(org, c["provider"]))
        for tid in trigs:   # trials begin at the next day boundary; until then nothing runs
            if schedule is not None:
                wf.set_status(ctx, tid, "paused", "trial starts tomorrow")
        return {"install": iid, "grants": grants, "triggers": trigs, "schedule": schedule}

    def _own_install(self, ctx, iid):
        i = ctx.state.get("cap_installs", iid)
        if i is None:
            raise NotFound("no installation " + iid)
        ident(ctx.world).authorize(ctx, i["installer"], "org.admin")
        return i

    def set_on(self, ctx, iid: str, on: bool) -> None:
        i = ctx.state.tables["cap_installs"][iid]
        wf = ctx.world.modules["wf"]
        for tid in i["triggers"]:
            wf.set_status(ctx, tid, "active" if on else "paused", "trial schedule" if i["status"] == "trial" else "")
        if i["on_today"] != on:
            ctx.emit("caps.on", {"id": iid, "on": on}, vis=(i["installer"],))

    def uninstall(self, ctx, iid: str, reason: str) -> None:
        i = ctx.state.tables["cap_installs"][iid]
        if i["status"] == "removed":
            return
        wf = ctx.world.modules["wf"]
        for tid in i["triggers"]:
            wf.delete_trigger(ctx, tid)
        for gid in i["grants"]:
            g = ctx.state.get("grants", gid)
            if g is not None and g["status"] == "active":
                ctx.emit("id.revoke", {"id": gid}, vis=(g["grantee"], g["owner"]))
        ctx.emit("caps.install_status", {"id": iid, "status": "removed", "reason": reason[:200]},
                 vis=(i["installer"], ctx.state.tables["capabilities"][i["capability"]]["provider"]))

    @endpoint("uninstall", S.obj({"install": S.s_str(maxLength=40), "reason": S.s_str(maxLength=200)}, ["install"]))
    def ep_uninstall(self, ctx, install, reason=""):
        """Remove an installation: deletes its triggers and revokes every grant it created."""
        self._own_install(ctx, install)
        self.uninstall(ctx, install, reason or "uninstalled")
        return {"ok": True}

    @endpoint("adopt", S.obj({"install": S.s_str(maxLength=40)}, ["install"]))
    def ep_adopt(self, ctx, install):
        """End a trial early (or confirm after it) and keep the capability running every day."""
        i = self._own_install(ctx, install)
        if i["status"] not in ("trial", "evaluated", "paused"):
            raise Conflict(f"installation is {i['status']}")
        ctx.emit("caps.install_status", {"id": install, "status": "active", "reason": "adopted"},
                 vis=(i["installer"],))
        self.set_on(ctx, install, True)
        return {"ok": True}

    @endpoint("pause", S.obj({"install": S.s_str(maxLength=40)}, ["install"]))
    def ep_pause(self, ctx, install):
        """Pause an installation (its triggers stop; grants stay)."""
        i = self._own_install(ctx, install)
        if i["status"] in ("active", "trial", "evaluated"):
            ctx.emit("caps.install_status", {"id": install, "status": "paused", "reason": "by installer"},
                     vis=(i["installer"],))
            self.set_on(ctx, install, False)
        return {"ok": True}

    @endpoint("installs", S.obj({"org": S.s_str(maxLength=40)}), read_only=True)
    def ep_installs(self, ctx, org=None):
        """Installations of an organisation you administer (or capabilities you provide: their installers)."""
        st = ctx.state
        if org:
            o = ident(ctx.world).resolve_handle(st, org)
            ident(ctx.world).authorize(ctx, o, "org.admin")
            ids = st.find_ids("cap_installs", "installer", o)
        else:
            mine = ctx.world.principals_of(ctx.actor)
            ids = [i for i, r in st.scan("cap_installs")
                   if r["installer"] in mine or st.tables["capabilities"][r["capability"]]["provider"] in mine]
        return [{"install": i, **{k: st.tables["cap_installs"][i][k] for k in (
            "capability", "installer", "status", "bindings", "schedule", "start_day", "on_today", "evidence")}}
                for i in ids]

    # ------------------------------------------------------------------ daily: trial schedules, billing, outcomes
    def outcome(self, world, i: dict) -> dict:
        ops = world.modules["ops"]
        st = world.state
        on, off, rev = [], [], []
        for k, flag in enumerate(i["schedule"] or []):
            d = i["start_day"] + k
            r = ops.day(st, i["installer"], d)
            from ..economy.operations import NON_OPERATING, operating_result
            v = operating_result(r)
            (on if flag else off).append(v)
            if r:
                rev.append(sum(x for kk, x in r["flows"].items() if x > 0 and kk not in NON_OPERATING))
        if not on or not off:
            return {}
        mu_on, mu_off = float(np.mean(on)), float(np.mean(off))
        var = (float(np.var(on, ddof=1)) if len(on) > 1 else 0.0) / len(on) + \
              (float(np.var(off, ddof=1)) if len(off) > 1 else 0.0) / len(off)
        se = math.sqrt(var)
        mean_rev = float(np.mean(rev)) if rev else 0.0
        return {"effect_per_day": round(mu_on - mu_off, 2), "se": round(se, 2), "days_on": len(on),
                "days_off": len(off), "mean_on": round(mu_on, 2), "mean_off": round(mu_off, 2),
                "rel_effect": round((mu_on - mu_off) / mean_rev, 4) if mean_rev > 0 else None,
                "mean_revenue_per_day": round(mean_rev, 2), "metric": "operating_result_cents_per_day",
                "design": "randomised switchback (paired days)"}

    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        today = ctx.now // DAY + 1        # this process runs just before midnight: schedule tomorrow
        for iid in list(st.ids("cap_installs")):
            i = st.tables["cap_installs"][iid]
            c = st.tables["capabilities"][i["capability"]]
            if i["status"] == "removed":
                continue
            prov_dep = ctx.world.modules["cloud"].live_deployment(st, c["host"])
            if i["status"] == "trial":
                k = today - i["start_day"]
                if k >= len(i["schedule"]):
                    data = self.outcome(ctx.world, i)
                    eid = st.new_id("evid")
                    ctx.emit("caps.evidence", {"id": eid, "capability": i["capability"], "by": i["installer"],
                                               "kind": "outcome", "data": {**data, "install": iid},
                                               "independent": i["independent"], "public": i["publish_outcome"],
                                               "verified": True},
                             vis="public" if i["publish_outcome"] else (i["installer"], c["provider"]))
                    ctx.emit("caps.install_status", {"id": iid, "status": "evaluated", "reason": "trial complete",
                                                     "evidence": eid}, vis=(i["installer"], c["provider"]))
                    self.set_on(ctx, iid, False)
                    continue
                self.set_on(ctx, iid, bool(i["schedule"][k]))
                if c["monthly_cents"] and k >= c["trial_days"] and i["schedule"][k]:
                    self._bill(ctx, i, iid, c, bank)
            elif i["status"] == "active":
                if c["monthly_cents"]:
                    self._bill(ctx, i, iid, c, bank)
                if prov_dep is None:
                    ops = ctx.world.hooks.get("ops")
                    if ops is not None:
                        ops.record_loss(ctx, i["installer"], "dependency_failure", 0, 1, resource=c["host"])
        self._census(ctx)

    def _bill(self, ctx, i, iid, c, bank):
        st = ctx.state
        src = bank.primary_account(st, i["installer"])
        dst = bank.primary_account(st, c["provider"])
        amount = max(1, c["monthly_cents"] // 30)
        try:
            if src is None or dst is None:
                raise InvalidAction("no account")
            bank.move(ctx, src, dst, amount, "subscription", f"capability {i['capability']}", ref=f"svc:{c['host']}")
            if i["independent"]:
                self._payer(ctx, c["host"], i["installer"], amount, True)
        except CrucibleError:
            ctx.emit("caps.install_status", {"id": iid, "status": "paused", "reason": "subscription unpaid"},
                     vis=(i["installer"], c["provider"]))
            self.set_on(ctx, iid, False)

    # ------------------------------------------------------------------ evidence: tests, reports, reviews
    @endpoint("evidence", S.obj({"capability": S.s_str(maxLength=40)}, ["capability"]), read_only=True)
    def ep_evidence(self, ctx, capability):
        """Evidence records about a capability that you may see (public ones, and your own)."""
        st = ctx.state
        mine = ctx.world.principals_of(ctx.actor)
        c = st.get("capabilities", capability)
        if c is None:
            raise NotFound("no capability")
        out = []
        for eid in st.find_ids("cap_evidence", "capability", capability):
            e = st.tables["cap_evidence"][eid]
            if e["public"] or e["by"] in mine or c["provider"] in mine:
                out.append({"evidence": eid, **{k: e[k] for k in ("kind", "by", "t", "data", "independent",
                                                                  "verified", "public")}})
        return out

    @endpoint("get_evidence", S.obj({"evidence": S.s_str(maxLength=40)}, ["evidence"]), read_only=True)
    def ep_get_evidence(self, ctx, evidence):
        """One evidence record (if public or yours) - for checking a claim someone made."""
        e = ctx.state.get("cap_evidence", evidence)
        if e is None:
            raise NotFound("no evidence " + evidence)
        c = ctx.state.tables["capabilities"][e["capability"]]
        mine = ctx.world.principals_of(ctx.actor)
        if not (e["public"] or e["by"] in mine or c["provider"] in mine):
            raise Forbidden("private evidence")
        return {"evidence": evidence, "capability": e["capability"], **{k: e[k] for k in (
            "kind", "by", "t", "data", "independent", "verified", "public")}}

    @endpoint("publish_evidence", S.obj({"evidence": S.s_str(maxLength=40)}, ["evidence"]))
    def ep_publish_evidence(self, ctx, evidence):
        """Make an evidence record you own public."""
        e = ctx.state.get("cap_evidence", evidence)
        if e is None:
            raise NotFound("no evidence")
        ident(ctx.world).authorize(ctx, e["by"], "org.admin")
        if not e["public"]:
            ctx.emit("caps.evidence_public", {"id": evidence}, vis="public")
        return {"ok": True}

    @endpoint("review", S.obj({"capability": S.s_str(maxLength=40), "stars": S.s_int(minimum=1, maximum=5),
                               "text": S.s_str(maxLength=2000), "as_org": S.s_str(maxLength=40)},
                              ["capability", "stars"]))
    def ep_review(self, ctx, capability, stars, text="", as_org=None):
        """Subjective review (Mode D) - only by a principal that installed or paid for the capability."""
        st = ctx.state
        c = st.get("capabilities", capability)
        if c is None:
            raise NotFound("no capability")
        who = ident(ctx.world).resolve_handle(st, as_org) if as_org else ctx.actor
        if who != ctx.actor:
            ident(ctx.world).authorize(ctx, who, "org.admin")
        used = any(i["installer"] == who for i in st.find("cap_installs", "capability", capability)) or \
            st.has("cap_payers", f"{c['host']}|{who}")
        if not used:
            raise Forbidden("only users of a capability can review it")
        eid = st.new_id("evid")
        ctx.emit("caps.evidence", {"id": eid, "capability": capability, "by": who, "kind": "review",
                                   "data": {"stars": int(stars), "text": text}, "public": True, "verified": False,
                                   "independent": not self.related(ctx.world, who, c["provider"])}, vis="public")
        return {"evidence": eid}

    @endpoint("test", S.obj({"capability": S.s_str(maxLength=40), "endpoint": S.s_str(maxLength=40),
                             "suite_repo": S.s_str("your repo with suite.py: generate(args, caller) -> input; "
                                                   "check(args, caller) -> score in [0,1] for args "
                                                   "{input, output}", maxLength=81),
                             "cases": S.s_int(minimum=1, maximum=20), "public": S.s_bool()},
                            ["capability", "endpoint", "suite_repo"]), cost_s=300)
    def ep_test(self, ctx, capability, endpoint, suite_repo, cases=8, public=True):
        """Executable verification (Mode A): run your own test suite against a capability's live endpoint (you pay
        the calls) and record the scores, failure rate and CPU per call as evidence with full provenance."""
        st = ctx.state
        c = st.get("capabilities", capability)
        if c is None:
            raise NotFound("no capability")
        if endpoint not in c["endpoints"]:
            raise NotFound(f"{c['host']} has no endpoint {endpoint}")
        code = ctx.world.modules["code"]
        repo = code.repo(st, suite_repo)
        ident(ctx.world).authorize(ctx, repo["owner"], "code.read")
        files = code.files(st, suite_repo, None)
        if "suite.py" not in files:
            raise InvalidAction("suite repo needs suite.py")
        from ..sandbox.bridge import run_in_ctx
        pool = code.sandbox_pool(ctx.world)
        key = f"{ctx.world.world_id}|suite|{suite_repo}|{repo['head']}"

        def worker():
            return pool.get(key, lambda: code.files(st, suite_repo, None))
        scores, fails, cpus = [], 0, []
        for k in range(int(cases)):
            g = run_in_ctx(ctx, "caps.suite", worker, {"kind": "call", "target": "suite:generate",
                                                        "args": {"seed": k, "capability": capability},
                                                        "caller": ctx.actor}, ctx.actor, timeout=5.0,
                           world_access=False)
            if not g["ok"] or not isinstance(g["result"], dict):
                raise InvalidAction(f"suite.generate failed: {(g.get('error') or {}).get('message', 'not a dict')}")
            try:
                res = ctx.call(f"{c['host']}.{endpoint}", g["result"])
            except CrucibleError as e:
                res = {"status": 400, "error": e.message}
            if not isinstance(res, dict) or res.get("status") != 200:
                fails += 1
                scores.append(0.0)
                continue
            cpus.append(float(res.get("cpu_ms", 0.0)))
            ch = run_in_ctx(ctx, "caps.suite", worker, {"kind": "call", "target": "suite:check",
                                                         "args": {"input": g["result"], "output": res.get("body")},
                                                         "caller": ctx.actor}, ctx.actor, timeout=5.0,
                            world_access=False)
            try:
                scores.append(float(min(1.0, max(0.0, float(ch["result"])))) if ch["ok"] else 0.0)
            except (TypeError, ValueError):
                scores.append(0.0)
        data = {"cases": int(cases), "mean_score": round(float(np.mean(scores)), 4), "failures": fails,
                "mean_cpu_ms": round(float(np.mean(cpus)), 4) if cpus else None, "endpoint": endpoint,
                "suite": f"{suite_repo}@{repo['head']}", "capability_commit": c["commit"]}
        eid = st.new_id("evid")
        indep = not self.related(ctx.world, repo["owner"], c["provider"])
        ctx.emit("caps.evidence", {"id": eid, "capability": capability, "by": repo["owner"], "kind": "test",
                                   "data": data, "public": bool(public), "verified": True, "independent": indep},
                 vis="public" if public else (repo["owner"], c["provider"]))
        return {"evidence": eid, **data}

    # ------------------------------------------------------------------ observed composition & revealed preference
    def _payer(self, ctx, host, payer, cents, indep):
        ctx.emit("caps.payer", {"host": host, "payer": payer, "cents": int(cents), "independent": bool(indep)},
                 vis="internal")

    def on_service_call(self, ctx, caller, host, ep, status, price):
        st = ctx.state
        cp = st.get("principals", caller)
        if cp is not None and cp["kind"] == "service":
            src = cp["info"].get("host")
            if src and src != host:
                ctx.emit("caps.dep", {"src": src, "dst": host, "ok": status == 200}, vis="internal")
        if price > 0 and status == 200 and caller not in ("households",):
            payer = cp["info"].get("operator") if cp is not None and cp["kind"] == "service" else caller
            prov = self.provider_of_host(st, host)
            if prov and payer and payer != prov:
                self._payer(ctx, host, payer, price, not self.related(ctx.world, payer, prov))

    @endpoint("categories", S.obj({"min_adopters": S.s_int(minimum=1, maximum=100)}), read_only=True)
    def ep_categories(self, ctx, min_adopters=1):
        """Emergent categories: clusters of published capabilities (by descriptor vocabulary) that independent
        organisations actually adopted or paid for. Descriptive only."""
        return self.categories(ctx.world, min_adopters)

    def categories(self, world, min_adopters=1) -> list[dict]:
        st = world.state
        caps = []
        for cid, c in st.scan("capabilities"):
            s = self.summary(world, cid)
            n = max(s["independent_adopters"], s["independent_payers"])
            if n >= min_adopters:
                caps.append((cid, c, s, tokens(" ".join([c["title"], c["description"], " ".join(c["tags"]),
                                                          " ".join(c["endpoints"]), " ".join(r["cap"] for r in c["requires"])]))))
        parent = list(range(len(caps)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a in range(len(caps)):
            for b in range(a + 1, len(caps)):
                ta, tb = caps[a][3], caps[b][3]
                if ta and tb and len(ta & tb) / len(ta | tb) >= 0.2 or caps[b][1]["improves"] == caps[a][0] \
                        or caps[a][1]["improves"] == caps[b][0]:
                    parent[find(a)] = find(b)
        groups: dict[int, list] = {}
        for k in range(len(caps)):
            groups.setdefault(find(k), []).append(caps[k])
        out = []
        for g in groups.values():
            freq: dict[str, int] = {}
            for _, _, _, t in g:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            label = " ".join(sorted(freq, key=lambda x: (-freq[x], x))[:3])
            buyers = set()
            for cid, c, s, _ in g:
                buyers |= {i["installer"] for i in st.find("cap_installs", "capability", cid)
                           if i["independent"] and i["status"] in ("active", "evaluated", "trial")}
                buyers |= {p["payer"] for p in st.find("cap_payers", "host", c["host"]) if p["independent"]}
            out.append({"label": label, "capabilities": sorted(cid for cid, *_ in g),
                        "sellers": sorted({c["provider"] for _, c, _, _ in g}), "independent_buyers": sorted(buyers),
                        "first_published": min(c["t"] for _, c, _, _ in g)})
        return sorted(out, key=lambda x: x["first_published"])

    def _census(self, ctx):
        """Record when each emergent market (category with >= 2 independent buyers) is first observed."""
        st = ctx.state
        for cat in self.categories(ctx.world, 1):
            if len(cat["independent_buyers"]) < 2:
                continue
            key = cat["capabilities"][0]
            if not st.has("cap_markets", key):
                ctx.emit("caps.market", {"id": key, "label": cat["label"], "sellers": cat["sellers"],
                                         "buyers": cat["independent_buyers"]}, vis="public")

    @endpoint("graph", S.obj(), read_only=True)
    def ep_graph(self, ctx):
        """The observed technology graph: capabilities, who uses them, which services call which (composition),
        and improvement lineage."""
        return self.graph(ctx.world)

    def graph(self, world) -> dict:
        st = world.state
        nodes = [{"id": cid, "host": c["host"], "provider": c["provider"], "t": c["t"], "status": c["status"]}
                 for cid, c in st.scan("capabilities")]
        edges = [{"src": d["src"], "dst": d["dst"], "kind": "calls", "n": d["n"]} for _, d in st.scan("cap_deps")]
        edges += [{"src": c["improves"], "dst": cid, "kind": "improves"} for cid, c in st.scan("capabilities")
                  if c["improves"]]
        edges += [{"src": i["installer"], "dst": i["capability"], "kind": "uses"} for _, i in st.scan("cap_installs")
                  if i["status"] in ("active", "trial", "evaluated")]
        return {"nodes": nodes, "edges": edges, "markets": [dict(m, id=k) for k, m in st.scan("cap_markets")]}

    # ------------------------------------------------------------------ appliers
    @applier("caps.published")
    def ap_published(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        for old in p["supersedes"]:
            st.patch("capabilities", old, status="superseded")
        p.update(status="active", t=ev.t)
        st.insert("capabilities", p.pop("id"), p)

    @applier("caps.installed")
    def ap_installed(self, ev):
        p = dict(ev.payload)
        p.update(status="trial" if p["schedule"] is not None else "active", t=ev.t,
                 on_today=p["schedule"] is None, evidence=None)
        self.world.state.insert("cap_installs", p.pop("id"), p)

    @applier("caps.on")
    def ap_on(self, ev):
        self.world.state.patch("cap_installs", ev.payload["id"], on_today=ev.payload["on"])

    @applier("caps.install_status")
    def ap_install_status(self, ev):
        p = ev.payload
        ch = {"status": p["status"]}
        if p.get("evidence"):
            ch["evidence"] = p["evidence"]
        if p["status"] in ("removed", "paused", "evaluated"):
            ch["on_today"] = False
        self.world.state.patch("cap_installs", p["id"], **ch)

    @applier("caps.evidence")
    def ap_evidence(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("cap_evidence", p["id"], p)

    @applier("caps.evidence_public")
    def ap_evidence_public(self, ev):
        self.world.state.patch("cap_evidence", ev.payload["id"], public=True)

    @applier("caps.dep")
    def ap_dep(self, ev):
        st = self.world.state
        p = ev.payload
        k = f"{p['src']}|{p['dst']}"
        r = st.tables["cap_deps"].get(k)
        if r is None:
            st.insert("cap_deps", k, {"src": p["src"], "dst": p["dst"], "n": 1, "fails": 0 if p["ok"] else 1,
                                      "first": ev.t, "last": ev.t})
        else:
            st.patch("cap_deps", k, n=r["n"] + 1, fails=r["fails"] + (0 if p["ok"] else 1), last=ev.t)

    @applier("caps.payer")
    def ap_payer(self, ev):
        st = self.world.state
        p = ev.payload
        k = f"{p['host']}|{p['payer']}"
        r = st.tables["cap_payers"].get(k)
        if r is None:
            st.insert("cap_payers", k, {"host": p["host"], "payer": p["payer"], "cents": p["cents"], "n": 1,
                                        "independent": p["independent"], "first": ev.t})
        else:
            st.patch("cap_payers", k, cents=r["cents"] + p["cents"], n=r["n"] + 1)

    @applier("caps.market")
    def ap_market(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("cap_markets", p.pop("id"), p)

    def check_invariants(self, world):
        errs = []
        st = world.state
        for iid, i in st.scan("cap_installs"):
            if i["status"] == "removed":
                for gid in i["grants"]:
                    g = st.get("grants", gid)
                    if g is not None and g["status"] == "active":
                        errs.append(f"{iid}: removed installation left grant {gid} active")
        return errs
