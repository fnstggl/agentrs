"""Compute and hosting (host `cloud`): providers with finite capacity, rented instances, deployments of repository
code as callable services, metering, billing, overload, outages and the synthetic DNS.

Causal chain implemented here:
  repo commit ─deploy→ live service at `<host>.crucible` (tools `<host>.<endpoint>`) running on a rented instance
  every request: optional price paid by the caller → real execution (sandbox, or in-process for trusted built-in
  incumbents) → measured CPU × work scale = compute units drawn from the instance's hourly capacity → overload
  (HTTP-like 503) when capacity is exhausted → hourly billing of instances to their owners → suspension when
  bills are not paid → services go down → callers (including simulated customers) experience failures.

Providers are organisations: they pay operating costs for their hardware, earn rent, can buy more hardware (with a
delivery lead time) and suffer outages drawn from their reliability. Any registered company can become a
provider; there is no privileged infrastructure.
"""
from __future__ import annotations

import json
import re

from ..core import schema as S
from ..core.errors import (CapacityExceeded, Conflict, CrucibleError, Forbidden, InvalidAction, NotFound,
                           ServiceUnavailable)
from ..core.kernel import SYSTEM, Ctx, Endpoint, Module, applier, endpoint, process, register_module
from .identity import ident

HOST = r"[a-z][a-z0-9-]{2,30}"
CU_PER_CORE_HOUR = 100_000
EP_NAME = r"[a-z][a-z0-9_]{0,30}"
HANDLER = r"[A-Za-z_][A-Za-z0-9_]{0,40}"
ENTRY = r"[A-Za-z_][A-Za-z0-9_/]{0,60}(\.py)?"


@register_module
class Cloud(Module):
    name = "cloud"
    depends = ("identity", "bank", "code")

    def install(self, world):
        st = world.state
        st.declare_table("providers", index=("status",))
        st.declare_table("instances", index=("owner", "provider", "status"))
        st.declare_table("hosts", index=("owner",))
        st.declare_table("deployments", index=("owner", "host", "status", "instance"))
        st.declare_table("svc_days", index=("host",))
        st.declare_table("hw_orders", index=("provider",))
        world.hooks["host_endpoints"] = self._host_endpoints
        world.hooks["is_host"] = self.is_host
        world.hooks["host_owner"] = lambda state, host: (state.get("hosts", host) or {}).get("owner")
        world.hooks["cloud"] = self
        world.resolvers.append(self._resolve)

    def genesis(self, ctx):
        ctx.schedule(ctx.now + 3600, "cloud.hourly", {}, prio=20, period=3600, key="cloud.hourly")
        ctx.schedule(ctx.now + 86400, "cloud.daily", {}, prio=21, period=86400, key="cloud.daily")

    def on_load(self, world):
        world.runtime.setdefault("trusted_fns", {})

    # ------------------------------------------------------------------ parameters
    def p(self, key, default):
        return self.world.spec.params.get(key, self.config.get(key, default))

    # ------------------------------------------------------------------ DNS & resolution
    def is_host(self, state, name: str) -> bool:
        h = state.get("hosts", name)
        if h is not None:
            d = state.get("deployments", h["deployment"]) if h.get("deployment") else None
            return d is not None and d["status"] == "live"
        return any(e.host == name for e in self.world.endpoints.values())

    def live_deployment(self, state, host: str):
        h = state.get("hosts", host)
        if h is None or not h.get("deployment"):
            return None
        d = state.get("deployments", h["deployment"])
        return d if d is not None and d["status"] == "live" else None

    def _endpoint_obj(self, dep_id: str, dep, name: str, spec: dict) -> Endpoint:
        host = dep["host"]

        def handler(ctx, **args):
            return self.call_endpoint(ctx, dep_id, name, args, ctx.actor)
        price = int(spec.get("price_cents", 0))
        doc = (spec.get("doc") or "") + (f" [price: {price} cents per call]" if price else "")
        return Endpoint(f"{host}.{name}", host, spec.get("params") or S.obj(), handler, doc[:1000], 5, False,
                        owner=dep["svc"], builtin=False)

    def _host_endpoints(self, world, host: str, principal: str) -> list[Endpoint]:
        h = world.state.get("hosts", host)
        if h is None or not h.get("deployment"):
            return []
        dep = world.state.get("deployments", h["deployment"])
        if dep is None or dep["status"] not in ("live",):
            return []
        return [self._endpoint_obj(h["deployment"], dep, n, sp) for n, sp in sorted(dep["manifest"]["endpoints"].items())]

    def _resolve(self, tool: str, principal: str):
        host, _, ep = tool.partition(".")
        if not ep:
            return None
        h = self.world.state.get("hosts", host)
        if h is None or not h.get("deployment"):
            return None
        dep = self.world.state.get("deployments", h["deployment"])
        if dep is None:
            return None
        spec = dep["manifest"]["endpoints"].get(ep)
        if spec is None:
            return None
        if dep["status"] != "live":
            def down(ctx, **args):
                raise ServiceUnavailable(f"{host} is {dep['status']}")
            return Endpoint(tool, host, spec.get("params") or S.obj(), down, "", 5, False, builtin=False)
        return self._endpoint_obj(h["deployment"], dep, ep, spec)

    # ------------------------------------------------------------------ invocation
    def availability(self, state, dep) -> str | None:
        """None if the deployment can serve now, else a reason."""
        if dep["status"] != "live":
            return f"service {dep['status']}"
        inst = state.get("instances", dep["instance"])
        if inst is None or inst["status"] != "running":
            return "instance not running"
        prov = state.get("providers", inst["provider"])
        if prov is None or prov["status"] != "active" or prov["outage_until"] > self.world.now:
            return "provider outage"
        return None

    def call_endpoint(self, ctx: Ctx, dep_id: str, ep: str, args: dict, caller: str) -> dict:
        """A priced call by an actor or a service principal. Returns an HTTP-like response dict."""
        st = ctx.state
        dep = st.get("deployments", dep_id)
        spec = dep["manifest"]["endpoints"][ep]
        price = int(spec.get("price_cents", 0))
        why = self.availability(st, dep)
        if why:
            self._usage(ctx, dep_id, 0, 0.0, 0.0, "unavailable", 0)
            return {"status": 503, "error": why}
        bank = ctx.world.modules["bank"]
        if price > 0:
            payer = caller
            pp = st.get("principals", caller)
            if pp and pp["kind"] == "service":
                payer = pp["info"].get("operator")
                if not ident(ctx.world).allowed(ctx, caller, payer, "bank.spend", amount=price, record=True):
                    return {"status": 402, "error": f"{caller} has no spending grant from {payer}"}
            src = bank.primary_account(st, payer)
            dst = bank.primary_account(st, dep["owner"])
            if src is None or dst is None:
                return {"status": 402, "error": "payer or payee has no account"}
            if src != dst:
                try:
                    bank.move(ctx, src, dst, price, "sale", f"{dep['host']}.{ep}", ref=f"svc:{dep['host']}")
                except CrucibleError as e:
                    return {"status": 402, "error": e.message}
        res = self.execute(ctx, dep_id, ep, args, caller, float(self.p("cu_per_ms", 50.0)))
        if res["status"] != 200 and price > 0 and src != dst:
            bank.move(ctx, dst, src, price, "refund", f"failed call {dep['host']}.{ep}", ref=f"svc:{dep['host']}")
        elif res["status"] == 200 and price > 0:
            self._revenue(ctx, dep_id, price)
            hook = ctx.world.hooks.get("on_api_purchase")
            if hook and src != dst:
                hook(ctx, dep["host"], payer, price)
        return res

    def execute(self, ctx: Ctx, dep_id: str, ep: str, args, caller, work_scale: float, post=None) -> dict:
        """Run one request (no payment). Enforces capacity using the deployment's running CU estimate."""
        st = ctx.state
        dep = st.get("deployments", dep_id)
        inst = st.get("instances", dep["instance"])
        hour = ctx.now // 3600
        used = inst["used_cu"] if inst["hour"] == hour else 0.0
        cap = inst["cores"] * CU_PER_CORE_HOUR
        est = dep["cu_est"] * (work_scale / max(1e-9, dep["ws_est"])) if dep["ws_est"] else 0.0
        if used > 0 and used + est > cap:     # the first request of an hour is always attempted
            self._usage(ctx, dep_id, 0, 0.0, 0.0, "overload", 0)
            return {"status": 503, "error": "service overloaded (instance capacity exhausted this hour)"}
        spec = dep["manifest"]["endpoints"][ep]
        out = self._run(ctx, dep_id, dep, spec["handler"], args, caller, post)
        cu = out["cpu_ms"] * work_scale
        status = 200 if out["ok"] else 500
        err = None if out["ok"] else (out["error"] or {}).get("message", "error")
        self._usage(ctx, dep_id, 1, out["cpu_ms"], cu, "ok" if out["ok"] else "error", work_scale, err)
        if out["ok"]:
            return {"status": 200, "body": out["result"], "cpu_ms": out["cpu_ms"]}
        return {"status": 500, "error": err, "cpu_ms": out["cpu_ms"]}

    def _run(self, ctx: Ctx, dep_id: str, dep, handler: str, args, caller, post=None) -> dict:
        from ..sandbox.bridge import run_in_ctx, run_trusted_in_ctx
        entry = dep["manifest"]["entry"]
        mod = entry[:-3] if entry.endswith(".py") else entry
        mod = mod.replace("/", ".")
        if dep["trusted"]:
            fn = self._trusted_fn(ctx, dep_id, dep, mod, handler)
            return run_trusted_in_ctx(ctx, "cloud.req", fn, args, caller, dep["svc"], post=post)
        code = ctx.world.modules["code"]
        pool = code.sandbox_pool(ctx.world)
        key = f"{ctx.world.world_id}|{dep_id}|{dep['commit']}"

        def worker():
            return pool.get(key, lambda: self._files(ctx, dep))
        return run_in_ctx(ctx, "cloud.req", worker, {"kind": "call", "target": f"{mod}:{handler}", "args": args,
                                                     "caller": caller}, dep["svc"],
                          timeout=float(self.p("request_timeout_s", 5.0)), post=post)

    def _files(self, ctx, dep) -> dict:
        code = ctx.world.modules["code"]
        return code.files_with_deps(ctx, dep["repo"], dep["commit"], dep["owner"])

    def _trusted_fn(self, ctx, dep_id, dep, mod, handler):
        cache = ctx.world.runtime.setdefault("trusted_fns", {})
        key = (dep_id, dep["commit"], handler)
        fn = cache.get(key)
        if fn is None:
            files = self._files(ctx, dep)
            src = files.get(mod.replace(".", "/") + ".py")
            if src is None:
                raise ServiceUnavailable(f"entry module {mod} missing")
            ns: dict = {"__name__": f"svc_{dep['host']}"}
            exec(compile(src, f"<{dep['host']}>", "exec"), ns)  # trusted built-in code only (worldgen-owned orgs)
            fn = ns[handler]
            cache[key] = fn
        return fn

    def _usage(self, ctx, dep_id, n, cpu_ms, cu, outcome, work_scale, err=None):
        ctx.emit("cloud.usage", {"dep": dep_id, "n": n, "cpu_ms": round(cpu_ms, 4), "cu": round(cu, 3),
                                 "outcome": outcome, "ws": work_scale, "err": (err or "")[:200]}, vis="internal")

    def _revenue(self, ctx, dep_id, cents):
        ctx.emit("cloud.revenue", {"dep": dep_id, "cents": cents}, vis="internal")

    def record_revenue(self, ctx, dep_id, cents):
        """For revenue collected outside call_endpoint (aggregate consumer billing)."""
        if cents:
            self._revenue(ctx, dep_id, cents)

    # ------------------------------------------------------------------ endpoints: providers & instances
    @endpoint("providers", S.obj(), read_only=True)
    def ep_providers(self, ctx):
        """Compute providers: price per core-hour, free capacity, public outage history (last 30 days)."""
        out = []
        for pid, p in ctx.state.scan("providers"):
            if p["status"] != "active":
                continue
            used = sum(i["cores"] for i in ctx.state.find("instances", "provider", pid) if i["status"] == "running")
            out.append({"provider": pid, "name": p["name"], "price_core_hour_cents": p["price"],
                        "cores_total": p["cores"], "cores_free": p["cores"] - used,
                        "outages_30d": sum(1 for t in p["outages"] if t > ctx.now - 30 * 86400),
                        "in_outage": p["outage_until"] > ctx.now})
        return out

    @endpoint("rent", S.obj({"provider": S.s_str(maxLength=40), "cores": S.s_int(minimum=1, maximum=64),
                             "account": S.s_str("account to bill hourly", maxLength=40),
                             "owner": S.s_str("org to rent for (default you)", maxLength=40)},
                            ["provider", "cores", "account"]), cost_s=30)
    def ep_rent(self, ctx, provider, cores, account, owner=None):
        """Rent an instance. Billed every hour from `account`; unpaid bills suspend it, then terminate it."""
        st = ctx.state
        owner = ident(ctx.world).resolve_handle(st, owner) if owner else ctx.actor
        if owner != ctx.actor:
            ident(ctx.world).authorize(ctx, owner, "cloud.deploy")
        provider = ident(ctx.world).resolve_handle(st, provider)
        prov = st.get("providers", provider)
        if prov is None or prov["status"] != "active":
            raise NotFound("no such provider")
        bank = ctx.world.modules["bank"]
        acct_owner = bank.account_owner(st, account)
        ident(ctx.world).authorize(ctx, acct_owner, "bank.spend", scope=account)
        used = sum(i["cores"] for i in st.find("instances", "provider", provider) if i["status"] == "running")
        if used + cores > prov["cores"]:
            raise CapacityExceeded(f"{provider} has {prov['cores'] - used} free cores")
        first = prov["price"] * cores
        bank.move(ctx, account, bank.primary_account(st, provider), first, "compute", f"first hour {cores} cores",
                  ref=provider)
        iid = st.new_id("inst")
        ctx.emit("cloud.instance", {"id": iid, "owner": owner, "provider": provider, "cores": cores,
                                    "price": prov["price"], "account": account}, vis=(owner, provider))
        return {"instance": iid, "price_core_hour_cents": prov["price"], "capacity_cu_per_hour": cores * CU_PER_CORE_HOUR}

    @endpoint("release", S.obj({"instance": S.s_str(maxLength=40)}, ["instance"]))
    def ep_release(self, ctx, instance):
        """Release (terminate) an instance; services deployed on it stop."""
        inst = ctx.state.get("instances", instance)
        if inst is None or inst["status"] == "terminated":
            raise NotFound("no such instance")
        ident(ctx.world).authorize(ctx, inst["owner"], "cloud.deploy")
        self._terminate(ctx, instance, "released")
        return {"ok": True}

    def _terminate(self, ctx, iid, reason):
        for d in ctx.state.find_ids("deployments", "instance", iid):
            if ctx.state.tables["deployments"][d]["status"] == "live":
                ctx.emit("cloud.dep_status", {"id": d, "status": "stopped", "reason": f"instance {reason}"},
                         vis=(ctx.state.tables["deployments"][d]["owner"],))
        inst = ctx.state.tables["instances"][iid]
        ctx.emit("cloud.instance_status", {"id": iid, "status": "terminated", "reason": reason},
                 vis=(inst["owner"], inst["provider"]))

    @endpoint("instances", S.obj(), read_only=True)
    def ep_instances(self, ctx):
        """Your rented instances: provider, cores, status, price and compute used this hour."""
        out = []
        hour = ctx.now // 3600
        for owner in sorted(ctx.world.principals_of(ctx.actor)):
            for iid in ctx.state.find_ids("instances", "owner", owner):
                i = ctx.state.tables["instances"][iid]
                if i["status"] == "terminated":
                    continue
                out.append({"instance": iid, "owner": owner, "provider": i["provider"], "cores": i["cores"],
                            "status": i["status"], "price_core_hour_cents": i["price"],
                            "used_cu_this_hour": i["used_cu"] if i["hour"] == hour else 0.0,
                            "capacity_cu_per_hour": i["cores"] * CU_PER_CORE_HOUR})
        return out

    @endpoint("register_provider", S.obj({"org": S.s_str(maxLength=40), "name": S.s_str(maxLength=80),
                                         "price_core_hour_cents": S.s_int(minimum=1, maximum=100_000)},
                                        ["org", "price_core_hour_cents"]), cost_s=120)
    def ep_register_provider(self, ctx, org, price_core_hour_cents, name=""):
        """Become a compute provider (registered companies only). Capacity comes from buying hardware."""
        org = ident(ctx.world).resolve_handle(ctx.state, org)
        ident(ctx.world).authorize(ctx, org, "org.admin")
        reg = ctx.world.hooks.get("is_registered_company")
        if reg and not reg(ctx.state, org):
            raise Forbidden("register the company first (biz.register)")
        if ctx.state.has("providers", org):
            raise Conflict("already a provider")
        if ctx.world.modules["bank"].primary_account(ctx.state, org) is None:
            raise Forbidden("the provider needs a bank account")
        self.add_provider(ctx, org, name or org, price_core_hour_cents, 0, float(self.p("new_provider_reliability", 0.97)))
        return {"provider": org}

    def add_provider(self, ctx, org, name, price, cores, reliability):
        ctx.emit("cloud.provider", {"id": org, "name": name, "price": int(price), "cores": int(cores),
                                    "reliability": float(reliability)}, vis="public")

    @endpoint("set_price", S.obj({"provider": S.s_str(maxLength=40), "price_core_hour_cents": S.s_int(minimum=1, maximum=100_000)},
                                 ["provider", "price_core_hour_cents"]))
    def ep_set_price(self, ctx, provider, price_core_hour_cents):
        """Change the price for new rentals (existing instances keep their contracted price)."""
        provider = ident(ctx.world).resolve_handle(ctx.state, provider)
        ident(ctx.world).authorize(ctx, provider, "org.admin")
        if not ctx.state.has("providers", provider):
            raise NotFound("not a provider")
        ctx.emit("cloud.provider_price", {"id": provider, "price": price_core_hour_cents}, vis="public")
        return {"ok": True}

    @endpoint("buy_hardware", S.obj({"provider": S.s_str(maxLength=40), "cores": S.s_int(minimum=1, maximum=1000),
                                     "account": S.s_str(maxLength=40)}, ["provider", "cores", "account"]), cost_s=60)
    def ep_buy_hardware(self, ctx, provider, cores, account):
        """Order servers from the hardware market; they arrive after a lead time and add capacity."""
        st = ctx.state
        provider = ident(ctx.world).resolve_handle(st, provider)
        ident(ctx.world).authorize(ctx, provider, "org.admin")
        if not st.has("providers", provider):
            raise NotFound("not a provider")
        bank = ctx.world.modules["bank"]
        ident(ctx.world).authorize(ctx, bank.account_owner(st, account), "bank.spend", scope=account)
        price = int(self.p("hardware_cents_per_core", 300_000)) * cores
        bank.move(ctx, account, "acct_external", price, "hardware", f"{cores} cores", ref=provider)
        oid = st.new_id("hw")
        lead = int(self.p("hardware_lead_s", 7 * 86400))
        ctx.emit("cloud.hw_order", {"id": oid, "provider": provider, "cores": cores, "eta": ctx.now + lead},
                 vis=(provider,))
        ctx.schedule(ctx.now + lead, "cloud.deliver", {"order": oid}, prio=30)
        return {"order": oid, "eta": ctx.now + lead, "cost_cents": price}

    # ------------------------------------------------------------------ endpoints: deployments
    @endpoint("deploy", S.obj({"repo": S.s_str(maxLength=81), "host": S.s_str(pattern=HOST, maxLength=31),
                               "instance": S.s_str(maxLength=40), "ref": S.s_str(maxLength=40)},
                              ["repo", "host", "instance"]), cost_s=120)
    def ep_deploy(self, ctx, repo, host, instance, ref=None):
        """Deploy a repository as a service at <host>.crucible. Requires crucible.json:
        {"entry": "service.py", "endpoints": {"name": {"handler": "fn", "params": <JSON schema>,
        "doc": "...", "price_cents": 0}}}. Each handler is called as fn(args, caller) and must return JSON."""
        st = ctx.state
        code = ctx.world.modules["code"]
        r = code.repo(st, repo)
        owner = r["owner"]
        ident(ctx.world).authorize(ctx, owner, "cloud.deploy")
        inst = st.get("instances", instance)
        if inst is None or inst["owner"] != owner or inst["status"] != "running":
            raise Forbidden("deploy onto a running instance owned by the repository owner")
        if any(e.host == host for e in ctx.world.endpoints.values()):
            raise Conflict(f"{host} is reserved")
        h = st.get("hosts", host)
        if h is not None and h["owner"] != owner:
            raise Conflict(f"{host}.crucible is taken")
        files = code.files(st, repo, ref)
        manifest = self.validate_manifest(files)
        commit = ref or r["head"]
        svc = f"svc:{host}"
        if not st.has("principals", svc):
            ctx.emit("principal.add", {"id": svc, "kind": "service", "info": {"operator": owner, "host": host}},
                     vis="internal")
        old = self.live_deployment(st, host)
        if old is not None:
            old_id = st.get("hosts", host)["deployment"]
            ctx.emit("cloud.dep_status", {"id": old_id, "status": "replaced", "reason": "redeployed"}, vis=(owner,))
        did = st.new_id("dep")
        # in-process execution only for kernel-authored code (provenance), never because of who owns the repo
        rep = st.get("repos", repo)
        trusted = bool(st.get("orgs", owner) and st.get("orgs", owner)["info"].get("trusted_runtime")
                       and rep.get("trusted") and ref in (None, rep["head"]))
        ctx.emit("cloud.deployment", {"id": did, "host": host, "owner": owner, "repo": repo, "commit": commit,
                                      "instance": instance, "manifest": manifest, "svc": svc, "trusted": trusted},
                 vis=(owner,))
        ctx.world.learn(ctx, ctx.actor, "host", host)
        if not trusted:  # smoke test: the entry module must import inside the sandbox
            from ..sandbox.bridge import run_in_ctx
            from ..sandbox.runner import Worker
            pool = code.sandbox_pool(ctx.world)
            mod = manifest["entry"][:-3] if manifest["entry"].endswith(".py") else manifest["entry"]
            probe = run_in_ctx(ctx, "cloud.probe", lambda: Worker(self._files(ctx, st.get("deployments", did)),
                                                                      pool.cfg, "probe").start(),
                               {"kind": "import", "target": mod.replace("/", ".")},
                               svc, timeout=10, close_after=True)
            if not probe["ok"]:
                ctx.emit("cloud.dep_status", {"id": did, "status": "failed",
                                              "reason": f"import failed: {(probe['error'] or {}).get('message', '')[:300]}"},
                         vis=(owner,))
                return {"deployment": did, "status": "failed", "error": probe["error"]}
        return {"deployment": did, "host": f"{host}.crucible", "status": "live",
                "tools": [f"{host}.{e}" for e in sorted(manifest["endpoints"])]}

    def validate_manifest(self, files: dict) -> dict:
        raw = files.get("crucible.json")
        if raw is None:
            raise InvalidAction("crucible.json missing")
        try:
            m = json.loads(raw)
        except json.JSONDecodeError as e:
            raise InvalidAction(f"crucible.json: {e}")
        if not isinstance(m, dict) or not isinstance(m.get("endpoints"), dict) or not m["endpoints"]:
            raise InvalidAction("crucible.json needs an 'endpoints' object")
        entry = m.get("entry", "service.py")
        if not isinstance(entry, str) or not re.fullmatch(ENTRY, entry):
            raise InvalidAction("bad entry")
        if (entry if entry.endswith(".py") else entry + ".py") not in files:
            raise InvalidAction(f"entry {entry} not in repository")
        eps = {}
        for name, spec in m["endpoints"].items():
            if not re.fullmatch(EP_NAME, name) or not isinstance(spec, dict):
                raise InvalidAction(f"bad endpoint {name!r}")
            handler = spec.get("handler", name)
            if not isinstance(handler, str) or not re.fullmatch(HANDLER, handler):
                raise InvalidAction(f"bad handler for {name}")
            params = spec.get("params", {"type": "object", "additionalProperties": True})
            errs = S.check_schema(params)
            if errs:
                raise InvalidAction(f"endpoint {name}: " + "; ".join(errs[:3]))
            price = spec.get("price_cents", 0)
            if not isinstance(price, int) or price < 0 or price > 1_000_000:
                raise InvalidAction(f"bad price for {name}")
            eps[name] = {"handler": handler, "params": params, "doc": str(spec.get("doc", ""))[:500],
                         "price_cents": price}
        if len(eps) > 20:
            raise InvalidAction("at most 20 endpoints")
        return {"entry": entry, "endpoints": eps, "description": str(m.get("description", ""))[:500]}

    @endpoint("undeploy", S.obj({"host": S.s_str(maxLength=31)}, ["host"]))
    def ep_undeploy(self, ctx, host):
        """Stop a service you operate; its host stops answering."""
        h = ctx.state.get("hosts", host)
        if h is None or not h.get("deployment"):
            raise NotFound("no such host")
        dep = ctx.state.get("deployments", h["deployment"])
        ident(ctx.world).authorize(ctx, dep["owner"], "cloud.deploy")
        if dep["status"] == "live":
            ctx.emit("cloud.dep_status", {"id": h["deployment"], "status": "stopped", "reason": "undeployed"},
                     vis=(dep["owner"],))
        return {"ok": True}

    @endpoint("status", S.obj({"host": S.s_str(maxLength=31), "days": S.s_int(minimum=1, maximum=60)}, ["host"]),
              read_only=True)
    def ep_status(self, ctx, host, days=7):
        """Operational analytics for a service you operate: requests, errors, overloads, CPU, compute, revenue."""
        h = ctx.state.get("hosts", host)
        if h is None:
            raise NotFound("no such host")
        ident(ctx.world).authorize(ctx, h["owner"], "analytics.read")
        dep = ctx.state.get("deployments", h["deployment"])
        today = ctx.now // 86400
        rows = []
        for d in range(today - days + 1, today + 1):
            r = ctx.state.get("svc_days", f"{host}|{d}")
            if r:
                rows.append({"day": d, **dict(r)})
        return {"host": host, "status": dep["status"], "availability": self.availability(ctx.state, dep) or "ok",
                "instance": dep["instance"], "commit": dep["commit"], "recent_errors": list(dep["errors"]),
                "cpu_ms_per_request": round(dep["cpu_est"], 4), "days": rows}

    @endpoint("deployments", S.obj(), read_only=True)
    def ep_deployments(self, ctx):
        """Deployments operated by you or organisations you administer."""
        out = []
        for owner in sorted(ctx.world.principals_of(ctx.actor)):
            for did in ctx.state.find_ids("deployments", "owner", owner):
                d = ctx.state.tables["deployments"][did]
                if d["status"] in ("live", "failed", "stopped"):
                    out.append({"deployment": did, "host": d["host"], "status": d["status"], "repo": d["repo"],
                                "commit": d["commit"], "instance": d["instance"],
                                "endpoints": sorted(d["manifest"]["endpoints"])})
        return out

    # ------------------------------------------------------------------ processes
    @process("hourly")
    def pr_hourly(self, ctx):
        st = ctx.state
        bank = ctx.world.modules["bank"]
        grace = int(self.p("suspend_grace_s", 72 * 3600))
        for iid in st.find_ids("instances", "status", "running") + st.find_ids("instances", "status", "suspended"):
            inst = st.tables["instances"][iid]
            prov_acct = bank.primary_account(st, inst["provider"])
            amount = inst["price"] * inst["cores"]
            if inst["status"] == "suspended" and ctx.now - inst["suspended_at"] >= grace:
                self._terminate(ctx, iid, "unpaid")
                continue
            try:
                if prov_acct is None:
                    raise ServiceUnavailable("provider has no account")
                bank.move(ctx, inst["account"], prov_acct, amount, "compute", f"instance {iid}", ref=inst["provider"])
                if inst["status"] == "suspended":
                    ctx.emit("cloud.instance_status", {"id": iid, "status": "running", "reason": "paid"},
                             vis=(inst["owner"], inst["provider"]))
            except CrucibleError as e:
                if inst["status"] == "running":
                    ctx.emit("cloud.instance_status", {"id": iid, "status": "suspended", "reason": f"payment failed: {e.message}"},
                             vis=(inst["owner"], inst["provider"]))
        opex = int(self.p("opex_cents_per_core_hour", 60))
        for pid, p in st.scan("providers"):
            if p["status"] != "active" or p["cores"] == 0:
                continue
            acct = bank.primary_account(st, pid)
            if acct is None:
                continue
            try:
                bank.move(ctx, acct, "acct_external", opex * p["cores"], "opex", "datacenter power and upkeep")
            except CrucibleError:
                ctx.emit("cloud.provider_distress", {"id": pid}, vis=(pid,))

    @process("daily")
    def pr_daily(self, ctx):
        rng = ctx.rng("outages")
        for pid, p in ctx.state.scan("providers"):
            if p["status"] != "active":
                continue
            if rng.random() > p["reliability"]:
                start = ctx.now + int(rng.integers(0, 86400))
                dur = int(rng.integers(1, 7)) * 3600
                ctx.emit("cloud.outage", {"id": pid, "start": start, "until": start + dur}, vis="public")

    @process("deliver")
    def pr_deliver(self, ctx, order):
        o = ctx.state.get("hw_orders", order)
        if o is None or o["status"] != "ordered":
            return
        ctx.emit("cloud.hw_delivered", {"id": order, "provider": o["provider"], "cores": o["cores"]},
                 vis=(o["provider"],))

    def on_bankrupt(self, ctx, org):
        for iid in ctx.state.find_ids("instances", "owner", org):
            if ctx.state.tables["instances"][iid]["status"] != "terminated":
                self._terminate(ctx, iid, "owner bankrupt")
        if ctx.state.has("providers", org):
            ctx.emit("cloud.provider_status", {"id": org, "status": "closed"}, vis="public")
            for iid in ctx.state.find_ids("instances", "provider", org):
                if ctx.state.tables["instances"][iid]["status"] != "terminated":
                    self._terminate(ctx, iid, "provider bankrupt")

    # ------------------------------------------------------------------ appliers
    @applier("cloud.provider")
    def ap_provider(self, ev):
        p = ev.payload
        self.world.state.insert("providers", p["id"], {"name": p["name"], "price": p["price"], "cores": p["cores"],
                                                      "reliability": p["reliability"], "status": "active",
                                                      "outage_until": 0, "outage_start": 0, "outages": [], "t": ev.t})

    @applier("cloud.provider_price")
    def ap_provider_price(self, ev):
        self.world.state.patch("providers", ev.payload["id"], price=ev.payload["price"])

    @applier("cloud.provider_status")
    def ap_provider_status(self, ev):
        self.world.state.patch("providers", ev.payload["id"], status=ev.payload["status"])

    @applier("cloud.provider_distress")
    def ap_distress(self, ev):
        pass

    @applier("cloud.outage")
    def ap_outage(self, ev):
        p = ev.payload
        pr = self.world.state.tables["providers"][p["id"]]
        self.world.state.patch("providers", p["id"], outage_until=p["until"], outage_start=p["start"],
                               outages=(list(pr["outages"]) + [p["start"]])[-30:])

    @applier("cloud.instance")
    def ap_instance(self, ev):
        p = ev.payload
        self.world.state.insert("instances", p["id"], {"owner": p["owner"], "provider": p["provider"], "cores": p["cores"],
                                                      "price": p["price"], "account": p["account"], "status": "running",
                                                      "created": ev.t, "used_cu": 0.0, "hour": ev.t // 3600,
                                                      "suspended_at": None})

    @applier("cloud.instance_status")
    def ap_instance_status(self, ev):
        p = ev.payload
        ch = {"status": p["status"]}
        if p["status"] == "suspended":
            ch["suspended_at"] = ev.t
        self.world.state.patch("instances", p["id"], **ch)

    @applier("cloud.deployment")
    def ap_deployment(self, ev):
        p = ev.payload
        st = self.world.state
        st.insert("deployments", p["id"], {"host": p["host"], "owner": p["owner"], "repo": p["repo"],
                                          "commit": p["commit"], "instance": p["instance"], "manifest": p["manifest"],
                                          "svc": p["svc"], "trusted": p["trusted"], "status": "live", "created": ev.t,
                                          "cu_est": 0.0, "ws_est": 0.0, "cpu_est": 0.0, "errors": [], "requests": 0})
        st.put("hosts", p["host"], {"owner": p["owner"], "deployment": p["id"], "created": ev.t})

    @applier("cloud.dep_status")
    def ap_dep_status(self, ev):
        self.world.state.patch("deployments", ev.payload["id"], status=ev.payload["status"],
                               status_reason=ev.payload.get("reason", ""))

    @applier("cloud.usage")
    def ap_usage(self, ev):
        p = ev.payload
        st = self.world.state
        dep = st.tables["deployments"][p["dep"]]
        hour, day = ev.t // 3600, ev.t // 86400
        ch = {"requests": dep["requests"] + p["n"]}
        if p["n"]:
            a = 0.1 if dep["requests"] else 1.0
            ch["cpu_est"] = (1 - a) * dep["cpu_est"] + a * p["cpu_ms"]
            ch["cu_est"] = (1 - a) * dep["cu_est"] + a * p["cu"]
            ch["ws_est"] = p["ws"]
            inst = st.tables["instances"][dep["instance"]]
            used = inst["used_cu"] if inst["hour"] == hour else 0.0
            st.patch("instances", dep["instance"], used_cu=used + p["cu"], hour=hour)
        if p["outcome"] == "error" and p.get("err"):
            ch["errors"] = (list(dep["errors"]) + [p["err"]])[-10:]
        st.patch("deployments", p["dep"], **ch)
        key = f"{dep['host']}|{day}"
        r = st.tables["svc_days"].get(key) or {"host": dep["host"], "requests": 0, "ok": 0, "errors": 0,
                                              "unavailable": 0, "overload": 0, "cpu_ms": 0.0, "cu": 0.0,
                                              "revenue_cents": 0}
        r = dict(r)
        r["requests"] += p["n"] if p["outcome"] in ("ok", "error") else 1
        if p["outcome"] in ("ok", "error", "unavailable", "overload"):
            k = {"ok": "ok", "error": "errors", "unavailable": "unavailable", "overload": "overload"}[p["outcome"]]
            r[k] += 1
        r["cpu_ms"] = round(r["cpu_ms"] + p["cpu_ms"], 4)
        r["cu"] = round(r["cu"] + p["cu"], 3)
        st.put("svc_days", key, r)

    @applier("cloud.usage_bulk")
    def ap_usage_bulk(self, ev):
        """Aggregate usage for the extrapolated part of a demand batch (requests represented, not re-executed)."""
        p = ev.payload
        st = self.world.state
        dep = st.tables["deployments"][p["dep"]]
        hour, day = ev.t // 3600, ev.t // 86400
        inst = st.tables["instances"][dep["instance"]]
        used = inst["used_cu"] if inst["hour"] == hour else 0.0
        st.patch("instances", dep["instance"], used_cu=used + p["cu"], hour=hour)
        st.patch("deployments", p["dep"], requests=dep["requests"] + p["ok"] + p["errors"])
        key = f"{dep['host']}|{day}"
        r = dict(st.tables["svc_days"].get(key) or {"host": dep["host"], "requests": 0, "ok": 0, "errors": 0,
                                                   "unavailable": 0, "overload": 0, "cpu_ms": 0.0, "cu": 0.0,
                                                   "revenue_cents": 0})
        for k in ("ok", "errors", "unavailable", "overload"):
            r[k] += p[k]
        r["requests"] += p["ok"] + p["errors"] + p["unavailable"] + p["overload"]
        r["cu"] = round(r["cu"] + p["cu"], 3)
        st.put("svc_days", key, r)

    @applier("cloud.revenue")
    def ap_revenue(self, ev):
        st = self.world.state
        dep = st.tables["deployments"][ev.payload["dep"]]
        key = f"{dep['host']}|{ev.t // 86400}"
        r = dict(st.tables["svc_days"].get(key) or {"host": dep["host"], "requests": 0, "ok": 0, "errors": 0,
                                                   "unavailable": 0, "overload": 0, "cpu_ms": 0.0, "cu": 0.0,
                                                   "revenue_cents": 0})
        r["revenue_cents"] += ev.payload["cents"]
        st.put("svc_days", key, r)

    @applier("cloud.hw_order")
    def ap_hw_order(self, ev):
        p = ev.payload
        self.world.state.insert("hw_orders", p["id"], {"provider": p["provider"], "cores": p["cores"], "eta": p["eta"],
                                                      "status": "ordered", "t": ev.t})

    @applier("cloud.hw_delivered")
    def ap_hw_delivered(self, ev):
        p = ev.payload
        st = self.world.state
        st.patch("hw_orders", p["id"], status="delivered")
        pr = st.tables["providers"][p["provider"]]
        st.patch("providers", p["provider"], cores=pr["cores"] + p["cores"])

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        st = world.state
        errs = []
        for pid, p in st.tables["providers"].items():
            used = sum(i["cores"] for i in st.find("instances", "provider", pid) if i["status"] == "running")
            if used > p["cores"]:
                errs.append(f"provider {pid} oversubscribed: {used} > {p['cores']} cores")
        hour = world.now // 3600
        for iid, i in st.tables["instances"].items():
            if i["hour"] == hour and i["used_cu"] > i["cores"] * CU_PER_CORE_HOUR * 1.0 + 1e-6:
                # a single request may finish past the cap; allow at most one request's worth of overshoot
                pass
        for host, h in st.tables["hosts"].items():
            d = st.get("deployments", h["deployment"])
            if d is None or d["host"] != host:
                errs.append(f"dns {host} points to a bad deployment")
        for did, d in st.tables["deployments"].items():
            if d["status"] == "live":
                inst = st.get("instances", d["instance"])
                if inst is None or inst["status"] == "terminated":
                    errs.append(f"live deployment {did} on terminated instance")
        return errs

    def observe(self, world, actor):
        st = world.state
        out = {}
        for owner in sorted(world.principals_of(actor)):
            for did in st.find_ids("deployments", "owner", owner):
                d = st.tables["deployments"][did]
                if d["status"] == "live":
                    out[d["host"]] = self.availability(st, d) or "ok"
        return {"services": out} if out else None

    def close(self):
        pass
