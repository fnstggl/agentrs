"""Capability registry v5 (module `caps`, host `caps`; INTERFACE_CONTRACTS section 7).

This module extends the 4.0 capability registry (`crucible.services.capabilities.Capabilities`, which it subclasses
and replaces under the same module name) with *typed capability declarations* for agent-created services:

    caps.declare {service_host, capability: {name, inputs, outputs, doc, endpoint?, tags?}, tests_ref?}
    caps.search  {query}                      (the 4.0 form {q} keeps its exact 4.0 behaviour)
    caps.try     {service_host, capability, input}

A declaration says: the live service `<service_host>` offers capability `name` at one of its endpoints; requests
must satisfy the JSON schema `inputs`, responses satisfy `outputs`. Nothing in the world enumerates which
capabilities may exist; the registry only indexes what operators declare.

* `caps.try` executes the real service through the 4.0 cloud path (payment of the endpoint price, sandboxed
  execution, metering against the instance's capacity) and checks the response against the declared output schema.
  The trial is recorded for the caller (`caps.tried`). Whether the output *solves the caller's problem* is for the
  caller to judge with its own tests; the registry never scores usefulness itself.
* `caps.verify` runs an independently written test suite (a repository with `suite.py: generate(args, caller) ->
  input` and `check(args, caller) -> score in [0, 1]`) against the live endpoint in the sandbox, on freshly drawn
  seeds, and records the scores as evidence. Suites written by the provider (`tests_ref`) are recorded as
  provider-supplied, never as independent.
* `caps.adopt_service` records that a principal relies on a capability. Adoption requires that the adopter has
  itself executed the service successfully (`caps.try`); afterwards every real call it makes to that endpoint is
  counted as use (`caps.use`). Nothing is ever counted from what a provider says about itself.

Observation-only (D-19, audit G 5.4): public statistics are raw counts anyone could tally - trials, successful
trials, active adopters, public verifications. Whether a trial or an adoption was by a party independent of the
provider is an evaluator verdict (`innovation.measure`); it is never computed in-world, stored in an event, or
returned by a tool (an `independent` flag would be a Sybil oracle). Adoption notes are kept with the adopter's own
record and are not relayed to the provider.

The v5 tables and endpoints exist only when the module is configured with `{"registry": true}`: worlds built before
5.0 keep their exact tool catalogue, state digests and event logs (the 4.0 endpoints, appliers and processes are
inherited unchanged, and `caps.search` keeps its 4.0 schema there).
"""
from __future__ import annotations

import types

from ..core import schema as S
from ..core.canonical import digest
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Endpoint, applier, endpoint, register_module
from ..services.capabilities import Capabilities, tokens
from ..services.identity import ident

CAP_NAME = r"[a-z][a-z0-9_]*(\.[a-z0-9_]+){0,3}"
HOST_RE = r"[a-z][a-z0-9-]{2,30}"
REPO_REF = r"[a-z0-9][a-z0-9_-]{0,39}/[a-z][a-z0-9_-]{0,39}(@c[0-9a-f]{20})?"
V5_ENDPOINTS = ("declare", "declaration", "declarations", "try", "verify", "adopt_service", "drop_service",
                "adoptions")
V5_TABLES = {
    "cap_decls": ("host", "provider", "name", "status"),
    "cap_trials": ("decl", "by"),
    "cap_adoptions": ("decl", "adopter", "status"),
    "cap_seen": ("decl", "by"),
    "cap_verifs": ("decl", "by"),
}
CAPABILITY_SCHEMA = S.obj({
    "name": S.s_str("capability name, e.g. route.shortest or invoice.extract", pattern=CAP_NAME, maxLength=60),
    "inputs": S.s_any_obj("JSON schema (type object) every request satisfies"),
    "outputs": S.s_any_obj("JSON schema (type object) every successful response satisfies"),
    "doc": S.s_str("what it does and how to call it", maxLength=4000),
    "endpoint": S.s_str("the service endpoint that implements it (default: the only endpoint, or one named like "
                        "the capability's last part)", maxLength=31),
    "tags": S.s_arr(S.s_str(maxLength=30), maxItems=12)}, ["name", "inputs", "outputs", "doc"])


def schema_props(schema: dict) -> dict:
    return dict(schema.get("properties") or {}) if isinstance(schema, dict) else {}


def _types(s: dict) -> set:
    t = (s or {}).get("type")
    if t is None:
        return {"any"}
    return set(t if isinstance(t, list) else [t])


def compatible(out_schema: dict, in_schema: dict) -> bool:
    """Structural compatibility: every required input property of `in_schema` is produced by `out_schema` with an
    overlapping type (used for the needs graph's `complements` relation)."""
    req = list((in_schema or {}).get("required") or [])
    if not req:
        return False
    outp, inp = schema_props(out_schema), schema_props(in_schema)
    for k in req:
        if k not in outp:
            return False
        a, b = _types(outp[k]), _types(inp.get(k, {}))
        if "any" not in a and "any" not in b and not (a & b) and not ({"integer", "number"} & a and "number" in b):
            return False
    return True


@register_module
class CapabilityRegistry(Capabilities):
    name = "caps"

    # ------------------------------------------------------------------ setup
    @property
    def registry(self) -> bool:
        return bool(self.config.get("registry", False))

    def install(self, world):
        super().install(world)
        world.hooks["caps_registry"] = self
        if not self.registry:
            # a world without the v5 registry keeps the exact 4.0 tool surface: the v5 endpoints are withdrawn and
            # caps.search is the 4.0 endpoint (schema, handler and version), so tool catalogues are unchanged
            host = self.host or self.name
            for ep in V5_ENDPOINTS:
                world.endpoints.pop(f"{host}.{ep}", None)
            e = Capabilities.ep_search.__crucible_endpoint__
            world.endpoints[f"{host}.search"] = Endpoint(
                f"{host}.search", host, e["params"], types.MethodType(Capabilities.ep_search, self), e["doc"],
                e["cost_s"], e["read_only"], e["returns"], version=str(e.get("version") or self.version))
            return
        for t, idx in V5_TABLES.items():
            world.state.declare_table(t, index=idx)
        prev = world.hooks.get("on_service_call")

        def on_call(ctx, caller, host, ep, status, price):
            if prev:
                prev(ctx, caller, host, ep, status, price)
            self._count_use(ctx, caller, host, ep, status)
        world.hooks["on_service_call"] = on_call

    def _need_registry(self):
        if not self.registry:
            raise InvalidAction("the typed capability registry is not enabled in this world "
                                "(module caps config {\"registry\": true})")

    # ------------------------------------------------------------------ helpers
    def decl(self, st, ref: str, host: str | None = None):
        """A declaration by id, or the active one named `ref` on `host`."""
        d = st.get("cap_decls", ref)
        if d is not None:
            return ref, d
        if host:
            for did in st.find_ids("cap_decls", "host", host):
                r = st.tables["cap_decls"][did]
                if r["name"] == ref and r["status"] == "active":
                    return did, r
        raise NotFound(f"no active capability {ref!r}" + (f" on {host}" if host else ""))

    def price_of(self, st, d) -> int:
        dep = self.world.modules["cloud"].live_deployment(st, d["host"])
        if dep is None:
            return 0
        return int(dep["manifest"]["endpoints"].get(d["endpoint"], {}).get("price_cents", 0))

    def decl_summary(self, world, did: str) -> dict:
        st = world.state
        d = st.tables["cap_decls"][did]
        live = world.modules["cloud"].live_deployment(st, d["host"]) is not None
        verifs = [st.tables["cap_verifs"][v] for v in st.find_ids("cap_verifs", "decl", did)]
        pub = [v for v in verifs if v["public"]]
        return {"declaration": did, "host": d["host"], "name": d["name"], "endpoint": d["endpoint"],
                "provider": d["provider"], "doc": d["doc"][:600], "tags": list(d["tags"]),
                "inputs": d["inputs"], "outputs": d["outputs"], "status": d["status"], "live": live,
                "price_cents": self.price_of(st, d), "declared_t": d["t"], "commit": d["commit"],
                "tests_ref": d["tests_ref"],
                "trials": d["trials"], "trial_ok_rate": round(d["trials_ok"] / d["trials"], 4) if d["trials"] else None,
                "adopters": d["adopters"],
                "public_verifications": [{"verification": v["id"], "by": v["by"],
                                          "provider_supplied": v["provider_supplied"], "cases": v["cases"],
                                          "mean_score": v["mean_score"], "failures": v["failures"],
                                          "schema_failures": v["schema_failures"], "commit": v["commit"]}
                                         for v in pub][-5:]}


    # ------------------------------------------------------------------ declare
    @endpoint("declare", S.obj({
        "service_host": S.s_str("your live service host", pattern=HOST_RE, maxLength=31),
        "capability": CAPABILITY_SCHEMA,
        "tests_ref": S.s_str("optional test suite repository owner/repo[@commit] with suite.py (generate, check); "
                             "recorded as provider-supplied", pattern=REPO_REF, maxLength=110)},
        ["service_host", "capability"]), cost_s=60, version="5")
    def ep_declare(self, ctx, service_host, capability, tests_ref=None):
        """Declare what one endpoint of your live service does: a capability name, the JSON schema of its inputs
        and of its outputs, and documentation. Declarations are public and searchable (caps.search); re-declaring
        the same name on the same host supersedes the previous declaration."""
        self._need_registry()
        st = ctx.state
        cloud = ctx.world.modules["cloud"]
        dep = cloud.live_deployment(st, service_host)
        if dep is None:
            raise NotFound(f"{service_host} is not a live service")
        owner = dep["owner"]
        ident(ctx.world).authorize(ctx, owner, "cloud.deploy", scope=service_host)
        cap = capability
        for k in ("inputs", "outputs"):
            errs = S.check_schema(cap[k])
            if errs:
                raise InvalidAction(f"capability.{k}: " + "; ".join(errs[:3]))
        eps = dep["manifest"]["endpoints"]
        ep = cap.get("endpoint")
        if ep is None:
            last = cap["name"].split(".")[-1]
            ep = next(iter(eps)) if len(eps) == 1 else (last if last in eps else None)
        if ep not in eps:
            raise InvalidAction(f"capability endpoint {ep!r} is not served by {service_host} "
                                f"(endpoints: {', '.join(sorted(eps))})")
        suite = None
        if tests_ref:
            repo, _, ref = tests_ref.partition("@")
            code = ctx.world.modules["code"]
            r = code.repo(st, repo)
            files = code.files(st, repo, ref or None)
            if "suite.py" not in files:
                raise InvalidAction("tests_ref repository needs suite.py")
            suite = f"{repo}@{ref or r['head']}"
        prev = [did for did in st.find_ids("cap_decls", "host", service_host)
                if st.tables["cap_decls"][did]["name"] == cap["name"] and st.tables["cap_decls"][did]["status"] == "active"]
        did = st.new_id("cdecl")
        tags = sorted({t.lower() for t in cap.get("tags") or []})
        ctx.emit("caps.declared", {"id": did, "host": service_host, "provider": owner, "by": ctx.actor,
                                   "name": cap["name"], "endpoint": ep, "inputs": cap["inputs"],
                                   "outputs": cap["outputs"], "doc": cap["doc"], "tags": tags,
                                   "tests_ref": suite, "commit": dep["commit"],
                                   "deployment": st.get("hosts", service_host)["deployment"], "supersedes": prev},
                 vis="public")
        from ..services.search import index
        fields = " ".join(sorted(schema_props(cap["inputs"]))) + " -> " + " ".join(sorted(schema_props(cap["outputs"])))
        index(ctx, f"capdecl:{did}", "capability", cap["name"], f"{cap['doc']}\nCapability: {cap['name']}\n"
              f"Tags: {' '.join(tags)}\nFields: {fields}\nHost: {service_host}", owner, ref=did, hosts=[service_host])
        for old in prev:
            from ..services.search import unindex
            unindex(ctx, f"capdecl:{old}")
        ctx.world.learn(ctx, ctx.actor, "host", service_host)
        return {"declaration": did, "supersedes": prev, "endpoint": ep}

    # ------------------------------------------------------------------ discovery
    @endpoint("search", S.obj({"query": S.s_str("what you need, in words, or a capability name or host",
                                                maxLength=300),
                               "q": S.s_str("4.0 form: full-text search over published (installable) capabilities",
                                            maxLength=300),
                               "limit": S.s_int(minimum=1, maximum=50)}), read_only=True, version="5")
    def ep_search(self, ctx, query=None, q=None, limit=10):
        """Search capabilities. `query` searches typed declarations of agent-created services (name, docs, tags,
        fields, host) and the installable capabilities published with caps.publish; each result carries public
        statistics (trials, successful trials, adopters, public verifications). Hosts of results become known to you.
        The older form `q` searches published capabilities only."""
        if query is None:
            if q is None:
                raise InvalidAction("give a query")
            return Capabilities.ep_search(self, ctx, q, limit)
        out = []
        st = ctx.state
        if self.registry:
            seen_now = []
            for did in self.match(ctx.world, query, limit):
                s = self.decl_summary(ctx.world, did)
                s["kind"] = "declaration"
                out.append(s)
                seen_now.append(did)
                ctx.world.learn(ctx, ctx.actor, "host", s["host"])
            new = [d for d in seen_now if not st.has("cap_seen", f"{d}|{ctx.actor}")]
            if new:
                ctx.emit("caps.seen", {"by": ctx.actor, "decls": new, "via": "caps.search"}, vis=(ctx.actor,))
        room = max(0, limit - len(out))
        if room:
            for r in Capabilities.ep_search(self, ctx, query, room):
                out.append({**r, "kind": "published"})
        return out

    def match(self, world, query: str, limit: int) -> list[str]:
        """Active declarations matching a query: exact name/host matches first, then BM25 over the index."""
        st = world.state
        qn = query.strip().lower()
        exact = []
        for did in st.find_ids("cap_decls", "status", "active"):
            d = st.tables["cap_decls"][did]
            if qn in (d["name"], d["host"], did):
                exact.append(did)
        ranked = []
        search = world.modules.get("search")
        if search is not None:
            sp, org, _ = search.search(world, query, "capability", limit * 3)
            for r in sp + org:
                ref = r.get("ref")
                if ref and st.has("cap_decls", ref) and st.tables["cap_decls"][ref]["status"] == "active":
                    ranked.append(ref)
        else:   # no search engine: token overlap over declarations
            qt = tokens(query)
            sc = []
            for did in st.find_ids("cap_decls", "status", "active"):
                d = st.tables["cap_decls"][did]
                n = len(qt & tokens(" ".join([d["name"].replace(".", " "), d["doc"], " ".join(d["tags"])])))
                if n:
                    sc.append((-n, did))
            ranked = [d for _, d in sorted(sc)]
        out = []
        for did in exact + ranked:
            if did not in out:
                out.append(did)
        return out[:limit]

    @endpoint("declaration", S.obj({"declaration": S.s_str(maxLength=40)}, ["declaration"]), read_only=True,
              version="5")
    def ep_declaration(self, ctx, declaration):
        """One capability declaration with its public statistics (you learn its host)."""
        self._need_registry()
        did, d = self.decl(ctx.state, declaration)
        ctx.world.learn(ctx, ctx.actor, "host", d["host"])
        return self.decl_summary(ctx.world, did)

    @endpoint("declarations", S.obj({"host": S.s_str(maxLength=31), "provider": S.s_str(maxLength=40),
                                     "limit": S.s_int(minimum=1, maximum=200)}), read_only=True, version="5")
    def ep_declarations(self, ctx, host=None, provider=None, limit=50):
        """Active declarations, newest first, optionally of one host or provider."""
        self._need_registry()
        st = ctx.state
        ids = st.find_ids("cap_decls", "status", "active")
        if host:
            ids = [i for i in ids if st.tables["cap_decls"][i]["host"] == host]
        if provider:
            pv = ident(ctx.world).resolve_handle(st, provider)
            ids = [i for i in ids if st.tables["cap_decls"][i]["provider"] == pv]
        return [self.decl_summary(ctx.world, i) for i in list(reversed(ids))[:limit]]

    # ------------------------------------------------------------------ trying (real execution)
    @endpoint("try", S.obj({"service_host": S.s_str(pattern=HOST_RE, maxLength=31),
                            "capability": S.s_str("capability name (or declaration id)", maxLength=60),
                            "input": S.s_any_obj("one request, valid against the declared inputs schema")},
                           ["service_host", "capability", "input"]), cost_s=10, version="5")
    def ep_try(self, ctx, service_host, capability, input):
        """Execute a declared capability once on your own input: the real service runs (you pay its price, the
        call uses its compute) and the response is checked against the declared outputs schema. Returns the
        output so you can judge it with your own tests; the trial is recorded for you."""
        self._need_registry()
        st = ctx.state
        did, d = self.decl(st, capability, service_host)
        if d["host"] != service_host:
            raise InvalidAction(f"{capability} is declared on {d['host']}, not {service_host}")
        errs = S.validate(d["inputs"], input, "input")
        if errs:
            raise InvalidAction("input does not satisfy the declared inputs schema: " + "; ".join(errs[:4]))
        try:
            res = ctx.call(f"{service_host}.{d['endpoint']}", input)
        except CrucibleError as e:
            res = {"status": 400, "error": e.message}
        if not isinstance(res, dict):
            res = {"status": 500, "error": "malformed response"}
        status = int(res.get("status", 500))
        body = res.get("body")
        out_errs = S.validate(d["outputs"], body, "output") if status == 200 else []
        ok = status == 200 and not out_errs
        dep = ctx.world.modules["cloud"].live_deployment(st, service_host)
        tid = st.new_id("ctry")
        ctx.emit("caps.tried", {"id": tid, "decl": did, "host": service_host, "by": ctx.actor, "status": status,
                                "ok": ok, "schema_errors": out_errs[:5], "cpu_ms": res.get("cpu_ms"),
                                "commit": dep["commit"] if dep else d["commit"],
                                "input_digest": digest(input), "output_digest": digest(body)}, vis=(ctx.actor,))
        return {"trial": tid, "declaration": did, "status": status, "ok": ok, "output": body,
                "schema_errors": out_errs[:5], "cpu_ms": res.get("cpu_ms"), "error": res.get("error")}

    # ------------------------------------------------------------------ verification against test suites
    @endpoint("verify", S.obj({"declaration": S.s_str(maxLength=40),
                               "suite_repo": S.s_str("your repo with suite.py: generate(args, caller) -> input; "
                                                     "check(args, caller) -> score in [0,1] for args {input, output}; "
                                                     "default: the provider's tests_ref", maxLength=110),
                               "cases": S.s_int(minimum=1, maximum=20), "public": S.s_bool()},
                              ["declaration"]), cost_s=300, version="5")
    def ep_verify(self, ctx, declaration, suite_repo=None, cases=6, public=True):
        """Executable verification: generate fresh test cases with a suite (yours, or the provider's tests_ref),
        run them against the live service and record mean score, failures and schema violations as evidence.
        Seeds are drawn fresh for every verification, so providers cannot tune to known cases. You pay the calls."""
        self._need_registry()
        st = ctx.state
        did, d = self.decl(st, declaration)
        code = ctx.world.modules["code"]
        provider_supplied = suite_repo is None
        if provider_supplied:
            if not d["tests_ref"]:
                raise InvalidAction("the declaration has no tests_ref; give suite_repo")
            repo, _, ref = d["tests_ref"].partition("@")
        else:
            repo, _, ref = suite_repo.partition("@")
        r = code.repo(st, repo)
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        ref = ref or r["head"]
        files = code.files(st, repo, ref)
        if "suite.py" not in files:
            raise InvalidAction("suite repo needs suite.py")
        from ..sandbox.bridge import run_in_ctx
        pool = code.sandbox_pool(ctx.world)
        key = f"{ctx.world.world_id}|suite|{repo}|{ref}"

        def worker():
            return pool.get(key, lambda: code.files(st, repo, ref))
        rng = ctx.rng("verify", did, ctx.actor)
        seeds = [int(x) for x in rng.integers(0, 2**31 - 1, int(cases))]
        scores, fails, schema_fails, cpus = [], 0, 0, []
        for seed in seeds:
            g = run_in_ctx(ctx, "caps.suite", worker, {"kind": "call", "target": "suite:generate",
                                                        "args": {"seed": seed, "capability": d["name"]},
                                                        "caller": ctx.actor}, ctx.actor, timeout=5.0,
                           world_access=False)
            if not g["ok"] or not isinstance(g["result"], dict):
                raise InvalidAction(f"suite.generate failed: {(g.get('error') or {}).get('message', 'not a dict')}")
            if S.validate(d["inputs"], g["result"], "input"):
                raise InvalidAction("the suite generated an input that violates the declared inputs schema")
            try:
                res = ctx.call(f"{d['host']}.{d['endpoint']}", g["result"])
            except CrucibleError as e:
                res = {"status": 400, "error": e.message}
            if not isinstance(res, dict) or res.get("status") != 200:
                fails += 1
                scores.append(0.0)
                continue
            if S.validate(d["outputs"], res.get("body"), "output"):
                schema_fails += 1
            cpus.append(float(res.get("cpu_ms") or 0.0))
            ch = run_in_ctx(ctx, "caps.suite", worker, {"kind": "call", "target": "suite:check",
                                                         "args": {"input": g["result"], "output": res.get("body")},
                                                         "caller": ctx.actor}, ctx.actor, timeout=5.0,
                            world_access=False)
            try:
                scores.append(float(min(1.0, max(0.0, float(ch["result"])))) if ch["ok"] else 0.0)
            except (TypeError, ValueError):
                scores.append(0.0)
        dep = ctx.world.modules["cloud"].live_deployment(st, d["host"])
        vid = st.new_id("cver")
        by = r["owner"]
        rec = {"id": vid, "decl": did, "by": by, "requested_by": ctx.actor, "suite": f"{repo}@{ref}",
               "provider_supplied": provider_supplied,
               "cases": int(cases), "mean_score": round(sum(scores) / len(scores), 4), "failures": fails,
               "schema_failures": schema_fails, "mean_cpu_ms": round(sum(cpus) / len(cpus), 4) if cpus else None,
               "commit": dep["commit"] if dep else d["commit"], "public": bool(public)}
        ctx.emit("caps.verification", rec, vis="public" if public else (ctx.actor, by, d["provider"]))
        return {"verification": vid, **{k: rec[k] for k in ("mean_score", "failures", "schema_failures", "cases",
                                                              "provider_supplied", "commit")}}

    # ------------------------------------------------------------------ adoption (only after own execution)
    @endpoint("adopt_service", S.obj({"declaration": S.s_str(maxLength=40),
                                      "note": S.s_str("why (kept with the adoption record)", maxLength=500)},
                                     ["declaration"]), version="5")
    def ep_adopt_service(self, ctx, declaration, note=""):
        """Record that you now rely on a declared capability. Requires that you executed it successfully yourself
        (caps.try); afterwards your real calls to it are counted as use."""
        self._need_registry()
        st = ctx.state
        did, d = self.decl(st, declaration)
        trials = [st.tables["cap_trials"][t] for t in st.find_ids("cap_trials", "by", ctx.actor)]
        mine = [t for t in trials if t["decl"] == did]
        if not any(t["ok"] for t in mine):
            raise Forbidden("adopt only what you have executed successfully yourself (caps.try)")
        if any(st.tables["cap_adoptions"][a]["decl"] == did and st.tables["cap_adoptions"][a]["status"] == "active"
               for a in st.find_ids("cap_adoptions", "adopter", ctx.actor)):
            raise Conflict("already adopted")
        aid = st.new_id("cadopt")
        ctx.emit("caps.adopted", {"id": aid, "decl": did, "host": d["host"], "adopter": ctx.actor,
                                  "provider": d["provider"],
                                  "trials": len(mine), "trials_ok": sum(1 for t in mine if t["ok"])},
                 vis=(ctx.actor, d["provider"]))
        if note:
            ctx.emit("caps.adoption_note", {"id": aid, "note": note[:500]}, vis=(ctx.actor,))
        return {"adoption": aid}

    @endpoint("drop_service", S.obj({"declaration": S.s_str(maxLength=40), "reason": S.s_str(maxLength=300)},
                                    ["declaration"]), version="5")
    def ep_drop_service(self, ctx, declaration, reason=""):
        """Stop relying on a capability you adopted."""
        self._need_registry()
        st = ctx.state
        did, d = self.decl(st, declaration)
        act = [a for a in st.find_ids("cap_adoptions", "adopter", ctx.actor)
               if st.tables["cap_adoptions"][a]["decl"] == did and st.tables["cap_adoptions"][a]["status"] == "active"]
        if not act:
            raise NotFound("you have not adopted " + did)
        for a in act:
            ctx.emit("caps.dropped", {"id": a, "reason": reason[:300]}, vis=(ctx.actor, d["provider"]))
        return {"ok": True}

    @endpoint("adoptions", S.obj(), read_only=True, version="5")
    def ep_adoptions(self, ctx):
        """Capabilities you (or organisations you administer) have adopted, with your own use counts."""
        self._need_registry()
        st = ctx.state
        out = []
        for p in sorted(ctx.world.principals_of(ctx.actor)):
            for a in st.find_ids("cap_adoptions", "adopter", p):
                r = st.tables["cap_adoptions"][a]
                out.append({"adoption": a, **{k: r[k] for k in ("decl", "host", "adopter", "status", "uses",
                                                                "uses_ok", "t")}})
        return out

    def _count_use(self, ctx, caller, host, ep, status):
        st = ctx.state
        cp = st.get("principals", caller)
        who = caller
        if cp is not None and cp["kind"] == "service":
            who = cp["info"].get("operator") or caller
        for a in st.find_ids("cap_adoptions", "adopter", who):
            r = st.tables["cap_adoptions"][a]
            if r["status"] == "active" and r["host"] == host and r["endpoint"] == ep:
                ctx.emit("caps.use", {"id": a, "ok": status == 200}, vis="internal")

    # ------------------------------------------------------------------ appliers (v5)
    @applier("caps.declared")
    def ap_declared(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        for old in p["supersedes"]:
            st.patch("cap_decls", old, status="superseded")
        p.update(status="active", t=ev.t, trials=0, trials_ok=0, adopters=0)
        st.insert("cap_decls", p.pop("id"), p)

    @applier("caps.seen")
    def ap_seen(self, ev):
        st = self.world.state
        for did in ev.payload["decls"]:
            k = f"{did}|{ev.payload['by']}"
            if not st.has("cap_seen", k):
                st.insert("cap_seen", k, {"decl": did, "by": ev.payload["by"], "via": ev.payload["via"], "t": ev.t})

    @applier("caps.tried")
    def ap_tried(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        p["t"] = ev.t
        st.insert("cap_trials", p.pop("id"), p)
        d = st.tables["cap_decls"][p["decl"]]
        st.patch("cap_decls", p["decl"], trials=d["trials"] + 1, trials_ok=d["trials_ok"] + (1 if p["ok"] else 0))

    @applier("caps.verification")
    def ap_verification(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("cap_verifs", p["id"], p)

    @applier("caps.adopted")
    def ap_adopted(self, ev):
        st = self.world.state
        p = dict(ev.payload)
        d = st.tables["cap_decls"][p["decl"]]
        p.update(status="active", t=ev.t, uses=0, uses_ok=0, endpoint=d["endpoint"])
        st.insert("cap_adoptions", p.pop("id"), p)
        st.patch("cap_decls", p["decl"], adopters=d["adopters"] + 1)

    @applier("caps.adoption_note")
    def ap_adoption_note(self, ev):
        self.world.state.patch("cap_adoptions", ev.payload["id"], note=ev.payload["note"])

    @applier("caps.dropped")
    def ap_dropped(self, ev):
        st = self.world.state
        a = st.tables["cap_adoptions"][ev.payload["id"]]
        st.patch("cap_adoptions", ev.payload["id"], status="dropped", dropped_t=ev.t, reason=ev.payload["reason"])
        d = st.tables["cap_decls"][a["decl"]]
        st.patch("cap_decls", a["decl"], adopters=d["adopters"] - 1)

    @applier("caps.use")
    def ap_use(self, ev):
        st = self.world.state
        a = st.tables["cap_adoptions"][ev.payload["id"]]
        st.patch("cap_adoptions", ev.payload["id"], uses=a["uses"] + 1, uses_ok=a["uses_ok"] + (1 if ev.payload["ok"] else 0),
                 last_use=ev.t)

    # ------------------------------------------------------------------ invariants
    def check_invariants(self, world):
        errs = super().check_invariants(world)
        if not self.registry:
            return errs
        st = world.state
        trials: dict[str, list] = {}
        for _, t in st.scan("cap_trials"):
            trials.setdefault(t["decl"], []).append(t)
        for did, d in st.scan("cap_decls"):
            ts = trials.get(did, [])
            if d["trials"] != len(ts) or d["trials_ok"] != sum(t["ok"] for t in ts):
                errs.append(f"{did}: trial counters do not match the trial records")
            act = [a for a in st.find("cap_adoptions", "decl", did) if a["status"] == "active"]
            if d["adopters"] != len(act):
                errs.append(f"{did}: adopter counter does not match adoption records")
        for aid, a in st.scan("cap_adoptions"):
            if not any(t["decl"] == a["decl"] and t["ok"] and t["t"] <= a["t"] for t in
                       st.find("cap_trials", "by", a["adopter"])):
                errs.append(f"{aid}: adoption without a successful own execution")
        return errs
