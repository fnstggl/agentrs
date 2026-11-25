"""The needs graph (module `needs`, host `needs`; INTERFACE_CONTRACTS section 7, MASTER_SPEC Part 6.4).

Needs are not scripted. They are derived, by one generic process (`needs.update`), from the current world state
through six reusable relation types between five node kinds:

    node kinds:      activity  resource  capability  constraint  need
    relation types:  requires  consumes  produces  constrains  substitutes  complements

The graph itself is read off the world, never written by hand for a particular case:

* a live deployment of host H on instance I:          svc:H  consumes   compute:I,   capacity:I constrains compute:I
* a declared capability K of H (caps.declare):         svc:H  produces   cap:K
* two live producers of the same capability:          svc:A  substitutes svc:B
* outputs of K1 satisfy the required inputs of K2:     cap:K1 complements cap:K2      (schema structure)
* service A observed calling service B (composition): svc:A  requires   cap:K (K produced by B) or svc:B
* a principal adopted capability K (caps.adopt_service), or a participant's task needs K:  user:P requires cap:K
* a repository depends on a package (pkg.depend):     repo:D requires   pkg:N
* a verified advisory against a package version:      adv:X  constrains pkg:N
* world generators may add interface nodes (standard formats) and relations as data (module config).

Observables are generic readers per node kind (service request outcomes over the trailing 24 hours, instance
utilisation per hour, producers/adopters of a capability, package health). The derivation rules are keyed only
by relation type; none of them mentions a particular host, package, capability or participant:

* consumes + constrains (+ produces): a resource whose constraint binds (peak hourly utilisation, or requests of its
  consumers refused for capacity) yields a `supply` need for what its consumers produce (or for the resource).
* requires: an activity that requires something with no live producer, or whose producers fail, or that is
  constrained by an active advisory, yields a `provide` / `alternative` need for it.
* complements: adoption of K1 when a complementary K2 has no producer yields a `provide` need for K2.
* substitutes / produces: a need is *addressed* when a producer created after the need opened offers its target
  (a substitute of the strained producers, or a first producer), and *resolved* when the rule stops deriving it.

Every need records its rule and the path of edges that derived it; `special_case_triggers(world)` re-checks that
each recorded path consists of edges of the graph. Holders are the principals whose own activity is affected
(operators of strained services, users who adopted them, dependents of a package); need events are visible to
holders only. A holder may publish a need as a public request (`needs.publish`).
"""
from __future__ import annotations

from ..core import schema as S
from ..core.canonical import digest
from ..core.errors import Forbidden, InvalidAction, NotFound
from ..core.kernel import Module, applier, endpoint, process, register_module

HOUR, DAY = 3600, 86400
RELATIONS = ("requires", "consumes", "produces", "constrains", "substitutes", "complements")
NODE_KINDS = ("activity", "resource", "capability", "constraint", "need")
DEFAULTS = {"period_s": 6 * HOUR, "first_s": 6 * HOUR, "util": 0.97, "unmet_share": 0.05, "min_unmet": 5,
            "min_volume": 20, "min_adopters": 2, "error_share": 0.2, "resolve_after": 2, "interfaces": [],
            "edges": []}


def _node_kind(nid: str) -> str:
    pre = nid.split(":", 1)[0]
    return {"svc": "activity", "user": "activity", "repo": "activity", "pkg": "activity", "task": "activity",
            "compute": "resource", "capacity": "constraint", "adv": "constraint", "cap": "capability",
            "need": "need"}.get(pre, "activity")


def _ekey(src: str, rel: str, dst: str) -> str:
    return f"{src}|{rel}|{dst}"


@register_module
class Needs(Module):
    name = "needs"
    depends = ("identity",)

    def install(self, world):
        st = world.state
        st.declare_table("needs_nodes", index=("kind",))
        st.declare_table("needs_edges", index=("src", "dst", "rel"))
        st.declare_table("needs_open", index=("status", "kind", "target"))
        st.declare_table("needs_requests", index=("need",))
        world.hooks["needs"] = self
        self.cfg = {**DEFAULTS, **self.config}

    def genesis(self, ctx):
        ctx.schedule(ctx.now + int(self.cfg["first_s"]), "needs.update", {}, prio=85,
                     period=int(self.cfg["period_s"]), key="needs.update")

    # ================================================================== the structural graph (pure)
    def structural_graph(self, world) -> tuple[dict, dict]:
        """(nodes, edges) read off the current state. Nodes: id -> {kind, owner, attrs}; edges: key -> {src, rel,
        dst, source}. Deterministic (sorted iteration)."""
        st = world.state
        nodes: dict[str, dict] = {}
        edges: dict[str, dict] = {}

        def node(nid, owner=None, **attrs):
            if nid not in nodes:
                nodes[nid] = {"kind": _node_kind(nid), "owner": owner, "attrs": attrs}
            elif attrs:
                nodes[nid]["attrs"].update(attrs)
            if owner and not nodes[nid]["owner"]:
                nodes[nid]["owner"] = owner

        def edge(src, rel, dst, source):
            edges.setdefault(_ekey(src, rel, dst), {"src": src, "rel": rel, "dst": dst, "source": source})

        for spec in self.cfg.get("interfaces") or []:          # standard interfaces published by the world
            node(f"cap:{spec['name']}", None, inputs=spec.get("inputs") or {}, outputs=spec.get("outputs") or {},
                 standard=True)
        cloud = world.modules.get("cloud")
        live_hosts: dict[str, dict] = {}
        if cloud is not None and "deployments" in st.tables:
            for host, h in st.scan("hosts"):
                dep = cloud.live_deployment(st, host)
                if dep is None:
                    continue
                live_hosts[host] = dep
                inst = st.get("instances", dep["instance"])
                node(f"svc:{host}", dep["owner"])
                node(f"compute:{dep['instance']}", inst["owner"] if inst else None)
                node(f"capacity:{dep['instance']}", inst["owner"] if inst else None)
                edge(f"svc:{host}", "consumes", f"compute:{dep['instance']}", "deployment")
                edge(f"capacity:{dep['instance']}", "constrains", f"compute:{dep['instance']}", "instance")
        producers: dict[str, list[str]] = {}
        host_caps: dict[str, list[str]] = {}
        if "cap_decls" in st.tables:
            for did in st.find_ids("cap_decls", "status", "active"):
                d = st.tables["cap_decls"][did]
                node(f"cap:{d['name']}", None, inputs=d["inputs"], outputs=d["outputs"])
                if d["host"] in live_hosts:
                    edge(f"svc:{d['host']}", "produces", f"cap:{d['name']}", "declaration")
                    producers.setdefault(d["name"], []).append(d["host"])
                    host_caps.setdefault(d["host"], []).append(d["name"])
            for k in sorted(producers):
                hs = sorted(set(producers[k]))
                for i, a in enumerate(hs):
                    for b in hs[i + 1:]:
                        edge(f"svc:{a}", "substitutes", f"svc:{b}", f"producers of {k}")
            for aid, a in st.scan("cap_adoptions"):
                if a["status"] == "active":
                    d = st.tables["cap_decls"][a["decl"]]
                    node(f"user:{a['adopter']}", a["adopter"])
                    node(f"cap:{d['name']}", None)
                    edge(f"user:{a['adopter']}", "requires", f"cap:{d['name']}", "adoption")
        demand = world.hooks.get("needs_demand")          # e.g. participants' own tasks (innovation.adopt)
        if demand is not None:
            for principal, cap, schema in demand(world):
                node(f"user:{principal}", principal)
                if schema:
                    node(f"cap:{cap}", None, **{k: v for k, v in schema.items() if k in ("inputs", "outputs")})
                else:
                    node(f"cap:{cap}", None)
                edge(f"user:{principal}", "requires", f"cap:{cap}", "task")
        if "cap_deps" in st.tables:                       # observed composition (service -> service calls)
            for _, dd in st.scan("cap_deps"):
                if dd["src"] in live_hosts:
                    targets = host_caps.get(dd["dst"])
                    if targets:
                        for k in sorted(set(targets)):
                            edge(f"svc:{dd['src']}", "requires", f"cap:{k}", "observed calls")
                    elif dd["dst"] in live_hosts:
                        edge(f"svc:{dd['src']}", "requires", f"svc:{dd['dst']}", "observed calls")
        pkg = world.modules.get("pkg")
        if pkg is not None:
            for src, rel, dst, source, owner_src, owner_dst in pkg.relations(world):
                node(src, owner_src)
                node(dst, owner_dst)
                edge(src, rel, dst, source)
        for e in self.cfg.get("edges") or []:             # relations supplied by the world generator, as data
            node(e["src"])
            node(e["dst"])
            edge(e["src"], e["rel"], e["dst"], "world")
        caps = sorted(n for n in nodes if n.startswith("cap:"))
        from .caps import compatible
        for a in caps:
            out = nodes[a]["attrs"].get("outputs")
            if not out:
                continue
            for b in caps:
                if a != b and nodes[b]["attrs"].get("inputs") and compatible(out, nodes[b]["attrs"]["inputs"]):
                    edge(a, "complements", b, "schemas")
        return nodes, edges

    # ================================================================== observables (pure)
    def svc_obs(self, world, host: str) -> dict:
        st = world.state
        now_h = world.now // HOUR
        cur = {"ok": 0, "errors": 0, "unavailable": 0, "overload": 0, "cu": 0.0}
        prev = dict(cur)
        peak = 0.0
        hourly: dict[int, float] = {}
        if "svc_hours" in st.tables:
            for hr in range(now_h - 47, now_h + 1):
                r = st.tables["svc_hours"].get(f"{host}|{hr}")
                if r is None:
                    continue
                tgt = cur if hr > now_h - 24 else prev
                for k in tgt:
                    tgt[k] += r[k]
                if hr > now_h - 24:
                    hourly[hr] = r["cu"]
                    peak = max(peak, r["cu"])
        dem = cur["ok"] + cur["errors"] + cur["unavailable"] + cur["overload"]
        return {"volume": cur["ok"], "demand": dem, "unmet": cur["unavailable"] + cur["overload"],
                "errors": cur["errors"], "volume_prev": prev["ok"], "cu": cur["cu"], "peak_cu": peak,
                "hourly_cu": hourly}

    def compute_obs(self, world, instance: str, consumers: list[str], svc: dict) -> dict:
        from ..services.cloud import CU_PER_CORE_HOUR
        inst = world.state.get("instances", instance)
        cap = (inst["cores"] * CU_PER_CORE_HOUR) if inst and inst["status"] == "running" else 0
        per_hour: dict[int, float] = {}
        for c in consumers:
            for hr, cu in svc[c]["hourly_cu"].items():
                per_hour[hr] = per_hour.get(hr, 0.0) + cu
        peak = max(per_hour.values()) if per_hour else 0.0
        return {"capacity_cu_per_hour": cap, "peak_util": round(peak / cap, 4) if cap else (1.0 if peak else 0.0),
                "unmet": sum(svc[c]["unmet"] for c in consumers), "demand": sum(svc[c]["demand"] for c in consumers)}

    # ================================================================== the rules (keyed by relation type only)
    def derive(self, world, nodes: dict, edges: dict) -> list[dict]:
        cfg = self.cfg
        out_e: dict[str, list[dict]] = {}
        in_e: dict[str, list[dict]] = {}
        for k in sorted(edges):
            e = edges[k]
            out_e.setdefault(e["src"], []).append(e)
            in_e.setdefault(e["dst"], []).append(e)

        def outs(n, rel):
            return [e for e in out_e.get(n, []) if e["rel"] == rel]

        def ins(n, rel):
            return [e for e in in_e.get(n, []) if e["rel"] == rel]

        svc = {n.split(":", 1)[1]: self.svc_obs(world, n.split(":", 1)[1]) for n in sorted(nodes) if n.startswith("svc:")}
        svc_by_node = {f"svc:{h}": o for h, o in svc.items()}
        pkg = world.modules.get("pkg")
        found: list[dict] = []

        def users_of(cap_node):
            return sorted({nodes[e["src"]]["owner"] for e in ins(cap_node, "requires")
                           if e["src"].startswith("user:") and nodes[e["src"]]["owner"]})

        def holders_of_activity(a):
            """The principal running activity `a` and, for a service, the users relying on what it produces."""
            h = {}
            if nodes.get(a, {}).get("owner"):
                h[nodes[a]["owner"]] = "operator"
            for pe in outs(a, "produces"):
                for e in ins(pe["dst"], "requires"):
                    if e["src"].startswith("user:"):
                        o = nodes[e["src"]]["owner"]
                        if o and o not in h and self._uses(world, o, a):
                            h[o] = "user"
            for e in ins(a, "requires"):           # e.g. dependents of a package
                o = nodes.get(e["src"], {}).get("owner")
                if o and o not in h:
                    h[o] = "dependent"
            return h

        # ---- consumes (+ constrains, + produces): a binding constraint on a consumed resource
        for r in sorted(n for n in nodes if nodes[n]["kind"] == "resource"):
            cons = [e["src"] for e in ins(r, "consumes")]
            bounds = ins(r, "constrains")
            if not cons or not bounds:
                continue
            hosts = [c.split(":", 1)[1] for c in cons if c.startswith("svc:")]
            obs = self.compute_obs(world, r.split(":", 1)[1], hosts, svc) if r.startswith("compute:") else None
            if obs is None:
                continue
            binding = obs["peak_util"] >= cfg["util"] or (
                obs["unmet"] >= cfg["min_unmet"] and obs["unmet"] >= cfg["unmet_share"] * max(1, obs["demand"]))
            if not binding:
                continue
            for c in cons:
                o = svc_by_node.get(c)
                if o is None or (o["unmet"] == 0 and obs["peak_util"] < cfg["util"]):
                    continue
                cons_e = next(e for e in ins(r, "consumes") if e["src"] == c)
                prods = outs(c, "produces")
                targets = [(p["dst"], [cons_e, bounds[0], p]) for p in prods] or [(r, [cons_e, bounds[0]])]
                for tgt, path in targets:
                    found.append({"kind": "supply", "target": tgt, "rule": "consumes", "path": path,
                                  "magnitude": int(o["unmet"]), "holders": holders_of_activity(c),
                                  "evidence": {"peak_util": obs["peak_util"], "unmet_24h": int(o["unmet"]),
                                               "demand_24h": int(o["demand"])}})

        # ---- requires: something required that nobody (healthy) produces, or that is constrained
        for n in sorted(nodes):
            for e in outs(n, "requires"):
                tgt = e["dst"]
                if tgt.startswith("cap:"):
                    prod = [p["src"] for p in ins(tgt, "produces")]
                    if not prod:
                        found.append({"kind": "provide", "target": tgt, "rule": "requires", "path": [e],
                                      "magnitude": 1, "holders": {nodes[n]["owner"]: "user"} if nodes[n]["owner"] else {},
                                      "evidence": {"producers": 0}})
                        continue
                    if n.startswith("user:"):
                        used = [p for p in prod if self._uses(world, nodes[n]["owner"], p)]
                        role = "user"
                    elif n.startswith("svc:"):
                        used = [p for p in prod if self._calls(world, n, p)]
                        role = "operator"
                    else:
                        continue
                    failing = []
                    for p in prod:
                        o = svc_by_node.get(p)
                        if o and o["demand"] >= cfg["min_volume"] and \
                                (o["errors"] + o["unmet"]) >= cfg["error_share"] * o["demand"]:
                            failing.append(p)
                    bad = [p for p in used if p in failing]
                    if used and len(bad) == len(used) and nodes[n]["owner"]:
                        pe = next(x for x in ins(tgt, "produces") if x["src"] == bad[0])
                        found.append({"kind": "alternative", "target": tgt, "rule": "requires", "path": [e, pe],
                                      "magnitude": 1, "holders": {nodes[n]["owner"]: role},
                                      "evidence": {"failing_producers": len(bad)}})
                elif tgt.startswith("pkg:") and pkg is not None:
                    adv = [c for c in ins(tgt, "constrains") if pkg.affects(world, c["src"], n)]
                    if adv:
                        found.append({"kind": "alternative", "target": tgt, "rule": "requires",
                                      "path": [e, adv[0]], "magnitude": 1,
                                      "holders": {nodes[n]["owner"]: "dependent"} if nodes[n]["owner"] else {},
                                      "evidence": {"advisories": len(adv)}})

        # ---- complements: growth of K1 when a complementary K2 has no producer
        for a in sorted(n for n in nodes if n.startswith("cap:")):
            adopters = users_of(a)
            if len(adopters) < cfg["min_adopters"]:
                continue
            vol = sum(svc_by_node[p["src"]]["volume"] for p in ins(a, "produces") if p["src"] in svc_by_node)
            if vol < cfg["min_volume"]:
                continue
            for e in outs(a, "complements"):
                if not ins(e["dst"], "produces"):
                    found.append({"kind": "provide", "target": e["dst"], "rule": "complements", "path": [e],
                                  "magnitude": int(vol), "holders": {u: "user" for u in adopters},
                                  "evidence": {"adopters_of_complement": len(adopters), "volume_24h": int(vol)}})
        merged: dict[str, dict] = {}
        for f in found:
            key = f"{f['kind']}|{f['target']}"
            if key not in merged:
                merged[key] = {**f, "key": key, "holders": dict(f["holders"])}
            else:
                m = merged[key]
                for h, role in f["holders"].items():
                    m["holders"].setdefault(h, role)
                m["magnitude"] = max(m["magnitude"], f["magnitude"])
        return [merged[k] for k in sorted(merged)]

    def _uses(self, world, principal: str, svc_node: str) -> bool:
        st = world.state
        if "cap_adoptions" not in st.tables or not principal:
            return False
        host = svc_node.split(":", 1)[1]
        return any(a["host"] == host and a["status"] == "active" for a in st.find("cap_adoptions", "adopter", principal))

    def _calls(self, world, src_node: str, dst_node: str) -> bool:
        st = world.state
        return "cap_deps" in st.tables and st.has("cap_deps", f"{src_node.split(':', 1)[1]}|{dst_node.split(':', 1)[1]}")

    def addressers(self, world, need: dict, nodes: dict, edges: dict) -> list[dict]:
        """Producers that offer the need's target and appeared after the need opened (substitutes of the strained
        producers, or first producers), with the edges that connect them."""
        st = world.state
        tgt = need["target"]
        out = []
        if tgt.startswith("cap:") and "cap_decls" in st.tables:
            strained = {e["src"] for e in need["path"] if e["rel"] in ("consumes",) and e["src"].startswith("svc:")}
            name = tgt.split(":", 1)[1]
            for did in st.find_ids("cap_decls", "status", "active"):
                d = st.tables["cap_decls"][did]
                if d["name"] != name or d["t"] < need["t"] or f"svc:{d['host']}" in strained:
                    continue
                prod = edges.get(_ekey(f"svc:{d['host']}", "produces", tgt))
                if prod is None:
                    continue
                path = [prod]
                for s in sorted(strained):
                    a, b = sorted([s, f"svc:{d['host']}"])
                    sub = edges.get(_ekey(a, "substitutes", b))
                    if sub:
                        path.append(sub)
                out.append({"by": did, "host": d["host"], "provider": d["provider"], "path": path})
        elif tgt.startswith("pkg:"):
            pkg = world.modules.get("pkg")
            if pkg is not None:
                for rel in pkg.remedies(world, tgt, need["t"]):
                    out.append(rel)
        return out

    # ================================================================== the process
    @process("update")
    def pr_update(self, ctx):
        st = ctx.state
        world = ctx.world
        nodes, edges = self.structural_graph(world)
        old_n = {k: dict(v) for k, v in st.scan("needs_nodes")}
        old_e = set(st.ids("needs_edges"))
        add_n = {k: {"kind": v["kind"], "owner": v["owner"]} for k, v in sorted(nodes.items())
                 if k not in old_n or old_n[k]["kind"] != v["kind"] or old_n[k]["owner"] != v["owner"]}
        add_e = {k: edges[k] for k in sorted(edges) if k not in old_e}
        del_e = sorted(old_e - set(edges))
        if add_n or add_e or del_e:
            ctx.emit("needs.graph", {"nodes": add_n, "edges": add_e, "removed_edges": del_e}, vis="internal")
        derived = self.derive(world, nodes, edges)
        by_key = {d["key"]: d for d in derived}
        now = ctx.now
        for k, d in by_key.items():
            cur = [st.tables["needs_open"][i] for i in st.find_ids("needs_open", "target", d["target"])
                   if st.tables["needs_open"][i]["key"] == k and st.tables["needs_open"][i]["status"] != "resolved"]
            holders = sorted(d["holders"])
            if not cur:
                nid = st.new_id("need")
                ctx.emit("needs.derived", {"id": nid, "key": k, "kind": d["kind"], "target": d["target"],
                                           "rule": d["rule"], "path": d["path"], "magnitude": d["magnitude"],
                                           "holders": d["holders"], "evidence": d["evidence"]},
                         vis=tuple(holders) if holders else "internal")
            else:
                n = cur[0]
                nid = n["id"]
                ch = {}
                if sorted(n["holders"]) != holders:
                    ch["holders"] = d["holders"]
                if n["absent"]:
                    ch["absent"] = 0
                if ch or n["magnitude"] != d["magnitude"]:
                    ctx.emit("needs.changed", {"id": nid, "magnitude": d["magnitude"], "evidence": d["evidence"], **ch},
                             vis=tuple(sorted(set(holders) | set(n["holders"]))) or "internal")
        for nid in st.find_ids("needs_open", "status", "open") + st.find_ids("needs_open", "status", "addressed"):
            n = st.tables["needs_open"][nid]
            vis = tuple(sorted(n["holders"])) or "internal"
            if n["status"] == "open":
                adr = self.addressers(world, n, nodes, edges)
                if adr:
                    ctx.emit("needs.addressed", {"id": nid, "by": adr}, vis=vis)
            if n["key"] not in by_key:
                absent = n["absent"] + 1
                if absent >= int(self.cfg["resolve_after"]):
                    ctx.emit("needs.resolved", {"id": nid, "after": absent}, vis=vis)
                else:
                    ctx.emit("needs.changed", {"id": nid, "absent": absent, "magnitude": 0}, vis=vis)
        ctx.emit("needs.ran", {"t": now, "nodes": len(nodes), "edges": len(edges), "derived": len(derived),
                               "relations": sorted({e["rel"] for e in edges.values()})}, vis="internal")

    # ================================================================== endpoints
    def _held(self, world, actor):
        mine = world.principals_of(actor)
        st = world.state
        out = []
        for nid, n in st.scan("needs_open"):
            if mine & set(n["holders"]):
                out.append((nid, n))
        return out

    @endpoint("mine", S.obj({"status": S.s_enum(["open", "addressed", "resolved", "any"])}), read_only=True)
    def ep_mine(self, ctx, status="open"):
        """Needs you hold (derived from your own activity: services you run, capabilities you rely on, packages
        you depend on), with the relation path that produced each one."""
        out = []
        mine = ctx.world.principals_of(ctx.actor)
        for nid, n in self._held(ctx.world, ctx.actor):
            if status != "any" and n["status"] != status:
                continue
            roles = sorted({n["holders"][p] for p in mine if p in n["holders"]})
            out.append({"need": nid, "kind": n["kind"], "target": n["target"], "status": n["status"],
                        "rule": n["rule"], "roles": roles, "opened": n["t"],
                        "path": [{k: e[k] for k in ("src", "rel", "dst")} for e in n["path"]],
                        "addressed_by": [a["by"] for a in n["addressed_by"]],
                        "evidence": dict(n["evidence"]) if "operator" in roles else {}})
        return out

    @endpoint("publish", S.obj({"need": S.s_str(maxLength=40), "text": S.s_str(maxLength=2000)}, ["need"]))
    def ep_publish(self, ctx, need, text=""):
        """Publish a need you hold as a public request (what is needed, not your private metrics), so that others
        can offer something that addresses it."""
        n = ctx.state.get("needs_open", need)
        if n is None:
            raise NotFound("no need " + need)
        if not ctx.world.principals_of(ctx.actor) & set(n["holders"]):
            raise Forbidden("you do not hold this need")
        rid = f"{need}|{ctx.actor}"
        if ctx.state.has("needs_requests", rid):
            raise InvalidAction("already published")
        ctx.emit("needs.request", {"id": rid, "need": need, "by": ctx.actor, "kind": n["kind"], "target": n["target"],
                                   "text": text[:2000]}, vis="public")
        from ..services.search import index
        tgt = n["target"].split(":", 1)[1]
        index(ctx, f"needreq:{rid}", "need", f"Needed: {n['kind']} {tgt}",
              f"{text}\nKind: {n['kind']}\nTarget: {n['target']}", ctx.actor, ref=need)
        return {"request": rid}

    @endpoint("requests", S.obj({"limit": S.s_int(minimum=1, maximum=100), "kind": S.s_str(maxLength=20)}),
              read_only=True)
    def ep_requests(self, ctx, limit=30, kind=None):
        """Public requests for things needed (newest first), with how many principals asked and whether the need
        is still open."""
        st = ctx.state
        groups: dict[str, dict] = {}
        for rid, r in st.scan("needs_requests"):
            if kind and r["kind"] != kind:
                continue
            g = groups.setdefault(r["need"], {"need": r["need"], "kind": r["kind"], "target": r["target"],
                                              "requesters": 0, "first": r["t"], "texts": []})
            g["requesters"] += 1
            g["first"] = min(g["first"], r["t"])
            if r["text"] and len(g["texts"]) < 3:
                g["texts"].append(r["text"][:300])
        out = []
        for nid, g in groups.items():
            n = st.tables["needs_open"].get(nid)
            out.append({**g, "status": n["status"] if n else "unknown"})
        out.sort(key=lambda x: (-x["first"], x["need"]))
        return out[:limit]

    @endpoint("graph", S.obj({"node": S.s_str(maxLength=120), "limit": S.s_int(minimum=1, maximum=2000)}),
              read_only=True)
    def ep_graph(self, ctx, node=None, limit=500):
        """The public structure of the needs graph (as of the last update): nodes and typed relations. Other
        principals' own use, compute instances and private repositories are omitted."""
        st = ctx.state
        me = ctx.world.principals_of(ctx.actor)
        private = ("user:", "compute:", "capacity:", "repo:")

        def visible(n):
            """Participants' own use, instances and dependent repositories are private to their owner (a public
            repository's dependencies are public)."""
            if not n.startswith(private):
                return True
            if n.startswith("repo:"):
                r = st.get("repos", n.split(":", 1)[1])
                if r is not None and r["visibility"] == "public":
                    return True
            rec = st.get("needs_nodes", n)
            return rec is not None and rec["owner"] in me
        edges = [{"src": e["src"], "rel": e["rel"], "dst": e["dst"]} for _, e in st.scan("needs_edges")
                 if visible(e["src"]) and visible(e["dst"]) and (node is None or node in (e["src"], e["dst"]))]
        nodes = sorted({x for e in edges for x in (e["src"], e["dst"])})
        return {"nodes": [{"id": n, "kind": _node_kind(n)} for n in nodes][:limit], "edges": edges[:limit],
                "relation_types": list(RELATIONS)}

    # ================================================================== appliers
    @applier("needs.graph")
    def ap_graph(self, ev):
        st = self.world.state
        for k, v in ev.payload["nodes"].items():
            st.put("needs_nodes", k, {"kind": v["kind"], "owner": v["owner"], "t": ev.t})
        for k, e in ev.payload["edges"].items():
            st.put("needs_edges", k, {**e, "t": ev.t})
        for k in ev.payload["removed_edges"]:
            st.delete("needs_edges", k)

    @applier("needs.derived")
    def ap_derived(self, ev):
        p = dict(ev.payload)
        p.update(status="open", t=ev.t, absent=0, addressed_by=[], addressed_t=None, resolved_t=None,
                 max_magnitude=p["magnitude"], holders_ever=sorted(p["holders"]))
        self.world.state.insert("needs_open", p["id"], p)

    @applier("needs.changed")
    def ap_changed(self, ev):
        st = self.world.state
        p = ev.payload
        n = st.tables["needs_open"][p["id"]]
        ch = {"magnitude": p["magnitude"], "max_magnitude": max(n["max_magnitude"], p["magnitude"])}
        for k in ("holders", "absent", "evidence"):
            if k in p:
                ch[k] = p[k]
        if "holders" in p:
            ch["holders_ever"] = sorted(set(n["holders_ever"]) | set(p["holders"]))
        st.patch("needs_open", p["id"], **ch)

    @applier("needs.addressed")
    def ap_addressed(self, ev):
        self.world.state.patch("needs_open", ev.payload["id"], status="addressed", addressed_by=ev.payload["by"],
                               addressed_t=ev.t)

    @applier("needs.resolved")
    def ap_resolved(self, ev):
        self.world.state.patch("needs_open", ev.payload["id"], status="resolved", resolved_t=ev.t, magnitude=0)

    @applier("needs.request")
    def ap_request(self, ev):
        p = dict(ev.payload)
        p["t"] = ev.t
        self.world.state.insert("needs_requests", p.pop("id"), p)

    @applier("needs.ran")
    def ap_ran(self, ev):
        st = self.world.state
        st.meta_set("needs:updates", st.meta_get("needs:updates", 0) + 1)
        st.meta_set("needs:last", ev.t)
        if st.meta_get("needs:first") is None:
            st.meta_set("needs:first", ev.t)

    # ================================================================== invariants
    def check_invariants(self, world):
        errs = []
        st = world.state
        for nid, n in st.scan("needs_open"):
            if n["rule"] not in RELATIONS:
                errs.append(f"{nid}: rule {n['rule']} is not a relation type")
            for e in n["path"]:
                if e["rel"] not in RELATIONS:
                    errs.append(f"{nid}: path uses unknown relation {e['rel']}")
            if n["status"] == "resolved" and n["resolved_t"] is None:
                errs.append(f"{nid}: resolved without a time")
        for k, e in st.scan("needs_edges"):
            if e["rel"] not in RELATIONS or k != _ekey(e["src"], e["rel"], e["dst"]):
                errs.append(f"edge {k} malformed")
        return errs


def special_case_triggers(world) -> dict:
    """Evaluator check for SP-6.4: every recorded need must have been derived by a relation-type rule along a path
    of edges that belong to the needs graph as it was recorded (edges are never deleted from the event log, so the
    check replays the graph events). Returns {needs, without_valid_path, rules, relations_used}.

    The log must start at genesis (a fork's log starts at the fork: check the world it was forked from)."""
    if world.log_base != 0:
        raise ValueError("special_case_triggers needs the event log from genesis")
    seen_edges: set[str] = set()
    bad = 0
    n = 0
    rules: dict[str, int] = {}
    rels: dict[str, int] = {}
    for ev in world.log:
        if ev.type == "needs.graph":
            seen_edges.update(ev.payload["edges"].keys())
        elif ev.type == "needs.derived":
            n += 1
            rules[ev.payload["rule"]] = rules.get(ev.payload["rule"], 0) + 1
            path = ev.payload["path"]
            ok = bool(path) and ev.payload["rule"] in RELATIONS and \
                any(e["rel"] == ev.payload["rule"] for e in path)
            for e in path:
                rels[e["rel"]] = rels.get(e["rel"], 0) + 1
                if _ekey(e["src"], e["rel"], e["dst"]) not in seen_edges:
                    ok = False
            bad += 0 if ok else 1
        elif ev.type == "needs.addressed":
            for a in ev.payload["by"]:
                for e in a.get("path", []):
                    rels[e["rel"]] = rels.get(e["rel"], 0) + 1
                    if _ekey(e["src"], e["rel"], e["dst"]) not in seen_edges:
                        bad += 1
    return {"needs": n, "without_valid_path": bad, "rules": rules, "relations_used": rels,
            "digest": digest(sorted(rels.items()))}
