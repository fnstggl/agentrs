"""Autonomous NPC invention (routine `innovator` on person minds).

Entrepreneurs (dynamics.py) imitate: they fork an open-source capability that already sells. Innovators invent.
An innovator is a simulated developer with savings who

1. **identifies a problem** from public evidence only: open problem reports (`ops.requests`) whose verified loss
   kind has a program grammar here, reported by >= 2 independent firms, and not yet solved (no capability with an
   adopted - kept after trial - install whose tags match the problem);
2. **proposes alternatives** by sampling a program from a small grammar of sizing / placement controllers
   (forecast expression x safety margin x floor x ceiling x growth term x placement premium) and generating
   Python source for it;
3. **tests** it - a compile check before deployment, then field trials: firms trial it with their own books
   and the innovator reads the outcome evidence its capability receives (evidence is visible to the provider);
4. **creates artifacts**: an organisation, a repository with the generated code, a rented instance, a deployed
   service and a published capability; it answers the reports it found;
5. **iterates**: when a version's trials come back negative, or nobody installs it within two weeks, it mutates
   the best version's parameters (hill climbing on the evidence it has) and ships a new version that declares it
   `improves` the previous one; old versions are shut down once nobody uses them; it gives up after
   `max_versions` or when it can no longer pay for compute.

Everything is executed through the same endpoints as any agent: nothing about the result is decided here.
`ablate_innovation` disables the routine.
"""
from __future__ import annotations

import copy
import json

from ..core.errors import CrucibleError

DAY = 86400

FORECASTS = {
    "yesterday": "_demand(hrs.get(target - 24))",
    "max_two_days": "max(_demand(hrs.get(target - 24)), 0.9 * _demand(hrs.get(target - 48)))",
    "last_hour": "_demand(hrs.get(now_h))",
    "blend": "0.5 * _demand(hrs.get(now_h)) + 0.5 * _demand(hrs.get(target - 24))",
}
SPACE = {"forecast": sorted(FORECASTS), "margin": [1.0, 1.1, 1.2, 1.35, 1.5], "floor": [1, 2], "ceiling": [8, 16, 32],
         "growth": [False, True], "premium": [1.2, 1.5, 2.0]}
GRAMMAR_KINDS = {"capacity_overload": "sizing", "cost:compute": "sizing", "capacity_unavailable": "placement"}
TAGS = {"capacity_overload": ["capacity", "overload", "scaling", "instance", "demand", "peak"],
        "cost:compute": ["compute", "cost", "idle", "cores", "scaling", "instance", "capacity"],
        "capacity_unavailable": ["capacity", "provider", "full", "unavailable", "placement", "migrate", "peak"]}

HEAD = '''import math
import crucible_sdk as sdk


def _demand(r):
    if not r:
        return 0.0
    served = r["ok"] + r["errors"]
    return r["cu"] * (served + r["overload"]) / served if served else 0.0


def tick(args, caller):
    host = args["host"]
    m = sdk.call("cloud.metrics", host=host, hours=50)
    hrs = {r["hour"]: r for r in m["hours"]}
    now_h = max(hrs)
    target = now_h + 1
    cur = max(1, m["cores"])
    per_core = m["capacity_cu_per_hour"] / cur
    fc = (FORECAST) * MARGIN
'''
GROWTH = '''    last = sum(_demand(hrs.get(h)) for h in range(now_h - 23, now_h + 1))
    prev = sum(_demand(hrs.get(h)) for h in range(now_h - 47, now_h - 23))
    fc = fc * (min(1.5, max(0.8, last / prev)) if prev > 0 else 1.0)
'''
SIZE = '''    need = max(FLOOR, min(CEILING, int(math.ceil(fc / per_core)))) if fc > 0 else cur
'''
PLACE = '''    if need > cur:
        provs = {p["provider"]: p for p in sdk.call("cloud.providers")}
        here = provs.get(m["provider"])
        if here is not None and here["cores_free"] < need - cur:
            opts = sorted((p["price_core_hour_cents"], n) for n, p in provs.items()
                          if n != m["provider"] and p["cores_free"] >= need and not p["in_outage"])
            if opts and opts[0][0] <= PREMIUM * m["price_core_hour_cents"]:
                new = sdk.call("cloud.rent", provider=opts[0][1], cores=need, account=args["account"],
                               owner=m["owner"])["instance"]
                sdk.call("cloud.deploy", repo=m["repo"], host=host, instance=new)
                sdk.call("cloud.release", instance=m["instance"])
                return {"moved_to": opts[0][1], "cores": need}
'''
TAIL = '''    if need != cur:
        try:
            sdk.call("cloud.resize", instance=m["instance"], cores=need)
        except sdk.WorldError:
            return {"cores": cur, "resize_failed": True}
    return {"cores": need}
'''


def _save(soc, ctx, mid, m, cfg) -> None:
    """Persist the routine's state as a *copy*: the emitted payload must never alias objects mutated later
    (that would break the hash chain on replay)."""
    soc.set(ctx, mid, routines=dict(m["routines"], innovator=copy.deepcopy(cfg)))


def program(kind: str, p: dict) -> str:
    """Generate the controller's source from grammar choices."""
    src = HEAD.replace("FORECAST", FORECASTS[p["forecast"]]).replace("MARGIN", repr(float(p["margin"])))
    if p["growth"]:
        src += GROWTH
    src += SIZE.replace("FLOOR", str(int(p["floor"]))).replace("CEILING", str(int(p["ceiling"])))
    if GRAMMAR_KINDS[kind] == "placement":
        src += PLACE.replace("PREMIUM", repr(float(p["premium"])))
    return src + TAIL


def sample(rng) -> dict:
    return {k: v[int(rng.integers(0, len(v)))] for k, v in SPACE.items()}


def mutate(rng, p: dict) -> dict:
    q = dict(p)
    for k in rng.choice(sorted(SPACE), size=2, replace=False):
        opts = SPACE[str(k)]
        i = opts.index(q[str(k)])
        j = int(min(len(opts) - 1, max(0, i + int(rng.choice([-1, 1])))))
        q[str(k)] = opts[j] if j != i else opts[int(rng.integers(0, len(opts)))]
    return q


def _solved(st, kind: str) -> bool:
    tags = set(TAGS.get(kind, []))
    for cid, c in st.scan("capabilities"):
        if c["status"] != "active" or len(tags & set(c["tags"])) < 3:
            continue
        if any(i["status"] == "active" and i["independent"] for i in st.find("cap_installs", "capability", cid)):
            return True
    return False


def _problems(st) -> dict[str, set]:
    out: dict[str, set] = {}
    for _, r in st.scan("ops_requests"):
        if r["status"] != "open":
            continue
        ev = r.get("evidence") or {}
        for k, n in (ev.get("loss_events") or {}).items():
            if n and k in GRAMMAR_KINDS:
                out.setdefault(k, set()).add(r["by"])
        if "cost-compute" in r["tags"]:          # paying for idle cores is reported as a cost, not a loss event
            out.setdefault("cost:compute", set()).add(r["by"])
    return out


def _score(st, cid: str) -> tuple[float, int, int]:
    """(mean z of outcome evidence, outcomes, installs) for one of the innovator's capabilities."""
    zs = [e["data"]["effect_per_day"] / max(1.0, e["data"].get("se", 1.0)) for e in st.find("cap_evidence", "capability", cid)
          if e["kind"] == "outcome"]
    return (sum(zs) / len(zs) if zs else 0.0, len(zs), len(st.find_ids("cap_installs", "capability", cid)))


def innovator_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    if soc.p("ablate_innovation", False) or "caps" not in ctx.world.modules or "ops_requests" not in st.tables:
        return
    cfg = copy.deepcopy(m["routines"].get("innovator", {}))      # never mutate state or emitted payloads
    phase = cfg.get("phase", "scan")
    me = m["owner"]
    bank = ctx.world.modules["bank"]
    if phase == "scan":
        if rng.random() > 0.5:
            return
        probs = {k: v for k, v in _problems(st).items() if len(v) >= 2 and not _solved(st, k)}
        acct = bank.primary_account(st, mid)
        cash = bank.balance(st, acct) if acct else 0
        if not probs or cash < cfg.get("min_cash", 150_000):
            return
        kind = sorted(probs, key=lambda k: (-len(probs[k]), k))[0]
        handle = mid.split(":", 1)[1][:12]
        try:
            org = ctx.call("id.create_org", {"handle": f"{handle}-lab", "name": f"{handle.title()} Lab", "kind": "company"},
                           as_principal=me)["org"]
            ctx.call("biz.register", {"org": org, "account": acct, "sector": "software"}, as_principal=me)
            oacct = ctx.call("bank.open_account", {"owner": org}, as_principal=me)["account"]
            ctx.call("bank.transfer", {"from_account": acct, "to": oacct, "amount": int(cash * 0.7), "memo": "capital"},
                     as_principal=me)
        except CrucibleError as e:
            ctx.emit("society.failed", {"who": me, "tool": "innovator", "error": e.as_dict()}, vis="internal")
            soc.set(ctx, mid, routines=dict(m["routines"], innovator=dict(cfg, phase="failed", error=e.message[:120])))
            return
        cfg.update(phase="build", kind=kind, org=org, acct=oacct, versions=[], params=sample(rng), started=ctx.now)
        _save(soc, ctx, mid, m, cfg)
        soc.remember(ctx, mid, m, "innovating", org, {"problem": kind, "reporters": sorted(probs[kind])})
        m = dict(st.tables["minds"][mid])
        phase = "build"
    if phase == "build":
        _build(soc, ctx, mid, m, cfg, rng)
        return
    if phase == "observe":
        _observe(soc, ctx, mid, m, cfg, rng)


def _build(soc, ctx, mid, m, cfg, rng):
    st = ctx.state
    me, kind, org, acct = m["owner"], cfg["kind"], cfg["org"], cfg["acct"]
    n = len(cfg["versions"]) + 1
    src = program(kind, cfg["params"])
    try:
        compile(src, "service.py", "exec")                     # test 1: the generated program is valid Python
    except SyntaxError as e:
        cfg.update(phase="failed", error=f"generated program invalid: {e}")
        _save(soc, ctx, mid, m, cfg)
        return
    price = int(cfg.get("price", 25))
    man = {"entry": "service.py", "endpoints": {"tick": {"handler": "tick", "price_cents": price,
                                                         "doc": "Hourly controller for your service's capacity.",
                                                         "params": {"type": "object", "additionalProperties": True}}},
           "description": f"{GRAMMAR_KINDS[kind]} controller v{n}"}
    host = f"{mid.split(':', 1)[1][:10]}-{GRAMMAR_KINDS[kind][:5]}{n}"
    placement = GRAMMAR_KINDS[kind] == "placement"
    try:
        if not cfg.get("repo"):
            cfg["repo"] = ctx.call("code.create_repo", {"name": f"{GRAMMAR_KINDS[kind]}-controller", "owner": org},
                                   as_principal=me)["repo"]
        ctx.call("code.commit", {"repo": cfg["repo"], "files": {"crucible.json": json.dumps(man), "service.py": src},
                                 "message": f"v{n}: {json.dumps(cfg['params'], sort_keys=True)}"}, as_principal=me)
        provs = sorted((p["price_core_hour_cents"], p["provider"]) for p in ctx.call("cloud.providers", {}, as_principal=me)
                       if p["cores_free"] >= 1 and not p["in_outage"])
        inst = ctx.call("cloud.rent", {"provider": provs[0][1], "cores": 1, "account": acct, "owner": org},
                        as_principal=me)["instance"]
        d = ctx.call("cloud.deploy", {"repo": cfg["repo"], "host": host, "instance": inst}, as_principal=me)
        if d.get("status") != "live":
            raise CrucibleError("deploy failed")
        prev = cfg["versions"][-1]["capability"] if cfg["versions"] else None
        req = [{"cap": "cloud.manage", "scope": "$host"}, {"cap": "analytics.read", "scope": "$host"}]
        params = {"host": {"kind": "host"}}
        trig_args = {"host": "$host"}
        if placement:
            req += [{"cap": "cloud.deploy", "scope": "$host"}, {"cap": "bank.spend", "scope": "$account",
                                                                "limit": 2_000_000}]
            params["account"] = {"kind": "account"}
            trig_args["account"] = "$account"
        r = ctx.call("caps.publish", {
            "host": host, "title": f"{'Placement-aware' if placement else 'Adaptive'} capacity controller v{n}",
            "description": f"Generated controller ({cfg['params']['forecast']} forecast, margin "
                           f"{cfg['params']['margin']}) that resizes your service every hour"
                           + (" and moves it to a provider with free cores when yours is full" if placement else "")
                           + ". Judge it on your own books.",
            "tags": TAGS[kind], "params": params, "requires": req,
            "triggers": [{"endpoint": "tick", "every_s": 3600, "args": trig_args}],
            **({"improves": prev} if prev else {})}, as_principal=me)
    except CrucibleError as e:
        ctx.emit("society.failed", {"who": me, "tool": "innovator.build", "error": e.as_dict()}, vis="internal")
        cfg.update(phase="failed", error=e.message[:120])
        _save(soc, ctx, mid, m, cfg)
        return
    cap = r["capability"]
    cfg["versions"] = list(cfg["versions"]) + [{"capability": cap, "host": host, "instance": inst,
                                                "params": dict(cfg["params"]), "t": ctx.now}]
    cfg["phase"] = "observe"
    for rid, q in sorted(st.scan("ops_requests")):
        if q["status"] == "open" and (((q.get("evidence") or {}).get("loss_events") or {}).get(kind) or
                                      (kind == "cost:compute" and "cost-compute" in q["tags"])):
            soc.act(ctx, me, "ops.respond", {"request": rid, "capability": cap,
                                             "text": f"Capability {cap} is a new controller for this problem; trial it "
                                                     "on your own books."})
    _save(soc, ctx, mid, m, cfg)
    ctx.emit("note", {"what": "invented", "who": me, "org": org, "capability": cap, "kind": kind, "version": n,
                      "params": dict(cfg["params"]), "improves": prev}, vis="public")


def _observe(soc, ctx, mid, m, cfg, rng):
    st = ctx.state
    me = m["owner"]
    cur = cfg["versions"][-1]
    z, n_out, n_inst = _score(st, cur["capability"])
    adopted = any(i["status"] == "active" and i["independent"] for v in cfg["versions"]
                  for i in st.find("cap_installs", "capability", v["capability"]))
    # shut down versions nobody uses any more (pay for compute only where it earns)
    for v in cfg["versions"][:-1]:
        live = [i for i in st.find("cap_installs", "capability", v["capability"]) if i["status"] in ("active", "trial")]
        if not live and not v.get("closed"):
            inst = st.get("instances", v["instance"])
            if inst and inst["status"] == "running":
                soc.act(ctx, me, "cloud.release", {"instance": v["instance"]})
            v["closed"] = ctx.now
    age = (ctx.now - cur["t"]) / DAY
    bank = ctx.world.modules["bank"]
    cash = bank.balance(st, cfg["acct"])
    stale = (n_out >= 2 and z <= 0) or (n_inst == 0 and age >= 14)
    if adopted or not stale:
        _save(soc, ctx, mid, m, cfg)
        return
    if len(cfg["versions"]) >= int(cfg.get("max_versions", 4)) or cash < 100_000:
        cfg["phase"] = "gave_up"
        _save(soc, ctx, mid, m, cfg)
        soc.remember(ctx, mid, m, "gave_up", cfg["org"], {"versions": len(cfg["versions"])})
        return
    scored = [(_score(st, v["capability"])[0], k) for k, v in enumerate(cfg["versions"])]
    best = cfg["versions"][max(scored)[1]]
    cfg["params"] = mutate(rng, best["params"])
    cfg["phase"] = "build"
    _save(soc, ctx, mid, m, cfg)
    _build(soc, ctx, mid, dict(st.tables["minds"][mid]), cfg, rng)


def register_innovators(world) -> None:
    world.hooks.setdefault("mind_routines", {}).setdefault("innovator", innovator_routine)
