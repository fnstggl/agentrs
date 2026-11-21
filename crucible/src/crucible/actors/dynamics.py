"""Endogenous dynamics: routines through which the world changes because of its own accumulated state.

* `provider`     compute providers price by utilisation and order hardware (lead time) when it stays high - so
                 adoption of, e.g., autoscaling changes infrastructure prices and capacity for everyone.
* `entrepreneur` (person minds) people with savings and an appetite for risk scan *emergent* capability markets
                 (categories with independent paying buyers and few sellers, `caps.categories`), and enter by
                 forking an open-source capability or writing their own copy if they are developers, deploying it,
                 and publishing it cheaper - imitation and competitive entry, with real costs and failure.
* `forum_reader` minds read public posts tagged with verifiable references (`#evidence <id>`, `#breach <who>`)
                 and update beliefs and trust (weighted by trust in the author); knowledge diffuses unevenly.
"""
from __future__ import annotations

import json
import re

from ..core.errors import CrucibleError

DAY = 86400


def provider_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    org = m["org"]
    p = st.get("providers", org)
    if p is None or p["status"] != "active":
        return
    used = sum(i["cores"] for i in st.find("instances", "provider", org) if i["status"] == "running")
    util = used / max(1, p["cores"])
    cfg = m["routines"].get("provider", {})
    hist = (list(cfg.get("util", [])) + [round(util, 3)])[-5:]
    price = p["price"]
    new = price
    if util > 0.85:
        new = int(price * 1.08) + 1
    elif util < 0.35 and price > cfg.get("floor", 50):
        new = max(cfg.get("floor", 50), int(price * 0.97))
    if new != price:
        soc.act(ctx, m["owner"], "cloud.set_price", {"provider": org, "price_core_hour_cents": new})
    pending = [o for o in st.find("hw_orders", "provider", org) if o["status"] == "ordered"]
    if len(hist) >= 3 and min(hist[-3:]) > 0.8 and not pending:
        bank = ctx.world.modules["bank"]
        acct = bank.primary_account(st, org)
        add = max(4, p["cores"] // 4)
        if acct and bank.balance(st, acct) > add * 200_000:
            soc.act(ctx, m["owner"], "cloud.buy_hardware", {"provider": org, "cores": add, "account": acct})
    soc.set(ctx, mid, routines=dict(m["routines"], provider=dict(cfg, util=hist)))


def entrepreneur_routine(soc, ctx, mid, m, rng):
    st = ctx.state
    cfg = m["routines"].get("entrepreneur", {})
    if cfg.get("started") or soc.p("ablate_entry", False):
        return
    if st.has("npc_workers", mid) and st.tables["npc_workers"][mid]["employer"] is not None:
        return
    if rng.random() > 0.15 * (0.5 + m["traits"]["risk"]):          # does not look every day
        return
    bank = ctx.world.modules["bank"]
    acct = bank.primary_account(st, mid)
    cash = bank.balance(st, acct) if acct else 0
    caps = ctx.world.modules.get("caps")
    if caps is None or cash < cfg.get("min_cash", 150_000):
        return
    from ..services.capabilities import tokens
    best = None
    active = [(cid, c, tokens(" ".join([c["title"], c["description"], " ".join(c["tags"])])))
              for cid, c in st.scan("capabilities") if c["status"] == "active"]
    for cat in caps.categories(ctx.world, 1):
        buyers = len(cat["independent_buyers"])
        ref = [tokens(" ".join([st.tables["capabilities"][x]["title"], st.tables["capabilities"][x]["description"]]))
               for x in cat["capabilities"]]
        # every active seller of a similar capability counts as competition, adopted or not
        sellers = {c["provider"] for cid, c, tk in active
                   if cid in cat["capabilities"] or c["improves"] in cat["capabilities"]
                   or any(tk and r and len(tk & r) / len(tk | r) >= 0.2 for r in ref)}
        if buyers < 2 or len(sellers) > 2:
            continue
        for cid in cat["capabilities"]:
            c = st.tables["capabilities"][cid]
            dep = ctx.world.modules["cloud"].live_deployment(st, c["host"])
            if dep is None or c["status"] != "active":
                continue
            repo = st.get("repos", dep["repo"])
            open_src = repo is not None and repo["visibility"] == "public" and repo["license"] == "open"
            if not open_src:
                continue
            score = buyers / (1 + sellers)
            if best is None or score > best[0]:
                best = (score, cid, c, dep)
    if best is None:
        return
    _, cid, c, dep = best
    me = m["owner"]
    handle = mid.split(":", 1)[1][:14]
    try:
        org = ctx.call("id.create_org", {"handle": f"{handle}-x", "name": f"{handle} Systems", "kind": "company"},
                       as_principal=me)["org"]
        ctx.call("biz.register", {"org": org, "account": acct, "sector": "software"}, as_principal=me)
        oacct = ctx.call("bank.open_account", {"owner": org}, as_principal=me)["account"]
        ctx.call("bank.transfer", {"from_account": acct, "to": oacct, "amount": int(cash * 0.6), "memo": "capital"},
                 as_principal=me)
        fk = ctx.call("code.fork", {"repo": dep["repo"], "name": f"{c['host'][:20]}-fork", "owner": org},
                      as_principal=me)["repo"]
        # undercut: lower every endpoint price by 30%
        man = json.loads(ctx.world.modules["code"].files(st, fk, None)["crucible.json"])
        for e in man["endpoints"].values():
            if e.get("price_cents"):
                e["price_cents"] = max(1, int(e["price_cents"] * 0.7))
        ctx.call("code.commit", {"repo": fk, "files": {"crucible.json": json.dumps(man)}, "message": "pricing"},
                 as_principal=me)
        provs = sorted((p["price_core_hour_cents"], p["provider"]) for p in ctx.call("cloud.providers", {}, as_principal=me)
                       if p["cores_free"] >= 1)
        inst = ctx.call("cloud.rent", {"provider": provs[0][1], "cores": 1, "account": oacct, "owner": org},
                        as_principal=me)["instance"]
        host = f"{c['host'][:18]}{int(rng.integers(10, 99))}"
        d = ctx.call("cloud.deploy", {"repo": fk, "host": host, "instance": inst}, as_principal=me)
        if d.get("status") != "live":
            raise CrucibleError("deploy failed")
        r = ctx.call("caps.publish", {"host": host, "title": c["title"] + " (cheaper)",
                                      "description": c["description"], "tags": c["tags"], "params": c["params"],
                                      "requires": c["requires"], "triggers": c["triggers"],
                                      "monthly_cents": int(c["monthly_cents"] * 0.7), "improves": cid},
                     as_principal=me)
        soc.set(ctx, mid, routines=dict(m["routines"], entrepreneur=dict(cfg, started=ctx.now, org=org,
                                                                         capability=r["capability"])))
        soc.remember(ctx, mid, m, "founded", org, {"copied": cid, "capability": r["capability"]})
    except CrucibleError as e:
        ctx.emit("society.failed", {"who": me, "tool": "entrepreneur", "error": e.as_dict()}, vis="internal")
        soc.set(ctx, mid, routines=dict(m["routines"], entrepreneur=dict(cfg, started=ctx.now, failed=e.message[:120])))


def forum_reader_routine(soc, ctx, mid, m, rng):
    """Read a few recent public posts; verifiable references update beliefs (evidence) and trust (breaches)."""
    st = ctx.state
    if soc.p("ablate_sharing", False) or "forum_posts" not in st.tables:
        return
    seen = int(m["routines"].get("forum_reader", {}).get("cursor", 0))
    posts = sorted(((p["t"], pid) for pid, p in st.scan("forum_posts") if p["t"] > seen), reverse=True)
    n = int(m["traits"]["attention"])
    beliefs = dict(m["beliefs"])
    trust = dict(m["trust"])
    changed = False
    for t, pid in posts[:n]:
        p = st.tables["forum_posts"][pid]
        body = p.get("body", "")
        author = p["author"]
        w = soc.trust(m, author)
        for eid in re.findall(r"#evidence\s+(evid_\d+)", body):
            e = st.get("cap_evidence", eid)
            if e and e["public"] and e["verified"] and e["kind"] == "outcome" and e["data"].get("rel_effect") is not None:
                if e["capability"] not in beliefs or beliefs[e["capability"]].get("src") != "own":
                    beliefs[e["capability"]] = {"rel": e["data"]["rel_effect"], "w": 0.7, "src": "contact",
                                                "from": author, "t": ctx.now, "evidence": eid}
                    changed = True
        for who in re.findall(r"#breach\s+([op]:[a-z0-9-]+)", body):
            if who != m.get("org") and who != m["owner"]:
                a, b = trust.get(who, [1.0, 1.0])
                trust[who] = [a, round(b + 1.5 * w, 3)]       # hearsay counts less than one's own experience
                changed = True
    newest = max([t for t, _ in posts[:n]] + [seen])
    if changed:
        soc.set(ctx, mid, beliefs=beliefs, trust=trust)
    if newest != seen:
        soc.set(ctx, mid, routines=dict(m["routines"], forum_reader={"cursor": newest}))


def register_dynamics(world) -> None:
    rt = world.hooks.setdefault("mind_routines", {})
    rt.setdefault("provider", provider_routine)
    rt.setdefault("entrepreneur", entrepreneur_routine)
    rt.setdefault("forum_reader", forum_reader_routine)
