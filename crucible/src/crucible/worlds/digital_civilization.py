"""The flagship world: a synthetic digital civilization.

Module `civ` writes the initial society at genesis (all through ordinary internal APIs and recorded events):
a standards institute that publishes the public need formats, compute providers, incumbent software firms (each
with an owner, a registered company, a bank account, a repository containing a real implementation, a rented
instance, a live deployment and a market listing), a labour pool, lenders and some open bounties.

Module `shocks` fires scheduled exogenous changes: preference drift, demand surges, compute shortages, competitor
entry, reliability storms and the emergence of new needs. They are mechanisms acting on state, announced only
through their consequences (and optionally a public news broadcast), never as instructions to agents.
"""
from __future__ import annotations

import json

import numpy as np

from ..core.errors import CrucibleError
from ..core.kernel import Module, applier, process, register_module
from ..economy.needs import NEEDS, tier_source

DAY = 86400


def service_repo_files(need: str, tier: int, price: int) -> dict[str, str]:
    manifest = {"entry": "service.py", "description": f"{NEEDS[need].title} as a service",
                "endpoints": {"solve": {"handler": "solve", "doc": NEEDS[need].doc[:400], "price_cents": price,
                                        "params": {"type": "object", "additionalProperties": True}}}}
    return {"crucible.json": json.dumps(manifest, indent=1), "service.py": tier_source(need, tier),
            "README.md": f"# {NEEDS[need].title}\n\nServes the public `{need}` format.\n"}


@register_module
class Civ(Module):
    name = "civ"
    depends = ("identity", "bank", "code", "cloud", "market", "business", "jobs", "npcs")

    def genesis(self, ctx):
        cfg = self.config
        w = ctx.world
        idm, bank, cloud, npcs = w.modules["identity"], w.modules["bank"], w.modules["cloud"], w.modules["npcs"]
        biz = w.modules["business"]
        rng = w.rng_for("civ-genesis")

        def org_with_owner(handle, name, kind, cash, trusted=False, sector=""):
            owner = idm.create_person(ctx, f"{handle}-ceo", f"{name} CEO", "npc", {"role": "owner"})
            bank.open_account(ctx, owner, "checking")
            oid = idm.create_org(ctx, handle, name, owner, "company", {"trusted_runtime": trusted})
            biz.register_company(ctx, oid, owner, sector, fee=0)
            acct = bank.open_account(ctx, oid, "operating")
            if cash:
                bank.move(ctx, "acct_genesis", acct, int(cash), "endowment", f"initial capital of {oid}")
            return oid, owner, acct

        # standards institute: publishes the need formats as public documents
        docs = w.modules.get("docs")
        sid, sowner, _ = org_with_owner("standards", "Crucible Standards Institute", "institution", 0)
        if docs is not None:
            for need in sorted(cfg.get("needs", NEEDS)):
                docs.create_doc(ctx, sid, f"Format: {need}", NEEDS[need].spec_text() +
                                f"\nKeywords: {', '.join(NEEDS[need].keywords)}\n", "public")
            docs.create_doc(ctx, sid, "Directory of public services",
                            "Built-in services: id (identity), bank, mail, forum, search, docs, code, issues, cal, "
                            "support, ads, market, cloud, biz (company registry), contracts, jobs, research (surveys). "
                            "Use world.describe(host) to read any service's API. Hosted services live at "
                            "<host>.crucible and are found through search, the market and other people.", "public")
        # compute providers
        for p in cfg.get("providers", []):
            oid, owner, acct = org_with_owner(p["handle"], p["name"], "company", p.get("cash", 5_000_000), sector="compute")
            cloud.add_provider(ctx, oid, p["name"], p["price"], p["cores"], p["reliability"])
            if cfg.get("firm_minds") and "society" in w.modules:
                soc = w.modules["society"]
                soc.add_mind(ctx, oid, "firm", oid, owner, soc.draw_traits(rng), role="provider",
                             extra={"routines": {"provider": {"floor": int(p["price"] * 0.5)}}})
        # incumbent firms
        for f in cfg.get("firms", []):
            oid, owner, acct = org_with_owner(f["handle"], f["name"], "company", f.get("cash", 2_000_000),
                                              trusted=True, sector=f["need"])
            code = w.modules["code"]
            rid = code.create_repo(ctx, oid, f"{f['need'].replace('.', '-')}-svc", "private",
                                   f"{f['name']} {f['need']} service")
            code.commit_files(ctx, rid, service_repo_files(f["need"], f["tier"], f["price"]), "initial", owner,
                              trusted=True)
            inst = ctx.call("cloud.rent", {"provider": f["provider"], "cores": f.get("cores", 1), "account": acct,
                                           "owner": oid}, as_principal=owner)
            ctx.call("cloud.deploy", {"repo": rid, "host": f["host"], "instance": inst["instance"]}, as_principal=owner)
            lst = ctx.call("market.list", {"kind": "api", "title": f"{f['name']}: {NEEDS[f['need']].title}",
                                           "description": f"{NEEDS[f['need']].title}. Call {f['host']}.solve. "
                                                          f"Keywords: {' '.join(NEEDS[f['need']].keywords)}",
                                           "price_cents": f["price"], "seller": oid, "host": f["host"],
                                           "endpoint": "solve", "need": f["need"]}, as_principal=owner)
            if cfg.get("firm_minds") and "society" in w.modules:
                soc = w.modules["society"]
                soc.add_mind(ctx, oid, "firm", oid, owner, f.get("traits") or soc.draw_traits(rng),
                             role="service", extra={"routines": {"capacity": {}, "pricing": {
                                 "step": f.get("policy", {}).get("price_step", 0.06)}, "staffing": {},
                                 "forum_reader": {}}})
            else:
                npcs.add_firm(ctx, oid, owner, f["need"], f["host"], lst["listing"], rid, f["tier"], f.get("policy"))
        # labour pool (the first `innovators` developers also run the NPC invention routine, with extra savings)
        n_innov = int(cfg.get("innovators", 0))
        for i in range(int(cfg.get("workers", 0))):
            role = str(rng.choice(cfg.get("worker_roles") or ["developer", "support", "sales", "developer", "support"]))
            skill = float(np.clip(rng.beta(2, 2.5), 0.05, 0.99))
            person = idm.create_person(ctx, f"worker-{i:03d}", f"Worker {i}", "npc", {"role": role})
            acct = bank.open_account(ctx, person, "checking")
            bank.move(ctx, "acct_genesis", acct, int(rng.integers(20_000, 200_000)), "endowment", "savings")
            base = cfg.get("wage_base", 300_000)
            res = int(base * (0.6 + 0.8 * skill) * rng.uniform(0.85, 1.15))
            npcs.add_worker(ctx, person, {role: skill}, res, role)
            innov = role == "developer" and n_innov > 0
            if innov:
                n_innov -= 1
                bank.move(ctx, "acct_genesis", acct, int(cfg.get("innovator_cash", 400_000)), "endowment", "savings")
            if "society" in w.modules:
                soc = w.modules["society"]
                soc.add_mind(ctx, person, "person", None, person, soc.draw_traits(rng), role=role,
                             extra={"reservation": res, "greed": round(float(rng.uniform(0.03, 0.3)), 3),
                                    "affinity": {}, "employer": None, "skill": round(skill, 3),
                                    "effort_cost": int(res * 0.5),
                                    "routines": {"forum_reader": {},
                                                 **({"entrepreneur": {"min_cash": 150_000}}
                                                    if cfg.get("entrepreneurs", True) else {}),
                                                 **({"innovator": {"min_cash": 150_000}} if innov else {})}})
        # lenders
        for L in cfg.get("lenders", []):
            oid, owner, acct = org_with_owner(L["handle"], L["name"], "company", L.get("cash", 20_000_000), sector="finance")
            npcs.add_lender(ctx, oid, owner, L.get("rate", 0.02), L.get("max", 2_000_000))
        # open bounties from incumbents (outsourced R&D)
        for b in cfg.get("bounties", []):
            org = f"o:{b['poster']}"
            acct = bank.primary_account(ctx.state, org)
            owner = idm.org_owner_persons(ctx.state, org)[0]
            ctx.call("jobs.post_bounty", {"need": b["need"], "reward_cents": b["reward"], "min_quality": b["min_q"],
                                          "account": acct, "deadline_s": b.get("deadline_days", 60) * DAY,
                                          **({"max_cpu_ms": b["max_cpu_ms"]} if b.get("max_cpu_ms") else {})},
                     as_principal=owner)


@register_module
class Shocks(Module):
    """Scheduled exogenous changes. Config: {"events": [{"day": d, "kind": k, ...params}]}."""
    name = "shocks"
    depends = ("identity", "bank", "cloud")

    def genesis(self, ctx):
        for i, e in enumerate(self.config.get("events", [])):
            ctx.schedule(int(e["day"] * DAY), "shocks.fire", {"i": i}, prio=1)

    @process("fire")
    def pr_fire(self, ctx, i):
        e = dict(self.config["events"][i])
        kind = e["kind"]
        st = ctx.state
        if kind in ("pref_drift", "demand_surge", "new_need"):
            ctx.emit("shocks.population", {"i": i, **e}, vis="internal")
        elif kind == "compute_shortage":
            for pid in st.find_ids("providers", "status", "active"):
                p = st.tables["providers"][pid]
                lose = int(p["cores"] * e.get("capacity_loss", 0.3))
                ctx.emit("shocks.capacity", {"provider": pid, "cores": max(0, p["cores"] - lose),
                                             "price": int(p["price"] * e.get("price_mult", 1.5))}, vis="internal")
        elif kind == "reliability":
            for pid in st.find_ids("providers", "status", "active"):
                ctx.emit("shocks.reliability", {"provider": pid, "reliability": e.get("reliability", 0.8)}, vis="internal")
        elif kind == "entrant":
            civ = Civ({"firms": [e["firm"]], "needs": []})
            civ.world = ctx.world
            try:
                civ.genesis(ctx)
            except CrucibleError as err:
                ctx.emit("note", {"what": "entrant failed", "error": err.as_dict()}, vis="internal")
        if e.get("news"):
            ctx.emit("note", {"what": "news", "text": e["news"]}, vis="public")

    @applier("shocks.population")
    def ap_population(self, ev):
        st = self.world.state
        e = ev.payload
        names = st.meta_get("cons_needs") or []
        if "c_need" not in st.arrays:
            return
        rng = self.world.rng_for("shock", e["i"])
        need = st.arrays["c_need"]
        if e["kind"] == "pref_drift":
            pref = st.array_mut("c_pref")
            mask = need == names.index(e["need"]) if e.get("need") in names else np.ones(len(need), bool)
            target = np.array(e.get("toward", [0.2, 0.2, 0.6]), np.float32)
            a = float(e.get("strength", 0.5))
            pref[mask] = (1 - a) * pref[mask] + a * target
        elif e["kind"] == "demand_surge":
            rate = st.array_mut("c_rate")
            mask = need == names.index(e["need"]) if e.get("need") in names else np.ones(len(need), bool)
            rate[mask] = rate[mask] * float(e.get("mult", 2.0))
        elif e["kind"] == "new_need":
            if e["need"] not in names:
                names = names + [e["need"]]
                st.meta_set("cons_needs", names)
            ni = names.index(e["need"])
            frac = float(e.get("fraction", 0.15))
            sel = np.flatnonzero(rng.random(len(need)) < frac)
            nn = st.array_mut("c_need")
            nn[sel] = ni
            prov = st.array_mut("c_prov")
            prov[sel] = -1
            st.array_mut("c_aware")[sel] = -1
            wtp = st.array_mut("c_wtp")
            wtp[sel] = NEEDS[e["need"]].base_price_cents * rng.uniform(0.8, 3.0, len(sel)).astype(np.float32)

    @applier("shocks.capacity")
    def ap_capacity(self, ev):
        p = ev.payload
        self.world.state.patch("providers", p["provider"], cores=p["cores"], price=p["price"])

    @applier("shocks.reliability")
    def ap_reliability(self, ev):
        self.world.state.patch("providers", ev.payload["provider"], reliability=ev.payload["reliability"])
