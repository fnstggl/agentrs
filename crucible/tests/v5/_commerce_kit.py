"""Shared builders for the commerce gate tests (K, SP-6.1): real worlds driven through ordinary tools."""
from __future__ import annotations

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World, WorldSpec  # noqa: E402

DAY = 86400
MODS = ["identity", "bank", "admin", "code", "search", "docs", "support", "contracts", "business", "cloud", "market",
        "products", "subs", "ventures", "xchg"]
PROVIDER = {"handle": "cloudco", "name": "CloudCo", "cores": 64, "price": 200, "reliability": 1.0, "cash": 5_000_000}


def make_world(seed: int = 0, configs: dict | None = None, extra: list[str] = (), params: dict | None = None,
               providers: bool = False) -> World:
    configs = dict(configs or {})
    if providers:
        configs.setdefault("ventures", {}).setdefault("providers", [PROVIDER])
    mods = [{"name": m, "config": configs.get(m, {})} for m in MODS + [x for x in extra if x not in MODS]]
    spec = WorldSpec("commerce-gate", seed, mods, dict({"cost_model": "deterministic"}, **(params or {})))
    return World.create(spec)


class Actor:
    def __init__(self, w: World, pid: str):
        self.w, self.id = w, pid

    def do(self, tool: str, **args):
        r = self.w.execute(self.id, {"tool": tool, "args": args})
        if not r.ok:
            raise AssertionError(f"{self.id} {tool} {args}: {r.error}")
        return r.result

    def attempt(self, tool: str, **args):
        return self.w.execute(self.id, {"tool": tool, "args": args})

    @property
    def acct(self) -> str:
        return self.w.modules["bank"].primary_account(self.w.state, self.id)


def join(w: World, handle: str, cents: int = 0, controller: str = "") -> Actor:
    pid = w.system_call("admin.join", {"handle": handle, "endowment": int(cents), "controller": controller})["principal"]
    return Actor(w, pid)


def balance(w: World, acct: str) -> int:
    return int(w.state.tables["accounts"][acct]["balance"])


def org_account(w: World, org: str) -> str:
    return w.modules["bank"].primary_account(w.state, org)


def media_business(w: World, handle: str, capital: int = 500_000, monthly: int | None = 2_000,
                   annual: int | None = None, trial_days: int = 0, controller: str = ""):
    """A founder with a company selling read access to a document by subscription."""
    f = join(w, f"{handle}-f", capital + 200_000, controller=controller)
    org = f.do("venture.form", handle=handle, name=handle.title(), capital_cents=capital)["org"]
    doc = f.do("docs.create", title=f"{handle} premium", body="premium content", visibility="private", org=org)["doc"]
    prod = f.do("prod.create", name=f"{handle} premium", kind="media", owner=org,
                delivery={"type": "content", "doc": doc})["product"]
    f.do("prod.update", product=prod, status="launched")
    plans = {}
    if monthly:
        plans["month"] = f.do("subs.create_plan", product=prod, price_cents=monthly, interval="month",
                              trial_days=trial_days)["plan"]
    if annual:
        plans["year"] = f.do("subs.create_plan", product=prod, price_cents=annual, interval="year",
                             trial_days=trial_days)["plan"]
    return f, org, prod, plans, doc


def saas_business(w: World, handle: str, need: str, tier: int, capital: int = 1_000_000, monthly: int = 3_000,
                  trial_days: int = 0, cores: int = 1, kind: str = "saas"):
    """A founder with a company running a solver of `need` (tier of the public reference ladder) as a service."""
    from crucible.economy.needs import NEEDS
    from crucible.worlds.digital_civilization import service_repo_files
    f = join(w, f"{handle}-f", capital + 300_000)
    out = f.do("venture.form", handle=handle, name=handle.title(), capital_cents=capital)
    org, acct = out["org"], out["account"]
    repo = f.do("code.create_repo", name="svc", owner=org)["repo"]
    commit = f.do("code.commit", repo=repo, files=service_repo_files(need, tier, 100_000), message="v1")["commit"]
    prov = sorted(p["provider"] for p in f.do("cloud.providers"))[0]
    inst = f.do("cloud.rent", provider=prov, cores=cores, account=acct, owner=org)["instance"]
    host = f"{handle}-svc"
    assert f.do("cloud.deploy", repo=repo, host=host, instance=inst)["status"] == "live"
    prod = f.do("prod.create", name=f"{handle} {need}", kind=kind, owner=org,
                delivery={"type": "service", "host": host, "endpoint": "solve", "need": need})["product"]
    f.do("prod.release", product=prod, version="1.0", commit=commit)
    f.do("prod.update", product=prod, status="launched")
    plan = f.do("subs.create_plan", product=prod, price_cents=monthly, interval="month", trial_days=trial_days)["plan"]
    return f, org, prod, plan, {"repo": repo, "host": host, "instance": inst, "commit": commit,
                                "need": NEEDS[need]}


def mint(w: World, owner: str, sku: str, qty: int) -> None:
    """World generation of conserved goods (a recorded system transaction, as industry/genesis does)."""
    w._run_system("test.mint", lambda ctx: w.modules["market"].mint_goods(ctx, owner, sku, qty))


def failures(w: World) -> list:
    """Records of contained failures in commerce processes (should be empty)."""
    return [e.payload for e in w.log if e.type in ("subs.bill_failed", "patrons.failed", "venture.daily_failed",
                                                     "xchg.sweep_failed", "proc.failed")]


def healthy(w: World) -> list[str]:
    return w.check_invariants() + w.modules["bank"].check_ledger_consistency(w)
