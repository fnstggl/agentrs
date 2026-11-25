"""Scripted reference agents that act only through the `ExperienceEnv` interface.

Clarification 1 asks for environment capabilities to be validated with scripted reference agents (besides neural,
statistical and language-model learners). These are deliberately plain: a fixed sequence of ordinary tool calls,
each checked, so a capability test fails loudly when the world does not do what its tools say.

* `call(env, agent, tool, **args)` - execute and require ``ok``.
* `found_company(...)`, `deploy_service(...)` - an organisation with a bank account, a repository with a service,
  rented compute and a live deployment (the 4.0 economy's own path: `id.create_org`, `biz.register`,
  `bank.open_account`, `code.create_repo`, `code.commit`, `cloud.rent`, `cloud.deploy`).
* `discover(env, agent, host)` - learn a host through `world.describe`; the host's endpoints then appear in the
  agent's tool list (an affordance a participant created).
"""
from __future__ import annotations

from typing import Any


class ActionFailed(RuntimeError):
    pass


def call(env, agent: str, tool: str, **args) -> Any:
    r = env.execute(agent, {"tool": tool, "args": args})
    if not r.get("ok"):
        raise ActionFailed(f"{agent} {tool}: {r.get('error')}")
    return r["result"]


def personal_account(env, agent: str) -> str:
    accts = [a for a in call(env, agent, "bank.accounts") if a["owner"] == agent]
    if not accts:
        raise ActionFailed(f"{agent} has no account")
    return accts[0]["account"]


def found_company(env, agent: str, handle: str, capital: int) -> dict:
    me = personal_account(env, agent)
    org = call(env, agent, "id.create_org", handle=handle, name=handle.title(), kind="company")["org"]
    call(env, agent, "biz.register", org=org, account=me)
    acct = call(env, agent, "bank.open_account", owner=org, name="operating")["account"]
    call(env, agent, "bank.transfer", from_account=me, to=acct, amount=int(capital), memo="founder capital")
    return {"org": org, "account": acct, "personal": me}


def deploy_service(env, agent: str, *, handle: str, host: str, need: str = "sort.records", tier: int | None = None,
                   price: int = 5, capital: int = 1_000_000) -> dict:
    """A live agent-operated service at `host` whose endpoint `solve` serves the public `need` format."""
    from ..economy.needs import best_tier
    from ..worlds.digital_civilization import service_repo_files
    co = found_company(env, agent, handle, capital)
    repo = call(env, agent, "code.create_repo", name=host, owner=co["org"])["repo"]
    files = service_repo_files(need, best_tier(need) if tier is None else int(tier), int(price))
    call(env, agent, "code.commit", repo=repo, files=files, message="first version")
    provs = sorted((p for p in call(env, agent, "cloud.providers") if p["cores_free"] >= 1 and not p["in_outage"]),
                   key=lambda p: (p["price_core_hour_cents"], p["provider"]))
    if not provs:
        raise ActionFailed("no compute provider with a free core")
    inst = call(env, agent, "cloud.rent", provider=provs[0]["provider"], cores=1, account=co["account"],
                owner=co["org"])["instance"]
    dep = call(env, agent, "cloud.deploy", repo=repo, host=host, instance=inst)
    if dep.get("status") != "live":
        raise ActionFailed(f"deployment of {host} is {dep.get('status')}: {dep}")
    return {**co, "repo": repo, "instance": inst, "host": host, "need": need, "price": int(price)}


def discover(env, agent: str, host: str) -> list[str]:
    """`world.describe(host)`; returns the tool names the agent gained."""
    before = {t["name"] for t in env.tools(agent)}
    call(env, agent, "world.describe", host=host)
    return sorted({t["name"] for t in env.tools(agent)} - before)


def need_input(need: str, seed: int = 0) -> dict:
    import numpy as np
    from ..economy.needs import NEEDS
    inp, _ = NEEDS[need].generate(np.random.default_rng(seed), 0.5)
    return inp
