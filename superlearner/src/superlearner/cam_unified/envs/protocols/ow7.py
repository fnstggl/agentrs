"""OW-7 Cooperation and coordination (spec 16): a useful outcome that depends on independently controlled
participants.

The world holds, besides its own minds, firms and consumers, independently controlled participants (here a
Crucible `ExampleInductor` operating a service of its own, and a random actor). The agent's outcome counts only
through `envs.goals.joint_utility`: third-party revenue of the agent's group and of its partners' groups (each
excluding what they paid each other), and only when a verified dependency exists - the agent paid the partners for
their work (ledger flows) or completed a contract with them. Success needs all of: joint utility >= target over 14
days, the agent's own independent revenue >= a floor (it must contribute, not just pay), and payments to partners
>= a floor (it must actually use their work), on 3 consecutive daily checks within 30 days. Messages, proposals
and agreements count nothing.

Control: `RelayIntegrator` (non-privileged): finds a partner's service through the public market, learns its public
category examples, deploys its own service that calls the partner's service for every request (paying it), lists
it, and operates it. Whether independent customers buy such a composed service is exactly what calibration
measures.
"""
from __future__ import annotations

from .. import goals as _goals          # noqa: F401 - registers cam_joint_utility / cam_partner_payments
from .base import COMMON_UNSUPPORTED, Control, TaskProtocol, idle_control

DAY = 86400
PARTNERS = ("partner-01",)
THRESHOLDS = {"joint_14d": 30_000, "own_revenue_14d": 2_000, "partner_payments_14d": 500}


def participants():
    from crucible.agents.scripted import RandomPolicy
    from crucible.usefulness.agents import ExampleInductor
    return [("partner-01", ExampleInductor(), 3_000_000), ("bystander-02", RandomPolicy(), 50_000)]


def make_task(world_family: str, seed: int, horizon_days: int = 30, sustain: int = 3, endowment: int = 3_000_000,
              joint: int = THRESHOLDS["joint_14d"], own: int = THRESHOLDS["own_revenue_14d"],
              paid: int = THRESHOLDS["partner_payments_14d"]):
    from crucible.goals.contracts import GoalContract
    w14 = 14 * DAY
    partners = [f"p:{h}" for h in PARTNERS]
    pred = {"all": [{"metric": "cam_joint_utility", "args": {"principal": "$org", "partners": partners,
                                                             "window_s": w14, "min_exchange_cents": int(paid)},
                     "op": ">=", "value": int(joint)},
                    {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": w14}, "op": ">=",
                     "value": int(own)},
                    {"metric": "cam_partner_payments", "args": {"principal": "$org", "partners": partners,
                                                                "window_s": w14}, "op": ">=", "value": int(paid)}]}
    text = (f"Produce value together with the independent participants {', '.join(partners)}. Success is verified "
            f"over the preceding 14 days on an organisation you own: you paid them at least {int(paid)} cents for "
            f"their work, you earned at least {int(own)} cents from independent customers, and together you and "
            f"they earned at least {int(joint)} cents from third parties - on {sustain} consecutive daily checks "
            f"within {horizon_days} days.")
    g = GoalContract("ow7_cooperation", text, {"family": world_family, "seed": int(seed), "overrides": {}}, "$self",
                     pred, {"metric": "cam_joint_utility", "args": {"principal": "$org", "partners": partners,
                                                                    "window_s": w14, "min_exchange_cents": int(paid)},
                            "target": int(joint)}, horizon_days * DAY, sustain, endowment=endowment,
                     tags=("ow7", "outcome_only"), progress_every_s=6 * 3600)
    g.protocol = "ow7"
    return g


RELAY_SOURCE = '''import crucible_sdk
UPSTREAM = {upstream!r}


def solve(args, caller):
    r = crucible_sdk.call(UPSTREAM, **args)
    if r.get("status") != 200:
        raise RuntimeError("upstream unavailable")
    return r["body"]
'''


def relay_integrator(markup: float = 1.2, capital: int = 600_000):
    from crucible.agents.scripted import ScriptPolicy, call, ok

    class RelayIntegrator(ScriptPolicy):
        name = "relay_integrator"

        def script(self):
            lst = None
            while lst is None:                                     # wait for a partner's public listing
                r = yield call("market.browse", kind="api", limit=30)
                for x in (r["result"] if ok(r) else []):
                    if x.get("host") and any(str(x.get("seller", "")).startswith(f"o:{p}") for p in PARTNERS):
                        lst = x
                        break
                if lst is None:
                    yield from self.wait(6)
            r = yield call("world.describe", host=lst["host"])
            if not ok(r):
                return
            r = yield call("use.categories", category=lst["need"])
            cat = r["result"][0] if ok(r) and r["result"] else {"title": lst["title"], "keywords": [], "doc": ""}
            org, acct, _ = yield from self.found_company("relay", capital)
            if acct is None:
                return
            host = f"{self.handle[:12]}-relay"
            r = yield call("code.create_repo", name=host, owner=org, description=cat.get("title", ""))
            if not ok(r):
                return
            repo = r["result"]["repo"]
            price = max(int(lst["price_cents"]) + 1, int(lst["price_cents"] * markup))
            manifest = {"entry": "service.py", "description": cat.get("title", lst["title"]),
                        "endpoints": {"solve": {"handler": "solve", "doc": str(cat.get("doc", ""))[:300],
                                                "price_cents": price,
                                                "params": {"type": "object", "additionalProperties": True}}}}
            import json as _json
            yield call("code.commit", repo=repo, files={"crucible.json": _json.dumps(manifest),
                                                         "service.py": RELAY_SOURCE.format(
                                                             upstream=lst["host"] + "." + lst["endpoint"])},
                       message="relay")
            r = yield call("cloud.providers")
            provs = sorted((p for p in (r or {}).get("result", []) if p["cores_free"] >= 1 and not p["in_outage"]),
                           key=lambda p: (p["price_core_hour_cents"], p["provider"]))
            if not provs:
                return
            r = yield call("cloud.rent", provider=provs[0]["provider"], cores=1, account=acct, owner=org)
            if not ok(r):
                return
            r = yield call("cloud.deploy", repo=repo, host=host, instance=r["result"]["instance"])
            if not ok(r):
                return
            yield call("id.grant", grantee=f"svc:{host}", cap="bank.spend", owner=org, limit=capital // 2)
            kws = " ".join(cat.get("keywords", []))
            yield call("market.list", kind="api", title=f"{host}: {cat.get('title', lst['title'])}", price_cents=price,
                       seller=org, host=host, endpoint="solve", need=lst["need"],
                       description=f"{cat.get('title', '')}. Always available. Keywords: {kws} {host}.crucible")
            yield call("forum.post", board="market", title=f"Launch: {host}.crucible",
                       body=f"{cat.get('title', '')} at {price} cents per request. {host}.crucible")
            while True:
                yield from self.wait(24)
                yield call("use.report", host=host, days=7)
    return RelayIntegrator()


PROTOCOL = TaskProtocol(
    name="ow7", title="Cooperation: jointly produced third-party value with independent participants",
    spec_ref="16 OW-7", make_task=make_task,
    splits={"dev": [("ow_tiny", 1)], "train": [("ow_market", 1), ("ow_frontier", 2)], "val": [("ow_composite", 2)],
            "test": [("ow_heldout", 5)]},
    horizon_days=30, thresholds=THRESHOLDS, participants=participants,
    controls=[Control("relay_integrator", relay_integrator, "control",
                      note="composes the partner's service into its own listed service (public information only)"),
              idle_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "NPC firms do not negotiate alliances or joint ventures with agents beyond employment, lending, licences and "
        "paid API calls; cooperation is measured through verified payments and contracts."],
    arms=["CAM Unified", "CAM with identical-copy partners only", "frozen semantic agent"])
