"""IH-13: the environment returns observations and consequences only -- no handcrafted scalar reward or bonus for
completing an operation (spec 12.1, 12A.15, 13.1).

The complete harness as the runtime composes it (``factories.all_executors``: internal, code, visual, api with
HTTP / publishing / Discord, skills; the browser in the ``-m browser`` variant) behind the real authorization
service, every operation authorized and executed for real against local staging services. Every successful
result, observation, event, measured outcome, metric, resource state and capability description is searched for
a reward-like field, and the contract types themselves are checked for one. Nothing here is a stand-in."""
import dataclasses
import json
import os
import shutil
import tempfile

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.factories import all_executors
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer

from e_fix_support import CountingServer, html

REWARDISH = ("reward", "bonus", "return_to_go", "advantage")


def keys(o, depth=0):
    if depth > 40:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k).lower()
            yield from keys(v, depth + 1)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from keys(v, depth + 1)


def rewardish(o):
    blob = json.loads(json.dumps(o, default=lambda x: x.to_dict() if hasattr(x, "to_dict") else str(x)))
    return sorted({k for k in keys(blob) if any(w in k for w in REWARDISH) or k in ("score", "points", "utility")})


def quantity_names(q):
    return [x.name.lower() for x in q]


def test_the_contract_types_have_no_reward_field():
    for cls in (C.ActionResult, C.Observation, C.OutcomeObservation, C.OutcomeQuery, C.Quantity, C.PrimitiveOp,
                C.ToolDescriptor, C.Cost, C.Provenance):
        names = [f.name.lower() for f in dataclasses.fields(cls)]
        assert not [n for n in names if any(w in n for w in REWARDISH)], (cls.__name__, names)
    assert not [m for m in dir(C.EnvironmentAdapter) if "reward" in m.lower()]


def _run(tmp_path, browser):
    site, discord = CountingServer(), StagingDiscordServer(StagingDiscord(token="staging-" + os.urandom(12).hex(),
                                                                          rate_limit=(1000, 1.0)))
    site.page("/", html("home", "<h1>Home</h1><a href='/about'>About</a><form method=post action='/notes'>"
                                "<input name=t><button>Save</button></form>"))
    site.page("/about", html("about", "<p>About this site</p>"))
    site.routes["/api/items"] = lambda *a: (200, {"Content-Type": "application/json"}, '{"items": [1, 2, 3]}')
    site.routes["/notes"] = lambda *a: (201, {"Content-Type": "application/json"}, '{"saved": true}')
    st = discord.state
    st.add_human("alice")
    base = tempfile.mkdtemp(prefix="camlive-hx-", dir="/tmp")     # the sandbox uid must be able to traverse it
    os.chmod(base, 0o755)
    run_dir = os.path.join(base, "run")
    rest = discord.base_url.split("//", 1)[1].split("/", 1)[0]
    owned = [site.host, f"account:discord:{st.guild_id}", "channel:discord:*"]
    task = C.TaskSpec(objective_id="obj-hx-consequences", objective="operate the staging sites", mode="staging",
                      resources={"owned": owned, "preauthorized": owned,
                                 "discord": {"guild_id": st.guild_id, "base_url": discord.base_url}},
                      data_permissions={"aggregate_metrics": True},
                      limits=C.Limits(duration_s=3600, max_external_actions=200, max_decisions=200, compute_s=600))
    svc = PolicyService(run_dir, fsync=False)
    svc.put_secret("discord_bot_token", st.token, [rest], ["discord"], by="test")
    svc.register_local_service(rest, by="test", note="staging Discord service")
    svc.register_task(task, "operator")
    env = ComputerEnvironment(run_dir, svc, secrets=svc.secrets_handle(), provenance="staging")
    for ex in all_executors(run_dir, env=env, policy=svc, task=task, browser_enabled=browser,
                            discord_base_url=discord.base_url):
        env.register(ex)
    ops = [("wait", C.Modality.INTERNAL, {"seconds": 0}),
           ("sandbox.python", C.Modality.CODE, {"code": "print(sum(range(10)))"}),
           ("http.get", C.Modality.API, {"url": site.url + "/api/items"}),
           ("http.request", C.Modality.API, {"method": "POST", "url": site.url + "/notes", "json": {"t": "a note"}}),
           ("discord.get_guild_stats", C.Modality.API, {"guild_id": st.guild_id}),
           ("skill.list", C.Modality.SKILL, {})]
    if browser:
        ops += [("browser.navigate", C.Modality.STRUCTURED_BROWSER, {"url": site.url + "/"}),
                ("browser.inspect", C.Modality.STRUCTURED_BROWSER, {}),
                ("visual.screenshot", C.Modality.VISUAL, {})]
    seen, cursor = [], ""
    try:
        for op, m, args in ops:
            a = C.ActionSpec(action_id=C.new_id("act"), operation=op, modality=m, arguments=args,
                             verify=C.VerificationCondition())
            dec = svc.authorize(a, task, {})
            assert dec.verdict == C.AuthVerdict.ALLOW, (op, dec.reasons)
            res = env.execute(a, dec)
            assert res.status == C.ResultStatus.OK, (op, res.failure_class, res.failure_detail)
            assert rewardish(res.to_dict()) == [], (op, rewardish(res.to_dict()))
            for o in res.observations:
                assert not [n for n in quantity_names(o.quantities) if any(w in n for w in REWARDISH)], (op, o.quantities)
            seen.append(op)
            evs, cursor = env.poll_events(cursor)
            assert rewardish([e.to_dict() for e in evs]) == []
        pending = env.get_pending_outcomes()
        measured = env.measure(pending) if pending else []
        caps = json.loads(json.dumps(env.get_capabilities(), default=str))
        assert caps["features"].pop("reward") is False          # the one place the word appears: "there is none"
        for blob in ([o.to_dict() for o in env.observe()], [q.to_dict() for q in env.get_observable_metrics()],
                     env.get_resource_state(), [m_.to_dict() for m_ in measured], caps,
                     [t.to_dict() for t in env.discover_tools()]):
            assert rewardish(blob) == [], rewardish(blob)
        assert not [q for q in env.get_observable_metrics() if any(w in q.name.lower() for w in REWARDISH)]
        return seen
    finally:
        env.close()
        svc.close()
        site.close()
        discord.close()
        shutil.rmtree(base, ignore_errors=True)


def test_no_modality_returns_a_reward_for_a_completed_operation(tmp_path):
    seen = _run(tmp_path, browser=False)
    assert {"wait", "sandbox.python", "http.get", "http.request", "discord.get_guild_stats", "skill.list"} <= set(seen)


@pytest.mark.browser
def test_no_modality_returns_a_reward_including_the_browser_and_visual(tmp_path):
    seen = _run(tmp_path, browser=True)
    assert {"browser.navigate", "browser.inspect", "visual.screenshot"} <= set(seen)
