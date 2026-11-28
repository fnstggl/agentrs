"""AU end to end: the composed computer environment (``factories.all_executors``: internal, api with HTTP /
publishing / Discord, skills) behind the real authorization service, on a *live* task whose operator-signed record
was built by ``tools/authorize_live.py --standing`` -- against a staging Discord server and a local "hosting API"
(no external contact). Routine operations inside CAM's designated infrastructure execute without per-item approval
and without repeated operator intervention; irreversible operations, new credentials and the emergency stop keep the
operator in charge; the hosting credential the operator entered through ``harness.credentials`` never appears in
an observation, a result, a log or any other file of the run outside the vault; a credential an API mints is
redacted from the observation."""
import json
import os
import shutil
import sys
import tempfile
import time

import pytest

from cam_live import contracts as C
from cam_live.harness import credentials as CR
from cam_live.harness.auth import PolicyService, liveauth
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.factories import all_executors
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer

from au_support import ALLOW, DENY, NEEDS, PROFILE, files_containing, install_operator_key
from e_fix_support import CountingServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools")))
import authorize_live  # noqa: E402


def _token():
    return "-".join(["hst", os.urandom(9).hex(), os.urandom(9).hex()])


@pytest.fixture
def world(tmp_path):
    hosting, minted, tok = CountingServer(), _token(), _token()
    files = {}

    def site(method, path, q, headers, body):
        if headers.get("Authorization") != f"Bearer {tok}":
            return 401, {"Content-Type": "application/json"}, '{"error": "unauthorized"}'
        if method == "PUT":
            files[path] = body
            return 201, {"Content-Type": "application/json"}, json.dumps({"stored": path, "bytes": len(body)})
        if method == "DELETE":
            for k in [k for k in files if k.startswith(path)]:
                del files[k]
            return 204, {}, ""
        return 405, {}, "no"
    class Routes(dict):                                    # every path of the site is served by ``site``
        def get(self, path, default=None):
            return site if path == "/sites/cam" or path.startswith("/sites/cam/") else dict.get(self, path, default)
    hosting.routes = Routes(hosting.routes)
    hosting.routes["/v1/tokens"] = lambda m, p, q, h, b: (
        201, {"Content-Type": "application/json"}, json.dumps({"id": 7, "token": minted, "scope": "deploy"}))
    discord = StagingDiscordServer(StagingDiscord(token="staging-" + os.urandom(12).hex(), rate_limit=(1000, 1.0)))
    st = discord.state
    base = tempfile.mkdtemp(prefix="camlive-au-e2e-", dir="/tmp")
    run_dir = os.path.join(base, "run")
    os.makedirs(run_dir)
    rest = discord.base_url.split("//", 1)[1].split("/", 1)[0]
    task = C.TaskSpec(objective_id="obj-au-e2e", objective="operate CAM's own infrastructure", mode="live",
                      interfaces=["api", "internal", "skill"],
                      resources={"discord": {"guild_id": st.guild_id, "base_url": discord.base_url,
                                             "token_secret": "discord_bot_token"}},
                      data_permissions={"aggregate_metrics": True},
                      limits=C.Limits(duration_s=3600, money=1.0, max_external_actions=200, max_decisions=200,
                                      compute_s=600))
    svc = PolicyService(run_dir, fsync=False)
    for h in (rest, hosting.host):
        svc.register_local_service(h, by="operator", note="staging")
    svc.put_secret("discord_bot_token", st.token, [rest], ["discord"], by="operator")
    # the operator enters the hosting credential: bound to the hosting API, to HTTP operations and to skills (a skill
    # is granted as skill.run / skill.induce; each of its steps is authorized on its own), 1 hour
    CR.store(run_dir, "hosting_token", tok, dest=[hosting.url + "/*"],
             operations=["http.request", "http.get", "skill.*"], expires_hours=1.0, verifier=False)
    priv = install_operator_key(svc, tmp_path)
    rec = authorize_live.build_record(task, hours=1.0, owned=[], read=[rest, hosting.host], preauthorized=[],
                                      accounts=[], issued_by="operator", standing=PROFILE,
                                      hosting=[hosting.url + "/*"])
    svc.register_live_authorization(liveauth.sign_live_authorization(rec, priv), "operator")
    svc.register_task(task, "operator")
    env = ComputerEnvironment(run_dir, svc, secrets=svc.secrets_handle(), provenance="staging")
    for ex in all_executors(run_dir, env=env, policy=svc, task=task, browser_enabled=False,
                            discord_base_url=discord.base_url, exclude=("code", "visual")):
        env.register(ex)
    w = dict(hosting=hosting, files=files, tok=tok, minted=minted, discord=discord, st=st, svc=svc, env=env,
             task=task, run_dir=run_dir, rec=rec)
    yield w
    env.close()
    svc.close()
    hosting.close()
    discord.close()
    shutil.rmtree(base, ignore_errors=True)


def run(w, op, modality=C.Modality.API, expect=ALLOW, **args):
    a = C.ActionSpec(action_id=C.new_id("act"), operation=op, modality=modality, arguments=args,
                     verify=C.VerificationCondition())
    d = w["svc"].authorize(a, w["task"], {})
    assert d.verdict == expect, (op, args, d.verdict, d.reasons)
    if d.verdict != ALLOW:
        return a, d, None
    res = w["env"].execute(a, d)
    return a, d, res


def approve_run(w, a, d):
    w["svc"].resolve_approval(d.approval_id, True, "operator")
    d2 = w["svc"].authorize(a, w["task"], {})
    assert d2.verdict == ALLOW, d2.reasons
    return w["env"].execute(a, d2)


def test_routine_operations_run_without_approval_and_retained_ones_wait_for_the_operator(world):
    w, st = world, world["st"]
    assert w["rec"]["standing"] == {"profile": PROFILE, "code_network": []}
    assert w["hosting"].url + "/*" in w["rec"]["preauthorized"] and "channel:discord:*" in w["rec"]["preauthorized"]
    # Discord: routine operations in CAM's own server
    _, _, res = run(w, "discord.create_channel", guild_id=st.guild_id, name="notes", type="text")
    assert res.status == C.ResultStatus.OK, (res.failure_class, res.failure_detail)
    cid = [c["id"] for c in st.channels.values() if c["name"] == "notes"][0]
    _, _, res = run(w, "discord.send_message", guild_id=st.guild_id, channel_id=cid, content="Session notes follow.")
    assert res.status == C.ResultStatus.OK, (res.failure_class, res.failure_detail)
    # hosting: two deployments with the operator's credential, by reference; no approval for either
    for name in ("index.html", "about.html"):
        _, _, res = run(w, "http.request", method="PUT", url=f"{w['hosting'].url}/sites/cam/{name}",
                        body=f"<h1>{name}</h1>", headers={"Content-Type": "text/html"},
                        auth={"secret": "secret:hosting_token"})
        assert res.status == C.ResultStatus.OK and res.observations[0].data["status"] == 201, res.failure_detail
    assert set(w["files"]) == {"/sites/cam/index.html", "/sites/cam/about.html"}
    assert w["svc"].approvals() == [] or all(x["status"] != "pending" for x in w["svc"].approvals())
    # irreversible: deleting the site waits for the operator; nothing is deleted before the approval
    a, d, _ = run(w, "http.request", expect=NEEDS, method="DELETE", url=f"{w['hosting'].url}/sites/cam",
                  auth={"secret": "secret:hosting_token"})
    assert "R-IRREVERSIBLE-HTTP" in d.policy_ids and len(w["files"]) == 2
    res = approve_run(w, a, d)
    assert res.status == C.ResultStatus.OK and w["files"] == {}, (res.failure_class, res.failure_detail)
    a, d, _ = run(w, "discord.delete_channel", expect=NEEDS, guild_id=st.guild_id, channel_id=cid)
    assert "R-IRREVERSIBLE" in d.policy_ids and cid in st.channels
    res = approve_run(w, a, d)
    assert res.status == C.ResultStatus.OK and cid not in st.channels, (res.failure_class, res.failure_detail)
    # a new credential on the hosting service: approval retained; the minted value never reaches the learner
    a, d, _ = run(w, "http.request", expect=NEEDS, method="POST", url=f"{w['hosting'].url}/v1/tokens",
                  json={"scope": "deploy"}, auth={"secret": "secret:hosting_token"})
    assert "R-NEW-PERMISSION" in d.policy_ids
    res = approve_run(w, a, d)
    assert res.status == C.ResultStatus.OK
    obs = res.observations[0]
    assert w["minted"] not in json.dumps(obs.to_dict()) and obs.data["redacted_credential_fields"] == 1
    assert obs.data["json"]["token"].startswith("[redacted") and obs.data["json"]["id"] == 7
    # the operator's credential and the minted one appear in no file of the run except the vault
    assert files_containing(w["run_dir"], w["tok"]) == ["vault/hosting_token.secret.json"]
    assert files_containing(w["run_dir"], w["minted"]) == []
    assert files_containing(w["run_dir"], st.token) == ["vault/discord_bot_token.secret.json"]


def test_emergency_stop_revocation_and_expiry_end_the_standing_use_of_a_credential(world):
    w = world
    put = dict(method="PUT", url=f"{w['hosting'].url}/sites/cam/news.html", body="<p>x</p>",
               auth={"secret": "secret:hosting_token"})
    _, _, res = run(w, "http.request", **put)
    assert res.status == C.ResultStatus.OK
    w["svc"].emergency_stop("operator: stop everything", "operator")
    run(w, "http.request", expect=DENY, **put)
    assert w["svc"].reset_stop("operator")
    n = len(w["hosting"].log)
    # the operator revokes the credential (CLI): the next deployment is not sent with it
    assert CR.main(["revoke", "hosting_token", "--run-dir", w["run_dir"]]) == 0
    _, _, res = run(w, "http.request", **put)
    assert res.status != C.ResultStatus.OK
    assert all(h.get("Authorization") != f"Bearer {w['tok']}" for _, _, h, _ in w["hosting"].log[n:])
    listing = CR.listing(w["run_dir"])
    entry = [x for x in listing if x["name"] == "hosting_token"][0]
    assert entry["revoked"] and entry["operations"] == ["http.request", "http.get", "skill.*"] and entry["expires_at"] > time.time()


def test_a_skill_induced_from_successful_deployments_is_reused_without_operator_intervention(world):
    w = world
    traces, inputs = [], []
    for page in ("about.html", "guide.html"):
        trace = []
        for url, body in ((f"{w['hosting'].url}/sites/cam/{page}", f"<h1>{page}</h1>"),
                          (f"{w['hosting'].url}/sites/cam/index.html", f"<a href='{page}'>{page}</a>")):
            a, _, res = run(w, "http.request", method="PUT", url=url, body=body, auth={"secret": "secret:hosting_token"})
            assert res.status == C.ResultStatus.OK
            trace.append(a.action_id)
        traces.append(trace)
        inputs.append({"page": page})
    _, d, res = run(w, "skill.induce", C.Modality.SKILL, traces=traces, name="deploy a page and link it",
                    inputs=inputs)
    assert res.status == C.ResultStatus.OK, (res.failure_class, res.failure_detail)
    data = res.observations[-1].data
    sid = (res.artifacts or {}).get("skill_id") or data.get("skill_id") or (data.get("skill") or {}).get("skill_id")
    assert sid, data
    _, _, desc = run(w, "skill.describe", C.Modality.SKILL, skill_id=sid)
    dd = desc.observations[-1].data
    sk = (dd.get("skills") or [dd.get("skill") or dd])[0]
    params = sorted((sk.get("arg_schema") or sk.get("args_schema") or {}).get("properties", {}))
    assert params, json.dumps(sk, default=str)[:3000]
    before = dict(w["files"])
    _, d, res = run(w, "skill.run", C.Modality.SKILL, skill_id=sid,
                    arguments={p: "faq.html" for p in params})
    assert res.status == C.ResultStatus.OK, (res.failure_class, res.failure_detail)
    assert "/sites/cam/faq.html" in w["files"] and "/sites/cam/faq.html" not in before
    assert "faq.html" in w["files"]["/sites/cam/index.html"].decode()
    assert not [x for x in w["svc"].approvals() if x["status"] == "pending"]
