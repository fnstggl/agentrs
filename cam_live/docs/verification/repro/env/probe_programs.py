"""G-env probe 8: programs (ActionSpec.steps) across modalities under the real authorization service.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/probe_programs.py
"""
import json
import sys

from g_common import C, HostileServer, action, brief, real_policy, report, scratch, staging_task, summary

from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.failures import RetryPolicy
from cam_live.harness.http_api import ApiModality, HttpApiExecutor
from cam_live.harness.internal import InternalExecutor

A, B = HostileServer("A"), HostileServer("B")
J = {"Content-Type": "application/json"}
A.routes["/w"] = lambda *a: (200, J, json.dumps({"ok": True}))
B.routes["/w"] = lambda *a: (200, J, "{}")


def world(tag, **resources):
    run = scratch("prog_" + tag)
    task = staging_task("g_env_prog_" + tag, **resources)
    pol = real_policy(run, task)
    env = ComputerEnvironment(run, pol, secrets=pol.secrets_handle())
    env.register(InternalExecutor(run, clock=env.clock))
    env.register(ApiModality([HttpApiExecutor(retry=RetryPolicy(rules={}))]))
    return env, pol, task


def post(url, i=0):
    return action("http.request", C.Modality.API, method="POST", url=url + "/w", json={"i": i})


def program(steps, **kw):
    return action("program", C.Modality.INTERNAL, level="procedure", steps=steps, **kw)


def run(env, pol, task, a):
    dec = pol.authorize(a, task, {})
    return dec, env.execute(a, dec)


env, pol, task = world("mix", deny=[B.host])
p = program([action("internal.note", C.Modality.INTERNAL, text="before"), post(A.url, 1),
             action("wait", C.Modality.INTERNAL, seconds=1, block=False), post(A.url, 2)])
dec, res = run(env, pol, task, p)
ok1 = res.status == "ok" and A.count("/w", "POST") == 2 and [o.modality for o in res.ops] == ["internal", "api", "internal", "api"]
bad = program([post(A.url, 3), post(B.url, 4), post(A.url, 5)])
dec2, res2 = run(env, pol, task, bad)
sent_before_refusal = A.count("/w", "POST") - 2
deep = program([program([program([program([post(A.url)])])])])
dec3, res3 = run(env, pol, task, deep)
unknown = program([action("totally.unknown", C.Modality.API)])
dec4, res4 = run(env, pol, task, unknown)
vprog = action("wait", C.Modality.INTERNAL, seconds=0, verify=C.VerificationCondition("", "program", {"steps": [post(A.url, 9).to_dict()]}))
n_before = A.count("/w", "POST")
dec5, res5 = run(env, pol, task, vprog)
denied_all = dec.verdict == "deny" and "tool table" in " ".join(dec.reasons)
report("PG-PROGRAMS-REAL-POLICY", "DEFECT" if denied_all else "HOLDS",
       "ActionSpec(operation='program', steps=[...]) - the environment's only way to run a composition of steps - under "
       "the real authorization service with the default policy: "
       + ("every program is DENIED at authorization because the operation 'program' is not in the policy's tool table "
          "(default_policy.yaml 'tools:'); EB's program tests pass only with EB's own test policy" if denied_all else
          "programs are authorized"),
       mixed_modalities_program=f"{dec.verdict} -> {brief(res)[:120]}", posts_sent=A.count("/w", "POST"),
       with_forbidden_step=f"{dec2.verdict} -> {brief(res2)[:60]}", forbidden_destination_requests=B.count("/w"),
       verification_program_with_write=f"verified={res5.verified}, posts={A.count('/w', 'POST') - n_before}")
env.close()

# the write rate limit: are writes inside programs counted like single writes?
env, pol, task = world("rate")
n0, sent, refused = A.count("/w", "POST"), 0, ""
for k in range(8):
    dec, res = run(env, pol, task, program([post(A.url, k * 10 + j) for j in range(4)]))
    if res.status != "ok":
        refused = f"{dec.verdict} -> {brief(res)[:120]}"
        break
sent = A.count("/w", "POST") - n0
report("PG-RATE-LIMIT", "DEFECT" if sent > 10 else ("NOTE" if sent == 0 else "HOLDS"),
       f"default policy, write limit 10 per hour per destination: programs of 4 POSTs each sent {sent} POSTs before a refusal",
       posts_sent=sent, first_refusal=refused)
env.close()
A.close(); B.close()
sys.exit(summary("probe_programs"))
