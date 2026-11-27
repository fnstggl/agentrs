"""G-env probe 6: generic HTTP API executor, publishing adapter, failure taxonomy / retries, modality selector (ES).

    cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/probe_api_publish_modality.py

Real ComputerEnvironment + PolicyService (default policy unless stated) + HttpApiExecutor + PublishingExecutor.
Destinations are local HTTP servers that count what they receive. The Discord adapter is not touched.
"""
import json
import os
import sys
import time

from g_common import C, CAM, HostileServer, action, brief, report, scratch, staging_task, summary

from cam_live.harness.auth import PolicyService
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.failures import FailureEvidence, RetryPolicy, classify
from cam_live.harness.http_api import ApiModality, HttpApiExecutor
from cam_live.harness.internal import InternalExecutor
from cam_live.harness.modality import ExecContextFeatures, ModalitySelector, Realisation
from cam_live.harness.publishing import Destination, PublishingExecutor

A, B, E = HostileServer("A"), HostileServer("B"), HostileServer("E")
J = {"Content-Type": "application/json"}
SECRET = "k3y-g-env-PROBE-51f0aa77"
created = []


def create(m, p, q, h, b):
    created.append(b)
    return 503, J, json.dumps({"error": "upstream hiccup", "stored": len(created)})


def create_slow(m, p, q, h, b):
    created.append(b)
    time.sleep(2.5)
    return 200, J, "{}"


A.routes["/create-503"] = create
A.routes["/create-slow"] = create_slow
A.routes["/forbidden"] = lambda *a: (403, J, json.dumps({"error": "forbidden"}))
A.routes["/legal"] = lambda *a: (451, J, json.dumps({"error": "unavailable for legal reasons"}))
A.routes["/limited"] = lambda *a: (429, dict(J, **{"Retry-After": "600"}), json.dumps({"error": "slow down"}))
A.routes["/ok"] = lambda *a: (200, J, json.dumps({"ok": True, "members": 17}))
A.routes["/to-b"] = lambda *a: (302, {"Location": B.url + "/landing"}, "")
A.routes["/to-e"] = lambda *a: (307, {"Location": E.url + "/landing"}, "")
B.routes["/landing"] = lambda *a: (200, J, "{}")
E.routes["/landing"] = lambda m, p, q, h, b: (200, J, json.dumps({"got_authorization_header": "Authorization" in h or "authorization" in h}))
A.routes["/echo-body"] = lambda m, p, q, h, b: (200, J, json.dumps({"you_sent": h.get("Authorization", "")}))
A.routes["/echo-etag"] = lambda m, p, q, h, b: (200, dict(J, ETag=h.get("Authorization", "")), json.dumps({"ok": 1}))
A.routes["/echo-upper"] = lambda m, p, q, h, b: (200, J, json.dumps({"you_sent": h.get("Authorization", "").upper()}))
A.routes["/echo-loc"] = lambda m, p, q, h, b: (201, dict(J, Location="/items/1?api_key=" + q.get("api_key", "")), json.dumps({"ok": 1}))
A.routes["/canon"] = lambda m, p, q, h, b: ((302, {"Location": "/canonical?api_key=" + q.get("api_key", "")}, "") )
A.routes["/canonical"] = lambda *a: (200, J, json.dumps({"ok": 1}))
A.routes["/huge"] = lambda *a: (200, J, json.dumps({"blob": "x" * 5_000_000, "n": 5}))


class StripPrefix:
    """The real ExecutorSecrets handle of the authorization service; only removes the 'secret:' prefix from the
    name before asking the vault (the step missing between the executors and the service, see HT-SECRET-REFERENCE)."""

    def __init__(self, inner):
        self._inner = inner

    def get(self, name, action, decision, destination):
        return self._inner.get(str(name).split("secret:", 1)[-1], action, decision, destination)

    def scrub(self, text):
        return self._inner.scrub(text)


class World:
    def __init__(self, tag, policy_path=None, retry=None, site=False, shim=False, **resources):
        self.run = scratch("api_" + tag)
        self.site = os.path.join(self.run, "site")
        if site:
            os.makedirs(self.site)
            resources.setdefault("owned", []).append(f"file:{self.site}")
        self.task = staging_task("g_env_api_" + tag, **resources)
        self.pol = PolicyService(self.run, policy_path, fsync=False)
        self.pol.register_task(self.task, "operator")
        self.pol.put_secret("api_key", SECRET, [A.host])
        handle = StripPrefix(self.pol.secrets_handle()) if shim else self.pol.secrets_handle()
        self.env = ComputerEnvironment(self.run, self.pol, secrets=handle)
        members = [HttpApiExecutor(retry=retry, timeout_s=1.5)]
        if site:
            self.pub = PublishingExecutor(os.path.join(self.run, "state"),
                                          [Destination("site", "static_site", root=self.site, base_url=A.url,
                                                       access_log=os.path.join(self.run, "access.log"))], clock=self.env.clock)
            members.append(self.pub)
        self.env.register(InternalExecutor(self.run, clock=self.env.clock))
        self.env.register(ApiModality(members))

    def go(self, op, aid="", timeout=60.0, **args):
        a = action(op, C.Modality.API, aid=aid, timeout=timeout, **args)
        return self.submit(a)

    def submit(self, a):
        dec = self.pol.authorize(a, self.task, {})
        res = self.env.execute(a, dec)
        res._action, res._dec = a, dec
        return res

    def checks(self, res):
        return [(r["op"]["op"], r["op"]["arguments"].get("url", r["op"]["arguments"].get("destination", "")), r["verdict"])
                for r in self.env.records("env.primitive_check", 500, res._action.action_id)]

    def close(self):
        self.env.close()


# ---- HT-1 retries -------------------------------------------------------------------------------------------
w = World("retry", deny=[B.host])
created.clear()
r = w.go("http.request", method="POST", url=A.url + "/create-503", json={"title": "one new thing"})
n503 = len(created)
created.clear()
r2 = w.go("http.request", method="POST", url=A.url + "/create-slow", json={"title": "one new thing"})
time.sleep(3)
nslow = len(created)
report("HT-POST-RETRY", "DEFECT" if (n503 > 1 or nslow > 1) else "HOLDS",
       "a POST (not idempotent, no idempotency key) whose effect the server applied is re-sent automatically after a "
       f"503 ({n503} sends) and after a client-side timeout ({nslow} sends); each re-send passes its own primitive check",
       after_503=brief(r)[:60], retries_503=r.retries, sends_503=n503, after_timeout=brief(r2)[:60],
       retries_timeout=r2.retries, sends_timeout=nslow, primitive_checks_503=len(w.checks(r)))
vals = {}
before_b = B.count("/landing")
dn = w.go("http.get", url=B.url + "/landing")
vals["policy deny"] = {"status": dn.status, "class": dn.failure_class, "requests": B.count("/landing") - before_b, "retries": dn.retries}
rb = w.go("http.get", url=A.url + "/to-b")
vals["redirect to forbidden"] = {"status": rb.status, "class": rb.failure_class, "requests_to_forbidden": B.count("/landing") - before_b,
                                 "checks": w.checks(rb)}
for name, path in (("403", "/forbidden"), ("451", "/legal"), ("429 retry-after 600", "/limited")):
    before = A.count(path)
    x = w.go("http.get", url=A.url + path)
    vals[name] = {"class": x.failure_class, "requests": A.count(path) - before, "retries": x.retries}
after_429 = w.go("http.get", url=A.url + "/ok")
vals["next request to the host that said Retry-After: 600"] = {"status": after_429.status, "class": after_429.failure_class,
                                                               "requests": A.count("/ok")}
ok = (vals["403"] == {"class": "missing_permission", "requests": 1, "retries": 0}
      and vals["451"] == {"class": "platform_restriction", "requests": 1, "retries": 0}
      and vals["429 retry-after 600"]["requests"] == 1 and vals["429 retry-after 600"]["class"] == "resource_limit"
      and vals["policy deny"]["requests"] == 0 and vals["policy deny"]["status"] == "denied"
      and vals["redirect to forbidden"]["requests_to_forbidden"] == 0 and vals["redirect to forbidden"]["status"] == "denied")
report("HT-HARD-NEVER-RETRIED", "HOLDS" if ok else "DEFECT",
       "refusals are final: HTTP 403 / 451 are sent once and classified hard, a 429 with a long Retry-After is not "
       "retried, a policy denial sends nothing, and a redirect to a forbidden destination is checked BEFORE the hop "
       "is followed", detail=vals)
w.close()

# ---- HT-2 secrets by reference --------------------------------------------------------------------------------
w = World("secret")
lit = w.go("http.request", method="GET", url=A.url + "/ok", headers={"Authorization": "Bearer " + SECRET})
lit2 = w.go("http.request", method="GET", url=A.url + "/ok", headers={"X-Api-Key": SECRET})
lit3 = w.go("http.get", url=A.url + "/ok?api_key=" + SECRET)
by_name = w.go("http.get", url=A.url + "/echo-body", auth={"secret": "api_key"})
by_ref = w.go("http.get", url=A.url + "/echo-body", auth={"secret": "secret:api_key"})
usable = by_name if by_name.status == "ok" else by_ref
form = "name" if by_name.status == "ok" else ("secret:name" if by_ref.status == "ok" else "none")
report("HT-LITERAL-CREDENTIALS", "HOLDS" if all(x.status != "ok" for x in (lit, lit2, lit3)) else "DEFECT",
       "literal credentials in a header or in the URL are refused by the authorization service",
       authorization_header=brief(lit)[:90], api_key_header=brief(lit2)[:90], in_query=brief(lit3)[:90])
report("HT-SECRET-REFERENCE", "HOLDS" if form != "none" else "DEFECT",
       "http.get / http.request with auth = {secret: <name>} under the real authorization service: the form the tool "
       "schema documents is refused as a literal credential, and the form the service asks for ('secret:<name>') is "
       "passed to the vault with its prefix, which has no such secret - the generic API executor cannot use a stored "
       "credential at all",
       literal_authorization_header=brief(lit)[:110], literal_api_key_header=brief(lit2)[:110],
       literal_in_query=brief(lit3)[:110], auth_secret_by_name=brief(by_name)[:110],
       auth_secret_reference_form=brief(by_ref)[:110], working_form=form)
w.close()
w = World("secret2", shim=True)             # scrubbing / binding checks need a credential that is actually sent
if True:
    ref = "secret:api_key"
    leaks = {}
    log_path = os.path.join(w.run, "events.jsonl")
    for name, path, kw in (("echoed in body", "/echo-body", {}), ("echoed in ETag header", "/echo-etag", {}),
                           ("echoed upper-cased in body", "/echo-upper", {}),
                           ("query credential echoed in Location header", "/echo-loc", {"scheme": "query", "name": "api_key"}),
                           ("query credential echoed by a redirect", "/canon", {"scheme": "query", "name": "api_key"})):
        off = os.path.getsize(log_path)
        x = w.go("http.get", url=A.url + path, auth=dict({"secret": ref}, **kw))
        blob = json.dumps(x.to_dict(), default=str)
        with open(log_path) as f:
            f.seek(off)
            new_log = f.read()
        leaks[name] = {"status": x.status, "in_result": SECRET in blob or SECRET.upper() in blob,
                       "in_events_log": SECRET in new_log or SECRET.upper() in new_log,
                       "detail": x.failure_detail[:70]}
    audit = json.dumps(w.pol.audit_tail(400), default=str)
    leaks["authorization-service audit log contains the value"] = SECRET in audit
    cross = w.go("http.get", url=E.url + "/landing", auth={"secret": ref})
    red = w.go("http.get", url=A.url + "/to-e", auth={"secret": ref})
    e_saw = [h for (m, p, h, b) in E.log if SECRET in json.dumps(h)]
    bad = [k for k, v in leaks.items() if isinstance(v, dict) and (v["in_result"] or v["in_events_log"])] + \
          (["audit"] if leaks["authorization-service audit log contains the value"] else [])
    report("HT-SECRET-SCRUB", "DEFECT" if bad else "HOLDS",
           "a destination that echoes the credential it was sent: which echoes reach the ActionResult returned to "
           "the learner or the logs", leaked_through=bad, detail=leaks)
    report("HT-SECRET-BINDING", "HOLDS" if (cross.status != "ok" and not e_saw) else "DEFECT",
           "a secret bound to one destination is not released for another, and is not forwarded across a redirect "
           "to another origin", other_destination=brief(cross)[:120], redirect_to_other_origin=brief(red)[:80],
           other_destination_received_the_secret=bool(e_saw))
big = w.go("http.get", url=A.url + "/huge")
blob = json.dumps(big.to_dict(), default=str)
report("HT-SIZE", "HOLDS" if len(blob) < 200_000 else "DEFECT", "a 5 MB response is cut to a bounded observation",
       result=brief(big)[:40], result_json_bytes=len(blob), truncated=big.observations[0].data.get("truncated") if big.observations else None)
bad_scheme = [w.go("http.get", url=u) for u in ("file:///etc/passwd", "ftp://127.0.0.1/x", "http://user:pw@127.0.0.1/x")]
portal = action("http.get", C.Modality.API, url="http://127.0.0.1:11111/capabilities")
report("HT-DESTINATIONS", "HOLDS" if all(x.status != "ok" for x in bad_scheme) else "DEFECT",
       "non-http schemes and embedded credentials are refused; destinations are otherwise whatever the authorization "
       "service allows (the executor has no list of its own) - in staging mode that includes every loopback port",
       refused=[brief(x)[:70] for x in bad_scheme],
       authorize_only_instance_portal=w.pol.authorize(portal, w.task, {}).verdict)
w.close()

# ---- PB publishing --------------------------------------------------------------------------------------------
pol_src = open(os.path.join(CAM, "src", "cam_live", "harness", "auth", "default_policy.yaml")).read()
w = World("publish", site=True)
bad_paths = {p: w.go("publish.draft", destination="site", path=p, content="x").status
             for p in ("../escape.html", "/etc/cron.d/x", "a/../../b.html", ".htaccess", "sub/.hidden", "a\\b.html", "")}
os.symlink("/tmp", os.path.join(w.site, "ln"))
bad_paths["ln/evil.html (symlinked directory)"] = w.go("publish.draft", destination="site", path="ln/g_env_evil.html", content="x").status
elsewhere = w.go("publish.draft", destination="not-mine", path="a.html", content="x")
dr = w.go("publish.draft", destination="site", path="index.html", content="<h1>first</h1>")
did = dr.artifacts.get("draft_id", "")
unprev = w.go("publish.submit", draft_id=did)
pv = w.go("publish.preview", draft_id=did)
wrong_dest = w.go("publish.submit", draft_id=did, destination="file:/etc/passwd")
sub = w.go("publish.submit", draft_id=did, destination=pv.observations[0].data["where"]["address"])
live = open(os.path.join(w.site, "index.html")).read() if os.path.exists(os.path.join(w.site, "index.html")) else None
mtime = os.path.getmtime(os.path.join(w.site, "index.html")) if live else 0
again = w.go("publish.submit", draft_id=did, destination=pv.observations[0].data["where"]["address"])
same_id = w.submit(sub._action)
report("PB-FLOW", "HOLDS" if (all(v != "ok" for v in bad_paths.values()) and elsewhere.status != "ok" and unprev.status != "ok"
                              and wrong_dest.status != "ok" and sub.status == "ok" and live == "<h1>first</h1>"
                              and again.status == "ok" and os.path.getmtime(os.path.join(w.site, "index.html")) == mtime
                              and not os.path.exists("/tmp/g_env_evil.html")) else "DEFECT",
       "draft -> preview -> submit on an owned static site: paths leaving the site root (.., absolute, hidden, "
       "symlinked directory) and unknown destinations are refused, an unpreviewed draft is not published, a stated "
       "destination must be the draft's, the file equals the previewed bytes, publishing again does not rewrite it",
       bad_paths=bad_paths, unknown_destination=brief(elsewhere)[:70], unpreviewed=brief(unprev)[:80],
       wrong_destination=brief(wrong_dest)[:90], submit=brief(sub)[:30], submit_authorization=sub._dec.verdict,
       submit_checks=w.checks(sub), resubmit=again.observations[0].text[:60] if again.observations else brief(again),
       same_action_again=same_id.status, trust=[o.provenance.trust for o in sub.observations])
report("PB-APPROVAL-DEFAULT", "NOTE",
       "under the default staging policy a publication to an owned static site is authorized without operator "
       "approval (publishing.require_approval = unless_preauthorized and the task's own file root counts as such)",
       authorization=sub._dec.verdict, reasons=sub._dec.reasons[:2])
with open(os.path.join(w.run, "access.log"), "w") as f:
    for i in range(6):
        f.write(f'10.0.0.{i % 3} - - [27/Nov/2026:10:00:0{i} +0000] "GET /index.html HTTP/1.1" 200 120 "http://ref.example/x" "Mozilla/5.0"\n')
    f.write('10.0.0.9 - - [27/Nov/2026:10:00:07 +0000] "GET /index.html HTTP/1.1" 200 120 "-" "curl/8.0"\n')
    f.write('10.0.0.9 - - [27/Nov/2026:10:00:08 +0000] "GET /style.css HTTP/1.1" 200 120 "-" "Mozilla/5.0"\n')
    f.write('10.0.0.3 - - [27/Nov/2026:10:00:09 +0000] "GET /missing HTTP/1.1" 404 10 "-" "Mozilla/5.0"\n')
    f.write('garbage line\n')
an = w.go("publish.get_analytics", destination="site")
q = {x.name: x.value for x in an.observations[0].quantities} if an.observations else {}
blob = json.dumps(an.to_dict(), default=str)
report("PB-ANALYTICS", "HOLDS" if (q.get("site.site.page_views") == 6 and q.get("site.site.visitors") == 3
                                   and q.get("site.site.page_views.declared_automated") == 1 and q.get("site.site.requests") == 9
                                   and "10.0.0." not in blob and an.observations[0].provenance.trust == "untrusted") else "DEFECT",
       "site analytics from an access log: counts only (page views 6, distinct clients 3, declared automated 1, "
       "requests 9), no client address in the result, labelled untrusted", quantities=q, client_addresses_in_result="10.0.0." in blob)
w.close()

# approval and what the approval is bound to (operator policy: staging writes are NOT pre-authorized)
needle = 'preauthorized: ["127.0.0.1", "localhost", "[::1]", "channel:staging:*", "account:staging:*"]'
custom = os.path.join(scratch("api_policy"), "policy.yaml")
open(custom, "w").write(pol_src.replace(needle, "preauthorized: []").replace("require_approval: unless_preauthorized", "require_approval: always")
                        .replace("policy_id: default-1", "policy_id: g-env-publish-approval"))
w = World("approval", policy_path=custom, site=True)
dr = w.go("publish.draft", destination="site", path="page.html", content="APPROVED TEXT A")
did = dr.artifacts.get("draft_id", "")
pv = w.go("publish.preview", draft_id=did)
addr = pv.observations[0].data["where"]["address"] if pv.observations else ""
sub_action = action("publish.submit", C.Modality.API, draft_id=did, destination=addr)
first = w.submit(sub_action)
page = os.path.join(w.site, "page.html")
queue = w.pol.approvals()
shown = json.dumps(queue, default=str)
entry = queue[0] if queue else {}
facts = {"approval_entry_keys": sorted(entry), "approval_entry_action_arguments": str((entry.get("action") or {}).get("arguments", entry.get("preview", "")))[:300],
         "draft": brief(dr)[:40], "first_submit": brief(first)[:120], "first_authorization": first._dec.verdict,
         "published_before_approval": os.path.exists(page), "approvals_queued": [(x.get("kind"), x.get("policy_ids")) for x in queue],
         "approval_entry_shows_content": "APPROVED TEXT A" in shown}
if queue:
    # the learner changes the draft while the approval is pending, then the operator approves what was shown
    upd = w.go("publish.update", draft_id=did, content="UNSEEN TEXT B", destination=addr)
    facts["update_while_pending"] = brief(upd)[:130]
    facts["update_authorization"] = upd._dec.verdict
    pv2 = w.go("publish.preview", draft_id=did)
    facts["approvals_after_update"] = [(x.get("kind"), x.get("policy_ids")) for x in w.pol.approvals()]
    ap_id = first._dec.approval_id or first.artifacts.get("approval_id", "") or queue[0].get("approval_id", "")
    dec2 = w.pol.resolve_approval(ap_id, True, by="operator")
    facts["approved_decision"] = dec2.verdict
    after = w.env.execute(sub_action, dec2) if dec2.verdict == "allow" else w.submit(sub_action)
    after._action = sub_action
    facts["submit_after_approval"] = brief(after)[:150]
    facts["checks_after_approval"] = w.checks(after)[-2:]
    facts["file_content_now"] = open(page).read() if os.path.exists(page) else None
    swapped = facts["file_content_now"] == "UNSEEN TEXT B"
    report("PB-APPROVAL-BINDING", "DEFECT" if swapped else "HOLDS",
           "approval queue with 'publishing.require_approval: always': a draft is submitted, waits for approval, is "
           "changed while waiting, and the operator then approves the request that showed the first text. "
           + ("The CHANGED text was published under that approval." if swapped else
              "The changed text is not published under the approval given for the first text."), **facts)
else:
    report("PB-APPROVAL-BINDING", "NOTE", "no approval was queued for the publication under this policy", **facts)
w.close()

# the same sequence with ES change request 3 applied to the operator policy (publish.update classified as a local
# draft operation - the exact line ES proposes in docs/change_requests/ES.md)
old_line = 'ops: [publish.draft, publish.preview, "publish.list*", "publish.get*", publish.status,'
assert old_line in pol_src
custom3 = os.path.join(scratch("api_policy3"), "policy.yaml")
open(custom3, "w").write(pol_src.replace(needle, "preauthorized: []").replace("require_approval: unless_preauthorized", "require_approval: always")
                         .replace(old_line, 'ops: [publish.draft, publish.preview, publish.update, "publish.list*", "publish.get*", publish.status,')
                         .replace("policy_id: default-1", "policy_id: g-env-es3"))
w = World("approval_es3", policy_path=custom3, site=True)
dr = w.go("publish.draft", destination="site", path="page.html", content="APPROVED TEXT A")
did = dr.artifacts.get("draft_id", "")
pv = w.go("publish.preview", draft_id=did)
addr = pv.observations[0].data["where"]["address"] if pv.observations else ""
sub_action = action("publish.submit", C.Modality.API, draft_id=did, destination=addr)
first = w.submit(sub_action)
queue = w.pol.approvals()
page = os.path.join(w.site, "page.html")
f3 = {"first_submit": first.status, "approvals_queued": len(queue),
      "operator_is_shown": str((queue[0].get("preview") or {}).get("content"))[:160] if queue else None}
if queue:
    upd = w.go("publish.update", draft_id=did, content="UNSEEN TEXT B")
    pv2 = w.go("publish.preview", draft_id=did)
    f3["update_while_pending"] = f"{upd._dec.verdict} -> {upd.status}"
    dec2 = w.pol.resolve_approval(first._dec.approval_id or queue[0].get("approval_id", ""), True, by="operator")
    after = w.env.execute(sub_action, dec2)
    f3["submit_after_approval"] = brief(after)[:80]
    f3["file_content_now"] = open(page).read() if os.path.exists(page) else None
    swapped = f3["file_content_now"] == "UNSEEN TEXT B"
    report("PB-APPROVAL-BINDING-ES3", "DEFECT" if swapped else "HOLDS",
           "the approval of publish.submit is bound to the action (draft id + destination), and the queue entry shows "
           "the operator the draft id, not the text. It holds today only because publish.update is itself treated as a "
           "publication. With ES's own pending change request ES-3 applied (publish.update = local draft operation): "
           + ("the text was replaced while the approval was pending and the REPLACED text was published under the "
              "approval given before the change." if swapped else "the replaced text was not published."), **f3)
w.close()

# ---- MS modality selector and failure taxonomy ---------------------------------------------------------------
sel = ModalitySelector(seed=0)
ctx = ExecContextFeatures(tool="request")
api = action("http.request", C.Modality.API, method="POST", url=A.url + "/x")
brw = action("browser.click", C.Modality.STRUCTURED_BROWSER, role="button", name="Send")
api.target = brw.target = A.url + "/x"
api.expected = brw.expected = "the thing is sent"
opts = [Realisation("api", api), Realisation("structured_browser", brw)]
denied = C.ActionResult(action_id=api.action_id, status="denied", failure_class="missing_permission", failure_detail="not allowed")
ch = sel.choose(api, opts, ctx, rule="greedy")
before = sel.estimate("api", ctx).p_success
sel.update_choice(ch, ctx, denied)
after = sel.estimate("api", ctx).p_success
ch2 = sel.choose(api, opts, ctx)                  # the same intent again
restored = ModalitySelector(seed=1)
restored.load_state_dict(json.loads(json.dumps(sel.state_dict())))
ch3 = restored.choose(api, opts, ctx)
forbidden = C.ActionResult(action_id=api.action_id, status="failed", failure_class="platform_restriction")
sel2 = ModalitySelector(seed=0)
c0 = sel2.choose("intent-1", opts, ctx, rule="greedy")
sel2.update_choice(c0, ctx, forbidden)
c1 = sel2.choose("intent-1", opts, ctx)
report("MS-NO-SWITCH", "HOLDS" if (ch2.blocked and not ch2.options and ch3.blocked and c1.blocked and before == after) else "DEFECT",
       "after a denial (or a platform restriction) of one realisation of an intent the selector offers NO realisation "
       "of that intent, also after a save / load; the denial is not counted as evidence about the modality",
       second_choice_blocked=ch2.blocked, reason=ch2.reason[:110], after_reload_blocked=ch3.blocked,
       platform_restriction_blocked=c1.blocked, p_success_before_after=[before, after])
# the same operation proposed again under a new decision id
api2 = C.ActionSpec.from_dict(api.to_dict()); brw2 = C.ActionSpec.from_dict(brw.to_dict())
api.parent_decision_id = brw.parent_decision_id = "dec_0001"
sel3 = ModalitySelector(seed=0)
k0 = sel3.choose(api, [Realisation("api", api), Realisation("structured_browser", brw)], ctx, rule="greedy")
sel3.update_choice(k0, ctx, denied)
api2.parent_decision_id = brw2.parent_decision_id = "dec_0002"
k1 = sel3.choose(brw2, [Realisation("api", api2), Realisation("structured_browser", brw2)], ctx)
sel5 = ModalitySelector(seed=0)
api3 = C.ActionSpec.from_dict(api2.to_dict()); brw3 = C.ActionSpec.from_dict(brw2.to_dict())
api3.parent_decision_id = brw3.parent_decision_id = ""
m0 = sel5.choose(api3, [Realisation("api", api3), Realisation("structured_browser", brw3)], ctx, rule="greedy")
sel5.update_choice(m0, ctx, denied)
m1 = sel5.choose(brw3, [Realisation("structured_browser", brw3)], ctx)      # the browser realisation named as the intent
sel4 = ModalitySelector(seed=0)
sel4.update("api", ctx, denied)                    # update() without the intent
k2 = sel4.choose(api2, [Realisation("api", api2), Realisation("structured_browser", brw2)], ctx)
report("MS-BLOCK-SCOPE", "DEFECT" if (not k1.blocked or not k2.blocked or not m1.blocked) else "HOLDS",
       "how far the block reaches: it is keyed by intent_key(), which for an ActionSpec is its parent_decision_id, "
       "else '<operation suffix>|target|expected'. So (a) the identical operation on the identical target proposed "
       "under the next decision id is offered again with every modality, (b) when the other realisation's own "
       "ActionSpec is given as the intent (browser.click instead of http.request, same target, same expectation) it "
       "is offered although the API route to that target was just denied, (c) update() without an intent records "
       "no block at all",
       next_decision_blocked=k1.blocked, next_decision_offers=[o["kind"] for o in k1.options],
       other_realisation_as_intent_blocked=m1.blocked, other_realisation_offers=[o["kind"] for o in m1.options],
       update_without_intent_blocked=k2.blocked)
rp = RetryPolicy()
hard_rules = []
for cls in C.FailureClass.HARD:
    try:
        RetryPolicy(rules={cls: __import__("cam_live.harness.failures", fromlist=["RetryRule"]).RetryRule(max_retries=1)})
        hard_rules.append(cls)
    except ValueError:
        pass
dec = [rp.decide(c, 0).retry for c in C.FailureClass.HARD]
bounds = {c: sum(1 for n in range(50) if rp.decide(c, n, retry_after_s=0.0).retry) for c in RetryPolicy.TRANSIENT}
mixed = classify(FailureEvidence(http_status=503, auth_verdict="deny", timed_out=True))
report("FT-RETRY-BOUNDS", "HOLDS" if (not hard_rules and not any(dec) and max(bounds.values()) <= 5 and mixed.hard and not mixed.retryable) else "DEFECT",
       "retry rules cannot be created for the hard classes, hard classes are never retried, transient classes are "
       "bounded, and a denial dominates any transient evidence that accompanies it",
       retries_allowed_per_class=bounds, hard_rule_accepted=hard_rules, denial_plus_timeout=[mixed.failure_class, mixed.retryable])
for s in (A, B, E):
    s.close()
sys.exit(summary("probe_api_publish_modality"))
