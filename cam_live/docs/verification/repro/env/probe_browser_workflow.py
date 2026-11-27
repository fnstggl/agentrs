"""G-env probe 3: EB's own multi-step staging workflow, authorized by the real PolicyService (default policy).

    cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/probe_browser_workflow.py

The EB test-suite runs this workflow with its test policy (GrantPolicy). Here the same steps on the same staging
site (cam_live.harness.staging, structured site) go through ComputerEnvironment + BrowserExecutor (real Chromium)
with workstream EA's PolicyService and default_policy.yaml, task mode "staging".
"""
import sys

from g_common import C, Drv, brief, real_policy, report, scratch, staging_task, summary

from cam_live.harness.browser import BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor
from cam_live.harness.staging import StagingSites

sites = StagingSites().start()
s = sites.urls["structured"]
run = scratch("workflow")
task = staging_task("g_env_workflow")
pol = real_policy(run, task)
env = ComputerEnvironment(run, pol, secrets=pol.secrets_handle())
env.register(InternalExecutor(run, clock=env.clock))
env.register(BrowserExecutor(run, clock=env.clock))
d = Drv(env, pol, task)
V = C.VerificationCondition
count = {"interactions_ok": 0, "workflows_completed": 0}
first_refusal = None


def step(op, **kw):
    global first_refusal
    r = d.run(op, **kw)
    if r.status == "ok":
        if op not in ("browser.navigate", "browser.inspect"):
            count["interactions_ok"] += 1
        return r
    if first_refusal is None:
        first_refusal = (op, brief(r))
    raise StopIteration


def workflow(i):
    step("browser.navigate", url=s + "/items/new")
    step("browser.fill", role="textbox", name="Item name", text=f"G item {i}")
    step("browser.select_option", role="combobox", name="Category", label="Misc")
    step("browser.fill", role="spinbutton", name="Quantity", text="7")
    r = step("browser.click", role="button", name="Continue", verify=V("", "text_present", {"text": "Step 2 of 3"}))
    assert r.verified is True
    step("browser.click", role="radio", name="High")
    step("browser.click", role="checkbox", name="Fragile")
    step("browser.fill", role="textbox", name="Notes", text="n")
    step("browser.click", role="button", name="Continue", verify=V("", "url_matches", {"pattern": r"/items/new/review\?draft="}))
    r = step("browser.click", role="button", name="Confirm", level=C.Level.INTERVENTION,
             verify=V("", "element_state", {"role": "status", "state": "visible"}))
    assert r.verified is True
    count["workflows_completed"] += 1


n0 = len(sites.structured.items)
try:
    for i in range(3):
        workflow(i)
except StopIteration:
    pass
created = len(sites.structured.items) - n0
report("BR-WORKFLOW-REAL-POLICY", "DEFECT" if count["workflows_completed"] < 3 else "HOLDS",
       "EB's three-step staging workflow (9 interaction operations) repeated under the real authorization service "
       "with the default policy: default_policy.yaml rate_limits.write = 10 per 3600 s per destination counts every "
       "click / fill / select on an owned site, so the second workflow is refused part-way",
       workflows_completed=count["workflows_completed"], interactions_ok=count["interactions_ok"],
       items_created_on_site=created, first_refusal=first_refusal,
       policy_limits=str(pol.limits_state())[:300])
env.close()
sites.stop()
sys.exit(summary("probe_browser_workflow"))
