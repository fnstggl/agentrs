"""G2-integration probe p02 — internal.request_operator (OP-06): recorded, answered as an event, grants nothing.

Real authorization service (harness.auth.PolicyService) and computer environment (harness.service.build_environment,
no browser), the operator's CLI (python -m cam_live.harness.operator_requests), and the integrated agent on the tiny
CPU configuration for the ingestion step. Nothing leaves the machine: the refused action targets 127.0.0.1:9 and is
refused before execution (the probe asserts it was never executed).

Usage (from /workspace/camlive):
    PYTHONPATH=<copy>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2integration/p02_operator_requests.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

from cam_live import contracts as C
from cam_live.harness import operator_requests as oreq
from cam_live.harness.auth import PolicyService
from cam_live.harness.service import build_environment

SCRATCH = os.environ.get("G2I_SCRATCH", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/"
                                        "G2INTEGRATION/p02")
broken = 0


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def tree_hash(root, skip):
    h = hashlib.sha256()
    for d, _dirs, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(d, f)
            if any(s in p for s in skip):
                continue
            h.update(p.encode())
            try:
                h.update(open(p, "rb").read())
            except OSError:
                pass
    return h.hexdigest()


def main() -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="p02_", dir=SCRATCH)
    R = os.path.join(tmp, "run")
    svc = PolicyService(R, fsync=False)
    env = build_environment(R, svc, secrets=svc.secrets_handle(), browser=False)
    task = C.TaskSpec(objective_id="req", objective="exercise operator requests", mode="staging",
                      interfaces=["internal"],
                      limits=C.Limits(duration_s=3600, money=0, max_decisions=100, max_external_actions=10))
    tools = {t.name: t for t in env.discover_tools()}
    check("1 internal.request_operator is offered by the computer environment", "internal.request_operator" in tools)

    # an action the task does not permit (interface http is not authorized): refused before execution
    http_tool = next((n for n in tools if n.startswith("http.")), "")
    target = C.ActionSpec(action_id=C.new_id("a"), operation=http_tool or "http.get", modality=C.Modality.API,
                          arguments={"url": "http://127.0.0.1:9/g2i-probe", "method": "GET"})
    d0 = svc.authorize(target, task, {})
    check("2 baseline: the action outside the task's interfaces is not allowed", d0.verdict != "allow",
          f"{http_tool}: {d0.verdict} {list(d0.reasons)[:2]}")

    req = C.ActionSpec(action_id=C.new_id("a"), operation="internal.request_operator", modality=C.Modality.INTERNAL,
                       arguments={"kind": "permission", "title": "Permission to use the http interface",
                                  "description": "the http tool is refused for this task",
                                  "justification": "a fetch was refused by the authorization service"})
    res = env.execute(req, svc.authorize(req, task, {}))
    check("3 the request executes and is recorded", res.status == C.ResultStatus.OK, res.failure_detail)
    d = oreq.find_file(R)
    rows = oreq.OperatorRequests(d).open_requests()
    check("4 one open request with the learner's text and action id",
          len(rows) == 1 and rows[0]["action_id"] == req.action_id and rows[0]["kind"] == "permission")
    rid = rows[0]["request_id"]
    before = tree_hash(os.path.join(R, "auth"), skip=())          # the authorization service's whole state
    code = oreq.main(["--run-dir", R, "grant", rid, "--note", "granted: you may now use http"])
    check("5 the operator CLI grants it", code == 0)
    evs, cur = env.poll_events("")
    ans = [e for e in evs if (e.data or {}).get("event") == "operator.answer"]
    ctx = [e for e in evs if (e.data or {}).get("event") == "operator.request.context"]
    check("6 the answer returns once as an event observation with operator trust",
          len(ans) == 1 and ans[0].provenance.trust == C.Trust.OPERATOR and "granted" in ans[0].text)
    check("7 the learner's own request text comes back separately under self trust, never under the operator label",
          len(ctx) == 1 and ctx[0].provenance.trust == C.Trust.SELF and "Permission to use" not in ans[0].text,
          ans[0].text[:120] if ans else "")
    evs2, _ = env.poll_events(cur)
    check("8 delivered exactly once", not [e for e in evs2 if (e.data or {}).get("event") == "operator.answer"])

    # the grant changed nothing at the authorization service
    after = tree_hash(os.path.join(R, "auth"), skip=())           # after the grant and its delivery
    d1 = svc.authorize(dict_action(target), task, {})
    check("9 after the grant the same action is still not allowed (an answer grants nothing by itself)",
          d1.verdict == d0.verdict, f"{d0.verdict} -> {d1.verdict}")
    check("10 the authorization service's state (R/auth: keys, sqlite state, audit head) is byte-identical before and after the grant and its delivery", before == after)
    res_t = env.execute(dict_action(target), d1)
    check("11 executing with that decision does not run the action", res_t.status != C.ResultStatus.OK
          and not res_t.ops, f"{res_t.status} ops={len(res_t.ops)}")

    # the answer enters the learner's experience store (integrated agent, tiny CPU configuration)
    from cam_live.agent import CamLiveAgent
    from cam_live.config import CamLiveConfig
    cfg = CamLiveConfig.debug(seed=0)
    agent = CamLiveAgent(cfg, os.path.join(tmp, "agent"), clock=C.SimClock(1000.0), task=task)
    agent.ingest(ans + ctx)
    items = {it.ref: it for it in agent.store.items([e.obs_id for e in ans + ctx]) if it is not None} \
        if hasattr(agent.store, "items") else {}
    if not items:
        got = [agent.store.get_item(e.obs_id) for e in ans + ctx] if hasattr(agent.store, "get_item") else []
        items = {it.ref: it for it in got if it is not None}
    a_item, c_item = items.get(ans[0].obs_id), items.get(ctx[0].obs_id)
    check("12 the answer is an item of the learner's experience store with operator trust",
          a_item is not None and a_item.trust == C.Trust.OPERATOR, repr(getattr(a_item, "trust", None)))
    check("13 the request context is stored with self trust", c_item is not None and c_item.trust == C.Trust.SELF,
          repr(getattr(c_item, "trust", None)))
    check("14 the answer is in the context window of the next decision", ans[0].obs_id in agent._window())
    agent.close()
    print(f"== p02_operator_requests: {broken} broken of 14 checks ({tmp})")
    return 1 if broken else 0


def dict_action(a: C.ActionSpec) -> C.ActionSpec:
    return C.ActionSpec(action_id=C.new_id("a"), operation=a.operation, modality=a.modality,
                        arguments=dict(a.arguments))


if __name__ == "__main__":
    sys.exit(main())
