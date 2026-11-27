"""G-rt probe 04 — limits and stop conditions: duration, decisions, external actions, money, compute; the run ends
with the right status and never exceeds a limit by more than one in-flight action.

    .venv/bin/python cam_live/docs/verification/repro/rt/p04_limits.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, StandInAgent, make_slow_run, read_log, slow_task  # noqa: E402

from cam_live.contracts import (ActionSpec, AuthorizationDecision, AuthVerdict, Cost, Limits, RunFiles, SimClock,  # noqa: E402
                                new_id)
from cam_live.diagnostics.world import AllowAllPolicy  # noqa: E402
from cam_live.runtime.control import OperatorControl  # noqa: E402

R = Report("p04_limits")


def state(L):
    return json.load(open(os.path.join(L.run_dir, RunFiles.STATE)))


def finished(L):
    ev = [e for e in read_log(L.run_dir, RunFiles.EVENTS) if e["type"] == "objective_finished"]
    return ev[-1]["data"] if ev else {}


# ---- decisions ----
L = make_slow_run("p04dec", SimClock(), limits={"max_decisions": 5})
s = L.rt.run(max_iterations=200)
st = state(L)
R.check("1 max_decisions=5: status exhausted, exactly 5 decisions, checkpoint + journal written",
        s == "exhausted" and st["counters"]["decisions"] == 5 and len(L.env.executed) == 5
        and "max_decisions" in st["status_reason"] and bool(st["committed"]["ckpt"])
        and os.path.exists(os.path.join(L.run_dir, RunFiles.JOURNAL)), {"status": s, "reason": st["status_reason"],
                                                                         "decisions": st["counters"]["decisions"]})
R.check("1b state says the operator is awaited", st["awaiting_operator"] is True)

# ---- external actions ----
L = make_slow_run("p04ext", SimClock(), limits={"max_external_actions": 3})
s = L.rt.run(max_iterations=200)
st = state(L)
R.check("2 max_external_actions=3: status exhausted with exactly 3 executions", s == "exhausted" and len(L.env.executed) == 3
        and "max_external_actions" in st["status_reason"], {"status": s, "executed": len(L.env.executed)})

# ---- money: honest estimate, then an under-estimate ----
L = make_slow_run("p04money", SimClock(), env_kw={"money": 2.0, "est_money": 2.0}, limits={"money": 5.0, "max_decisions": 12})
s = L.rt.run(max_iterations=200)
st = state(L)
spent = 2.0 * len(L.env.executed)
R.check("3 money=5, cost 2 with an honest estimate: never over budget (refused before execution)",
        spent <= 5.0 and abs(st["counters"]["money"] - spent) < 1e-9, {"spent": spent, "status": s,
                                                                       "decisions": st["counters"]["decisions"]})
L = make_slow_run("p04money0", SimClock(), env_kw={"money": 2.0, "est_money": 0.0}, limits={"money": 5.0, "max_decisions": 12})
s = L.rt.run(max_iterations=200)
st = state(L)
spent = 2.0 * len(L.env.executed)
R.check("3b money=5, cost 2 but estimated 0: over by at most one action, then exhausted",
        s == "exhausted" and 5.0 <= spent <= 5.0 + 2.0 and "money" in st["status_reason"], {"spent": spent, "status": s})

# ---- duration ----
L = make_slow_run("p04dur", SimClock(), env_kw={"exec_seconds": 3600.0}, limits={"duration_s": 5000.0})
s = L.rt.run(max_iterations=200)
st = state(L)
el = st["limits"]["duration_s"]["used"]
R.check("4 duration_s=5000 with 3600 s actions: exhausted, over by at most one in-flight action",
        s == "exhausted" and len(L.env.executed) == 2 and el <= 5000 + 3600, {"elapsed": el, "executed": len(L.env.executed)})
L = make_slow_run("p04durw", SimClock(), script=("wait",), wait_seconds=10 ** 7, limits={"duration_s": 5000.0})
s = L.rt.run(max_iterations=200)
st = state(L)
R.check("4b a wait far beyond the deadline is cut at the deadline (not slept through)",
        s == "exhausted" and abs(st["limits"]["duration_s"]["used"] - 5000.0) < 2.0, {"elapsed": st["limits"]["duration_s"]["used"]})

# ---- compute ----
class SlowLearner(StandInAgent):
    def learn(self, budget_s):
        time.sleep(0.05)
        return super().learn(budget_s)


c = SimClock()
L = make_slow_run("p04cmp", c, agent=SlowLearner(c, script=("slow.act",), durable=False), limits={"compute_s": 0.3})
s = L.rt.run(max_iterations=200)
st = state(L)
R.check("5 compute_s=0.3 with 0.05 s of learner time per cycle: exhausted, over by at most one cycle",
        s == "exhausted" and "compute_s" in st["status_reason"] and st["counters"]["compute_s"] <= 0.3 + 0.1,
        {"compute_s": st["counters"]["compute_s"], "executed": len(L.env.executed)})

# ---- NaN / infinite limit ----
t = slow_task(max_decisions=12)
t.limits.money = float("nan")
L = make_slow_run("p04nan", SimClock(), env_kw={"money": 1000.0, "est_money": 1000.0}, task=t)
s = L.rt.run(max_iterations=200)
st = state(L)
started = [e for e in read_log(L.run_dir, RunFiles.EVENTS) if e["type"] == "run_started"][0]["data"]
R.check("6 a NaN money limit is either refused at start or reported as not enforced (never silently unlimited)",
        "money" in started["limits_not_enforced"], {"limits_not_enforced": started["limits_not_enforced"],
                                                    "spent": 1000.0 * len(L.env.executed), "status": s})

# ---- approvals approved in a batch vs. the limits ----
class ApprovalPolicy(AllowAllPolicy):
    """Every slow.act needs operator approval (G-rt stand-in)."""

    def __init__(self):
        super().__init__()
        self.q = {}

    def authorize(self, action, task, context=None):
        h = action.content_hash()
        if self._stopped:
            return AuthorizationDecision(AuthVerdict.DENY, h, ["stopped"])
        for e in self.q.values():
            if e["hash"] == h:
                return AuthorizationDecision({"approved": AuthVerdict.ALLOW, "rejected": AuthVerdict.DENY}.get(
                    e["status"], AuthVerdict.NEEDS_APPROVAL), h, [e["status"]], grant="g", approval_id=e["id"])
        aid = f"apr_{len(self.q) + 1}"
        self.q[aid] = {"id": aid, "hash": h, "status": "pending"}
        return AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, h, ["operator approval required"], approval_id=aid)

    def approvals(self):
        return [{"approval_id": e["id"], "status": e["status"]} for e in self.q.values() if e["status"] == "pending"]

    def resolve_approval(self, approval_id, approve, by, note=""):
        e = self.q[approval_id]
        e["status"] = "approved" if approve else "rejected"
        return AuthorizationDecision(AuthVerdict.ALLOW if approve else AuthVerdict.DENY, e["hash"], [e["status"]],
                                     grant="g", approval_id=approval_id)


class Distinct(StandInAgent):
    """Proposes slow.act with a different argument each time, so each proposal is a different action."""

    def propose(self, tools, k):
        self._t("propose", tools, k)
        t = {x.name: x for x in tools}["slow.act"]
        return [ActionSpec(action_id=new_id("act"), operation="slow.act", modality=t.modality, level=t.level,
                           arguments={"i": self.n_select}, est_cost=Cost(money=t.est_cost.money), provenance="g-standin")]


c, pol = SimClock(), ApprovalPolicy()
L = make_slow_run("p04apr", c, agent=Distinct(c, durable=False), policy=pol, env_kw={"money": 2.0, "est_money": 2.0},
                  limits={"max_external_actions": 2, "money": 4.0, "max_decisions": 50})
for _ in range(30):
    L.rt.step()
    if len(L.rt.approvals) >= 6:
        break
n_q = len(L.rt.approvals)
op = OperatorControl(L.run_dir, policy=pol)
for aid in list(L.rt.approvals):
    op.approve(aid)
for _ in range(10):
    L.rt.step()
st = state(L)
R.check("7 six queued actions approved together: executions stay within max_external_actions=2 (+1 in flight) and money=4 (+1 action)",
        len(L.env.executed) <= 3 and 2.0 * len(L.env.executed) <= 6.0,
        {"queued": n_q, "executed": len(L.env.executed), "limit_external_actions": 2,
         "money_spent": 2.0 * len(L.env.executed), "limit_money": 4.0, "status": st["status"]})

# ---- stop condition: authorization_revoked ----
class Revocable(AllowAllPolicy):
    revoked = False

    def authorize(self, action, task, context=None):
        if self.revoked:
            return AuthorizationDecision(AuthVerdict.DENY, action.content_hash(), ["authorization for this task was revoked"])
        return super().authorize(action, task, context)

    def limits_state(self):
        return {"tasks": {"obj_slow": {"unavailable": ["revoked"]}}} if self.revoked else {}


pol = Revocable()
t = slow_task(max_decisions=40)
t.stop_conditions = ["verified_success", "verified_failure", "authorization_revoked", "operator_stop", "emergency_stop"]
L = make_slow_run("p04rev", SimClock(), policy=pol, task=t)
for _ in range(3):
    L.rt.step()
pol.revoked = True
s = L.rt.run(max_iterations=400)
st = state(L)
denied = sum(1 for r in L.agent.results if r[1] == "denied")
R.check("8 stop condition 'authorization_revoked' ends the run when the authorization is revoked",
        s in ("stopped", "paused") or "revok" in st.get("status_reason", ""),
        {"status": s, "reason": st.get("status_reason"), "decisions_taken_after_revocation_all_denied": denied})
sys.exit(R.finish())
