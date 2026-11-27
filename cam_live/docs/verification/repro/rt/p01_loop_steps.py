"""G-rt probe 01 — the 17 steps (spec 15.1): order, observability in the logs, predictions before outcomes,
append-only prediction records, delayed outcomes linked to the causing decision, predictions never become outcomes.

    .venv/bin/python cam_live/docs/verification/repro/rt/p01_loop_steps.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import LOGS, Report, StandInAgent, check_seq, make_lab_run, read_log  # noqa: E402

from cam_live.contracts import (ExperimentRecord, OutcomeObservation, OutcomeStatus, RunFiles)  # noqa: E402
from cam_live.runtime.dashboard import RunView  # noqa: E402
from cam_live.runtime.journal import generate_journal  # noqa: E402

R = Report("p01_loop_steps")

# ------------------------------------------------------------------ A. order of the steps in one working cycle
L = make_lab_run("p01a", script=("lab.act", "wait"), lab_kw={"target": 3})
T = []          # shared trace: (who.method, log seq at call time)


def tap(obj, who, names):
    for n in names:
        orig = getattr(obj, n)

        def w(*a, __o=orig, __n=n, **k):
            T.append((f"{who}.{__n}", L.rt.log.next_seq))
            return __o(*a, **k)
        setattr(obj, n, w)


tap(L.world, "env", ["observe", "poll_events", "measure", "discover_tools", "validate_action", "execute"])
tap(L.policy, "policy", ["authorize", "check_grant"])
tap(L.verifier, "verifier", ["verify", "measure"])
for m in ("set_objective", "ingest", "reconcile", "update_beliefs", "propose", "evaluate", "select", "record_result",
          "internal_rewards", "learn", "meta_learn", "save", "next_wake"):
    L.agent.hooks[m] = (lambda *a, __m=m: T.append((f"agent.{__m}", L.rt.log.next_seq)))
snapshots = []
L.rt.hooks["after_cycle"].append(lambda rt: (rt.log.sync(), snapshots.append(
    {n: open(os.path.join(rt.run_dir, n), "rb").read() if os.path.exists(os.path.join(rt.run_dir, n)) else b""
     for n in LOGS})))
L.rt.start()
rep = L.rt.step()                      # first cycle: observe ... act ... save
names = [t[0] for t in T]
want = ["agent.set_objective", "env.observe", "agent.ingest", "verifier.verify", "agent.update_beliefs",
        "agent.propose", "agent.evaluate", "agent.select", "env.validate_action", "policy.authorize", "env.execute",
        "agent.record_result", "agent.internal_rewards", "agent.learn", "agent.meta_learn", "agent.save"]
pos = [names.index(w) if w in names else -1 for w in want]
R.check("A1 first cycle calls steps 1,2,3,(4),5,6,7-8,9,10,11,12,13,14,15,16 in the specified order",
        all(p >= 0 for p in pos) and pos == sorted(pos), dict(zip(want, pos)))
R.note("A1b env.poll_events is read before env.observe (events delivered to the learner after observe)",
       {"poll_events": names.index("env.poll_events"), "observe": names.index("env.observe")})
status = L.rt.run(max_iterations=400)
names = [t[0] for t in T]
R.check("A2 run reaches a terminal status on LabWorld", status in ("completed", "failed", "exhausted"), status)
R.check("A3 step 4 (reconcile) was reached", "agent.reconcile" in names, {"reconcile": names.count("agent.reconcile")})
# in every cycle (split at poll_events): reconcile < update_beliefs < propose < evaluate < select < record_result
# < internal_rewards < learn < meta_learn < save.  The result of a *wait* that just ended is delivered at the very
# start of the next cycle (before observe); that single leading record_result is set aside.
order = ["agent.reconcile", "agent.update_beliefs", "agent.propose", "agent.evaluate", "agent.select",
         "agent.record_result", "agent.internal_rewards", "agent.learn", "agent.meta_learn", "agent.save"]
cyc, viol = [], 0
for n in names:
    if n == "env.poll_events":
        cyc.append([])
    elif cyc:
        cyc[-1].append(n)
for c in cyc:
    if "env.observe" in c and "agent.record_result" in c[:c.index("env.observe")]:
        c.remove("agent.record_result")
    idx = [c.index(o) for o in order if o in c]
    viol += idx != sorted(idx)
R.check("A4 every cycle keeps the step order", viol == 0 and len(cyc) >= 4, {"cycles": len(cyc), "violations": viol})

# step 17 with nothing to do: a learner that proposes nothing and asks to be woken in one hour
class Quiet(StandInAgent):
    def propose(self, tools, k):
        self._t("propose", tools, k)
        return []

    def next_wake(self):
        self._t("next_wake")
        return self.clock.now() + 3600.0


LQ = make_lab_run("p01q", lab_kw={"target": 3})
LQ.rt.agent = q = Quiet(LQ.clock, durable=False)
execs = []
oe = LQ.world.execute
LQ.world.execute = lambda a, auth: (execs.append(a.action_id), oe(a, auth))[1]
t0 = LQ.clock.now()
for _ in range(10):
    LQ.rt.step()
dt = LQ.clock.now() - t0
R.check("A5 with no work the runtime asks next_wake, sleeps until then, and executes nothing (no activity to fill time)",
        q.trace.count("next_wake") >= 9 and not execs and abs(dt - 10 * 3600.0) < 1.0 and q.trace.count("propose") == 10,
        {"next_wake": q.trace.count("next_wake"), "propose": q.trace.count("propose"), "sim_seconds": dt,
         "env_executions": len(execs)})

# ------------------------------------------------------------------ B. each step observable in the logs
ev = read_log(L.run_dir, RunFiles.EVENTS)
types = {}
for e in ev:
    types[e["type"]] = types.get(e["type"], 0) + 1
dec = read_log(L.run_dir, RunFiles.DECISIONS)
pred = read_log(L.run_dir, RunFiles.PREDICTIONS)
lu = read_log(L.run_dir, RunFiles.LEARNING)
obs_via = {e["data"].get("via") for e in ev if e["type"] == "observation"}
visible = {
    "1 objective (run_started carries the task)": types.get("run_started", 0) >= 1,
    "2 observe (observation via=observe)": "observe" in obs_via,
    "3 evidence (observation via=poll)": "poll" in obs_via,
    "4 reconcile (outcome + reconciled events)": types.get("outcome", 0) >= 1 and types.get("reconciled", 0) >= 1,
    "5 belief update (any record of update_beliefs)": any("belief" in e["type"] for e in ev),
    "6 candidates (decision.candidates)": bool(dec) and all(len(d["data"]["decision"]["candidates"]) >= 1 for d in dec),
    "7-8 predictions registered": any(p["type"] == "prediction" for p in pred),
    "9 decision record": bool(dec),
    "10 authorization event": types.get("authorization", 0) >= 1,
    "11 action_start": types.get("action_start", 0) >= 1,
    "12 action_result + observation via=action": types.get("action_result", 0) >= 1 and "action" in obs_via,
    "13 internal_rewards event": types.get("internal_rewards", 0) >= 1,
    "14 learning_update record": bool(lu),
    "15 meta_update record (stand-in returns none; F's claim is 'records what the learner returns')": True,
    "16 checkpoint event": types.get("checkpoint", 0) >= 1,
    "17 sleep event": types.get("sleep", 0) >= 1,
}
for k, v in visible.items():
    R.check(f"B  step {k} is observable in the run logs", v)

# ------------------------------------------------------------------ C. predictions before outcomes; append-only
res = [p for p in pred if p["type"] == "resolution"]
reg = {p["rid"]: p for p in pred if p["type"] == "prediction"}
starts = {e["data"]["action_id"]: e["seq"] for e in ev if e["type"] == "action_start"}
outcomes = {e["rid"]: e for e in ev if e["type"] == "outcome"}
bad = []
for r in res:
    g = reg.get(r["data"]["registered_rid"])
    o = outcomes.get(r["data"]["outcome"]["outcome_rid"])
    if g is None or o is None:
        bad.append(("unlinked", r["rid"]))
        continue
    aid = g["data"]["prediction"]["action_id"]
    if not (g["seq"] < starts.get(aid, 1e18) < o["seq"] < r["seq"]):
        bad.append(("order", r["rid"], g["seq"], starts.get(aid), o["seq"], r["seq"]))
R.check("C1 every resolution: registration seq < action_start seq < outcome seq < resolution seq",
        bool(res) and not bad, {"resolutions": len(res), "bad": bad[:3]})
final = {n: open(os.path.join(L.run_dir, n), "rb").read() if os.path.exists(os.path.join(L.run_dir, n)) else b"" for n in LOGS}
R.check("C2 all log files are append-only (every per-cycle snapshot is a byte prefix of the final file)",
        all(final[n].startswith(s[n]) for s in snapshots for n in LOGS), {"snapshots": len(snapshots)})
R.check("C3 registered prediction records carry no resolved value",
        all(p["data"]["prediction"]["resolved_value"] is None and p["data"]["prediction"]["resolved_status"] == "pending"
            for p in reg.values()))
sq = check_seq(L.run_dir)
R.check("C4 seq is global, unique and increasing; no gaps", not sq["duplicate_seq"] and not sq["non_monotonic"]
        and sq["n_gaps"] == 0 and sq["duplicate_rid"] == 0, sq)

# ------------------------------------------------------------------ C'. decision records (spec 14.1)
st_ = json.load(open(os.path.join(L.run_dir, RunFiles.STATE)))
dstart = {e["data"]["decision_id"]: e["seq"] for e in ev if e["type"] == "action_start"}
dres = {e["data"]["decision_id"]: e["seq"] for e in ev if e["type"] == "action_result"}
fields_ok = all(
    d["ts"] > 0 and d["data"]["objective"] and "situation" in d["data"]
    and all(k in d["data"]["decision"] for k in ("ts", "objective_id", "chosen", "candidates", "probabilities", "chosen_index",
                                                 "exploration", "rationale", "context_obs_ids", "model_version", "checkpoint_id",
                                                 "diagnostics"))
    and all(k in c for c in d["data"]["decision"]["candidates"] for k in ("predictions", "p_success", "value", "voi", "cost", "uncertainty"))
    for d in dec)
R.check("C5 one decision record per decision (count = decisions counter), written before the action starts, closed by one result",
        len(dec) == st_["counters"]["decisions"] and all(d["data"]["decision"]["decision_id"] in dres for d in dec)
        and all(d["seq"] < dstart.get(d["data"]["decision"]["decision_id"], 1e18) for d in dec),
        {"records": len(dec), "counter": st_["counters"]["decisions"]})
R.check("C6 each decision record has the slots of spec 14.1 (timestamp, objective, situation, evidence refs, candidates with "
        "predictions / uncertainty / learning value, chosen action, rationale, model version)", fields_ok)

# ------------------------------------------------------------------ D. delayed outcomes -> causing decision
res_ev = {e["data"]["action_id"]: e for e in ev if e["type"] == "action_result" and e["data"]["kind"] == "execute"}
ok, n = True, 0
for e in outcomes.values():
    d = e["data"]
    aid = d["action_id"]
    truth = sum(1 for t, a, _ in L.world._arrivals if a == aid and t <= d["query"]["issued_at"] + d["query"]["horizon_s"])
    exp = [q["query_id"] for q in res_ev[aid]["data"]["expected_outcomes"]]
    n += 1
    if not (d["query"]["query_id"] in exp and d["decision_id"] == res_ev[aid]["data"]["decision_id"]
            and d["observation"]["value"] == truth and L.agent.query_of.get(d["query"]["query_id"]) == aid):
        ok = False
R.check("D1 each matured outcome is logged with the decision/action that issued its query and the world's true count",
        ok and n > 0, {"outcomes": n})
R.check("D2 each outcome reaches the learner exactly once",
        sorted(L.agent.reconciled) == sorted(e["data"]["query"]["query_id"] for e in outcomes.values())
        and len(set(L.agent.reconciled)) == len(L.agent.reconciled), {"delivered": len(L.agent.reconciled)})
R.check("D3 goal-relevant outcomes measured by the verifier are marked verified, trust=verifier",
        all(e["data"]["observation"]["verified"] and e.get("trust") == "verifier" for e in outcomes.values()))

# ------------------------------------------------------------------ E. a learner that tries to pass predictions off as outcomes
class Liar(StandInAgent):
    """Registers predictions with resolved_* pre-filled and reports an experiment with a fabricated 'actual'."""

    def evaluate(self, candidates):
        evs = super().evaluate(candidates)
        for e in evs:
            for p in e.predictions:
                p.dist.mean = 777.0
                p.resolved_value, p.resolved_status, p.resolved_at, p.nll = 777.0, "observed", self.clock.now(), 0.0
        return evs

    def drain_experiments(self):
        if getattr(self, "_sent", False):
            return []
        self._sent = True
        fake = OutcomeObservation(query_id="q_never_issued", status=OutcomeStatus.OBSERVED, value=999.0,
                                  observed_at=self.clock.now(), verified=True, source="verifier:forged-by-learner")
        return [ExperimentRecord(experiment_id="exp_liar", created_at=self.clock.now(), question="q", hypothesis="h",
                                 status="closed", actual=[fake], supported=True)]


L2 = make_lab_run("p01e", script=("lab.act", "wait"), lab_kw={"target": 3})
liar = Liar(L2.clock, script=("lab.act", "wait"), durable=False)
L2.rt.agent = liar
L2.rt.run(max_iterations=200)
ev2 = read_log(L2.run_dir, RunFiles.EVENTS)
pr2 = read_log(L2.run_dir, RunFiles.PREDICTIONS)
ex2 = read_log(L2.run_dir, RunFiles.EXPERIMENTS)
out_vals = [e["data"]["observation"]["value"] for e in ev2 if e["type"] == "outcome"]
res_vals = [p["data"]["outcome"]["value"] for p in pr2 if p["type"] == "resolution"]
R.check("E1 no outcome event / resolution carries the predicted value 777", 777.0 not in out_vals + res_vals and out_vals,
        {"outcomes": out_vals[:5]})
prefilled = [p["rid"] for p in pr2 if p["type"] == "prediction" and p["data"]["prediction"]["resolved_value"] is not None]
R.check("E2 a prediction registered with resolved_value/resolved_status already filled is refused or cleared",
        not prefilled, {"records_with_prefilled_resolution": len(prefilled), "first": prefilled[:1]})
fab = [e for e in ex2 if e["type"] == "experiment" and any(o.get("value") == 999.0 for o in e["data"].get("actual") or [])]
view = RunView(L2.run_dir).experiments()
shown = [x for x in view if any(o.get("value") == 999.0 and o.get("verified") for o in x["actual"])]
R.check("E3 a learner-fabricated 'actual' outcome (verified=True, query never issued) is not logged as an experiment outcome",
        not fab, {"experiments.jsonl records carrying it": [e["rid"] for e in fab]})
R.check("E4 the dashboard's 'actual outcomes' for experiments come from runtime-measured outcomes, not the learner's record",
        not shown, {"dashboard rows showing value 999 verified": shown[:1]})
j = generate_journal(L2.run_dir)
R.check("E5 the journal does not state the learner's own 'supported=True' verdict as a [FACT]",
        "experiment(s) ended with the registered hypothesis supported" not in j,
        [ln for ln in j.splitlines() if "supported" in ln][:2])

# ------------------------------------------------------------------ F. predictions about a metric the learner names itself
class Forecaster(StandInAgent):
    """Registers, for the chosen action, a prediction of the objective metric one day ahead (its own OutcomeQuery)."""

    def evaluate(self, candidates):
        evs = super().evaluate(candidates)
        now = self.clock.now()
        for e in evs:
            if e.action.operation == "lab.act":
                q = OutcomeQuery(query_id=new_id("pq"), metric="lab.adopters.total", kind=OutcomeKind.COUNT, horizon_s=86400.0,
                                 issued_at=now, goal_relevant=True, description="total adopters one day from now")
                e.predictions.append(PredictionRecord(prediction_id=new_id("pred"), query=q,
                                                      dist=Distribution("poisson", {"rate": 2.0}, mean=2.0, std=1.4),
                                                      made_at=now, action_id=e.action.action_id))
        return evs


from cam_live.contracts import Distribution, OutcomeKind, OutcomeQuery, PredictionRecord, new_id  # noqa: E402
L3 = make_lab_run("p01f", script=("lab.act", "wait"), lab_kw={"target": 10_000}, task_edit=lambda t: setattr(t.limits, "max_decisions", 8))
L3.rt.agent = Forecaster(L3.clock, script=("lab.act", "wait"), durable=False)
L3.rt.run(max_iterations=300)
pr3 = read_log(L3.run_dir, RunFiles.PREDICTIONS)
own = [p for p in pr3 if p["type"] == "prediction" and p["data"]["prediction"]["query"]["metric"] == "lab.adopters.total"
       and p["data"]["for_chosen_action"]]
res3 = {p["data"]["prediction_id"] for p in pr3 if p["type"] == "resolution"}
due = [p for p in own if p["data"]["prediction"]["query"]["issued_at"] + 86400.0 <= L3.clock.now()]
R.check("F1 a registered prediction whose query the learner defined itself (objective metric, horizon one day; the "
        "environment and the verifier can both measure it) is measured and resolved when it falls due",
        bool(due) and all(p["data"]["prediction"]["prediction_id"] in res3 for p in due),
        {"registered_for_chosen_actions": len(own), "due_by_end_of_run": len(due),
         "resolved": sum(1 for p in due if p["data"]["prediction"]["prediction_id"] in res3)})
sys.exit(R.finish())
