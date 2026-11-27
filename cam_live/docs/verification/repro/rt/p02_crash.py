"""G-rt probe 02 — crash recovery: kill -9 the runtime process at many points, restart, check invariants.

    .venv/bin/python cam_live/docs/verification/repro/rt/p02_crash.py            # all kill points
    .venv/bin/python cam_live/docs/verification/repro/rt/p02_crash.py after_wal_end:2 [--no-dedup] [--keep]

The world, verifier, policy and clock live in THIS process (served over the project RPC), so they survive the kill
and their ground truth (what really executed) can be compared with what the restarted runtime believes and logged.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, check_seq, read_log, scratch  # noqa: E402

from cam_live.contracts import RunFiles, SimClock  # noqa: E402
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task  # noqa: E402
from cam_live.harness.rpc import RpcServer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


class Host:
    ENV = ("observe", "discover_tools", "validate_action", "execute", "poll_events", "get_resource_state",
           "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure")
    METHODS = ENV + ("v_verify", "v_check_signature", "v_measure", "p_authorize", "p_check_grant", "p_record",
                     "p_approvals", "p_resolve_approval", "p_stopped", "p_limits_state", "clock_now", "clock_advance")

    def __init__(self, dedup=True):
        self.clock = SimClock()
        self.world = LabWorld(LabConfig(seed=5, target=10_000, budget=1000.0), self.clock)
        self.verifier, self.policy, self.dedup = LabVerifier(self.world), AllowAllPolicy(), dedup
        self.results, self.calls, self.real = {}, {}, {}

    def __getattr__(self, name):
        if name in ("observe", "discover_tools", "validate_action", "poll_events", "get_resource_state",
                    "get_pending_outcomes", "get_observable_metrics", "measure"):
            return getattr(self.world, name)
        raise AttributeError(name)

    def get_capabilities(self):
        return {**self.world.get_capabilities(), "idempotent_actions": self.dedup}

    def execute(self, action, authorization):
        aid = action.action_id
        self.calls[aid] = self.calls.get(aid, 0) + 1
        if self.dedup and aid in self.results:
            return self.results[aid]
        res = self.world.execute(action, authorization)
        self.real[aid] = self.real.get(aid, 0) + 1
        self.results[aid] = res
        return res

    def v_verify(self, task): return self.verifier.verify(task)
    def v_check_signature(self, o): return self.verifier.check_signature(o)
    def v_measure(self, q): return self.verifier.measure(q)
    def p_authorize(self, a, t, c=None): return self.policy.authorize(a, t, c)
    def p_check_grant(self, a, d): return self.policy.check_grant(a, d)
    def p_record(self, a, r): return self.policy.record(a, r)
    def p_approvals(self): return self.policy.approvals()
    def p_resolve_approval(self, i, ap, by, note=""): return self.policy.resolve_approval(i, ap, by, note)
    def p_stopped(self): return self.policy.stopped()
    def p_limits_state(self): return self.policy.limits_state()
    def clock_now(self): return self.clock.now()
    def clock_advance(self, dt): return self.clock.advance(dt)


def child(sock, run_dir, summary, kill="", env_extra=None, max_iter=400):
    env = dict(os.environ, G_KILL=kill, **(env_extra or {}))
    return subprocess.run([PY, os.path.join(HERE, "crash_child.py"), sock, run_dir, str(max_iter), summary],
                          env=env, capture_output=True, text=True, timeout=300)


def scenario(kills, dedup=True, max_decisions=9, env_extra=None, tamper=None, first_env=None):
    """kills: list of 'point:n' applied to successive process lifetimes; then one clean run to the end."""
    base = scratch("p02_" + "_".join(k.replace(":", "") for k in kills)[:60])
    # unix socket paths are limited to 108 bytes: the socket gets its own short directory
    run_dir, sock, summary = os.path.join(base, "run"), os.path.join(scratch("c"), "s"), os.path.join(base, "sum.json")
    host = Host(dedup)
    task = lab_task(host.world, "obj_crash")
    task.limits.max_decisions = max_decisions
    json.dump(task.to_dict(), open(os.path.join(base, "task.json"), "w"))
    srv = RpcServer(sock, host, Host.METHODS, name="g-host").start()
    info = {"kills": kills, "dedup": dedup, "base": base, "died": []}
    try:
        for i, k in enumerate(kills):
            p = child(sock, run_dir, summary, k, {**(env_extra or {}), **((first_env or {}) if i == 0 else {})})
            info["died"].append(p.returncode)
            if i == 0:
                info["first"] = {"rc": p.returncode, "err": (p.stderr.strip().splitlines() or [""])[-1][:200],
                                 "n_exec": host.world._n_exec}
            if tamper:
                tamper(run_dir)
        p = child(sock, run_dir, summary, "", env_extra)
        info["final_rc"], info["stderr"] = p.returncode, p.stderr[-600:]
        info["summary"] = json.load(open(summary)) if os.path.exists(summary) and p.returncode == 0 else None
    finally:
        srv.stop()
    info["host"] = host
    info["run_dir"] = run_dir
    return info


def judge(info):
    """Invariants after the final clean run. Returns {name: (holds, detail)}."""
    out = {}
    host, run_dir, s = info["host"], info["run_dir"], info["summary"]
    out["killed_as_planned"] = (all(rc == -9 for rc in info["died"]), info["died"])
    if s is None:
        out["restart_completes"] = (False, {"rc": info.get("final_rc"), "stderr": info.get("stderr")})
        return out
    out["restart_completes"] = (s["status"] in ("exhausted", "completed", "failed"), s["status"])
    ev = read_log(run_dir, RunFiles.EVENTS)
    dec = read_log(run_dir, RunFiles.DECISIONS)
    pred = read_log(run_dir, RunFiles.PREDICTIONS)
    st = json.load(open(os.path.join(run_dir, RunFiles.STATE)))
    # 1 nothing executes twice in the world
    twice = {a: n for a, n in host.real.items() if n != 1}
    out["no_action_executed_twice"] = (not twice, {"real_executions": len(host.real), "more_than_once": twice})
    # 2 every action the world executed is known to the restarted runtime: one action_result in events.jsonl ...
    results = {}
    for e in ev:
        if e.get("type") == "action_result":
            results.setdefault(e["data"]["action_id"], []).append(e)
    no_log = [a for a in host.real if a not in results]
    out["every_world_execution_has_an_action_result_record"] = (not no_log, {"executed_without_record": no_log})
    dup_res = {a: len(v) for a, v in results.items() if len(v) > 1}
    out["no_duplicate_action_result_record"] = (not dup_res, dup_res)
    # ... and one delivery to the learner (its persisted experience after replay)
    delivered = [r[0] for r in s["results"]]
    lost = [a for a in host.real if a not in delivered]
    unknown = [a for a in host.real if a in delivered and dict(s["results"])[a] == "cancelled"]
    out["every_world_execution_delivered_to_learner_once"] = (
        not lost and len(delivered) == len(set(delivered)),
        {"lost": lost, "duplicates": len(delivered) - len(set(delivered)), "reported_unknown(cancelled)": unknown})
    # 3 decisions: each decision record is closed by exactly one result; counters agree
    dec_ids = [d["data"]["decision"]["decision_id"] for d in dec if "data" in d]
    res_by_dec = {}
    for e in ev:
        if e.get("type") == "action_result":
            res_by_dec.setdefault(e["data"]["decision_id"], 0)
            res_by_dec[e["data"]["decision_id"]] += 1
    orphan = [d for d in dec_ids if d not in res_by_dec]
    out["every_decision_record_has_a_result_record"] = (not orphan, {"decisions": len(dec_ids), "without_result": orphan})
    out["decision_counter_matches_learner_experience"] = (
        st["counters"]["decisions"] == len(delivered) == s["counters"]["decisions"],
        {"state.decisions": st["counters"]["decisions"], "delivered": len(delivered), "decision_records": len(dec_ids)})
    out["decision_limit_respected"] = (st["counters"]["decisions"] <= st["task"]["limits"]["max_decisions"],
                                       st["counters"]["decisions"])
    # 4 accounting against ground truth
    out["money_counter_matches_world"] = (abs(st["counters"]["money"] - host.world._spent) < 1e-6,
                                          {"runtime": st["counters"]["money"], "world": host.world._spent})
    out["external_action_counter_matches_world"] = (st["counters"]["external_actions"] == host.world._n_exec,
                                                    {"runtime": st["counters"]["external_actions"], "world": host.world._n_exec})
    # 5 logs: seq and torn lines
    sq = check_seq(run_dir)
    out["seq_unique_and_monotonic_no_torn_line"] = (
        not sq["duplicate_seq"] and not sq["non_monotonic"] and sq["torn_or_corrupt"] == 0 and sq["duplicate_rid"] == 0, sq)
    # 6 outcomes: one outcome record per query; learner got each once
    oq = [e["data"]["query"]["query_id"] for e in ev if e.get("type") == "outcome"]
    out["one_outcome_record_per_query"] = (len(oq) == len(set(oq)), {"records": len(oq), "queries": len(set(oq))})
    rc = s["reconciled"]
    out["each_outcome_delivered_to_learner_once"] = (len(rc) == len(set(rc)) and set(rc) == set(oq),
                                                     {"delivered": len(rc), "distinct": len(set(rc)), "logged": len(set(oq))})
    res = [p["data"]["prediction_id"] for p in pred if p.get("type") == "resolution"]
    out["one_resolution_per_prediction"] = (len(res) == len(set(res)), {"resolutions": len(res), "distinct": len(set(res))})
    # 7 the checkpoint the state points at exists and loads
    ck = (st.get("committed") or {}).get("ckpt") or {}
    okck = bool(ck) and os.path.exists(os.path.join(ck["path"], "agent.json"))
    if okck:
        try:
            json.load(open(os.path.join(ck["path"], "agent.json")))
        except Exception:  # noqa: BLE001
            okck = False
    out["committed_checkpoint_is_complete"] = (okck, ck.get("id"))
    return out


POINTS = ["after_decision_log:2", "after_wal_begin:2", "before_execute:2", "after_execute:2", "before_wal_end:2",
          "after_wal_end:2", "torn_wal_end:2", "after_action_result_log:2", "torn_event:2", "in_record_result:2",
          "after_internal_rewards_log:2", "in_learn:3", "in_meta_learn:3", "in_save_partial:3", "after_save:3",
          "before_wal_reset:3", "after_wal_reset:3", "after_outcome_log:1", "in_reconcile:2", "after_checkpoint_log:2",
          "in_wait_sleep:1", "in_wait_sleep:2", "torn_wal_begin:2"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dedup = "--no-dedup" not in sys.argv
    R = Report("p02_crash" + ("" if dedup else "_nodedup") + ("_" + args[0].replace(":", "") if args else ""))
    table = {}
    for k in (args or POINTS):
        t0 = time.time()
        info = scenario([k], dedup=dedup)
        j = judge(info)
        table[k] = {n: v[0] for n, v in j.items()}
        for n, (ok, detail) in j.items():
            R.check(f"{k:28s} {n}", ok, detail if not ok else "")
        print(f"   ({k}: {time.time() - t0:.1f}s, run dir {info['run_dir']})", flush=True)
    if not args:
        # two consecutive crashes (in-flight twice; torn journal line then a second crash before the next commit)
        for kills, extra in ((["after_execute:2", "after_execute:1"], None),
                             (["torn_wal_begin:2", "in_learn:1"], None),
                             (["torn_wal_end:2", "after_wal_end:1"], {"G_CKPT_EVERY": "1e12"}),
                             (["torn_wal_end:2", "before_wal_end:2"], {"G_CKPT_EVERY": "1e12"})):
            info = scenario(kills, dedup=dedup, env_extra=extra)
            j = judge(info)
            name = "+".join(kills) + (" (checkpoint_every_s large)" if extra else "")
            table[name] = {n: v[0] for n, v in j.items()}
            for n, (ok, detail) in j.items():
                R.check(f"{name} {n}", ok, detail if not ok else "")
        # "disk full" (cheap simulation): no file may grow past N bytes during the first process lifetime
        for limit in ("30000", "60000", "120000"):
            info = scenario(["none:0"], dedup=dedup, env_extra=None, first_env={"G_FSIZE": limit}, max_decisions=30)
            j = judge(info)
            j.pop("killed_as_planned", None)
            name = f"file size limit {limit} bytes in the first lifetime"
            first = info.get("first", {})
            R.note(f"{name}: how the first process ended", {"returncode": first.get("rc"), "last_stderr_line": first.get("err"),
                                                           "world_executions_at_that_point": first.get("n_exec")})
            table[name] = {n: v[0] for n, v in j.items()}
            for n, (ok, detail) in j.items():
                R.check(f"{name} {n}", ok, detail if not ok else "")
    names = sorted({n for row in table.values() for n in row})
    print("\nsummary (rows: kill point; columns: broken invariants)")
    for k, row in table.items():
        print(f"  {k:60s} " + (", ".join(n for n in names if row.get(n) is False) or "all hold"))
    return R.finish()


if __name__ == "__main__":
    sys.exit(main())
