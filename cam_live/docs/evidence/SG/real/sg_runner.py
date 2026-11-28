"""SG experiment driver: one CamLiveAgent, several objectives in ONE GPU-lock hold (or on CPU at debug scale), the real
runtime (runtime.loop.LiveRuntime through eval.oe_runner.run_objective) on the persistent staging computer.

    python sg_runner.py --plan CUFREE:20,CU:store_entry,CU:store_entry,CU:record_item --decisions 15 --seed 0 \
        [--no-skills] [--real] --out R.json --root DIR [--hold-s 840]

Families: CU:<problem> (staging computer-use objective, verified by the sites' own state), CUFREE:<n> (a goal-free
period of n decisions on the same staging computer: no success criterion, nothing checked), SW/MECH/GOALFREE as in
eval.oe_runner. Recorded per objective: what oe_runner.run_objective records, plus the skill library (every skill
with its provenance, validation history, runs, versions), the goal-free report (goals, transfer evidence) and the
agent's counters.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch


def lib_state(computers):
    cu = getattr(computers, "_cu", None)
    if cu is None or cu.sc.env is None:
        return None
    from cam_live.harness.skills import SkillExecutor
    ex = next((e for e in cu.sc.env._executors.values() if isinstance(e, SkillExecutor)), None)
    if ex is None:
        return {"skills": "excluded"}
    out = []
    for s in ex.library.list():
        out.append({"skill_id": s.skill_id, "version": s.version, "status": s.status, "operations": s.operations(),
                    "params": sorted(s.arg_schema.get("properties", {})), "runs": s.runs, "successes": s.successes,
                    "reliability": s.reliability()["mean"], "applicability_n": s.applicability_model().n,
                    "method": s.provenance.get("method"), "composed_of": s.provenance.get("composed_of"),
                    "source_action_ids": s.provenance.get("action_ids"), "versions": ex.library.versions(s.skill_id),
                    "validation": [{k: h.get(k) for k in ("ok", "termination", "n_ops")} for h in s.validation_history],
                    "run_history": [{k: h.get(k) for k in ("ok", "termination", "n_ops", "elapsed_s")}
                                    for h in s.run_history]})
    rejected = 0
    path = os.path.join(ex.library.dir, "rejected.jsonl")
    if os.path.exists(path):
        reasons = [json.loads(line)["report"].get("reason", "")[:160] for line in open(path)]
        rejected = len(reasons)
    else:
        reasons = []
    return {"skills": out, "rejected": rejected, "rejected_reasons": reasons[-10:], "buffer": len(ex._buffer)}


class Computers:
    """oe_runner._Computers plus CUFREE (a goal-free period on the staging computer)."""

    def __init__(self, root, clock, no_skills):
        from cam_live.eval.oe_runner import _Computers
        ns = argparse.Namespace(no_skills=no_skills)
        self.inner = _Computers(root, clock, ns)
        self.clock = clock

    @property
    def _cu(self):
        return self.inner._cu

    def open(self, family, problem, seed, objective_id, *, max_decisions):
        if family == "CU2":                          # the same objective with two more records needed (repetition)
            if self.inner._cu is None:
                from cam_live.eval.oe_runner import CUComputer
                self.inner._cu = CUComputer(self.inner.root, self.inner.clock, skills=not self.inner.a.no_skills)
            sc = self.inner._cu.sc
            from cam_live.validation import STAGING_OBJECTIVES
            metric = STAGING_OBJECTIVES[problem][2]
            now = {q.name: q.value for q in sc.metrics.read()}[metric]
            task = sc.task(problem, max_decisions=max_decisions, objective_id=objective_id, target=int(now) + 2)
            env, policy, verifier = sc.open(task, sc.start_url(problem))
            return env, policy, verifier, task
        if family != "CUFREE":
            return self.inner.open(family, problem, seed, objective_id, max_decisions=max_decisions)
        from cam_live.eval.openended import NoCriterionVerifier
        env, policy, _v, task = self.inner.open("CU", "store_entry", seed, objective_id, max_decisions=max_decisions)
        task.success = {}
        task.objective = ("No objective is assigned for this period and nothing is checked. The computer and its tools "
                          "are available.")
        task.family = "staging/goal-free"
        policy.register_task(task, "operator")
        return env, policy, NoCriterionVerifier(self.clock), task

    def close(self):
        self.inner.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--decisions", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-skills", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--hold-s", type=float, default=840.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    from cam_live.eval.oe_runner import _build, describe, run_objective, substantive_config, _meta_records, \
        _learning_records
    from cam_live.validation import OffsetClock
    os.makedirs(a.root, exist_ok=True)
    clock = OffsetClock()
    plan = []
    for part in a.plan.split(","):
        fam, _, prob = part.partition(":")
        plan.append((fam, prob))
    res = {"plan": a.plan, "seed": a.seed, "skills": not a.no_skills, "real": a.real, "decisions": a.decisions,
           "pythonpath": os.environ.get("PYTHONPATH", ""), "objectives": [], "started": time.time()}
    lock = contextlib.nullcontext()
    if a.real and not a.no_lock:
        from cam_live.nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=14.0, timeout_s=7200.0)
    t_wait = time.time()
    with lock:
        t0 = time.time()
        res["lock_wait_s"] = t0 - t_wait
        cfg = substantive_config("E", a.seed, real=a.real)
        agent = _build("E", cfg, os.path.join(a.root, "agent"), clock)
        res["configuration"] = describe(agent)
        comp = Computers(a.root, clock, a.no_skills)
        k_assigned = 0
        for i, (fam, prob) in enumerate(plan):
            left = a.hold_s - (time.time() - t0)
            per = [o["wall_s"] for o in res["objectives"]] or [0.0]
            if i > 0 and left < 0.6 * max(per) + 30:
                res.setdefault("not_started", []).append(f"{fam}:{prob}")
                continue
            n = int(prob) if fam == "CUFREE" and prob else a.decisions
            wseed = 1000 * (k_assigned + 1) + a.seed
            k_assigned += int(fam not in ("CUFREE", "GOALFREE"))
            ob = run_objective(agent, comp, fam, prob if fam != "CUFREE" else "free", wseed,
                               f"obj_{fam}_{prob}_{i}_s{a.seed}", os.path.join(a.root, f"obj{i:02d}"), clock,
                               max_decisions=n, budget_s=max(60.0, left - 30.0))
            ob["index"] = i
            ob["meta_updates"] = _meta_records(os.path.join(a.root, f"obj{i:02d}"))[-20:]
            ob["learning_updates_n"] = len(_learning_records(os.path.join(a.root, f"obj{i:02d}")))
            ob["library"] = lib_state(comp)
            gf = agent.goal_free.report()
            ob["goal_free"] = {k: gf.get(k) for k in ("goals", "resolved", "achieved", "abandoned", "by_key",
                                                      "transfer", "budget_remaining")}
            ob["agent_counters"] = {k: v for k, v in dict(agent.counters).items()}
            ob["skill_catalog"] = list(agent.skill_catalog)
            res["objectives"].append(ob)
            print(f"[{'skills' if not a.no_skills else 'primitive'} s{a.seed}] {i} {fam}:{prob} "
                  f"{ob['verified_status']} decisions {ob['decisions']} ops {ob.get('primitive_ops_total')} "
                  f"skill_runs {ob.get('skill_runs')}/{ob.get('skill_runs_ok')} wall {ob['wall_s']:.0f}s "
                  f"lib {[(s['operations'], s['runs']) for s in (ob['library'] or {}).get('skills', [])] if isinstance((ob['library'] or {}).get('skills'), list) else ob['library']}",
                  flush=True)
            with open(a.out, "w") as f:
                json.dump(res, f, default=str)
        res["hold_s"] = time.time() - t0
        res["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        comp.close()
        agent.close()
    res["executed"] = True
    with open(a.out, "w") as f:
        json.dump(res, f, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
