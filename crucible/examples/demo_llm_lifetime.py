"""A general-purpose frozen-LLM agent living in a persistent Crucible world with a broad objective.

    PYTHONPATH=crucible/src:src python crucible/examples/demo_llm_lifetime.py --model Qwen/Qwen3-4B --days 5

The agent is told only: "Establish and maintain a sustainable source of value in this digital economy." It is not
told any industry, strategy or procedure. It sees what any actor sees (observations + its tool list) and acts once
per `dt`. The full trajectory (prompts' hashes, generated texts, actions, results) and an outcome summary are
written to docs/experiments/demo_llm_lifetime.json + .jsonl. Outcomes are measured by the evaluator, never by the
model's own text.
"""
import argparse
import collections
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.llm import LLMPolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.audit import integrity_report  # noqa: E402
from crucible.goals.metrics import m_cash, m_independent_revenue  # noqa: E402
from crucible.training.trajectories import export_jsonl  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

OBJECTIVE = "Establish and maintain a sustainable source of value in this digital economy."
DAY = 86400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--days", type=float, default=5)
    ap.add_argument("--dt", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/experiments/demo_llm_lifetime.json"))
    a = ap.parse_args()
    w = World.create(generate("bazaar", a.seed))
    me = w.system_call("admin.join", {"handle": "gaia", "endowment": 3_000_000, "kind": "agent"})["principal"]
    pol = LLMPolicy(model=a.model, goal=OBJECTIVE)
    run = Runner(w, [(me, pol)], dt=a.dt, seed=a.seed)
    t0 = time.time()
    rounds = int(a.days * DAY / a.dt)
    for i in range(rounds):
        run.round()
        if (i + 1) % 24 == 0:
            print(f"day {(i + 1) * a.dt / DAY:.1f}: actions={w.state.get('principals', me)['actions']} "
                  f"cash={m_cash(w, me)} wall={time.time() - t0:.0f}s", flush=True)
    steps = run.trajs[me].steps
    tools = collections.Counter(s.action["tool"] for s in steps if s.action)
    codes = collections.Counter((s.result or {}).get("error", {}).get("code", "ok") if s.result and not s.result.get("ok")
                                else "ok" for s in steps if s.action)
    orgs = sorted(m["org"] for m in w.state.find("members", "person", me))
    deps = sorted(d["host"] for o in orgs for d in w.state.find("deployments", "owner", o))
    repos = sorted(r for o in [me] + orgs for r in w.state.find_ids("repos", "owner", o))
    res = {"model": a.model, "objective": OBJECTIVE, "world": {"family": "bazaar", "seed": a.seed}, "days": a.days,
           "dt": a.dt, "decisions": len(steps), "tool_calls": sum(tools.values()), "tools": tools.most_common(25),
           "outcomes": dict(codes), "distinct_tools": len(tools), "orgs": orgs, "repos": repos, "deployments": deps,
           "cash_start": 3_000_000, "cash_end": m_cash(w, me) + sum(m_cash(w, o) for o in orgs),
           "independent_revenue_7d": {o: m_independent_revenue(w, o, 7 * DAY) for o in orgs},
           "integrity": integrity_report(w, me), "usage": pol.usage(), "wall_s": round(time.time() - t0, 1),
           "first_actions": [{"t": s.t, "action": s.action, "ok": (s.result or {}).get("ok"),
                              "error": (s.result or {}).get("error")} for s in steps[:12]],
           "invariants": w.check_invariants(),
           "replay_equal": World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()}
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    export_jsonl(run.trajs, a.out.replace(".json", ".jsonl"))
    print(json.dumps({k: res[k] for k in ("decisions", "tool_calls", "outcomes", "distinct_tools", "orgs", "repos",
                                          "deployments", "cash_end", "independent_revenue_7d", "usage", "replay_equal")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
