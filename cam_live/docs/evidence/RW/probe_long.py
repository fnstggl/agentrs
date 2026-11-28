"""RW scratch probe: one long LabWorld objective through the real runtime (debug agent, CPU); records evidence rates
of the reward meta-learning pathway (meta-update skips, groups, matured units, rewards vs verified progress)."""
import argparse, json, math, os, sys, time, faulthandler
faulthandler.dump_traceback_later(int(os.environ.get("RW_DUMP_S", "900")), repeat=True)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--decisions", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--world-seed", type=int, default=20000)
ap.add_argument("--meta", default="{}")
ap.add_argument("--agent", default="{}")
ap.add_argument("--real", action="store_true")
a = ap.parse_args()

import torch
from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.eval.envs import TruthTap
from cam_live.eval.runner import DecisionTap, runtime_config, close_out
from cam_live.eval.conditions import Timed
from cam_live.runtime.loop import LiveRuntime
from cam_live.reward.inner import verified_progress
from cam_live.reward.windows import independence_key
from cam_live.validation import staging_config

DAY = 86400.0
cfg = staging_config(None, a.real, a.seed)
cfg.sections["agent"].update(json.loads(a.agent))
rw = dict(cfg.sections.get("reward", {}))
rw["meta"] = {**rw.get("meta", {}), **json.loads(a.meta)}
cfg.sections["reward"] = rw
work = os.path.splitext(a.out)[0] + "_work"
os.makedirs(work, exist_ok=True)
clock = SimClock()
agent = CamLiveAgent(cfg, os.path.join(work, "agent"), clock=clock)
meta = agent.meta
lc = LabConfig(seed=a.world_seed, n_channels=4, target=10_000, budget=400.0, horizon_s=40 * DAY,
               outcome_horizon_s=6 * 3600.0, mean_delay_s=(1800.0, 4 * 3600.0), at_most_once=True, family="lab/first")
world = LabWorld(lc, clock)
task = lab_task(world, f"obj_long_{a.world_seed}")
task.limits.max_decisions = a.decisions
verifier = LabVerifier(world)
tap = TruthTap(world)
agent.attach_verifier(verifier)
timed = Timed(agent)
dtap = DecisionTap(timed, tap)
rt = LiveRuntime(dtap, tap, AllowAllPolicy(), verifier, task, os.path.join(work, "rt"), clock, runtime_config(16))
rew_log = []
orig_ir = meta.internal_rewards


def ir(records, **kw):
    out = orig_ir(records, **kw)
    ids = [r.exp_id for r in records]
    new = ids[-out["n"]:] if out["n"] else []
    for i, b, l in zip(new, out["bootstrap"], out["learned"]):
        rew_log.append({"exp_id": i, "bootstrap": b, "learned": l, "reward_version": out["reward_version"],
                        "mix": out["mix"]})
    return out


meta.internal_rewards = ir
mix_trace = []
orig_mu = meta.meta_update
def mu(now=None):
    r = orig_mu(now)
    mix_trace.append([len(rew_log), meta.mixer.mix, r.applied, r.skipped_reason[:60]])
    return r
meta.meta_update = mu
t0 = time.time()
rt.start()
steps = 0
while rt.status == "running" and steps < 6 * a.decisions:
    rt.step()
    steps += 1
if rt.status == "running":
    rt._finish("stopped", "step cap")
rt.close()
closing = close_out(timed, world, verifier, task)
wall = time.time() - t0
recs = [r for r in agent.store.trajectory(task.objective_id)]
units = []
for r in recs:
    units.append({"exp_id": r.exp_id, "ts": r.decision.ts, "op": r.decision.chosen.operation,
                  "y": verified_progress(r), "group": independence_key(r), "n_outcomes": len(r.outcomes)})
report = {"decisions": len(recs), "wall_s": wall, "s_per_decision": wall / max(1, len(recs)),
          "rewards": rew_log, "units": units,
          "meta": [{"applied": m.applied, "reason": m.skipped_reason, "n_groups": m.n_independent_outcomes,
                    "diag": {k: v for k, v in (m.diagnostics or {}).items() if isinstance(v, (int, float))}}
                   for m in meta.records],
          "mixer": [vars(d) for d in meta.mixer.decisions], "mix": meta.mixer.mix,
          "stats": dict(meta.stats), "ledger": len(meta.ledger), "closing": closing.get("meta_updates"),
          "validity": meta.validity.summary() if hasattr(meta, "validity") else None,
          "mix_trace": mix_trace}
with open(a.out, "w") as f:
    json.dump(report, f, indent=1, default=str)
from collections import Counter
print("decisions", len(recs), "wall", round(wall), "s/dec", round(wall / max(1, len(recs)), 2))
print("units with y", sum(1 for u in units if u["y"] is not None), "nonzero y", sum(1 for u in units if u["y"]))
print("meta calls", len(meta.records), Counter((m.skipped_reason or "APPLIED").split(":")[0] for m in meta.records))
print("mixer", [(d.action, d.n_independent) for d in meta.mixer.decisions], "mix", meta.mixer.mix)
if hasattr(meta, "validity"):
    v = meta.validity.summary(); print("validity", {k: v[k] for k in ("units", "pairs", "informative_pairs")},
                                       v["concordance_bootstrap"], v["concordance_learned"], v["comparisons"]["1->0.75"])
agent.close()
