"""Experiment E: observational prediction vs controlled interventions in SocialWebLab.

In one lab run, simulated users post naturally (observational data, where experienced authors with
many followers also tend to write more detailed posts), while our agent runs a randomized 2x2x2
experiment with its own test accounts: content (detailed+polite vs vague), audience (account with 0
vs 30 followers) and timing (posting at 04:00 vs 19:00 UTC), assigned at random per post.

Compared, for P(answered within 24 h):
  randomized    difference in means between arms of the experiment (identified causal effect)
  naive_obs     difference in means among third-party posts by content (confounded)
  adjusted_obs  same, within follower strata (adjusts for the observed confounder)
  model_obs     temporal world model trained on third-party observations only: mean predicted
                difference when ONLY the content (or only the followers / time) of each experimental
                post is swapped
  model_nocov   same model family trained without the audience/time covariates in its context
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from common import RUNS, Timer, make_wm, save_json, strip_context, threads  # noqa: E402

from agentrs.actions.schema import Action  # noqa: E402
from agentrs.agent.runtime import Decision  # noqa: E402
from agentrs.agent.temporal_runtime import act_with_delayed_outcome  # noqa: E402
from agentrs.evaluation.metrics import bootstrap_ci  # noqa: E402
from agentrs.sensors.observation_network import NetworkConfig  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.socialweblab.agent_tasks import HelpTeacher, SocialWebLabContentEnv, make_help_task, op_for  # noqa: E402
from agentrs.socialweblab.apps import content_features  # noqa: E402
from agentrs.socialweblab.engine import LAB_EPOCH, SocialWebLab  # noqa: E402
from agentrs.temporal.system import build_network  # noqa: E402
from agentrs.temporal.temporal_evaluation import event_truth  # noqa: E402
from agentrs.temporal.temporal_world_model import TemporalQuery  # noqa: E402
from agentrs.training.common import OptimConfig  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)

DAY = 86400.0
DETAILED = dict(p_right_space=1.0, p_detail=1.0, p_polite=1.0, p_shout=0.0, p_clickbait=0.0)
VAGUE = dict(p_right_space=1.0, p_detail=0.0, p_polite=0.0, p_shout=0.0, p_clickbait=0.0)


def write(task, detailed: bool, seed: int) -> str:
    t = HelpTeacher(seed=seed, **(DETAILED if detailed else VAGUE))
    t.rng = random.Random(seed)  # same sentence choices for both variants
    return t.write(task)[1]


class Fixed:
    name = "experiment"

    def __init__(self, op, content):
        self.op, self.content = op, content

    def begin_episode(self, b):
        pass

    def decide(self, belief, cands):
        return Decision(action=Action.submit(self.op, self.content), selection="randomized_experiment")


def diff(y, x):
    y, x = np.asarray(y, float), np.asarray(x, bool)
    d = y[x].mean() - y[~x].mean()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        yy, xx = y[i], x[i]
        if xx.any() and (~xx).any():
            bs.append(yy[xx].mean() - yy[~xx].mean())
    return {"effect": float(d), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_treated": int(x.sum()), "n_control": int((~x).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--per-slot", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    threads()
    T = Timer()
    out = RUNS / f"expE_s{a.seed}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lab = SocialWebLab(out / "lab.db", seed=a.seed, n_users=80, lab_id=f"causal{a.seed}")
    net = build_network(out / "net", lab, None, NetworkConfig(scheduling="fixed", forecasts_enabled=False),
                        split_fn=lambda o: "observational")
    env = SocialWebLabContentEnv(lab)
    for h in ("L0", "L1"):
        env.ensure_account(f"agent{h}", followers=0)
    for h in ("H0", "H1"):
        env.ensure_account(f"agent{h}", followers=30)
    rng = random.Random(a.seed)
    design = []
    k = 0
    for step in range(a.days * 96):
        lab.advance(900)
        hour = int(((lab.now() - LAB_EPOCH) % DAY) // 3600)
        minute = int(((lab.now() - LAB_EPOCH) % 3600) // 60)
        if minute == 0 and hour in (4, 19) and lab.now() - LAB_EPOCH >= DAY and lab.now() - LAB_EPOCH < (a.days - 1) * DAY:
            for _ in range(a.per_slot):
                detailed, high = rng.random() < 0.5, rng.random() < 0.5
                acc = f"agent{'H' if high else 'L'}{k % 2}"
                task = make_help_task("train", 5000 + k, account=acc)
                content = write(task, detailed, seed=k)
                r = act_with_delayed_outcome(env, task, Fixed(op_for(task.private["space"]), content), net,
                                             allow_decline=False)
                net.ledger._x("UPDATE actions SET split='experiment' WHERE action_id=?", (r["action_id"],))
                design.append({"action_id": r["action_id"], "item": r["item_id"], "detailed": detailed, "high": high,
                               "peak": hour == 19, "task": task, "content": content})
                k += 1
        net.step()
    spec = lab_temporal_spec()
    b = TemporalTrainingDataBuilder(net.ledger, spec)
    obs_ex = b.build(splits=["observational"], cutoffs=[0], groups=["forum"])
    exp_ex = {e.action_id: e for e in b.build(splits=["experiment"], cutoffs=[0])}
    # randomized estimates
    rows = []
    for d in design:
        e = exp_ex.get(d["action_id"])
        y = event_truth(e, "answered", DAY) if e else None
        if y is None:
            continue
        help_ = lab.helpful_answer_within(d["item"], DAY)["success"]
        views = [o for o in net.ledger.observations(d["action_id"]) if abs(o["elapsed"] - DAY) < 3 * 3600]
        rows.append({**{k_: d[k_] for k_ in ("detailed", "high", "peak")}, "answered": y, "helpful": int(help_),
                     "views": views[0]["payload"]["counts"]["views"] if views else None})
    res = {"seed": a.seed, "n_experiment": len(rows), "n_observational": len(obs_ex), "randomized": {}, "observational": {}}
    for outcome in ("answered", "helpful", "views"):
        rr = [r for r in rows if r[outcome] is not None]
        res["randomized"][outcome] = {f: diff([r[outcome] for r in rr], [r[f] for r in rr])
                                      for f in ("detailed", "high", "peak")}
    # observational associations among third-party forum posts
    ys, det, fol = [], [], []
    for e in obs_ex:
        y = event_truth(e, "answered", DAY)
        if y is None or e.source_type != "third_party":
            continue
        f = content_features(e.complete_action, e.group.split(":")[-1])
        followers = int(e.initial_state.split("followers=")[1].split()[0]) if "followers=" in e.initial_state else 0
        ys.append(y)
        det.append(f.detail and f.polite)
        fol.append(followers)
    ys, det, fol = np.array(ys), np.array(det), np.array(fol)
    res["observational"]["naive_detail_and_polite"] = diff(ys, det)
    strata = np.digitize(fol, [3, 10])
    adj, wsum = 0.0, 0
    for s in np.unique(strata):
        m = strata == s
        if det[m].any() and (~det[m]).any():
            adj += m.sum() * (ys[m][det[m]].mean() - ys[m][~det[m]].mean())
            wsum += m.sum()
    res["observational"]["adjusted_by_followers"] = adj / max(1, wsum)
    res["observational"]["corr_detail_followers"] = float(np.corrcoef(det.astype(float), fol)[0, 1])
    res["observational"]["followers_effect_naive"] = diff(ys, fol >= 10)
    # model-implied effects
    cfg = TemporalTrainConfig(steps=a.steps, batch_size=16, seed=0,
                              optim=OptimConfig(lr=2e-4, total_steps=a.steps, schedule="cosine", warmup_steps=30))
    third = [e for e in obs_ex if e.source_type == "third_party"]
    for name, data, ctxfn in (("model_obs", third, lambda c: c), ("model_nocov", [strip_context(e) for e in third],
                                                                     lambda c: strip_context_text(c))):
        m = make_wm(0)
        TemporalWorldModelTrainer(m, cfg, name=name).train(data)
        eff = {"content": [], "audience": [], "timing": []}
        for d in design:
            e = exp_ex.get(d["action_id"])
            if e is None:
                continue
            ctx = e.initial_state
            alt = write(d["task"], not d["detailed"], seed=0)
            op = op_for(d["task"].private["space"])
            c_det, c_vag = (d["content"], alt) if d["detailed"] else (alt, d["content"])

            def p(context, content):
                q = TemporalQuery(context=ctxfn(context), action=Action.submit(op, content).to_str(), source_type="self",
                                  actor_type="agent")
                return m.predict(q, horizons=[DAY]).horizons[0].event_prob["answered"]

            eff["content"].append(p(ctx, c_det) - p(ctx, c_vag))
            f_hi = _set_followers(ctx, 30)
            f_lo = _set_followers(ctx, 0)
            eff["audience"].append(p(f_hi, d["content"]) - p(f_lo, d["content"]))
            eff["timing"].append(p(_set_hour(ctx, 19), d["content"]) - p(_set_hour(ctx, 4), d["content"]))
        res[name] = {k_: {"mean": float(np.mean(v)), "ci95": list(bootstrap_ci(v))} for k_, v in eff.items()}
        print(name, res[name], flush=True)
    res["wall_s"] = T.s()
    save_json(f"expE_s{a.seed}.json", res)
    print(res["randomized"]["answered"], res["observational"])


def strip_context_text(c: str) -> str:
    return "\n".join(line for line in c.splitlines() if line.split(":")[0] in ("ENV", "TOPIC"))


def _set_followers(ctx: str, n: int) -> str:
    return "\n".join(f"AUTHOR: followers={n}" if line.startswith("AUTHOR:") else line for line in ctx.splitlines())


def _set_hour(ctx: str, h: int) -> str:
    out = []
    for line in ctx.splitlines():
        if line.startswith("TIME:"):
            day = line.split()[1]
            line = f"TIME: {day} {h:02d}:00 UTC" + (" (simulated clock)" if "simulated" in line else "")
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    main()
