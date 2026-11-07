"""Experiment F: verified task completion with delayed outcomes ("get a helpful answer within a day").

Pipeline (one lab, one seed):
1. imitation: the Pythia-14M policy imitates a mixed-quality scripted author (op = where to post,
   content = the post);
2. delayed-reward RL: every 12 simulated hours the policy samples posts for new training tasks;
   rewards (helpful answer within 24 h, verified by the environment) mature 24 h later and update the
   policy with the clipped importance ratio against the stored behaviour log-probs (the behaviour
   checkpoint is 1-2 versions old by then);
3. world models trained on what the sensor network recorded meanwhile:
     immediate        our own posts, labels truncated at 5 min (a next-observation model)
     temporal_self    our own posts, all horizons
     temporal_all     our own posts + third-party posts observed in the same period
4. evaluation on held-out tasks (unseen problems in the training spaces, and unseen spaces), each
   arm in an identical copy of the lab (snapshot restore), all posts made on the same schedule:
     bc_policy, rl_policy (one sample), planner_immediate, planner_temporal_self,
     planner_temporal_all, planner_proxy_views (temporal_all model, objective = views at 1 h)
   Success = helpful answer within 24 h (environment-verified, privileged truth).
"""

from __future__ import annotations

import argparse
import copy
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import RUNS, Timer, make_wm, save_json, threads  # noqa: E402

from agentrs.agent.belief import BeliefState  # noqa: E402
from agentrs.agent.temporal_runtime import act_with_delayed_outcome  # noqa: E402
from agentrs.content.policy import ContentGenConfig, ContentGeneratingPolicy, ContentPolicyActor, generation_prompt, op_text  # noqa: E402
from agentrs.content.training import ContentDemo, ContentRLConfig, content_bc_loss  # noqa: E402
from agentrs.evaluation.metrics import bootstrap_ci  # noqa: E402
from agentrs.evaluation.temporal_baselines import cap_example  # noqa: E402
from agentrs.models.backbone import BackboneConfig  # noqa: E402
from agentrs.models.policy import LMPolicy, PolicyConfig  # noqa: E402
from agentrs.planning.temporal_planner import TemporalContentPlanner, TemporalPlannerConfig  # noqa: E402
from agentrs.sensors.observation_network import NetworkConfig  # noqa: E402
from agentrs.sensors.socialweblab_source import lab_temporal_spec  # noqa: E402
from agentrs.socialweblab.agent_tasks import (  # noqa: E402
    HelpTeacher, SocialWebLabContentEnv, make_help_task, own_context, space_for, verify_help)
from agentrs.socialweblab.apps import content_features  # noqa: E402
from agentrs.socialweblab.engine import SocialWebLab  # noqa: E402
from agentrs.temporal.system import build_network  # noqa: E402
from agentrs.training.common import GradientTrainer, OptimConfig, param_change, param_snapshot  # noqa: E402
from agentrs.training.delayed_reward_trainer import DelayedRewardConfig, DelayedRewardTrainer, record_matured_rewards  # noqa: E402
from agentrs.training.temporal_world_model_trainer import (  # noqa: E402
    TemporalTrainConfig, TemporalTrainingDataBuilder, TemporalWorldModelTrainer)

DAY = 86400.0


def help_verifier(lab):
    def v(a):
        item = (a["context"] or {}).get("object_ref")
        if item is None:
            return "censored", None, None
        ver = verify_help(lab, int(item), a["action_time"], DAY)
        if ver.evidence.get("pending"):
            return "pending", None, None
        return "matured", float(ver.success), a["action_time"] + DAY
    return v


def advance(lab, net, seconds, step=900.0):
    end = lab.now() + seconds
    while lab.now() < end - 1e-6:
        lab.advance(min(step, end - lab.now()))
        if net is not None:
            net.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-demos", type=int, default=400)
    ap.add_argument("--bc-steps", type=int, default=400)
    ap.add_argument("--rl-iters", type=int, default=16)
    ap.add_argument("--tasks-per-iter", type=int, default=8)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--wm-steps", type=int, default=500)
    ap.add_argument("--eval-test", type=int, default=48)
    ap.add_argument("--eval-env", type=int, default=24)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    threads()
    torch.manual_seed(a.seed)
    out = RUNS / f"expF_s{a.seed}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    TT = Timer()
    res = {"seed": a.seed, "args": vars(a), "phases": {}}
    lab = SocialWebLab(out / "lab.db", seed=100 + a.seed, n_users=80, lab_id=f"taskF{a.seed}")
    net = build_network(out / "net", lab, None, NetworkConfig(scheduling="fixed", forecasts_enabled=False),
                        split_fn=lambda o: "train")
    env = SocialWebLabContentEnv(lab)
    accounts = [f"agent{i}" for i in range(4)]
    for acc in accounts:
        env.ensure_account(acc, followers=3)
    advance(lab, net, DAY)
    # ---------------------------------------------------------------- 1. imitation
    T = Timer()
    bb = BackboneConfig(name="EleutherAI/pythia-14m", max_prompt_tokens=320, seed=a.seed)
    cp = ContentGeneratingPolicy(LMPolicy(PolicyConfig(backbone=bb)), ContentGenConfig(max_content_tokens=64),
                                 checkpoint_id="bc")
    teacher = HelpTeacher(seed=a.seed)
    demos = []
    for i in range(a.bc_demos):
        t = make_help_task("train", 20000 + i, account=accounts[i % 4])
        obs = env.reset(t)
        bel = BeliefState(task=t.public())
        bel.reset(obs)
        cands = [c for c in env.available_actions() if c.type.value != "finish"]
        op, content = teacher.write(t)
        ops = [op_text(c) for c in cands]
        k = [c.name for c in cands].index(op)
        demos.append(ContentDemo(bel.prompt(), ops, k, generation_prompt(bel.prompt(), op), cp.demo_token_ids(content), 1.0))
    opt = GradientTrainer(cp.policy, OptimConfig(lr=5e-4, total_steps=a.bc_steps, warmup_steps=20, schedule="cosine"))
    rng = random.Random(a.seed)
    cp.policy.train()
    for _ in range(a.bc_steps):
        loss, st = content_bc_loss(cp, rng.sample(demos, 16))
        opt.step(loss)
    cp.policy.eval()
    res["phases"]["bc"] = {"demos": len(demos), "last_loss": float(loss.detach()), **st, "wall_s": T.s()}
    bc_policy = copy.deepcopy(cp.policy)
    ref = ContentGeneratingPolicy(copy.deepcopy(cp.policy), cp.cfg, checkpoint_id="bc-ref")
    # ---------------------------------------------------------------- 2. delayed-reward RL
    T = Timer()
    rl_cfg = DelayedRewardConfig(rl=ContentRLConfig(optim=OptimConfig(lr=3e-5, total_steps=10_000), minibatch_size=16,
                                                    kl_coef=0.02, epochs=2))
    prt = DelayedRewardTrainer(cp, net.ledger, rl_cfg, ref=ref, seed=a.seed)
    snap = param_snapshot(cp.policy)
    curve = []
    actor = ContentPolicyActor(cp, mode="sample", seed=a.seed)
    task_i = 0
    for it in range(a.rl_iters):
        posted = []
        for j in range(a.tasks_per_iter):
            t = make_help_task("train", task_i, account=accounts[j % 4])
            task_i += 1
            for _g in range(a.group):
                posted.append(act_with_delayed_outcome(env, t, actor, net, allow_decline=False))
        advance(lab, net, DAY / 2)
        rec = record_matured_rewards(net.ledger, help_verifier(lab), rl_cfg.objective)
        upd = prt.update()
        curve.append({"iter": it, "posted": len(posted), "rewards": rec, "mean_matured_reward": upd.get("mean_reward"),
                      "n_matured": upd.get("n_matured"), "policy": cp.checkpoint_id,
                      "stale": upd.get("too_stale_or_unknown_checkpoint", 0), "kl": upd.get("kl_ref"),
                      "token_ratio": upd.get("token_ratio_mean")})
        print("rl", it, curve[-1], flush=True)
    advance(lab, net, DAY)
    record_matured_rewards(net.ledger, help_verifier(lab), rl_cfg.objective)
    final = prt.update()
    res["phases"]["rl"] = {"curve": curve, "final_update": {k: v for k, v in final.items() if not isinstance(v, dict)},
                           "param_change": param_change(cp.policy, snap), "wall_s": T.s(),
                           "self_posts": task_i * a.group}
    # ---------------------------------------------------------------- 3. world models
    T = Timer()
    spec = lab_temporal_spec()
    b = TemporalTrainingDataBuilder(net.ledger, spec)
    self_ex = b.build(source_types=["self"], cutoffs="observations", max_cutoffs=2)
    third_ex = b.build(source_types=["third_party"], cutoffs="observations", max_cutoffs=2, groups=["forum"])
    wcfg = TemporalTrainConfig(steps=a.wm_steps, batch_size=16, seed=a.seed,
                               optim=OptimConfig(lr=2e-4, total_steps=a.wm_steps, schedule="cosine", warmup_steps=30))
    wms = {}
    for name, data in (("immediate", [cap_example(e, 300.0) for e in self_ex if e.cutoff_elapsed_s == 0]),
                       ("temporal_self", self_ex), ("temporal_all", self_ex + third_ex)):
        m = make_wm(a.seed)
        info = TemporalWorldModelTrainer(m, wcfg, name=name).train(data)
        wms[name] = m
        res["phases"].setdefault("wm", {})[name] = {"n_examples": len(data), "last_loss": info["last_loss"]}
    res["phases"]["wm"]["wall_s"] = T.s()
    res["data"] = {"self_examples": len(self_ex), "third_party_examples": len(third_ex),
                   "third_party_actions_observed": net.ledger._q("SELECT COUNT(*) AS n FROM actions WHERE "
                                                                 "source_type='third_party'")[0]["n"]}
    # ---------------------------------------------------------------- 4. evaluation
    T = Timer()
    snap_path = out / "eval_start.db"
    lab.snapshot(snap_path)
    tasks = ([make_help_task("test", i, account=accounts[i % 4]) for i in range(a.eval_test)]
             + [make_help_task("test_env", i, account=accounts[i % 4]) for i in range(a.eval_env)])
    rl_policy = cp.policy

    def ctx(op):
        return own_context(lab, env.account_uid(), space_for(op) or "linux")

    arms = {
        "bc_policy": lambda: ContentPolicyActor(ContentGeneratingPolicy(bc_policy, cp.cfg, "bc"), "sample", a.seed + 7),
        "rl_policy": lambda: ContentPolicyActor(ContentGeneratingPolicy(rl_policy, cp.cfg, "rl"), "sample", a.seed + 7),
        "planner_immediate": lambda: TemporalContentPlanner(cp, wms["immediate"], TemporalPlannerConfig(
            num_candidates=a.k, horizon_s=300.0, seed=a.seed + 7), ctx),
        "planner_temporal_self": lambda: TemporalContentPlanner(cp, wms["temporal_self"], TemporalPlannerConfig(
            num_candidates=a.k, horizon_s=DAY, seed=a.seed + 7), ctx),
        "planner_temporal_all": lambda: TemporalContentPlanner(cp, wms["temporal_all"], TemporalPlannerConfig(
            num_candidates=a.k, horizon_s=DAY, seed=a.seed + 7), ctx),
        "planner_proxy_views": lambda: TemporalContentPlanner(cp, wms["temporal_all"], TemporalPlannerConfig(
            num_candidates=a.k, horizon_s=3600.0, objective="views", objective_kind="count", seed=a.seed + 7), ctx),
    }
    res["eval"] = {}
    for name, mk in arms.items():
        lab.restore(snap_path)
        actor_ = mk()
        rows = []
        for i, t in enumerate(tasks):
            r = act_with_delayed_outcome(env, t, actor_, None if False else _NullNet(), allow_decline=False)
            rows.append({"task": t.task_id, "split": t.split, "item": r["item_id"], "posted_at": lab.now(),
                         "space": r["space"], "right_space": r["space"] == t.private["space"], "content": r["content"],
                         "pred": (r["wm"].reward_prob if r["wm"] is not None else None)})
            lab.advance(DAY / 2 / len(tasks) * 2)
        lab.advance(DAY + 3600)
        for row in rows:
            v = verify_help(lab, row["item"], row["posted_at"], DAY)
            row["success"] = bool(v.success)
            it = lab.items("id=?", (row["item"],))[0] if row["item"] else None
            row["views_total"] = it["views"] if it else 0
            f = content_features(row["content"] or "", row["space"])
            row.update({"detail": f.detail, "shouting": f.shouting, "clickbait": f.clickbait, "polite": f.polite})
        summ = {}
        for split in ("test", "test_env", "all"):
            rr = [r for r in rows if split == "all" or r["split"] == split]
            s = [float(r["success"]) for r in rr]
            summ[split] = {"n": len(rr), "success": float(np.mean(s)), "ci95": list(bootstrap_ci(s)),
                           "right_space": float(np.mean([r["right_space"] for r in rr])),
                           "detail": float(np.mean([r["detail"] for r in rr])),
                           "shouting_or_clickbait": float(np.mean([r["shouting"] or r["clickbait"] for r in rr])),
                           "views": float(np.mean([r["views_total"] for r in rr]))}
        res["eval"][name] = {"summary": summ, "rows": rows}
        print("eval", name, {k: (round(v["success"], 3), round(v["right_space"], 2), round(v["detail"], 2))
                             for k, v in summ.items()}, flush=True)
        save_json(f"expF_s{a.seed}.json", res)
    res["phases"]["eval_wall_s"] = T.s()
    res["wall_s"] = TT.s()
    save_json(f"expF_s{a.seed}.json", res)


class _NullNet:
    """Evaluation episodes are not fed back into learning: a ledger-free stand-in."""

    class _L:
        def set_eligibility(self, *a, **k):
            pass

    def __init__(self):
        self.ledger = self._L()

    def register_self_action(self, rec, horizons=None):
        return rec.action_id

    def link_self_action(self, *a, **k):
        pass


if __name__ == "__main__":
    main()
