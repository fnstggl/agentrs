"""Collect an observational corpus from SocialWebLab: simulated users act for N days; our agent's
test accounts post help requests (scripted author: self experience); a sensor network observes
everything through the public feed and public item pages (fixed horizon schedule) and stores it in
the delayed-outcome ledger. Splits are by *time* (train: actions in the first ``train_days``; test:
actions in the following days, outcomes followed to the end) and by *environment* (spaces
``databases`` and ``webdev`` and the whole ``codereview`` app never enter training)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, Timer, save_json  # noqa: E402

from agentrs.actions.schema import Action  # noqa: E402
from agentrs.agent.temporal_runtime import act_with_delayed_outcome  # noqa: E402
from agentrs.sensors.observation_network import NetworkConfig  # noqa: E402
from agentrs.socialweblab.agent_tasks import HelpTeacher, SocialWebLabContentEnv, make_help_task  # noqa: E402
from agentrs.socialweblab.engine import LAB_EPOCH, SocialWebLab  # noqa: E402
from agentrs.temporal.system import build_network  # noqa: E402

HELD_ENV_SPACES = {"databases", "webdev", "repo-core", "repo-ui"}


class _Teacher:
    name = "help-teacher"

    def __init__(self, seed, tasks):
        self.t = HelpTeacher(seed)
        self.tasks = tasks

    def begin_episode(self, b):
        pass

    def decide(self, belief, cands):
        from agentrs.agent.runtime import Decision

        op, c = self.t.write(self.tasks[belief.task.task_id])
        return Decision(action=Action.submit(op, c), selection="teacher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--train-days", type=float, default=9)
    ap.add_argument("--test-days", type=float, default=2)
    ap.add_argument("--users", type=int, default=80)
    ap.add_argument("--self-per-day", type=int, default=12)
    a = ap.parse_args()
    out = RUNS / f"corpus_s{a.seed}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    T = Timer()
    lab = SocialWebLab(out / "lab.db", seed=a.seed, n_users=a.users, lab_id=f"corpus{a.seed}")
    t_train_end = LAB_EPOCH + a.train_days * 86400
    t_test_end = t_train_end + a.test_days * 86400

    def split_of(t, space):
        if space in HELD_ENV_SPACES:
            return "test_env" if t >= t_train_end and t < t_test_end else "env_other"
        if t < t_train_end:
            return "train"
        if t < t_test_end:
            return "test"
        return "late"

    net = build_network(out / "net", lab, None, NetworkConfig(scheduling="fixed", forecasts_enabled=False),
                        split_fn=lambda o: split_of(o.effective_action_time(), o.context_before.get("space")))
    env = SocialWebLabContentEnv(lab)
    for i in range(4):
        env.ensure_account(f"agent{i}", followers=[1, 3, 8, 20][i])
    tasks = {}
    actor = _Teacher(a.seed, tasks)
    k = 0
    n_steps = int(a.days * 96)
    for step in range(n_steps):
        lab.advance(900)
        if step % max(1, 96 // a.self_per_day) == 0 and lab.now() < t_test_end:
            split = "train" if lab.now() < t_train_end else "test"
            t = make_help_task("train" if split == "train" else "test", k, account=f"agent{k % 4}")
            tasks[t.task_id] = t
            res = act_with_delayed_outcome(env, t, actor, net, allow_decline=False)
            net.ledger._x("UPDATE actions SET split=? WHERE action_id=?", (split, res["action_id"]))
            k += 1
        net.step()
        if step % 96 == 0:
            print(f"day {step // 96} t={T.s()}s ledger={net.ledger.stats()}", flush=True)
    stats = {"seed": a.seed, "days": a.days, "users": a.users, "wall_s": T.s(), "lab": lab.stats(),
             "ledger": net.ledger.stats(), "network": net.stats, "scheduler": net.scheduler.stats(),
             "self_actions": k, "splits": {r["split"]: r["n"] for r in net.ledger._q(
                 "SELECT split, COUNT(*) AS n FROM actions GROUP BY split")}}
    save_json(f"corpus_s{a.seed}.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
