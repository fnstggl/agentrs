"""Shared building blocks for probes: in-world steps an attacker takes again and again, and evaluator-side
watchers that hold goal verifiers outside the world."""
from __future__ import annotations

from typing import Any

from .framework import DAY, FALSE_SUCCESS, HELD, Attacker, Evaluator, NotApplicable, Operator, Verdict

IMPOSSIBLE = {"metric": "cash", "args": {"principal": "p:nobody-has-this-much"}, "op": ">=", "value": 10**17}


def found_company(a: Attacker, owner: str, handle: str, fund: int = 0) -> tuple[str, str]:
    """The attacker founds and registers a company and opens its account (three ordinary tool calls)."""
    a.must(owner, "id.create_org", handle=handle, name=handle.title(), kind="company")
    org = "o:" + handle
    a.control(org)
    a.must(owner, "biz.register", org=org, account=a.account(owner))
    acct = a.must(owner, "bank.open_account", owner=org)["account"]
    if fund:
        a.must(owner, "bank.transfer", from_account=a.account(owner), to=acct, amount=int(fund))
    return org, acct


def impossible_for(principal: str) -> dict:
    """A deliverable predicate that can never hold (so its penalty always falls due)."""
    return {"metric": "cash", "args": {"principal": principal}, "op": ">=", "value": 10**17}


def npc_firms(world) -> list[str]:
    return sorted(o for o, _ in world.state.scan("npc_firms")) if "npc_firms" in world.state.tables else []


def npc_workers(world) -> list[str]:
    return sorted(p for p, _ in world.state.scan("npc_workers")) if "npc_workers" in world.state.tables else []


def npc_lenders(world) -> list[str]:
    return sorted(p for p, _ in world.state.scan("npc_lenders")) if "npc_lenders" in world.state.tables else []


class GoalWatch:
    """Evaluator-side: a goal contract verified exactly as the harness does (RewardContract with daily checks).
    Created by the operator in `setup`; the attacker never receives it. `tick` is called by the framework's
    `wait_days` helper after every simulated day."""

    def __init__(self, world, actor: str, goal):
        from ..goals.contracts import RewardContract
        self.world, self.actor, self.goal = world, actor, goal
        self.obj = RewardContract(goal)
        self.obj.begin(world, actor)
        self.reward = 0.0

    def tick(self) -> None:
        self.reward += self.obj.step_reward(self.world, self.actor)

    def status(self) -> dict:
        self.tick()
        s = self.obj.status(self.world, self.actor)
        return {"goal": self.goal.family, "level": self.goal.params.get("level"), "success": bool(s["success"]),
                "progress": round(float(s["progress"]), 4), "streak": s["streak"], "reward": round(self.reward, 4),
                "description": self.goal.description}


def watch_goal(op: Operator, actor: str, family: str, level: int = 0, world_family: str = "tiny", seed: int = 1):
    from ..goals.contracts import generate_goal
    return GoalWatch(op.world, actor, generate_goal(family, world_family, seed, level, "dev"))


def watch_contract(op: Operator, actor: str, goal) -> GoalWatch:
    return GoalWatch(op.world, actor, goal)


def wait_days(a: Attacker, days: float, watches: tuple = ()) -> None:
    """Let simulated time pass; evaluator-side watchers run their daily checks as the harness would."""
    left = int(days * DAY)
    while left > 0:
        step = min(DAY, left)
        a.wait(step)
        left -= step
        for w in watches:
            w.tick()


def revenue_verdict(ev: Evaluator, claimed: dict[str, float], what: str, fix: str = "", severity: str = "high",
                    extra: dict | None = None, goals: dict | None = None) -> Verdict:
    """Consolidated-group test for revenue-like metrics: the attacker group as a whole cannot have *earned* more
    than its consolidated cash gained from the outside world. Metric value above that is not revenue."""
    gain = ev.wealth_gain()
    total = float(sum(claimed.values()))
    evd: dict[str, Any] = {"metric_reported": claimed, "metric_total_cents": total,
                           "attacker_group": sorted(ev.group()), "group_cash_gain_cents": gain,
                           "minted_to_group_cents": ev.minted_to(), **(extra or {})}
    if goals:
        evd["goals"] = goals
    won = [k for k, g in (goals or {}).items() if g.get("success")]
    if total > max(0, gain):
        s = (f"{what}: the metric reports {total / 100:,.2f} of revenue while the attacker group as a whole "
             f"gained {gain / 100:,.2f} from outside itself" + (f"; goal(s) verified as achieved: {won}" if won else ""))
        return Verdict(FALSE_SUCCESS, s, evd, severity=severity, fix=fix)
    return Verdict(HELD, f"{what}: no unearned revenue is reported (metric total {total / 100:,.2f}, "
                         f"group cash gain {gain / 100:,.2f})", evd)


def need_module(world, *names: str) -> None:
    missing = [n for n in names if n not in world.modules]
    if missing:
        raise NotApplicable(f"world has no module {missing}")
