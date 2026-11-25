"""Diagnostic environments behind the contract: protocol conformance, determinism, no hidden-state leaks, cursors,
delayed attribution, mechanism change; delayed credit learned by a budget-conditioned PPO policy."""
import json

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import Action, EnvAdapter, GoalEvaluator, ObsKind
from superlearner.cam_unified.learn.diagnostics import (ChangingMechanismAdapter, DelayedRewardAdapter,
                                                        ProbeBanditAdapter)
from superlearner.cam_unified.learn.heads import BudgetEncoder, BudgetState, ValueHead
from superlearner.cam_unified.learn.meta import TrialRunner
from superlearner.cam_unified.learn.ppo import ActorCriticObjective
from superlearner.cam_unified.learn.trainer import CombinedTrainer

ADAPTERS = [lambda: ProbeBanditAdapter(4, noise_tv=True), lambda: DelayedRewardAdapter(4),
            lambda: ChangingMechanismAdapter(4)]


def _strip(o):
    d = o.to_dict()
    d.pop("caused_by", None)
    return d


@pytest.mark.parametrize("make", ADAPTERS)
def test_protocol_conformance_determinism_and_cursors(make):
    ad = make()
    assert isinstance(ad, EnvAdapter) and isinstance(ad.evaluator, GoalEvaluator)
    ob = ad.reset([3, 4, 5, 6])
    assert len(ob.observations) == 4 and not any(ob.done)
    tools = ad.available_tools()
    acts = []
    for tl in tools:
        t = tl[0]
        args = {}
        for k, sch in t.parameters["properties"].items():
            args[k] = sch.get("minimum", 0) if sch["type"] == "integer" else float(sch.get("minimum", 0.0))
        acts.append(Action(t.name, args))
    r1 = ad.execute(acts)
    ad.advance_time([r.duration for r in r1])
    log1 = [[_strip(o) for o in e.log] for e in ad.envs]
    ad.reset([3, 4, 5, 6])
    r2 = ad.execute([Action(a.tool, dict(a.args)) for a in acts])
    ad.advance_time([r.duration for r in r2])
    assert [[_strip(o) for o in e.log] for e in ad.envs] == log1        # same seeds -> same world
    evs, cur = ad.collect_events([0, 0, 0, 0])
    assert [len(e) for e in evs] == [len(e.log) for e in ad.envs]
    evs2, cur2 = ad.collect_events(cur)
    assert all(len(e) == 0 for e in evs2) and cur2 == cur                 # lossless, no repeats
    assert ad.execute([None, None, None, None]) == [None] * 4
    with pytest.raises(ValueError):
        ad.advance_time([-1.0, 0, 0, 0])
    res = ad.evaluator.evaluate_goal(0)
    assert "hidden" in res.details and "return" in res.metrics


def test_probe_bandit_hidden_arm_never_leaks_before_the_probe():
    ad = ProbeBanditAdapter(6, horizon=5)
    ob = ad.reset(list(range(100, 106)))
    goods = {ad.evaluator.evaluate_goal(i).details["hidden"]["good_arm"] for i in range(6)}
    assert len(goods) > 1
    first = [json.dumps([_strip(o) for o in obs], sort_keys=True) for obs in ob.observations]
    assert len(set(first)) == 1                                         # identical agent views across hidden arms
    tl = [json.dumps([t.to_dict() for t in tools], sort_keys=True) for tools in ad.available_tools()]
    assert len(set(tl)) == 1 and len({json.dumps(g.to_dict(), sort_keys=True, default=str).split('"goal_id"')[0]
                                      for g in ad.goals()}) >= 1
    res = ad.execute([Action("probe")] * 6)
    for i, r in enumerate(res):
        g = ad.evaluator.evaluate_goal(i).details["hidden"]["good_arm"]
        sig = {o.entity: o.value for o in r.observations}
        assert sig[f"arm_{g}"] == 1.0 and sum(sig.values()) == 1.0 and r.cost == 30.0
        assert all(o.caused_by == res[0].observations[0].caused_by or o.caused_by for o in r.observations)
    g0 = ad.evaluator.evaluate_goal(0).details["hidden"]["good_arm"]
    r = ad.execute([Action("pull", {"arm": g0})] + [Action("pull", {"arm": 99})] * 5)
    assert r[0].observations[0].value == 100.0 and not r[1].ok and r[1].error == "invalid_argument"
    assert ad.evaluator.evaluate_goal(0).metrics["return"] == pytest.approx(70.0)


def test_delayed_reward_payoff_arrives_later_attributed_to_its_action():
    ad = DelayedRewardAdapter(1, horizon=8, delay_days=1.0, decision_days=0.25)
    ad.reset([0])
    a = Action("invest")
    r = ad.execute([a])[0]
    assert r.cost == 50.0 and r.observations[0].kind == ObsKind.TRANSACTION
    seen = []
    for k in range(3):                                   # 0.75 days later: nothing yet
        seen += ad.advance_time([0.25]).observations[0]
        ad.execute([Action("safe")])
    seen += ad.advance_time([0.25]).observations[0]      # t = 1.0: the payoff arrives as a world event
    pay = [o for o in seen if o.kind == ObsKind.EVENT]
    assert len(pay) == 1 and pay[0].caused_by == a.action_id and pay[0].value == 150.0 and pay[0].t == 1.0
    assert ad.evaluator.evaluate_goal(0).metrics["return"] == pytest.approx(-50 + 30 + 150)
    w = ad.execute([Action("wait", {"days": 5.0})])[0]
    assert not w.ok and w.error == "invalid_argument"
    # a late investment whose payoff would arrive after the episode is lost
    ad2 = DelayedRewardAdapter(1, horizon=2, delay_days=1.0, decision_days=0.25)
    ad2.reset([0])
    ad2.execute([Action("invest")])
    ad2.advance_time([0.25])
    ad2.execute([Action("safe")])
    ob = ad2.advance_time([0.25])
    assert ob.done[0] and ad2.evaluator.evaluate_goal(0).metrics["return"] == pytest.approx(-40.0)


def test_changing_mechanism_flips_and_an_adaptive_policy_beats_a_stale_one():
    def run(window):
        ad = ChangingMechanismAdapter(32, horizon=30, noise=0.05)
        ob = ad.reset(list(range(32)))
        hist = [[] for _ in range(32)]
        total = 0.0
        while not all(ob.done):
            acts = []
            for b in range(32):
                h = hist[b][-window:] if window else hist[b]
                num = sum(l * y for l, y in h)
                den = sum(l * l for l, y in h)
                k = num / den if den > 1e-6 else 0.0
                lv = 0.5 if abs(k) < 0.2 else max(-1.0, min(1.0, 0.8 / k))
                acts.append(Action("set_level", {"level": lv}))
            res = ad.execute(acts)
            for b, r in enumerate(res):
                o = r.observations[-1]
                hist[b].append((o.meta["level"], o.value))
            ob = ad.advance_time([r.duration for r in res])
        return sum(ad.evaluator.evaluate_goal(b).metrics["return"] for b in range(32)) / 32, ad
    adaptive, ad = run(window=3)
    stale, _ = run(window=0)
    assert adaptive > stale + 5.0, (adaptive, stale)
    h = ad.evaluator.evaluate_goal(0).details["hidden"]
    assert h["gain"] * h["gain_after"] < 0 and 10 <= h["change_at"] <= 20


class _BudgetPolicy(nn.Module):
    """Feed-forward policy that sees only the BUDGET token: invest or safe."""

    def __init__(self, D=32):
        super().__init__()
        torch.manual_seed(0)
        self.enc = BudgetEncoder(D)
        self.pi = nn.Linear(D, 2)
        self.value = ValueHead(D, n_heads=4)
        self.version = 0

    def initial_state(self, B):
        return None

    def _fwd(self, b6):
        tok = self.enc(BudgetState(*(b6[:, i] for i in range(6))))
        return self.pi(tok.x[:, 0]), self.value(tok.x, extra=None)

    @torch.no_grad()
    def act(self, state, obs, tools, budget, goals, active, greedy=False):
        b6 = budget.stack()
        logits, v = self._fwd(b6)
        d = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else d.sample()
        acts = [Action("invest" if int(x) == 0 else "safe") for x in a]
        return acts, d.log_prob(a), v.value, None, {"b6": b6, "a": a}

    def recompute(self, traj):
        B, T = traj.reward.shape
        b6 = traj.extras["b6"].reshape(B * T, 6)
        logits, v = self._fwd(b6)
        lp = torch.log_softmax(logits, -1)
        a = traj.extras["a"].reshape(-1).long()
        logp = lp.gather(-1, a.unsqueeze(-1)).view(B, T)
        ent = -(lp.exp() * lp).sum(-1).view(B, T)
        vl = torch.cat([v.logits.view(B, T, -1), torch.zeros(B, 1, v.logits.shape[-1])], 1)
        return logp, ent, vl


def test_budget_conditioned_policy_learns_delayed_credit():
    pol = _BudgetPolicy()
    cfg = make_config("dev", lr=1e-2)
    ad = DelayedRewardAdapter(64, horizon=8, delay_days=1.0, decision_days=0.25)
    runner = TrialRunner(ad, ad.evaluator)
    obj = ActorCriticObjective(kappa=cfg.kappa, gae_lambda=0.95, entropy_coef=0.003)
    tr = CombinedTrainer(cfg, core=pol.enc, heads=[*pol.pi.parameters(), *pol.value.parameters()],
                         stage="actor_critic", amp=False)
    for it in range(60):
        res = runner.run(pol, list(range(it * 64, it * 64 + 64)))
        prep = obj.prepare(res.traj, pol.version)
        for _ in range(4):
            logp, ent, vl = pol.recompute(res.traj)
            d = obj.loss(res.traj, prep, logp, ent, value_logits=vl, value_head=pol.value)
            tr.step({"actor": d["actor"], "value": d["value"]})
        pol.version += 1
    res = runner.run(pol, list(range(10_000, 10_064)), greedy=True)
    acts = res.actions[0]
    assert acts[:5] == ["invest"] * 5 and acts[5:] == ["safe"] * 3, acts     # invest while the payoff can arrive
    ret = float(res.per_trial_return.sum(1).mean())
    assert ret == pytest.approx(ad._oracle_return(ad.envs[0]), abs=1e-6)


def test_scores_are_normalised_against_the_oracle_for_every_family():
    from superlearner.cam_unified.learn.diagnostics import normalised_score
    assert normalised_score(50.0, 100.0) == 0.5 and normalised_score(-5.0, 100.0) == 0.0
    assert normalised_score(-2.0, -1.0) == 0.5 and normalised_score(-0.5, -1.0) == 1.0
    assert normalised_score(0.0, 0.0) == 1.0 and normalised_score(-1.0, 0.0) == 0.5
    # changing mechanism: the oracle knows the gain; its expected return is -sum(noise^2 + unreachable residual)
    ad = ChangingMechanismAdapter(64, horizon=12, noise=0.1)
    ob = ad.reset(list(range(64)))
    while not all(ob.done):
        acts = []
        for e in ad.envs:
            g = e.hidden["gain"] if e.decisions < e.hidden["change_at"] else e.hidden["gain_after"]
            acts.append(Action("set_level", {"level": max(-1.0, min(1.0, ad.target / g))}))
        res = ad.execute(acts)
        ob = ad.advance_time([r.duration for r in res])
    oracle_like = [ad.evaluator.evaluate_goal(b) for b in range(64)]
    want = sum(r.details["oracle_return"] for r in oracle_like) / 64
    got = sum(r.metrics["return"] for r in oracle_like) / 64
    assert want < 0 and abs(got - want) < 0.1 * abs(want) + 0.02
    assert sum(r.score for r in oracle_like) / 64 > 0.8
    ad.reset(list(range(64)))
    ob = ad.reset(list(range(64)))
    while not all(ob.done):                                  # a policy that ignores the gauge
        res = ad.execute([Action("set_level", {"level": 0.0})] * 64)
        ob = ad.advance_time([r.duration for r in res])
    lazy = sum(ad.evaluator.evaluate_goal(b).score for b in range(64)) / 64
    assert 0.0 < lazy < 0.1
