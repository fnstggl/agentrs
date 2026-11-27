"""W1 (plan 2026-11-26 §2): runtime.run_lifetime is the ONE act -> execute -> advance loop of the package, the old
loops are wrappers over it, and the loop's v4 contracts hold: observation-only delivery (no reward reaches the
agent, beliefs or archived targets; the agent's state is bit-identical when the evaluator's payout changes), tool
results delivered by seq merge, cam-event/4 version stamps, the action-authorisation assertion, the LearnerHook
schedule (on_step / step / meta_step), labels to the LabelStore only, update-triggered checkpoints.

Behavioural tests use the CPU doubles of integration_doubles (unit-test doubles; the real dom_dev / dom checks are
test_dom_e2e_real.py and test_memory_budget_1b.py)."""
import ast
import importlib
import inspect
import os

import pytest
import torch

from conftest import Stack, make_cfg
from superlearner.cam_unified import runtime as R
from superlearner.cam_unified.envs.labels import LabelRecord, LabelStore
from superlearner.cam_unified.interfaces import (EVENT_SCHEMA, EVENT_VERSION_KEYS, Action, Observation, ObsKind)

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "superlearner", "cam_unified"))

# Loops that remain outside runtime.py, each with its reason and owner (change requests filed by W1). A NEW loop
# anywhere else fails this test; migrating one of these only shrinks the set.
KNOWN_OTHER_LOOPS = {
    "dist/actor.py": "declared non-DOM (plan §1 Distributed: DOM runs are single-process; dist deferred)",
    "dist/toyworld.py": "toy mechanism proofs (RETAIN, not CAM evidence)",
    "envs/integrity.py": "operator-side oracle reference on forks of an integrity world (no agent acts) [W2]",
    "learn/meta.py": "MetaRLObjective baseline harness (kept only as a baseline) [W6: migrate to run_lifetime]",
    "experiments/predict.py": "support collection for Experiment 1 predictions [W9: migrate to run_lifetime]",
    "capabilities/causal.py": "scripted intervention sequences of the causal diagnostic [W6: migrate]",
    "capabilities/memory.py": "cue evaluation of the memory diagnostic [unowned: migrate to run_lifetime]",
    "selfeval/integrity_harness.py": "Exp F harness: a state-independent tabular tool policy on integrity worlds (no "
                                     "CAM agent acts; psi meta-gradient sanity check) [W8]",
}
# the W1-owned runners that must be wrappers (no loop of their own)
WRAPPERS = ("envs/harness.py", "experiments/mechrun.py", "experiments/train.py", "capabilities/loop.py",
            "stage1/runner.py", "stage1/__main__.py", "stage1/system.py", "build.py")


def _attrs(node) -> set:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.add(n.func.id)
    return out


def act_execute_advance_loops() -> dict:
    """{relative path: [function names]} of every function with a for/while loop whose body references `execute` and
    `advance_time` / `advance` (called or passed as a callable)."""
    found: dict = {}
    for dp, _, fs in os.walk(PKG):
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            tree = ast.parse(open(p).read())
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for loop in ast.walk(fn):
                    if isinstance(loop, (ast.For, ast.While)):
                        a = _attrs(loop)
                        if "execute" in a and a & {"advance_time", "advance"}:
                            found.setdefault(os.path.relpath(p, PKG), []).append(fn.name)
                            break
    return found


def test_only_runtime_contains_the_act_execute_advance_loop():
    found = act_execute_advance_loops()
    assert found.get("runtime.py") == ["run_lifetime"], found.get("runtime.py")
    others = set(found) - {"runtime.py"}
    unexpected = others - set(KNOWN_OTHER_LOOPS)
    assert not unexpected, f"new act->execute->advance loops outside runtime.run_lifetime: " \
                           f"{ {k: found[k] for k in unexpected} }"
    assert not set(WRAPPERS) & set(found)


def test_old_entry_points_are_wrappers_over_run_lifetime():
    from superlearner.cam_unified.capabilities import loop as L
    from superlearner.cam_unified.envs import harness as H
    from superlearner.cam_unified.experiments import mechrun as M
    from superlearner.cam_unified.stage1 import runner as S1
    for fn in (H.run_lifetime, R.run_steps, M.run_batch, S1.run):
        assert "run_lifetime(" in inspect.getsource(fn), fn
    assert "run_steps(" in inspect.getsource(L.train_agent)
    assert H.LifetimeResult is R.LifetimeResult


def test_merge_by_seq_drops_redelivered_events_and_keeps_unsequenced():
    def o(seq, txt="x"):
        return Observation(ObsKind.EVENT if seq is not None else ObsKind.TOOL_RESULT, txt, seq=seq)
    out, top = R.merge_by_seq(3, [o(None, "res"), o(2), o(4)], [o(4), o(5)], None)
    assert [x.seq for x in out] == [None, 4, 5] and top == 5
    # one event expanded into several observations (same seq, one list) is delivered whole; a later list's copy is not
    out, top = R.merge_by_seq(0, [o(1, "a"), o(1, "b")], [o(1, "c"), o(2, "d")])
    assert [x.content for x in out] == ["a", "b", "d"] and top == 2


# ============================================================================ observation-only delivery

class ScaledEvaluator:
    """The toy evaluator with its payout multiplied by `k` (operator side only)."""

    def __init__(self, inner, k):
        self.inner, self.k = inner, k

    def evaluate_goal(self, b):
        r = self.inner.evaluate_goal(b)
        r.score = r.score * self.k
        return r


def _obs_only_run(k: float, rounds: int = 6, mode: str = "observation_only"):
    st = Stack(cfg=make_cfg(reward_mode=mode, reward_in_beliefs=True))
    seen = []
    res = R.run_lifetime(st.agent, st.env, evaluator=ScaledEvaluator(st.evaluator, k), max_decisions=rounds,
                         update_every=0, on_step=lambda s, tr: seen.append(tr), keep_rounds=True)
    return st, res, seen


def _fast_equal(a, b) -> bool:
    return all(torch.equal(getattr(a, n), getattr(b, n)) for n in ("particles", "logw", "pscale", "bayes_mu", "bayes_P",
                                                                      "bayes_noise", "bayes_n", "surprise"))


def test_observation_only_no_reward_reaches_the_agent_and_state_ignores_payout():
    st1, res1, seen = _obs_only_run(1.0)
    assert res1.reward_mode == "observation_only"
    assert all(tr.rewards is None for tr in seen)
    assert all(rec.reward == 0.0 and rec.extra["reward_mode"] == "observation_only"
               for tr in seen for rec in tr.records)
    rows = [r for r in st1.archive.rows if r.action is not None]
    assert rows and all(r.reward == 0.0 for r in rows)
    for r in rows:                                   # the archived reward target is unobserved
        rew = [o for o in r.extra["outcomes"] if o["key"] == "reward"]
        assert rew and all(not o["observed"] for o in rew)
    assert all(float(f.reward.abs().sum()) == 0.0 for f in st1.agent.fresh)
    assert st1.model.reward_in_beliefs is True       # restored after the masked belief update
    # the operator still records the verified return
    assert len(res1.rounds) == 6
    # altering the evaluator's payout changes nothing the agent holds (beliefs, Kalman layer, Z)
    st2, res2, _ = _obs_only_run(100.0)
    assert _fast_equal(st1.agent.state.fast, st2.agent.state.fast)
    assert torch.equal(st1.agent.state.Z, st2.agent.state.Z)
    # power check: with the reward delivered (legacy, reward in beliefs) the same payout change does move the beliefs
    l1, _, _ = _obs_only_run(1.0, mode="legacy")
    l2, _, _ = _obs_only_run(100.0, mode="legacy")
    assert not _fast_equal(l1.agent.state.fast, l2.agent.state.fast)
    assert [r.action.to_dict()["args"] for r in st1.archive.rows if r.action] == \
           [r.action.to_dict()["args"] for r in st2.archive.rows if r.action]


def test_legacy_mode_delivers_verified_increments():
    st = Stack()
    seen = []
    res = R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=5, update_every=0,
                         on_step=lambda s, tr: seen.append(tr))
    assert res.reward_mode == "legacy"               # the double config keeps the pre-v4 default
    assert sum(sum(tr.rewards) for tr in seen) == pytest.approx(sum(res.returns))
    assert all(rec.extra["reward_mode"] == "legacy" for tr in seen for rec in tr.records)


def test_tool_results_reach_the_next_decision_and_versions_are_stamped():
    st = Stack()
    inputs = []
    act0 = st.agent.act

    def spy(obs, tools):
        inputs.append([list(o) for o in obs.observations])
        return act0(obs, tools)
    st.agent.act = spy
    res = R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=3, update_every=0,
                         env_version="toyshop/1")
    assert all(any(o.kind is ObsKind.TOOL_RESULT for o in rows[b]) for rows in inputs[1:] for b in range(2))
    v = res.versions
    assert set(v) == set(EVENT_VERSION_KEYS) and v["env"] == "toyshop/1" and v["policy"] == st.agent.version
    assert v["reward_model"] is None                 # legacy: no actor reward model in force
    recs = [r for r in st.archive.rows if r.action is not None]
    assert recs and all(r.extra["schema"] == EVENT_SCHEMA and set(r.extra["versions"]) == set(EVENT_VERSION_KEYS)
                        for r in recs)


# ============================================================================ action authorisation

class Stamping:
    """Wraps an agent and stamps every action with a cam_eval record (stands in for W5's decision.select)."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def act(self, obs, tools):
        S = importlib.import_module("superlearner.cam_unified.decision.select")
        acts = self.inner.act(obs, tools)
        for a in acts:
            if a is not None:        # one candidate, selected with probability 1: a real (HMAC) stamp
                _, _, recs = S.select(torch.zeros(1, 1), torch.zeros(1, 1), greedy=True)
                S.attach_stamp(a, recs[0], [a], {"policy": 0, "reward_model": None})
        return acts


class CountingEnv:
    def __init__(self, inner):
        self.inner, self.n_exec = inner, 0

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def execute(self, actions):
        self.n_exec += 1
        return self.inner.execute(actions)


def test_dom_decision_path_refuses_unstamped_actions_before_execution():
    st = Stack(cfg=make_cfg(n_candidates=2))
    env = CountingEnv(st.env)
    with pytest.raises(R.ActionNotAuthorised):
        R.run_lifetime(st.agent, env, evaluator=st.evaluator, max_decisions=2, update_every=0)
    assert env.n_exec == 0
    st = Stack(cfg=make_cfg(n_candidates=2))
    res = R.run_lifetime(Stamping(st.agent), st.env, evaluator=st.evaluator, max_decisions=2, update_every=0)
    assert res.authorisation == {"required": True, "checked": 4, "source": "config"}
    st = Stack(cfg=make_cfg(n_candidates=2))
    res = R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=2, update_every=0,
                         require_cam_eval=False)
    assert res.authorisation["required"] is False and res.authorisation["source"] == "explicit"
    assert R.cam_eval_stamp(Action("x")) is None and R.dom_decision_path(make_cfg(n_intent=4))
    # a stamp that is present but forged / no longer matches the action is refused too (W5 verify_stamp)
    a = Stamping(st.agent).act(st.env.reset(), st.env.available_tools())[0]
    assert R.authorise([a], 0) == 1
    a.proposal["cam_eval"] = {**a.proposal["cam_eval"], "q": [9.0]}
    with pytest.raises(R.ActionNotAuthorised, match="invalid cam_eval stamp"):
        R.authorise([a], 0)
    with pytest.raises(R.ActionNotAuthorised, match="invalid"):
        R.authorise([Action("x", proposal={"cam_eval": {"q": 0.0, "k": 0}})], 0)


# ============================================================================ learner hook, labels, checkpoints

class RecLearner:
    def __init__(self):
        self.on, self.steps, self.metas = 0, [], 0
        self.reward_model = type("RM", (), {"version": 7})()

    def on_step(self, tr):
        self.on += 1

    def step(self):
        self.steps.append(self.on)
        return {"updated": True, "update_id": f"u{len(self.steps)}"}

    def meta_step(self):
        self.metas += 1
        return {"updated": True}

    def state_dict(self):
        return {"steps": list(self.steps)}

    def load_state_dict(self, d):
        self.steps = list(d["steps"])


class FakeLabels:
    def __init__(self):
        self.n = 0

    def take_labels(self):
        self.n += 1
        return [LabelRecord("shop0", self.n, "success", 1.0, 5.0, 0.1, "toy")] if self.n % 2 == 0 else []


def test_learner_hook_schedule_labels_and_update_checkpoints(tmp_path):
    st = Stack(cfg=make_cfg(reward_mode="observation_only", reward_in_beliefs=False, meta_every=2))
    ln = RecLearner()
    store = LabelStore(str(tmp_path / "labels"))
    res = R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, labels=FakeLabels(), label_store=store,
                         learner=ln, max_decisions=8, update_every=2,
                         checkpoint=R.CheckpointSpec(str(tmp_path / "ck"), every_updates=2))
    assert ln.on == 8 and ln.steps == [2, 4, 6, 8] and ln.metas == 2 and len(res.meta_updates) == 2
    assert res.labels_taken == 4 and res.labels_appended == 4 and len(store) == 4
    assert res.versions["reward_model"] == 7
    assert len(res.checkpoints) == 2 and all(os.path.exists(os.path.join(d, "learner.pt")) for d in res.checkpoints)
    assert torch.load(os.path.join(res.checkpoints[-1], "learner.pt"), weights_only=False) == {"steps": [2, 4, 6, 8]}


def test_labels_need_a_store_and_the_agent_never_holds_them(tmp_path):
    st = Stack()
    with pytest.raises(ValueError, match="label_store"):
        R.run_lifetime(st.agent, st.env, labels=FakeLabels(), max_decisions=1)
    store = LabelStore(str(tmp_path / "labels"))
    st.agent.side = {"store": store}
    with pytest.raises(ValueError, match="holds"):
        R.run_lifetime(st.agent, st.env, labels=FakeLabels(), label_store=store, max_decisions=1)


def test_default_learner_is_the_agent_update_with_a_reward_model_in_observation_only():
    st = Stack(cfg=make_cfg(reward_mode="observation_only", reward_in_beliefs=False))
    res = R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=4, update_every=2)
    assert len(res.updates) == 2 and all(u.get("updated") for u in res.updates)
    assert res.versions["reward_model"] == 0          # the seam RewardModel stub (W8 replaces it)
    assert st.agent.version == 2


def test_build_refuses_test_doubles():
    from superlearner.cam_unified.build import assert_no_doubles, make_cam_config
    st = Stack()
    with pytest.raises(ValueError, match="doubles"):
        assert_no_doubles(st.model)
    assert make_cam_config("dom_dev").preset == "dom_dev" and make_cam_config("dev", {"n_layers": 2}).n_layers == 2


def test_update_checkpoints_save_a_plain_agent_through_checkpoint_ops(tmp_path):
    class Ops:
        saved = []

        def save_full_lifetime(self, path):
            os.makedirs(path, exist_ok=True)
            self.saved.append(path)
    st = Stack()
    res = R.run_lifetime(st.agent, st.env, evaluator=st.evaluator, max_decisions=4, update_every=2,
                         checkpoint=R.CheckpointSpec(str(tmp_path / "ck"), every_updates=1, ops=Ops()))
    assert len(Ops.saved) == 2 and all(p.endswith("agent") for p in Ops.saved) and len(res.checkpoints) == 2


def test_guarded_env_turns_an_env_exception_into_a_reported_error_result():
    class Crashy(CountingEnv):
        def execute(self, actions):
            if actions[1] is not None:
                raise KeyError("re68")
            return self.inner.execute(actions)
    st = Stack()
    env = R.GuardedEnv(Crashy(st.env))
    res = R.run_lifetime(st.agent, env, evaluator=st.evaluator, max_decisions=2, update_every=0)
    assert res.steps == 2 and len(env.errors) == 2 and all(e["row"] == 1 for e in env.errors)
    bad = [r for r in st.archive.rows if r.action is not None and r.env_id == "shop1"]
    assert bad and all(not r.result.ok and r.result.error.startswith("env_exception") for r in bad)
