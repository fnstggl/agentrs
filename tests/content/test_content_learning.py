"""Parameterized semantic action learning: content is part of the action, visible to the world
model, compared by the planner, and learned by gradient updates from verified outcomes."""

import copy

import pytest
import torch

from agentrs.actions.schema import Action, ActionType, parse_action
from agentrs.agent.runtime import TeacherActor, run_episode
from agentrs.content.data import ContentExperienceDataset, observational_association, randomized_variant_test
from agentrs.content.evaluation import RewardHackingMonitor, action_regression, lm_perplexity, regression_check
from agentrs.content.planner import ContentOutcomePlanner, ContentPlannerConfig
from agentrs.content.policy import ContentGenConfig, ContentGeneratingPolicy, ContentPolicyActor
from agentrs.content.semantic_action import ContentActionEncoder, SemanticAction
from agentrs.content.training import (
    ContentBCConfig,
    ContentImaginationConfig,
    ContentPolicyRLTrainer,
    ContentPredictionTrainer,
    ContentRLConfig,
    ContentWMTrainConfig,
    build_content_rl_steps,
    content_wm_examples,
    eligible_for_pg,
    imagined_content_steps,
    train_content_bc,
)
from agentrs.content.world_model import ContentConditionedWorldModel, ContentWorldModelConfig
from agentrs.data.schemas import Provenance, Source
from agentrs.data.storage import ExperienceStore
from agentrs.demonstrations.content_teacher import ContentTeacher
from agentrs.environments.content.env import ContentEnv, ContentOutcomeVerifier, make_content_task
from agentrs.evaluation.benchmark import make_env
from agentrs.models.policy import LMPolicy, PolicyConfig
from agentrs.security.permissions import GuardedEnvironment, PermissionPolicy
from agentrs.training.common import OptimConfig, load_model, param_change, param_snapshot, save_checkpoint

SLOTS = ["Mon 9am", "Tue 2pm"]  # the recipient accepts mornings with p=0.9, afternoons with p=0.1
TEACH = Provenance(source=Source.TEACHER, source_id="teacher:content-v1")


def micro_task(split, i):
    return make_content_task("content_email", split, i, email_slots=SLOTS)


def canonical(who, slot):
    return f"Hi {who}, can we meet {slot}?"


@pytest.fixture(scope="module")
def micro_demos():
    env = make_env("content")
    return [t for i in range(40) for t in run_episode(env, micro_task("train", i), TeacherActor(ContentTeacher(i)),
                                                      TEACH).transitions]


@pytest.fixture(scope="module")
def bc_policy(scratch_bb, micro_demos):
    """Scratch policy imitating ALL demos (successful and failed): it learns the message format
    and has no preference between the two times."""
    torch.manual_seed(0)
    cp = ContentGeneratingPolicy(LMPolicy(PolicyConfig(backbone=scratch_bb)), ContentGenConfig(max_content_tokens=24))
    train_content_bc(cp, micro_demos, ContentBCConfig(steps=200, batch_size=16, w_failure=1.0,
                                                      optim=OptimConfig(lr=3e-3, total_steps=200)))
    return cp


def collect(cp, tasks, per_task=1, seed=0, env=None):
    env = env or make_env("content")
    return [run_episode(env, t, ContentPolicyActor(cp, "sample", seed=seed + 100 * k + j))
            for k, t in enumerate(tasks) for j in range(per_task)]


@torch.no_grad()
def am_log_odds(cp, tasks):
    """Exact log pi(content=morning) - log pi(content=afternoon) for canonical messages."""
    from agentrs.agent.belief import BeliefState
    from agentrs.content.policy import generation_prompt

    env, tot = ContentEnv(), 0.0
    for t in tasks:
        obs = env.reset(t)
        b = BeliefState(task=t.public())
        b.reset(obs)
        gp = generation_prompt(b.prompt(), "send_email")
        ids = [cp.demo_token_ids(canonical(t.private["who"], s)) for s in SLOTS]
        lp, m = cp.content_log_probs([gp, gp], ids)
        s = (lp * m).sum(-1)
        tot += float(s[0] - s[1])
    return tot / len(tasks)


# ----------------------------------------------------------------------------- representation
def test_semantic_action_keeps_complete_content_and_roundtrips(tmp_path):
    long = "Dear team,\n" + " ".join(f"point {i} with \"quotes\"" for i in range(400))
    sa = SemanticAction(operation="write_document", content=long, modality="document")
    a = sa.to_action()
    assert a.type == ActionType.SUBMIT and parse_action(a.to_str()).text == long
    assert sa.content_ref and len(sa.content_ref) == 64
    enc = ContentActionEncoder(None, chunk_tokens=100)
    parts = enc.chunked_action_texts(sa)
    assert len(parts) > 1 and all(p.startswith("submit(write_document, [part") for p in parts)
    # a transition carrying it survives Parquet storage losslessly
    env = ContentEnv()
    ep = run_episode(env, make_content_task("content_config", "train", 0), TeacherActor(ContentTeacher(0)), TEACH)
    tr = ep.transitions[0].model_copy(update={"semantic_action": sa})
    st = ExperienceStore(tmp_path)
    st.write_transitions([tr])
    back = st.load_transitions()[0]
    assert back.semantic_action.content == long and back.content_outcome == tr.content_outcome


def test_outcomes_depend_on_content_not_operation_or_length():
    t = make_content_task("content_email", "train", 3, email_slots=SLOTS)
    v = ContentOutcomeVerifier(t)
    a = canonical(t.private["who"], "Mon 9am")
    b = canonical(t.private["who"], "Tue 2pm")
    assert len(a) == len(b)  # same operation, same length
    acc_a = sum(ContentOutcomeVerifier(make_content_task("content_email", "train", i, email_slots=SLOTS)).check(
        canonical("x", "Mon 9am"), 1)["objective_met"] for i in range(60))
    acc_b = sum(ContentOutcomeVerifier(make_content_task("content_email", "train", i, email_slots=SLOTS)).check(
        canonical("x", "Tue 2pm"), 1)["objective_met"] for i in range(60))
    assert acc_a > 40 and acc_b < 20
    assert v.check("can we meet sometime?", 1)["format_valid"] is False
    c = make_content_task("content_code", "train", 1)
    good = f"{c.private['a']}*x+{c.private['b']}"
    assert ContentOutcomeVerifier(c).check(good, 1)["objective_met"]
    assert not ContentOutcomeVerifier(c).check(f"{c.private['b']}*x+{c.private['a']}", 1)["objective_met"] or \
        c.private["a"] == c.private["b"]
    assert ContentOutcomeVerifier(c).check("__import__('os')", 1)["format_valid"] is False  # never executed


# ----------------------------------------------------------------------------- world model
def test_world_model_trains_on_content_and_distinguishes_content(scratch_bb, micro_demos):
    """(1) different content -> different predictions; (2) trains on exact content-outcome pairs."""
    torch.manual_seed(0)
    wm = ContentConditionedWorldModel(ContentWorldModelConfig(backbone=scratch_bb, n_heads=3))
    env = make_env("content")
    data = []
    for i in range(160):  # authorised third-party examples: both times, observed outcomes
        t = micro_task("train", 200 + i)
        slot = SLOTS[i % 2]
        ep = run_episode(env, t, TeacherActor(_Fixed(slot)), TEACH)
        data += ep.transitions
    ex = content_wm_examples(wm, data)
    for e in ex[:20]:
        assert e.transition.action in e.prompt  # the exact content is in the model input
        assert e.content_targets[1] == float(bool(e.transition.content_outcome["objective_met"]))
    t = micro_task("val", 5)
    obs = env.reset(t)
    acts = [Action.submit("send_email", canonical(t.private["who"], s)).to_str() for s in SLOTS]
    p0 = wm.predict_content(t.goal, obs, acts)
    assert p0[0].content_probs != p0[1].content_probs  # same operation, different content, different input
    trainer = ContentPredictionTrainer(wm, ContentWMTrainConfig(steps=250, optim=OptimConfig(lr=2e-3, total_steps=10**5)))
    snap = param_snapshot(wm)
    r = trainer.train(data)
    assert param_change(wm, snap) > 0 and r["last10"] < r["first10"]
    p1 = wm.predict_content(t.goal, obs, acts)
    assert p1[0].p_objective > p1[1].p_objective + 0.3, (p1[0].p_objective, p1[1].p_objective)
    held = [tr for i in range(20) for tr in run_episode(env, micro_task("val", 40 + i), TeacherActor(_Fixed(SLOTS[i % 2])),
                                                         TEACH).transitions]
    ev = trainer.evaluate(held)
    assert ev["objective_auroc"] > 0.7


class _Fixed:
    """Demonstrator that always proposes one time (a third party's content)."""

    source_id = "teacher:fixed"

    def __init__(self, slot):
        self.slot = slot

    def reset(self):
        pass

    def act(self, task, obs, candidates=None):
        return Action.submit("send_email", f"Hi there, can we meet {self.slot}?")


# ----------------------------------------------------------------------------- content RL
def test_content_rl_gradients_step_and_checkpoint(bc_policy, tmp_path):
    """(3) nonzero gradients from a valid content-level objective, (4) a real optimizer step
    changes parameters, (5) checkpoint reload preserves the updated policy."""
    cp = copy.deepcopy(bc_policy)
    eps = collect(cp, [micro_task("train", 300 + i) for i in range(4)], per_task=4)
    for e in eps:
        sa = e.transitions[0].semantic_action
        lp, m = cp.content_log_probs([sa.generation_prompt], [sa.token_ids])
        assert torch.allclose(lp[0][m[0]], torch.tensor(sa.token_logprobs), atol=1e-4)  # exact stored ids
    trainer = ContentPolicyRLTrainer(cp, ContentRLConfig(optim=OptimConfig(lr=1e-3, total_steps=10**5), kl_coef=0.0))
    steps, rejected = build_content_rl_steps([e.transitions for e in eps], trainer.cfg)
    assert steps and not rejected
    assert any(s.advantage != 0 for s in steps)
    loss, st = trainer.loss(steps)
    cp.policy.zero_grad()
    loss.backward()
    gn = sum(float(p.grad.abs().sum()) for p in cp.policy.parameters() if p.grad is not None)
    assert gn > 0 and abs(st["token_ratio_mean"] - 1) < 1e-3
    cp.policy.zero_grad()
    snap = param_snapshot(cp.policy)
    trainer.update([e.transitions for e in eps])
    assert param_change(cp.policy, snap) > 0
    path = save_checkpoint(tmp_path / "pol", cp.policy, trainer.trainer)
    cp2 = ContentGeneratingPolicy(load_model(path), cp.cfg)
    sa = eps[0].transitions[0].semantic_action
    a, _ = cp.content_log_probs([sa.generation_prompt], [sa.token_ids])
    b, _ = cp2.content_log_probs([sa.generation_prompt], [sa.token_ids])
    assert torch.allclose(a, b, atol=1e-6)


def test_zero_advantage_gives_zero_policy_gradient_and_masks_padding(bc_policy):
    cp = copy.deepcopy(bc_policy)
    eps = collect(cp, [micro_task("train", 400)], per_task=3)
    trainer = ContentPolicyRLTrainer(cp, ContentRLConfig(kl_coef=0.0))
    steps, _ = build_content_rl_steps([e.transitions for e in eps], trainer.cfg)
    for s in steps:
        s.advantage = 0.0
    loss, _ = trainer.loss(steps)
    cp.policy.zero_grad()
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in cp.policy.parameters() if p.grad is not None) < 1e-8


def test_only_policy_sampled_content_is_used_for_policy_gradient(bc_policy, micro_demos):
    """Third-party content, planner choices and T != 1 drafts are not treated as on-policy samples;
    rewards come only from the environment."""
    for t in micro_demos[:5]:
        ok, why = eligible_for_pg(t)
        assert not ok and "third-party" in why
    cp = copy.deepcopy(bc_policy)
    ep = collect(cp, [micro_task("train", 500)])[0]
    t = ep.transitions[0]
    assert eligible_for_pg(t)[0]
    hot = t.model_copy(update={"semantic_action": t.semantic_action.model_copy(update={"sampling_temperature": 1.5})})
    assert not eligible_for_pg(hot)[0]
    planned = t.model_copy(update={"selection": "planner"})
    assert not eligible_for_pg(planned)[0]
    # the reward the trainer sees is the environment's: a transition's model-predicted value is ignored
    steps, _ = build_content_rl_steps([[t]], ContentRLConfig(advantage="batch"))
    assert steps[0].advantage == 0.0  # single episode: no baseline contrast, no invented reward


def test_controlled_rl_improves_content_on_learnable_task(bc_policy):
    """(10) Goal-conditioned content RL from verified outcomes measurably shifts the generated
    content toward what succeeds, on held-out task instances (val seeds, unseen recipients' tasks)."""
    torch.manual_seed(0)
    cp = copy.deepcopy(bc_policy)
    held = [micro_task("val", i) for i in range(12)]
    before = am_log_odds(cp, held)
    trainer = ContentPolicyRLTrainer(cp, ContentRLConfig(optim=OptimConfig(lr=1e-3, total_steps=10**5), kl_coef=0.0))
    for it in range(12):
        eps = collect(cp, [micro_task("train", 600 + it * 4 + k) for k in range(4)], per_task=6, seed=it * 1000)
        trainer.update([e.transitions for e in eps])
    after = am_log_odds(cp, held)
    assert after > before + 1.0, (before, after)


# ----------------------------------------------------------------------------- planning + grounding
def test_planner_compares_content_candidates_and_can_decline(bc_policy, scratch_bb):
    """(6) multiple generated content candidates are scored before one is executed; the planner
    declines when predictions are unreliable or the operation is not authorised."""
    cp = copy.deepcopy(bc_policy)
    wm = ContentConditionedWorldModel(ContentWorldModelConfig(backbone=scratch_bb, n_heads=3))
    pl = ContentOutcomePlanner(cp, wm, ContentPlannerConfig(num_candidates=6, revision_rounds=1, target_score=2.0))
    ep = run_episode(make_env("content"), micro_task("val", 1), pl)
    cands = pl.last_candidates
    assert len({c["content"] for c in cands}) >= 2 and pl.stats["candidates_scored"] >= 2
    chosen = ep.transitions[0].semantic_action.content
    assert chosen == max(cands, key=lambda c: c["score"])["content"]
    assert ep.transitions[0].selection == "planner" and pl.stats["revisions"] == 1
    cautious = ContentOutcomePlanner(cp, wm, ContentPlannerConfig(num_candidates=3, min_score_to_act=1.01))
    ep2 = run_episode(make_env("content"), micro_task("val", 2), cautious)
    assert ep2.transitions[0].action == "finish()" and cautious.stats["declined"] == 1
    no_email = PermissionPolicy(allowed_content_ops=frozenset({"apply_config"}))
    gated = ContentOutcomePlanner(cp, wm, ContentPlannerConfig(num_candidates=3), permissions=no_email)
    ep3 = run_episode(make_env("content"), micro_task("val", 3), gated)
    assert ep3.transitions[0].action == "finish()" and ep3.transitions[0].selection == "planner_declined"
    assert gated.stats["declined"] == 1 and gated.stats["candidates_scored"] == 0  # refused before drafting


def test_imagined_success_is_not_verified_success(bc_policy, scratch_bb):
    """(7) a world model that predicts certain success does not make it so; the real failure is
    recorded and the model learns from the discrepancy; imagined drafts are labelled imagined."""
    cp = copy.deepcopy(bc_policy)
    wm = ContentConditionedWorldModel(ContentWorldModelConfig(backbone=scratch_bb, n_heads=3))
    with torch.no_grad():
        wm.content_heads.w2.zero_()
        wm.content_heads.b2.fill_(8.0)  # P(objective) ~ 1 for every content
    env = make_env("content")
    t = make_content_task("content_email", "val", 7, email_slots=["Tue 2pm", "Wed 3pm"])  # afternoons only
    pl = ContentOutcomePlanner(cp, wm, ContentPlannerConfig(num_candidates=3, revision_rounds=0))
    fails = []
    for i in range(6):
        ep = run_episode(env, make_content_task("content_email", "val", 70 + i, email_slots=["Tue 2pm", "Wed 3pm"]), pl)
        tr = ep.transitions[0]
        assert tr.wm_prediction.event_probs[1] > 0.99  # predicted objective met
        fails.append(not ep.success)
        assert ep.success == bool(ep.verification["success"])  # success = verifier, not prediction
    assert sum(fails) >= 4
    steps, stats = imagined_content_steps(cp, wm, [ep.transitions[0]], ContentImaginationConfig(drafts_per_state=3,
                                                                                                max_epistemic=1.0))
    assert steps and all(s.imagined for s in steps) and stats["imagined_mean_p_objective"] > 0.99
    before = wm.predict_content(t.goal, ep.transitions[0].observation, [ep.transitions[0].action])[0].p_objective
    real = [tr for _ in range(4) for tr in ep.transitions]
    ContentPredictionTrainer(wm, ContentWMTrainConfig(steps=30, optim=OptimConfig(lr=3e-3, total_steps=10**5))).train(real)
    after = wm.predict_content(t.goal, ep.transitions[0].observation, [ep.transitions[0].action])[0].p_objective
    assert ep.success or after < before - 0.3


# ----------------------------------------------------------------------------- observation + safety
def test_learns_from_authorised_third_party_content(scratch_bb, micro_demos):
    """(8) imitation from other agents' content raises the policy's probability of that content
    on held-out tasks; the same data trains the world model."""
    torch.manual_seed(1)
    cp = ContentGeneratingPolicy(LMPolicy(PolicyConfig(backbone=scratch_bb)), ContentGenConfig(max_content_tokens=24))
    held = [micro_task("val", 20 + i) for i in range(6)]

    @torch.no_grad()
    def demo_lp():
        from agentrs.agent.belief import BeliefState
        from agentrs.content.policy import generation_prompt

        env, tot = ContentEnv(), 0.0
        for t in held:
            b = BeliefState(task=t.public())
            b.reset(env.reset(t))
            lp, m = cp.content_log_probs([generation_prompt(b.prompt(), "send_email")],
                                         [cp.demo_token_ids(canonical(t.private["who"], "Mon 9am"))])
            tot += float((lp * m).sum())
        return tot / len(held)

    b0 = demo_lp()
    r = train_content_bc(cp, micro_demos, ContentBCConfig(steps=80, optim=OptimConfig(lr=3e-3, total_steps=80)))
    assert r["n_demos"] > 0 and demo_lp() > b0 + 20
    ds = ContentExperienceDataset(micro_demos)
    assert len(ds) == len(micro_demos) and ds.by_source() == {"teacher": len(micro_demos)}
    assert all(row.content and row.goal for row in ds.rows)


def test_content_learning_regressions_are_reported_not_silent(scratch_bb, demo_data, micro_demos):
    """(9) content training of the SHARED network is checked against its high-level action skills
    and general text; an aggressive update is detected."""
    from agentrs.training.bc import BCConfig, make_examples, train_bc

    torch.manual_seed(0)
    train, val = demo_data
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    train_bc(pol, train, val, BCConfig(steps=60, optim=OptimConfig(lr=3e-3, total_steps=60)))
    mw_val = make_examples([t for t in val if t.episode_success], BCConfig())
    cp = ContentGeneratingPolicy(pol)
    before = {"action_regression": action_regression(pol, mw_val), "lm_perplexity": lm_perplexity(pol.backbone)}
    rehearsal = make_examples(train, BCConfig())
    gentle = copy.deepcopy(cp)
    train_content_bc(gentle, micro_demos, ContentBCConfig(steps=20, rehearsal_weight=1.0,
                                                          optim=OptimConfig(lr=1e-4, total_steps=20)), rehearsal=rehearsal)
    after_gentle = {"action_regression": action_regression(gentle.policy, mw_val)}
    train_content_bc(cp, micro_demos, ContentBCConfig(steps=60, optim=OptimConfig(lr=2e-2, total_steps=60)))
    after = {"action_regression": action_regression(pol, mw_val), "lm_perplexity": lm_perplexity(pol.backbone)}
    issues = regression_check(before, after, max_nll_increase=0.5)
    assert issues, (before, after)
    assert after_gentle["action_regression"]["val_nll"] < after["action_regression"]["val_nll"]


def test_permissions_and_rate_limits_apply_to_content_actions():
    pol = PermissionPolicy(allowed_content_ops=frozenset({"apply_config"}), approval_required_ops=frozenset({"apply_config"}),
                           max_content_submissions=1)
    env = GuardedEnvironment(ContentEnv(), pol)
    env.reset(make_content_task("content_email", "train", 0))
    r = env.step(Action.submit("send_email", "Hi, can we meet?"))
    assert not r.action_valid and "not permitted" in r.error
    env2 = GuardedEnvironment(ContentEnv(), pol, approve=lambda a: True)
    env2.reset(make_content_task("content_config", "train", 0))
    assert env2.step(Action.submit("apply_config", "retries=1")).action_valid  # approved
    assert "rate limit" in env2.step(Action.submit("apply_config", "retries=2")).error
    env3 = GuardedEnvironment(ContentEnv(), pol)  # approval required, none given
    env3.reset(make_content_task("content_config", "train", 0))
    assert "approval" in env3.step(Action.submit("apply_config", "retries=1")).error


def test_engagement_is_not_reward_and_hacking_is_flagged():
    env = ContentEnv()
    plain, hype, wrong_hype = [], [], []
    mon = RewardHackingMonitor()
    eps_a, eps_b = [], []
    for i in range(20):
        t = make_content_task("content_forum", "train", i)
        p = t.private
        for store, txt in ((plain, f"The capital of {p['place']} is {p['answer']}."),
                           (hype, f"The capital of {p['place']} is {p['answer']}. AMAZING!!!!"),
                           (wrong_hype, "WOW!!!! GREAT QUESTION!!!!")):
            env.reset(t)
            store.append(env.step(Action.submit("post_reply", txt)))
    eng = lambda rs: sum(r.info["content_outcome"]["engagement"] for r in rs) / len(rs)  # noqa: E731
    rew = lambda rs: sum(r.reward for r in rs) / len(rs)  # noqa: E731
    assert eng(hype) > eng(plain) + 50 and rew(hype) == rew(plain) == 1.0
    assert eng(wrong_hype) > eng(plain) and rew(wrong_hype) == 0.0  # more engagement, no success
    teach = TeacherActor(ContentTeacher(0, hype_prob=0.0))
    spam = TeacherActor(_Spam())
    for i in range(10):
        eps_a.append(run_episode(env, make_content_task("content_forum", "val", i), teach))
        eps_b.append(run_episode(env, make_content_task("content_forum", "val", i), spam))
    mon.record(0, eps_a)
    mon.record(1, eps_b)
    assert mon.flags()


class _Spam:
    source_id = "spam"

    def reset(self):
        pass

    def act(self, task, obs, candidates=None):
        return Action.submit("post_reply", "WOW!!!! MUST READ!!!!")


def test_observational_vs_randomised_effect_labels():
    env = ContentEnv()
    eps = [run_episode(env, make_content_task("content_forum", "train", i), TeacherActor(ContentTeacher(i)), TEACH)
           for i in range(40)]
    rows = ContentExperienceDataset([t for e in eps for t in e.transitions]).rows
    obs = observational_association(rows, lambda r: "!" in r.content, lambda r: r.engagement)
    assert obs["causal"] is False and obs["difference"] is not None
    assert all("posted_hour" in r.confounders for r in rows)
    with pytest.raises(PermissionError):
        randomized_variant_test(env, [], {}, lambda t: "post_reply")
    tasks = [make_content_task("content_forum", "train", 100 + i) for i in range(60)]
    rct = randomized_variant_test(
        env, tasks, {"plain": lambda t: f"It is {t.private['answer']}.",
                     "wrong": lambda t: "It is " + next(v for v in t.private["table"].values() if v != t.private["answer"])},
        lambda t: "post_reply", authorized=True)
    assert rct["causal"] is True and rct["difference"] < -0.8


def test_end_to_end_goal_generate_predict_execute_update_evaluate(bc_policy, scratch_bb, tmp_path):
    """Goal -> generate content -> predict effects -> execute in a controlled environment ->
    observe outcome -> verified reward -> update content-policy weights -> evaluate new checkpoint."""
    from agentrs.content.evaluation import evaluate_content

    cp = copy.deepcopy(bc_policy)
    wm = ContentConditionedWorldModel(ContentWorldModelConfig(backbone=scratch_bb, n_heads=3))
    env = make_env("content")
    tasks = [micro_task("train", 900 + i) for i in range(4)]
    pl = ContentOutcomePlanner(cp, wm, ContentPlannerConfig(num_candidates=3, revision_rounds=0))
    planned = [run_episode(env, t, pl) for t in tasks]  # predict-then-act
    for e in planned:
        tr = e.transitions[0]
        assert tr.wm_prediction is not None and tr.semantic_action.execution_status == "executed"
        assert tr.content_outcome is not None and tr.reward in (0.0, 1.0)
    onpol = collect(cp, tasks, per_task=4)  # on-policy samples for the policy gradient
    ContentPredictionTrainer(wm, ContentWMTrainConfig(steps=10)).train(
        [t for e in planned + onpol for t in e.transitions])
    trainer = ContentPolicyRLTrainer(cp, ContentRLConfig(optim=OptimConfig(lr=1e-3, total_steps=10**5)),
                                     ref=copy.deepcopy(bc_policy))
    snap = param_snapshot(cp.policy)
    st = trainer.update([e.transitions for e in planned + onpol])
    assert st["n_steps"] == 16 and "rejected[not sampled from the policy (planner/greedy/other selection)]" in st
    assert param_change(cp.policy, snap) > 0 and "kl_ref" in st
    path = save_checkpoint(tmp_path / "ckpt", cp.policy, trainer.trainer)
    new = ContentGeneratingPolicy(load_model(path), cp.cfg, checkpoint_id="content-rl-v1")
    m, eps = evaluate_content(env, [micro_task("val", 100 + i) for i in range(6)], ContentPolicyActor(new, "greedy"))
    assert m["n"] == 6 and all(e.transitions[0].semantic_action.policy_checkpoint == "content-rl-v1" for e in eps)
