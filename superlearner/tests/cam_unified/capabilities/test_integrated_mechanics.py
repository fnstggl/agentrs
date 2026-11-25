"""Default-suite integrated tests (workstream Y): capabilities exercised through `build_model` with the real A-E
components and `CamUnifiedModel.decide` / `observe_outcomes` / `CamUnifiedAgent` (no test doubles), on the diagnostic
environments of `cam_unified.capabilities.envs`. These need no long training; the learned versions (trained agents,
several seeds) are the slow tests in test_capabilities_learned.py with their committed numbers in results/."""
from __future__ import annotations

import ast
import os

import pytest
import torch

from superlearner.cam_unified.capabilities import envs as E
from superlearner.cam_unified.capabilities.beliefs import BeliefSettings, evaluate_mechanism
from superlearner.cam_unified.capabilities.causal import measure_imagination_isolated, same_fast
from superlearner.cam_unified.capabilities.imagination import (config_override, frozen_imagined_beliefs,
                                                               imagination_settings)
from superlearner.cam_unified.capabilities.loop import Episode, mean
from superlearner.cam_unified.capabilities.reasoning import ReasoningSettings, evaluate_chain
from superlearner.cam_unified.capabilities.stack import build_stack, capability_config

CAP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "superlearner", "cam_unified",
                       "capabilities")


def test_capability_code_imports_no_test_doubles():
    """Production code of the capability package never imports cam_unified.testing or integration_doubles."""
    for fn in sorted(os.listdir(CAP_DIR)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(CAP_DIR, fn)).read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                assert "testing" not in n.split(".") and "integration_doubles" not in n, (fn, n)


@pytest.fixture(scope="module")
def stack(semantic_ok, device, tmp_path_factory):
    cfg = capability_config(use_imagination=False, use_memory=False)
    return build_stack(cfg, device=device, seed=0, memory_root=str(tmp_path_factory.mktemp("mem")))


def test_stack_is_the_real_assembly(stack):
    rep = stack.real_component_report()
    for name in ("semantic", "tokenizer", "core", "entity_encoder", "reasoner", "beliefs", "bayes", "outcome_head",
                 "world_model", "action_generator", "value_head", "exploration_head"):
        assert rep[name] is not None and rep[name].startswith("superlearner.cam_unified."), (name, rep[name])
        assert "testing" not in rep[name] and "integration_doubles" not in rep[name]
    assert rep["semantic"].endswith("QwenSemantic") and rep["action_generator"].endswith("ActionGenerator")
    assert stack.semantic.model_name == stack.cfg.sem_test_model


def test_a05_new_observations_change_the_predictive_reward_distribution(stack):
    """A05 in the integrated agent: after k observed payouts the predicted reward of the next pull (the value
    recorded in the DecisionTrace) moves toward the active mechanism, in opposite directions for the two mechanisms,
    and the movement is caused by the fast state (same action-conditioned latent scored with the current vs the
    episode's initial FastState)."""
    st = BeliefSettings(eval_rows=16, horizon=6)
    hi = evaluate_mechanism(stack, st, "high", 100_000)["steps"]
    lo = evaluate_mechanism(stack, st, "low", 101_000)["steps"]
    assert hi[-1]["pred_mean"] - hi[0]["pred_mean"] > 0.5, [r["pred_mean"] for r in hi]
    assert lo[-1]["pred_mean"] - lo[0]["pred_mean"] < -0.5, [r["pred_mean"] for r in lo]
    assert hi[-1]["abs_err"] < hi[0]["abs_err"] and lo[-1]["abs_err"] < lo[0]["abs_err"]
    assert hi[-1]["belief_effect_full"] > 0.5 and lo[-1]["belief_effect_full"] < -0.5
    assert hi[-1]["pred_var"] < hi[1]["pred_var"]          # uncertainty contracts as evidence accrues


def test_a06_misspecification_expands_uncertainty_only_for_the_unfamiliar_mechanism(stack):
    """A06 (misspecification): a payout mechanism far outside the familiar ones drives the predictive surprise up,
    triggers B's CUSUM alarm and uncertainty expansion, and the reward predictive variance exceeds the familiar
    one; a familiar mechanism triggers no expansion."""
    st = BeliefSettings(eval_rows=16, horizon=8)
    fam = evaluate_mechanism(stack, st, "high", 102_000)
    unf = evaluate_mechanism(stack, st, "extreme", 103_000)
    assert fam["expansions_total"] == 0
    assert unf["expansions_total"] > 0
    assert mean(r["surprise_z"] for r in unf["steps"][1:]) > mean(r["surprise_z"] for r in fam["steps"][1:]) + 1.0
    assert max(r["pred_var"] for r in unf["steps"]) > 3 * fam["steps"][-1]["pred_var"]


def test_a03_reasoning_state_reaches_the_actual_decisions(stack):
    """A03 mechanics: with the same weights and inputs, forcing the number of latent iterations changes the decisions
    `decide` returns (the tool pointer reads R), and the adaptive halting policy reports the iterations it computed."""
    st = ReasoningSettings(eval_rows=64)
    model = stack.model
    base = evaluate_chain(model, st)
    changed = []
    for j in (1, 16):
        r = evaluate_chain(model, st, j)
        changed.append(mean(float(a != b) for a, b in zip(r["tools"], base["tools"])))
    assert max(changed) > 0.05, changed
    iters = [v["iters"] for v in base["per_hop"].values()]
    assert all(1 <= i <= model.cfg.max_reason_iters for i in iters)


def test_c03_7_imagination_never_touches_environment_archive_or_real_beliefs(stack):
    """C03 (6.6(7)) and A08 mechanics: a planner decision over imagined multistep futures leaves the environment,
    the archive and the real FastState bit-identical; imagined belief updates ran; provenance is IMAGINED."""
    out = measure_imagination_isolated(stack, 100_500)
    assert out["env_and_archive_unchanged"] and out["fast_state_unchanged"], out
    assert out["imagined_items"] > 0 and out["trajectory_provenance"] == "imagined"
    assert out["imagination_belief_updates"] > 0


def test_a08_planner_imagines_multistep_with_revised_beliefs(stack, device):
    """A08 mechanics in the integrated planner (LatentImaginationPlanner over CAM's own candidates): every candidate
    first action is rolled forward 2 imagined decisions; the FastState clone of each branch is revised by the
    imagined outcomes (its particle ESS departs from the real belief's), the second imagined action is sampled from
    the policy on that revised belief, and the real action is the argmax of the imagined scores. The
    frozen-belief ablation used by the learned test keeps every branch at the real belief."""
    model = stack.model
    env = E.ProbeEnv(8, horizon=3)
    for mode in ("full", "frozen"):
        ep = Episode(model, env, E.ScoreEvaluator(env), list(range(100_700, 100_708)), greedy=True,
                     generator=torch.Generator().manual_seed(5))
        w = torch.softmax(ep.state.fast.logw.float(), -1)
        real_ess = (1.0 / (w * w).sum(-1)).cpu()
        ctx = frozen_imagined_beliefs(model) if mode == "frozen" else torch.no_grad()
        with config_override(model, use_imagination=True), imagination_settings(model, 2, 4, 2), ctx:
            dec, _ = ep.decide()
        tr = dec.trajectory
        assert tr is not None and len(tr.steps) == 2 and tr.provenance.value == "imagined"
        rows = tr.branch_of.cpu()
        ess = torch.stack([s.belief_ess.float().cpu() for s in tr.steps], 1)       # [Bb, 2]
        dev_ess = (ess - real_ess[rows].unsqueeze(1)).abs()
        if mode == "full":
            assert float(dev_ess.max()) > 1e-3, dev_ess
        else:
            assert float(dev_ess.max()) < 1e-5, dev_ess
        for b, a in enumerate(dec.actions):
            sc = a.proposal["scores"]
            assert a.proposal["chosen"] == max(range(len(sc)), key=sc.__getitem__)
            assert a.source.value == "cam_planner"
        assert all(it["horizon"] == 2 and len(it["next_tools"]) == 2 for row in dec.imagined for it in row)


def test_fast_state_comparison_is_nan_aware(stack):
    f = stack.model.init_fast(2)
    g = f.clone()
    g.surprise[:] = float("nan")
    f.surprise[:] = float("nan")
    assert same_fast(f, g)
    g.logw[0, 0] += 1.0
    assert not same_fast(f, g)
