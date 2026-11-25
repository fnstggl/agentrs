"""CamUnifiedAgent + run_steps: 30-step end-to-end run, ablation switches, archive/trace contents, update path."""
import ast
import json
import math
import os

import pytest
import torch

from conftest import Stack, make_cfg
from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.agent import FreshStep, gae_semi_markov
from superlearner.cam_unified.interfaces import (EventRecord, ExecutionResult, Provenance, SequenceBatch, TokenKind,
                                                 TokenSet)
from superlearner.cam_unified.runtime import run_steps

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "superlearner", "cam_unified"))
HEADS = ("tokenizer", "entity_encoder", "reasoner", "beliefs", "outcome_head", "world_model", "action_generator",
         "value_head", "option_policy", "exploration_head", "query_encoder")


def params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def test_30_step_run_exercises_every_call_path():
    st = Stack()
    core0 = params(st.model.core)
    heads0 = {h: params(getattr(st.model, h)) for h in HEADS}
    rep = run_steps(st.agent, st.env, 30, evaluator=st.evaluator, update_every=5)
    calls = st.model.calls
    # every component of the step equation ran
    for k in ("core", "reasoner", "beliefs.tokens", "retriever.retrieve", "planner", "action_generator.sample",
              "world_model.prior", "imagination.beliefs.update", "beliefs.update", "bayes.update",
              "exploration_head"):
        assert calls.get(k, 0) > 0, k
    assert st.model.tokenizer.tokens_encoded > 0 and st.model.retriever.n_retrieve == 30
    # ---- archive: one REAL EventRecord per decision with predictions, uncertainty, behaviour logp, version
    rows = st.archive.rows
    assert len(rows) == 60 == rep.compute.decisions
    assert all(r.provenance == Provenance.REAL for r in rows)
    for r in rows:
        assert r.action is not None and r.result is not None and r.behavior_logp is not None
        assert {"ok", "error_code", "cost", "duration", "reward"} <= set(r.predicted) == set(r.uncertainty)
        assert all(v >= 0 for v in r.uncertainty.values())
        assert r.extra["outcomes"] and r.extra["goal"]["text"] and "family" not in r.extra["goal"]
        assert r.action.source.value == "cam_planner" and r.action.proposal["n"] == st.model.imag_candidates
    by_env = {}
    for r in rows:
        by_env.setdefault(r.env_id, []).append(r)
    for env_rows in by_env.values():
        vs = [r.model_version for r in sorted(env_rows, key=lambda r: r.step)]
        assert vs == sorted(vs) and vs[0] == 0 and vs[-1] == 5          # 6 updates; last 5 decisions after update 5
        assert [r.step for r in env_rows] == list(range(30))
        ts = [r.t_sim for r in env_rows]
        assert all(b > a for a, b in zip(ts, ts[1:]))                    # SMDP clock advanced every decision
    assert any(r.extra["lp_prev"] != 0.0 for r in rows)                   # hindsight learning progress
    assert any(r.extra["retrieved"] for r in rows[4:])                    # memory retrieved from the archive index
    # ---- traces: spec 18.1 fields
    tr = st.recorder.traces
    assert len(tr) == 60
    for t in tr:
        assert t["objective"] and t["context_refs"] and t["predicted"] and t["uncertainty"]
        assert len(t["candidates"]) == st.model.imag_candidates and len(t["imagined"]) == st.model.imag_candidates
        assert t["selected"]["tool"] and t["belief_before"]["ess"] > 0 and t["belief_after"]["ess"] > 0
        assert t["reasoning"]["iters"] >= 1.0 and t["outcome"]["values"] and t["result_ref"] == t["event_id"]
    assert sum(1 for t in tr if t["update_refs"]) >= 50                   # decisions consumed by updates
    assert any(len(t["update_refs"]) >= 2 for t in tr)                  # fresh PPO data, later replayed
    assert all(set(u["replay_decisions"]) <= {t["decision_id"] for t in tr} for u in rep.updates)
    assert any(t["retrieved"] for t in tr)
    # ---- updates: all objectives, gradients into the core from each, heads and core actually changed
    ups = rep.updates
    assert len(ups) == 6 and all(u["updated"] for u in ups) and st.agent.version == 6
    for u in ups:
        assert {"world", "adapt", "memory", "actor", "value", "reason_compute"} <= set(u["losses"])
        for obj in ("world", "adapt", "actor", "value"):
            assert u[f"trainer/grad_core/{obj}"] > 0, (obj, u)
        assert u["trainer/param_delta/core"] > 0 and u["replay_events"]
        for h in ("tokenizer", "reasoner", "beliefs", "outcome_head", "world_model", "action_generator",
                  "value_head", "option_policy", "exploration_head", "query_encoder", "entity_encoder"):
            assert u[f"trainer/gradnorm/{h}"] > 0, h
    assert abs(ups[0]["policy/ratio_mean"] - 1.0) < 1e-3 and ups[0]["policy/ratio_maxdev"] < 1e-3
    assert ups[0]["policy/max_version_lag"] == 0.0 and ups[1]["policy/max_version_lag"] <= 1.0
    assert any(not torch.equal(core0[n], p.detach()) for n, p in st.model.core.named_parameters())
    for h in HEADS:
        assert any(not torch.equal(heads0[h][n], p.detach()) for n, p in getattr(st.model, h).named_parameters()), h
    # ---- compute accounting
    c = rep.compute
    assert c.env_steps == 60 and c.updates == 6 and c.sim_days > 0 and c.wall_s > 0 and c.cpu_s > 0
    assert c.inference_tokens and c.archive_records == 60 and c.max_rss_bytes > 0
    assert len(rep.returns) == 2 and len(rep.goal_scores) == 2


def test_evaluator_only_state_never_reaches_learner_inputs():
    st = Stack()
    run_steps(st.agent, st.env, 8, evaluator=st.evaluator, update_every=4)
    blob = json.dumps([r.to_dict() for r in st.archive.rows]) + json.dumps(st.recorder.traces)
    assert "elasticity_secret" not in blob and "base_rate" not in blob and "toy_shop" not in blob
    # the reward is the evaluator's score increment, nothing else
    total = sum(r.reward for r in st.archive.rows)
    assert math.isclose(total, sum(st.evaluator.evaluate_goal(b).score for b in range(2)), abs_tol=1e-6)


@pytest.mark.parametrize("switch", ["use_memory", "use_imagination", "use_fast_adaptation", "use_reasoning",
                                    "use_learned_exploration", "continual_updates"])
def test_ablation_switches_change_the_path(switch):
    cfg = make_cfg(**{switch: False})
    comps = dbl.double_components(cfg)
    st = Stack(cfg=cfg, components=comps)
    p0 = params(st.model)
    seen_kinds = set()
    orig = st.model.core.forward

    def spy(Z, ctx):
        seen_kinds.update(ctx.kind[ctx.mask].tolist())
        return orig(Z, ctx)

    st.model.core.forward = spy
    rep = run_steps(st.agent, st.env, 8, evaluator=st.evaluator, update_every=4)
    calls, rows, tr = st.model.calls, st.archive.rows, st.recorder.traces
    losses = set().union(*[set(u.get("losses", {})) for u in rep.updates])
    if switch == "use_memory":
        assert st.model.retriever.n_retrieve == 0 and "retriever.retrieve" not in calls
        assert TokenKind.MEMORY not in seen_kinds and not st.model.retriever.ids
        assert all(not t["retrieved"] for t in tr) and "memory" not in losses
    elif switch == "use_imagination":
        assert "planner" not in calls and "imagination.beliefs.update" not in calls
        assert all(r.action.source.value == "cam_decoder" and r.action.proposal is None for r in rows)
        assert all(not t["imagined"] for t in tr)
    elif switch == "use_fast_adaptation":
        assert st.agent.fast is None and "beliefs.update" not in calls and "bayes.update" not in calls
        assert TokenKind.BELIEF not in seen_kinds and "adapt" not in losses
        assert all(r.extra["lp_prev"] == 0.0 for r in rows) and all(not t["belief_after"] for t in tr)
    elif switch == "use_reasoning":
        assert "reasoner" not in calls and "reason_compute" not in losses
        assert all(t["reasoning"]["iters"] == 0.0 for t in tr)
    elif switch == "use_learned_exploration":
        assert "exploration_head" not in calls
        assert all(r.extra["lp_prev"] == 0.0 for r in rows)
    elif switch == "continual_updates":
        assert all(not u["updated"] for u in rep.updates) and st.agent.version == 0
        assert all(torch.equal(p0[n], p) for n, p in st.model.named_parameters())
        assert all(r.model_version == 0 for r in rows)
    if switch != "continual_updates":
        assert all(u["updated"] for u in rep.updates) and st.agent.version == 2
    assert len(rows) == 16                                     # experience is archived under every condition


def test_default_config_uses_every_path_the_ablations_remove():
    st = Stack()
    seen = set()
    orig = st.model.core.forward

    def spy(Z, ctx):
        seen.update(ctx.kind[ctx.mask].tolist())
        return orig(Z, ctx)

    st.model.core.forward = spy
    run_steps(st.agent, st.env, 4, evaluator=st.evaluator)
    assert {TokenKind.MEMORY, TokenKind.BELIEF, TokenKind.ACTION, TokenKind.OPTION, TokenKind.GOAL} <= seen
    assert all(r.action.source.value == "cam_planner" for r in st.archive.rows)


def test_reset_rows_only_and_checkpoint_attribute_contract():
    st = Stack()
    run_steps(st.agent, st.env, 3, evaluator=st.evaluator)
    ag = st.agent
    for name in ("model", "config", "optimizers", "fast", "archive", "index", "replay", "rng", "version"):
        assert hasattr(ag, name), name
    assert set(ag.rng) == {"python", "torch"} and "all" in ag.optimizers
    Z0, logw0, ep0 = ag.state.Z.clone(), ag.fast.logw.clone(), list(ag.episode_ids)
    ag.reset(["shop1"])
    init = st.model.core.init_state(1)
    assert torch.allclose(ag.state.Z[1], init[0]) and torch.equal(ag.state.Z[0], Z0[0])
    assert torch.equal(ag.fast.logw[0], logw0[0]) and ag.episode_ids[0] == ep0[0] and ag.episode_ids[1] != ep0[1]
    assert ag.state.last_actions[1] is None and ag.state.last_actions[0] is not None
    # reset_fast_state: uniform weights, empty channel tables
    ag.reset_fast_state()
    K = st.cfg.n_particles
    assert torch.allclose(ag.fast.logw, torch.full_like(ag.fast.logw, -math.log(K)))
    assert all(not c for c in ag.fast.channels)
    # reset_optimizer clears moments
    ag.update()
    opt = ag.optimizers["all"]
    assert len(opt.state) > 0
    ag.reset_optimizer()
    assert len(opt.state) == 0
    # reset_experience_access: no archive/replay/index, retriever bank emptied -> retrieval returns nothing
    ag.reset_experience_access()
    assert ag.archive is None and ag.replay is None and ag.index is None and not st.model.retriever.ids
    ob = st.env.observe()
    ag.act(ob, st.env.available_tools())
    assert all(not r for r in ag._pending[1].retrieved)


def test_archive_rejects_imagined_records():
    arch = dbl.DoubleArchive()
    rec = EventRecord("e1", "env", "a", "ep", 0, 0, 0.0, [], provenance=Provenance.IMAGINED)
    with pytest.raises(ValueError):
        arch.append([rec])
    assert arch.count() == 0


def test_tokenise_sequences_from_archived_records():
    st = Stack()
    run_steps(st.agent, st.env, 5, evaluator=st.evaluator)
    eps = {}
    for r in st.archive.rows:
        eps.setdefault(r.episode_id, []).append(r)
    seqs = [sorted(v, key=lambda r: r.step) for v in eps.values()]
    seqs[1] = seqs[1][:3]                                                         # ragged -> padded
    seq, tables = st.agent.tokenise_sequences(seqs)
    assert isinstance(seq, SequenceBatch) and len(seq.steps) == 5
    D = st.cfg.d_model
    for t, step in enumerate(seq.steps):
        assert step.obs.x.shape[0] == 2 and step.obs.x.shape[2] == D and step.act.x.shape[0] == 2
        assert step.outcomes.y.shape == step.outcomes.lik.shape and step.outcomes.q.x.shape[2] == D
        assert step.dt.shape == (2,) and step.reward.shape == (2,) and step.goal.x.shape[0] == 2
        assert step.event_ids[0] == seqs[0][t].event_id
        assert step.outcomes.y_mask[0].any()
        assert (step.outcomes.channel[0][step.outcomes.y_mask[0]] >= 0).all()
        if t >= 3:
            assert bool(step.done[1]) and not step.outcomes.y_mask[1].any() and not step.obs.mask[1].any()
    for t in range(5):
        assert abs(float(seq.steps[t].dt[0]) - seqs[0][t].extra["dt"]) < 1e-6
        k = seq.steps[t].outcomes.keys[0].index("reward")
        assert abs(float(seq.steps[t].outcomes.y[0, k]) - seqs[0][t].reward) < 1e-5
    assert len(tables[0]) > 5
    # the world model consumes it (gradients into the shared core)
    out = st.model.world_model.loss(seq)
    out["world"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in st.model.core.parameters())


def _fs(rewards, values, dts, dones, ep="ep"):
    n = len(rewards)
    t = lambda v: torch.tensor(v, dtype=torch.float)
    return FreshStep([0] * n, [ep] * n, None, [None] * n, t([0.0] * n), t(values), t(rewards), t(dts),
                     torch.tensor(dones), 0, ["d"] * n, t([0.0] * n))


def test_gae_semi_markov_matches_manual_computation():
    k, lam = 0.5, 0.9
    steps = [_fs([1.0], [0.5], [1.0], [False]), _fs([0.0], [0.2], [2.0], [False]), _fs([2.0], [0.1], [0.0], [True]),
             _fs([5.0], [0.3], [1.0], [False], ep="ep2"), _fs([1.0], [0.4], [1.0], [False], ep="ep2")]
    out = gae_semi_markov([(s, 0) for s in steps], k, lam, lp_coef=0.0)
    g = [math.exp(-k * 1.0), math.exp(-k * 2.0), 1.0]
    d2 = 2.0 - 0.1
    d1 = 0.0 + g[1] * 0.1 - 0.2
    d0 = 1.0 + g[0] * 0.2 - 0.5
    a2 = d2
    a1 = d1 + g[1] * lam * a2
    a0 = d0 + g[0] * lam * a1
    close = lambda a, b: math.isclose(a, b, rel_tol=1e-6)
    assert close(out[(id(steps[2]), 0)][0], a2) and close(out[(id(steps[1]), 0)][0], a1)
    assert close(out[(id(steps[0]), 0)][0], a0)
    assert close(out[(id(steps[0]), 0)][1], a0 + 0.5)
    # second episode: its last undone entry only bootstraps; the first gets r + gamma V(next) - V
    assert (id(steps[4]), 0) not in out
    assert close(out[(id(steps[3]), 0)][0], 5.0 + math.exp(-k) * 0.4 - 0.3)
    # learning progress enters as an intrinsic reward
    steps[2].lp[0] = 1.0
    out2 = gae_semi_markov([(s, 0) for s in steps[:3]], k, lam, lp_coef=0.5)
    assert close(out2[(id(steps[2]), 0)][0], a2 + 0.5)


def test_runtime_delivers_each_event_once_and_advances_the_clock():
    st = Stack()
    seen = []
    orig = st.agent.observe

    def spy(results, rewards, next_obs):
        for b, obs in enumerate(next_obs.observations):
            seen.extend((b, o.seq) for o in obs if o.seq is not None)
        return orig(results, rewards, next_obs)

    st.agent.observe = spy
    rep = run_steps(st.agent, st.env, 6, evaluator=st.evaluator)
    assert len(seen) == len(set(seen)) and seen                    # advance_time + collect_events deduplicated
    think = st.cfg.think_s / 86400
    for b in range(2):
        rows = sorted((r for r in st.archive.rows if r.env_id == f"shop{b}"), key=lambda r: r.step)
        for r in rows:
            assert math.isclose(r.extra["dt"], r.result.duration + think, rel_tol=1e-6, abs_tol=1e-9)
    assert math.isclose(rep.compute.sim_days, sum(r.extra["dt"] for r in st.archive.rows), rel_tol=1e-6)


def test_episode_ends_skip_done_envs_and_restart():
    st = Stack()
    st.env.horizon = 0.25                                       # days: episodes end after a few decisions
    rep = run_steps(st.agent, st.env, 24, evaluator=st.evaluator, update_every=6)
    eps = {}
    for r in st.archive.rows:
        eps.setdefault((r.env_id, r.episode_id), []).append(r)
    assert len(eps) > 2
    for v in eps.values():
        v = sorted(v, key=lambda r: r.step)
        assert [r.step for r in v] == list(range(len(v)))       # step counter restarts per episode
        assert not any(r.done for r in v[:-1])
        assert all(r.t_sim < st.env.horizon for r in v)
    assert sum(1 for v in eps.values() if max(v, key=lambda r: r.step).done) >= 2
    skipped = [(i, b) for i, s in enumerate(rep.steps) for b, tool in enumerate(s["tools"]) if tool is None]
    assert skipped                                              # a finished env waits (None action) for the others
    n_real = sum(1 for s in rep.steps for tool in s["tools"] if tool is not None)
    assert n_real == len(st.archive.rows) == rep.compute.decisions
    assert all(u["updated"] for u in rep.updates)


def test_production_code_never_imports_doubles():
    for fn in ("model.py", "agent.py", "runtime.py", "outcome_spec.py"):
        tree = ast.parse(open(os.path.join(PKG, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "")
                assert "testing" not in mod and "integration_doubles" not in mod, (fn, mod)
                assert not any(a.name in ("testing", "integration_doubles") for a in node.names), fn
            elif isinstance(node, ast.Import):
                assert not any("testing" in a.name or "integration_doubles" in a.name for a in node.names), fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_bf16_run_reports_gpu_accounting():
    st = Stack(device="cuda")
    assert st.model.device.type == "cuda"
    rep = run_steps(st.agent, st.env, 6, evaluator=st.evaluator, update_every=3)
    assert all(u["updated"] for u in rep.updates)
    assert rep.compute.gpu_s > 0 and rep.compute.peak_gpu_mem_bytes > 0
    assert st.agent.fast.particles.device.type == "cuda"
    assert all(math.isfinite(v) for r in st.archive.rows for v in r.predicted.values())


@pytest.mark.slow
def test_dev_scale_30_step_run():
    """dev scale (8 x 384, 32 state tokens) with doubles: throughput and update numbers for the report."""
    from superlearner.cam_unified.config import make_config
    cfg = make_config("dev", imag_branches=4, imag_samples=2, imag_horizons=(4,), mem_top_k=8, n_options=8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = Stack(cfg=cfg, device=dev)
    rep = run_steps(st.agent, st.env, 30, evaluator=st.evaluator, update_every=5)
    assert len(st.archive.rows) == 60 and all(u["updated"] for u in rep.updates)
    print("dev-scale compute:", json.dumps(rep.compute.to_dict()))
    print("dev-scale params:", json.dumps(st.model.count_parameters()["cam_total"]))
