"""CamUnifiedAgent + run_steps: 30-step end-to-end run, ablation switches, archive/trace contents, update path,
checkpoint attribute contract, lossless archiving, channel budget, behaviour provenance, replay-context consistency."""
import ast
import copy
import json
import math
import os

import pytest
import torch

from conftest import Stack, make_cfg
from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.agent import FreshStep, gae_semi_markov
from superlearner.cam_unified.interfaces import (Action, EventRecord, ObservationBatch, Observation, ObsKind,
                                                 Provenance, SequenceBatch, TokenKind)
from superlearner.cam_unified.runtime import run_steps

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "superlearner", "cam_unified"))
HEADS = ("tokenizer", "entity_encoder", "reasoner", "beliefs", "bayes", "outcome_head", "world_model", "retriever",
         "action_generator", "value_head", "option_policy", "exploration_head", "query_encoder")
# parameters that must each change (the ones the first verifier found frozen, plus the Kalman prior / noise)
MUST_CHANGE = ("bayes.w.weight", "bayes.log_prior_var", "bayes.log_noise", "beliefs.m0", "entity_encoder.rel.0.weight",
               "retriever.mem.weight")


def params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def actions_only(rows):
    return [r for r in rows if r.action is not None]


def test_30_step_run_exercises_every_call_path():
    st = Stack()
    st.env.horizon = 100.0                                  # one 30-decision episode per environment
    core0 = params(st.model.core)
    heads0 = {h: params(getattr(st.model, h)) for h in HEADS}
    all0 = params(st.model)
    rep = run_steps(st.agent, st.env, 30, evaluator=st.evaluator, update_every=5)
    calls = st.model.calls
    # every component of the step equation ran
    for k in ("core", "reasoner", "beliefs.tokens", "retriever.retrieve", "planner", "action_generator.sample",
              "world_model.prior", "imagination.beliefs.update", "imagination.bayes.update", "beliefs.update",
              "bayes.update", "exploration_head", "option_policy.step"):
        assert calls.get(k, 0) > 0, k
    assert st.model.tokenizer.tokens_encoded > 0 and st.model.retriever.n_retrieve == 30
    # ---- archive: one REAL EventRecord per decision with predictions, uncertainty, behaviour logp, version
    rows = actions_only(st.archive.rows)
    world = [r for r in st.archive.rows if r.action is None]
    assert len(rows) == 60 == rep.compute.decisions
    assert len(world) == 2 and all(w.extra["reason"] == "end_of_run" and w.observations for w in world)
    assert all(r.provenance == Provenance.REAL for r in st.archive.rows)
    for r in rows:
        assert r.action is not None and r.result is not None and r.behavior_logp is not None
        assert {"ok", "error_code", "cost", "duration", "reward"} <= set(r.predicted) == set(r.uncertainty)
        assert all(v >= 0 for v in r.uncertainty.values())
        assert r.extra["outcomes"] and r.extra["goal"]["text"] and "family" not in r.extra["goal"]
        assert r.action.source.value == "cam_planner" and r.action.proposal["n"] == st.model.imag_candidates
        assert r.extra["behavior"]["policy"] == "planner_argmax"
        assert r.extra["behavior"]["logp_kind"] == "generator_proposal"
        assert r.extra["budget"]["days_left"] <= r.extra["budget"]["days_total"] == st.env.horizon
    by_env = {}
    for r in rows:
        by_env.setdefault(r.env_id, []).append(r)
    for env_rows in by_env.values():
        vs = [r.model_version for r in sorted(env_rows, key=lambda r: r.step)]
        assert vs == sorted(vs) and vs[0] == 0 and vs[-1] == 5          # 6 updates; last 5 decisions after update 5
        assert [r.step for r in env_rows] == list(range(30))
        ts = [r.t_sim for r in env_rows]
        assert all(b > a for a, b in zip(ts, ts[1:]))                    # SMDP clock advanced every decision
        days = [r.extra["budget"]["days_left"] for r in sorted(env_rows, key=lambda r: r.step)]
        assert all(b < a for a, b in zip(days, days[1:]))                # the budget token tracks remaining time
    assert any(r.extra["lp_prev"] != 0.0 for r in rows)                   # hindsight learning progress
    assert any(r.extra["retrieved"] for r in rows[4:])                    # memory retrieved from the archive index
    # ---- traces: spec 18.1 fields; context refs resolve against the archive
    tr = st.recorder.traces
    assert len(tr) == 60
    ids = {r.event_id: r for r in st.archive.rows}
    for t in tr:
        assert t["objective"] and t["context_refs"] and t["predicted"] and t["uncertainty"]
        assert len(t["candidates"]) == st.model.imag_candidates and len(t["imagined"]) == st.model.imag_candidates
        assert t["selected"]["tool"] and t["belief_before"]["ess"] > 0 and t["belief_after"]["ess"] > 0
        assert t["reasoning"]["iters"] >= 1.0 and t["outcome"]["values"] and t["result_ref"] == t["event_id"]
        for ref in t["context_refs"]:
            if "#obs:" in ref:
                eid, i = ref.split("#obs:")
                assert eid == t["event_id"] and int(i) < len(ids[eid].observations)
            elif ref.startswith(("e_", "w_", "a_")):                    # event ids (= action ids of decisions)
                assert ref in ids, ref                                   # memory / previous-action event ids
            else:
                assert ref.startswith(("entity:", "relation:", "goal", "tool:", "option:", "budget")), ref
                if ref.startswith(("entity:", "relation:")):                 # persistent ids, never slots
                    assert not any(x.isdigit() for x in ref.split(":", 1)[1].split("->")), ref
        assert not any(r.startswith(("obs:", "mem:", "act:", "entity_slot:")) for r in t["context_refs"])
    assert any(any(r.startswith("entity:") for r in t["context_refs"]) for t in tr)
    assert any(any(r.startswith("relation:") for r in t["context_refs"]) for t in tr)
    assert sum(1 for t in tr if t["update_refs"]) >= 50                   # decisions consumed by updates
    assert any(len(t["update_refs"]) >= 2 for t in tr)                  # fresh data, later replayed
    assert all(set(u["replay_decisions"]) <= {t["decision_id"] for t in tr} for u in rep.updates)
    assert any(t["retrieved"] for t in tr)
    # ---- updates: all objectives, gradients into the core from each, heads and core actually changed
    ups = rep.updates
    assert len(ups) == 6 and all(u["updated"] for u in ups) and st.agent.version == 6
    for u in ups:
        assert {"world", "adapt", "memory", "actor", "value"} <= set(u["losses"])
        assert set(u["losses"]) <= set(st.cfg.loss_weights)            # only weighted objectives (D's trainer)
        assert u["policy/reason_compute"] > 0 and u["policy/ponder_mean"] >= 1.0
        for obj in ("world", "adapt", "actor", "value"):
            assert u[f"trainer/grad_core/{obj}"] > 0, (obj, u)
        assert u["trainer/param_delta/core"] > 0 and u["replay_events"]
        for h in ("tokenizer", "reasoner", "beliefs", "bayes", "outcome_head", "world_model", "action_generator",
                  "value_head", "option_policy", "exploration_head", "query_encoder", "entity_encoder", "retriever"):
            assert u[f"trainer/gradnorm/{h}"] > 0, h
        for p in ("bayes.w.weight", "bayes.log_prior_var", "bayes.log_noise"):
            assert u[f"trainer/grad/{p}"] > 0, p                         # the Kalman layer is trained (adapt loss)
        assert u["adapt/adapt_kalman_outcomes"] > 0 or u["adapt/adapt_k"] == 0
        assert u["policy/n_awr"] > 0 and u["policy/n_ppo"] == 0          # planner choices -> AWR, not PPO ratios
        assert u["policy/memory_rebuilt"]
    assert ups[0]["policy/logp_maxdev_lag0"] < 1e-3                      # exact behaviour-logp recomputation
    assert ups[0]["policy/max_version_lag"] == 0.0 and ups[1]["policy/max_version_lag"] <= 1.0
    assert any(not torch.equal(core0[n], p.detach()) for n, p in st.model.core.named_parameters())
    for h in HEADS:
        assert any(not torch.equal(heads0[h][n], p.detach()) for n, p in getattr(st.model, h).named_parameters()), h
    now = dict(st.model.named_parameters())
    for n in MUST_CHANGE:
        assert not torch.equal(all0[n], now[n].detach()), n
    # ---- compute accounting and budget drops
    c = rep.compute
    assert c.env_steps == 60 and c.updates == 6 and c.sim_days > 0 and c.wall_s > 0 and c.cpu_s > 0
    assert c.inference_tokens and c.archive_records == 62 and c.max_rss_bytes > 0
    assert c.gpu_span_s == 0.0 and c.gpu_busy_s is None                  # CPU run
    assert len(rep.returns) == 2 and len(rep.goal_scores) == 2
    assert {"ctx_tokens", "channels_evicted", "channels_unbound", "fresh_steps", "outcomes"} <= set(rep.dropped)
    assert rep.dropped["channels_unbound"] == 0 and rep.dropped["fresh_steps"] == 0


def test_evaluator_only_state_never_reaches_learner_inputs():
    st = Stack()
    seen_goals, seen_obs = [], []
    tok = st.model.tokenizer
    og, oo = tok.goal, tok.observations

    def spy_goal(goals):
        seen_goals.extend(goals)
        return og(goals)

    def spy_obs(obs_lists, regs, now):
        seen_obs.extend(o for ol in obs_lists for o in ol)
        return oo(obs_lists, regs, now)

    tok.goal, tok.observations = spy_goal, spy_obs
    run_steps(st.agent, st.env, 8, evaluator=st.evaluator, update_every=4)
    assert seen_goals and all(g.family == "" and g.meta == {} for g in seen_goals)   # bookkeeping fields stripped
    hidden = [v for h in st.env._hidden for p in h.values() for v in p.values()]
    for o in seen_obs:                                                   # no hidden mechanism value is an input
        assert o.value is None or all(abs(o.value - v) > 1e-9 for v in hidden)
        assert "elasticity" not in o.content and "base_rate" not in o.content
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
    calls, rows, tr = st.model.calls, actions_only(st.archive.rows), st.recorder.traces
    losses = set().union(*[set(u.get("losses", {})) for u in rep.updates])
    if switch == "use_memory":
        assert st.model.retriever.n_retrieve == 0 and "retriever.retrieve" not in calls
        assert TokenKind.MEMORY not in seen_kinds and not st.model.retriever.ids
        assert all(not t["retrieved"] for t in tr) and "memory" not in losses
    elif switch == "use_imagination":
        assert "planner" not in calls and "imagination.beliefs.update" not in calls
        assert all(r.action.source.value == "cam_decoder" and r.action.proposal is None for r in rows)
        assert all(r.extra["behavior"]["policy"] == "decoder_sample" for r in rows)
        assert all(not t["imagined"] for t in tr)
        assert all(u["policy/n_ppo"] > 0 and u["policy/n_awr"] == 0 for u in rep.updates)   # exact ratios -> PPO
        assert abs(rep.updates[0]["policy/ratio_mean"] - 1.0) < 1e-3
    elif switch == "use_fast_adaptation":
        assert st.agent.fast_state is None and "beliefs.update" not in calls and "bayes.update" not in calls
        assert TokenKind.BELIEF not in seen_kinds and "adapt" not in losses
        assert all(r.extra["lp_prev"] == 0.0 for r in rows) and all(not t["belief_after"] for t in tr)
    elif switch == "use_reasoning":
        assert "reasoner" not in calls and all("policy/reason_compute" not in u for u in rep.updates)
        assert all(t["reasoning"]["iters"] == 0.0 for t in tr)
    elif switch == "use_learned_exploration":
        assert "exploration_head" not in calls and TokenKind.BUDGET not in seen_kinds
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
    assert {TokenKind.MEMORY, TokenKind.BELIEF, TokenKind.ACTION, TokenKind.OPTION, TokenKind.GOAL, TokenKind.TIME,
            TokenKind.RELATION, TokenKind.ENTITY} <= seen
    assert all(r.action.source.value == "cam_planner" for r in actions_only(st.archive.rows))


def test_replay_context_matches_the_acting_context():
    """filter_sequence (world / adapt / memory objectives) feeds the core the same kinds of context as acting, and
    for a sequence starting at an episode start reproduces the acting Z_0 exactly."""
    st = Stack(cfg=make_cfg(use_imagination=False))
    acting_Z, seen_fit = {}, []
    od = st.model.decide

    def spy_decide(state, *a, **k):
        dec = od(state, *a, **k)
        acting_Z[len(acting_Z)] = dec.Z.clone()
        return dec

    st.model.decide = spy_decide
    run_steps(st.agent, st.env, 4, evaluator=st.evaluator)
    eps = {}
    for r in actions_only(st.archive.rows):
        eps.setdefault(r.env_id, []).append(r)
    seqs = [sorted(eps[f"shop{b}"], key=lambda r: r.step) for b in range(2)]
    seq, aux = st.agent.tokenise_sequences(seqs)
    orig = st.model.core.forward

    def spy(Z, ctx):
        seen_fit.append(set(ctx.kind[ctx.mask].tolist()))
        return orig(Z, ctx)

    st.model.core.forward = spy
    with torch.no_grad():
        filt = st.model.filter_sequence(seq, aux)
    st.model.core.forward = orig
    assert {TokenKind.BELIEF, TokenKind.MEMORY, TokenKind.OPTION, TokenKind.TIME, TokenKind.ACTION,
            TokenKind.GOAL} <= set().union(*seen_fit[1:])
    assert filt.n_belief_updates == 4 and filt.memory_rebuilt and len(filt.beliefs) == 4
    assert torch.allclose(filt.Zs[0], acting_Z[0], atol=1e-5)
    # the world model receives the same belief / memory context
    got = {}
    ol = st.model.world_model.loss

    def spy_loss(s, beliefs=None, memory=None):
        got["b"], got["m"] = beliefs, memory
        return ol(s, beliefs=beliefs, memory=memory)

    st.model.world_model.loss = spy_loss
    st.model.world_loss(seq, filt)
    assert got["b"] is filt.beliefs and got["m"] is filt.memory


def test_adapt_loss_trains_the_kalman_last_layer():
    st = Stack()
    run_steps(st.agent, st.env, 8, evaluator=st.evaluator)
    eps = {}
    for r in actions_only(st.archive.rows):
        eps.setdefault(r.env_id, []).append(r)
    seq, aux = st.agent.tokenise_sequences([sorted(v, key=lambda r: r.step) for v in eps.values()])
    st.model.train()
    filt = st.model.filter_sequence(seq, aux)
    loss, mets = st.model.adapt_loss(seq, filt, 4)
    assert mets["adapt_kalman_outcomes"] > 0 and mets["adapt_mixture_outcomes"] > 0 and torch.isfinite(loss)
    st.model.zero_grad()
    loss.backward()
    for n in ("bayes.w.weight", "bayes.log_prior_var", "bayes.log_noise", "beliefs.m0", "outcome_head.out.weight",
              "core.layer.linear1.weight"):
        g = dict(st.model.named_parameters())[n].grad
        assert g is not None and float(g.abs().sum()) > 0, n
    # k = 0: no support, no Kalman statistics -> every query scored by the particle mixture only
    _, m0 = st.model.adapt_loss(seq, st.model.filter_sequence(seq, aux), 0)
    assert m0["adapt_kalman_outcomes"] == 0


def test_adapt_objective_samples_long_sequences_and_logs_k():
    st = Stack(seq_len=4, adapt_seq_len=65)
    rep = run_steps(st.agent, st.env, 20, evaluator=st.evaluator, update_every=10)
    u = rep.updates[-1]
    assert u["adapt_seq_T"] > 4 and u["adapt_seq_T"] <= 20           # longer than the world-model windows
    assert sum(u["adapt_k_hist"].values()) == 2 and all(k < u["adapt_seq_T"] for k in u["adapt_k_hist"])
    T = u["adapt_seq_T"]
    allowed = [k for k in (0, 1, 2, 4, 8, 16, 32, 64) if k < T]
    assert max(allowed) >= 8


def test_more_outcome_keys_than_kalman_channels_is_counted_and_stable():
    cfg = make_cfg()
    comps = dbl.double_components(cfg)
    comps["beliefs"] = dbl.DoubleBeliefs(cfg, n_channels=8)
    st = Stack(cfg=cfg, components=comps)
    st.env = dbl.ToyShopEnv(B=2, seed=0, n_products=6)          # 13 numeric readings + 3 continuous core keys
    st.evaluator = dbl.ToyShopEvaluator(st.env)
    rep = run_steps(st.agent, st.env, 6, evaluator=st.evaluator)
    rows = actions_only(st.archive.rows)
    n_cont = [sum(1 for s in r.extra["outcomes"] if s["observed"] and s["lik"] in (0, 1)) for r in rows]
    assert all(n == 16 for n in n_cont)
    assert all(r.extra["channels"]["unbound"] == 8 for r in rows)            # 16 realised keys, 8 slots: counted
    assert all(r.extra["channels"]["evicted"] == 0 for r in rows)            # realised keys are never evicted
    assert rep.dropped["channels_unbound"] == 8 * 12 and rep.dropped["channels_evicted"] == 0
    fast = st.agent.fast_state
    for b in range(2):
        assert len(fast.channels[b]) == 8
        assert torch.all(fast.bayes_n[b] == 6)                               # posteriors kept, never reset
    tr = st.recorder.traces
    assert all(t["outcome"]["channels"]["unbound"] == 8 for t in tr)


def test_every_delivered_observation_is_archived():
    st = Stack()
    st.env.horizon = 0.25                                     # episodes end after a few decisions
    rep = run_steps(st.agent, st.env, 24, evaluator=st.evaluator, update_every=6)
    delivered = set()
    for s in rep.steps:
        for b, seqs in enumerate(s["delivered"]):
            delivered.update((f"shop{b}", q) for q in seqs)
        for b, seqs in enumerate(s.get("reset_delivered", [])):
            delivered.update((f"shop{b}", q) for q in seqs)
    archived = {(r.env_id, o.seq) for r in st.archive.rows for o in r.observations if o.seq is not None}
    assert delivered and delivered <= archived, sorted(delivered - archived)[:10]
    world = [r for r in st.archive.rows if r.action is None]
    reasons = {w.extra["reason"] for w in world}
    assert "episode_end" in reasons
    by_id = {r.event_id: r for r in st.archive.rows}
    for w in world:
        assert w.attributed_to in by_id and by_id[w.attributed_to].action is not None
        assert w.provenance == Provenance.REAL and w.observations
    # world-event records are archived but never used as training steps
    assert all(by_id[e].action is not None for u in rep.updates for e in u["replay_events"])
    # a short run: the last round's events reach the archive through flush()
    st2 = Stack()
    rep2 = run_steps(st2.agent, st2.env, 5, evaluator=st2.evaluator)
    last = {(f"shop{b}", q) for b, seqs in enumerate(rep2.steps[-1]["delivered"]) for q in seqs}
    arch2 = {(r.env_id, o.seq) for r in st2.archive.rows for o in r.observations if o.seq is not None}
    assert last and last <= arch2


def test_post_action_observations_not_handed_to_the_next_decision_are_archived():
    st = Stack()
    ob = st.env.reset()
    st.agent.reset(ob.env_ids)
    st.agent.set_goals(st.env.goals())
    acts = st.agent.act(ob, st.env.available_tools())
    res = st.env.execute(acts)
    adv = st.env.advance_time([r.duration + 0.01 for r in res])
    nxt = ObservationBatch(ob.env_ids, [list(r.observations) + list(o) for r, o in zip(res, adv.observations)],
                           adv.t, adv.done)
    st.agent.observe(res, [0.0, 0.0], nxt)
    # a harness that hands the next decision only part of what it delivered: the rest is archived, not lost
    partial = ObservationBatch(ob.env_ids, [o[:1] for o in nxt.observations], adv.t, adv.done)
    st.agent.act(partial, st.env.available_tools())
    world = [r for r in st.archive.rows if r.action is None]
    assert len(world) == 2 and all(w.extra["reason"] == "not_in_next_decision" for w in world)
    for b, w in enumerate(world):
        assert len(w.observations) == len(nxt.observations[b]) - 1


def test_checkpoint_attribute_contract_of_workstream_E():
    """The duck-typed attributes E's CheckpointOps reads and assigns (fast_state, latent_state, registries,
    init_* hooks, counters, version, optimizers / make_optimizers, weights_exclude, rng)."""
    st = Stack()
    run_steps(st.agent, st.env, 3, evaluator=st.evaluator)
    ag = st.agent
    for name in ("model", "config", "optimizers", "fast_state", "latent_state", "registries", "archive", "index",
                 "replay", "rng", "version", "counters", "weights_exclude", "schedulers"):
        assert hasattr(ag, name), name
    assert callable(ag.init_fast_state) and callable(ag.init_latent_state) and callable(ag.make_optimizers)
    assert set(ag.rng) == {"python", "torch"} and "all" in ag.optimizers and ag.weights_exclude == ()
    # fast_state / latent_state are the live per-environment state (not copies)
    assert ag.fast_state is ag.state.fast and ag.latent_state is ag.state.Z
    saved_fast, saved_Z = ag.fast_state.clone(), ag.latent_state.clone()
    ag.fast_state = ag.init_fast_state()                     # what E's reset_fast_state does
    ag.latent_state = ag.init_latent_state()
    ag.registries = []
    K = st.cfg.n_particles
    assert torch.allclose(ag.state.fast.logw, torch.full_like(ag.state.fast.logw, -math.log(K)))
    assert all(not c for c in ag.state.fast.channels) and float(ag.state.fast.bayes_n.sum()) == 0
    assert torch.allclose(ag.state.Z, st.model.core.init_state(2))
    assert len(ag.registries) == 2 and all(len(r.slot) == 0 for r in ag.registries)
    ag.fast_state, ag.latent_state = saved_fast, saved_Z     # what E's load_parts(fast_state=True) does
    assert torch.equal(ag.state.fast.logw, saved_fast.logw) and torch.equal(ag.state.Z, saved_Z)
    with pytest.raises(ValueError):
        ag.latent_state = saved_Z[:1]
    # counters / version reset (load_weights_only) and the weights epoch
    c = ag.counters
    assert c["n_decisions"] == 6 and c["steps"] == [4, 4]          # 3 decisions + the end-of-run world event
    ep = st.model.weights_epoch
    ag.version, ag.counters = 0, {}
    assert ag.version == 0 and st.model.weights_epoch > ep and ag.n_decisions == 0 and ag.steps == [0, 0]
    ag.counters = c
    assert ag.n_decisions == 6 and ag.steps == [4, 4]
    # reset_optimizer clears moments; make_optimizers replaces them
    ag.update()
    opt = ag.optimizers["all"]
    assert len(opt.state) > 0
    ag.reset_optimizer()
    assert len(opt.state) == 0
    ag.optimizers = ag.make_optimizers()
    assert ag.optimizers["all"] is not opt
    # reset_experience_access: no archive/replay/index, retriever bank emptied -> retrieval returns nothing
    ag.reset_experience_access()
    assert ag.archive is None and ag.replay is None and ag.index is None and not st.model.retriever.ids
    ob = st.env.observe()
    ag.act(ob, st.env.available_tools())
    assert all(not r for r in ag._pending[1].retrieved)


def test_reset_rows_only():
    st = Stack()
    run_steps(st.agent, st.env, 3, evaluator=st.evaluator)
    ag = st.agent
    ob = st.env.observe()                                     # one more decision whose consequences stay pending
    acts = ag.act(ob, st.env.available_tools())
    res = st.env.execute(acts)
    adv = st.env.advance_time([r.duration + 0.01 for r in res])
    ag.observe(res, [0.0, 0.0], ObservationBatch(ob.env_ids, [list(r.observations) + list(o) for r, o in
                                                               zip(res, adv.observations)], adv.t, adv.done))
    Z0, logw0, ep0 = ag.state.Z.clone(), ag.fast_state.logw.clone(), list(ag.episode_ids)
    n_arch = st.archive.count()
    ag.reset(["shop1"])
    init = st.model.core.init_state(1)
    assert torch.allclose(ag.state.Z[1], init[0]) and torch.equal(ag.state.Z[0], Z0[0])
    assert torch.equal(ag.fast_state.logw[0], logw0[0]) and ag.episode_ids[0] == ep0[0] and ag.episode_ids[1] != ep0[1]
    assert ag.state.last_actions[1] is None and ag.state.last_actions[0] is not None
    assert float(ag.state.last_now[1]) == -1.0 and float(ag.state.last_now[0]) >= 0
    # the reset row's undelivered post-action observations were archived (reason "reset")
    assert st.archive.count() == n_arch + 1 and st.archive.rows[-1].extra["reason"] == "reset"
    assert st.archive.rows[-1].env_id == "shop1"


def test_checkpoint_ops_of_workstream_E_round_trip(tmp_path):
    ck = pytest.importorskip("superlearner.cam_unified.memory.checkpoint",
                             reason="workstream E's CheckpointOps not merged yet")
    st = Stack()
    run_steps(st.agent, st.env, 3, evaluator=st.evaluator, update_every=3)
    st.archive.pointer = lambda: None
    ops = ck.CheckpointOps(st.agent)
    fast, Z = st.agent.fast_state.clone(), st.agent.latent_state.clone()
    st.agent.archive = st.agent.replay = st.agent.index = None
    ops.save_full_lifetime(str(tmp_path / "life"))
    ops.reset_fast_state()
    assert not torch.equal(st.agent.fast_state.logw, fast.logw) or float(fast.bayes_n.sum()) > 0
    assert float(st.agent.fast_state.bayes_n.sum()) == 0 and all(not c for c in st.agent.fast_state.channels)
    ops.load_parts(str(tmp_path / "life"), fast_state=True, counters=True)
    assert torch.equal(st.agent.fast_state.bayes_mu, fast.bayes_mu) and torch.equal(st.agent.latent_state, Z)
    assert st.agent.fast_state.channels == fast.channels and st.agent.version == 1
    ops.save_weights(str(tmp_path / "w.safetensors"))
    ep = st.model.weights_epoch
    ops.load_weights_only(str(tmp_path / "w.safetensors"))
    assert st.agent.version == 0 and st.model.weights_epoch > ep and float(st.agent.fast_state.bayes_n.sum()) == 0


def test_planner_choices_never_reach_the_archive_as_imagined_data():
    st = Stack()
    trajs, chosen = [], []
    od = st.model.decide

    def spy_decide(state, *a, **k):
        dec = od(state, *a, **k)
        trajs.append(dec.trajectory)
        chosen.extend(x.action_id for x in dec.actions)
        return dec

    st.model.decide = spy_decide
    appended = []
    oa = st.archive.append

    def spy_append(records):
        appended.extend(records)
        return oa(records)

    st.archive.append = spy_append
    run_steps(st.agent, st.env, 6, evaluator=st.evaluator, update_every=3)
    assert trajs and all(t is not None and t.provenance == Provenance.IMAGINED for t in trajs)
    assert all(r.provenance == Provenance.REAL for r in appended)
    archived = {r.action.action_id for r in appended if r.action is not None}
    assert archived == set(chosen)
    imagined_ids = {a.action_id for t in trajs for s in t.steps for a in s.action}
    later = {a.action_id for t in trajs for s in t.steps[1:] for a in s.action}
    assert later and not (later & archived)                  # imagined follow-up actions are never archived
    assert (imagined_ids - set(chosen)).isdisjoint(archived)  # nor the candidates that were not chosen
    for r in appended:
        assert "trajectory" not in r.extra and r.extra.get("kind") != "imagined"


def test_tokenise_sequences_from_archived_records():
    st = Stack()
    run_steps(st.agent, st.env, 5, evaluator=st.evaluator)
    eps = {}
    for r in actions_only(st.archive.rows):
        eps.setdefault(r.episode_id, []).append(r)
    seqs = [sorted(v, key=lambda r: r.step) for v in eps.values()]
    seqs[1] = seqs[1][:3]                                                         # ragged -> padded
    seq, aux = st.agent.tokenise_sequences(seqs)
    assert isinstance(seq, SequenceBatch) and len(seq.steps) == 5
    D = st.cfg.d_model
    for t, step in enumerate(seq.steps):
        assert step.obs.x.shape[0] == 2 and step.obs.x.shape[2] == D and step.act.x.shape[0] == 2
        assert step.outcomes.y.shape == step.outcomes.lik.shape and step.outcomes.q.x.shape[2] == D
        assert step.dt.shape == (2,) and step.reward.shape == (2,) and step.goal.x.shape[0] == 2
        assert step.event_ids[0] == seqs[0][t].event_id
        assert step.outcomes.y_mask[0].any()
        cont = (step.outcomes.lik[0] <= 1) & step.outcomes.y_mask[0]
        assert (step.outcomes.channel[0][cont] >= 0).all() and (step.outcomes.channel[0][~cont] == -1).all()
        assert aux.retrieved[t][0] == seqs[0][t].extra["retrieved"]
        assert aux.option_prev[t][0] == (-1 if t == 0 else seqs[0][t - 1].extra["option"])
        if t >= 3:
            assert bool(step.done[1]) and not step.outcomes.y_mask[1].any() and not step.obs.mask[1].any()
    for t in range(5):
        assert abs(float(seq.steps[t].dt[0]) - seqs[0][t].extra["dt"]) < 1e-6
        k = seq.steps[t].outcomes.keys[0].index("reward")
        assert abs(float(seq.steps[t].outcomes.y[0, k]) - seqs[0][t].reward) < 1e-5
    assert len(aux.tables[0]) > 5
    # the world model consumes it (gradients into the shared core)
    out = st.model.world_model.loss(seq)
    out["world"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in st.model.core.parameters())


def _fs(rewards, values, dts, dones, ep="ep"):
    n = len(rewards)
    t = lambda v: torch.tensor(v, dtype=torch.float)
    return FreshStep([0] * n, [ep] * n, None, [None] * n, t([0.0] * n), t(values), t(rewards), t(dts),
                     torch.tensor(dones), 0, ["d"] * n, t([0.0] * n), torch.zeros(n, dtype=torch.bool))


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
        rows = sorted((r for r in actions_only(st.archive.rows) if r.env_id == f"shop{b}"), key=lambda r: r.step)
        for r in rows:
            assert math.isclose(r.extra["dt"], r.result.duration + think, rel_tol=1e-6, abs_tol=1e-9)
            if r.action.tool == "wait":                            # expected dt from the wait arguments
                assert math.isclose(r.extra["dt_pred"], r.extra["dt"], rel_tol=1e-5, abs_tol=1e-7)
    assert math.isclose(rep.compute.sim_days, sum(r.extra["dt"] for r in actions_only(st.archive.rows)),
                        rel_tol=1e-6)


class SeqZeroEnv(dbl.ToyShopEnv):
    """ToyShop without a cursors() method whose event numbering starts at 0 (the first event has seq 0)."""
    cursors = None

    def _emit(self, b, o):
        o.seq = self.seq[b]
        self.seq[b] += 1
        self._events[b].append(o)
        return o

    def collect_events(self, since):
        out, cur = [], []
        for b in range(self.B):
            out.append([o for o in self._events[b] if o.seq > int(since[b])])
            cur.append(self.seq[b] - 1)
        return out, cur


def test_runtime_cursor_sentinel_and_distinct_episode_seeds():
    env = SeqZeroEnv(B=2, seed=0)
    env.horizon = 0.2
    st = Stack()
    st.env, st.evaluator = env, dbl.ToyShopEvaluator(env)
    reset_obs = env.reset()
    assert min(o.seq for o in reset_obs.observations[0] if o.seq is not None) == 0
    rep = run_steps(st.agent, env, 16, evaluator=st.evaluator)
    archived = {(r.env_id, o.seq) for r in st.archive.rows for o in r.observations if o.seq is not None}
    assert ("shop0", 0) in archived and ("shop1", 0) in archived      # the seq-0 event was not lost
    seeds = [tuple(s) if s is not None else None for s in rep.episode_seeds]
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds)          # every episode is a different world
    assert rep.compute.episodes == len(seeds)


def test_episode_ends_skip_done_envs_and_restart():
    st = Stack()
    st.env.horizon = 0.25                                       # days: episodes end after a few decisions
    rep = run_steps(st.agent, st.env, 24, evaluator=st.evaluator, update_every=6)
    eps = {}
    for r in actions_only(st.archive.rows):
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
    assert n_real == len(actions_only(st.archive.rows)) == rep.compute.decisions
    assert all(u["updated"] for u in rep.updates)


def test_budget_drops_are_counted_and_reported():
    cfg = make_cfg(max_ctx_tokens=12)
    st = Stack(cfg=cfg, components=dbl.double_components(cfg))
    st.model.max_numeric_outcomes = 2
    for tr in st.agent.state.trackers if st.agent.state is not None else []:
        tr.max_numeric = 2
    rep = run_steps(st.agent, st.env, 4, evaluator=st.evaluator)
    assert rep.dropped["ctx_tokens"] and sum(rep.dropped["ctx_tokens"].values()) > 0
    assert rep.dropped["outcomes"]["numeric_targets"] > 0
    t = st.recorder.traces[-1]
    assert sum(t["outcome"]["dropped"]["ctx_tokens"].values()) > 0
    assert t["outcome"]["dropped"]["outcome_targets"]["numeric_targets"] > 0
    assert all(st.model.calls[f"ctx_dropped/{k}"] == v for k, v in rep.dropped["ctx_tokens"].items())


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
    rep = run_steps(st.agent, st.env, 6, evaluator=st.evaluator, update_every=3, profile_gpu=True)
    assert all(u["updated"] for u in rep.updates)
    c = rep.compute
    assert c.gpu_span_s > 0 and c.peak_gpu_mem_bytes > 0 and c.gpu_kernels > 0
    assert 0 < c.gpu_busy_s <= c.gpu_span_s                     # kernel time is a part of the CUDA-event span
    assert st.agent.fast_state.particles.device.type == "cuda"
    assert all(math.isfinite(v) for r in actions_only(st.archive.rows) for v in r.predicted.values())


@pytest.mark.slow
def test_dev_scale_30_step_run():
    """dev scale (8 x 384, 32 state tokens) with doubles: throughput and update numbers for the report."""
    from superlearner.cam_unified.config import make_config
    cfg = make_config("dev", imag_branches=4, imag_samples=2, imag_horizons=(4,), mem_top_k=8, n_options=8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = Stack(cfg=cfg, device=dev)
    rep = run_steps(st.agent, st.env, 30, evaluator=st.evaluator, update_every=5)   # unprofiled: throughput
    assert len(actions_only(st.archive.rows)) == 60 and all(u["updated"] for u in rep.updates)
    print("dev-scale compute:", json.dumps(rep.compute.to_dict()))
    print("dev-scale params:", json.dumps(st.model.count_parameters()["cam_total"]))


class RecordingConsolidator:
    """Shape of E's memory/consolidate.py:Consolidator as the agent uses it: step() reads the archive itself;
    link_weights(seg_ids, update_id, model_version) records parametric-knowledge provenance."""

    def __init__(self, archive):
        self.archive, self.cursor, self.steps, self.links = archive, 0, 0, []

    def step(self):
        n = self.archive.count() - self.cursor
        self.cursor = self.archive.count()
        self.steps += 1
        return {"events": n}

    def link_weights(self, seg_ids, update_id, model_version):
        self.links.append((list(seg_ids), update_id, model_version))


class SegmentIndex:
    """E-shaped index: add(version, seg_ids, keys, values) must never be called with records."""

    def add(self, version, seg_ids, keys, values):
        raise AssertionError("the agent must not feed records to a segment index")


def test_consolidator_and_segment_index_wiring():
    st = Stack(index=SegmentIndex())
    st.agent.consolidator = RecordingConsolidator(st.archive)
    rep = run_steps(st.agent, st.env, 6, evaluator=st.evaluator, update_every=3)
    con = st.agent.consolidator
    assert con.steps == 2 and rep.updates[0]["consolidated"] == {"events": 6}
    # the double retriever's bank is filled only through a records-level index; with the segment index it stays
    # empty, so nothing is retrieved and nothing is linked
    assert all(not u["memory_linked"] for u in rep.updates) and not con.links
    # with retrieved memory, the consumed decisions' memory ids are linked to the update
    st2 = Stack()
    st2.agent.consolidator = RecordingConsolidator(st2.archive)
    rep2 = run_steps(st2.agent, st2.env, 6, evaluator=st2.evaluator, update_every=3)
    linked = [l for l in st2.agent.consolidator.links]
    assert linked and linked[-1][1] == rep2.updates[-1]["update_id"] and linked[-1][2] == st2.agent.version
    assert set(linked[-1][0]) <= {r.event_id for r in st2.archive.rows}
