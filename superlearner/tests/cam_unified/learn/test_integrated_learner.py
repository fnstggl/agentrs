"""W7 (plan 2026-11-26 §3): IntegratedLearner, named-group gradient probe, loss balancing, role optimisers and
per-group clipping, PPO / V-trace routing by policy lag, reward conditions, observation-only integrity.

Unit tests on CPU with the master's reference doubles (testing.py / integration_doubles.py); they are not evidence
of the real stack (that is learn/integrated_e2e.py on dom_dev, docs/evidence/dom/W7)."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.agent import CamUnifiedAgent
from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.envs.labels import LabelRecord, LabelStore
from superlearner.cam_unified.learn import integrated as I
from superlearner.cam_unified.learn.balance import LossBalancer
from superlearner.cam_unified.learn.grad_probe import (DOM08_REQUIRED, classify, named_groups,
                                                       objective_gradients)
from superlearner.cam_unified.learn.ppo import ActorCriticObjective, StalePolicyError
from superlearner.cam_unified.learn.trainer import CombinedTrainer
from superlearner.cam_unified.learn.trajectory import Trajectory
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.runtime import run_steps
from superlearner.cam_unified.testing import tiny_config

torch.set_num_threads(4)


# ============================================================================ helpers

def _cfg(**over):
    base = dict(use_imagination=False, use_memory=False, reward_mode="observation_only", reward_in_beliefs=False,
                ppo_epochs=2, target_kl=0.5, clip_mode="per_group", balance_every=25, world_core_share_max=0.45,
                lr_core=1e-3, lr_heads=1e-3, lr_warmup_updates=0)
    base.update(over)
    return tiny_config(**base)


class _Stack:
    def __init__(self, cfg=None, B=2, seed=0, horizon=100.0):
        torch.manual_seed(seed)
        self.cfg = cfg or _cfg()
        self.model = build_model(self.cfg, components=dbl.double_components(self.cfg))
        self.archive = dbl.DoubleArchive()
        self.recorder = dbl.ListTraceRecorder()
        self.replay = dbl.DoubleReplay(self.archive, seed=seed)
        self.agent = CamUnifiedAgent(self.model, archive=self.archive, recorder=self.recorder, replay=self.replay,
                                     trainer=dbl.DoubleTrainer(self.model, self.cfg), seq_len=4, n_seqs=2,
                                     adapt_seq_len=6, adapt_n_seqs=1, seed=seed)
        self.env = dbl.ToyShopEnv(B=B, seed=seed)
        self.env.horizon = horizon
        self.evaluator = dbl.ToyShopEvaluator(self.env)


def _run(st, learner, rounds, every=4, evaluator=None):
    infos = []

    def on_step(i, row):
        if (i + 1) % every == 0:
            infos.append(learner.step())
    run_steps(st.agent, st.env, rounds, evaluator=evaluator, update_every=None, on_step=on_step)
    return infos


class ConstReward(nn.Module):
    """RewardModelP double: a constant reward, a psi parameter that must never receive an actor gradient."""

    def __init__(self, value=1.0):
        super().__init__()
        self.psi = nn.Parameter(torch.tensor(0.3))
        self.value = value
        self.version = 0
        self.calls = []

    def meta_parameters(self):
        return [self.psi]

    def reward(self, R, R_next=None, goal=None, dt=None, **kw):
        self.calls.append({"R": tuple(R.shape), "R_next": tuple(R_next.shape), "kw": sorted(kw),
                           "terminal": kw["signals"]["terminal"].clone(), "signals": sorted(kw["signals"])})
        return (torch.full((R.shape[0],), self.value) * self.psi).detach()

    def potential(self, R, G=None):
        return torch.zeros(R.shape[0])

    def losses(self, label_batch):
        self.last_batch = label_batch
        return {}


# ============================================================================ trainer: named groups, roles, clipping

class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.core = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)
        self.psi = nn.Linear(4, 1)


def test_named_groups_learning_rates_roles_and_warmup():
    m = _Toy()
    cfg = make_config("dev", lr=1e-2, lr_core=3e-5, lr_heads=1e-4, lr_psi=1e-3, lr_warmup_updates=4)
    tr = CombinedTrainer(cfg, groups={"core.block0": m.core, "policy": m.head, "psi": m.psi}, amp=False)
    assert set(tr.opts) == {"slow", "meta"} and tr.roles["meta"] == ["psi"]
    lr = {g["name"]: g["base_lr"] for g in tr.opts["slow"].param_groups}
    assert lr["core.block0/decay"] == 3e-5 and lr["policy/decay"] == 1e-4
    assert tr.opts["meta"].param_groups[0]["base_lr"] == 1e-3
    (m.head(m.core(torch.randn(3, 4)))).sum().backward()
    out = tr.apply("slow")
    assert out["lr"]["core.block0/decay"] == pytest.approx(3e-5 / 4)       # warmup on the core only
    assert out["lr"]["policy/decay"] == pytest.approx(1e-4)
    with pytest.raises(ValueError):
        CombinedTrainer(cfg, core=m.core, groups={"a": m.head})
    with pytest.raises(ValueError):
        CombinedTrainer(cfg, groups={"a": m.head, "b": m.head})


def test_slow_step_refuses_gradients_on_psi_and_meta_role_steps_alone():
    m = _Toy()
    tr = CombinedTrainer(make_config("dev"), groups={"core": m.core, "heads": m.head, "psi": m.psi}, amp=False)
    m.psi(torch.randn(2, 4)).sum().backward()
    with pytest.raises(RuntimeError, match="other-role"):
        tr.apply("slow")
    w_core = m.core.weight.detach().clone()
    w_psi = m.psi.weight.detach().clone()
    tr.apply("meta")
    assert torch.equal(w_core, m.core.weight) and not torch.equal(w_psi, m.psi.weight)


def test_per_group_clipping_uses_the_running_median_of_each_group():
    m = _Toy()
    cfg = make_config("dev", clip_mode="per_group", clip_median_mult=5.0)
    tr = CombinedTrainer(cfg, groups={"core": m.core, "heads": m.head}, amp=False)
    x = torch.randn(8, 4)
    for _ in range(4):                                     # build a history of ordinary norms
        m.head(m.core(x)).sum().backward()
        out = tr.apply("slow")
        assert not out["clipped"]
    med = sorted(tr.clip_hist["core"])[len(tr.clip_hist["core"]) // 2]
    (1e3 * m.head(m.core(x)).sum()).backward()
    g_before = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.core.parameters()))
    out = tr.apply("slow")
    assert out["clip"]["core"]["clipped"] and out["clip"]["core"]["threshold"] == pytest.approx(5.0 * med)
    assert out["clip"]["core"]["norm"] == pytest.approx(g_before, rel=1e-5)
    assert 0 < tr.clip_rate() < 0.5
    s = tr.state_dict()
    tr2 = CombinedTrainer(cfg, groups={"core": _Toy().core, "heads": _Toy().head}, amp=False)
    tr2.load_state_dict(s)
    assert list(tr2.clip_hist["core"]) == list(tr.clip_hist["core"]) and tr2.clip_counts == tr.clip_counts


# ============================================================================ grad probe

def test_classify_assigns_dom08_groups_by_name():
    L = 8
    assert classify("core.blocks.0.attn.q.weight", L) == "core.block0"
    assert classify("core.blocks.4.ffn.w1.weight", L) == "core.block_mid"
    assert classify("core.blocks.7.ffn.w1.weight", L) == "core.block_last"
    assert classify("core.blocks.2.ffn.w1.weight", L) == "core.blocks_other"
    assert classify("core.xattn.3.gate_in.weight", L) == "core.sem_xattn_gate"
    assert classify("core.xattn.3.attn.q.weight", L) == "core.sem_xattn"
    assert classify("core.sem_proj.proj.weight", L) == "sem_proj"
    assert classify("core.state_pos", L) == "core.embed"
    assert classify("tokenizer.num_mlp.0.weight", L) == "sem_proj"
    assert classify("action_generator.bridge.w.weight", L) == "dec_bridge"
    assert classify("action_generator.tool_q.weight", L) == "policy"
    assert classify("value_head.mlp.0.weight", L) == "v_int"
    assert classify("exploration_head.mlp.0.weight", L) == "exploration"
    assert classify("bayes.l1.weight", L) == "beliefs"
    assert classify("intent_head.q", L) == "intent"
    assert classify("x.GatedSemanticXAttn.gate_out.weight", L, hint="core.sem_xattn") == "core.sem_xattn_gate"


def test_objective_gradients_match_autograd_and_accumulate_the_sum():
    torch.manual_seed(0)
    core = nn.Linear(6, 6)
    head_a, head_b = nn.Linear(6, 1), nn.Linear(6, 1)
    groups = {"core.block0": list(core.named_parameters()), "policy": list(head_a.named_parameters()),
              "world_model": list(head_b.named_parameters()), "intent": []}
    x = torch.randn(5, 6)
    h = core(x)
    la, lb = head_a(h).pow(2).mean(), head_b(h).sum()
    ref = {}
    for n, l in (("actor", la), ("world", lb)):
        g = torch.autograd.grad(l, list(core.parameters()), retain_graph=True)
        ref[n] = float(torch.cat([t.flatten() for t in g]).norm())
    gsum = torch.autograd.grad(la + lb, list(core.parameters()), retain_graph=True)
    tab = objective_gradients({"actor": la, "world": lb}, groups)
    assert tab.core_norm["actor"] == pytest.approx(ref["actor"], rel=1e-5)
    assert tab.core_norm["world"] == pytest.approx(ref["world"], rel=1e-5)
    assert tab.norms["actor"]["world_model"] == 0.0 and tab.norms["world"]["policy"] == 0.0
    for p, g in zip(core.parameters(), gsum):
        assert torch.allclose(p.grad, g, atol=1e-6)       # .grad = gradient of the summed loss
    assert sum(tab.share().values()) == pytest.approx(1.0)
    d = tab.dom08(("core.block0", "policy", "intent"))
    assert d == {"core.block0": "nonzero", "policy": "nonzero", "intent": "absent"}
    assert "intent" in tab.to_dict()["absent"]
    h2 = core(x).sum()
    t2 = objective_gradients({"a": h2, "b": -h2}, {"core.x": list(core.named_parameters())}, accumulate=False)
    assert t2.cosine["a|b"] == pytest.approx(-1.0, abs=1e-5)


def test_named_groups_cover_every_trainable_parameter_once_with_psi_split():
    st = _Stack()
    rm = ConstReward()
    rm.pred = nn.Linear(3, 1)
    g = named_groups(st.model, rm)
    flat = [p for ps in g.values() for _, p in ps]
    assert len(flat) == len({id(p) for p in flat})
    want = {id(p) for p in st.model.parameters() if p.requires_grad} | {id(p) for p in rm.parameters()}
    assert {id(p) for p in flat} == want
    assert [n for n, _ in g["psi"]] == ["reward_model.psi"]
    assert {n for n, _ in g["predictors"]} == {"reward_model.pred.weight", "reward_model.pred.bias"}
    assert set(DOM08_REQUIRED) <= set(g)


# ============================================================================ balancer

def test_balancer_calibration_meets_targets_and_refinement_is_clipped():
    bal = LossBalancer({"world": ("max", 0.45), "actor": ("min", 0.10)})
    norms = {"world": 96.0, "adapt": 3.0, "actor": 0.25, "value": 0.1}
    e = bal.update(1, {"core_norm": norms})
    assert e["mode"] == "calibrate"
    after = {k: v * bal.factor(k) for k, v in norms.items()}
    tot = sum(after.values())
    assert after["world"] / tot <= 0.45 + 1e-9 and after["actor"] / tot >= 0.10 - 1e-9
    f0 = dict(bal.factors)
    e2 = bal.update(2, {"core_norm": {"world": 0.9 * 1e3, "adapt": 1.0, "actor": 1e-3, "value": 0.0}},
                    calibrate=False)
    assert e2["mode"] == "clipped_sqrt"
    assert bal.factors["world"] >= 0.5 * f0["world"] - 1e-12 and bal.factors["actor"] <= 2.0 * f0["actor"] + 1e-12
    assert "share_ema" in e2                                       # refinement acts on the smoothed shares
    fresh = LossBalancer({"world": ("max", 0.45), "actor": ("min", 0.10)})
    assert fresh.update(1, {"core_norm": {"world": 1.0, "actor": 1.0, "adapt": 1.0}}, calibrate=False) == {}
    s = bal.state_dict()
    b2 = LossBalancer()
    b2.load_state_dict(s)
    assert b2.factors == bal.factors and b2.calibrated


def test_balancer_ranges_bound_a_dominating_decision_objective_too():
    bal = LossBalancer()                       # world (0.2, 0.45), actor (0.1, 0.4), pred (0.1, 0.4)
    e = bal.update(1, {"core_norm": {"world": 1.0, "adapt": 1.0, "actor": 98.0, "value": 0.0}})
    assert e["mode"] == "calibrate" and bal.factor("actor") < 1.0 and bal.factor("world") > 1.0
    after = {"world": 1.0 * bal.factor("world"), "adapt": 1.0, "actor": 98.0 * bal.factor("actor")}
    tot = sum(after.values())
    assert 0.2 - 1e-9 <= after["world"] / tot <= 0.45 + 1e-9 and 0.1 - 1e-9 <= after["actor"] / tot <= 0.4 + 1e-9
    assert bal.violated({"world": 0.9, "actor": 0.05}) == ["world", "actor"]
    assert bal.violated({"world": 0.6, "actor": 0.07}, slack=1.5) == []      # within 1.5 x hi and lo / 1.5
    with pytest.raises(ValueError):
        LossBalancer({"a": (0.6, 0.9), "b": (0.5, 0.9)})      # infeasible lower bounds


# ============================================================================ PPO objective extensions

def _traj(B=2, T=3, lag=0):
    g = torch.Generator().manual_seed(0)
    return Trajectory(reward=torch.randn(B, T, generator=g), dt=torch.ones(B, T), done=torch.zeros(B, T, dtype=torch.bool),
                      behavior_logp=torch.full((B, T), -1.0), model_version=torch.full((B, T), 5 - lag),
                      value=torch.zeros(B, T + 1), reward_int=torch.ones(B, T))


def test_advantage_fn_reward_key_and_tool_entropy_bonus():
    tr = _traj()
    seen = {}

    def adv_fn(traj, r, v, gam, lam):
        seen["r"] = r.clone()
        return torch.full_like(r, 2.0), torch.full_like(r, 3.0)
    obj = ActorCriticObjective(advantage_fn=adv_fn, reward_key="reward_int", tool_entropy_coef=0.5,
                               normalize_advantages=False)
    prep = obj.prepare(tr, 5)
    assert torch.equal(seen["r"], tr.reward_int) and torch.equal(prep.ret, torch.full((2, 3), 3.0))
    lp = torch.full((2, 3), -1.0, requires_grad=True)
    ent = torch.zeros(2, 3)
    v = torch.zeros(2, 4, requires_grad=True)
    d0 = obj.loss(tr, prep, lp, ent, v)
    d1 = obj.loss(tr, prep, lp, ent, v, tool_entropy=torch.full((2, 3), 0.8))
    assert float((d0["actor"] - d1["actor"]).detach()) == pytest.approx(0.5 * 0.8) and d1["tool_entropy"] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        ActorCriticObjective(reward_key="bogus")


def test_lag_rows_go_to_vtrace_and_never_see_ppo_ratios():
    obj = ActorCriticObjective()
    tr = _traj(lag=1)
    prep = obj.prepare(tr, 5)
    assert not bool(prep.fresh.any())
    d = obj.loss(tr, prep, torch.full((2, 3), -0.5, requires_grad=True), torch.zeros(2, 3),
                 torch.zeros(2, 4, requires_grad=True))
    assert d["n_stale_steps"] == 6 and d["n_fresh_steps"] == 0 and "vtrace_rho_mean" in d


# ============================================================================ the integrated learner

def test_step_routes_survivors_to_vtrace_logs_the_group_table_and_bumps_the_version():
    st = _Stack()
    rm = ConstReward(0.5)
    L = I.IntegratedLearner(st.model, rm, None, agent=st.agent, probe_every=1)
    infos = _run(st, L, 8, every=4)
    assert [i["updated"] for i in infos] == [True, True]
    a, b = infos
    assert a["policy"]["n_stale_steps"] == 0 and a["policy"]["n_fresh_steps"] > 0
    assert b["policy"]["n_stale_steps"] > 0 and b["policy"]["n_fresh_steps"] > 0        # V-trace on survivors
    assert b["policy"]["max_lag"] == 1
    assert st.model.version == 2 and st.agent.version == 2
    for i in infos:
        assert set(i["losses"]) >= {"world", "actor", "value"}
        assert sum(i["core_share"].values()) == pytest.approx(1.0)
        assert set(i["dom08"]) == set(DOM08_REQUIRED)
        assert i["actor_psi_grad_norm"] == 0.0
        assert i["reward"]["mode"] == "observation_only" and i["reward"]["condition"] == "D0"
        assert i["reward"]["mean"] == pytest.approx(0.5 * 0.3, rel=1e-5)
        assert i["epochs"][0]["stepped"]
    assert float(rm.psi.detach()) == pytest.approx(0.3)                  # no actor / world loss moved psi
    assert all(len(s.rows) == 1 for s in st.agent.fresh) and len(st.agent.fresh) == st.env.B  # survivors only
    assert len(st.agent.update_log) == 2 and st.agent.update_log[-1]["learner"] == "w7"
    consumed = set(st.agent.update_log[0]["consumed_decisions"])
    linked = [t for t in st.recorder.traces if t["update_refs"]]
    assert consumed and {t["decision_id"] for t in linked} >= consumed     # provenance of consumed decisions
    c = rm.calls[0]
    assert c["R"][1:] == (st.cfg.n_state, st.cfg.d_model) and c["R"] == c["R_next"]
    assert c["kw"] == ["belief", "belief_next", "cost", "info_gain", "intent", "signals"]
    assert c["signals"] == ["action", "lp", "terminal"]          # executed-action tokens for e_a


def test_balancing_moves_the_weights_toward_the_core_share_targets():
    st = _Stack()
    L = I.IntegratedLearner(st.model, ConstReward(1.0), None, agent=st.agent, probe_every=1)
    infos = _run(st, L, 12, every=4)
    b0 = infos[0]["balance"]
    assert b0["mode"] == "calibrate"
    sh, fa = b0["share_before"], b0["factors_after"]
    if sh["world"] > 0.45:
        assert fa["world"] < 1.0
    if sh["actor"] < 0.10:
        assert fa["actor"] > 1.0
    # the predicted shares at the calibrated factors satisfy the targets (linear model of the probe)
    pred = {k: v * fa.get(k, 1.0) for k, v in sh.items()}
    tot = sum(pred.values())
    assert pred["world"] / tot <= 0.45 + 1e-6 and pred["actor"] / tot >= 0.10 - 1e-6
    for i in infos[1:]:                       # weights in force = base weights x the factors of the previous probes
        assert i["weights"]["world"] == pytest.approx(st.cfg.loss_weights["world"] * i["balance_factors_used"]["world"])


def test_observation_only_refuses_environment_rewards_everywhere():
    st = _Stack()
    L = I.IntegratedLearner(st.model, ConstReward(), None, agent=st.agent)
    with pytest.raises(I.RewardIntegrityError):
        L.on_step(SimpleNamespace(rewards=[0.0, 1.5], records=[]))
    L.on_step(SimpleNamespace(rewards=None, records=[]))           # no reward channel: fine
    L.on_step(SimpleNamespace(rewards=[0.0, 0.0], records=[]))
    # the evaluator's verified increments reach the agent's fresh buffer -> the learner refuses the batch
    run_steps(st.agent, st.env, 4, evaluator=st.evaluator, update_every=None)
    assert any(bool((s.reward != 0).any()) for s in st.agent.fresh)       # the toy evaluator pays revenue
    with pytest.raises(I.RewardIntegrityError, match="observation_only"):
        L.step()                               # the replay windows (world-model targets) are checked first
    assert L.integrity["replay_reward_nonzero"] > 0
    st.agent.replay = None                     # without replay the fresh buffer's guard refuses the batch
    with pytest.raises(I.RewardIntegrityError, match="fresh decisions"):
        L.step()
    assert L.integrity["fresh_reward_nonzero"] > 0


def test_reward_model_output_must_be_detached():
    class Leaky(ConstReward):
        def reward(self, R, R_next=None, goal=None, dt=None, **kw):
            return torch.ones(R.shape[0]) * self.psi
    st = _Stack()
    L = I.IntegratedLearner(st.model, Leaky(), None, agent=st.agent)
    run_steps(st.agent, st.env, 4, evaluator=None, update_every=None)
    with pytest.raises(I.RewardIntegrityError, match="detached"):
        L.step()


def test_legacy_condition_uses_the_verified_increment_and_sparse_condition_the_joined_label(tmp_path):
    st = _Stack(_cfg(reward_mode="legacy"))
    L = I.IntegratedLearner(st.model, ConstReward(), None, agent=st.agent)
    run_steps(st.agent, st.env, 4, evaluator=st.evaluator, update_every=None)
    want = [float(s.reward[j]) + st.cfg.lp_coef * float(s.lp[j]) for s in st.agent.fresh for j in range(len(s.rows))]
    info = L.step()
    assert info["reward"]["condition"] == "D1"
    assert info["reward"]["n"] > 0 and min(want) - 1e-6 <= info["reward"]["min"] <= info["reward"]["max"] <= max(want) + 1e-6
    # sparse verified: episodes end (horizon), the operator releases labels, only terminal decisions are rewarded
    st2 = _Stack(_cfg(reward_mode="sparse_verified"), horizon=0.02)
    store = LabelStore(str(tmp_path / "labels"))
    L2 = I.IntegratedLearner(st2.model, ConstReward(), store, agent=st2.agent, sparse_beta=2.0)
    run_steps(st2.agent, st2.env, 6, evaluator=None, update_every=None)
    terms = [(s.env_ids[j], s.episode_ids[j]) for s in st2.agent.fresh for j in range(len(s.rows)) if bool(s.done[j])]
    assert terms, "the short horizon must end episodes"
    for env, ep in terms:
        o = L2._register_episode(env, ep)
        store.append(LabelRecord(env, o, "success", 0.75, 1.0, 0.1, "test-operator"))
    info2 = L2.step()
    assert info2["reward"]["condition"] == "D2" and info2["reward"]["labelled"] == len(terms)
    assert info2["reward"]["max"] == pytest.approx(1.5)


def test_label_batch_reaches_the_reward_model_with_a_grad_carrying_encoder(tmp_path):
    st = _Stack()
    store = LabelStore(str(tmp_path / "labels"))
    store.append(LabelRecord("shop0", 1, "failure", 0.0, 0.0, 1.0, "op"))
    rm = ConstReward()
    L = I.IntegratedLearner(st.model, rm, store, agent=st.agent)
    run_steps(st.agent, st.env, 4, evaluator=None, update_every=None)
    L.step()
    lb = rm.last_batch
    assert len(lb.labels) == 1 and len(lb.new) == 1 and lb.events and lb.R is not None
    R, _ = lb.encode([0, 1])
    assert R.requires_grad and R.shape[0] == 2


def test_state_dict_round_trip_and_meta_step_delegation():
    st = _Stack()
    L = I.IntegratedLearner(st.model, ConstReward(), None, agent=st.agent)
    _run(st, L, 4, every=4)
    s = L.state_dict()
    L2 = I.IntegratedLearner(st.model, ConstReward(), None, agent=st.agent)
    L2.load_state_dict(s)
    assert L2.n_updates == L.n_updates and L2.balancer.factors == L.balancer.factors
    assert L.meta_step()["meta"] is False
    calls = []
    L.meta_trainer = SimpleNamespace(meta_step=lambda learner: calls.append(learner) or {"psi_changed": True})
    out = L.meta_step()
    assert out["meta"] and out["psi_changed"] and calls == [L]


def test_microbatched_policy_gradient_equals_the_full_batch_gradient():
    st = _Stack(B=3)
    L = I.IntegratedLearner(st.model, ConstReward(0.7), None, agent=st.agent)
    _run(st, L, 4, every=4)                                   # one update: survivors (lag 1) + new decisions
    run_steps(st.agent, st.env, 4, evaluator=None, update_every=None)
    pb = L.policy_batch(st.agent.fresh)
    pb.prep = L.objective.prepare(pb.traj, int(st.model.version))
    # one-step evaluator targets r + Gamma * V_old(next) on every trainable node with a successor or terminal
    tr_nodes = sorted({int(i) for i in pb.col[pb.traj.mask].tolist()})
    assert torch.isfinite(pb.eval_target[tr_nodes]).all() and torch.isfinite(pb.eval_target).sum() == len(tr_nodes)
    st.model.train()
    params = [p for p in st.model.parameters() if p.requires_grad]

    def grads(fn):
        for p in params:
            p.grad = None
        fn()
        return [None if p.grad is None else p.grad.clone() for p in params]

    def full():
        d = L._policy_losses(pb)
        (d["actor"] + 0.5 * d["value"]).backward()

    def micro():
        mbs = [list(range(i, min(len(pb.nodes), i + 3))) for i in range(0, len(pb.nodes), 3)]
        assert len(mbs) > 1
        with torch.no_grad():
            base = L._node_eval(pb)
        for idx in mbs:
            d = L._policy_losses(pb, idx, base)
            (d["actor"] + 0.5 * d["value"]).backward()
    g_full = grads(full)
    g_micro = grads(micro)
    n_cmp = 0
    for a, b in zip(g_full, g_micro):
        if a is None:
            assert b is None or float(b.abs().max()) < 1e-7
            continue
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
        n_cmp += 1
    assert n_cmp > 10
    # the learner uses it when cfg.microbatch is set
    st2 = _Stack(_cfg(microbatch=3), B=3)
    L2 = I.IntegratedLearner(st2.model, ConstReward(0.7), None, agent=st2.agent, probe_every=1)
    infos = _run(st2, L2, 8, every=4)
    assert all(i["updated"] for i in infos) and infos[0]["microbatches"][0] == 3
    assert infos[0]["probe_policy"].startswith("actor/value norms from microbatch 0")


def test_ppo_clip_loss_still_refuses_stale_rows_directly():
    from superlearner.cam_unified.learn.ppo import ppo_clip_loss
    m = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(StalePolicyError):
        ppo_clip_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2), m, 0.2, lag=torch.ones(1, 2))
