"""Seam commit (redesign plan 2026-11-26 §3, tag seam-0): the moved runtime bodies live in their owners' modules,
CamUnifiedModel / CamUnifiedAgent keep one-line delegations (every existing caller keeps working), interfaces v4 is
additive, and the DOM presets build."""
import ast
import dataclasses
import importlib
import inspect
import textwrap

import pytest
import torch

from superlearner.cam_unified import agent as agent_mod
from superlearner.cam_unified import config as config_mod
from superlearner.cam_unified import interfaces as itf
from superlearner.cam_unified import model as model_mod
from superlearner.cam_unified.agent import CamUnifiedAgent
from superlearner.cam_unified.config import DOM_SWITCHES, PRESETS, CamConfig, make_config, make_preset
from superlearner.cam_unified.model import CamUnifiedModel

PKG = "superlearner.cam_unified"

# model method -> (new module, function); methods that were private keep their name on the model
MODEL_DELEGATIONS = {
    "filter_sequence": ("world.train_seq", "filter_sequence"),
    "world_loss": ("world.train_seq", "world_loss"),
    "observe_outcomes": ("core.belief_runtime", "observe_outcomes"),
    "predict_outcomes": ("core.belief_runtime", "predict_outcomes"),
    "base_moments": ("core.belief_runtime", "base_moments"),
    "kalman_bound": ("core.belief_runtime", "kalman_bound"),
    "kalman_predict": ("core.belief_runtime", "kalman_predict"),
    "kalman_log_prob": ("core.belief_runtime", "kalman_log_prob"),
    "kalman_update": ("core.belief_runtime", "kalman_update"),
    "mixture_ll": ("core.belief_runtime", "mixture_ll"),
    "mixture_loglik": ("core.belief_runtime", "mixture_loglik"),
    "adapt_loss": ("core.belief_runtime", "adapt_loss"),
    "channels_of": ("core.belief_runtime", "channels_of"),
    "combine": ("core.belief_runtime", "combine"),
    "residual_batch": ("core.belief_runtime", "residual_batch"),
    "belief_view": ("core.belief_runtime", "belief_view"),
    "_particle_dist": ("core.belief_runtime", "particle_dist"),
    "_kalman_on": ("core.belief_runtime", "kalman_on"),
    "_bayes_prior": ("core.belief_runtime", "bayes_prior"),
    "retrieve": ("memory.runtime", "retrieve"),
    "memory_tokens": ("memory.runtime", "memory_tokens"),
    "stored_memory_tokens": ("memory.runtime", "stored_memory_tokens"),
}
MODEL_STATIC = {"_continuous": ("core.belief_runtime", "continuous_mask"),
                "_padded": ("core.belief_runtime", "padded_rows")}
AGENT_REEXPORTS = ("objective_world", "objective_adapt", "objective_memory", "objective_policy", "gae_semi_markov",
                   "DEFAULT_OBJECTIVES", "SUPPORT_SIZES")


def _mod(rel):
    return importlib.import_module(f"{PKG}.{rel}")


def _is_delegation(fn, target: str) -> bool:
    """fn's body is exactly `return <target>(...)` (one statement, no decorators)."""
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    if not isinstance(node, ast.FunctionDef) or node.decorator_list or len(node.body) != 1:
        return False
    ret = node.body[0]
    return isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call) and ast.unparse(ret.value.func) == target


# ============================================================================ moved bodies and delegations

def test_planner_lives_in_planner_runtime_and_is_re_exported():
    pr = _mod("world.planner_runtime")
    assert model_mod.LatentImaginationPlanner is pr.LatentImaginationPlanner
    assert pr.LatentImaginationPlanner.__module__ == f"{PKG}.world.planner_runtime"
    assert "class LatentImaginationPlanner" not in inspect.getsource(model_mod)
    assert callable(pr.LatentImaginationPlanner().plan)


@pytest.mark.parametrize("method", sorted(MODEL_DELEGATIONS))
def test_model_method_is_a_one_line_delegation(method, monkeypatch):
    rel, fname = MODEL_DELEGATIONS[method]
    mod = _mod(rel)
    fn = getattr(mod, fname)
    assert fn.__module__ == mod.__name__, f"{fname} must be defined in {rel}"
    meth = CamUnifiedModel.__dict__[method]
    alias = {"world.train_seq": "_train_seq", "core.belief_runtime": "_belief", "memory.runtime": "_memory"}[rel]
    assert getattr(model_mod, alias) is mod
    assert _is_delegation(meth, f"{alias}.{fname}"), f"CamUnifiedModel.{method} is not a one-line delegation"
    n_args = len(inspect.signature(fn).parameters) - 1          # without `model`
    calls = []

    def spy(*a, **k):
        calls.append((a, k))
        return "sentinel"
    monkeypatch.setattr(mod, fname, spy)
    me = object()
    args = tuple(object() for _ in range(n_args))
    assert meth(me, *args) == "sentinel"
    assert calls == [((me,) + args, {})]


@pytest.mark.parametrize("name", sorted(MODEL_STATIC))
def test_model_static_helpers_are_the_runtime_functions(name):
    rel, fname = MODEL_STATIC[name]
    mod = _mod(rel)
    raw = CamUnifiedModel.__dict__[name]
    assert isinstance(raw, staticmethod) and raw.__func__ is getattr(mod, fname)


def test_model_py_no_longer_holds_the_moved_bodies():
    src = inspect.getsource(model_mod)
    for marker in ("LP_{t-1} = log p(o_t | B_t)", "rep = torch.arange(B, device=dev).repeat_interleave(n)",
                   "n_kal = n_mix = 0", "kids = idx.kids_of(version, flat)", "res.n_belief_updates += 1"):
        assert marker not in src, marker


def test_agent_objectives_live_in_learn_integrated_and_are_re_exported():
    integ = _mod("learn.integrated")
    for name in AGENT_REEXPORTS:
        assert getattr(agent_mod, name) is getattr(integ, name), name
    for name in ("objective_world", "objective_adapt", "objective_memory", "objective_policy", "gae_semi_markov",
                 "collect_losses", "cap_losses", "trainer_step"):
        assert getattr(integ, name).__module__ == integ.__name__
    src = inspect.getsource(agent_mod)
    assert "def objective_policy" not in src and "def gae_semi_markov" not in src
    upd = inspect.getsource(CamUnifiedAgent.update)
    assert "integrated.collect_losses(self, batch)" in upd and "integrated.cap_losses(self, losses, metrics)" in upd
    assert _is_delegation(CamUnifiedAgent._trainer_step, "integrated.trainer_step")
    from superlearner.cam_unified.capabilities import reasoning      # an existing `from ..agent import` caller
    assert reasoning is not None


def test_agent_trainer_step_delegates(monkeypatch):
    integ = _mod("learn.integrated")
    seen = []
    monkeypatch.setattr(integ, "trainer_step", lambda ag, losses, consumed: seen.append((ag, losses, consumed)) or {})
    me, losses, consumed = object(), {"a": 1}, ["d"]
    assert CamUnifiedAgent._trainer_step(me, losses, consumed) == {}
    assert seen == [(me, losses, consumed)]


class _FakeAgent:
    def __init__(self, objectives, loss_cap=None):
        self.objectives = objectives
        self.loss_cap = loss_cap
        self.loss_cap_beta = 0.9
        self.loss_mag = {}


def test_collect_and_cap_losses_keep_the_update_semantics():
    integ = _mod("learn.integrated")
    w = torch.tensor(2.0, requires_grad=True)

    def obj_a(agent, batch):
        return {"losses": {"world": w * 50.0, "flat": torch.tensor(1.0)}, "metrics": {"m": 1},
                "consumed": ["d1"], "consumed_events": [("e1", "env0")]}
    ag = _FakeAgent({"a": obj_a, "empty": lambda agent, batch: {}}, loss_cap=10.0)
    losses, mets, consumed, events = integ.collect_losses(ag, None)
    assert set(losses) == {"world"} and mets["skipped/flat"] == 1 and mets["m"] == 1
    assert consumed == ["d1"] and events == [("e1", "env0")]
    raw = integ.cap_losses(ag, losses, mets)
    assert raw == {"world": 100.0} and mets["balance/world"] == pytest.approx(0.1)
    assert float(losses["world"].detach()) == pytest.approx(10.0) and losses["world"].requires_grad
    bad = _FakeAgent({"a": lambda agent, batch: {"losses": {"x": w * float("nan")}}})
    with pytest.raises(FloatingPointError):
        integ.collect_losses(bad, None)


# ============================================================================ interfaces v4

def test_interfaces_v4_is_versioned_and_additive():
    assert itf.CONTRACT_VERSION == 4
    assert itf.EVENT_SCHEMA == "cam-event/4"
    assert itf.EVENT_VERSION_KEYS == ("policy", "reward_model", "evaluator", "env", "code")
    v = itf.event_versions(policy=3, code="abc")
    assert tuple(v) == itf.EVENT_VERSION_KEYS and v["policy"] == 3 and v["reward_model"] is None
    rec = itf.EventRecord("e", "env", "a", "ep", 0, 3, 0.0, [], extra={"versions": v, "schema": itf.EVENT_SCHEMA})
    back = itf.EventRecord.from_dict(rec.to_dict())
    assert back.extra["versions"] == v and back.extra["schema"] == "cam-event/4"


def test_v4_protocols_are_structural():
    class Labels:
        def take_labels(self):
            return []

    class Learner:
        def on_step(self, transition):
            pass

        def step(self):
            return {"updated": False}

    assert isinstance(Labels(), itf.LabelSource) and not isinstance(Learner(), itf.LabelSource)
    assert isinstance(Learner(), itf.LearnerHook) and not isinstance(Labels(), itf.LearnerHook)


def test_reward_model_stub_returns_detached_zeros():
    from superlearner.cam_unified.selfeval.reward_model import RewardModel
    rm = RewardModel(make_config("dev"))
    assert isinstance(rm, itf.RewardModelP) and rm.version == 0
    assert sum(p.numel() for p in rm.parameters()) == 0
    R = torch.randn(3, 4, 8, requires_grad=True)
    r = rm.reward(R, R, None, torch.ones(3), info_gain=torch.ones(3), cost=torch.ones(3))
    assert r.shape == (3,) and r.dtype == torch.float32 and not r.requires_grad and bool((r == 0).all())
    assert rm.reward(None, None, None, torch.ones(5)).shape == (5,)
    phi = rm.potential(R, None)
    assert phi.shape == (3,) and not phi.requires_grad and bool((phi == 0).all())
    assert rm.losses([]) == {}


# ============================================================================ config fields and presets

LEGACY = dict(state_gate=False, state_emb_std=0.5, sem_xattn_every=0, dec_xattn_every=0, dec_alpha_init=0.05,
              n_intent=0, n_candidates=1, select_tau=1.0, veto_q=None, value_channel=False, d_satellite=0,
              wm_transition_blocks=0, wm_ema_target=True, kl_free_bits_group=0.0, wm_consistency=0.0, burn_in=0,
              bptt_steps=0, reason_iters_fixed=0, reward_mode="legacy", reward_in_beliefs=True, meta_every=0,
              sem_lora=True, ppo_epochs=1, target_kl=0.0, lr_core=0.0, lr_heads=0.0, lr_warmup_updates=0,
              clip_mode="global", balance_every=0, world_core_share_max=1.0, microbatch=0, preset="")


def test_new_config_fields_default_to_the_pre_seam_behaviour():
    names = {f.name for f in dataclasses.fields(CamConfig)}
    for scale in config_mod.SCALES:
        cfg = make_config(scale)
        for k, v in LEGACY.items():
            assert k in names and getattr(cfg, k) == v, (scale, k)
        assert cfg.use_imagination and cfg.use_memory and cfg.max_reason_iters == 16


def test_model_reads_reward_in_beliefs_and_fixed_reasoning_from_the_config():
    from superlearner.cam_unified import integration_doubles as dbl
    from superlearner.cam_unified.model import build_model
    from superlearner.cam_unified.testing import tiny_config
    m = build_model(tiny_config(), components=dbl.double_components(tiny_config()))
    assert m.reward_in_beliefs is True and m.reason_iters is None            # pre-seam defaults
    cfg = tiny_config(reward_in_beliefs=False, reason_iters_fixed=2)
    m = build_model(cfg, components=dbl.double_components(cfg))
    assert m.reward_in_beliefs is False and m.reason_iters == 2
    m = build_model(cfg, components=dbl.double_components(cfg), reward_in_beliefs=True, reason_iters=3)
    assert m.reward_in_beliefs is True and m.reason_iters == 3               # explicit kwargs still win


@pytest.mark.parametrize("name,dims,sem", [("dom", (32, 1536, 24, 128, 4096), "Qwen/Qwen3-4B"),
                                           ("dom_dev", (8, 384, 6, 32, 1024), "Qwen/Qwen3-0.6B"),
                                           ("dom_150m", (16, 768, 12, 64, 2048), "Qwen/Qwen3-4B")])
def test_dom_presets_build_configs(name, dims, sem):
    cfg = make_preset(name)
    assert isinstance(cfg, CamConfig) and cfg.preset == name
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_state, cfg.d_ff) == dims
    assert cfg.sem_model == sem and cfg.max_ctx_tokens == 512
    for k, v in DOM_SWITCHES.items():                     # same switches in every dom preset
        assert getattr(cfg, k) == v, (name, k)
    assert cfg.state_gate and cfg.state_emb_std == 0.02 and cfg.sem_xattn_every == 4
    assert cfg.sem_xattn_max_tokens == 256 and cfg.dec_xattn_every == 6 and cfg.dec_alpha_init == 0.0
    assert (cfg.n_intent, cfg.n_candidates, cfg.select_tau) == (16, 4, 1.0) and cfg.value_channel
    assert (cfg.wm_transition_blocks, cfg.wm_ema_target, cfg.kl_free_bits_group, cfg.wm_consistency,
            cfg.burn_in) == (6, False, 0.02, 1.0, 16)
    assert (cfg.reason_iters_fixed, cfg.max_reason_iters) == (2, 32)
    assert not cfg.use_imagination and not cfg.use_memory
    assert (cfg.reward_mode, cfg.reward_in_beliefs, cfg.meta_every) == ("observation_only", False, 16)
    assert cfg.sem_xattn_dim == {"Qwen/Qwen3-4B": 2560, "Qwen/Qwen3-0.6B": 1024}[sem]
    assert cfg.d_satellite == (768 if name == "dom" else 0)
    assert make_preset(name, n_candidates=2).n_candidates == 2
    assert set(PRESETS) == {"dom", "dom_dev", "dom_150m"}


def test_unknown_preset_is_refused():
    with pytest.raises(KeyError):
        make_preset("dom_xl")


def _build_components(cfg, device):
    """Every non-semantic component of the preset through the registry's constructor injection; the core is the
    Mechanism C GatedUnifiedCore. On the meta device, components whose initialisation reads tensor values are built
    on the CPU instead (recorded)."""
    from superlearner.cam_unified.core.semantic_xattn import GatedUnifiedCore
    from superlearner.cam_unified.model import DEFAULT_COMPONENT_PATHS, N_CONTINUOUS_CORE, _instantiate
    with torch.device(device):
        built = {"core": GatedUnifiedCore(cfg)}
    where = {"core": device}
    for n in ("entity_encoder", "reasoner", "outcome_head", "beliefs", "bayes", "world_model", "value_head",
              "option_policy", "exploration_head"):
        kw = {"n_channels": N_CONTINUOUS_CORE + 32} if n == "beliefs" else {}
        try:
            with torch.device(device):
                built[n] = _instantiate(n, DEFAULT_COMPONENT_PATHS[n], cfg, built, kw)
            where[n] = device
        except RuntimeError as e:
            if device != "meta" or "meta" not in str(e):
                raise
            built[n] = _instantiate(n, DEFAULT_COMPONENT_PATHS[n], cfg, built, kw)
            where[n] = "cpu"
    return built, where


@pytest.mark.parametrize("name,device", [("dom_dev", "cpu"), ("dom_150m", "meta"), ("dom", "meta")])
def test_dom_presets_instantiate(name, device):
    cfg = make_preset(name)
    built, where = _build_components(cfg, device)
    core = built["core"]
    assert where["core"] == device and next(core.parameters()).device.type == device
    assert core.xattn_layers == [i for i in range(cfg.n_layers) if (i + 1) % 4 == 0] and core.sem_proj is not None
    n_core = sum(p.numel() for p in core.parameters())
    if name == "dom":
        assert 0.9e9 < n_core < 1.1e9, n_core                   # the ~1B Superlearner core
    assert built["world_model"].core is core
    if device == "cpu":
        Z = core.init_state(2)
        assert tuple(Z.shape) == (2, cfg.n_state, cfg.d_model)
