"""CombinedTrainer: optimiser groups, staged objectives, per-objective gradient contributions into the shared backbone
(norms, shares, cosines), clipping, AMP, non-finite guard, provenance, state."""
import json

import pytest
import torch
import torch.nn as nn

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import DecisionTrace, EventRecord
from superlearner.cam_unified.learn.trainer import STAGES, CombinedTrainer


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.backbone = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 16))
        self.world = nn.Linear(16, 1)
        self.actor = nn.Linear(16, 3)
        self.value = nn.Linear(16, 1)
        self.bridge = nn.Linear(16, 4)

    def losses(self, x):
        h = self.backbone(x)
        return {"world": (self.world(h) - 1).pow(2).mean(),
                "actor": -torch.log_softmax(self.actor(h), -1)[:, 0].mean(),
                "value": self.value(h.detach()).pow(2).mean(),               # cut from the backbone on purpose
                "semantic": self.bridge(h).pow(2).mean()}


def _trainer(m, **kw):
    cfg = make_config("dev", lr=1e-2, lr_sem=1e-3, grad_clip=1.0)
    return CombinedTrainer(cfg, core=m.backbone, heads=[*m.world.parameters(), *m.actor.parameters(),
                                                        *m.value.parameters()], semantic=m.bridge, amp=False, **kw)


def test_stages_are_cumulative_and_cover_the_seven_spec_stages():
    names = [s.name for s in STAGES]
    assert names == ["grounding", "world", "adapt", "actor_critic", "explore_meta", "imagination", "continual"]
    for a, b in zip(STAGES, STAGES[1:]):
        assert a.objectives < b.objectives
    assert STAGES[4].flag("lp_reward") and not STAGES[3].flag("lp_reward")
    assert {"world", "adapt", "actor", "value", "action", "memory", "retention", "meta", "semantic",
            "imagined"} == set(STAGES[-1].objectives)


def test_param_groups_learning_rates_and_disjointness():
    m = _Model()
    tr = _trainer(m)
    lrs = {g["name"]: g["lr"] for g in tr.opt.param_groups}
    assert lrs["core/decay"] == 1e-2 and lrs["semantic/decay"] == 1e-3
    assert all(g["weight_decay"] == 0.0 for g in tr.opt.param_groups if g["name"].endswith("no_decay"))
    with pytest.raises(ValueError):
        CombinedTrainer(make_config("dev"), core=m.backbone, heads=m.backbone)


def test_per_objective_backbone_contributions_shares_and_cosines():
    m = _Model()
    tr = _trainer(m, stage="actor_critic", grad_probe_every=1)
    x = torch.randn(32, 8)
    L = m.losses(x)
    w = tr.weights
    bb = list(m.backbone.parameters())
    want = {}
    for k in ("world", "actor", "value", "semantic"):
        g = torch.autograd.grad(w[k] * L[k], bb, retain_graph=True, allow_unused=True)
        want[k] = float(torch.cat([(torch.zeros_like(p) if gi is None else gi).flatten() for gi, p in zip(g, bb)]).norm())
    info = tr.step(L)
    assert set(info.backbone["norm"]) == {"world", "actor", "value", "semantic"}
    for k, v in want.items():
        assert info.backbone["norm"][k] == pytest.approx(v, rel=1e-4, abs=1e-8)
    assert info.backbone["norm"]["value"] == 0.0          # the value objective is cut from the backbone
    assert sum(info.backbone["share"].values()) == pytest.approx(1.0)
    # opposite objectives have cosine -1; the same objective twice +1
    m2 = _Model()
    tr2 = _trainer(m2, stage="explore_meta", grad_probe_every=1, loss_weights={"world": 1.0, "meta": 1.0})
    h = m2.backbone(torch.randn(16, 8)).sum()
    info2 = tr2.step({"world": h, "meta": -h})
    assert info2.backbone["cosine"]["world|meta"] == pytest.approx(-1.0, abs=1e-5)
    # probing happens every N updates only
    m3 = _Model()
    tr3 = _trainer(m3, stage="actor_critic", grad_probe_every=3)
    probed = [tr3.step(m3.losses(torch.randn(4, 8))).backbone is not None for _ in range(5)]
    assert probed == [True, False, False, True, False]


def test_stage_gating_schedule_and_unknown_objectives():
    m = _Model()
    tr = _trainer(m, stage="grounding", stage_schedule=[(2, "actor_critic")])
    a0 = m.actor.weight.detach().clone()
    info = tr.step(m.losses(torch.randn(8, 8)))
    assert info.stage == "grounding" and set(info.disabled) == {"world", "actor", "value"}
    assert torch.equal(m.actor.weight, a0)                 # disabled objective: its head is not trained
    tr.step(m.losses(torch.randn(8, 8)))
    info = tr.step(m.losses(torch.randn(8, 8)))
    assert info.stage == "actor_critic" and info.disabled == []
    assert not torch.equal(m.actor.weight, a0)
    with pytest.raises(KeyError):
        tr.step({"bogus": torch.tensor(1.0, requires_grad=True)})
    with pytest.raises(ValueError):
        _trainer(_Model(), stage="grounding").step({"actor": m.losses(torch.randn(2, 8))["actor"]})


def test_clipping_nonfinite_guard_provenance_logging_and_state(tmp_path):
    m = _Model()
    log = tmp_path / "upd.jsonl"
    tr = _trainer(m, stage="continual", log_path=str(log))
    big = {"world": 1e4 * m.losses(torch.randn(8, 8))["world"]}
    ev = EventRecord("e1", "env", "a", "ep", 0, 0, 0.0, [])
    dt = DecisionTrace("d1", "e1", "env", 0, 0.0, "goal")
    info = tr.step(big, consumed=[ev, dt])
    assert info.clipped and info.grad_norm > 1.0
    assert ev.training_provenance["used_in_updates"] == [info.update_id] and dt.update_refs == [info.update_id]
    before = [p.detach().clone() for p in m.parameters()]
    info = tr.step({"world": torch.tensor(float("nan"), requires_grad=True) * m.losses(torch.randn(2, 8))["world"]})
    assert info.skipped and tr.n_skipped == 1
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    tr_raise = _trainer(_Model(), stage="world", on_nonfinite="raise")
    with pytest.raises(FloatingPointError):
        tr_raise.step({"world": torch.tensor(float("inf"), requires_grad=True)})
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["update_id"] == ev.training_provenance["used_in_updates"][0]
    st = tr.state_dict()
    tr2 = _trainer(_Model(), stage="grounding")
    tr2.load_state_dict(st)
    assert tr2.n_updates == 2 and tr2.stage.name == "continual"
    tr.reset_optimizer()
    assert len(tr.opt.state) == 0


def test_bf16_autocast_training_step():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _Model().to(dev)
    tr = CombinedTrainer(make_config("dev", lr=1e-2), core=m.backbone, heads=[*m.world.parameters()], amp=True,
                         stage="world", grad_probe_every=1)
    with tr.autocast():
        h = m.backbone(torch.randn(16, 8, device=dev))
        assert h.dtype == torch.bfloat16
        loss = (m.world(h).float() - 1).pow(2).mean()
    w0 = m.world.weight.detach().clone()
    info = tr.step({"world": loss})
    assert not info.skipped and info.backbone["norm"]["world"] > 0
    assert not torch.equal(w0, m.world.weight)


def test_backbone_probe_on_a_large_backbone_uses_exact_norms_and_sampled_cosines():
    m = _Model()
    tr = _trainer(m, stage="explore_meta", grad_probe_every=1, loss_weights={"world": 1.0, "meta": 1.0},
                  cos_exact_numel=100, cos_sample=64)
    x = torch.randn(16, 8)
    h = m.backbone(x)
    la, lb = h[:, :8].pow(2).sum(), h[:, 8:].sum()
    bb = list(m.backbone.parameters())
    ga = torch.autograd.grad(la, bb, retain_graph=True)
    gb = torch.autograd.grad(lb, bb, retain_graph=True)
    fa = torch.cat([g.flatten() for g in ga])
    fb = torch.cat([g.flatten() for g in gb])
    info = tr.step({"world": la, "meta": lb})
    assert info.backbone["cosine_exact"] is False
    assert info.backbone["norm"]["world"] == pytest.approx(float(fa.norm()), rel=1e-5)
    assert info.backbone["norm"]["meta"] == pytest.approx(float(fb.norm()), rel=1e-5)
    true_cos = float(fa @ fb / (fa.norm() * fb.norm()))
    assert -1.0 <= info.backbone["cosine"]["world|meta"] <= 1.0
    assert abs(info.backbone["cosine"]["world|meta"] - true_cos) < 0.5
    h2 = m.backbone(x).sum()
    info2 = tr.step({"world": h2, "meta": -h2})
    assert info2.backbone["cosine"]["world|meta"] == pytest.approx(-1.0, abs=1e-5)   # exact even when sampled
    info3 = tr.step({"world": m.backbone(x).sum(), "meta": torch.tensor(0.5)})        # objective without a graph
    assert info3.backbone["norm"]["meta"] == 0.0


def test_critic_weight_must_be_unambiguous():
    m = _Model()
    with pytest.raises(ValueError):
        CombinedTrainer(make_config("dev", value_coef=1.0), core=m.backbone, amp=False)
    tr = CombinedTrainer(make_config("dev", value_coef=1.0), core=m.backbone, amp=False,
                         loss_weights={"value": 1.0})
    assert tr.weights["value"] == 1.0
    cfg = make_config("dev", value_coef=2.0)
    cfg.loss_weights["value"] = 2.0
    assert CombinedTrainer(cfg, core=m.backbone, amp=False).weights["value"] == 2.0


def test_sat_optimizer_adafactor_splits_core_adamw_from_satellite_adafactor():
    """cfg.sat_optimizer="adafactor" (plan 2026-11-26 §4 G3): AdamW for the core groups, Adafactor for the others,
    one role optimiser to the trainer (lr writes, apply, zero_grad, state_dict round trip)."""
    from superlearner.cam_unified.learn.trainer import SplitOptimizer
    m = _Model()
    cfg = make_config("dev", lr=1e-2, grad_clip=1.0, sat_optimizer="adafactor")
    tr = CombinedTrainer(cfg, groups={"core": list(m.backbone.parameters()),
                                      "heads": [*m.world.parameters(), *m.actor.parameters()]}, amp=False)
    assert isinstance(tr.opt, SplitOptimizer)
    kinds = [type(o).__name__ for o in tr.opt.parts]
    assert kinds == ["AdamW", "Adafactor"]
    core_ids = {id(p) for p in m.backbone.parameters()}
    assert all(id(p) in core_ids for g in tr.opt.parts[0].param_groups for p in g["params"])
    assert not any(id(p) in core_ids for g in tr.opt.parts[1].param_groups for p in g["params"])
    x = torch.randn(16, 8)
    before = [p.detach().clone() for p in m.parameters()]
    L = m.losses(x)
    (L["world"] + L["actor"]).backward()
    out = tr.apply("slow")
    assert not out["skipped"]
    moved = [not torch.equal(a, p.detach()) for a, p in zip(before, m.parameters())]
    assert moved[0] and any(moved[4:8])                   # a core and a satellite parameter moved
    assert all(p.grad is None for p in m.parameters())
    sd = tr.state_dict()
    tr2 = CombinedTrainer(cfg, groups={"core": list(m.backbone.parameters()),
                                       "heads": [*m.world.parameters(), *m.actor.parameters()]}, amp=False)
    tr2.load_state_dict(sd)
    assert len(tr2.opt.state) == len(tr.opt.state) > 0
