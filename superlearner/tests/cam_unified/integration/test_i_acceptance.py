"""Acceptance tests of the assembly over the real workstream classes (default dotted paths). Each test skips while
the needed workstreams are not merged; once they are, the full assembly runs end to end on them. Verified before the
merge on an overlay of the B/C/D/E worktrees (see the workstream-I report)."""
import importlib

import pytest
import torch

from conftest import make_cfg
from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.agent import CamUnifiedAgent
from superlearner.cam_unified.model import DEFAULT_COMPONENT_PATHS, build_model
from superlearner.cam_unified.runtime import run_steps

REAL = ("core", "entity_encoder", "reasoner", "beliefs", "bayes", "outcome_head", "world_model", "retriever",
        "value_head", "option_policy", "exploration_head")


def _need(*names):
    for n in names:
        mod = DEFAULT_COMPONENT_PATHS[n].split(":")[0]
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            pytest.skip(f"{n} ({mod}) not merged yet")


def _trainer(model, cfg):
    try:
        from superlearner.cam_unified.learn.trainer import CombinedTrainer
    except ModuleNotFoundError:
        return dbl.DoubleTrainer(model, cfg), "double"
    core = {id(p) for p in model.core.parameters()}
    heads = [p for p in model.parameters() if id(p) not in core and p.requires_grad]
    return CombinedTrainer(cfg, core=model.core, heads=heads, amp=False, grad_probe_every=1, stage="continual"), "D"


def test_real_b_c_d_e_components_run_end_to_end():
    _need(*REAL)
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build_model(cfg, components={"tokenizer": dbl.DoubleTokenizer(cfg),
                                         "action_generator": dbl.DoubleActionGenerator(cfg)})
    for n in REAL:
        assert model.build_report[n].startswith("built from superlearner.cam_unified."), (n, model.build_report[n])
    assert model.n_channels == 3 + model.max_numeric_outcomes          # Kalman channels sized to the key budget
    trainer, kind = _trainer(model, cfg)
    arch = dbl.DoubleArchive()
    rec = dbl.ListTraceRecorder()
    agent = CamUnifiedAgent(model, archive=arch, recorder=rec, replay=dbl.DoubleReplay(arch), trainer=trainer,
                            seq_len=4, n_seqs=2, adapt_seq_len=12, adapt_n_seqs=2)
    env = dbl.ToyShopEnv(B=2, seed=0)
    env.horizon = 100.0
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    rep = run_steps(agent, env, 10, evaluator=dbl.ToyShopEvaluator(env), update_every=5)
    assert len(rep.updates) == 2 and all(u["updated"] for u in rep.updates)
    for u in rep.updates:
        assert {"world", "adapt", "actor", "value"} <= set(u["losses"])
        assert u["policy/logp_maxdev_lag0"] < 1e-3                       # acting-mode recomputation is exact
        assert u["adapt/adapt_kalman_outcomes"] > 0
    now = dict(model.named_parameters())
    for prefix in ("core.", "reasoner.", "beliefs.", "bayes.", "outcome_head.", "world_model.", "value_head.",
                   "option_policy.", "exploration_head.", "entity_encoder."):
        assert any(not torch.equal(p0[n], now[n].detach()) for n in now if n.startswith(prefix)), prefix
    for n in [k for k in now if k.startswith("bayes.") and "log_" in k]:
        assert not torch.equal(p0[n], now[n].detach()), n               # B's learned prior / noise are trained
    assert len(rec.traces) == 20 and all(t["update_refs"] for t in rec.traces[:10])
    assert model.calls.get("imagination.bayes.update", 0) > 0 and rep.dropped["channels_unbound"] == 0
