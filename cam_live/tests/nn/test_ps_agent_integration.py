"""PS — the persistent core state on the agent's real runtime path (change request PS-1 applied to agent.py).

Skipped while ``CamLiveAgent`` has no ``persist`` attribute (agent.py is ST's; PS-1 forwards the exact diff). Run on
a patched copy of the tree with the diff applied (``docs/PS.md`` §3) these tests passed there.
"""
import os

import pytest
import torch

from cam_live.agent import CamLiveAgent
from cam_live.config import CamLiveConfig
from cam_live.contracts import SimClock

pytestmark = pytest.mark.skipif(not hasattr(CamLiveAgent, "_live"), reason="change request PS-1 not applied to agent.py")


def _cfg(enabled=True):
    from cam_live.eval.conditions import base_config
    cfg = base_config(real=False, seed=0)
    sec = dict(cfg.sections.get("agent", {}))
    sec["use_backbone"] = False
    cfg.sections["agent"] = sec
    cfg.sections["persistent_state"] = {"enabled": enabled, "unroll": 2}
    return cfg


def _lifetime(agent, root, clock, n=4, seed=7):
    from cam_live.eval import envs
    from cam_live.eval.runner import run_lifetime
    envs._TARGETS[("F1", seed)] = 4                    # skip the oracle calibration run (test speed)
    return run_lifetime(agent, "F1", seed, os.path.join(root, f"lt{seed}"), clock, max_decisions=n, k=8)


def test_runtime_commits_every_decision_and_the_learner_replays_with_the_carried_state(tmp_path):
    torch.set_num_threads(4)
    clock = SimClock(1.8e9)
    agent = CamLiveAgent(_cfg(), str(tmp_path / "agent"), clock=clock)
    assert agent.model.persistent is agent.persist
    lt = _lifetime(agent, str(tmp_path), clock)
    st = agent.persist.status()
    n = agent.store.count()
    assert lt["decisions"] >= 2 and st["positions"] == n, (lt["decisions"], st, n)
    c = st["counters"]
    assert c["rows_live"] > 0 and c["commits"] == n                     # decisions read the carried state
    assert c["rows_context"] + c["rows_post"] > 0                       # the learner re-encoded stored decisions
    # a stored decision re-encoded now equals what the decision-time path computes for its position
    rec = agent.store.all()[-1]
    with torch.no_grad():
        b = agent.model.core_forward(agent.model.tokenize([agent.store.items(rec.context_item_refs)]), None).b
        pos = agent.persist._by_decision[rec.decision.decision_id]
        want = agent.model.exp.core(agent.model.tokenize([agent.store.items(rec.context_item_refs)]),
                                    agent.persist.state_for([pos])).b
    assert torch.allclose(b, want, atol=1e-6)
    # checkpoint: a rebuilt agent restores the chain consistently and continues from the same state
    live = agent.persist.live_state()
    ck = agent.save(str(tmp_path / "ck"))
    assert ck
    agent.close()
    again = CamLiveAgent(_cfg(), str(tmp_path / "agent"), clock=clock)
    again.load(str(tmp_path / "ck"))
    rep = again.restore_report["persistent_state"]
    assert rep["legacy_checkpoint"] is False and rep["consistent"] and rep["positions_at_checkpoint"] == n
    assert torch.allclose(again.persist.live_state().slots, live.slots, atol=1e-6)
    assert again.status()["persistent_state"]["positions"] == n
    again.close()


def test_checkpoint_without_the_chain_restores_closed(tmp_path):
    torch.set_num_threads(4)
    clock = SimClock(1.8e9)
    old = CamLiveAgent(_cfg(enabled=False), str(tmp_path / "a"), clock=clock)
    assert old.persist is None and old.model.persistent is None
    _lifetime(old, str(tmp_path), clock, n=2)
    old.save(str(tmp_path / "ck"))
    recs = old.store.all()
    with torch.no_grad():
        ref = [old.model.core_forward(old.model.tokenize([old.store.items(r.context_item_refs)]), None).b for r in recs]
    old.close()
    new = CamLiveAgent(_cfg(), str(tmp_path / "a"), clock=clock)
    new.load(str(tmp_path / "ck"))
    rep = new.restore_report["persistent_state"]
    assert rep["legacy_checkpoint"] is True and rep["rho"] == [0.0] * new.model.exp.core.n_slots
    with torch.no_grad():
        for r, b in zip(recs, ref):
            got = new.model.core_forward(new.model.tokenize([new.store.items(r.context_item_refs)]), None).b
            assert torch.equal(got, b)
        with new.persist.live():
            items = new._items()
            b_live = new.model.core_forward(new.model.tokenize([items]), None).b
        assert torch.equal(b_live, new.model.exp.core(new.model.tokenize([items]), None).b)
    new.close()


def test_weights_carry_the_learned_behaviour_at_equal_chain_state_and_export_starts_initial(tmp_path):
    """With the chain on: restoring the initial weights restores the initial decision distribution when both are read
    at the initial chain state (the chain is experience-dependent state, like the fast beliefs); a weights-only export
    rebuilt by ``load_exported_model`` computes the agent's distribution at the initial chain state."""
    from cam_live.agent import register_agent_heads
    from cam_live.credit.checkpoint import load_exported_model
    from cam_live.validation import HeldOutProbe, weight_swap
    torch.set_num_threads(4)
    clock = SimClock(1.8e9)
    agent = CamLiveAgent(_cfg(), str(tmp_path / "agent"), clock=clock)
    probe = HeldOutProbe(agent)
    p0, s0 = probe.distribution(), agent.model.trainable_state()        # nothing committed: the initial chain state
    _lifetime(agent, str(tmp_path), clock, n=6)
    assert agent.persist.n >= 2
    with agent.persist.initial():
        p1 = probe.distribution()
    carried = probe.distribution()
    assert float(0.5 * (p0 - p1).abs().sum()) > 1e-6                   # learning changed the weights' behaviour
    s1 = agent.model.trainable_state()
    weight_swap(agent.model, s0, ("experiential", "heads", "semantic_adapters", "reward"))
    with agent.persist.initial():
        assert float(0.5 * (probe.distribution() - p0).abs().sum()) < 1e-9
    weight_swap(agent.model, s1, ("experiential", "heads", "semantic_adapters", "reward"))
    assert float(0.5 * (probe.distribution() - carried).abs().sum()) < 1e-9
    dest = str(tmp_path / "export")
    agent.export_weights(dest)
    holder = {}
    ex = load_exported_model(dest, register_heads=lambda m: holder.setdefault("heads", register_agent_heads(m)),
                             strict=True)
    model, heads = ex.model, holder["heads"]
    assert getattr(model, "persistent", None) is None                   # an export carries no chain
    with torch.no_grad():
        b = model.core_forward(model.tokenize([list(probe.items)]), None).b
        e_a, _ = model.encode_actions([list(probe.candidates)])
        z = model.decision_latent(b, e_a, model.encode_goals([probe.goal]))
        mask = torch.ones(1, len(probe.candidates), dtype=torch.bool, device=z.device)
        p_ex = heads.policy(z, mask, b).probs[0].double().cpu()
    assert float((p_ex - p1).abs().max()) < 1e-6
    agent.close()
