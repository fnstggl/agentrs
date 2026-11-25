"""Workstream X: the integrated stack's adapters and the checkpoint conditions the experiments rely on (real
components, Qwen3-0.6B frozen encoder on the GPU)."""
from __future__ import annotations

import os

import pytest
import torch

from superlearner.cam_unified.experiments import mechrun as M
from superlearner.cam_unified.experiments import stack as S
from superlearner.cam_unified.interfaces import EntityRegistry, TokenSet


def test_safe_entity_encoder_handles_empty_steps_and_delegates_otherwise():
    from superlearner.cam_unified.core.entities import EntityEncoder
    cfg = S.experiment_config()
    inner = EntityEncoder(cfg)
    enc = S.SafeEntityEncoder(inner)
    empty = TokenSet.empty(3, cfg.d_model)
    with pytest.raises(IndexError):
        inner(empty)                                           # the defect the adapter works around
    out = enc(empty)
    assert out.B == 3 and out.N == 0
    x = torch.randn(2, 5, cfg.d_model)
    ts = TokenSet.from_dense(x, 3, entity=torch.tensor([[0, 0, 1, -1, 1], [2, 2, 2, -1, -1]]))
    torch.manual_seed(0)
    a = enc(ts)
    torch.manual_seed(0)
    b = inner(ts)
    assert torch.equal(a.x, b.x) and a.N == b.N > 0


def test_memo_tokenizer_slot_codes_are_bit_identical(sem):
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer
    cfg = S.experiment_config()
    memo = S.memo_tokenizer_class()(cfg, sem)
    plain = ObservationTokenizer(cfg, sem)
    for seed in (0, 7, 123):
        r1, r2, r3 = EntityRegistry(cfg.max_entities, seed=seed), EntityRegistry(cfg.max_entities, seed=seed), \
            EntityRegistry(cfg.max_entities, seed=seed)
        assert torch.equal(memo.slot_codes(r1), plain.slot_codes(r2))
        assert torch.equal(memo.slot_codes(r3), plain.slot_codes(r2))       # served from the memo for a new registry
    assert not torch.equal(memo.slot_codes(EntityRegistry(cfg.max_entities, seed=1)),
                           memo.slot_codes(EntityRegistry(cfg.max_entities, seed=2)))


def test_syncing_replay_delivers_sequences_the_raw_sampler_misses(sem, tmp_path, dev_tasks):
    st = S.build_stack(S.experiment_config(), run_dir=str(tmp_path / "run"), seed=0, sem=sem)
    w = M.make_world(dev_tasks[:1], seed=0)
    M.run_batch(st.agent, w, learn=False, max_rounds=12)
    assert st.archive.count() >= 12
    raw = st.replay.sampler
    assert raw.sample(2, 4) == []                              # E's sampler without sync sees nothing
    got = st.replay.sample(2, 4)
    assert len(got) == 2 and all(len(s.records) >= 1 for s in got) and st.replay.syncs == 1


def test_checkpoint_conditions_restore_exactly_their_parts(sem, tmp_path, dev_tasks):
    """B restores weights only (no optimiser moments, no experience, version 0); C adds the optimiser; F also the
    experience handles, version and the tool registry - the separation the negative-transfer diagnosis needs."""
    run = str(tmp_path / "src")
    st = S.build_stack(S.experiment_config(), run_dir=run, seed=0, sem=sem)
    w = M.make_world(dev_tasks[:1], seed=0)
    M.run_batch(st.agent, w, learn=True, update_every=6, max_rounds=12)
    assert st.agent.version >= 1 and sum(len(o.state) for o in st.agent.optimizers.values()) > 0
    ck = str(tmp_path / "ckpt")
    info = S.save_checkpoint(st, ck)
    assert info["tools"] > 0 and os.path.exists(os.path.join(ck, "lifetime", "lifetime.json"))
    ref = {k: v.detach().clone() for k, v in st.model.state_dict().items()}
    st.archive.close()                                         # the full-lifetime restore reopens the source archive
    for cond, mom, exp in (("B", False, False), ("C", True, False), ("F", True, True)):
        fresh = S.build_stack(S.experiment_config(), run_dir=str(tmp_path / f"t{cond}"), seed=5, sem=sem,
                              experience=cond == "F")
        if cond == "F":
            fresh.agent.archive = fresh.agent.index = fresh.agent.replay = fresh.agent.consolidator = None
            fresh.model.retriever.detach_index()
        S.load_condition(fresh, ck, cond)
        rep = S.state_report(fresh)
        sd = fresh.model.state_dict()
        assert all(torch.equal(sd[k].cpu(), ref[k].cpu()) for k in ref), cond      # identical weights
        assert (rep["optimizer_state_tensors"] > 0) == mom, (cond, rep)
        assert rep["archive"] == exp and rep["replay"] == exp, (cond, rep)
        assert (rep["version"] > 0) == (cond == "F"), (cond, rep)
    a = S.build_stack(S.experiment_config(), experience=False, seed=9, sem=sem)
    S.load_condition(a, None, "A")
    assert not all(torch.equal(a.model.state_dict()[k].cpu(), ref[k].cpu()) for k in ref)
