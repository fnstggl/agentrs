"""Inner-learner correctness on the in-process test fixture with the tiny random Qwen3 (plumbing, not results):
GAE terminal/truncation handling, exact behaviour log-probability replay, parameters change on optimiser steps,
evaluation never changes parameters, checkpoint save/resume is exact, retention mechanics, and an end-to-end
outer-learner search with lineage and resume."""
import json

import numpy as np
import pytest
import torch

from agentrs.cam2.backbone import load_llm
from agentrs.cam2.envs.base import TaskRef
from agentrs.cam2.learner import Episode, Transition, gae
from agentrs.cam2.model import param_checksum
from agentrs.cam2.trainer import RunConfig, Trainer

CAM_SMALL = {"d": 64, "integ_layers": 1, "heads": 4, "core_dim": 64, "wm_hidden": 64, "head_hidden": 64,
             "prefix_len": 4, "prefix_rank": 16, "mem_k": 4, "wm_ensemble": 2}


@pytest.fixture(scope="module")
def llm():
    return load_llm("tiny-random", device="cpu")


def _tr(**kw):
    t = Transition(step=None, h0={}, mem=None, mem_m=None, cand=0, logp_cand=0.0, delib=0, logp_delib=0.0, bias=None,
                   logits=torch.zeros(1), gen=None, logp_text=0.0, value=kw.pop("value", 0.5), f=torch.zeros(1))
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_gae_terminal_vs_truncated():
    term = Episode(tr=[_tr(reward=0.0), _tr(reward=1.0, terminated=True)], task=None)
    trunc = Episode(tr=[_tr(reward=0.0), _tr(reward=0.0, truncated=True, boot_value=2.0)], task=None)
    a1, r1, _ = gae(term, 0.9, 1.0)
    a2, r2, _ = gae(trunc, 0.9, 1.0)
    assert abs(r1[1] - 1.0) < 1e-6                      # terminal: no bootstrap
    assert abs(r2[1] - 0.9 * 2.0) < 1e-6                # truncated: bootstrap with V(final observation)
    assert abs(r1[0] - 0.9) < 1e-6 and abs(r2[0] - 0.81 * 2.0) < 1e-6


def _trainer(tmp_path, llm, name, pool, **kw):
    rc = RunConfig(run_dir=str(tmp_path / name), env="fixture", llm=llm.name, workers=pool.n, seed=kw.pop("seed", 0),
                   env_steps=kw.pop("env_steps", 120), max_steps=4, eval_every=0, ckpt_every=1, cam=dict(CAM_SMALL),
                   learn={"episodes_per_update": 16, "mb_episodes": 8, "lr": 3e-4, "warmup_steps": 1, **kw.pop("learn", {})},
                   plan=kw.pop("plan", {}), **kw)
    return Trainer(rc, llm=llm, pool=pool, device="cpu")


@pytest.fixture(scope="module")
def pool():
    from agentrs.cam2.envs.pool import EnvPool

    p = EnvPool("fixture", 4, {"max_steps": 4})
    yield p
    p.close()


def test_training_updates_parameters_and_replays_exactly(tmp_path, llm, pool):
    t = _trainer(tmp_path, llm, "a", pool, plan={"enabled": True, "min_reliability": 0.0},
                 learn={"prefix_replay_correction": True})
    c0 = param_checksum(t.cam)
    fp = llm.fingerprint()
    t.train()
    logs = [json.loads(x) for x in (tmp_path / "a" / "log.jsonl").read_text().splitlines()]
    ups = [r for r in logs if r["type"] == "update"]
    assert ups and all(r["params_changed"] > 0 and r["update_rel_norm"] > 0 for r in ups if r["opt_steps"] > 0)
    # behaviour log-probabilities (candidate + LLM text under CAM's control vectors) are reproduced by the learner
    assert max(r["replay_mismatch_max"] for r in ups) < 1e-3
    assert all(r.get("prefix_deltas", 0) >= 0 for r in ups) and any(r.get("prefix_deltas", 0) > 0 for r in ups)
    assert all(x.gen is None or "prefix" not in x.gen for e in t.replay for x in e.tr)   # dropped before replay
    assert any(r["llm_gen_tokens"] > 0 for r in ups)
    assert param_checksum(t.cam) != c0 and llm.fingerprint() == fp
    # evaluation: no parameter or memory change (asserted inside evaluate as well)
    c1, w1 = param_checksum(t.cam), t.mem.writes
    ev = t.evaluate([TaskRef("fixture", "type", 50_000 + i) for i in range(4)])
    assert ev["n"] == 4 and param_checksum(t.cam) == c1 and t.mem.writes == w1
    t.close()


def test_checkpoint_resume_is_exact(tmp_path, llm, pool):
    t = _trainer(tmp_path, llm, "b", pool, env_steps=60)
    t.train()
    ck = t.save("check")
    sd = {k: v.clone() for k, v in t.cam.state_dict().items()}
    opt_steps = t.learner.opt_steps_total
    t.close()
    t2 = _trainer(tmp_path, llm, "b", pool, env_steps=60)        # resumes from ckpt/latest.pt automatically
    assert all(torch.equal(sd[k], v) for k, v in t2.cam.state_dict().items())
    assert t2.learner.opt_steps_total == opt_steps and t2.counters["env_steps"] >= 60
    assert len(t2.replay) > 0 and t2.mem.n > 0
    t2.close()


def _kl_to_behaviour(t, eps) -> float:
    """Mean KL(stored behaviour policy || current policy) over every decision of ``eps`` (what the anchor targets)."""
    out = t.learner.forward_episodes(eps, need_text=False)
    kls = []
    with torch.no_grad():
        for s in out["steps"]:
            Nc = s["lp_all"].shape[1]
            beh = torch.log_softmax(s["beh"][:, :Nc].masked_fill(~s["cmask"], -1e9), -1)
            kls.append((beh.exp() * (beh - s["lp_all"]).masked_fill(~s["cmask"], 0.0)).sum(-1))
    return float(torch.cat(kls).mean())


def test_retention_replay_anchor_limits_forgetting(tmp_path, llm, pool):
    """Train on family A, then on family B with and without the replay behaviour anchor (same seed, same budget).
    The anchor must keep the policy on A's replayed decisions closer to the policy that generated them
    (mechanism test on the fixture, not a performance claim; performance retention is measured on MiniWoB)."""
    tA = _trainer(tmp_path, llm, "A", pool, env_steps=160, train_families=["press:3"], seed=1)
    tA.train()
    ref = tA.save("final")
    replayA = list(tA.replay)[-24:]
    tA.close()

    def phase_b(name, distill):
        t = _trainer(tmp_path, llm, name, pool, env_steps=240, train_families=["type"], seed=2, init_from=ref,
                     learn={"distill_coef": distill, "replay_episodes": 32})
        assert len(t.replay) > 0                       # A's experience travels with the weights
        t.train()
        kl = _kl_to_behaviour(t, replayA)
        t.close()
        return kl

    kl_free = phase_b("B0", 0.0)
    kl_anchor = phase_b("B1", 2.0)
    assert kl_anchor < kl_free, (kl_anchor, kl_free)


def test_outer_search_end_to_end_with_resume(tmp_path, llm, pool):
    from agentrs.cam2.search.outer import OuterLearner, SearchConfig

    base = {"arch": {**{k: v for k, v in CAM_SMALL.items()}, "d": 256, "core_dim": 512, "wm_hidden": 512,
                     "head_hidden": 256, "prefix_rank": 64}}
    cfg = SearchConfig(root=str(tmp_path / "search"), env="fixture", llm=llm.name, workers=pool.n, controller="adaptive",
                       operators=["learn.optim", "arch.depth", "rule.expr"], b0=40, eta=2, max_rung=1, base=base,
                       fixed_learn={"episodes_per_update": 16, "mb_episodes": 8, "warmup_steps": 1}, max_steps=4,
                       val_per_family=3, budget={"max_candidates": 4, "gpu_hours": 1.0, "wall_hours": 1.0}, check_device="cpu")
    ol = OuterLearner(cfg, llm=llm, pool=pool)
    ol.dev = "cpu"
    res = ol.run()
    pop = ol.pop
    evald = [c for c in pop.cands.values() if c.rung >= 0]
    assert len(pop.order) == 4 and len(evald) >= 3
    assert len({c.op for c in pop.cands.values() if c.parent}) >= 2 or len(pop.hypotheses) >= 3
    assert all(c.history[0][0] == "seed" for c in pop.cands.values())
    assert any(c.rung == 1 for c in pop.cands.values())          # ASHA promoted at least one candidate
    assert res["champion"] in pop.cands and (tmp_path / "search" / "population.json").exists()
    # resume reconstructs the same population and does not redo finished work
    ol2 = OuterLearner(cfg, llm=llm, pool=pool)
    assert ol2.pop.order == pop.order and ol2.pop.hypotheses == json.loads(json.dumps(pop.hypotheses, default=str))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_checkpoint_resume_on_gpu(tmp_path, pool):
    """Resume with CUDA tensors (map_location=cuda moves the saved RNG states to the GPU; they must be restored)."""
    gllm = load_llm("tiny-random", device="cuda")
    rc = RunConfig(run_dir=str(tmp_path / "g"), env="fixture", llm=gllm.name, workers=pool.n, env_steps=40, max_steps=4,
                   eval_every=0, ckpt_every=1, cam=dict(CAM_SMALL), learn={"episodes_per_update": 8, "mb_episodes": 8})
    t = Trainer(rc, llm=gllm, pool=pool, device="cuda")
    t.train()
    sd = {k: v.clone() for k, v in t.cam.state_dict().items()}
    t.close()
    rc.env_steps = 80
    t2 = Trainer(rc, llm=gllm, pool=pool, device="cuda")          # resumes from latest.pt
    assert all(torch.equal(sd[k], v) for k, v in t2.cam.state_dict().items())
    t2.train()                                                    # continues to the larger budget
    assert t2.counters["env_steps"] >= 80
    t2.close()
