"""CAM + frozen pretrained semantic foundation: the LM stays frozen and bit-identical, the trainable bridge
and CAM learn, PPO ratios are exact, and semantic runs train / checkpoint / resume."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from agentrs.cam.hybrid.semantic import HashingTextEncoder
from agentrs.cam.v2.collector import Collector
from agentrs.cam.v2.data import Replay
from agentrs.cam.v2.learner import OffPolicy, ppo_update, wm_step
from agentrs.cam.v2.model import CAM2, CAM2Config

TINY = dict(d=48, deter=64, stoch=4, classes=8, hidden=64, layers=1, heads=2, n_options=3, ens=3)


def _sem_model(dim=96):
    torch.manual_seed(0)
    return CAM2(CAM2Config(**TINY, obs_source="semantic", sem_dim=dim))


def test_semantic_cam_collects_from_text_and_trains_the_bridge_and_world_model():
    enc = HashingTextEncoder(96)
    m = _sem_model()
    col = Collector(m, 8, {"negotiation": 0.5, "product": 0.5}, "train", gen_seed=1, semantic=enc, store_feat=True)
    eps = col.collect(min_episodes=12)
    assert all(s.sem is not None and s.sem.shape == (96,) for e in eps for s in e.steps)
    assert all(e.goal_sem is not None and e.final_sem is not None for e in eps)
    op = OffPolicy(m, {"actor": "ppo"})
    g = m.groups()
    snap = {k: [p.detach().clone() for p in v] for k, v in g.items()}
    st = ppo_update(op, eps, {}, {"epochs": 1, "mb_episodes": 6, "target_kl": 10.0}, random.Random(0))
    assert st["mismatch"] < 1e-5  # stored actor inputs: recorded and re-scored log-probs are identical
    assert not any((p.detach() - s).abs().max() > 0 for p, s in zip(g["wm"], snap["wm"]))  # PPO never touches the WM
    rp = Replay(10_000, 1_000)
    rp.add(eps, persist=False)
    bridge0 = [p.detach().clone() for p in m.sem_obs.parameters()]
    wm_step(op, rp.sample(6, random.Random(0)), random.Random(0))
    assert any((p.detach() - b).abs().max() > 0 for p, b in zip(m.sem_obs.parameters(), bridge0))  # bridge learns


def test_offpolicy_learner_runs_on_semantic_inputs_with_hindsight():
    enc = HashingTextEncoder(96)
    m = _sem_model()
    eps = Collector(m, 8, {"bizdev": 0.5, "product": 0.5}, "train", gen_seed=2, semantic=enc).collect(min_episodes=16)
    rp = Replay(10_000, 1_000)
    rp.add(eps, persist=False)
    for e in eps:
        for h in e.hindsight:
            assert len(h) == 3 and h[2].shape == (96,)  # hindsight goals carry their own frozen goal vectors
    st = OffPolicy(m, {"actor": "awr"}).update(rp.sample(6, random.Random(0), her_p=1.0), {}, random.Random(0))
    assert np.isfinite(st["q_loss"]) and np.isfinite(st["wm_kl_dyn"])


def test_semantic_trainer_ppo_checkpoint_resume_and_final_controls(tmp_path):
    from agentrs.cam.v2.train import Trainer2, TrainConfig2

    cfg = TrainConfig2(run_dir=str(tmp_path / "r"), model=TINY, workers=8, steps_per_iter=120, max_iters=2, updates_per_iter=2,
                       batch_episodes=4, learn_start_steps=50, eval_every=100, eval_episodes=2, test_episodes=2, cf_states=1,
                       audit_episodes=20, learner={"actor": "ppo", "epochs": 1, "mb_episodes": 8}, final_reserve_h=0,
                       semantic={"model": "hashing-test", "dim": 96}, eval_at_interactions=[100], eval_heldout_wording=True,
                       final_reset_evals=True, final_nl_episodes=2, final_frontends=["parser"])
    t = Trainer2(cfg)
    t.train_iter()
    w = {k: v.clone() for k, v in t.model.state_dict().items()}
    t2 = Trainer2(cfg)
    assert t2.state["iter"] == 1 and all(torch.equal(w[k], v) for k, v in t2.model.state_dict().items())
    assert t2.run() == 0
    hf = t2.state["hybrid_final"]
    assert hf["semantic_model_unchanged"] and "reset_bridge" in {e["tag"].split("-")[0] for e in t2.state["evals"]} | {"reset_bridge"}
    tags = {e["tag"] for e in t2.state["evals"]}
    assert {"m100", "m100-heldout", "final-heldout", "reset_all-train", "reset_bridge-heldout"} <= tags


@pytest.mark.skipif(not __import__("pathlib").Path("/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B").exists(), reason="no local Qwen3-0.6B")
def test_real_frozen_qwen_gets_no_gradients_and_stays_bit_identical():
    from agentrs.cam.hybrid.semantic import FrozenSemanticEncoder

    enc = FrozenSemanticEncoder("Qwen/Qwen3-0.6B", (0.5, 1.0), device="cpu", batch=8)
    fp0 = enc.fingerprint()
    m = _sem_model(enc.dim)
    eps = Collector(m, 4, {"negotiation": 1.0}, "train", gen_seed=3, semantic=enc, store_feat=True).collect(min_episodes=4)
    rp = Replay(10_000, 1_000)
    rp.add(eps, persist=False)
    op = OffPolicy(m, {"actor": "awr"})
    op.update(rp.sample(4, random.Random(0), her_p=0.5), {}, random.Random(0))
    assert all(p.grad is None and not p.requires_grad for p in enc.model.parameters())
    assert enc.fingerprint() == fp0
    v1, v2 = enc.encode(["The supplier countered."]), enc.encode(["The supplier countered."])
    assert np.array_equal(v1, v2) and enc.stats["cache_hits"] >= 1


def test_llm_actor_scoring_matches_full_logits_reference():
    """Arm D scores options from hidden states at option positions only; it must equal full-logits scoring."""
    import pytest

    pytest.importorskip("transformers")
    from agentrs.cam.hybrid.semantic import FrozenLLMActor

    try:
        actor = FrozenLLMActor("Qwen/Qwen3-0.6B", device="cpu", dtype=torch.float32)
    except Exception as e:  # noqa: BLE001 - model not available offline
        pytest.skip(f"Qwen3-0.6B unavailable: {e}")
    prefix = "You are a buyer negotiating with a supplier.\\nObservation: the supplier offers N5 N3 N2 N1.\\nNext action:"
    options = ["ASK", "PROPOSE", "ACCEPT", "WALK"]
    got = actor._score(prefix, options)
    p_ids = actor.tok(prefix, add_special_tokens=False)["input_ids"]
    ref = []
    with torch.no_grad():
        for o in options:
            o_ids = actor.tok(" " + o, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([p_ids + o_ids])
            lp = torch.log_softmax(actor.model(input_ids=ids).logits.float(), -1)[0]
            ref.append(sum(float(lp[len(p_ids) - 1 + k, t]) for k, t in enumerate(o_ids)))
    assert np.allclose(got, ref, atol=1e-3), (got, ref)
