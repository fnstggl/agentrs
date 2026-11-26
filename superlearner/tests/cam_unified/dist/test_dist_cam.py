"""The integrated CAM Unified model in the distributed stack on the real Crucible OpenWorld (manifest A16 / F02 / F03).

The in-process tests drive the real pipeline — `ActorProcess` (CamActorPolicy: the full CamUnifiedModel built from
workstreams A-E with Qwen3-0.6B, tiny core dimensions) on its own process-isolated ow_tiny world -> durable queue ->
`CentralLearner` (CamLearnerModel: D's CombinedTrainer + ActorCriticObjective) -> atomic weights-only publication ->
the other actor's refresh — and check what would differ if the capability were fake: which actor's decisions an
update consumed, that the learner recomputes the recorded behaviour log-probabilities exactly at lag 0, that lagged
data goes through V-trace, that weights versions reach the actors' records, and that the latent state is rebuilt
under new weights. They need a CUDA GPU and the cached Qwen3-0.6B (marker ``gpu``). The multiprocess launcher run is
``slow``.
"""
from __future__ import annotations

import json
import os
import random

import pytest
import torch

from superlearner.cam_unified.dist.actor import ActorProcess, ActorSpec, merge_by_seq, run_paths
from superlearner.cam_unified.dist.cam_learner import rho_stats
from superlearner.cam_unified.dist.cam_policy import (_registry_dict, _registry_from, inputs_bytes, inputs_from_bytes,
                                                      tools_from_payload, tools_payload)
from superlearner.cam_unified.interfaces import EntityRegistry, Observation, ObsKind, ToolSchema

TINY = dict(n_layers=2, d_model=64, n_heads=4, n_state=8, d_particle=32, n_particles=8, bayes_feat=8, stoch_groups=8,
            stoch_classes=8, mem_key_dim=32, reason_layers=1, max_ctx_tokens=64, n_prefix=4, max_entities=32,
            n_options=4)
POLICY = "superlearner.cam_unified.dist.cam_policy:make_cam_policy"
ENV = "superlearner.cam_unified.dist.cam_env:make_openworld_env"


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:                                    # noqa: BLE001
        pytest.skip("Qwen3-0.6B is not in the local HF cache")


# ============================================================================ cheap unit tests (no model)

def test_tool_payload_preserves_schema_key_order():
    """The decoder prompt renders a schema in its own key order: a sorted copy is a different prompt (the replay of
    a sampled action is then refused). The stored tool list must round-trip key order exactly."""
    params = {"type": "object", "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer"}},
              "required": ["zeta"], "additionalProperties": False}
    tools = [ToolSchema("x.do", "Do x. Costs 5 cents.", params, duration_hint=0.01)]
    back = tools_from_payload(tools_payload(tools))
    assert list(back[0].parameters) == list(params)
    assert list(back[0].parameters["properties"]) == ["zeta", "alpha"]
    assert json.dumps(back[0].parameters) == json.dumps(params)
    assert back[0].key == tools[0].key and back[0].duration_hint == 0.01


def test_registry_round_trip_reproduces_slots_and_codes():
    r = EntityRegistry(8, seed=1234)
    for i, e in enumerate(["p:a", "o:b", "acct_1", "p:c"]):
        r.bind(e, float(i), "person")
    d = inputs_from_bytes(inputs_bytes({"format": "cam-step-inputs/1", "r": _registry_dict(r)}))["r"]
    q = _registry_from(d)
    assert q.slot == r.slot and q.free == r.free and q.last_seen == r.last_seen and q.etype == r.etype
    assert q.rng.getstate() == r.rng.getstate()           # the tokenizer's slot codes are seeded from this state
    assert q.bind("p:new", 9.0) == r.bind("p:new", 9.0)    # identical future bindings


def test_merge_by_seq_delivers_each_event_once():
    o = lambda s: Observation(ObsKind.EVENT, content=str(s), seq=s)        # noqa: E731
    res = [Observation(ObsKind.TOOL_RESULT, content="{}")]
    merged, top = merge_by_seq(5, res, [o(4), o(6), o(7)], [o(6), o(7), o(8)])
    assert [x.seq for x in merged] == [None, 6, 7, 8] and top == 8


def test_rho_stats_truncation_and_ess():
    s = rho_stats(torch.zeros(10))
    assert s["rho_mean"] == 1.0 and s["ess_frac"] == pytest.approx(1.0) and s["clipped_frac"] == 0.0
    lr = torch.tensor([0.0, 0.0, -2.0, 3.0])
    s = rho_stats(lr)
    assert s["clipped_frac"] == 0.25 and s["below_half_frac"] == 0.25
    assert s["rho_mean"] == pytest.approx((1 + 1 + float(torch.tensor(-2.0).exp()) + 1) / 4)
    assert 0 < s["ess_frac"] < 1


def test_openworld_actor_env_uses_dev_seeds_and_verified_rewards():
    """Each actor's world is a real process-isolated ow_tiny world; world seeds come from the dev range only."""
    from superlearner.cam_unified.dist.cam_env import OpenWorldActorEnv
    with pytest.raises(ValueError):
        OpenWorldActorEnv(seed=0, seed_base=200_000)                        # confirmation range refused
    env = OpenWorldActorEnv(seed=7, days=2)
    try:
        ob = env.reset()
        ob2 = env.reset()
        assert all(100_000 <= s < 200_000 for s in env.world_seeds) and env.world_seeds[0] != env.world_seeds[1]
        assert ob.env_ids != ob2.env_ids
        assert env.operator.pid != os.getpid()                              # the worlds live in another process
        tools = env.available_tools()[0]
        assert len(tools) > 100 and any(t.name == "wait" for t in tools)
        assert env.take_rewards() == [0.0]
        assert env.initial_cursors()[0] >= 0
        # nothing in the agent-side records names the world family or seed
        blob = json.dumps([o.to_dict() for o in ob2.observations[0]])
        assert "ow_tiny" not in blob and str(env.world_seeds[1]) not in blob
    finally:
        env.close()


# ============================================================================ the integrated pipeline (GPU)

def _actor(run, aid, seed, n, **kw):
    spec = ActorSpec(actor_id=aid, run_dir=run, env_factory=ENV, env_kwargs={"days": 8}, policy_factory=POLICY,
                     policy_kwargs={"cfg": TINY, "rebuild_window": 8}, seed=seed, refresh_at="decision",
                     max_decisions=n, cuda=True, batch_records=4, **kw)
    return ActorProcess(spec)


def _steps(L, k=12):
    out = []
    for _ in range(k):
        r = L.step()
        if r is not None:
            out.append(r)
    return out


@pytest.mark.gpu
def test_cam_actors_feed_one_learner_exactly_and_versions_propagate(tmp_path):
    _gpu()
    from superlearner.cam_unified.dist.cam_learner import CamLearnerModel
    from superlearner.cam_unified.dist.learner import CentralLearner
    from superlearner.cam_unified.dist.sinks import JsonlArchive
    run = str(tmp_path / "run")
    model = CamLearnerModel(run_dir=run, cfg=TINY, seed=0, recompute_state="always", fsync=False)
    L = CentralLearner(run, model, sink=JsonlArchive(run_paths(run)["archive"], fsync=False), max_lag=8,
                       min_decisions=6, max_batch_decisions=6, segment_len=6, burn_in=2, settle=1, fsync=False,
                       queue_kwargs={"fsync": False})
    p0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # ---- actor0 acts under v0; the learner trains on its decisions only
    a0 = _actor(run, "actor0", 1, 8)
    a0.run()
    ups = _steps(L)
    assert ups, "no update from actor0's experience"
    u = ups[0]
    assert set(u["by_actor"]) == {"actor0"} and u["n_decisions"] >= 5
    m = u["metrics"]
    assert m["n_policy_trained"] == u["n_decisions"] and not m.get("model_errors")
    assert m["n_fresh_steps"] == u["n_decisions"] and m["n_stale_steps"] == 0       # lag 0 -> PPO rows
    assert m["lag0_logp_dev_max"] < 1e-3                       # recorded behaviour logp recomputed exactly
    assert m["z_recompute_dev_max"] < 1e-4                     # latent state re-filtered exactly under v0
    assert {"world", "adapt", "actor", "value"} <= set(m["losses"])
    assert L.version >= 1
    sd = model.state_dict()
    assert any(not torch.equal(p0[k], sd[k]) for k in sd if k.startswith("core."))
    assert any(not torch.equal(p0[k], sd[k]) for k in sd if k.startswith("action_generator."))
    assert len(sd) < len(model.model.state_dict())             # shared backbone published once
    v_after_a0 = L.version
    # ---- actor1 starts after the update: it acts with the weights actor0's experience produced
    a1 = _actor(run, "actor1", 2, 8)
    a1.run()
    assert a1.policy.version == v_after_a0 and a1.stats["versions_used"][0] == v_after_a0
    ups1 = _steps(L)
    assert any("actor1" in x["by_actor"] for x in ups1)
    rec1 = [r for ch in L.sink.scan(actor_id="actor1") for r in ch if r.action is not None]
    assert rec1 and all(int(r.model_version) == v_after_a0 for r in rec1)
    assert all(r.extra.get("inputs_ref") and r.extra.get("tools_ref") for r in rec1)
    m1 = [x["metrics"] for x in ups1 if "actor1" in x["by_actor"]][0]
    assert m1["lag0_logp_dev_max"] is not None and m1["lag0_logp_dev_max"] < 1e-3
    # durable index: both actors' decisions were trained
    trained = {r[0] for r in L.db.execute("SELECT event_id FROM consumed WHERE status='trained'")}
    ids0 = {r.event_id for ch in L.sink.scan(actor_id="actor0") for r in ch if r.action is not None}
    ids1 = {r.event_id for r in rec1}
    assert trained & ids0 and trained & ids1
    # ---- stale behaviour: actor0 keeps acting with its old weights -> lag > 0 -> V-trace, never PPO
    a0.client.maybe_refresh = lambda policy, boundary: False
    a0.spec.max_decisions = a0.stats["decisions"] + 8
    v_a0 = a0.policy.version
    a0.run()
    ups2 = _steps(L)
    stale = [x for x in ups2 if any(int(k) > 0 for k in x["lag"]["lag_hist"])]
    assert stale, "no lagged decision was trained"
    ms = stale[0]["metrics"]
    assert ms["n_stale_steps"] > 0 and ms["vtrace"]["n"] > 0
    lags = [int(k) for k in stale[0]["lag"]["lag_hist"]]
    assert max(lags) >= stale[0]["lag"]["learner_version"] - v_a0 > 0     # actor0's decisions were made under v_a0
    L.close()


@pytest.mark.gpu
def test_decision_boundary_refresh_rebuilds_latent_state_under_new_weights(tmp_path):
    """After a refresh whose drift exceeds the threshold the actor reconstructs Z from the initial state over its
    stored raw inputs under the new weights (not the stale Z), and later records say the state was built under the
    new version."""
    _gpu()
    from superlearner.cam_unified.dist.cam_policy import CamActorPolicy, unique_state_dict, unpack_step_inputs
    from superlearner.cam_unified.dist.publish import CheckpointPublisher
    from superlearner.cam_unified.model import StepInputs, acting_mode
    run = str(tmp_path / "run")
    pol0 = CamActorPolicy(seed=0, run_dir=run, cfg=TINY, fsync=False)
    pub = CheckpointPublisher(run_paths(run)["ckpt"], fsync=False)
    sd0 = {k: v.detach().cpu().clone() for k, v in unique_state_dict(pol0.model).items()}
    pub.publish(sd0, 0)
    del pol0
    torch.cuda.empty_cache()
    a = _actor(run, "actor0", 3, 6, rebuild_threshold=1e-9)
    a.client.wait_for_version(a.policy, 0, timeout_s=5)
    # run 6 decisions under v0 (in a thread-free way: the actor's own loop)
    a.run()
    pol = a.policy
    assert pol.counters["decisions"] == 6
    win = [list(w) for w in pol._windows]
    # publish v1 = perturbed weights and let the actor refresh at a decision boundary
    g = torch.Generator().manual_seed(0)
    sd1 = {k: (v + 0.05 * torch.randn(v.shape, generator=g) if v.is_floating_point() else v) for k, v in sd0.items()}
    pub.publish(sd1, 1)
    Z_stale = pol.agent.state.Z.clone()
    a._refresh_decision([[]])
    assert a.client.version == 1 and a.stats["rebuilds"] == 1 and a.client.state_version == 1
    m = pol.model
    # the rebuilt Z equals Z recomputed from the initial state over the stored window under v1
    with torch.no_grad(), acting_mode(m), m._autocast():
        z = m.core.init_state(1, device=m.device).float()
        for packed, obs, tref in win[0]:
            inp = unpack_step_inputs(packed, obs, pol.store.tools(tref), m.device, Z_prev=z)
            z = m.encode_step(StepInputs.cat([inp])).Z.float()
    assert torch.allclose(pol.agent.state.Z[0].float(), z[0], atol=1e-5)
    assert not torch.allclose(pol.agent.state.Z[0].float(), Z_stale[0].float(), atol=1e-3)
    assert pol.counters["rebuild_decisions"] == len(win[0])


@pytest.mark.gpu
@pytest.mark.slow
def test_multiprocess_cam_run_two_actors_share_one_learner(tmp_path):
    """Real processes: learner + 2 actors (each with its own GPU model and isolated world) under the launcher."""
    _gpu()
    from superlearner.cam_unified.dist.cam_runs import cam_run
    ev = cam_run(str(tmp_path), 2, 150.0, label="mp", cfg=TINY, min_decisions=6, segment_len=6, burn_in=2,
                 max_lag=8, stop_timeout_s=300)
    assert ev["checks"]["every_actor_trained"], ev["per_actor"]
    assert ev["checks"]["every_actor_acted_with_updated_weights"], ev["per_actor"]
    assert ev["throughput"]["updates"] >= 2
    for aid, a in ev["per_actor"].items():
        assert a["trained_decisions_durable"] > 0 and (a["max_version_acted"] or 0) >= 1
