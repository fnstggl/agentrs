"""End-to-end observational learning on SocialWebLab (integration tests 1, 2, 3, 4, 5, 8).

A virtual-clock lab with simulated users runs for several simulated days while the sensor network
observes it through the public feed and public item pages, forecasts every new third-party action
before its outcome exists, follows it up on schedule and stores outcomes in the ledger. A small
randomly initialised temporal world model (scratch backbone) is trained on the matured outcomes.
"""

import hashlib

import numpy as np
import pytest
import torch

from agentrs.sensors.deduplication import stable_action_id
from agentrs.sensors.observation_network import NetworkConfig
from agentrs.sensors.socialweblab_source import lab_temporal_spec
from agentrs.socialweblab.engine import SocialWebLab
from agentrs.temporal.delayed_outcome_ledger import ActionRecord
from agentrs.temporal.outcome_labels import build_examples
from agentrs.temporal.system import build_network, run_lab
from agentrs.temporal.temporal_evaluation import evaluate_temporal, marginal_predict_fn
from agentrs.temporal.temporal_world_model import (
    TemporalWorldModel,
    TemporalWorldModelConfig,
)
from agentrs.temporal.time_encoding import DAY, HOUR
from agentrs.training.common import param_change, param_snapshot
from agentrs.training.temporal_world_model_trainer import (
    TemporalTrainConfig,
    TemporalTrainingDataBuilder,
    TemporalWorldModelTrainer,
)

HELD_OUT_SPACE = "databases"


def _split(o):
    if o.context_before.get("space") == HELD_OUT_SPACE:
        return "test_env"
    h = int(hashlib.md5((o.actor_ref or "").encode()).hexdigest(), 16) % 5
    return "test_actor" if h == 0 else "train"


@pytest.fixture(scope="module")
def world(tmp_path_factory, scratch_bb):
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("obs")
    lab = SocialWebLab(d / "lab.db", seed=11, n_users=50, lab_id="e2e11")
    model = TemporalWorldModel(TemporalWorldModelConfig(backbone=scratch_bb, n_heads=3, spec=lab_temporal_spec()))
    net = build_network(d / "net", lab, model, NetworkConfig(scheduling="fixed", updated_forecasts=True), split_fn=_split)
    run_lab(lab, net, 3.5 * DAY, step_s=900)
    yield {"lab": lab, "net": net, "model": model, "dir": d}
    lab.close()


def test_2_third_party_prediction_trains_real_weights_and_improves_held_out(world):
    lab, net, m = world["lab"], world["net"], world["model"]
    L = net.ledger
    third = L.actions("source_type='third_party'")
    assert len(third) > 80
    # every observed action was forecast before any of its follow-up observations, by a recorded checkpoint
    for a in third[:50]:
        fs = L.forecasts(a["action_id"])
        obs = L.observations(a["action_id"])
        assert fs and fs[0]["kind"] == "initial" and fs[0]["checkpoint"]
        if obs:
            assert fs[0]["created_time"] <= obs[0]["actual_time"]
    original = {a["action_id"]: L.forecasts(a["action_id"])[0]["prediction"] for a in third[:20]}
    b = TemporalTrainingDataBuilder(L, m.cfg.spec)
    now = {lab.time_base: lab.now()}
    train = b.build(as_of=now, splits=["train"])
    held = b.build(as_of=now, splits=["test_actor"], cutoffs=[0])
    assert len(train) > 100 and len(held) > 12
    T = TemporalWorldModelTrainer(m, TemporalTrainConfig(steps=160, batch_size=16, seed=0))
    ev = ["answered", "first_response", "positive_feedback", "removed"]
    before = T.evaluate(held, [HOUR, 6 * HOUR, DAY], ev)["overall"]
    snap = param_snapshot(m)
    r = T.train(train)
    assert param_change(m, snap) > 0 and r["last_loss"] < r["first_loss"]
    after = T.evaluate(held, [HOUR, 6 * HOUR, DAY], ev)["overall"]
    km = evaluate_temporal(marginal_predict_fn(train, ev, []), held, [HOUR, 6 * HOUR, DAY], ev)["overall"]
    world["trainer"] = T
    assert after["brier"] < before["brier"] - 0.01
    assert after["nll"] < km["nll"] + 0.02  # at least as good as the censoring-aware lookup table
    # predictions recorded earlier are unchanged by training (immutable forecasts)
    for aid, p in original.items():
        assert L.forecasts(aid)[0]["prediction"] == p


def test_8_generalises_to_held_out_actors_and_environment(world):
    lab, net, m = world["lab"], world["net"], world["model"]
    T = world.get("trainer")
    if T is None:
        pytest.skip("depends on test_2 training")
    b = TemporalTrainingDataBuilder(net.ledger, m.cfg.spec)
    now = {lab.time_base: lab.now()}
    train = b.build(as_of=now, splits=["train"], cutoffs=[0])
    for split in ("test_actor", "test_env"):
        held = b.build(as_of=now, splits=[split], cutoffs=[0])
        # no memorisation shortcut: score only held-out actions whose exact content never occurs in training
        seen = {e.complete_action for e in train}
        novel = [e for e in held if e.complete_action not in seen]
        assert len(novel) >= 0.7 * len(held)
        held = novel
        res = T.evaluate(held, [6 * HOUR, DAY], ["answered"])
        aur = [v["auroc"] for v in res["events"]["answered"].values() if v["auroc"] is not None]
        assert aur and float(np.mean(aur)) > 0.6, (split, res["events"]["answered"])
    actors_train = {e.action_id for e in train}
    assert actors_train


def test_3_multiple_actors_are_not_confused(world):
    lab, net = world["lab"], world["net"]
    L = net.ledger
    third = L.actions("source_type='third_party'")
    assert len({a["actor_ref"] for a in third}) >= 10
    checked = 0
    for a in third:
        item = int(a["context"]["object_ref"])
        truth = {e["id"]: e for e in lab.all_events(item) if e["public"]}
        author = lab.items("id=?", (item,))[0]["author"]
        for ev in L.events(a["action_id"]):
            t = truth.get(int(ev["provider_event_id"]))
            assert t is not None and t["item_id"] == item  # attached to the right action
            if ev["type"] in ("response_created", "answer_posted", "reaction_up"):
                assert bool(ev["by_original_actor"]) == (t["actor"] == author)
            checked += 1
    assert checked > 100


def test_4_changing_conditions_original_forecast_immutable_and_no_leakage(world):
    """A (simulated) user publishes a post, edits it later, then receives a reply — all through the
    application's service layer, observed only through the public feed and item pages."""
    lab, net, m = world["lab"], world["net"], world["model"]
    L = net.ledger
    users = [u for u in lab._q("SELECT id, role FROM users WHERE kind='simulated' AND role='regular'")]
    author, replier = users[0]["id"], users[1]["id"]
    iid = lab.create_item(author, "linux", "How do I fix ssh when cron hangs?", "How do I fix ssh when cron hangs? I tried.")
    lab.advance(300)
    net.step()
    aid = stable_action_id("swl-feed", str(iid))
    f0 = L.forecasts(aid)
    assert len(f0) == 1 and f0[0]["kind"] == "initial"
    original = f0[0]["prediction"]
    lab.advance(2 * HOUR)
    net.step()
    lab.edit_item(author, iid, "How do I fix ssh when cron hangs? Using version 2.1.4, error E1234 on line 7.")
    lab.advance(600)
    net.step()
    lab.advance(HOUR)
    rid = lab.respond(replier, iid, "Solution: restart the cron service.", kind="answer", helpful=True)
    assert rid
    lab.advance(4 * HOUR)
    net.step()
    evs = L.events(aid)
    t_edit = next(e for e in evs if e["type"] == "item_edited")
    t_reply = next(e for e in evs if e["type"] == "answer_posted")
    assert t_edit["by_original_actor"] and not t_reply["by_original_actor"]
    fs = L.forecasts(aid)
    # the original forecast is unchanged; its cutoff is when we first observed the post (feed lag)
    assert fs[0]["prediction"] == original and fs[0]["cutoff_elapsed"] <= 300.0 + 1e-6
    upd = [f for f in fs if f["kind"] == "updated"]
    assert upd and all(f["info_cutoff_time"] > fs[0]["info_cutoff_time"] for f in upd)
    obs = L.observations(aid)
    c_before = max(o["elapsed"] for o in obs if o["actual_time"] < t_edit["observed_time"])
    t0 = L.action(aid)["action_time"]
    c_mid = t_edit["observed_time"] - t0 + 1.0  # after the edit was observed, before the reply was observed
    assert t0 + c_mid < t_reply["observed_time"]
    exs = build_examples(L, aid, m.cfg.spec, cutoffs=[c_before, c_mid])
    assert not any("item_edited" in h for _, h in exs[0].history)  # nothing from the future leaks back
    assert any("item_edited" in h for _, h in exs[1].history)
    assert not any("answer_posted" in h for _, h in exs[1].history)
    assert exs[1].events["answered"].kind == "exact"  # still a target of the updated forecast
    p_upd = m.predict(net._query_for(L.action(aid), exs[1].cutoff_elapsed_s, exs[1].history, exs[1].known_now),
                      horizons=[DAY])
    assert 0.0 <= p_upd.horizons[0].event_prob["answered"] <= 1.0


def test_5_pending_outcomes_are_censored_not_failures(world):
    lab, net, m = world["lab"], world["net"], world["model"]
    L = net.ledger
    recent = [a for a in L.actions("source_type='third_party' AND status='open'") if lab.now() - a["action_time"] < 12 * HOUR]
    assert recent
    statuses = set()
    for a in recent:
        ex = build_examples(L, a["action_id"], m.cfg.spec)[0]
        for name, lab_ in ex.events.items():
            if lab_.kind == "right_censored":
                statuses.add(lab_.status)
                assert lab_.status in ("pending", "precluded")  # never a closed negative while the window is open
    assert "pending" in statuses


def test_1_delayed_consequences_of_our_own_action_are_linked(tmp_path, scratch_bb):
    """Our agent submits a patch: immediate effect (published, visible), delayed effect (CI result
    minutes later), longer-term effect (review approval / merge hours later). All three are linked to
    the original action and each horizon yields a trainable label."""
    lab = SocialWebLab(tmp_path / "lab.db", seed=4, n_users=40, spaces=["repo-core", "repo-ui", "linux"])
    model = TemporalWorldModel(TemporalWorldModelConfig(backbone=scratch_bb, n_heads=2, spec=lab_temporal_spec()))
    net = build_network(tmp_path / "net", lab, model, NetworkConfig(scheduling="event"))
    lab.advance(10 * HOUR)
    uid = lab.create_account("agent-a", "pw", spaces=["repo-core"], followers=5)
    content = ("Fix retry handling in scheduler\nFix retry when cache times out. Fixes issue #412 (error E2001 on line 88).\n"
               "```\ndef retry_cache(x, retries=3):\n    return handle(x, retries)\n```\n"
               "Adds def test_retry_cache(): assert retry_cache(1) == 1\nThanks!")
    rec = ActionRecord(source_id="self", source_type="self", actor_type="agent", time_base=lab.time_base,
                       action_time=lab.now(), observed_time=lab.now(), goal="get the retry fix merged",
                       context_text="ENV: socialweblab/codereview space=repo-core\nAUTHOR: followers=5",
                       context={"env": "socialweblab/codereview", "space": "repo-core"},
                       action_str=f'submit(submit_patch, "{content[:40]}")',
                       action={"operation": "submit_patch", "content": content}, policy_checkpoint="policy-v0")
    aid = net.register_self_action(rec)
    assert net.ledger.forecasts(aid)[0]["created_time"] <= lab.now()  # predicted before execution
    title, body = content.split("\n", 1)
    iid = lab.create_item(uid, "repo-core", title, body)
    net.link_self_action(aid, "swl-feed", str(iid))
    run_lab(lab, net, 3 * DAY, step_s=600)
    ev = {e["type"]: e for e in reversed(net.ledger.events(aid))}
    ex = build_examples(net.ledger, aid, model.cfg.spec)[0]
    assert ex.source_type == "self" and ex.goal == "get the retry fix merged"
    assert ex.states and ex.states[0].state == "visible"  # immediate: published and visible
    ci = ev.get("response_created")
    assert ci is not None and ci["env_event_time"] - rec.action_time < 2 * HOUR  # delayed: CI result
    later = [n for n in ("positive_feedback", "answered", "resolved") if ex.events[n].kind in ("exact", "interval")]
    assert later, ex.events  # longer-term: review approval / review / merge
    t_long = max(ex.events[n].b_s for n in later)
    assert t_long > ci["env_event_time"] - rec.action_time
    for o in net.ledger.observations(aid):
        assert o["action_id"] == aid
    loss, st = model.temporal_loss([ex])
    assert torch.isfinite(loss) and st["n_event_labels"] >= 3 and st["n_count_labels"] >= 3
    assert stable_action_id("swl-feed", str(iid)) not in {a["action_id"] for a in net.ledger.actions()}
    lab.close()
