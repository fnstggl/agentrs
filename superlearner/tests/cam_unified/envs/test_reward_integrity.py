"""W2 reward-integrity and canary tests (plan 2026-11-26 §3 W2; audit 'observation-only reward mode' actions 1-3, 6):

(1) mode invariance - what the actor receives (reset / advance / observe batches, tools, results, goals) is identical
    in the three reward modes for a fixed action script, and observation_only gives a 0.0 reward stream;
(2) canaries planted in the task's private params, the evaluator's objective state and the label channel never
    reach any actor-side record (positive controls: each canary is present where it was planted);
(3) no evaluator-status / private-task key appears in actor records;
(6) nothing reachable from the factory adapter is an operator, evaluator, label source or world kernel;
plus the split guard (held-out families never enter a training stream) and the declared-mode rule.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

CRUCIBLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../crucible/src"))
if CRUCIBLE not in sys.path:
    sys.path.insert(0, CRUCIBLE)

from superlearner.cam_unified.envs import integrity as I  # noqa: E402
from superlearner.cam_unified.envs.factory import default_guard, make_env  # noqa: E402
from superlearner.cam_unified.envs.labels import LabelStore  # noqa: E402
from superlearner.cam_unified.envs.worldgen import HeldOutFamilyError  # noqa: E402
from superlearner.cam_unified.interfaces import Action  # noqa: E402

DAY = 86400.0
MODES = ("legacy", "verified", "observation_only")
SCRIPT = [("world.whoami", {}), ("world.hosts", {}), ("bank.balance", {}), ("market.inventory", {}),
          ("no.such_tool", {}), ("wait", {"seconds": 1800}), ("jobs.openings", {}), ("world.clock", {})]


def _norm(text: str) -> str:
    text = re.sub(r"ow\d+-[0-9a-f]{10}", "ENV", text)
    text = re.sub(r"x\d+-[0-9a-f]{10}", "ENV", text)
    return re.sub(r"g_[0-9a-f]{32}", "GOAL", text)


def _drive(ad, ev, script, days_per_step=600 / DAY):
    """The actor-side stream of a fixed script (fixed action ids, so provenance links compare across runs)."""
    stream = [ad.reset(), ad.available_tools(), ad.goals()]
    rewards = []
    for k, (tool, args) in enumerate(script):
        a = Action(tool, dict(args), action_id=f"a_script_{k:03d}")
        res = ad.execute([a] * 1)
        stream.append(res)
        stream.append(ad.advance_time([float(res[0].duration) + days_per_step]))
        stream.append(ad.available_tools())
        rewards.append(ev.take_rewards() if hasattr(ev, "take_rewards") else [0.0])
    stream.append(ad.observe())
    stream.append(ad.collect_events([0]))
    stream.append(ad.goals())
    return stream, rewards


def test_openworld_actor_stream_is_identical_in_all_reward_modes():
    dumps, rewards = {}, {}
    for m in MODES:
        es = make_env("openworld", tasks=["open_economy#100000"], isolated=False, reward_mode=m,
                      allow_reward_modes=(m,), seed=5)
        ad, op, ev = es
        try:
            stream, rw = _drive(ad, ev, SCRIPT)
        finally:
            es.close()
        dumps[m], rewards[m] = _norm(I.actor_dump(*stream)), rw
    assert len(dumps["observation_only"]) > 50_000
    assert dumps["legacy"] == dumps["verified"] == dumps["observation_only"]
    assert all(r == [0.0] for r in rewards["observation_only"])


def test_experience_actor_stream_is_identical_in_all_reward_modes():
    dumps = {}
    for m in MODES:
        es = make_env("rapid", ["dl_delayed_returns#4"], reward_mode=m, allow_reward_modes=(m,), salt="fixed",
                      compute_references=False)
        ad, op, src = es
        tool = ad.available_tools()[0][0].name
        script = [(tool, {"kind": "levers", "values": {}})] * 6 + [(tool, {"kind": "query", "target": "spec"})]
        stream, _ = _drive(ad, src, script, days_per_step=1 / 24)
        dumps[m] = _norm(I.actor_dump(*stream))
    assert dumps["legacy"] == dumps["verified"] == dumps["observation_only"]


def test_canaries_in_task_params_evaluator_state_and_labels_never_reach_the_actor(tmp_path):
    from superlearner.cam_unified.envs.harness import run_lifetime
    from superlearner.cam_unified.envs.protocols.ow1 import open_economy_task
    c_task, c_desc, c_state, c_status, c_label = (I.new_canary(t) for t in ("task", "taskattr", "state", "status",
                                                                           "label"))
    task = open_economy_task(seed=100_004, days=1)
    task.params = {**task.params, "canary": c_task}
    task.private_note = c_desc
    es = make_env("openworld", tasks=[task], isolated=False,
                  evaluator_kw={"objective_factory": I.canary_objective_factory(
                      {"state": c_state, "status": c_status, "label": c_label}, payout_bias=0.25)})
    ad, op, ev = es
    seen, labels, rewards = [], [], []

    class Recorder:
        def act(self, obs, tools):
            seen.extend([obs, tools])
            k = len(seen) // 2
            tool, args = SCRIPT[k % len(SCRIPT)]
            return [Action(tool, dict(args))]

        def observe(self, tr):
            seen.extend([tr.records, tr.next_obs])
            rewards.extend(tr.rewards)

        def update(self):
            return {}

    path = str(tmp_path / "transcript.jsonl")
    try:
        res = run_lifetime(Recorder(), ad, ev, think_s=3600, transcript=path,
                           on_step=lambda k, tr: labels.extend(ev.take_labels()))
        seen.append(ad.goals())
        # positive controls: every canary is where it was planted
        assert op.task(0).params["canary"] == c_task and op.task(0).private_note == c_desc
        assert ev.objectives[0].secret["note"] == c_state
        assert ev.evaluate_goal(0).details["canary"] == c_status
        assert len(labels) == 1 and c_label in json.dumps(labels[0].meta)
        assert ev.total[0] > 0.0                                    # the canary objective paid operator-side
        # negative: none of them reaches the actor (inputs, records, transcript, adapter goals)
        dump = I.actor_dump(*seen) + open(path).read()
        assert I.canary_hits(dump, [c_task, c_desc, c_state, c_status, c_label]) == []
        assert I.privileged_keys(seen[:2] + [ad.goals()]) == []
        assert all(r == 0.0 for r in rewards) and res.returns[0] == 0.0
        scan = I.scan_run_dir(str(tmp_path), [c_task, c_state, c_status, c_label])
        assert scan["files"] >= 1 and scan["hits"] == {}
        store_dir = tmp_path / "labels"
        LabelStore(str(store_dir)).append(labels[0])               # the label channel holds c_label ...
        assert I.scan_run_dir(str(tmp_path), [c_label], exclude=())["hits"]
        assert I.scan_run_dir(str(tmp_path), [c_label])["hits"] == {}   # ... which run-dir scans exclude
    finally:
        es.close()


def test_no_privileged_keys_in_experience_actor_records():
    es = make_env("rapid", ["hc_confounded#2", "dl_delayed_returns#1"], compute_references=False)
    ad, op, src = es
    tools = ad.available_tools()
    script = []
    for ts in tools:
        for t in ts:
            for v in (t.parameters or {}).get("properties", {}).get("target", {}).get("enum", []):
                script.append((t.name, {"kind": "query", "target": v}))
    stream = [ad.reset(), tools]
    for tool, args in script[:12]:
        acts = [Action(tool, dict(args)) if tool in {t.name for t in ts} else None for ts in tools]
        stream += [ad.execute(acts), ad.advance_time([1 / 24] * ad.B)]
    assert I.privileged_keys(stream) == []
    assert not [e for e in ad.env_ids if "hc_confounded" in e or "#" in e]


def test_factory_adapters_reach_no_operator_evaluator_or_label_source():
    from crucible.api import WorldServer
    from crucible.core.kernel import World
    from superlearner.cam_unified.envs.experience import ExperienceOperator, _World
    from superlearner.cam_unified.envs.openworld import OpenWorldOperator
    from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator
    from test_openworld import _walk                     # noqa: E402 - the existing attribute-graph walk
    forbidden = (World, WorldServer, OpenWorldOperator, OpenWorldEvaluator, ExperienceOperator, _World, LabelStore)
    es = make_env("openworld", tasks=["open_economy#100000"], isolated=False)
    try:
        ad = es.adapter
        ad.reset()
        ad.execute([Action("world.whoami", {})])
        ad.advance_time([0.01])
        assert not [x for x in _walk(ad) if isinstance(x, forbidden)]
        assert [x for x in _walk(es.labels) if isinstance(x, World)]          # negative control
        assert not {n for n in dir(ad) if not n.startswith("__")} & {"operator", "evaluator", "labels", "split",
                                                                   "take_labels", "take_rewards"}
    finally:
        es.close()
    es2 = make_env("rapid", ["dl_delayed_returns"], compute_references=False)
    ad2 = es2.adapter
    ad2.reset()
    assert not [x for x in _walk(ad2) if isinstance(x, forbidden)]
    assert not {n for n in dir(ad2) if not n.startswith("__")} & {"take_labels", "release_labels", "verify",
                                                                "private_verify", "worlds"}


@pytest.mark.parametrize("kind,spec", [
    ("rapid", "srch_distractors"),                       # a rapid val family (structure-fingerprint holdout)
    ("rapid", "dl_delayed_returns#1000000"),             # a training family at a validation / sealed seed
    ("rapid", "no_such_family"),
    ("mech", "scm_invu#3"),                              # mechanism val family
    ("mech", "scm_feedback#3"),                          # mechanism test family
    ("kernel", "ow_heldout#1"),
])
def test_split_guard_refuses_heldout_worlds_in_training_streams(kind, spec):
    with pytest.raises(HeldOutFamilyError):
        make_env(kind, [spec], compute_references=False)


@pytest.mark.parametrize("task", ["ow1:test:0", "open_economy#1000001"])
def test_split_guard_refuses_heldout_openworld_tasks(task):
    with pytest.raises(HeldOutFamilyError):
        make_env("openworld", tasks=[task], isolated=False)


def test_split_guard_honours_protocol_pair_declarations():
    from superlearner.cam_unified.envs.factory import protocol_pairs
    assert protocol_pairs()[("open_industry", 1)] == "test"     # a train family whose seed 1 a protocol holds out
    g = default_guard()
    with pytest.raises(HeldOutFamilyError):
        g.check("open_industry", 1)
    g.check("open_industry", 7)


def test_evaluation_streams_build_held_out_worlds_and_tag_their_labels():
    es = make_env("rapid", ["srch_distractors"], split="val", compute_references=False)
    assert es.split == "val" and es.guarded == 0 and es.operator.stream == "eval"
    es2 = make_env("rapid", ["dl_delayed_returns"], n_envs=3, compute_references=False)
    assert es2.guarded == 3 and es2.n_envs == 3 and len(set(es2.adapter.env_ids)) == 3


def test_reward_modes_other_than_observation_only_must_be_declared():
    with pytest.raises(ValueError):
        make_env("rapid", ["dl_delayed_returns"], reward_mode="legacy", compute_references=False)
    es = make_env("rapid", ["dl_delayed_returns"], reward_mode="legacy", allow_reward_modes=("legacy",),
                  compute_references=False)
    assert es.reward_mode == "legacy_shaped_reward"
    with pytest.raises(ValueError):
        make_env("openworld", tasks=["open_economy#100000"], reward_mode="verified")
    with pytest.raises(ValueError):
        make_env("rapid", ["mech:scm_additive"], compute_references=False)     # kind / spec mismatch


def test_scan_run_dir_reads_parquet_and_plain_files(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    c1, c2 = I.new_canary("pq"), I.new_canary("txt")
    pq.write_table(pa.table({"x": [c1]}), str(tmp_path / "chunk.lz4.parquet"), compression="lz4")
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "t.jsonl").write_text(json.dumps({"y": c2}))
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "labels.jsonl").write_text(c2)
    rep = I.scan_run_dir(str(tmp_path), [c1, c2, I.new_canary("absent")])
    assert set(rep["hits"]) == {c1, c2} and rep["files"] == 2
    assert all("labels" not in p for ps in rep["hits"].values() for p in ps)
