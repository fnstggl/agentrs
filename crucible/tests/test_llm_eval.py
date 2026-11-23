"""The frozen-LLM baseline harness (`research.llm_eval`): what it measures is the numeric learners' measurement,
what it returns is complete, a shortened horizon is its own consistently normalised task, and a batch resumes.

No GPU: the language model is the scripted test double `EchoLLM` (never used for reported results) and the
competent control is the scripted operator `RigExperimenter`.
"""
import dataclasses
import json
import os

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.llm import EchoLLM, LLMPolicy  # noqa: E402
from crucible.core import discovery as D  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.research import llm_eval as L  # noqa: E402
from crucible.research import tasks as T  # noqa: E402
from crucible.research.envs import MechVecEnv  # noqa: E402
from crucible.research.learner import EnvSpec  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

SPEC = EnvSpec()
WAIT = '{"tool": "wait", "args": {}}'


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("crucible-cache")
    old = os.environ.get("CRUCIBLE_CACHE")
    os.environ["CRUCIBLE_CACHE"] = str(d)
    yield d
    if old is None:
        os.environ.pop("CRUCIBLE_CACHE", None)
    else:
        os.environ["CRUCIBLE_CACHE"] = old


@pytest.fixture(scope="module")
def task(cache):
    """One validated task (structural causal model engine)."""
    return T.build_tasks([("scm_additive", T.VAL_SEED0)])[0]


def _with_made_up_reference(family: str, seed: int, horizon: int) -> T.Task:
    """An unvalidated draw of another engine dressed with a MADE-UP reference (defaults 0, reference 5% of the
    demand income per tick), cut to `horizon` ticks. Validating an agent-based or technical mechanism costs a
    minute of CPU; for the tests that compare two measurement paths any fixed normalisation serves, because both
    paths apply the same one."""
    t = T.make_task(family, seed, validate=False)
    ref = 0.05 * t.income
    return dataclasses.replace(t, validated=True, horizon=horizon, default_net=0.0, reference_net=ref,
                               random_net=-ref, goal_net_cents=int(round(horizon * 0.5 * ref)))


def _levers(task):
    return task.sim(money=False).engine.levers(0)


def _settings(task, a: np.ndarray) -> dict:
    """Lever values of a normalised action in [-1, 1], by the arithmetic of `MechVecEnv.step` (and of the bridge)."""
    return {lv.name: float(lv.lo + (a[j] + 1.0) * 0.5 * (lv.hi - lv.lo)) for j, lv in enumerate(_levers(task))}


def _set(settings: dict) -> str:
    return json.dumps({"tool": "rig.set", "args": {"settings": settings}})


def _vec_record(task, actions: np.ndarray) -> dict:
    """The numeric learners' episode record for a sequence of actions (one row per tick)."""
    env = MechVecEnv([task], SPEC)
    env.reset()
    for a in actions:
        pad = np.zeros((1, SPEC.max_levers))
        pad[0, :len(a)] = a
        env.step(pad)
    assert env.done.all()
    return env.drain()[0]


def _llm(replies, **kw):
    kw = {"interface": "v4", "tool_chars": 40_000, "result_chars": 12_000, "history": 8, "history_chars": 1200,
          "last_action_line": False, **kw}
    return L.EvalLLMPolicy(backend=EchoLLM(replies), **kw)


# ====================================================================== the record
def test_record_carries_outcome_transcript_accounting_stamp_audit_and_diagnosis(task):
    short = dataclasses.replace(task, horizon=8)       # the full objective on 8 ticks: unreachable, which is fine here
    names = [lv.name for lv in _levers(task)]
    replies = ['{"tool": "rig.spec", "args": {}}',
               "I will look at the facility first.",                                   # no tool call: syntax
               '{"tool": "rig.sett", "args": {"settings": {}}}',                        # near name: discovery failure
               '{"tool": "rig.set", "args": {"levers": {"%s": 1}}}' % names[0],         # schema: wrong argument name
               '{"tool": "rig.set", "args": {"settings": {"nolever": 1}}}',             # the tool's own rule
               _set({names[0]: 1e9}),                                                   # accepted, clipped
               '{"tool": "rig.read", "args": {}}',
               '{"tool": "rig.hypothesize", "args": {"effects": [{"lever": "%s", "target": "cost", "sign": 1}]}}' % names[0],
               WAIT]
    pol = _llm(replies)
    rec = L.run_mechanism(short, pol, run="unit")
    assert rec["format"] == L.FORMAT and rec["kind"] == "mechanism" and rec["run"] == "unit"
    # ---- the goal the model was given is the public objective text and nothing else
    assert rec["goal_text"] == short.public()["objective"] and pol.goal == rec["goal_text"]
    first_prompt = pol.backend.seen[0][1]["content"]
    assert first_prompt.startswith("Goal: " + short.public()["objective"])
    for secret in (short.reference_net, short.default_net, short.random_net):
        assert f"{secret:.3f}"[:8] not in json.dumps(rec["task"]) and f"{secret:.3f}"[:8] not in first_prompt
    assert rec["task"]["shortened"] and rec["task"]["full_horizon_ticks"] == task.horizon
    # ---- runner: two turns per 3600 s tick (the stated default), exactly the horizon
    assert L.RUNNER_DT == 1800
    assert rec["runner"] == {"dt_s": 1800, "tick_s": 3600, "turns_per_tick": 2, "turns": 16, "seed": 0}
    # ---- outcome: the numeric environments' record, plus progress toward the objective
    vec = _vec_record(short, np.zeros((8, len(names))))
    assert set(rec["outcome"]) == (set(vec) - {"wall_s_batch"}) | {"progress"}
    assert rec["outcome"]["ticks"] == 8 and rec["outcome"]["success"] is False
    assert rec["outcome"]["net_cents"] == rec["outcome"]["revenue_cents"] - rec["outcome"]["cost_cents"]
    # ---- transcript: one record per turn with the model's text, the action, the complete result and its class
    tr = rec["transcript"]
    assert len(tr) == 16 and [r["turn"] for r in tr] == list(range(16)) and tr[3]["tick"] == 1
    assert tr[0]["text"] == replies[0] and tr[0]["result"]["ok"] and "levers" in tr[0]["result"]["result"]
    assert [r.get("class") for r in tr[:9]] == ["ok_read", "syntax", "unknown_tool", "schema", "rule", "ok_effect",
                                                "ok_read", "ok_effect", "wait"]
    assert tr[2]["sub"] == "near_name" and tr[1]["action"] is None and tr[1]["parse_error"]
    assert tr[5]["result"]["result"]["clipped"] == [names[0]]
    assert all(r["prompt_tokens"] > 0 for r in tr)
    json.dumps(rec)                                                    # the whole record is JSON
    # ---- accounting
    acc = rec["accounting"]
    assert acc["turns"] == acc["calls"] == 16 and acc["parse_failures"] == 1 and acc["waits"] == 8
    assert acc["prompt_tokens"] == sum(r["prompt_tokens"] for r in tr) and acc["simulated_s"] == 8 * 3600
    assert acc["wall_s"] >= acc["model_s"] >= 0
    # ---- the interface stamp is the world's, and the audit says what the prompt cut: nothing
    w = World.create(generate(*short.world_config()))
    me = w.system_call("admin.join", {"handle": "agent", "endowment": short.endowment})["principal"]
    assert rec["interface"]["stamp"] == D.interface_stamp(w, me) and rec["interface"]["stamp"]["tools"] == 40
    w.close()
    audit = rec["interface"]["audit"]
    assert audit["applies"] and audit["clean"] and audit["catalogue"]["level"] == "full"
    assert audit["prompts"]["truncated"] == 0 and audit["results"]["cut_as_last_result"] == 0
    assert audit["results"]["n"] == 7 and audit["generation"]["at_limit"] == 0
    # ---- diagnosis: every class counted once, the cause named, nothing guessed about difficulty
    d = rec["diagnosis"]
    assert d["format"] == "crucible-diagnosis/1" and "actions" not in d
    assert d["counts"] == {"turns": 16, "attempts": 8, "accepted": 4, "rejected": 3, "unparsed": 1, "waits": 8,
                           "no_action": 0, "distinct_tools": 5}
    assert {c: v["n"] for c, v in d["classes"].items()} == {"ok_effect": 2, "ok_read": 2, "wait": 8, "syntax": 1,
                                                            "unknown_tool": 1, "schema": 1, "rule": 1}
    assert d["primary"] == "syntax" or d["primary"].startswith(("unknown_tool", "schema", "rule"))
    assert d["context"]["status"] == "absent" and d["context"]["tool_list"]["level"] == "full"
    assert any(x.startswith("task_difficulty: not established") for x in d["not_determinable"])
    # ---- what it did with the facility, and its one filed claim scored against the hidden truth
    b = rec["behaviour"]
    assert b["read_spec"] and b["set_levers"] and b["filed_hypotheses"] and b["levers_moved"] == [names[0]]
    assert b["setting_changes"] == 1 and b["single_lever_changes"] == 1 and b["final_differs_from_default"]
    assert b["rig_calls_accepted"] == {"hypothesize": 1, "read": 1, "set": 1, "spec": 1}
    h = rec["hypotheses"]
    assert h["filings"] == 1 and h["n_claims"] == 1 and h["sign_accuracy"] in (0.0, 1.0) and h["n_pairs"] > 20
    assert rec["sim_check"] == {"applicable": True, "equal": True, "differences": {}} and rec["invariants"] == []


def test_audit_reports_what_a_small_budget_cut_and_eval_policy_equals_base_policy_at_default_widths(task):
    short = dataclasses.replace(task, horizon=3)
    replies = ['{"tool": "rig.spec", "args": {}}', '{"tool": "rig.read", "args": {}}', '{"tool": "rig.reed", "args": {}}',
               '{"tool": "rig.read", "args": {}}', '{"tool": "rig.history", "args": {"n": 3}}', WAIT]
    # a starved v4 prompt: the catalogue degrades (uniformly) and the specification does not fit its budget
    small = L.run_mechanism(short, _llm(replies, tool_chars=3000, result_chars=300, history_chars=240))
    a = small["interface"]["audit"]
    assert not a["clean"] and a["catalogue"]["level"] not in ("full",) and a["catalogue"]["chars"] <= 3000
    assert a["results"]["cut_as_last_result"] >= 2 and "rig.spec" in a["results"]["cut_as_last_result_by_tool"]
    assert a["results"]["cut_in_history"] >= a["results"]["cut_as_last_result"]
    # the 3.0 prompt: first-line descriptions, a fixed 600 characters for the last result, no error detail
    pol3 = LLMPolicy(backend=EchoLLM(replies), interface="v3", tool_chars=9000)
    old = L.run_mechanism(short, pol3)
    a3 = old["interface"]["audit"]
    assert a3["version"] == "v3" and a3["catalogue"]["first_line_descriptions"] and not a3["results"]["error_detail_shown"]
    assert a3["results"]["last_result_budget_chars"] == 600 and "rig.spec" in a3["results"]["cut_as_last_result_by_tool"]
    assert not a3["clean"] and old["policy"]["interface"] == "v3" and old["policy"]["history_chars"] == 240
    assert "bank.accounts" not in pol3.backend.seen[3][1]["content"].split("Tools (")[0]      # no near-name detail
    # the generous v4 prompt cuts nothing, and shows the near name of the misspelt tool
    pol = _llm(replies)
    full = L.run_mechanism(short, pol)
    assert full["interface"]["audit"]["clean"]
    assert "rig.read" in pol.backend.seen[3][1]["content"].split("Result of your last action:")[1].split("Tools (")[0]
    # with history_chars=1200 an earlier report is still whole two turns later; at the base width it is cut
    wide = pol.backend.seen[5][1]["content"].split("Your recent actions:")[1].split("Observation:")[0]
    base = L.EvalLLMPolicy(backend=EchoLLM(replies), interface="v4", tool_chars=40_000)
    L.run_mechanism(short, base)
    narrow = base.backend.seen[5][1]["content"].split("Your recent actions:")[1].split("Observation:")[0]
    assert '"shortfall_cents"' in wide and '"shortfall_cents"' not in narrow and "..." in narrow
    # with its default arguments the subclass builds the base class's prompt, byte for byte
    ref = LLMPolicy(backend=EchoLLM(replies), interface="v4", tool_chars=40_000)
    L.run_mechanism(short, ref)
    assert [m[1]["content"] for m in base.backend.seen] == [m[1]["content"] for m in ref.backend.seen]
    assert base.usage()["interface"]["history_chars"] == 240 and base.usage()["interface"]["last_action_line"] is True
    # the base class's observation says "last action: none" on every turn, also right after an action; the
    # corrected prompt says it only when the agent did nothing in its previous turn (turn 5 follows a wait)
    stale = [m[1]["content"].split("Tools (")[0] for m in ref.backend.seen]
    fixed = [m[1]["content"].split("Tools (")[0] for m in pol.backend.seen]
    assert all("last action: none" in x for x in stale) and "Result of your last action:" in stale[1]
    assert ["last action: none" in x for x in fixed] == [True, False, False, False, False, False]
    assert all("Result of your last action:" in x for x in fixed[1:])
    waited = _llm([WAIT, '{"tool": "rig.read", "args": {}}'])
    L.run_mechanism(short, waited)
    after_wait = waited.backend.seen[1][1]["content"].split("Tools (")[0]
    assert "last action: none" in after_wait and "Result of your last action:" not in after_wait
    assert full["policy"]["last_action_line"] is False and old["policy"]["last_action_line"] is True


def test_conduct_tells_interface_failure_from_what_the_agent_chose_to_do(task):
    """`conduct` names the kind of failure by fixed rules, including the one `diagnose` leaves unnamed: an agent
    whose calls are all valid and never change anything."""
    short = dataclasses.replace(task, horizon=6)
    names = [lv.name for lv in _levers(task)]
    read = '{"tool": "rig.read", "args": {}}'

    def kind(replies, **kw):
        rec = L.run_mechanism(short, _llm(replies, **kw))
        return rec, L.conduct(rec)
    # only ever reads: every call valid, no effect - the diagnosis has no name for it
    rec, c = kind([read])
    assert rec["diagnosis"]["primary"] == "not_determinable" and rec["diagnosis"]["families"]["valid"]["rate"] == 1.0
    assert c["kind"] == "passive" and c["effects"] == 0 and c["accepted_share"] == 1.0 and c["interface_share"] == 0.0
    assert not c["interface_confound"]
    # prose instead of tool calls, and misspelt tools: lost to the interface
    _, c = kind(["Let me think about the levers.", '{"tool": "rig.reed", "args": {}}'])
    assert c["kind"] == "interface" and c["interface_share"] == 1.0 and c["accepted_share"] == 0.0
    # well-formed calls the world refuses (a lever that does not exist): invalid actions, not the interface
    _, c = kind([_set({"nolever": 1.0})])
    assert c["kind"] == "invalid_actions" and c["largest"] == "rule" and c["interface_share"] == 0.0
    # the same accepted setting over and over
    _, c = kind([_set({names[0]: 1.0})])
    assert c["kind"] == "repetition" and c["effects"] > 0 and c["top_call_share"] == 1.0
    # reads, sets different levers, files a claim: valid, effective and varied - a strategic failure
    seq = ['{"tool": "rig.spec", "args": {}}', read] + [x for i in range(5) for x in (_set({names[i % len(names)]: float(i)}), read)]
    rec, c = kind(seq + [WAIT])
    assert c["kind"] == "strategic" and rec["diagnosis"]["strategy"]["valid_and_varied"]
    assert rec["diagnosis"]["primary"] == "strategic_failure"
    # a prompt that withheld something is flagged whatever the kind
    _, c = kind([read], tool_chars=3000)
    assert c["kind"] == "passive" and c["interface_confound"]
    row = L.summary_rows([rec])[0]
    assert row["conduct"] == "strategic" and row["interface_confound"] is False and row["effect_share"] > 0


def test_a_tool_call_inside_an_unfinished_thought_is_not_executed(task):
    """A thinking model whose generation budget ends inside ``<think>`` has not answered. The 3.0 parser would
    take a tool call quoted in the unfinished thought for the action; the evaluated policy does not."""
    short = dataclasses.replace(task, horizon=2)
    name = _levers(task)[0].name
    cut = '<think>\nMaybe I should try %s first, but I am not sure it' % _set({name: 1e9})
    done = '<think>\nI will read first, not %s.\n</think>\n\n{"tool": "rig.read", "args": {}}' % _set({name: 1e9})
    rec = L.run_mechanism(short, _llm([cut, done, WAIT]))
    tr = rec["transcript"]
    assert tr[0]["action"] is None and tr[0]["class"] == "syntax" and tr[0]["text"].endswith("</think>")
    assert tr[1]["action"] == {"tool": "rig.read", "args": {}} and rec["behaviour"]["set_levers"] is False
    gen = rec["interface"]["audit"]["generation"]
    assert gen["replies_with_thought"] == 2 and gen["thought_without_answer"] == 1 and gen["thought_without_answer_turns"] == [0]
    # the base class executes the call it found inside the unfinished thought
    base = L.run_mechanism(short, LLMPolicy(backend=EchoLLM([cut, WAIT]), interface="v4", tool_chars=40_000))
    assert base["transcript"][0]["action"]["tool"] == "rig.set" and base["behaviour"]["set_levers"] is True


# ====================================================================== the same measurement as the numeric table
@pytest.mark.parametrize("family, seed", [("scm_additive", None), ("abm_utility", 3), ("tech_pipeline", 3)])
def test_outcome_equals_mechvecenv_for_the_same_settings_on_every_engine(task, family, seed):
    """A tool-using policy and a numeric learner that apply the same settings tick by tick get the identical
    episode record: net, revenue, cost, objective quantity, fallback and idle ticks, final funds, the normalised
    score, both half-episode scores and success. (`abm` and `techsys` draws carry a made-up reference - see
    `_with_made_up_reference`; the equality does not depend on it.)"""
    t = dataclasses.replace(task, horizon=12) if seed is None else _with_made_up_reference(family, seed, 12)
    n = len(_levers(t))
    rng = np.random.default_rng(7)
    acts = rng.uniform(-1.0, 1.0, size=(t.horizon, n))
    acts[4] = acts[3]                                           # a tick without a change
    vec = _vec_record(t, acts)
    vec.pop("wall_s_batch", None)
    # one turn per tick: every turn sets the levers of the coming tick
    one = L.run_mechanism(t, _llm([_set(_settings(t, a)) for a in acts]), dt=3600)
    # two turns per tick (the default): read, set
    two = L.run_mechanism(t, _llm([x for a in acts for x in ('{"tool": "rig.read", "args": {}}', _set(_settings(t, a)))]))
    # three turns per tick: set, read, wait
    three = L.run_mechanism(t, _llm([x for a in acts for x in (_set(_settings(t, a)), '{"tool": "rig.read", "args": {}}', WAIT)]),
                            dt=1200)
    for rec in (one, two, three):
        out = dict(rec["outcome"])
        assert out.pop("progress") == min(1.0, max(0.0, out["score"] / T.GOAL_FRACTION))
        assert out == vec, {k: (out[k], vec[k]) for k in vec if out[k] != vec[k]}
        assert rec["sim_check"]["equal"] and rec["invariants"] == []
        assert [rec["settings_by_tick"][k][lv.name] for k in (0, 5) for lv in _levers(t)] == \
               [lv.clip(_settings(t, acts[k])[lv.name]) for k in (0, 5) for lv in _levers(t)]
    assert (one["runner"]["turns"], two["runner"]["turns"], three["runner"]["turns"]) == (12, 24, 36)
    assert vec["net_cents"] != 0.0 and vec["ticks"] == 12 and vec["engine"] == t.engine
    # a setting made in the last turn before a tick applies to it; one made after the tick does not
    late = L.run_mechanism(t, _llm([x for a in acts for x in (WAIT, WAIT, _set(_settings(t, a)))]), dt=1200)
    assert {k: v for k, v in late["outcome"].items() if k != "progress"} == vec


def test_money_rules_and_tampering_are_seen_by_the_measurement(task):
    """Fallback and idle ticks are counted like the numeric environments count them, and a run in which money
    moved outside the facility is not passed off as simulator-equal."""
    t = _with_made_up_reference("tech_pipeline", 3, 16)
    hi = {lv.name: float(lv.hi) for lv in _levers(t)}
    acts = np.ones((16, len(hi)))
    vec = _vec_record(t, acts)
    rec = L.run_mechanism(t, _llm([_set(hi), WAIT]), dt=3600)
    assert rec["outcome"]["ticks_fallback"] == vec["ticks_fallback"] and rec["outcome"]["ticks_idle"] == vec["ticks_idle"]
    assert rec["outcome"]["net_cents"] == vec["net_cents"] and rec["outcome"]["funds_end_cents"] == vec["funds_end_cents"]
    # the agent opens a second account and moves most of its balance there: the facility now sees less money
    replies = ['{"tool": "bank.open_account", "args": {"name": "side"}}',
               json.dumps({"tool": "bank.transfer", "args": {"from_account": "acct_000002", "to": "acct_000003",
                                                              "amount": int(t.endowment * 0.9)}}), _set(hi), WAIT]
    moved = L.run_mechanism(t, _llm(replies), dt=3600)
    assert [r["result"]["ok"] for r in moved["transcript"][:3]] == [True, True, True]
    assert moved["sim_check"]["applicable"] is False and moved["outcome"]["funds_end_cents"] < 0.2 * t.endowment


# ====================================================================== controls
def test_scripted_control_uses_only_facility_tools_learns_and_is_the_reference_of_the_diagnosis(task):
    ctl = L.RigExperimenter(horizon=task.horizon)
    rec = L.run_mechanism(task, ctl, run="control")
    idle = L.run_mechanism(task, L.IdlePolicy(), run="idle")
    rnd = L.run_mechanism(task, L.RandomRigPolicy(), run="random")
    # tool calls only, all accepted, nothing but the facility's own tools
    calls = rec["behaviour"]["calls"]
    assert set(calls) == {"rig.spec", "rig.read", "rig.set", "rig.history", "rig.hypothesize"}
    assert rec["diagnosis"]["counts"]["rejected"] == 0 and rec["interface"]["audit"] == {"applies": False}
    # a one-lever-at-a-time screen: every screening arm differs from the defaults in exactly one lever
    dflt = {lv.name: lv.default for lv in _levers(task)}
    screen = [a for a in ctl.arms if a["phase"] == "screen"]
    assert len(screen) >= len(dflt) and all(a["valid"] for a in screen)
    for a in screen:
        ran = rec["settings_by_tick"][a["first_tick"]]
        assert [k for k in dflt if ran[k] != dflt[k]] == [a["lever"]] and ran[a["lever"]] == a["value"]
    # then exploitation: the settings it ended with stayed for the rest of the horizon and paid
    o = rec["outcome"]
    assert rec["behaviour"]["final_settings_held_ticks"] >= 0.4 * task.horizon
    assert rec["settings_by_tick"][-1] == {**dflt, **ctl.best} and ctl.best != dflt
    assert o["score_second_half"] > 0.3 > abs(idle["outcome"]["score_second_half"]) and o["score"] > idle["outcome"]["score"]
    assert rnd["outcome"]["score"] < idle["outcome"]["score"] < o["score"]
    # the kernel world and the simulator agree on its episode, independently replayed here
    sim = task.sim()
    net = sum(float(r.revenue[0]) - float(r.cost[0]) for r in (sim.step(x) for x in rec["settings_by_tick"]))
    assert net == o["net_cents"] and rec["sim_check"]["equal"] and idle["sim_check"]["equal"] and rnd["sim_check"]["equal"]
    # the signs it filed are mostly right (scored by the private evaluator)
    h = rec["hypotheses"]
    assert h["effects_filed"] == len(ctl.claims) >= 0.8 * h["n_pairs"] and h["invalid"] == 0
    assert h["sign_accuracy"] > 0.75 and h["coverage"] > 0.5 and h["first_filed_at_tick"] <= task.horizon // 2 + 3
    # the random control changes every lever every tick; the idle control does nothing
    assert rnd["behaviour"]["setting_changes"] == task.horizon and rnd["behaviour"]["rig_calls_accepted"]["set"] == task.horizon
    assert idle["diagnosis"]["primary"] == "no_activity" and idle["behaviour"]["ticks_at_default"] == task.horizon
    # ---- as references of another run's diagnosis
    refs = [L.reference_of(rec, competent=True), L.reference_of(idle, competent=False), L.reference_of(rnd, competent=False)]
    assert refs[0]["name"] == "scripted_experimenter" and refs[0]["success"] == o["success"] and refs[0]["score"] == o["score"]
    names = [lv.name for lv in _levers(task)]
    agent = L.run_mechanism(dataclasses.replace(task, horizon=8),
                            _llm(['{"tool": "rig.spec", "args": {}}', _set({names[0]: 1.0}), '{"tool": "rig.read", "args": {}}',
                                  '{"tool": "rig.history", "args": {"n": 5}}', WAIT]), references=refs)
    graded = L.conduct(agent)
    assert graded["reference_score"] == o["score"] and graded["reference_succeeded"] == o["success"]
    assert graded["reference_progress"] == min(1.0, max(0.0, o["score"] / T.GOAL_FRACTION)) > graded["agent_progress"]
    diff = agent["diagnosis"]["difficulty"]
    assert [r["name"] for r in diff["references"]] == ["scripted_experimenter", "idle", "random_settings"]
    assert diff["status"] == ("absent" if o["success"] else "established")
    assert not any(x.startswith("task_difficulty: not established") for x in agent["diagnosis"]["not_determinable"])


def test_private_reference_played_through_the_tools_scores_about_one(task):
    sched = L.reference_schedule(task)
    rec = L.run_mechanism(task, L.SchedulePolicy(sched), run="reference")
    assert rec["policy"]["kind"] == "reference" and rec["behaviour"]["calls"] == {"rig.set": len(sched)}
    assert rec["outcome"]["success"] is True and 0.8 < rec["outcome"]["score"] < 1.2 and rec["sim_check"]["equal"]


# ====================================================================== shortened horizons
def test_shortened_horizon_is_renormalised_not_rescaled(task, cache):
    kw = dict(budget=48, n_random=24)
    short = L.shorten(task, 32, **kw)
    assert short.horizon == 32 and short.id == task.id and short.validated
    assert (short.mech_seed, short.world_seed, short.income, short.endowment) == \
           (task.mech_seed, task.world_seed, task.income, task.endowment)
    # the public objective is restated for 32 ticks at the same fraction of the re-estimated gap
    d, r = short.default_net, short.reference_net
    assert r > d and short.goal_net_cents == int(round(32 * (d + T.GOAL_FRACTION * (r - d))))
    assert "first 32 ticks" in short.public()["objective"] and str(short.goal_net_cents) in short.public()["objective"]
    assert short.goal_net_cents != int(round(task.goal_net_cents * 32 / task.horizon))       # not a rescaling
    assert (d, r) != (task.default_net, task.reference_net)
    # deterministic, cached, and the full horizon is returned unchanged
    assert L.shorten(task, 32, **kw) == short == L.shorten(task, 32, use_cache=False, **kw)
    assert L.shorten(task, task.horizon) is task
    with pytest.raises(ValueError):
        L.shorten(T.make_task("scm_additive", 5, validate=False), 32)
    # ---- the normalisation means on 32 ticks what it means on the full horizon: defaults ~0, the reference ~1
    idle = L.run_mechanism(short, L.IdlePolicy())
    ref = L.run_mechanism(short, L.SchedulePolicy(L.reference_schedule(short, budget=48)))
    assert idle["outcome"]["ticks"] == ref["outcome"]["ticks"] == 32
    assert abs(idle["outcome"]["score"]) < 0.35 and idle["outcome"]["success"] is False
    assert 0.7 < ref["outcome"]["score"] < 1.3 and ref["outcome"]["success"] is True
    assert idle["outcome"]["net_per_tick"] == idle["outcome"]["net_cents"] / 32
    assert idle["outcome"]["score"] == short.score(idle["outcome"]["net_cents"] / 32)
    # ---- and the short episode is the first 32 ticks of the same world, measured like the numeric table
    vec = _vec_record(short, np.asarray([[2.0 * (lv.default - lv.lo) / (lv.hi - lv.lo) - 1.0 for lv in _levers(task)]] * 32))
    assert vec["ticks"] == 32 and abs(vec["net_cents"] - idle["outcome"]["net_cents"]) <= 0.02 * abs(short.reference_net) * 32
    sim = task.sim()
    assert idle["outcome"]["net_cents"] == sum(float(x.revenue[0]) - float(x.cost[0]) for x in (sim.step(None) for _ in range(32)))
    assert idle["sim_check"]["equal"] and ref["sim_check"]["equal"]


# ====================================================================== resumable batches
def test_resuming_skips_finished_runs_and_reruns_a_changed_configuration(task, tmp_path):
    short = dataclasses.replace(task, horizon=4)
    calls = []

    def job(name, replies):
        def fn():
            calls.append(name)
            return L.run_mechanism(short, _llm(replies), run=name)
        return fn
    path = str(tmp_path / "ow_llm_unit.json")
    store = L.RunStore(path, prefix="ow_llm_unit_")
    cfg = {"model": "echo", "ticks": 4, "interface": "v4"}
    a = store.run("mech/a", job("a", ['{"tool": "rig.read", "args": {}}', WAIT]), cfg)
    b = store.run("mech/b", job("b", [WAIT]), cfg)
    assert calls == ["a", "b"] and store.executed == ["mech/a", "mech/b"] and not store.skipped
    # the JSON holds the records without transcripts; each transcript is a JSONL file, one turn per line
    data = json.load(open(path))
    assert data["format"] == L.FORMAT and set(data["runs"]) == {"mech/a", "mech/b"}
    assert "transcript" not in data["runs"]["mech/a"] and data["runs"]["mech/a"]["transcript_turns"] == 8
    lines = [json.loads(x) for x in open(store.transcript_path("mech/a"))]
    assert len(lines) == 8 and lines[0]["run"] == "mech/a" and lines[0]["action"]["tool"] == "rig.read"
    assert os.path.basename(store.transcript_path("mech/a")) == "ow_llm_unit_mech_a.jsonl" == a["transcript_file"]
    # the same batch again, in the same process and after a restart: nothing runs, the stored records come back
    assert store.run("mech/a", job("a", [WAIT]), cfg) == a
    again = L.RunStore(path, prefix="ow_llm_unit_")
    assert again.run("mech/a", job("a", [WAIT]), cfg) == a and again.run("mech/b", job("b", [WAIT]), cfg) == b
    assert calls == ["a", "b"] and again.skipped == ["mech/a", "mech/b"] and again.executed == []
    # a run that was never finished, and a finished run whose configuration changed, do run
    again.run("mech/c", job("c", [WAIT]), cfg)
    again.run("mech/a", job("a2", [WAIT]), {**cfg, "interface": "v3"})
    assert calls == ["a", "b", "c", "a2"] and again.get("mech/a")["behaviour"]["calls"] == {}
    assert set(json.load(open(path))["runs"]) == {"mech/a", "mech/b", "mech/c"}
    # two writers on one store (stages run side by side) never drop each other's runs
    one, two = L.RunStore(path, prefix="ow_llm_unit_"), L.RunStore(path, prefix="ow_llm_unit_")
    one.put("mech/x", {**a, "note": "from one"})
    two.put("mech/y", {**b, "note": "from two"})
    one.put("mech/z", {**a, "note": "from one again"})
    assert set(json.load(open(path))["runs"]) == {"mech/a", "mech/b", "mech/c", "mech/x", "mech/y", "mech/z"}
    assert two.run("mech/x", job("never", [WAIT]))["note"] == "from one" and calls == ["a", "b", "c", "a2"]
    again = L.RunStore(path, prefix="ow_llm_unit_")
    for extra in ("mech/x", "mech/y", "mech/z"):
        again.runs.pop(extra)
    # the stored records make the comparison table
    rows = L.summary_rows(again.runs.values())
    assert [r["run"] for r in rows] == ["mech/a", "mech/b", "mech/c"] and rows[0]["policy"] == "llm:echo-test-double"
    assert {"score", "success", "primary", "rate_valid", "rate_form", "loop_rate", "prompts_truncated", "catalogue",
            "prompt_tokens", "wall_s", "sim_equal", "setting_changes"} <= set(rows[0])
    table = L.render_table(rows, ["run", "score", "success", "primary", "catalogue"])
    assert table.count("\n") == 4 and "mech/b" in table and "| full" in table
    with pytest.raises(ValueError):
        open(str(tmp_path / "other.json"), "w").write('{"format": "something-else", "runs": {}}')
        L.RunStore(str(tmp_path / "other.json"))


# ====================================================================== (b) the open economy
def test_open_economy_run_is_verified_from_state_and_idle_is_no_activity():
    goal = L.open_economy_goal("ow_tiny", seed=1, overrides={"n_consumers": 300}, days=2, window_days=1,
                               revenue_cents=5000, requests=50, sustain_checks=1, endowment=1_000_000)
    assert goal.validate() == [] and "5000 cents" in goal.description and "50 successful requests" in goal.description
    for word in ("deploy", "repository", "listing", "category", "use.c", "cloud", "code", "market"):   # an outcome, not a recipe
        assert word not in goal.description
    replies = ['{"tool": "use.categories", "args": {}}',
               '{"tool": "id.create_org", "args": {"handle": "nova-labs", "name": "Nova Labs", "kind": "company"}}',
               '{"tool": "cloud.deploy", "args": {"repo": "nova-labs/x", "host": "novasvc", "instance": "inst_000099"}}',
               WAIT]
    pol = _llm(replies)
    rec = L.run_open_economy(goal, pol, run="open/unit", battery=0)
    assert rec["kind"] == "open_economy" and rec["goal_text"] == goal.description == pol.goal
    assert rec["runner"]["turns"] == 48 and len(rec["transcript"]) == 48 and rec["accounting"]["simulated_s"] == 2 * 86400
    o = rec["outcome"]
    assert o["success"] is False and o["progress"] == 0.0 and [c["day"] for c in o["daily_checks"]] == [1, 2]
    assert o["measures"] == {"artifact_creation": 0.0, "technical_correctness": 0.0, "real_usage": 0.0,
                             "customer_value": 0.0, "financial_viability": 0.0, "long_term_adoption": 0.0}
    e = rec["economy"]
    assert e["organisations"] == ["o:nova-labs"] and e["deployments"] == 0 and e["hosts"] == [] and e["commits"] == 0
    assert e["cash_start_cents"] == 1_000_000 and set(e["independent_revenue_cents"]) == {"p:nova", "o:nova-labs"}
    assert [r.get("class") for r in rec["transcript"][:4]] == ["ok_read", "ok_effect", "nonexistent", "wait"]
    assert rec["interface"]["stamp"]["tools"] > 100 and rec["interface"]["audit"]["catalogue"]["level"] == "full"
    assert rec["diagnosis"]["code"]["status"] in ("absent", "established") and rec["invariants"] == []
    assert rec["behaviour"]["calls_by_host"] == {"cloud": 1, "id": 1, "use": 1}
    idle = L.run_open_economy(goal, L.IdlePolicy(), run="open/idle", battery=0)
    assert idle["outcome"]["success"] is False and idle["diagnosis"]["primary"] == "no_activity"
    assert idle["economy"]["organisations"] == [] and idle["economy"]["cash_end_cents"] == 1_000_000
    rows = L.summary_rows([rec, idle])
    assert rows[0]["kind"] == "open_economy" and rows[0]["organisations"] == 1 and rows[1]["live_deployments"] == 0
    assert rows[1]["conduct"] == "no_activity" and rows[0]["conduct"] in ("strategic", "invalid_actions")
    json.dumps(rec)
