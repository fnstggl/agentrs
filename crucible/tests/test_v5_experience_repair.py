"""Workstream L (wave 3b repair pass): regression tests for the defects an independent verifier found on c5c/w3b-obs
at 507adb24 (R.4 sensitivity, seat controller labels, evaluation-script guard, ...)."""
import pytest

from crucible.core import evaluator_only as EO
from crucible.experience import leakage

pytestmark = pytest.mark.timeout(900)

LEGIT = ("amount", "memo", "title", "body")


class _OverHide(dict):
    """An injected defect: the evaluator-only registry hides legitimate payload fields of every event type."""
    def get(self, k, d=None):
        return tuple(dict.get(self, k, ())) + LEGIT


def _missing(res):
    return {r["test"]: int(r["metrics"].get("legitimate_fields_missing", 0)) for r in res
            if "legitimate_fields_missing" in r["metrics"]}


def test_the_reviewed_registry_covers_every_hidden_field():
    assert leakage.unreviewed_hidden_fields() == []
    for (kind, name), fields in leakage.REVIEWED_EVALUATOR_FIELDS.items():
        assert kind in ("event", "result", "summary") and fields and all(fields.values()), (kind, name)


def test_a_field_hidden_without_review_is_named(monkeypatch):
    monkeypatch.setitem(EO.EVENT_FIELDS, "bank.transfer", ("amount",))
    monkeypatch.setitem(EO.RESULT_FIELDS, "forum.post", ("title",))
    assert leakage.unreviewed_hidden_fields() == ["event:bank.transfer.amount", "result:forum.post.title"]


def test_over_hiding_legitimate_fields_is_counted_missing_by_every_legacy_comparison(monkeypatch):
    """The verifier's probe p6: with the registry wrongly hiding amount / memo / title / body, the Gym, PettingZoo,
    kernel-invariance and WorldEnv comparisons each count missing legitimate fields (before the fix only WorldEnv
    did: the legacy side was stripped of every registry field before the comparison)."""
    tests = (leakage.test_gym, leakage.test_pettingzoo, leakage.test_worldenv, leakage.test_kernel_invariance)
    base = _missing(leakage.run_battery(tests))
    assert all(v == 0 for v in base.values()), base
    monkeypatch.setattr(EO, "EVENT_FIELDS", _OverHide(EO.EVENT_FIELDS))
    over = _missing(leakage.run_battery(tests))
    assert all(over[t.__name__] > 0 for t in tests), over
    # a reviewed field (the controller label) is still normalised: hiding only it leaves nothing missing
    monkeypatch.setattr(EO, "EVENT_FIELDS", dict(EO.EVENT_FIELDS.items()))
    assert all(v == 0 for v in _missing(leakage.run_battery(tests)).values())


def _controller(world, p):
    rec = world.state.get("persons", p)
    return (rec.get("info") or {}).get("controller", "")


def test_controller_labels_reach_tier2_seats_or_raise():
    """The verifier's probe p11: make(2, 'crisis', 3, controllers='learner-X') left the seat's controller '' with no
    error (a seat is an existing principal). The label is now written into the seat configuration before the world
    is created; relabelling an existing principal raises; another seat sees no label in its observations."""
    from crucible.commerce.integrity import related_set
    from crucible.experience import make
    from crucible.experience.kernel import KernelExperienceEnv
    env = make(2, "crisis", 3, controllers="learner-X")
    a = env.agents()[0]
    assert _controller(env.world, a) == "learner-X"
    spec_seats = [m for m in env.world.spec.modules if m["name"] == "seat"][0]["config"]["seats"]
    assert all(s.get("controller") == "learner-X" for s in spec_seats)      # recorded in the spec: replay keeps it
    env2 = make(2, "crisis", 3, controllers={a: "lab-1"})                     # per seat, keyed by principal
    assert _controller(env2.world, env2.agents()[0]) == "lab-1"
    env3 = make(2, "crisis", 3)
    assert _controller(env3.world, env3.agents()[0]) == ""
    with pytest.raises(ValueError, match="controller label"):
        KernelExperienceEnv(env3.world, 3, agents=[a], controllers="someone-else")
    # a second participant of the same controller is related to the seat; nothing it observes names the label
    b = env.add_agent("second", controller="learner-X")
    assert b in related_set(env.world, a)
    assert "learner-X" not in leakage.dump([env.observe(b), env.observe_since(b, 0), env.observe(a)])


_EVALUATOR_SCRIPT = '''
import os, sys
sys.path.insert(0, {src!r})
open(os.path.join({marks!r}, f"ran-{{os.getpid()}}"), "w").close()      # top-level code outside any guard
HIDDEN = {{"grader_note": {secret!r}}}                                   # evaluator data built at import time
if __name__ == "__main__":                                              # only the evaluation call is guarded
    import contextlib, json
    from crucible.experience import remote
    if {control!r}:                               # positive control: the processes started as before the fix
        remote._without_main = contextlib.nullcontext
    out = remote.evaluate({{"tier": 1, "name": "rapid:net_routing"}}, {learner!r}, {{"hint": {secret!r}}})
    print(json.dumps({{"pid": os.getpid(), "status": out["learner_status"], "result": out["learner_result"],
                      "error": out["learner_error"], "processes": out["processes"]}}))
'''

_LEARNER = '''
import gc, os, sys
def learn(part, hint):
    a = part.agents()[0]
    part.observe(a)
    # dicts of atomic values are not tracked by the garbage collector: scan module globals as well
    pool = [o for o in gc.get_objects() if isinstance(o, dict)]
    pool += [v for m in list(sys.modules.values()) if m is not None for v in list(vars(m).values())
             if isinstance(v, dict)]
    found = [o for o in pool if any(isinstance(v, str) and v.startswith(hint) for v in list(o.values()))]
    return {"pid": os.getpid(), "found": len(found)}
'''


def test_an_unguarded_evaluator_script_never_runs_in_the_learner_process(tmp_path):
    """The verifier's nit: spawn re-imported the evaluator's __main__ in the learner (and world) process, so an
    evaluator script without a __main__ guard put the objects its top-level code built (a world, hidden values) in
    the learner's process, where a garbage-collector scan found them. The processes now start without the caller's
    __main__: the script's top-level code runs once (the evaluator) and the learner finds nothing."""
    import json
    import os
    import subprocess
    import sys
    import uuid
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    (tmp_path / "learner.py").write_text(_LEARNER)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=src)

    def run(control):
        marks = tmp_path / f"marks-{int(control)}"
        marks.mkdir()
        secret = f"SECRET-{uuid.uuid4().hex}"
        script = tmp_path / f"evaluate_me_{int(control)}.py"
        script.write_text(_EVALUATOR_SCRIPT.format(src=src, marks=str(marks), secret=secret, control=control,
                                                   learner=f"{tmp_path / 'learner.py'}:learn"))
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600, env=env)
        assert p.returncode == 0, p.stderr[-3000:]
        out = json.loads(p.stdout.strip().splitlines()[-1])
        assert out["status"] == "ok", out["error"]
        assert out["result"]["pid"] == out["processes"]["learner"] != out["pid"]
        return out, sorted(os.listdir(marks))

    out, ran = run(False)
    assert ran == [f"ran-{out['pid']}"]                     # neither the world nor the learner ran the script
    assert out["result"]["found"] == 0
    out, ran = run(True)                                    # control: the old start re-ran it in both children
    assert len(ran) == 3 and out["result"]["found"] >= 1


def test_every_domain_verdict_channel_is_probed_with_a_planted_verdict(monkeypatch):
    """The verifier found organization's verdict channel unprobed (its verdict never had a distinctive number). The
    Tier 2 probe now plants a canary and a distinctive number in the domain evaluator's verdict for every domain and
    looks for them in everything the seat reaches; positive control: an observation that forwards the evaluator's
    verdict is caught."""
    from crucible.experience import probes_tiers as PT
    from crucible.experience.kernel import KernelExperienceEnv
    from crucible.domains import base as DB
    name = "crisis:storm_radial"
    r = PT.probe_domain(names=[name], reference_days=1.0, max_reference_days=1.0)
    assert r["verdicts"]["planted"] == [name] and r["n_leaks"] == 0, r["leaks"][:3]
    assert r["timing"][name]["total_s"] > 0 and "judge" not in vars(DB.REGISTRY["crisis"])   # restored
    real = KernelExperienceEnv.observe

    def forwarding(self, agent):
        o = real(self, agent)
        if getattr(self, "domain", None):
            o = dict(o, status_note=(self.verify().get("evaluation") or {}).get(agent))
        return o
    monkeypatch.setattr(KernelExperienceEnv, "observe", forwarding)
    bad = PT.probe_domain(names=[name], reference_days=1.0, max_reference_days=1.0)
    assert any("verdict (planted)" in leak["what"] for leak in bad["leaks"]), bad["leaks"][:3]


def test_learners_set_the_fields_of_a_set_any_subset_object():
    """The rapid Tier 1 levers are a nested object whose fields are all optional ("set any subset"); both reference
    learners sent it empty, so they could never change a lever (the frozen neural copy's episode equalled doing
    nothing in des_inventory, mk_signal, net_routing, ...). Now they set such fields (the neural learner decides
    per field with a learned Bernoulli; the statistical one with p = 1/2, or every field at an arm level)."""
    from crucible.experience import make
    from crucible.experience.reference import ReferenceLearner
    from crucible.experience.statistical import ToolValueLearner
    env = make(1, "rapid:mk_signal", 3)
    a = env.agents()[0]
    tools = [t for t in env.tools(a) if t["name"] == "rapid.levers"]
    obs = env.observe(a)
    levers = sorted(tools[0]["inputSchema"]["properties"]["values"]["properties"])
    seen = set()
    for L in (ReferenceLearner(seed=1), ToolValueLearner(seed=1)):
        for _ in range(40):
            act = L.act(obs, tools)
            if act is not None:
                seen |= set(act["args"]["values"])
                r = env.execute(a, act)
                assert r["ok"], r.get("error")
    assert seen == set(levers)
    arms = ToolValueLearner(seed=1, arg_levels=5)
    acts = [arms.act(obs, tools) for _ in range(60)]
    vals = {json_dumps(x["args"]["values"]) for x in acts if x is not None}
    spec = tools[0]["inputSchema"]["properties"]["values"]["properties"]
    for v in vals:                                   # every lever at one common fraction of its range
        d = json_loads(v)
        assert set(d) == set(levers)
        fr = {round((d[k] - spec[k]["minimum"]) / (spec[k]["maximum"] - spec[k]["minimum"]), 1) for k in d}
        assert len(fr) == 1 and fr.pop() in (0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0)
    assert len(vals) >= 3


def test_arm_level_values_survive_a_checkpoint(tmp_path):
    from crucible.experience import make
    from crucible.experience.loop import run_learner
    from crucible.experience.statistical import ToolValueLearner
    env = make(1, "rapid:mk_signal", 2)
    L = ToolValueLearner(seed=3, arg_levels=5)
    run_learner(env, L, 120, update_every=16)
    assert any("@" in k for k in L.values)
    L.save(str(tmp_path / "tv.json"))
    M = ToolValueLearner(seed=0)
    M.load(str(tmp_path / "tv.json"))
    assert M.arg_levels == 5 and M.params_hash() == L.params_hash()


def json_dumps(x):
    import json
    return json.dumps(x, sort_keys=True)


def json_loads(x):
    import json
    return json.loads(x)


def test_sp_11_6c_earned_money_excludes_grants_endowments_and_related_parties():
    """SP-11.6c's kernel-tier outcome (the verifier: the crisis 'improvement' was a $60,000 grant held unspent):
    money paid to the learner by an unrelated participant counts; an endowment, a grant and a transfer from a
    participant of the same controller do not."""
    import importlib.util
    import os
    from crucible.experience import make
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5", "test_gate_SP_11_6c.py")
    spec = importlib.util.spec_from_file_location("gate_sp_11_6c", path)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    env = make(0, "tiny", 2, agents=["learner"], controllers={"learner": "lab"}, endowment=100_000,
               overrides={"n_consumers": 20})
    a = env.agents()[0]
    w = env.world
    other = w.system_call("admin.join", {"handle": "stranger", "endowment": 50_000})["principal"]
    twin = w.system_call("admin.join", {"handle": "twin", "endowment": 50_000, "controller": "lab"})["principal"]
    start = g._start(env)
    bank = w.modules["bank"]
    mine = bank.primary_account(w.state, a)

    def pay(src, amount):
        r = w.execute(src, {"tool": "bank.transfer", "args": {"from_account": bank.primary_account(w.state, src),
                                                              "to": a, "amount": amount, "memo": "x"}})
        assert r.ok, r.error
    pay(other, 1_234)
    pay(twin, 20_000)
    w.system_call("admin.endow", {"principal": a, "amount": 9_000})
    assert mine and g._earned_cents(env, start) == 1_234.0
