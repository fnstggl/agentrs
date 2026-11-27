"""cam_live.eval.openended: the substantive families' independent verifiers (no agent; fast)."""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

from cam_live.contracts import GoalStatus, OutcomeQuery, OutcomeStatus, SimClock
from cam_live.eval.openended import (SW_PROBLEMS, HeldOutVerifier, HiddenTestVerifier, MechanismService, mech_task,
                                     run_hidden, sw_task)

REF = {
    "rle": "def solve(s):\n    out=[];i=0\n    while i<len(s):\n        j=i\n        while j<len(s) and s[j]==s[i]:\n"
           "            j+=1\n        n=j-i\n        out.append(f'{n}{s[i]}' if n>1 else s[i])\n        i=j\n"
           "    return ''.join(out)\n",
    "base": "def solve(n, b):\n    d='0123456789abcdefghijklmnopqrstuvwxyz'\n    if n == 0:\n        return '0'\n"
            "    neg, n, o = n < 0, abs(n), ''\n    while n:\n        o = d[n % b] + o\n        n //= b\n"
            "    return ('-' if neg else '') + o\n",
}


@pytest.mark.parametrize("problem", ["rle", "base"])
def test_hidden_tests_accept_a_correct_program_and_reject_others(tmp_path, problem):
    clock = SimClock()
    v = HiddenTestVerifier(str(tmp_path), problem, 3, clock)
    task = sw_task(problem, "o", max_decisions=5)
    assert v.verify(task).status == GoalStatus.PENDING and v.score()["missing"]
    (tmp_path / "solution.py").write_text("def solve(*a):\n    return None\n")
    assert v.verify(task).value < 0.2
    (tmp_path / "solution.py").write_text(REF[problem])
    out = v.verify(task)
    assert out.status == GoalStatus.ACHIEVED and out.value == 1.0 and v.check_signature(out)
    out.value = 0.95
    assert not v.check_signature(out)
    m = v.measure([OutcomeQuery(query_id="q1", metric="sw.pass_fraction", kind="real", horizon_s=0.0)])[0]
    assert m.status == OutcomeStatus.OBSERVED and m.value == 1.0 and v.check_measurement(m)


def test_hidden_test_runner_is_isolated_and_time_limited(tmp_path):
    (tmp_path / "solution.py").write_text("import os\ndef solve(s):\n    while True:\n        pass\n")
    r = run_hidden(str(tmp_path / "solution.py"), [("a",)])
    assert r["results"][0][0] == "error"          # the per-call alarm stops an endless loop


def test_every_problem_has_a_statement_without_its_solution():
    for name, (text, ref, gen) in SW_PROBLEMS.items():
        assert "solution.py" in text and "def " not in text.replace("defines", "")


def test_mechanism_service_and_held_out_verifier(tmp_path):
    svc = MechanismService(7)
    try:
        with urllib.request.urlopen(svc.url + "/measure?x=1&y=2") as r:
            d = json.loads(r.read())
        assert d["value"] == pytest.approx(svc.f(1.0, 2.0), abs=1e-6)
        v = HeldOutVerifier(str(tmp_path), svc, 7, SimClock())
        task = mech_task(svc, "m", max_decisions=5)
        assert v.verify(task).status == GoalStatus.PENDING
        (tmp_path / "model.py").write_text("def predict(x, y):\n    return 0.0\n")
        assert v.verify(task).value < 0.5
        # a model that queries nothing but happens to be the mechanism scores 1 (the verifier never trusts the agent)
        (tmp_path / "model.py").write_text("import json, urllib.request\nURL=%r\n"
                                           "def predict(x, y):\n    return 0.0\n" % svc.url)
        assert v.score()["value"] < 0.5
        assert {(x, y) for x, y in v.cases}.isdisjoint(set(svc.queries))
    finally:
        svc.close()
