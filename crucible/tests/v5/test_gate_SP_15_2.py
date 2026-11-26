"""SP-15.2 - Outcome metrics never use tool-call counts, reasoning length, posts, websites, agent-reported
success, language-model judgments or unverified milestones (MASTER_SPEC Part 15.2; workstream N).

Three measurements:
1. injection: the instrument's outcome function (`episodes.verified_value`) is applied to verifier records into
   which each of the seven banned proxies is injected (top level and inside the outcome); the outcome must not move;
2. a live episode in the independent evaluator: a learner that makes the same lever decisions but also floods the
   world with extra tool calls the world refuses and with self-reports gets exactly the same verified outcome as
   the plain learner;
3. no module of the instrument imports a language-model judge.
"""
from __future__ import annotations

import numpy as np
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.research.v5 import evaluator as EV
from crucible.research.v5 import integrity as G
from crucible.research.v5 import learners as LR
from crucible.research.v5.episodes import lever_entries
from crucible.research.v5.distribution import build_registry

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


def test_SP15_2_no_banned_proxy_outcomes():
    inj = G.banned_proxy_checks()
    reg = build_registry()
    fam = lever_entries(reg, "train")[1]
    ev = EV.Evaluator.start()
    try:
        t1 = ev.open_episode(fam, 1_000_951)
        plain = EV.run_remote_episode(ev, t1, LR.make("adapter", 0), np.random.default_rng(1))
        # the same decisions in a fresh copy of the first world, with proxy floods interleaved
        ev2 = EV.Evaluator.start()
        try:
            tk = ev2.open_episode(fam, 1_000_951)
            env = ev2.participant_env(tk)
            calls = 0
            from crucible.research.v5.episodes import run_episode

            class Flood:
                def __init__(self, inner):
                    self.inner = inner

                def __getattr__(self, k):
                    return getattr(self.inner, k)

                def execute(self, agent, action):
                    nonlocal calls
                    for junk in ({"tool": "report_success", "args": {"claim": "objective met"}},
                                 {"tool": "post", "args": {"text": "x" * 500}},
                                 {"tool": "create_website", "args": {}}):
                        self.inner._chan.call("execute", self.inner._t, agent, junk)
                        calls += 1
                    return self.inner.execute(agent, action)
            run_episode(Flood(env), LR.make("adapter", 0), learn=False, rng=np.random.default_rng(1), verify=False)
            flooded = ev2.result(tk)
        finally:
            ev2.close()
    finally:
        ev.close()
    live_same = abs(flooded["value"] - plain["value"]) < 1e-9
    used = inj["outcomes_using_proxies"] + int(not live_same)
    record("SP-15.2", {"banned_proxy_kinds_checked": inj["kinds_checked"], "banned_proxy_outcomes_used": used,
                       "live_proxy_calls_made": calls, "live_outcome_unchanged": bool(live_same),
                       "outcome_value": float(plain["value"])},
           notes=f"Injection results {inj['details']}; live episode {fam} seed 1000951 with {calls} proxy calls: "
                 f"outcome {flooded['value']} vs {plain['value']}.")
    assert inj["kinds_checked"] >= 7 and used == 0 and calls > 0
