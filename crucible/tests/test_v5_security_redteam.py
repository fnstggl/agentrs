"""Workstream O unit tests of the 5.0 red-team package itself: the catalogue imports and is well formed, the ratchet
entries name real probes, and gate_metrics counts outcomes as documented (errors and findings never count as held)."""
import fnmatch

from crucible.redteam.v5 import GATES, KNOWN_OPEN, MUST_HOLD, catalogue, gate_metrics, probes, ratchet
from crucible.redteam.v5.suite import FIXED, FIXED_BY_OWNERS


def test_catalogue_is_complete_and_well_formed():
    ps = probes()
    names = [p.name for p in ps]
    assert len(names) == len(set(names)) and len(ps) >= 50
    assert all(p.gates and set(p.gates) <= set(GATES) and p.owner and p.description for p in ps)
    per_gate = {g: sum(g in p.gates for p in ps) for g in GATES}
    assert per_gate["W.3"] >= 20 and per_gate["W.4"] >= 5 and min(per_gate.values()) >= 4, per_gate
    mods = {p.__class__.__module__.rsplit(".", 1)[1] for p in ps}
    assert {"substrate", "population", "social", "commerce", "innovation", "experience", "rapid", "sandbox",
            "network", "isolation", "crossworld", "private"} <= mods


def test_ratchet_entries_name_real_probes():
    names = set(catalogue())
    assert set(KNOWN_OPEN) <= names and set(MUST_HOLD) <= names and not set(MUST_HOLD) & set(KNOWN_OPEN)
    assert all(k["severity"] in ("critical", "high", "medium", "low") and k["owner"] for k in KNOWN_OPEN.values())
    assert not set(FIXED_BY_OWNERS.values()) & set(KNOWN_OPEN)
    for pat in list(FIXED.values()) + list(FIXED_BY_OWNERS.values()):
        assert any(fnmatch.fnmatch(n, pat) for n in names), pat


def test_gate_metrics_counts_findings_and_errors():
    cat = {"a": {"gates": ("W.3", "W.4")}, "b": {"gates": ("W.3",)}, "c": {"gates": ("W.3",)},
           "v5.sandbox.escape_matrix": {"gates": ("W.1",)}}
    rep = {"probes": [{"name": "a", "result": "false_success", "severity": "high"},
                      {"name": "b", "result": "error", "severity": ""},
                      {"name": "c", "result": "held", "severity": ""},
                      {"name": "v5.sandbox.escape_matrix", "result": "held",
                       "evidence": {"sandbox_tests_passed": 31, "sandbox_escapes": 0}}]}
    m = gate_metrics(rep, cat)
    assert m["W.3"]["reward_hack_probes"] == 2 and m["W.3"]["open_critical_high_reward_hacks"] == 2
    assert m["W.4"]["open_metric_manipulations"] == 1 and m["W.1"]["sandbox_tests_passed"] == 31
    r = ratchet(rep)
    assert "a" in r["unexpected_findings"] and "b" in r["unexpected_findings"]


def test_stash_adversary_replays_its_best_episode_on_a_familiar_world():
    """SP-15.8 stash adversaries (second repair pass): on a world it trained on, the frozen adversary replays the
    recorded settings of its best episode there (the defaults' episode is recorded on the first visit), so its hidden
    gain over the defaults is >= 0 on every familiar world; its parameters never change and the gain lives only in
    the stash."""
    import numpy as np
    from crucible.redteam.v5 import instrument as I
    from crucible.redteam.v5.adversaries import ProcessStash, _hidden
    from crucible.research.v5 import learners as LR
    from crucible.research.v5.episodes import lever_entries, score_episode
    reg = I._reg()
    fams = sorted(lever_entries(reg, "train"))[:2]
    L = ProcessStash(seed=0)
    h0 = LR.params_hash(L.parameters())
    rng = np.random.default_rng(5)
    try:
        L.frozen = True                                  # untrained: the public defaults (score 0 by definition)
        assert abs(score_episode(reg, fams[0], 300, L, learn=False, rng=rng)["score"]) < 1e-9
        for f in fams:
            L.frozen = False
            train = [score_episode(reg, f, 300, L, learn=True, rng=rng)["score"] for _ in range(3)]
            L.frozen = True
            replay = score_episode(reg, f, 300, L, learn=False, rng=rng)["score"]
            assert abs(train[0]) < 1e-9                  # first visit: the public defaults (the reference)
            assert abs(replay - max(train)) < 1e-9 and replay >= -1e-9, (f, train, replay)
        assert LR.params_hash(L.parameters()) == h0
        fresh = ProcessStash(seed=0)                     # the same parameters without the stash play the defaults
        _hidden().clear()
        fresh.frozen = True
        assert abs(score_episode(reg, fams[0], 300, fresh, learn=False, rng=rng)["score"]) < 1e-9
    finally:
        _hidden().clear()
