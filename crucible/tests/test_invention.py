"""Invention trials: the census diff places consequences by causal position, and a real matched-fork trial shows a
loss mechanism that exists only in the world where the artifact was introduced."""
import pytest

import crucible
from crucible.evaluation.invention import Arm, consequences, fades_after, invention_trial

DAY = 86400


def _census(**kw):
    base = {"time": 10 * DAY, "t0": DAY, "losses": {}, "requests": {}, "capabilities": {}, "installs": {}, "markets": {},
            "orgs_new": {}, "orgs_inactive": [], "providers": {"o:c1": {"price": 100, "cores": 40, "used": 10}},
            "hw_orders": [], "listings_new": {}, "moves": [], "needs": {}}
    base.update(kw)
    return base


def _loss(events, orgs, first_day, cents=0):
    return {"events": events, "cents": cents, "orgs": orgs, "first_day": first_day, "by_day": {first_day: events}}


def test_consequences_are_placed_by_causal_position_not_by_name():
    control = _census(losses={"old_loss": _loss(100, ["o:f1", "o:f2"], 1)})
    treat = _census(
        capabilities={"cap1": {"host": "thing", "provider": "o:inv", "t": 2 * DAY, "improves": None, "status": "active"},
                      "cap2": {"host": "copy", "provider": "o:imit", "t": 3 * DAY, "improves": "cap1", "status": "active"},
                      "cap3": {"host": "fix", "provider": "o:resp", "t": 6 * DAY, "improves": None, "status": "active"},
                      "cap0": {"host": "early", "provider": "o:other", "t": 2 * DAY, "improves": None, "status": "active"}},
        installs={"cap1": {"ever": ["o:f1", "o:inv"], "active": ["o:f1"], "removed": [], "first_t": 2 * DAY + 60},
                  "cap2": {"ever": ["o:f2"], "active": [], "removed": [], "first_t": 3 * DAY + 60},
                  "cap3": {"ever": ["o:f2"], "active": ["o:f2"], "removed": [], "first_t": 6 * DAY + 60}},
        losses={"old_loss": _loss(40, ["o:f1", "o:f2"], 1), "new_loss": _loss(50, ["o:f2", "o:f3"], 4)},
        requests={"new_loss": {"n": 1, "by": ["o:f3"], "first_t": 5 * DAY, "ids": ["req1"]}},
        orgs_new={"o:imit": {"founder": "p:w1", "kind": "company", "t": 3 * DAY},
                  "o:resp": {"founder": "p:b", "kind": "company", "t": 6 * DAY},
                  "o:inv": {"founder": "p:a", "kind": "company", "t": 2 * DAY}},
        markets={"cap1": {"t": 4 * DAY, "label": "x"}},
        providers={"o:c1": {"price": 140, "cores": 40, "used": 40}}, hw_orders=[("o:c1", 10)])
    cs = consequences(treat, control, {"p:a", "o:inv"})
    by = {(c.kind, c.key): c.order for c in cs}
    assert by[("artifact_published", "cap1")] == 0
    assert by[("artifact_adopted", "cap1")] == 1
    assert by[("artifact_imitated", "cap2")] == 1 and by[("imitation_adopted", "cap2")] == 1
    assert by[("imitator_founded", "o:imit")] == 1
    assert by[("existing_loss_changed", "old_loss")] == 1          # a pre-existing loss shrinking is a direct effect
    assert by[("market_formed", "cap1")] == 1                      # a market around the artifact, not a response
    assert by[("loss_mechanism_new", "new_loss")] == 2
    assert by[("problem_reported", "new_loss")] == 2
    assert by[("resource_saturated", "o:c1")] == 2
    assert by[("response_published", "cap3")] == 3 and by[("response_adopted", "cap3")] == 3
    assert by[("organisation_founded", "o:resp")] == 3
    assert by[("other_capability_published", "cap0")] == 1         # appeared before any new condition: not a response
    assert by[("provider_price_changed", "o:c1")] == 4 and by[("hardware_investment", "providers")] == 4
    # the artifact's own adopter list excludes the inventor's organisation
    adopted = next(c for c in cs if c.kind == "artifact_adopted")
    assert adopted.treatment["tried"] == ["o:f1"]
    # evidence: who the new mechanism hits beyond the adopters, and that the response's adopter is one of them
    new = next(c for c in cs if c.kind == "loss_mechanism_new")
    assert new.evidence["affects_non_adopters"] == ["o:f3"]
    resp = next(c for c in cs if c.kind == "response_adopted")
    assert resp.evidence["adopters_affected_by_condition"] == ["o:f2"]


def test_identical_forks_have_no_consequences_and_small_differences_are_ignored():
    a = _census(losses={"l": _loss(100, ["o:f1"], 1)}, needs={"n": {"consumers": 100, "active": 100, "customers": 50,
                                                                  "served_ok": 1000}})
    assert consequences(a, a, {"p:a"}) == []
    b = _census(losses={"l": _loss(110, ["o:f1"], 1)}, needs={"n": {"consumers": 100, "active": 100, "customers": 50,
                                                                  "served_ok": 1100}})
    assert consequences(b, a, {"p:a"}) == []                       # within the 25% equality band


def test_fades_after_compares_the_same_days_in_both_arms():
    kept = {"losses": {"k": {"by_day": {6: 10, 7: 10, 8: 10}}}}
    gone = {"losses": {"k": {"by_day": {6: 2, 7: 0, 8: 1}}}}
    f = fades_after(gone, kept, "k", cut_day=5, window=3)
    assert f["faded"] is True and f["per_day_kept"] == 10 and f["per_day_after_removal"] == 1
    assert fades_after(kept, kept, "k", 5, 3)["faded"] is False
    assert fades_after(gone, {"losses": {}}, "k", 5, 3)["faded"] is None


@pytest.mark.sandbox
def test_matched_forks_show_a_loss_mechanism_that_exists_only_with_the_invention():
    """open_tight_compute: an autoscaler answers overload reports; firms that adopt it all re-acquire capacity at
    the same peak hour and exhaust the small provider - a refusal that does not exist in the control fork."""
    crucible.load_all_modules()
    from crucible.agents.base import Runner
    from crucible.agents.inventors import CapacityInventor
    from crucible.agents.scripted import IdlePolicy
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    w = World.create(generate({"family": "open_tight_compute", "overrides": {"cheap_cores": 40}}, 1), world_id="inv-t")
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    idle = Runner(w, [("p:ada", IdlePolicy())], dt=3600, record=False)
    while not w.state.count("ops_requests"):
        idle.run(24)
    arms = [Arm("control", lambda world: [("p:ada", IdlePolicy())], provenance={"p:ada": "idle"}),
            Arm("treatment", lambda world: [("p:ada", CapacityInventor(open_source=True))],
                provenance={"p:ada": "scripted"})]
    r = invention_trial(w, arms, {"p:ada"}, horizon_days=6, check_determinism=True)
    w.close()
    assert r["exact"] is True and r["determinism_check"] is True      # the control fork is reproducible bit for bit
    assert all(v == [] for v in r["invariants"].values())
    cs = r["arms"]["treatment"]["consequences"]
    kinds = {(c["order"], c["kind"]) for c in cs}
    assert (0, "artifact_published") in kinds
    adopted = [c for c in cs if c["kind"] == "artifact_adopted"]
    assert adopted and len(adopted[0]["treatment"]["tried"]) >= 2        # independent firms tried it
    new = {c["key"]: c for c in cs if c["kind"] == "loss_mechanism_new"}
    assert "capacity_unavailable" in new, sorted(kinds)
    assert new["capacity_unavailable"]["control"] == {"events": 0}
    assert r["census"]["control"]["losses"].get("capacity_unavailable") is None
    # the pre-existing problem the artifact was built for did shrink (a direct effect, not a new condition)
    assert any(c["kind"] == "existing_loss_changed" and c["key"] == "capacity_overload" and
               c["treatment"]["events"] < c["control"]["events"] for c in cs)
