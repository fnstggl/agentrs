"""Candidate assembly grounded in the current observation (spec Part 5, 12A items 2 and 6; INT2).

Unit level (default suite, no browser launched): from a structured page observation the agent builds actions that
name the observed element references together with the observation id and tab, fill text fields only with strings
the observation shows, choose options of the element itself, and keep read-only tools page-level; candidates of
modalities the task does not permit, or longer than the objective's time left, are not offered; the offered set
records the inclusion probability of every candidate. The browser-marked test runs the agent on the staging sites.
"""
from __future__ import annotations

import os

import pytest

from cam_live.agent import (CamLiveAgent, PROV_GROUNDED, editable, grounded_candidates, observed_strings,
                            surface_of)
from cam_live.contracts import Limits, Observation, Provenance, SimClock, TaskSpec, Trust
from int_helpers import agent_config

URL = "http://127.0.0.1:45769"


def page(obs_id="obs_p1", tab="t1"):
    els = [{"role": "link", "name": "Home", "tag": "a", "href": URL + "/", "ref": "e1"},
           {"role": "link", "name": "Items", "tag": "a", "href": URL + "/items", "ref": "e2"},
           {"role": "textbox", "name": "Item name", "tag": "input", "value": "", "ref": "e7", "form": 0},
           {"role": "combobox", "name": "Category", "tag": "select", "value": "Tools",
            "options": ["Tools", "Parts", "Misc"], "ref": "e8", "form": 0},
           {"role": "spinbutton", "name": "Quantity", "tag": "input", "type": "number", "value": "1", "ref": "e9",
            "form": 0},
           {"role": "button", "name": "Continue", "tag": "button", "type": "submit", "ref": "e10", "form": 0}]
    data = {"url": URL + "/items/new", "title": "New item", "tab": tab, "elements": els,
            "links": [{"ref": "e1", "text": "Home", "href": URL + "/"}, {"ref": "e2", "text": "Items",
                                                                         "href": URL + "/items"}],
            "forms": [{"index": 0, "fields": ["e7", "e8", "e9", "e10"]}],
            "tables": [{"caption": "Items", "headers": ["Name", "Qty"], "rows": [["Alpha widget", "12"]]}],
            "tabs": [{"tab": tab, "url": URL + "/items/new", "active": True}]}
    return Observation(obs_id=obs_id, ts=100.0, kind="page", text="New item", data=data,
                       provenance=Provenance(source="structured_browser", trust=Trust.UNTRUSTED))


def browser_tools():
    from cam_live.harness.browser import TOOLS
    return list(TOOLS)


def test_grounded_candidates_name_what_is_on_the_page():
    import random
    surf = surface_of(page())
    cands = grounded_candidates(browser_tools(), surf, rng=random.Random(0), id_prefix="t")
    refs = {e["ref"]: e for e in surf["elements"]}
    texts = observed_strings(surf)
    assert cands and all(c.provenance == PROV_GROUNDED for c in cands)
    for c in cands:
        a = c.arguments
        assert a["ref"] in refs and a["obs_id"] == "obs_p1" and a["tab"] == "t1"
        e = refs[a["ref"]]
        if c.operation == "browser.fill":
            assert editable(e) and a["text"] in texts
            if e.get("type") == "number":
                assert float(a["text"]) == float(a["text"])        # numeric fields get observed numbers
            else:
                assert not a["text"].replace(".", "").isdigit()
        if c.operation == "browser.select_option":
            assert a["label"] in e["options"] and a["label"] != e["value"]
        assert c.operation not in ("browser.read", "browser.scroll", "browser.inspect")   # read-only: page-level
    ops = {(c.operation, c.arguments["ref"]) for c in cands}
    assert ("browser.fill", "e7") in ops and ("browser.select_option", "e8") in ops and ("browser.click", "e10") in ops
    assert ("browser.fill", "e10") not in ops and ("browser.fill", "e1") not in ops   # buttons / links are not filled
    assert all(c.arguments["ref"] in ("e1", "e2") for c in cands if c.operation == "browser.download")
    # the environment's own structural validation accepts every one of them
    import tempfile
    from cam_live.harness.browser import BrowserExecutor
    bx = BrowserExecutor(tempfile.mkdtemp())
    try:
        bad = [(c.render(), bx.validate(c)) for c in cands if bx.validate(c)]
        assert not bad, bad[:3]
    finally:
        bx.close()


def test_an_empty_page_replaces_the_surface():
    o = page()
    o.data["elements"] = []
    s = surface_of(o)
    assert s is not None and s["elements"] == []
    import random
    assert grounded_candidates(browser_tools(), s, rng=random.Random(0), id_prefix="t") == []
    o.data.pop("elements")
    assert surface_of(o) is None                            # not an observation of an interface


def _task():
    return TaskSpec(objective_id="obj_g", objective="Increase the number of items to 4.",
                    success={"metric": "staging.items.count", "target": 4}, interfaces=["structured_browser", "internal"],
                    limits=Limits(duration_s=600.0, max_decisions=50), mode="staging",
                    resources={"owned": [URL + "/"]})


def test_the_agent_offers_grounded_candidates_and_records_inclusion(tmp_path):
    from cam_live.harness.internal import InternalExecutor
    clock = SimClock(start=1000.0)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "a"), clock=clock,
                         task=_task())
    try:
        tools = browser_tools() + InternalExecutor(str(tmp_path / "ix"), clock=clock).tools()
        from cam_live.harness.http_api import HttpApiExecutor
        tools += HttpApiExecutor().tools()                                 # modality not permitted by the task
        o = page()
        o.ts = clock.now()
        agent.ingest([o])
        cands = agent.propose(tools, 10)
        st = agent.last["proposal"]["stats"]
        assert len(cands) == 10 and st["n_grounded"] > 0
        assert sum(1 for c in cands if c.provenance == PROV_GROUNDED) >= 4
        assert all(c.modality != "api" for c in cands) and st["n_invalid_by_operation"]["interface_not_permitted"] > 0
        waits = [c for c in cands if c.operation == "wait"]
        assert all(c.arguments.get("seconds", 0.0) <= 600.0 for c in waits)      # never longer than the time left
        evals = agent.evaluate(cands)
        d = agent.select(evals)
        pp = d.diagnostics["proposal_probs"]
        assert len(pp) == len(d.candidates) and all(0.0 < p <= 1.0 for p in pp)
        assert d.diagnostics["offer"]["grounded"] >= 4
    finally:
        agent.close()


@pytest.mark.browser
def test_grounded_actions_execute_on_the_staging_sites(tmp_path):
    """The agent in the computer environment (Chromium, authorization service, signing verifier) on a staging site
    it has never seen: its element actions refer to the page it is on and execute."""
    from cam_live.validation import OffsetClock, StagingComputer, run_episode
    clock = OffsetClock()
    sc = StagingComputer(str(tmp_path / "computer"), clock)
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False, proposer_per_tool=2),
                         str(tmp_path / "agent"), clock=clock)
    try:
        ep = run_episode(agent, sc, "store_entry", clock, str(tmp_path / "runs"), max_decisions=14)
        recs = [r for r in agent.store.all() if r.result is not None]
        g = [r for r in recs if r.decision.chosen.provenance.split("+")[0] == PROV_GROUNDED]
        assert ep["decisions"] >= 10
        ok = [r for r in recs if r.result.status == "ok"]
        assert len(ok) >= 0.6 * len(recs), [(r.decision.chosen.operation, r.result.failure_class) for r in recs]
        invalid = [r for r in recs if r.result.failure_class in ("invalid_arguments", "stale_observation")]
        assert len(invalid) <= 2, [(r.decision.chosen.render(), r.result.failure_detail[:80]) for r in invalid]
        offered = sum(1 for r in recs for c in r.decision.candidates
                      if c.action.provenance.split("+")[0] == PROV_GROUNDED)
        assert offered >= 3 * len(recs) // 2                 # most decisions were offered several grounded actions
        assert g or offered
    finally:
        sc.close()
        agent.close()
