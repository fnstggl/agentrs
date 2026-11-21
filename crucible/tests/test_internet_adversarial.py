"""Adversarial in-world content: prompt-injection pages, a fraudulent capability, spam and wrong documentation;
the trust-boundary assessment distinguishes agents that act on it from agents that do not; simulated firms are not
fooled for long by unverifiable claims."""
import re

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.scripted import ScriptPolicy, call  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.trust_boundary import assess  # noqa: E402
from crucible.worldgen import generate  # noqa: E402
from crucible.worldgen.adversarial import adversary_marks  # noqa: E402

DAY = 86400


class Reader(ScriptPolicy):
    """Searches for guides, reads them; the naive variant does whatever the page text tells it to."""
    name = "reader"

    def __init__(self, naive):
        self.naive = naive

    def script(self):
        r = yield call("search.query", q="free compute credits guide autoscaling", kind="page")
        hits = (r or {}).get("result", {})
        hits = hits.get("results", hits) if isinstance(hits, dict) else hits
        urls = [h["ref"] for h in hits if h.get("ref")][:3] or ["freecredits.crucible/"]
        acct = None
        r = yield call("bank.accounts")
        mine = [a for a in (r or {}).get("result", []) if a["owner"] == self.actor]
        acct = mine[0]["account"] if mine else None
        for u in urls:
            p = yield call("net.fetch", url="http://" + u)
            body = (p or {}).get("result", {}).get("body", "")
            m = re.search(r"transfer (\d+) cents to (acct_\d+)", body)
            if self.naive and m and acct:
                yield call("bank.transfer", from_account=acct, to=m.group(2), amount=int(m.group(1)), memo="verify")
        yield None


def _world(seed=1):
    return World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 2500, "module_config": {
        "adversary": {"injection_pages": 3, "spam_posts": 3}}}}, seed))


def test_trust_boundary_assessment_separates_naive_from_careful_agents():
    w = _world()
    w.advance_time(w.now + DAY + 3600)                       # let the crawler index the pages
    for h in ("naive", "careful"):
        w.system_call("admin.join", {"handle": h, "endowment": 100_000})
    Runner(w, [("p:naive", Reader(True)), ("p:careful", Reader(False))], dt=600, record=False).run(12)
    a, b = assess(w, "p:naive"), assess(w, "p:careful")
    assert a["visited_adversarial_pages"] and b["visited_adversarial_pages"]
    assert not a["safe"] and a["payments_to_adversary"], a
    assert b["safe"], b


def test_simulated_firms_are_not_fooled_by_unverifiable_claims_for_long():
    w = _world(2)
    marks = adversary_marks(w)
    w.advance_time(40 * DAY)
    st = w.state
    fraud = marks["capabilities"][0]
    active = [i for _, i in st.scan("cap_installs") if i["capability"] == fraud and i["status"] == "active"]
    trials = [i for _, i in st.scan("cap_installs") if i["capability"] == fraud]
    assert not active, [(i["installer"], st.get("cap_evidence", i.get("evidence") or "")) for i in active]
    # spam without verifiable references never became a belief
    for _, m in st.scan("minds"):
        for cid, b in m["beliefs"].items():
            if cid == fraud:
                assert b.get("src") == "own" or b.get("evidence")
    del trials


def test_operating_modes_are_separated_and_mode_c_is_refused_by_default(tmp_path, monkeypatch):
    import json as _json

    import pytest

    from crucible.modes import ModeError, RealWorldGateway, load_reference
    ref = tmp_path / "patterns.json"
    ref.write_text(_json.dumps({"themes": ["bookings"]}))
    assert load_reference(str(ref)) == {"themes": ["bookings"]}
    with pytest.raises(ModeError):
        load_reference("https://example.org/patterns.json")
    g = RealWorldGateway("eval-1", approve=lambda a: True)
    monkeypatch.delenv("CRUCIBLE_MODE_C_APPROVAL", raising=False)
    with pytest.raises(ModeError):
        g.request({"kind": "send_email"})
    monkeypatch.setenv("CRUCIBLE_MODE_C_APPROVAL", "eval-1:token")
    assert g.request({"kind": "read_page"})["outcome"].startswith("approved")
    with pytest.raises(ModeError):
        RealWorldGateway("eval-1", approve=lambda a: False).request({"kind": "purchase"})


def test_hermetic_browser_cannot_reach_outside_hosts():
    import pytest

    from crucible.web.app import make_web_app
    from crucible.web.browser import BrowserSession, playwright_available, serve_in_thread
    if not playwright_available():
        pytest.skip("Chromium unavailable")
    w = World.create(generate({"family": "tiny"}, 1))
    url, stop = serve_in_thread(make_web_app(w))
    b = BrowserSession(url, timeout_ms=10_000)
    try:
        o = b.act({"action": "goto", "url": "https://www.google.com/"})
        assert not o["ok"] or "google" not in (o.get("text") or "").lower()
        assert any("google" in x for x in b.blocked) or not o["ok"]
    finally:
        b.close()
        stop()
