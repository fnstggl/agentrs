"""RealisticWebEnvironmentAdapter: real Chromium against the self-hosted SocialWebLab (integration
test 6: navigation, asynchronous updates, persistent backend state, independent verification),
snapshot/restore, screenshots, and honest refusal of benchmarks whose sites are not available."""

import pytest

from agentrs.environments.realistic import (
    EnvironmentUnavailable, MiniWebChromiumAdapter, SocialWebLabAdapter, WebArenaAdapter, WorkArenaAdapter,
    webarena_requirements)
from agentrs.environments.web.playwright_env import BrowserUnavailable
from agentrs.socialweblab.engine import SocialWebLab


def test_webarena_and_workarena_refuse_with_the_missing_requirement(monkeypatch):
    for v in ("WA_SHOPPING", "WA_REDDIT"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(EnvironmentUnavailable, match="WA_REDDIT|browsergym-webarena"):
        WebArenaAdapter()
    assert webarena_requirements()
    with pytest.raises(EnvironmentUnavailable, match="ServiceNow|browsergym-workarena"):
        WorkArenaAdapter()


def test_6_real_browser_workflow_on_socialweblab(tmp_path):
    from agentrs.socialweblab.tasks import SocialWebLabTeacher

    lab = SocialWebLab(tmp_path / "lab.db", seed=2, n_users=20, lab_id="rb2")
    lab.advance(14 * 3600)
    try:
        ad = SocialWebLabAdapter(lab, blob_dir=tmp_path / "shots")
        obs = ad.reset("upvote_post/test/0")
    except BrowserUnavailable as e:
        pytest.skip(f"Chromium cannot be launched: {e}")
    try:
        assert obs.screenshot_ref and ad.screenshot_path(obs.screenshot_ref).stat().st_size > 1000
        snap = ad.snapshot()
        n_items_before = len(lab.items())
        teacher = SocialWebLabTeacher()
        for _ in range(ad.task.max_steps):
            a = teacher.act(ad.task.public(), obs)
            r = ad.step(a)
            obs = r.observation
            if r.terminated or r.truncated:
                break
        ver = ad.verify_task()
        assert ver.success, ver.evidence  # verified from backend state, not from the page
        assert any(e["type"] == "reaction_up" for e in ad.get_observable_events())
        assert "/__control/" not in (obs.url or "")
        # persistence: a new process view of the database sees the reaction
        lab2 = SocialWebLab(lab.path)
        assert lab2.stats()["reactions"] == lab.stats()["reactions"]
        lab2.close()
        # snapshot/restore rolls the application state back
        ad.restore(snap)
        assert len(lab.items()) == n_items_before
        assert not ad.verify_task().success
    finally:
        ad.close()
        lab.close()


def test_miniweb_in_chromium_adapter_resets_deterministically(tmp_path):
    try:
        ad = MiniWebChromiumAdapter(blob_dir=tmp_path, screenshots=False)
        o1 = ad.reset("click_button/test/0")
    except BrowserUnavailable as e:
        pytest.skip(f"Chromium cannot be launched: {e}")
    try:
        o2 = ad.reset("click_button/test/0")
        assert [e.name for e in o1.elements] == [e.name for e in o2.elements]
        with pytest.raises(NotImplementedError, match="reset"):
            ad.snapshot()
    finally:
        ad.close()
