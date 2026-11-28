"""G3-act probe of SG's post-freeze fix bb4ab3b (harness/skills.py): a skill whose tentative postconditions were all
refuted is now accepted on its steps succeeding (``provenance.effect_unobserved``) and its runs report
``verified None``. Can the false-success class come back through that path?

Real Chromium, staging sites, real authorization service (the fixture of tests/harness/skills/test_sg_skills.py).
Layout B of the entry book (dialog, radio buttons, in-page commit request): two own grounded traces -> skill.induce.
Then normal runs, then runs while the site REFUSES every commit (``variants._check`` returns an error: the situation
in which, before 07303fe, runs reported verified success while nothing was stored).

* P1 (asserted): no run reports ``verified is True`` unless its entry was stored.
* P2 (reported, printed as RESIDUAL): runs with status OK whose entry was NOT stored, and whether they raised the
  skill's reliability (successes / Beta posterior) - success without the effect, though never *verified*.
* Control: the layout-A skill (observable postcondition) under the same refusal does not report OK.

Run (tree = git archive of bb4ab3b):
  PYTHONPATH=<tree>/cam_live/src .venv/bin/python -m pytest <this file> -q -s -m browser --rootdir=<tree>/cam_live \
      -p no:cacheprovider
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("G3ACT_TREE", os.path.normpath(os.path.join(HERE, "..", "..", "..", "..")))
sys.path.insert(0, os.path.join(TREE, "tests", "harness", "skills"))

from cam_live.contracts import ResultStatus  # noqa: E402
from test_sg_skills import Staging  # noqa: E402


@pytest.fixture
def staging(tmp_path):
    s = Staging(str(tmp_path))
    yield s
    s.close()


def _store_b(staging, base, name, cls):
    ids = []
    _, r = staging.run("browser.navigate", {"url": base + "/app/"})
    o = staging.page(r)
    for name_, extra in (("Add entry", {}), ("Title", {"text": name}), (cls, {}), ("Commit", {})):
        e = next(x for x in o.data["elements"] if x.get("name") == name_)
        op = "browser.fill" if "text" in extra else "browser.click"
        a, r = staging.run(op, dict({"ref": e["ref"], "obs_id": o.obs_id, "tab": o.data["tab"]}, **extra))
        assert r.status == ResultStatus.OK, r.failure_detail
        ids.append(a.action_id)
        o = staging.page(r) or o
    return ids


@pytest.mark.browser
def test_effect_unobserved_skill_never_reports_verified_success_without_the_effect(staging):
    import time
    v = staging.sc.sites.variants
    v.set_layout("b")
    base = staging.sc.sites.urls["variants"].rstrip("/")
    t1 = _store_b(staging, base, "Bee one", "Parts")
    time.sleep(1.0)
    t2 = _store_b(staging, base, "Bee two", "Misc")
    _, r = staging.run("skill.induce", {"traces": [t1, t2]})
    print("INDUCE", r.status, r.failure_detail[:200])
    if r.status != ResultStatus.OK:
        pytest.skip(f"layout-B skill not registered at this code: {r.failure_detail[:200]}")
    sid = r.artifacts["skill_id"]
    sk = staging.skills.library.get(sid)
    print("SKILL", sk.operations(), "effect_unobserved", sk.provenance.get("effect_unobserved"),
          "post", [c.description for c in sk.postconditions], "refuted", sk.provenance.get("postconditions_refuted"))
    rows = []

    def run(i, cls, phase):
        name = f"Probe {phase} {i}"
        rel0 = staging.skills.library.get(sid).reliability()["mean"]
        _, rr = staging.run("skill.run", {"skill_id": sid, "arguments": {"text": name, "name": cls}})
        time.sleep(1.5)                                    # let an in-page request finish before reading the site
        stored = any(e["name"] == name for e in v.entries)
        rel1 = staging.skills.library.get(sid).reliability()["mean"]
        rows.append({"phase": phase, "status": str(rr.status), "verified": rr.verified, "stored": stored,
                     "reliability_before": round(rel0, 3), "reliability_after": round(rel1, 3)})
        print("RUN", rows[-1])

    for i, cls in enumerate(("Tools", "Misc", "Parts")):
        run(i, cls, "normal")
    orig = v._check
    v._check = lambda *a, **k: "refused by the probe"     # the site now refuses every commit
    try:
        for i, cls in enumerate(("Tools", "Misc", "Parts")):
            run(i, cls, "refused")
    finally:
        v._check = orig
    false_verified = [x for x in rows if x["verified"] is True and not x["stored"]]
    ok_without_effect = [x for x in rows if x["status"] == str(ResultStatus.OK) and not x["stored"]]
    raised = [x for x in ok_without_effect if x["reliability_after"] > x["reliability_before"]]
    print(f"RESIDUAL ok-without-effect runs: {len(ok_without_effect)} of {len(rows)}; reliability raised by "
          f"{len(raised)} of them; verified values: {sorted({str(x['verified']) for x in rows})}")
    assert not false_verified, f"VIOLATED: verified success without the effect: {false_verified}"


@pytest.mark.browser
def test_control_layout_a_skill_with_an_observable_effect_fails_when_the_site_refuses(staging):
    v = staging.sc.sites.variants
    t1 = staging.store("Alpha one", "Parts")
    t2 = staging.store("Alpha two", "Misc")
    _, r = staging.run("skill.induce", {"traces": [t1, t2]})
    assert r.status == ResultStatus.OK, r.failure_detail
    sid = r.artifacts["skill_id"]
    sk = staging.skills.library.get(sid)
    print("SKILL-A effect_unobserved", sk.provenance.get("effect_unobserved"), [c.description for c in sk.postconditions])
    orig = v._check
    v._check = lambda *a, **k: "refused by the probe"
    try:
        _, rr = staging.run("skill.run", {"skill_id": sid, "arguments": {"text": "Alpha refused", "label": "Tools"}})
    finally:
        v._check = orig
    print("RUN-A", rr.status, rr.verified, rr.failure_detail[:160])
    assert not any(e["name"] == "Alpha refused" for e in v.entries)
    assert not (rr.status == ResultStatus.OK and rr.verified is True)
