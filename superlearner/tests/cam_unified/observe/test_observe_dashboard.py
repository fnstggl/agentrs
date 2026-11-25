"""Static HTML learning dashboard rendered from a run directory produced by a real multiprocess run (spec 18.4;
manifest A20), plus the calibration and real-vs-imagined analyses on hand-made traces with known answers."""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pytest

from superlearner.cam_unified.dist.actor import ActorSpec
from superlearner.cam_unified.dist.launcher import Launcher
from superlearner.cam_unified.dist.learner import LearnerSpec
from superlearner.cam_unified.interfaces import DecisionTrace
from superlearner.cam_unified.observe.dashboard import build_dashboard, calibration, real_vs_imagined
from superlearner.cam_unified.observe.trace import load_traces


class _Checker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.ids, self.links, self.external, self.svgs = [], set(), [], [], 0
        self.void = {"meta", "br", "img", "input", "link", "hr", "path", "line", "circle", "rect"}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        for k in ("src", "href"):
            v = a.get(k)
            if v:
                if re.match(r"^(https?:)?//", v):
                    self.external.append(v)
                elif k == "href":
                    self.links.append(v)
        if tag == "link" or (tag == "script" and a.get("src")):
            self.external.append(tag)
        if tag == "svg":
            self.svgs += 1
        if tag not in self.void:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        assert self.stack and self.stack[-1] == tag, (self.stack[-3:], tag)
        self.stack.pop()


def _check_page(path):
    src = open(path).read()
    c = _Checker()
    c.feed(src)
    assert c.stack == [], c.stack[-5:]
    assert c.external == []                                   # single file: no external assets
    for svg in re.findall(r"<svg.*?</svg>", src, flags=re.S):
        ET.fromstring(svg.replace("&nbsp;", " "))              # every chart is well-formed SVG
    return src, c


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    run = str(tmp_path_factory.mktemp("dash") / "run")
    L = LearnerSpec(run_dir=run, model_kwargs={"seed": 0, "lr": 1e-2}, min_decisions=128, max_batch_decisions=256)
    A = [ActorSpec(actor_id=f"actor{i}", run_dir=run, env_kwargs={"B": 2, "mechanism_seed": 0}, seed=20 + i,
                   batch_records=128) for i in range(2)]
    Launcher(run, L, A).run(duration_s=8.0)
    os.makedirs(os.path.join(run, "eval"))
    with open(os.path.join(run, "eval", "retention.jsonl"), "w") as fh:     # as an evaluation harness writes them
        for v, val in ((0, 0.25), (5, 0.6), (10, 0.8)):
            fh.write(json.dumps({"metric": "retention_score", "family": "toy_panel", "version": v, "value": val}) + "\n")
    return run


def test_dashboard_renders_every_section_from_a_real_run(run_dir):
    path = build_dashboard(run_dir)
    src, c = _check_page(path)
    for sec in ("worlds", "checkpoints", "experience", "learning", "calibration", "retention", "imagined",
                "lifetimes", "updates", "compute"):
        assert sec in c.ids, sec
    latest = json.load(open(os.path.join(run_dir, "ckpt", "LATEST")))
    assert f"LATEST → version <b>{latest['version']:,}</b>" in src
    assert "actor0" in src and "actor1" in src and "Earn as many panel payouts" in src
    assert "retention_score" in src and "No retention or transfer evaluations" not in src
    assert "reliability" in src and "Brier" in src                    # calibration from real traces
    assert "No imagined trajectories" not in src and "payoff@h1" in src
    assert "Policy-lag routing" in src and src.count("<svg") == c.svgs >= 6
    assert "Table view" in src and "prefers-color-scheme:dark" in src
    lifetime_links = [l for l in c.links if l.startswith("lifetime-")]
    traces = load_traces(os.path.join(run_dir, "traces"))
    envs = {t.env_id for t in traces}
    assert len(lifetime_links) == len(envs) == 4
    for l in lifetime_links:
        lsrc, _ = _check_page(os.path.join(os.path.dirname(path), l))
        assert "Retrospective summary" in lsrc and "not direct access to the model" in lsrc
        assert "Action timeline" in lsrc and "Predicted vs observed: payoff" in lsrc
        assert 'href="index.html"' in lsrc
    # numbers shown are the run's numbers
    ups = [json.loads(x) for x in open(os.path.join(run_dir, "learner", "updates.jsonl"))]
    last = [u for u in ups if u["trained"]][-1]
    assert f">{last['update_id']}<" in src


def test_dashboard_states_missing_data_instead_of_inventing_it(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    src, c = _check_page(build_dashboard(str(run)))
    assert "No actor heartbeats recorded" in src and "No checkpoints published" in src
    assert "No retention or transfer evaluations recorded" in src and "No decision traces recorded" in src
    assert c.svgs == 0


def _t(i, p, o, u=None, imagined=None):
    return DecisionTrace(decision_id=f"d{i}", event_id=f"e{i}", env_id="w", policy_version=0, t_sim=float(i),
                         objective="", predicted={"k": p}, uncertainty={} if u is None else {"k": u},
                         candidates=[{"action": {"tool": "a"}}, {"action": {"tool": "b"}}],
                         selected={"tool": "b", "args": {}}, imagined=imagined or [], outcome={"k": o})


def test_calibration_and_real_vs_imagined_analyses():
    # probabilities: 10 predictions of 0.8 with 8 successes, 10 of 0.2 with 2 -> perfectly calibrated
    ts = [_t(i, 0.8, 1.0 if i < 8 else 0.0) for i in range(10)] + \
         [_t(10 + i, 0.2, 1.0 if i < 2 else 0.0) for i in range(10)]
    c = calibration(ts)["k"]
    assert c["type"] == "probability" and c["n"] == 20
    assert c["ece"] == pytest.approx(0.0, abs=1e-9) and c["brier"] == pytest.approx(0.16)
    assert sorted((round(a, 3), round(b, 3), n) for a, b, n in c["reliability"]) == [(0.2, 0.2, 10), (0.8, 0.8, 10)]
    # continuous: errors of 0.5 with sigma 1 -> inside both intervals; errors of 3 -> outside both
    cs = [_t(i, 0.0, 0.5, u=1.0) for i in range(3)] + [_t(3, 0.0, 3.0, u=1.0)]
    cc = calibration(cs)["k"]
    assert cc["type"] == "continuous" and cc["coverage_1sd"] == 0.75 and cc["coverage_2sd"] == 0.75
    assert cc["mae"] == pytest.approx((0.5 * 3 + 3.0) / 4)
    im = [_t(i, 0.0, 1.0, imagined=[{"candidate": 0, "horizon": 1, "outcomes": {"k": 9.0}},
                                    {"candidate": 1, "horizon": 1, "outcomes": {"k": 0.7}}]) for i in range(5)]
    r = real_vs_imagined(im)["k@h1"]
    assert r["n"] == 5 and r["mean_imagined"] == pytest.approx(0.7) and r["mean_real"] == 1.0   # selected = 'b'
    assert r["mae"] == pytest.approx(0.3)
