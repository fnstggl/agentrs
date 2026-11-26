"""A throwaway repository with the real acceptance package, one requirement (Z.1) and its gate test, for testing the
checker against honest and fabricated evidence. Everything runs for real: git commits, official runs in fresh clones,
pytest subprocesses, reviews that re-execute the tests."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from crucible.acceptance.v5 import checker, manifest as M, official

PKG = os.path.dirname(os.path.abspath(checker.__file__))
CRIT = [{"metric": "widgets", "op": ">=", "value": 10}, {"metric": "leaks", "op": "==", "value": 0}]

REQS = '''"""mini requirements for checker tests"""
GATE_ITEMS = {{}}
CLARIFIED = {{}}
GATE_TITLE_OVERRIDE = {{}}
GATE_NOTES = []
SPEC_ITEMS = []
TOLERANCES = {tol!r}
AMENDMENT_ITEMS = {items!r}
'''

GOOD_TEST = '''from crucible.acceptance.v5 import evidence
import widgets


def test_z():
    n = widgets.count()
    leaks = widgets.leaks()
    evidence.record("Z.1", {"widgets": n, "leaks": leaks})
    assert n >= 10 and leaks == 0
'''

WIDGETS = "def count():\n    return 12\n\n\ndef leaks():\n    return 0\n"


def git(root: str, *a: str) -> str:
    return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


class Repo:
    def __init__(self, tmp: str, monkeypatch, *, items=None, tol=None, test_src: str = GOOD_TEST,
                 widgets: str = WIDGETS, extra_files: dict | None = None):
        self.tmp, self.mp = tmp, monkeypatch
        self.top = os.path.join(tmp, "repo")
        self.root = os.path.join(self.top, "crucible")
        self.scratch = os.path.join(tmp, "scratch")
        os.makedirs(self.scratch)
        src = os.path.join(self.root, "src")
        shutil.copytree(PKG, os.path.join(src, "crucible", "acceptance", "v5"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        hooks = os.path.join(M.crucible_root(), "docs", "v5", "baselines", "python_site_hooks.json")
        if os.path.exists(hooks):                   # the reference interpreter's start-up hooks (checker.env_problems)
            self.write("docs/v5/baselines/python_site_hooks.json", open(hooks).read())
        self.write("src/crucible/__init__.py", '"""mini crucible"""\n')
        self.write("src/crucible/acceptance/__init__.py", "")
        self.items = items if items is not None else [("Z.1", "Z", "make widgets", "A", [], CRIT, 3)]
        self.tol = tol or {}
        self.write_requirements()
        self.write("src/widgets/__init__.py", widgets)
        self.write("tests/v5/test_gate_Z.py", test_src)
        self.write("docs/v5/DECISION_LOG.md", "# log\n\n## D-1 raise the bar for Z.1\nZ.1 needs more widgets.\n")
        self.write("docs/v5/BLOCKERS.md", "# blockers\n\n## Open blockers\n\n### BLK-1 the widget mine is closed\n"
                                          "Blocks Z.2 (no hard widgets without the mine).\n\n## Risks being managed\n")
        for p, s in (extra_files or {}).items():
            self.write(p, s)
        git(self.top, "init", "-q")
        git(self.top, "config", "user.email", "t@t")
        git(self.top, "config", "user.name", "t")
        M.build(self.root)
        for rid, *_ in self.items:
            M.add_files(self.root, rid, implementation_files=["src/widgets/__init__.py"])
        M.sync(self.root)
        self.commit("impl")

    def write(self, rel: str, text: str) -> None:
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)

    def read(self, rel: str) -> str:
        return open(os.path.join(self.root, rel)).read()

    def write_requirements(self) -> None:
        self.write("src/crucible/acceptance/v5/requirements.py", REQS.format(items=self.items, tol=self.tol))

    def commit(self, msg: str) -> str:
        git(self.top, "add", "-A")
        git(self.top, "commit", "-qm", msg, "--allow-empty")
        return git(self.top, "rev-parse", "HEAD")

    def official_run(self, agent: str = "builder", session: str = "s-builder", requirement: str = "Z.1", **kw) -> dict:
        self.mp.setenv("CRUCIBLE_V5_SESSION", session)
        return official.run(self.root, requirement=requirement, agent=agent, scratch=self.scratch, **kw)

    def review(self, agent: str = "qa", session: str = "s-qa", requirement: str = "Z.1", verdict: str = "approved"):
        self.mp.setenv("CRUCIBLE_V5_SESSION", session)
        return official.review(self.root, requirement=requirement, agent=agent, verdict=verdict, notes="",
                               scratch=self.scratch)

    def honest(self) -> None:
        """The full legitimate pipeline: official run, commit, independent review, commit."""
        self.official_run()
        self.commit("evidence")
        rv = self.review()
        assert rv["verdict"] == "approved", rv["reproduction"]
        self.commit("review")

    def verify(self, **kw) -> tuple[dict, dict]:
        res = checker.verify_all(self.root, **kw)
        rows = {r["id"]: r for r in res["rows"]}
        return res, rows

    def manifest(self) -> dict:
        return M.load(self.root)

    def save_manifest(self, m: dict) -> None:
        M.save(m, self.root)

    def json(self, rel: str) -> dict:
        return json.load(open(os.path.join(self.root, rel)))

    def dump(self, rel: str, obj) -> None:
        self.write(rel, json.dumps(obj, indent=1, sort_keys=True))
